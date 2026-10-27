"""JSON API v1 (``/api/v1``): published briefs, stories, search, Ask Newsren, finance trackers, alerts, metrics.

The API follows the web visibility rules:

* only approved (published) briefs are public; editors may also fetch
  unpublished briefs, which then carry their editorial data;
* a story's sourced timeline is public, but its AI story-intelligence summary
  is included only once the story has an approved brief (or for editors);
* finance responses carry the not-investment-advice ``disclaimer``.

Authentication uses the web session cookie (sign in at ``/login``).
``/api/v1/me/*`` needs a signed-in reader (401 otherwise) and
``/api/v1/metrics`` an editor (401 / 403).

State-changing requests (``POST /api/v1/ask``) need the session's CSRF token
in the ``X-CSRF-Token`` header. Fetch it from ``GET /api/v1/csrf`` (same
cookie jar) first; a request without a valid token gets 403. All timestamps
are ISO-8601 UTC (``2026-10-27T09:30:00Z``).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from newsren.agents.research import ask
from newsren.agents.tools import archive_search, query_terms
from newsren.editorial.guardrails import FINANCE_DISCLAIMER
from newsren.finance.ai_claims import CLASSIFICATION_LABELS
from newsren.finance.funding import STAGE_LABELS, funding_trends
from newsren.intelligence.clustering import related_stories
from newsren.intelligence.timeline import member_articles, story_citations
from newsren.metrics import all_metrics
from newsren.models import (
    AIFinanceClaim,
    Alert,
    Article,
    Brief,
    BriefClaim,
    BriefStatus,
    Correction,
    FundingRound,
    Story,
    StoryEvent,
    User,
    Vertical,
)
from newsren.textutil import iso
from newsren.users.service import user_alerts
from newsren.web.deps import CSRF, DB, CurrentUser, csrf_token, session_key

router = APIRouter(prefix="/api/v1", tags=["api"])

API_VERSION = "v1"
MAX_LIMIT = 100
MAX_QUESTION_CHARS = 1000
MAX_SEARCH_TERMS = 8
SEARCH_CANDIDATES = 200
FINANCE = Vertical.FINANCE.value


# --------------------------------------------------------------------------
# Access helpers
# --------------------------------------------------------------------------


def is_editor(user: User | None) -> bool:
    return user is not None and user.is_editor


def require_api_user(user: CurrentUser) -> User:
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in to use this endpoint (session cookie from /login).")
    return user


def require_api_editor(user: CurrentUser) -> User:
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in as an editor to use this endpoint.")
    if not user.is_editor:
        raise HTTPException(status_code=403, detail="Editors only.")
    return user


ApiUser = Annotated[User, Depends(require_api_user)]
ApiEditor = Annotated[User, Depends(require_api_editor)]


def visible_brief(db: Session, brief_id: int, user: User | None) -> Brief:
    brief = db.get(Brief, brief_id)
    if brief is None or (brief.status != BriefStatus.APPROVED and not is_editor(user)):
        raise HTTPException(status_code=404, detail="No published brief with that id.")
    return brief


def approved_brief_id(db: Session, story_id: int) -> int | None:
    return db.scalar(
        select(Brief.id)
        .where(Brief.story_id == story_id, Brief.status == BriefStatus.APPROVED)
        .order_by(Brief.published_at.desc().nulls_last(), Brief.id.desc())
        .limit(1)
    )


# --------------------------------------------------------------------------
# Serializers (plain JSON-ready dicts)
# --------------------------------------------------------------------------


def claim_json(claim: BriefClaim, editor: bool = False) -> dict[str, Any]:
    out = {
        "id": claim.id,
        "text": claim.text,
        "claim_type": claim.claim_type,
        "support_status": claim.support_status,
        "citation_ids": list(claim.citation_ids or []),
        "performance_label": claim.performance_label,
        "performance_period": claim.performance_period,
        "benchmark": claim.benchmark,
        "high_impact": bool(claim.high_impact),
        "notes": claim.notes,
    }
    if editor:
        out["editor_verdict"] = claim.editor_verdict
        out["verdict_at"] = iso(claim.verdict_at)
    return out


def brief_json(db: Session, brief: Brief, user: User | None) -> dict[str, Any]:
    editor = is_editor(user)
    corrections = db.scalars(
        select(Correction).where(Correction.brief_id == brief.id).order_by(Correction.created_at.asc())
    ).all()
    out: dict[str, Any] = {
        "id": brief.id,
        "url": f"/briefs/{brief.id}",
        "story_id": brief.story_id,
        "vertical": brief.vertical,
        "kind": brief.kind,
        "status": brief.status,
        "published": brief.status == BriefStatus.APPROVED,
        "headline": brief.headline,
        "summary": brief.summary,
        "why_it_matters": brief.why_it_matters,
        "what_changed": brief.what_changed,
        "key_points": list(brief.key_points or []),
        "sections": list(brief.sections or []),
        "uncertainty": list(brief.uncertainty or []),
        "disagreements": list(brief.disagreements or []),
        "watch_next": list(brief.watch_next or []),
        "claims": [claim_json(c, editor) for c in brief.claims or []],
        "citations": list(brief.citations or []),
        "generated_by": brief.generated_by,
        "generated_at": iso(brief.generated_at),
        "published_at": iso(brief.published_at),
        "reviewed_by_human": brief.reviewer_id is not None and brief.status == BriefStatus.APPROVED,
        "version": brief.version,
        "is_demo": bool(brief.is_demo),
        "corrections": [
            {"description": c.description, "material": bool(c.material), "created_at": iso(c.created_at)}
            for c in corrections
        ],
        "disclaimer": FINANCE_DISCLAIMER if brief.vertical == FINANCE else None,
    }
    if editor:
        out["editorial"] = {
            "risk_level": brief.risk_level,
            "requires_human_review": bool(brief.requires_human_review),
            "check_results": list(brief.check_results or []),
            "review_notes": brief.review_notes,
            "review_started_at": iso(brief.review_started_at),
            "reviewed_at": iso(brief.reviewed_at),
            "agent_run_id": brief.agent_run_id,
            "review_url": f"/editor/briefs/{brief.id}",
        }
    return out


def brief_summary_json(brief: Brief) -> dict[str, Any]:
    return {
        "id": brief.id,
        "url": f"/briefs/{brief.id}",
        "vertical": brief.vertical,
        "kind": brief.kind,
        "headline": brief.headline,
        "summary": brief.summary,
        "published_at": iso(brief.published_at),
        "story_id": brief.story_id,
        "citations": len(brief.citations or []),
        "is_demo": bool(brief.is_demo),
    }


def article_json(article: Article) -> dict[str, Any]:
    return {
        "id": article.id,
        "url": article.url,
        "title": article.title,
        "publisher": article.source.name if article.source else None,
        "summary": article.summary,
        "vertical": article.vertical,
        "doc_type": article.doc_type,
        "trust_tier": article.trust_tier,
        "published_at": iso(article.published_at),
        "retrieved_at": iso(article.retrieved_at),
        "story_id": article.story_id,
        "is_demo": bool(article.is_demo),
    }


def event_json(event: StoryEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "occurred_at": iso(event.occurred_at),
        "event_type": event.event_type,
        "title": event.title,
        "description": event.description,
        "is_correction": bool(event.is_correction),
        "article": article_json(event.article) if event.article is not None else None,
    }


def funding_round_json(r: FundingRound) -> dict[str, Any]:
    return {
        "id": r.id,
        "company_name": r.company_name,
        "sector": r.sector,
        "stage": r.stage,
        "stage_label": STAGE_LABELS.get(r.stage, r.stage),
        "amount_text": r.amount_text,
        "amount_usd": r.amount_usd,
        "lead_investors": list(r.lead_investors or []),
        "investors": list(r.investors or []),
        "announced_at": iso(r.announced_at),
        "source_url": r.source_url,
        "source_title": r.source_title,
        "retrieved_at": iso(r.retrieved_at),
        "extraction_confidence": r.extraction_confidence,
        "verified_by_editor": bool(r.verified),
        "is_demo": bool(r.is_demo),
    }


def ai_claim_json(c: AIFinanceClaim) -> dict[str, Any]:
    return {
        "id": c.id,
        "firm": c.firm,
        "area": c.area,
        "claim_text": c.claim_text,
        "classification": c.classification,
        "classification_label": CLASSIFICATION_LABELS.get(c.classification, c.classification),
        "performance_label": c.performance_label,
        "period": c.period,
        "benchmark": c.benchmark,
        "costs_assumed": c.costs_assumed,
        "authors_caveats": c.authors_caveats,
        "evidence_note": c.evidence_note,
        "source_url": c.source_url,
        "source_title": c.source_title,
        "published_at": iso(c.published_at),
        "retrieved_at": iso(c.retrieved_at),
        "article_id": c.article_id,
        "from_plan_research": bool(c.is_seed),
    }


def alert_json(alert: Alert) -> dict[str, Any]:
    return {
        "id": alert.id,
        "title": alert.title,
        "reason": alert.reason,
        "event_type": alert.event_type,
        "matched_criteria": list(alert.matched_criteria or []),
        "citations": list(alert.citations or []),
        "story_id": alert.story_id,
        "article_id": alert.article_id,
        "follow": {"id": alert.follow_id, "question": alert.follow.question if alert.follow else None},
        "created_at": iso(alert.created_at),
        "read_at": iso(alert.read_at),
        "feedback": alert.feedback,
    }


# --------------------------------------------------------------------------
# Index and CSRF
# --------------------------------------------------------------------------


@router.get("")
def api_index() -> dict[str, Any]:
    """What the API offers and how to authenticate."""
    return {
        "name": "The Newsren API",
        "version": API_VERSION,
        "auth": "Session cookie from /login. /me endpoints need a signed-in reader; /metrics needs an editor.",
        "csrf": "POST requests need the X-CSRF-Token header; get the token from GET /api/v1/csrf with the same cookies.",
        "timestamps": "ISO-8601 UTC",
        "endpoints": [
            {"method": "GET", "path": "/api/v1/briefs", "params": ["vertical", "limit"], "access": "public"},
            {"method": "GET", "path": "/api/v1/briefs/{id}", "access": "public (published) / editor (any)"},
            {"method": "GET", "path": "/api/v1/stories/{id}", "access": "public"},
            {"method": "GET", "path": "/api/v1/search", "params": ["q", "limit"], "access": "public"},
            {
                "method": "POST",
                "path": "/api/v1/ask",
                "body": "{question, brief_id?, audience?}",
                "access": "CSRF",
            },
            {"method": "GET", "path": "/api/v1/finance/funding", "params": ["days", "limit"], "access": "public"},
            {
                "method": "GET",
                "path": "/api/v1/finance/ai-claims",
                "params": ["classification"],
                "access": "public",
            },
            {"method": "GET", "path": "/api/v1/me/alerts", "params": ["unread", "limit"], "access": "signed in"},
            {"method": "GET", "path": "/api/v1/metrics", "access": "editor"},
            {"method": "GET", "path": "/api/v1/csrf", "access": "public"},
        ],
        "finance_disclaimer": FINANCE_DISCLAIMER,
    }


@router.get("/csrf")
def api_csrf(request: Request) -> dict[str, str]:
    """The session's CSRF token, to send as ``X-CSRF-Token`` on POST requests (same-origin readable only)."""
    return {"csrf_token": csrf_token(request), "header": "X-CSRF-Token"}


# --------------------------------------------------------------------------
# Briefs, stories, search
# --------------------------------------------------------------------------


@router.get("/briefs")
def api_briefs(
    db: DB,
    vertical: Vertical | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 20,
) -> dict[str, Any]:
    """Published briefs, newest first, with their claims and citations."""
    query = select(Brief).where(Brief.status == BriefStatus.APPROVED)
    if vertical is not None:
        query = query.where(Brief.vertical == vertical.value)
    briefs = db.scalars(query.order_by(Brief.published_at.desc().nulls_last(), Brief.id.desc()).limit(limit)).all()
    return {
        "vertical": vertical.value if vertical is not None else None,
        "count": len(briefs),
        "briefs": [brief_json(db, b, None) for b in briefs],
        "disclaimer": FINANCE_DISCLAIMER
        if vertical == Vertical.FINANCE or any(b.vertical == FINANCE for b in briefs)
        else None,
    }


@router.get("/briefs/{brief_id}")
def api_brief(brief_id: int, db: DB, user: CurrentUser) -> dict[str, Any]:
    """One brief: published for everyone; unpublished ones only for editors (with editorial data)."""
    return brief_json(db, visible_brief(db, brief_id, user), user)


@router.get("/stories/{story_id}")
def api_story(story_id: int, db: DB, user: CurrentUser) -> dict[str, Any]:
    """A story with its sourced timeline, related story ids, and (when published) its intelligence summary."""
    story = db.get(Story, story_id)
    if story is None:
        raise HTTPException(status_code=404, detail="No story with that id.")
    brief_id = approved_brief_id(db, story.id)
    show_intel = brief_id is not None or is_editor(user)
    related = related_stories(db, story)
    intelligence: dict[str, Any] | None = None
    if show_intel:
        intelligence = {
            "what_happened_first": story.what_happened_first,
            "whats_new": story.whats_new,
            "disputed": list(story.disputed or []),
            "evidence_changes": list(story.evidence_changes or []),
            "citations": [c.model_dump() for c in story_citations(story)],
            "preview": brief_id is None,
        }
    return {
        "id": story.id,
        "url": f"/stories/{story.id}",
        "title": story.title,
        "vertical": story.vertical,
        "status": story.status,
        "first_seen_at": iso(story.first_seen_at),
        "last_updated_at": iso(story.last_updated_at),
        "version": story.version,
        "is_demo": bool(story.is_demo),
        "key_entities": list(story.key_entities or []),
        "keywords": list(story.keywords or []),
        "article_count": len(member_articles(story)),
        "brief_id": brief_id,
        "intelligence": intelligence,
        "intelligence_note": None
        if show_intel
        else "The AI story-intelligence summary is published once an editor approves a brief for this story.",
        "events": [event_json(e) for e in story.events],
        "related_story_ids": [s.id for s in related],
        "related": [{"id": s.id, "title": s.title, "vertical": s.vertical} for s in related],
        "disclaimer": FINANCE_DISCLAIMER if story.vertical == FINANCE else None,
    }


def search_published_briefs(db: Session, query: str, limit: int) -> list[Brief]:
    """Published briefs whose headline or summary match at least half the query's terms (headline weighs more)."""
    terms = query_terms(query)[:MAX_SEARCH_TERMS]
    if not terms:
        return []
    conditions = []
    for term in terms:
        pattern = "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        conditions += [Brief.headline.ilike(pattern, escape="\\"), Brief.summary.ilike(pattern, escape="\\")]
    candidates = db.scalars(
        select(Brief)
        .where(Brief.status == BriefStatus.APPROVED, or_(*conditions))
        .order_by(Brief.published_at.desc().nulls_last(), Brief.id.desc())
        .limit(SEARCH_CANDIDATES)
    ).all()
    scored = []
    for position, brief in enumerate(candidates):
        headline, summary = brief.headline.lower(), (brief.summary or "").lower()
        matched = [t for t in terms if t in headline or t in summary]
        if len(matched) * 2 >= len(terms):
            score = sum(3 for t in terms if t in headline) + sum(1 for t in terms if t in summary)
            scored.append((score, -position, brief))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [b for _, _, b in scored[:limit]]


@router.get("/search")
def api_search(
    db: DB,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> dict[str, Any]:
    """Published briefs and archived articles (approved sources only) matching ``q``."""
    query = " ".join(q.split())
    if not query:
        raise HTTPException(status_code=422, detail="Give a search query in q.")
    briefs = search_published_briefs(db, query, limit)
    articles = archive_search(db, query, limit=limit)
    return {
        "query": query,
        "briefs": [brief_summary_json(b) for b in briefs],
        "articles": [article_json(a) for a in articles],
    }


# --------------------------------------------------------------------------
# Ask Newsren
# --------------------------------------------------------------------------


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    brief_id: int | None = None
    audience: Literal["student", "educator", "professional"] | None = None


async def ask_body(request: Request) -> AskIn:
    """Parse the JSON body by hand: the CSRF dependency reads form fields, so FastAPI can't also bind a JSON model."""
    try:
        data = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail='Send a JSON body: {"question": "..."}.') from exc
    try:
        return AskIn.model_validate(data)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors(include_url=False, include_context=False)) from exc


@router.post("/ask", dependencies=[CSRF], status_code=201)
def api_ask(
    request: Request,
    db: DB,
    user: CurrentUser,
    payload: Annotated[AskIn, Depends(ask_body)],
) -> dict[str, Any]:
    """Ask the research agent a question (optionally about a published brief) and get a sourced answer.

    Requires the ``X-CSRF-Token`` header (see ``GET /api/v1/csrf``). Finance
    advice requests are answered educationally and flagged ``refused``.
    """
    text = " ".join(payload.question.split())
    if not text:
        raise HTTPException(status_code=422, detail="The question is empty.")
    brief = visible_brief(db, payload.brief_id, user) if payload.brief_id is not None else None
    audience = payload.audience or (user.audience if user is not None else None)
    row = ask(db, text, brief=brief, audience=audience, user=user, session_key=session_key(request))
    result = dict(row.result or {})
    finance = (brief is not None and brief.vertical == FINANCE) or bool(row.refused)
    return {
        "question_id": row.id,
        "url": f"/questions/{row.id}",
        "question": row.question,
        "brief_id": row.brief_id,
        "story_id": row.story_id,
        "audience": row.audience,
        "answer": result.get("answer", row.answer),
        "plan": result.get("plan", []),
        "verified": result.get("verified", []),
        "could_not_verify": result.get("could_not_verify", []),
        "citations": result.get("citations", row.citations or []),
        "follow_up_questions": result.get("follow_up_questions", []),
        "refused": bool(row.refused),
        "refusal_reason": result.get("refusal_reason"),
        "agent_run_id": row.agent_run_id,
        "created_at": iso(row.created_at),
        "disclaimer": FINANCE_DISCLAIMER if finance else None,
    }


# --------------------------------------------------------------------------
# Finance trackers
# --------------------------------------------------------------------------


@router.get("/finance/funding")
def api_funding(
    db: DB,
    days: Annotated[int, Query(ge=1, le=3650)] = 90,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 50,
) -> dict[str, Any]:
    """Funding Round Tracker: trends over ``days`` (never a ranking of startups) plus logged rounds."""
    trends = funding_trends(db, days=days)
    rounds = db.scalars(
        select(FundingRound)
        .order_by(FundingRound.announced_at.desc().nulls_last(), FundingRound.id.desc())
        .limit(limit)
    ).all()
    return {
        "disclaimer": FINANCE_DISCLAIMER,
        "trends": {
            "window_days": trends["window_days"],
            "since": trends["since"],
            "total_rounds": trends["total_rounds"],
            "total_disclosed_usd": trends["total_disclosed_usd"],
            "rounds_with_disclosed_usd": trends["rounds_with_disclosed_usd"],
            "by_sector": trends["by_sector"],
            "by_stage": trends["by_stage"],
            "investor_participation": [{"name": name, "rounds": n} for name, n in trends["top_investors"]],
            "includes_demo": trends["includes_demo"],
            "note": trends["note"],
        },
        "rounds": [funding_round_json(r) for r in rounds],
    }


@router.get("/finance/ai-claims")
def api_ai_claims(db: DB, classification: str | None = None) -> dict[str, Any]:
    """AI-in-Finance Reality Check: claims classified as announcement, pilot, backtest, live deployment, or regulatory finding."""
    if classification is not None and classification not in CLASSIFICATION_LABELS:
        raise HTTPException(
            status_code=422, detail="classification must be one of: " + ", ".join(CLASSIFICATION_LABELS)
        )
    query = select(AIFinanceClaim)
    if classification:
        query = query.where(AIFinanceClaim.classification == classification)
    claims = db.scalars(
        query.order_by(AIFinanceClaim.published_at.desc().nulls_last(), AIFinanceClaim.id.desc())
    ).all()
    return {
        "disclaimer": FINANCE_DISCLAIMER,
        "classifications": CLASSIFICATION_LABELS,
        "count": len(claims),
        "claims": [ai_claim_json(c) for c in claims],
    }


# --------------------------------------------------------------------------
# Reader and editor endpoints
# --------------------------------------------------------------------------


@router.get("/me/alerts")
def api_my_alerts(
    db: DB,
    user: ApiUser,
    unread: bool = False,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 50,
) -> dict[str, Any]:
    """The signed-in reader's Follow This Story alerts, newest first."""
    alerts = user_alerts(db, user, unread_only=unread, limit=limit)
    return {"count": len(alerts), "alerts": [alert_json(a) for a in alerts]}


@router.get("/metrics")
def api_metrics(db: DB, editor: ApiEditor) -> dict[str, Any]:
    """Every success metric of plan §9 (editors only)."""
    return all_metrics(db)
