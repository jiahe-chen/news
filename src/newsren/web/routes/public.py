"""Public reading pages: front page, vertical sections, AI Story Briefs, stories, search, sources, about.

Reading stays central (plan §5): these pages are a recognizable editorial
front end. The intelligence layer (Connected stories, Learn this topic,
Inspect evidence, Industry impact, Follow developments, Ask about this story,
editorial products) is reached from the brief page's tool bar, whose partials
live in ``newsren.web.routes.tools``.

Visibility rules:
* only approved briefs are public; editors may preview unpublished briefs
  (shown with an "Unpublished draft" banner), everyone else gets a 404;
* a story's sourced timeline is public, but its AI story-intelligence summary
  is shown only once the story has an approved brief (or to editors).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from newsren.agents.learning import strip_citation_marks
from newsren.agents.products import PRODUCTS, available_products
from newsren.agents.tools import archive_search, doc_type_label, query_terms, trust_tier_label
from newsren.config import get_settings
from newsren.editorial.guardrails import FINANCE_DISCLAIMER
from newsren.editorial.workflow import published_briefs
from newsren.intelligence.clustering import related_stories
from newsren.intelligence.events import event_label
from newsren.intelligence.timeline import AGENT_NAME as STORY_AGENT_NAME
from newsren.intelligence.timeline import member_articles, story_citations
from newsren.llm import claude_credentials_available
from newsren.models import (
    AgentRun,
    AgentRunStatus,
    Article,
    Brief,
    BriefStatus,
    Correction,
    DocType,
    EventType,
    Source,
    Story,
    User,
    Vertical,
)
from newsren.web.deps import DB, CurrentUser, ReaderAudience
from newsren.web.templating import render
from newsren.web.tracking import record_view

router = APIRouter()

FINANCE = Vertical.FINANCE.value
SECTION_VERTICALS = [Vertical.TECHNOLOGY.value, Vertical.EDUCATION.value, Vertical.INDUSTRY.value, FINANCE]
PAGE_VERTICALS = [Vertical.TECHNOLOGY.value, Vertical.EDUCATION.value, Vertical.INDUSTRY.value]

# Plan §3: what each vertical monitors and how it goes beyond summarizing.
VERTICAL_INTROS: dict[str, dict[str, Any]] = {
    "technology": {
        "headline": "From announcements to technical understanding",
        "monitors": (
            "Model releases, research publications, benchmarks, open-source repositories, "
            "cybersecurity advisories, and product updates."
        ),
        "beyond": (
            "Beyond summarizing, The Newsren distinguishes vendor claims from third-party evaluations, "
            "explains technical tradeoffs, and tracks how capabilities evolve."
        ),
        "entry_title": "Understand the next technology breakthrough",
        "entry_text": "Compare primary research, product announcements, and independent evaluations.",
        "prompts": [
            "Which recent AI model claims have been checked by an independent benchmark?",
            "What is technically new in the latest model release, and what evidence exists?",
            "Which security advisories should schools and small organizations act on?",
        ],
    },
    "education": {
        "headline": "From headlines to useful learning",
        "monitors": (
            "Education policy, curriculum changes, AI literacy, research on learning, higher education, "
            "and access to educational opportunities."
        ),
        "beyond": (
            "The Newsren creates age-appropriate explainers, classroom discussions, comparisons of "
            "instructional strategies, and evidence-based assessments of new education tools."
        ),
        "entry_title": "Find out what the evidence says about learning",
        "entry_text": "Compare study results with press releases, and see who a new policy affects and when.",
        "prompts": [
            "Does AI tutoring improve learning, or only speed up practice?",
            "Who is affected by new AI literacy requirements, and on what timeline?",
            "What did the latest study measure, and who took part?",
        ],
    },
    "industry": {
        "headline": "From isolated news to strategic implications",
        "monitors": ("Companies, supply chains, funding, regulations, market trends, and technology adoption."),
        "beyond": (
            "The Newsren connects developments across organizations, explains plausible operational "
            "implications, and identifies questions professionals should investigate before taking action."
        ),
        "entry_title": "Trace how a development ripples through an industry",
        "entry_text": "Connect company announcements, supply chains, and workforce changes — with scenarios labelled as such.",
        "prompts": [
            "How could new chip fab capacity affect suppliers and device makers?",
            "Which skills are manufacturers retraining workers for?",
            "Which announced capacity has actually been delivered so far?",
        ],
    },
    "finance": {
        "headline": "From market headlines to evidence-based understanding",
        "monitors": (
            "Earnings releases and calls, regulatory filings, central-bank and regulator announcements, "
            "venture funding rounds, and how financial institutions are adopting AI."
        ),
        "beyond": (
            "The Newsren explains what a development means for investors, founders, and finance "
            "professionals, separates reported facts from forecasts, and teaches the underlying concepts."
        ),
    },
}

# The homepage agent entry point (plan §5) with one suggested prompt per vertical.
HOME_PROMPTS = [
    ("technology", "Which AI model claims have been checked by independent benchmarks?"),
    ("education", "Does AI tutoring improve learning, or only speed up practice?"),
    ("industry", "How could new semiconductor capacity affect suppliers and device makers?"),
    ("finance", "What is the difference between a backtest and a live track record?"),
]

# Intelligence tools on the brief page (URLs implemented by the tools routes).
TOOL_TABS = [
    ("connected", "Connected stories"),
    ("learn", "Learn this topic"),
    ("evidence", "Inspect evidence"),
    ("impact", "Industry impact"),
    ("follow", "Follow developments"),
    ("ask", "Ask about this story"),
]

STORY_STATUS_LABELS = {"developing": "Developing", "active": "Active", "dormant": "Dormant"}

DOC_TYPE_ORDER = [d.value for d in DocType]

MAX_SEARCH_TERMS = 8
BRIEF_SEARCH_CANDIDATES = 200
SEARCH_BRIEF_LIMIT = 10
SEARCH_ARTICLE_LIMIT = 20


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def is_editor(user: User | None) -> bool:
    return user is not None and user.is_editor


def is_opinion_brief(brief: Brief) -> bool:
    """True when every source the brief cites is opinion or commentary (a column, not a news report)."""
    cites = brief.citations or []
    return bool(cites) and all(
        c.get("doc_type") == DocType.OPINION or c.get("trust_tier") == "commentary" for c in cites
    )


def load_brief(db: Session, brief_id: int, user: User | None) -> Brief:
    """The brief if the reader may see it: approved for everyone, any status for editors; else 404."""
    brief = db.get(Brief, brief_id)
    if brief is None:
        raise HTTPException(status_code=404, detail="No brief with that id.")
    if brief.status != BriefStatus.APPROVED and not is_editor(user):
        raise HTTPException(status_code=404, detail="No published brief with that id.")
    return brief


def approved_brief_for_story(db: Session, story_id: int) -> Brief | None:
    return db.scalars(
        select(Brief)
        .where(Brief.story_id == story_id, Brief.status == BriefStatus.APPROVED)
        .order_by(Brief.published_at.desc().nulls_last(), Brief.id.desc())
        .limit(1)
    ).first()


def latest_brief_for_story(db: Session, story_id: int) -> Brief | None:
    return db.scalars(select(Brief).where(Brief.story_id == story_id).order_by(Brief.id.desc()).limit(1)).first()


def member_counts(db: Session) -> dict[int, int]:
    """Non-duplicate member articles per story."""
    rows = db.execute(
        select(Article.story_id, func.count(Article.id))
        .where(Article.story_id.is_not(None), Article.duplicate_of_id.is_(None))
        .group_by(Article.story_id)
    ).all()
    return {int(story_id): int(n) for story_id, n in rows}


def story_event_types(story: Story, limit: int = 4) -> list[str]:
    """Distinct event types on a story's timeline, in order of first occurrence (generic ones skipped)."""
    seen: list[str] = []
    for event in story.events:
        etype = event.event_type or EventType.OTHER
        if etype != EventType.OTHER and etype not in seen:
            seen.append(etype)
    return seen[:limit]


def developing_stories(
    db: Session, counts: dict[int, int], vertical: str | None = None, limit: int = 8
) -> list[Story]:
    """Stories with two or more sourced developments, most recently updated first."""
    ids = [sid for sid, n in counts.items() if n >= 2]
    if not ids:
        return []
    query = select(Story).where(Story.id.in_(ids))
    if vertical:
        query = query.where(Story.vertical == vertical)
    return list(db.scalars(query.order_by(Story.last_updated_at.desc(), Story.id.desc()).limit(limit)))


def has_demo_content(db: Session) -> bool:
    return bool(db.scalar(select(func.count(Article.id)).where(Article.is_demo.is_(True))))


def story_intel_run(db: Session, story: Story) -> AgentRun | None:
    """The latest story-intelligence agent run for ``story`` (who drafted the summary, and when)."""
    return db.scalars(
        select(AgentRun)
        .where(
            AgentRun.agent == STORY_AGENT_NAME, AgentRun.subject_type == "story", AgentRun.subject_id == story.id
        )
        .order_by(AgentRun.id.desc())
        .limit(1)
    ).first()


def run_engine(run: AgentRun | None) -> str:
    """``generated_by`` label for an agent run: "offline", or the provider/model that answered."""
    if run is None or run.provider == "offline" or run.status == AgentRunStatus.FALLBACK:
        return "offline"
    return f"{run.provider}:{run.model}" if run.model else run.provider


def tool_tabs(brief: Brief, audience: str) -> list[dict[str, str]]:
    tabs = []
    for key, label in TOOL_TABS:
        url = f"/briefs/{brief.id}/tools/{key}"
        if key == "learn":
            url += f"?audience={audience}"
        tabs.append({"key": key, "label": label, "url": url})
    return tabs


def search_briefs(db: Session, query: str, limit: int = SEARCH_BRIEF_LIMIT) -> list[Brief]:
    """Approved briefs whose headline or summary match the query's terms (headline hits weigh more)."""
    terms = query_terms(query)[:MAX_SEARCH_TERMS]
    if not terms:
        return []
    conditions = []
    for term in terms:
        literal = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{literal}%"
        conditions += [Brief.headline.ilike(pattern, escape="\\"), Brief.summary.ilike(pattern, escape="\\")]
    candidates = db.scalars(
        select(Brief)
        .where(Brief.status == BriefStatus.APPROVED, or_(*conditions))
        .order_by(Brief.published_at.desc().nulls_last(), Brief.id.desc())
        .limit(BRIEF_SEARCH_CANDIDATES)
    ).all()
    scored: list[tuple[int, int, Brief]] = []
    for position, brief in enumerate(candidates):
        headline = brief.headline.lower()
        summary = (brief.summary or "").lower()
        matched = [t for t in terms if t in headline or t in summary]
        score = sum(3 for t in terms if t in headline) + sum(1 for t in terms if t in summary)
        # Multi-word queries: require at least half the terms so one common word does not match everything.
        if len(matched) * 2 >= len(terms):
            scored.append((score, -position, brief))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [brief for _, _, brief in scored[:limit]]


def page(request: Request, template: str, ctx: dict[str, Any], status_code: int = 200):
    """Render a public page with the shared helpers its templates use."""
    ctx.setdefault("FINANCE_DISCLAIMER", FINANCE_DISCLAIMER)
    ctx.setdefault("plain", strip_citation_marks)
    ctx.setdefault("event_label", event_label)
    ctx.setdefault("doc_type_label", doc_type_label)
    ctx.setdefault("trust_tier_label", trust_tier_label)
    ctx.setdefault("is_opinion_brief", is_opinion_brief)
    ctx.setdefault("story_status_label", lambda s: STORY_STATUS_LABELS.get(s or "", (s or "").capitalize()))
    return render(request, template, ctx, status_code=status_code)


# --------------------------------------------------------------------------
# Front page
# --------------------------------------------------------------------------


@router.get("/")
def home(request: Request, db: DB):
    recent = published_briefs(db, limit=60)
    news = [b for b in recent if not is_opinion_brief(b)]
    lead: Brief | None = None
    if news or recent:
        pool = (news or recent)[:10]
        lead = max(pool, key=lambda b: (len(b.citations or []), b.published_at or b.generated_at, b.id))

    sections = []
    for vertical in SECTION_VERTICALS:
        briefs = [b for b in published_briefs(db, vertical=vertical, limit=12) if lead is None or b.id != lead.id]
        # News reports before columns; each keeps newest-first order.
        briefs.sort(key=lambda b: is_opinion_brief(b))
        sections.append(
            {
                "vertical": vertical,
                "intro": VERTICAL_INTROS[vertical],
                "briefs": briefs[:4],
                "url": "/finance" if vertical == FINANCE else f"/v/{vertical}",
            }
        )

    counts = member_counts(db)
    stories = developing_stories(db, counts, limit=8)
    return page(
        request,
        "public/home.html",
        {
            "lead": lead,
            "sections": sections,
            "stories": stories,
            "counts": counts,
            "story_types": {s.id: story_event_types(s) for s in stories},
            "prompts": HOME_PROMPTS,
            "demo": has_demo_content(db),
        },
    )


# --------------------------------------------------------------------------
# Vertical sections
# --------------------------------------------------------------------------


@router.get("/v/{vertical}")
def vertical_page(request: Request, db: DB, vertical: str):
    if vertical == FINANCE:
        return RedirectResponse("/finance", status_code=303)
    if vertical not in PAGE_VERTICALS:
        raise HTTPException(status_code=404, detail="We cover Technology, Education, Industry, and Finance.")
    briefs = published_briefs(db, vertical=vertical, limit=24)
    briefs.sort(key=lambda b: is_opinion_brief(b))
    counts = member_counts(db)
    stories = list(
        db.scalars(
            select(Story)
            .where(Story.vertical == vertical, Story.status != "dormant")
            .order_by(Story.last_updated_at.desc(), Story.id.desc())
            .limit(30)
        )
    )
    stories = [s for s in stories if counts.get(s.id, 0) >= 1]
    # Multi-source storylines first (they have a timeline worth previewing), then single reports.
    stories.sort(key=lambda s: counts.get(s.id, 0) < 2)
    stories = stories[:6]
    products = [spec for spec in PRODUCTS.values() if spec.vertical == vertical]
    return page(
        request,
        "public/vertical.html",
        {
            "vertical": vertical,
            "intro": VERTICAL_INTROS[vertical],
            "lead": briefs[0] if briefs else None,
            "briefs": briefs[1:13],
            "stories": stories,
            "counts": counts,
            "previews": {s.id: list(s.events)[-3:] for s in stories},
            "products": products,
            "active_vertical": vertical,
            "demo": has_demo_content(db),
        },
    )


# --------------------------------------------------------------------------
# AI Story Brief
# --------------------------------------------------------------------------


@router.get("/briefs/{brief_id}")
def brief_page(request: Request, db: DB, user: CurrentUser, audience: ReaderAudience, brief_id: int):
    brief = load_brief(db, brief_id, user)
    corrections = list(
        db.scalars(
            select(Correction)
            .where(Correction.brief_id == brief.id)
            .order_by(Correction.created_at.desc(), Correction.id.desc())
        )
    )
    checks = brief.check_results or []
    record_view(db, user, request.url.path, story_id=brief.story_id, brief_id=brief.id)
    ctx = {
        "brief": brief,
        "story": brief.story,
        "is_finance": brief.vertical == FINANCE,
        "preview": brief.status != BriefStatus.APPROVED,
        "corrections": corrections,
        "checks_passed": sum(1 for c in checks if c.get("passed")),
        "checks_total": len(checks),
        "products": available_products(brief),
        "active_vertical": brief.vertical,
    }
    ctx["tabs"] = tool_tabs(brief, audience)
    return page(request, "public/brief.html", ctx)


# --------------------------------------------------------------------------
# Stories
# --------------------------------------------------------------------------


@router.get("/stories/{story_id}")
def story_page(request: Request, db: DB, user: CurrentUser, story_id: int):
    story = db.get(Story, story_id)
    if story is None:
        raise HTTPException(status_code=404, detail="No story with that id.")
    approved = approved_brief_for_story(db, story.id)
    latest = latest_brief_for_story(db, story.id)
    show_intel = approved is not None or is_editor(user)
    intel_citations = [c.model_dump() for c in story_citations(story)] if show_intel else []
    run = story_intel_run(db, story) if show_intel else None

    members = member_articles(story)
    by_doc_type: dict[str, list[Article]] = defaultdict(list)
    for article in sorted(members, key=lambda a: a.display_time, reverse=True):
        by_doc_type[article.doc_type or DocType.NEWS].append(article)
    grouped = sorted(
        by_doc_type.items(),
        key=lambda kv: DOC_TYPE_ORDER.index(kv[0]) if kv[0] in DOC_TYPE_ORDER else len(DOC_TYPE_ORDER),
    )

    related = related_stories(db, story)
    counts = member_counts(db)
    record_view(db, user, request.url.path, story_id=story.id)
    return page(
        request,
        "public/story.html",
        {
            "story": story,
            "events": list(story.events),
            "approved_brief": approved,
            "latest_brief": latest,
            "show_intel": show_intel,
            "intel_preview": show_intel and approved is None,
            "intel_citations": intel_citations,
            "intel_run": run,
            "intel_engine": run_engine(run),
            "members": members,
            "grouped": grouped,
            "related": related,
            "counts": counts,
            "is_finance": story.vertical == FINANCE,
            "active_vertical": story.vertical,
        },
    )


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


@router.get("/search")
def search(request: Request, db: DB, q: str = ""):
    query = q.strip()[:200]
    briefs: list[Brief] = []
    articles: list[Article] = []
    if query:
        briefs = search_briefs(db, query)
        articles = archive_search(db, query, limit=SEARCH_ARTICLE_LIMIT)
    return page(request, "public/search.html", {"q": query, "briefs": briefs, "articles": articles})


# --------------------------------------------------------------------------
# Sources and About
# --------------------------------------------------------------------------


@router.get("/sources")
def sources_page(request: Request, db: DB):
    rows = sorted(db.scalars(select(Source).where(Source.approved.is_(True))), key=lambda s: s.name.casefold())
    grouped = {v: [s for s in rows if s.vertical == v] for v in SECTION_VERTICALS}
    other = [s for s in rows if s.vertical not in grouped]
    if other:
        grouped["other"] = other
    tiers: dict[str, int] = defaultdict(int)
    for source in rows:
        tiers[source.trust_tier] += 1
    article_counts = dict(
        db.execute(
            select(Article.source_id, func.count(Article.id))
            .where(Article.duplicate_of_id.is_(None))
            .group_by(Article.source_id)
        ).all()
    )
    return page(
        request,
        "public/sources.html",
        {
            "grouped": grouped,
            "total": len(rows),
            "tiers": dict(tiers),
            "article_counts": article_counts,
            "demo_count": sum(1 for s in rows if s.kind == "demo"),
        },
    )


@router.get("/about")
def about_page(request: Request, db: DB):
    settings = get_settings()
    engines = dict(db.execute(select(Brief.generated_by, func.count(Brief.id)).group_by(Brief.generated_by)).all())
    demo_articles = db.scalar(select(func.count(Article.id)).where(Article.is_demo.is_(True))) or 0
    live_articles = db.scalar(select(func.count(Article.id)).where(Article.is_demo.is_(False))) or 0
    if settings.llm_mode == "offline":
        mode = "offline"
    elif settings.llm_mode == "claude" or claude_credentials_available():
        mode = "claude"
    else:
        mode = "offline"
    return page(
        request,
        "public/about.html",
        {
            "engines": engines,
            "claude_briefs": sum(n for k, n in engines.items() if k and k != "offline"),
            "offline_briefs": engines.get("offline", 0),
            "mode": mode,
            "model": settings.model,
            "demo_articles": demo_articles,
            "live_articles": live_articles,
            "products": list(PRODUCTS.values()),
        },
    )
