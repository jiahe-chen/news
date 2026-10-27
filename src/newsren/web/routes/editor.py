"""Editor desk: the editorial quality gate UI (plan §6, "Editorial quality gate").

AI agents research and draft; nothing reaches readers until it passes the
automated checks and, where the rules require it, a human editor approves it
("Require human approval for publishing initially"; finance is *always*
reviewed, plan §4). This module is the editor's side of that gate:

* ``GET  /editor`` — the review queue (risk, failed checks, why a human must
  look), plus recently published and rejected briefs;
* ``GET  /editor/briefs/{id}`` — one brief under review: content, automated
  checks, claim-by-claim source sampling (feeds the Trust metric), approve /
  reject / correct / regenerate, the audit trail and the agent run behind it;
* ``POST /editor/claims/{claim_id}/verdict`` — Accurate / Inaccurate;
* ``POST /editor/briefs/{id}/approve|reject|corrections|regenerate``;
* ``GET  /editor/sources`` + ``POST /editor/sources/{id}/toggle`` — the approved
  source registry (agents can only read approved, active sources);
* ``GET  /editor/runs`` (+ ``/editor/runs/{id}``) — the orchestrator's AgentRun log;
* ``GET|POST /editor/pipeline`` — run the editorially controlled workflow;
* ``GET  /editor/funding`` + ``POST /editor/funding/{id}/verify`` — editor
  checks of auto-extracted funding rounds.

Every route requires an editor (anonymous readers are sent to sign in, other
readers get 403). Every POST is CSRF-protected and follows Post/Redirect/Get;
claim verdicts and source toggles also answer htmx requests with a fragment.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from markupsafe import Markup, escape
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsren.config import get_settings
from newsren.editorial.checks import BLOCKER, INFO, WARNING, run_checks
from newsren.editorial.guardrails import FINANCE_DISCLAIMER
from newsren.editorial.workflow import (
    OVERRIDE_PREFIX,
    approve,
    editorial_history,
    published_briefs,
    record_correction,
    reject,
    review_queue,
    review_reasons,
    set_claim_verdict,
    start_review,
)
from newsren.finance.funding import STAGE_LABELS
from newsren.models import (
    AgentRun,
    AgentRunStatus,
    Article,
    Brief,
    BriefClaim,
    BriefStatus,
    Correction,
    EarningsReport,
    FundingRound,
    Source,
    User,
    Vertical,
    utcnow,
)
from newsren.schemas import CheckResult
from newsren.web.deps import CSRF, DB, EditorUser, csrf_token
from newsren.web.routes.auth import safe_next
from newsren.web.templating import flash, render, templates

router = APIRouter()

VERTICAL_SLUGS = [v.value for v in (Vertical.TECHNOLOGY, Vertical.EDUCATION, Vertical.INDUSTRY, Vertical.FINANCE)]

# Desk navigation: (key, label, url).
DESK_NAV = [
    ("queue", "Review queue", "/editor"),
    ("funding", "Funding rounds", "/editor/funding"),
    ("sources", "Sources", "/editor/sources"),
    ("runs", "Agent runs", "/editor/runs"),
    ("pipeline", "Pipeline", "/editor/pipeline"),
    ("analytics", "Analytics", "/analytics"),
]

SEVERITY_ORDER = {BLOCKER: 0, WARNING: 1, INFO: 2}
TOGGLE_FIELDS = {"approved": "Approved", "active": "Active"}

RUN_STATUS_LABELS = {
    AgentRunStatus.RUNNING.value: "Running",
    AgentRunStatus.OK.value: "OK",
    AgentRunStatus.ERROR.value: "Error",
    AgentRunStatus.BUDGET_EXCEEDED.value: "Budget exceeded",
    AgentRunStatus.REFUSED.value: "Refused",
    AgentRunStatus.FALLBACK.value: "Fallback (offline engine answered)",
}
RUN_STATUS_HELP = {
    AgentRunStatus.OK.value: "The agent finished within its budget.",
    AgentRunStatus.FALLBACK.value: "Claude was unavailable or returned invalid output; the offline evidence engine answered.",
    AgentRunStatus.REFUSED.value: "The model declined; the offline engine answered and the refusal is recorded.",
    AgentRunStatus.BUDGET_EXCEEDED.value: "The orchestrator stopped the run at its tool-call, time, or cost limit.",
    AgentRunStatus.ERROR.value: "The run failed; see the error.",
    AgentRunStatus.RUNNING.value: "Started and not yet finished.",
}

RUNS_DEFAULT_LIMIT = 50
RUNS_MAX_LIMIT = 200
PIPELINE_SESSION_KEY = "editor_pipeline_report"
PIPELINE_MAX_BRIEFS = 60
PIPELINE_MAX_ERRORS = 8

# What each pipeline step does (plan §1 workflow, §6 components), shown on the pipeline page.
PIPELINE_STEPS = [
    {
        "name": "Discover",
        "component": "Source Discovery",
        "text": (
            "Sync the approved source registry, then fetch new items from approved, active sources only "
            "(RSS feeds, arXiv, SEC EDGAR, ERIC). Items are normalized, deduplicated (exact URL, content hash, "
            "near-duplicate SimHash) and indexed. Fetching needs network access; with it off, this step only "
            "syncs the registry."
        ),
    },
    {
        "name": "Connect",
        "component": "Story Intelligence",
        "text": (
            "Tag entities, cluster new articles into evolving stories, rebuild each changed story's timeline, "
            "and refresh what happened first, what's new, what is disputed, and how the evidence changed."
        ),
    },
    {
        "name": "Explain",
        "component": "News Explanation",
        "text": (
            "Draft an AI Story Brief for each story whose sources changed (up to the limit below), from an "
            "evidence pack that puts primary and independent sources first. Every claim cites its sources."
        ),
    },
    {
        "name": "Verify",
        "component": "Editorial quality gate",
        "text": (
            "Run the automated checks on every new brief. Briefs go to this desk's review queue; only "
            "low-risk, non-finance briefs with no blockers may publish automatically, and only when human "
            "approval is not required by configuration."
        ),
    },
    {
        "name": "Finance trackers",
        "component": "Finance section",
        "text": (
            "Log funding rounds from public announcements, classify AI-in-finance claims (announcement, pilot, "
            "backtest, live deployment, regulatory finding), and, with network access, refresh SEC EDGAR "
            "earnings data. Finance briefs always wait for an editor."
        ),
    },
    {
        "name": "Output",
        "component": "Output",
        "text": (
            "Check readers' Follow This Story requests and create alerts only when their criteria are met; "
            "build low-risk personal briefings from published briefs and the reader's own alerts."
        ),
    },
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def is_htmx(request: Request) -> bool:
    return (
        request.headers.get("HX-Request") == "true" and request.headers.get("HX-History-Restore-Request") != "true"
    )


CITE_MARK = re.compile(r"\s*\[(c\d+)\]")


def cited(text: str | None, citations: list[Any] | None) -> Markup:
    """Escape ``text`` and turn its ``[c1]`` markers into links to the matching source entries.

    Markers that point at no citation in the brief are kept visible (flagged) so
    the editor can see an unknown-citation problem instead of having it hidden.
    """
    known = {str(c.get("id")): c for c in citations or [] if isinstance(c, dict)}
    text = text or ""
    out: list[str] = []
    pos = 0
    for match in CITE_MARK.finditer(text):
        out.append(str(escape(text[pos : match.start()])))
        cid = match.group(1)
        cit = known.get(cid)
        if cit is not None:
            title = f"{cit.get('title', '')} — {cit.get('publisher', '')}".strip(" —")
            out.append(
                f'<a class="cite" href="#cite-{escape(cid)}" title="{escape(title)}">[{escape(cid[1:])}]</a>'
            )
        else:
            out.append(f'<span class="cite cite-unknown" title="No source with this id">[{escape(cid)}?]</span>')
        pos = match.end()
    out.append(str(escape(text[pos:])))
    return Markup("".join(out))


def fmt_duration(seconds: float | None) -> str:
    """Human review / run durations: "45s", "3m 12s", "1h 05m"."""
    if seconds is None:
        return ""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


ACRONYMS = {"ai": "AI", "edgar": "EDGAR", "usd": "USD", "url": "URL", "llm": "LLM", "id": "ID"}


def humanize(key: str) -> str:
    """``ai_claims_seeded`` -> "AI claims seeded" (labels for report and budget keys)."""
    words = str(key).replace("_", " ").split()
    words = [ACRONYMS.get(w.lower(), w) for w in words]
    if words and words[0] not in ACRONYMS.values():
        words[0] = words[0].capitalize()
    return " ".join(words)


def helpers(ctx: dict[str, Any]) -> dict[str, Any]:
    ctx.setdefault("FINANCE_DISCLAIMER", FINANCE_DISCLAIMER)
    ctx.setdefault("run_status_label", run_status_label)
    ctx.setdefault("cited", cited)
    ctx.setdefault("fmt_duration", fmt_duration)
    ctx.setdefault("humanize", humanize)
    return ctx


def desk(request: Request, template: str, ctx: dict[str, Any], active: str, status_code: int = 200) -> Response:
    """Render a full editor-desk page with the desk navigation."""
    ctx.setdefault("desk_nav", DESK_NAV)
    ctx.setdefault("desk_active", active)
    return render(request, template, helpers(ctx), status_code=status_code)


def fragment(request: Request, template: str, ctx: dict[str, Any], status_code: int = 200) -> Response:
    """Render an htmx fragment (no page chrome; the session's flash message is left alone)."""
    ctx = dict(ctx)
    ctx.setdefault("current_user", getattr(request.state, "user", None))
    ctx.setdefault("csrf_token", csrf_token(request))
    ctx.setdefault("flash", None)
    response = templates.TemplateResponse(request, template, helpers(ctx), status_code=status_code)
    response.headers["Vary"] = "HX-Request"
    return response


def run_status_label(status: str | None) -> str:
    return RUN_STATUS_LABELS.get(status or "", (status or "unknown").replace("_", " ").capitalize())


def get_brief(db: Session, brief_id: int) -> Brief:
    brief = db.get(Brief, brief_id)
    if brief is None:
        raise HTTPException(status_code=404, detail="No brief with that id.")
    return brief


def review_url(brief: Brief | int, anchor: str = "") -> str:
    brief_id = brief if isinstance(brief, int) else brief.id
    return f"/editor/briefs/{brief_id}" + (f"#{anchor}" if anchor else "")


def redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


def parse_checks(raw: list[Any] | None) -> list[CheckResult]:
    """Stored check results as ``CheckResult`` models (malformed rows are skipped, not fatal)."""
    out: list[CheckResult] = []
    for item in raw or []:
        try:
            out.append(item if isinstance(item, CheckResult) else CheckResult.model_validate(item))
        except ValidationError:
            continue
    return out


def failed(checks: list[CheckResult], severity: str) -> list[CheckResult]:
    return [c for c in checks if not c.passed and c.severity == severity]


def ordered_checks(checks: list[CheckResult]) -> list[CheckResult]:
    """Failed checks first (blockers, then warnings, then info), then the passed ones, stable otherwise."""
    return sorted(checks, key=lambda c: (c.passed, SEVERITY_ORDER.get(c.severity, 3)))


def check_summary(checks: list[CheckResult]) -> dict[str, int]:
    return {
        "total": len(checks),
        "passed": sum(1 for c in checks if c.passed),
        "blockers": len(failed(checks, BLOCKER)),
        "warnings": len(failed(checks, WARNING)),
        "info": len(failed(checks, INFO)),
    }


def user_names(db: Session, ids: set[int | None]) -> dict[int, str]:
    wanted = {i for i in ids if i is not None}
    if not wanted:
        return {}
    return {u.id: u.display_name for u in db.scalars(select(User).where(User.id.in_(wanted)))}


def citation_map(brief: Brief) -> dict[str, dict[str, Any]]:
    return {str(c.get("id")): c for c in brief.citations or [] if isinstance(c, dict) and c.get("id")}


def earnings_report_for(db: Session, brief: Brief) -> EarningsReport | None:
    if brief.kind != "earnings":
        return None
    return db.scalars(select(EarningsReport).where(EarningsReport.brief_id == brief.id).limit(1)).first()


def can_regenerate(db: Session, brief: Brief) -> bool:
    if brief.kind == "story":
        return brief.story is not None
    return earnings_report_for(db, brief) is not None


def queue_item(brief: Brief, settings, names: dict[int, str]) -> dict[str, Any]:
    checks = parse_checks(brief.check_results)
    return {
        "brief": brief,
        "checks": check_summary(checks),
        "blocker_names": [c.name for c in failed(checks, BLOCKER)],
        "reasons": review_reasons(brief, checks, settings),
        "reviewer": names.get(brief.reviewer_id) if brief.review_started_at else None,
        "claims": len(brief.claims or []),
        "sources": len(brief.citations or []),
    }


# --------------------------------------------------------------------------
# Review queue
# --------------------------------------------------------------------------


@router.get("/editor")
def queue_page(request: Request, db: DB, editor: EditorUser, vertical: str | None = None):
    settings = get_settings()
    vertical = vertical if vertical in VERTICAL_SLUGS else None
    queue = review_queue(db)
    names = user_names(db, {b.reviewer_id for b in queue})
    items = [queue_item(b, settings, names) for b in queue]
    by_vertical = Counter(b.vertical for b in queue)
    shown = [i for i in items if vertical is None or i["brief"].vertical == vertical]

    published = published_briefs(db, limit=12)
    rejected = list(
        db.scalars(
            select(Brief)
            .where(Brief.status == BriefStatus.REJECTED, Brief.reviewer_id.is_not(None))
            .order_by(Brief.reviewed_at.desc().nulls_last(), Brief.id.desc())
            .limit(6)
        )
    )
    names.update(user_names(db, {b.reviewer_id for b in published + rejected}))
    week_ago = utcnow() - timedelta(days=7)
    stats = {
        "awaiting": len(queue),
        "finance": by_vertical.get(Vertical.FINANCE.value, 0),
        "high_risk": sum(1 for b in queue if b.risk_level == "high"),
        "with_blockers": sum(1 for i in items if i["checks"]["blockers"]),
        "published_total": int(
            db.scalar(select(func.count(Brief.id)).where(Brief.status == BriefStatus.APPROVED)) or 0
        ),
        "published_week": int(
            db.scalar(
                select(func.count(Brief.id)).where(
                    Brief.status == BriefStatus.APPROVED, Brief.published_at >= week_ago
                )
            )
            or 0
        ),
        "unverified_rounds": int(
            db.scalar(select(func.count(FundingRound.id)).where(FundingRound.verified.is_(False))) or 0
        ),
    }
    return desk(
        request,
        "editor/queue.html",
        {
            "items": shown,
            "stats": stats,
            "by_vertical": by_vertical,
            "vertical": vertical,
            "published": published,
            "rejected": rejected,
            "names": names,
            "require_human_approval": settings.require_human_approval,
            "override_prefix": OVERRIDE_PREFIX,
        },
        "queue",
    )


# --------------------------------------------------------------------------
# Review one brief
# --------------------------------------------------------------------------


@router.get("/editor/briefs/{brief_id}")
def review_page(request: Request, db: DB, editor: EditorUser, brief_id: int):
    brief = get_brief(db, brief_id)
    if brief.status == BriefStatus.IN_REVIEW and brief.review_started_at is None:
        start_review(db, brief, editor)

    settings = get_settings()
    # Re-run the deterministic, read-only checks so the editor sees exactly what approval will evaluate.
    live_checks = run_checks(db, brief)
    stored_summary = check_summary(parse_checks(brief.check_results))
    history = editorial_history(db, brief)
    corrections = list(
        db.scalars(
            select(Correction)
            .where(Correction.brief_id == brief.id)
            .order_by(Correction.created_at.desc(), Correction.id.desc())
        )
    )
    names = user_names(db, {a.actor_id for a in history} | {c.actor_id for c in corrections} | {brief.reviewer_id})
    run = db.get(AgentRun, brief.agent_run_id) if brief.agent_run_id else None
    claims = list(brief.claims or [])
    judged = [c for c in claims if c.editor_verdict]
    return desk(
        request,
        "editor/review.html",
        {
            "brief": brief,
            "story": brief.story,
            "is_finance": brief.vertical == Vertical.FINANCE.value,
            "checks": ordered_checks(live_checks),
            "summary": check_summary(live_checks),
            "stored_summary": stored_summary,
            "blockers": failed(live_checks, BLOCKER),
            "reasons": review_reasons(brief, live_checks, settings),
            "claims": claims,
            "claims_judged": len(judged),
            "claims_accurate": sum(1 for c in judged if c.editor_verdict == "accurate"),
            "cites": brief.citations or [],
            "cite_map": citation_map(brief),
            "history": history,
            "corrections": corrections,
            "names": names,
            "run": run,
            "can_regenerate": can_regenerate(db, brief),
            "override_prefix": OVERRIDE_PREFIX,
            "unpublished": brief.status != BriefStatus.APPROVED,
        },
        "queue",
    )


@router.post("/editor/claims/{claim_id}/verdict", dependencies=[CSRF])
def claim_verdict(
    request: Request, db: DB, editor: EditorUser, claim_id: int, verdict: Annotated[str, Form()] = ""
):
    claim = db.get(BriefClaim, claim_id)
    if claim is None:
        raise HTTPException(status_code=404, detail="No claim with that id.")
    brief = claim.brief
    error = None
    try:
        set_claim_verdict(db, claim, editor, verdict)
    except ValueError as exc:
        error = str(exc)
    if is_htmx(request):
        ctx = {"claim": claim, "brief": brief, "cite_map": citation_map(brief), "claim_error": error}
        # 200 even on a validation error: htmx only swaps successful responses, and the claim's
        # fragment carries the error message in place.
        return fragment(request, "editor/_claim.html", ctx)
    if error:
        flash(request, error, "error")
    else:
        flash(
            request, f"Claim recorded as {claim.editor_verdict}. It now counts toward the Trust metric.", "success"
        )
    return redirect(review_url(brief, f"claim-{claim.id}"))


@router.post("/editor/briefs/{brief_id}/approve", dependencies=[CSRF])
def approve_brief(request: Request, db: DB, editor: EditorUser, brief_id: int, notes: Annotated[str, Form()] = ""):
    brief = get_brief(db, brief_id)
    if brief.status == BriefStatus.APPROVED:
        flash(request, "This brief is already published.", "info")
        return redirect(review_url(brief))
    if brief.status == BriefStatus.REJECTED:
        flash(
            request,
            "This brief was rejected or superseded. Regenerate it to send a fresh draft through the quality gate.",
            "error",
        )
        return redirect(review_url(brief))
    try:
        approve(db, brief, editor, notes)
    except ValueError as exc:
        flash(request, str(exc), "error")
        return redirect(review_url(brief, "decision"))
    flash(request, f"Approved and published: “{brief.headline}”. Readers can now see it.", "success")
    return redirect("/editor")


@router.post("/editor/briefs/{brief_id}/reject", dependencies=[CSRF])
def reject_brief(request: Request, db: DB, editor: EditorUser, brief_id: int, notes: Annotated[str, Form()] = ""):
    brief = get_brief(db, brief_id)
    if brief.status == BriefStatus.APPROVED:
        flash(request, "Published briefs are corrected, not rejected. Record a correction instead.", "error")
        return redirect(review_url(brief, "decision"))
    try:
        reject(db, brief, editor, notes)
    except ValueError as exc:
        flash(
            request, f"{exc} Notes are required so the agent's next draft and the audit trail record why.", "error"
        )
        return redirect(review_url(brief, "decision"))
    flash(request, f"Rejected: “{brief.headline}”. It will not be published.", "success")
    return redirect("/editor")


@router.post("/editor/briefs/{brief_id}/corrections", dependencies=[CSRF])
def add_correction(
    request: Request,
    db: DB,
    editor: EditorUser,
    brief_id: int,
    description: Annotated[str, Form()] = "",
    material: Annotated[str | None, Form()] = None,
):
    brief = get_brief(db, brief_id)
    if brief.status != BriefStatus.APPROVED:
        flash(request, "Corrections are for published briefs. Reject or regenerate an unpublished draft.", "error")
        return redirect(review_url(brief, "decision"))
    is_material = material not in (None, "", "0", "false", "off")
    try:
        record_correction(db, brief, editor, description, material=is_material)
    except ValueError as exc:
        flash(request, str(exc), "error")
        return redirect(review_url(brief, "decision"))
    kind = "Material correction" if is_material else "Correction"
    flash(
        request, f"{kind} recorded; the brief is now version {brief.version} and readers see the note.", "success"
    )
    return redirect(review_url(brief, "history"))


@router.post("/editor/briefs/{brief_id}/regenerate", dependencies=[CSRF])
def regenerate_brief(request: Request, db: DB, editor: EditorUser, brief_id: int):
    brief = get_brief(db, brief_id)
    new_brief: Brief | None = None
    try:
        if brief.kind == "story" and brief.story is not None:
            from newsren.agents.brief import generate_brief

            new_brief = generate_brief(db, brief.story, regenerate=True)
        else:
            report = earnings_report_for(db, brief)
            if report is not None:
                from newsren.finance.edgar import earnings_brief

                new_brief = earnings_brief(db, report, regenerate=True)
    except ValueError as exc:
        flash(request, f"Could not regenerate: {exc}", "error")
        return redirect(review_url(brief))
    if new_brief is None:
        flash(request, "This brief has no story or report to regenerate from.", "error")
        return redirect(review_url(brief))
    if new_brief.id == brief.id:
        flash(request, "The agent returned the current brief; nothing changed.", "info")
    else:
        flash(
            request,
            f"Regenerated as brief #{new_brief.id} and sent through the quality gate"
            + (f"; brief #{brief.id} was superseded." if brief.status == BriefStatus.REJECTED else "."),
            "success",
        )
    return redirect(review_url(new_brief))


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def source_rows(db: Session) -> tuple[list[Source], dict[int, int]]:
    sources = list(db.scalars(select(Source)))
    order = {v: i for i, v in enumerate(VERTICAL_SLUGS)}
    sources.sort(key=lambda s: (order.get(s.vertical, len(order)), s.name.casefold()))
    counts = dict(
        db.execute(
            select(Article.source_id, func.count(Article.id))
            .where(Article.duplicate_of_id.is_(None))
            .group_by(Article.source_id)
        ).all()
    )
    return sources, {int(k): int(v) for k, v in counts.items()}


@router.get("/editor/sources")
def sources_page(request: Request, db: DB, editor: EditorUser):
    sources, counts = source_rows(db)
    stats = {
        "total": len(sources),
        "approved": sum(1 for s in sources if s.approved),
        "usable": sum(1 for s in sources if s.approved and s.active),
        "failing": sum(1 for s in sources if s.last_status and not s.last_status.lower().startswith("ok")),
    }
    return desk(request, "editor/sources.html", {"sources": sources, "counts": counts, "stats": stats}, "sources")


@router.post("/editor/sources/{source_id}/toggle", dependencies=[CSRF])
def toggle_source(
    request: Request, db: DB, editor: EditorUser, source_id: int, field: Annotated[str, Form()] = ""
):
    source = db.get(Source, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="No source with that id.")
    if field not in TOGGLE_FIELDS:
        raise HTTPException(status_code=400, detail="Choose a field to toggle: approved or active.")
    setattr(source, field, not bool(getattr(source, field)))
    db.flush()
    state = "on" if getattr(source, field) else "off"
    message = f"{TOGGLE_FIELDS[field]} is now {state} for {source.name}."
    if is_htmx(request):
        counts = {
            source.id: int(
                db.scalar(
                    select(func.count(Article.id)).where(
                        Article.source_id == source.id, Article.duplicate_of_id.is_(None)
                    )
                )
                or 0
            )
        }
        return fragment(request, "editor/_source_row.html", {"s": source, "counts": counts, "message": message})
    flash(request, message, "success")
    return redirect(f"/editor/sources#source-{source.id}")


# --------------------------------------------------------------------------
# Agent runs
# --------------------------------------------------------------------------


@router.get("/editor/runs")
def runs_page(
    request: Request,
    db: DB,
    editor: EditorUser,
    agent: str | None = None,
    status: str | None = None,
    limit: int = RUNS_DEFAULT_LIMIT,
    offset: int = 0,
):
    limit = max(1, min(limit, RUNS_MAX_LIMIT))
    offset = max(0, offset)
    agent = agent or None
    status = status or None
    query = select(AgentRun)
    count_query = select(func.count(AgentRun.id))
    if agent:
        query = query.where(AgentRun.agent == agent)
        count_query = count_query.where(AgentRun.agent == agent)
    if status:
        query = query.where(AgentRun.status == status)
        count_query = count_query.where(AgentRun.status == status)
    total = int(db.scalar(count_query) or 0)
    runs = list(
        db.scalars(query.order_by(AgentRun.started_at.desc(), AgentRun.id.desc()).limit(limit).offset(offset))
    )
    agents = sorted(a for a in db.scalars(select(AgentRun.agent).distinct()) if a)
    status_counts = dict(
        db.execute(select(AgentRun.status, func.count(AgentRun.id)).group_by(AgentRun.status)).all()
    )
    totals = db.execute(select(func.count(AgentRun.id), func.coalesce(func.sum(AgentRun.cost_usd), 0.0))).one()

    def page_url(new_offset: int) -> str:
        params = [f"limit={limit}", f"offset={new_offset}"]
        if agent:
            params.insert(0, f"agent={agent}")
        if status:
            params.insert(1 if agent else 0, f"status={status}")
        return "/editor/runs?" + "&".join(params)

    return desk(
        request,
        "editor/runs.html",
        {
            "runs": runs,
            "agents": agents,
            "statuses": list(RUN_STATUS_LABELS),
            "status_counts": status_counts,
            "status_help": RUN_STATUS_HELP,
            "agent": agent,
            "status": status,
            "limit": limit,
            "offset": offset,
            "total": total,
            "all_runs": int(totals[0] or 0),
            "all_cost": float(totals[1] or 0.0),
            "prev_url": page_url(max(0, offset - limit)) if offset > 0 else None,
            "next_url": page_url(offset + limit) if offset + limit < total else None,
        },
        "runs",
    )


@router.get("/editor/runs/{run_id}")
def run_detail(request: Request, db: DB, editor: EditorUser, run_id: int):
    run = db.get(AgentRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="No agent run with that id.")
    briefs = list(db.scalars(select(Brief).where(Brief.agent_run_id == run.id)))
    return desk(
        request,
        "editor/run.html",
        {"run": run, "briefs": briefs, "status_help": RUN_STATUS_HELP},
        "runs",
    )


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def _trim_report(report: dict[str, Any]) -> dict[str, Any]:
    """Keep the stored report small enough for the signed session cookie."""
    errors = [str(e)[:200] for e in report.get("errors") or []]
    report = dict(report)
    report["errors"] = errors[:PIPELINE_MAX_ERRORS]
    report["errors_total"] = len(errors)
    for key in ("finance", "monitoring", "briefings"):
        block = report.get(key)
        if isinstance(block, dict):
            report[key] = {
                k: (v[:3] if isinstance(v, list) else v)
                for k, v in block.items()
                if isinstance(v, int | float | bool | str | list) or v is None
            }
    return report


@router.get("/editor/pipeline")
def pipeline_page(request: Request, db: DB, editor: EditorUser):
    stored = request.session.get(PIPELINE_SESSION_KEY)
    last_runs = list(
        db.scalars(select(AgentRun).order_by(AgentRun.started_at.desc(), AgentRun.id.desc()).limit(5))
    )
    return desk(
        request,
        "editor/pipeline.html",
        {
            "steps": PIPELINE_STEPS,
            "last": stored,
            "max_briefs_limit": PIPELINE_MAX_BRIEFS,
            "recent_runs": last_runs,
        },
        "pipeline",
    )


@router.post("/editor/pipeline", dependencies=[CSRF])
def pipeline_run(
    request: Request,
    db: DB,
    editor: EditorUser,
    network: Annotated[str | None, Form()] = None,
    max_briefs: Annotated[str, Form()] = "10",
):
    from newsren.pipeline import run_pipeline

    try:
        limit = int(max_briefs)
    except (TypeError, ValueError):
        limit = 10
    limit = max(0, min(limit, PIPELINE_MAX_BRIEFS))
    use_network = network not in (None, "", "0", "false", "off")
    started = utcnow()
    report = run_pipeline(db, network=use_network, max_briefs=limit)
    finished = utcnow()
    request.session[PIPELINE_SESSION_KEY] = {
        "report": _trim_report(report.as_dict()),
        "network": use_network,
        "max_briefs": limit,
        "started_at": started.isoformat(),
        "seconds": round((finished - started).total_seconds(), 2),
        "by": editor.display_name,
    }
    errors = len(report.errors)
    flash(
        request,
        f"Pipeline finished: {report.briefs_generated} brief(s) drafted, {report.briefs_in_review} sent to review"
        + (f", {errors} step error(s) recorded." if errors else "."),
        "warning" if errors else "success",
    )
    return redirect("/editor/pipeline#report")


# --------------------------------------------------------------------------
# Funding rounds (editor verification)
# --------------------------------------------------------------------------


@router.get("/editor/funding")
def funding_page(request: Request, db: DB, editor: EditorUser):
    rounds = list(
        db.scalars(
            select(FundingRound).order_by(
                FundingRound.verified.asc(),
                FundingRound.announced_at.desc().nulls_last(),
                FundingRound.id.desc(),
            )
        )
    )
    return desk(
        request,
        "editor/funding.html",
        {
            "rounds": rounds,
            "unverified": sum(1 for r in rounds if not r.verified),
            "stage_labels": STAGE_LABELS,
            "is_finance": True,
        },
        "funding",
    )


@router.post("/editor/funding/{round_id}/verify", dependencies=[CSRF])
def verify_funding(
    request: Request, db: DB, editor: EditorUser, round_id: int, next: Annotated[str, Form()] = "/editor/funding"
):
    row = db.get(FundingRound, round_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No funding round with that id.")
    row.verified = True
    db.flush()
    flash(
        request,
        f"Marked the {row.company_name} round as verified against its public announcement.",
        "success",
    )
    return redirect(safe_next(next, "/editor/funding"))
