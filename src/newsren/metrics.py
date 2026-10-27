"""Success metrics (plan §9), computed from the database.

Article volume and chatbot usage are deliberately not headline metrics. These
are:

* **Trust** - share of editor-sampled claims judged accurately supported by
  their linked sources; material corrections per 100 published briefs;
  citation coverage of fact and performance claims.
* **Usefulness** - reader ratings of whether an item answered their question
  and reduced time spent searching.
* **Education** - comprehension and transfer accuracy after reading a
  Newsren explanation versus an ordinary summary.
* **Engagement** - meaningful return visits to followed stories; useful versus
  unwanted alerts.
* **Editorial efficiency** - human review time and the estimated time saved
  per approved brief, net of that review.
* **Cost** - average inference and retrieval cost per story brief, question,
  and monitoring check, from ``AgentRun`` (offline runs cost $0).

Every function returns a plain JSON-serialisable dict, works on an empty
database (rates are ``None`` when their denominator is 0, counts are 0), and
reports the ``n`` behind each rate so small samples are visible.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import timedelta
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from newsren.models import (
    AgentRun,
    AgentRunStatus,
    Alert,
    Brief,
    BriefClaim,
    BriefStatus,
    ClaimType,
    Correction,
    EditorialAction,
    Feedback,
    Follow,
    PageView,
    QuizAttempt,
    utcnow,
)
from newsren.textutil import iso

DEFAULT_BASELINE_MANUAL_MINUTES = 45.0
# A view this soon after following is usually the page reload that follows the
# "Follow" click, not a return visit.
RETURN_VISIT_MIN_GAP_MINUTES = 30
CONDITIONS = ("newsren", "summary")
STORY_BRIEF_AGENT = "story_brief"
QUESTION_AGENT = "ask_newsren"
MONITOR_AGENT = "follow_monitor"


def _rate(numerator: float, denominator: float, digits: int = 4) -> float | None:
    return round(numerator / denominator, digits) if denominator else None


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(float(value), digits) if value is not None else None


def _mean(values: list[float], digits: int = 4) -> float | None:
    return round(statistics.fmean(values), digits) if values else None


def _median(values: list[float], digits: int = 4) -> float | None:
    return round(statistics.median(values), digits) if values else None


def _diff(a: float | None, b: float | None) -> float | None:
    return round(a - b, 4) if a is not None and b is not None else None


# --------------------------------------------------------------------------
# Trust
# --------------------------------------------------------------------------


def trust_metrics(session: Session) -> dict[str, Any]:
    """Sampled-claim accuracy, material corrections per 100 published briefs, and claim citation coverage."""
    verdicts = session.execute(
        select(Brief.vertical, BriefClaim.editor_verdict, func.count(BriefClaim.id))
        .join(Brief, Brief.id == BriefClaim.brief_id)
        .where(BriefClaim.editor_verdict.in_(("accurate", "inaccurate")))
        .group_by(Brief.vertical, BriefClaim.editor_verdict)
    ).all()
    by_vertical: dict[str, Counter[str]] = defaultdict(Counter)
    for vertical, verdict, count in verdicts:
        by_vertical[vertical or "unknown"][verdict] += count
    accurate = sum(c["accurate"] for c in by_vertical.values())
    inaccurate = sum(c["inaccurate"] for c in by_vertical.values())
    judged = accurate + inaccurate

    published = int(
        session.scalar(select(func.count(Brief.id)).where(Brief.status == BriefStatus.APPROVED.value)) or 0
    )
    correction_rows = session.execute(
        select(Correction.material, func.count(Correction.id))
        .join(Brief, Brief.id == Correction.brief_id)
        .where(Brief.status == BriefStatus.APPROVED.value)
        .group_by(Correction.material)
    ).all()
    material = sum(n for is_material, n in correction_rows if is_material)
    corrections_total = sum(n for _, n in correction_rows)

    claim_rows = session.execute(
        select(BriefClaim.citation_ids, Brief.status)
        .join(Brief, Brief.id == BriefClaim.brief_id)
        .where(BriefClaim.claim_type.in_((ClaimType.FACT.value, ClaimType.PERFORMANCE.value)))
    ).all()
    fact_claims = len(claim_rows)
    cited = sum(1 for ids, _ in claim_rows if ids)
    published_rows = [ids for ids, status in claim_rows if status == BriefStatus.APPROVED.value]
    published_cited = sum(1 for ids in published_rows if ids)

    return {
        "definition": (
            "Share of editor-sampled claims whose linked sources accurately support them; material "
            "corrections per 100 published briefs; share of fact and performance claims citing at least "
            "one source."
        ),
        "claims_judged": judged,
        "claims_accurate": accurate,
        "claims_inaccurate": inaccurate,
        "claim_accuracy_rate": _rate(accurate, judged),
        "accuracy_by_vertical": {
            vertical: {
                "judged": c["accurate"] + c["inaccurate"],
                "accurate": c["accurate"],
                "rate": _rate(c["accurate"], c["accurate"] + c["inaccurate"]),
            }
            for vertical, c in sorted(by_vertical.items())
        },
        "published_briefs": published,
        "corrections_total": corrections_total,
        "material_corrections": material,
        "material_corrections_per_100_published": _rate(material * 100, published, 2),
        "fact_claims": fact_claims,
        "fact_claims_cited": cited,
        "citation_coverage": _rate(cited, fact_claims),
        "published_fact_claims": len(published_rows),
        "published_citation_coverage": _rate(published_cited, len(published_rows)),
    }


# --------------------------------------------------------------------------
# Usefulness
# --------------------------------------------------------------------------


def _usefulness_block(rows: list[tuple[bool | None, bool | None, int | None]]) -> dict[str, Any]:
    answered = [a for a, _, _ in rows if a is not None]
    saved = [s for _, s, _ in rows if s is not None]
    ratings = [float(r) for _, _, r in rows if r is not None]
    return {
        "responses": len(rows),
        "answered_question_rate": _rate(sum(answered), len(answered)),
        "answered_question_n": len(answered),
        "saved_time_rate": _rate(sum(saved), len(saved)),
        "saved_time_n": len(saved),
        "mean_rating": _mean(ratings, 2),
        "rating_n": len(ratings),
    }


def usefulness_metrics(session: Session) -> dict[str, Any]:
    """Reader feedback: answered-question rate, saved-time rate, and mean 1-5 rating, overall and by item type."""
    rows = session.execute(
        select(Feedback.target_type, Feedback.answered_question, Feedback.saved_time, Feedback.rating)
    ).all()
    grouped: dict[str, list[tuple[bool | None, bool | None, int | None]]] = defaultdict(list)
    for target_type, answered, saved, rating in rows:
        grouped[target_type or "unknown"].append((answered, saved, rating))
    everything = [item for items in grouped.values() for item in items]
    return {
        "definition": (
            "Reader ratings of whether a briefing, answer, or explanation answered their question and "
            "reduced time spent searching (rates over readers who answered each question)."
        ),
        **_usefulness_block(everything),
        "by_target_type": {target: _usefulness_block(items) for target, items in sorted(grouped.items())},
    }


# --------------------------------------------------------------------------
# Education
# --------------------------------------------------------------------------


def education_metrics(session: Session) -> dict[str, Any]:
    """Comprehension and transfer accuracy by reading condition, and the Newsren-minus-summary difference."""
    rows = session.execute(
        select(
            QuizAttempt.condition,
            func.count(QuizAttempt.id),
            func.count(func.distinct(QuizAttempt.user_id)),
            func.sum(QuizAttempt.comprehension_correct),
            func.sum(QuizAttempt.comprehension_total),
            func.sum(QuizAttempt.transfer_correct),
            func.sum(QuizAttempt.transfer_total),
        ).group_by(QuizAttempt.condition)
    ).all()
    by_condition: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        by_condition[condition] = _education_block(0, 0, 0, 0, 0, 0)
    for condition, attempts, readers, c_ok, c_total, t_ok, t_total in rows:
        by_condition[condition or "unknown"] = _education_block(
            attempts, readers, c_ok or 0, c_total or 0, t_ok or 0, t_total or 0
        )
    newsren, summary = by_condition["newsren"], by_condition["summary"]
    return {
        "definition": (
            "Accuracy on short comprehension questions and on transfer questions (applying the idea to an "
            "unfamiliar case) after reading a Newsren explanation versus an ordinary summary."
        ),
        "attempts": sum(b["attempts"] for b in by_condition.values()),
        "by_condition": by_condition,
        "comprehension_delta": _diff(newsren["comprehension_accuracy"], summary["comprehension_accuracy"]),
        "transfer_delta": _diff(newsren["transfer_accuracy"], summary["transfer_accuracy"]),
        "note": (
            "Deltas are Newsren minus summary, pooled over all questions answered. They describe the readers "
            "who took quizzes and are causal only when readers were randomly assigned to a condition."
        ),
    }


def _education_block(attempts: int, readers: int, c_ok: int, c_total: int, t_ok: int, t_total: int) -> dict:
    return {
        "attempts": int(attempts),
        "readers": int(readers),
        "comprehension_correct": int(c_ok),
        "comprehension_total": int(c_total),
        "comprehension_accuracy": _rate(c_ok, c_total),
        "transfer_correct": int(t_ok),
        "transfer_total": int(t_total),
        "transfer_accuracy": _rate(t_ok, t_total),
    }


# --------------------------------------------------------------------------
# Engagement
# --------------------------------------------------------------------------


def engagement_metrics(session: Session, min_gap_minutes: int = RETURN_VISIT_MIN_GAP_MINUTES) -> dict[str, Any]:
    """Return visits to followed stories and the useful-versus-unwanted alert rate.

    A return visit is a signed-in reader viewing a story they follow (or one
    of its briefs) at least ``min_gap_minutes`` after following it; repeat
    views on the same day count once (per reader, story, and day).
    """
    follow_rows = session.execute(
        select(Follow.target_type, Follow.active, func.count(Follow.id), func.sum(Follow.check_count)).group_by(
            Follow.target_type, Follow.active
        )
    ).all()
    follows_total = sum(n for _, _, n, _ in follow_rows)
    follows_active = sum(n for _, active, n, _ in follow_rows if active)
    by_target: Counter[str] = Counter()
    for target, active, n, _ in follow_rows:
        if active:
            by_target[target or "topic"] += n
    checks = int(sum(c or 0 for _, _, _, c in follow_rows))

    story_follows = session.execute(
        select(Follow.id, Follow.user_id, Follow.story_id).where(Follow.story_id.is_not(None))
    ).all()
    viewed_story = func.coalesce(PageView.story_id, Brief.story_id)
    visit_rows = session.execute(
        select(Follow.id, Follow.user_id, viewed_story, PageView.created_at, Follow.created_at)
        .select_from(PageView)
        .outerjoin(Brief, Brief.id == PageView.brief_id)
        .join(Follow, and_(Follow.user_id == PageView.user_id, Follow.story_id == viewed_story))
        .where(PageView.user_id.is_not(None), Follow.story_id.is_not(None))
    ).all()
    gap = timedelta(minutes=max(0, min_gap_minutes))
    visits: set[tuple[int, int, str]] = set()
    visit_days: set[tuple[int, str]] = set()
    follows_returned: set[int] = set()
    for follow_id, user_id, story_id, viewed_at, followed_at in visit_rows:
        if viewed_at is None or followed_at is None or viewed_at < followed_at + gap:
            continue
        day = viewed_at.date().isoformat()
        visits.add((user_id, story_id, day))
        visit_days.add((user_id, day))
        follows_returned.add(follow_id)

    alert_rows = session.execute(
        select(Alert.feedback, func.count(Alert.id), func.count(Alert.read_at)).group_by(Alert.feedback)
    ).all()
    alerts_total = sum(n for _, n, _ in alert_rows)
    alerts_read = sum(r for _, _, r in alert_rows)
    useful = sum(n for fb, n, _ in alert_rows if fb == "useful")
    unwanted = sum(n for fb, n, _ in alert_rows if fb == "unwanted")

    return {
        "definition": (
            "Meaningful return visits to followed stories (distinct reader-story-days, at least "
            f"{min_gap_minutes} minutes after following) and the share of rated alerts readers found useful."
        ),
        "follows_total": follows_total,
        "follows_active": follows_active,
        "follows_active_by_target": dict(sorted(by_target.items())),
        "monitoring_checks": checks,
        "story_follows": len(story_follows),
        "readers_following_stories": len({user_id for _, user_id, _ in story_follows}),
        "story_follows_with_return_visit": len(follows_returned),
        "return_visit_rate": _rate(len(follows_returned), len(story_follows)),
        "return_visits": len(visits),
        "return_visit_days": len(visit_days),
        "readers_returning": len({user_id for user_id, _ in visit_days}),
        "alerts_total": alerts_total,
        "alerts_read": alerts_read,
        "alerts_read_rate": _rate(alerts_read, alerts_total),
        "alerts_useful": useful,
        "alerts_unwanted": unwanted,
        "alerts_rated": useful + unwanted,
        "useful_alert_rate": _rate(useful, useful + unwanted),
        "unwanted_alert_rate": _rate(unwanted, useful + unwanted),
        "alerts_per_active_follow": _rate(alerts_total, follows_active, 2),
    }


# --------------------------------------------------------------------------
# Editorial efficiency
# --------------------------------------------------------------------------


def editorial_metrics(
    session: Session, baseline_manual_minutes: float = DEFAULT_BASELINE_MANUAL_MINUTES
) -> dict[str, Any]:
    """Human review time, approval/rejection counts, corrections, and estimated time saved per approved brief.

    Time saved is an estimate: ``baseline_manual_minutes`` (how long a
    journalist would take to research and write a comparable sourced brief by
    hand) minus the median measured review time of approved briefs.
    """
    action_rows = session.execute(
        select(EditorialAction.action, func.count(EditorialAction.id))
        .where(EditorialAction.action.in_(("approve", "reject", "correct", "claim_verdict")))
        .group_by(EditorialAction.action)
    ).all()
    actions = {action: int(n) for action, n in action_rows}
    approvals, rejections = actions.get("approve", 0), actions.get("reject", 0)
    durations = [
        float(d)
        for d in session.scalars(
            select(EditorialAction.duration_seconds).where(
                EditorialAction.action == "approve", EditorialAction.duration_seconds.is_not(None)
            )
        )
    ]
    reject_durations = [
        float(d)
        for d in session.scalars(
            select(EditorialAction.duration_seconds).where(
                EditorialAction.action == "reject", EditorialAction.duration_seconds.is_not(None)
            )
        )
    ]
    status_rows = session.execute(
        select(Brief.status, Brief.requires_human_review, func.count(Brief.id)).group_by(
            Brief.status, Brief.requires_human_review
        )
    ).all()
    awaiting = sum(n for status, _, n in status_rows if status in (BriefStatus.IN_REVIEW, BriefStatus.DRAFT))
    auto_published = sum(n for status, needs, n in status_rows if status == BriefStatus.APPROVED and not needs)
    correction_rows = session.execute(
        select(Correction.material, func.count(Correction.id)).group_by(Correction.material)
    ).all()

    median_seconds = _median(durations, 1)
    median_minutes = round(median_seconds / 60, 2) if median_seconds is not None else None
    saved_per_brief = round(baseline_manual_minutes - median_minutes, 2) if median_minutes is not None else None
    return {
        "definition": (
            "Time saved per approved brief, net of human review: baseline manual research-and-writing "
            "minutes minus the median measured review time."
        ),
        "approvals": approvals,
        "rejections": rejections,
        "approval_rate": _rate(approvals, approvals + rejections),
        "auto_published": auto_published,
        "awaiting_review": awaiting,
        "corrections": sum(n for _, n in correction_rows),
        "material_corrections": sum(n for material, n in correction_rows if material),
        "claim_verdicts": actions.get("claim_verdict", 0),
        "reviews_timed": len(durations),
        "median_review_seconds": median_seconds,
        "mean_review_seconds": _mean(durations, 1),
        "median_review_minutes": median_minutes,
        "median_rejection_review_seconds": _median(reject_durations, 1),
        "baseline_manual_minutes": float(baseline_manual_minutes),
        "estimated_minutes_saved_per_approved_brief": saved_per_brief,
        "estimated_hours_saved_total": (
            round(saved_per_brief * approvals / 60, 2) if saved_per_brief is not None else None
        ),
        "time_saved_is_estimate": True,
        "time_saved_method": (
            f"Estimate, not a measurement: an assumed {baseline_manual_minutes:g} minutes to research and "
            "write a comparable sourced brief by hand, minus the median measured human review time of "
            "approved briefs. It includes the cost of human review but not inference time or later "
            "corrections; a negative value means review took longer than the assumed manual baseline."
        ),
    }


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def _cost_block(rows: list[Any]) -> dict[str, Any]:
    runs = sum(r.n for r in rows)
    statuses: Counter[str] = Counter()
    for r in rows:
        statuses[r.status] += r.n
    cost = sum(r.cost or 0.0 for r in rows)
    tokens_in = sum(r.tokens_in or 0 for r in rows)
    tokens_out = sum(r.tokens_out or 0 for r in rows)
    timed = sum(r.timed or 0 for r in rows)
    return {
        "runs": runs,
        "total_cost_usd": round(cost, 6),
        "avg_cost_usd": _rate(cost, runs, 6),
        "avg_input_tokens": _rate(tokens_in, runs, 1),
        "avg_output_tokens": _rate(tokens_out, runs, 1),
        "avg_tokens": _rate(tokens_in + tokens_out, runs, 1),
        "avg_duration_ms": _rate(sum(r.duration or 0 for r in rows), timed, 1),
        "avg_tool_calls": _rate(sum(r.tools or 0 for r in rows), runs, 2),
        "avg_retrieval_requests": _rate(sum(r.retrievals or 0 for r in rows), runs, 2),
        "claude_share": _rate(sum(r.n for r in rows if r.provider == "claude"), runs),
        "fallback_share": _rate(statuses[AgentRunStatus.FALLBACK.value], runs),
        "refused_share": _rate(statuses[AgentRunStatus.REFUSED.value], runs),
        "budget_exceeded_share": _rate(statuses[AgentRunStatus.BUDGET_EXCEEDED.value], runs),
        "error_share": _rate(statuses[AgentRunStatus.ERROR.value], runs),
    }


def cost_metrics(session: Session) -> dict[str, Any]:
    """Inference and retrieval cost per agent, per story brief, per question, and per monitoring check."""
    rows = session.execute(
        select(
            AgentRun.agent.label("agent"),
            AgentRun.status.label("status"),
            AgentRun.provider.label("provider"),
            func.count(AgentRun.id).label("n"),
            func.sum(AgentRun.cost_usd).label("cost"),
            func.sum(AgentRun.input_tokens).label("tokens_in"),
            func.sum(AgentRun.output_tokens).label("tokens_out"),
            func.sum(AgentRun.duration_ms).label("duration"),
            func.count(AgentRun.duration_ms).label("timed"),
            func.sum(AgentRun.tool_calls).label("tools"),
            func.sum(AgentRun.retrieval_requests).label("retrievals"),
        ).group_by(AgentRun.agent, AgentRun.status, AgentRun.provider)
    ).all()
    by_agent_rows: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        by_agent_rows[row.agent].append(row)
    by_agent = {agent: _cost_block(agent_rows) for agent, agent_rows in sorted(by_agent_rows.items())}
    overall = _cost_block(list(rows))

    def _per(agent: str) -> dict[str, Any]:
        block = by_agent.get(agent) or _cost_block([])
        return {
            "runs": block["runs"],
            "avg_cost_usd": block["avg_cost_usd"],
            "avg_tokens": block["avg_tokens"],
            "avg_duration_ms": block["avg_duration_ms"],
            "avg_retrieval_requests": block["avg_retrieval_requests"],
        }

    monitor = by_agent.get(MONITOR_AGENT) or _cost_block([])
    checks = int(session.scalar(select(func.sum(Follow.check_count))) or 0)
    active_follows = int(session.scalar(select(func.count(Follow.id)).where(Follow.active.is_(True))) or 0)
    return {
        "definition": (
            "Average inference cost (USD, from recorded token usage) and retrieval requests per story brief, "
            "Ask Newsren question, and monitoring check. Offline heuristic runs cost $0."
        ),
        "runs": overall["runs"],
        "total_cost_usd": overall["total_cost_usd"],
        "avg_cost_usd_per_run": overall["avg_cost_usd"],
        "claude_share": overall["claude_share"],
        "fallback_share": overall["fallback_share"],
        "refused_share": overall["refused_share"],
        "budget_exceeded_share": overall["budget_exceeded_share"],
        "error_share": overall["error_share"],
        "per_story_brief": _per(STORY_BRIEF_AGENT),
        "per_question": _per(QUESTION_AGENT),
        "per_monitoring_check": {
            # The monitoring agent only runs when a follow has a free-text
            # criterion, so cost per check spreads agent cost over all checks.
            "agent_runs": monitor["runs"],
            "avg_cost_usd_per_run": monitor["avg_cost_usd"],
            "checks": checks,
            "avg_cost_usd_per_check": _rate(monitor["total_cost_usd"], checks, 6),
            "active_follows": active_follows,
            "cost_usd_per_active_follow": _rate(monitor["total_cost_usd"], active_follows, 6),
        },
        "by_agent": by_agent,
        "note": (
            "Costs are estimates computed from token usage at list prices when each run finished. Retrieval "
            "from approved public sources (archive, arXiv, SEC EDGAR) has no per-request fee, so retrieval "
            "expense is reported as request counts."
        ),
    }


# --------------------------------------------------------------------------
# All
# --------------------------------------------------------------------------


def all_metrics(
    session: Session, baseline_manual_minutes: float = DEFAULT_BASELINE_MANUAL_MINUTES
) -> dict[str, Any]:
    """Every metric group in one JSON-serialisable dict (for the analytics page and ``newsren metrics``)."""
    return {
        "generated_at": iso(utcnow()),
        "trust": trust_metrics(session),
        "usefulness": usefulness_metrics(session),
        "education": education_metrics(session),
        "engagement": engagement_metrics(session),
        "editorial": editorial_metrics(session, baseline_manual_minutes),
        "cost": cost_metrics(session),
    }
