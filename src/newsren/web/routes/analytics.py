"""Analytics: the success metrics of plan §9, for editors.

Total article output and chatbot usage are deliberately *not* the primary
metrics. The page shows Trust, Usefulness, Education, Engagement, Editorial
efficiency and Cost — each with a plain-language definition, the sample size
(n) behind every rate, and an explicit "no data yet" state instead of a
misleading zero. The numbers come from ``newsren.metrics.all_metrics``; the
same dict is served as JSON at ``/api/v1/metrics``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from newsren.metrics import all_metrics
from newsren.web.deps import DB, EditorUser
from newsren.web.routes.editor import desk, fmt_duration

router = APIRouter()

# Plan §9 wording, shown under each heading.
PLAN_TEXT = {
    "trust": "Percentage of sampled claims accurately supported by linked sources; frequency of material corrections.",
    "usefulness": "Reader ratings of whether a briefing answered their question and reduced time spent searching.",
    "education": "Improvement on short comprehension and transfer questions, compared with reading ordinary summaries.",
    "engagement": "Meaningful return visits to followed stories; rate of useful versus unwanted alerts.",
    "editorial": "Time saved per approved briefing, including the cost of human review.",
    "cost": "Average inference and retrieval expense per story, query, and active monitoring task.",
}

TARGET_LABELS = {
    "briefing": "Personal briefings",
    "answer": "Ask Newsren answers",
    "brief": "AI Story Briefs",
    "learning": "Learning materials",
    "impact": "Industry impact",
    "evidence": "Evidence inspection",
}


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def fmt_rate(value: float | None) -> str | None:
    return None if value is None else f"{value * 100:.1f}%"


def fmt_points(value: float | None) -> str | None:
    """A difference between two rates, in percentage points with a sign."""
    return None if value is None else f"{value * 100:+.1f} pts"


def fmt_usd(value: float | None) -> str | None:
    if value is None:
        return None
    if value == 0:
        return "$0.00"
    return f"${value:,.4f}" if abs(value) < 1 else f"${value:,.2f}"


def fmt_number(value: float | int | None, digits: int = 0) -> str | None:
    if value is None:
        return None
    return f"{value:,.{digits}f}"


def tile(
    label: str,
    value: str | None,
    *,
    n: int | None = None,
    n_label: str = "",
    note: str = "",
    bar: float | None = None,
    empty: str = "No data yet",
) -> dict[str, Any]:
    """One stat tile. ``value`` None renders the "no data yet" state; ``bar`` is a 0-1 share."""
    return {
        "label": label,
        "value": value,
        "n": n,
        "n_label": n_label,
        "note": note,
        "bar": None if bar is None else max(0.0, min(1.0, float(bar))),
        "empty": empty,
    }


# --------------------------------------------------------------------------
# Sections (one per plan §9 measure)
# --------------------------------------------------------------------------


def trust_section(t: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "label": vertical.capitalize(),
            "cells": [
                fmt_number(v["judged"]),
                fmt_number(v["accurate"]),
                fmt_rate(v["rate"]) or "—",
            ],
        }
        for vertical, v in (t.get("accuracy_by_vertical") or {}).items()
    ]
    return {
        "key": "trust",
        "title": "Trust",
        "plan": PLAN_TEXT["trust"],
        "definition": t["definition"],
        "tiles": [
            tile(
                "Claims accurately supported",
                fmt_rate(t["claim_accuracy_rate"]),
                n=t["claims_judged"],
                n_label="claims sampled by editors",
                bar=t["claim_accuracy_rate"],
                empty="No claims sampled yet",
            ),
            tile(
                "Material corrections per 100 published",
                fmt_number(t["material_corrections_per_100_published"], 2),
                n=t["published_briefs"],
                n_label="published briefs",
                note=f"{t['material_corrections']} material of {t['corrections_total']} corrections",
                empty="Nothing published yet",
            ),
            tile(
                "Claims citing a source",
                fmt_rate(t["citation_coverage"]),
                n=t["fact_claims"],
                n_label="fact and performance claims",
                bar=t["citation_coverage"],
            ),
            tile(
                "Published claims citing a source",
                fmt_rate(t["published_citation_coverage"]),
                n=t["published_fact_claims"],
                n_label="claims in published briefs",
                bar=t["published_citation_coverage"],
            ),
        ],
        "table": {
            "caption": "Sampled-claim accuracy by section",
            "head": ["Section", "Sampled", "Accurate", "Accuracy"],
            "rows": rows,
        }
        if rows
        else None,
        "empty_hint": (
            "Editors sample claims on each brief's review page (Accurate / Inaccurate) — that sample is the "
            "trust measure."
            if not t["claims_judged"]
            else ""
        ),
    }


def usefulness_section(u: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "label": TARGET_LABELS.get(target, target.replace("_", " ").capitalize()),
            "cells": [
                fmt_number(block["responses"]),
                _rate_n(block["answered_question_rate"], block["answered_question_n"]),
                _rate_n(block["saved_time_rate"], block["saved_time_n"]),
                f"{block['mean_rating']:.2f} (n={block['rating_n']})" if block["mean_rating"] is not None else "—",
            ],
        }
        for target, block in (u.get("by_target_type") or {}).items()
    ]
    return {
        "key": "usefulness",
        "title": "Usefulness",
        "plan": PLAN_TEXT["usefulness"],
        "definition": u["definition"],
        "tiles": [
            tile(
                "Answered the reader's question",
                fmt_rate(u["answered_question_rate"]),
                n=u["answered_question_n"],
                n_label="readers who said yes or no",
                bar=u["answered_question_rate"],
            ),
            tile(
                "Reduced time spent searching",
                fmt_rate(u["saved_time_rate"]),
                n=u["saved_time_n"],
                n_label="readers who said yes or no",
                bar=u["saved_time_rate"],
            ),
            tile(
                "Mean rating",
                f"{u['mean_rating']:.2f} / 5" if u["mean_rating"] is not None else None,
                n=u["rating_n"],
                n_label="ratings",
                bar=(u["mean_rating"] / 5) if u["mean_rating"] is not None else None,
            ),
            tile("Feedback responses", fmt_number(u["responses"]), note="briefings, answers, briefs and tools"),
        ],
        "table": {
            "caption": "Usefulness by what readers rated",
            "head": ["Item", "Responses", "Answered question", "Saved time", "Mean rating"],
            "rows": rows,
        }
        if rows
        else None,
        "empty_hint": (
            "Readers rate answers, briefings and tools with the “Did this answer your question?” prompts."
            if not u["responses"]
            else ""
        ),
    }


def _rate_n(rate: float | None, n: int) -> str:
    return f"{fmt_rate(rate)} (n={n})" if rate is not None else "—"


def education_section(ed: dict[str, Any]) -> dict[str, Any]:
    by = ed["by_condition"]
    newsren, summary = by.get("newsren", {}), by.get("summary", {})
    condition_labels = {"newsren": "Newsren explanation", "summary": "Ordinary summary"}
    rows = [
        {
            "label": condition_labels.get(cond, cond),
            "cells": [
                fmt_number(block["attempts"]),
                fmt_number(block["readers"]),
                f"{fmt_rate(block['comprehension_accuracy'])} ({block['comprehension_correct']}/{block['comprehension_total']})"
                if block["comprehension_accuracy"] is not None
                else "—",
                f"{fmt_rate(block['transfer_accuracy'])} ({block['transfer_correct']}/{block['transfer_total']})"
                if block["transfer_accuracy"] is not None
                else "—",
            ],
        }
        for cond, block in by.items()
    ]
    return {
        "key": "education",
        "title": "Education",
        "plan": PLAN_TEXT["education"],
        "definition": ed["definition"],
        "tiles": [
            tile(
                "Comprehension · Newsren",
                fmt_rate(newsren.get("comprehension_accuracy")),
                n=newsren.get("comprehension_total", 0),
                n_label="questions answered",
                bar=newsren.get("comprehension_accuracy"),
            ),
            tile(
                "Comprehension · summary",
                fmt_rate(summary.get("comprehension_accuracy")),
                n=summary.get("comprehension_total", 0),
                n_label="questions answered",
                bar=summary.get("comprehension_accuracy"),
            ),
            tile(
                "Transfer · Newsren",
                fmt_rate(newsren.get("transfer_accuracy")),
                n=newsren.get("transfer_total", 0),
                n_label="questions answered",
                bar=newsren.get("transfer_accuracy"),
            ),
            tile(
                "Transfer · summary",
                fmt_rate(summary.get("transfer_accuracy")),
                n=summary.get("transfer_total", 0),
                n_label="questions answered",
                bar=summary.get("transfer_accuracy"),
            ),
            tile(
                "Comprehension difference",
                fmt_points(ed["comprehension_delta"]),
                note="Newsren minus summary",
                empty="Needs both conditions",
            ),
            tile(
                "Transfer difference",
                fmt_points(ed["transfer_delta"]),
                note="Newsren minus summary",
                empty="Needs both conditions",
            ),
        ],
        "table": {
            "caption": "Quiz results by reading condition",
            "head": ["Condition", "Attempts", "Readers", "Comprehension", "Transfer"],
            "rows": rows,
        },
        "note": ed.get("note", ""),
        "empty_hint": (
            "Readers take the short comprehension and transfer quiz in “Learn this topic”."
            if not ed["attempts"]
            else ""
        ),
    }


def engagement_section(en: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": "engagement",
        "title": "Engagement",
        "plan": PLAN_TEXT["engagement"],
        "definition": en["definition"],
        "tiles": [
            tile(
                "Followed stories revisited",
                fmt_rate(en["return_visit_rate"]),
                n=en["story_follows"],
                n_label="story follows",
                bar=en["return_visit_rate"],
                note=f"{en['return_visits']} return visit{'s' if en['return_visits'] != 1 else ''} by "
                f"{en['readers_returning']} reader{'s' if en['readers_returning'] != 1 else ''}",
            ),
            tile(
                "Useful alerts",
                fmt_rate(en["useful_alert_rate"]),
                n=en["alerts_rated"],
                n_label="alerts rated by readers",
                bar=en["useful_alert_rate"],
            ),
            tile(
                "Unwanted alerts",
                fmt_rate(en["unwanted_alert_rate"]),
                n=en["alerts_rated"],
                n_label="alerts rated by readers",
                bar=en["unwanted_alert_rate"],
            ),
            tile(
                "Alerts opened",
                fmt_rate(en["alerts_read_rate"]),
                n=en["alerts_total"],
                n_label="alerts sent",
                bar=en["alerts_read_rate"],
            ),
            tile(
                "Active follows",
                fmt_number(en["follows_active"]),
                note=f"{en['monitoring_checks']} monitoring check{'s' if en['monitoring_checks'] != 1 else ''} run"
                + (
                    f" · {en['alerts_per_active_follow']:.2f} alerts per follow"
                    if en["alerts_per_active_follow"] is not None
                    else ""
                ),
            ),
        ],
        "table": None,
        "empty_hint": (
            "Readers follow stories and questions from “Follow developments”; alerts carry a useful / not useful prompt."
            if not en["follows_total"]
            else ""
        ),
    }


def editorial_section(ed: dict[str, Any]) -> dict[str, Any]:
    decided = ed["approvals"] + ed["rejections"]
    return {
        "key": "editorial",
        "title": "Editorial efficiency",
        "plan": PLAN_TEXT["editorial"],
        "definition": ed["definition"],
        "tiles": [
            tile(
                "Estimated minutes saved per approved brief",
                fmt_number(ed["estimated_minutes_saved_per_approved_brief"], 1),
                n=ed["reviews_timed"],
                n_label="timed reviews",
                note="Estimate, net of human review time",
                empty="No timed reviews yet",
            ),
            tile(
                "Median review time",
                fmt_duration(ed["median_review_seconds"]) if ed["median_review_seconds"] is not None else None,
                n=ed["reviews_timed"],
                n_label="timed approvals",
                note=f"Mean {fmt_duration(ed['mean_review_seconds'])}"
                if ed["mean_review_seconds"] is not None
                else "",
                empty="No timed reviews yet",
            ),
            tile(
                "Approval rate",
                fmt_rate(ed["approval_rate"]),
                n=decided,
                n_label="human decisions",
                bar=ed["approval_rate"],
                note=f"{ed['approvals']} approved · {ed['rejections']} rejected",
            ),
            tile("Awaiting review", fmt_number(ed["awaiting_review"]), note="in the queue now"),
            tile(
                "Published automatically",
                fmt_number(ed["auto_published"]),
                note="low risk, no blockers, human approval not required",
            ),
            tile(
                "Estimated hours saved in total",
                fmt_number(ed["estimated_hours_saved_total"], 1),
                n=ed["approvals"],
                n_label="approvals",
                empty="No timed reviews yet",
            ),
        ],
        "facts": [
            ("Corrections recorded", f"{ed['corrections']} ({ed['material_corrections']} material)"),
            ("Claim verdicts recorded", str(ed["claim_verdicts"])),
            ("Assumed manual baseline", f"{ed['baseline_manual_minutes']:g} minutes per sourced brief"),
        ],
        "table": None,
        "note": ed["time_saved_method"],
        "empty_hint": "Review times are measured from opening a brief on the editor desk to approving it."
        if not ed["reviews_timed"]
        else "",
    }


def cost_section(c: dict[str, Any]) -> dict[str, Any]:
    per_brief, per_question, per_check = c["per_story_brief"], c["per_question"], c["per_monitoring_check"]
    rows = [
        {
            "label": agent,
            "cells": [
                fmt_number(b["runs"]),
                fmt_usd(b["total_cost_usd"]) or "—",
                fmt_usd(b["avg_cost_usd"]) or "—",
                fmt_number(b["avg_tokens"]) or "—",
                f"{b['avg_duration_ms']:,.0f} ms" if b["avg_duration_ms"] is not None else "—",
                fmt_number(b["avg_tool_calls"], 2) or "—",
                fmt_number(b["avg_retrieval_requests"], 2) or "—",
                " · ".join(
                    part
                    for part in (
                        f"fallback {fmt_rate(b['fallback_share'])}" if b["fallback_share"] else "",
                        f"refused {fmt_rate(b['refused_share'])}" if b["refused_share"] else "",
                        f"over budget {fmt_rate(b['budget_exceeded_share'])}"
                        if b["budget_exceeded_share"]
                        else "",
                        f"error {fmt_rate(b['error_share'])}" if b["error_share"] else "",
                    )
                    if part
                )
                or "none",
            ],
        }
        for agent, b in (c.get("by_agent") or {}).items()
    ]
    return {
        "key": "cost",
        "title": "Cost",
        "plan": PLAN_TEXT["cost"],
        "definition": c["definition"],
        "tiles": [
            tile(
                "Per story brief",
                fmt_usd(per_brief["avg_cost_usd"]),
                n=per_brief["runs"],
                n_label="brief runs",
                note=f"{fmt_number(per_brief['avg_retrieval_requests'], 2)} retrieval requests on average"
                if per_brief["avg_retrieval_requests"] is not None
                else "",
            ),
            tile(
                "Per Ask Newsren question",
                fmt_usd(per_question["avg_cost_usd"]),
                n=per_question["runs"],
                n_label="questions",
                note=f"{fmt_number(per_question['avg_retrieval_requests'], 2)} retrieval requests on average"
                if per_question["avg_retrieval_requests"] is not None
                else "",
            ),
            tile(
                "Per monitoring check",
                fmt_usd(per_check["avg_cost_usd_per_check"]),
                n=per_check["checks"],
                n_label="checks",
                note=f"{per_check['active_follows']} active follow{'s' if per_check['active_follows'] != 1 else ''}",
            ),
            tile(
                "Total inference cost",
                fmt_usd(c["total_cost_usd"]) if c["runs"] else None,
                n=c["runs"],
                n_label="agent runs",
                note=f"Claude answered {fmt_rate(c['claude_share'])} of runs"
                if c["claude_share"] is not None
                else "",
            ),
            tile(
                "Offline fallback share",
                fmt_rate(c["fallback_share"]),
                n=c["runs"],
                n_label="agent runs",
                bar=c["fallback_share"],
                note="Claude unavailable or invalid; offline engine answered",
            ),
        ],
        "table": {
            "caption": "Cost by agent",
            "head": [
                "Agent",
                "Runs",
                "Total cost",
                "Avg cost",
                "Avg tokens",
                "Avg duration",
                "Avg tool calls",
                "Avg retrievals",
                "Fallback / refused / errors",
            ],
            "rows": rows,
        }
        if rows
        else None,
        "note": c.get("note", ""),
        "empty_hint": "Costs are recorded for every agent run by the orchestrator." if not c["runs"] else "",
    }


def build_sections(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        trust_section(metrics["trust"]),
        usefulness_section(metrics["usefulness"]),
        education_section(metrics["education"]),
        engagement_section(metrics["engagement"]),
        editorial_section(metrics["editorial"]),
        cost_section(metrics["cost"]),
    ]


@router.get("/analytics")
def analytics_page(request: Request, db: DB, editor: EditorUser):
    metrics = all_metrics(db)
    return desk(
        request,
        "analytics/index.html",
        {"sections": build_sections(metrics), "generated_at": metrics.get("generated_at")},
        "analytics",
    )
