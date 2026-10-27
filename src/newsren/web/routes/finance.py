"""Finance section (plan §3 "Finance news", §4): the section front and its four editorial products.

Routes
* ``GET /finance`` — section front: what the section tracks and never does,
  the latest approved finance briefs, previews of every module, and the
  "Follow a company or sector" form (posts to ``POST /follows``).
* ``GET /finance/earnings`` and ``/finance/earnings/{id}`` — Earnings Decoder:
  reported figures from SEC EDGAR filings, each with its filing link and
  source quote, compared with prior periods and (when quoted from the
  company's release) guidance.
* ``GET /finance/funding`` — Funding Round Tracker: trends by sector, stage,
  and investor activity (counts), and the rounds table; filters ``sector``,
  ``stage``, ``days``. htmx requests get only the results fragment.
* ``GET /finance/ai-reality-check`` — AI-in-Finance Reality Check: claims
  grouped by announcement / pilot / backtest / live deployment / regulatory
  finding; filter ``area``. htmx requests get only the results fragment.
* ``GET /finance/regulation`` — Regulation Watch: who is affected, what
  changes, and when, quoted from regulators' documents.
* ``GET /finance/concepts`` and ``/finance/concepts/{slug}`` — Finance
  concepts for learners, linked to the published stories that use them.

Guardrails (plan §4): every page shows the not-investment-advice notice at
the top and a data note (sources, retrieval times, delays; public or licensed
data only). Nothing here ranks companies or startups, issues ratings or price
targets, or connects to brokerage or trading. Only approved briefs are linked
for readers; editors also see in-review briefs, clearly flagged as unpublished.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsren.agents.learning import strip_citation_marks
from newsren.editorial.guardrails import FINANCE_DISCLAIMER
from newsren.editorial.workflow import published_briefs
from newsren.finance.ai_claims import (
    AREA_LABELS,
    AREA_NOTES,
    CLASSIFICATION_HELP,
    CLASSIFICATION_LABELS,
    REGULATORY_CONTEXT,
    claims_by_classification,
)
from newsren.finance.concepts import CATEGORY_LABELS, Concept, concept_mentions, get_concept, related_concepts
from newsren.finance.concepts import DISCLAIMER as CONCEPTS_DISCLAIMER
from newsren.finance.concepts import concepts_by_category as grouped_concepts
from newsren.finance.edgar import GUIDANCE_NOTE, fiscal_label, format_value, period_text
from newsren.finance.funding import STAGE_LABELS, STAGES, TRENDS_NOTE, funding_trends
from newsren.finance.regulation import WATCH_NOTE, finance_regulation_items, regulation_watch
from newsren.finance.service import DATA_NOTE, REGULATION_WINDOW_DAYS, finance_overview
from newsren.models import (
    AIFinanceClaim,
    Article,
    Brief,
    BriefStatus,
    EarningsReport,
    FundingRound,
    User,
    Vertical,
    utcnow,
)
from newsren.textutil import domain_of, iso
from newsren.web.deps import DB, CurrentUser, csrf_token
from newsren.web.templating import render, templates

router = APIRouter()

FINANCE = Vertical.FINANCE.value

# The section's own navigation (plan §3 finance products + §4 learners' concepts).
SECTION_NAV: tuple[tuple[str, str, str], ...] = (
    ("overview", "/finance", "Overview"),
    ("earnings", "/finance/earnings", "Earnings Decoder"),
    ("funding", "/finance/funding", "Funding Round Tracker"),
    ("ai", "/finance/ai-reality-check", "AI-in-Finance Reality Check"),
    ("regulation", "/finance/regulation", "Regulation Watch"),
    ("concepts", "/finance/concepts", "Concepts for learners"),
)

# Plan §3/§4: what the section tracks, and the guardrails readers can rely on.
TRACKS: tuple[str, ...] = (
    "Earnings releases and calls, and the 10-Q, 10-K, and 8-K filings behind them (SEC EDGAR).",
    "Regulatory filings and actions by the SEC, FINRA, central banks, and other regulators.",
    "Venture funding rounds announced publicly, by sector, stage, and investor.",
    "How banks, asset managers, and other financial firms are adopting AI, and what is documented versus claimed.",
)
NEVER: tuple[str, ...] = (
    "No buy, sell, or hold calls, and no price targets.",
    "No portfolio recommendations and no personalized investment advice.",
    "No trade execution and no brokerage connections.",
    "No forecasts presented as facts, and no claim that an AI agent can reliably outperform markets.",
)

FINANCE_PROMPTS: tuple[str, ...] = (
    "What is the difference between a backtest and a live track record?",
    "What did the latest AI-washing enforcement action find?",
    "How does dilution work in a Series A round?",
)

# Follow a company or sector (plan §4): material events, never price moves.
FOLLOW_EVENTS: tuple[tuple[str, str], ...] = (
    ("filing", "New filings (10-Q, 10-K, 8-K)"),
    ("earnings", "Earnings releases and calls"),
    ("funding", "Funding rounds"),
    ("enforcement", "Enforcement actions"),
)
FOLLOW_DEFAULT_QUESTION = (
    "Alert me to material events — new filings, earnings, funding rounds, and enforcement actions — "
    "not price moves."
)

EARNINGS_CONCEPTS: tuple[str, ...] = (
    "eps",
    "earnings-guidance",
    "revenue-vs-net-income",
    "operating-margin",
    "sec-forms-10q-10k-8k",
    "xbrl",
)
COMPARISON_LABELS: dict[str, str] = {
    "same_quarter_prior_year": "Same quarter a year earlier",
    "prior_quarter": "Previous quarter",
    "prior_fiscal_year": "Previous fiscal year",
}
# Where a reported figure sits relative to the company's own guidance (positions, not verdicts).
GUIDANCE_POSITIONS: dict[str, str] = {
    "above": "Above the guided figure or range",
    "within": "Within the guided range",
    "below": "Below the guided figure or range",
    "not_comparable": "Not directly comparable",
    "unknown": "Not determined",
}
EARNINGS_LIST_LIMIT = 100

FUNDING_DAY_CHOICES: tuple[int, ...] = (30, 90, 180, 365)
FUNDING_DEFAULT_DAYS = 90
FUNDING_NOT_A_RANKING = "Trends from public announcements — not an investment ranking or recommendation."

REGULATION_DAY_CHOICES: tuple[int, ...] = (30, 60, 180, 365)
REGULATION_DEFAULT_DAYS = 60
WHEN_LABELS: dict[str, str] = {
    "comment_deadline": "Comments due",
    "compliance": "Compliance date",
    "effective": "Takes effect",
    "deadline": "Deadline",
}

# Why the five AI-in-finance classifications matter (plan §3 finance use case).
WHY_CLASSIFY = (
    "A bank reporting that AI agents beat a benchmark portfolio can mean very different things. JPMorgan "
    "researchers reported in July 2026 that AI portfolio agents beat a 60/40 portfolio in a two-decade "
    "backtest, while warning that an in-sample result is not proof of future outperformance. A backtest is a "
    "simulation on historical data designed with hindsight; a pilot is a limited test; a live deployment is "
    "documented use with real users or money; a regulatory finding is a regulator's own conclusion. The "
    "Reality Check labels each claim so the headline is never mistaken for the evidence."
)
REALITY_CHECK_CONCEPTS: tuple[str, ...] = (
    "backtesting",
    "in-sample-vs-out-of-sample",
    "overfitting",
    "paper-trading",
    "benchmark",
    "ai-washing",
)

# Published briefs scanned to link concepts to the stories that use them.
CONCEPT_SCAN_LIMIT = 300
CONCEPT_STORIES_LIMIT = 12


# --------------------------------------------------------------------------
# Request and rendering helpers
# --------------------------------------------------------------------------


def is_htmx(request: Request) -> bool:
    """True for an htmx fragment request (a history restore needs the full page)."""
    return (
        request.headers.get("HX-Request") == "true" and request.headers.get("HX-History-Restore-Request") != "true"
    )


def is_editor(user: User | None) -> bool:
    return user is not None and user.is_editor


def _nav(section: str) -> list[dict[str, Any]]:
    return [{"key": k, "url": url, "label": label, "active": k == section} for k, url, label in SECTION_NAV]


def _base_context(section: str) -> dict[str, Any]:
    return {
        "FINANCE_DISCLAIMER": FINANCE_DISCLAIMER,
        "DATA_NOTE": DATA_NOTE,
        "active_vertical": FINANCE,
        "finance_nav": _nav(section),
        "section": section,
        "plain": strip_citation_marks,
        "page_generated_at": utcnow(),
    }


def _page(
    request: Request, template: str, ctx: dict[str, Any], *, section: str, status_code: int = 200
) -> Response:
    response = render(request, template, {**_base_context(section), **ctx}, status_code=status_code)
    response.headers["Vary"] = "HX-Request"
    return response


def _partial(request: Request, template: str, ctx: dict[str, Any], *, section: str) -> Response:
    """An htmx fragment, rendered without consuming the session's flash message."""
    context = {**_base_context(section), **ctx}
    context.setdefault("current_user", getattr(request.state, "user", None))
    context.setdefault("csrf_token", csrf_token(request))
    context.setdefault("flash", None)
    response = templates.TemplateResponse(request, template, context)
    response.headers["Vary"] = "HX-Request"
    return response


def http_url(url: str | None) -> str:
    """The URL if it is an absolute http(s) link, else "" (never render javascript: or data: links)."""
    value = (url or "").strip()
    return value if value.lower().startswith(("https://", "http://")) else ""


def _int_choice(raw: str | None, choices: tuple[int, ...], default: int) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value in choices else default


def _query_url(path: str, **params: Any) -> str:
    query = urlencode({k: v for k, v in params.items() if v not in (None, "")})
    return f"{path}?{query}" if query else path


def _has_demo(db: Session) -> bool:
    return bool(db.scalar(select(func.count(Article.id)).where(Article.is_demo.is_(True))))


# --------------------------------------------------------------------------
# Briefs and concepts
# --------------------------------------------------------------------------


def _brief_text(brief: Brief) -> str:
    parts: list[str] = [brief.headline, brief.summary or "", brief.why_it_matters or "", brief.what_changed or ""]
    parts += [str(p) for p in brief.key_points or []]
    for section in brief.sections or []:
        if isinstance(section, dict):
            parts += [str(section.get("title") or ""), str(section.get("body") or "")]
    return strip_citation_marks(" ".join(p for p in parts if p))


def _concept_index(db: Session) -> list[tuple[Brief, list[tuple[Concept, str]]]]:
    """Published briefs (newest first) with the concepts their text mentions and the phrase that matched."""
    out = []
    for brief in published_briefs(db, limit=CONCEPT_SCAN_LIMIT):
        mentions = concept_mentions(_brief_text(brief))
        if mentions:
            out.append((brief, mentions))
    return out


def _brief_link(brief: Brief | None, user: User | None) -> dict[str, Any] | None:
    if brief is None:
        return None
    approved = brief.status == BriefStatus.APPROVED
    return {
        "id": brief.id,
        "headline": brief.headline,
        "status": brief.status,
        "approved": approved,
        "can_open": approved or is_editor(user),
        "published_at": brief.published_at,
        "generated_by": brief.generated_by,
        "generated_at": brief.generated_at,
    }


def _concept_links(slugs: tuple[str, ...]) -> list[Concept]:
    return [c for slug in slugs if (c := get_concept(slug)) is not None]


# --------------------------------------------------------------------------
# Earnings Decoder views
# --------------------------------------------------------------------------


def _change(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    pct = float(value)
    direction = "up" if pct > 0 else "down" if pct < 0 else "flat"
    word = {"up": "increase", "down": "decrease", "flat": "no change"}[direction]
    return {"value": pct, "text": f"{pct:+.1f}%", "direction": direction, "word": word}


def _metric_row(metric: dict[str, Any]) -> dict[str, Any]:
    unit = str(metric.get("unit") or "USD")
    prior = metric.get("prior_value")
    prior_quarter = metric.get("prior_quarter_value")
    comparison = str(metric.get("comparison") or "")
    second = prior_quarter is not None and comparison == "same_quarter_prior_year"
    return {
        "name": metric.get("name") or metric.get("key") or "Reported figure",
        "display": format_value(metric.get("value"), unit),
        "period": period_text(metric.get("period")),
        "annual": metric.get("period_kind") == "fiscal_year",
        "derived": bool(metric.get("derived")),
        "comparison_label": COMPARISON_LABELS.get(comparison, "Earlier period"),
        "year_ago": comparison == "same_quarter_prior_year",
        "has_prior": prior is not None,
        "prior_display": format_value(prior, unit) if prior is not None else "",
        "prior_period": period_text(metric.get("prior_period")) if metric.get("prior_period") else "",
        "change": _change(metric.get("change_pct")),
        "has_prior_quarter": second,
        "prior_quarter_display": format_value(prior_quarter, unit) if second else "",
        "prior_quarter_period": period_text(metric.get("prior_quarter_period")) if second else "",
        "prior_quarter_change": _change(metric.get("prior_quarter_change_pct")) if second else None,
        "source_url": http_url(metric.get("source_url")),
        "prior_source_url": http_url(metric.get("prior_source_url")),
        "source_quote": str(metric.get("source_quote") or ""),
        "xbrl_tag": str(metric.get("xbrl_tag") or ""),
        "form": str(metric.get("form") or ""),
        "filed": str(metric.get("filed") or ""),
    }


def _guidance_row(item: dict[str, Any]) -> dict[str, Any]:
    position = str(item.get("comparison") or "unknown")
    return {
        "metric": str(item.get("metric") or ""),
        "guidance": str(item.get("guidance") or ""),
        "actual": str(item.get("actual") or ""),
        "position": position,
        "position_label": GUIDANCE_POSITIONS.get(position, GUIDANCE_POSITIONS["unknown"]),
        "source_url": http_url(item.get("source_url")),
        "source_quote": str(item.get("source_quote") or ""),
    }


def _report_summary(report: EarningsReport, user: User | None) -> dict[str, Any]:
    company = report.company
    return {
        "id": report.id,
        "company": company.name if company else "Unknown company",
        "ticker": company.ticker if company else None,
        "cik": company.cik if company else None,
        "sector": company.sector if company else None,
        "fiscal_period": report.fiscal_period,
        "fiscal_label": fiscal_label(report.fiscal_period),
        "form": report.form,
        "filed_at": report.filed_at,
        "retrieved_at": report.retrieved_at,
        "filing_url": http_url(report.filing_url),
        "metric_count": sum(1 for m in report.metrics or [] if isinstance(m, dict)),
        "brief": _brief_link(report.brief, user),
        "is_demo": bool(report.is_demo),
    }


def _readable_note(note: str | None, retrieved_at: datetime | None) -> str:
    """The stored delay note with its ISO retrieval stamp written for people ("Oct 27, 2026 22:23 UTC")."""
    text = note or ""
    stamp = iso(retrieved_at)
    if retrieved_at is not None and stamp:
        text = text.replace(stamp, retrieved_at.strftime("%b %d, %Y %H:%M UTC").replace(" 0", " "))
    return text


def _report_detail(report: EarningsReport, user: User | None) -> dict[str, Any]:
    view = _report_summary(report, user)
    narrative = report.narrative if isinstance(report.narrative, dict) else {}
    filing = narrative.get("filing") if isinstance(narrative.get("filing"), dict) else {}
    release = narrative.get("earnings_release") if isinstance(narrative.get("earnings_release"), dict) else None
    metrics = [_metric_row(m) for m in report.metrics or [] if isinstance(m, dict)]
    view.update(
        {
            "quarterly": [m for m in metrics if not m["annual"]],
            "annual": [m for m in metrics if m["annual"]],
            "guidance": [_guidance_row(g) for g in report.guidance or [] if isinstance(g, dict)],
            "guidance_note": narrative.get("guidance_note") or GUIDANCE_NOTE,
            "filing_period": period_text(filing.get("period")) if filing.get("period") else "",
            "filing_index_url": http_url(filing.get("index_url")),
            "xbrl_source": http_url(filing.get("xbrl_source")),
            "accession": report.accession or filing.get("accession") or "",
            "release": (
                {
                    "form": str(release.get("form") or ""),
                    "filed": str(release.get("filed") or ""),
                    "url": http_url(release.get("url")),
                    "note": str(release.get("note") or ""),
                }
                if release
                else None
            ),
            "data_delay_note": _readable_note(report.data_delay_note, report.retrieved_at),
        }
    )
    return view


# --------------------------------------------------------------------------
# Funding Round Tracker views
# --------------------------------------------------------------------------


def _round_time(row: FundingRound) -> datetime | None:
    return row.announced_at or row.retrieved_at


def _rounds_in_window(db: Session, days: int, now: datetime) -> list[FundingRound]:
    since = now - timedelta(days=days)
    rows = [r for r in db.scalars(select(FundingRound)) if (t := _round_time(r)) is not None and since <= t <= now]
    return sorted(rows, key=lambda r: (_round_time(r), r.id or 0), reverse=True)


def _round_view(row: FundingRound) -> dict[str, Any]:
    leads = [str(n) for n in row.lead_investors or []]
    others = [str(n) for n in row.investors or [] if n not in leads]
    url = http_url(row.source_url)
    return {
        "id": row.id,
        "company": row.company_name,
        "sector": row.sector or "Other",
        "stage": row.stage,
        "stage_label": STAGE_LABELS.get(row.stage, row.stage),
        "amount_text": row.amount_text or "Not disclosed",
        "amount_usd": row.amount_usd,
        "lead_investors": leads,
        "other_investors": others,
        "announced_at": row.announced_at,
        "retrieved_at": row.retrieved_at,
        "source_url": url,
        "source_title": row.source_title or (domain_of(url) if url else "Source"),
        "publisher": domain_of(url) if url else "",
        "verified": bool(row.verified),
        "confidence": round(float(row.extraction_confidence or 0) * 100),
        "is_demo": bool(row.is_demo),
    }


def _bars(rows: list[dict[str, Any]], label_key: str) -> list[dict[str, Any]]:
    top = max((r["rounds"] for r in rows), default=0) or 1
    return [
        {
            "key": r.get("stage") or r.get("sector"),
            "label": r[label_key],
            "rounds": r["rounds"],
            "disclosed_usd": r.get("disclosed_usd") or 0.0,
            "width": max(4, round(r["rounds"] / top * 100)),
        }
        for r in rows
    ]


def _funding_context(db: Session, sector: str | None, stage: str | None, days: int) -> dict[str, Any]:
    now = utcnow()
    trends = funding_trends(db, days, now=now)
    window = _rounds_in_window(db, days, now)
    all_sectors = sorted({r.sector or "Other" for r in db.scalars(select(FundingRound))}, key=str.lower)
    sector = sector if sector in all_sectors else None
    stage = stage if stage in STAGES else None
    rounds = [
        r
        for r in window
        if (sector is None or (r.sector or "Other") == sector) and (stage is None or r.stage == stage)
    ]
    investors = trends["top_investors"]
    top_count = max((count for _, count in investors), default=0) or 1
    return {
        "trends": trends,
        "sector_bars": _bars(trends["by_sector"], "sector"),
        "stage_bars": _bars(trends["by_stage"], "label"),
        "investor_bars": [
            {"name": name, "rounds": count, "width": max(4, round(count / top_count * 100))}
            for name, count in investors
        ],
        "rounds": [_round_view(r) for r in rounds],
        "window_count": len(window),
        "filters": {"sector": sector, "stage": stage, "days": days},
        "filtered": sector is not None or stage is not None,
        "sector_options": all_sectors,
        "stage_options": [(s, STAGE_LABELS.get(s, s)) for s in STAGES],
        "day_options": FUNDING_DAY_CHOICES,
        "stage_label": STAGE_LABELS.get(stage, "") if stage else "",
        "filter_url": lambda **kw: _query_url(
            "/finance/funding", **{**{"sector": sector, "stage": stage, "days": days}, **kw}
        ),
        "not_a_ranking": FUNDING_NOT_A_RANKING,
        "trends_note": TRENDS_NOTE,
        "verified_count": sum(1 for r in rounds if r.verified),
    }


# --------------------------------------------------------------------------
# AI-in-Finance Reality Check views
# --------------------------------------------------------------------------


def _claim_view(claim: AIFinanceClaim, demo_articles: set[int]) -> dict[str, Any]:
    performance = bool(claim.performance_label) or claim.classification == "backtest"
    return {
        "id": claim.id,
        "firm": claim.firm,
        "area": claim.area,
        "area_label": AREA_LABELS.get(claim.area, claim.area),
        "claim_text": claim.claim_text,
        "classification": claim.classification,
        "classification_label": CLASSIFICATION_LABELS.get(claim.classification, claim.classification),
        "performance": performance,
        "performance_label": claim.performance_label,
        "period": claim.period,
        "benchmark": claim.benchmark,
        "costs_assumed": claim.costs_assumed,
        "authors_caveats": claim.authors_caveats,
        "evidence_note": claim.evidence_note,
        "source_url": http_url(claim.source_url),
        "source_title": claim.source_title,
        "publisher": domain_of(claim.source_url) if http_url(claim.source_url) else "",
        "published_at": claim.published_at,
        "retrieved_at": claim.retrieved_at,
        "is_seed": bool(claim.is_seed),
        "is_demo": claim.article_id in demo_articles if claim.article_id else False,
    }


def _claims_context(db: Session, area: str | None) -> dict[str, Any]:
    area = area if area in AREA_LABELS else None
    groups = claims_by_classification(db)
    article_ids = {c.article_id for g in groups for c in g["claims"] if c.article_id}
    demo_articles = (
        set(db.scalars(select(Article.id).where(Article.id.in_(article_ids), Article.is_demo.is_(True))))
        if article_ids
        else set()
    )
    area_counts: Counter[str] = Counter(c.area for g in groups for c in g["claims"])
    shown = []
    for group in groups:
        claims = [c for c in group["claims"] if area is None or c.area == area]
        shown.append(
            {
                "classification": group["classification"],
                "label": group["label"],
                "help": group["help"],
                "total": len(group["claims"]),
                "claims": [_claim_view(c, demo_articles) for c in claims],
            }
        )
    return {
        "groups": shown,
        "area": area,
        "area_label": AREA_LABELS.get(area, "") if area else "",
        "area_note": AREA_NOTES.get(area, "") if area else "",
        "area_options": [
            {
                "key": key,
                "label": label,
                "count": area_counts.get(key, 0),
                "url": _query_url("/finance/ai-reality-check", area=key),
            }
            for key, label in AREA_LABELS.items()
            if area_counts.get(key, 0)
        ],
        "all_count": sum(area_counts.values()),
        "shown_count": sum(len(g["claims"]) for g in shown),
    }


# --------------------------------------------------------------------------
# Regulation Watch views
# --------------------------------------------------------------------------


def _when_view(entry: dict[str, Any]) -> dict[str, Any]:
    raw = entry.get("date")
    if isinstance(raw, str) and len(raw) == 10:
        try:
            display = datetime.fromisoformat(raw).strftime("%B %d, %Y").replace(" 0", " ")
        except ValueError:
            display = raw
    elif isinstance(raw, str) and len(raw) == 7:
        try:
            display = datetime.strptime(raw, "%Y-%m").strftime("%B %Y")
        except ValueError:
            display = raw
    else:
        display = raw or ""
    return {
        "label": WHEN_LABELS.get(str(entry.get("kind")), "Date"),
        "date": display,
        "phrase": str(entry.get("phrase") or ""),
        "estimated": bool(entry.get("estimated")),
        "unknown": raw is None,
        "sentence": str(entry.get("sentence") or ""),
    }


def _regulation_view(item: dict[str, Any]) -> dict[str, Any]:
    citations = [{**c, "url": http_url(c.get("url"))} for c in item.get("citations") or [] if isinstance(c, dict)]
    return {
        **item,
        "url": http_url(item.get("url")),
        "when_view": [_when_view(w) for w in item.get("when") or [] if isinstance(w, dict)],
        "citations": citations,
        "coverage": [
            {**c, "url": http_url(c.get("url"))} for c in item.get("coverage") or [] if isinstance(c, dict)
        ],
    }


# --------------------------------------------------------------------------
# Section front
# --------------------------------------------------------------------------


@router.get("/finance")
def finance_front(request: Request, db: DB, user: CurrentUser):
    overview = finance_overview(db)
    briefs = published_briefs(db, vertical=FINANCE, limit=6)
    report_count = db.scalar(select(func.count(EarningsReport.id))) or 0
    in_review = 0
    if is_editor(user):
        in_review = (
            db.scalar(
                select(func.count(Brief.id)).where(
                    Brief.vertical == FINANCE,
                    Brief.status.in_([BriefStatus.IN_REVIEW.value, BriefStatus.DRAFT.value]),
                )
            )
            or 0
        )
    claim_counts = [
        {"classification": g["classification"], "label": g["label"], "count": g["count"]}
        for g in overview["ai_claims"]
    ]
    categories = [
        {"key": key, "label": CATEGORY_LABELS.get(key, key), "concepts": members[:4]}
        for key, members in grouped_concepts().items()
        if members
    ]
    ctx = {
        "overview": overview,
        "briefs": briefs,
        "tracks": TRACKS,
        "never": NEVER,
        "report_count": report_count,
        "latest_report": overview["earnings"][0] if overview["earnings"] else None,
        "claim_counts": claim_counts,
        "regulation": [_regulation_view(i) for i in overview["regulation"][:2]],
        "regulation_days": REGULATION_WINDOW_DAYS,
        "concept_categories": categories,
        "in_review": in_review,
        "demo": _has_demo(db),
        "follow_events": FOLLOW_EVENTS,
        "follow_question": FOLLOW_DEFAULT_QUESTION,
        "login_url": "/login?" + urlencode({"next": "/finance#follow"}),
        "prompts": FINANCE_PROMPTS,
    }
    return _page(request, "finance/index.html", ctx, section="overview")


# --------------------------------------------------------------------------
# Earnings Decoder
# --------------------------------------------------------------------------


@router.get("/finance/earnings")
def earnings_list(request: Request, db: DB, user: CurrentUser):
    reports = db.scalars(
        select(EarningsReport)
        .order_by(
            EarningsReport.filed_at.desc().nulls_last(),
            EarningsReport.retrieved_at.desc(),
            EarningsReport.id.desc(),
        )
        .limit(EARNINGS_LIST_LIMIT)
    ).all()
    ctx = {
        "reports": [_report_summary(r, user) for r in reports],
        "concepts": _concept_links(EARNINGS_CONCEPTS),
        "guidance_note": GUIDANCE_NOTE,
    }
    return _page(request, "finance/earnings.html", ctx, section="earnings")


@router.get("/finance/earnings/{report_id}")
def earnings_detail(request: Request, report_id: int, db: DB, user: CurrentUser):
    report = db.get(EarningsReport, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="We couldn't find that Earnings Decoder report.")
    ctx = {
        "report": _report_detail(report, user),
        "concepts": _concept_links(EARNINGS_CONCEPTS),
    }
    return _page(request, "finance/earnings_detail.html", ctx, section="earnings")


# --------------------------------------------------------------------------
# Funding Round Tracker
# --------------------------------------------------------------------------


@router.get("/finance/funding")
def funding_tracker(
    request: Request,
    db: DB,
    sector: str | None = None,
    stage: str | None = None,
    days: str | None = None,
):
    window = _int_choice(days, FUNDING_DAY_CHOICES, FUNDING_DEFAULT_DAYS)
    ctx = _funding_context(db, (sector or "").strip() or None, (stage or "").strip() or None, window)
    if is_htmx(request):
        return _partial(request, "finance/_funding_results.html", ctx, section="funding")
    ctx["concepts"] = _concept_links(
        ("funding-rounds", "dilution", "pre-money-vs-post-money-valuation", "term-sheet")
    )
    return _page(request, "finance/funding.html", ctx, section="funding")


# --------------------------------------------------------------------------
# AI-in-Finance Reality Check
# --------------------------------------------------------------------------


@router.get("/finance/ai-reality-check")
def ai_reality_check(request: Request, db: DB, area: str | None = None):
    ctx = _claims_context(db, (area or "").strip() or None)
    if is_htmx(request):
        return _partial(request, "finance/_claims.html", ctx, section="ai")
    ctx.update(
        {
            "classifications": [
                {"key": key, "label": label, "help": CLASSIFICATION_HELP.get(key, "")}
                for key, label in CLASSIFICATION_LABELS.items()
            ],
            "why": WHY_CLASSIFY,
            "area_notes": [{"label": AREA_LABELS[k], "note": v} for k, v in AREA_NOTES.items()],
            "regulatory_context": [{**c, "source_url": http_url(c.get("source_url"))} for c in REGULATORY_CONTEXT],
            "concepts": _concept_links(REALITY_CHECK_CONCEPTS),
        }
    )
    return _page(request, "finance/ai_reality_check.html", ctx, section="ai")


# --------------------------------------------------------------------------
# Regulation Watch
# --------------------------------------------------------------------------


@router.get("/finance/regulation")
def regulation(request: Request, db: DB, days: str | None = None):
    window = _int_choice(days, REGULATION_DAY_CHOICES, REGULATION_DEFAULT_DAYS)
    items = finance_regulation_items(regulation_watch(db, window))
    ctx = {
        "items": [_regulation_view(i) for i in items],
        "days": window,
        "day_options": REGULATION_DAY_CHOICES,
        "watch_note": WATCH_NOTE,
        "regulatory_context": [{**c, "source_url": http_url(c.get("source_url"))} for c in REGULATORY_CONTEXT],
        "concepts": _concept_links(("ai-washing", "investment-adviser-registration", "reg-bi-fiduciary-duty")),
    }
    return _page(request, "finance/regulation.html", ctx, section="regulation")


# --------------------------------------------------------------------------
# Finance concepts for learners
# --------------------------------------------------------------------------


@router.get("/finance/concepts")
def concepts_index(request: Request, db: DB):
    usage: Counter[str] = Counter()
    for _, mentions in _concept_index(db):
        usage.update({c.slug for c, _ in mentions})
    categories = [
        {"key": key, "label": CATEGORY_LABELS.get(key, key), "concepts": members}
        for key, members in grouped_concepts().items()
        if members
    ]
    ctx = {"categories": categories, "usage": usage, "concepts_disclaimer": CONCEPTS_DISCLAIMER}
    return _page(request, "finance/concepts.html", ctx, section="concepts")


@router.get("/finance/concepts/{slug}")
def concept_page(request: Request, slug: str, db: DB):
    concept = get_concept(slug)
    if concept is None:
        raise HTTPException(status_code=404, detail="We don't have an explainer for that finance concept yet.")
    if concept.slug != slug:
        # Short or alternative names ("drawdown", "10-q") resolve to the canonical page.
        return RedirectResponse(f"/finance/concepts/{concept.slug}", status_code=301)
    stories = [
        {"brief": brief, "phrase": phrase}
        for brief, mentions in _concept_index(db)
        for c, phrase in mentions
        if c.slug == concept.slug
    ][:CONCEPT_STORIES_LIMIT]
    ctx = {
        "concept": concept,
        "category_label": CATEGORY_LABELS.get(concept.category, concept.category),
        "related": related_concepts(concept),
        "stories": stories,
        "concepts_disclaimer": CONCEPTS_DISCLAIMER,
    }
    return _page(request, "finance/concept.html", ctx, section="concepts")
