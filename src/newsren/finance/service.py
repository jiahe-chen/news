"""Finance section service: the Finance page's data and the finance refresh job.

``finance_overview`` gathers everything the Finance page shows in one dict:
the not-investment-advice notice, the latest Earnings Decoder reports, funding
trends, AI-in-finance claims grouped by classification, Regulation Watch
items, and the concepts library. ``refresh_finance`` is the pipeline step
that keeps the trackers current: it seeds the plan's AI-in-finance claims,
logs funding rounds and classifies AI-in-finance claims from new articles,
refreshes earnings reports from SEC EDGAR for tracked companies (only when
``network`` is True), and writes an Earnings and Filings Brief for every
report that lacks a current one. Both functions flush but never commit.
"""

from __future__ import annotations

import logging
import time as _time
from datetime import datetime
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from newsren.agents.base import Orchestrator
from newsren.config import Settings, get_settings
from newsren.editorial.guardrails import FINANCE_DISCLAIMER
from newsren.finance.ai_claims import (
    AREA_LABELS,
    CLASSIFICATION_HELP,
    CLASSIFICATION_LABELS,
    seed_ai_claims,
    update_ai_claims,
)
from newsren.finance.concepts import CATEGORY_LABELS, CONCEPTS, concepts_by_category
from newsren.finance.concepts import DISCLAIMER as CONCEPTS_DISCLAIMER
from newsren.finance.edgar import (
    EdgarError,
    build_earnings_report,
    earnings_brief,
    fiscal_label,
    format_value,
    period_text,
)
from newsren.finance.funding import funding_trends, update_funding_tracker
from newsren.finance.regulation import WATCH_NOTE, finance_regulation_items, regulation_watch
from newsren.models import AIFinanceClaim, Brief, Company, EarningsReport, utcnow
from newsren.textutil import iso

log = logging.getLogger(__name__)

FUNDING_WINDOW_DAYS = 90
REGULATION_WINDOW_DAYS = 60
EARNINGS_LIMIT = 8
CLAIMS_PER_CLASSIFICATION = 6
MAX_EDGAR_COMPANIES = 25
# SEC asks automated clients to stay well under 10 requests per second.
EDGAR_PAUSE_SECONDS = 0.25
DATA_NOTE = (
    "Earnings figures come from SEC EDGAR filings (XBRL company facts) and companies' own releases; funding "
    "rounds and AI-in-finance claims come from public announcements and news reports. Every item shows its "
    "source and retrieval time. The Newsren shows no live market prices, and EDGAR data can lag filings."
)


# --------------------------------------------------------------------------
# Finance page
# --------------------------------------------------------------------------


def _metric_view(metric: dict[str, Any]) -> dict[str, Any]:
    unit = str(metric.get("unit") or "USD")
    return {
        "key": metric.get("key") or "",
        "name": metric.get("name") or "",
        "value": metric.get("value"),
        "display": format_value(metric.get("value"), unit),
        "unit": unit,
        "period": metric.get("period") or "",
        "period_text": period_text(metric.get("period")),
        "prior_value": metric.get("prior_value"),
        "prior_display": format_value(metric.get("prior_value"), unit)
        if metric.get("prior_value") is not None
        else "",
        "prior_period": metric.get("prior_period") or "",
        "comparison": metric.get("comparison") or "",
        "change_pct": metric.get("change_pct"),
        "prior_quarter_value": metric.get("prior_quarter_value"),
        "prior_quarter_change_pct": metric.get("prior_quarter_change_pct"),
        "derived": bool(metric.get("derived")),
        "source_url": metric.get("source_url") or "",
        "source_quote": metric.get("source_quote") or "",
    }


def _earnings_view(report: EarningsReport, brief: Brief | None) -> dict[str, Any]:
    company = report.company
    narrative = report.narrative or {}
    metrics = [m for m in report.metrics or [] if isinstance(m, dict)]
    return {
        "report_id": report.id,
        "company": company.name if company else "",
        "company_id": report.company_id,
        "ticker": company.ticker if company else None,
        "cik": company.cik if company else None,
        "sector": company.sector if company else None,
        "fiscal_period": report.fiscal_period,
        "fiscal_label": fiscal_label(report.fiscal_period),
        "form": report.form,
        "filed_at": iso(report.filed_at),
        "accession": report.accession,
        "filing_url": report.filing_url,
        "retrieved_at": iso(report.retrieved_at),
        "data_delay_note": report.data_delay_note,
        "metrics": [_metric_view(m) for m in metrics],
        "guidance": [g for g in report.guidance or [] if isinstance(g, dict)],
        "guidance_available": bool(report.guidance),
        "guidance_note": narrative.get("guidance_note"),
        "earnings_release": narrative.get("earnings_release"),
        "brief_id": brief.id if brief else None,
        "brief_status": brief.status if brief else None,
        "brief_headline": brief.headline if brief else None,
        "is_demo": bool(report.is_demo),
    }


def _latest_reports(session: Session, limit: int) -> list[EarningsReport]:
    """The most recent report per company, newest filing first."""
    rows = session.scalars(
        select(EarningsReport).order_by(
            EarningsReport.filed_at.desc().nulls_last(),
            EarningsReport.retrieved_at.desc(),
            EarningsReport.id.desc(),
        )
    )
    seen: set[int] = set()
    latest: list[EarningsReport] = []
    for report in rows:
        if report.company_id in seen:
            continue
        seen.add(report.company_id)
        latest.append(report)
        if len(latest) >= limit:
            break
    return latest


def _claim_view(claim: AIFinanceClaim) -> dict[str, Any]:
    return {
        "id": claim.id,
        "firm": claim.firm,
        "area": claim.area,
        "area_label": AREA_LABELS.get(claim.area, claim.area),
        "claim_text": claim.claim_text,
        "classification": claim.classification,
        "classification_label": CLASSIFICATION_LABELS.get(claim.classification, claim.classification),
        "performance_label": claim.performance_label,
        "period": claim.period,
        "benchmark": claim.benchmark,
        "costs_assumed": claim.costs_assumed,
        "authors_caveats": claim.authors_caveats,
        "evidence_note": claim.evidence_note,
        "source_url": claim.source_url,
        "source_title": claim.source_title,
        "published_at": iso(claim.published_at),
        "retrieved_at": iso(claim.retrieved_at),
        "article_id": claim.article_id,
        "is_seed": claim.is_seed,
    }


def _claim_groups(session: Session, per_group: int) -> list[dict[str, Any]]:
    rows = list(
        session.scalars(
            select(AIFinanceClaim).order_by(
                AIFinanceClaim.published_at.desc().nulls_last(), AIFinanceClaim.id.desc()
            )
        )
    )
    groups = []
    for key, label in CLASSIFICATION_LABELS.items():
        members = [r for r in rows if r.classification == key]
        groups.append(
            {
                "classification": key,
                "label": label,
                "help": CLASSIFICATION_HELP.get(key, ""),
                "count": len(members),
                "claims": [_claim_view(r) for r in members[:per_group]],
            }
        )
    return groups


def _concept_views() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    concepts = [
        {
            "slug": c.slug,
            "term": c.term,
            "short": c.short,
            "category": c.category,
            "category_label": CATEGORY_LABELS.get(c.category, c.category),
            "related": list(c.related),
        }
        for c in CONCEPTS
    ]
    categories = [
        {"key": key, "label": CATEGORY_LABELS.get(key, key), "slugs": [c.slug for c in members]}
        for key, members in concepts_by_category().items()
        if members
    ]
    return concepts, categories


def finance_overview(session: Session, *, now: datetime | None = None) -> dict[str, Any]:
    """Everything the Finance page shows, as plain data.

    Keys:

    * ``disclaimer`` — ``FINANCE_DISCLAIMER`` (shown on every finance page);
      ``data_note`` — where the figures come from and their delays;
    * ``earnings`` — the latest Earnings Decoder report per company (newest
      first): company, ticker, form, fiscal period, filing links, ``metrics``
      (each with a ready-to-show ``display`` value, its comparison, change,
      and ``source_url``/``source_quote``), ``guidance`` and
      ``guidance_available``, the linked brief's id/status/headline, and
      ``data_delay_note``;
    * ``funding`` — ``funding_trends(session, 90)`` (by sector, stage, and
      investor activity; never a ranking of startups);
    * ``ai_claims`` — AI-in-finance claims grouped by classification
      (announcement, pilot, backtest, live deployment, regulatory finding):
      ``[{classification, label, help, count, claims: [...]}]``, newest first;
    * ``regulation`` — finance-relevant ``regulation_watch(session, 60)``
      items; ``regulation_note`` — the Regulation Watch caveat;
    * ``concepts`` — the learners' concept library ``[{slug, term, short,
      category, category_label, related}]``, ``concept_categories``, and
      ``concepts_disclaimer``;
    * ``counts`` and ``generated_at``.
    """
    now = now or utcnow()
    reports = _latest_reports(session, EARNINGS_LIMIT)
    briefs = {
        b.id: b
        for b in session.scalars(select(Brief).where(Brief.id.in_([r.brief_id for r in reports if r.brief_id])))
    }
    earnings = [_earnings_view(r, briefs.get(r.brief_id) if r.brief_id else None) for r in reports]
    funding = funding_trends(session, FUNDING_WINDOW_DAYS, now=now)
    claims = _claim_groups(session, CLAIMS_PER_CLASSIFICATION)
    regulation = finance_regulation_items(regulation_watch(session, REGULATION_WINDOW_DAYS, now=now))
    concepts, categories = _concept_views()
    return {
        "generated_at": iso(now),
        "disclaimer": FINANCE_DISCLAIMER,
        "data_note": DATA_NOTE,
        "earnings": earnings,
        "funding": funding,
        "ai_claims": claims,
        "regulation": regulation,
        "regulation_note": WATCH_NOTE,
        "concepts": concepts,
        "concept_categories": categories,
        "concepts_disclaimer": CONCEPTS_DISCLAIMER,
        "counts": {
            "earnings_reports": len(earnings),
            "funding_rounds": funding["total_rounds"],
            "ai_claims": sum(g["count"] for g in claims),
            "regulatory_items": len(regulation),
            "concepts": len(concepts),
        },
    }


# --------------------------------------------------------------------------
# Refresh job
# --------------------------------------------------------------------------


def _refresh_reports(session: Session, settings: Settings, result: dict[str, Any]) -> list[EarningsReport]:
    """Rebuild the latest EarningsReport from EDGAR for each company with a CIK or ticker."""
    companies = session.scalars(
        select(Company)
        .where(or_(Company.cik.is_not(None), Company.ticker.is_not(None)))
        .order_by(Company.id)
        .limit(MAX_EDGAR_COMPANIES)
    ).all()
    refreshed: list[EarningsReport] = []
    for index, company in enumerate(companies):
        if index:
            _time.sleep(EDGAR_PAUSE_SECONDS)
        result["companies_checked"] += 1
        try:
            with session.begin_nested():
                refreshed.append(build_earnings_report(session, company, settings))
        except (EdgarError, ValueError) as exc:
            log.warning("EDGAR refresh failed for %s: %s", company.name, exc)
            result["errors"].append(f"{company.name}: {exc}")
    result["earnings_reports_refreshed"] = len(refreshed)
    return refreshed


def refresh_finance(
    session: Session,
    orchestrator: Orchestrator | None = None,
    settings: Settings | None = None,
    network: bool = False,
) -> dict[str, Any]:
    """Refresh the finance trackers; returns counts of what changed and any errors.

    Runs ``seed_ai_claims``, ``update_funding_tracker`` and
    ``update_ai_claims`` (with ``orchestrator`` when given), then, only when
    ``network`` is True, rebuilds the latest ``EarningsReport`` from SEC
    EDGAR for companies in the companies table that have a CIK or ticker
    (a failure for one company is recorded in ``errors`` and the rest
    continue). Finally it writes an earnings brief (sent to human review) for
    every report without one and for every refreshed report whose evidence
    changed. Idempotent: a second run with nothing new returns zero counts.
    """
    settings = settings or get_settings()
    result: dict[str, Any] = {
        "network": bool(network),
        "ai_claims_seeded": 0,
        "funding_rounds_added": 0,
        "ai_claims_added": 0,
        "companies_checked": 0,
        "earnings_reports_refreshed": 0,
        "earnings_briefs": 0,
        "errors": [],
    }
    result["ai_claims_seeded"] = seed_ai_claims(session)
    result["funding_rounds_added"] = len(update_funding_tracker(session, orchestrator=orchestrator))
    result["ai_claims_added"] = len(update_ai_claims(session, orchestrator=orchestrator))

    refreshed = _refresh_reports(session, settings, result) if network else []
    pending = {r.id: r for r in refreshed}
    for report in session.scalars(select(EarningsReport).where(EarningsReport.brief_id.is_(None))).all():
        pending.setdefault(report.id, report)
    for report in pending.values():
        if not any(isinstance(m, dict) and m.get("value") is not None for m in report.metrics or []):
            continue
        before = report.brief_id
        try:
            with session.begin_nested():
                brief = earnings_brief(session, report, orchestrator)
        except (EdgarError, ValueError) as exc:
            log.warning("Earnings brief failed for report %s: %s", report.id, exc)
            result["errors"].append(f"earnings brief for report {report.id}: {exc}")
            continue
        if brief.id != before:
            result["earnings_briefs"] += 1
    session.flush()
    return result
