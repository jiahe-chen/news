"""Finance page data (``finance_overview``) and the finance refresh job (``refresh_finance``).

Offline only: EDGAR access is replaced by the saved IRADIMED samples under
tests/data/finance/, and any unexpected network call fails the test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import func, select

from newsren.agents.base import Orchestrator
from newsren.editorial.guardrails import FINANCE_DISCLAIMER
from newsren.finance import edgar, service
from newsren.finance.ai_claims import CLASSIFICATION_LABELS, SEED_CLAIMS
from newsren.finance.concepts import CONCEPTS
from newsren.finance.edgar import EdgarError, build_earnings_report, get_or_create_company
from newsren.finance.service import finance_overview, refresh_finance
from newsren.ingest.fixtures import load_demo_data
from newsren.models import AIFinanceClaim, Brief, BriefStatus, EarningsReport, FundingRound

DATA = Path(__file__).parent / "data" / "finance"
FACTS = json.loads((DATA / "irmd_companyfacts.json").read_text())
SUBMISSIONS = json.loads((DATA / "irmd_submissions.json").read_text())


@pytest.fixture()
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("no network access expected")

    monkeypatch.setattr(edgar, "_get_json", refuse)
    monkeypatch.setattr(service, "EDGAR_PAUSE_SECONDS", 0)


@pytest.fixture()
def finance_db(session, settings, no_network):
    """Demo corpus plus IRADIMED's latest earnings report built from the saved EDGAR samples."""
    load_demo_data(session)
    company = get_or_create_company(session, "IRADIMED CORP", ticker="IRMD", cik="1325618")
    build_earnings_report(session, company, settings, facts=FACTS, submissions=SUBMISSIONS)
    return session


def counts(session) -> dict[str, int]:
    return {
        model.__name__: session.scalar(select(func.count()).select_from(model))
        for model in (FundingRound, AIFinanceClaim, Brief, EarningsReport)
    }


# --------------------------------------------------------------------------- refresh_finance


def test_refresh_finance_offline_runs_every_tracker_and_is_idempotent(finance_db):
    session = finance_db
    first = refresh_finance(session)

    assert first["network"] is False and first["errors"] == []
    assert first["ai_claims_seeded"] == len(SEED_CLAIMS)
    assert first["funding_rounds_added"] >= 4
    assert first["ai_claims_added"] >= 1
    assert first["companies_checked"] == first["earnings_reports_refreshed"] == 0
    assert first["earnings_briefs"] == 1
    report = session.scalars(select(EarningsReport)).one()
    brief = session.get(Brief, report.brief_id)
    assert (brief.kind, brief.status) == ("earnings", BriefStatus.IN_REVIEW)

    before = counts(session)
    second = refresh_finance(session)
    assert {k: v for k, v in second.items() if k not in ("network", "errors")} == {
        "ai_claims_seeded": 0,
        "funding_rounds_added": 0,
        "ai_claims_added": 0,
        "companies_checked": 0,
        "earnings_reports_refreshed": 0,
        "earnings_briefs": 0,
    }
    assert second["errors"] == []
    assert counts(session) == before


def test_refresh_finance_accepts_the_pipeline_call(finance_db, settings):
    # newsren.pipeline calls refresh_finance(session, orchestrator, settings, network) positionally.
    result = refresh_finance(finance_db, Orchestrator(finance_db), settings, False)
    assert result["earnings_briefs"] == 1 and result["errors"] == []


def test_refresh_finance_with_network_refreshes_tracked_companies(finance_db, settings, monkeypatch):
    session = finance_db
    monkeypatch.setattr(edgar, "company_facts", lambda cik, settings=None, client=None: FACTS)
    monkeypatch.setattr(edgar, "company_submissions", lambda cik, settings=None, client=None: SUBMISSIONS)
    monkeypatch.setattr(edgar, "lookup_cik", lambda ticker, settings=None, client=None: None)
    get_or_create_company(session, "Unlisted Example Co", ticker="ZZZZ")
    get_or_create_company(session, "Corvane Semiconductor")  # no identifiers: never sent to EDGAR

    first = refresh_finance(session, network=True)
    assert first["companies_checked"] == 2
    assert first["earnings_reports_refreshed"] == 1
    assert first["earnings_briefs"] == 1
    assert len(first["errors"]) == 1 and "Unlisted Example Co" in first["errors"][0]

    second = refresh_finance(session, network=True)
    assert second["earnings_reports_refreshed"] == 1
    assert second["earnings_briefs"] == 0, "unchanged evidence keeps the existing brief"
    assert session.scalar(select(func.count()).select_from(EarningsReport)) == 1


def test_edgar_errors_never_escape_refresh(session, settings, monkeypatch, no_network):
    def unavailable(cik, settings=None, client=None):
        raise EdgarError("EDGAR returned HTTP 503")

    monkeypatch.setattr(edgar, "company_facts", unavailable)
    get_or_create_company(session, "IRADIMED CORP", ticker="IRMD", cik="1325618")
    result = refresh_finance(session, network=True)
    assert result["earnings_reports_refreshed"] == 0
    assert result["errors"] == ["IRADIMED CORP: EDGAR returned HTTP 503"]


# --------------------------------------------------------------------------- finance_overview


def test_finance_overview_shape(finance_db):
    session = finance_db
    refresh_finance(session)
    overview = finance_overview(session)

    assert set(overview) == {
        "generated_at",
        "disclaimer",
        "data_note",
        "earnings",
        "funding",
        "ai_claims",
        "regulation",
        "regulation_note",
        "concepts",
        "concept_categories",
        "concepts_disclaimer",
        "counts",
    }
    assert overview["disclaimer"] == FINANCE_DISCLAIMER
    json.dumps(overview)  # plain data, ready for templates and APIs

    (report,) = overview["earnings"]
    assert (report["company"], report["form"], report["fiscal_period"]) == ("IRADIMED CORP", "10-Q", "FY2026Q2")
    assert report["brief_status"] == BriefStatus.IN_REVIEW and report["brief_id"]
    assert report["guidance_available"] is False and report["data_delay_note"]
    revenue = report["metrics"][0]
    assert (revenue["name"], revenue["display"], revenue["change_pct"]) == ("Revenue", "$20,466,000", 0.3)
    assert revenue["source_url"].startswith("https://www.sec.gov/Archives/edgar/data/1325618/")

    funding = overview["funding"]
    assert funding["window_days"] == 90 and funding["total_rounds"] >= 4
    assert {"by_sector", "by_stage", "top_investors", "recent", "note"} <= set(funding)

    groups = overview["ai_claims"]
    assert [g["classification"] for g in groups] == list(CLASSIFICATION_LABELS)
    assert sum(g["count"] for g in groups) == overview["counts"]["ai_claims"] >= len(SEED_CLAIMS)
    backtests = next(g for g in groups if g["classification"] == "backtest")["claims"]
    # Backtest-class claims are always labelled (backtest, or simulated for paper trading), never live.
    assert backtests and all(c["performance_label"] in ("backtest", "simulated") for c in backtests)
    assert all(c["source_url"] for g in groups for c in g["claims"])

    regulation = overview["regulation"]
    assert {(i["regulator"], i["action_type"]) for i in regulation} == {
        ("Markets Integrity Commission", "enforcement"),
        ("Broker-Dealer Standards Authority", "guidance"),
    }
    assert all(i["regulator_kind"] != "education agency" for i in regulation)

    slugs = {c["slug"] for c in overview["concepts"]}
    assert {"backtesting", "overfitting", "sharpe-ratio", "maximum-drawdown", "dilution", "term-sheet"} <= slugs
    assert len(overview["concepts"]) == len(CONCEPTS) == overview["counts"]["concepts"]
    assert {c["key"] for c in overview["concept_categories"]} >= {
        "investing",
        "startups",
        "earnings",
        "regulation",
    }


def test_finance_overview_on_an_empty_database(session):
    overview = finance_overview(session)
    assert overview["earnings"] == [] and overview["regulation"] == []
    assert overview["funding"]["total_rounds"] == 0
    assert all(g["count"] == 0 for g in overview["ai_claims"])
    assert overview["counts"]["concepts"] == len(CONCEPTS)
    assert overview["disclaimer"] == FINANCE_DISCLAIMER
