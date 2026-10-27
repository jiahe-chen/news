"""SEC EDGAR access, the XBRL quarterly-metrics parser, and earnings report persistence.

Samples under tests/data/finance/ are trimmed real EDGAR responses for
IRADIMED Corp (CIK 1325618): companyfacts (six us-gaap tags, periods from
2024 on) and submissions (10-Q/10-K/8-K filings since 2025).
"""

from __future__ import annotations

import copy
import json
from datetime import date, datetime
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select

from newsren.finance import edgar
from newsren.finance.edgar import (
    EdgarError,
    QuarterlyMetric,
    build_earnings_report,
    change_pct,
    company_facts,
    company_submissions,
    extract_quarterly_metrics,
    filing_index_url,
    get_or_create_company,
    latest_periodic_filing,
    lookup_cik,
    normalize_cik,
    normalize_ticker,
    parse_company_tickers,
)
from newsren.models import Company, EarningsReport
from newsren.schemas import EarningsMetric, GuidanceItem

DATA = Path(__file__).parent / "data" / "finance"
CIK = "0001325618"
Q2_2026_ACCESSION = "0001628280-26-051402"
Q2_2026_INDEX = "https://www.sec.gov/Archives/edgar/data/1325618/000162828026051402/"


@pytest.fixture()
def facts() -> dict:
    return json.loads((DATA / "irmd_companyfacts.json").read_text())


@pytest.fixture()
def submissions() -> dict:
    return json.loads((DATA / "irmd_submissions.json").read_text())


@pytest.fixture(autouse=True)
def _clear_ticker_cache():
    edgar._ticker_cache.clear()
    yield
    edgar._ticker_cache.clear()


def by_key(metrics: list[QuarterlyMetric]) -> dict[str, QuarterlyMetric]:
    return {m.key: m for m in metrics}


def mock_client(routes: dict[str, httpx.Response], seen: list[httpx.Request] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return routes.get(str(request.url), httpx.Response(404))

    return httpx.Client(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------- identifiers and network


def test_identifier_normalization():
    assert normalize_cik(1325618) == CIK
    assert normalize_cik("CIK0001325618") == CIK
    assert normalize_ticker(" brk.b ") == "BRK-B"
    assert normalize_ticker("$irmd") == "IRMD"
    assert filing_index_url("1325618", Q2_2026_ACCESSION) == Q2_2026_INDEX
    with pytest.raises(ValueError):
        normalize_cik("not a cik")


def test_lookup_cik_uses_ticker_file_with_user_agent(settings):
    seen: list[httpx.Request] = []
    body = (DATA / "company_tickers_sample.json").read_bytes()
    client = mock_client({edgar.COMPANY_TICKERS_URL: httpx.Response(200, content=body)}, seen)

    assert lookup_cik("irmd", settings, client=client) == CIK
    assert lookup_cik("BRK.B", settings, client=client) == "0001067983"
    assert lookup_cik("ZZZZ", settings, client=client) is None
    assert len(seen) == 1, "ticker file is cached after the first request"
    assert seen[0].headers["User-Agent"] == settings.user_agent
    assert "newsroom@newsren.example" in settings.user_agent


def test_parse_company_tickers_skips_bad_rows():
    mapping = parse_company_tickers(
        {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, "1": {"x": 1}}
    )
    assert mapping == {"AAPL": ("0000320193", "Apple Inc.")}


def test_network_failures_raise_edgar_error(settings):
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(EdgarError, match="request failed"):
        lookup_cik("IRMD", settings, client=httpx.Client(transport=httpx.MockTransport(boom)))
    with pytest.raises(EdgarError, match="404"):
        company_facts(CIK, settings, client=mock_client({}))
    bad = mock_client({edgar.SUBMISSIONS_URL.format(cik=CIK): httpx.Response(503)})
    with pytest.raises(EdgarError, match="503"):
        company_submissions(CIK, settings, client=bad)


def test_company_facts_and_submissions_fetch_padded_cik_urls(settings, facts, submissions):
    client = mock_client(
        {
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0001325618.json": httpx.Response(200, json=facts),
            "https://data.sec.gov/submissions/CIK0001325618.json": httpx.Response(200, json=submissions),
        }
    )
    assert company_facts(1325618, settings, client=client)["entityName"] == "IRADIMED CORPORATION"
    assert company_submissions("1325618", settings, client=client)["tickers"] == ["IRMD"]


# --------------------------------------------------------------------------- XBRL parser


def test_latest_filing_is_most_recent_10q(facts):
    filing = latest_periodic_filing(facts)
    assert filing is not None
    assert (filing.form, filing.accession, filing.label) == ("10-Q", Q2_2026_ACCESSION, "FY2026Q2")
    assert filing.period == "2026-04-01..2026-06-30"
    assert filing.filed == date(2026, 7, 31)
    assert not filing.is_annual


def test_quarterly_metrics_compare_with_year_earlier_and_prior_quarter(facts):
    metrics = extract_quarterly_metrics(facts)
    assert [m.key for m in metrics] == ["revenue", "gross_profit", "operating_income", "net_income", "eps_diluted"]
    assert all(isinstance(m, EarningsMetric) for m in metrics)
    m = by_key(metrics)

    revenue = m["revenue"]
    assert revenue.value == 20_466_000
    assert revenue.period == "2026-04-01..2026-06-30"
    assert revenue.comparison == "same_quarter_prior_year"
    assert revenue.prior_value == 20_409_000  # comparative column of the same 10-Q
    assert revenue.prior_period == "2025-04-01..2025-06-30"
    assert revenue.change_pct == 0.3
    assert revenue.prior_quarter_value == 21_979_000
    assert revenue.prior_quarter_period == "2026-01-01..2026-03-31"
    assert revenue.prior_quarter_change_pct == -6.9
    assert revenue.xbrl_tag == "us-gaap:Revenues"
    assert (revenue.form, revenue.filed, revenue.accession) == ("10-Q", "2026-07-31", Q2_2026_ACCESSION)
    assert not revenue.derived

    assert m["net_income"].value == 5_218_000 and m["net_income"].prior_value == 5_774_000
    assert m["net_income"].change_pct == -9.6
    eps = m["eps_diluted"]
    assert (eps.value, eps.prior_value, eps.unit, eps.change_pct) == (0.41, 0.45, "USD/share", -8.9)


def test_metric_provenance_points_to_filing_and_names_tag_form_period(facts):
    revenue = by_key(extract_quarterly_metrics(facts))["revenue"]
    assert revenue.source_url == Q2_2026_INDEX
    assert revenue.prior_source_url == Q2_2026_INDEX
    # The prior quarter was first reported in the Q1 10-Q (a different filing).
    assert (
        revenue.prior_quarter_source_url == "https://www.sec.gov/Archives/edgar/data/1325618/000162828026029325/"
    )
    quote = revenue.source_quote
    assert quote.startswith("us-gaap:Revenues, 10-Q filed 2026-07-31, period 2026-04-01..2026-06-30: $20,466,000.")
    assert "period 2025-04-01..2025-06-30: $20,409,000 (comparative column of the same filing)" in quote
    assert "10-Q filed 2026-05-01, period 2026-01-01..2026-03-31: $21,979,000" in quote


def test_stored_metric_dict_round_trips_as_earnings_metric(facts):
    dumped = by_key(extract_quarterly_metrics(facts))["eps_diluted"].model_dump()
    metric = EarningsMetric.model_validate(dumped)
    assert metric.value == 0.41 and metric.source_url == Q2_2026_INDEX
    assert dumped["prior_quarter_value"] == 0.45  # extra provenance keys survive in the stored dict


def test_10k_anchor_derives_fourth_quarter_for_flow_items_only(facts):
    as_of = date(2026, 3, 10)  # after the FY2025 10-K, before the Q1 2026 10-Q
    filing = latest_periodic_filing(facts, as_of)
    assert filing is not None and (filing.form, filing.label, filing.is_annual) == ("10-K", "FY2025", True)

    m = by_key(extract_quarterly_metrics(facts, as_of=as_of))
    revenue = m["revenue"]
    # FY2025 revenue 83,814,000 (10-K, thousands) minus nine months 61,122,101 (10-Q, dollars),
    # rounded to the coarser precision.
    assert revenue.derived and revenue.value == 22_692_000
    assert revenue.period == "2025-10-01..2025-12-31"
    assert revenue.prior_value == 19_389_000  # FY2024 73,242,000 minus nine months 53,852,954
    assert revenue.change_pct == 17.0
    assert revenue.prior_quarter_value == 21_202_064 and revenue.prior_quarter_period == "2025-07-01..2025-09-30"
    assert "minus nine months 2025-01-01..2025-09-30" in revenue.source_quote
    assert "derived; the quarter is not tagged separately" in revenue.source_quote
    assert revenue.source_url == "https://www.sec.gov/Archives/edgar/data/1325618/000110465926024691/"

    # Per-share figures do not add across quarters, so no derived Q4 EPS...
    assert "eps_diluted" not in m
    # ...but the fiscal-year figures from the 10-K are reported with the prior year.
    eps_fy = m["eps_diluted_fy"]
    assert (eps_fy.value, eps_fy.prior_value, eps_fy.change_pct) == (1.75, 1.5, 16.7)
    assert eps_fy.period_kind == "fiscal_year" and eps_fy.comparison == "prior_fiscal_year"
    assert m["revenue_fy"].value == 83_814_000 and m["revenue_fy"].prior_value == 73_242_000


def test_earlier_as_of_uses_that_quarters_filing(facts):
    m = by_key(extract_quarterly_metrics(facts, as_of=date(2025, 11, 10)))
    revenue = m["revenue"]
    assert revenue.period == "2025-07-01..2025-09-30" and revenue.value == 21_202_064
    assert revenue.prior_value == 18_325_959 and revenue.change_pct == 15.7
    assert revenue.prior_quarter_value == 20_409_400


def test_revenue_tag_fallback_chain(facts):
    trimmed = copy.deepcopy(facts)
    del trimmed["facts"]["us-gaap"]["Revenues"]
    revenue = by_key(extract_quarterly_metrics(trimmed))["revenue"]
    assert revenue.xbrl_tag == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    assert revenue.value == 20_466_000
    assert revenue.source_quote.startswith("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax, 10-Q")


def test_non_periodic_forms_and_late_filings_are_ignored(facts):
    # A DEF 14A (proxy statement) filed 2026-04-30 repeats FY2025 net income; it must not become the anchor.
    filing = latest_periodic_filing(facts, as_of=date(2026, 4, 30))
    assert filing is not None and filing.form == "10-K"
    assert extract_quarterly_metrics({"cik": 1, "facts": {}}) == []
    assert latest_periodic_filing({}) is None


@pytest.mark.parametrize(
    ("current", "prior", "expected"),
    [
        (110.0, 100.0, 10.0),
        (90.0, 100.0, -10.0),
        (-5.0, 10.0, None),
        (5.0, -10.0, None),
        (5.0, 0.0, None),
        (5.0, None, None),
    ],
)
def test_change_pct_is_omitted_when_not_meaningful(current, prior, expected):
    assert change_pct(current, prior) == expected


# --------------------------------------------------------------------------- persistence


def test_get_or_create_company_matches_by_cik_then_ticker(session):
    company = get_or_create_company(session, "IRMD", ticker="irmd", cik="1325618")
    assert (company.ticker, company.cik) == ("IRMD", CIK)
    assert get_or_create_company(session, "IRMD", cik=1325618).id == company.id
    assert get_or_create_company(session, "IRADIMED Corp", ticker="IRMD").id == company.id
    assert company.name == "IRADIMED Corp", "a placeholder name equal to the ticker is replaced"
    get_or_create_company(session, "Another name", cik=1325618)
    assert company.name == "IRADIMED Corp", "a real name is never overwritten"

    named = get_or_create_company(session, "Northwind Analytics", sector="Software")
    assert get_or_create_company(session, "northwind analytics").id == named.id
    filled = get_or_create_company(session, "Northwind Analytics", ticker="NWND")
    assert filled.id == named.id and filled.ticker == "NWND" and filled.sector == "Software"
    other = get_or_create_company(session, "Northwind Analytics", ticker="NWX")
    assert other.id != named.id, "a conflicting ticker means a different company"
    assert session.scalar(select(func.count()).select_from(Company)) == 3


def test_build_earnings_report_from_saved_samples(session, settings, facts, submissions):
    company = get_or_create_company(session, "IRMD", ticker="IRMD", cik=CIK)
    report = build_earnings_report(session, company, settings, facts=facts, submissions=submissions)

    assert isinstance(report, EarningsReport) and report.id is not None
    assert (report.fiscal_period, report.form, report.accession) == ("FY2026Q2", "10-Q", Q2_2026_ACCESSION)
    assert report.filed_at == datetime(2026, 7, 31)
    assert report.filing_url == Q2_2026_INDEX + "irmd-20260630.htm"
    assert report.guidance == []
    assert report.data_delay_note.startswith(
        "Figures from SEC XBRL company facts; EDGAR data can lag filings and is not real-time market data. Retrieved "
    )
    metrics = [EarningsMetric.model_validate(m) for m in report.metrics]
    assert {m.name for m in metrics} >= {"Revenue", "Net income (loss)", "Diluted EPS"}
    assert all(m.source_url.startswith("https://www.sec.gov/Archives/edgar/data/1325618/") for m in metrics)

    release = report.narrative["earnings_release"]
    assert release["form"] == "8-K/A" and "2.02" in release["items"]
    assert release["url"].endswith("/000162828026057617/irmd-20260731.htm")
    assert "not part of XBRL" in report.narrative["guidance_note"]
    assert report.narrative["filing"]["period"] == "2026-04-01..2026-06-30"
    # Company details are filled from the filings.
    assert company.name == "IRADIMED CORPORATION"
    assert company.sector == "Surgical & Medical Instruments & Apparatus"


@pytest.mark.parametrize(
    ("as_of", "expected_accession"),
    [
        (None, "0001628280-26-057617"),  # Q2 2026 10-Q -> 8-K/A filed 2026-08-18
        (date(2026, 3, 10), "0001104659-26-012356"),  # FY2025 10-K -> 8-K filed 2026-02-10
        (date(2025, 11, 10), "0001104659-25-105298"),  # Q3 2025 10-Q -> 8-K filed 2025-11-03
        # FY2024 10-K (filed 2025-03-06): the Feb 13 release, not the preliminary Jan 10 one.
        (date(2025, 3, 10), "0001558370-25-000907"),
    ],
)
def test_find_earnings_release_matches_the_period(facts, submissions, as_of, expected_accession):
    filing = latest_periodic_filing(facts, as_of)
    assert filing is not None
    release = edgar.find_earnings_release(submissions, filing, CIK)
    assert release is not None and release["accession"] == expected_accession
    assert "2.02" in release["items"]


def test_build_earnings_report_is_an_upsert_that_keeps_guidance(session, settings, facts):
    company = get_or_create_company(session, "IRADIMED", ticker="IRMD", cik=CIK)
    guidance = [GuidanceItem(metric="Revenue", guidance="$20.0M-$20.5M", source_url="https://example.com/8k")]
    first = build_earnings_report(session, company, settings, facts=facts, guidance=guidance)
    first.narrative = {**first.narrative, "brief_note": "kept"}
    second = build_earnings_report(session, company, settings, facts=facts)
    assert second.id == first.id
    assert session.scalar(select(func.count()).select_from(EarningsReport)) == 1
    assert second.guidance[0]["guidance"] == "$20.0M-$20.5M"
    assert second.narrative["brief_note"] == "kept"
    assert second.narrative["earnings_release"] is None  # no submissions supplied


def test_build_earnings_report_fetches_when_no_facts_given(session, settings, facts, submissions, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(edgar, "company_facts", lambda cik, s: calls.append(f"facts:{cik}") or facts)
    monkeypatch.setattr(edgar, "company_submissions", lambda cik, s: calls.append(f"subs:{cik}") or submissions)
    company = get_or_create_company(session, "IRMD", ticker="IRMD", cik=CIK)
    report = build_earnings_report(session, company, settings)
    assert calls == [f"facts:{CIK}", f"subs:{CIK}"]
    assert report.narrative["earnings_release"] is not None


def test_build_earnings_report_requires_cik_or_ticker(session, settings, facts):
    company = get_or_create_company(session, "Private Startup Inc")
    with pytest.raises(EdgarError, match="No SEC CIK"):
        build_earnings_report(session, company, settings, facts=facts)
    company2 = get_or_create_company(session, "Empty Co", cik="0000000001")
    with pytest.raises(EdgarError, match="No 10-Q/10-K"):
        build_earnings_report(session, company2, settings, facts={"cik": 1, "facts": {"us-gaap": {}}})
