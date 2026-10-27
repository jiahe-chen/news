"""Finance editorial products: sourced figures, labels, the disclaimer, and guardrail-clean text (offline mode)."""

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from newsren.agents.base import Orchestrator
from newsren.agents.products import PRODUCTS, run_product
from newsren.editorial.guardrails import FINANCE_DISCLAIMER, lint_text
from newsren.llm.base import LLMResult
from newsren.models import Company, EarningsReport, FundingRound, GeneratedContent, utcnow
from newsren.schemas import ProductOut, ProductSection


def _load_newsroom():
    name = "explain_newsroom"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).parent / "data" / "explain" / "newsroom.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture()
def newsroom(factory):
    return _load_newsroom().Newsroom(factory)


FINANCE_CASES = [
    ("earnings", "earnings_decoder"),
    ("funding", "funding_round_tracker"),
    ("backtest", "ai_finance_reality_check"),
    ("enforcement", "ai_finance_reality_check"),
    ("enforcement", "regulation_watch"),
]


def _out(row: GeneratedContent) -> ProductOut:
    return ProductOut.model_validate(row.content)


def _all_text(out: ProductOut) -> str:
    return " ".join([out.title, out.summary, *(f"{s.heading}. {s.body}" for s in out.sections), *out.caveats])


@pytest.mark.parametrize(("brief_name", "key"), FINANCE_CASES)
def test_finance_products_carry_disclaimer_and_pass_lint(session, newsroom, brief_name, key):
    out = _out(run_product(session, key, newsroom.brief(brief_name)))
    assert PRODUCTS[key].is_finance
    assert FINANCE_DISCLAIMER in out.caveats
    assert lint_text(_all_text(out), "finance") == []


def _quotes(text: str) -> list[str]:
    return [q.rstrip(".") for q in re.findall(r"“([^”]+)”", text)]


def test_earnings_decoder_quotes_every_number(session, newsroom):
    brief = newsroom.brief("earnings")
    row = run_product(session, "earnings_decoder", brief)
    out = _out(row)
    sources = " ".join(f"{a.summary} {a.content}" for a in newsroom.articles.values())
    reported = [s for s in out.sections if s.label.startswith("reported")]
    assert {"Revenue", "Net income and earnings per share"} <= {s.heading for s in reported}
    for section in reported:
        assert section.citation_ids
        quotes = _quotes(section.body)
        assert quotes
        for quote in quotes:
            assert quote in sources, quote
        # Every figure in the section comes from a quote of a cited source.
        figures = re.findall(r"\$\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?%", section.body)
        assert all(any(f in q for q in quotes) for f in figures), section.body

    versus = next(s for s in out.sections if s.heading == "Versus earlier guidance")
    assert versus.body.startswith("The reported figure is above the top of the company's earlier guidance range")
    forecast = next(s for s in out.sections if s.label == "forecast")
    assert "$3.60 billion to $3.80 billion" in forecast.body and "not a result" in forecast.heading
    assert any("Guidance is the company's own forecast" in c for c in out.caveats)


def test_earnings_decoder_wraps_linked_edgar_report(session, newsroom):
    brief = newsroom.brief("earnings")
    company = Company(name="Corvane Semiconductor", ticker="CRVN", cik="0000000001")
    session.add(company)
    session.flush()
    report = EarningsReport(
        company_id=company.id,
        fiscal_period="FY2026Q3",
        form="10-Q",
        filing_url="https://www.sec.gov/Archives/edgar/data/1/000000000126000001/0000000001-26-000001-index.htm",
        filed_at=utcnow() - timedelta(days=4),
        metrics=[
            {
                "name": "Revenue",
                "value": 3420000000.0,
                "unit": "USD",
                "period": "2026-06-29..2026-09-27",
                "prior_value": 2900000000.0,
                "prior_period": "2025-06-30..2025-09-28",
                "change_pct": 17.9,
                "source_url": "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/0000000001-26-000001-index.htm",
                "source_quote": "us-gaap:Revenues, 10-Q filed 2026-10-23, period 2026-06-29..2026-09-27: $3,420,000,000.",
            }
        ],
        guidance=[],
        data_delay_note="Figures from SEC XBRL company facts; EDGAR data can lag filings. Retrieved 2026-10-23.",
        brief_id=brief.id,
    )
    session.add(report)
    session.flush()
    row = run_product(session, "earnings_decoder", brief)
    out = _out(row)
    filing = next(c for c in row.citations if c["doc_type"] == "filing")
    assert filing["publisher"] == "SEC EDGAR" and filing["url"] == report.filing_url
    section = next(s for s in out.sections if s.label == "reported (filing)")
    assert section.citation_ids == [filing["id"]]
    assert (
        "$3,420,000,000" in section.body
        and "$2,900,000,000" in section.body
        and "us-gaap:Revenues" in section.body
    )
    assert report.data_delay_note in out.caveats
    assert lint_text(_all_text(out), "finance") == []


def test_funding_round_tracker_is_not_a_ranking(session, newsroom):
    brief = newsroom.brief("funding")
    out = _out(run_product(session, "funding_round_tracker", brief))
    rounds = [s for s in out.sections if s.label.startswith("reported (public announcement)")]
    assert {s.heading.split(":")[0] for s in rounds} == {"Tarnwell Compute", "Quillfield"}
    tarnwell = next(s for s in rounds if s.heading.startswith("Tarnwell"))
    assert "Orbit Growth Partners" in tarnwell.body and "$220 million" in tarnwell.heading
    note = next(s for s in out.sections if s.heading == "How to read this tracker")
    assert "not an investment ranking" in note.body
    assert "Not an investment ranking" in out.summary


def test_funding_round_tracker_shows_tracker_trends(session, newsroom):
    brief = newsroom.brief("funding")
    tarnwell = newsroom.articles["round_tarnwell"]
    session.add_all(
        [
            FundingRound(
                company_name="Tarnwell Compute",
                sector="Semiconductors",
                stage="series_d_plus",
                amount_usd=220e6,
                amount_text="$220 million",
                investors=["Orbit Growth Partners", "Corvane Ventures"],
                lead_investors=["Orbit Growth Partners"],
                announced_at=tarnwell.published_at,
                article_id=tarnwell.id,
                source_url=tarnwell.url,
            ),
            FundingRound(
                company_name="Copperline Security",
                sector="Cybersecurity",
                stage="seed",
                amount_usd=6.5e6,
                amount_text="$6.5 million",
                investors=["Orbit Growth Partners"],
                announced_at=utcnow() - timedelta(days=5),
                source_url="https://seedandseries.example/copperline",
            ),
        ]
    )
    session.flush()
    out = _out(run_product(session, "funding_round_tracker", brief, refresh=True))
    trends = next(s for s in out.sections if s.heading.startswith("Sector and stage activity"))
    assert trends.label == "aggregate counts (not a ranking)"
    assert "Rounds logged: 2" in trends.body and "Orbit Growth Partners (2)" in trends.body
    assert lint_text(_all_text(out), "finance") == []


def test_ai_finance_reality_check_labels_backtests(session, newsroom):
    out = _out(run_product(session, "ai_finance_reality_check", newsroom.brief("backtest")))
    claims = [s for s in out.sections if s.label.startswith("backtest")]
    assert claims
    note = next(s for s in claims if "Northbridge Bank Newsroom" in s.body)
    assert "Performance label: backtest." in note.body
    assert "Period:" in note.body and "Benchmark:" in note.body and "Costs assumed:" in note.body
    assert "Authors' caveats:" in note.body
    perf = next(s for s in out.sections if s.label == "performance labels")
    assert "backtest, period January 2005 through December 2024" in perf.body
    assert "not live results" in out.summary

    enforcement = _out(run_product(session, "ai_finance_reality_check", newsroom.brief("enforcement")))
    assert any(s.label == "regulatory_finding" for s in enforcement.sections)


def test_regulation_watch_status_and_dates(session, newsroom):
    out = _out(run_product(session, "regulation_watch", newsroom.brief("enforcement")))
    labels = [s.label for s in out.sections]
    assert "enforcement action (settled)" in labels and "proposal (not final)" in labels
    when = next(s for s in out.sections if s.heading == "When?")
    assert "60 days" in when.body and "estimate" in when.body
    assert re.search(r"about [A-Z][a-z]{2} \d{1,2}, \d{4}", when.body)
    changes = next(s for s in out.sections if s.heading == "What changes?")
    assert changes.label == "proposed" and "authority limits" in changes.body
    assert any("not legal advice" in c for c in out.caveats)


class _FakeClaude:
    name = "claude"
    model = "fake-claude"

    def __init__(self, output: ProductOut):
        self.output = output

    def generate(self, task, budget):
        return LLMResult(output=self.output, provider="claude", model=self.model)

    def run_tools(self, task, budget):  # pragma: no cover - not used by products
        raise NotImplementedError


def test_model_advice_is_removed_and_disclaimer_added(session, newsroom):
    brief = newsroom.brief("earnings")
    raw = ProductOut(
        product="earnings_decoder",
        title="Earnings Decoder: Corvane",
        summary="Corvane is a strong buy after this quarter.",
        sections=[
            ProductSection(
                heading="Revenue", body="Revenue was $3.42 billion [c1].", label="reported", citation_ids=["c1"]
            ),
            ProductSection(
                heading="Our view", body="We raise our price target to $95 per share.", label="opinion"
            ),
            ProductSection(heading="Tip", body="You should buy the shares now.", label="advice"),
        ],
        caveats=["Shares will soar next quarter."],
    )
    row = run_product(
        session, "earnings_decoder", brief, orchestrator=Orchestrator(session, provider=_FakeClaude(raw))
    )
    out = _out(row)
    assert [s.heading for s in out.sections] == ["Revenue"]
    assert "strong buy" not in out.summary
    assert out.caveats == [FINANCE_DISCLAIMER]
    assert lint_text(_all_text(out), "finance") == []
