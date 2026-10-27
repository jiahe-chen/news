"""Earnings and Filings Brief (``newsren.finance.edgar.earnings_brief``).

Uses the saved EDGAR samples for IRADIMED Corp (tests/data/finance/) and the
fictional demo corpus (Corvane Semiconductor's earnings release and coverage).
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from newsren.agents.base import Orchestrator
from newsren.editorial.checks import failed_checks, run_checks
from newsren.editorial.guardrails import is_hedged, lint_text
from newsren.finance.edgar import (
    EARNINGS_SECTIONS,
    EARNINGS_SYSTEM_PROMPT,
    build_earnings_report,
    change_pct,
    company_name_variants,
    earnings_brief,
    extract_guidance,
    format_value,
    get_or_create_company,
    period_text,
)
from newsren.ingest.fixtures import load_demo_data
from newsren.llm import LLMResult
from newsren.models import AgentRun, Article, Brief, BriefStatus, EarningsReport, EditorialAction, utcnow
from newsren.schemas import BriefOut, ClaimOut, SectionOut

DATA = Path(__file__).parent / "data" / "finance"
IRMD_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/1325618/"
Q1_2026_INDEX = "https://www.sec.gov/Archives/edgar/data/1325618/000162828026029325/"
IRMD_RELEASE = "https://www.sec.gov/Archives/edgar/data/1325618/000162828026057617/irmd-20260731.htm"
CORVANE_FILING = "https://investors.corvane-semi.example/filings/fy2026-q3-10q"


@pytest.fixture()
def irmd_report(session, settings) -> EarningsReport:
    company = get_or_create_company(session, "IRADIMED CORP", ticker="IRMD", cik="1325618")
    return build_earnings_report(
        session,
        company,
        settings,
        facts=json.loads((DATA / "irmd_companyfacts.json").read_text()),
        submissions=json.loads((DATA / "irmd_submissions.json").read_text()),
    )


def _demo_metric(key: str, name: str, value: float, prior: float, start, end, filed, unit: str = "USD") -> dict:
    prior_start, prior_end = start.replace(year=start.year - 1), end.replace(year=end.year - 1)
    return {
        "key": key,
        "name": name,
        "value": value,
        "unit": unit,
        "period": f"{start}..{end}",
        "prior_value": prior,
        "prior_period": f"{prior_start}..{prior_end}",
        "change_pct": change_pct(value, prior),
        "comparison": "same_quarter_prior_year",
        "form": "10-Q",
        "filed": filed.isoformat(),
        "source_url": CORVANE_FILING,
        "prior_source_url": CORVANE_FILING,
        "source_quote": (
            f"Fictional demo 10-Q filed {filed}, statements of operations, period {start}..{end}: {name} "
            f"{format_value(value, unit)}. Same quarter a year earlier, period {prior_start}..{prior_end}: "
            f"{format_value(prior, unit)} (comparative column of the same filing)."
        ),
    }


@pytest.fixture()
def corvane_report(session) -> EarningsReport:
    """A fictional 10-Q for the demo corpus's Corvane Semiconductor, filed after its earnings release."""
    load_demo_data(session)
    now = utcnow()
    end = (now - timedelta(days=40)).date()
    start, filed = end - timedelta(days=90), (now - timedelta(days=3)).date()
    company = get_or_create_company(session, "Corvane Semiconductor", sector="Semiconductors")
    report = EarningsReport(
        company_id=company.id,
        fiscal_period="FY2026Q3",
        form="10-Q",
        filed_at=now - timedelta(days=3),
        filing_url=CORVANE_FILING,
        metrics=[
            _demo_metric("revenue", "Revenue", 3_420_000_000, 2_900_000_000, start, end, filed),
            _demo_metric("net_income", "Net income (loss)", 612_000_000, 498_000_000, start, end, filed),
            _demo_metric("eps_diluted", "Diluted EPS", 1.12, 0.91, start, end, filed, unit="USD/share"),
        ],
        guidance=[],
        narrative={},
        data_delay_note="Fictional demo figures.",
        is_demo=True,
    )
    session.add(report)
    session.flush()
    return report


def brief_texts(brief: Brief) -> list[str]:
    texts = [brief.headline, brief.summary, brief.why_it_matters or "", brief.what_changed or ""]
    texts += list(brief.key_points) + list(brief.uncertainty) + list(brief.watch_next)
    texts += [s["body"] for s in brief.sections] + [c.text for c in brief.claims]
    texts += [p["position"] for d in brief.disagreements for p in d["positions"]]
    return [t for t in texts if t]


def section(brief: Brief, title: str) -> dict:
    return next(s for s in brief.sections if s["title"] == title)


def demo_article(session, key: str) -> Article:
    return next(a for a in session.scalars(select(Article)) if (a.extra or {}).get("demo_key") == key)


# --------------------------------------------------------------------------- figures and citations


def test_every_reported_figure_is_a_claim_citing_its_filing(session, irmd_report):
    brief = earnings_brief(session, irmd_report)
    citations = {c["id"]: c for c in brief.citations}
    figure_claims = [c for c in brief.claims if c.high_impact]

    assert len(figure_claims) == len(irmd_report.metrics) == 5
    for metric, claim in zip(irmd_report.metrics, figure_claims, strict=True):
        cited = [citations[cid] for cid in claim.citation_ids]
        assert cited[0]["url"] == metric["source_url"]
        assert cited[0]["snippet"].startswith(metric["source_quote"])
        assert cited[0]["doc_type"] == "filing" and cited[0]["trust_tier"] == "primary"
        assert cited[0]["retrieved_at"] and cited[0]["published_at"]
        assert all(c["url"].startswith(IRMD_ARCHIVE) for c in cited)
        assert format_value(metric["value"], metric["unit"]) in claim.text
        assert format_value(metric["prior_value"], metric["unit"]) in claim.text
        assert f"{metric['change_pct']:+.1f}%" in claim.text
        assert claim.claim_type == "fact" and claim.support_status == "supported"

    checks = {c.name: c for c in run_checks(session, brief)}
    # Every number in every claim is found in the evidence the claim cites.
    assert checks["claim_support"].passed, checks["claim_support"].details
    assert checks["citation_coverage"].passed
    assert checks["unknown_citation_ids"].passed
    assert checks["approved_sources"].passed


def test_comparison_figures_link_to_the_filing_they_come_from(session, irmd_report):
    brief = earnings_brief(session, irmd_report)
    earlier = [c for c in brief.citations if c["url"] == Q1_2026_INDEX]
    assert len(earlier) == 1
    assert "10-Q filed 2026-05-01" in earlier[0]["title"]
    assert "$21,979,000" in earlier[0]["snippet"]  # prior-quarter revenue as filed in Q1
    versus_prior = section(brief, "Versus the prior period")
    assert earlier[0]["id"] in versus_prior["citation_ids"]
    assert "$21,979,000 in the prior quarter (-6.9%)" in versus_prior["body"]


def test_brief_is_a_finance_brief_waiting_for_human_review(session, irmd_report):
    brief = earnings_brief(session, irmd_report)

    assert (brief.vertical, brief.kind) == ("finance", "earnings")
    assert brief.status == BriefStatus.IN_REVIEW
    assert brief.requires_human_review is True
    assert brief.risk_level == "high"
    assert brief.published_at is None
    assert irmd_report.brief_id == brief.id
    assert not failed_checks(brief.check_results, "blocker")
    for text in brief_texts(brief):
        assert lint_text(text, "finance") == [], text
    actions = session.scalars(select(EditorialAction.action).where(EditorialAction.brief_id == brief.id)).all()
    assert "submit" in actions
    run = session.get(AgentRun, brief.agent_run_id)
    assert (run.agent, run.subject_type, run.subject_id) == ("earnings_brief", "earnings_report", irmd_report.id)
    assert brief.generated_by == "offline"


def test_sections_say_guidance_was_not_available_and_link_concepts(session, irmd_report):
    brief = earnings_brief(session, irmd_report)

    assert [s["title"] for s in brief.sections] == list(EARNINGS_SECTIONS)
    guidance = section(brief, "Versus guidance")
    assert "Guidance was not available from the filing" in guidance["body"]
    release = next(c for c in brief.citations if c["url"] == IRMD_RELEASE)
    assert guidance["citation_ids"] == [release["id"]]
    assert irmd_report.guidance == [], "guidance is never inferred"

    concepts = section(brief, "Concepts")
    assert {"eps", "revenue-vs-net-income", "operating-margin", "sec-forms-10q-10k-8k"} <= set(
        concepts["concepts"]
    )
    meaning = section(brief, "What it means")["body"]
    assert "Operating margin was 31.2% of revenue, versus 33.3% in the same quarter a year earlier" in meaning
    assert "not a forecast" in meaning
    assert brief.headline == "IRADIMED CORP fiscal 2026 Q2 10-Q: revenue up 0.3% from a year earlier"
    assert period_text("2026-04-01..2026-06-30") in brief.summary


def test_rerun_keeps_the_brief_and_regenerate_supersedes_it(session, irmd_report):
    first = earnings_brief(session, irmd_report)
    assert earnings_brief(session, irmd_report).id == first.id
    assert session.scalars(select(Brief).where(Brief.kind == "earnings")).all() == [first]

    second = earnings_brief(session, irmd_report, regenerate=True)
    assert second.id != first.id
    assert irmd_report.brief_id == second.id
    assert first.status == BriefStatus.REJECTED
    assert f"#{second.id}" in first.review_notes
    regenerate = session.scalars(select(EditorialAction).where(EditorialAction.action == "regenerate")).all()
    assert {a.brief_id for a in regenerate} == {first.id, second.id}


def test_report_without_figures_is_refused(session):
    company = get_or_create_company(session, "Empty Filer Inc", ticker="EMPT")
    report = EarningsReport(company_id=company.id, fiscal_period="FY2026Q1", form="10-Q", metrics=[])
    session.add(report)
    session.flush()
    with pytest.raises(ValueError, match="no reported figures"):
        earnings_brief(session, report)


# --------------------------------------------------------------------------- release guidance (demo corpus)


def test_release_guidance_is_quoted_compared_and_labelled(session, corvane_report):
    brief = earnings_brief(session, corvane_report)
    release = demo_article(session, "corvane-q3-results")
    citations = {c["id"]: c for c in brief.citations}
    release_cid = next(cid for cid, c in citations.items() if c["article_id"] == release.id)

    # The company's own release is the guidance source and the brief's lead article.
    assert brief.article_id == release.id
    stored = {(g["guidance"], g["comparison"], g["actual"]) for g in corvane_report.guidance}
    assert stored == {
        ("Third-quarter revenue of $3.25 billion to $3.40 billion", "above", "$3,420,000,000"),
        ("Gross margin of 46% to 48%", "within", "47.8%"),
        ("Fourth-quarter revenue of $3.60 billion to $3.80 billion", "not_comparable", ""),
        ("Fourth-quarter gross margin of 47% to 49%", "not_comparable", ""),
    }
    assert all(g["source_url"] == release.url and g["source_quote"] for g in corvane_report.guidance)
    assert corvane_report.narrative["guidance_source"]["url"] == release.url

    forecasts = [c for c in brief.claims if c.claim_type == "forecast"]
    assert len(forecasts) == 2
    assert all(is_hedged(c.text) and c.citation_ids == [release_cid] for c in forecasts)
    above = next(c for c in brief.claims if "above the top of that range" in c.text)
    assert "$3.25 billion to $3.40 billion" in above.text and "$3,420,000,000" in above.text
    assert release_cid in above.citation_ids and "c1" in above.citation_ids

    guidance = section(brief, "Versus guidance")["body"]
    assert "not available" not in guidance
    assert "forecast" in guidance

    checks = {c.name: c for c in run_checks(session, brief)}
    assert not failed_checks(list(checks.values()), "blocker")
    for name in ("claim_support", "forecast_labelling", "vendor_attribution", "guardrail_lint"):
        assert checks[name].passed, (name, checks[name].details)
    for text in brief_texts(brief):
        assert lint_text(text, "finance") == [], text


def test_rerun_uses_stored_guidance_with_the_same_evidence(session, corvane_report):
    first = earnings_brief(session, corvane_report)
    assert earnings_brief(session, corvane_report).id == first.id  # stored guidance, same evidence

    again = earnings_brief(session, corvane_report, regenerate=True)
    assert again.citations == first.citations
    assert [c.text for c in again.claims] == [c.text for c in first.claims]
    above = next(c for c in again.claims if "above the top of that range" in c.text)
    assert "c1" in above.citation_ids  # the filed actual still cites the filing
    checks = {c.name: c for c in run_checks(session, again)}
    assert checks["claim_support"].passed, checks["claim_support"].details


def test_only_earnings_documents_about_the_company_are_used(session, corvane_report):
    brief = earnings_brief(session, corvane_report)
    cited = {c["article_id"] for c in brief.citations if c["article_id"]}
    assert demo_article(session, "corvane-q3-coverage").id in cited
    assert demo_article(session, "corvane-guidance-opinion").id not in cited  # opinion is not evidence
    assert demo_article(session, "corvane-capacity").id not in cited  # a capacity announcement, not results
    assert demo_article(session, "tarnwell-series-d").id not in cited
    assert company_name_variants("Corvane Semiconductor Inc.") == [
        "Corvane Semiconductor Inc",
        "Corvane Semiconductor",
        "Corvane",
    ]
    assert company_name_variants("IRADIMED CORP") == ["IRADIMED CORP", "IRADIMED"]


def test_coverage_that_disagrees_with_the_filing_is_described(session, corvane_report):
    metrics = [dict(m) for m in corvane_report.metrics]
    metrics[0]["value"] = 3_520_000_000  # the filing now differs from the coverage's "$3.42 billion"
    metrics[0]["change_pct"] = change_pct(3_520_000_000, 2_900_000_000)
    metrics[0]["source_quote"] = metrics[0]["source_quote"].replace("$3,420,000,000", "$3,520,000,000")
    corvane_report.metrics = metrics

    brief = earnings_brief(session, corvane_report)
    coverage = demo_article(session, "corvane-q3-coverage")
    coverage_cid = next(c["id"] for c in brief.citations if c["article_id"] == coverage.id)

    assert len(brief.disagreements) == 1
    positions = brief.disagreements[0]["positions"]
    assert positions[0]["citation_ids"] == ["c1"] and "$3,520,000,000" in positions[0]["position"]
    assert positions[1]["citation_ids"] == [coverage_cid] and "$3.42 billion" in positions[1]["position"]
    checks = {c.name: c for c in run_checks(session, brief)}
    assert checks["disagreements_described"].passed


def test_extract_guidance_reads_ranges_without_inventing_any():
    text = (
        "The company had guided third-quarter revenue of $3.25 billion to $3.40 billion. "
        "Gross margin was 47.8%, within the company's guidance of 46% to 48%. "
        "Fourth-quarter guidance: revenue of $3.60 billion to $3.80 billion and gross margin of 47% to 49%. "
        "Revenue was $3.42 billion and net income was $612 million. "
        "The company expects to hire between 200 and 300 engineers."
    )
    items = extract_guidance(text, report_quarter=3, actuals={"revenue": (3_420_000_000.0, "USD")})
    assert [(i.metric, i.comparison) for i in items] == [
        ("Revenue", "above"),
        ("Gross margin", "within"),
        ("Revenue", "not_comparable"),
        ("Gross margin", "not_comparable"),
    ]
    assert items[0].actual == "$3,420,000,000"
    assert items[1].actual == "47.8%"
    assert items[2].source_quote.startswith("Fourth-quarter guidance:")
    assert extract_guidance("Revenue was $3.42 billion, up from $2.90 billion.") == []


# --------------------------------------------------------------------------- agent output is checked


class StubClaude:
    """A provider returning a fixed BriefOut, standing in for Claude."""

    name = "claude"
    model = "stub-model"

    def __init__(self, output: BriefOut):
        self.output = output
        self.tasks = []

    def generate(self, task, budget):
        self.tasks.append(task)
        return LLMResult(output=self.output, provider="claude", model=self.model)

    def run_tools(self, task, budget):  # pragma: no cover - not used by this agent
        raise NotImplementedError


def test_agent_text_is_kept_only_where_it_passes_the_guardrails(session, irmd_report):
    meaning = "Revenue was roughly flat (+0.3%) while net income fell (-9.6%) [c1][c4]."
    stub = StubClaude(
        BriefOut(
            headline="IRADIMED is a strong buy after a 0.3% revenue gain",
            summary="IRADIMED reported revenue of $21 million [c1].",
            why_it_matters="Quarterly figures show how the business changed from a year earlier [c99].",
            what_changed="Revenue fell -6.9% from the prior quarter [c1].",
            sections=[
                SectionOut(title="What it means", body=meaning, citation_ids=["c1", "c4"]),
                SectionOut(title="Versus guidance", body="Results beat guidance of $25 million.", citation_ids=[]),
            ],
            claims=[
                ClaimOut(text="Revenue will soar next quarter.", claim_type="forecast", citation_ids=["c1"]),
                ClaimOut(
                    text="Net income fell more than revenue, so profit per dollar of sales narrowed.",
                    claim_type="interpretation",
                    citation_ids=["c1", "c4"],
                ),
                ClaimOut(text="Margins compressed sharply.", claim_type="interpretation", citation_ids=["c42"]),
            ],
            risk_level="low",
        )
    )
    brief = earnings_brief(session, irmd_report, Orchestrator(session, provider=stub))

    task = stub.tasks[0]
    assert task.system == EARNINGS_SYSTEM_PROMPT and "Never infer" in task.system
    assert '<source id="c1">' in task.prompt and "Guidance quoted in the evidence:" in task.prompt
    assert brief.generated_by == "stub-model"
    assert brief.headline.startswith("IRADIMED CORP fiscal 2026 Q2 10-Q")  # rating call replaced
    assert "$21 million" not in brief.summary  # number not in the evidence
    assert brief.why_it_matters == "Quarterly figures show how the business changed from a year earlier."
    assert brief.what_changed == "Revenue fell -6.9% from the prior quarter [c1]."
    assert section(brief, "What it means")["body"] == meaning
    assert "Guidance was not available from the filing" in section(brief, "Versus guidance")["body"]
    assert brief.risk_level == "high"

    texts = [c.text for c in brief.claims]
    assert len(texts) == 6  # five figure claims + the one valid agent claim
    assert "Net income fell more than revenue, so profit per dollar of sales narrowed." in texts
    assert not any("soar" in t or "compressed" in t for t in texts)
    assert not failed_checks(brief.check_results, "blocker")
    for text in brief_texts(brief):
        assert lint_text(text, "finance") == [], text
