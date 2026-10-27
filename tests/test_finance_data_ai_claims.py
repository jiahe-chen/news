"""AI-in-Finance Reality Check: seed claims, rule-based classification, tracker updates, and the agent path."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import func, select

from newsren.agents.base import Orchestrator
from newsren.finance.ai_claims import (
    AGENT_NAME,
    AREA_LABELS,
    CLASSIFICATION_LABELS,
    REGULATORY_CONTEXT,
    SEED_CLAIMS,
    claims_by_classification,
    classify_claim,
    classify_claim_rules,
    mentions_ai_in_finance,
    seed_ai_claims,
    update_ai_claims,
)
from newsren.llm import LLMResult
from newsren.models import AgentRun, AIFinanceClaim, DocType, TrustTier, Vertical, utcnow
from newsren.schemas import AIFinanceClaimOut

# --------------------------------------------------------------------------- seeds


def seed(firm: str, fragment: str = ""):
    return next(s for s in SEED_CLAIMS if s.firm == firm and fragment in s.claim_text)


def test_seed_claims_cover_the_plans_research_table():
    assert seed("FINRA", "agent types").classification == "regulatory_finding"
    assert (
        seed("FINRA", "agent types").source_url
        == "https://www.finra.org/media-center/blog/observations-on-ai-agents"
    )
    assert seed("FINRA", "agent types").published_at == datetime(2026, 1, 27)
    assert seed("Deutsche Bank").classification == "pilot"
    assert "Google Cloud" in seed("Deutsche Bank").claim_text
    assert seed("Goldman Sachs").classification == "announcement"
    assert seed("JPMorgan", "LLM Suite").classification == "live_deployment"
    assert "50,000" in seed("JPMorgan", "LLM Suite").claim_text
    assert seed("Metal").classification == "announcement" and seed("Metal").area == "founders"
    survey = seed("Wealth managers (vendor-cited surveys)")
    assert survey.classification == "announcement" and "16%" in survey.claim_text and "8%" in survey.claim_text
    assert "vendor" in survey.evidence_note.lower()

    trading_agents = next(s for s in SEED_CLAIMS if "TradingAgents" in s.firm)
    assert trading_agents.source_url == "https://arxiv.org/abs/2412.20138"
    assert (trading_agents.classification, trading_agents.performance_label) == ("backtest", "backtest")
    assert "preprint" in trading_agents.evidence_note

    delphia, global_predictions = seed("Delphia"), seed("Global Predictions")
    assert "$225,000" in delphia.claim_text and "$175,000" in global_predictions.claim_text
    assert {delphia.classification, global_predictions.classification} == {"regulatory_finding"}
    assert delphia.published_at == datetime(2024, 3, 1) and delphia.published_label == "Mar 2024"


def test_jpmorgan_backtest_seed_carries_full_performance_label():
    claim = seed("JPMorgan", "60/40")
    assert claim.classification == "backtest" and claim.performance_label == "backtest"
    assert claim.period and "two-decade" in claim.period.lower()
    assert claim.benchmark == "60/40 portfolio"
    assert "in-sample result is not proof of future outperformance" in claim.authors_caveats
    assert claim.published_at == datetime(2026, 7, 9)
    assert "investsuite" in claim.source_url


def test_seed_claims_are_well_formed():
    keys = [(s.firm, s.source_url) for s in SEED_CLAIMS]
    assert len(keys) == len(set(keys)), "(firm, source_url) identifies a seed"
    for s in SEED_CLAIMS:
        assert s.classification in CLASSIFICATION_LABELS and s.area in AREA_LABELS
        assert s.source_url.startswith("https://") and s.source_title and s.claim_text and s.evidence_note
        if s.classification == "backtest":
            assert s.performance_label in ("backtest", "simulated")
    assert any("investment-adviser registration" in r["text"] for r in REGULATORY_CONTEXT)


def test_seed_ai_claims_is_idempotent_and_refreshes_text(session):
    assert seed_ai_claims(session) == len(SEED_CLAIMS)
    assert seed_ai_claims(session) == 0
    count = session.scalar(select(func.count()).select_from(AIFinanceClaim))
    assert count == len(SEED_CLAIMS)
    row = session.scalars(select(AIFinanceClaim).where(AIFinanceClaim.firm == "Goldman Sachs")).one()
    assert row.is_seed and row.article_id is None
    row.claim_text = "edited"
    seed_ai_claims(session)
    assert row.claim_text == seed("Goldman Sachs").claim_text
    assert session.scalar(select(func.count()).select_from(AIFinanceClaim)) == count


# --------------------------------------------------------------------------- rule-based classification


@pytest.fixture()
def fin(factory):
    return factory.source(vertical=Vertical.FINANCE, name="Finance Wire")


def art(factory, source, title, body="", **kwargs):
    return factory.article(title, body, source=source, **kwargs)


CASES = [
    (
        "Northbridge researchers: AI portfolio agents outperformed a 60/40 benchmark in a 20-year backtest",
        "Researchers at Northbridge Bank published a note on AI portfolio agents. In a backtest covering January 2005 "
        "through December 2024, the agents returned 7.9% a year versus 6.6% for a benchmark portfolio of 60% global "
        "equities and 40% bonds. The backtest assumed transaction costs of 5 basis points per trade. The authors "
        "caution that the result is in-sample and is not proof of future outperformance.",
        dict(
            classification="backtest", performance_label="backtest", firm="Northbridge Bank", area="asset_wealth"
        ),
    ),
    (
        "Northbridge says AI portfolio agents will enter a year of paper trading",
        "Northbridge Bank said its AI portfolio agents will run in a 12-month paper-trading pilot with no client money.",
        dict(classification="backtest", performance_label="simulated"),
    ),
    (
        "Harbor Bank pilots AI agents for anti-money laundering alerts",
        "Harbor Bank is testing AI agents that triage AML alerts in two branches.",
        dict(classification="pilot", firm="Harbor Bank"),
    ),
    (
        "Meridian Securities rolls out AI research assistant to 5,000 advisers",
        "The generative AI assistant has been deployed across the firm's wealth management unit and is in production.",
        dict(classification="live_deployment", firm="Meridian Securities", area="asset_wealth"),
    ),
    (
        "SEC fines adviser over AI claims",
        "The SEC said Brightvane Advisors settled charges that it overstated its use of artificial intelligence, "
        "agreeing to a $310,000 penalty without admitting or denying the findings.",
        dict(classification="regulatory_finding", firm="Brightvane Advisors", area="regulation"),
    ),
    (
        "Regulatory Notice: FINRA proposes supervision expectations for AI agents at member firms",
        "FINRA requested comment on guidance for member firms that use AI agents.",
        dict(classification="regulatory_finding", firm="FINRA"),
    ),
    (
        "Goldman Sachs is exploring agentic AI for trade surveillance",
        "Goldman Sachs is exploring how AI agents could support trade surveillance.",
        dict(classification="announcement", firm="Goldman Sachs", area="trading_markets"),
    ),
]


@pytest.mark.parametrize(("title", "body", "expected"), CASES, ids=[c[0][:40] for c in CASES])
def test_classify_claim_rules(factory, fin, title, body, expected):
    out = classify_claim_rules(art(factory, fin, title, body))
    assert out.is_ai_finance_claim
    for field, value in expected.items():
        assert getattr(out, field) == value, field
    assert out.claim_text and out.evidence_note


def test_backtest_details_are_extracted_from_the_source(factory, fin):
    title, body, _ = CASES[0]
    out = classify_claim_rules(art(factory, fin, title, body))
    assert out.period.startswith("January 2005 through December 2024")
    assert out.benchmark.startswith("60% global equities and 40% bonds")
    assert "5 basis points per trade" in out.costs_assumed
    assert "in-sample" in out.authors_caveats and "not proof of future outperformance" in out.authors_caveats
    assert "not a live track record" in out.evidence_note


def test_missing_backtest_details_are_flagged_not_invented(factory, fin):
    out = classify_claim_rules(
        art(factory, fin, "Fund says AI trading model beat the market in backtests", "No details.")
    )
    assert out.classification == "backtest" and out.benchmark == "" and out.costs_assumed == ""
    assert "does not state" in out.evidence_note


def test_negated_deployment_is_not_live(factory, fin):
    out = classify_claim_rules(
        art(
            factory,
            fin,
            "Bank's AI trading agents have not been deployed",
            "The bank's AI agents have not been deployed.",
        )
    )
    assert out.is_ai_finance_claim and out.classification != "live_deployment"


@pytest.mark.parametrize(
    ("title", "body", "doc_type"),
    [
        ("Lumen Labs launches a new AI model for math tutoring", "Students use it in classrooms.", None),
        ("Corvane reports record quarterly revenue", "Revenue rose 18% on chip demand.", None),
        (
            "Opinion: AI agents at banks are overhyped",
            "A columnist argues banks' AI portfolio claims are hype.",
            DocType.OPINION,
        ),
        (
            "Quillfield raises $45 million to build AI tools for teachers",
            "Investors include Harrow Ventures.",
            None,
        ),
    ],
)
def test_non_claims_are_not_classified(factory, fin, title, body, doc_type):
    article = art(factory, fin, title, body, doc_type=doc_type)
    assert not classify_claim_rules(article).is_ai_finance_claim


def test_vendor_press_release_is_labelled_as_vendor_claim(factory):
    vendor = factory.source(vertical=Vertical.FINANCE, trust_tier=TrustTier.VENDOR, doc_type=DocType.PRESS_RELEASE)
    out = classify_claim_rules(art(factory, vendor, "Acme Capital launches AI agents for portfolio rebalancing"))
    assert out.classification == "announcement"
    assert "vendor claim" in out.evidence_note


# --------------------------------------------------------------------------- tracker


def test_update_ai_claims_stores_new_claims_once(session, factory, fin):
    now = utcnow()
    regulator = factory.source(
        vertical=Vertical.FINANCE, trust_tier=TrustTier.PRIMARY, doc_type=DocType.REGULATORY
    )
    wire = art(
        factory,
        fin,
        "Regulator fines adviser $310,000 for claiming AI agents it did not use",
        "Brightvane Advisors agreed to pay a $310,000 penalty to settle charges that it marketed autonomous AI "
        "portfolio management it did not use.",
        published_at=now - timedelta(days=1),
    )
    release = art(
        factory,
        regulator,
        "Markets Integrity Commission charges Brightvane Advisors with misleading clients about AI use",
        "The Commission settled charges that Brightvane Advisors falsely marketed portfolios as managed by AI agents.",
        published_at=now - timedelta(days=2),
    )
    backtest = art(factory, fin, *CASES[0][:2], published_at=now - timedelta(days=3))
    art(
        factory,
        fin,
        "Opinion: AI agents at banks are overhyped",
        "Bank AI portfolio claims.",
        doc_type=DocType.OPINION,
    )
    art(factory, fin, "Central bank holds rates steady", "No AI here.")

    created = update_ai_claims(session)
    by_article = {c.article_id: c for c in created}
    assert set(by_article) == {release.id, backtest.id}, "the primary release is kept over the wire copy"
    assert wire.id not in by_article
    claim = by_article[backtest.id]
    assert (claim.source_url, claim.source_title, claim.published_at) == (
        backtest.url,
        backtest.title,
        backtest.published_at,
    )
    assert claim.classification == "backtest" and claim.performance_label == "backtest" and not claim.is_seed
    assert by_article[release.id].classification == "regulatory_finding"

    assert update_ai_claims(session) == []
    assert update_ai_claims(session, articles=[backtest]) == []


def test_claims_by_classification_groups_seed_and_new_claims(session, factory, fin):
    seed_ai_claims(session)
    update_ai_claims(session, articles=[art(factory, fin, *CASES[2][:2])])
    groups = {g["classification"]: g for g in claims_by_classification(session)}
    assert list(groups) == list(CLASSIFICATION_LABELS)
    assert any(c.firm == "Harbor Bank" for c in groups["pilot"]["claims"])
    assert any(c.firm == "Deutsche Bank" for c in groups["pilot"]["claims"])
    non_seed = {g["classification"]: g for g in claims_by_classification(session, include_seed=False)}
    assert [c.firm for c in non_seed["pilot"]["claims"]] == ["Harbor Bank"]


# --------------------------------------------------------------------------- agent path


class FakeClaude:
    name = "claude"
    model = "claude-test"

    def __init__(self, output: AIFinanceClaimOut):
        self.output = output
        self.calls = 0

    def generate(self, task, budget):
        self.calls += 1
        assert task.output_model is AIFinanceClaimOut
        assert '<source id="c1">' in task.prompt
        assert "backtest" in task.system and "never state" in task.system.lower()
        return LLMResult(output=self.output, provider="claude", model=self.model, llm_calls=1)

    def run_tools(self, task, budget):  # pragma: no cover - not used
        raise NotImplementedError


def test_agent_output_gets_required_performance_label(session, factory, fin):
    article = art(factory, fin, *CASES[0][:2])
    fake = FakeClaude(
        AIFinanceClaimOut(
            is_ai_finance_claim=True,
            firm="",
            area="asset_wealth",
            claim_text="Northbridge's AI agents beat a 60/40 benchmark in a backtest.",
            classification="backtest",
            performance_label=None,
        )
    )
    orchestrator = Orchestrator(session, provider=fake)
    out = classify_claim(article, orchestrator)
    assert out.performance_label == "backtest", "a backtest is always labelled"
    assert out.firm == "Northbridge Bank", "missing fields are filled from the rule-based reading"
    assert classify_claim(article, orchestrator).classification == "backtest"
    assert fake.calls == 1, "model answers are cached per article version"


def test_offline_orchestrator_path_records_agent_runs(session, factory, fin):
    article = art(factory, fin, *CASES[2][:2])
    art(factory, fin, "Central bank holds rates steady", "No AI here.")
    assert mentions_ai_in_finance(article)
    created = update_ai_claims(session, orchestrator=Orchestrator(session))
    assert [c.firm for c in created] == ["Harbor Bank"]
    runs = session.scalars(select(AgentRun).where(AgentRun.agent == AGENT_NAME)).all()
    assert [(r.subject_type, r.subject_id) for r in runs] == [("article", article.id)]
