"""Editorial products registry and applies_to selection on hand-built briefs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from newsren.agents.products import PRODUCTS, available_products, get_product, run_product
from newsren.editorial.guardrails import FINANCE_DISCLAIMER
from newsren.models import Brief


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


# The plan's "Proposed editorial products" table.
PLAN_TABLE = {
    "launch_reality_check": ("technology", "Launch Reality Check", "independent benchmarks"),
    "research_paper_decoder": (
        "technology",
        "Research Paper Decoder",
        "hypothesis, method, findings, limitations",
    ),
    "technology_timeline": (
        "technology",
        "Technology Evolution Timeline",
        "evaluations, corrections, and adoption",
    ),
    "policy_impact_explorer": ("education", "Policy Impact Explorer", "implementation timelines"),
    "classroom_discussion_builder": ("education", "Classroom Discussion Builder", "age-appropriate questions"),
    "education_evidence_checker": ("education", "Education Evidence Checker", "measured learning outcomes"),
    "company_watch": ("industry", "Company and Competitor Watch", "publicly documented company actions"),
    "supply_chain_ripple_map": ("industry", "Supply-Chain Ripple Map", "labeled as hypotheses"),
    "skills_career_trends": ("industry", "Skills and Career Trends", "emerging skills, training options"),
    "earnings_decoder": ("finance", "Earnings Decoder", "quote the filing or call transcript"),
    "funding_round_tracker": ("finance", "Funding Round Tracker", "without ranking startups"),
    "ai_finance_reality_check": ("finance", "AI-in-Finance Reality Check", "backtests from documented live"),
    "regulation_watch": ("finance", "Regulation Watch", "SEC, FINRA, and central-bank"),
}


def test_registry_matches_plan_table():
    assert set(PRODUCTS) == set(PLAN_TABLE)
    for key, (vertical, name, value_phrase) in PLAN_TABLE.items():
        spec = PRODUCTS[key]
        assert spec.key == key and spec.vertical == vertical and spec.name == name
        assert value_phrase in spec.reader_value
        assert spec.description and callable(spec.builder) and callable(spec.applies_to)
        assert spec.name in spec.system_prompt and "citation" in spec.system_prompt
        if vertical == "finance":
            assert spec.is_finance and FINANCE_DISCLAIMER in spec.system_prompt
            assert "beat the market" in spec.system_prompt
    assert "hypothesis" in PRODUCTS["supply_chain_ripple_map"].system_prompt
    assert "preprint" in PRODUCTS["research_paper_decoder"].system_prompt


def _keys(brief: Brief) -> set[str]:
    return {spec.key for spec in available_products(brief)}


def test_technology_selection(newsroom):
    launch = newsroom.brief("model_launch")
    assert _keys(launch) >= {"launch_reality_check", "research_paper_decoder", "technology_timeline"}
    assert not _keys(launch) & {"earnings_decoder", "classroom_discussion_builder", "policy_impact_explorer"}
    tutor = newsroom.brief("tutor_launch")
    keys = _keys(tutor)
    assert "launch_reality_check" in keys
    # One source, no research paper: no timeline and no paper to decode.
    assert "technology_timeline" not in keys and "research_paper_decoder" not in keys


def test_education_selection(newsroom):
    study = _keys(newsroom.brief("learning_study"))
    assert study >= {"classroom_discussion_builder", "education_evidence_checker", "research_paper_decoder"}
    policy = _keys(newsroom.brief("policy"))
    assert policy >= {"policy_impact_explorer", "classroom_discussion_builder"}
    assert "education_evidence_checker" not in policy and "research_paper_decoder" not in policy
    # The brief's own vertical comes first.
    products = available_products(newsroom.brief("learning_study"))
    assert products[0].vertical == "education" and products[-1].key == "research_paper_decoder"


def test_industry_selection(newsroom):
    keys = _keys(newsroom.brief("semiconductor"))
    assert keys == {"company_watch", "supply_chain_ripple_map", "skills_career_trends"}


def test_finance_selection(newsroom):
    assert _keys(newsroom.brief("earnings")) == {"earnings_decoder"}
    assert _keys(newsroom.brief("funding")) == {"funding_round_tracker"}
    assert _keys(newsroom.brief("backtest")) == {"ai_finance_reality_check"}
    assert _keys(newsroom.brief("enforcement")) == {"ai_finance_reality_check", "regulation_watch"}


def test_unknown_product_key(session, newsroom):
    brief = newsroom.brief("policy")
    with pytest.raises(KeyError):
        get_product("horoscope")
    with pytest.raises(KeyError):
        run_product(session, "horoscope", brief)


def test_non_applicable_product_reports_missing_evidence(session, newsroom):
    brief = newsroom.brief("policy")
    row = run_product(session, "research_paper_decoder", brief)
    sections = row.content["sections"]
    assert [s["label"] for s in sections] == ["missing evidence"]
