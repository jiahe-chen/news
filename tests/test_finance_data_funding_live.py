"""Funding Round Tracker precision on real live-source rows (headline + summary, Oct 2026).

A run against the 27 approved feeds logged "Williams $860,000" from an SEC
enforcement release and "Foursquare $411 million" from a story about a
lifetime funding total. These tests replay the real rows from
``tests/data/finance/live_funding_samples.json`` (each with the expected
outcome and why) through the rule path, the tracker, and the agent path.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from newsren.agents.base import Orchestrator
from newsren.finance.funding import (
    MIN_CONFIDENCE,
    STAGES,
    extract_funding,
    extract_funding_from_text,
    is_regulator_source,
    looks_like_funding,
    negative_context,
    update_funding_tracker,
)
from newsren.llm import LLMResult
from newsren.models import AgentRun, Vertical
from newsren.schemas import FundingExtractOut

SAMPLES = json.loads((Path(__file__).parent / "data" / "finance" / "live_funding_samples.json").read_text())[
    "samples"
]
ROUNDS = [s for s in SAMPLES if s["expected"].get("is_funding_round")]


def _sample(live_id: int) -> dict:
    return next(s for s in SAMPLES if s["live_article_id"] == live_id)


def live_article(factory, sample: dict):
    """An Article with the live row's text, URL, doc type, trust tier, tags, and source URLs."""
    source = factory.source(vertical=sample["vertical"], trust_tier=sample["source_trust_tier"])
    source.url, source.homepage = sample["source_url"], sample["source_homepage"]
    return factory.article(
        sample["title"],
        sample["content"],
        source=source,
        summary=sample["summary"],
        url=sample["url"],
        doc_type=sample["doc_type"],
        trust_tier=sample["trust_tier"],
        published_at=datetime.fromisoformat(sample["published_at"]),
        extra=dict(sample["extra"]),
    )


@pytest.mark.parametrize("sample", SAMPLES, ids=[f"{s['live_article_id']}-{s['title'][:40]}" for s in SAMPLES])
def test_live_rows_extract_as_expected(factory, sample):
    out = extract_funding(live_article(factory, sample))
    expected = dict(sample["expected"])
    expected.pop("why")
    never = expected.pop("never_amount_text", None)
    if never is not None:
        assert out.amount_text != never, "a valuation is never logged as the round size"
        return
    if not expected["is_funding_round"]:
        assert not (out.is_funding_round and out.confidence >= MIN_CONFIDENCE), sample["expected"]["why"]
        return
    assert out.is_funding_round and out.confidence >= MIN_CONFIDENCE and out.stage in STAGES
    for field, value in expected.items():
        assert getattr(out, field) == value, field


def test_live_tracker_logs_only_real_rounds(session, factory):
    for sample in SAMPLES:
        live_article(factory, sample)
    created = update_funding_tracker(session)
    assert sorted(r.company_name for r in created) == sorted(s["expected"]["company_name"] for s in ROUNDS)
    assert not {"Williams", "Foursquare"} & {r.company_name for r in created}
    legora = next(r for r in created if r.company_name == "Legora")
    assert (legora.stage, legora.amount_usd) == ("series_d_plus", 550e6), (
        "Series E is logged as Series D and later"
    )


def test_regulator_releases_never_reach_the_extractor(factory):
    sec = live_article(factory, _sample(253))
    assert is_regulator_source(sec)
    assert not looks_like_funding(sec)
    assert not extract_funding(sec).is_funding_round
    # Even the summary's text alone, read as news, has no company raising a round.
    s = _sample(253)
    assert not extract_funding_from_text(s["title"], s["summary"]).is_funding_round
    assert not extract_funding_from_text("Wire digest", s["summary"]).is_funding_round
    assert not is_regulator_source(live_article(factory, _sample(275)))


@pytest.mark.parametrize(
    "text",
    [
        "SEC Charges South Florida Resident and His Company for Alleged Investment Scheme Defrauding Law Enforcement",
        "Foursquare co-founder says equity holders saw no returns from the company’s acquisition",
        "SEC Charges Founder and His Two New Jersey-Based Companies in Alleged $16 Million Ponzi Scheme",
        "SEC Charges Two Individuals With Orchestrating Fraud Scheme That Targeted Veterans",
    ],
)
def test_negative_contexts_from_live_headlines(text):
    assert negative_context(text)


@pytest.mark.parametrize(
    "text",
    [
        "Fraud-detection startup Sardine raises $70M Series C",
        "Payments fraud startup Pinmoor raises $30M Series B",
        "Pinmoor, which helps banks fight wire fraud, raises $20M seed round",
        "Acme raises $30M Series B to fund acquisitions",
        "Legal AI startup Legora raises $550M Series E at $8.8B valuation",
    ],
)
def test_benign_wording_is_not_a_negative_context(text):
    assert not negative_context(text)


class EagerClaude:
    """A model that always reports a round; it must never be asked about enforcement releases."""

    name = "claude"
    model = "claude-test"

    def __init__(self):
        self.calls = 0

    def generate(self, task, budget):
        self.calls += 1
        out = FundingExtractOut(
            is_funding_round=True,
            company_name="Williams",
            amount_text="$860,000",
            amount_usd=860_000,
            confidence=0.9,
        )
        return LLMResult(output=out, provider="claude", model=self.model, llm_calls=1)

    def run_tools(self, task, budget):  # pragma: no cover - not used
        raise NotImplementedError


def test_agent_path_skips_regulator_and_negative_headlines(session, factory):
    excluded = [live_article(factory, _sample(i)) for i in (253, 248, 259, 272)]
    fake = EagerClaude()
    created = update_funding_tracker(session, articles=excluded, orchestrator=Orchestrator(session, provider=fake))
    assert created == [] and fake.calls == 0
    assert session.scalars(select(AgentRun)).all() == []


def test_agent_path_still_reads_real_rounds(session, factory):
    town = live_article(factory, _sample(275))
    assert town.vertical == Vertical.FINANCE and looks_like_funding(town)
    created = update_funding_tracker(session, articles=[town], orchestrator=Orchestrator(session))
    assert [(r.company_name, r.amount_text) for r in created] == [("Town", "$108 million")]
