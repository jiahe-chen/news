"""AI-in-Finance Reality Check precision on real live-source rows (headline + summary, Oct 2026).

A run against the 27 approved feeds logged two non-claims: a sponsored
storage-hardware post (firm "From Cloud") and a guest explainer about
founders' securities-fraud liability (firm "When Traction Becomes
Personal"). The real rows in ``tests/data/finance/live_ai_claims_samples.json``
are replayed here; a claim now needs a financial-sector actor and an AI cue
in the same sentence, sponsored/commentary/marketing items are skipped, and
the firm is never a run of capitalised headline words.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from newsren.agents.base import Orchestrator
from newsren.finance.ai_claims import (
    classify_claim,
    classify_claim_rules,
    is_ai_finance_sentence,
    mentions_ai_in_finance,
    skip_reason,
    update_ai_claims,
)
from newsren.llm import LLMResult
from newsren.models import TrustTier, Vertical
from newsren.schemas import AIFinanceClaimOut

SAMPLES = json.loads((Path(__file__).parent / "data" / "finance" / "live_ai_claims_samples.json").read_text())[
    "samples"
]


def _sample(live_id: int) -> dict:
    return next(s for s in SAMPLES if s["live_article_id"] == live_id)


def live_article(factory, sample: dict, **overrides):
    """An Article with the live row's text, URL, doc type, trust tier, and tags (``overrides`` replace fields)."""
    source = factory.source(vertical=sample["vertical"], trust_tier=sample["source_trust_tier"])
    title = overrides.pop("title", sample["title"])
    fields = dict(
        summary=sample["summary"],
        url=sample["url"],
        doc_type=sample["doc_type"],
        trust_tier=sample["trust_tier"],
        published_at=datetime.fromisoformat(sample["published_at"]),
        extra=dict(sample["extra"]),
    )
    fields.update(overrides)
    return factory.article(title, sample["content"], source=source, **fields)


@pytest.mark.parametrize("sample", SAMPLES, ids=[f"{s['live_article_id']}-{s['title'][:40]}" for s in SAMPLES])
def test_live_rows_are_not_ai_finance_claims(factory, sample):
    article = live_article(factory, sample)
    assert not mentions_ai_in_finance(article), sample["expected"]["why"]
    assert not classify_claim_rules(article).is_ai_finance_claim, sample["expected"]["why"]


def test_live_tracker_logs_no_claims(session, factory):
    for sample in SAMPLES:
        live_article(factory, sample)
    assert update_ai_claims(session) == []


def test_sponsored_hardware_post_is_skipped_and_has_no_finance_actor(factory):
    biwin = _sample(318)
    assert "Sponsored" in skip_reason(live_article(factory, biwin))
    # Without the publisher's tag it is still marketing copy ("See how ...") with no financial actor:
    # "memory portfolio" is a product line, not a portfolio manager.
    untagged = live_article(factory, biwin, extra={"tags": ["AI and Big Data"]}, url=biwin["url"] + "?copy=1")
    assert skip_reason(untagged).startswith("Marketing")
    assert not is_ai_finance_sentence(biwin["summary"])


def test_guest_explainer_is_skipped_and_has_no_finance_actor(factory):
    joonko = _sample(139)
    assert "commentary" in skip_reason(live_article(factory, joonko)).lower()
    first_sentence = joonko["summary"].split(". Guest author")[0]
    assert not is_ai_finance_sentence(first_sentence), (
        "securities-fraud liability and D&O insurance are not actors"
    )


def test_funding_announcements_are_not_ai_claims():
    slingshot, town = _sample(144), _sample(275)
    assert not is_ai_finance_sentence(slingshot["summary"]), (
        "'AI-native ... insurance startup' is a self-description"
    )
    assert not is_ai_finance_sentence(town["summary"]), "the round's venture investors are not the AI actor"


@pytest.mark.parametrize(
    ("title", "summary", "firm"),
    [
        # Live headlines whose leading capitals were logged as firm names; with a real actor sentence the firm
        # comes from that sentence, never from the Title Case headline.
        (
            "From Cloud AI to Edge AI: How BIWIN Enables the Next Generation of Storage and Memory Solutions",
            "Harbor Bank is testing AI agents that triage fraud alerts.",
            "Harbor Bank",
        ),
        (
            "When Traction Becomes Personal Exposure: What The Joonko Guilty Plea Means For Founders",
            "Banks are piloting AI agents for client onboarding.",
            "Unnamed firm",
        ),
        ("Calder Mutual says its AI now settles a third of claims", "The insurer uses AI to triage claims.", None),
    ],
)
def test_firm_is_a_real_name_or_unnamed(factory, title, summary, firm):
    fin = factory.source(vertical=Vertical.FINANCE)
    out = classify_claim_rules(factory.article(title, "", source=fin, summary=summary))
    assert out.is_ai_finance_claim
    assert out.firm == (firm or "Calder Mutual")
    assert out.firm not in {"From Cloud", "When Traction Becomes Personal"}


@pytest.mark.parametrize(
    ("title", "summary"),
    [
        ("Insurer Calder Mutual says its AI handles a third of claims", ""),
        ("Ferrow, a payments startup, uses AI to flag card fraud", ""),
        ("FINRA flags risks of AI agents at broker-dealers", ""),
        (
            "Fintech startup Ferrow raises $20M Series A",
            "Ferrow, a fintech, raised $20 million to build AI underwriting tools for lenders.",
        ),
    ],
)
def test_financial_actor_claims_are_still_found(factory, title, summary):
    fin = factory.source(vertical=Vertical.FINANCE)
    assert classify_claim_rules(factory.article(title, "", source=fin, summary=summary)).is_ai_finance_claim


@pytest.mark.parametrize(
    ("sentence", "is_claim"),
    [
        # Mentions that are not a financial-sector actor making or facing an AI claim.
        ("The Financial Times reported that Lumen Labs is building AI agents", False),
        ("Lumen Labs disclosed its AI revenue in an SEC filing", False),
        ("Nasdaq-listed AI chipmaker Tarnwell Compute jumped 20%", False),
        ("Bloomberg reported that the AI lab is raising money", False),
        ("AI Companies Brace for $100,000 H-1B Visa Fee", False),
        ("A startup used its funds to build AI tutors", False),
        # Financial-sector actors.
        ("Vanguard, the asset manager, is testing AI agents for client service", True),
        ("The SEC charged an adviser over AI claims", True),
        ("Researchers at Northbridge Bank published a note on AI portfolio agents.", True),
    ],
)
def test_financial_actor_gate(sentence, is_claim):
    assert is_ai_finance_sentence(sentence) is is_claim


class CountingClaude:
    name = "claude"
    model = "claude-test"

    def __init__(self, output: AIFinanceClaimOut):
        self.output = output
        self.calls = 0

    def generate(self, task, budget):
        self.calls += 1
        return LLMResult(output=self.output, provider="claude", model=self.model, llm_calls=1)

    def run_tools(self, task, budget):  # pragma: no cover - not used
        raise NotImplementedError


def test_agent_is_not_asked_about_sponsored_or_commentary(session, factory):
    fake = CountingClaude(AIFinanceClaimOut(is_ai_finance_claim=True, firm="From Cloud", claim_text="x"))
    orchestrator = Orchestrator(session, provider=fake)
    for live_id in (318, 139):
        assert not classify_claim(live_article(factory, _sample(live_id)), orchestrator).is_ai_finance_claim
    assert fake.calls == 0


def test_agent_firm_absent_from_the_document_is_replaced(session, factory):
    fin = factory.source(vertical=Vertical.FINANCE, trust_tier=TrustTier.INDEPENDENT)
    article = factory.article(
        "Harbor Bank pilots AI agents for anti-money laundering alerts",
        "Harbor Bank is testing AI agents that triage AML alerts in two branches.",
        source=fin,
    )
    fake = CountingClaude(
        AIFinanceClaimOut(
            is_ai_finance_claim=True, firm="Harbour Banking Group", claim_text="Harbor Bank pilots AI agents."
        )
    )
    out = classify_claim(article, Orchestrator(session, provider=fake))
    assert out.firm == "Harbor Bank" and fake.calls == 1
