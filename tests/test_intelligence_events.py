"""Rule-based event classification (Story Intelligence + monitoring)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from newsren.intelligence.events import classify_event, classify_event_with_reason, event_label, event_phrase
from newsren.models import EventType

LIVE = {
    row["key"]: row
    for row in json.loads((Path(__file__).parent / "data" / "briefs" / "live_samples.json").read_text())[
        "articles"
    ]
}


def subject(title: str, summary: str = "", doc_type: str = "news", trust_tier: str = "independent"):
    return SimpleNamespace(title=title, summary=summary, content="", doc_type=doc_type, trust_tier=trust_tier)


CASES = [
    # correction outranks everything
    ("Correction: Atlas-3 benchmark figure revised", "", "news", "independent", EventType.CORRECTION),
    (
        "SEC charges adviser over AI claims",
        "An earlier version of this article misstated the penalty.",
        "news",
        "independent",
        EventType.CORRECTION,
    ),
    (
        "Journal retracts AI tutoring study",
        "The paper was withdrawn.",
        "news",
        "independent",
        EventType.CORRECTION,
    ),
    ("Market correction wipes out chip gains", "Stocks fell 10%.", "news", "independent", EventType.OTHER),
    # commentary by format beats cue words about benchmarks or launches
    (
        "The Atlas-3 benchmark is beside the point",
        "Independent benchmarks miss what teachers need.",
        "opinion",
        "commentary",
        EventType.OPINION,
    ),
    ("Analysis: what the SEC's AI-washing cases mean", "", "news", "independent", EventType.ANALYSIS),
    ("Why independent benchmarks matter", "", "blog", "commentary", EventType.ANALYSIS),
    # enforcement
    ("SEC charges two advisers with AI washing", "", "news", "independent", EventType.ENFORCEMENT),
    ("FINRA fines broker-dealer over surveillance lapses", "", "regulatory", "primary", EventType.ENFORCEMENT),
    ("Authors sue AI startup over training data", "", "news", "independent", EventType.OTHER),
    # security advisories
    (
        "CISA warns of actively exploited flaw in VPN appliances",
        "",
        "news",
        "independent",
        EventType.SECURITY_ADVISORY,
    ),
    ("Vendor bulletin for router firmware", "", "advisory", "primary", EventType.SECURITY_ADVISORY),
    # earnings and filings
    (
        "Kestrel Semiconductor reports third-quarter results",
        "Revenue rose 12% and net income doubled.",
        "news",
        "independent",
        EventType.EARNINGS,
    ),
    ("Kestrel Semiconductor Q3 2026 call", "", "earnings_call", "primary", EventType.EARNINGS),
    ("Kestrel Semiconductor 10-Q", "Quarterly report for the period.", "filing", "primary", EventType.FILING),
    ("Kestrel files 8-K disclosing fab delay", "", "news", "independent", EventType.FILING),
    # funding
    ("Brightpath Robotics raises $40 million Series B", "", "news", "independent", EventType.FUNDING),
    (
        "Tutoring startup closes seed round led by Harbor Point Ventures",
        "",
        "news",
        "independent",
        EventType.FUNDING,
    ),
    # evaluations vs vendor benchmark claims
    (
        "Independent benchmark finds Atlas-3 scores lower than claimed",
        "Researchers at the Open Evaluation Institute evaluated the model.",
        "news",
        "independent",
        EventType.INDEPENDENT_EVALUATION,
    ),
    (
        "Atlas-3 struggles on unseen algebra problems",
        "Researchers at State University tested the model and found errors.",
        "news",
        "independent",
        EventType.INDEPENDENT_EVALUATION,
    ),
    (
        "Atlas-3 tops MathBench benchmark",
        "Nimbus Labs says its model beat rivals.",
        "press_release",
        "vendor",
        EventType.BENCHMARK,
    ),
    (
        "Atlas-3 tops MathBench benchmark, independent tests confirm",
        "",
        "press_release",
        "vendor",
        EventType.BENCHMARK,
    ),
    (
        "Atlas-3 tops math leaderboard",
        "Nimbus Labs claims its model is state of the art.",
        "news",
        "independent",
        EventType.BENCHMARK,
    ),
    (
        "Evaluating Atlas-3 on high-school algebra",
        "We benchmark the model.",
        "research_preprint",
        "primary",
        EventType.INDEPENDENT_EVALUATION,
    ),
    # research
    (
        "Scaling laws for tutoring models",
        "We report benchmark gains.",
        "research_preprint",
        "primary",
        EventType.RESEARCH_PUBLICATION,
    ),
    (
        "Study finds AI tutor improves exam scores",
        "A randomized controlled trial with 1,200 students.",
        "news",
        "independent",
        EventType.RESEARCH_PUBLICATION,
    ),
    # regulation and policy
    ("SEC proposes new rule on predictive data analytics", "", "news", "independent", EventType.REGULATION),
    ("Oversight report highlights agent risks", "", "regulatory", "primary", EventType.REGULATION),
    ("Department of Education issues AI guidance for schools", "", "news", "independent", EventType.POLICY),
    (
        "Fed holds interest rates steady",
        "The Federal Reserve left its benchmark rate unchanged.",
        "news",
        "independent",
        EventType.POLICY,
    ),
    # launches and adoption
    (
        "Nimbus Labs launches Atlas-3",
        "Atlas-3 scores 92% on our internal benchmark.",
        "press_release",
        "vendor",
        EventType.PRODUCT_LAUNCH,
    ),
    (
        "Nimbus Labs debuts Atlas-3 math model",
        "The company claimed a top score on the MathBench benchmark.",
        "news",
        "independent",
        EventType.PRODUCT_LAUNCH,
    ),
    ("Riverside district adopts Atlas-3 after launch", "", "news", "independent", EventType.ADOPTION),
    ("JPMorgan rolls out LLM Suite to 50,000 staff", "", "news", "independent", EventType.ADOPTION),
    # opinion / analysis by cue, announcement, other
    ("The case against AI tutors in algebra class", "", "news", "independent", EventType.ANALYSIS),
    (
        "Kestrel Semiconductor announces plans to expand Arizona fab",
        "",
        "press_release",
        "vendor",
        EventType.ANNOUNCEMENT,
    ),
    ("Weather delays school openings", "", "news", "independent", EventType.OTHER),
    # regressions found on the demo corpus
    (
        "Lumen Labs unveils Lumen-4; benchmark claims not yet independently verified",
        "",
        "news",
        "independent",
        EventType.PRODUCT_LAUNCH,
    ),
    (
        "Atlas-3 benchmark results await outside review",
        "No independent evaluation has been published; Nimbus Labs says its model leads.",
        "news",
        "independent",
        EventType.BENCHMARK,
    ),
    (
        "Regulatory Notice: authority proposes supervision expectations for AI agents",
        "Firms should evaluate and test agents before deployment.",
        "regulatory",
        "primary",
        EventType.REGULATION,
    ),
    (
        "Northbridge says AI portfolio agents will enter a year of paper trading",
        "",
        "news",
        "independent",
        EventType.OTHER,
    ),
    (
        "Pallas Networks patches VPN flaw already used in attacks",
        "",
        "news",
        "independent",
        EventType.SECURITY_ADVISORY,
    ),
    (
        "Corvane's capacity plan: what 40,000 wafers a month would mean",
        "",
        "news",
        "independent",
        EventType.ANALYSIS,
    ),
    (
        "Survey: 61% of manufacturers plan to retrain inspectors",
        "",
        "news",
        "independent",
        EventType.RESEARCH_PUBLICATION,
    ),
]


@pytest.mark.parametrize(("title", "summary", "doc_type", "tier", "expected"), CASES)
def test_classify_event_rules(title, summary, doc_type, tier, expected):
    event_type, rule = classify_event_with_reason(subject(title, summary, doc_type, tier))
    assert event_type == expected, f"rule {rule!r} fired for {title!r}"


def test_vendor_benchmark_claim_is_never_independent_evaluation():
    vendor = subject("Independent third-party benchmark confirms Atlas-3 lead", "", "press_release", "vendor")
    assert classify_event(vendor) == EventType.BENCHMARK


def test_classify_event_returns_plain_string_for_real_articles(factory):
    art = factory.article(
        "Brightpath Robotics raises $40 million Series B",
        "Brightpath Robotics raised $40 million led by Harbor Point Ventures.",
        vertical="finance",
    )
    result = classify_event(art)
    assert result == "funding"
    assert type(result) is str


def test_uses_body_when_summary_missing():
    art = SimpleNamespace(
        title="Update on the program",
        summary=None,
        content="Kestrel Semiconductor raised $200 million in a Series C round.",
        doc_type="news",
        trust_tier="independent",
    )
    assert classify_event(art) == EventType.FUNDING


def test_event_labels_are_reader_facing():
    assert event_label("independent_evaluation") == "independent evaluation"
    assert event_label("benchmark") == "benchmark claim"
    assert event_label(None) == "development"


def test_event_phrases_take_the_right_article_in_running_text():
    assert event_phrase("independent_evaluation") == "an independent evaluation"
    assert event_phrase("research_publication") == "a research publication"
    assert event_phrase("adoption") == "adoption"
    assert event_phrase("earnings") == "earnings"
    assert event_phrase("opinion") == "an opinion piece"
    assert event_phrase(None) == "a development"


# --------------------------------------------------------------------------
# Regressions from the live run (real titles and summaries, tests/data/briefs/live_samples.json)
# --------------------------------------------------------------------------

LIVE_CASES = [
    # ERIC papers: "Evaluation" as a topic in a title or the study design is not an evaluation of claims.
    ("eric_literacy_1", EventType.RESEARCH_PUBLICATION),  # "... Student Adoption, Interaction, Evaluation and ..."
    ("evaluation_study_1", EventType.RESEARCH_PUBLICATION),  # "...: An Evaluation Study"
    ("arxiv_off_policy_evaluation", EventType.RESEARCH_PUBLICATION),  # off-policy evaluation is a method
    ("arxiv_model_evaluation", EventType.INDEPENDENT_EVALUATION),  # "Evaluating DeepSeek V4.1 ..."
    ("arxiv_vulnerability_detection", EventType.RESEARCH_PUBLICATION),  # a paper, not a security advisory
    ("imec_research_summary", EventType.RESEARCH_PUBLICATION),  # "Researchers at imec published ..."
    # A superintendent's job review is not an independent evaluation.
    ("denver_board_1", EventType.OTHER),
    # "Code-execution flaw ... allows attackers to execute code" is a security story.
    ("uefi_flaw_1", EventType.SECURITY_ADVISORY),
    # "energy bill help" is not legislation; "- what could be in the Budget?" is a preview.
    ("bbc_budget_2", EventType.ANALYSIS),
    # Benchmark the venture firm.
    ("benchmark_firm_partners", EventType.OTHER),
    ("benchmark_firm_pact", EventType.OTHER),
    # A statement about AI evaluations, and "a shift toward more deliberate evaluation" in a summary.
    ("ncsc_statement", EventType.ANNOUNCEMENT),
    ("school_ai_bans", EventType.POLICY),
]


@pytest.mark.parametrize(("key", "expected"), LIVE_CASES)
def test_live_articles_are_classified_by_what_they_are(key, expected):
    row = LIVE[key]
    article = subject(row["title"], row["summary"] or "", row["doc_type"], row["trust_tier"])
    event_type, rule = classify_event_with_reason(article)
    assert event_type == expected, f"rule {rule!r} fired for {row['title']!r}"


def test_two_research_index_papers_are_both_research_publications():
    """The live story that read "moved from an independent evaluation to research publication"."""
    types = {
        classify_event(subject(LIVE[k]["title"], LIVE[k]["summary"], LIVE[k]["doc_type"], LIVE[k]["trust_tier"]))
        for k in ("eric_literacy_1", "eric_literacy_2")
    }
    assert types == {EventType.RESEARCH_PUBLICATION}
