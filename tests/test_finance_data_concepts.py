"""Finance concepts for learners: content checks, lookup, and linking text to concepts."""

from __future__ import annotations

import re

import pytest

from newsren.finance.concepts import (
    CATEGORY_LABELS,
    CONCEPTS,
    concept_mentions,
    concepts_by_category,
    concepts_in_text,
    get_concept,
    related_concepts,
)

REQUIRED = {
    "backtesting",
    "overfitting",
    "in-sample-vs-out-of-sample",
    "look-ahead-bias",
    "survivorship-bias",
    "sharpe-ratio",
    "maximum-drawdown",
    "benchmark",
    "transaction-costs",
    "dilution",
    "term-sheet",
    "pre-money-vs-post-money-valuation",
    "liquidation-preference",
    "earnings-guidance",
    "eps",
    "revenue-vs-net-income",
    "operating-margin",
    "sec-forms-10q-10k-8k",
    "xbrl",
    "ai-washing",
    "reg-bi-fiduciary-duty",
    "investment-adviser-registration",
}


def slugs(text: str) -> list[str]:
    return [c.slug for c in concepts_in_text(text)]


def test_required_concepts_present_with_unique_slugs():
    all_slugs = [c.slug for c in CONCEPTS]
    assert len(all_slugs) == len(set(all_slugs))
    assert set(all_slugs) >= REQUIRED


@pytest.mark.parametrize("concept", CONCEPTS, ids=lambda c: c.slug)
def test_concept_content_is_complete_and_educational(concept):
    assert concept.term and concept.short.endswith(".")
    assert 2 <= len(concept.paragraphs) <= 4, "explainer should be 2-4 paragraphs"
    assert concept.example.startswith("Hypothetical example:")
    assert concept.pitfalls, "every concept lists pitfalls"
    assert concept.category in CATEGORY_LABELS
    assert all(get_concept(slug) is not None for slug in concept.related), "related slugs must resolve"
    assert concept.slug not in concept.related
    text = " ".join([concept.short, concept.explainer, concept.example, *concept.pitfalls]).lower()
    for banned in (
        "you should buy",
        "you should sell",
        "price target of",
        "guaranteed return",
        "will beat the market",
    ):
        assert banned not in text


def test_concepts_link_to_themselves_by_their_own_aliases():
    for concept in CONCEPTS:
        for alias in concept.aliases:
            assert concept in concepts_in_text(f"A story mentioning {alias} here."), (concept.slug, alias)


def test_get_concept_by_slug_alias_and_term():
    assert get_concept("sharpe-ratio").term == "Sharpe ratio"
    assert get_concept("drawdown").slug == "maximum-drawdown"
    assert get_concept("guidance").slug == "earnings-guidance"
    assert get_concept("10-Q").slug == "sec-forms-10q-10k-8k"
    assert get_concept("Sharpe ratio").slug == "sharpe-ratio"
    assert get_concept("XBRL").slug == "xbrl"
    assert get_concept("no-such-concept") is None
    assert get_concept("") is None


def test_concepts_in_text_orders_by_first_mention():
    text = (
        "In a 2005-2024 backtest, the agents beat a 60/40 portfolio with a higher Sharpe ratio and a smaller "
        "maximum drawdown, but the authors warn the result is in-sample and assumed low transaction costs."
    )
    assert slugs(text) == [
        "backtesting",
        "benchmark",
        "sharpe-ratio",
        "maximum-drawdown",
        "in-sample-vs-out-of-sample",
        "transaction-costs",
    ]
    mentions = dict((c.slug, phrase) for c, phrase in concept_mentions(text))
    assert mentions["backtesting"] == "backtest"


def test_startup_and_earnings_vocabulary():
    assert slugs("Quillfield raised a $45 million Series B at an undisclosed valuation.") == [
        "funding-rounds",
        "pre-money-vs-post-money-valuation",
    ]
    assert slugs("The term sheet includes a 2x liquidation preference and an expanded option pool.") == [
        "term-sheet",
        "liquidation-preference",
        "dilution",
    ]
    earnings = (
        "Corvane guides fourth-quarter revenue of $3.6 billion; diluted EPS was $1.12 and gross margin 47.8%."
    )
    assert slugs(earnings) == ["earnings-guidance", "eps", "operating-margin"]
    assert "sec-forms-10q-10k-8k" in slugs("The company filed its 10-Q and an 8-K with the SEC.")
    assert "ai-washing" in slugs("Regulators are cracking down on AI-washing by advisers.")


def test_word_boundaries_and_case_avoid_false_links():
    # Generic words outside finance phrasing do not link.
    assert slugs("Lumen-4 scored 84.3% on the MathBench-Pro benchmark in an independent evaluation.") == []
    assert slugs("The board proposed guidance requiring member firms to inventory AI agents.") == []
    # Acronyms are matched case-sensitively and on word boundaries.
    assert "eps" not in slugs("The steps were reviewed; reps and preps continued.")
    assert "eps" in slugs("EPS rose 10%.")
    # "Series" needs a capitalised round letter.
    assert "funding-rounds" not in slugs("a TV series about money and a series of events")
    assert "backtesting" not in slugs("Their feedback testing went well.")
    assert "backtesting" in slugs("The back-tested strategy was published.")


def test_grouping_and_related_helpers():
    grouped = concepts_by_category()
    assert list(grouped)[: len(CATEGORY_LABELS)] == list(CATEGORY_LABELS)
    assert sum(len(v) for v in grouped.values()) == len(CONCEPTS)
    assert [c.slug for c in related_concepts(get_concept("backtesting"))][:2] == [
        "overfitting",
        "in-sample-vs-out-of-sample",
    ]


def test_ai_washing_explainer_cites_the_documented_penalties():
    text = get_concept("ai-washing").explainer
    assert re.search(r"Delphia.*Global\s+Predictions", text, re.S)
    assert "$225,000" in text and "$175,000" in text and "March 2024" in text
