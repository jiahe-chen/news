"""Ask Newsren retrieval on live-shaped data: query analysis, archive ranking, arXiv query terms, dates.

The articles are real items from the approved live sources (headline + feed
summary, as ingested), saved in tests/data/research/live_articles.json; the
regressions they pin were found by running Ask Newsren against the live
archive.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from newsren.agents.tools import (
    ResearchToolbox,
    archive_search,
    arxiv_query_params,
    arxiv_terms,
    query_profile,
    stem_supported,
    stem_word,
    text_stems,
)
from newsren.models import Article, Source
from newsren.textutil import canonical_url, content_hash, parse_datetime, simhash_hex

DATA = Path(__file__).parent / "data" / "research"


def load_live_articles(session) -> dict[int, Article]:
    """Load the saved live articles (keyed by their id in the live archive) into the test database."""
    rows = json.loads((DATA / "live_articles.json").read_text())["articles"]
    sources: dict[str, Source] = {}
    out: dict[int, Article] = {}
    for row in rows:
        meta = row["source"]
        source = sources.get(meta["slug"])
        if source is None:
            source = Source(
                slug=meta["slug"],
                name=meta["name"],
                url=meta["url"],
                homepage=meta["homepage"],
                vertical=meta["vertical"],
                trust_tier=meta["trust_tier"],
                default_doc_type=meta["default_doc_type"],
                approved=True,
            )
            session.add(source)
            sources[meta["slug"]] = source
        article = Article(
            source=source,
            url=row["url"],
            canonical_url=canonical_url(row["url"]),
            title=row["title"],
            summary=row["summary"],
            content=row["content"],
            published_at=parse_datetime(row["published_at"]) if row["published_at"] else None,
            vertical=row["vertical"],
            doc_type=row["doc_type"],
            trust_tier=row["trust_tier"],
            content_hash=content_hash(row["title"], row["summary"]),
            simhash=simhash_hex(f"{row['title']} {row['summary']}"),
            extra=row["extra"],
        )
        session.add(article)
        out[row["live_id"]] = article
    session.flush()
    return out


@pytest.fixture()
def live(session) -> dict[int, Article]:
    return load_live_articles(session)


# --------------------------------------------------------------------------
# Query analysis
# --------------------------------------------------------------------------


def test_query_units_keep_names_whole_and_split_question_words():
    def units(q: str) -> list[tuple[str, str]]:
        return [(u.text, u.kind) for u in query_profile(None, q).units]

    assert units("What is happening with the UK Budget and income tax?") == [
        ("UK", "acronym"),
        ("Budget", "name"),
        ("income", "word"),
        ("tax", "word"),
    ]
    # "Is Intel" is a question opener plus a name, not a two-word name.
    assert units("Is Intel getting more CHIPS Act subsidies?")[:2] == [("Intel", "name"), ("CHIPS Act", "name")]
    assert units("What is the status of NASA's Mars Sample Return mission?")[:2] == [
        ("NASA", "acronym"),
        ("Mars Sample Return", "name"),
    ]
    assert units("What did the Federal Reserve announce about enforcement actions?")[0] == (
        "Federal Reserve",
        "name",
    )


def test_question_verbs_and_generic_nouns_are_weak():
    profile = query_profile(None, "What has the SEC charged firms with recently?")
    weak = {u.text for u in profile.units if u.weak}
    assert weak == {"firms"}
    ncsc = query_profile(None, "What security vulnerabilities has the NCSC warned about?")
    assert {u.text for u in ncsc.units if u.weak} == {"warned"}


def test_acronyms_match_case_sensitively_and_through_aliases():
    sec = query_profile(None, "SEC charges").units[0]
    assert sec.matches("The Securities and Exchange Commission today charged ...")
    assert sec.matches("SEC Charges Two Individuals")
    ups = query_profile(None, "UPS rates").units[0]
    assert ups.matches("UPS plans 5.9% rate increase on Dec. 21")
    assert not ups.matches("The start-ups hoping to return battery making to the US")
    # A one-word name is a proper noun: "Carrier" (the company) is not "the carrier's" (UPS).
    carrier = query_profile(None, "How much did Carrier pay for its Fort Worth campus?").units[0]
    assert carrier.text == "Carrier" and carrier.matches("CARRIER GLOBAL CORP files Form 8-K")
    assert not carrier.matches("The carrier's average rate bump will match FedEx's upcoming increase")
    ai = query_profile(None, "AI tutors").units[0]
    assert ai.matches("The Implications of Generative Artificial Intelligence for Mathematics Education")


def test_stem_supported_accepts_derived_forms_only():
    assert stem_supported(stem_word("robotics"), text_stems("robot makers"))
    assert stem_supported(stem_word("tutors"), text_stems("high-dosage tutoring programs"))
    # "election" is not "electric" (live regression: a truck-battery article ranked for an election question).
    assert not stem_supported(stem_word("election"), text_stems("Simulation App Speeds Up Electric Truck Battery"))


def test_weights_follow_archive_rarity(session, live):
    profile = query_profile(session, "What security vulnerabilities has the NCSC warned about?")
    weights = {u.text: u.weight for u in profile.units}
    assert weights["vulnerabilities"] > weights["security"], "rarer words weigh more"
    assert weights["warned"] < weights["security"], "question verbs weigh little"
    absent = query_profile(session, "What did the European Central Bank decide about interest rates?")
    assert absent.subject_absent and not absent.on_topic("5 Interesting Startup Deals You May Have Missed")


# --------------------------------------------------------------------------
# Archive ranking
# --------------------------------------------------------------------------


def test_archive_search_ranks_specific_matches_first(session, settings, live):
    def titles(query: str) -> list[str]:
        return [a.title for a in archive_search(session, query)]

    assert titles("What is UPS changing about its rates?")[0] == "UPS plans 5.9% rate increase on Dec. 21"
    assert "The start-ups hoping to return battery making to the US" not in titles("UPS rates")[:3]
    assert titles("What did NIST publish about post-quantum cryptography?")[0] == (
        "NIST Report Provides Guidelines for Transition to Post-Quantum Cryptography Standards"
    )
    assert titles("How is Kroger using RFID and computer vision?")[0].startswith("Kroger taps RFID")
    # Before: the SEC releases outranked Federal Reserve items whose only text is the headline.
    fed = titles("What did the Federal Reserve announce about enforcement actions?")[:4]
    assert all(t.startswith("Federal Reserve Board") and "enforcement action" in t for t in fed)

    box = ResearchToolbox(session, settings, allow_network=False)
    result = box.search_archive({"query": "UPS rate increase", "vertical": None, "since_days": None})
    relevance = [row["relevance"] for row in result.data]
    assert relevance[0] == max(relevance) > 0.5


def test_headline_only_items_get_a_snippet(session, settings, live):
    box = ResearchToolbox(session, settings, allow_network=False)
    result = box.search_archive(
        {"query": "Federal Reserve enforcement action", "vertical": None, "since_days": None}
    )
    assert result.data and all(row["snippet"] for row in result.data)


def test_year_only_publication_dates_are_shown(session, settings, live):
    eric = live[61]  # ERIC record: publication year only
    assert eric.published_at is None and eric.extra["publication_year"] == 2025
    box = ResearchToolbox(session, settings, allow_network=False)
    result = box.get_article({"article_id": eric.id})
    assert "published 2025" in result.content and "publication date unknown" not in result.content
    assert result.data["date_label"] == "2025"


# --------------------------------------------------------------------------
# arXiv query construction
# --------------------------------------------------------------------------


def test_arxiv_terms_drop_evidence_words_and_phrase_acronyms():
    tutor = "Is there independent evidence that AI tutors improve learning?"
    assert arxiv_terms(tutor) == ["ai tutors", "improve", "learning"]
    assert arxiv_query_params(tutor)["search_query"] == 'all:"ai tutors" AND all:improve AND all:learning'
    literacy = "What does recent research say about students' AI literacy?"
    assert arxiv_query_params(literacy)["search_query"] == 'all:students AND all:"ai literacy"'
    assert arxiv_terms("What did NIST publish about post-quantum cryptography?") == [
        "nist",
        "post-quantum",
        "cryptography",
    ]
