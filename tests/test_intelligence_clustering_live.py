"""Story clustering on real headlines from approved live sources (tests/data/intelligence/live_sample.json).

A run on live feeds merged unrelated items through publisher boilerplate:
60 EDGAR filings from different companies ("X files Form 10-Q (quarterly
report)"), Federal Reserve releases about different banks, SEC enforcement
releases against different respondents, NIST News items that only share
"NIST", and BBC "Tech Life" episodes that only share the programme name.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pytest

from newsren.intelligence.clustering import assign_stories, title_anchors
from newsren.models import Article, DocType, Source, utcnow

DATA = Path(__file__).parent / "data" / "intelligence" / "live_sample.json"
SAMPLE = json.loads(DATA.read_text())


class Live:
    """The loaded sample: sources by slug and articles in file order."""

    def __init__(self, factory) -> None:
        self.factory = factory
        self.sources: dict[str, Source] = {}
        for row in SAMPLE["sources"]:
            src = Source(
                slug=row["slug"],
                name=row["name"],
                url=row["url"],
                homepage=row["homepage"],
                vertical=row["vertical"],
                trust_tier=row["trust_tier"],
                default_doc_type=row["default_doc_type"],
            )
            factory.session.add(src)
            self.sources[row["slug"]] = src
        factory.session.flush()
        now = utcnow()
        self.articles: list[Article] = [
            factory.article(
                row["title"],
                "",
                source=self.sources[row["source"]],
                vertical=row["vertical"],
                doc_type=row["doc_type"],
                trust_tier=row["trust_tier"],
                summary=row["summary"],
                published_at=now - timedelta(hours=row["hours_ago"] + 1),
                extra=row["extra"],
            )
            for row in SAMPLE["articles"]
        ]

    def of(self, slug: str) -> list[Article]:
        return [a for a in self.articles if a.source.slug == slug]

    def titled(self, text: str) -> list[Article]:
        found = [a for a in self.articles if text in a.title]
        assert found, text
        return found

    def one(self, text: str) -> Article:
        (article,) = self.titled(text)
        return article


@pytest.fixture()
def live(factory) -> Live:
    return Live(factory)


def story_members(articles: list[Article]) -> dict[int, list[Article]]:
    groups: dict[int, list[Article]] = defaultdict(list)
    for art in articles:
        assert art.story_id is not None
        groups[art.story_id].append(art)
    return groups


# --------------------------------------------------------------------------- SEC filings


def test_filings_from_different_companies_never_share_a_story(session, live):
    assign_stories(session)
    filings = [a for a in live.articles if a.doc_type == DocType.FILING]
    assert len(filings) == 14
    assert len({a.extra["cik"] for a in filings}) == len(filings)
    # Before: all 14 joined one "X files Form 10-Q (quarterly report)" story.
    assert len({a.story_id for a in filings}) == len(filings)
    for story_id in story_members(filings):
        in_story = [a for a in live.articles if a.story_id == story_id]
        assert len({a.extra.get("cik") for a in in_story if a.doc_type == DocType.FILING}) == 1


def test_same_company_filings_share_a_company_story(session, factory, live):
    """Rush Enterprises' earnings 8-K and its 10-Q (same CIK) form one company story; other filers stay apart."""
    assign_stories(session)
    earnings_8k = live.one("RUSH ENTERPRISES")
    assert earnings_8k.extra["is_earnings_release"] is True
    template = live.one("PayPal Holdings")  # a real 10-Q item, re-addressed to Rush's CIK
    company, cik = earnings_8k.extra["company"], earnings_8k.extra["cik"]
    ten_q = factory.article(
        f"{company} files Form 10-Q (quarterly report)",
        "",
        source=template.source,
        vertical=template.vertical,
        doc_type=DocType.FILING,
        trust_tier=template.trust_tier,
        summary=template.summary.replace(template.extra["company"], company).replace(template.extra["cik"], cik),
        published_at=earnings_8k.published_at + timedelta(days=6),
        extra={**template.extra, "cik": cik, "company": company},
    )
    other_10q = factory.article(
        template.title,
        "",
        source=template.source,
        vertical=template.vertical,
        doc_type=DocType.FILING,
        trust_tier=template.trust_tier,
        summary=template.summary,
        published_at=earnings_8k.published_at + timedelta(days=6, minutes=5),
        extra={**template.extra, "accession": "0001633917-26-000099"},
    )
    assign_stories(session, [ten_q, other_10q])
    assert ten_q.story_id == earnings_8k.story_id
    assert other_10q.story_id == template.story_id != earnings_8k.story_id


def test_news_that_names_the_filer_joins_its_filing_story_and_other_filing_news_does_not(session, factory, live):
    assign_stories(session)
    filing = live.one("PayPal Holdings")
    # Real coverage of a different company's filing must not join any EDGAR story.
    anthropic = live.one("Anthropic’s IPO Filing")
    assert all(a.story_id != anthropic.story_id for a in live.articles if a.doc_type == DocType.FILING)

    news = factory.article(
        "PayPal raises full-year profit forecast as branded checkout grows",
        "",
        source=live.sources["crunchbase-news"],
        vertical=filing.vertical,
        summary=(
            "PayPal said branded checkout volume grew in the third quarter, according to the quarterly report "
            "PayPal filed with the SEC, and raised its full-year profit forecast."
        ),
        published_at=filing.published_at + timedelta(hours=3),
    )
    assign_stories(session, [news])
    assert news.story_id == filing.story_id


# --------------------------------------------------------------------------- same-publisher releases


def test_federal_reserve_releases_about_different_banks_are_separate_stories(session, live):
    assign_stories(session)
    fed = live.of("federal-reserve-press")
    actions = [a for a in fed if "enforcement action" in a.title or "approval of application" in a.title.lower()]
    assert len(actions) >= 8
    # Before: twelve of these releases formed one "Federal Reserve Board announces ..." story.
    assert len({a.story_id for a in actions}) == len(actions)
    # Releases about one FOMC meeting are still one story.
    statement = live.one("Federal Reserve issues FOMC statement")
    projections = live.one("release economic projections from the September 15-16")
    assert statement.story_id == projections.story_id
    groups = story_members(fed)
    assert max(len(m) for m in groups.values()) <= 3


def test_sec_press_releases_are_not_grouped_by_their_template(session, live):
    assign_stories(session)
    sec = live.of("sec-press-releases")
    charges = [a for a in sec if a.title.startswith("SEC Charges")]
    assert len(charges) >= 4
    # Distinct enforcement actions against different respondents are different stories.
    assert len({a.story_id for a in charges}) == len(charges)
    assert len({a.story_id for a in sec}) == len(sec)


def test_nist_news_items_are_not_grouped_by_the_agency_name(session, live):
    assign_stories(session)
    nist = live.of("nist-news")
    assert sum("NIST" in a.title for a in nist) >= 10
    assert len({a.story_id for a in nist}) == len(nist)


def test_a_publisher_naming_itself_is_not_a_headline_anchor(session, live):
    assert "nist" not in title_anchors(session, live.one("NIST Expands AI Consortium"))
    assert "federal reserve" not in title_anchors(
        session, live.one("Federal Reserve Board announces approval of application by BancFirst")
    )
    assert "sec" not in title_anchors(session, live.one("SEC Charges Meyer Global Management"))
    # Other publishers' headlines keep their anchors.
    assert "php" in title_anchors(session, live.one("Critical security issue affecting PHP"))


# --------------------------------------------------------------------------- repeated templates


def test_repeated_programme_titles_do_not_form_a_story(session, live):
    assign_stories(session)
    episodes = live.titled("Tech Life")
    assert len(episodes) == 4 and len({a.title for a in episodes}) == 1
    assert len({a.story_id for a in episodes}) == 4  # before: three episodes formed one "Tech Life" story


def test_newsletter_template_sentence_is_not_a_shared_story(session, live):
    assign_stories(session)
    editions = live.titled("The Download:")
    assert len(editions) == 3
    assert len({a.story_id for a in editions}) == 3


def test_real_multi_source_story_still_forms(session, live):
    """Positive control: the PHP vulnerability advisory and the BBC's PHP coverage remain one story."""
    assign_stories(session)
    advisory = live.one("Critical security issue affecting PHP")
    coverage = [live.one("What is PHP and why"), live.one("frontline of PHP cyber-security scramble")]
    assert {a.story_id for a in coverage} == {advisory.story_id}
    china = live.titled("Defending against China-nexus covert networks")
    assert len({a.story_id for a in china}) == 1
