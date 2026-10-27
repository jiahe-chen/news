"""Regulation Watch (``newsren.finance.regulation``): regulator, action, who/what/when.

Runs on the fictional demo corpus (an enforcement action, proposed broker-dealer
guidance, and two education-board rules), on the saved FINRA news-release feed
(tests/data/ingest/finra_news_rss.xml), and on small factory articles naming
real regulators.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from newsren.finance.regulation import (
    ACTION_TYPES,
    REGULATOR_GROUPS,
    classify_action,
    finance_regulation_items,
    parse_when,
    regulation_watch,
)
from newsren.ingest.fetchers import parse_rss
from newsren.ingest.fixtures import load_demo_data
from newsren.ingest.pipeline import store_item
from newsren.ingest.sources import sync_sources
from newsren.models import Article, DocType, Source, TrustTier, Vertical

INGEST_DATA = Path(__file__).parent / "data" / "ingest"


@pytest.fixture()
def demo_items(session) -> list[dict]:
    load_demo_data(session)
    return regulation_watch(session, days=60)


def item_for(items: list[dict], fragment: str) -> dict:
    matches = [i for i in items if fragment in i["title"]]
    assert len(matches) == 1, [i["title"] for i in items]
    return matches[0]


def published(item: dict) -> date:
    return datetime.fromisoformat(item["published_at"].removesuffix("Z")).date()


def when_of(item: dict, kind: str) -> list[dict]:
    return [w for w in item["when"] if w["kind"] == kind]


# --------------------------------------------------------------------------- demo corpus


def test_enforcement_action_regulator_and_sanction(demo_items):
    item = item_for(demo_items, "Brightvane")

    assert item["regulator"] == "Markets Integrity Commission"
    assert item["regulator_group"] == "other"
    assert item["regulator_kind"] == "securities regulator"
    assert (item["action_type"], item["stage"]) == ("enforcement", "settled")
    assert any("civil penalty of $310,000" in s for s in item["what_changes"])
    assert any("registered investment adviser" in s for s in item["who_is_affected"])
    assert item["when"] == []
    assert "Settled without admitting or denying the findings." in item["caveats"]
    # The wire story about the same case is coverage of this item, not a second item.
    assert [c["title"] for c in item["coverage"]] == [
        "Regulator fines adviser $310,000 for claiming AI agents it did not use"
    ]
    assert [c["id"] for c in item["citations"]] == ["c1", "c2"]
    assert item["citations"][0]["url"] == item["url"]
    assert item["citations"][1]["url"] == item["coverage"][0]["url"]
    assert not [i for i in demo_items if i["title"].startswith("Regulator fines adviser")]


def test_proposed_guidance_who_what_and_when(demo_items):
    item = item_for(demo_items, "BDSA proposes supervision expectations")

    assert item["regulator"] == "Broker-Dealer Standards Authority"
    assert item["regulator_kind"] == "self-regulatory organization"
    assert (item["action_type"], item["stage"]) == ("guidance", "proposed")
    assert any(s.startswith("Who is affected: all member firms") for s in item["who_is_affected"])
    assert any("firms would be expected to" in s for s in item["what_changes"])

    comment = when_of(item, "comment_deadline")
    assert len(comment) == 1
    assert comment[0]["date"] == (published(item) + timedelta(days=60)).isoformat()
    assert comment[0]["estimated"] is True
    effective = when_of(item, "effective")
    assert [(w["phrase"], w["date"]) for w in effective] == [("six months after the final notice", None)]
    assert "Proposed: the regulator may change it before it is final." in item["caveats"]


def test_education_rules_with_effective_and_comment_dates(demo_items):
    tutor = item_for(demo_items, "proposes evidence and privacy rules for AI tutors")
    assert tutor["regulator"] == "Calder State Board of Education"
    assert tutor["regulator_kind"] == "education agency"
    assert (tutor["action_type"], tutor["stage"]) == ("rule", "proposed")
    assert [(w["date"], w["estimated"]) for w in when_of(tutor, "effective")] == [("2027-07-01", False)]
    assert [w["date"] for w in when_of(tutor, "comment_deadline")] == [
        (published(tutor) + timedelta(days=45)).isoformat()
    ]
    assert any("would apply to districts and charter schools" in s for s in tutor["who_is_affected"])
    assert any("could not purchase an AI tutoring product unless" in s for s in tutor["what_changes"])

    literacy = item_for(demo_items, "Board adopts AI literacy requirement")
    assert (literacy["action_type"], literacy["stage"]) == ("rule", "final")
    dates = {(w["kind"], w["date"]) for w in literacy["when"]}
    assert {("effective", "2028"), ("effective", "2029"), ("deadline", "2027-03")} <= dates
    # The explainer about the same requirement attaches as coverage.
    assert [c["title"] for c in literacy["coverage"]] == [
        "What Calder's new AI literacy requirement means for teachers and students"
    ]


def test_items_are_well_formed_and_exclude_opinion_and_duplicates(session, demo_items):
    assert len(demo_items) == 4
    for item in demo_items:
        assert item["regulator_group"] in REGULATOR_GROUPS
        assert item["action_type"] in ACTION_TYPES
        assert item["citations"] and all(c["retrieved_at"] for c in item["citations"])
        article = session.get(Article, item["article_id"])
        assert article.duplicate_of_id is None
        assert article.doc_type != DocType.OPINION
    assert not [i for i in demo_items if i["title"].startswith(("Commentary", "Opinion", "Op-Ed"))]
    times = [i["published_at"] for i in demo_items]
    assert times == sorted(times, reverse=True)


def test_window_and_vertical_filters(session, demo_items):
    recent = regulation_watch(session, days=4)
    assert [i["title"] for i in recent] == [
        "Calder education board proposes evidence and privacy rules for AI tutors in public schools"
    ]
    finance = regulation_watch(session, days=60, vertical=Vertical.FINANCE)
    assert {i["regulator"] for i in finance} == {
        "Markets Integrity Commission",
        "Broker-Dealer Standards Authority",
    }
    assert {i["title"] for i in finance_regulation_items(demo_items)} == {i["title"] for i in finance}


# --------------------------------------------------------------------------- real regulator samples


def test_saved_finra_feed(session):
    sync_sources(session)
    source = session.scalars(select(Source).where(Source.slug == "finra-news-releases")).one()
    for fetched in parse_rss((INGEST_DATA / "finra_news_rss.xml").read_bytes(), source):
        store_item(session, source, fetched)
    items = regulation_watch(session, days=60, now=datetime(2026, 10, 27))

    revere = item_for(items, "Revere Securities")
    assert (revere["regulator"], revere["regulator_group"]) == ("FINRA", "FINRA")
    assert revere["regulator_kind"] == "self-regulatory organization"
    assert revere["action_type"] == "enforcement"
    assert any("required Revere to retain a third-party consultant" in s for s in revere["what_changes"])
    board = item_for(items, "Board of Governors")
    assert (board["action_type"], board["stage"]) == ("rule", "proposed")
    # Investor-education launches and research releases are not regulatory actions.
    assert len(items) == 2


def _regulator_article(factory, title: str, body: str, *, name: str, vertical: str = Vertical.FINANCE) -> Article:
    source = factory.source(
        vertical=vertical, trust_tier=TrustTier.PRIMARY, doc_type=DocType.REGULATORY, name=name
    )
    return factory.article(title, body, source=source, published_at=datetime(2026, 10, 1, 12))


def test_known_regulators_actions_and_dates(session, factory):
    _regulator_article(
        factory,
        "SEC Charges Investment Adviser With Misleading Claims About AI",
        "The Securities and Exchange Commission today charged an investment adviser with misleading clients. "
        "The firm agreed to pay a $250,000 civil penalty.",
        name="SEC Press Releases",
    )
    _regulator_article(
        factory,
        "Federal Reserve Board announces final rule on bank model risk",
        "The final rule is effective January 1, 2027. Covered banks must comply by June 30, 2027.",
        name="Federal Reserve Board - Press Releases",
    )
    _regulator_article(
        factory,
        "Department of Education issues guidance on AI tools in schools",
        "Schools that receive federal funds must publish their AI use policies by March 31, 2027.",
        name="U.S. Department of Education",
        vertical=Vertical.EDUCATION,
    )
    items = regulation_watch(session, days=60, now=datetime(2026, 10, 27))

    sec = item_for(items, "SEC Charges")
    assert (sec["regulator"], sec["regulator_group"], sec["action_type"]) == ("SEC", "SEC", "enforcement")
    fed = item_for(items, "Federal Reserve")
    assert (fed["regulator_group"], fed["regulator_kind"]) == ("Federal Reserve", "central bank")
    assert (fed["action_type"], fed["stage"]) == ("rule", "final")
    assert {(w["kind"], w["date"]) for w in fed["when"]} == {
        ("effective", "2027-01-01"),
        ("compliance", "2027-06-30"),
    }
    edu = item_for(items, "Department of Education")
    assert (edu["regulator_group"], edu["action_type"]) == ("Department of Education", "guidance")
    assert [(w["kind"], w["date"]) for w in edu["when"]] == [("deadline", "2027-03-31")]
    assert {i["regulator"] for i in finance_regulation_items(items)} == {"SEC", "Federal Reserve"}


def test_parse_when_and_classify_action_units():
    mentions = parse_when(
        "Comments are due within 30 days. The rule takes effect 90 days after publication in the Federal "
        "Register. Firms must comply by June 30, 2027. The program was first announced on March 3, 2024.",
        datetime(2026, 10, 1),
    )
    assert [(m["kind"], m["date"], m["estimated"]) for m in mentions] == [
        ("comment_deadline", "2026-10-31", True),
        ("effective", "2026-12-30", True),
        ("compliance", "2027-06-30", False),
    ]
    assert parse_when("Comments are due within 30 days.") == []  # no publication date to count from

    assert classify_action("FINRA Fines Firm $800,000") == ("enforcement", "announced")
    assert classify_action("Regulator requests comment on proposed rule changes") == ("rule", "proposed")
    assert classify_action("FOMC statement on the federal funds rate")[0] == "statement"
    assert classify_action("FINRA Makes Investor Education Content Available")[0] is None
