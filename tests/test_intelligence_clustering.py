"""Incremental story clustering and cross-story connections (Story Intelligence)."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from newsren.intelligence.clustering import (
    CROSS_VERTICAL_MARGIN,
    assign_stories,
    match_score,
    related_stories,
)
from newsren.models import Article, Story, utcnow

DATA = Path(__file__).parent / "data" / "intelligence" / "storylines.json"


def load_storylines(factory) -> dict[str, Article]:
    """Create the synthetic storyline articles; returns key -> Article."""
    now = utcnow()
    sources = {}
    articles: dict[str, Article] = {}
    rows = json.loads(DATA.read_text())["articles"]
    for row in rows:
        src_key = (row["publisher"], row["vertical"])
        if src_key not in sources:
            sources[src_key] = factory.source(
                vertical=row["vertical"],
                trust_tier=row["trust_tier"],
                doc_type=row["doc_type"],
                name=row["publisher"],
            )
        articles[row["key"]] = factory.article(
            row["title"],
            row["summary"],
            source=sources[src_key],
            vertical=row["vertical"],
            doc_type=row["doc_type"],
            trust_tier=row["trust_tier"],
            summary=row["summary"],
            published_at=now - timedelta(days=row["days_ago"]) + timedelta(hours=row.get("hours_offset", 0)),
        )
    for row in rows:
        if row.get("duplicate_of"):
            articles[row["key"]].duplicate_of_id = articles[row["duplicate_of"]].id
    factory.session.flush()
    return articles


def storyline_keys(articles: dict[str, Article], storyline: str) -> list[str]:
    rows = json.loads(DATA.read_text())["articles"]
    return [r["key"] for r in rows if r["storyline"] == storyline]


@pytest.fixture()
def storylines(factory) -> dict[str, Article]:
    return load_storylines(factory)


def test_two_storylines_and_noise_are_grouped_correctly(session, storylines):
    changed = assign_stories(session)

    atlas = {storylines[k].story_id for k in storyline_keys(storylines, "atlas")}
    kestrel = {storylines[k].story_id for k in storyline_keys(storylines, "kestrel")}
    assert len(atlas) == 1 and None not in atlas
    assert len(kestrel) == 1 and None not in kestrel
    assert atlas != kestrel

    noise_stories = [storylines[k].story_id for k in storyline_keys(storylines, "noise")]
    assert len(set(noise_stories)) == len(noise_stories)  # each noise article stands alone
    assert not (set(noise_stories) & (atlas | kestrel))

    total = session.scalar(select(func.count()).select_from(Story))
    assert total == 2 + len(noise_stories)
    assert {s.id for s in changed} == {s for s in atlas | kestrel | set(noise_stories)}


def test_story_fields_are_maintained(session, storylines):
    assign_stories(session)
    story = storylines["atlas_launch"].story
    members = [a for a in story.articles if a.duplicate_of_id is None]
    assert story.first_seen_at == min(a.published_at for a in members)
    assert story.last_updated_at == max(a.published_at for a in members)
    assert story.status == "active"
    assert story.vertical == "technology"
    assert "atlas-3" in story.keywords
    assert story.key_entities[:2] == ["Atlas-3", "Nimbus Labs"]
    # Provisional title is a cleaned member headline, never the opinion piece.
    assert story.title in {a.title for a in members}
    assert not story.title.lower().startswith("opinion")


def test_duplicates_follow_their_original(session, storylines):
    assign_stories(session)
    original = storylines["atlas_eval_news"]
    copy = storylines["atlas_eval_syndicated"]
    assert copy.story_id == original.story_id


def test_duplicate_alone_pulls_in_its_original_and_is_not_a_story_change(session, storylines):
    copy = storylines["atlas_eval_syndicated"]
    original = storylines["atlas_eval_news"]
    changed = assign_stories(session, [copy])
    assert original.story_id is not None
    assert copy.story_id == original.story_id
    assert [s.id for s in changed] == [original.story_id]

    # A second syndicated copy joining an existing story is not a change.
    late_copy = original.__class__(
        source=copy.source,
        url="https://wire.example/copy-2",
        canonical_url="https://wire.example/copy-2",
        title=copy.title,
        summary=copy.summary,
        content=copy.content,
        published_at=copy.published_at,
        vertical=copy.vertical,
        doc_type=copy.doc_type,
        trust_tier=copy.trust_tier,
        content_hash=copy.content_hash,
        simhash=copy.simhash,
        duplicate_of_id=original.id,
    )
    session.add(late_copy)
    session.flush()
    assert assign_stories(session, [late_copy]) == []
    assert late_copy.story_id == original.story_id


def test_assign_is_incremental_and_idempotent(session, storylines):
    assign_stories(session)
    before = session.scalar(select(func.count()).select_from(Story))
    assert assign_stories(session) == []
    assert session.scalar(select(func.count()).select_from(Story)) == before


def test_new_article_joins_existing_story(session, factory, storylines):
    assign_stories(session)
    story = storylines["atlas_launch"].story
    follow_up = factory.article(
        "Nimbus Labs releases Atlas-3.1 update after independent algebra evaluation",
        "Nimbus Labs said the Atlas-3.1 update improves the Atlas-3 math model on unseen high school algebra "
        "problems, responding to the Open Evaluation Institute evaluation.",
        vertical="technology",
        published_at=utcnow() - timedelta(days=1),
    )
    changed = assign_stories(session, [follow_up])
    assert follow_up.story_id == story.id
    assert changed == [story]
    assert story.last_updated_at == follow_up.published_at


def _pair(factory, vertical_b: str):
    a = factory.article(
        "Nimbus Labs launches Atlas-3 math tutoring model for high schools",
        "Nimbus Labs released the Atlas-3 math model for high school algebra classes.",
        vertical="technology",
        published_at=utcnow() - timedelta(days=3),
    )
    b = factory.article(
        "High schools weigh the Atlas-3 math tutoring model from Nimbus Labs",
        "District leaders are reviewing the Atlas-3 math model from Nimbus Labs for algebra classes.",
        vertical=vertical_b,
        published_at=utcnow() - timedelta(days=1),
    )
    return a, b


@pytest.mark.parametrize("vertical_b", ["technology", "education"])
def test_cross_vertical_match_needs_a_higher_score(session, factory, vertical_b):
    a, b = _pair(factory, vertical_b)
    assign_stories(session, [a])
    score = match_score(session, b, a.story)
    assert score > 0.2  # genuinely related text

    # Threshold just below the raw score: same vertical joins, cross-vertical must clear the margin.
    assign_stories(session, [b], threshold=score - 0.02)
    if vertical_b == "technology":
        assert b.story_id == a.story_id
    else:
        assert score - CROSS_VERTICAL_MARGIN < score - 0.02
        assert b.story_id != a.story_id


def test_strong_cross_vertical_match_still_joins(session, factory):
    a, b = _pair(factory, "education")
    assign_stories(session, [a])
    score = match_score(session, b, a.story)
    assign_stories(session, [b], threshold=score - CROSS_VERTICAL_MARGIN - 0.01)
    assert b.story_id == a.story_id


def test_recency_window_limits_candidates(session, factory):
    old = factory.article(
        "Kestrel Semiconductor announces Arizona fab capacity",
        "Kestrel Semiconductor plans 20,000 wafers per month in Arizona.",
        vertical="industry",
        published_at=utcnow() - timedelta(days=90),
    )
    assign_stories(session, [old])
    new = factory.article(
        "Kestrel Semiconductor announces more Arizona fab capacity",
        "Kestrel Semiconductor plans 30,000 wafers per month in Arizona.",
        vertical="industry",
        published_at=utcnow(),
    )
    assign_stories(session, [new], window_days=21)
    assert new.story_id != old.story_id
    assert old.story.status == "dormant"


def test_related_stories_connect_across_verticals(session, factory):
    tech = factory.article(
        "Nimbus Labs launches Atlas-3 math model",
        "Nimbus Labs released Atlas-3, a math tutoring model, with claims of strong algebra scores.",
        vertical="technology",
        published_at=utcnow() - timedelta(days=5),
    )
    edu = factory.article(
        "Riverside Unified School District pilots Atlas-3 in algebra classes",
        "Riverside Unified School District will pilot Atlas-3 from Nimbus Labs with ninth-grade students.",
        vertical="education",
        published_at=utcnow() - timedelta(days=2),
    )
    unrelated = factory.article(
        "Fed holds interest rates steady",
        "The Federal Reserve kept rates unchanged and pointed to inflation data.",
        vertical="finance",
        published_at=utcnow() - timedelta(days=1),
    )
    # A high threshold keeps every article in its own story so we can test the links.
    assign_stories(session, [tech, edu, unrelated], threshold=0.95)
    assert len({tech.story_id, edu.story_id, unrelated.story_id}) == 3

    related = related_stories(session, tech.story)
    assert related[0] == edu.story
    assert tech.story not in related
    assert unrelated.story not in related
    assert related_stories(session, edu.story)[0] == tech.story
