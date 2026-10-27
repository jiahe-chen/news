"""Story clustering on the fictional demo corpus (``newsren demo``): every storyline stays pure.

Guards the per-source boilerplate, publisher-name and same-publisher release
rules against regressions on the storylines the product demo relies on.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest
from sqlalchemy import select

from newsren.ingest.fixtures import load_demo_data
from newsren.intelligence.clustering import assign_stories
from newsren.models import Article

DEMO = Path(__file__).parents[1] / "src" / "newsren" / "ingest" / "data" / "demo_articles.json"
STORYLINE = {row["key"]: row["storyline"] for row in json.loads(DEMO.read_text())["articles"]}


@pytest.fixture()
def demo(session) -> dict[str, Article]:
    load_demo_data(session)
    assign_stories(session)
    articles = {(a.extra or {}).get("demo_key"): a for a in session.scalars(select(Article))}
    assert set(STORYLINE) <= set(articles)
    return articles


def test_every_demo_story_holds_a_single_storyline(demo):
    lines: dict[int, set[str]] = defaultdict(set)
    for key, article in demo.items():
        assert article.story_id is not None
        lines[article.story_id].add(STORYLINE[key])
    assert all(len(found) == 1 for found in lines.values()), dict(lines)


def test_model_launch_and_product_launch_share_the_story(demo):
    story = demo["lumen4-launch"].story_id
    for key in (
        "lumen4-tech-report",
        "lumen4-coverage",
        "lumen4-independent-eval",
        "lumen4-eval-coverage",
        "lumen4-correction",
        "lumen-tutor-launch",  # the vendor's own product launch, anchored by "Lumen-4" (not by "Lumen Labs")
    ):
        assert demo[key].story_id == story, key


def test_vendor_clarification_joins_its_study_release(demo):
    """Same-publisher releases may share a story when they name a shared subject (the StepWise study)."""
    story = demo["halbrook-press-release"].story_id
    for key in ("halbrook-preprint", "halbrook-dispatch-analysis", "halbrook-clarification"):
        assert demo[key].story_id == story, key


def test_independent_coverage_anchors_to_the_companys_own_announcement(demo):
    story = demo["corvane-capacity"].story_id
    for key in ("corvane-output-q3", "corvane-capacity-analysis", "pinecrest-supply-deal"):
        assert demo[key].story_id == story, key
    # The same company's earnings release (another vertical, another storyline) stays apart.
    assert demo["corvane-q3-results"].story_id != story
    assert demo["corvane-q3-coverage"].story_id == demo["corvane-q3-results"].story_id


def test_advisory_and_its_update_share_the_story(demo):
    story = demo["ccrc-pallas-advisory"].story_id
    assert demo["ccrc-pallas-update"].story_id == story
    assert demo["pallas-exploited-coverage"].story_id == story


def test_funding_rounds_of_different_companies_are_separate_stories(demo):
    """Shared "raises $X million in a Series Y led by ..." wording is the funding desk's template, not a story."""
    assert demo["quillfield-series-b"].story_id != demo["tarnwell-series-d"].story_id
