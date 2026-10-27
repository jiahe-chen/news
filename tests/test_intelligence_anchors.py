"""Headline anchors: a distinctive subject named in both headlines lowers the match bar."""

from __future__ import annotations

from datetime import timedelta

from newsren.intelligence.clustering import assign_stories, title_anchors
from newsren.models import TrustTier, Vertical, utcnow


def test_title_anchors_come_from_headline_entities(factory, session):
    art = factory.article(
        "Quillfeather Labs ships Quill-7 tutor for algebra classes",
        "Quillfeather Labs said its Quill-7 model now powers a tutoring product for algebra classes.",
    )
    anchors = title_anchors(session, art)
    assert "quillfeather" in anchors or "quillfeather labs" in anchors


def test_product_launch_joins_its_model_story_through_a_headline_anchor(factory, session):
    now = utcnow()
    vendor = factory.source(vertical=Vertical.TECHNOLOGY, trust_tier=TrustTier.VENDOR, name="Quillfeather Blog")
    press = factory.source(vertical=Vertical.TECHNOLOGY, name="Tech Desk")
    launch = factory.article(
        "Quillfeather Labs unveils Quill-7 model with record algebra benchmark scores",
        "Quillfeather Labs released Quill-7, a language model it says sets records on algebra benchmarks. "
        "The company published its own benchmark results; no independent evaluation is available yet.",
        source=vendor,
        published_at=now - timedelta(days=6),
    )
    evaluation = factory.article(
        "Independent evaluation finds Quill-7 strong on algebra drills, weaker on unfamiliar problems",
        "Researchers evaluated Quill-7 on held-out problem sets. The model matched vendor claims on drills "
        "but scored lower on unfamiliar multi-step problems.",
        source=press,
        published_at=now - timedelta(days=4),
    )
    product = factory.article(
        "Quillfeather Labs launches Quill Tutor for high school classrooms",
        "The tutoring product, built on Quill-7, offers step-by-step hints for high school mathematics.",
        source=vendor,
        published_at=now - timedelta(days=2),
    )
    unrelated = factory.article(
        "City council approves new bus routes downtown",
        "The council voted to add three bus routes and extend evening service.",
        source=press,
        published_at=now - timedelta(days=2),
    )
    assign_stories(session)
    assert launch.story_id is not None
    assert evaluation.story_id == launch.story_id
    assert product.story_id == launch.story_id
    assert unrelated.story_id != launch.story_id
