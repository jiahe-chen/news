"""Fictional demo corpus: idempotent loading, coverage of the plan's use cases, dedup."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import timedelta

import pytest
from sqlalchemy import select

from newsren import db
from newsren.ingest.fixtures import DEMO_ARTICLES_PATH, load_demo_data, load_demo_dataset
from newsren.models import Article, DocType, Source, SourceKind, TrustTier, Vertical, utcnow


@pytest.fixture()
def demo(session):
    now = utcnow()
    loaded = load_demo_data(session, now=now)
    articles = session.scalars(select(Article)).all()
    return loaded, articles, now


def _by_key(articles):
    return {a.extra["demo_key"]: a for a in articles}


def test_load_is_idempotent(session, demo):
    loaded, articles, _ = demo
    assert loaded == len(articles) >= 40
    assert load_demo_data(session) == 0
    assert len(session.scalars(select(Article)).all()) == loaded
    assert all(s.kind == SourceKind.DEMO for s in session.scalars(select(Source)))


def test_all_demo_content_is_marked_fictional(demo):
    _, articles, _ = demo
    assert all(a.is_demo and a.extra["is_demo"] for a in articles)
    assert all(re.match(r"https://[a-z0-9.-]+\.example/", a.url) for a in articles)
    for src in {a.source for a in articles}:
        assert src.homepage.endswith(".example") and "Fictional demo content" in src.license_note


def test_demo_text_does_not_name_real_organizations():
    raw = DEMO_ARTICLES_PATH.read_text(encoding="utf-8")
    for real in [
        "JPMorgan", "FINRA", "Securities and Exchange Commission", "Federal Reserve", "OpenAI", "Anthropic",
        "Google", "Nvidia", "TSMC", "Intel", "Samsung", "Apple", "Microsoft", "Goldman", "Deutsche Bank",
    ]:  # fmt: skip
        assert real not in raw, real
    assert not re.search(r"\bSEC\b|\bCVE-\d", raw)


def test_covers_four_verticals_and_spans_three_weeks(demo):
    _, articles, now = demo
    per_vertical = defaultdict(int)
    for a in articles:
        per_vertical[a.vertical] += 1
    assert set(per_vertical) == {v.value for v in Vertical}
    assert min(per_vertical.values()) >= 8
    dates = [a.published_at for a in articles]
    assert all(d <= now for d in dates)
    assert max(dates) - min(dates) >= timedelta(days=18)
    assert now - max(dates) < timedelta(days=3)


def test_every_storyline_has_multiple_articles_and_an_opinion_piece(demo):
    _, articles, _ = demo
    stories = defaultdict(list)
    for a in articles:
        stories[a.extra["storyline"]].append(a)
    assert len(stories) >= 10
    for name, members in stories.items():
        assert len(members) >= 3, name
        opinions = [a for a in members if a.doc_type == DocType.OPINION]
        assert opinions, name
        assert all(a.trust_tier == TrustTier.COMMENTARY for a in opinions)
        assert len({a.published_at.date() for a in members}) >= 2, name


def test_syndicated_copies_detected_as_duplicates(demo):
    _, articles, _ = demo
    by_key = _by_key(articles)
    copies = [a for a in articles if a.extra.get("republished_from_demo_key")]
    assert len(copies) >= 3
    for copy in copies:
        original = by_key[copy.extra["republished_from_demo_key"]]
        assert copy.duplicate_of_id == original.id, copy.extra["demo_key"]
        assert copy.source_id != original.source_id
    # Duplicates are the only articles marked as such.
    assert sum(1 for a in articles if a.is_duplicate) == len(copies)


def test_plan_use_cases_are_present(demo):
    _, articles, _ = demo
    by_key = _by_key(articles)
    text = {k: f"{a.title} {a.summary} {a.content}" for k, a in by_key.items()}

    # (a) model launch: vendor claim vs independent evaluation, correction, adoption, regulation
    assert by_key["lumen4-launch"].trust_tier == TrustTier.VENDOR
    assert by_key["lumen4-independent-eval"].trust_tier == TrustTier.INDEPENDENT
    assert "independent evaluation" in text["lumen4-independent-eval"].lower()
    assert by_key["lumen4-correction"].extra["is_correction"]
    assert by_key["calder-ai-tutor-rules"].doc_type == DocType.POLICY
    assert "Lumen Labs" in text["westhollow-pilot"]
    # (b) press release vs study: participants, outcome measures, comparison group, preprint status
    study = text["halbrook-preprint"]
    assert by_key["halbrook-preprint"].doc_type == DocType.RESEARCH_PREPRINT
    for needle in ["212", "transfer", "comparison", "Limitations", "not been peer reviewed"]:
        assert needle in study
    assert by_key["halbrook-press-release"].trust_tier == TrustTier.VENDOR
    # (c) capacity projection vs reported output
    assert "projects" in text["corvane-capacity"] and "11,500" in text["corvane-output-q3"]
    # (d) backtest vs live, with authors' caveat
    note = text["northbridge-backtest"]
    assert "backtest" in note and "60%" in note and "not proof of future" in note
    assert "paper-trading" in text["northbridge-paper-trading"]
    # (e) funding rounds across stages
    rounds = [t for t in text.values() if re.search(r"raise[sd]? \$[\d.]+ million", t)]
    assert len(rounds) >= 5
    for stage in ["Series B", "seed", "Series C", "Series D"]:
        assert any(stage in t for t in rounds), stage
    # (f) earnings with guidance, (g) enforcement + guidance notice, (h) policy, (i) advisory
    assert "guidance" in text["corvane-q3-results"].lower()
    assert by_key["corvane-q3-results"].trust_tier == TrustTier.PRIMARY
    assert "AI washing" in text["mic-ai-washing"] and by_key["mic-ai-washing"].doc_type == DocType.REGULATORY
    assert by_key["bdsa-agent-guidance"].doc_type == DocType.REGULATORY
    assert "fall 2028" in text["calder-ai-literacy-adopted"]
    assert by_key["ccrc-pallas-advisory"].doc_type == DocType.ADVISORY
    # (j) supply chain and skills
    assert "certificate" in text["cleanroom-technicians"]


def test_demo_articles_are_searchable(session, demo):
    hits = db.search_article_ids(session, "Lumen Tutor")
    keys = {session.get(Article, i).extra["demo_key"] for i, _ in hits}
    assert "lumen-tutor-launch" in keys


def test_dataset_validation_rejects_unknown_source(tmp_path):
    bad = DEMO_ARTICLES_PATH.read_text(encoding="utf-8").replace(
        '"source_slug": "demo-lumen-labs"', '"source_slug": "nope"', 1
    )
    path = tmp_path / "demo.json"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown source"):
        load_demo_dataset(path)
