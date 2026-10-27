"""AI Story Brief generation (offline): evidence pack, claims, sections, guardrails, and regeneration."""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsren import db
from newsren.agents.base import build_evidence_pack
from newsren.agents.brief import (
    AGENT_NAME,
    MAX_EVIDENCE_DOCS,
    VERTICAL_SECTIONS,
    brief_for_story,
    generate_brief,
    sanitize_brief,
    select_pack_articles,
    source_docs,
    tier_rank,
)
from newsren.config import Settings, set_settings
from newsren.editorial.guardrails import is_hedged, lint_text
from newsren.ingest.fixtures import load_demo_data
from newsren.intelligence.clustering import assign_stories
from newsren.intelligence.entities import tag_article_entities
from newsren.intelligence.timeline import refresh_story
from newsren.llm import OfflineProvider, set_provider
from newsren.models import (
    AgentRun,
    Article,
    Brief,
    BriefStatus,
    ClaimType,
    DocType,
    EditorialAction,
    Story,
    SupportStatus,
    TrustTier,
    Vertical,
    utcnow,
)
from newsren.schemas import BriefOut, ClaimOut, DisagreementOut, DisagreementPosition, SectionOut
from newsren.textutil import canonical_url, content_hash, simhash_hex

CITE = re.compile(r"\[(c\d+)\]")

LUMEN = "Lumen Labs launches Lumen-4"
HALBROOK = "Halbrook study: AI practice assistant"
CORVANE = "Corvane Semiconductor to add 40,000 wafers"
NORTHBRIDGE = "Northbridge researchers: AI portfolio agents"


# --------------------------------------------------------------------------
# Fixtures: the fictional demo corpus clustered into stories, built once per module
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def story_db(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("brief-stories")
    settings = Settings(data_dir=root, database_url=f"sqlite:///{root / 'stories.db'}", llm_mode="offline")
    set_settings(settings)
    set_provider(OfflineProvider())
    db.reset_engine()
    try:
        db.init_db()
        with db.session_scope() as session:
            load_demo_data(session)
            for article in session.scalars(select(Article).where(Article.duplicate_of_id.is_(None))):
                tag_article_entities(session, article)
            for story in assign_stories(session):
                refresh_story(session, story)
    finally:
        db.reset_engine()
        set_provider(None)
        set_settings(None)
    return root / "stories.db"


@pytest.fixture()
def stories(tmp_path, story_db) -> Iterator[Session]:
    target = tmp_path / "stories.db"
    shutil.copyfile(story_db, target)
    settings = Settings(
        data_dir=tmp_path, database_url=f"sqlite:///{target}", llm_mode="offline", require_human_approval=True
    )
    set_settings(settings)
    set_provider(OfflineProvider())
    db.reset_engine()
    db.init_db()
    session = db.new_session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        db.reset_engine()
        set_provider(None)
        set_settings(None)


def story_of(session: Session, title_start: str) -> Story:
    article = session.scalars(select(Article).where(Article.title.startswith(title_start))).one()
    assert article.story is not None
    return article.story


def brief_texts(brief: Brief) -> list[str]:
    return [
        brief.headline,
        brief.summary,
        brief.why_it_matters or "",
        brief.what_changed or "",
        *brief.key_points,
        *brief.uncertainty,
        *brief.watch_next,
        *(s["body"] for s in brief.sections),
        *(c.text for c in brief.claims),
        *(p["position"] for d in brief.disagreements for p in d["positions"]),
    ]


def assert_well_cited(brief: Brief) -> set[str]:
    valid = {c["id"] for c in brief.citations}
    assert valid, "a brief must record its sources"
    for claim in brief.claims:
        assert set(claim.citation_ids) <= valid
        if claim.claim_type in (ClaimType.FACT, ClaimType.PERFORMANCE):
            assert claim.citation_ids, f"uncited {claim.claim_type}: {claim.text}"
    for section in brief.sections:
        assert set(section["citation_ids"]) <= valid
    for text in brief_texts(brief):
        assert set(CITE.findall(text)) <= valid, text
    return valid


# --------------------------------------------------------------------------
# One brief per vertical
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title_start", "vertical"),
    [
        (LUMEN, Vertical.TECHNOLOGY),
        (HALBROOK, Vertical.EDUCATION),
        (CORVANE, Vertical.INDUSTRY),
        (NORTHBRIDGE, Vertical.FINANCE),
    ],
)
def test_brief_for_each_vertical_is_cited_sectioned_and_reviewed(stories, title_start, vertical):
    story = story_of(stories, title_start)
    brief = generate_brief(stories, story)

    assert brief.id is not None and brief.story_id == story.id and brief.kind == "story"
    assert brief.vertical == vertical == story.vertical
    assert brief.headline and brief.summary and brief.why_it_matters and brief.what_changed
    assert [s["title"] for s in brief.sections] == list(VERTICAL_SECTIONS[vertical])
    assert all(s["body"] for s in brief.sections)
    assert 2 <= len(brief.claims) <= 8
    assert [c.position for c in brief.claims] == list(range(len(brief.claims)))
    assert brief.key_points and brief.watch_next
    assert_well_cited(brief)

    # Most authoritative member, provenance, and the run that produced it.
    members = {a.id for a in story.articles}
    assert brief.article_id in members
    assert brief.is_demo is True
    assert brief.generated_by == "offline"
    run = stories.get(AgentRun, brief.agent_run_id)
    assert run.agent == AGENT_NAME and run.subject_type == "story" and run.subject_id == story.id
    for cit in brief.citations:
        assert cit["url"].startswith("https://") and cit["retrieved_at"] and cit["article_id"] in members

    # Editorial quality gate ran: human review queue, no blocking checks, no guardrail language.
    assert brief.status == BriefStatus.IN_REVIEW and brief.requires_human_review
    assert brief.check_results
    blockers = [c for c in brief.check_results if not c["passed"] and c["severity"] == "blocker"]
    assert blockers == []
    assert all(not lint_text(text, vertical) for text in brief_texts(brief))


def test_evidence_pack_puts_primary_and_independent_sources_first(stories):
    story = story_of(stories, LUMEN)
    members = [a for a in story.articles if a.duplicate_of_id is None]
    selected = select_pack_articles(members, {})
    ranks = [tier_rank(a) for a in selected]
    assert ranks == sorted(ranks)
    assert len(selected) <= MAX_EVIDENCE_DOCS

    brief = generate_brief(stories, story)
    tiers = [c["trust_tier"] for c in brief.citations]
    assert tiers[0] in (TrustTier.PRIMARY, TrustTier.INDEPENDENT)
    assert tiers.index(TrustTier.VENDOR) < tiers.index(TrustTier.COMMENTARY)


def test_large_story_pack_is_capped_but_keeps_vendor_commentary_and_corrections(session, factory):
    vendor = factory.source(trust_tier=TrustTier.VENDOR, doc_type=DocType.PRESS_RELEASE, name="Acme Newsroom")
    news = factory.source(name="Daily Wire")
    opinion = factory.source(trust_tier=TrustTier.COMMENTARY, doc_type=DocType.OPINION, name="Op-Ed Desk")
    now = utcnow()
    story = Story(title="Acme model", vertical=Vertical.TECHNOLOGY)
    session.add(story)
    claim = factory.article("Acme launches Model Z", "Acme says Model Z is fast.", source=vendor,
                            published_at=now - timedelta(days=20))  # fmt: skip
    reports = [
        factory.article(
            f"Report {i} on Model Z",
            f"Model Z coverage number {i}.",
            source=news,
            published_at=now - timedelta(days=19 - i),
        )  # fmt: skip
        for i in range(10)
    ]
    fix = factory.article("Correction: Model Z figure", "An earlier report misstated it.", source=news,
                          published_at=now - timedelta(days=15))  # fmt: skip
    column = factory.article("Opinion: Model Z hype", "A columnist argues it is hype.", source=opinion,
                             published_at=now - timedelta(days=1))  # fmt: skip
    for art in [claim, *reports, fix, column]:
        art.story = story
    session.flush()

    selected = select_pack_articles(story.articles, {fix.id: "correction"})
    assert len(selected) == MAX_EVIDENCE_DOCS
    assert claim in selected and column in selected and fix in selected
    assert [tier_rank(a) for a in selected] == sorted(tier_rank(a) for a in selected)


# --------------------------------------------------------------------------
# Guardrails by vertical
# --------------------------------------------------------------------------


def test_finance_brief_always_needs_review_and_labels_performance(stories, settings_override):
    settings_override(require_human_approval=False)
    brief = generate_brief(stories, story_of(stories, NORTHBRIDGE))

    assert brief.status == BriefStatus.IN_REVIEW
    assert brief.requires_human_review is True
    assert brief.risk_level == "high"
    performance = [c for c in brief.claims if c.claim_type == ClaimType.PERFORMANCE]
    assert performance
    for claim in performance:
        assert claim.performance_label in ("backtest", "simulated", "live")
        assert claim.performance_period and claim.benchmark and claim.citation_ids
        assert claim.high_impact is True
    assert any(c.performance_label == "backtest" for c in performance)
    assert any("60/40" in (c.benchmark or "") for c in performance)
    checks = {c["name"]: c for c in brief.check_results}
    assert checks["performance_labels"]["passed"] and checks["guardrail_lint"]["passed"]
    sections = {s["title"]: s["body"] for s in brief.sections}
    assert "not investment advice" in sections["What it means"]
    assert "Backtesting" in sections["Concepts to know"]
    assert "not investment advice" in brief.why_it_matters


def test_education_brief_names_the_preprint_population_and_outcomes(stories):
    brief = generate_brief(stories, story_of(stories, HALBROOK))
    text = " ".join(brief_texts(brief))
    assert "preprint" in brief.headline.lower()
    assert re.search(r"not (yet )?(been )?peer reviewed", text)
    assert re.search(r"\b212\b|ninth-grade", text)
    assert "transfer" in text
    checks = {c["name"]: c for c in brief.check_results}
    assert checks["preprint_labelling"]["passed"]
    assert checks["study_population"]["passed"]
    who = next(s for s in brief.sections if s["title"] == "Who is affected?")
    assert re.search(r"students|classrooms", who["body"])


def test_industry_brief_labels_projections_and_scenarios(stories):
    brief = generate_brief(stories, story_of(stories, CORVANE))
    forecasts = [c for c in brief.claims if c.claim_type == ClaimType.FORECAST]
    assert forecasts and all(is_hedged(c.text) for c in forecasts)
    scenarios = next(s for s in brief.sections if s["title"].startswith("Scenarios"))
    assert "not a confirmed outcome" in scenarios["body"]
    assert "11,500" in scenarios["body"]  # projection compared with the reported result
    assert "scenarios, not confirmed outcomes" in brief.why_it_matters


def test_technology_brief_contrasts_vendor_claims_with_independent_evaluation(stories):
    brief = generate_brief(stories, story_of(stories, LUMEN))
    sections = {s["title"]: s["body"] for s in brief.sections}
    assert "Independent evaluation:" in sections["What evidence exists?"]
    assert "own testing or account" in sections["What evidence exists?"]
    assert "Independent evidence is now available" in brief.why_it_matters
    # The correction withdrew 81.3%: the figure is never repeated except when the correction states it.
    for text in brief_texts(brief)[:-1]:
        if "81.3" in text:
            assert "not 81.3" in text or "84.3" in text, text


def test_contested_claims_come_with_disagreements_citing_both_sides(stories):
    for title in (LUMEN, HALBROOK):
        brief = generate_brief(stories, story_of(stories, title))
        by_id = {c["id"]: c for c in brief.citations}
        contested = [c for c in brief.claims if c.support_status == SupportStatus.CONTESTED]
        assert contested, title
        assert brief.disagreements
        for claim in contested:
            assert any(by_id[cid]["trust_tier"] == TrustTier.VENDOR for cid in claim.citation_ids)
        for item in brief.disagreements:
            assert len(item["positions"]) == 2
            sides = [set(p["citation_ids"]) for p in item["positions"]]
            assert all(sides) and not sides[0] & sides[1]
            assert any(by_id[c]["trust_tier"] == TrustTier.VENDOR for c in sides[0])
            assert all(by_id[c]["trust_tier"] != TrustTier.VENDOR for c in sides[1])
        checks = {c["name"]: c for c in brief.check_results}
        assert checks["disagreements_described"]["passed"]


def test_corroborated_requires_two_independent_publishers(stories):
    brief = generate_brief(stories, story_of(stories, HALBROOK))
    by_id = {c["id"]: c for c in brief.citations}
    corroborated = [c for c in brief.claims if c.support_status == SupportStatus.CORROBORATED]
    assert corroborated
    for claim in corroborated:
        publishers = {
            by_id[c]["publisher"]
            for c in claim.citation_ids
            if by_id[c]["trust_tier"] not in (TrustTier.VENDOR, TrustTier.COMMENTARY)
        }
        assert len(publishers) >= 2


def test_opinion_only_story_is_labelled_as_commentary(stories):
    story = story_of(stories, "Opinion: Faster isn't smarter")
    brief = generate_brief(stories, story)
    assert brief.headline.endswith("(opinion)")
    assert [c.claim_type for c in brief.claims] == [ClaimType.OPINION]
    assert any("only on commentary" in u for u in brief.uncertainty)
    assert_well_cited(brief)


# --------------------------------------------------------------------------
# Regeneration and lookup
# --------------------------------------------------------------------------


def test_regenerate_semantics(stories):
    story = story_of(stories, LUMEN)
    first = generate_brief(stories, story)
    runs = stories.scalar(select(func.count()).select_from(AgentRun).where(AgentRun.agent == AGENT_NAME))

    again = generate_brief(stories, story)
    assert again.id == first.id
    assert stories.scalar(select(func.count()).select_from(AgentRun).where(AgentRun.agent == AGENT_NAME)) == runs

    forced = generate_brief(stories, story, regenerate=True)
    assert forced.id != first.id
    assert brief_for_story(stories, story).id == forced.id
    assert first.status == BriefStatus.REJECTED and "Superseded" in (first.review_notes or "")
    actions = stories.scalars(select(EditorialAction).where(EditorialAction.action == "regenerate")).all()
    assert {a.brief_id for a in actions} == {first.id, forced.id}
    assert forced.status == BriefStatus.IN_REVIEW

    # A new development joins the story: the brief is out of date and is rebuilt without forcing.
    source = story.articles[0].source
    url = "https://circuit-ledger.example/stories/lumen-4-second-evaluation"
    article = Article(
        source=source,
        url=url,
        canonical_url=canonical_url(url),
        title="Second independent group replicates Lumen-4 held-out results",
        summary="A second independent group reported Lumen-4 scored 83.9% on its own held-out math set.",
        content="A second independent group reported Lumen-4 scored 83.9% on its own held-out math set.",
        published_at=utcnow() + timedelta(minutes=1),
        vertical=story.vertical,
        doc_type=DocType.NEWS,
        trust_tier=TrustTier.INDEPENDENT,
        content_hash=content_hash("second evaluation"),
        simhash=simhash_hex("second evaluation"),
        is_demo=True,
        story=story,
    )
    stories.add(article)
    stories.flush()
    assert refresh_story(stories, story)
    updated = generate_brief(stories, story)
    assert updated.id != forced.id
    assert article.id in {c["article_id"] for c in updated.citations}
    assert forced.status == BriefStatus.REJECTED


def test_brief_for_story_and_empty_story(session):
    story = Story(title="Empty", vertical=Vertical.TECHNOLOGY)
    session.add(story)
    session.flush()
    assert brief_for_story(session, story) is None
    with pytest.raises(ValueError):
        generate_brief(session, story)


def test_story_never_refreshed_still_gets_a_timeline_based_brief(session, factory):
    vendor = factory.source(trust_tier=TrustTier.VENDOR, doc_type=DocType.PRESS_RELEASE, name="Quill Labs")
    news = factory.source(name="Tech Daily")
    now = utcnow()
    launch = factory.article(
        "Quill Labs launches Quill-2 coding model",
        "Quill Labs says Quill-2 solves 71% of its internal coding benchmark, up from 58% for Quill-1.",
        source=vendor,
        published_at=now - timedelta(days=3),
    )
    coverage = factory.article(
        "Quill-2 arrives; coding benchmark claims not yet independently verified",
        "Quill Labs says Quill-2 solves 71% of its internal coding benchmark. No independent group has tested "
        "the model yet, and the benchmark has not been published.",
        source=news,
        published_at=now - timedelta(days=2),
    )
    story = Story(title="Quill-2", vertical=Vertical.TECHNOLOGY)
    session.add(story)
    launch.story = coverage.story = story
    session.flush()

    brief = generate_brief(session, story)
    assert brief.what_changed and "First reported" in brief.what_changed
    assert any("independent evaluation" in w.lower() for w in brief.watch_next)
    assert "No independent evaluation is in the evidence yet." in brief.why_it_matters
    assert_well_cited(brief)


# --------------------------------------------------------------------------
# Output hygiene
# --------------------------------------------------------------------------


def test_sanitize_drops_unknown_ids_and_enforces_labels(session, factory):
    vendor = factory.source(trust_tier=TrustTier.VENDOR, doc_type=DocType.PRESS_RELEASE, name="Bank Newsroom")
    wire = factory.source(trust_tier=TrustTier.INDEPENDENT, vertical=Vertical.FINANCE, name="Market Wire")
    note = factory.article(
        "Bank research note on AI agents",
        "In a backtest from 2005 to 2024, the agents returned 7.9% a year versus 6.6% for a 60/40 portfolio.",
        source=vendor,
        doc_type=DocType.RESEARCH_PREPRINT,
    )
    report = factory.article("Wire: bank agents in a backtest", "The agents returned 7.9% a year.", source=wire)
    docs = source_docs(build_evidence_pack([note, report]), {note.id: note, report.id: report})
    out = BriefOut(
        headline="Bank AI agents",
        summary="A bank reported a backtest [c1][c9].",
        why_it_matters="Readers should learn what a backtest is [c2].",
        key_points=["Only an unknown source says so [c7]."],
        sections=[
            SectionOut(title="What was reported?", body="Backtest figures [c1].", citation_ids=["c1", "c5"])
        ],
        claims=[
            ClaimOut(
                text="The agents returned 7.9% a year in a backtest.",
                claim_type="performance",
                support_status="corroborated",
                citation_ids=["c1", "c2"],
            ),  # fmt: skip
            ClaimOut(text="An invented fact.", support_status="supported", citation_ids=["c8"]),
        ],
        disagreements=[
            DisagreementOut(
                topic="Returns",
                positions=[DisagreementPosition(position="Bank view [c1]", citation_ids=["c1", "c4"])],
            )
        ],
    )
    clean = sanitize_brief(out, docs)
    assert clean.summary == "A bank reported a backtest [c1]."
    assert clean.key_points == ["Only an unknown source says so."]
    assert clean.sections[0].citation_ids == ["c1"]
    perf, invented = clean.claims
    # The vendor's note and one wire report are not two independent publishers.
    assert perf.support_status == "supported"
    assert perf.performance_label == "backtest"
    assert perf.performance_period and "2005" in perf.performance_period
    assert perf.benchmark and "60/40" in perf.benchmark
    assert invented.citation_ids == [] and invented.support_status == "unverified"
    assert clean.disagreements[0].positions[0].citation_ids == ["c1"]
    # c1 is a preprint the text never names as one: a notice is added.
    assert any("preprint" in u for u in clean.uncertainty)


@pytest.fixture()
def settings_override(stories):
    """Swap the active settings for this test (the stories fixture restores them)."""
    from newsren.config import get_settings

    def apply(**changes):
        set_settings(get_settings().with_overrides(**changes))

    return apply
