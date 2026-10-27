"""Story timelines, change detection, and the story-intelligence agent."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from newsren.agents.base import Orchestrator
from newsren.intelligence.clustering import assign_stories
from newsren.intelligence.timeline import (
    AGENT_NAME,
    member_articles,
    member_fingerprint,
    refresh_story,
    story_changes_since,
    story_citations,
    update_stories,
)
from newsren.llm import LLMResult
from newsren.models import AgentRun, Article, EventType, Story, utcnow
from newsren.schemas import StoryUpdateOut, TimelineEventOut

DATA = Path(__file__).parent / "data" / "intelligence" / "storylines.json"
CITE = re.compile(r"\[(c\d+)\]")


def load_storylines(factory) -> dict[str, Article]:
    now = utcnow()
    sources = {}
    articles: dict[str, Article] = {}
    rows = json.loads(DATA.read_text())["articles"]
    for row in rows:
        key = (row["publisher"], row["vertical"])
        if key not in sources:
            sources[key] = factory.source(
                vertical=row["vertical"],
                trust_tier=row["trust_tier"],
                doc_type=row["doc_type"],
                name=row["publisher"],
            )
        articles[row["key"]] = factory.article(
            row["title"],
            row["summary"],
            source=sources[key],
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


@pytest.fixture()
def atlas(session, factory) -> Story:
    articles = load_storylines(factory)
    assign_stories(session)
    return articles["atlas_launch"].story


@pytest.fixture()
def kestrel(session, factory) -> Story:
    articles = load_storylines(factory)
    assign_stories(session)
    return articles["kestrel_announce"].story


def cited_ids(story: Story) -> set[str]:
    texts = [story.what_happened_first or "", story.whats_new or "", *story.disputed, *story.evidence_changes]
    return {cid for text in texts for cid in CITE.findall(text)}


def agent_runs(session) -> int:
    return session.scalar(select(func.count()).select_from(AgentRun).where(AgentRun.agent == AGENT_NAME))


def test_refresh_builds_timeline_and_offline_intelligence(session, atlas):
    assert refresh_story(session, atlas) is True

    members = member_articles(atlas)
    assert len(members) == 7  # the syndicated copy is not a member event
    assert [ev.article_id for ev in atlas.events] == [a.id for a in members]
    assert (
        atlas.member_fingerprint
        == hashlib.sha256(",".join(str(i) for i in sorted(a.id for a in members)).encode()).hexdigest()
    )
    assert atlas.version == 1

    by_type = {ev.event_type: ev for ev in atlas.events}
    assert by_type[EventType.CORRECTION].is_correction is True
    assert sum(ev.is_correction for ev in atlas.events) == 1
    assert {
        EventType.PRODUCT_LAUNCH,
        EventType.INDEPENDENT_EVALUATION,
        EventType.ADOPTION,
        EventType.OPINION,
    } <= set(by_type)
    assert all(a.event_type for a in members)  # cached on the article for monitoring

    # Evidence-grounded summary: earliest member first, citations only from the pack.
    assert atlas.what_happened_first.startswith("The story began on")
    assert "[c1]" in atlas.what_happened_first
    assert "company's own claims" in atlas.what_happened_first
    assert "correction" in atlas.whats_new
    assert cited_ids(atlas) <= {f"c{i}" for i in range(1, 8)}
    assert any("disputes the methodology" in d for d in atlas.disputed)
    assert all(CITE.search(d) for d in atlas.disputed)
    changes = " ".join(atlas.evidence_changes)
    assert "independent evaluation" in changes and "preprint" in changes
    assert "correction was issued" in changes
    assert not atlas.title.lower().startswith("opinion")

    run = session.scalars(select(AgentRun).where(AgentRun.agent == AGENT_NAME)).one()
    assert run.subject_type == "story" and run.subject_id == atlas.id
    assert run.provider == "offline"


def test_story_citations_resolve_inline_markers(session, atlas):
    refresh_story(session, atlas)
    citations = story_citations(atlas)
    assert [c.id for c in citations] == [f"c{i}" for i in range(1, 8)]
    assert [c.article_id for c in citations] == [a.id for a in member_articles(atlas)]
    assert cited_ids(atlas) <= {c.id for c in citations}
    assert citations[0].trust_tier == "vendor" and citations[0].doc_type == "press_release"


def test_refresh_is_idempotent_until_membership_changes(session, atlas):
    assert refresh_story(session, atlas) is True
    fingerprint = atlas.member_fingerprint
    assert refresh_story(session, atlas) is False
    assert agent_runs(session) == 1
    assert atlas.version == 1 and atlas.member_fingerprint == fingerprint

    assert refresh_story(session, atlas, force=True) is True
    assert agent_runs(session) == 2
    assert atlas.version == 2


def test_adding_an_article_changes_fingerprint_version_and_changes_since(session, factory, atlas):
    refresh_story(session, atlas)
    fingerprint, version = atlas.member_fingerprint, atlas.version
    event_ids = {ev.id for ev in atlas.events}
    marker = utcnow()
    assert story_changes_since(atlas, marker) == []

    follow_up = factory.article(
        "Nimbus Labs releases Atlas-3.1 update after independent algebra evaluation",
        "Nimbus Labs said the Atlas-3.1 update improves the Atlas-3 math model on unseen high school algebra "
        "problems, responding to the Open Evaluation Institute evaluation.",
        vertical="technology",
        published_at=utcnow() + timedelta(seconds=1),
    )
    assign_stories(session, [follow_up])
    assert follow_up.story_id == atlas.id

    assert refresh_story(session, atlas) is True
    assert atlas.member_fingerprint != fingerprint
    assert atlas.member_fingerprint == member_fingerprint(member_articles(atlas))
    assert atlas.version == version + 1
    assert len(atlas.events) == 8
    assert event_ids < {ev.id for ev in atlas.events}  # existing rows were kept, not recreated

    changes = story_changes_since(atlas, marker)
    assert [ev.article_id for ev in changes] == [follow_up.id]
    assert story_changes_since(atlas, marker.replace(tzinfo=UTC)) == changes
    week_ago = story_changes_since(atlas, utcnow() - timedelta(days=7) - timedelta(minutes=1))
    assert [ev.occurred_at for ev in week_ago] == sorted(ev.occurred_at for ev in week_ago)
    assert follow_up.id in {ev.article_id for ev in week_ago}


def test_duplicate_joining_does_not_trigger_refresh(session, factory, atlas):
    refresh_story(session, atlas)
    original = next(a for a in member_articles(atlas) if a.event_type == EventType.INDEPENDENT_EVALUATION)
    copy = factory.article(original.title, original.summary or "", vertical="technology")
    copy.duplicate_of_id = original.id
    session.flush()
    assert assign_stories(session, [copy]) == []
    assert copy.story_id == atlas.id
    assert refresh_story(session, atlas) is False


def test_removed_member_is_dropped_from_timeline(session, atlas):
    refresh_story(session, atlas)
    opinion = next(a for a in member_articles(atlas) if a.event_type == EventType.OPINION)
    opinion.story = None
    session.flush()
    assert refresh_story(session, atlas) is True
    assert opinion.id not in {ev.article_id for ev in atlas.events}
    assert len(atlas.events) == 6


def test_projection_story_labels_forward_looking_claims(session, kestrel):
    assert refresh_story(session, kestrel) is True
    assert "forward-looking statement" in kestrel.what_happened_first
    changes = " ".join(kestrel.evidence_changes)
    assert "primary document" in changes
    assert "Projections" in changes
    assert "could" not in kestrel.title.lower()
    assert "scenario" in kestrel.whats_new  # supplier knock-on effects are labelled as scenarios


def test_single_vendor_claim_flags_missing_independent_evidence(session, factory):
    vendor = factory.source(
        vertical="finance", trust_tier="vendor", doc_type="press_release", name="Harborview Bank IR"
    )
    art = factory.article(
        "Harborview Bank says AI portfolio agents beat a 60/40 portfolio",
        "Harborview Bank said its AI portfolio agents outperformed a 60/40 portfolio by two points a year over "
        "two decades.",
        source=vendor,
        vertical="finance",
    )
    (story,) = assign_stories(session, [art])
    assert refresh_story(session, story) is True
    assert "backtest, simulated, or live" in story.what_happened_first
    assert story.whats_new.startswith("No later developments yet")
    assert any("No independent evaluation is in the evidence yet" in c for c in story.evidence_changes)
    assert story.disputed == []
    assert cited_ids(story) == {"c1"}


class FakeClaude:
    """Stands in for the Claude provider: returns a scripted StoryUpdateOut."""

    name = "claude"
    model = "claude-test"

    def __init__(self, build):
        self.build = build
        self.prompts: list[str] = []

    def generate(self, task, budget):
        self.prompts.append(task.prompt)
        assert task.output_model is StoryUpdateOut
        assert "[c1]" in task.prompt and "<evidence>" in task.prompt
        return LLMResult(output=self.build(task), provider="claude", model=self.model, llm_calls=1)

    def run_tools(self, task, budget):  # pragma: no cover - not used by this agent
        raise NotImplementedError


def test_claude_output_is_sanitized_and_event_descriptions_mapped(session, atlas):
    members = member_articles(atlas)
    first, last = members[0], members[-1]

    def build(task):
        return StoryUpdateOut(
            title="Atlas-3 math model: vendor scores, an independent evaluation, and a correction",
            what_happened_first="Nimbus Labs launched Atlas-3 and reported a 92% score [c1][c42].",
            whats_new="An independent evaluation found lower scores [c3]; a correction followed [c7].",
            disputed=["Nimbus Labs disputes the evaluation's methodology [c4]", "Unsupported claim [c99]"],
            evidence_changes=["The vendor claim [c1] was later checked independently [c3]."],
            events=[
                TimelineEventOut(
                    article_id=first.id,
                    event_type="product_launch",
                    title="Launch",
                    description="Vendor launch claim [c1].",
                ),
                TimelineEventOut(
                    article_id=999_999, event_type="other", title="Not a member", description="Ignored."
                ),
            ],
        )

    provider = FakeClaude(build)
    assert refresh_story(session, atlas, orchestrator=Orchestrator(session, provider=provider)) is True

    assert atlas.title.startswith("Atlas-3 math model")
    assert "[c42]" not in atlas.what_happened_first and "[c1]" in atlas.what_happened_first
    assert atlas.disputed == ["Nimbus Labs disputes the evaluation's methodology [c4]", "Unsupported claim"]
    events = {ev.article_id: ev for ev in atlas.events}
    assert events[first.id].description == "Vendor launch claim [c1]."
    assert events[last.id].description != "Ignored."
    assert 999_999 not in events
    # Rule-based event types stay authoritative for monitoring.
    assert events[last.id].event_type == EventType.CORRECTION

    run = session.scalars(select(AgentRun).where(AgentRun.agent == AGENT_NAME)).one()
    assert run.provider == "claude" and run.model == "claude-test"
    prompt = provider.prompts[0]
    assert "Previous summary" not in prompt  # first build
    assert "article_id=" in prompt


def test_empty_claude_summary_fields_fall_back_to_offline(session, atlas):
    provider = FakeClaude(lambda task: StoryUpdateOut(title="Atlas-3 story", what_happened_first="", whats_new=""))
    refresh_story(session, atlas, orchestrator=Orchestrator(session, provider=provider))
    assert atlas.title == "Atlas-3 story"
    assert atlas.what_happened_first.startswith("The story began on")
    assert atlas.whats_new


def test_prompt_includes_previous_summary_on_rebuild(session, atlas):
    refresh_story(session, atlas)
    provider = FakeClaude(lambda task: StoryUpdateOut(title="t", what_happened_first="a [c1]", whats_new="b [c2]"))
    refresh_story(session, atlas, orchestrator=Orchestrator(session, provider=provider), force=True)
    assert "Previous summary (version 1)" in provider.prompts[0]


def test_update_stories_clusters_and_refreshes(session, factory):
    load_storylines(factory)
    refreshed = update_stories(session)
    assert len(refreshed) == 6
    assert all(s.member_fingerprint and s.what_happened_first for s in refreshed)
    assert agent_runs(session) == 6
    assert update_stories(session) == []


def test_finance_performance_basis_is_labelled_backtest_simulated_or_live(session, factory):
    now = utcnow()
    vendor = factory.source(
        vertical="finance", trust_tier="vendor", doc_type="press_release", name="Harborview IR"
    )
    wire = factory.source(vertical="finance", name="Market Desk")
    articles = [
        factory.article(
            "Harborview Bank researchers: AI portfolio agents beat a 60/40 portfolio in a backtest",
            "Harborview Bank said its AI portfolio agents beat a 60/40 portfolio in a 2005-2024 backtest.",
            source=vendor,
            vertical="finance",
            published_at=now - timedelta(days=5),
        ),
        factory.article(
            "Harborview Bank's AI portfolio agents 'beat' a 60/40 portfolio, in a backtest",
            "The Harborview Bank result for its AI portfolio agents is a two-decade backtest, not live results.",
            source=wire,
            vertical="finance",
            published_at=now - timedelta(days=4),
        ),
        factory.article(
            "Harborview Bank to run AI portfolio agents in paper trading before any live use",
            "Harborview Bank said its AI portfolio agents will run a 12-month simulated paper-trading pilot "
            "before any live use.",
            source=wire,
            vertical="finance",
            published_at=now - timedelta(days=1),
        ),
    ]
    (story,) = assign_stories(session, articles)
    refresh_story(session, story)
    basis = next(c for c in story.evidence_changes if c.startswith("Performance evidence"))
    assert "a backtest [c1][c2]" in basis
    assert "simulated (paper-trading) results [c3]" in basis
    assert "no live track record is cited" in basis


def test_ai_benchmark_claims_are_not_given_finance_performance_labels(session, factory):
    vendor = factory.source(trust_tier="vendor", doc_type="press_release", name="Nimbus Labs Newsroom")
    art = factory.article(
        "Nimbus Labs launches Atlas-3",
        "Nimbus Labs said Atlas-3 outperforms every model it tested on algebra problems.",
        source=vendor,
    )
    (story,) = assign_stories(session, [art])
    refresh_story(session, story)
    assert "backtest" not in story.what_happened_first
    assert "company's own claims" in story.what_happened_first
