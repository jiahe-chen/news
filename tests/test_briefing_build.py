"""My Newsren personal briefings: matching, ranking, sections, citations, novelty, depth, delivery, agent output."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from newsren.agents.base import Orchestrator, citation_for
from newsren.agents.briefing import (
    AGENT_NAME,
    ALERTS_HEADING,
    build_briefing,
    deliver_briefing,
    run_briefings,
)
from newsren.config import set_settings
from newsren.editorial.guardrails import FINANCE_DISCLAIMER
from newsren.llm import LLMResult
from newsren.models import (
    AgentRun,
    Alert,
    Article,
    Brief,
    BriefClaim,
    Briefing,
    DocType,
    Story,
    TrustTier,
    Vertical,
    utcnow,
)
from newsren.schemas import BriefingOut
from newsren.users.service import add_interest, create_follow, set_preferences


def _brief(
    session,
    story: Story | None,
    articles: list[Article],
    *,
    headline: str,
    summary: str,
    what_changed: str,
    why: str,
    vertical: str,
    published_at: datetime,
    claims: list[dict] | None = None,
    status: str = "approved",
    uncertainty: list[str] | None = None,
) -> Brief:
    brief = Brief(
        story=story,
        vertical=vertical,
        headline=headline,
        summary=summary,
        what_changed=what_changed,
        why_it_matters=why,
        citations=[citation_for(a, f"c{i}").model_dump() for i, a in enumerate(articles, start=1)],
        status=status,
        published_at=published_at if status == "approved" else None,
        requires_human_review=False,
        uncertainty=uncertainty or [],
    )
    for position, claim in enumerate(claims or []):
        brief.claims.append(BriefClaim(position=position, support_status="supported", **claim))
    session.add(brief)
    session.flush()
    return brief


def _story(session, title: str, vertical: str, articles: list[Article], **kwargs) -> Story:
    story = Story(title=title, vertical=vertical, **kwargs)
    session.add(story)
    for article in articles:
        article.story = story
    session.flush()
    return story


@pytest.fixture()
def now() -> datetime:
    return utcnow()


@pytest.fixture()
def world(session, factory, now) -> SimpleNamespace:
    """Approved briefs across verticals (created directly, as the brief agent would leave them)."""
    vendor = factory.source(
        trust_tier=TrustTier.VENDOR, doc_type=DocType.PRESS_RELEASE, name="Lumen Labs Newsroom"
    )
    lab = factory.source(
        trust_tier=TrustTier.INDEPENDENT, doc_type=DocType.RESEARCH_PREPRINT, name="Open Eval Lab"
    )
    corvane_pr = factory.source(
        vertical=Vertical.INDUSTRY, trust_tier=TrustTier.VENDOR, doc_type=DocType.PRESS_RELEASE, name="Corvane"
    )
    fabline = factory.source(vertical=Vertical.INDUSTRY, name="FabLine Weekly")
    bank = factory.source(
        vertical=Vertical.FINANCE, trust_tier=TrustTier.VENDOR, doc_type=DocType.PRESS_RELEASE, name="Northbridge"
    )
    wire = factory.source(vertical=Vertical.FINANCE, name="Ledger Wire")
    city = factory.source(vertical=Vertical.INDUSTRY, name="City Desk")

    launch = factory.article(
        "Lumen Labs launches Lumen-4", "Lumen Labs says Lumen-4 scores 92.1% on MathBench-Pro.", source=vendor
    )
    evaluation = factory.article(
        "Independent evaluation of Lumen-4",
        "Open Eval Lab found Lumen-4 scored 84.3% on 1,200 held-out math problems. This preprint has not been "
        "peer reviewed.",
        source=lab,
    )
    lumen_story = _story(
        session,
        "Lumen Labs' Lumen-4 math model: vendor claims and an independent evaluation",
        Vertical.TECHNOLOGY,
        [launch, evaluation],
        key_entities=["Lumen Labs", "Lumen-4", "Open Eval Lab"],
        keywords=["lumen-4", "math", "benchmark"],
    )
    lumen = _brief(
        session,
        lumen_story,
        [launch, evaluation],
        headline="Independent evaluation finds Lumen-4 weaker on unfamiliar math word problems",
        summary="Lumen Labs says Lumen-4 scores 92.1% on MathBench-Pro [c1]. An independent preprint found 84.3% [c2].",
        what_changed=(
            "An independent evaluation, still a preprint, found Lumen-4 scored 84.3% on 1,200 held-out problems "
            "[c2]. The company had reported 92.1% on its own benchmark [c1]. Open Eval Lab plans a follow-up on "
            "geometry [c2]."
        ),
        why=(
            "Schools weighing AI math tutors now have independent evidence that the model is weaker on unfamiliar "
            "problems [c2]. The evaluation has not been peer reviewed [c2]."
        ),
        vertical=Vertical.TECHNOLOGY,
        published_at=now - timedelta(days=1),
        claims=[
            {
                "text": "Open Eval Lab measured 84.3% on 1,200 held-out problems written after the training cutoff [c2].",
                "claim_type": "fact",
                "citation_ids": ["c2"],
            },
            {
                "text": "Lumen Labs reported 92.1% on MathBench-Pro [c1].",
                "claim_type": "fact",
                "citation_ids": ["c1"],
            },
        ],
        uncertainty=["Whether the gap holds on geometry problems is not yet known [c2]."],
    )

    capacity = factory.article(
        "Corvane Semiconductor to add capacity at Calder Valley",
        "Corvane Semiconductor plans 40,000 wafers per month of 3nm-class capacity by late 2027.",
        source=corvane_pr,
    )
    output = factory.article(
        "Corvane's Calder Valley line reached 11,500 wafers a month",
        "Corvane reported its line reached 11,500 wafers a month, behind its 15,000 interim target.",
        source=fabline,
    )
    corvane_story = _story(
        session, "Corvane Semiconductor capacity at Calder Valley", Vertical.INDUSTRY, [capacity, output]
    )
    corvane = _brief(
        session,
        corvane_story,
        [capacity, output],
        headline="Corvane's Calder Valley line reached 11,500 wafers a month, behind its 15,000 target",
        summary="Corvane Semiconductor plans 40,000 wafers per month of capacity [c1]. Output reached 11,500 [c2].",
        what_changed="Reported output of 11,500 wafers a month trails the 15,000 interim target [c2].",
        why="Capacity plans are projections; reported output shows the semiconductor capacity ramp is slower [c2][c1].",
        vertical=Vertical.INDUSTRY,
        published_at=now - timedelta(days=2),
    )

    backtest = factory.article(
        "Northbridge researchers: AI portfolio agents outperformed a 60/40 benchmark in a 20-year backtest",
        "Northbridge researchers said the result is in-sample and does not prove future outperformance.",
        source=bank,
    )
    wire_story_article = factory.article(
        "Bank's AI agents 'beat' a 60/40 portfolio, in a backtest",
        "The researchers cautioned that a backtest is not a live track record.",
        source=wire,
    )
    finance_story = _story(
        session, "Northbridge AI portfolio agents backtest", Vertical.FINANCE, [backtest, wire_story_article]
    )
    northbridge = _brief(
        session,
        finance_story,
        [backtest, wire_story_article],
        headline="Northbridge's AI portfolio agents beat a 60/40 benchmark in a 20-year backtest",
        summary="Northbridge researchers reported a backtest result [c1].",
        what_changed=(
            "Northbridge researchers reported that AI portfolio agents outperformed a 60/40 portfolio in a 20-year "
            "backtest [c1]. The researchers caution that in-sample results do not prove future performance [c2]."
        ),
        why="The result is a backtest, not a live track record [c2].",
        vertical=Vertical.FINANCE,
        published_at=now - timedelta(days=1),
        claims=[
            {
                "text": "AI portfolio agents outperformed a 60/40 portfolio over 2006-2026 [c1].",
                "claim_type": "performance",
                "citation_ids": ["c1"],
                "performance_label": "backtest",
                "performance_period": "2006-2026",
                "benchmark": "60/40 portfolio",
            }
        ],
    )

    lanes = factory.article("City council approves bike lanes", "The council voted 7-2.", source=city)
    unrelated = _brief(
        session,
        _story(session, "City bike lanes", Vertical.INDUSTRY, [lanes]),
        [lanes],
        headline="City council approves new bike lanes",
        summary="The council voted 7-2 for bike lanes [c1].",
        what_changed="The council approved the plan [c1].",
        why="Cyclists get new lanes [c1].",
        vertical=Vertical.INDUSTRY,
        published_at=now - timedelta(days=1),
    )
    old_launch = factory.article("Lumen Labs teases Lumen-4", "Lumen Labs teased a math model.", source=vendor)
    old = _brief(
        session,
        _story(session, "Lumen Labs teases Lumen-4", Vertical.TECHNOLOGY, [old_launch]),
        [old_launch],
        headline="Lumen Labs teases its Lumen-4 math model",
        summary="Lumen Labs teased Lumen-4 [c1].",
        what_changed="Lumen Labs teased a model [c1].",
        why="It may matter for math tutoring [c1].",
        vertical=Vertical.TECHNOLOGY,
        published_at=now - timedelta(days=10),
    )
    draft = _brief(
        session,
        lumen_story,
        [evaluation],
        headline="Draft: Lumen Labs Lumen-4 follow-up awaiting review",
        summary="Draft [c1].",
        what_changed="Draft [c1].",
        why="Draft [c1].",
        vertical=Vertical.TECHNOLOGY,
        published_at=now,
        status="in_review",
    )
    return SimpleNamespace(
        lumen=lumen,
        corvane=corvane,
        northbridge=northbridge,
        unrelated=unrelated,
        old=old,
        draft=draft,
        lumen_story=lumen_story,
        corvane_story=corvane_story,
        output=output,
        fabline=fabline,
        evaluation=evaluation,
    )


@pytest.fixture()
def reader(session, factory):
    user = factory.user(email="rae@example.com", display_name="Rae")
    add_interest(session, user, "topic", "AI models for learning mathematics")
    add_interest(session, user, "organization", "Lumen Labs")
    add_interest(session, user, "topic", "semiconductor capacity")
    add_interest(session, user, "sector", "Finance")
    return user


def _items(briefing: Briefing) -> list[dict]:
    return [item for section in briefing.content["sections"] for item in section["items"]]


# --------------------------------------------------------------------------
# Building a briefing
# --------------------------------------------------------------------------


def test_briefing_has_sections_per_interest_and_renumbered_citations(session, world, reader, now):
    briefing = build_briefing(session, reader, now=now)
    assert briefing is not None
    content = briefing.content

    # Only published, in-period, matching briefs; one item each.
    expected = {world.lumen.id, world.corvane.id, world.northbridge.id}
    assert set(content["brief_ids"]) == expected
    assert sorted(i["brief_id"] for i in _items(briefing)) == sorted(expected)
    assert briefing.item_count == 3
    headings = {s["heading"]: [i["brief_id"] for i in s["items"]] for s in content["sections"]}
    assert headings == {
        "Lumen Labs": [world.lumen.id],
        "Semiconductor capacity": [world.corvane.id],
        "Finance": [world.northbridge.id],
    }
    assert content["story_ids"] and world.lumen_story.id in content["story_ids"]
    for item in _items(briefing):
        assert item["interest"] in headings
        assert item["what_changed"] and item["why_it_matters"]
        assert "[c" not in item["what_changed"] + item["why_it_matters"]

    # One merged citation list renumbered c1..cN, every entry cited by some item, mapped to the right sources.
    ids = [c["id"] for c in briefing.citations]
    assert ids == [f"c{n}" for n in range(1, len(ids) + 1)]
    cited = [cid for item in _items(briefing) for cid in item["citation_ids"]]
    assert set(cited) == set(ids)
    by_id = {c["id"]: c for c in briefing.citations}
    by_brief = {i["brief_id"]: i for i in _items(briefing)}
    for brief in (world.lumen, world.corvane, world.northbridge):
        urls = {by_id[cid]["url"] for cid in by_brief[brief.id]["citation_ids"]}
        assert urls and urls <= {c["url"] for c in brief.citations}
    assert world.output.url in {by_id[c]["url"] for c in by_brief[world.corvane.id]["citation_ids"]}
    assert len({c["url"] for c in briefing.citations}) == len(ids)

    # Evidence labels, guardrails, provenance.
    assert "preprint" in by_brief[world.lumen.id]["why_it_matters"]
    assert "backtest" in by_brief[world.northbridge.id]["why_it_matters"]
    assert FINANCE_DISCLAIMER in content["closing_note"]
    assert briefing.generated_by == "offline" and briefing.delivered_at is None
    assert session.get(AgentRun, briefing.agent_run_id).agent == AGENT_NAME
    assert briefing.period_start == now - timedelta(days=7) and briefing.period_end == now
    assert "Lumen Labs" in briefing.title or "Semiconductor capacity" in briefing.title


def test_shared_sources_are_merged_once(session, factory, world, now):
    user = factory.user()
    add_interest(session, user, "topic", "semiconductor capacity")
    _brief(
        session,
        _story(session, "Calder Valley wafer output", Vertical.INDUSTRY, []),
        [world.output],
        headline="Semiconductor capacity: FabLine tracks Calder Valley output",
        summary="FabLine reported output at 11,500 wafers a month [c1].",
        what_changed="The semiconductor capacity ramp at Calder Valley is slower than planned [c1].",
        why="Customers depend on that capacity [c1].",
        vertical=Vertical.INDUSTRY,
        published_at=now - timedelta(hours=5),
    )
    briefing = build_briefing(session, user, now=now)
    assert briefing.item_count == 2
    urls = [c["url"] for c in briefing.citations]
    assert urls.count(world.output.url) == 1
    output_id = next(c["id"] for c in briefing.citations if c["url"] == world.output.url)
    assert all(output_id in item["citation_ids"] for item in _items(briefing))


def test_no_repeats_next_time_and_none_when_nothing_new(session, factory, world, reader, now):
    first = build_briefing(session, reader, now=now)
    assert first is not None
    assert build_briefing(session, reader, now=now + timedelta(hours=1)) is None
    reader.briefing_frequency = "weekly"  # still needs at least one item
    assert build_briefing(session, reader, now=now + timedelta(hours=1)) is None
    assert build_briefing(session, reader, now=now + timedelta(hours=1), force=True) is None

    update = _brief(
        session,
        world.corvane_story,
        [world.output],
        headline="Corvane delays its Calder Valley interim capacity target",
        summary="Corvane moved its 15,000-wafer target to next quarter [c1].",
        what_changed="Corvane moved its 15,000-wafer interim target to next quarter [c1].",
        why="Customers waiting on semiconductor capacity face a later ramp [c1].",
        vertical=Vertical.INDUSTRY,
        published_at=now + timedelta(hours=2),
    )
    third = build_briefing(session, reader, now=now + timedelta(hours=3))
    assert third is not None
    assert third.content["brief_ids"] == [update.id]
    assert _items(third)[0]["what_changed"].startswith("Corvane moved its 15,000-wafer interim target")
    assert [c["id"] for c in third.citations] == ["c1"]


def test_returns_none_when_no_interest_matches(session, factory, world, now):
    user = factory.user()
    add_interest(session, user, "topic", "marine biology field research")
    assert build_briefing(session, user, now=now) is None
    assert build_briefing(session, user, now=now, force=True) is None
    nobody = factory.user()
    assert build_briefing(session, nobody, now=now) is None


def test_meaningfulness_threshold_respects_frequency_and_force(session, factory, world, now):
    user = factory.user()
    add_interest(session, user, "topic", "semiconductor capacity")
    later = now + timedelta(days=4)  # the Corvane brief is now six days old: relevant but stale
    assert build_briefing(session, user, now=later) is None
    forced = build_briefing(session, user, now=later, force=True)
    assert forced is not None and forced.content["brief_ids"] == [world.corvane.id]

    weekly = factory.user(briefing_frequency="weekly")
    add_interest(session, weekly, "topic", "semiconductor capacity")
    assert build_briefing(session, weekly, now=later).content["brief_ids"] == [world.corvane.id]


def test_depth_controls_detail(session, factory, world, now):
    technical = factory.user()
    set_preferences(session, technical, depth="technical", audience="professional")
    add_interest(session, technical, "organization", "Lumen Labs")
    add_interest(session, technical, "sector", "finance")
    tech_items = {i["brief_id"]: i for i in _items(build_briefing(session, technical, now=now))}
    lumen = tech_items[world.lumen.id]
    assert "Key technical claims: Open Eval Lab measured 84.3% on 1,200 held-out problems" in lumen["what_changed"]
    assert "Still uncertain: Whether the gap holds on geometry" in lumen["why_it_matters"]
    assert "(backtest, 2006-2026, 60/40 portfolio)" in tech_items[world.northbridge.id]["what_changed"]

    intro = factory.user()
    set_preferences(session, intro, depth="intro", audience="student")
    add_interest(session, intro, "organization", "Lumen Labs")
    briefing = build_briefing(session, intro, now=now)
    item = _items(briefing)[0]
    assert item["what_changed"] == (
        "An independent evaluation, still a preprint, found Lumen-4 scored 84.3% on 1,200 held-out problems."
    )
    assert "not yet checked by independent reviewers" in item["why_it_matters"]
    assert "Key technical claims" not in item["what_changed"]
    assert "plain language" in briefing.content["intro"] and "a student" in briefing.content["intro"]
    assert briefing.content["depth"] == "intro"


def test_unread_alerts_are_included_once(session, factory, world, now):
    user = factory.user()
    follow = create_follow(
        session,
        user,
        "Tell me when an independent benchmark is published for this model.",
        story=world.lumen_story,
    )
    on_story = Alert(
        follow_id=follow.id,
        user_id=user.id,
        article_id=world.evaluation.id,
        story_id=world.lumen_story.id,
        event_type="independent_evaluation",
        title="Independent evaluation of Lumen-4",
        reason="An independent evaluation was published by Open Eval Lab; matched your criteria: not opinion.",
        citations=[citation_for(world.evaluation, "c1").model_dump()],
        dedup_key="article:1",
        created_at=now - timedelta(hours=3),
    )
    elsewhere = Alert(
        follow_id=follow.id,
        user_id=user.id,
        title="Second lab benchmarks Lumen-4 on geometry",
        reason="An independent evaluation was reported by Tech Wire; matched your criteria: not opinion.",
        citations=[{"id": "c1", "url": "https://wire.example/geometry", "title": "Second lab benchmarks Lumen-4"}],
        dedup_key="article:2",
        created_at=now - timedelta(hours=2),
    )
    read = Alert(
        follow_id=follow.id,
        user_id=user.id,
        title="Already read",
        reason="Read earlier.",
        dedup_key="article:3",
        created_at=now - timedelta(hours=2),
        read_at=now - timedelta(hours=1),
    )
    session.add_all([on_story, elsewhere, read])
    session.flush()

    briefing = build_briefing(session, user, now=now)
    sections = {s["heading"]: s["items"] for s in briefing.content["sections"]}
    assert [i["headline"] for i in sections[ALERTS_HEADING]] == ["Second lab benchmarks Lumen-4 on geometry"]
    story_section = next(items for heading, items in sections.items() if heading.startswith("Following:"))
    assert story_section[0]["brief_id"] == world.lumen.id
    assert "met a question you follow" in story_section[0]["why_it_matters"]
    assert sorted(briefing.content["alert_ids"]) == sorted([on_story.id, elsewhere.id])
    assert "https://wire.example/geometry" in {c["url"] for c in briefing.citations}
    assert build_briefing(session, user, now=now + timedelta(minutes=5)) is None


# --------------------------------------------------------------------------
# Delivery and scheduling
# --------------------------------------------------------------------------


def test_deliver_emails_opted_in_reader_and_advances_last_briefing(session, settings, world, reader, now):
    set_preferences(session, reader, email_opt_in=True)
    briefing = build_briefing(session, reader, now=now)
    deliver_briefing(session, briefing, reader)
    assert briefing.delivery_channel == "email"
    assert briefing.delivered_at is not None
    assert reader.last_briefing_at == briefing.period_end
    outbox = list(settings.outbox_dir.glob("*.eml"))
    assert len(outbox) == 1
    raw = outbox[0].read_text()
    assert "rae@example.com" in raw and "Corvane" in raw and "Sources" in raw


def test_deliver_web_only_without_opt_in_or_when_auto_delivery_is_off(session, settings, factory, world, now):
    user = factory.user()
    add_interest(session, user, "topic", "semiconductor capacity")
    briefing = deliver_briefing(session, build_briefing(session, user, now=now), user)
    assert briefing.delivery_channel == "web" and briefing.delivered_at is not None

    set_settings(settings.with_overrides(auto_deliver_personal_briefings=False))
    opted = factory.user(email_opt_in=True)
    add_interest(session, opted, "organization", "Lumen Labs")
    briefing = deliver_briefing(session, build_briefing(session, opted, now=now), opted)
    assert briefing.delivery_channel == "web"
    assert not list(settings.outbox_dir.glob("*.eml"))


def test_run_briefings_builds_delivers_and_respects_cadence(session, factory, world, reader, now):
    factory.user()  # no interests or follows: not considered
    nothing = factory.user()
    add_interest(session, nothing, "topic", "marine biology field research")

    counts = run_briefings(session, now=now)
    assert counts == {
        "users_considered": 2,
        "briefings_built": 1,
        "delivered_web": 1,
        "delivered_email": 0,
        "not_due": 0,
        "nothing_new": 1,
        "errors": 0,
    }
    assert reader.last_briefing_at == now
    assert session.query(Briefing).filter_by(user_id=reader.id).count() == 1

    again = run_briefings(session, now=now + timedelta(hours=1))
    assert again["not_due"] == 1 and again["briefings_built"] == 0 and again["nothing_new"] == 1


# --------------------------------------------------------------------------
# Agent output validation
# --------------------------------------------------------------------------


class _RewritingProvider:
    """Stands in for Claude: rewrites the planned briefing, including mistakes the validator must catch."""

    name = "claude"
    model = "claude-opus-5-5"

    def __init__(self, drop_item_in_section: int | None = None):
        self.tasks = []
        self.drop_item_in_section = drop_item_in_section

    def generate(self, task, budget):
        self.tasks.append(task)
        plan: BriefingOut = task.offline()
        out = plan.model_copy(deep=True)
        out.title = "Capacity, backtests, and what changed this week"
        out.closing_note = "Check the sources."  # finance disclaimer missing
        for s_index, section in enumerate(out.sections):
            for item in section.items:
                if item.brief_id is None:
                    continue
                item.what_changed = f"Rewritten for the reader: {item.what_changed}"
                item.citation_ids = [*item.citation_ids, "c99"]
                if "backtest" in item.headline.lower():
                    item.why_it_matters = (
                        "You should buy Northbridge shares now; AI agents reliably beat the market."
                    )
            if self.drop_item_in_section == s_index:
                section.items = section.items[:-1]
        return LLMResult(output=out, provider="claude", model=self.model)

    def run_tools(self, task, budget):  # pragma: no cover - not used by this agent
        raise NotImplementedError


def test_agent_output_is_validated_against_plan_and_guardrails(session, factory, world, now):
    user = factory.user()
    add_interest(session, user, "topic", "semiconductor capacity")
    add_interest(session, user, "sector", "Finance")
    provider = _RewritingProvider()
    briefing = build_briefing(session, user, orchestrator=Orchestrator(session, provider=provider), now=now)

    task = provider.tasks[0]
    assert "My Newsren briefing agent" in task.system
    assert "allowed_citation_ids" in task.prompt and "Depth: standard" in task.prompt
    assert briefing.generated_by == "claude-opus-5-5"
    assert briefing.title == "Capacity, backtests, and what changed this week"
    items = {i["brief_id"]: i for i in _items(briefing)}
    assert items[world.corvane.id]["what_changed"].startswith("Rewritten for the reader:")
    all_ids = {c["id"] for c in briefing.citations}
    assert all(set(i["citation_ids"]) <= all_ids for i in items.values())
    assert len(all_ids) <= 4  # c99 never survives renumbering
    finance_item = items[world.northbridge.id]
    assert "buy" not in finance_item["why_it_matters"].lower()
    assert not finance_item["what_changed"].startswith("Rewritten")  # the whole item fell back
    assert FINANCE_DISCLAIMER in briefing.content["closing_note"]


def test_agent_output_with_wrong_structure_falls_back_per_section(session, factory, world, now):
    user = factory.user()
    add_interest(session, user, "topic", "semiconductor capacity")
    _brief(
        session,
        _story(session, "Orrin lead times", Vertical.INDUSTRY, []),
        [world.output],
        headline="Lithography lead times stretch as semiconductor capacity expands",
        summary="Tool lead times reached 18 months [c1].",
        what_changed="Semiconductor capacity expansions pushed lithography lead times to 18 months [c1].",
        why="Fab ramps depend on tools [c1].",
        vertical=Vertical.INDUSTRY,
        published_at=now - timedelta(hours=6),
    )
    provider = _RewritingProvider(drop_item_in_section=0)
    briefing = build_briefing(session, user, orchestrator=Orchestrator(session, provider=provider), now=now)
    items = _items(briefing)
    assert len(items) == 2
    assert not any(i["what_changed"].startswith("Rewritten") for i in items)
