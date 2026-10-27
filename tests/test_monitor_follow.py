"""Follow This Story monitoring: the plan's example follow, subject matching, criteria, dedup, and email."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from newsren.agents import monitor
from newsren.agents.base import Orchestrator
from newsren.agents.monitor import (
    AGENT_NAME,
    check_follow,
    is_strong_keyword,
    offline_custom_decision,
    run_monitoring,
    topic_match,
)
from newsren.intelligence.entities import tag_article_entities
from newsren.llm.claude import ClaudeProvider
from newsren.models import (
    AgentRun,
    Alert,
    Article,
    DocType,
    EventType,
    Story,
    TrustTier,
    Vertical,
    utcnow,
)
from newsren.schemas import FollowCriteria, MonitorDecisionOut
from newsren.users.service import create_follow, deactivate_follow

PLAN_EXAMPLE = (
    "Tell me when an independent benchmark is published for this model, "
    "not whenever someone writes another opinion article about it."
)
EVAL_BODY = (
    "Researchers at the Open Eval Lab evaluated Lumen-4 on 1,200 held-out competition problems and found it "
    "scored 84.3%, below the 92.1% the vendor reported. The model was strong on algebra but weaker on "
    "unfamiliar multi-step word problems. This preprint has not been peer reviewed."
)


def _at(
    factory, title: str, body: str, source, retrieved_at: datetime, story: Story | None = None, **kw
) -> Article:
    article = factory.article(title, body, source=source, published_at=retrieved_at - timedelta(minutes=30), **kw)
    article.retrieved_at = retrieved_at
    if story is not None:
        article.story = story
    factory.session.flush()
    return article


@pytest.fixture()
def t0() -> datetime:
    return utcnow() - timedelta(days=3)


@pytest.fixture()
def sources(factory) -> SimpleNamespace:
    return SimpleNamespace(
        vendor=factory.source(
            trust_tier=TrustTier.VENDOR, doc_type=DocType.PRESS_RELEASE, name="Lumen Labs Newsroom"
        ),
        opinion=factory.source(trust_tier=TrustTier.COMMENTARY, doc_type=DocType.OPINION, name="The Op-Ed Desk"),
        lab=factory.source(
            trust_tier=TrustTier.INDEPENDENT, doc_type=DocType.RESEARCH_PREPRINT, name="Open Eval Lab"
        ),
        wire=factory.source(trust_tier=TrustTier.INDEPENDENT, doc_type=DocType.NEWS, name="Tech Wire"),
        harbor=factory.source(trust_tier=TrustTier.INDEPENDENT, doc_type=DocType.NEWS, name="Harbor Daily"),
        journal=factory.source(
            trust_tier=TrustTier.INDEPENDENT, doc_type=DocType.RESEARCH_PEER_REVIEWED, name="Journal of Learning"
        ),
        unapproved=factory.source(trust_tier=TrustTier.INDEPENDENT, name="Unvetted Blog", approved=False),
    )


@pytest.fixture()
def story(session, factory, sources, t0) -> Story:
    story = Story(title="Lumen-4 launch and math benchmark claims", vertical=Vertical.TECHNOLOGY)
    session.add(story)
    session.flush()
    _at(
        factory,
        "Lumen Labs launches Lumen-4, claims state-of-the-art math reasoning",
        "Lumen Labs says Lumen-4 is its most capable math model.",
        sources.vendor,
        t0 - timedelta(days=1),
        story,
    )
    return story


@pytest.fixture()
def follow(session, factory, story, t0):
    user = factory.user(display_name="Rae Reader")
    follow = create_follow(session, user, PLAN_EXAMPLE, story=story)
    follow.created_at = t0
    session.flush()
    return follow


# --------------------------------------------------------------------------
# The plan's example follow on a synthetic storyline
# --------------------------------------------------------------------------


def test_plan_example_storyline_alerts_once_on_the_independent_evaluation(
    session, factory, sources, story, follow, t0
):
    assert follow.criteria["require_independent"] is True
    assert "opinion" in follow.criteria["exclude_doc_types"]

    # 1. Another opinion article about the model: no alert.
    _at(
        factory,
        "Opinion: Lumen-4's benchmark score is a press release, not a measurement",
        "We should wait for independent testing before believing the hype about this model.",
        sources.opinion,
        t0 + timedelta(hours=1),
        story,
    )
    assert check_follow(session, follow, now=t0 + timedelta(hours=2)) == []
    assert follow.last_checked_at == t0 + timedelta(hours=2)
    assert follow.check_count == 1

    # 2. The vendor's own benchmark claim, and a news report relaying it: no alert.
    _at(
        factory,
        "Lumen Labs reports Lumen-4 scores 92.1% on MathBench-Pro benchmark",
        "Lumen Labs says its new model is the best on the MathBench-Pro benchmark.",
        sources.vendor,
        t0 + timedelta(hours=3),
        story,
    )
    _at(
        factory,
        "Lumen Labs says its Lumen-4 model tops math benchmark",
        "The company says its model scored 92.1% on MathBench-Pro, according to the company.",
        sources.wire,
        t0 + timedelta(hours=3, minutes=10),
        story,
    )
    assert check_follow(session, follow, now=t0 + timedelta(hours=4)) == []

    # 3. An independent evaluation: exactly one alert.
    evaluation = _at(
        factory,
        "Independent evaluation of Lumen-4: strong on algebra, weaker on unfamiliar word problems",
        EVAL_BODY,
        sources.lab,
        t0 + timedelta(hours=5),
        story,
    )
    alerts = check_follow(session, follow, now=t0 + timedelta(hours=6))
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.article_id == evaluation.id
    assert alert.story_id == story.id
    assert alert.event_type == EventType.INDEPENDENT_EVALUATION
    assert alert.user_id == follow.user_id
    assert alert.dedup_key.startswith(f"story:{story.id}:independent_evaluation:")
    assert "An independent evaluation was published by Open Eval Lab" in alert.reason
    assert (
        "matched your criteria: independent evaluation, independent (non-vendor) source, not opinion"
        in alert.reason
    )
    assert "preprint" in alert.reason and "not been peer reviewed" in alert.reason
    assert alert.matched_criteria[0].startswith("part of the story you follow")
    assert len(alert.citations) == 1
    citation = alert.citations[0]
    assert citation["id"] == "c1"
    assert citation["url"] == evaluation.url
    assert citation["article_id"] == evaluation.id
    assert citation["publisher"] == "Open Eval Lab"
    assert citation["doc_type"] == DocType.RESEARCH_PREPRINT
    assert citation["retrieved_at"] and citation["snippet"]

    # 4. Syndicated copies of it (one flagged by ingest, one not): no second alert.
    flagged = _at(factory, evaluation.title, EVAL_BODY, sources.harbor, t0 + timedelta(hours=7), story)
    flagged.duplicate_of_id = evaluation.id
    _at(
        factory,
        "Independent evaluation finds Lumen-4 strong on algebra but weaker on unfamiliar word problems",
        EVAL_BODY,
        sources.wire,
        t0 + timedelta(hours=7, minutes=5),
        story,
    )
    assert check_follow(session, follow, now=t0 + timedelta(hours=8)) == []
    assert session.query(Alert).filter_by(follow_id=follow.id).count() == 1

    # last_checked_at advances with every check.
    assert follow.last_checked_at == t0 + timedelta(hours=8)
    assert follow.check_count == 4


def test_articles_from_before_the_follow_or_unapproved_sources_never_alert(
    session, factory, sources, story, follow, t0
):
    # Retrieved two hours before the follow existed (beyond the creation grace period).
    _at(
        factory,
        "Independent evaluation of Lumen-4 on algebra",
        EVAL_BODY,
        sources.lab,
        t0 - timedelta(hours=2),
        story,
    )
    _at(
        factory,
        "Independent evaluation of Lumen-4 by an unvetted blog",
        "Researchers at a hobby lab independently evaluated Lumen-4 and found 80%.",
        sources.unapproved,
        t0 + timedelta(hours=1),
        story,
    )
    # Retrieved half an hour before the follow was created: within the grace period, so it counts as new.
    within_grace = _at(
        factory,
        "Independent evaluation of Lumen-4 on geometry",
        "Researchers at Bramley University independently evaluated Lumen-4 on 500 geometry problems: 71%.",
        sources.wire,
        t0 - timedelta(minutes=30),
        story,
    )
    alerts = check_follow(session, follow, now=t0 + timedelta(hours=2))
    assert [a.article_id for a in alerts] == [within_grace.id]


def test_same_story_event_is_suppressed_within_24_hours_but_new_evidence_later_alerts(
    session, factory, sources, story, follow, t0
):
    _at(
        factory,
        "Independent evaluation of Lumen-4 on algebra",
        EVAL_BODY,
        sources.lab,
        t0 + timedelta(hours=1),
        story,
    )
    assert len(check_follow(session, follow, now=t0 + timedelta(hours=2))) == 1

    # A different independent evaluation the same day is the same development for this follow.
    _at(
        factory,
        "Second lab benchmarks Lumen-4 on geometry problems",
        "Researchers at Bramley University evaluated Lumen-4 on 500 geometry problems and found 71%.",
        sources.wire,
        t0 + timedelta(hours=3),
        story,
    )
    assert check_follow(session, follow, now=t0 + timedelta(hours=4)) == []

    # Two days later: a syndicated copy of the first evaluation is still not new ...
    _at(
        factory,
        "Lumen-4 independent evaluation: strong on algebra, weaker on unfamiliar word problems",
        EVAL_BODY,
        sources.harbor,
        t0 + timedelta(days=2),
        story,
    )
    # ... but a genuinely new independent evaluation is.
    replication = _at(
        factory,
        "University team could not replicate Lumen-4's reported math scores",
        "Researchers at Calder Institute of Technology evaluated Lumen-4 on 3,000 olympiad problems and could not "
        "replicate the 92.1% the company reported; they measured 77.5% under the vendor's own prompt format.",
        sources.wire,
        t0 + timedelta(days=2, minutes=5),
        story,
    )
    alerts = check_follow(session, follow, now=t0 + timedelta(days=2, hours=1))
    assert [a.article_id for a in alerts] == [replication.id]


def test_simhash_near_duplicate_of_an_alerted_article_is_skipped(session, factory, sources, follow, t0):
    follow.story_id = None
    follow.entity_name = "Lumen-4"
    session.flush()
    original = _at(
        factory, "Independent evaluation of Lumen-4 on algebra", EVAL_BODY, sources.lab, t0 + timedelta(hours=1)
    )
    assert len(check_follow(session, follow, now=t0 + timedelta(hours=2))) == 1
    copy = _at(
        factory,
        "Lumen-4 evaluated independently on held-out problems",
        "Different wording entirely about the Lumen-4 evaluation with new sentences throughout.",
        sources.harbor,
        t0 + timedelta(days=2),
    )
    copy.simhash = original.simhash  # an ingest-level near-duplicate fingerprint
    session.flush()
    assert check_follow(session, follow, now=t0 + timedelta(days=2, hours=1)) == []


# --------------------------------------------------------------------------
# Entity and topic follows
# --------------------------------------------------------------------------


def test_entity_follow_matches_tags_and_names_but_not_price_moves(session, factory, sources, t0):
    user = factory.user()
    follow = create_follow(
        session, user, "Tell me about Corvane Semiconductor", entity_name="Corvane Semiconductor"
    )
    follow.created_at = t0
    assert follow.target_type == "company"
    corp = factory.source(trust_tier=TrustTier.VENDOR, doc_type=DocType.PRESS_RELEASE, name="Corvane Newsroom")
    results = _at(
        factory,
        "Corvane Semiconductor reports third-quarter fiscal 2026 results",
        "Revenue rose 12% from a year earlier, the company said.",
        corp,
        t0 + timedelta(hours=1),
        vertical=Vertical.FINANCE,
    )
    body_only = _at(
        factory,
        "Pinecrest Mobile signs multi-year wafer supply agreement",
        "Pinecrest Mobile signed a supply deal on Monday. The supplier, Corvane Semiconductor, will ship wafers "
        "from its Calder Valley line starting in 2027.",
        sources.wire,
        t0 + timedelta(hours=1, minutes=5),
        summary="Pinecrest Mobile signed a supply deal on Monday.",
    )
    assert "corvane semiconductor" in {e.normalized for e in tag_article_entities(session, body_only)}
    _at(
        factory,
        "Corvane Semiconductor shares jumped 8% on Tuesday",
        "Shares of the chipmaker rose after an analyst note.",
        sources.wire,
        t0 + timedelta(hours=1, minutes=10),
    )
    _at(
        factory,
        "Halbrook study on algebra practice",
        "An unrelated education study.",
        sources.wire,
        t0 + timedelta(hours=1),
    )
    alerts = check_follow(session, follow, now=t0 + timedelta(hours=2))
    assert sorted(a.article_id for a in alerts) == sorted([results.id, body_only.id])
    by_article = {a.article_id: a for a in alerts}
    assert by_article[results.id].event_type == EventType.EARNINGS
    assert "company's own account" in by_article[results.id].reason
    assert by_article[results.id].matched_criteria[0] == "mentions Corvane Semiconductor"
    assert all(a.dedup_key.startswith("article:") for a in alerts)


def test_topic_follow_uses_full_text_search_with_min_overlap(session, factory, sources, t0):
    user = factory.user()
    follow = create_follow(session, user, "Follow semiconductor capacity in Calder Valley")
    follow.created_at = t0
    assert follow.keywords == ["semiconductor capacity", "calder valley"]
    corp = factory.source(trust_tier=TrustTier.VENDOR, doc_type=DocType.PRESS_RELEASE, name="Corvane Newsroom")
    capacity = _at(
        factory,
        "Corvane Semiconductor to add 40,000 wafers per month of capacity at Calder Valley",
        "The company plans to add 3nm-class capacity at its Calder Valley site by 2028, a projection.",
        corp,
        t0 + timedelta(hours=1),
    )
    _at(  # only one of the two subjects
        factory,
        "Calder Valley school board adopts AI literacy rules",
        "The Calder Valley board voted on Tuesday.",
        sources.wire,
        t0 + timedelta(hours=1),
    )
    _at(  # both terms only deep in the body, never in the headline or summary
        factory,
        "City council debates bike lanes",
        "Council members met for three hours.",
        sources.wire,
        t0 + timedelta(hours=1),
        summary="Council members met for three hours.",
    )
    alerts = check_follow(session, follow, now=t0 + timedelta(hours=2))
    assert [a.article_id for a in alerts] == [capacity.id]
    assert alerts[0].matched_criteria[0] == "matches your keywords: semiconductor capacity, calder valley"
    assert alerts[0].event_type == EventType.ANNOUNCEMENT


def test_topic_follow_ignores_weak_single_word_matches(session, factory, sources, t0):
    user = factory.user()
    follow = create_follow(
        session, user, "Track new AI models that could change how high school students learn mathematics."
    )
    follow.created_at = t0
    _at(
        factory,
        "New AI model released for photo editing",
        "The AI model edits photos for designers.",
        sources.wire,
        t0 + timedelta(hours=1),
    )
    pilot = _at(
        factory,
        "Westhollow Unified pilots Lumen Tutor in 12 high schools",
        "Students in 12 high schools will use the AI model to learn mathematics with a tutor.",
        sources.wire,
        t0 + timedelta(hours=1),
    )
    alerts = check_follow(session, follow, now=t0 + timedelta(hours=2))
    assert [a.article_id for a in alerts] == [pilot.id]
    assert alerts[0].event_type == EventType.ADOPTION


def test_topic_match_rules():
    def art(title: str, summary: str = "", content: str = "") -> Article:
        return Article(title=title, summary=summary, content=content)

    keywords = ["independent benchmark", "model"]
    assert topic_match(keywords, art("Independent benchmarks published for new model")) == keywords
    assert topic_match(keywords, art("New model launched", "A model for chat.")) == []
    assert topic_match(["lumen labs"], art("Lumen Labs raises funding")) == ["lumen labs"]
    assert topic_match(["ai"], art("Why schools ban phones", "", "Some AI tools.")) == []
    assert is_strong_keyword("semiconductor") and is_strong_keyword("calder valley") and is_strong_keyword("sec")
    assert not is_strong_keyword("model") and not is_strong_keyword("ai")


# --------------------------------------------------------------------------
# Custom criterion (offline path and agent)
# --------------------------------------------------------------------------


def test_custom_criterion_offline_path_requires_peer_review(session, factory, sources, story, t0):
    user = factory.user()
    criteria = FollowCriteria(
        event_types=[EventType.RESEARCH_PUBLICATION, EventType.INDEPENDENT_EVALUATION],
        custom="Only peer-reviewed or formally published research meets this follow; preprints do not.",
    )
    follow = create_follow(session, user, "Peer-reviewed research on Lumen-4", story=story, criteria=criteria)
    follow.created_at = t0
    _at(
        factory,
        "Independent evaluation of Lumen-4 on algebra",
        EVAL_BODY,
        sources.lab,
        t0 + timedelta(hours=1),
        story,
    )
    paper = _at(
        factory,
        "Peer-reviewed study: Lumen-4 tutoring and algebra transfer",
        "Published in the Journal of Learning Research, the cluster-randomized study of 1,400 students found "
        "small gains on transfer problems.",
        sources.journal,
        t0 + timedelta(days=2),
        story,
    )
    alerts = check_follow(session, follow, now=t0 + timedelta(days=2, hours=1))
    assert [a.article_id for a in alerts] == [paper.id]
    assert "peer-reviewed publication" in alerts[0].matched_criteria
    runs = session.query(AgentRun).filter_by(agent=AGENT_NAME, subject_type="follow", subject_id=follow.id).all()
    assert len(runs) == 2 and all(r.provider == "offline" for r in runs)


def test_offline_custom_decision_keyword_rule(factory, sources):
    pilot = factory.article(
        "Westhollow district pilots Lumen Tutor",
        "The school district pilot covers 12 high schools this spring.",
        source=sources.wire,
    )
    pricing = factory.article(
        "Lumen Tutor district pilot pricing revealed",
        "The school district pilot pricing is $40 per student.",
        source=sources.wire,
    )
    custom = "mentions a school district pilot, not pricing"
    yes = offline_custom_decision(custom, pilot, EventType.ADOPTION)
    assert yes.matches and yes.is_material and yes.criteria_met == ["mentions school district pilot"]
    no = offline_custom_decision(custom, pricing, EventType.ADOPTION)
    assert not no.matches and "pricing" in no.reason
    opinion = factory.article(
        "Opinion: district pilots", "A school district pilot is a bad idea.", source=sources.opinion
    )
    assert not offline_custom_decision(custom, opinion, EventType.OPINION).matches


def test_agent_calls_are_capped_per_check(session, factory, sources, story, t0, monkeypatch):
    monkeypatch.setattr(monitor, "MAX_AGENT_DECISIONS_PER_CHECK", 1)
    user = factory.user()
    criteria = FollowCriteria(custom="mentions held-out problems")
    follow = create_follow(session, user, "Lumen-4 evaluations", story=story, criteria=criteria)
    follow.created_at = t0
    _at(
        factory,
        "Independent evaluation of Lumen-4 on algebra",
        EVAL_BODY,
        sources.lab,
        t0 + timedelta(hours=1),
        story,
    )
    _at(
        factory,
        "Lumen Tutor reaches 12 high schools",
        "Westhollow says students used the tutor on held-out problems.",
        sources.wire,
        t0 + timedelta(hours=1, minutes=5),
        story,
    )
    check_follow(session, follow, now=t0 + timedelta(hours=2))
    assert session.query(AgentRun).filter_by(agent=AGENT_NAME).count() == 1


class _FakeMessages:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=900, output_tokens=150, cache_read_input_tokens=0, cache_creation_input_tokens=0
            ),
            model="claude-opus-5-5",
        )


def test_follow_monitor_agent_judges_custom_condition_with_claude(session, factory, sources, story, t0):
    decision = MonitorDecisionOut(
        matches=True,
        event_type="not-a-real-type",
        is_material=True,
        is_opinion=False,
        criteria_met=["evaluation on held-out problems"],
        reason="Open Eval Lab evaluated Lumen-4 on held-out problems [c1][c7].",
    )
    fake = _FakeMessages(json.dumps(decision.model_dump()))
    provider = ClaudeProvider(model="claude-opus-5-5", client=SimpleNamespace(beta=SimpleNamespace(messages=fake)))
    orchestrator = Orchestrator(session, provider=provider)
    user = factory.user()
    criteria = FollowCriteria(require_independent=True, custom="an evaluation on held-out problems")
    follow = create_follow(session, user, "Held-out evaluations of Lumen-4", story=story, criteria=criteria)
    follow.created_at = t0
    _at(  # excluded by rules before the agent is asked
        factory, "Opinion: held-out problems matter", "Commentary on held-out problems.", sources.opinion,
        t0 + timedelta(hours=1), story,
    )  # fmt: skip
    evaluation = _at(
        factory, "Independent evaluation of Lumen-4", EVAL_BODY, sources.lab, t0 + timedelta(hours=1), story
    )
    alerts = check_follow(session, follow, orchestrator, now=t0 + timedelta(hours=2))

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert "Follow This Story monitoring agent" in call["system"][0]["text"]
    prompt = call["messages"][0]["content"]
    assert "Held-out evaluations of Lumen-4" in prompt and "an evaluation on held-out problems" in prompt
    assert '<source id="c1">' in prompt and evaluation.url in prompt
    assert [a.article_id for a in alerts] == [evaluation.id]
    alert = alerts[0]
    assert alert.event_type == EventType.INDEPENDENT_EVALUATION  # unknown agent type replaced by the rule type
    assert "evaluation on held-out problems" in alert.matched_criteria
    assert "[c7]" not in alert.reason and "[c1]" not in alert.reason
    run = session.query(AgentRun).filter_by(agent=AGENT_NAME).one()
    assert run.provider == "claude" and run.input_tokens == 900


# --------------------------------------------------------------------------
# Scheduler and email digests
# --------------------------------------------------------------------------


def test_run_monitoring_counts_and_emails_opted_in_readers(session, settings, factory, sources, story, t0):
    opted_in = factory.user(email="rae@example.com", display_name="Rae", email_opt_in=True)
    quiet = factory.user(email="quiet@example.com")
    f1 = create_follow(session, opted_in, PLAN_EXAMPLE, story=story)
    f2 = create_follow(session, quiet, PLAN_EXAMPLE, story=story)
    f3 = create_follow(session, quiet, "Lumen Labs news", entity_name="Lumen Labs")
    for f in (f1, f2, f3):
        f.created_at = t0
    deactivate_follow(session, f3)
    evaluation = _at(
        factory, "Independent evaluation of Lumen-4", EVAL_BODY, sources.lab, t0 + timedelta(hours=1), story
    )

    counts = run_monitoring(session, now=t0 + timedelta(hours=2))
    assert counts == {"follows_checked": 2, "alerts_created": 2, "emails_sent": 1, "errors": 0}
    a1 = session.query(Alert).filter_by(follow_id=f1.id).one()
    a2 = session.query(Alert).filter_by(follow_id=f2.id).one()
    assert a1.emailed_at is not None and a2.emailed_at is None
    assert f3.check_count == 0

    outbox = list(settings.outbox_dir.glob("*.eml"))
    assert len(outbox) == 1
    raw = outbox[0].read_text()
    assert "rae@example.com" in raw
    assert "independent benchmark is published" in raw
    assert evaluation.url in raw

    # Nothing new: no alerts, no emails, but the checks are counted.
    assert run_monitoring(session, now=t0 + timedelta(hours=3))["alerts_created"] == 0
    assert len(list(settings.outbox_dir.glob("*.eml"))) == 1
    assert f1.check_count == 2


def test_alert_email_respects_automatic_delivery_switch(session, settings, factory, sources, story, t0):
    from newsren.config import set_settings

    set_settings(settings.with_overrides(auto_deliver_personal_briefings=False))
    user = factory.user(email_opt_in=True)
    follow = create_follow(session, user, PLAN_EXAMPLE, story=story)
    follow.created_at = t0
    _at(factory, "Independent evaluation of Lumen-4", EVAL_BODY, sources.lab, t0 + timedelta(hours=1), story)
    alerts = check_follow(session, follow, now=t0 + timedelta(hours=2))
    assert len(alerts) == 1 and alerts[0].emailed_at is None
    assert not list(settings.outbox_dir.glob("*.eml"))


def test_correction_to_an_alerted_development_is_always_sent(session, factory, sources, story, follow, t0):
    _at(
        factory,
        "Independent evaluation of Lumen-4 on algebra",
        EVAL_BODY,
        sources.lab,
        t0 + timedelta(hours=1),
        story,
    )
    assert len(check_follow(session, follow, now=t0 + timedelta(hours=2))) == 1
    # A reader who started following after the evaluation has no alert it could correct.
    latecomer = create_follow(session, factory.user(), PLAN_EXAMPLE, story=story)
    latecomer.created_at = t0 + timedelta(hours=2, minutes=30)

    correction = _at(
        factory,
        "Correction: Independent benchmark score for Lumen-4 was 84.3%, not 81.3%",
        "An earlier version of this article misstated Lumen-4's score on the held-out benchmark.",
        sources.wire,
        t0 + timedelta(hours=3),
        story,
    )
    alerts = check_follow(session, follow, now=t0 + timedelta(hours=4))
    assert [a.article_id for a in alerts] == [correction.id]
    assert alerts[0].event_type == EventType.CORRECTION
    assert "correction to a development we alerted you about" in alerts[0].reason
    assert "It corrects earlier reporting." in alerts[0].reason
    assert check_follow(session, latecomer, now=t0 + timedelta(hours=4)) == []
