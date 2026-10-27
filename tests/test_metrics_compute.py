"""Success metrics (plan §9) on an empty database and on seeded rows."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from newsren.metrics import (
    all_metrics,
    cost_metrics,
    editorial_metrics,
    education_metrics,
    engagement_metrics,
    trust_metrics,
    usefulness_metrics,
)
from newsren.models import (
    AgentRun,
    Alert,
    Brief,
    BriefClaim,
    Correction,
    EditorialAction,
    Feedback,
    Follow,
    GeneratedContent,
    PageView,
    QuizAttempt,
    Story,
    utcnow,
)


def _brief(session, status: str = "approved", vertical: str = "technology", **kwargs) -> Brief:
    brief = Brief(vertical=vertical, headline="H", summary="S", status=status, **kwargs)
    session.add(brief)
    session.flush()
    return brief


def _claim(session, brief: Brief, *, claim_type="fact", citation_ids=("c1",), verdict=None) -> BriefClaim:
    claim = BriefClaim(
        brief_id=brief.id,
        text="claim",
        claim_type=claim_type,
        citation_ids=list(citation_ids),
        editor_verdict=verdict,
    )
    session.add(claim)
    return claim


# --------------------------------------------------------------------------- empty database


def test_empty_database(session):
    metrics = all_metrics(session)
    json.dumps(metrics)  # plain JSON-serialisable values only
    assert set(metrics) == {"generated_at", "trust", "usefulness", "education", "engagement", "editorial", "cost"}

    trust = metrics["trust"]
    assert trust["claims_judged"] == 0 and trust["claim_accuracy_rate"] is None
    assert trust["published_briefs"] == 0 and trust["material_corrections_per_100_published"] is None
    assert trust["citation_coverage"] is None and trust["accuracy_by_vertical"] == {}

    useful = metrics["usefulness"]
    assert useful["responses"] == 0 and useful["mean_rating"] is None and useful["by_target_type"] == {}

    education = metrics["education"]
    assert education["attempts"] == 0
    assert education["by_condition"]["newsren"]["comprehension_accuracy"] is None
    assert education["comprehension_delta"] is None and education["transfer_delta"] is None

    engagement = metrics["engagement"]
    assert engagement["follows_total"] == 0 and engagement["return_visits"] == 0
    assert engagement["return_visit_rate"] is None and engagement["useful_alert_rate"] is None

    editorial = metrics["editorial"]
    assert editorial["approvals"] == 0 and editorial["median_review_seconds"] is None
    assert editorial["estimated_minutes_saved_per_approved_brief"] is None
    assert editorial["time_saved_is_estimate"] is True

    cost = metrics["cost"]
    assert cost["runs"] == 0 and cost["total_cost_usd"] == 0 and cost["avg_cost_usd_per_run"] is None
    assert cost["per_story_brief"]["runs"] == 0 and cost["per_story_brief"]["avg_cost_usd"] is None
    assert cost["per_monitoring_check"]["avg_cost_usd_per_check"] is None
    assert cost["by_agent"] == {}


# --------------------------------------------------------------------------- trust


def test_trust_metrics(session):
    published = _brief(session, vertical="technology")
    finance = _brief(session, vertical="finance")
    draft = _brief(session, status="in_review")
    _claim(session, published, verdict="accurate")
    _claim(session, published, verdict="accurate", claim_type="performance")
    _claim(session, published, verdict="inaccurate", citation_ids=())
    _claim(session, published, claim_type="interpretation", citation_ids=())  # not a fact claim
    _claim(session, finance, verdict="accurate", claim_type="performance")
    _claim(session, draft, citation_ids=())
    _claim(session, draft)
    session.add_all(
        [
            Correction(brief_id=published.id, description="Wrong date", material=True),
            Correction(brief_id=published.id, description="Typo", material=False),
            Correction(brief_id=draft.id, description="Unpublished draft fix", material=True),
        ]
    )
    session.flush()

    trust = trust_metrics(session)
    assert trust["claims_judged"] == 4
    assert trust["claims_accurate"] == 3 and trust["claims_inaccurate"] == 1
    assert trust["claim_accuracy_rate"] == 0.75
    assert trust["accuracy_by_vertical"] == {
        "finance": {"judged": 1, "accurate": 1, "rate": 1.0},
        "technology": {"judged": 3, "accurate": 2, "rate": 0.6667},
    }
    assert trust["published_briefs"] == 2
    # Only corrections to published briefs count toward the published-correction rate.
    assert trust["corrections_total"] == 2 and trust["material_corrections"] == 1
    assert trust["material_corrections_per_100_published"] == 50.0
    assert trust["fact_claims"] == 6 and trust["fact_claims_cited"] == 4
    assert trust["citation_coverage"] == 0.6667
    assert trust["published_fact_claims"] == 4 and trust["published_citation_coverage"] == 0.75


# --------------------------------------------------------------------------- usefulness


def test_usefulness_metrics(session):
    session.add_all(
        [
            Feedback(target_type="briefing", target_id=1, answered_question=True, saved_time=True, rating=5),
            Feedback(target_type="briefing", target_id=1, answered_question=False, saved_time=True, rating=3),
            Feedback(target_type="answer", target_id=2, answered_question=True, saved_time=None, rating=None),
            Feedback(target_type="answer", target_id=3, answered_question=None, saved_time=False, rating=4),
        ]
    )
    session.flush()
    useful = usefulness_metrics(session)
    assert useful["responses"] == 4
    assert useful["answered_question_rate"] == 0.6667 and useful["answered_question_n"] == 3
    assert useful["saved_time_rate"] == 0.6667 and useful["saved_time_n"] == 3
    assert useful["mean_rating"] == 4.0 and useful["rating_n"] == 3
    assert useful["by_target_type"]["briefing"]["answered_question_rate"] == 0.5
    assert useful["by_target_type"]["briefing"]["mean_rating"] == 4.0
    assert useful["by_target_type"]["answer"]["answered_question_rate"] == 1.0
    assert useful["by_target_type"]["answer"]["saved_time_rate"] == 0.0


# --------------------------------------------------------------------------- education


def test_education_metrics(session, factory):
    content = GeneratedContent(kind="learning", cache_key="k", content={})
    session.add(content)
    session.flush()
    reader = factory.user()
    rows = [
        ("newsren", reader.id, 3, 4, 2, 2),
        ("newsren", None, 4, 4, 1, 2),
        ("summary", None, 2, 4, 0, 2),
        ("summary", None, 3, 4, 1, 2),
    ]
    for condition, user_id, c_ok, c_total, t_ok, t_total in rows:
        session.add(
            QuizAttempt(
                content_id=content.id,
                user_id=user_id,
                condition=condition,
                comprehension_correct=c_ok,
                comprehension_total=c_total,
                transfer_correct=t_ok,
                transfer_total=t_total,
            )
        )
    session.flush()
    education = education_metrics(session)
    newsren, summary = education["by_condition"]["newsren"], education["by_condition"]["summary"]
    assert education["attempts"] == 4
    assert newsren["attempts"] == 2 and newsren["readers"] == 1
    assert newsren["comprehension_accuracy"] == 0.875 and newsren["transfer_accuracy"] == 0.75
    assert summary["comprehension_accuracy"] == 0.625 and summary["transfer_accuracy"] == 0.25
    assert education["comprehension_delta"] == 0.25
    assert education["transfer_delta"] == 0.5


# --------------------------------------------------------------------------- engagement


def test_engagement_metrics(session, factory):
    alice, bob = factory.user(), factory.user()
    story = Story(title="Lumen-4", vertical="technology")
    other_story = Story(title="Chips", vertical="industry")
    session.add_all([story, other_story])
    session.flush()
    brief = _brief(session, story_id=story.id)
    t0 = utcnow() - timedelta(days=5)
    follow = Follow(
        user_id=alice.id, question="q", target_type="story", story_id=story.id, created_at=t0, check_count=3
    )
    unvisited = Follow(user_id=bob.id, question="q", target_type="story", story_id=other_story.id, created_at=t0)
    topic = Follow(
        user_id=bob.id, question="chips", target_type="topic", active=False, created_at=t0, check_count=2
    )
    session.add_all([follow, unvisited, topic])
    session.flush()
    views = [
        (alice.id, story.id, None, t0 - timedelta(days=1)),  # before following
        (alice.id, story.id, None, t0 + timedelta(minutes=5)),  # the reload right after following
        (alice.id, story.id, None, t0 + timedelta(days=1)),
        (alice.id, story.id, None, t0 + timedelta(days=1, hours=2)),  # same day: counted once
        (alice.id, None, brief.id, t0 + timedelta(days=2)),  # a brief of the followed story
        (alice.id, other_story.id, None, t0 + timedelta(days=2)),  # not followed by alice
        (bob.id, story.id, None, t0 + timedelta(days=2)),  # bob does not follow this story
        (None, story.id, None, t0 + timedelta(days=3)),  # anonymous
    ]
    for user_id, story_id, brief_id, at in views:
        session.add(PageView(user_id=user_id, path="/x", story_id=story_id, brief_id=brief_id, created_at=at))
    for i, (feedback, read) in enumerate([("useful", True), ("useful", True), ("unwanted", True), (None, False)]):
        session.add(
            Alert(
                follow_id=follow.id,
                user_id=alice.id,
                title="a",
                reason="r",
                dedup_key=f"alert-{i}",
                feedback=feedback,
                read_at=utcnow() if read else None,
            )
        )
    session.flush()

    engagement = engagement_metrics(session)
    assert engagement["follows_total"] == 3 and engagement["follows_active"] == 2
    assert engagement["follows_active_by_target"] == {"story": 2}
    assert engagement["monitoring_checks"] == 5
    assert engagement["story_follows"] == 2 and engagement["readers_following_stories"] == 2
    assert engagement["return_visits"] == 2  # day 1 (twice) + day 2 via the brief
    assert engagement["return_visit_days"] == 2
    assert engagement["readers_returning"] == 1
    assert engagement["story_follows_with_return_visit"] == 1
    assert engagement["return_visit_rate"] == 0.5
    assert engagement["alerts_total"] == 4 and engagement["alerts_read"] == 3
    assert engagement["alerts_useful"] == 2 and engagement["alerts_unwanted"] == 1
    assert engagement["useful_alert_rate"] == 0.6667 and engagement["unwanted_alert_rate"] == 0.3333
    assert engagement["alerts_per_active_follow"] == 2.0
    # Counting the immediate reload would change the picture: with no gap it is a return visit.
    assert engagement_metrics(session, min_gap_minutes=0)["return_visits"] == 3


# --------------------------------------------------------------------------- editorial


def test_editorial_metrics(session):
    approved = _brief(session, requires_human_review=True)
    _brief(session, requires_human_review=False)  # auto-published
    _brief(session, status="in_review")
    rejected = _brief(session, status="rejected")
    for seconds in (60.0, 120.0, 600.0):
        session.add(EditorialAction(brief_id=approved.id, action="approve", duration_seconds=seconds))
    session.add_all(
        [
            EditorialAction(brief_id=approved.id, action="approve", duration_seconds=None),
            EditorialAction(brief_id=rejected.id, action="reject", duration_seconds=300.0),
            EditorialAction(brief_id=approved.id, action="submit"),
            EditorialAction(brief_id=approved.id, action="claim_verdict"),
            Correction(brief_id=approved.id, description="d", material=True),
            Correction(brief_id=approved.id, description="d", material=False),
        ]
    )
    session.flush()
    editorial = editorial_metrics(session)
    assert editorial["approvals"] == 4 and editorial["rejections"] == 1
    assert editorial["approval_rate"] == 0.8
    assert editorial["auto_published"] == 1 and editorial["awaiting_review"] == 1
    assert editorial["corrections"] == 2 and editorial["material_corrections"] == 1
    assert editorial["claim_verdicts"] == 1
    assert editorial["reviews_timed"] == 3
    assert editorial["median_review_seconds"] == 120.0
    assert editorial["mean_review_seconds"] == 260.0
    assert editorial["median_review_minutes"] == 2.0
    assert editorial["median_rejection_review_seconds"] == 300.0
    assert editorial["baseline_manual_minutes"] == 45.0
    assert editorial["estimated_minutes_saved_per_approved_brief"] == 43.0
    assert editorial["estimated_hours_saved_total"] == pytest.approx(43.0 * 4 / 60, abs=0.01)
    assert "Estimate" in editorial["time_saved_method"]
    custom = editorial_metrics(session, baseline_manual_minutes=30)
    assert custom["estimated_minutes_saved_per_approved_brief"] == 28.0


# --------------------------------------------------------------------------- cost


def test_cost_metrics(session, factory):
    def run(agent, status="ok", provider="claude", cost=0.0, tokens=(0, 0), duration=None, tools=0, retrievals=0):
        session.add(
            AgentRun(
                agent=agent,
                task="t",
                status=status,
                provider=provider,
                cost_usd=cost,
                input_tokens=tokens[0],
                output_tokens=tokens[1],
                duration_ms=duration,
                tool_calls=tools,
                retrieval_requests=retrievals,
            )
        )

    run("story_brief", cost=0.10, tokens=(1000, 500), duration=2000)
    run("story_brief", cost=0.30, tokens=(3000, 1500), duration=4000)
    run("ask_newsren", status="fallback", provider="offline", cost=0.0, duration=100, tools=3, retrievals=2)
    run("ask_newsren", cost=0.20, tokens=(2000, 1000), duration=None, tools=5, retrievals=4)
    run("follow_monitor", status="budget_exceeded", provider="offline", cost=0.02)
    run("learning", status="refused", provider="offline")
    run("story_intelligence", status="error", provider="claude")
    session.add_all(
        [
            Follow(user_id=factory.user().id, question="q", check_count=3, active=True),
            Follow(user_id=factory.user().id, question="q2", check_count=1, active=False),
        ]
    )
    session.flush()

    cost = cost_metrics(session)
    assert cost["runs"] == 7
    assert cost["total_cost_usd"] == pytest.approx(0.62)
    assert cost["avg_cost_usd_per_run"] == pytest.approx(0.62 / 7, abs=1e-6)
    assert cost["fallback_share"] == pytest.approx(1 / 7, abs=1e-4)
    assert cost["refused_share"] == pytest.approx(1 / 7, abs=1e-4)
    assert cost["budget_exceeded_share"] == pytest.approx(1 / 7, abs=1e-4)
    assert cost["error_share"] == pytest.approx(1 / 7, abs=1e-4)

    per_brief = cost["per_story_brief"]
    assert per_brief["runs"] == 2 and per_brief["avg_cost_usd"] == pytest.approx(0.2)
    assert per_brief["avg_tokens"] == 3000.0 and per_brief["avg_duration_ms"] == 3000.0

    per_question = cost["per_question"]
    assert per_question["runs"] == 2 and per_question["avg_cost_usd"] == pytest.approx(0.1)
    assert per_question["avg_duration_ms"] == 100.0  # only timed runs count toward duration
    assert per_question["avg_retrieval_requests"] == 3.0
    assert cost["by_agent"]["ask_newsren"]["fallback_share"] == 0.5
    assert cost["by_agent"]["ask_newsren"]["claude_share"] == 0.5
    assert cost["by_agent"]["ask_newsren"]["avg_tool_calls"] == 4.0

    monitor = cost["per_monitoring_check"]
    assert monitor["agent_runs"] == 1 and monitor["checks"] == 4
    assert monitor["avg_cost_usd_per_check"] == pytest.approx(0.005)
    assert monitor["active_follows"] == 1 and monitor["cost_usd_per_active_follow"] == pytest.approx(0.02)
    assert cost["by_agent"]["follow_monitor"]["budget_exceeded_share"] == 1.0


def test_all_metrics_on_seeded_rows_is_json_serialisable(session):
    brief = _brief(session)
    _claim(session, brief, verdict="accurate")
    session.add(AgentRun(agent="story_brief", task="t", status="ok", provider="offline", cost_usd=0.0))
    session.flush()
    metrics = all_metrics(session, baseline_manual_minutes=60)
    json.dumps(metrics)
    assert metrics["trust"]["claim_accuracy_rate"] == 1.0
    assert metrics["editorial"]["baseline_manual_minutes"] == 60.0
    assert metrics["cost"]["per_story_brief"]["avg_cost_usd"] == 0.0  # offline runs cost nothing
