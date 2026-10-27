"""Personalization: preferences, interests, Follow This Story criteria derivation, alerts, and feedback."""

from __future__ import annotations

from datetime import timedelta

import pytest

from newsren.models import Alert, Feedback, Follow, Interest, Story, utcnow
from newsren.schemas import FollowCriteria
from newsren.users.service import (
    PEER_REVIEW_NOTE,
    PUBLISHED_STUDY_NOTE,
    add_interest,
    create_follow,
    deactivate_follow,
    derive_criteria,
    derive_keywords,
    describe_criteria,
    mark_alert,
    record_feedback,
    remove_interest,
    set_preferences,
    unread_alert_count,
    user_alerts,
    user_follows,
)

PLAN_EXAMPLE = (
    "Tell me when an independent benchmark is published for this model, "
    "not whenever someone writes another opinion article about it."
)


def _story(session, **kwargs) -> Story:
    story = Story(
        title=kwargs.pop("title", "Lumen Labs releases Lumen-4 math tutor"),
        vertical=kwargs.pop("vertical", "technology"),
        keywords=kwargs.pop("keywords", ["lumen-4", "lumen labs", "math tutoring"]),
        **kwargs,
    )
    session.add(story)
    session.flush()
    return story


def _alert(session, follow: Follow, title: str, minutes_ago: int, **kwargs) -> Alert:
    alert = Alert(
        follow_id=follow.id,
        user_id=follow.user_id,
        title=title,
        reason="Matched: independent evaluation",
        dedup_key=f"k-{title}",
        created_at=utcnow() - timedelta(minutes=minutes_ago),
        **kwargs,
    )
    session.add(alert)
    session.flush()
    return alert


# --------------------------------------------------------------------------- preferences & interests


def test_set_preferences_updates_only_given_fields(session, factory):
    user = factory.user()
    set_preferences(session, user, audience="Student", email_opt_in=True)
    assert (user.audience, user.depth, user.email_opt_in, user.briefing_frequency) == (
        "student",
        "standard",
        True,
        "when_meaningful",
    )
    set_preferences(session, user, depth="technical", briefing_frequency="weekly", email_opt_in="false")
    assert (user.audience, user.depth, user.email_opt_in, user.briefing_frequency) == (
        "student",
        "technical",
        False,
        "weekly",
    )


@pytest.mark.parametrize("kwargs", [{"audience": "parent"}, {"depth": "deep"}, {"briefing_frequency": "hourly"}])
def test_set_preferences_validates(session, factory, kwargs):
    with pytest.raises(ValueError):
        set_preferences(session, factory.user(), **kwargs)


def test_add_interest_dedupes_and_validates(session, factory):
    user = factory.user()
    first = add_interest(session, user, "topic", "  AI models for   learning mathematics ")
    again = add_interest(session, user, "Topic", "ai models for learning mathematics", vertical="education")
    assert again is first
    assert first.value == "AI models for learning mathematics"
    assert first.vertical == "education"
    org = add_interest(session, user, "organization", "Lumen Labs", vertical="technology")
    assert {i.kind for i in user.interests} == {"topic", "organization"}
    assert org.id != first.id
    with pytest.raises(ValueError):
        add_interest(session, user, "hobby", "chess")
    with pytest.raises(ValueError):
        add_interest(session, user, "topic", "   ")
    with pytest.raises(ValueError):
        add_interest(session, user, "topic", "x", vertical="sports")


def test_remove_interest_only_removes_own(session, factory):
    alice, bob = factory.user(), factory.user()
    interest = add_interest(session, alice, "sector", "semiconductors")
    assert remove_interest(session, bob, interest.id) is False
    assert session.get(Interest, interest.id) is not None
    assert remove_interest(session, alice, interest.id) is True
    assert session.get(Interest, interest.id) is None
    assert alice.interests == []
    assert remove_interest(session, alice, 999_999) is False


# --------------------------------------------------------------------------- criteria derivation


def test_plan_example_derives_independent_benchmark_criteria():
    criteria = derive_criteria(PLAN_EXAMPLE)
    assert criteria.event_types == ["independent_evaluation", "benchmark"]
    assert criteria.require_independent is True
    assert criteria.exclude_doc_types == ["opinion"]
    assert "opinion" not in criteria.event_types
    assert criteria.require_primary is False
    assert criteria.custom == ""


def test_plan_example_keywords_skip_filler_and_negated_words():
    keywords = derive_keywords(PLAN_EXAMPLE)
    assert keywords[0] == "independent benchmark"
    assert "model" in keywords
    for filler in ("tell", "whenever", "someone", "opinion", "article", "writes", "benchmark model"):
        assert filler not in keywords


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Notify me when Acme Robotics files its 10-Q", {"filing"}),
        ("Alert me to any new 8-K from Harbor Bank", {"filing"}),
        ("Tell me about SEC filings by chip suppliers", {"filing"}),
        ("Let me know when Harbor Bank reports earnings", {"earnings"}),
        ("Tell me when an edtech startup raises a Series B", {"funding"}),
        ("Follow funding for AI tutoring companies", {"funding"}),
        ("Notify me of SEC enforcement actions about AI washing", {"enforcement"}),
        ("Tell me if FINRA issues a fine or penalty against a broker using AI", {"enforcement"}),
        ("Track the new rule on AI disclosures", {"regulation", "policy"}),
        ("Tell me when state policy on AI in schools changes", {"regulation", "policy"}),
        ("Tell me if the newspaper issues a correction", {"correction"}),
        ("Tell me when Lumen Labs launches a new model", {"product_launch"}),
        ("Tell me when school districts adopt AI tutors", {"adoption"}),
        ("Tell me when a published study on AI tutoring appears", {"research_publication"}),
        ("Track new AI models that could change how high school students learn mathematics", set()),
    ],
)
def test_phrase_table(question, expected):
    assert set(derive_criteria(question).event_types) == expected


def test_peer_reviewed_adds_research_type_custom_note_and_drops_preprints():
    criteria = derive_criteria("Tell me when a peer-reviewed study of AI tutoring outcomes is published")
    assert criteria.event_types == ["research_publication"]
    assert criteria.custom == PEER_REVIEW_NOTE
    assert "research_preprint" in criteria.exclude_doc_types
    assert "opinion" in criteria.exclude_doc_types


def test_published_study_custom_note():
    criteria = derive_criteria("Notify me when the published study behind the press release is available")
    assert criteria.event_types == ["research_publication"]
    assert criteria.custom == PUBLISHED_STUDY_NOTE
    assert "research_preprint" not in criteria.exclude_doc_types


def test_negated_phrases_never_add_event_types():
    criteria = derive_criteria("Tell me about Harbor Bank earnings, but not funding rumors or opinion columns")
    assert criteria.event_types == ["earnings"]
    assert criteria.exclude_doc_types == ["opinion"]
    assert derive_keywords("Tell me about Harbor Bank earnings, but not funding rumors") == [
        "harbor bank earnings"
    ]


def test_negated_vendor_claims_and_blogs_tighten_criteria():
    criteria = derive_criteria("Tell me when Lumen-4 is evaluated, not vendor claims or blog posts")
    assert criteria.require_independent is True
    assert "blog" in criteria.exclude_doc_types
    assert criteria.event_types == ["benchmark", "independent_evaluation"]
    third = derive_criteria("Tell me when a third-party evaluation of Lumen-4 appears")
    assert third.event_types == ["independent_evaluation", "benchmark"]
    assert third.require_independent is True


def test_false_friends_do_not_trigger_rules():
    assert derive_criteria("Tell me about fine-tuning costs for small models").event_types == []
    journal = derive_criteria("Tell me when the Wall Street Journal covers Lumen Labs")
    assert journal.custom == "" and "research_preprint" not in journal.exclude_doc_types
    assert derive_criteria("Track the press release cadence of Lumen Labs").event_types == []


def test_explicit_request_for_opinion_keeps_commentary():
    criteria = derive_criteria("Follow opinion columns about AI regulation")
    assert "opinion" not in criteria.exclude_doc_types
    assert {"opinion", "analysis", "regulation"} <= set(criteria.event_types)


def test_primary_source_requirement():
    assert derive_criteria("Tell me when there is a primary source for the capacity numbers").require_primary


def test_describe_criteria_lines():
    lines = describe_criteria(derive_criteria(PLAN_EXAMPLE))
    assert lines[0] == "Alerts only for: independent evaluation, benchmark."
    assert "Requires an independent (non-vendor) source." in lines
    assert "Ignores: opinion pieces." in lines
    assert describe_criteria({})[0] == "Alerts for any material development."


# --------------------------------------------------------------------------- follows


def test_create_follow_plan_example(session, factory):
    user = factory.user()
    follow = create_follow(session, user, f"  {PLAN_EXAMPLE}  ")
    assert follow.id is not None and follow.active
    assert follow.question == PLAN_EXAMPLE
    assert follow.target_type == "topic"
    assert (
        follow.criteria
        == FollowCriteria(
            event_types=["independent_evaluation", "benchmark"], require_independent=True
        ).model_dump()
    )
    assert follow.keywords == ["independent benchmark", "model"]
    assert user_follows(session, user) == [follow]


def test_create_story_follow_adds_story_keywords_and_vertical(session, factory):
    user = factory.user()
    story = _story(session)
    follow = create_follow(session, user, PLAN_EXAMPLE, story=story)
    assert follow.target_type == "story"
    assert follow.story_id == story.id
    assert follow.vertical == "technology"
    assert follow.keywords[:2] == ["independent benchmark", "model"]
    assert {"lumen-4", "lumen labs", "math tutoring"} <= set(follow.keywords)


def test_create_follow_for_company_and_explicit_criteria_override(session, factory):
    user = factory.user()
    explicit = FollowCriteria(event_types=["earnings"], exclude_doc_types=[], require_primary=True)
    follow = create_follow(
        session,
        user,
        "Tell me when an independent benchmark is published",
        entity_name=" Harbor  Bank ",
        vertical="finance",
        criteria=explicit,
    )
    assert follow.target_type == "company"
    assert follow.entity_name == "Harbor Bank"
    assert follow.keywords[0] == "harbor bank"
    assert follow.criteria == explicit.model_dump()
    assert follow.vertical == "finance"


def test_create_follow_validation(session, factory):
    user = factory.user()
    with pytest.raises(ValueError):
        create_follow(session, user, "   ")
    with pytest.raises(ValueError):
        create_follow(session, user, "Track chips", target_type="planet")
    with pytest.raises(ValueError):
        create_follow(session, user, "Track chips", target_type="story")
    with pytest.raises(ValueError):
        create_follow(session, user, "Track chips", vertical="sports")


def test_following_the_same_question_again_reuses_the_follow(session, factory):
    user = factory.user()
    first = create_follow(session, user, "Track semiconductor capacity announcements")
    deactivate_follow(session, first)
    assert not first.active
    again = create_follow(session, user, "track semiconductor capacity announcements")
    assert again is first and again.active
    assert len(user.follows) == 1


def test_deactivate_follow_both_call_styles(session, factory):
    alice, bob = factory.user(), factory.user()
    follow = create_follow(session, alice, "Track Lumen Labs funding")
    assert deactivate_follow(session, bob, follow.id) is None
    assert follow.active
    assert deactivate_follow(session, alice, follow.id) is follow
    assert not follow.active
    assert user_follows(session, alice) == []
    assert user_follows(session, alice, active_only=False) == [follow]
    assert deactivate_follow(session, alice, 999_999) is None


# --------------------------------------------------------------------------- alerts


def test_user_alerts_newest_first_and_unread_filter(session, factory):
    user, other = factory.user(), factory.user()
    follow = create_follow(session, user, PLAN_EXAMPLE)
    other_follow = create_follow(session, other, PLAN_EXAMPLE)
    old = _alert(session, follow, "old", minutes_ago=120)
    new = _alert(session, follow, "new", minutes_ago=5)
    read = _alert(session, follow, "read", minutes_ago=60, read_at=utcnow())
    _alert(session, other_follow, "someone else's", minutes_ago=1)
    assert user_alerts(session, user) == [new, read, old]
    assert user_alerts(session, user, unread_only=True) == [new, old]
    assert user_alerts(session, user, limit=1) == [new]
    assert unread_alert_count(session, user) == 2


def test_mark_alert_feedback_and_read_state(session, factory):
    user = factory.user()
    follow = create_follow(session, user, PLAN_EXAMPLE)
    alert = _alert(session, follow, "benchmark", minutes_ago=10)
    mark_alert(session, alert, feedback="Useful")
    assert alert.feedback == "useful"
    first_read = alert.read_at
    assert first_read is not None
    mark_alert(session, alert)  # reading again keeps the first read time and the feedback
    assert alert.read_at == first_read and alert.feedback == "useful"
    mark_alert(session, alert, feedback="  ")  # an empty form field leaves feedback alone
    assert alert.feedback == "useful"
    mark_alert(session, alert, feedback="unwanted", read=False)
    assert alert.feedback == "unwanted" and alert.read_at is None
    with pytest.raises(ValueError):
        mark_alert(session, alert, feedback="meh")


# --------------------------------------------------------------------------- feedback


def test_record_feedback_upserts_per_reader_and_validates(session, factory):
    user = factory.user()
    row = record_feedback(session, user, "briefing", 3, answered_question=True, rating=4)
    again = record_feedback(session, user, "briefing", 3, saved_time=False, comment="  Helpful  ")
    assert again is row
    assert (row.answered_question, row.saved_time, row.rating, row.comment) == (True, False, 4, "Helpful")
    anon_a = record_feedback(session, None, "answer", 1, answered_question=False)
    anon_b = record_feedback(session, None, "answer", 1, answered_question=True)
    assert anon_a.id != anon_b.id
    assert session.query(Feedback).count() == 3
    with pytest.raises(ValueError):
        record_feedback(session, user, "briefing", 3, rating=6)
    with pytest.raises(ValueError):
        record_feedback(session, user, "homepage", 1)
