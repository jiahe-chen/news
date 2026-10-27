"""Analytics page (plan §9 success metrics) for editors."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from newsren.metrics import all_metrics
from newsren.models import Brief, BriefClaim, BriefStatus, Feedback, QuizAttempt, User
from newsren.web.routes.analytics import build_sections

SECTIONS = ["Trust", "Usefulness", "Education", "Engagement", "Editorial efficiency", "Cost"]


def _fresh(session) -> None:
    session.rollback()
    session.expire_all()


def test_analytics_redirects_anonymous(client):
    resp = client.get("/analytics", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login?next=/analytics"


def test_analytics_forbidden_for_readers(reader_client):
    assert reader_client.get("/analytics").status_code == 403


def test_analytics_renders_every_plan_section(editor_client):
    resp = editor_client.get("/analytics")
    assert resp.status_code == 200
    html = resp.text
    for title in SECTIONS:
        assert f">{title}</h2>" in html
    for key in ("trust", "usefulness", "education", "engagement", "editorial", "cost"):
        assert f'id="{key}"' in html
    # Plan §9: output volume and chatbot usage are deliberately not primary metrics.
    assert "Deliberately not primary metrics" in html
    assert "total article output and chatbot usage" in html
    # Plain-language definitions, sample sizes and the cost table.
    assert "Measures</span>" in html
    assert '<abbr title="sample size">n</abbr>' in html
    assert "Cost by agent" in html and "story_brief" in html
    assert 'aria-label="Editor desk"' in html


def test_analytics_shows_no_data_states(editor_client):
    html = editor_client.get("/analytics").text
    # The demo has no sampled claims, reader ratings or quiz attempts yet.
    assert "No claims sampled yet" in html
    assert "No data yet" in html
    assert "Needs both conditions" in html
    assert "an-empty" in html


def test_analytics_reflects_editor_and_reader_activity(editor_client, demo_session):
    _fresh(demo_session)
    brief = demo_session.scalars(select(Brief).where(Brief.status == BriefStatus.IN_REVIEW)).first()
    claim_id = brief.claims[0].id
    editor_client.post(
        f"/editor/claims/{claim_id}/verdict",
        data={"verdict": "accurate", "csrf_token": editor_client.csrf},
        follow_redirects=False,
    )
    _fresh(demo_session)
    assert demo_session.get(BriefClaim, claim_id).editor_verdict == "accurate"
    reader = demo_session.scalars(select(User).where(User.email == "reader@newsren.example")).one()
    demo_session.add_all(
        [
            Feedback(
                user_id=reader.id,
                target_type="answer",
                target_id=1,
                answered_question=True,
                saved_time=True,
                rating=5,
            ),
            Feedback(user_id=reader.id, target_type="answer", target_id=2, answered_question=False, rating=3),
            QuizAttempt(
                user_id=reader.id,
                content_id=1,
                condition="newsren",
                comprehension_correct=3,
                comprehension_total=4,
                transfer_correct=1,
                transfer_total=2,
            ),
        ]
    )
    demo_session.commit()
    html = editor_client.get("/analytics").text
    assert "n</abbr> = 1 claims sampled by editors" in html
    assert "100.0%" in html
    assert "50.0%" in html  # answered-question rate over 2 responses
    assert "4.00 / 5" in html
    assert "Ask Newsren answers" in html
    assert "75.0%" in html  # comprehension accuracy, Newsren condition


def test_build_sections_on_empty_database(session):
    sections = build_sections(all_metrics(session))
    assert [s["title"] for s in sections] == SECTIONS
    for section in sections:
        assert section["definition"] and section["plan"]
        assert section["tiles"]
    trust = sections[0]
    assert trust["tiles"][0]["value"] is None
    assert trust["empty_hint"]
    cost = sections[-1]
    assert cost["table"] is None


@pytest.mark.parametrize("value,expected", [(None, None), (0.5, "50.0%"), (1, "100.0%")])
def test_rate_formatting(value, expected):
    from newsren.web.routes.analytics import fmt_rate

    assert fmt_rate(value) == expected


def test_usd_formatting():
    from newsren.web.routes.analytics import fmt_usd

    assert fmt_usd(None) is None
    assert fmt_usd(0) == "$0.00"
    assert fmt_usd(0.01234) == "$0.0123"
    assert fmt_usd(12.5) == "$12.50"
