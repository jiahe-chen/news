"""My Newsren (plan §2.1) and Follow This Story (§2.6) web pages: dashboard, briefings, follows, alerts."""

from __future__ import annotations

import re
from datetime import timedelta

import pytest
from markupsafe import escape
from sqlalchemy import func, select

from conftest import _csrf, login
from newsren.agents.monitor import check_follow
from newsren.editorial.guardrails import FINANCE_DISCLAIMER
from newsren.models import Alert, Article, Briefing, Follow, Interest, Source, Story, User, utcnow
from newsren.schemas import CitationRef
from newsren.textutil import canonical_url, content_hash, simhash_hex
from newsren.users.auth import create_user

READER_EMAIL = "reader@newsren.example"
EDITOR_EMAIL = "editor@newsren.example"
HX = {"HX-Request": "true"}
DISCLAIMER_HTML = str(escape(FINANCE_DISCLAIMER))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _fresh(session) -> None:
    """End the test session's snapshot and forget cached rows, so reads see what the app committed."""
    session.rollback()
    session.expire_all()


def _user(session, email: str = READER_EMAIL) -> User:
    _fresh(session)
    return session.scalars(select(User).where(User.email == email)).one()


def _flash(client) -> str:
    """The flash message shown on the next page (consumes it)."""
    html = client.get("/about").text
    match = re.search(r'<div class="flash flash-(\w+)" role="status">(.*?)</div>', html, re.S)
    return f"{match.group(1)}: {match.group(2).strip()}" if match else ""


def _make_alert(
    session, user: User, *, title: str = "Independent benchmark of Lumen-4 published", **kwargs
) -> Alert:
    follow = kwargs.pop("follow", None) or session.scalars(select(Follow).where(Follow.user_id == user.id)).first()
    assert follow is not None
    article = session.scalars(select(Article).where(Article.duplicate_of_id.is_(None))).first()
    citation = CitationRef(
        id="c1",
        url=article.url,
        title=article.title,
        publisher=article.source.name,
        published_at=article.published_at.isoformat() + "Z",
        retrieved_at=article.retrieved_at.isoformat() + "Z",
        article_id=article.id,
        snippet="The evaluation used 1,200 held-out problems.",
        trust_tier="independent",
        doc_type="research_preprint",
    ).model_dump()
    alert = Alert(
        follow=follow,
        user_id=user.id,
        article_id=article.id,
        story_id=article.story_id,
        event_type="independent_evaluation",
        title=title,
        reason="An independent evaluation was published; matched your criteria: independent evaluation.",
        matched_criteria=kwargs.pop("matched_criteria", ["independent evaluation", "not opinion"]),
        citations=[citation],
        dedup_key=kwargs.pop("dedup_key", f"test:{title}"),
        **kwargs,
    )
    session.add(alert)
    session.commit()
    return alert


def _new_reader(session, email: str = "fresh@example.com", password: str = "fresh-reader-pass") -> User:
    user = create_user(session, email, password, "Fresh Reader")
    session.commit()
    return user


# --------------------------------------------------------------------------
# Sign-in and CSRF
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/me", "/me/briefings", "/me/briefings/1", "/me/alerts"])
def test_pages_require_sign_in(client, path):
    resp = client.get(path, follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/login?next={path}"


def test_signed_out_post_goes_to_sign_in(client):
    token = _csrf(client)
    resp = client.post("/me/preferences", data={"csrf_token": token, "depth": "intro"}, follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login?next=/me"


def test_signed_out_follow_redirects_to_login_with_next(client, demo_session):
    token = _csrf(client)
    story = demo_session.scalars(select(Story)).first()
    resp = client.post(
        "/follows",
        data={
            "csrf_token": token,
            "question": "Tell me about material developments",
            "target_type": "story",
            "story_id": str(story.id),
            "next": f"/stories/{story.id}",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/login?next=%2Fstories%2F{story.id}"
    assert demo_session.scalars(select(Follow).where(Follow.story_id == story.id)).first() is None


def test_signed_out_follow_rejects_open_redirect(client):
    token = _csrf(client)
    resp = client.post(
        "/follows",
        data={"csrf_token": token, "question": "x", "next": "https://evil.example/"},
        follow_redirects=False,
    )
    assert resp.headers["location"] == "/login?next=%2Fme"


@pytest.mark.parametrize(
    "path,data",
    [
        ("/me/preferences", {"depth": "intro"}),
        ("/me/interests", {"kind": "topic", "value": "chips"}),
        ("/me/interests/1/delete", {}),
        ("/me/briefings/generate", {}),
        ("/follows", {"question": "Tell me about Lumen-4"}),
        ("/follows/1/delete", {}),
        ("/alerts/1/feedback", {"feedback": "useful"}),
        ("/alerts/1/read", {}),
    ],
)
def test_posts_require_csrf(reader_client, demo_session, path, data):
    before = _user(demo_session)
    interests_before = len(before.interests)
    resp = reader_client.post(path, data=data, follow_redirects=False)
    assert resp.status_code == 403
    bad = reader_client.post(path, data={**data, "csrf_token": "wrong"}, follow_redirects=False)
    assert bad.status_code == 403
    assert len(_user(demo_session).interests) == interests_before


def test_htmx_header_token_is_accepted(reader_client, demo_session):
    resp = reader_client.post(
        "/me/interests",
        data={"kind": "topic", "value": "AI literacy mandates"},
        headers={"X-CSRF-Token": reader_client.csrf},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert any(i.value == "AI literacy mandates" for i in _user(demo_session).interests)


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------


def test_dashboard_shows_reader_sections(reader_client, demo_session):
    user = _user(demo_session)
    briefing = demo_session.scalars(select(Briefing).where(Briefing.user_id == user.id)).first()
    follow = demo_session.scalars(select(Follow).where(Follow.user_id == user.id)).first()
    resp = reader_client.get("/me")
    assert resp.status_code == 200
    html = resp.text
    assert "Hello, Demo Reader" in html
    assert 'href="/static/css/account.css"' in html
    # Your briefing: latest briefing summary with a link, provenance, and "Brief me now".
    assert "Your briefing" in html
    assert f'href="/me/briefings/{briefing.id}"' in html
    assert "Drafted by" in html  # generated note
    assert 'action="/me/briefings/generate"' in html
    assert "Brief me now" in html
    # Alerts inbox: `newsren demo` runs monitoring once, so the story follow has alerts.
    assert 'id="alerts"' in html
    assert "Independent evaluation of Lumen-4" in html
    # Follows with their criteria in plain words and a stop button.
    assert "independent benchmark is published for Lumen-4" in html
    assert "Requires an independent (non-vendor) source." in html
    assert "Ignores: opinion pieces." in html
    assert follow.last_checked_at is not None
    assert f'action="/follows/{follow.id}/delete"' in html
    # Interests grouped by kind, each removable, and an add form.
    for value in ("AI models for learning mathematics", "Lumen Labs", "semiconductor capacity"):
        assert value in html
    for interest in user.interests:
        assert f'action="/me/interests/{interest.id}/delete"' in html
    assert 'action="/me/interests"' in html
    assert 'name="vertical"' in html
    # Preferences reflect the profile (student, email opt-in, when_meaningful).
    assert re.search(r'id="pref-audience-student" name="audience" value="student" checked', html)
    assert re.search(r'id="pref-email" name="email_opt_in" value="1" checked', html)
    assert re.search(r'<option value="when_meaningful" selected>', html)
    for value in ("intro", "standard", "technical"):
        assert f'name="depth" value="{value}"' in html
    # How monitoring works.
    assert "How monitoring works" in html
    assert "Only material developments that meet your criteria." in html
    assert "Opinion is excluded unless you opt in." in html
    assert "Approved sources only." in html
    assert "Automatic delivery is limited to low-risk personal briefings." in html
    # The demo reader follows a company's filings and earnings, so the dashboard carries the finance notice.
    assert "finance-notice" in html
    # My Newsren is the active nav item.
    assert 'class="is-active">My Newsren' in html


def test_dashboard_escapes_reader_text(reader_client, demo_session):
    reader_client.post(
        "/me/interests",
        data={"kind": "topic", "value": "<script>alert(1)</script>", "csrf_token": reader_client.csrf},
    )
    html = reader_client.get("/me").text
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_dashboard_shows_finance_notice_for_finance_interests(reader_client, demo_session):
    reader_client.post(
        "/me/interests",
        data={
            "kind": "sector",
            "value": "Venture funding",
            "vertical": "finance",
            "csrf_token": reader_client.csrf,
        },
    )
    html = reader_client.get("/me").text
    assert "finance-notice" in html
    assert DISCLAIMER_HTML in html


def test_new_reader_sees_empty_states(client, demo_session):
    _new_reader(demo_session)
    client.csrf = login(client, "fresh@example.com", "fresh-reader-pass")
    html = client.get("/me").text
    assert "Hello, Fresh Reader" in html
    assert "No briefing yet" in html
    assert "No alerts yet" in html
    assert "Follow a new question" in html
    assert "None yet" in html


# --------------------------------------------------------------------------
# Preferences and interests
# --------------------------------------------------------------------------


def test_preferences_update_persists(reader_client, demo_session):
    resp = reader_client.post(
        "/me/preferences",
        data={
            "audience": "educator",
            "depth": "technical",
            "briefing_frequency": "weekly",
            "csrf_token": reader_client.csrf,
        },  # email box unticked: browsers omit it
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/me#preferences"
    assert "newsren_audience=educator" in resp.headers.get("set-cookie", "")
    user = _user(demo_session)
    assert (user.audience, user.depth, user.briefing_frequency, user.email_opt_in) == (
        "educator",
        "technical",
        "weekly",
        False,
    )
    assert _flash(reader_client).startswith("success: Preferences saved.")
    html = reader_client.get("/me").text
    assert re.search(r'name="audience" value="educator" checked', html)
    assert re.search(r'name="depth" value="technical" checked', html)
    assert not re.search(r'name="email_opt_in" value="1" checked', html)

    reader_client.post(
        "/me/preferences",
        data={
            "audience": "student",
            "depth": "intro",
            "email_opt_in": "1",
            "briefing_frequency": "daily",
            "csrf_token": reader_client.csrf,
        },
    )
    user = _user(demo_session)
    assert (user.depth, user.email_opt_in, user.briefing_frequency) == ("intro", True, "daily")


def test_invalid_preferences_change_nothing(reader_client, demo_session):
    resp = reader_client.post(
        "/me/preferences",
        data={"audience": "educator", "depth": "expert", "csrf_token": reader_client.csrf},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert _flash(reader_client).startswith("error:")
    user = _user(demo_session)
    assert (user.audience, user.depth) == ("student", "standard")


def test_interest_add_and_remove(reader_client, demo_session):
    resp = reader_client.post(
        "/me/interests",
        data={
            "kind": "organization",
            "value": "Corvane Semiconductor",
            "vertical": "industry",
            "csrf_token": reader_client.csrf,
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/me#interests"
    assert "Organization added" in _flash(reader_client)
    added = next(i for i in _user(demo_session).interests if i.value == "Corvane Semiconductor")
    assert (added.kind, added.vertical) == ("organization", "industry")
    assert "Corvane Semiconductor" in reader_client.get("/me").text

    # Adding it again does not duplicate it.
    reader_client.post(
        "/me/interests",
        data={"kind": "organization", "value": "corvane semiconductor", "csrf_token": reader_client.csrf},
        follow_redirects=False,
    )
    assert "already in your interests" in _flash(reader_client)
    assert sum(i.value.lower() == "corvane semiconductor" for i in _user(demo_session).interests) == 1

    resp = reader_client.post(
        f"/me/interests/{added.id}/delete", data={"csrf_token": reader_client.csrf}, follow_redirects=False
    )
    assert resp.status_code == 303
    assert "Removed" in _flash(reader_client)
    added_id = added.id
    _fresh(demo_session)
    demo_session.expunge_all()
    assert demo_session.get(Interest, added_id) is None
    # Removing it again (or someone else's) is a 404.
    again = reader_client.post(f"/me/interests/{added_id}/delete", data={"csrf_token": reader_client.csrf})
    assert again.status_code == 404


def test_interest_validation(reader_client, demo_session):
    count = len(_user(demo_session).interests)
    for data in (
        {"kind": "hobby", "value": "chess"},
        {"kind": "topic", "value": "   "},
        {"kind": "topic", "value": "x", "vertical": "sports"},
    ):
        reader_client.post(
            "/me/interests", data={**data, "csrf_token": reader_client.csrf}, follow_redirects=False
        )
        assert _flash(reader_client).startswith("error:")
    assert len(_user(demo_session).interests) == count


def test_cannot_remove_someone_elses_interest(client, demo_session):
    reader = _user(demo_session)
    interest_id = reader.interests[0].id
    _new_reader(demo_session)
    client.csrf = login(client, "fresh@example.com", "fresh-reader-pass")
    resp = client.post(f"/me/interests/{interest_id}/delete", data={"csrf_token": client.csrf})
    assert resp.status_code == 404
    assert demo_session.get(Interest, interest_id) is not None


# --------------------------------------------------------------------------
# Follow This Story
# --------------------------------------------------------------------------


def _latest_follow(session, email: str = READER_EMAIL) -> Follow:
    user = _user(session, email)
    return session.scalars(select(Follow).where(Follow.user_id == user.id).order_by(Follow.id.desc())).first()


def test_follow_derives_criteria_from_question(reader_client, demo_session):
    resp = reader_client.post(
        "/follows",
        data={
            "question": "Tell me when Corvane files a 10-Q or reports earnings, not opinion columns",
            "target_type": "topic",
            "next": "/me",
            "csrf_token": reader_client.csrf,
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/me"
    assert _flash(reader_client).startswith("success: Following: “Tell me when Corvane files")
    follow = _latest_follow(demo_session)
    assert follow.active and follow.target_type == "topic"
    assert "filing" in follow.criteria["event_types"]
    assert "earnings" in follow.criteria["event_types"]
    assert "opinion" in follow.criteria["exclude_doc_types"]
    assert follow.last_checked_at is None  # created only; monitoring looks forward
    assert demo_session.scalars(select(Alert).where(Alert.follow_id == follow.id)).first() is None
    html = reader_client.get("/me").text
    assert "Alerts only for: filing, earnings." in html


def test_follow_with_explicit_criteria(reader_client, demo_session):
    reader_client.post(
        "/follows",
        data={
            "question": "Anything new about Lumen Tutor in schools",
            "event_types": ["adoption", "research_publication", "not-a-type"],
            "require_primary": "1",
            "custom": "only if a school district reports measured learning outcomes",
            "vertical": "education",
            "csrf_token": reader_client.csrf,
        },
    )
    follow = _latest_follow(demo_session)
    assert follow.criteria["event_types"] == ["adoption", "research_publication"]
    assert follow.criteria["require_primary"] is True
    assert follow.criteria["require_independent"] is False
    assert follow.criteria["exclude_doc_types"][0] == "opinion"
    assert follow.criteria["custom"].startswith("only if a school district")
    assert follow.vertical == "education"


def test_follow_including_opinion(reader_client, demo_session):
    reader_client.post(
        "/follows",
        data={
            "question": "Debate about AI literacy mandates",
            "event_types": ["policy"],
            "include_opinion": "1",
            "csrf_token": reader_client.csrf,
        },
    )
    follow = _latest_follow(demo_session)
    assert "opinion" not in follow.criteria["exclude_doc_types"]
    assert {"policy", "opinion", "analysis"} <= set(follow.criteria["event_types"])


def test_story_follow(reader_client, demo_session):
    story = demo_session.scalars(select(Story).order_by(Story.id)).first()
    resp = reader_client.post(
        "/follows",
        data={
            "question": f"Tell me about material developments in: {story.title}",
            "target_type": "story",
            "story_id": str(story.id),
            "vertical": story.vertical,
            "next": f"/stories/{story.id}",
            "csrf_token": reader_client.csrf,
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/stories/{story.id}"
    follow = _latest_follow(demo_session)
    assert (follow.target_type, follow.story_id, follow.vertical) == ("story", story.id, story.vertical)
    html = reader_client.get("/me").text
    assert f'href="/stories/{story.id}"' in html

    # Following the same story again reactivates rather than duplicating.
    reader_client.post(
        "/follows",
        data={
            "question": f"Tell me about material developments in: {story.title}",
            "target_type": "story",
            "story_id": str(story.id),
            "csrf_token": reader_client.csrf,
        },
    )
    user = _user(demo_session)
    assert len([f for f in user.follows if f.story_id == story.id]) == 1


def test_story_follow_validation(reader_client, demo_session):
    count = len(_user(demo_session).follows)
    missing = reader_client.post(
        "/follows",
        data={"question": "x", "target_type": "story", "story_id": "999999", "csrf_token": reader_client.csrf},
    )
    assert missing.status_code == 404
    junk = reader_client.post(
        "/follows",
        data={"question": "x", "target_type": "story", "story_id": "abc", "csrf_token": reader_client.csrf},
    )
    assert junk.status_code == 404
    reader_client.post(
        "/follows",
        data={"question": "x", "target_type": "story", "csrf_token": reader_client.csrf},
        follow_redirects=False,
    )
    assert _flash(reader_client).startswith("error:")
    for data in (
        {"question": "", "target_type": "topic"},
        {"question": "Corvane news", "target_type": "company"},
        {"question": "Chips", "target_type": "planet"},
        {"question": "Chips", "vertical": "sports"},
    ):
        resp = reader_client.post(
            "/follows", data={**data, "csrf_token": reader_client.csrf}, follow_redirects=False
        )
        assert resp.status_code == 303
        assert _flash(reader_client).startswith("error:"), data
    assert len(_user(demo_session).follows) == count


def test_company_follow(reader_client, demo_session):
    reader_client.post(
        "/follows",
        data={
            "question": "Filings and enforcement actions",
            "target_type": "company",
            "entity_name": "Brightvane Advisors",
            "vertical": "finance",
            "csrf_token": reader_client.csrf,
        },
    )
    follow = _latest_follow(demo_session)
    assert (follow.target_type, follow.entity_name) == ("company", "Brightvane Advisors")
    html = reader_client.get("/me").text
    assert "Brightvane Advisors" in html
    assert "finance-notice" in html  # a finance follow makes the dashboard a finance page


def test_stop_following(reader_client, demo_session):
    follow = _latest_follow(demo_session)
    resp = reader_client.post(
        f"/follows/{follow.id}/delete",
        data={"csrf_token": reader_client.csrf, "next": "/me#follows"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/me#follows"
    assert "Stopped following" in _flash(reader_client)
    _fresh(demo_session)
    assert demo_session.get(Follow, follow.id).active is False
    html = reader_client.get("/me").text
    assert f'action="/follows/{follow.id}/delete"' not in html


def test_stop_following_htmx_returns_fragment(reader_client, demo_session):
    follow = _latest_follow(demo_session)
    resp = reader_client.post(f"/follows/{follow.id}/delete", data={"csrf_token": reader_client.csrf}, headers=HX)
    assert resp.status_code == 200
    assert "<html" not in resp.text
    assert f'id="follow-{follow.id}"' in resp.text
    assert 'role="status"' in resp.text and "Stopped following" in resp.text


def test_cannot_stop_someone_elses_follow(client, demo_session):
    follow = _latest_follow(demo_session)
    _new_reader(demo_session)
    client.csrf = login(client, "fresh@example.com", "fresh-reader-pass")
    resp = client.post(f"/follows/{follow.id}/delete", data={"csrf_token": client.csrf})
    assert resp.status_code == 404
    assert client.post("/follows/999999/delete", data={"csrf_token": client.csrf}).status_code == 404
    _fresh(demo_session)
    assert demo_session.get(Follow, follow.id).active is True


# --------------------------------------------------------------------------
# Alerts
# --------------------------------------------------------------------------


def test_alert_in_inbox_with_provenance(reader_client, demo_session):
    alert = _make_alert(demo_session, _user(demo_session))
    html = reader_client.get("/me").text
    assert f'id="alert-{alert.id}"' in html
    assert alert.title in html
    assert "matched your criteria: independent evaluation" in html
    assert '<li class="pill">independent evaluation</li>' in html
    assert "published " in html and "retrieved " in html
    assert "badge-independent" in html and "badge-research_preprint" in html
    assert "Preprint · not peer reviewed" in html
    assert f'action="/alerts/{alert.id}/feedback"' in html
    assert f'action="/alerts/{alert.id}/read"' in html
    unread = demo_session.scalar(
        select(func.count(Alert.id)).where(Alert.user_id == alert.user_id, Alert.read_at.is_(None))
    )
    assert f"{unread} unread" in html


def test_alert_feedback_persists(reader_client, demo_session):
    alert = _make_alert(demo_session, _user(demo_session))
    resp = reader_client.post(
        f"/alerts/{alert.id}/feedback",
        data={"feedback": "useful", "next": "/me", "csrf_token": reader_client.csrf},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/me"
    assert "Marked useful" in _flash(reader_client)
    _fresh(demo_session)
    stored = demo_session.get(Alert, alert.id)
    assert stored.feedback == "useful"
    assert stored.read_at is not None
    # Read alerts leave the unread inbox but stay on the alerts page.
    assert f'id="alert-{alert.id}"' not in reader_client.get("/me").text
    page = reader_client.get("/me/alerts").text
    assert f'id="alert-{alert.id}"' in page
    assert "1 marked useful" in page


def test_alert_feedback_htmx_returns_updated_row(reader_client, demo_session):
    alert = _make_alert(demo_session, _user(demo_session))
    resp = reader_client.post(
        f"/alerts/{alert.id}/feedback", data={"feedback": "unwanted", "csrf_token": reader_client.csrf}, headers=HX
    )
    assert resp.status_code == 200
    html = resp.text
    assert "<html" not in html
    assert html.lstrip().startswith(f'<li class="me-alert" id="alert-{alert.id}">')
    assert re.search(r'value="unwanted"\s+aria-describedby="[^"]+" aria-pressed="true"', html)
    assert "Marked not useful" in html and 'role="status"' in html
    # The flash is not consumed by the fragment and none was set.
    assert _flash(reader_client) == ""
    _fresh(demo_session)
    assert demo_session.get(Alert, alert.id).feedback == "unwanted"


def test_alert_feedback_validation_and_ownership(client, demo_session):
    alert = _make_alert(demo_session, _user(demo_session))
    client.csrf = login(client, READER_EMAIL, "reader-demo-pass")
    bad = client.post(f"/alerts/{alert.id}/feedback", data={"feedback": "meh", "csrf_token": client.csrf})
    assert bad.status_code == 400
    assert (
        client.post("/alerts/999999/feedback", data={"feedback": "useful", "csrf_token": client.csrf}).status_code
        == 404
    )
    client.post("/logout", data={"csrf_token": client.csrf})
    _new_reader(demo_session)
    client.csrf = login(client, "fresh@example.com", "fresh-reader-pass")
    other = client.post(f"/alerts/{alert.id}/feedback", data={"feedback": "useful", "csrf_token": client.csrf})
    assert other.status_code == 404
    assert client.post(f"/alerts/{alert.id}/read", data={"csrf_token": client.csrf}).status_code == 404
    _fresh(demo_session)
    assert demo_session.get(Alert, alert.id).feedback is None


def test_alert_mark_read_and_unread(reader_client, demo_session):
    alert = _make_alert(demo_session, _user(demo_session))
    resp = reader_client.post(f"/alerts/{alert.id}/read", data={"csrf_token": reader_client.csrf}, headers=HX)
    assert resp.status_code == 200
    assert "Mark unread" in resp.text and "is-unread" not in resp.text
    _fresh(demo_session)
    assert demo_session.get(Alert, alert.id).read_at is not None
    resp = reader_client.post(
        f"/alerts/{alert.id}/read", data={"read": "0", "csrf_token": reader_client.csrf}, follow_redirects=False
    )
    assert resp.status_code == 303
    _fresh(demo_session)
    assert demo_session.get(Alert, alert.id).read_at is None
    assert "Marked as unread" in _flash(reader_client)


def test_alerts_page_filters(reader_client, demo_session):
    user = _user(demo_session)
    first = _make_alert(demo_session, user, title="First development")
    second = _make_alert(demo_session, user, title="Second development", read_at=utcnow())
    all_page = reader_client.get("/me/alerts").text
    assert "First development" in all_page and "Second development" in all_page
    unread = reader_client.get("/me/alerts?show=unread").text
    assert "First development" in unread and "Second development" not in unread
    assert 'aria-current="page"' in unread
    assert first.id != second.id


def test_monitoring_alert_appears_on_dashboard(reader_client, demo_session):
    """End to end: a new independent benchmark article meets the demo follow's criteria."""
    user = _user(demo_session)
    follow = demo_session.scalars(select(Follow).where(Follow.user_id == user.id)).first()
    # Pretend the follow was checked after the demo articles arrived: only the new article is a candidate.
    latest = demo_session.scalar(select(func.max(Article.retrieved_at)))
    follow.last_checked_at = latest + timedelta(seconds=1)
    source = demo_session.scalars(select(Source).where(Source.slug == "demo-circuit-ledger")).one()
    title = "Independent benchmark finds Lumen-4 model trails rivals on unfamiliar geometry problems"
    body = (
        "An independent benchmark run by the Open Evaluation Collective found that the Lumen-4 model scored "
        "71 percent on new geometry problems written after its training cutoff, behind two rival models."
    )
    url = "https://circuitledger.example/stories/lumen-4-geometry-benchmark-test"
    demo_session.add(
        Article(
            source=source,
            url=url,
            canonical_url=canonical_url(url),
            title=title,
            summary=body,
            content=body,
            published_at=utcnow(),
            retrieved_at=latest + timedelta(seconds=2),
            vertical="technology",
            doc_type="news",
            trust_tier="independent",
            content_hash=content_hash(title, body),
            simhash=simhash_hex(f"{title} {body}"),
            is_demo=True,
        )
    )
    demo_session.flush()
    # Demo monitoring already alerted on today's independent evaluation of this story; same-day alerts
    # for the same story development are deduplicated, so start from an empty inbox for this follow.
    for old in list(follow.alerts):
        demo_session.delete(old)
    # The pipeline clusters new articles before monitoring: the article joins the followed Lumen-4 story.
    from newsren.intelligence.clustering import assign_stories

    assign_stories(demo_session)
    alerts = check_follow(demo_session, follow, now=max(utcnow(), latest + timedelta(seconds=3)))
    demo_session.commit()
    assert [a.title for a in alerts] == [title]
    html = reader_client.get("/me").text
    assert title in html
    assert "The Circuit Ledger" in html
    assert "independent (non-vendor) source" in html
    assert "Demo content" in html


# --------------------------------------------------------------------------
# Briefings
# --------------------------------------------------------------------------


def test_briefing_view_for_owner(reader_client, demo_session):
    user = _user(demo_session)
    briefing = demo_session.scalars(select(Briefing).where(Briefing.user_id == user.id)).one()
    assert briefing.opened_at is None
    resp = reader_client.get(f"/me/briefings/{briefing.id}")
    assert resp.status_code == 200
    html = resp.text
    content = briefing.content
    assert content["title"].replace("&", "&amp;") in html or "Your Newsren briefing" in html
    for section in content["sections"]:
        assert section["heading"].split("…")[0][:40] in html
    item = next(i for s in content["sections"] for i in s["items"] if i.get("brief_id"))
    assert f'href="/briefs/{item["brief_id"]}"' in html
    assert "What changed" in html and "Why it matters" in html
    assert 'class="cite" href="#cite-c1"' in html  # citation refs resolve against the briefing's sources
    assert 'id="cite-c1"' in html and "retrieved" in html
    assert "Drafted by" in html  # generated note
    assert "Demo content" in html
    assert "Before you rely on this" in html
    # Feedback goes to the shared /feedback endpoint.
    assert 'action="/feedback"' in html
    assert 'name="target_type" value="briefing"' in html
    assert f'name="target_id" value="{briefing.id}"' in html
    _fresh(demo_session)
    opened = demo_session.get(Briefing, briefing.id).opened_at
    assert opened is not None
    reader_client.get(f"/me/briefings/{briefing.id}")
    _fresh(demo_session)
    assert demo_session.get(Briefing, briefing.id).opened_at == opened  # first open is kept


def test_briefing_is_private_to_its_owner(editor_client, demo_session):
    reader = _user(demo_session)
    briefing = demo_session.scalars(select(Briefing).where(Briefing.user_id == reader.id)).one()
    assert editor_client.get(f"/me/briefings/{briefing.id}").status_code == 404
    assert editor_client.get("/me/briefings/999999").status_code == 404
    assert f"/me/briefings/{briefing.id}" not in editor_client.get("/me/briefings").text
    _fresh(demo_session)
    assert demo_session.get(Briefing, briefing.id).opened_at is None


def test_briefings_list(reader_client, demo_session):
    briefing = demo_session.scalars(select(Briefing).where(Briefing.user_id == _user(demo_session).id)).one()
    html = reader_client.get("/me/briefings").text
    assert f'href="/me/briefings/{briefing.id}"' in html
    assert "New" in html  # not opened yet
    assert "Brief me now" in html


def test_finance_briefing_shows_notice(reader_client, demo_session):
    user = _user(demo_session)
    briefing = Briefing(
        user_id=user.id,
        title="Your Newsren briefing: funding rounds",
        content={
            "title": "Your Newsren briefing: funding rounds",
            "intro": "One funding round matched your interests.",
            "sections": [
                {
                    "heading": "Venture funding",
                    "items": [
                        {
                            "headline": "Quillfield raises $45 million",
                            "what_changed": "A Series B was announced.",
                            "why_it_matters": "It shows investor interest in AI feedback tools.",
                            "citation_ids": ["c1"],
                        }
                    ],
                }
            ],
            "closing_note": f"Check the linked sources. {FINANCE_DISCLAIMER}",
        },
        citations=[
            CitationRef(
                id="c1",
                url="https://seedandseries.example/q",
                title="Quillfield raises",
                publisher="Seed & Series",
            ).model_dump()
        ],
        item_count=1,
    )
    demo_session.add(briefing)
    demo_session.commit()
    html = reader_client.get(f"/me/briefings/{briefing.id}").text
    assert 'class="finance-notice"' in html
    assert html.index('class="finance-notice"') < html.index("What changed")
    assert html.count(DISCLAIMER_HTML) == 2  # notice near the top and the closing note
    # The newest briefing is finance, so the dashboard carries the notice too.
    assert 'class="finance-notice"' in reader_client.get("/me").text
    assert 'class="finance-notice"' in reader_client.get("/me/briefings").text


def test_generate_briefing_nothing_new(reader_client, demo_session):
    count = demo_session.query(Briefing).count()
    resp = reader_client.post(
        "/me/briefings/generate", data={"csrf_token": reader_client.csrf}, follow_redirects=False
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/me#briefing"
    assert "Nothing new since your last briefing" in _flash(reader_client)
    _fresh(demo_session)
    assert demo_session.query(Briefing).count() == count


def test_generate_briefing_builds_and_delivers(client, demo_session):
    user = _new_reader(demo_session)
    from newsren.users.service import add_interest

    add_interest(demo_session, user, "topic", "semiconductor capacity")
    add_interest(demo_session, user, "organization", "Lumen Labs")
    demo_session.commit()
    client.csrf = login(client, "fresh@example.com", "fresh-reader-pass")
    resp = client.post("/me/briefings/generate", data={"csrf_token": client.csrf}, follow_redirects=False)
    assert resp.status_code == 303
    _fresh(demo_session)
    briefing = demo_session.scalars(select(Briefing).where(Briefing.user_id == user.id)).one()
    assert resp.headers["location"] == f"/me/briefings/{briefing.id}"
    assert briefing.delivered_at is not None and briefing.delivery_channel == "web"
    page = client.get(resp.headers["location"])
    assert "Your briefing is ready" in page.text
    assert "Semiconductor capacity" in page.text


def test_generate_briefing_without_interests(client, demo_session):
    _new_reader(demo_session)
    client.csrf = login(client, "fresh@example.com", "fresh-reader-pass")
    resp = client.post("/me/briefings/generate", data={"csrf_token": client.csrf}, follow_redirects=False)
    assert resp.headers["location"] == "/me#interests"
    assert "Add an interest or follow a question first" in _flash(client)
