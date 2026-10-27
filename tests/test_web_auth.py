"""Sign-in, registration, CSRF, and the audience switch."""

from __future__ import annotations

from conftest import DEMO_READER_PASSWORD, _csrf


def test_login_requires_csrf(client):
    resp = client.post("/login", data={"email": "reader@newsren.example", "password": DEMO_READER_PASSWORD})
    assert resp.status_code == 403


def test_login_success_and_bad_password(client):
    token = _csrf(client)
    bad = client.post(
        "/login", data={"email": "reader@newsren.example", "password": "nope-nope", "csrf_token": token}
    )
    assert bad.status_code == 400
    assert "match an account" in bad.text
    ok = client.post(
        "/login",
        data={"email": "reader@newsren.example", "password": DEMO_READER_PASSWORD, "csrf_token": token},
        follow_redirects=False,
    )
    assert ok.status_code == 303


def test_login_rejects_open_redirect(client):
    token = _csrf(client)
    resp = client.post(
        "/login",
        data={
            "email": "reader@newsren.example",
            "password": DEMO_READER_PASSWORD,
            "csrf_token": token,
            "next": "https://evil.example/",
        },
        follow_redirects=False,
    )
    assert resp.headers["location"] == "/me"


def test_register_and_duplicate(client):
    token = _csrf(client)
    data = {"email": "new@example.com", "password": "long-enough-pw", "display_name": "New", "csrf_token": token}
    first = client.post("/register", data=data, follow_redirects=False)
    assert first.status_code == 303
    client.cookies.clear()
    token = _csrf(client)
    again = client.post("/register", data={**data, "csrf_token": token})
    assert again.status_code == 400


def test_audience_switch_sets_cookie(client):
    token = _csrf(client)
    resp = client.post(
        "/audience", data={"audience": "student", "next": "/about", "csrf_token": token}, follow_redirects=False
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/about"
    assert "newsren_audience=student" in resp.headers.get("set-cookie", "")
