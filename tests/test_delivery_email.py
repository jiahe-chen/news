"""Email delivery: outbox .eml writing, SMTP (faked), header safety, and briefing rendering with escaping."""

from __future__ import annotations

import email
import json
import logging
import smtplib
from email.policy import default as default_policy
from pathlib import Path

import pytest

from newsren.delivery import email as delivery
from newsren.delivery.email import (
    FOOTER_TEXT,
    NO_SOURCE_TEXT,
    EmailDeliveryError,
    render_briefing_email,
    send_email,
)
from newsren.models import Briefing, User

DATA = Path(__file__).parent / "data" / "delivery" / "briefing.json"


def _briefing() -> Briefing:
    raw = json.loads(DATA.read_text())
    return Briefing(id=5, user_id=1, title=raw["title"], content=raw["content"], citations=raw["citations"])


def _reader() -> User:
    return User(id=1, email="reader@example.com", display_name="Ada <Reader>", password_hash="x")


def _parse(path: str) -> email.message.EmailMessage:
    return email.message_from_bytes(Path(path).read_bytes(), policy=default_policy)


# --------------------------------------------------------------------------- outbox


def test_send_email_writes_eml_to_outbox_when_smtp_is_not_configured(settings):
    result = send_email("Ada <ada@example.com>", "Your briefing", "<p>Hello</p>", "Hello", settings)
    path = Path(result)
    assert path.parent == settings.outbox_dir and path.suffix == ".eml" and path.exists()
    msg = _parse(result)
    assert msg["To"] == "Ada <ada@example.com>"
    assert msg["From"] == settings.smtp_from
    assert msg["Subject"] == "Your briefing"
    assert msg["Message-ID"].endswith("@newsren.example>")
    assert msg["List-Unsubscribe"] == "<http://localhost:8000/me>"
    assert msg["Auto-Submitted"] == "auto-generated"
    assert msg.get_content_type() == "multipart/alternative"
    assert msg.get_body(("plain",)).get_content().strip() == "Hello"
    assert msg.get_body(("html",)).get_content().strip() == "<p>Hello</p>"
    assert b"\r\n" in path.read_bytes()  # RFC 822 line endings


def test_send_email_uses_global_settings_by_default(settings):
    path = send_email("reader@example.com", "Hi", "<p>x</p>", "x")
    assert Path(path).parent == settings.outbox_dir


def test_send_email_blocks_header_injection(settings):
    path = send_email("ada@example.com", "Hi\r\nBcc: victim@example.com", "<p>x</p>", "x", settings)
    msg = _parse(path)
    assert msg["Bcc"] is None
    assert msg["Subject"] == "Hi Bcc: victim@example.com"
    with pytest.raises(ValueError):
        send_email("ada@example.com\r\nBcc: victim@example.com", "Hi", "x", "x", settings)
    with pytest.raises(ValueError):
        send_email("ada@example.com, eve@example.com", "Hi", "x", "x", settings)
    with pytest.raises(ValueError):
        send_email("not-an-address", "Hi", "x", "x", settings)


# --------------------------------------------------------------------------- SMTP


class FakeSMTP:
    instances: list[FakeSMTP] = []
    fail_login = False

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.calls: list[str] = []
        self.sent: list = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.calls.append("quit")
        return False

    def ehlo(self):
        self.calls.append("ehlo")

    def starttls(self, context=None):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append(f"login:{user}")
        if FakeSMTP.fail_login:
            raise smtplib.SMTPAuthenticationError(535, f"bad credentials for {user} / {password}".encode())

    def send_message(self, msg):
        self.calls.append("send")
        self.sent.append(msg)


@pytest.fixture()
def fake_smtp(monkeypatch):
    FakeSMTP.instances = []
    FakeSMTP.fail_login = False
    monkeypatch.setattr(delivery.smtplib, "SMTP", FakeSMTP)
    return FakeSMTP


def test_send_email_over_smtp_with_starttls_and_login(settings, fake_smtp, caplog):
    smtp = settings.with_overrides(
        smtp_host="smtp.example.com", smtp_user="mailer", smtp_password="hunter2-secret", smtp_port=587
    )
    with caplog.at_level(logging.DEBUG):
        message_id = send_email("ada@example.com", "Subject", "<p>x</p>", "x", smtp)
    client = fake_smtp.instances[0]
    assert (client.host, client.port) == ("smtp.example.com", 587)
    assert client.calls == ["ehlo", "starttls", "ehlo", "login:mailer", "send", "quit"]
    assert client.sent[0]["Message-ID"] == message_id
    assert "hunter2-secret" not in caplog.text
    assert not settings.outbox_dir.exists()


def test_send_email_smtp_without_tls_or_login(settings, fake_smtp):
    smtp = settings.with_overrides(smtp_host="relay.internal", smtp_port=25, smtp_starttls=False)
    send_email("ada@example.com", "Subject", "<p>x</p>", "x", smtp)
    assert fake_smtp.instances[0].calls == ["ehlo", "send", "quit"]


def test_smtp_failure_never_leaks_the_password(settings, fake_smtp, caplog):
    fake_smtp.fail_login = True
    smtp = settings.with_overrides(
        smtp_host="smtp.example.com", smtp_user="mailer", smtp_password="hunter2-secret"
    )
    with caplog.at_level(logging.DEBUG), pytest.raises(EmailDeliveryError) as excinfo:
        send_email("ada@example.com", "Subject", "<p>x</p>", "x", smtp)
    assert "hunter2-secret" not in str(excinfo.value)
    assert "***" in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert "hunter2-secret" not in caplog.text


# --------------------------------------------------------------------------- rendering


def test_render_briefing_email_structure_and_sources(settings):
    subject, html, text = render_briefing_email(_briefing(), _reader())
    assert subject == "Your briefing: an independent test of Lumen-4 and new AI-tutoring guidance | The Newsren"
    assert "<h2" in html and "AI models for learning mathematics" in html
    assert "Independent test finds Lumen-4 solves 61% of held-out problems" in html
    assert "<strong>What changed:</strong>" in html and "<strong>Why it matters:</strong>" in html
    assert "Matches your interest: AI models for learning mathematics" in html
    # Sources are resolved from the briefing's citations and numbered in order of first use.
    assert 'href="https://evals.example/reports/lumen-4-math"' in html
    assert "[1] Independent evaluation of Lumen-4" in html
    assert "[2] Lumen Labs launches Lumen-4" in html
    assert "Open Eval Lab (demo), published 24 Oct 2026, retrieved 24 Oct 2026" in html
    assert 'href="http://localhost:8000/briefs/12"' in html
    assert NO_SOURCE_TEXT in html  # the policy item cites nothing, so it is flagged as unverified
    assert "c99" not in html  # unknown citation ids are dropped, never invented
    assert 'href="http://localhost:8000/me"' in html and "Manage preferences or unsubscribe" in html
    assert "automated personal briefing generated by The Newsren from the sources cited" in html
    assert "not investment advice" in html


def test_render_briefing_email_escapes_everything(settings):
    _, html, text = render_briefing_email(_briefing(), _reader())
    assert "<script>" not in html
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt; &amp; &quot;quotes&quot;" in html
    assert "Hi Ada &lt;Reader&gt;," in html
    assert "<b>link</b>" not in html and "Unsafe &lt;b&gt;link&lt;/b&gt; source (link unavailable)" in html
    assert 'href="javascript:' not in html and "document.cookie" not in html
    assert "Example &amp; Co" in html
    # Plain text is not HTML: it keeps the characters as written.
    assert "Tutoring <script>alert('x')</script>" in text
    assert "javascript:alert" not in text


def test_render_briefing_email_plain_text(settings):
    _, _, text = render_briefing_email(_briefing(), _reader())
    assert text.startswith("THE NEWSREN - MY NEWSREN BRIEFING")
    assert "Hi Ada <Reader>," in text
    assert "AI MODELS FOR LEARNING MATHEMATICS" in text
    assert "  Sources: [1] [2]" in text
    assert "[1] Independent evaluation of Lumen-4 on 1,200 held-out math problems -" in text
    assert "* Independent test finds Lumen-4 solves 61% of held-out problems, below the\n  company's" in text
    assert "    https://evals.example/reports/lumen-4-math" in text
    assert "  Full brief: http://localhost:8000/briefs/12" in text
    assert "Manage preferences or unsubscribe: http://localhost:8000/me" in text
    assert "automated personal briefing generated by The Newsren" in text
    assert all(len(line) <= 78 for line in text.splitlines() if not line.strip().startswith("http"))


def test_render_uses_configured_base_url_and_tolerates_empty_content(settings):
    custom = settings.with_overrides(base_url="https://news.example/")
    briefing = Briefing(user_id=1, title="Quiet week", content={}, citations=[])
    subject, html, text = render_briefing_email(briefing, _reader(), custom)
    assert subject == "Quiet week | The Newsren"
    assert 'href="https://news.example/me"' in html
    assert "Sources</h2>" not in html
    assert FOOTER_TEXT in " ".join(text.split())


def test_rendered_briefing_round_trips_through_the_outbox(settings):
    subject, html, text = render_briefing_email(_briefing(), _reader())
    msg = _parse(send_email("reader@example.com", subject, html, text, settings))
    assert msg["Subject"] == subject
    html_part = msg.get_body(("html",)).get_content()
    assert "&lt;script&gt;" in html_part and "<script>" not in html_part
    assert "https://evals.example/reports/lumen-4-math" in msg.get_body(("plain",)).get_content()
