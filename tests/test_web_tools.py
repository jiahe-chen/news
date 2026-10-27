"""Brief intelligence tools, Ask Newsren pages, feedback, and quizzes (web layer)."""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from conftest import _csrf
from newsren import db
from newsren.agents.products import available_products
from newsren.models import Brief, BriefStatus, Feedback, GeneratedContent, Question, QuizAttempt, User
from newsren.schemas import LearningOut
from newsren.users.service import create_follow
from newsren.web.routes.tools import cite_markers, label_badges, story_intel_visible, text_blocks

HX = {"HX-Request": "true"}
TOOLS = ["connected", "learn", "evidence", "impact", "follow", "ask"]
NOTICE = "Not investment advice."


@pytest.fixture()
def dbs(demo_settings) -> Iterator[Session]:
    """A separate session for arranging data and checking what the app stored."""
    s = db.new_session()
    try:
        yield s
    finally:
        s.rollback()
        s.close()


def _brief(
    s: Session, *, status: str = BriefStatus.APPROVED, vertical: str | None = None, contains: str = ""
) -> Brief:
    stmt = select(Brief).where(Brief.status == status)
    if vertical:
        stmt = stmt.where(Brief.vertical == vertical)
    if contains:
        stmt = stmt.where(Brief.headline.contains(contains))
    brief = s.scalars(stmt.order_by(Brief.id)).first()
    assert brief is not None, f"demo data has no {status} {vertical or ''} brief containing {contains!r}"
    return brief


def _lumen(s: Session) -> Brief:
    return _brief(s, vertical="technology", contains="Independent evaluation of Lumen-4")


def _finance_draft(s: Session) -> Brief:
    return _brief(s, status=BriefStatus.IN_REVIEW, vertical="finance", contains="Northbridge")


def _token(client) -> str:
    return getattr(client, "csrf", None) or _csrf(client)


# --------------------------------------------------------------------------
# Template helpers
# --------------------------------------------------------------------------


def test_cite_markers_escapes_text_and_links_known_ids():
    out = cite_markers(
        '<script>alert("x")</script> Lumen-4 scored 84.3% [c1] but see [c9].',
        [{"id": "c1", "title": 'Eval <b>"quoted"</b>', "publisher": "Open Eval"}],
        "evidence-",
    )
    html = str(out)
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert '<a class="cite" href="#evidence-cite-c1"' in html
    assert "[1]</a>" in html
    assert "&lt;b&gt;" in html and "<b>" not in html  # titles are escaped inside the attribute
    assert "[c9]" not in html and "c9" not in html  # unknown ids are dropped, not shown broken
    assert str(cite_markers("", [])) == "" and str(cite_markers(None, None)) == ""


def test_cite_markers_without_citations_drops_markers():
    assert str(cite_markers("A fact [c1][c2].", None)) == "A fact."


def test_text_blocks_headings_paragraphs_lists():
    blocks = text_blocks(
        "What happened\n\nFirst sentence [c1]. Second.\n\n- one [c2].\n- two.\n\nWhy it matters\n\nBecause."
    )
    assert [b["kind"] for b in blocks] == ["heading", "paragraph", "list", "heading", "paragraph"]
    assert blocks[2]["entries"] == ["one [c2].", "two."]
    assert text_blocks("Lead line\n- a\n- b")[0] == {"kind": "paragraph", "text": "Lead line"}
    assert text_blocks(None) == []


def test_label_badges_split_and_dedupe():
    html = str(label_badges("backtest, simulated, backtest; independently tested"))
    assert html.count("badge-backtest") == 1 and "badge-simulated" in html
    assert "badge-independently_tested" in html and "Independently tested" in html
    assert str(label_badges("")) == ""


def test_story_intel_visible_rules(dbs):
    draft = _finance_draft(dbs)
    story = draft.story
    assert story is not None
    has_approved = any(b.status == BriefStatus.APPROVED for b in story.briefs)
    editor = dbs.scalars(select(User).where(User.role == "editor")).first()
    reader = dbs.scalars(select(User).where(User.role == "reader")).first()
    assert story_intel_visible(story, editor) is True
    assert story_intel_visible(story, reader) is has_approved
    assert story_intel_visible(story, None) is has_approved
    assert story_intel_visible(_lumen(dbs).story, None) is True
    assert story_intel_visible(None, editor) is False


# --------------------------------------------------------------------------
# Partials vs full pages, visibility
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tool", TOOLS)
def test_tool_partial_and_full_page(client, dbs, tool):
    brief = _lumen(dbs)
    url = f"/briefs/{brief.id}/tools/{tool}"
    full = client.get(url)
    assert full.status_code == 200, full.text[:500]
    assert "<html" in full.text and "data-tool-root" in full.text
    assert "Back to the brief" in full.text and f'href="/briefs/{brief.id}"' in full.text
    assert brief.headline in full.text.replace("&#39;", "'")
    assert 'aria-current="page"' in full.text  # the active tool in the tool nav
    assert "/static/css/tools.css" in full.text
    part = client.get(url, headers=HX)
    assert part.status_code == 200
    assert "<html" not in part.text and "<body" not in part.text
    assert "data-tool-root" in part.text
    assert "Back to the brief" not in part.text
    assert "HX-Request" in part.headers.get("vary", "")


def test_history_restore_gets_full_page(client, dbs):
    brief = _lumen(dbs)
    resp = client.get(f"/briefs/{brief.id}/tools/evidence", headers={**HX, "HX-History-Restore-Request": "true"})
    assert "<html" in resp.text


def test_missing_brief_404(client):
    for tool in TOOLS:
        assert client.get(f"/briefs/999999/tools/{tool}").status_code == 404
    assert client.get("/briefs/999999/tools/products/launch_reality_check").status_code == 404


def test_unpublished_brief_hidden_from_readers(client, dbs):
    draft = _finance_draft(dbs)
    for tool in TOOLS:
        assert client.get(f"/briefs/{draft.id}/tools/{tool}").status_code == 404
        assert client.get(f"/briefs/{draft.id}/tools/{tool}", headers=HX).status_code == 404


def test_unpublished_brief_hidden_from_signed_in_reader(reader_client, dbs):
    draft = _finance_draft(dbs)
    assert reader_client.get(f"/briefs/{draft.id}/tools/connected").status_code == 404


def test_editor_previews_unpublished_brief_with_banner_and_notice(editor_client, dbs):
    draft = _finance_draft(dbs)
    for tool in ("connected", "evidence", "learn"):
        full = editor_client.get(f"/briefs/{draft.id}/tools/{tool}")
        assert full.status_code == 200
        assert "Unpublished draft — in editorial review" in full.text
        assert NOTICE in full.text
        part = editor_client.get(f"/briefs/{draft.id}/tools/{tool}", headers=HX)
        assert "Unpublished draft — in editorial review" in part.text
        assert NOTICE in part.text


# --------------------------------------------------------------------------
# Connected stories
# --------------------------------------------------------------------------


def test_connected_storyline_crosses_verticals(client, dbs):
    brief = _lumen(dbs)
    html = client.get(f"/briefs/{brief.id}/tools/connected", headers=HX).text
    assert 'class="timeline storyline"' in html
    assert "Story intelligence" in html and "What happened first" in html and "What's new" in html
    assert "What remains disputed" in html and "How the evidence changed" in html
    assert 'href="#connected-cite-c1"' in html and 'id="connected-cite-c1"' in html
    assert "[c1]" not in html  # markers resolved to links
    assert "badge-product_launch" in html and "badge-independent_evaluation" in html
    assert "is-correction" in html  # the published correction is highlighted
    assert f'href="/stories/{brief.story_id}"' in html
    assert "Connected story:" in html and "vertical-tag-education" in html  # cross-vertical storyline
    assert "Technology Evolution Timeline" in html
    assert 'rel="noopener noreferrer"' in html and "https://" in html  # events link to their sources
    dates = re.findall(r'<time datetime="([^"]+)"', html)
    assert dates == sorted(dates) and len(dates) >= 5


# --------------------------------------------------------------------------
# Learn this topic + quiz
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("audience", "expected"),
    [
        ("student", ["Explained for students", "Glossary"]),
        ("educator", ["Classroom guide", "Classroom activities", "Discussion questions"]),
        ("professional", ["Professional briefing", "Implications for professionals", "Advanced reading"]),
    ],
)
def test_learn_for_each_audience(client, dbs, audience, expected):
    brief = _lumen(dbs)
    resp = client.get(f"/briefs/{brief.id}/tools/learn?audience={audience}", headers=HX)
    assert resp.status_code == 200
    html = resp.text
    for text in expected:
        assert text in html
    assert f'href="/briefs/{brief.id}/tools/learn?audience={audience}"' in html
    assert re.search(rf'learn\?audience={audience}"[^>]*aria-current="true"', html)
    assert "data-quiz-question" in html and "data-quiz-option" in html and "data-quiz-result" in html
    assert re.search(r'data-answer="\d"', html)
    assert "badge-quiz-transfer" in html and "badge-quiz-comprehension" in html
    assert 'class="glossary' in html
    assert 'href="#learn-cite-c1"' in html and "retrieved" in html
    assert "Was this explanation useful?" in html and 'name="target_type" value="learning"' in html
    assert "offline evidence engine" in html  # generated note
    assert re.search(r'action="/learning/\d+/quiz"', html)


def test_learn_uses_reader_audience_cookie(client, dbs):
    brief = _lumen(dbs)
    client.cookies.set("newsren_audience", "educator")
    html = client.get(f"/briefs/{brief.id}/tools/learn", headers=HX).text
    assert "Classroom guide" in html


def _learning_content(html: str, s: Session) -> GeneratedContent:
    content_id = int(re.search(r'action="/learning/(\d+)/quiz"', html).group(1))
    s.rollback()  # end the read snapshot so the app's commits are visible
    content = s.get(GeneratedContent, content_id)
    assert content is not None and content.kind == "learning"
    return content


def test_quiz_attempt_recorded_with_comprehension_and_transfer_split(client, dbs):
    brief = _lumen(dbs)
    html = client.get(f"/briefs/{brief.id}/tools/learn?audience=student", headers=HX).text
    content = _learning_content(html, dbs)
    quiz = LearningOut.model_validate(content.content).quiz
    comp = sum(1 for q in quiz if q.kind == "comprehension")
    transfer = sum(1 for q in quiz if q.kind == "transfer")
    assert comp and transfer
    data = {f"q{i}": str(q.answer_index) for i, q in enumerate(quiz)}
    data["q0"] = str((quiz[0].answer_index + 1) % len(quiz[0].options))  # one wrong comprehension answer
    resp = client.post(f"/learning/{content.id}/quiz", data={**data, "csrf_token": _token(client)}, headers=HX)
    assert resp.status_code == 200
    assert "<html" not in resp.text
    assert f"comprehension <strong>{comp - 1}/{comp}</strong>" in resp.text
    assert f"transfer <strong>{transfer}/{transfer}</strong>" in resp.text
    assert "Not quite" in resp.text
    dbs.rollback()  # end the read snapshot so the app's commits are visible
    attempt = dbs.scalars(select(QuizAttempt).where(QuizAttempt.content_id == content.id)).one()
    assert (attempt.comprehension_correct, attempt.comprehension_total) == (comp - 1, comp)
    assert (attempt.transfer_correct, attempt.transfer_total) == (transfer, transfer)
    assert attempt.condition == "newsren" and attempt.user_id is None


def test_quiz_without_js_redirects_and_shows_score(reader_client, dbs):
    brief = _lumen(dbs)
    html = reader_client.get(f"/briefs/{brief.id}/tools/learn?audience=professional").text
    content = _learning_content(html, dbs)
    resp = reader_client.post(
        f"/learning/{content.id}/quiz", data={"q0": "0", "csrf_token": reader_client.csrf}, follow_redirects=False
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/briefs/{brief.id}/tools/learn?audience=professional#quiz-{content.id}"
    page = reader_client.get(resp.headers["location"])
    assert "Your result:" in page.text and "Not answered" in page.text
    dbs.rollback()  # end the read snapshot so the app's commits are visible
    attempt = dbs.scalars(select(QuizAttempt).where(QuizAttempt.content_id == content.id)).one()
    reader = dbs.scalars(select(User).where(User.email == "reader@newsren.example")).one()
    assert attempt.user_id == reader.id
    # The stashed result is shown once.
    assert "Your result:" not in reader_client.get(resp.headers["location"]).text


def test_quiz_requires_csrf_and_valid_content(client, dbs):
    brief = _lumen(dbs)
    html = client.get(f"/briefs/{brief.id}/tools/learn?audience=student", headers=HX).text
    content = _learning_content(html, dbs)
    assert client.post(f"/learning/{content.id}/quiz", data={"q0": "1"}).status_code == 403
    token = _token(client)
    assert client.post("/learning/999999/quiz", data={"csrf_token": token}).status_code == 404
    evidence_html = client.get(f"/briefs/{brief.id}/tools/evidence", headers=HX).text
    evidence_id = re.search(r'name="target_id" value="(\d+)"', evidence_html).group(1)
    assert client.post(f"/learning/{evidence_id}/quiz", data={"csrf_token": token}).status_code == 404


def test_finance_learning_shows_notice(editor_client, dbs):
    draft = _finance_draft(dbs)
    part = editor_client.get(f"/briefs/{draft.id}/tools/learn?audience=student", headers=HX)
    assert part.status_code == 200 and NOTICE in part.text


# --------------------------------------------------------------------------
# Evidence, impact, products
# --------------------------------------------------------------------------


def test_evidence_inspection_renders_roles_claims_and_framing(client, dbs):
    brief = _lumen(dbs)
    html = client.get(f"/briefs/{brief.id}/tools/evidence", headers=HX).text
    assert "Where the evidence comes from" in html
    assert "badge-vendor_claim" in html and "Vendor claim" in html
    assert "badge-independent_evaluation" in html and "Independent evaluation" in html
    assert "Claims compared across sources" in html and "badge-contested" in html
    assert "Supported by:" in html and "Disputed by:" in html
    assert "Reported facts" in html and "Interpretations and claims of significance" in html
    assert "We don" in html and "biased" in html  # framing explainer
    assert 'href="#evidence-cite-c1"' in html and 'id="evidence-cite-c1"' in html
    assert "badge-research_preprint" in html  # preprints are labelled
    assert "Did this help you judge the evidence?" in html


def test_evidence_study_design_card(client, dbs):
    study = _brief(dbs, vertical="education", contains="AI practice assistant")
    html = client.get(f"/briefs/{study.id}/tools/evidence", headers=HX).text
    assert "Study design" in html
    assert "Participants" in html and "Outcome measures" in html and "Comparison group" in html
    assert "Limitations" in html and "Peer reviewed?" in html
    assert "Measures transfer to unfamiliar problems?" in html


def test_industry_impact_labels_and_custom_question(client, dbs):
    brief = _lumen(dbs)
    html = client.get(f"/briefs/{brief.id}/tools/impact", headers=HX).text
    assert "badge-supported" in html and "badge-scenario" in html
    assert 'class="panel stakeholder"' in html
    assert 'href="#impact-cite-c' in html
    assert 'method="get"' in html and 'name="question"' in html
    asked = "How could this affect schools, parents, and app developers?"
    custom = client.get(f"/briefs/{brief.id}/tools/impact", params={"question": asked}, headers=HX)
    assert custom.status_code == 200
    assert "How could this affect schools, parents, and app developers?" in custom.text
    assert "Back to the default analysis" in custom.text


def test_product_renders_with_labels_sections_and_citations(client, dbs):
    brief = _lumen(dbs)
    keys = [spec.key for spec in available_products(brief)]
    assert "launch_reality_check" in keys
    html = client.get(f"/briefs/{brief.id}/tools/products/launch_reality_check", headers=HX).text
    assert "Launch Reality Check" in html
    assert "product-section" in html and "badge-" in html
    assert 'href="#product-cite-c' in html and 'id="product-cite-c' in html
    assert NOTICE not in html
    full = client.get(f"/briefs/{brief.id}/tools/products/launch_reality_check")
    assert full.status_code == 200 and "Editorial products" in full.text


def test_product_unknown_or_unavailable_is_404(client, dbs):
    brief = _lumen(dbs)
    assert client.get(f"/briefs/{brief.id}/tools/products/no_such_product").status_code == 404
    assert client.get(f"/briefs/{brief.id}/tools/products/earnings_decoder").status_code == 404


def test_finance_product_shows_notice(editor_client, dbs):
    draft = _finance_draft(dbs)
    key = available_products(draft)[0].key
    part = editor_client.get(f"/briefs/{draft.id}/tools/products/{key}", headers=HX)
    assert part.status_code == 200 and NOTICE in part.text
    full = editor_client.get(f"/briefs/{draft.id}/tools/products/{key}")
    assert full.text.count(NOTICE) == 1  # once, near the top of the full page


# --------------------------------------------------------------------------
# Follow developments
# --------------------------------------------------------------------------


def test_follow_anonymous_sees_sign_in_prompt(client, dbs):
    brief = _lumen(dbs)
    html = client.get(f"/briefs/{brief.id}/tools/follow", headers=HX).text
    assert "Sign in to follow" in html
    assert f'href="/login?next=%2Fbriefs%2F{brief.id}"' in html
    assert 'action="/follows"' not in html


def test_follow_form_for_signed_in_reader(reader_client, dbs):
    brief = _lumen(dbs)
    html = reader_client.get(f"/briefs/{brief.id}/tools/follow", headers=HX).text
    assert 'action="/follows"' in html and 'method="post"' in html
    assert 'name="csrf_token"' in html
    assert re.search(r'<textarea[^>]*name="question"[^>]*>Tell me when', html)
    assert 'name="target_type" value="story"' in html
    assert f'name="story_id" value="{brief.story_id}"' in html
    assert f'name="next" value="/briefs/{brief.id}"' in html
    assert 'name="event_types" value="independent_evaluation" checked' in html
    assert 'name="event_types" value="correction"' in html
    assert 'name="require_independent" value="1" checked' in html
    assert re.search(r'name="include_opinion" value="1">', html)  # opinion excluded by default
    assert 'name="custom"' in html


def test_follow_shows_existing_follow(reader_client, dbs):
    brief = _lumen(dbs)
    reader = dbs.scalars(select(User).where(User.email == "reader@newsren.example")).one()
    create_follow(dbs, reader, "Tell me when a second lab replicates the evaluation.", story=brief.story)
    dbs.commit()
    html = reader_client.get(f"/briefs/{brief.id}/tools/follow", headers=HX).text
    assert "You're following this story" in html
    assert "a second lab replicates the evaluation" in html
    assert 'href="/me"' in html


# --------------------------------------------------------------------------
# Ask about this story / Ask Newsren
# --------------------------------------------------------------------------


def test_ask_tool_form_and_suggested_prompts(client, dbs):
    brief = _lumen(dbs)
    html = client.get(f"/briefs/{brief.id}/tools/ask", headers=HX).text
    assert f'action="/briefs/{brief.id}/ask"' in html and f'hx-post="/briefs/{brief.id}/ask"' in html
    assert 'name="question"' in html and "data-ask-form" in html
    assert html.count("data-prompt=") >= 3
    assert "Any new evidence on" in html
    prefilled = client.get(
        f"/briefs/{brief.id}/tools/ask", params={"question": "What about <b>x</b>?"}, headers=HX
    )
    assert "What about &lt;b&gt;x&lt;/b&gt;?" in prefilled.text


def test_ask_about_story_htmx_answer_with_citations(client, dbs):
    brief = _lumen(dbs)
    resp = client.post(
        f"/briefs/{brief.id}/ask",
        data={
            "question": "Did anyone independently verify the Lumen-4 benchmark score?",
            "csrf_token": _token(client),
        },
        headers=HX,
    )
    assert resp.status_code == 200
    html = resp.text
    assert "<html" not in html
    qid = int(re.search(r'href="/questions/(\d+)"', html).group(1))
    assert f'href="#answer-{qid}-cite-c1"' in html and f'id="answer-{qid}-cite-c1"' in html
    assert 'class="plan-steps"' in html and 'class="verify-grid"' in html
    assert "Verified in the sources" in html and "Could not verify" in html
    assert 'name="target_type" value="answer"' in html and 'name="rating"' in html
    dbs.rollback()  # end the read snapshot so the app's commits are visible
    q = dbs.get(Question, qid)
    assert q.brief_id == brief.id and q.citations and not q.refused


def test_ask_about_story_uses_the_forms_audience(client, dbs):
    brief = _lumen(dbs)
    resp = client.post(
        f"/briefs/{brief.id}/ask",
        data={"question": "What is MathBench-Pro?", "audience": "educator", "csrf_token": _token(client)},
        follow_redirects=False,
    )
    qid = int(resp.headers["location"].rsplit("/", 1)[1])
    dbs.rollback()
    assert dbs.get(Question, qid).audience == "educator"
    page = client.get(f"/questions/{qid}")
    assert "explained for educators" in page.text


def test_ask_about_story_without_js_redirects_to_private_permalink(client, dbs):
    brief = _lumen(dbs)
    resp = client.post(
        f"/briefs/{brief.id}/ask",
        data={"prompt": "Has the claimed performance been independently verified?", "csrf_token": _token(client)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    location = resp.headers["location"]
    assert re.fullmatch(r"/questions/\d+", location)
    page = client.get(location)
    assert page.status_code == 200
    assert "Has the claimed performance been independently verified?" in page.text
    assert f'href="/briefs/{brief.id}"' in page.text
    assert re.search(r'<h1 class="answer-question"', page.text)
    assert "data-prompt=" not in page.text  # no ask box here, so follow-ups are plain links
    # Another browser cannot open it.
    client.cookies.clear()
    assert client.get(location).status_code == 404


def test_permalink_visible_to_editor_and_owner_only(client, dbs):
    q = dbs.scalars(select(Question).order_by(Question.id)).first()
    if q is None:
        token = _csrf(client)
        client.post("/ask", data={"question": "What changed in AI tutoring?", "csrf_token": token})
        client.cookies.clear()
        dbs.rollback()  # end the read snapshot so the app's commits are visible
        q = dbs.scalars(select(Question).order_by(Question.id)).first()
    assert client.get(f"/questions/{q.id}").status_code == 404
    from conftest import DEMO_EDITOR_PASSWORD, login

    login(client, "editor@newsren.example", DEMO_EDITOR_PASSWORD)
    assert client.get(f"/questions/{q.id}").status_code == 200
    assert client.get("/questions/999999").status_code == 404


def test_signed_in_reader_sees_own_question_after_new_session(reader_client, dbs):
    resp = reader_client.post(
        "/ask",
        data={"question": "What did the board propose for AI tutors?", "csrf_token": reader_client.csrf},
        follow_redirects=False,
    )
    location = resp.headers["location"]
    from conftest import DEMO_READER_PASSWORD, login

    reader_client.cookies.clear()
    login(reader_client, "reader@newsren.example", DEMO_READER_PASSWORD)
    assert reader_client.get(location).status_code == 200


def test_finance_advice_question_is_refused_with_education(client):
    resp = client.post(
        "/ask", data={"question": "Should I buy Corvane stock right now?", "csrf_token": _token(client)}
    )
    assert resp.status_code == 200  # followed the 303 to the permalink
    html = resp.text
    assert "We can't give personalized investment advice." in html
    assert 'class="refusal"' in html
    assert NOTICE in html
    assert 'href="/finance/concepts"' in html
    assert "verify-grid" not in html


def test_finance_advice_on_brief_tool_refused_htmx(client, dbs):
    brief = _lumen(dbs)
    resp = client.post(
        f"/briefs/{brief.id}/ask",
        data={"question": "Is Lumen Labs a good investment? Should I buy?", "csrf_token": _token(client)},
        headers=HX,
    )
    assert resp.status_code == 200
    assert 'class="refusal"' in resp.text and NOTICE in resp.text


def test_ask_page_and_recent_questions(client):
    page = client.get("/ask")
    assert page.status_code == 200
    html = page.text
    assert "Understand the next technology breakthrough" in html
    assert 'action="/ask"' in html and 'hx-post="/ask"' in html and 'method="post"' in html
    assert "approved" in html and "could not verify" in html and "investment advice" in html
    for vertical in ("technology", "education", "industry", "finance"):
        assert f"vertical-tag-{vertical}" in html
    assert "No questions yet" in html
    assert 'class="nav-ask is-active"' in html
    assert 'hx-trigger="submit"' in html  # nothing to research yet
    prefilled = client.get("/ask", params={"question": "Hello <script>"})
    assert 'value="Hello &lt;script&gt;"' in prefilled.text
    assert 'hx-trigger="submit, load"' in prefilled.text  # handed-over question runs on load (POST + CSRF)
    resp = client.post(
        "/ask", data={"question": "What is MathBench-Pro?", "csrf_token": _token(client)}, follow_redirects=False
    )
    assert resp.status_code == 303 and resp.headers["location"].startswith("/questions/")
    again = client.get("/ask").text
    assert "What is MathBench-Pro?" in again and resp.headers["location"] in again


def test_ask_htmx_returns_answer_partial(client):
    resp = client.post(
        "/ask", data={"question": "What is MathBench-Pro?", "csrf_token": _token(client)}, headers=HX
    )
    assert resp.status_code == 200
    assert "<html" not in resp.text and 'class="answer-block"' in resp.text


def test_ask_empty_or_too_long_question(client, dbs):
    token = _token(client)
    resp = client.post("/ask", data={"question": "   ", "csrf_token": token})
    assert resp.status_code == 400 and "Write a question" in resp.text
    part = client.post("/ask", data={"question": "", "csrf_token": token}, headers=HX)
    assert part.status_code == 200 and 'role="alert"' in part.text
    long = client.post("/ask", data={"question": "x" * 1200, "csrf_token": token})
    assert long.status_code == 400
    brief = _lumen(dbs)
    nojs = client.post(
        f"/briefs/{brief.id}/ask", data={"question": "", "csrf_token": token}, follow_redirects=False
    )
    assert nojs.status_code == 303 and nojs.headers["location"] == f"/briefs/{brief.id}/tools/ask"
    dbs.rollback()  # end the read snapshot so the app's commits are visible
    assert dbs.scalar(select(func.count()).select_from(Question)) == 0


def test_ask_requires_csrf(client, dbs):
    brief = _lumen(dbs)
    assert client.post("/ask", data={"question": "Anything?"}).status_code == 403
    assert client.post(f"/briefs/{brief.id}/ask", data={"question": "Anything?"}).status_code == 403
    assert client.post("/ask", data={"question": "Anything?", "csrf_token": "wrong"}).status_code == 403


def test_ask_about_unpublished_brief_404(client, dbs):
    draft = _finance_draft(dbs)
    resp = client.post(
        f"/briefs/{draft.id}/ask", data={"question": "What happened?", "csrf_token": _token(client)}
    )
    assert resp.status_code == 404


# --------------------------------------------------------------------------
# Feedback
# --------------------------------------------------------------------------


def _ask_one(client) -> int:
    resp = client.post(
        "/ask", data={"question": "What is MathBench-Pro?", "csrf_token": _token(client)}, follow_redirects=False
    )
    return int(resp.headers["location"].rsplit("/", 1)[1])


def test_feedback_htmx_stores_answer_rating(client, dbs):
    qid = _ask_one(client)
    resp = client.post(
        "/feedback",
        data={
            "target_type": "answer",
            "target_id": str(qid),
            "answered_question": "yes",
            "saved_time": "no",
            "rating": "4",
            "csrf_token": _token(client),
        },
        headers=HX,
    )
    assert resp.status_code == 200 and "Thanks" in resp.text and "<html" not in resp.text
    dbs.rollback()  # end the read snapshot so the app's commits are visible
    row = dbs.scalars(select(Feedback).where(Feedback.target_type == "answer", Feedback.target_id == qid)).one()
    assert (row.answered_question, row.saved_time, row.rating) == (True, False, 4)


def test_feedback_without_js_redirects_back_safely(reader_client, dbs):
    brief = _lumen(dbs)
    token = reader_client.csrf
    resp = reader_client.post(
        "/feedback",
        data={
            "target_type": "brief",
            "target_id": str(brief.id),
            "answered_question": "no",
            "next": f"/briefs/{brief.id}",
            "csrf_token": token,
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303 and resp.headers["location"] == f"/briefs/{brief.id}"
    evil = reader_client.post(
        "/feedback",
        data={
            "target_type": "brief",
            "target_id": str(brief.id),
            "answered_question": "yes",
            "next": "https://evil.example/",
            "csrf_token": token,
        },
        follow_redirects=False,
    )
    assert evil.headers["location"] == "/"
    dbs.rollback()  # end the read snapshot so the app's commits are visible
    rows = dbs.scalars(
        select(Feedback).where(Feedback.target_type == "brief", Feedback.target_id == brief.id)
    ).all()
    assert len(rows) == 1 and rows[0].answered_question is True  # a reader's second rating updates the first


def test_feedback_on_generated_content(client, dbs):
    brief = _lumen(dbs)
    html = client.get(f"/briefs/{brief.id}/tools/learn?audience=student", headers=HX).text
    content = _learning_content(html, dbs)
    resp = client.post(
        "/feedback",
        data={
            "target_type": "learning",
            "target_id": str(content.id),
            "answered_question": "yes",
            "csrf_token": _token(client),
        },
        headers=HX,
    )
    assert resp.status_code == 200
    # The wrong kind for that content id is rejected.
    wrong = client.post(
        "/feedback",
        data={"target_type": "impact", "target_id": str(content.id), "csrf_token": _token(client)},
    )
    assert wrong.status_code == 404


def test_feedback_validation_and_csrf(client, dbs):
    brief = _lumen(dbs)
    draft = _finance_draft(dbs)
    token = _token(client)
    assert client.post("/feedback", data={"target_type": "brief", "target_id": str(brief.id)}).status_code == 403
    assert (
        client.post(
            "/feedback", data={"target_type": "nonsense", "target_id": "1", "csrf_token": token}
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/feedback", data={"target_type": "brief", "target_id": "abc", "csrf_token": token}
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/feedback", data={"target_type": "brief", "target_id": "999999", "csrf_token": token}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/feedback", data={"target_type": "brief", "target_id": str(draft.id), "csrf_token": token}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/feedback",
            data={"target_type": "brief", "target_id": str(brief.id), "rating": "9", "csrf_token": token},
        ).status_code
        == 400
    )
    dbs.rollback()  # end the read snapshot so the app's commits are visible
    assert dbs.scalar(select(func.count()).select_from(Feedback)) == 0
