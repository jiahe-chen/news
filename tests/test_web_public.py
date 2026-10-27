"""Public reading pages: front page, sections, AI Story Brief, story, search, sources, about."""

from __future__ import annotations

import re
from html import unescape

import pytest
from sqlalchemy import func, select

from newsren.agents.products import PRODUCTS, available_products
from newsren.editorial.guardrails import FINANCE_DISCLAIMER
from newsren.models import Article, Brief, BriefStatus, PageView, Story, User

FINANCE_SNIPPET = "this is not investment advice"
TOOL_KEYS = ["connected", "learn", "evidence", "impact", "follow", "ask"]


def _approved(session, vertical=None):
    query = select(Brief).where(Brief.status == BriefStatus.APPROVED)
    if vertical:
        query = query.where(Brief.vertical == vertical)
    return list(session.scalars(query.order_by(Brief.id)))


def _story_brief(session) -> Brief:
    """An approved brief on a multi-source story (the richest page in the demo)."""
    best = None
    for brief in _approved(session):
        if brief.story is None:
            continue
        if best is None or len(brief.citations) > len(best.citations):
            best = brief
    assert best is not None
    return best


def _in_review_finance(session) -> Brief:
    brief = session.scalars(
        select(Brief).where(Brief.status != BriefStatus.APPROVED, Brief.vertical == "finance").order_by(Brief.id)
    ).first()
    assert brief is not None, "demo should keep finance briefs in review"
    return brief


def _done(session):
    """End the test session's read transaction so the app can write to the shared SQLite file."""
    session.rollback()


# --------------------------------------------------------------------------
# Front page
# --------------------------------------------------------------------------


def test_home_has_sections_lead_and_agent_entry(client, demo_session):
    lead = _story_brief(demo_session)
    lead_headline = lead.headline
    _done(demo_session)
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    for heading in ("Technology", "Education", "Industry", "Finance", "Developing stories"):
        assert heading in html
    # Lead story: the most-cited recent approved brief.
    assert 'class="brief-card lead' in html
    assert unescape(html).count(lead_headline) >= 1
    # Agent entry point (plan §5) with one suggested prompt per vertical.
    assert "Understand the next technology breakthrough" in html
    assert "compare primary research, product announcements, and independent evaluations" in html.lower()
    assert 'action="/ask"' in html and 'name="question"' in html
    assert html.count("data-prompt=") >= 4
    # Finance section carries the not-advice line; demo content is flagged.
    assert "Not investment advice." in html
    assert "fictional demo content" in html
    assert "newsren pipeline" in html


def test_home_only_shows_published_briefs(client, demo_session):
    hidden = [b.id for b in demo_session.scalars(select(Brief).where(Brief.status != BriefStatus.APPROVED))]
    shown = [b.id for b in _approved(demo_session)]
    _done(demo_session)
    assert hidden
    html = client.get("/").text
    for brief_id in hidden:
        assert f'href="/briefs/{brief_id}"' not in html
    linked = {int(x) for x in re.findall(r'href="/briefs/(\d+)"', html)}
    assert linked and linked <= set(shown)


def test_home_developing_stories_have_counts_and_event_badges(client, demo_session):
    counts = dict(
        demo_session.execute(
            select(Article.story_id, func.count(Article.id))
            .where(Article.story_id.is_not(None), Article.duplicate_of_id.is_(None))
            .group_by(Article.story_id)
        ).all()
    )
    multi = [s for s in demo_session.scalars(select(Story)) if counts.get(s.id, 0) >= 2]
    single = [s.id for s in demo_session.scalars(select(Story)) if counts.get(s.id, 0) == 1]
    newest = max(multi, key=lambda s: (s.last_updated_at, s.id))
    story_id, n = newest.id, counts[newest.id]
    _done(demo_session)
    html = client.get("/").text
    aside = html.split('id="developing-heading"')[1].split("</aside>")[0]
    assert f'href="/stories/{story_id}"' in aside
    assert f"{n} sourced developments" in aside
    assert "badge-event" in aside
    for sid in single:  # single-report stories are not "developing"
        assert f'href="/stories/{sid}"' not in aside


# --------------------------------------------------------------------------
# Vertical sections
# --------------------------------------------------------------------------


@pytest.mark.parametrize("vertical", ["technology", "education", "industry"])
def test_vertical_page(client, vertical):
    resp = client.get(f"/v/{vertical}")
    assert resp.status_code == 200
    html = unescape(resp.text)
    assert "What we monitor" in html
    assert 'class="agent-entry"' in resp.text and 'action="/ask"' in resp.text
    for spec in PRODUCTS.values():
        if spec.vertical == vertical:
            assert spec.name in html
            assert spec.reader_value in html
        else:
            assert f'<p class="product-name">{spec.name}</p>' not in html
    assert 'aria-current="page"' in resp.text


def test_vertical_intro_goes_beyond_summaries(client):
    html = client.get("/v/technology").text
    assert "distinguishes vendor claims from third-party evaluations" in html


def test_finance_vertical_redirects_and_unknown_is_404(client):
    resp = client.get("/v/finance", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/finance"
    assert client.get("/v/sports").status_code == 404


# --------------------------------------------------------------------------
# AI Story Brief page
# --------------------------------------------------------------------------


def test_approved_brief_is_public_with_full_structure(client, demo_session):
    brief = _story_brief(demo_session)
    brief_id, headline, n_sources = brief.id, brief.headline, len(brief.citations)
    first_claim = brief.claims[0].text
    has_disagreement = bool(brief.disagreements)
    _done(demo_session)
    resp = client.get(f"/briefs/{brief_id}")
    assert resp.status_code == 200
    html = resp.text
    text = unescape(html)
    assert headline in text
    assert "Why it matters" in html
    assert "Key points" in html
    assert 'class="expandable"' in html
    assert "Claims and their evidence" in html and first_claim in text
    assert "What remains uncertain" in html
    if has_disagreement:
        assert "Where credible sources disagree" in html and 'class="position"' in html
    assert "What to watch next" in html
    assert 'id="corrections"' in html
    assert f"{n_sources} sources" in html
    assert 'class="source-list"' in html and "retrieved" in html
    assert "offline evidence engine" in html  # generated_note
    assert "Demo content" in html
    assert "Unpublished draft" not in html
    # Inline citation markers become links to the source list, never raw "[c1]" text.
    assert 'href="#cite-c1"' in html
    lede = re.search(r'<p class="lede">(.*?)</p>', html, re.S).group(1)
    assert "[c1]" not in lede and 'class="cite"' in lede


def test_brief_page_has_tools_tablist_and_products(client, demo_session):
    brief = _story_brief(demo_session)
    brief_id = brief.id
    product_keys = [spec.key for spec in available_products(brief)]
    _done(demo_session)
    html = client.get(f"/briefs/{brief_id}").text
    assert 'role="tablist"' in html
    assert html.count('role="tab"') == len(TOOL_KEYS)
    for key in TOOL_KEYS:
        url = f"/briefs/{brief_id}/tools/{key}"
        assert f'href="{url}' in html, key
        assert f'hx-get="{url}' in html, key
    assert f"/briefs/{brief_id}/tools/learn?audience=professional" in html
    assert 'hx-target="#tool-panel"' in html and 'hx-push-url="false"' in html
    assert "data-tool-tab" in html
    # Default panel lazily loads Connected stories, with a plain link for no-JS readers.
    panel = re.search(r'<div id="tool-panel"[^>]*>', html).group(0)
    assert f'hx-get="/briefs/{brief_id}/tools/connected"' in panel and 'hx-trigger="load"' in panel
    assert product_keys
    for key in product_keys:
        assert f'href="/briefs/{brief_id}/tools/products/{key}"' in html
    assert "Explain for: Professional" in html


def test_brief_learn_tab_uses_reader_audience(client, demo_session):
    brief_id = _story_brief(demo_session).id
    _done(demo_session)
    html = client.get(f"/briefs/{brief_id}?audience=student").text
    assert f"/briefs/{brief_id}/tools/learn?audience=student" in html
    assert "Explain for: Student" in html


def test_unpublished_brief_hidden_from_readers_previewed_for_editors(client, demo_session):
    brief_id = _in_review_finance(demo_session).id
    _done(demo_session)
    assert client.get(f"/briefs/{brief_id}").status_code == 404
    assert client.get("/briefs/999999").status_code == 404


def test_unpublished_brief_404_for_signed_in_reader(reader_client, demo_session):
    brief_id = _in_review_finance(demo_session).id
    _done(demo_session)
    assert reader_client.get(f"/briefs/{brief_id}").status_code == 404


def test_editor_preview_of_unpublished_finance_brief(editor_client, demo_session):
    brief = _in_review_finance(demo_session)
    brief_id, headline = brief.id, brief.headline
    _done(demo_session)
    resp = editor_client.get(f"/briefs/{brief_id}")
    assert resp.status_code == 200
    html = resp.text
    assert "Unpublished draft — in editorial review" in html
    assert headline in unescape(html)
    assert FINANCE_SNIPPET in html
    # The finance notice sits near the top, before the headline.
    assert html.index('class="finance-notice"') < html.index('id="brief-title"')


def test_published_finance_brief_shows_finance_notice(client, demo_session):
    brief = _in_review_finance(demo_session)
    brief.status = BriefStatus.APPROVED.value
    brief_id = brief.id
    demo_session.commit()
    resp = client.get(f"/briefs/{brief_id}")
    assert resp.status_code == 200
    assert 'class="finance-notice"' in resp.text
    assert FINANCE_DISCLAIMER.split(":")[0] in resp.text
    assert "Unpublished draft" not in resp.text


def test_non_finance_brief_has_no_finance_notice(client, demo_session):
    brief_id = _approved(demo_session, "technology")[0].id
    _done(demo_session)
    assert 'class="finance-notice"' not in client.get(f"/briefs/{brief_id}").text


def test_brief_text_is_escaped(client, demo_session):
    brief = _story_brief(demo_session)
    brief.headline = "<script>alert(1)</script> Lumen"
    brief.summary = "A <b>bold</b> claim [c1]."
    brief_id = brief.id
    demo_session.commit()
    html = client.get(f"/briefs/{brief_id}").text
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<b>bold</b>" not in html and "&lt;b&gt;bold&lt;/b&gt;" in html


def test_brief_page_view_recorded_for_signed_in_reader(reader_client, demo_session):
    brief = _story_brief(demo_session)
    brief_id, story_id = brief.id, brief.story_id
    reader_id = demo_session.scalar(select(User.id).where(User.email == "reader@newsren.example"))
    _done(demo_session)
    assert reader_client.get(f"/briefs/{brief_id}").status_code == 200
    view = demo_session.scalars(
        select(PageView).where(PageView.user_id == reader_id, PageView.brief_id == brief_id)
    ).first()
    assert view is not None
    assert view.story_id == story_id and view.path == f"/briefs/{brief_id}"
    _done(demo_session)


def test_anonymous_views_are_not_recorded(client, demo_session):
    brief_id = _story_brief(demo_session).id
    before = demo_session.scalar(select(func.count(PageView.id)))
    _done(demo_session)
    client.get(f"/briefs/{brief_id}")
    assert demo_session.scalar(select(func.count(PageView.id))) == before
    _done(demo_session)


# --------------------------------------------------------------------------
# Story pages
# --------------------------------------------------------------------------


def _story_with_approved_brief(session) -> Story:
    return _story_brief(session).story


def _story_without_approved_brief(session) -> Story:
    approved_story_ids = {b.story_id for b in _approved(session)}
    for brief in session.scalars(select(Brief).where(Brief.status != BriefStatus.APPROVED).order_by(Brief.id)):
        if brief.story_id and brief.story_id not in approved_story_ids and brief.story.events:
            return brief.story
    raise AssertionError("demo should have a story whose brief is still in review")


def test_story_page_timeline_and_ai_summary(client, demo_session):
    story = _story_with_approved_brief(demo_session)
    story_id = story.id
    events = [(e.title, e.article.url) for e in story.events if e.article]
    _done(demo_session)
    resp = client.get(f"/stories/{story_id}")
    assert resp.status_code == 200
    html = resp.text
    text = unescape(html)
    assert 'class="timeline"' in html
    for title, url in events:
        assert title in text
        assert f'href="{url}"' in html
    assert "badge-event" in html
    assert "What happened first" in html and "What's new" in html.replace("&#39;", "'")
    assert "Sources for this summary" in html
    assert "Editor preview" not in html
    assert "Connected stories" in html
    assert "Source documents by type" in html
    assert re.search(r'href="/briefs/\d+"', html)


def test_story_summary_hidden_until_brief_approved(client, demo_session):
    story = _story_without_approved_brief(demo_session)
    story_id, first_event = story.id, story.events[0].title
    _done(demo_session)
    html = client.get(f"/stories/{story_id}").text
    assert first_event in unescape(html)  # timeline is public
    assert "What happened first" not in html
    assert "Brief in editorial review" in html


def test_story_summary_previewed_for_editor(editor_client, demo_session):
    story_id = _story_without_approved_brief(demo_session).id
    _done(demo_session)
    html = editor_client.get(f"/stories/{story_id}").text
    assert "What happened first" in html
    assert "Editor preview" in html


def test_story_follow_button_anonymous_and_reader(client, demo_session):
    story_id = _story_with_approved_brief(demo_session).id
    _done(demo_session)
    anon = client.get(f"/stories/{story_id}").text
    assert f'href="/login?next=/stories/{story_id}"' in anon
    assert 'action="/follows"' not in anon
    from conftest import DEMO_READER_PASSWORD, login

    token = login(client, "reader@newsren.example", DEMO_READER_PASSWORD)
    html = client.get(f"/stories/{story_id}").text
    form = re.search(r'<form method="post" action="/follows".*?</form>', html, re.S).group(0)
    assert 'name="target_type" value="story"' in form
    assert f'name="story_id" value="{story_id}"' in form
    assert f'name="csrf_token" value="{token}"' in form
    assert f'name="next" value="/stories/{story_id}"' in form


def test_story_page_view_recorded_and_404(reader_client, demo_session):
    story_id = _story_with_approved_brief(demo_session).id
    _done(demo_session)
    assert reader_client.get(f"/stories/{story_id}").status_code == 200
    assert reader_client.get("/stories/999999").status_code == 404
    count = demo_session.scalar(
        select(func.count(PageView.id)).where(PageView.story_id == story_id, PageView.brief_id.is_(None))
    )
    assert count == 1
    _done(demo_session)


def test_finance_story_has_finance_notice(client, demo_session):
    story_id = demo_session.scalar(select(Story.id).where(Story.vertical == "finance").order_by(Story.id))
    _done(demo_session)
    html = client.get(f"/stories/{story_id}").text
    assert 'class="finance-notice"' in html and FINANCE_SNIPPET in html


# --------------------------------------------------------------------------
# Search, sources, about
# --------------------------------------------------------------------------


def test_search_returns_briefs_then_articles(client, demo_session):
    brief = _story_brief(demo_session)
    word = next(w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{4,}", brief.headline))
    hidden = [b.id for b in demo_session.scalars(select(Brief).where(Brief.status != BriefStatus.APPROVED))]
    _done(demo_session)
    resp = client.get("/search", params={"q": word})
    assert resp.status_code == 200
    html = resp.text
    assert f'href="/briefs/{brief.id}"' in html
    assert html.index('id="brief-results"') < html.index('id="article-results"')
    assert 'class="article-line"' in html
    assert f"/ask?question={word}" in html and "Ask Newsren instead" in html
    for brief_id in hidden:
        assert f'href="/briefs/{brief_id}"' not in html


def test_search_collapses_duplicates_and_handles_odd_input(client, demo_session):
    dup = demo_session.scalars(select(Article).where(Article.duplicate_of_id.is_not(None))).first()
    _done(demo_session)
    if dup is not None:
        html = client.get("/search", params={"q": dup.title}).text
        assert f'href="{dup.url}"' not in html or dup.url == dup.canonical_url
    for q in ['"', "<script>", "AND OR NOT", "x" * 500, ""]:
        resp = client.get("/search", params={"q": q})
        assert resp.status_code == 200
        assert "<script>" not in resp.text.split('id="search-q"')[1]


def test_sources_page(client, demo_session):
    names = [s for s in demo_session.scalars(select(Article.source_id).distinct())]
    _done(demo_session)
    resp = client.get("/sources")
    assert resp.status_code == 200
    html = resp.text
    assert "Approved sources" in html
    for heading in ("Technology", "Education", "Industry", "Finance"):
        assert f'id="src-{heading.lower()}"' in html
    assert "Trust tier" in html and "License" in html and "Data delay" in html and "Last fetched" in html
    assert "SEC EDGAR" in html
    assert "Not investment advice" in html
    assert names


def test_about_page(client):
    resp = client.get("/about")
    assert resp.status_code == 200
    html = unescape(resp.text)
    for step in ("Discover", "Verify", "Connect", "Explain", "Quality gate", "Output", "Reader loop"):
        assert step in html
    assert "Conventional AI versus agentic AI" in html
    for feature in (
        "My Newsren",
        "Story Intelligence",
        "News-to-Learning",
        "Industry Impact",
        "Evidence and Bias Inspection",
        "Follow This Story",
    ):
        assert feature in html
    for metric in ("Trust", "Usefulness", "Education", "Engagement", "Editorial efficiency", "Cost"):
        assert f"<td>{metric}</td>" in html
    assert "chatbot usage are not our primary metrics" in re.sub(r"<[^>]+>", "", html)
    assert "backtest, simulated or live" in html
    assert "offline evidence engine" in html
    assert "fictional demo documents" in html


def test_public_pages_have_no_post_routes(client):
    # The public area is read-only; state changes live elsewhere and require CSRF.
    assert client.post("/search", data={"q": "x"}).status_code == 405
    assert client.post("/about").status_code == 405
