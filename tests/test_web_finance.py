"""Finance section web pages (plan §3 Finance, §4): front, Earnings Decoder, Funding Round Tracker,
AI-in-Finance Reality Check, Regulation Watch, and concepts for learners."""

from __future__ import annotations

import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import pytest
from sqlalchemy import select

from conftest import _csrf
from newsren.editorial.guardrails import FINANCE_DISCLAIMER, lint_text
from newsren.editorial.workflow import approve
from newsren.finance.concepts import CONCEPTS, concepts_in_text
from newsren.finance.edgar import build_earnings_report, earnings_brief, format_value, get_or_create_company
from newsren.finance.service import DATA_NOTE
from newsren.models import Brief, BriefStatus, EarningsReport, Follow, FundingRound, User
from newsren.schemas import GuidanceItem

DATA = Path(__file__).parent / "data" / "finance"
NOTICE = "Not investment advice."
FINANCE_PAGES = [
    "/finance",
    "/finance/earnings",
    "/finance/funding",
    "/finance/ai-reality-check",
    "/finance/regulation",
    "/finance/concepts",
    "/finance/concepts/backtesting",
]
NAV_LABELS = {
    "/finance": "Overview",
    "/finance/earnings": "Earnings Decoder",
    "/finance/funding": "Funding Round Tracker",
    "/finance/ai-reality-check": "AI-in-Finance Reality Check",
    "/finance/regulation": "Regulation Watch",
    "/finance/concepts": "Concepts for learners",
    "/finance/concepts/backtesting": "Concepts for learners",
}
JUDGEMENT_WORDS = re.compile(r"\b(?:beat|beats|missed|misses|miss|disappointing|strong|weak|impressive)\b", re.I)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class _VisibleText(HTMLParser):
    """Text a reader sees: everything in <body> outside <script>/<style>."""

    def __init__(self) -> None:
        super().__init__()
        self.skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip())


def visible_text(html: str) -> str:
    parser = _VisibleText()
    parser.feed(html)
    return "\n".join(parser.parts)


def _done(session) -> None:
    """End the test session's transaction so the app can use the shared SQLite file."""
    session.commit()


def _editor(session) -> User:
    return session.scalars(select(User).where(User.email == "editor@newsren.example")).one()


def _publish(session, brief: Brief) -> Brief:
    approve(session, brief, _editor(session), notes="OVERRIDE: approved for a web test")
    session.commit()
    return brief


def _irmd_report(session, guidance=None) -> EarningsReport:
    company = get_or_create_company(session, "IRADIMED CORP", ticker="IRMD", cik="1325618")
    report = build_earnings_report(
        session,
        company,
        None,
        facts=json.loads((DATA / "irmd_companyfacts.json").read_text()),
        submissions=json.loads((DATA / "irmd_submissions.json").read_text()),
        guidance=guidance,
    )
    session.commit()
    return report


def _in_review_finance(session) -> list[Brief]:
    return list(
        session.scalars(select(Brief).where(Brief.vertical == "finance", Brief.status != BriefStatus.APPROVED))
    )


def _brief_mentioning(session, slug: str) -> Brief:
    for brief in _in_review_finance(session):
        text = " ".join([brief.headline, brief.summary, brief.why_it_matters or ""])
        text += " " + " ".join(str(s.get("body", "")) for s in brief.sections or [])
        if any(c.slug == slug for c in concepts_in_text(text)):
            return brief
    raise AssertionError(f"no demo finance brief mentions {slug}")


# --------------------------------------------------------------------------
# Guardrails on every page
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", FINANCE_PAGES)
def test_every_finance_page_has_notice_data_note_and_section_nav(client, path):
    resp = client.get(path)
    assert resp.status_code == 200
    html = unescape(resp.text)
    # The not-investment-advice notice comes before the page's own content.
    assert NOTICE in html and FINANCE_DISCLAIMER in html
    assert html.index(NOTICE) < html.index('class="page-title')
    # Data note: sources, delays, public data only, timestamp.
    assert DATA_NOTE in html
    assert "No live market prices" in html and "Page generated" in html
    # Section navigation marks the current page; the masthead marks Finance.
    assert re.search(r'aria-current="page">' + re.escape(NAV_LABELS[path]) + "<", html)
    assert 'class="nav-finance is-active"' in html


def test_earnings_detail_has_notice_and_data_note(client, demo_session):
    report = _irmd_report(demo_session)
    html = unescape(client.get(f"/finance/earnings/{report.id}").text)
    assert NOTICE in html and FINANCE_DISCLAIMER in html and DATA_NOTE in html
    assert "EDGAR data can lag filings and is not real-time market data" in html


def test_no_advice_language_on_any_finance_page(editor_client, demo_session):
    """lint_text finds no buy/sell/hold calls, price targets, personal advice, or market-beating claims."""
    report = _irmd_report(
        demo_session,
        guidance=[
            GuidanceItem(metric="Revenue", guidance="$20.0M to $20.5M", actual="$20,466,000", comparison="within")
        ],
    )
    paths = FINANCE_PAGES + [
        f"/finance/earnings/{report.id}",
        "/finance/funding?stage=seed",
        "/finance/ai-reality-check?area=asset_wealth",
        "/finance/regulation?days=365",
    ]
    paths += [f"/finance/concepts/{c.slug}" for c in CONCEPTS]
    for path in paths:
        resp = editor_client.get(path)
        assert resp.status_code == 200, path
        assert lint_text(visible_text(resp.text), "finance") == [], path


def test_finance_pages_escape_source_text_and_drop_unsafe_links(client, demo_session):
    demo_session.add(
        FundingRound(
            company_name="<script>alert(1)</script> Labs",
            sector="Fintech",
            stage="seed",
            amount_text="$1 million",
            amount_usd=1_000_000.0,
            investors=["<b>Bold</b> Ventures"],
            lead_investors=["<b>Bold</b> Ventures"],
            source_url="javascript:alert(1)",
            source_title="<img src=x onerror=alert(1)>",
            extraction_confidence=0.7,
        )
    )
    _done(demo_session)
    html = client.get("/finance/funding").text
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;alert(1)&lt;/script&gt; Labs" in html
    assert "<b>Bold</b>" not in html and "<img src=x" not in html
    assert 'href="javascript:' not in html


# --------------------------------------------------------------------------
# Section front
# --------------------------------------------------------------------------


def test_front_explains_scope_guardrails_and_links_every_module(client):
    html = unescape(client.get("/finance").text)
    assert "From market headlines to evidence-based understanding" in html
    assert "What we track" in html and "What we never do" in html
    for line in (
        "No buy, sell, or hold calls, and no price targets.",
        "No trade execution and no brokerage connections.",
    ):
        assert line in html
    assert "personalized investment advice" in html
    for href in (
        "/finance/earnings",
        "/finance/funding",
        "/finance/ai-reality-check",
        "/finance/regulation",
        "/finance/concepts",
        "#follow",
    ):
        assert f'href="{href}"' in html
    assert "Follow a company or sector" in html and "not price moves" in html.lower()
    # Demo content is flagged; the agent entry point hands off to Ask Newsren.
    assert "Fictional demo content." in html
    assert 'action="/ask"' in html and "data-prompt=" in html


def test_front_lists_only_approved_finance_briefs(client, demo_session):
    pending = [b for b in _in_review_finance(demo_session) if b.kind == "story"]
    assert pending, "the demo keeps flagged finance briefs in review"
    ids = [b.id for b in pending]
    approved = list(
        demo_session.scalars(
            select(Brief).where(Brief.vertical == "finance", Brief.status == BriefStatus.APPROVED)
        )
    )
    _done(demo_session)
    html = unescape(client.get("/finance").text)
    assert approved and any(f'href="/briefs/{b.id}"' in html for b in approved)
    assert not any(f'href="/briefs/{i}"' in html for i in ids)

    published = _publish(demo_session, pending[0])
    html = unescape(client.get("/finance").text)
    assert published.headline in html and f'href="/briefs/{published.id}"' in html
    assert not any(f'href="/briefs/{i}"' in html for i in ids[1:])
    assert "No finance briefs published yet" not in html


def test_front_editor_sees_review_queue_count(editor_client, demo_session):
    count = len(_in_review_finance(demo_session))
    _done(demo_session)
    html = editor_client.get("/finance").text
    assert "Editor view:" in html and f"{count} finance briefs awaiting human review" in html
    assert 'href="/editor"' in html


def test_front_reader_does_not_see_review_queue(reader_client):
    assert "Editor view:" not in reader_client.get("/finance").text


def test_follow_anonymous_gets_sign_in_link(client):
    html = client.get("/finance").text
    assert 'href="/login?next=%2Ffinance%23follow"' in html
    assert 'action="/follows"' not in html


def test_follow_form_for_signed_in_reader(reader_client):
    html = reader_client.get("/finance").text
    form = html[html.index('action="/follows"') :]
    form = form[: form.index("</form>")]
    assert f'name="csrf_token" value="{reader_client.csrf}"' in form
    assert 'name="target_type" value="company"' in form and 'name="target_type" value="sector"' in form
    assert 'name="entity_name"' in form and 'name="vertical" value="finance"' in form
    for event in ("filing", "earnings", "funding", "enforcement"):
        assert f'name="event_types" value="{event}" checked' in form
    assert '<label for="fin-follow-entity">' in form and 'id="fin-follow-entity"' in form


def test_follow_form_creates_a_company_follow(reader_client, demo_session):
    resp = reader_client.post(
        "/follows",
        data={
            "csrf_token": reader_client.csrf,
            "target_type": "company",
            "entity_name": "Corvane Semiconductor",
            "vertical": "finance",
            "event_types": ["filing", "earnings", "funding", "enforcement"],
            "question": "Alert me to material events, not price moves.",
            "next": "/finance#follow",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303 and resp.headers["location"] == "/finance#follow"
    demo_session.expire_all()
    follow = demo_session.scalars(
        select(Follow).where(Follow.entity_name == "Corvane Semiconductor").order_by(Follow.id.desc())
    ).first()
    assert (follow.target_type, follow.vertical) == ("company", "finance")
    assert set(follow.criteria["event_types"]) >= {"filing", "earnings", "funding", "enforcement"}


def test_follow_form_post_requires_csrf(reader_client):
    resp = reader_client.post(
        "/follows",
        data={
            "target_type": "sector",
            "entity_name": "Semiconductors",
            "vertical": "finance",
            "question": "x y z",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 403


# --------------------------------------------------------------------------
# Earnings Decoder
# --------------------------------------------------------------------------


def test_earnings_empty_state_explains_how_to_build_a_report(client, demo_session):
    for report in demo_session.scalars(select(EarningsReport)):
        demo_session.delete(report)
    _done(demo_session)
    html = client.get("/finance/earnings").text
    assert "No Earnings Decoder reports yet" in html
    assert "newsren finance --ticker XYZ" in html and "SEC EDGAR" in html


def test_earnings_list_links_reports(client, demo_session):
    report = _irmd_report(demo_session)
    html = unescape(client.get("/finance/earnings").text)
    assert f'href="/finance/earnings/{report.id}"' in html
    for text in ("IRADIMED CORP", "IRMD", "fiscal 2026 Q2", "10-Q", "Jul 31, 2026"):
        assert text in html
    # `newsren demo` already wrote this report's earnings brief; finance briefs wait for an editor.
    assert "In editorial review" in html


def test_earnings_detail_quotes_the_filing_for_every_number(client, demo_session):
    report = _irmd_report(demo_session)
    metrics = [m for m in report.metrics if isinstance(m, dict)]
    assert metrics
    resp = client.get(f"/finance/earnings/{report.id}")
    assert resp.status_code == 200
    html = unescape(resp.text)
    for metric in metrics:
        assert metric["name"] in html
        assert format_value(metric["value"], metric["unit"]) in html
        assert f'href="{metric["source_url"]}"' in html
        assert metric["source_quote"] in html
        if metric.get("prior_value") is not None:
            assert format_value(metric["prior_value"], metric["unit"]) in html
        if metric.get("change_pct") is not None:
            assert f"{metric['change_pct']:+.1f}%" in html
    # Up/down styling, never verdicts about the results.
    assert "change-up" in html or "change-down" in html
    table = html[html.index('id="metrics-heading"') : html.index('id="guidance-heading"')]
    assert not JUDGEMENT_WORDS.search(visible_text(table))
    # No guidance in XBRL: the page says so instead of inventing a comparison.
    assert "not part of XBRL" in html
    # Filing provenance, earnings release, retrieval time and delay note.
    assert "0001628280-26-051402" in html or report.accession in html
    assert report.narrative["earnings_release"]["url"] in html
    assert "Retrieved" in html and "EDGAR data can lag filings" in html
    # Concepts for reading the report.
    for slug in ("eps", "earnings-guidance", "revenue-vs-net-income", "sec-forms-10q-10k-8k", "xbrl"):
        assert f'href="/finance/concepts/{slug}"' in html


def test_earnings_detail_guidance_comparison_is_labelled_forecast(client, demo_session):
    guidance = [
        GuidanceItem(
            metric="Revenue",
            guidance="$20.0 million to $20.5 million",
            actual="$20,466,000",
            comparison="within",
            source_url="https://www.sec.gov/Archives/edgar/data/1325618/000162828026057617/irmd-20260731.htm",
            source_quote="Revenue guidance quoted from the release.",
        )
    ]
    report = _irmd_report(demo_session, guidance=guidance)
    html = unescape(client.get(f"/finance/earnings/{report.id}").text)
    section = html[html.index('id="guidance-heading"') :]
    assert "$20.0 million to $20.5 million" in section and "Within the guided range" in section
    assert "Forecast" in section and "Revenue guidance quoted from the release." in section
    assert "not part of XBRL" not in section


def test_earnings_brief_link_only_when_approved(client, demo_session):
    report = _irmd_report(demo_session)
    brief = earnings_brief(demo_session, report)
    demo_session.commit()
    assert brief.status != BriefStatus.APPROVED
    html = client.get(f"/finance/earnings/{report.id}").text
    assert "In editorial review" in html
    assert f'href="/briefs/{brief.id}"' not in html

    _publish(demo_session, brief)
    html = unescape(client.get(f"/finance/earnings/{report.id}").text)
    assert f'href="/briefs/{brief.id}"' in html and brief.headline in html
    assert "Drafted by" in html  # provenance of the AI-drafted brief


def test_earnings_unpublished_brief_preview_link_for_editors(editor_client, demo_session):
    report = _irmd_report(demo_session)
    brief = earnings_brief(demo_session, report)
    demo_session.commit()
    html = editor_client.get(f"/finance/earnings/{report.id}").text
    assert "In editorial review" in html
    assert f'href="/briefs/{brief.id}"' in html and "Preview the unpublished draft" in html


def test_earnings_detail_missing_report_is_404(client):
    resp = client.get("/finance/earnings/999999")
    assert resp.status_code == 404
    assert client.get("/finance/earnings/not-a-number").status_code in (404, 422)


# --------------------------------------------------------------------------
# Funding Round Tracker
# --------------------------------------------------------------------------


def test_funding_shows_demo_rounds_trends_and_not_a_ranking(client, demo_session):
    rounds = demo_session.scalars(select(FundingRound)).all()
    assert rounds
    names = [r.company_name for r in rounds]
    _done(demo_session)
    resp = client.get("/finance/funding")
    assert resp.status_code == 200
    html = unescape(resp.text)
    assert "Trends from public announcements — not an investment ranking or recommendation." in html
    for name in names:
        assert name in html
    assert "By sector" in html and "By stage" in html and "Investor activity" in html
    assert "Auto-extracted" in html and "% extraction confidence" in html
    assert "Demo content" in html
    # Amount as reported plus the US-dollar figure, lead investors, source link with retrieval time.
    assert "$220 million" in html and "Orbit Growth Partners" in html
    assert 'rel="noopener noreferrer"' in html and "retrieved" in html


def test_funding_editor_verified_badge(client, demo_session):
    row = demo_session.scalars(select(FundingRound).order_by(FundingRound.id)).first()
    row.verified = True
    _done(demo_session)
    html = client.get("/finance/funding").text
    assert "Editor-verified" in html


@pytest.mark.parametrize(
    ("query", "present", "absent"),
    [
        ("sector=Edtech", ["Quillfield"], ["Tarnwell Compute", "Copperline Security"]),
        ("stage=seed", ["Copperline Security", "Tessellate Robotics"], ["Quillfield", "GridLoom"]),
        ("stage=seed&sector=Robotics", ["Tessellate Robotics"], ["Copperline Security", "Quillfield"]),
    ],
)
def test_funding_filters(client, query, present, absent):
    html = unescape(client.get(f"/finance/funding?{query}").text)
    rounds = html[html.index('id="fund-rounds-heading"') :]
    for name in present:
        assert name in rounds
    for name in absent:
        assert name not in rounds


def test_funding_bad_filters_fall_back_safely(client):
    for query in ("days=abc", "days=-5", "sector=Nope", "stage=<script>"):
        resp = client.get(f"/finance/funding?{query}")
        assert resp.status_code == 200, query
        assert "Quillfield" in resp.text
    html = client.get("/finance/funding?days=30").text
    assert '<option value="30" selected>' in html


def test_funding_htmx_partial_and_full_page(client):
    partial = client.get("/finance/funding?stage=seed", headers={"HX-Request": "true"})
    assert partial.status_code == 200
    assert partial.text.lstrip().startswith('<div id="funding-results"')
    assert "<html" not in partial.text and NOTICE not in partial.text
    assert "Copperline Security" in partial.text
    assert "HX-Request" in partial.headers.get("vary", "")

    full = client.get("/finance/funding?stage=seed")
    assert "<!doctype html>" in full.text.lower() and NOTICE in full.text
    assert 'id="funding-results"' in full.text and 'hx-get="/finance/funding"' in full.text

    restore = client.get("/finance/funding", headers={"HX-Request": "true", "HX-History-Restore-Request": "true"})
    assert "<!doctype html>" in restore.text.lower()


# --------------------------------------------------------------------------
# AI-in-Finance Reality Check
# --------------------------------------------------------------------------


def test_reality_check_explains_and_groups_by_classification(client):
    html = unescape(client.get("/finance/ai-reality-check").text)
    for label in (
        "Announcement",
        "Pilot or testing",
        "Backtest or simulation",
        "Live deployment",
        "Regulatory finding",
    ):
        assert label in html
    for key in ("announcement", "pilot", "backtest", "live_deployment", "regulatory_finding"):
        assert f'id="group-{key}"' in html
    # The plan's JPMorgan example and why the labels matter.
    assert "an in-sample result is not proof of future outperformance" in html
    # Seeded claims are marked, with source links and dates.
    assert "From the plan's research table" in html
    assert "Auto-classified from a news report" in html
    assert 'href="https://www.finra.org/media-center/blog/observations-on-ai-agents"' in html
    assert "published Jan 27, 2026" in html
    # Regulatory context and concepts.
    assert "Delphia ($225,000)" in html and 'href="/finance/concepts/backtesting"' in html


def test_reality_check_backtests_show_period_benchmark_costs_and_caveats(client):
    html = unescape(client.get("/finance/ai-reality-check").text)
    group = html[html.index('id="group-backtest"') : html.index('id="group-live_deployment"')]
    assert "JPMorgan" in group
    assert "Two-decade backtest on historical data" in group and "60/40 portfolio" in group
    assert "Not stated in the cited summary" in group
    assert "What the authors cautioned:" in group
    for label in ("Period", "Benchmark", "Costs assumed", "Result type"):
        assert label in group
    # A backtest without a stated period or benchmark says so rather than leaving it blank.
    assert "Not stated in the source" in group
    # The live deployment is not mixed in with backtests.
    assert "LLM Suite" not in group


def test_reality_check_area_filter_and_htmx(client):
    html = unescape(client.get("/finance/ai-reality-check?area=founders").text)
    claims = html[html.index('id="claims-results"') :]
    assert "Metal" in claims and "Deutsche Bank" not in claims
    assert 'aria-current="true">Entrepreneurs and founders' in claims

    partial = client.get("/finance/ai-reality-check?area=trading_markets", headers={"HX-Request": "true"})
    assert partial.text.lstrip().startswith('<div id="claims-results"')
    assert "<html" not in partial.text
    assert "Deutsche Bank" in partial.text and "Metal launched" not in partial.text

    everything = client.get("/finance/ai-reality-check?area=<bogus>")
    assert everything.status_code == 200 and "Metal" in everything.text and "Deutsche Bank" in everything.text


# --------------------------------------------------------------------------
# Regulation Watch
# --------------------------------------------------------------------------


def test_regulation_watch_shows_demo_enforcement_with_who_what_when(client):
    resp = client.get("/finance/regulation")
    assert resp.status_code == 200
    html = unescape(resp.text)
    assert "Markets Integrity Commission" in html and "Enforcement action" in html and "Settled" in html
    assert "Who is affected" in html and "What changes" in html and "When" in html
    assert "Settled without admitting or denying the findings." in html
    # A proposed rule with an estimated comment deadline and an unknown effective date.
    assert "Proposed (not final)" in html and "Comments due:" in html and "Estimated" in html
    assert "Date not yet known" in html
    # Citations carry publication and retrieval times; coverage is attached as a second source.
    assert "retrieved" in html and "Meridian Wire" in html
    assert "not legal advice" in html
    # Citation anchors stay unique across items.
    ids = re.findall(r'<li id="([^"]+)"', resp.text)
    assert ids and len(ids) == len(set(ids))


def test_regulation_window_parameter(client):
    html = client.get("/finance/regulation?days=365").text
    assert '<option value="365" selected>' in html
    assert client.get("/finance/regulation?days=zzz").status_code == 200


# --------------------------------------------------------------------------
# Concepts for learners
# --------------------------------------------------------------------------


def test_concepts_index_lists_every_concept_by_category(client):
    html = unescape(client.get("/finance/concepts").text)
    for concept in CONCEPTS:
        assert f'href="/finance/concepts/{concept.slug}"' in html
        assert concept.term in html
    for heading in ("Testing investment claims", "Startups and venture funding", "Earnings and company filings"):
        assert heading in html
    assert "not investment, legal, or tax advice" in html


def test_concept_page_explainer_example_pitfalls_related(client):
    concept = next(c for c in CONCEPTS if c.slug == "backtesting")
    html = unescape(client.get("/finance/concepts/backtesting").text)
    assert concept.term in html and concept.short in html
    for para in concept.paragraphs:
        assert para in html
    assert concept.example in html and "Hypothetical" in html
    for pitfall in concept.pitfalls:
        assert pitfall in html
    assert 'href="/finance/concepts/overfitting"' in html
    assert "No published stories use this concept yet" in html


def test_concept_alias_redirects_and_unknown_is_404(client):
    resp = client.get("/finance/concepts/drawdown", follow_redirects=False)
    assert resp.status_code == 301 and resp.headers["location"] == "/finance/concepts/maximum-drawdown"
    assert client.get("/finance/concepts/no-such-concept").status_code == 404


def test_concept_links_published_stories_only(client, demo_session):
    brief = _brief_mentioning(demo_session, "backtesting")
    headline = brief.headline
    _done(demo_session)
    html = unescape(client.get("/finance/concepts/backtesting").text)
    assert headline not in html, "unpublished briefs are not linked"

    _publish(demo_session, brief)
    html = unescape(client.get("/finance/concepts/backtesting").text)
    stories = html[html.index('id="concept-stories"') :]
    assert headline in stories and f'href="/briefs/{brief.id}"' in stories
    assert "Mentions" in stories
    index = unescape(client.get("/finance/concepts").text)
    assert "Used in 1 published story" in index


def test_finance_pages_have_no_post_routes_without_csrf(client):
    """The finance section is read-only; its only form posts to /follows, which rejects requests without CSRF."""
    token = _csrf(client)
    assert token
    assert client.post("/finance").status_code == 405
    assert client.post("/follows", data={"question": "x"}).status_code == 403
