"""Feed parsers on saved real samples (tests/data/ingest) and the HTTP fetcher (mocked)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from newsren.config import Settings
from newsren.ingest.fetchers import (
    FeedParseError,
    FetchError,
    expand_url,
    fetch_source,
    parse_arxiv,
    parse_edgar_atom,
    parse_eric,
    parse_rss,
)
from newsren.models import DocType, Source

DATA = Path(__file__).parent / "data" / "ingest"
PUBLISHER_LICENSE = "Headline + publisher summary + link; full text not stored."


def _source(
    kind: str = "rss", license_note: str = PUBLISHER_LICENSE, url: str = "https://feed.example/rss"
) -> Source:
    return Source(
        slug=f"test-{kind}",
        name="Test",
        kind=kind,
        url=url,
        vertical="technology",
        trust_tier="independent",
        default_doc_type="news",
        license_note=license_note,
    )


def _read(name: str) -> bytes:
    return (DATA / name).read_bytes()


# --------------------------------------------------------------------- RSS


def test_parse_rss_publisher_feed_keeps_summary_not_full_text():
    items = parse_rss(_read("the74_rss.xml"), _source())
    assert len(items) == 5
    first = items[0]
    assert first.title.startswith("Opinion:")
    assert first.author == "Bree Dusseault and Dana Harrison"
    assert first.published_at == datetime(2026, 10, 27, 16, 30)
    assert first.summary and "<" not in first.summary
    assert first.content == ""  # licence: headline + summary + link only
    assert "Opinion" in first.extra["tags"]
    assert all(i.url.startswith("https://www.the74million.org/") for i in items)


def test_parse_rss_keeps_body_when_licence_allows_and_strips_html():
    items = parse_rss(_read("the74_rss.xml"), _source(license_note="Public domain US government record."))
    assert any(len(i.content) > 500 for i in items)
    for item in items:
        assert "<p>" not in item.content and "<a " not in item.content
        assert "appeared first on" not in item.content


def test_parse_rss_without_source_never_keeps_body():
    assert all(i.content == "" for i in parse_rss(_read("the74_rss.xml")))


def test_parse_rss_strips_html_titles_and_two_digit_years():
    items = parse_rss(_read("finra_news_rss.xml"), _source())
    assert len(items) == 5
    assert items[0].title.startswith("FINRA Makes Investor Education Content Available")
    assert all("<a" not in i.title for i in items)
    assert all(i.published_at and i.published_at.year == 2026 for i in items)
    assert any("$800,000" in i.title for i in items)


def test_parse_rss_bbc_without_author():
    items = parse_rss(_read("bbc_technology_rss.xml"), _source())
    assert len(items) == 5
    assert all(i.author is None for i in items)
    assert "at_medium=RSS" in items[0].url  # tracking params are dropped by canonical_url at storage


def test_parse_rss_cms_bylines_replace_account_names():
    items = parse_rss(_read("insidehighered_rss.xml"), _source())
    assert len(items) == 5  # feed has a stray entity (bozo) but entries still parse
    authors = [i.author for i in items]
    assert authors[0] == "Nancy L. Thomas"
    assert all(a is None or "@" not in a for a in authors)
    assert items[0].summary == "Consider this checklist for the days—and weeks—after."
    assert "/opinion/" in items[0].url


def test_parse_rss_flags_sponsored_posts():
    items = parse_rss(_read("ieee_spectrum_rss.xml"), _source())
    sponsored = [i for i in items if i.extra.get("sponsored")]
    assert sponsored and sponsored[0].author == "VicOne"
    assert not all(i.extra.get("sponsored") for i in items)


def test_parse_rss_removes_feed_boilerplate():
    items = parse_rss(_read("hechinger_rss.xml"), _source())
    assert len(items) == 2
    assert all("appeared first on" not in i.summary for i in items)
    assert "originally published by Barn Raiser" in items[0].summary


def test_parse_rss_rejects_garbage():
    with pytest.raises(FeedParseError):
        parse_rss(b"<html><body>Access denied</body", _source())


# --------------------------------------------------------------------- arXiv


def test_parse_arxiv_preprints():
    items = parse_arxiv(_read("arxiv_cs_ai.atom"))
    assert len(items) == 5
    paper = items[0]
    assert paper.doc_type == DocType.RESEARCH_PREPRINT
    assert paper.url == "https://arxiv.org/abs/2610.35868"
    assert paper.extra["arxiv_id"] == "2610.35868" and paper.extra["arxiv_version"] == "v1"
    assert paper.extra["peer_review_status"] == "unknown"
    assert "not peer reviewed" in paper.extra["peer_review_note"]
    assert paper.extra["primary_category"] == "cs.AI"
    assert paper.author.startswith("Chao Wan, ")
    assert paper.published_at == datetime(2026, 10, 26, 17, 59, 44)
    assert "DecisionRAG" in paper.summary


def test_parse_arxiv_api_error_raises():
    error_feed = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>http://arxiv.org/api/errors#incorrect_id_format</id><title>Error</title>
      <summary>incorrect id format</summary></entry></feed>"""
    with pytest.raises(FeedParseError, match="arXiv API error"):
        parse_arxiv(error_feed)


# --------------------------------------------------------------------- EDGAR


def test_parse_edgar_8k_filings():
    items = parse_edgar_atom(_read("edgar_current_8k.atom"))
    assert len(items) == 6
    first = items[0]
    assert first.doc_type == DocType.FILING
    assert first.extra["company"] == "TRAVELZOO"
    assert first.extra["cik"] == "0001133311"
    assert first.extra["form_type"] == "8-K"
    assert first.extra["accession"] == "0001133311-26-000019"
    assert first.extra["is_earnings_release"] is True  # Item 2.02 Results of Operations
    assert first.extra["items"][0] == {"number": "2.02", "title": "Results of Operations and Financial Condition"}
    assert "acceptance" in first.extra["data_delay_note"]
    assert first.published_at == datetime(2026, 10, 27, 18, 43, 36)  # 14:43:36 -04:00 in UTC
    assert (
        first.title == "TRAVELZOO files Form 8-K (current report): Results of Operations and Financial Condition"
    )
    assert "accession number 0001133311-26-000019" in first.summary
    assert first.url.startswith("https://www.sec.gov/Archives/edgar/data/1133311/")


def test_parse_edgar_10q_filings():
    items = parse_edgar_atom(_read("edgar_current_10q.atom"))
    assert len(items) == 5
    assert {i.extra["form_type"] for i in items} == {"10-Q"}
    assert all("quarterly report" in i.title for i in items)
    assert all(len(i.extra["cik"]) == 10 for i in items)


def test_parse_edgar_dedupes_filer_roles():
    entry = """<entry><title>8-K - ACME WIDGETS INC (0000000123) ({role})</title>
      <link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/123/x-index.htm"/>
      <summary type="html">&lt;b&gt;Filed:&lt;/b&gt; 2026-10-20 &lt;b&gt;AccNo:&lt;/b&gt; 0000000123-26-000001</summary>
      <updated>2026-10-20T10:00:00-04:00</updated>
      <category scheme="https://www.sec.gov/" label="form type" term="8-K"/>
      <id>urn:tag:sec.gov,2008:accession-number=0000000123-26-000001</id></entry>"""
    feed = f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">{entry.format(role="Filer")}{entry.format(role="Subject")}</feed>'
    items = parse_edgar_atom(feed)
    assert len(items) == 1 and items[0].extra["filer_role"] == "Filer"
    assert items[0].extra["items"] == [] and items[0].extra["is_earnings_release"] is False


# --------------------------------------------------------------------- ERIC


def test_parse_eric_records():
    items = parse_eric(_read("eric_ai.json"))
    assert len(items) == 5
    first = items[0]
    assert first.url == "https://eric.ed.gov/?id=EJ1464545"
    assert first.doc_type == DocType.RESEARCH_PEER_REVIEWED
    assert first.published_at is None  # ERIC gives a year only
    assert first.extra["publication_year"] == 2025 and first.extra["date_precision"] == "year"
    assert first.extra["peer_reviewed"] is True
    assert first.summary and first.author


def test_parse_eric_peer_review_flag_and_opinion_papers():
    payload = {
        "response": {
            "docs": [
                {
                    "id": "ED1",
                    "title": "District report on AI pilots",
                    "peerreviewed": "F",
                    "publicationdateyear": 2026,
                },
                {
                    "id": "EJ2",
                    "title": "Teachers&apos; views",
                    "peerreviewed": "T",
                    "publicationtype": ["Journal Articles"],
                },
                {"id": "ED3", "title": "Why schools should slow down", "publicationtype": ["Opinion Papers"]},
                {"title": "missing id is skipped"},
            ]
        }
    }
    items = parse_eric(json.dumps(payload))
    assert [i.doc_type for i in items] == [
        DocType.EDUCATION_REPORT,
        DocType.RESEARCH_PEER_REVIEWED,
        DocType.OPINION,
    ]
    assert items[1].title == "Teachers' views"
    with pytest.raises(FeedParseError):
        parse_eric(b'{"error": "bad query"}')


# --------------------------------------------------------------------- fetch_source


def _settings(**kw) -> Settings:
    return Settings(contact_email="desk@newsren.example", fetch_timeout_seconds=5.0, **kw)


def test_expand_url_placeholders():
    url = "https://api.example/q?rows={max_items}&count={edgar_count}&years=[{prev_year} TO {year}]"
    expanded = expand_url(url, _settings(max_items_per_source=30), now=datetime(2026, 10, 27))
    assert expanded == "https://api.example/q?rows=30&count=40&years=[2025 TO 2026]"
    assert "count=100" in expand_url(url, _settings(max_items_per_source=500))


def test_parsers_never_fetch_url_strings():
    # A str argument is parsed as a document, never fetched as a URL by feedparser.
    with pytest.raises(FeedParseError):
        parse_rss("https://www.sec.gov/news/pressreleases.rss", _source())


def test_fetch_source_sends_user_agent_and_respects_item_limit():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers["user-agent"]
        seen["url"] = str(request.url)
        return httpx.Response(200, content=_read("arxiv_cs_ai.atom"))

    source = _source(kind="arxiv", url="https://export.arxiv.org/api/query?max_results={max_items}")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        items = fetch_source(source, _settings(max_items_per_source=3), client=client)
    assert len(items) == 3
    assert "desk@newsren.example" in seen["ua"]
    assert seen["url"].endswith("max_results=3")


@pytest.mark.parametrize(
    ("handler", "message"),
    [
        (lambda request: httpx.Response(503), "HTTP 503"),
        (lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused")), "network error"),
        (lambda request: httpx.Response(200, content=b"not a feed <<<"), "unparseable"),
    ],
)
def test_fetch_source_wraps_failures_in_fetch_error(handler, message):
    with httpx.Client(transport=httpx.MockTransport(handler)) as client, pytest.raises(FetchError, match=message):
        fetch_source(_source(), _settings(), client=client)


def test_fetch_source_rejects_unknown_kind():
    with pytest.raises(FetchError, match="unsupported source kind"):
        fetch_source(_source(kind="demo"), _settings())
