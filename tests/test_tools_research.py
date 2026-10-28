"""Ask Newsren retrieval tools: strict schemas, archive search, approved fetch, arXiv and SEC parsers.

No network: arXiv/SEC tests replay saved responses from tests/data/research/
through a patched HTTP layer, and the fetch tests use httpx.MockTransport.
"""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from newsren.agents import tools
from newsren.agents.tools import (
    HttpResponse,
    ResearchToolbox,
    arxiv_query_params,
    build_research_tools,
    parse_company_tickers,
    recent_material_filings,
    resolve_company,
)
from newsren.llm import Budget, ToolRunner
from newsren.models import DocType, TrustTier, Vertical, utcnow

DATA = Path(__file__).parent / "data" / "research"
NVDA_CIK = "0001045810"


@pytest.fixture(autouse=True)
def _fresh_caches():
    tools.clear_caches()
    yield
    tools.clear_caches()


def _walk(schema: Any, path: str = "$"):
    """Yield (path, node) for every dict in a JSON schema."""
    if isinstance(schema, dict):
        yield path, schema
        for key, value in schema.items():
            yield from _walk(value, f"{path}.{key}")
    elif isinstance(schema, list):
        for i, value in enumerate(schema):
            yield from _walk(value, f"{path}[{i}]")


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


def test_tool_schemas_are_strict_valid(session, settings):
    built = build_research_tools(session, settings, allow_network=False)
    assert [t.name for t in built] == [
        "search_archive",
        "get_article",
        "search_research_papers",
        "search_sec_filings",
        "fetch_approved_url",
    ]
    for tool in built:
        api = tool.to_api()
        assert api["strict"] is True
        assert len(tool.description) > 80
        for path, node in _walk(tool.input_schema):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, f"{tool.name} {path}"
                assert sorted(node.get("required", [])) == sorted(node["properties"]), f"{tool.name} {path}"
            # Strict tool schemas reject numeric and string-length constraints.
            for unsupported in ("minimum", "maximum", "minLength", "maxLength", "pattern", "default"):
                assert unsupported not in node, f"{tool.name} {path} uses {unsupported}"


def test_optional_inputs_are_nullable(session, settings):
    schema = build_research_tools(session, settings)[0].input_schema
    for name in ("vertical", "since_days"):
        variants = schema["properties"][name]["anyOf"]
        assert {"type": "null"} in variants
    assert schema["properties"]["vertical"]["anyOf"][0]["enum"] == [v.value for v in Vertical]


# --------------------------------------------------------------------------
# search_archive / get_article
# --------------------------------------------------------------------------


@pytest.fixture()
def archive(factory):
    tech = factory.source(vertical=Vertical.TECHNOLOGY, name="Tech Daily")
    fin = factory.source(vertical=Vertical.FINANCE, name="Money Wire")
    banned = factory.source(vertical=Vertical.TECHNOLOGY, name="Unapproved Blog", approved=False)
    now = utcnow()
    original = factory.article(
        "Quantum chip maker posts new benchmark",
        "Qubitron said its quantum chip set a benchmark record. Independent labs have not yet checked it.",
        source=tech,
        published_at=now - timedelta(days=2),
    )
    duplicate = factory.article(
        "Quantum chip maker posts new benchmark (syndicated)",
        "Qubitron said its quantum chip set a benchmark record. Reprinted with permission, syndicatedcredit.",
        source=fin,
        published_at=now - timedelta(days=1),
    )
    duplicate.duplicate_of_id = original.id
    finance = factory.article(
        "Investors weigh quantum computing startups",
        "Funding for quantum startups rose; analysts cited the benchmark claims.",
        source=fin,
        published_at=now - timedelta(days=3),
    )
    old = factory.article(
        "Quantum benchmark history",
        "An older look at quantum benchmark debates.",
        source=tech,
        published_at=now - timedelta(days=60),
    )
    hidden = factory.article(
        "Quantum benchmark rumor",
        "A quantum benchmark rumor from an unapproved site.",
        source=banned,
        published_at=now - timedelta(days=1),
    )
    factory.session.flush()
    return {"original": original, "duplicate": duplicate, "finance": finance, "old": old, "hidden": hidden}


def _ids(result) -> list[int]:
    return [row["article_id"] for row in result.data]


def test_search_archive_filters_and_skips_duplicates(session, settings, archive):
    box = ResearchToolbox(session, settings, allow_network=False)
    result = box.search_archive({"query": "quantum benchmark", "vertical": None, "since_days": None})
    assert not result.is_error
    ids = _ids(result)
    assert archive["original"].id in ids and archive["finance"].id in ids and archive["old"].id in ids
    assert archive["duplicate"].id not in ids, "syndicated copies collapse into the original"
    assert archive["hidden"].id not in ids, "unapproved sources are never searched"
    assert len(ids) == len(set(ids))

    # Numbered, human-readable results with handles, metadata, and citations.
    assert result.content.splitlines()[0].startswith('Archive search for "quantum benchmark": 3 results')
    assert f"[1] [A{ids[0]}]" in result.content
    assert "trust tier:" in result.content and "URL: https://" in result.content and "Snippet:" in result.content
    assert [c.id for c in result.citations] == [f"A{i}" for i in ids]
    assert all(c.article_id in ids and c.url.startswith("https://") for c in result.citations)

    tech_only = box.search_archive({"query": "quantum benchmark", "vertical": "technology", "since_days": None})
    assert set(_ids(tech_only)) == {archive["original"].id, archive["old"].id}
    recent = box.search_archive({"query": "quantum benchmark", "vertical": "technology", "since_days": 30})
    assert _ids(recent) == [archive["original"].id]
    assert "last 30 days" in recent.content


def test_search_archive_maps_duplicate_only_hits_to_original(session, settings, archive):
    box = ResearchToolbox(session, settings, allow_network=False)
    result = box.search_archive({"query": "syndicatedcredit", "vertical": None, "since_days": None})
    assert _ids(result) == [archive["original"].id]


def test_search_archive_caps_results_and_validates_input(session, settings, factory):
    src = factory.source()
    for i in range(12):
        factory.article(f"Robotics update {i}", f"Warehouse robotics news item number {i}.", source=src)
    box = ResearchToolbox(session, settings, allow_network=False)
    result = box.search_archive({"query": "warehouse robotics", "vertical": None, "since_days": None})
    assert len(result.data) == tools.ARCHIVE_RESULT_LIMIT
    assert box.search_archive({"query": "the of and what", "vertical": None, "since_days": None}).is_error
    assert box.search_archive({"query": "robotics", "vertical": "sports", "since_days": None}).is_error
    empty = box.search_archive({"query": "zebrafish", "vertical": None, "since_days": None})
    assert not empty.is_error and empty.data == [] and "No archived article matches" in empty.content


def test_get_article_returns_labelled_truncated_text(session, settings, factory, archive):
    preprint_src = factory.source(
        trust_tier=TrustTier.PRIMARY, doc_type=DocType.RESEARCH_PREPRINT, name="Preprints"
    )
    long_body = " ".join(f"Sentence {i} about tutoring outcomes in algebra classes." for i in range(300))
    paper = factory.article("A tutoring study", long_body, source=preprint_src)
    box = ResearchToolbox(session, settings, allow_network=False)

    result = box.get_article({"article_id": paper.id})
    assert not result.is_error
    assert result.content.startswith(f"[A{paper.id}] A tutoring study")
    assert "preprint" in result.content.lower() and "not peer reviewed" in result.content.lower()
    assert "Text truncated to 6000" in result.content
    assert len(result.data["text"]) <= tools.ARTICLE_TEXT_LIMIT
    assert result.citations[0].id == f"A{paper.id}" and result.citations[0].doc_type == DocType.RESEARCH_PREPRINT
    assert box.text_for(paper.url), "retrieved text is recorded for quote checks"

    dup = box.get_article({"article_id": archive["duplicate"].id})
    assert dup.data["article_id"] == archive["original"].id and "syndicated copy" in dup.content
    assert box.get_article({"article_id": 999999}).is_error
    assert box.get_article({"article_id": archive["hidden"].id}).is_error


def test_tools_record_trace_through_runner(session, settings, archive):
    runner = ToolRunner(build_research_tools(session, settings, allow_network=False), Budget(max_tool_calls=3))
    runner.call("search_archive", {"query": "quantum", "vertical": None, "since_days": None})
    runner.call("get_article", {"article_id": archive["original"].id})
    assert [t["tool"] for t in runner.trace] == ["search_archive", "get_article"]
    assert archive["original"].url in runner.citations


# --------------------------------------------------------------------------
# Network tools disabled
# --------------------------------------------------------------------------


def test_network_tools_return_errors_when_network_disabled(session, settings, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network must not be used")

    monkeypatch.setattr(tools, "http_get", no_network)
    box = ResearchToolbox(session, settings, allow_network=False)
    for result in (
        box.search_research_papers({"query": "multi-agent trading"}),
        box.search_sec_filings({"company_or_ticker": "NVDA"}),
        box.fetch_approved_url({"url": "https://www.sec.gov/cgi-bin/browse-edgar"}),
    ):
        assert result.is_error
        assert "network retrieval is disabled" in result.content
    refused = box.fetch_approved_url({"url": "https://unknown-site.example.net/page"})
    assert refused.is_error and "not an approved source" in refused.content


# --------------------------------------------------------------------------
# fetch_approved_url (httpx mocked)
# --------------------------------------------------------------------------


@pytest.fixture()
def mock_http(monkeypatch):
    """Route tools' httpx.Client through a MockTransport; returns the list of requests made."""
    requests: list[httpx.Request] = []
    routes: dict[str, httpx.Response] = {}
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = routes.get(str(request.url))
        if response is None:
            return httpx.Response(404, text="not found")
        return response

    def client_factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(tools.httpx, "Client", client_factory)
    return requests, routes


def test_fetch_refuses_unapproved_domain_without_requesting(session, settings, mock_http):
    requests, _ = mock_http
    box = ResearchToolbox(session, settings, allow_network=True)
    result = box.fetch_approved_url({"url": "https://evil.example.net/steal"})
    assert result.is_error and "not an approved source" in result.content
    assert box.fetch_approved_url({"url": "ftp://sec.gov/file"}).is_error
    assert requests == []


def test_fetch_approved_page_strips_html(session, settings, mock_http):
    requests, routes = mock_http
    url = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000075/nvda-20260726.htm"
    html = (
        "<html><head><title>NVIDIA 10-Q</title><script>var tracking = 1;</script></head><body>"
        "<nav>Home | Search EDGAR</nav><main><h1>Quarterly report</h1>"
        "<p>Revenue for the quarter was reported in the condensed consolidated statements of income, "
        "which present results for the three and six months ended July 26, 2026.</p>"
        "<p>Risk factors are described in Part II, Item 1A, and should be read with the financial "
        "statements and the notes that accompany them.</p></main><footer>Contact SEC</footer></body></html>"
    )
    routes[url] = httpx.Response(200, html=html)
    box = ResearchToolbox(session, settings, allow_network=True)
    result = box.fetch_approved_url({"url": url})
    assert not result.is_error, result.content
    assert "Revenue for the quarter" in result.content and "Risk factors" in result.content
    assert "tracking" not in result.content and "Search EDGAR" not in result.content
    citation = result.citations[0]
    assert citation.url == url and citation.title == "NVIDIA 10-Q"
    assert citation.trust_tier == TrustTier.PRIMARY and citation.doc_type == DocType.FILING
    assert requests[0].headers["user-agent"] == settings.user_agent
    assert "Revenue for the quarter" in box.text_for(url)


def test_fetch_rechecks_redirects(session, settings, mock_http):
    requests, routes = mock_http
    start = "https://www.finra.org/rules-guidance/notices/26-01"
    routes[start] = httpx.Response(302, headers={"location": "https://tracker.example.net/landing"})
    routes["https://tracker.example.net/landing"] = httpx.Response(200, text="should never be read")
    box = ResearchToolbox(session, settings, allow_network=True)
    result = box.fetch_approved_url({"url": start})
    assert result.is_error and "not an approved source" in result.content
    assert [str(r.url) for r in requests] == [start]


def test_fetch_rejects_pdf_and_http_errors(session, settings, mock_http):
    _, routes = mock_http
    pdf = "https://www.federalreserve.gov/report.pdf"
    routes[pdf] = httpx.Response(200, content=b"%PDF-1.7 binary", headers={"content-type": "application/pdf"})
    box = ResearchToolbox(session, settings, allow_network=True)
    assert "PDF" in box.fetch_approved_url({"url": pdf}).content
    missing = box.fetch_approved_url({"url": "https://www.federalreserve.gov/missing.htm"})
    assert missing.is_error and "HTTP 404" in missing.content


def test_fetch_allows_registered_source_domain(session, settings, factory, mock_http):
    src = factory.source(name="Tech Daily")
    _, routes = mock_http
    url = f"{src.homepage}/stories/today"
    routes[url] = httpx.Response(
        200, text="Plain text story about chips. " * 5, headers={"content-type": "text/plain"}
    )
    result = ResearchToolbox(session, settings, allow_network=True).fetch_approved_url({"url": url})
    assert not result.is_error
    assert result.citations[0].publisher == "Tech Daily"


# --------------------------------------------------------------------------
# arXiv (saved samples)
# --------------------------------------------------------------------------


def _fake_http(responses: dict[str, bytes], calls: list[str]):
    def fake(url, settings, **kwargs):
        calls.append(url)
        for prefix, body in responses.items():
            if url.startswith(prefix):
                return HttpResponse(url=url, status_code=200, content_type="application/xml", content=body)
        raise AssertionError(f"unexpected URL {url}")

    return fake


def test_arxiv_tool_parses_saved_sample(session, settings, monkeypatch):
    calls: list[str] = []
    sample = (DATA / "arxiv_tradingagents.xml").read_bytes()
    monkeypatch.setattr(tools, "http_get", _fake_http({tools.ARXIV_API_URL: sample}, calls))
    box = ResearchToolbox(session, settings, allow_network=True)
    result = box.search_research_papers({"query": "TradingAgents multi-agent trading"})
    assert not result.is_error, result.content
    assert calls[0].startswith("https://export.arxiv.org/api/query?search_query=")
    assert "search_query=all%3Atradingagents+AND+all%3A%22multi-agent%22" in calls[0]
    assert "[1] [arXiv:2412.20138] TradingAgents: Multi-Agents LLM Financial Trading Framework" in result.content
    assert "not peer reviewed" in result.content and "Abstract:" in result.content
    first = result.citations[0]
    assert first.id == "arXiv:2412.20138" and first.url == "https://arxiv.org/abs/2412.20138"
    assert first.doc_type == DocType.RESEARCH_PREPRINT and first.trust_tier == TrustTier.PRIMARY
    assert first.publisher == "arXiv" and first.published_at.startswith("2024-12-")
    assert result.data[0]["authors"][0] == "Yijia Xiao"
    assert "multi-agent" in box.text_for(first.url).lower()


def test_arxiv_query_params():
    assert arxiv_query_params("arXiv:2412.20138v7") == {"id_list": "2412.20138", "max_results": 1}
    assert (
        arxiv_query_params("ti:TradingAgents AND cat:q-fin.TR")["search_query"]
        == "ti:TradingAgents AND cat:q-fin.TR"
    )
    params = arxiv_query_params("What does the Lumen-4 technical report say about math?")
    assert params["search_query"] == 'all:"lumen-4" AND all:technical AND all:math'
    assert params["sortBy"] == "relevance"


def test_arxiv_error_and_broadening(session, settings, monkeypatch):
    calls: list[str] = []
    error = (DATA / "arxiv_error.xml").read_bytes()
    monkeypatch.setattr(tools, "http_get", _fake_http({tools.ARXIV_API_URL: error}, calls))
    box = ResearchToolbox(session, settings, allow_network=True)
    failed = box.search_research_papers({"query": "2412.99999"})
    assert failed.is_error and "incorrect id format" in failed.content

    empty, full = (DATA / "arxiv_empty.xml").read_bytes(), (DATA / "arxiv_tradingagents.xml").read_bytes()
    bodies = iter([empty, full])
    calls.clear()

    def sequenced(url, settings, **kwargs):
        calls.append(url)
        return HttpResponse(url=url, status_code=200, content_type="application/xml", content=next(bodies))

    monkeypatch.setattr(tools, "http_get", sequenced)
    result = box.search_research_papers({"query": "multi-agent trading framework zzqv"})
    assert not result.is_error and len(calls) == 2
    assert "broadened" in result.content and result.citations


# --------------------------------------------------------------------------
# SEC EDGAR (saved samples)
# --------------------------------------------------------------------------


def _sec_fake(calls: list[str]):
    tickers = (DATA / "sec_company_tickers_sample.json").read_bytes()
    submissions = (DATA / "sec_submissions_nvda_sample.json").read_bytes()

    def fake(url, settings, **kwargs):
        calls.append(url)
        assert settings.user_agent.startswith("TheNewsren/")
        if url == tools.SEC_TICKERS_URL:
            body = tickers
        elif url == tools.SEC_SUBMISSIONS_URL.format(cik=NVDA_CIK):
            body = submissions
        else:
            return HttpResponse(url=url, status_code=404, content_type="text/plain", content=b"")
        return HttpResponse(url=url, status_code=200, content_type="application/json", content=body)

    return fake


def test_sec_filings_tool_on_saved_samples(session, settings, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(tools, "http_get", _sec_fake(calls))
    box = ResearchToolbox(session, settings, allow_network=True)
    result = box.search_sec_filings({"company_or_ticker": "NVDA"})
    assert not result.is_error, result.content
    assert calls == [tools.SEC_TICKERS_URL, tools.SEC_SUBMISSIONS_URL.format(cik=NVDA_CIK)]
    assert "NVIDIA CORP" in result.content and f"CIK {NVDA_CIK}" in result.content
    assert "Form 10-Q (quarterly report) filed 2026-08-26, period ending 2026-07-26" in result.content
    assert "2.02 Results of Operations and Financial Condition" in result.content
    assert "Form 4 " not in result.content and "insider transactions" in result.content
    ten_q = next(c for c in result.citations if "10-Q" in c.title)
    assert ten_q.url == "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000075/nvda-20260726.htm"
    assert ten_q.id == "SEC:0001045810-26-000075"
    assert ten_q.doc_type == DocType.FILING and ten_q.trust_tier == TrustTier.PRIMARY
    assert ten_q.published_at.startswith("2026-08-26")
    assert all(f["form"] in tools.MATERIAL_FORMS for f in result.data["filings"])

    # The ticker file is cached: a second lookup by company name only fetches submissions.
    by_name = box.search_sec_filings({"company_or_ticker": "Nvidia"})
    assert not by_name.is_error and calls.count(tools.SEC_TICKERS_URL) == 1


def test_sec_filings_unknown_company(session, settings, monkeypatch):
    monkeypatch.setattr(tools, "http_get", _sec_fake([]))
    result = ResearchToolbox(session, settings, allow_network=True).search_sec_filings(
        {"company_or_ticker": "Corvane Semiconductor"}
    )
    assert result.is_error and "no company matching" in result.content


def test_resolve_company_by_ticker_cik_and_name():
    import json

    records = parse_company_tickers(json.loads((DATA / "sec_company_tickers_sample.json").read_text()))
    assert resolve_company(records, "$nvda")[0].cik == NVDA_CIK
    assert resolve_company(records, "brk.b")[0].ticker == "BRK-B"
    assert resolve_company(records, "CIK 19617")[0].title == "JPMORGAN CHASE & CO"
    assert resolve_company(records, "320193")[0].ticker == "AAPL"
    assert resolve_company(records, "Goldman Sachs")[0].ticker == "GS"
    jpm = resolve_company(records, "JPMorgan Chase & Co.")
    assert [r.cik for r in jpm] == ["0000019617"], "share classes of one filer collapse to one CIK"
    assert resolve_company(records, "Deutsche Bank")[0].ticker == "DB"
    assert resolve_company(records, "Corvane") == []


def test_recent_material_filings_skips_routine_forms():
    import json

    submissions = json.loads((DATA / "sec_submissions_nvda_sample.json").read_text())
    filings, skipped = recent_material_filings(submissions, NVDA_CIK, limit=3)
    assert len(filings) == 3
    assert [f["filed"] for f in filings] == sorted((f["filed"] for f in filings), reverse=True)
    assert skipped["4"] > 10 and "10-Q" not in skipped
    assert filings[0]["index_url"].startswith("https://www.sec.gov/Archives/edgar/data/1045810/")


def test_arxiv_requests_are_spaced_and_a_429_is_retried_once(monkeypatch):
    import newsren.agents.tools as tools
    from newsren.config import Settings

    calls: list[float] = []
    statuses = iter([429, 200])

    def fake_get(url, settings, **kwargs):
        calls.append(time.monotonic())
        return tools.HttpResponse(
            url=url, status_code=next(statuses), content=b"", content_type="", truncated=False
        )

    monkeypatch.setattr(tools, "http_get", fake_get)
    monkeypatch.setattr(tools, "ARXIV_MIN_INTERVAL", 0.2)
    monkeypatch.setattr(tools, "ARXIV_RETRY_WAIT", 0.0)
    monkeypatch.setattr(tools, "_arxiv_last_request", 0.0)
    response = tools._arxiv_get("https://export.arxiv.org/api/query?x=1", Settings())
    assert response.status_code == 200
    assert len(calls) == 2
    assert calls[1] - calls[0] >= 0.19  # the retry waited for the minimum interval
