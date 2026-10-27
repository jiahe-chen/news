"""Approved retrieval tools for the Ask Newsren research agent (reader loop).

The agent never browses freely. It gets five tools, each limited to an
approved source (plan §6: "Restrict agent access to approved source APIs and
retrieval tools"):

* ``search_archive`` — full-text search over articles The Newsren ingested
  from approved, active sources. Syndicated copies collapse onto the original.
* ``get_article`` — the (truncated) text of one archived article.
* ``search_research_papers`` — the arXiv API, an approved research index.
  Every result is labelled a preprint: arXiv does not peer review.
* ``search_sec_filings`` — SEC EDGAR company submissions (data.sec.gov),
  resolved from a ticker, CIK, or company name via SEC's ticker file.
* ``fetch_approved_url`` — one document, only when
  ``newsren.ingest.sources.is_approved_url`` allows its domain; redirects are
  followed by hand and every hop is re-checked.

Each handler returns a ``ToolResult``: numbered, human-readable text for the
model (handle, title, publisher, dates, source type, trust tier, URL, snippet)
plus ``CitationRef``s for provenance and a structured ``data`` payload for
offline builders. Handles: ``A<article_id>`` for archive articles,
``arXiv:<id>`` for papers, ``SEC:<accession>`` for filings, and the URL for
fetched pages. ``ResearchToolbox.retrieved_text`` records what each source
said, so the research agent can check the quotes a model attaches to its
citations.

Network tools honour ``allow_network``: when it is False (offline mode,
tests) they return an error result explaining that network retrieval is
disabled. All input schemas are strict (``additionalProperties: false``,
every property required, optional inputs nullable via ``anyOf`` with
``null``) because ``Tool.to_api`` sends ``strict: true``; numeric bounds are
stated in descriptions and enforced in the handlers, since strict schemas do
not support ``minimum``/``maximum``.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from newsren.agents.base import article_text, citation_for
from newsren.config import Settings, get_settings
from newsren.db import search_article_ids
from newsren.ingest.fetchers import FeedParseError, FetchedItem, parse_arxiv
from newsren.ingest.sources import is_approved_url
from newsren.llm import Tool, ToolResult
from newsren.models import Article, DocType, Source, TrustTier, Vertical, utcnow
from newsren.schemas import CitationRef
from newsren.textutil import (
    best_snippet,
    canonical_url,
    domain_of,
    iso,
    normalize_space,
    parse_datetime,
    split_sentences,
    strip_html,
    tokenize,
    truncate,
)

ARCHIVE_RESULT_LIMIT = 8
ARCHIVE_CANDIDATES = 80
ARTICLE_TEXT_LIMIT = 6000
FETCH_TEXT_LIMIT = 6000
PAPER_RESULT_LIMIT = 6
FILING_RESULT_LIMIT = 10
SNIPPET_CHARS = 280
MAX_SINCE_DAYS = 3650
MAX_FETCH_BYTES = 3_000_000
MAX_JSON_BYTES = 40_000_000  # SEC submissions for large filers run to several MB
MAX_REDIRECTS = 5
SEC_TICKER_CACHE_SECONDS = 24 * 3600

ARXIV_API_URL = "https://export.arxiv.org/api/query"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

VERTICALS: tuple[str, ...] = tuple(v.value for v in Vertical)

DOC_TYPE_LABELS: dict[str, str] = {
    DocType.NEWS: "news report",
    DocType.PRESS_RELEASE: "press release (the organization's own account)",
    DocType.RESEARCH_PREPRINT: "preprint, not peer reviewed",
    DocType.RESEARCH_PEER_REVIEWED: "peer-reviewed research",
    DocType.FILING: "regulatory filing",
    DocType.EARNINGS_CALL: "earnings call transcript",
    DocType.REGULATORY: "regulator release",
    DocType.POLICY: "policy document",
    DocType.EDUCATION_REPORT: "education report",
    DocType.ADVISORY: "security advisory",
    DocType.OPINION: "opinion piece",
    DocType.BLOG: "blog post",
}

TRUST_TIER_LABELS: dict[str, str] = {
    TrustTier.PRIMARY: "primary source (the original record)",
    TrustTier.INDEPENDENT: "independent reporting or evaluation",
    TrustTier.VENDOR: "vendor (a company describing itself)",
    TrustTier.COMMENTARY: "commentary",
}

# Words that make a reader's question a question but say nothing about its
# topic; dropping them keeps archive searches and term checks focused.
QUESTION_FILLER = frozenset(
    """
    tell explain explained happen happened happening happens know think really mean means meant thing things
    anything something everything latest recent recently current currently update updates story stories
    article articles newsren please exactly actually going does did done doing use used using way ways kind
    part matter matters mattered important need needs want wants show shows showed give given find found
    sure question questions answer anyone someone people lot lots
    """.split()
)

# Primary-record domains on the allow-list and what they publish.
_PRIMARY_DOMAIN_TYPES: tuple[tuple[str, str], ...] = (
    ("sec.gov", DocType.FILING),
    ("arxiv.org", DocType.RESEARCH_PREPRINT),
    ("federalreserve.gov", DocType.REGULATORY),
    ("finra.org", DocType.REGULATORY),
    ("ies.ed.gov", DocType.EDUCATION_REPORT),
    ("nces.ed.gov", DocType.EDUCATION_REPORT),
    ("ed.gov", DocType.POLICY),
    ("cisa.gov", DocType.ADVISORY),
    ("nist.gov", DocType.REGULATORY),
)

# SEC forms that carry company news; routine forms (insider Forms 3/4/5,
# Form 144, prospectus supplements) are counted but not listed.
MATERIAL_FORMS: dict[str, str] = {
    "10-K": "annual report",
    "10-K/A": "amended annual report",
    "10-KT": "transition annual report",
    "10-Q": "quarterly report",
    "10-Q/A": "amended quarterly report",
    "8-K": "current report",
    "8-K/A": "amended current report",
    "20-F": "foreign private issuer annual report",
    "20-F/A": "amended foreign private issuer annual report",
    "40-F": "Canadian issuer annual report",
    "6-K": "foreign private issuer current report",
    "DEF 14A": "proxy statement",
    "S-1": "registration statement",
    "S-1/A": "amended registration statement",
    "F-1": "foreign issuer registration statement",
    "S-3": "shelf registration statement",
    "S-4": "registration statement for a merger or exchange offer",
    "ARS": "annual report to shareholders",
    "11-K": "employee benefit plan annual report",
}

EIGHT_K_ITEMS: dict[str, str] = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.02": "Departure or Election of Directors or Officers; Compensation Arrangements",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}

_ROUTINE_FORM_LABELS: dict[str, str] = {
    "3": "insider ownership",
    "4": "insider transactions",
    "5": "insider ownership",
    "144": "proposed insider sales",
}

_ARXIV_ID = re.compile(r"^(?:arxiv:\s*)?(\d{4}\.\d{4,5})(v\d+)?$", re.IGNORECASE)
_ARXIV_FIELD = re.compile(r"\b(?:ti|au|abs|co|jr|cat|rn|id|all):", re.IGNORECASE)
_HTML_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_BOILERPLATE_BLOCKS = re.compile(r"(?is)<(nav|footer|noscript|svg|form|aside|script|style)\b.*?</\1\s*>")
_MAIN_BLOCK = re.compile(r"(?is)<main\b[^>]*>(.*)</main\s*>")
_ARTICLE_BLOCK = re.compile(r"(?is)<article\b[^>]*>(.*)</article\s*>")
_CHARSET = re.compile(r"charset=([\w-]+)", re.IGNORECASE)
_COMPANY_SUFFIXES = frozenset(
    """
    inc incorporated corp corporation co company companies ltd limited plc llc lp holdings holding group the
    sa ag nv se aktiengesellschaft
    """.split()
)


# --------------------------------------------------------------------------
# Small helpers shared with the research agent
# --------------------------------------------------------------------------


def query_terms(text: str | None) -> list[str]:
    """Distinct topic words of a query or question, in order (stopwords, filler, possessives removed)."""
    out: list[str] = []
    for token in tokenize(text):
        if token.endswith("'s"):
            token = token[:-2]
        if len(token) > 1 and token not in QUESTION_FILLER and token not in out:
            out.append(token)
    return out


def doc_type_label(doc_type: str | None) -> str:
    """Reader-facing label for a ``DocType`` value."""
    return DOC_TYPE_LABELS.get(doc_type or "", doc_type or "source")


def trust_tier_label(trust_tier: str | None) -> str:
    """Reader-facing label for a ``TrustTier`` value."""
    return TRUST_TIER_LABELS.get(trust_tier or "", trust_tier or "unknown")


def article_handle(article_id: int) -> str:
    """The handle tools use for an archived article, e.g. ``A42``."""
    return f"A{article_id}"


def _day(value: datetime | str | None) -> str:
    dt = parse_datetime(value) if isinstance(value, str) else value
    return dt.strftime("%Y-%m-%d") if dt else ""


def _int_arg(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _error(message: str) -> ToolResult:
    return ToolResult(content=message, is_error=True)


# --------------------------------------------------------------------------
# HTTP layer (patched in tests; every request carries the descriptive User-Agent)
# --------------------------------------------------------------------------


class RetrievalError(RuntimeError):
    """A network retrieval failed or was refused (unapproved redirect, too large, ...)."""


@dataclass
class HttpResponse:
    url: str
    status_code: int
    content_type: str
    content: bytes
    truncated: bool = False

    @property
    def text(self) -> str:
        match = _CHARSET.search(self.content_type or "")
        encoding = match.group(1) if match else "utf-8"
        try:
            return self.content.decode(encoding, errors="replace")
        except LookupError:
            return self.content.decode("utf-8", errors="replace")


def http_get(
    url: str,
    settings: Settings,
    *,
    accept: str = "*/*",
    allow: Callable[[str], bool] | None = None,
    max_bytes: int = MAX_FETCH_BYTES,
    partial_ok: bool = False,
) -> HttpResponse:
    """GET ``url`` with The Newsren's User-Agent, following redirects only to URLs ``allow`` accepts.

    Reads at most ``max_bytes``: beyond that the body is cut short when
    ``partial_ok`` (and ``truncated`` is set), otherwise ``RetrievalError``.
    Network failures also raise ``RetrievalError``; HTTP error statuses are
    returned for the caller to report.
    """
    headers = {"User-Agent": settings.user_agent, "Accept": accept}
    current = url
    try:
        with httpx.Client(
            timeout=settings.fetch_timeout_seconds, headers=headers, follow_redirects=False
        ) as client:
            for _hop in range(MAX_REDIRECTS + 1):
                if allow is not None and not allow(current):
                    raise RetrievalError(f"redirected to {current}, which is not an approved source")
                with client.stream("GET", current) as response:
                    if response.is_redirect and response.headers.get("location"):
                        current = str(response.url.join(response.headers["location"]))
                        continue
                    chunks: list[bytes] = []
                    size = 0
                    truncated = False
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            if not partial_ok:
                                raise RetrievalError(f"response from {current} exceeds {max_bytes} bytes")
                            chunks.append(chunk[: max(0, len(chunk) - (size - max_bytes))])
                            truncated = True
                            break
                        chunks.append(chunk)
                    return HttpResponse(
                        url=str(response.url),
                        status_code=response.status_code,
                        content_type=response.headers.get("content-type", ""),
                        content=b"".join(chunks),
                        truncated=truncated,
                    )
    except httpx.HTTPError as exc:
        raise RetrievalError(f"{type(exc).__name__}: {exc}") from exc
    raise RetrievalError(f"too many redirects starting from {url}")


def _get_json(url: str, settings: Settings) -> Any:
    response = http_get(url, settings, accept="application/json", max_bytes=MAX_JSON_BYTES)
    if response.status_code == 404:
        raise RetrievalError(f"no data at {url} (HTTP 404)")
    if response.status_code != 200:
        raise RetrievalError(f"HTTP {response.status_code} from {url}")
    try:
        return json.loads(response.content)
    except ValueError as exc:
        raise RetrievalError(f"invalid JSON from {url}") from exc


# --------------------------------------------------------------------------
# Archive search
# --------------------------------------------------------------------------


def _approved_source(article: Article) -> bool:
    src = article.source
    return src is not None and bool(src.approved) and bool(src.active)


def archive_search(
    session: Session,
    query: str,
    *,
    vertical: str | None = None,
    since_days: int | None = None,
    limit: int = ARCHIVE_RESULT_LIMIT,
) -> list[Article]:
    """Best-matching archived articles from approved, active sources (duplicates collapse to the original)."""
    terms = query_terms(query)
    if not terms:
        return []
    hits = search_article_ids(session, " ".join(terms), limit=ARCHIVE_CANDIDATES)
    if not hits:
        return []
    rows = {a.id: a for a in session.scalars(select(Article).where(Article.id.in_([aid for aid, _ in hits])))}
    cutoff = utcnow() - timedelta(days=since_days) if since_days else None
    out: list[Article] = []
    seen: set[int] = set()
    for aid, _score in hits:
        article = rows.get(aid)
        if article is not None and article.duplicate_of_id is not None:
            article = session.get(Article, article.duplicate_of_id)
        if article is None or article.id in seen:
            continue
        seen.add(article.id)
        if not _approved_source(article):
            continue
        if vertical and article.vertical != vertical:
            continue
        if cutoff is not None and article.display_time < cutoff:
            continue
        out.append(article)
        if len(out) >= limit:
            break
    return out


def _article_meta(article: Article) -> str:
    published = _day(article.published_at)
    parts = [
        article.source.name if article.source else "",
        f"published {published}" if published else "publication date unknown",
        f"retrieved {_day(article.retrieved_at)}",
        f"type: {doc_type_label(article.doc_type)}",
        f"trust tier: {trust_tier_label(article.trust_tier)}",
        f"vertical: {article.vertical}",
    ]
    return " · ".join(p for p in parts if p)


def _evidence_notes(article: Article) -> list[str]:
    """Labelling reminders for the model (preprint, vendor claim, opinion, correction, data delay)."""
    notes: list[str] = []
    extra = article.extra or {}
    if article.doc_type == DocType.RESEARCH_PREPRINT:
        notes.append(
            extra.get("peer_review_note") or "Preprint: not peer reviewed. Do not overstate causality from it."
        )
    elif article.doc_type == DocType.RESEARCH_PEER_REVIEWED:
        notes.append("Peer-reviewed research.")
    if article.trust_tier == TrustTier.VENDOR or article.doc_type == DocType.PRESS_RELEASE:
        notes.append(
            "This is the organization's own account: attribute its claims to it and treat capability or "
            "performance claims as claims until independently checked."
        )
    if article.doc_type == DocType.OPINION or article.trust_tier == TrustTier.COMMENTARY:
        notes.append("Opinion/commentary: an interpretation, not evidence that something happened.")
    if re.match(r"\s*(correction|clarification)\b", article.title or "", re.IGNORECASE):
        notes.append("This item corrects or clarifies earlier reporting.")
    if article.source is not None and article.source.data_delay_note:
        notes.append(f"Data note: {article.source.data_delay_note}")
    return notes


def _article_sentences(article: Article) -> str:
    """Summary + body without the title, for snippets."""
    return normalize_space(f"{article.summary or ''} {article.content or ''}")


# --------------------------------------------------------------------------
# arXiv
# --------------------------------------------------------------------------


def arxiv_query_params(query: str, max_terms: int = 5) -> dict[str, Any]:
    """arXiv API parameters for a free-text query, an arXiv id, or a query already in arXiv syntax."""
    q = normalize_space(query)
    id_match = _ARXIV_ID.match(q)
    if id_match:
        return {"id_list": id_match.group(1), "max_results": 1}
    if _ARXIV_FIELD.search(q):
        search_query = q
    else:
        terms = query_terms(q)[:max_terms] or [q]
        search_query = " AND ".join(f'all:"{t}"' if "-" in t or " " in t else f"all:{t}" for t in terms)
    return {
        "search_query": search_query,
        "start": 0,
        "max_results": PAPER_RESULT_LIMIT,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }


def _arxiv_entry(item: FetchedItem, index: int, query: str) -> tuple[str, CitationRef]:
    extra = item.extra or {}
    arxiv_id = extra.get("arxiv_id") or item.url.rsplit("/", 1)[-1]
    authors = extra.get("authors") or []
    author_text = ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "")
    updated = _day(extra.get("updated_at"))
    submitted = _day(item.published_at)
    status = "Preprint, not peer reviewed by arXiv."
    if extra.get("journal_ref"):
        status += f" Authors report a journal reference ({extra['journal_ref']}); not verified."
    meta = " · ".join(
        p
        for p in [
            f"authors: {author_text}" if author_text else "",
            f"submitted {submitted}" if submitted else "",
            f"last updated {updated}" if updated and updated != submitted else "",
            f"categories: {', '.join(extra.get('categories') or [])}" if extra.get("categories") else "",
        ]
        if p
    )
    version = extra.get("arxiv_version") or ""
    lines = [
        f"[{index}] [arXiv:{arxiv_id}] {item.title}" + (f" ({version})" if version else ""),
        f"    {meta}" if meta else "",
        f"    Status: {status}",
        f"    URL: {item.url}",
        f"    Abstract: {truncate(item.summary, 900)}",
    ]
    citation = CitationRef(
        id=f"arXiv:{arxiv_id}",
        url=item.url,
        title=item.title,
        publisher="arXiv",
        published_at=iso(item.published_at),
        retrieved_at=iso(utcnow()),
        snippet=best_snippet(item.summary or "", query, SNIPPET_CHARS) or truncate(item.summary, SNIPPET_CHARS),
        trust_tier=TrustTier.PRIMARY,
        doc_type=DocType.RESEARCH_PREPRINT,
    )
    return "\n".join(line for line in lines if line), citation


# --------------------------------------------------------------------------
# SEC EDGAR
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CompanyRecord:
    cik: str  # zero-padded, 10 digits
    ticker: str
    title: str


_sec_cache: dict[str, tuple[float, list[CompanyRecord]]] = {}


def clear_caches() -> None:
    """Forget the cached SEC ticker file (tests; or after SEC updates it)."""
    _sec_cache.clear()


def parse_company_tickers(data: Mapping[str, Any]) -> list[CompanyRecord]:
    """Rows of SEC's ``company_tickers.json`` in SEC's order (largest companies first)."""
    records: list[CompanyRecord] = []
    rows = data.values() if isinstance(data, Mapping) else data
    for row in rows:
        try:
            cik = str(int(row["cik_str"])).zfill(10)
            ticker = str(row["ticker"]).strip().upper()
        except (KeyError, TypeError, ValueError):
            continue
        records.append(CompanyRecord(cik=cik, ticker=ticker, title=str(row.get("title") or "")))
    return records


def _company_words(name: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", name.lower().replace("&", " and "))
    return [w for w in words if w not in _COMPANY_SUFFIXES]


def resolve_company(records: Iterable[CompanyRecord], query: str) -> list[CompanyRecord]:
    """Companies matching a ticker, CIK, or name, best first (one row per CIK); empty when nothing matches.

    Exact ticker or CIK wins; otherwise names match when every query word
    appears in the registered name (legal suffixes like "Inc" ignored), with
    exact names first.
    """
    q = normalize_space(query)
    if not q:
        return []
    records = list(records)
    # Tickers are never all digits, so a number is a CIK.
    cik_match = re.fullmatch(r"(?i)(?:cik\s*)?(\d{1,10})", q)
    if cik_match:
        cik = cik_match.group(1).zfill(10)
        return [r for r in records if r.cik == cik][:1]
    ticker = q.lstrip("$").upper().replace(".", "-").replace("/", "-")
    exact = [r for r in records if r.ticker == ticker]
    if exact:
        return exact[:1]
    words = _company_words(q)
    if not words:
        return []
    scored: list[tuple[int, int, CompanyRecord]] = []
    for order, record in enumerate(records):
        title_words = _company_words(record.title)
        if not title_words:
            continue
        if title_words == words:
            score = 3
        elif title_words[: len(words)] == words:
            score = 2
        elif all(w in title_words for w in words):
            score = 1
        else:
            continue
        scored.append((-score, order, record))
    scored.sort(key=lambda item: (item[0], item[1]))
    out: list[CompanyRecord] = []
    seen: set[str] = set()
    for _, _, record in scored:
        if record.cik not in seen:
            seen.add(record.cik)
            out.append(record)
    return out


def filing_url(cik: str, accession: str, document: str | None = None) -> str:
    """EDGAR archive URL of a filing's primary document (or its index when the document is unknown)."""
    base = f"{SEC_ARCHIVES_URL}/{int(cik)}/{accession.replace('-', '')}/"
    return f"{base}{document}" if document else base


def recent_material_filings(
    submissions: Mapping[str, Any], cik: str, limit: int = FILING_RESULT_LIMIT
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Newest material filings from a ``submissions`` payload, plus counts of the routine forms skipped."""
    recent = (submissions.get("filings") or {}).get("recent") or {}
    columns = {k: v for k, v in recent.items() if isinstance(v, list)}
    n = min((len(v) for v in columns.values()), default=0)
    rows = [{k: v[i] for k, v in columns.items()} for i in range(n)]
    rows.sort(key=lambda r: str(r.get("filingDate") or ""), reverse=True)
    filings: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        form = str(row.get("form") or "").strip()
        if form not in MATERIAL_FORMS:
            skipped[form or "unknown"] += 1
            continue
        if len(filings) >= limit:
            continue
        accession = str(row.get("accessionNumber") or "")
        if not accession:
            continue
        items = [i.strip() for i in str(row.get("items") or "").split(",") if i.strip()]
        filings.append(
            {
                "form": form,
                "form_description": MATERIAL_FORMS[form],
                "filed": str(row.get("filingDate") or ""),
                "report_date": str(row.get("reportDate") or ""),
                "accession": accession,
                "items": [f"{i} {EIGHT_K_ITEMS[i]}" if i in EIGHT_K_ITEMS else i for i in items],
                "url": filing_url(cik, accession, str(row.get("primaryDocument") or "") or None),
                "index_url": filing_url(cik, accession),
                "description": str(row.get("primaryDocDescription") or ""),
            }
        )
    return filings, skipped


def html_main_text(raw_html: str) -> str:
    """Readable text of an HTML page: the ``<main>``/``<article>`` element when present, without navigation."""
    cleaned = _BOILERPLATE_BLOCKS.sub(" ", raw_html)
    for pattern in (_MAIN_BLOCK, _ARTICLE_BLOCK):
        match = pattern.search(cleaned)
        if match:
            text = strip_html(match.group(1))
            if len(text) >= 200:
                return text
    return strip_html(cleaned)


def _skipped_summary(skipped: Counter[str]) -> str:
    if not skipped:
        return ""
    parts = []
    for form, count in skipped.most_common(5):
        label = _ROUTINE_FORM_LABELS.get(form.removesuffix("/A"))
        if label is None and form.startswith("424B"):
            label = "prospectus supplements"
        parts.append(f"{count} × {form}" + (f" ({label})" if label else ""))
    rest = sum(skipped.values()) - sum(c for _, c in skipped.most_common(5))
    if rest:
        parts.append(f"{rest} other")
    return f"Not listed: {sum(skipped.values())} other filings, mostly routine ({', '.join(parts)})."


# --------------------------------------------------------------------------
# Tool schemas (strict)
# --------------------------------------------------------------------------


def _nullable(schema: dict[str, Any], description: str) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}], "description": description}


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


SEARCH_ARCHIVE_SCHEMA = _object(
    {
        "query": {
            "type": "string",
            "description": "Keywords to search for: organizations, products, people, policies, or topics. "
            "Short keyword queries work better than full questions.",
        },
        "vertical": _nullable(
            {"type": "string", "enum": list(VERTICALS)},
            "Restrict to one news section, or null for all sections.",
        ),
        "since_days": _nullable(
            {"type": "integer"},
            f"Only articles published in the last N days (1 to {MAX_SINCE_DAYS}), or null for any date.",
        ),
    }
)
GET_ARTICLE_SCHEMA = _object(
    {"article_id": {"type": "integer", "description": "The number in an archive handle, e.g. 42 for [A42]."}}
)
SEARCH_PAPERS_SCHEMA = _object(
    {
        "query": {
            "type": "string",
            "description": "Keywords (title words, method, model or dataset names), an arXiv id such as "
            "2412.20138, or arXiv query syntax such as ti:TradingAgents.",
        }
    }
)
SEARCH_FILINGS_SCHEMA = _object(
    {
        "company_or_ticker": {
            "type": "string",
            "description": "A stock ticker (NVDA), SEC CIK number, or registered company name (NVIDIA Corp).",
        }
    }
)
FETCH_URL_SCHEMA = _object(
    {"url": {"type": "string", "description": "An absolute http(s) URL on an approved source domain."}}
)


# --------------------------------------------------------------------------
# The toolbox
# --------------------------------------------------------------------------


@dataclass
class ResearchToolbox:
    """The approved tools bound to one database session, plus a record of what they retrieved."""

    session: Session
    settings: Settings = field(default_factory=get_settings)
    allow_network: bool = True
    # canonical URL -> the text the agent was shown for that source
    retrieved_text: dict[str, str] = field(default_factory=dict)

    def tools(self) -> list[Tool]:
        """The five approved retrieval tools, with strict input schemas."""
        return [
            Tool(
                name="search_archive",
                description=(
                    "Search The Newsren's archive of articles ingested from approved sources (news outlets, "
                    "regulators, research indexes, company newsrooms). Returns up to 8 numbered results, best "
                    "match first, each with an archive handle such as [A42], title, publisher, publication "
                    "and retrieval dates, source type (news, press release, preprint, filing, regulator "
                    "release, opinion, ...), trust tier, URL, and a matching snippet. Syndicated copies are "
                    "collapsed into the original. Start every research task here. Snippets are not enough to "
                    "support a claim: read the articles you rely on with get_article."
                ),
                input_schema=SEARCH_ARCHIVE_SCHEMA,
                handler=self.search_archive,
            ),
            Tool(
                name="get_article",
                description=(
                    "Read one archived article by its handle number (42 for [A42]): title, publisher, dates, "
                    "source type, trust tier, labelling notes (preprint, vendor claim, opinion, correction), "
                    "URL, and the text, truncated to about 6,000 characters."
                ),
                input_schema=GET_ARTICLE_SCHEMA,
                handler=self.get_article,
            ),
            Tool(
                name="search_research_papers",
                description=(
                    "Search arXiv (an approved research index) for research papers. Use it when the question "
                    "concerns a study, paper, model evaluation, or benchmark methodology. Returns up to 6 "
                    "papers with arXiv id, title, authors, submission date, abstract, and URL. arXiv papers "
                    "are preprints that have not been peer reviewed; say so when you rely on one."
                ),
                input_schema=SEARCH_PAPERS_SCHEMA,
                handler=self.search_research_papers,
            ),
            Tool(
                name="search_sec_filings",
                description=(
                    "List a US-listed company's most recent material SEC EDGAR filings (10-K, 10-Q, 8-K with "
                    "item descriptions, 20-F, 6-K, proxy and registration statements): form, filing date, "
                    "period, accession number, and the primary document URL. Use it for questions about "
                    "earnings, filings, or disclosures, then read a filing with fetch_approved_url."
                ),
                input_schema=SEARCH_FILINGS_SCHEMA,
                handler=self.search_sec_filings,
            ),
            Tool(
                name="fetch_approved_url",
                description=(
                    "Fetch one document from an approved source: domains of The Newsren's registered sources "
                    "and primary-record sites such as sec.gov, arxiv.org, federalreserve.gov, finra.org, and "
                    "ed.gov. Returns the page title and plain text, truncated to about 6,000 characters. Any "
                    "other domain is refused. PDFs are not supported."
                ),
                input_schema=FETCH_URL_SCHEMA,
                handler=self.fetch_approved_url,
            ),
        ]

    # ------------------------------------------------------------ bookkeeping

    def _remember(self, url: str, text: str) -> None:
        try:
            key = canonical_url(url)
        except ValueError:
            return
        existing = self.retrieved_text.get(key, "")
        if text and text not in existing:
            self.retrieved_text[key] = f"{existing}\n{text}".strip()

    def text_for(self, url: str) -> str:
        """Everything the tools showed for ``url`` in this session (empty when never retrieved)."""
        try:
            return self.retrieved_text.get(canonical_url(url), "")
        except ValueError:
            return ""

    def _network_disabled(self, tool: str, what: str) -> ToolResult:
        return _error(
            f"{tool} is unavailable: network retrieval is disabled in this deployment (offline mode), so "
            f"{what} cannot be consulted right now. Answer from The Newsren archive (search_archive, "
            f"get_article) and list anything that needs {what} under could_not_verify."
        )

    # ------------------------------------------------------------ archive

    def search_archive(self, args: dict[str, Any]) -> ToolResult:
        """Tool handler: full-text search over approved archived articles."""
        query = normalize_space(args.get("query"))
        vertical = args.get("vertical") or None
        since_days = _int_arg(args.get("since_days"))
        if vertical is not None and vertical not in VERTICALS:
            return _error(f"Unknown vertical {vertical!r}. Use one of: {', '.join(VERTICALS)}, or null.")
        if since_days is not None:
            since_days = max(1, min(since_days, MAX_SINCE_DAYS))
        if not query_terms(query):
            return _error("search_archive needs a query with at least one meaningful keyword.")

        articles = archive_search(self.session, query, vertical=vertical, since_days=since_days)
        filters = [f"vertical={vertical}" if vertical else "", f"last {since_days} days" if since_days else ""]
        filter_text = f"; filters: {', '.join(f for f in filters if f)}" if any(filters) else ""
        header = (
            f'Archive search for "{query}": {len(articles)} result{"s" if len(articles) != 1 else ""} '
            f"(approved sources only; syndicated copies collapsed into the original{filter_text})."
        )
        if not articles:
            return ToolResult(
                content=f"{header}\nNo archived article matches. Try different or fewer keywords, or "
                "remove the filters.",
                data=[],
            )
        lines = [header, "Cite a result by its URL. Read it with get_article before relying on it.", ""]
        citations: list[CitationRef] = []
        data: list[dict[str, Any]] = []
        for index, article in enumerate(articles, start=1):
            handle = article_handle(article.id)
            snippet = best_snippet(_article_sentences(article), query, SNIPPET_CHARS) or truncate(
                article.summary, SNIPPET_CHARS
            )
            lines.extend(
                [
                    f"[{index}] [{handle}] {article.title}",
                    f"    {_article_meta(article)}",
                    f"    URL: {article.url}",
                    f'    Snippet: "{snippet}"',
                ]
            )
            citation = citation_for(article, handle, snippet)
            citations.append(citation)
            self._remember(article.url, f"{article.title}\n{snippet}")
            data.append(
                {
                    "article_id": article.id,
                    "handle": handle,
                    "title": article.title,
                    "url": article.url,
                    "snippet": snippet,
                    "vertical": article.vertical,
                    "doc_type": article.doc_type,
                    "trust_tier": article.trust_tier,
                    "published_at": iso(article.published_at),
                    "story_id": article.story_id,
                    "citation": citation,
                }
            )
        return ToolResult(content="\n".join(lines), citations=citations, data=data)

    def get_article(self, args: dict[str, Any]) -> ToolResult:
        """Tool handler: one archived article's text (truncated) with labels and its citation."""
        article_id = _int_arg(args.get("article_id"))
        if article_id is None:
            return _error("get_article needs an integer article_id (42 for [A42]).")
        article = self.session.get(Article, article_id)
        if article is None:
            return _error(f"No archived article has id {article_id}. Use search_archive to find article handles.")
        notes: list[str] = []
        if article.duplicate_of_id is not None:
            original = self.session.get(Article, article.duplicate_of_id)
            if original is not None:
                notes.append(
                    f"[A{article.id}] is a syndicated copy; showing the original "
                    f"[{article_handle(original.id)}] instead."
                )
                article = original
        if not _approved_source(article):
            return _error(f"Article [A{article.id}] is not from an approved, active source and cannot be used.")

        handle = article_handle(article.id)
        full = article_text(article, max_chars=10**9)
        text = article_text(article, max_chars=ARTICLE_TEXT_LIMIT)
        notes.extend(_evidence_notes(article))
        if len(full) > len(text):
            notes.append(f"Text truncated to {ARTICLE_TEXT_LIMIT} of {len(full)} characters.")
        lines = [
            f"[{handle}] {article.title}",
            _article_meta(article),
            f"URL: {article.url}",
            *(f"Note: {n}" for n in notes),
            "",
            text,
        ]
        self._remember(article.url, text)
        citation = citation_for(article, handle)
        data = {
            "article_id": article.id,
            "handle": handle,
            "title": article.title,
            "url": article.url,
            "publisher": article.source.name if article.source else "",
            "published_at": iso(article.published_at),
            "doc_type": article.doc_type,
            "trust_tier": article.trust_tier,
            "vertical": article.vertical,
            "story_id": article.story_id,
            "summary": truncate(article.summary, ARTICLE_TEXT_LIMIT),
            "content": truncate(article.content, ARTICLE_TEXT_LIMIT),
            "text": text,
            "citation": citation,
        }
        return ToolResult(content="\n".join(lines), citations=[citation], data=data)

    # ------------------------------------------------------------ arXiv

    def search_research_papers(self, args: dict[str, Any]) -> ToolResult:
        """Tool handler: arXiv API search; every result is labelled a preprint."""
        query = normalize_space(args.get("query"))
        if not query:
            return _error("search_research_papers needs a query.")
        if not self.allow_network:
            return self._network_disabled("search_research_papers", "arXiv")
        params = arxiv_query_params(query)
        items, error = self._arxiv(params)
        broadened = ""
        if error is None and not items and "search_query" in params and not _ARXIV_FIELD.search(query):
            terms = query_terms(query)
            if len(terms) > 2:
                params = arxiv_query_params(" ".join(terms[:2]))
                items, error = self._arxiv(params)
                broadened = f' No paper matched every keyword; broadened to "{" ".join(terms[:2])}".'
        if error is not None:
            return _error(error)
        header = (
            f'arXiv search for "{query}": {len(items)} paper{"s" if len(items) != 1 else ""} '
            f"(arXiv API, an approved research index).{broadened}"
        )
        if not items:
            return ToolResult(content=f"{header}\nNo papers found. Try other keywords or an arXiv id.", data=[])
        lines = [
            header,
            'Every arXiv paper is a preprint: arXiv does not peer review submissions. Say "preprint" when '
            "you rely on one and do not overstate causality.",
            "",
        ]
        citations: list[CitationRef] = []
        data: list[dict[str, Any]] = []
        for index, item in enumerate(items[:PAPER_RESULT_LIMIT], start=1):
            block, citation = _arxiv_entry(item, index, query)
            lines.extend([block, ""])
            citations.append(citation)
            self._remember(item.url, f"{item.title}\n{item.summary}")
            data.append(
                {
                    "arxiv_id": citation.id.removeprefix("arXiv:"),
                    "title": item.title,
                    "url": item.url,
                    "summary": item.summary or "",
                    "authors": (item.extra or {}).get("authors") or [],
                    "published_at": citation.published_at,
                    "doc_type": DocType.RESEARCH_PREPRINT,
                    "trust_tier": TrustTier.PRIMARY,
                    "citation": citation,
                }
            )
        return ToolResult(content="\n".join(lines).rstrip(), citations=citations, data=data)

    def _arxiv(self, params: dict[str, Any]) -> tuple[list[FetchedItem], str | None]:
        url = f"{ARXIV_API_URL}?{urlencode(params)}"
        try:
            response = http_get(url, self.settings, accept="application/atom+xml")
        except RetrievalError as exc:
            return [], f"arXiv could not be reached: {exc}"
        try:
            items = parse_arxiv(response.content)
        except FeedParseError as exc:
            return [], f"arXiv rejected the query: {exc}"
        if response.status_code != 200 and not items:
            return [], f"arXiv returned HTTP {response.status_code}."
        return items, None

    # ------------------------------------------------------------ SEC EDGAR

    def _sec_companies(self) -> list[CompanyRecord]:
        cached = _sec_cache.get(SEC_TICKERS_URL)
        if cached and time.monotonic() - cached[0] < SEC_TICKER_CACHE_SECONDS:
            return cached[1]
        records = parse_company_tickers(_get_json(SEC_TICKERS_URL, self.settings))
        if records:
            _sec_cache[SEC_TICKERS_URL] = (time.monotonic(), records)
        return records

    def search_sec_filings(self, args: dict[str, Any]) -> ToolResult:
        """Tool handler: a company's recent material SEC EDGAR filings."""
        name = normalize_space(args.get("company_or_ticker"))
        if not name:
            return _error("search_sec_filings needs a ticker, CIK, or company name.")
        if not self.allow_network:
            return self._network_disabled("search_sec_filings", "SEC EDGAR")
        try:
            matches = resolve_company(self._sec_companies(), name)
        except RetrievalError as exc:
            return _error(f"SEC EDGAR could not be reached: {exc}")
        if not matches:
            return _error(
                f'SEC EDGAR lists no company matching "{name}". Try the exact ticker (e.g. NVDA) or the '
                "registered company name. Companies that are not SEC registrants have no EDGAR filings."
            )
        company = matches[0]
        try:
            submissions = _get_json(SEC_SUBMISSIONS_URL.format(cik=company.cik), self.settings)
        except RetrievalError as exc:
            return _error(f"SEC EDGAR submissions for CIK {company.cik} could not be retrieved: {exc}")
        filings, skipped = recent_material_filings(submissions, company.cik)
        registered = str(submissions.get("name") or company.title)
        tickers = submissions.get("tickers") or [company.ticker]
        retrieved = iso(utcnow())
        lines = [
            f"SEC EDGAR filings for {registered} (ticker {', '.join(map(str, tickers[:3]))}, CIK {company.cik}); "
            f"primary source: data.sec.gov submissions, retrieved {retrieved}. EDGAR lists filings after "
            "acceptance; this is not real-time market data.",
        ]
        if len(matches) > 1:
            others = "; ".join(f"{m.title} ({m.ticker})" for m in matches[1:4])
            lines.append(f'Other companies matching "{name}": {others}.')
        if not filings:
            lines.append("No recent 10-K, 10-Q, 8-K, or other material filings are listed.")
        else:
            lines.append(f"The {len(filings)} most recent material filings, newest first:")
        skipped_text = _skipped_summary(skipped)
        if skipped_text:
            lines.append(skipped_text)
        lines.append("")
        citations: list[CitationRef] = []
        for index, filing in enumerate(filings, start=1):
            period = f", period ending {filing['report_date']}" if filing["report_date"] else ""
            items = f"\n    Items: {'; '.join(filing['items'])}" if filing["items"] else ""
            lines.append(
                f"[{index}] [SEC:{filing['accession']}] Form {filing['form']} ({filing['form_description']}) "
                f"filed {filing['filed']}{period}{items}\n    URL: {filing['url']}"
            )
            summary = (
                f"{registered} Form {filing['form']} ({filing['form_description']}) filed {filing['filed']}"
                f"{period}" + (f"; items {'; '.join(filing['items'])}" if filing["items"] else "")
            )
            citations.append(
                CitationRef(
                    id=f"SEC:{filing['accession']}",
                    url=filing["url"],
                    title=f"{registered} Form {filing['form']} ({filing['form_description']}), filed {filing['filed']}",
                    publisher="SEC EDGAR",
                    published_at=iso(parse_datetime(filing["filed"])),
                    retrieved_at=retrieved,
                    snippet=truncate(summary, SNIPPET_CHARS),
                    trust_tier=TrustTier.PRIMARY,
                    doc_type=DocType.FILING,
                )
            )
            self._remember(filing["url"], summary)
        if filings:
            lines.extend(
                [
                    "",
                    "Read a filing with fetch_approved_url(url); sec.gov is an approved primary source. Quote the "
                    "filing for every number you report.",
                ]
            )
        data = {"company": registered, "cik": company.cik, "tickers": tickers, "filings": filings}
        return ToolResult(content="\n".join(lines).rstrip(), citations=citations, data=data)

    # ------------------------------------------------------------ approved URLs

    def _source_for_domain(self, domain: str) -> Source | None:
        for src in self.session.scalars(select(Source).where(Source.approved.is_(True), Source.active.is_(True))):
            for url in (src.homepage, src.url):
                d = domain_of(url or "")
                if d and (domain == d or domain.endswith("." + d)):
                    return src
        return None

    def _page_citation(self, url: str, title: str, text: str) -> CitationRef:
        try:
            article = self.session.scalars(
                select(Article).where(Article.canonical_url == canonical_url(url))
            ).first()
        except ValueError:
            article = None
        if article is not None and article.duplicate_of_id is None:
            return citation_for(article, article_handle(article.id))
        domain = domain_of(url)
        source = self._source_for_domain(domain)
        doc_type = next(
            (dt for d, dt in _PRIMARY_DOMAIN_TYPES if domain == d or domain.endswith("." + d)),
            source.default_doc_type if source else DocType.NEWS,
        )
        is_primary_domain = any(domain == d or domain.endswith("." + d) for d, _ in _PRIMARY_DOMAIN_TYPES)
        trust = (
            TrustTier.PRIMARY if is_primary_domain else (source.trust_tier if source else TrustTier.INDEPENDENT)
        )
        lead = next((s for s in split_sentences(text) if len(s) > 60), "") or text
        return CitationRef(
            id=url,
            url=url,
            title=title or url,
            publisher=source.name if source else domain,
            published_at=None,
            retrieved_at=iso(utcnow()),
            snippet=truncate(lead, SNIPPET_CHARS),
            trust_tier=trust,
            doc_type=doc_type,
        )

    def fetch_approved_url(self, args: dict[str, Any]) -> ToolResult:
        """Tool handler: fetch one page from an approved domain (redirects re-checked) as plain text."""
        url = normalize_space(args.get("url"))
        if not url.lower().startswith(("http://", "https://")):
            return _error("fetch_approved_url needs an absolute http(s) URL.")
        if not is_approved_url(self.session, url):
            return _error(
                f"Refused: {url} is not an approved source. Only domains of The Newsren's registered sources "
                "and primary-record sites (for example sec.gov, arxiv.org, federalreserve.gov, finra.org, "
                "ed.gov) can be fetched. Use search_archive for coverage from other outlets."
            )
        if not self.allow_network:
            return self._network_disabled("fetch_approved_url", "live web pages")
        try:
            response = http_get(
                url,
                self.settings,
                accept="text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.8,*/*;q=0.5",
                allow=lambda u: is_approved_url(self.session, u),
                partial_ok=True,
            )
        except RetrievalError as exc:
            return _error(f"Could not fetch {url}: {exc}")
        if response.status_code >= 400:
            return _error(f"{url} returned HTTP {response.status_code}.")
        content_type = response.content_type.lower()
        if "pdf" in content_type or response.content[:5] == b"%PDF-":
            return _error(
                f"{url} is a PDF, which this tool cannot read. Look for an HTML version (for SEC filings, the "
                ".htm primary document listed by search_sec_filings)."
            )
        raw = response.text.lstrip("\ufeff")
        if "html" in content_type or raw.lstrip()[:200].lower().startswith(("<!doctype html", "<html")):
            match = _HTML_TITLE.search(raw)
            title = normalize_space(strip_html(match.group(1))) if match else ""
            text = html_main_text(raw)
        elif any(t in content_type for t in ("text/", "json", "xml")) or not content_type:
            title, text = "", raw
        else:
            return _error(f"{url} has an unsupported content type ({response.content_type}).")
        text = normalize_space(text)
        if not text:
            return _error(f"{url} returned no readable text.")
        shown = truncate(text, FETCH_TEXT_LIMIT)
        final_url = response.url or url
        citation = self._page_citation(final_url, title, text)
        self._remember(url, shown)
        self._remember(final_url, shown)
        notes = []
        if final_url != url:
            notes.append(f"Redirected to {final_url}.")
        if len(text) > len(shown) or response.truncated:
            notes.append(f"Text truncated to {FETCH_TEXT_LIMIT} characters.")
        lines = [
            f"{title or final_url}",
            f"{citation.publisher} · retrieved {_day(utcnow())} · type: {doc_type_label(citation.doc_type)} · "
            f"trust tier: {trust_tier_label(citation.trust_tier)}",
            f"URL: {final_url}",
            *(f"Note: {n}" for n in notes),
            "",
            shown,
        ]
        data = {"url": final_url, "title": title, "text": shown, "citation": citation}
        return ToolResult(content="\n".join(lines), citations=[citation], data=data)


def build_research_tools(
    session: Session, settings: Settings | None = None, allow_network: bool = True
) -> list[Tool]:
    """The Ask Newsren tools (archive, article, arXiv, SEC EDGAR, approved URL) with strict JSON schemas.

    With ``allow_network=False`` the arXiv, EDGAR, and URL tools return an
    error result explaining that network retrieval is disabled.
    """
    return ResearchToolbox(session, settings or get_settings(), allow_network).tools()
