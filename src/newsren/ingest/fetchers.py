"""Fetchers and parsers for approved sources (Discover).

``fetch_source`` downloads one approved source with a descriptive User-Agent
(SEC EDGAR requires one), a timeout, and the per-source item limit, then hands
the bytes to a pure parser. Parsers never touch the network so tests run on
saved samples under ``tests/data/ingest``.

Licensing is enforced here: publisher feeds whose registry ``license_note``
says "full text not stored" keep only the headline, publisher summary, and link.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import feedparser
import httpx

from newsren.config import Settings, get_settings
from newsren.ingest.sources import full_text_allowed
from newsren.models import DocType, Source, SourceKind
from newsren.textutil import domain_of, iso, normalize_space, parse_datetime, strip_html, truncate

SUMMARY_MAX_CHARS = 1500
LEDE_MAX_CHARS = 400
MAX_RESPONSE_BYTES = 15_000_000

PREPRINT_NOTE = (
    "arXiv preprint: peer-review status unknown. Treat as not peer reviewed unless a journal "
    "publication is independently confirmed."
)
EDGAR_DELAY_NOTE = "EDGAR filings appear with a delay after acceptance; times are filing acceptance times."

_ACCEPT = (
    "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.9, "
    "application/json;q=0.9, */*;q=0.5"
)


class FetchError(RuntimeError):
    """A source could not be fetched or parsed; recorded on ``Source.last_status``."""


class FeedParseError(FetchError):
    """The response was retrieved but is not a parseable feed of the expected kind."""


@dataclass
class FetchedItem:
    """One item from a source, normalized enough for ``pipeline.store_item``."""

    url: str
    title: str
    summary: str = ""
    content: str = ""
    author: str | None = None
    published_at: datetime | None = None
    doc_type: str | None = None  # explicit override (parsers that know the record type)
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------


# EDGAR's current-filings feed only honours these page sizes (others are rounded down).
EDGAR_PAGE_SIZES = (10, 20, 40, 80, 100)


def expand_url(url: str, settings: Settings, now: datetime | None = None) -> str:
    """Fill registry URL placeholders: ``{max_items}``, ``{edgar_count}``, ``{year}``, ``{prev_year}``."""
    year = (now or datetime.now(UTC)).year
    max_items = max(1, settings.max_items_per_source)
    edgar_count = next((n for n in EDGAR_PAGE_SIZES if n >= max_items), EDGAR_PAGE_SIZES[-1])
    return (
        url.replace("{max_items}", str(max_items))
        .replace("{edgar_count}", str(edgar_count))
        .replace("{prev_year}", str(year - 1))
        .replace("{year}", str(year))
    )


def http_get(url: str, settings: Settings, client: httpx.Client | None = None) -> bytes:
    """GET ``url`` with the platform User-Agent and timeout; raise ``FetchError`` on any failure."""
    headers = {"User-Agent": settings.user_agent, "Accept": _ACCEPT}
    own_client = client is None
    if client is None:
        client = httpx.Client(
            timeout=httpx.Timeout(settings.fetch_timeout_seconds), follow_redirects=True, headers=headers
        )
    host = domain_of(url) or url
    try:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise FetchError(f"response from {host} exceeds {MAX_RESPONSE_BYTES} bytes")
        return response.content
    except httpx.HTTPStatusError as exc:
        raise FetchError(f"HTTP {exc.response.status_code} from {host}") from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"network error fetching {host}: {type(exc).__name__}: {exc}") from exc
    finally:
        if own_client:
            client.close()


def fetch_source(
    source: Source, settings: Settings | None = None, *, client: httpx.Client | None = None
) -> list[FetchedItem]:
    """Fetch and parse one approved source, dispatching on ``source.kind``.

    Returns at most ``settings.max_items_per_source`` items. Network, HTTP, and
    parse failures raise ``FetchError`` (never a raw ``httpx`` exception) so the
    pipeline can record them on the source and carry on with the others.
    """
    settings = settings or get_settings()
    kind = source.kind
    if kind not in _PARSERS:
        raise FetchError(f"unsupported source kind {kind!r} for {source.slug}")
    raw = http_get(expand_url(source.url, settings), settings, client)
    try:
        items = _PARSERS[kind](raw, source)
    except FetchError:
        raise
    except Exception as exc:  # malformed payloads surface as a per-source failure
        raise FeedParseError(f"could not parse {kind} response: {type(exc).__name__}: {exc}") from exc
    return items[: max(0, settings.max_items_per_source)]


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

_BOILERPLATE = [
    re.compile(r"The post .{1,400}? appeared first on .{1,200}?\.?\s*$", re.S),
    re.compile(r"\s*(?:Continue reading|Read more|Read the full (?:story|article))\b.{0,120}$", re.I | re.S),
    re.compile(r"\s*\[(?:…|\.\.\.|&#8230;)\]\s*$"),
]
# Lead-in text that marks paid/sponsored posts (vendor content), shared with the pipeline's cues.
SPONSORED_LEAD = re.compile(
    r"^\s*(?:sponsored(?: content| post)?\b|paid (?:content|post)\b|partner content\b|"
    r"this (?:article|content|post) is (?:brought to you|sponsored|paid for|presented) by\b)",
    re.I,
)


_BYLINE_BLOCK = re.compile(
    r'<div class="field field--name-field-authors[^"]*">.*?Byline\(s\).*?</div>\s*</div>\s*</div>', re.S | re.I
)
_BYLINE_NAME = re.compile(r'field__item">\s*(?:<a[^>]*>)?\s*([^<]+?)\s*<', re.I)
_TIME_TAG = re.compile(r"<time\b[^>]*>.*?</time>", re.S | re.I)
_USERNAME_SPAN = re.compile(r"<span><span>[^<]{0,80}</span></span>", re.I)
_USERNAME_AUTHOR = re.compile(r"@|^[\w-]+\.[\w.-]+$")


def _preclean_html(html: str) -> tuple[str, list[str]]:
    """Remove CMS chrome (timestamps, account names, byline widgets) and return printed bylines."""
    bylines: list[str] = []
    block = _BYLINE_BLOCK.search(html)
    if block:
        bylines = [normalize_space(n) for n in _BYLINE_NAME.findall(block.group(0)) if normalize_space(n)]
        html = html[: block.start()] + html[block.end() :]
    html = _TIME_TAG.sub(" ", html)
    html = _USERNAME_SPAN.sub(" ", html)
    return html, bylines


def _clean_text(html_or_text: str | None) -> str:
    """Strip markup and feed boilerplate; keep paragraph breaks."""
    text = strip_html(html_or_text) if html_or_text else ""
    for pattern in _BOILERPLATE:
        text = pattern.sub("", text).strip()
    return text


def _clean_summary(raw: str | None, title: str) -> str:
    text = normalize_space(_clean_text(raw))
    # Some feeds (Inside Higher Ed) repeat the headline at the start of the summary.
    if title and text.lower().startswith(title.lower()):
        text = text[len(title) :].lstrip(" :-–—|")
    return truncate(text, SUMMARY_MAX_CHARS)


def _entry_datetime(entry: Any, *keys: str) -> datetime | None:
    for key in keys:
        value = entry.get(key)
        if value:
            parsed = parse_datetime(value)
            if parsed is not None:
                return parsed
        struct = entry.get(f"{key}_parsed")
        if struct:
            return datetime(*struct[:6])
    return None


def _entry_author(entry: Any, bylines: list[str] | None = None) -> str | None:
    """Printed byline when the feed exposes one, else feed authors (CMS account names/emails dropped)."""
    names: list[str] = list(bylines or [])
    if not names:
        candidates = [a.get("name") or "" for a in entry.get("authors") or []] or [entry.get("author") or ""]
        for candidate in candidates:
            name = normalize_space(strip_html(candidate))
            if name and name not in names and not _USERNAME_AUTHOR.search(name):
                names.append(name)
    return truncate(", ".join(names), 300) or None


def _entry_link(entry: Any) -> str:
    for candidate in (entry.get("link"), entry.get("id")):
        if candidate and str(candidate).lower().startswith(("http://", "https://")):
            return str(candidate).strip()
    return ""


def _parse_feed(raw: bytes | str) -> Any:
    # feedparser treats a str that looks like a URL or path as something to fetch;
    # bytes are always parsed as a document, so no unapproved retrieval can happen here.
    feed = feedparser.parse(raw.encode("utf-8") if isinstance(raw, str) else raw)
    if not feed.entries and feed.bozo:
        raise FeedParseError(f"unparseable feed: {feed.get('bozo_exception')}")
    return feed


# --------------------------------------------------------------------------
# RSS / Atom (publisher and regulator feeds)
# --------------------------------------------------------------------------


def parse_rss(raw: bytes | str, source: Source | None = None) -> list[FetchedItem]:
    """Parse an RSS/Atom feed into items.

    HTML is stripped from titles (some regulators wrap them in links),
    summaries, and bodies; dates become naive UTC; authors are captured. Full
    body text is kept only when the source's license allows it (and never when
    ``source`` is unknown).
    """
    feed = _parse_feed(raw)
    keep_body = source is not None and full_text_allowed(source)
    feed_title = normalize_space(strip_html(feed.feed.get("title") or ""))
    items: list[FetchedItem] = []
    seen: set[str] = set()
    for entry in feed.entries:
        url = _entry_link(entry)
        title = truncate(normalize_space(strip_html(entry.get("title") or "")), 500)
        if not url or not title or url in seen:
            continue
        seen.add(url)
        summary_html, bylines = _preclean_html(entry.get("summary") or entry.get("description") or "")
        body_html = ""
        if entry.get("content"):
            body_html, body_bylines = _preclean_html(entry["content"][0].get("value") or "")
            bylines = bylines or body_bylines
        summary = _clean_summary(summary_html, title)
        body = _clean_text(body_html)
        if not summary and body:
            # No publisher summary: keep only a short lede, not the article.
            summary = truncate(normalize_space(body), LEDE_MAX_CHARS)
        content = body if keep_body and normalize_space(body) != summary else ""
        tags = [normalize_space(t.get("term") or "") for t in entry.get("tags") or []]
        extra: dict[str, Any] = {
            "guid": entry.get("id") or url,
            "tags": [t for t in tags if t][:12],
            "feed_title": feed_title,
        }
        lead = normalize_space(_clean_text(summary_html or body_html))[:300]
        if SPONSORED_LEAD.search(lead):
            extra["sponsored"] = True
        items.append(
            FetchedItem(
                url=url,
                title=title,
                summary=summary,
                content=content,
                author=_entry_author(entry, bylines),
                published_at=_entry_datetime(entry, "published", "updated"),
                extra=extra,
            )
        )
    return items


# --------------------------------------------------------------------------
# arXiv API (Atom)
# --------------------------------------------------------------------------

_ARXIV_ID = re.compile(r"arxiv\.org/abs/(?P<id>[^\s?#]+?)(?:v(?P<ver>\d+))?$")


def parse_arxiv(raw: bytes | str) -> list[FetchedItem]:
    """Parse an arXiv API Atom response; every paper is a ``research_preprint``.

    arXiv does not peer review submissions, so the item records the
    peer-review status as unknown and keeps any author-supplied journal
    reference as an unverified note.
    """
    feed = _parse_feed(raw)
    items: list[FetchedItem] = []
    for entry in feed.entries:
        entry_id = str(entry.get("id") or "")
        if "api/errors" in entry_id or normalize_space(entry.get("title")).lower() == "error":
            raise FeedParseError(f"arXiv API error: {normalize_space(entry.get('summary'))[:200]}")
        match = _ARXIV_ID.search(entry_id)
        if not match:
            continue
        arxiv_id, version = match.group("id"), match.group("ver")
        authors = [normalize_space(a.get("name")) for a in entry.get("authors") or [] if a.get("name")]
        author = ", ".join(authors[:6]) + (" et al." if len(authors) > 6 else "")
        pdf_url = next(
            (
                link.get("href")
                for link in entry.get("links") or []
                if link.get("type") == "application/pdf" or link.get("title") == "pdf"
            ),
            f"https://arxiv.org/pdf/{arxiv_id}",
        )
        categories = [t.get("term") for t in entry.get("tags") or [] if t.get("term")]
        primary = (entry.get("arxiv_primary_category") or {}).get("term") or (categories[:1] or [None])[0]
        journal_ref = normalize_space(entry.get("arxiv_journal_ref")) or None
        updated = _entry_datetime(entry, "updated")
        extra: dict[str, Any] = {
            "arxiv_id": arxiv_id,
            "arxiv_version": f"v{version}" if version else None,
            "pdf_url": pdf_url,
            "authors": authors,
            "categories": categories,
            "primary_category": primary,
            "comment": normalize_space(entry.get("arxiv_comment")) or None,
            "doi": normalize_space(entry.get("arxiv_doi")) or None,
            "journal_ref": journal_ref,
            "updated_at": iso(updated),
            "peer_review_status": "unknown",
            "peer_review_note": PREPRINT_NOTE
            + (f" Authors report a journal reference ({journal_ref}); not verified." if journal_ref else ""),
        }
        items.append(
            FetchedItem(
                url=f"https://arxiv.org/abs/{arxiv_id}",
                title=truncate(normalize_space(entry.get("title")), 500),
                summary=normalize_space(entry.get("summary")),
                author=truncate(author, 300) or None,
                published_at=_entry_datetime(entry, "published", "updated"),
                doc_type=DocType.RESEARCH_PREPRINT,
                extra=extra,
            )
        )
    return items


# --------------------------------------------------------------------------
# SEC EDGAR "current filings" Atom feed
# --------------------------------------------------------------------------

_EDGAR_TITLE = re.compile(
    r"^(?P<form>\S+(?: \S+)?) - (?P<company>.+?) \((?P<cik>\d{4,10})\) \((?P<role>[^)]+)\)\s*$"
)
_EDGAR_FILED = re.compile(r"Filed:\s*(\d{4}-\d{2}-\d{2})")
_EDGAR_ACC = re.compile(r"(?:AccNo:\s*|accession-number=)(\d{10}-\d{2}-\d{6})")
_EDGAR_SIZE = re.compile(r"Size:\s*([\d.]+\s*[KMG]?B)")
_EDGAR_ITEM = re.compile(r"Item\s+(\d+\.\d+):\s*([^\n]+)")

FORM_DESCRIPTIONS = {
    "8-K": "current report",
    "8-K/A": "amended current report",
    "10-Q": "quarterly report",
    "10-Q/A": "amended quarterly report",
    "10-K": "annual report",
    "10-K/A": "amended annual report",
    "6-K": "foreign issuer current report",
    "20-F": "foreign issuer annual report",
    "S-1": "registration statement",
}


def parse_edgar_atom(raw: bytes | str) -> list[FetchedItem]:
    """Parse the EDGAR current-filings Atom feed (8-K, 10-Q, 10-K ...) into ``filing`` items.

    Company name, form type, accession number, CIK, and reported 8-K items go
    in ``extra``. The feed time is the filing acceptance time.
    """
    feed = _parse_feed(raw)
    items: list[FetchedItem] = []
    seen_accessions: set[str] = set()
    for entry in feed.entries:
        raw_title = normalize_space(entry.get("title"))
        match = _EDGAR_TITLE.match(raw_title)
        url = _entry_link(entry)
        if not match or not url:
            continue
        summary_text = strip_html(entry.get("summary") or "")
        acc_match = _EDGAR_ACC.search(summary_text) or _EDGAR_ACC.search(str(entry.get("id") or ""))
        accession = acc_match.group(1) if acc_match else ""
        if accession and accession in seen_accessions:
            continue  # the same filing is listed once per filer role
        seen_accessions.add(accession)
        form_terms = [t.get("term") for t in entry.get("tags") or [] if t.get("label") == "form type"]
        form = (form_terms[0] if form_terms else match.group("form")).strip()
        company = match.group("company").strip()
        cik = match.group("cik").zfill(10)
        filed = _EDGAR_FILED.search(summary_text)
        filed_date = filed.group(1) if filed else None
        size = _EDGAR_SIZE.search(summary_text)
        reported = [{"number": n, "title": normalize_space(t)} for n, t in _EDGAR_ITEM.findall(summary_text)]
        accepted = _entry_datetime(entry, "updated", "published")
        description = FORM_DESCRIPTIONS.get(form.upper(), "filing")
        title = f"{company} files Form {form} ({description})"
        if reported:
            title += f": {reported[0]['title']}"
        parts = [
            f"{company} (CIK {cik}) filed Form {form} ({description}) with the SEC"
            + (f" on {filed_date}" if filed_date else "")
            + (f", accession number {accession}." if accession else ".")
        ]
        if reported:
            parts.append(
                "Items reported: " + "; ".join(f"Item {r['number']} {r['title']}" for r in reported) + "."
            )
        parts.append(
            "Source: SEC EDGAR current filings feed"
            + (f"; accepted {iso(accepted)}" if accepted else "")
            + ". EDGAR times are filing acceptance times."
        )
        items.append(
            FetchedItem(
                url=url,
                title=truncate(title, 500),
                summary=" ".join(parts),
                author=company,
                published_at=accepted,
                doc_type=DocType.FILING,
                extra={
                    "cik": cik,
                    "company": company,
                    "form_type": form,
                    "accession": accession or None,
                    "filed_date": filed_date,
                    "accepted_at": iso(accepted),
                    "items": reported,
                    "filer_role": match.group("role"),
                    "size": size.group(1) if size else None,
                    "is_earnings_release": any(r["number"] == "2.02" for r in reported),
                    "data_delay_note": EDGAR_DELAY_NOTE,
                },
            )
        )
    return items


# --------------------------------------------------------------------------
# ERIC (IES) research index JSON API
# --------------------------------------------------------------------------


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_space(strip_html(str(v))) for v in value if v]
    return [normalize_space(strip_html(str(value)))]


def parse_eric(payload: bytes | str | dict[str, Any]) -> list[FetchedItem]:
    """Parse an ERIC API JSON response.

    ERIC's peer-review flag decides ``research_peer_reviewed`` versus
    ``education_report`` ("Opinion Papers" become ``opinion``). ERIC gives only
    a publication year, so ``published_at`` stays unknown and the year is kept
    in ``extra``.
    """
    data = json.loads(payload) if isinstance(payload, bytes | str) else payload
    response = data.get("response") if isinstance(data, dict) else None
    if not isinstance(response, dict) or not isinstance(response.get("docs"), list):
        raise FeedParseError("ERIC response has no response.docs list")
    items: list[FetchedItem] = []
    for doc in response["docs"]:
        eric_id = normalize_space(str(doc.get("id") or ""))
        title = truncate(normalize_space(strip_html(doc.get("title") or "")), 500)
        if not eric_id or not title:
            continue
        peer_reviewed = str(doc.get("peerreviewed") or "").upper() == "T"
        pub_types = _as_list(doc.get("publicationtype"))
        if any("opinion paper" in t.lower() for t in pub_types):
            doc_type = DocType.OPINION
        elif peer_reviewed:
            doc_type = DocType.RESEARCH_PEER_REVIEWED
        else:
            doc_type = DocType.EDUCATION_REPORT
        authors = _as_list(doc.get("author"))
        year = doc.get("publicationdateyear")
        items.append(
            FetchedItem(
                url=f"https://eric.ed.gov/?id={eric_id}",
                title=title,
                summary=normalize_space(strip_html(doc.get("description") or "")),
                author=truncate(", ".join(authors[:6]) + (" et al." if len(authors) > 6 else ""), 300) or None,
                published_at=None,
                doc_type=doc_type,
                extra={
                    "eric_id": eric_id,
                    "publication_year": int(year) if str(year or "").isdigit() else None,
                    "date_precision": "year",
                    "peer_reviewed": peer_reviewed,
                    "publication_types": pub_types,
                    "subjects": _as_list(doc.get("subject"))[:12],
                    "journal": normalize_space(strip_html(doc.get("source") or "")) or None,
                    "publisher_url": doc.get("url") or None,
                    "issn": _as_list(doc.get("issn")),
                    "authors": authors,
                },
            )
        )
    return items


_PARSERS = {
    SourceKind.RSS.value: lambda raw, source: parse_rss(raw, source),
    SourceKind.ARXIV.value: lambda raw, source: parse_arxiv(raw),
    SourceKind.EDGAR.value: lambda raw, source: parse_edgar_atom(raw),
    SourceKind.ERIC.value: lambda raw, source: parse_eric(raw),
}
