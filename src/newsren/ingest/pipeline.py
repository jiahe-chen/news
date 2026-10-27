"""Ingestion pipeline: normalize, classify, deduplicate, and store fetched items.

``ingest_all`` fetches every approved, active source and never lets one
failing source abort the run: failures are recorded on ``Source.last_status``
and in the ``IngestReport``. Functions flush but never commit.

Classification precedence for ``doc_type``: an explicit parser/item override
(arXiv, EDGAR, ERIC, demo data) wins; otherwise title/URL/feed-tag cues
(opinion, press release, sponsored, preprint, security advisory) apply;
otherwise the source's default. The trust tier follows the document type:
opinion is ``commentary``; press releases and sponsored posts are ``vendor``
unless the source is itself a primary record (a regulator's or company's
official release); everything else inherits the source's tier.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from newsren.config import Settings, get_settings
from newsren.ingest.dedup import find_duplicate
from newsren.ingest.fetchers import SPONSORED_LEAD, FetchedItem, FetchError, fetch_source
from newsren.models import Article, DocType, Source, SourceKind, TrustTier, utcnow
from newsren.textutil import (
    canonical_url,
    content_hash,
    normalize_space,
    parse_datetime,
    simhash_hex,
    strip_html,
    tokenize,
    truncate,
)

log = logging.getLogger(__name__)

# Texts shorter than this are too thin for content/SimHash matching (a bare
# headline would collide with unrelated items); they dedupe by URL only.
MIN_TOKENS_FOR_CONTENT_HASH = 15
MIN_TOKENS_FOR_SIMHASH = 25
SUMMARY_MAX_CHARS = 2000
CONTENT_MAX_CHARS = 60_000
MAX_ITEM_ERRORS_PER_SOURCE = 5

# Records that carry their own identifier are deduplicated by URL only;
# templated summaries (filings, index records) would otherwise look alike.
_IDENTIFIER_KEYS = ("accession", "arxiv_id", "eric_id")

_OPINION_PREFIX = re.compile(
    r"^\s*\[?(?:opinion|op-ed|op ed|commentary|column|editorial|guest essay|guest column|viewpoint|"
    r"my view|letter to the editor|letters)\]?\s*[:|\-–—]\s*",
    re.I,
)
_OPINION_SUFFIX = re.compile(r"[\(\[\|–—-]\s*(?:opinion|op-ed|commentary|column|editorial)\s*[\)\]]?\s*$", re.I)
_OPINION_PHRASE = re.compile(r"\bop-ed\b", re.I)
_OPINION_PATH = re.compile(r"/(?:opinion|opinions|op-eds?|commentary|columns?|editorials?|views)/", re.I)
_OPINION_TAGS = {"opinion", "op-ed", "op-eds", "commentary", "column", "columns", "editorial", "editorials"}
_PRESS_RELEASE_TITLE = re.compile(r"\bpress release\b", re.I)
_PRESS_RELEASE_LEAD = re.compile(
    r"^\s*(?:press release|for immediate release)\b|\((?:business wire|globe ?newswire|pr ?newswire|accesswire)\)",
    re.I,
)
_PREPRINT_PREFIX = re.compile(r"^\s*\[?preprint\]?\s*[:|\-–—]\s*", re.I)
_PREPRINT_MENTION = re.compile(r"\bpre-?print\b|\bnot (?:yet )?(?:been )?peer[- ]reviewed\b", re.I)
_CORRECTION = re.compile(
    r"^\s*\[?(?:correction|corrected|clarification)\]?\s*[:|\-–—]|\((?:corrected|correction)\)", re.I
)
_ADVISORY = re.compile(
    r"\bCVE-\d{4}-\d{3,}\b|\bsecurity (?:advisory|issue|update|alert|flaw)\b|\bvulnerabilit(?:y|ies)\b|"
    r"\bactively exploited\b|\bzero-day\b",
    re.I,
)
_SYNDICATION = re.compile(
    r"\b(?:originally (?:published|appeared) (?:by|in|on)|first (?:published|appeared) (?:by|in|on)|"
    r"(?:republished|reprinted) (?:from|with permission from))\s+(?:the\s+)?"
    r"([A-Z][\w&'’.-]*(?:\s+(?:(?:of|and|&)\s+)?[A-Z][\w&'’.-]*)*)"
)
_EARNINGS_CALL = re.compile(r"\bearnings call transcript\b|\bQ[1-4] \d{4} earnings call\b", re.I)


@dataclass
class IngestReport:
    """Outcome of one ``ingest_all`` run."""

    sources_ok: int = 0
    sources_failed: int = 0
    new_articles: int = 0
    duplicates: int = 0  # syndicated / near-duplicate copies stored with duplicate_of_id
    errors: list[str] = field(default_factory=list)
    already_seen: int = 0  # items whose URL was ingested on an earlier run
    per_source: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class Classification:
    doc_type: str
    trust_tier: str
    cues: list[str] = field(default_factory=list)
    flags: dict[str, bool] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def classify_item(source: Source, item: FetchedItem) -> Classification:
    """Decide doc_type and trust tier for an item (see module docstring for precedence)."""
    title = normalize_space(strip_html(item.title))
    lead = normalize_space(strip_html(item.summary or item.content))[:300]
    tags = {str(t).strip().lower() for t in item.extra.get("tags") or []}
    path = item.url.split("?", 1)[0]
    cues: list[str] = []
    flags: dict[str, bool] = {}

    doc_type = item.doc_type or source.default_doc_type or DocType.NEWS
    cue: tuple[str, str] | None = None  # (doc_type, cue label)
    if item.doc_type:
        cues.append("override")
    elif _OPINION_PREFIX.search(title) or _OPINION_SUFFIX.search(title) or _OPINION_PHRASE.search(title):
        cue = (DocType.OPINION, "title:opinion")
    elif _OPINION_PATH.search(path):
        cue = (DocType.OPINION, "url:opinion")
    elif tags & _OPINION_TAGS:
        cue = (DocType.OPINION, "tag:opinion")
    elif item.extra.get("sponsored") or SPONSORED_LEAD.search(lead):
        cue = (DocType.BLOG, "sponsored")
        flags["sponsored"] = True
    elif _PRESS_RELEASE_TITLE.search(title) or _PRESS_RELEASE_LEAD.search(lead):
        cue = (DocType.PRESS_RELEASE, "press_release")
    elif _PREPRINT_PREFIX.search(title):
        cue = (DocType.RESEARCH_PREPRINT, "title:preprint")
    elif _EARNINGS_CALL.search(title):
        cue = (DocType.EARNINGS_CALL, "title:earnings_call")
    elif (
        source.trust_tier == TrustTier.PRIMARY
        and doc_type in (DocType.NEWS, DocType.PRESS_RELEASE)
        and _ADVISORY.search(title)
    ):
        cue = (DocType.ADVISORY, "title:advisory")
    if cue is not None:
        doc_type = cue[0]
        cues.append(cue[1])

    source_tier = source.trust_tier or TrustTier.INDEPENDENT
    if doc_type == DocType.OPINION:
        trust_tier = TrustTier.COMMENTARY
    elif flags.get("sponsored"):
        trust_tier = TrustTier.VENDOR
    elif doc_type == DocType.PRESS_RELEASE:
        # A regulator's or company's own official release is the primary record.
        trust_tier = TrustTier.PRIMARY if source_tier == TrustTier.PRIMARY else TrustTier.VENDOR
    else:
        trust_tier = source_tier

    text = f"{title} {item.summary} {item.content[:4000]}"
    if doc_type != DocType.RESEARCH_PREPRINT and _PREPRINT_MENTION.search(text):
        flags["mentions_preprint"] = True
    if _CORRECTION.search(title):
        flags["is_correction"] = True
    return Classification(doc_type=str(doc_type), trust_tier=str(trust_tier), cues=cues, flags=flags)


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def _clean_body(text: str | None) -> str:
    """Plain text with paragraph breaks; markup stripped only when present."""
    if not text:
        return ""
    if re.search(r"<[a-zA-Z/][^>]*>", text):
        return strip_html(text)
    paragraphs = [normalize_space(p) for p in re.split(r"\n\s*\n", text.replace("\r\n", "\n"))]
    return "\n\n".join(p for p in paragraphs if p)


def _published_at(value: datetime | str | None, now: datetime) -> datetime | None:
    published = parse_datetime(value) if value is not None else None
    if published is not None and published > now + timedelta(hours=1):
        # Feeds occasionally carry future timestamps (bad time zones); never date news ahead of retrieval.
        return now
    return published


def _store_item(
    session: Session, source: Source, item: FetchedItem, now: datetime | None = None
) -> tuple[Article, str]:
    """Store one item. Status is "new", "duplicate" (copy stored), or "seen" (URL known)."""
    now = now or utcnow()
    url = (item.url or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"item has no http(s) URL: {url[:120]!r}")
    canonical = canonical_url(url)

    title = truncate(normalize_space(strip_html(item.title)), 500)
    summary = truncate(normalize_space(strip_html(item.summary)), SUMMARY_MAX_CHARS)
    content = _clean_body(item.content)[:CONTENT_MAX_CHARS]
    if not title:
        title = truncate(summary or normalize_space(content), 200)
    if not title:
        raise ValueError(f"item has no title or text: {url[:120]}")

    existing = session.scalars(select(Article).where(Article.canonical_url == canonical)).first()
    if existing is not None:
        return existing, "seen"

    published_at = _published_at(item.published_at, now)
    classification = classify_item(source, item)
    chash = content_hash(title, summary, content)
    shash = simhash_hex(f"{title} {summary} {content}")
    # Count only distinct text: many regulator feeds repeat the headline as the summary.
    distinct = " ".join(dict.fromkeys(part for part in (title, summary, content) if part))
    n_tokens = len(tokenize(distinct, keep_stopwords=True))
    has_identifier = any(item.extra.get(k) for k in _IDENTIFIER_KEYS)
    duplicate = find_duplicate(
        session,
        None,
        None if has_identifier or n_tokens < MIN_TOKENS_FOR_CONTENT_HASH else chash,
        None if has_identifier or n_tokens < MIN_TOKENS_FOR_SIMHASH else shash,
        around=published_at or now,
        text=f"{title} {summary} {content}",
    )

    extra = dict(item.extra)
    syndicated = _SYNDICATION.search(f"{summary} {content[:4000]}")
    if syndicated:
        extra.setdefault("syndicated_from", syndicated.group(1).rstrip(".’' "))
    if classification.cues:
        extra["classification_cues"] = classification.cues
    extra.update(classification.flags)
    is_demo = source.kind == SourceKind.DEMO or bool(extra.get("is_demo"))
    article = Article(
        source=source,
        url=url,
        canonical_url=canonical,
        title=title,
        author=truncate(normalize_space(strip_html(item.author)), 300) or None,
        summary=summary or None,
        content=content or None,
        published_at=published_at,
        retrieved_at=now,
        vertical=source.vertical,
        doc_type=classification.doc_type,
        trust_tier=classification.trust_tier,
        content_hash=chash,
        simhash=shash,
        duplicate_of_id=duplicate.id if duplicate is not None else None,  # already the root original
        is_demo=is_demo,
        extra=extra,
    )
    session.add(article)
    session.flush()
    return article, "duplicate" if duplicate is not None else "new"


def store_item(
    session: Session, source: Source, item: FetchedItem, now: datetime | None = None
) -> tuple[Article, bool]:
    """Normalize, classify, deduplicate, and store one fetched item.

    Returns ``(article, is_new)``:

    * ``(new_article, True)`` - a new original article was stored;
    * ``(copy, False)`` - a syndicated or near-duplicate copy was stored as a new
      row with ``duplicate_of_id`` pointing at the original (check
      ``article.is_duplicate``);
    * ``(existing, False)`` - the URL was already ingested; nothing is stored and
      the existing row is returned unchanged.

    Raises ``ValueError`` for items without an http(s) URL or any text.
    """
    article, status = _store_item(session, source, item, now)
    return article, status == "new"


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


def _sources_to_fetch(session: Session, source_slugs: Iterable[str] | None, report: IngestReport) -> list[Source]:
    query = select(Source).where(
        Source.approved.is_(True),
        Source.active.is_(True),
        Source.kind.not_in([SourceKind.DEMO.value, SourceKind.MANUAL.value]),
    )
    if source_slugs is None:
        return list(session.scalars(query.order_by(Source.vertical, Source.slug)))
    wanted = list(dict.fromkeys(source_slugs))
    found = {s.slug: s for s in session.scalars(query.where(Source.slug.in_(wanted)))}
    for slug in wanted:
        if slug not in found:
            report.errors.append(f"{slug}: not an approved, active, fetchable source; skipped")
    return [found[s] for s in wanted if s in found]


def ingest_source(
    session: Session, source: Source, settings: Settings, report: IngestReport, now: datetime | None = None
) -> None:
    """Fetch one source and store its items, recording the outcome on the source and report."""
    now = now or utcnow()
    stats = {"items": 0, "new": 0, "duplicates": 0, "seen": 0, "errors": 0}
    source.last_fetched_at = now
    try:
        items = fetch_source(source, settings)
    except Exception as exc:  # FetchError, or anything unexpected: contain it to this source
        message = str(exc) if isinstance(exc, FetchError) else f"{type(exc).__name__}: {exc}"
        source.last_status = truncate(f"error: {message}", 300)
        report.sources_failed += 1
        report.errors.append(f"{source.slug}: {message}")
        report.per_source[source.slug] = {**stats, "status": "error", "error": message}
        log.warning("Ingest failed for %s: %s", source.slug, message)
        session.flush()
        return

    for item in items:
        stats["items"] += 1
        try:
            with session.begin_nested():
                _, status = _store_item(session, source, item, now)
        except Exception as exc:  # one malformed item must not drop the rest of the feed
            stats["errors"] += 1
            if stats["errors"] <= MAX_ITEM_ERRORS_PER_SOURCE:
                report.errors.append(f"{source.slug}: item {item.url[:120]!r}: {type(exc).__name__}: {exc}")
            continue
        key = {"new": "new", "duplicate": "duplicates", "seen": "seen"}[status]
        stats[key] += 1

    report.sources_ok += 1
    report.new_articles += stats["new"]
    report.duplicates += stats["duplicates"]
    report.already_seen += stats["seen"]
    status_text = (
        f"ok: {stats['items']} items, {stats['new']} new, {stats['duplicates']} duplicates, "
        f"{stats['seen']} already seen" + (f", {stats['errors']} item errors" if stats["errors"] else "")
    )
    source.last_status = truncate(status_text, 300)
    report.per_source[source.slug] = {**stats, "status": "ok"}
    session.flush()


def ingest_all(
    session: Session, settings: Settings | None = None, source_slugs: Iterable[str] | None = None
) -> IngestReport:
    """Fetch every approved, active source (optionally only ``source_slugs``).

    Demo and manual sources are never fetched. Each source's
    ``last_fetched_at``/``last_status`` is updated; a failing source is
    recorded and skipped, never aborting the run.
    """
    settings = settings or get_settings()
    report = IngestReport()
    now = utcnow()
    for source in _sources_to_fetch(session, source_slugs, report):
        ingest_source(session, source, settings, report, now)
    session.flush()
    return report
