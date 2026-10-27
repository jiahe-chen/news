"""Duplicate and near-duplicate detection for ingested items.

Three tests, strongest first:

1. the same canonical URL (the item was already ingested);
2. identical normalized content (``content_hash``), e.g. a verbatim syndicated copy;
3. SimHash Hamming distance <= 3, e.g. a wire story republished with a credit line. Because 64-bit
   SimHash is noisy on short texts, candidates up to 12 bits away also count
   when the caller supplies the item text and their word 3-shingle Jaccard
   similarity is >= 0.8 (a retitled syndicated copy); unrelated articles never
   share that much verbatim text.

Tests 2 and 3 only consider articles dated within 14 days of the item (so a
headline that recurs every year is not a "copy"). Matches resolve to the
*original* article so syndicated copies
all point at one root (``Article.duplicate_of_id``).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsren.models import Article, utcnow
from newsren.textutil import hamming, jaccard, ngrams, tokenize

DUPLICATE_WINDOW_DAYS = 14
MAX_SIMHASH_DISTANCE = 3
VERIFY_SIMHASH_DISTANCE = 12
MIN_SHINGLE_JACCARD = 0.8
_ZERO_SIMHASH = "0" * 16


def shingles(text: str, n: int = 3) -> set[str]:
    """Word n-gram set used to verify near-duplicates."""
    tokens = tokenize(text, keep_stopwords=True)
    return set(ngrams(tokens, n)) if len(tokens) >= n else set(tokens)


def article_fingerprint_text(article: Article) -> str:
    """The text an article's hashes are computed from (title, summary, body)."""
    return f"{article.title} {article.summary or ''} {article.content or ''}"


def root_article(session: Session, article: Article) -> Article:
    """Follow ``duplicate_of_id`` to the original article."""
    seen: set[int] = set()
    while article.duplicate_of_id is not None and article.duplicate_of_id not in seen:
        seen.add(article.id)
        parent = session.get(Article, article.duplicate_of_id)
        if parent is None:
            break
        article = parent
    return article


def find_duplicate(
    session: Session,
    canonical_url: str | None,
    content_hash: str | None,
    simhash: str | None,
    *,
    around: datetime | None = None,
    window_days: int = DUPLICATE_WINDOW_DAYS,
    max_distance: int = MAX_SIMHASH_DISTANCE,
    text: str | None = None,
) -> Article | None:
    """Return an existing article this item duplicates, or None.

    * Exact canonical URL: returns that very article (callers compare
      ``canonical_url`` to tell "already ingested" from "syndicated copy").
    * Exact content hash, or SimHash within ``max_distance`` bits, among
      articles dated within ``window_days`` of ``around`` (default: now):
      returns the root original.

    Pass ``None`` for ``content_hash``/``simhash`` to skip those tests (the
    pipeline does so for very short texts and for records with their own
    identifiers, such as filings, where templated text would collide). Pass
    ``text`` (title + summary + body) to enable the shingle-verified fallback.
    """
    if canonical_url:
        hit = session.scalars(select(Article).where(Article.canonical_url == canonical_url)).first()
        if hit is not None:
            return hit

    center = around or utcnow()
    lo, hi = center - timedelta(days=window_days), center + timedelta(days=window_days)
    when = func.coalesce(Article.published_at, Article.retrieved_at)

    if content_hash:
        hit = session.scalars(
            select(Article)
            .where(Article.content_hash == content_hash, when >= lo, when <= hi)
            .order_by(Article.duplicate_of_id.is_not(None), Article.id)
        ).first()
        if hit is not None:
            return root_article(session, hit)

    if simhash and simhash != _ZERO_SIMHASH:
        rows = session.execute(
            select(Article.id, Article.simhash, Article.duplicate_of_id).where(
                when >= lo, when <= hi, Article.simhash != _ZERO_SIMHASH
            )
        ).all()
        candidates: list[tuple[int, bool, int]] = []  # (distance, is_copy, id): originals first on ties
        verify_limit = VERIFY_SIMHASH_DISTANCE if text else max_distance
        for art_id, art_hash, dup_of in rows:
            if not art_hash:
                continue
            distance = hamming(simhash, art_hash)
            if distance <= verify_limit:
                candidates.append((distance, dup_of is not None, art_id))
        candidates.sort()
        item_shingles: set[str] | None = None
        for distance, _, art_id in candidates:
            hit = session.get(Article, art_id)
            if hit is None:
                continue
            if distance <= max_distance:
                return root_article(session, hit)
            if item_shingles is None:
                item_shingles = shingles(text or "")
            if jaccard(item_shingles, shingles(article_fingerprint_text(hit))) >= MIN_SHINGLE_JACCARD:
                return root_article(session, hit)
    return None
