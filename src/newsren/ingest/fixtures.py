"""Fictional demo data loader.

``demo_articles.json`` holds a clearly fictional corpus (invented
organizations, people, and figures on reserved ``.example`` domains) that
exercises every use case in the plan: an AI model launch followed by vendor
benchmark claims, an independent evaluation, a correction, a product launch,
school adoption, and regulation; a university press release versus the
underlying preprint; a semiconductor capacity projection versus reported
output; a bank backtest versus live use; funding rounds; an earnings release
with guidance; regulator enforcement and guidance; an education policy; a
cybersecurity advisory; and supply-chain/skills stories. Each storyline has an
opinion piece, and three syndicated copies exercise duplicate detection.

Dates are stored as offsets (``days_ago``/``hours_ago``) and computed at load
time so the demo always looks current. Articles go through
``pipeline.store_item``, so classification and deduplication are the real ones.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from newsren.ingest.fetchers import FetchedItem
from newsren.ingest.pipeline import store_item
from newsren.models import Article, DocType, Source, SourceKind, TrustTier, Vertical, utcnow
from newsren.textutil import canonical_url, slugify

DEMO_ARTICLES_PATH: Path = Path(__file__).parent / "data" / "demo_articles.json"
DEMO_LICENSE_NOTE = (
    "Fictional demo content for The Newsren: organizations, people, products, and figures are invented. "
    "Not a real publication."
)


def load_demo_dataset(path: Path | None = None) -> dict[str, Any]:
    """Read and validate the demo corpus; resolves syndicated copies into full articles."""
    data = json.loads((path or DEMO_ARTICLES_PATH).read_text(encoding="utf-8"))
    sources = {s["slug"]: s for s in data["sources"]}
    for slug, src in sources.items():
        if src["vertical"] not in {v.value for v in Vertical}:
            raise ValueError(f"demo source {slug}: invalid vertical {src['vertical']!r}")
        if src["trust_tier"] not in {t.value for t in TrustTier}:
            raise ValueError(f"demo source {slug}: invalid trust tier {src['trust_tier']!r}")
        if not src["homepage"].split("/")[2].endswith(".example"):
            raise ValueError(f"demo source {slug}: homepage must use a reserved .example domain")

    by_key: dict[str, dict[str, Any]] = {}
    articles: list[dict[str, Any]] = []
    for raw in data["articles"]:
        if raw["key"] in by_key:
            raise ValueError(f"duplicate demo article key {raw['key']!r}")
        if raw["source_slug"] not in sources:
            raise ValueError(f"demo article {raw['key']}: unknown source {raw['source_slug']!r}")
        article = dict(raw)
        if "syndicate_of" in raw:
            original = by_key[raw["syndicate_of"]]  # originals are listed before their copies
            content = original["content"]
            if raw.get("credit_line"):
                content = f"{content}\n\n{raw['credit_line']}"
            article = {
                **{k: original[k] for k in ("summary", "author", "doc_type")},
                "title": raw.get("title") or original["title"],
                "content": content,
                **{k: v for k, v in raw.items() if k not in ("title", "credit_line")},
            }
        if article["doc_type"] not in {d.value for d in DocType}:
            raise ValueError(f"demo article {raw['key']}: invalid doc_type {article['doc_type']!r}")
        by_key[raw["key"]] = article
        articles.append(article)
    return {**data, "sources": list(sources.values()), "articles": articles}


def demo_article_url(source_homepage: str, key: str, title: str) -> str:
    """Stable URL for a demo article (independent of load date, so reloading is idempotent)."""
    return f"{source_homepage.rstrip('/')}/stories/{key}-{slugify(title, 50)}"


def _demo_source(session: Session, spec: dict[str, Any]) -> Source:
    src = session.scalars(select(Source).where(Source.slug == spec["slug"])).first()
    if src is None:
        src = Source(slug=spec["slug"])
        session.add(src)
    src.name = spec["name"]
    src.kind = SourceKind.DEMO
    src.homepage = spec["homepage"]
    src.url = spec.get("url") or f"{spec['homepage'].rstrip('/')}/feed"
    src.vertical = spec["vertical"]
    src.trust_tier = spec["trust_tier"]
    src.default_doc_type = spec["default_doc_type"]
    src.license_note = DEMO_LICENSE_NOTE
    src.approved = True
    src.active = True
    return src


def load_demo_data(session: Session, now: datetime | None = None, path: Path | None = None) -> int:
    """Load the fictional demo corpus; returns the number of articles newly stored.

    Idempotent: sources are upserted by slug and articles are skipped when
    their (stable) URL is already present, so a second call returns 0.
    Syndicated copies are stored as duplicates by the normal dedup logic.
    """
    now = now or utcnow()
    dataset = load_demo_dataset(path)
    sources = {spec["slug"]: _demo_source(session, spec) for spec in dataset["sources"]}
    session.flush()

    def published(article: dict[str, Any]) -> datetime:
        return now - timedelta(days=article["days_ago"], hours=article.get("hours_ago", 0))

    loaded = 0
    # Oldest first, so originals exist before their syndicated copies arrive.
    for article in sorted(dataset["articles"], key=published):
        source = sources[article["source_slug"]]
        url = demo_article_url(source.homepage or "", article["key"], article["title"])
        exists = session.scalars(select(Article.id).where(Article.canonical_url == canonical_url(url))).first()
        if exists is not None:
            continue
        extra = {
            "is_demo": True,
            "demo_key": article["key"],
            "storyline": article["storyline"],
            **article.get("extra", {}),
        }
        if article.get("syndicate_of"):
            extra["republished_from_demo_key"] = article["syndicate_of"]
        item = FetchedItem(
            url=url,
            title=article["title"],
            summary=article["summary"],
            content=article["content"],
            author=article.get("author"),
            published_at=published(article),
            doc_type=article["doc_type"],
            extra=extra,
        )
        store_item(session, source, item, now=now)
        loaded += 1
    session.flush()
    return loaded
