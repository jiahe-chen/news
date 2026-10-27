"""Approved source registry and the access gate used by agents.

Agents may only retrieve from approved sources: domains of approved, active
``Source`` rows plus a short allow-list of primary-record domains (regulators,
filings, research indexes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from newsren.models import DocType, Source, SourceKind, TrustTier, Vertical
from newsren.textutil import domain_of

APPROVED_SOURCES_PATH: Path = Path(__file__).parent / "data" / "sources.json"

# Registry fields copied onto ``Source`` rows by ``sync_sources``.
_REGISTRY_FIELDS = (
    "name",
    "kind",
    "url",
    "homepage",
    "vertical",
    "trust_tier",
    "default_doc_type",
    "license_note",
    "data_delay_note",
)

# Primary-record domains agents may consult directly (official records,
# research indexes). Keep this list short and reviewed.
PRIMARY_ALLOWLIST: frozenset[str] = frozenset(
    {
        "sec.gov",
        "efts.sec.gov",
        "data.sec.gov",
        "arxiv.org",
        "export.arxiv.org",
        "federalreserve.gov",
        "finra.org",
        "ies.ed.gov",
        "api.ies.ed.gov",
        "nces.ed.gov",
        "ed.gov",
        "nist.gov",
        "cisa.gov",
    }
)


def _domain_matches(domain: str, allowed: str) -> bool:
    return domain == allowed or domain.endswith("." + allowed)


def approved_domains(session: Session) -> set[str]:
    domains: set[str] = set()
    for src in session.scalars(select(Source).where(Source.approved.is_(True), Source.active.is_(True))):
        for url in (src.url, src.homepage):
            if url:
                d = domain_of(url)
                if d:
                    domains.add(d)
    return domains


def is_approved_url(session: Session, url: str) -> bool:
    """True only for http(s) URLs on an approved source domain or the primary allow-list."""
    if not url or not url.lower().startswith(("http://", "https://")):
        return False
    domain = domain_of(url)
    if not domain:
        return False
    if any(_domain_matches(domain, d) for d in PRIMARY_ALLOWLIST):
        return True
    return any(_domain_matches(domain, d) for d in approved_domains(session))


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def load_registry(path: Path | None = None) -> list[dict[str, Any]]:
    """Read and validate the approved source registry (``sources.json``).

    Raises ``ValueError`` on unknown vocabulary values or duplicate slugs so a
    bad registry edit fails loudly instead of silently ingesting mislabelled
    sources.
    """
    raw = json.loads((path or APPROVED_SOURCES_PATH).read_text(encoding="utf-8"))
    entries = raw["sources"] if isinstance(raw, dict) else raw
    seen: set[str] = set()
    for entry in entries:
        slug = entry.get("slug", "")
        if not slug or slug in seen:
            raise ValueError(f"source registry: missing or duplicate slug {slug!r}")
        seen.add(slug)
        for key, vocab in (
            ("kind", SourceKind),
            ("vertical", Vertical),
            ("trust_tier", TrustTier),
            ("default_doc_type", DocType),
        ):
            if entry.get(key) not in {v.value for v in vocab}:
                raise ValueError(f"source registry: {slug} has invalid {key} {entry.get(key)!r}")
        if not str(entry.get("url", "")).startswith("https://"):
            raise ValueError(f"source registry: {slug} url must be https")
    return entries


def sync_sources(session: Session, path: Path | None = None) -> list[Source]:
    """Upsert registry rows into ``sources`` (by slug) and return them.

    Registry fields overwrite the stored values; fetch bookkeeping
    (``last_fetched_at``/``last_status``) is preserved. Sources that are not in
    the registry (demo or manual rows) are left untouched.
    """
    rows: list[Source] = []
    for entry in load_registry(path):
        src = session.scalar(select(Source).where(Source.slug == entry["slug"]))
        if src is None:
            src = Source(slug=entry["slug"])
            session.add(src)
        for key in _REGISTRY_FIELDS:
            setattr(src, key, entry.get(key))
        src.approved = bool(entry.get("approved", True))
        src.active = bool(entry.get("active", True))
        rows.append(src)
    session.flush()
    return rows


def full_text_allowed(source: Source) -> bool:
    """Whether the license lets us store an item's full body text.

    Publisher feeds are stored as headline + summary + link only; the
    registry's ``license_note`` is the policy ("full text not stored").
    """
    note = (source.license_note or "").lower()
    return "full text not stored" not in note
