"""Approved source registry: validation, sync, and the agent access gate."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import func, select

from newsren.ingest.sources import (
    APPROVED_SOURCES_PATH,
    full_text_allowed,
    is_approved_url,
    load_registry,
    sync_sources,
)
from newsren.models import Source, SourceKind, TrustTier, Vertical


def test_registry_covers_all_verticals_with_verified_kinds():
    entries = load_registry()
    by_vertical: dict[str, list[dict]] = {}
    for e in entries:
        by_vertical.setdefault(e["vertical"], []).append(e)
    assert set(by_vertical) == {v.value for v in Vertical}
    for vertical, rows in by_vertical.items():
        assert len(rows) >= 4, vertical
    kinds = {e["kind"] for e in entries}
    assert {"rss", "arxiv", "edgar", "eric"} <= kinds
    assert all(e["url"].startswith("https://") for e in entries)
    assert all(e.get("license_note") for e in entries)


def test_registry_labels_trust_tiers_licences_and_delays():
    entries = {e["slug"]: e for e in load_registry()}
    for slug, e in entries.items():
        if e["kind"] == "edgar":
            assert e["trust_tier"] == TrustTier.PRIMARY
            assert e["default_doc_type"] == "filing"
            assert "acceptance" in e["data_delay_note"]
            assert "Public domain US government record" in e["license_note"]
        if e["kind"] == "arxiv":
            assert e["default_doc_type"] == "research_preprint"
            assert e["trust_tier"] == TrustTier.PRIMARY
        if e["kind"] == "rss" and e["trust_tier"] == TrustTier.INDEPENDENT:
            assert "full text not stored" in e["license_note"], slug
    assert "Public domain US government record" in entries["sec-press-releases"]["license_note"]
    assert "Public domain US government record" in entries["federal-reserve-press"]["license_note"]
    assert entries["eric-ai-learning"]["kind"] == "eric"


def test_registry_rejects_bad_vocabulary(tmp_path):
    bad = {"sources": [{**load_registry()[0], "trust_tier": "trusted"}]}
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="trust_tier"):
        load_registry(path)
    dup = {"sources": [load_registry()[0], load_registry()[0]]}
    path.write_text(json.dumps(dup))
    with pytest.raises(ValueError, match="duplicate"):
        load_registry(path)


def test_sync_sources_upserts_and_preserves_fetch_state(session, factory):
    demo = factory.source(name="Demo row")
    rows = sync_sources(session)
    assert len(rows) == len(load_registry())
    assert APPROVED_SOURCES_PATH.exists()
    bbc = session.scalars(select(Source).where(Source.slug == "bbc-technology")).one()
    assert bbc.kind == SourceKind.RSS and bbc.approved and bbc.active
    bbc.last_status = "ok: 5 items"
    bbc.name = "renamed locally"
    session.flush()

    sync_sources(session)
    total = session.scalar(select(func.count(Source.id)))
    assert total == len(load_registry()) + 1  # registry rows + the untouched non-registry row
    assert bbc.name == "BBC News - Technology"  # registry wins
    assert bbc.last_status == "ok: 5 items"  # fetch bookkeeping kept
    assert session.get(Source, demo.id).name == "Demo row"


def test_is_approved_url_uses_registry_and_primary_allowlist(session):
    sync_sources(session)
    assert is_approved_url(session, "https://www.bbc.co.uk/news/articles/c893zwlev2xpo")
    assert is_approved_url(session, "https://feeds.bbci.co.uk/news/technology/rss.xml")
    assert is_approved_url(session, "https://www.sec.gov/Archives/edgar/data/1/x-index.htm")
    assert is_approved_url(session, "https://eric.ed.gov/?id=EJ1")  # ERIC homepage domain
    assert not is_approved_url(session, "https://unknown-blog.example.com/post")
    assert not is_approved_url(session, "ftp://www.sec.gov/file")
    bbc = session.scalars(select(Source).where(Source.slug == "bbc-technology")).one()
    bbc_biz = session.scalars(select(Source).where(Source.slug == "bbc-business")).one()
    bbc.active = bbc_biz.active = False
    session.flush()
    assert not is_approved_url(session, "https://www.bbc.co.uk/news/articles/c893zwlev2xpo")


def test_full_text_policy_follows_license_note(factory):
    src = factory.source()
    src.license_note = "Headline + publisher summary + link; full text not stored."
    assert not full_text_allowed(src)
    src.license_note = "Public domain US government record (SEC)."
    assert full_text_allowed(src)
