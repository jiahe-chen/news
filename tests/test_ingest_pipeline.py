"""store_item normalization/classification/dedup and ingest_all orchestration."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from newsren import db
from newsren.ingest import pipeline
from newsren.ingest.fetchers import FetchedItem, FetchError, parse_edgar_atom, parse_rss
from newsren.ingest.pipeline import IngestReport, classify_item, ingest_all, store_item
from newsren.models import Article, DocType, Source, SourceKind, TrustTier, Vertical, utcnow

DATA = Path(__file__).parent / "data" / "ingest"
BODY = (
    "Pinecrest Mobile said Wednesday it has signed a multi-year agreement with Corvane Semiconductor for "
    "3nm-class wafers produced at Corvane's expanding Calder Valley Fab 3. The companies did not disclose "
    "volumes or prices. Supply-chain analysts said long-term agreements are common when capacity is tight."
)


def _item(title: str, url: str = "https://news.example/a", **kw) -> FetchedItem:
    kw.setdefault("summary", "A summary of the item for readers.")
    return FetchedItem(url=url, title=title, **kw)


def _count(session) -> int:
    return session.scalar(select(func.count(Article.id)))


# --------------------------------------------------------------------- store_item


def test_store_item_normalizes_and_stores(session, factory):
    src = factory.source(vertical=Vertical.INDUSTRY)
    published = utcnow() - timedelta(hours=3)
    item = _item(
        "Pinecrest <b>signs</b> wafer deal",
        url="https://www.News.example/a/?utm_source=rss&at_medium=RSS",
        summary="<p>Pinecrest &amp; Corvane sign a deal.</p>",
        content=BODY,
        author="Owen Marchetti",
        published_at=published,
        extra={"guid": "g1"},
    )
    article, is_new = store_item(session, src, item)
    assert is_new
    assert article.canonical_url == "https://news.example/a"
    assert article.title == "Pinecrest signs wafer deal"
    assert article.summary == "Pinecrest & Corvane sign a deal."
    assert article.vertical == Vertical.INDUSTRY
    assert article.doc_type == DocType.NEWS and article.trust_tier == TrustTier.INDEPENDENT
    assert article.published_at == published and article.retrieved_at is not None
    assert len(article.content_hash) == 64 and len(article.simhash) == 16
    assert article.extra["guid"] == "g1" and article.is_demo is False
    assert db.search_article_ids(session, "Pinecrest wafer")[0][0] == article.id


def test_store_item_same_url_is_seen_not_stored(session, factory):
    src = factory.source()
    first, _ = store_item(session, src, _item("Original headline", content=BODY))
    again, is_new = store_item(session, src, _item("Edited headline", url="https://news.example/a?utm_campaign=x"))
    assert not is_new and again.id == first.id
    assert _count(session) == 1


def test_store_item_syndicated_copy_is_stored_as_duplicate(session, factory):
    wire, outlet = factory.source(name="Wire"), factory.source(name="Outlet")
    original, _ = store_item(session, wire, _item("Pinecrest signs wafer deal", content=BODY))
    copy, is_new = store_item(
        session,
        outlet,
        _item(
            "Pinecrest signs wafer deal",
            url="https://outlet.example/business/pinecrest",
            content=BODY + "\n\nThis article first appeared in Supply Signal and is republished with permission.",
        ),
    )
    assert not is_new
    assert copy.id != original.id and copy.duplicate_of_id == original.id and copy.is_duplicate
    assert copy.extra["syndicated_from"] == "Supply Signal"
    assert _count(session) == 2


def test_short_items_are_not_content_deduplicated(session, factory):
    src = factory.source(trust_tier=TrustTier.PRIMARY, doc_type=DocType.REGULATORY)
    title = "Minutes of the Board's discount rate meetings"
    _, new1 = store_item(session, src, _item(title, url="https://fed.example/1", summary=title))
    _, new2 = store_item(session, src, _item(title, url="https://fed.example/2", summary=title))
    assert new1 and new2


def test_filings_with_identifiers_dedupe_by_url_only(session, factory):
    src = factory.source(vertical=Vertical.FINANCE, trust_tier=TrustTier.PRIMARY, doc_type=DocType.FILING)
    items = parse_edgar_atom((DATA / "edgar_current_10q.atom").read_bytes())
    results = [store_item(session, src, item) for item in items]
    assert all(is_new for _, is_new in results)  # templated summaries must not collide
    article = results[0][0]
    assert article.doc_type == DocType.FILING and article.extra["accession"]


def test_future_publication_dates_are_clamped(session, factory):
    now = utcnow()
    article, _ = store_item(
        session, factory.source(), _item("Clock skew", published_at=now + timedelta(days=2)), now=now
    )
    assert article.published_at == now


def test_store_item_rejects_items_without_url(session, factory):
    with pytest.raises(ValueError, match="URL"):
        store_item(session, factory.source(), _item("No link", url="urn:x"))


# --------------------------------------------------------------------- classification cues


@pytest.mark.parametrize(
    ("title", "url", "tags", "expected_doc", "expected_tier"),
    [
        ("Opinion: Benchmarks are claims", "https://n.example/a", [], DocType.OPINION, TrustTier.COMMENTARY),
        ("Op-Ed: Slow down the AI rollout", "https://n.example/b", [], DocType.OPINION, TrustTier.COMMENTARY),
        (
            "Commentary: Chip capacity is a promise",
            "https://n.example/c",
            [],
            DocType.OPINION,
            TrustTier.COMMENTARY,
        ),
        ("Column: Apprenticeships matter", "https://n.example/d", [], DocType.OPINION, TrustTier.COMMENTARY),
        ("Why the AI rules matter | Opinion", "https://n.example/e", [], DocType.OPINION, TrustTier.COMMENTARY),
        ("Election tips", "https://n.example/opinion/views/2026/x", [], DocType.OPINION, TrustTier.COMMENTARY),
        ("States and AI", "https://n.example/f", ["Opinion", "AI"], DocType.OPINION, TrustTier.COMMENTARY),
        (
            "Press release: Lumen Tutor launches",
            "https://n.example/g",
            [],
            DocType.PRESS_RELEASE,
            TrustTier.VENDOR,
        ),
        (
            "Preprint: AI tutors and transfer",
            "https://n.example/h",
            [],
            DocType.RESEARCH_PREPRINT,
            TrustTier.INDEPENDENT,
        ),
        (
            "Automaker recalls cars over steering column fault",
            "https://n.example/i",
            [],
            DocType.NEWS,
            TrustTier.INDEPENDENT,
        ),
        ("Pinecrest signs wafer deal", "https://n.example/j", [], DocType.NEWS, TrustTier.INDEPENDENT),
    ],
)
def test_doc_type_cues(factory, title, url, tags, expected_doc, expected_tier):
    src = factory.source()
    result = classify_item(src, _item(title, url=url, extra={"tags": tags}))
    assert (result.doc_type, result.trust_tier) == (expected_doc, expected_tier)


def test_press_release_from_primary_source_stays_primary(factory):
    regulator = factory.source(trust_tier=TrustTier.PRIMARY, doc_type=DocType.REGULATORY)
    result = classify_item(regulator, _item("Press release: Commission adopts rule"))
    assert (result.doc_type, result.trust_tier) == (DocType.PRESS_RELEASE, TrustTier.PRIMARY)
    wire = classify_item(
        factory.source(), _item("Acme opens plant", summary="AUSTIN (BUSINESS WIRE)--Acme today...")
    )
    assert (wire.doc_type, wire.trust_tier) == (DocType.PRESS_RELEASE, TrustTier.VENDOR)


def test_sponsored_and_advisory_cues(factory):
    sponsored = classify_item(factory.source(), _item("Robot safety", extra={"sponsored": True}))
    assert (sponsored.doc_type, sponsored.trust_tier) == (DocType.BLOG, TrustTier.VENDOR)
    agency = factory.source(trust_tier=TrustTier.PRIMARY)
    advisory = classify_item(agency, _item("Critical security issue affecting PHP (CVE-2026-109618)"))
    assert advisory.doc_type == DocType.ADVISORY and advisory.trust_tier == TrustTier.PRIMARY
    # Newsroom reporting on a vulnerability stays news.
    assert classify_item(factory.source(), _item("Hackers exploit VPN vulnerability")).doc_type == DocType.NEWS


def test_explicit_override_wins_and_flags_are_recorded(session, factory):
    src = factory.source()
    result = classify_item(src, _item("Opinion: looks like opinion", doc_type=DocType.RESEARCH_PREPRINT))
    assert result.doc_type == DocType.RESEARCH_PREPRINT and result.cues == ["override"]
    article, _ = store_item(
        session,
        src,
        _item(
            "Correction: Independent benchmark score was 84.3%",
            summary="The study, which has not been peer reviewed, was misreported.",
        ),
    )
    assert article.extra["is_correction"] and article.extra["mentions_preprint"]


def test_real_feed_sample_end_to_end(session, factory):
    src = factory.source(vertical=Vertical.EDUCATION)
    src.license_note = "Headline + publisher summary + link; full text not stored."
    for item in parse_rss((DATA / "the74_rss.xml").read_bytes(), src):
        store_item(session, src, item)
    rows = session.scalars(select(Article).where(Article.source_id == src.id)).all()
    assert len(rows) == 5
    opinions = [a for a in rows if a.doc_type == DocType.OPINION]
    assert len(opinions) == 4 and all(a.trust_tier == TrustTier.COMMENTARY for a in opinions)
    assert all(a.content is None for a in rows)


# --------------------------------------------------------------------- ingest_all


def _registry_source(session, slug: str, **kw) -> Source:
    src = Source(
        slug=slug,
        name=slug,
        kind=kw.pop("kind", SourceKind.RSS),
        url=f"https://{slug}.example/feed",
        homepage=f"https://{slug}.example",
        vertical=kw.pop("vertical", Vertical.TECHNOLOGY),
        trust_tier=TrustTier.INDEPENDENT,
        default_doc_type=DocType.NEWS,
        license_note="Headline + publisher summary + link; full text not stored.",
        **kw,
    )
    session.add(src)
    session.flush()
    return src


@pytest.fixture()
def sources(session):
    return {
        "good": _registry_source(session, "good"),
        "also-good": _registry_source(session, "also-good", vertical=Vertical.EDUCATION),
        "down": _registry_source(session, "down"),
        "buggy": _registry_source(session, "buggy"),
        "demo": _registry_source(session, "demo", kind=SourceKind.DEMO),
        "unapproved": _registry_source(session, "unapproved", approved=False),
        "inactive": _registry_source(session, "inactive", active=False),
    }


def _fake_fetch(calls: list[str]):
    sample = (DATA / "the74_rss.xml").read_bytes()

    def fetch(source, settings=None, **kw):
        calls.append(source.slug)
        if source.slug == "down":
            raise FetchError("HTTP 503 from down.example")
        if source.slug == "buggy":
            raise KeyError("unexpected parser bug")
        if source.slug == "also-good":
            return [
                _item("Calder adopts AI literacy requirement", url="https://also-good.example/1", content=BODY),
                _item("Broken item", url="not-a-url"),
            ]
        return parse_rss(sample, source)

    return fetch


def test_ingest_all_contains_failures_and_records_status(session, settings, sources, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "fetch_source", _fake_fetch(calls))
    report = ingest_all(session, settings)

    assert isinstance(report, IngestReport)
    assert sorted(calls) == ["also-good", "buggy", "down", "good"]  # demo/unapproved/inactive never fetched
    assert report.sources_ok == 2 and report.sources_failed == 2
    assert report.new_articles == 6 and report.duplicates == 0
    assert any("down: HTTP 503" in e for e in report.errors)
    assert any("buggy: KeyError" in e for e in report.errors)
    assert any("not-a-url" in e for e in report.errors)  # bad item recorded, rest of feed stored

    down, good, also = sources["down"], sources["good"], sources["also-good"]
    assert down.last_status.startswith("error: HTTP 503") and down.last_fetched_at is not None
    assert good.last_status.startswith("ok: 5 items, 5 new") and good.last_fetched_at is not None
    assert "1 item errors" in also.last_status
    assert sources["demo"].last_fetched_at is None
    assert report.per_source["good"]["new"] == 5

    again = ingest_all(session, settings)
    assert again.new_articles == 0 and again.already_seen == 6


def test_ingest_all_source_filter(session, settings, sources, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "fetch_source", _fake_fetch(calls))
    report = ingest_all(session, settings, source_slugs=["good", "unapproved", "nope"])
    assert calls == ["good"]
    assert report.sources_ok == 1 and report.new_articles == 5
    assert sum("not an approved, active, fetchable source" in e for e in report.errors) == 2
