"""Canonical URLs, exact and near-duplicate detection."""

from __future__ import annotations

from datetime import timedelta

from newsren.ingest.dedup import find_duplicate, root_article
from newsren.models import utcnow
from newsren.textutil import canonical_url, content_hash, hamming, simhash_hex

BODY = (
    "Corvane Semiconductor's Calder Valley Fab 3 exited the fiscal third quarter at 11,500 wafer starts per "
    "month, short of the interim target of 15,000 the company projected when it announced the expansion. "
    "Corvane attributed the shortfall to slower qualification of new tools and said yields were in line with "
    "plan. The company reiterated its projection of 40,000 wafer starts per month by late 2027. Customers have "
    "not commented on whether deliveries were affected, and analysts said reported output is the number to watch "
    "each quarter as the ramp continues. The expansion depends on extreme-ultraviolet lithography tools from Orrin "
    "Lithography, whose lead times have stretched to about 18 months, and on high-purity process gases. Analysts "
    "described scenarios rather than forecasts: if tools arrive on schedule the late-2027 target is achievable, "
    "while a two-quarter slip would leave capacity closer to 28,000 wafers per month."
)
TITLE = "Corvane's Calder Valley line reached 11,500 wafers a month"


def test_canonical_url_strips_tracking_and_normalizes():
    a = canonical_url("http://www.Example.com/news/story/?utm_source=rss&id=7&at_medium=RSS#comments")
    b = canonical_url("https://example.com/news/story?id=7")
    assert a == b == "https://example.com/news/story?id=7"


def test_exact_url_returns_same_article(factory, session):
    art = factory.article(TITLE, BODY, url="https://fabline.example/news/corvane-output?utm_source=x")
    hit = find_duplicate(session, canonical_url("https://fabline.example/news/corvane-output"), None, None)
    assert hit is not None and hit.id == art.id


def test_exact_content_hash_matches_original(factory, session):
    original = factory.article(TITLE, BODY)
    hit = find_duplicate(session, "https://other.example/copy", content_hash(TITLE, BODY), None)
    assert hit is not None and hit.id == original.id


def test_near_duplicate_simhash_within_window(factory, session):
    original = factory.article(TITLE, BODY)
    copy_text = f"{TITLE} {BODY} (FabLine Weekly)"
    copy_hash = simhash_hex(copy_text)
    assert hamming(copy_hash, original.simhash) <= 3
    hit = find_duplicate(session, None, None, copy_hash, around=original.published_at)
    assert hit is not None and hit.id == original.id
    # Outside the 14-day window the same fingerprint is not treated as a copy.
    later = original.published_at + timedelta(days=20)
    assert find_duplicate(session, None, None, copy_hash, around=later) is None


def test_retitled_copy_found_by_verified_fallback(factory, session):
    original = factory.article(TITLE, BODY)
    copy_title = "Corvane's Calder Valley chip line reached 11,500 wafers a month, behind its interim target"
    copy_text = (
        f"{copy_title} {BODY} This article first appeared in FabLine Weekly and is republished under a "
        "content-sharing agreement."
    )
    copy_hash = simhash_hex(copy_text)
    assert 3 < hamming(copy_hash, original.simhash) <= 12  # too far for the strict SimHash rule
    hit = find_duplicate(session, None, None, copy_hash, around=original.published_at, text=copy_text)
    assert hit is not None and hit.id == original.id
    # Without the text to verify, only the strict <= 3 bit rule applies.
    assert find_duplicate(session, None, None, copy_hash, around=original.published_at) is None


def test_same_text_outside_window_is_not_a_duplicate(factory, session):
    original = factory.article(TITLE, BODY)
    next_year = original.published_at + timedelta(days=365)
    assert find_duplicate(session, None, content_hash(TITLE, BODY), None, around=next_year) is None


def test_unrelated_article_is_not_a_duplicate(factory, session):
    factory.article(TITLE, BODY)
    other = (
        "Quillfield raised $45 million in a Series B led by Harrow Ventures to expand AI feedback tools for "
        "teachers, with participation from Larchmont Capital and Saltmarsh Partners."
    )
    assert find_duplicate(session, None, content_hash(other), simhash_hex(other), text=other) is None


def test_duplicates_resolve_to_root(factory, session):
    original = factory.article(TITLE, BODY)
    copy = factory.article(TITLE + " (wire)", BODY)
    copy.duplicate_of_id = original.id
    session.flush()
    assert root_article(session, copy).id == original.id
    hit = find_duplicate(session, None, copy.content_hash, None)
    assert hit is not None and hit.id == original.id


def test_window_uses_retrieval_time_when_publication_unknown(factory, session):
    original = factory.article(TITLE, BODY)
    original.published_at = None
    original.retrieved_at = utcnow() - timedelta(days=2)
    session.flush()
    hit = find_duplicate(session, None, None, simhash_hex(f"{TITLE} {BODY} (FabLine Weekly)"))
    assert hit is not None and hit.id == original.id
