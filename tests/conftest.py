"""Shared test fixtures: isolated SQLite database per test, offline LLM, factories."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from newsren import db
from newsren.config import Settings, set_settings
from newsren.llm import OfflineProvider, set_provider
from newsren.models import Article, DocType, Source, TrustTier, User, Vertical, utcnow
from newsren.textutil import canonical_url, content_hash, simhash_hex


@pytest.fixture()
def settings(tmp_path) -> Iterator[Settings]:
    s = Settings(
        data_dir=tmp_path,
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        llm_mode="offline",
        secret_key="test-secret",
        require_human_approval=True,
    )
    set_settings(s)
    set_provider(OfflineProvider())
    db.reset_engine()
    db.init_db()
    yield s
    db.reset_engine()
    set_provider(None)
    set_settings(None)


@pytest.fixture()
def session(settings) -> Iterator[Session]:
    s = db.new_session()
    try:
        yield s
    finally:
        s.rollback()
        s.close()


class Factory:
    """Small builders for sources/articles/users used across test modules."""

    def __init__(self, session: Session):
        self.session = session
        self._n = 0

    def source(
        self,
        vertical: str = Vertical.TECHNOLOGY,
        trust_tier: str = TrustTier.INDEPENDENT,
        doc_type: str = DocType.NEWS,
        name: str | None = None,
        approved: bool = True,
    ) -> Source:
        self._n += 1
        src = Source(
            slug=f"src-{self._n}",
            name=name or f"Source {self._n}",
            url=f"https://source{self._n}.example/feed",
            homepage=f"https://source{self._n}.example",
            vertical=vertical,
            trust_tier=trust_tier,
            default_doc_type=doc_type,
            approved=approved,
        )
        self.session.add(src)
        self.session.flush()
        return src

    def article(
        self,
        title: str,
        body: str = "",
        *,
        source: Source | None = None,
        vertical: str | None = None,
        doc_type: str | None = None,
        trust_tier: str | None = None,
        published_at: datetime | None = None,
        url: str | None = None,
        summary: str | None = None,
        extra: dict | None = None,
    ) -> Article:
        self._n += 1
        source = source or self.source(vertical=vertical or Vertical.TECHNOLOGY)
        url = url or f"https://{source.slug}.example/articles/{self._n}"
        art = Article(
            source=source,
            url=url,
            canonical_url=canonical_url(url),
            title=title,
            summary=summary if summary is not None else body[:300],
            content=body,
            published_at=published_at or (utcnow() - timedelta(hours=self._n)),
            vertical=vertical or source.vertical,
            doc_type=doc_type or source.default_doc_type,
            trust_tier=trust_tier or source.trust_tier,
            content_hash=content_hash(title, body),
            simhash=simhash_hex(f"{title} {body}"),
            extra=extra or {},
        )
        self.session.add(art)
        self.session.flush()
        return art

    def user(self, email: str | None = None, role: str = "reader", **kwargs) -> User:
        self._n += 1
        user = User(
            email=email or f"user{self._n}@example.com",
            display_name=kwargs.pop("display_name", f"User {self._n}"),
            password_hash=kwargs.pop("password_hash", "x"),
            role=role,
            **kwargs,
        )
        self.session.add(user)
        self.session.flush()
        return user


@pytest.fixture()
def factory(session) -> Factory:
    return Factory(session)
