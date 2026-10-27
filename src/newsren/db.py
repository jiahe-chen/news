"""Engine/session management and full-text search index.

The archive search used by agents (``search_articles``) is SQLite FTS5 over
article title, summary, and content, kept in sync with triggers.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from newsren.config import get_settings
from newsren.models import Base

_engine: Engine | None = None
_engine_url: str | None = None
_SessionLocal: sessionmaker[Session] | None = None

FTS_DDL = [
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
        title, summary, content,
        content='articles', content_rowid='id',
        tokenize='porter unicode61'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS articles_ai AFTER INSERT ON articles BEGIN
        INSERT INTO articles_fts(rowid, title, summary, content)
        VALUES (new.id, new.title, coalesce(new.summary, ''), coalesce(new.content, ''));
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS articles_ad AFTER DELETE ON articles BEGIN
        INSERT INTO articles_fts(articles_fts, rowid, title, summary, content)
        VALUES ('delete', old.id, old.title, coalesce(old.summary, ''), coalesce(old.content, ''));
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS articles_au AFTER UPDATE OF title, summary, content ON articles BEGIN
        INSERT INTO articles_fts(articles_fts, rowid, title, summary, content)
        VALUES ('delete', old.id, old.title, coalesce(old.summary, ''), coalesce(old.content, ''));
        INSERT INTO articles_fts(rowid, title, summary, content)
        VALUES (new.id, new.title, coalesce(new.summary, ''), coalesce(new.content, ''));
    END
    """,
]


def _make_engine(url: str) -> Engine:
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {"future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        if ":memory:" in url:
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
            # Let SQLAlchemy own transaction boundaries so SAVEPOINTs work
            # (pysqlite's implicit BEGIN handling breaks nested transactions).
            dbapi_conn.isolation_level = None
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            if ":memory:" not in url:
                cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

        @event.listens_for(engine, "begin")
        def _begin(conn):  # pragma: no cover - trivial
            conn.exec_driver_sql("BEGIN")

    return engine


def get_engine() -> Engine:
    global _engine, _engine_url, _SessionLocal
    url = get_settings().database_url
    if _engine is None or _engine_url != url:
        if _engine is not None:
            _engine.dispose()
        _engine = _make_engine(url)
        _engine_url = url
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def reset_engine() -> None:
    """Dispose the cached engine (tests switch databases between cases)."""
    global _engine, _engine_url, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _engine_url = None
    _SessionLocal = None


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)
    if engine.dialect.name == "sqlite":
        with engine.begin() as conn:
            for ddl in FTS_DDL:
                conn.execute(text(ddl))


def new_session() -> Session:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for jobs and CLI commands."""
    session = new_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


_FTS_TOKEN = re.compile(r"[\w][\w\-\.']*", re.UNICODE)


def fts_query(user_query: str, mode: str = "or") -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Each token is quoted (so punctuation and FTS operators in user input are
    inert); tokens are combined with OR by default for recall, ranked by bm25.
    """
    tokens = [t.strip(".'-") for t in _FTS_TOKEN.findall(user_query or "")]
    tokens = [t for t in tokens if len(t) > 1][:24]
    if not tokens:
        return ""
    joiner = " OR " if mode == "or" else " AND "
    return joiner.join('"' + t.replace('"', '""') + '"' for t in tokens)


def search_article_ids(session: Session, query: str, limit: int = 20, mode: str = "or") -> list[tuple[int, float]]:
    """Return (article_id, bm25 score) pairs, best first. Lower bm25 = better match."""
    match = fts_query(query, mode)
    if not match:
        return []
    rows = session.execute(
        text(
            "SELECT rowid, bm25(articles_fts, 10.0, 4.0, 1.0) AS score FROM articles_fts "
            "WHERE articles_fts MATCH :q ORDER BY score LIMIT :limit"
        ),
        {"q": match, "limit": limit},
    ).all()
    return [(int(r[0]), float(r[1])) for r in rows]
