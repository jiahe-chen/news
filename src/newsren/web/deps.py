"""Request-scoped dependencies: DB session, current reader, audience, CSRF."""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Form, HTTPException, Request
from sqlalchemy.orm import Session

from newsren.db import new_session
from newsren.models import Audience, User

AUDIENCE_COOKIE = "newsren_audience"
SESSION_KEY_COOKIE = "newsren_sid"


def get_db() -> Iterator[Session]:
    """One session per request; committed when the handler returns without error."""
    session = new_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


DB = Annotated[Session, Depends(get_db)]


def get_current_user(request: Request, db: DB) -> User | None:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    user = db.get(User, user_id)
    if user is None:
        request.session.pop("user_id", None)
    return user


CurrentUser = Annotated[User | None, Depends(get_current_user)]


class LoginRequired(Exception):
    """Raised for pages that need a signed-in reader; handled by redirecting to /login."""

    def __init__(self, next_url: str = "/"):
        self.next_url = next_url


def _return_url(request: Request) -> str:
    """Where to go after signing in: the same page for GETs, the front page after a POST."""
    if request.method != "GET":
        return "/"
    query = request.url.query
    return request.url.path + (f"?{query}" if query else "")


def require_user(request: Request, user: CurrentUser) -> User:
    if user is None:
        raise LoginRequired(next_url=_return_url(request))
    return user


def require_editor(request: Request, user: CurrentUser) -> User:
    if user is None:
        raise LoginRequired(next_url=_return_url(request))
    if not user.is_editor:
        raise HTTPException(status_code=403, detail="Editors only")
    return user


RequiredUser = Annotated[User, Depends(require_user)]
EditorUser = Annotated[User, Depends(require_editor)]


def get_audience(request: Request, user: CurrentUser) -> str:
    """The reader's "Explain for" choice: query param > cookie > profile > professional."""
    for candidate in (
        request.query_params.get("audience"),
        request.cookies.get(AUDIENCE_COOKIE),
        user.audience if user else None,
    ):
        if candidate in {a.value for a in Audience}:
            return candidate
    return Audience.PROFESSIONAL.value


ReaderAudience = Annotated[str, Depends(get_audience)]


def session_key(request: Request) -> str:
    """Anonymous, per-browser key used to group a reader's questions."""
    key = request.session.get("sid")
    if not key:
        key = secrets.token_hex(16)
        request.session["sid"] = key
    return key


def csrf_token(request: Request) -> str:
    token = request.session.get("csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf"] = token
    return token


async def verify_csrf(
    request: Request, csrf_token_field: Annotated[str | None, Form(alias="csrf_token")] = None
) -> None:
    """Reject state-changing requests without the session's CSRF token (form field or X-CSRF-Token header)."""
    expected = request.session.get("csrf")
    supplied = request.headers.get("X-CSRF-Token") or csrf_token_field
    if not expected or not supplied or not secrets.compare_digest(expected, supplied):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token")


CSRF = Depends(verify_csrf)
