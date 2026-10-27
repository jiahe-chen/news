"""Sign in, create account, sign out, and the "Explain for" audience switch."""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from newsren.models import Audience
from newsren.users.auth import authenticate, create_user
from newsren.users.service import set_preferences
from newsren.web.deps import AUDIENCE_COOKIE, CSRF, DB, CurrentUser
from newsren.web.templating import flash, render

router = APIRouter()


def safe_next(target: str | None, default: str = "/") -> str:
    """Only allow same-site relative redirects (prevents open redirects)."""
    if not target:
        return default
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or not target.startswith("/") or target.startswith("//"):
        return default
    return target


@router.get("/login")
def login_page(request: Request, user: CurrentUser, next: str = "/"):
    if user is not None:
        return RedirectResponse(safe_next(next), status_code=303)
    return render(request, "auth/login.html", {"next": safe_next(next), "error": None})


@router.post("/login", dependencies=[CSRF])
def login(
    request: Request,
    db: DB,
    email: Annotated[str, Form()],
    password: Annotated[str, Form()],
    next: Annotated[str, Form()] = "/",
):
    user = authenticate(db, email, password)
    if user is None:
        return render(
            request,
            "auth/login.html",
            {"next": safe_next(next), "error": "That email and password don't match an account.", "email": email},
            status_code=400,
        )
    request.session.clear()
    request.session["user_id"] = user.id
    flash(request, f"Welcome back, {user.display_name}.", "success")
    return RedirectResponse(safe_next(next, "/me"), status_code=303)


@router.get("/register")
def register_page(request: Request, user: CurrentUser):
    if user is not None:
        return RedirectResponse("/me", status_code=303)
    return render(request, "auth/register.html", {"error": None, "form": {}})


@router.post("/register", dependencies=[CSRF])
def register(
    request: Request,
    db: DB,
    email: Annotated[str, Form()],
    password: Annotated[str, Form()],
    display_name: Annotated[str, Form()] = "",
    audience: Annotated[str, Form()] = Audience.PROFESSIONAL.value,
):
    form = {"email": email, "display_name": display_name, "audience": audience}
    try:
        user = create_user(db, email, password, display_name.strip() or email.split("@")[0])
        if audience in {a.value for a in Audience}:
            set_preferences(db, user, audience=audience)
    except ValueError as exc:
        return render(request, "auth/register.html", {"error": str(exc), "form": form}, status_code=400)
    db.flush()
    request.session.clear()
    request.session["user_id"] = user.id
    flash(request, "Your account is ready. Tell The Newsren what to follow.", "success")
    return RedirectResponse("/me", status_code=303)


@router.post("/logout", dependencies=[CSRF])
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)


@router.post("/audience", dependencies=[CSRF])
def set_audience(
    request: Request,
    db: DB,
    user: CurrentUser,
    audience: Annotated[str, Form()],
    next: Annotated[str, Form()] = "/",
):
    """Persist the reader's "Explain for" choice (cookie; profile too when signed in)."""
    target = safe_next(next)
    response = RedirectResponse(target, status_code=303)
    if audience in {a.value for a in Audience}:
        response.set_cookie(AUDIENCE_COOKIE, audience, max_age=60 * 60 * 24 * 365, samesite="lax")
        if user is not None:
            set_preferences(db, user, audience=audience)
    return response
