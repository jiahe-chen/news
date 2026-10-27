"""FastAPI application factory."""

from __future__ import annotations

import importlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from newsren import __version__
from newsren.config import get_settings
from newsren.db import init_db, new_session
from newsren.models import User
from newsren.web.deps import LoginRequired
from newsren.web.templating import render

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Route modules, each exposing ``router``. Missing modules are skipped with a
# warning so the app still starts while a section is under construction.
ROUTE_MODULES = [
    "newsren.web.routes.auth",
    "newsren.web.routes.public",
    "newsren.web.routes.tools",
    "newsren.web.routes.account",
    "newsren.web.routes.finance",
    "newsren.web.routes.editor",
    "newsren.web.routes.analytics",
    "newsren.web.routes.api",
]


class CurrentUserMiddleware(BaseHTTPMiddleware):
    """Expose the signed-in reader as ``request.state.user`` for templates."""

    async def dispatch(self, request: Request, call_next):
        request.state.user = None
        user_id = request.session.get("user_id") if "session" in request.scope else None
        if user_id:
            session = new_session()
            try:
                request.state.user = session.get(User, user_id)
            finally:
                session.close()
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    scheduler = None
    if get_settings().scheduler_enabled:
        from newsren.scheduler import start_scheduler

        scheduler = start_scheduler()
    try:
        yield
    finally:
        if scheduler is not None:
            scheduler.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="The Newsren",
        version=__version__,
        description="Agentic news intelligence for technology, education, industry, and finance.",
        lifespan=lifespan,
    )
    app.add_middleware(CurrentUserMiddleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        session_cookie="newsren_session",
        same_site="lax",
        https_only=settings.base_url.startswith("https://"),
        max_age=60 * 60 * 24 * 30,
    )
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    for module_name in ROUTE_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                log.warning("Route module %s not found; skipping", module_name)
                continue
            raise
        app.include_router(module.router)

    @app.exception_handler(LoginRequired)
    async def _login_required(request: Request, exc: LoginRequired):
        return RedirectResponse(f"/login?next={quote(exc.next_url)}", status_code=303)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        if request.url.path.startswith("/api/") or "application/json" in request.headers.get("accept", ""):
            from fastapi.responses import JSONResponse

            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        return render(
            request,
            "error.html",
            {"status_code": exc.status_code, "detail": exc.detail},
            status_code=exc.status_code,
        )

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
