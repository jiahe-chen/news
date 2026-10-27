"""Jinja2 environment: filters and globals shared by every template."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from newsren.models import VERTICAL_LABELS, Audience, Vertical, utcnow

TEMPLATE_DIR = Path(__file__).parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

VERTICALS = [
    {"slug": v.value, "label": VERTICAL_LABELS[v]}
    for v in (Vertical.TECHNOLOGY, Vertical.EDUCATION, Vertical.INDUSTRY, Vertical.FINANCE)
]
AUDIENCES = [
    {"slug": Audience.STUDENT.value, "label": "Student"},
    {"slug": Audience.EDUCATOR.value, "label": "Educator"},
    {"slug": Audience.PROFESSIONAL.value, "label": "Professional"},
]

# Human-readable labels for evidence/claim vocabularies used as badges.
LABELS: dict[str, str] = {
    "research_preprint": "Preprint · not peer reviewed",
    "research_peer_reviewed": "Peer-reviewed",
    "press_release": "Press release",
    "filing": "Filing",
    "regulatory": "Regulatory",
    "policy": "Policy",
    "opinion": "Opinion",
    "advisory": "Advisory",
    "education_report": "Education report",
    "earnings_call": "Earnings call",
    "news": "News",
    "blog": "Blog",
    "primary": "Primary source",
    "independent": "Independent",
    "vendor": "Vendor claim",
    "commentary": "Commentary",
    "supported": "Supported",
    "corroborated": "Corroborated",
    "contested": "Contested",
    "unverified": "Unverified",
    "speculative": "Speculative",
    "scenario": "Scenario",
    "hypothesis": "Hypothesis",
    "projection": "Projection",
    "backtest": "Backtest",
    "simulated": "Simulated",
    "live": "Live",
    "fact": "Fact",
    "interpretation": "Interpretation",
    "forecast": "Forecast",
    "performance": "Performance claim",
    "announcement": "Announcement",
    "pilot": "Pilot",
    "live_deployment": "Live deployment",
    "regulatory_finding": "Regulatory finding",
}


def label(value: Any) -> str:
    if value is None:
        return ""
    key = str(value)
    return LABELS.get(key, key.replace("_", " ").capitalize())


def badge(value: Any, extra_class: str = "") -> Markup:
    """<span class="badge badge-<value>">Label</span>"""
    if value in (None, ""):
        return Markup("")
    key = str(value)
    cls = f"badge badge-{escape(key)}" + (f" {escape(extra_class)}" if extra_class else "")
    return Markup(f'<span class="{cls}">{escape(label(key))}</span>')


_CITE_MARK = re.compile(r"\s*\[c\d+\]")


def strip_cites(value: Any) -> str:
    """Drop inline [cN] citation markers (for teasers where footnotes aren't shown)."""
    return _CITE_MARK.sub("", str(value or ""))


def fmt_dt(value: datetime | str | None, fmt: str = "%b %d, %Y") -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return value
    return value.strftime(fmt).replace(" 0", " ")


def fmt_dt_time(value: datetime | str | None) -> str:
    return fmt_dt(value, "%b %d, %Y %H:%M UTC")


def timeago(value: datetime | str | None) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return value
    seconds = (utcnow() - value).total_seconds()
    if seconds < 0:
        return fmt_dt(value)
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} ago"
    return "just now"


def money(value: float | None) -> str:
    if value is None:
        return "Undisclosed"
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= size:
            return f"${value / size:,.1f}{unit}".replace(".0" + unit, unit)
    return f"${value:,.0f}"


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%" if abs(value) <= 1.5 else f"{value:.{digits}f}%"


templates.env.filters.update(
    {
        "label": label,
        "badge": badge,
        "fmt_dt": fmt_dt,
        "fmt_dt_time": fmt_dt_time,
        "timeago": timeago,
        "money": money,
        "pct": pct,
    }
)
templates.env.globals.update(
    {
        "VERTICALS": VERTICALS,
        "AUDIENCES": AUDIENCES,
        "VERTICAL_LABELS": {k.value: v for k, v in VERTICAL_LABELS.items()},
    }
)


def render(
    request: Request, name: str, context: dict[str, Any] | None = None, status_code: int = 200, **kwargs: Any
):
    """TemplateResponse with the shared per-request context (user, audience, csrf, flash)."""
    from newsren.web.deps import AUDIENCE_COOKIE, csrf_token

    ctx: dict[str, Any] = dict(context or {})
    ctx.setdefault("current_user", getattr(request.state, "user", None))
    audience = request.query_params.get("audience") or request.cookies.get(AUDIENCE_COOKIE)
    user = ctx["current_user"]
    if audience not in {a["slug"] for a in AUDIENCES}:
        audience = user.audience if user is not None else Audience.PROFESSIONAL.value
    ctx.setdefault("audience", audience)
    ctx.setdefault("csrf_token", csrf_token(request))
    if "flash" not in ctx:
        ctx["flash"] = request.session.pop("flash", None)
    ctx.setdefault("active_vertical", None)
    ctx.setdefault("now_label", utcnow().strftime("%A, %B %d, %Y").replace(" 0", " "))
    return templates.TemplateResponse(request, name, ctx, status_code=status_code, **kwargs)


def flash(request: Request, message: str, kind: str = "info") -> None:
    request.session["flash"] = {"message": message, "kind": kind}
