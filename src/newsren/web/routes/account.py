"""My Newsren (plan §2.1) and Follow This Story (§2.6): the signed-in reader's own pages.

My Newsren is the reader's personal news research agent: the reader chooses
topics, questions, organizations, and sectors, an "Explain for" audience and
a technical depth, and the agent prepares briefings that explain what changed
and why it matters. Follow This Story turns a reader's question into
monitoring criteria and alerts only when a development meets them.

Routes (all need a signed-in reader; every POST checks the CSRF token and
answers with a 303 redirect, except htmx requests, which get the updated
fragment):

* ``GET  /me`` — dashboard: latest briefing, unread alerts, follows,
  interests, preferences, and how monitoring works.
* ``GET  /me/alerts`` — every alert (``?show=unread`` for unread only).
* ``POST /me/preferences`` — audience, depth, email opt-in, frequency.
* ``POST /me/interests`` and ``POST /me/interests/{id}/delete``.
* ``GET  /me/briefings``, ``GET /me/briefings/{id}`` (owner only; sets
  ``opened_at``), ``POST /me/briefings/generate`` ("Brief me now").
* ``POST /follows`` — follow a question, story, company, or sector (the form
  contract shared with the brief tools and story pages).
* ``POST /follows/{id}/delete`` — stop following (alerts are kept).
* ``POST /alerts/{id}/feedback`` (useful | unwanted) and
  ``POST /alerts/{id}/read``.

Readers only ever see their own briefings, follows, and alerts: anything that
belongs to someone else is a 404, editors included.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsren.agents.briefing import build_briefing, deliver_briefing
from newsren.editorial.guardrails import FINANCE_DISCLAIMER
from newsren.models import (
    Alert,
    Audience,
    Brief,
    Briefing,
    BriefStatus,
    Depth,
    EventType,
    Follow,
    Story,
    User,
    Vertical,
    utcnow,
)
from newsren.schemas import BriefingOut, FollowCriteria
from newsren.textutil import truncate
from newsren.users.service import (
    ALERT_FEEDBACK,
    BRIEFING_FREQUENCIES,
    FOLLOW_TARGETS,
    MAX_INTEREST_LENGTH,
    MAX_QUESTION_LENGTH,
    add_interest,
    create_follow,
    deactivate_follow,
    derive_criteria,
    describe_criteria,
    interests_by_kind,
    mark_alert,
    remove_interest,
    set_preferences,
    user_alerts,
    user_follows,
)
from newsren.web.deps import AUDIENCE_COOKIE, CSRF, DB, CurrentUser, LoginRequired, RequiredUser, csrf_token
from newsren.web.routes.auth import safe_next
from newsren.web.templating import flash, render, templates

log = logging.getLogger(__name__)

router = APIRouter()

MAX_DASHBOARD_ALERTS = 20
MAX_ALERTS_PAGE = 200
MAX_BRIEFINGS_LIST = 100
MAX_BRIEFING_PREVIEW_ITEMS = 6
MAX_CUSTOM_CRITERION = 500

# --------------------------------------------------------------------------
# Vocabulary shown in forms (values come from the service vocabularies)
# --------------------------------------------------------------------------

AUDIENCE_CHOICES: list[tuple[str, str, str]] = [
    (
        Audience.STUDENT.value,
        "Student",
        "Plain-language explanations, a glossary, and quick checks of understanding.",
    ),
    (Audience.EDUCATOR.value, "Educator", "Classroom-ready framing, discussion questions, and activities."),
    (Audience.PROFESSIONAL.value, "Professional", "Implications for industries, occupations, and decisions."),
]
DEPTH_CHOICES: list[tuple[str, str, str]] = [
    (Depth.INTRO.value, "Introductory", "One plain sentence on what changed and one on why it matters."),
    (Depth.STANDARD.value, "Standard", "What changed and why it matters, with evidence notes."),
    (
        Depth.TECHNICAL.value,
        "Technical",
        "Adds the key technical claims with their labels (backtest, projection, contested).",
    ),
]
FREQUENCY_CHOICES: list[tuple[str, str]] = [
    ("when_meaningful", "Only when it matters"),
    ("daily", "At most daily"),
    ("weekly", "Weekly"),
]
INTEREST_KIND_CHOICES: list[tuple[str, str, str]] = [
    ("topic", "Topics", "e.g. “AI models for learning mathematics”"),
    ("question", "Questions", "e.g. “Do AI tutors improve transfer, not just speed?”"),
    ("organization", "Organizations", "e.g. a company, regulator, or university"),
    ("sector", "Sectors", "e.g. “semiconductors”, or a whole section such as Finance"),
]
INTEREST_KIND_SINGULAR = {
    "topic": "Topic",
    "question": "Question",
    "organization": "Organization",
    "sector": "Sector",
}
FOLLOW_TARGET_LABELS = {
    "topic": "Topic or question",
    "story": "Story",
    "company": "Company or organization",
    "sector": "Sector",
}
FOLLOW_EVENT_CHOICES: list[tuple[str, str]] = [
    (EventType.INDEPENDENT_EVALUATION.value, "Independent evaluation"),
    (EventType.BENCHMARK.value, "Benchmark result"),
    (EventType.RESEARCH_PUBLICATION.value, "Research publication"),
    (EventType.PRODUCT_LAUNCH.value, "Product launch"),
    (EventType.ADOPTION.value, "Adoption"),
    (EventType.REGULATION.value, "Regulation"),
    (EventType.POLICY.value, "Policy decision"),
    (EventType.FILING.value, "Filing"),
    (EventType.EARNINGS.value, "Earnings"),
    (EventType.FUNDING.value, "Funding round"),
    (EventType.ENFORCEMENT.value, "Enforcement action"),
    (EventType.SECURITY_ADVISORY.value, "Security advisory"),
    (EventType.CORRECTION.value, "Correction"),
]
_EVENT_VALUES = {e.value for e in EventType}
_VERTICAL_VALUES = {v.value for v in Vertical}
_TRUE = {"1", "true", "on", "yes"}

FEEDBACK_MESSAGES = {
    "useful": "Marked useful. Thanks — this tunes what counts as a useful alert.",
    "unwanted": "Marked not useful. Thanks — unwanted alerts tell us where the criteria are too loose.",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def is_htmx(request: Request) -> bool:
    """True for an htmx fragment request (not a history restore, which needs the full page)."""
    return (
        request.headers.get("HX-Request") == "true" and request.headers.get("HX-History-Restore-Request") != "true"
    )


def require_user_for_post(request: Request, user: CurrentUser) -> User:
    """Signed-in reader for form posts; signed-out readers go to sign-in and then back to My Newsren.

    (``RequiredUser`` would send them back to the POST-only URL after signing in.)
    """
    if user is None:
        raise LoginRequired(next_url="/me")
    return user


PostUser = Annotated[User, Depends(require_user_for_post)]


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUE


def _page(request: Request, name: str, context: dict[str, Any], status_code: int = 200) -> Response:
    ctx = dict(context)
    ctx.setdefault("FINANCE_DISCLAIMER", FINANCE_DISCLAIMER)
    ctx.setdefault("active_vertical", "me")
    _shared_context(ctx)
    response = render(request, name, ctx, status_code=status_code)
    response.headers["Vary"] = "HX-Request"
    response.headers["Cache-Control"] = "private, no-store"
    return response


def _partial(request: Request, name: str, context: dict[str, Any], status_code: int = 200) -> Response:
    """Render an htmx fragment without consuming the session's flash message."""
    ctx = dict(context)
    user = getattr(request.state, "user", None)
    ctx.setdefault("current_user", user)
    ctx.setdefault("audience", user.audience if user is not None else Audience.PROFESSIONAL.value)
    ctx.setdefault("csrf_token", csrf_token(request))
    ctx.setdefault("flash", None)
    ctx.setdefault("FINANCE_DISCLAIMER", FINANCE_DISCLAIMER)
    _shared_context(ctx)
    response = templates.TemplateResponse(request, name, ctx, status_code=status_code)
    response.headers["Vary"] = "HX-Request"
    response.headers["Cache-Control"] = "private, no-store"
    return response


def _shared_context(ctx: dict[str, Any]) -> None:
    ctx.setdefault("FOLLOW_TARGET_LABELS", FOLLOW_TARGET_LABELS)
    ctx.setdefault("INTEREST_KIND_SINGULAR", INTEREST_KIND_SINGULAR)


def _redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


def _owned_alert(db: Session, alert_id: int, user: User) -> Alert:
    alert = db.get(Alert, alert_id)
    if alert is None or alert.user_id != user.id:
        raise HTTPException(status_code=404, detail="We couldn't find that alert.")
    return alert


def _owned_briefing(db: Session, briefing_id: int, user: User) -> Briefing:
    briefing = db.get(Briefing, briefing_id)
    if briefing is None or briefing.user_id != user.id:
        raise HTTPException(status_code=404, detail="We couldn't find that briefing.")
    return briefing


def _briefing_content(briefing: Briefing) -> BriefingOut:
    """The stored ``BriefingOut``; a damaged record still renders its title instead of failing."""
    try:
        return BriefingOut.model_validate(briefing.content or {})
    except ValidationError:
        log.warning("Briefing %s has malformed content; showing its title only", briefing.id)
        return BriefingOut(title=briefing.title, intro="")


def _briefs_by_id(db: Session, ids: Iterable[int | None]) -> dict[int, Brief]:
    wanted = {int(i) for i in ids if i is not None}
    if not wanted:
        return {}
    return {b.id: b for b in db.scalars(select(Brief).where(Brief.id.in_(wanted)))}


def _stories_by_id(db: Session, ids: Iterable[int | None]) -> dict[int, Story]:
    wanted = {int(i) for i in ids if i is not None}
    if not wanted:
        return {}
    return {s.id: s for s in db.scalars(select(Story).where(Story.id.in_(wanted)))}


def briefing_view(db: Session, briefing: Briefing, user: User) -> dict[str, Any]:
    """Everything a template needs to show a briefing: content, linked briefs/stories, finance and demo flags."""
    content = _briefing_content(briefing)
    items = [item for section in content.sections for item in section.items]
    briefs = _briefs_by_id(db, (item.brief_id for item in items))
    stories = _stories_by_id(db, (item.story_id for item in items))
    sections = []
    verticals: set[str] = set()
    for section in content.sections:
        rows = []
        for item in section.items:
            brief = briefs.get(item.brief_id) if item.brief_id is not None else None
            story = stories.get(item.story_id) if item.story_id is not None else None
            vertical = (brief.vertical if brief else None) or (story.vertical if story else None)
            if vertical:
                verticals.add(str(vertical))
            visible = brief is not None and (brief.status == BriefStatus.APPROVED or user.is_editor)
            rows.append(
                {
                    "item": item,
                    "brief": brief,
                    "brief_url": f"/briefs/{brief.id}" if visible and brief is not None else None,
                    "brief_unpublished": brief is not None and brief.status != BriefStatus.APPROVED,
                    "story": story,
                    "vertical": vertical,
                    "is_demo": bool((brief and brief.is_demo) or (story and story.is_demo)),
                }
            )
        sections.append({"heading": section.heading, "rows": rows})
    finance = Vertical.FINANCE.value in verticals or FINANCE_DISCLAIMER in (content.closing_note or "")
    stored = briefing.content or {}
    return {
        "briefing": briefing,
        "content": content,
        "sections": sections,
        "item_total": len(items),
        "citations": list(briefing.citations or []),
        "is_finance": finance,
        "is_demo": any(row["is_demo"] for s in sections for row in s["rows"]),
        "verticals": sorted(verticals),
        "audience": stored.get("audience"),
        "depth": stored.get("depth"),
        "interests": [str(i) for i in stored.get("interests") or []],
    }


def _alert_is_finance(alert: Alert) -> bool:
    if alert.article is not None and alert.article.vertical == Vertical.FINANCE:
        return True
    return alert.follow is not None and alert.follow.vertical == Vertical.FINANCE


def _follow_view(follow: Follow) -> dict[str, Any]:
    criteria = follow.criteria or {}
    try:
        model = FollowCriteria.model_validate(criteria)
    except ValidationError:
        model = FollowCriteria()
    alerts = list(follow.alerts)
    return {
        "follow": follow,
        "criteria_lines": describe_criteria(model),
        "criteria": model,
        "alert_count": len(alerts),
        "unread_count": sum(1 for a in alerts if a.read_at is None),
        "useful_count": sum(1 for a in alerts if a.feedback == "useful"),
    }


def _interest_is_finance(interest: Any) -> bool:
    return interest.vertical == Vertical.FINANCE or str(interest.value).strip().lower() == "finance"


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------


@router.get("/me")
def dashboard(request: Request, db: DB, user: RequiredUser):
    latest = db.scalars(
        select(Briefing)
        .where(Briefing.user_id == user.id)
        .order_by(Briefing.generated_at.desc(), Briefing.id.desc())
        .limit(1)
    ).first()
    briefing_count = int(db.scalar(select(func.count(Briefing.id)).where(Briefing.user_id == user.id)) or 0)
    latest_view = briefing_view(db, latest, user) if latest is not None else None

    unread = user_alerts(db, user, unread_only=True, limit=MAX_DASHBOARD_ALERTS)
    unread_total = int(
        db.scalar(select(func.count(Alert.id)).where(Alert.user_id == user.id, Alert.read_at.is_(None))) or 0
    )
    alert_total = int(db.scalar(select(func.count(Alert.id)).where(Alert.user_id == user.id)) or 0)
    follows = [_follow_view(f) for f in user_follows(db, user)]
    interests = sorted(user.interests, key=lambda i: (i.created_at, i.id))
    grouped = interests_by_kind(interests)

    show_finance = (
        (latest_view is not None and latest_view["is_finance"])
        or any(_alert_is_finance(a) for a in unread)
        or any(f["follow"].vertical == Vertical.FINANCE for f in follows)
        or any(_interest_is_finance(i) for i in interests)
    )
    ctx = {
        "user": user,
        "latest": latest_view,
        "preview_limit": MAX_BRIEFING_PREVIEW_ITEMS,
        "briefing_count": briefing_count,
        "alerts": unread,
        "unread_total": unread_total,
        "alert_total": alert_total,
        "follows": follows,
        "interests": interests,
        "interest_groups": [
            (kind, label, hint, grouped.get(kind, [])) for kind, label, hint in INTEREST_KIND_CHOICES
        ],
        "audience_choices": AUDIENCE_CHOICES,
        "depth_choices": DEPTH_CHOICES,
        "frequency_choices": FREQUENCY_CHOICES,
        "interest_kinds": INTEREST_KIND_CHOICES,
        "event_choices": FOLLOW_EVENT_CHOICES,
        "show_finance": show_finance,
        "next_url": "/me",
        "max_interest_length": MAX_INTEREST_LENGTH,
        "max_question_length": MAX_QUESTION_LENGTH,
    }
    return _page(request, "account/dashboard.html", ctx)


@router.get("/me/alerts")
def alerts_page(request: Request, db: DB, user: RequiredUser, show: str = "all"):
    unread_only = show == "unread"
    alerts = user_alerts(db, user, unread_only=unread_only, limit=MAX_ALERTS_PAGE)
    counts = {
        "all": int(db.scalar(select(func.count(Alert.id)).where(Alert.user_id == user.id)) or 0),
        "unread": int(
            db.scalar(select(func.count(Alert.id)).where(Alert.user_id == user.id, Alert.read_at.is_(None))) or 0
        ),
        "useful": int(
            db.scalar(select(func.count(Alert.id)).where(Alert.user_id == user.id, Alert.feedback == "useful"))
            or 0
        ),
        "unwanted": int(
            db.scalar(select(func.count(Alert.id)).where(Alert.user_id == user.id, Alert.feedback == "unwanted"))
            or 0
        ),
    }
    ctx = {
        "alerts": alerts,
        "show": "unread" if unread_only else "all",
        "counts": counts,
        "show_finance": any(_alert_is_finance(a) for a in alerts),
        "next_url": "/me/alerts" + ("?show=unread" if unread_only else ""),
    }
    return _page(request, "account/alerts.html", ctx)


# --------------------------------------------------------------------------
# Preferences and interests
# --------------------------------------------------------------------------


@router.post("/me/preferences", dependencies=[CSRF])
def update_preferences(
    request: Request,
    db: DB,
    user: PostUser,
    audience: Annotated[str | None, Form()] = None,
    depth: Annotated[str | None, Form()] = None,
    email_opt_in: Annotated[str | None, Form()] = None,
    briefing_frequency: Annotated[str | None, Form()] = None,
):
    """Save the reader's preferences; an unticked email box means "no email" (browsers omit it)."""
    audience = (audience or "").strip().lower() or None
    depth = (depth or "").strip().lower() or None
    frequency = (briefing_frequency or "").strip().lower() or None
    problems = []
    if audience is not None and audience not in {a.value for a in Audience}:
        problems.append("Choose Student, Educator, or Professional for “Explain for”.")
    if depth is not None and depth not in {d.value for d in Depth}:
        problems.append("Choose introductory, standard, or technical depth.")
    if frequency is not None and frequency not in BRIEFING_FREQUENCIES:
        problems.append("Choose how often you want briefings.")
    if problems:
        flash(request, " ".join(problems), "error")
        return _redirect("/me#preferences")
    set_preferences(
        db, user, audience=audience, depth=depth, email_opt_in=_truthy(email_opt_in), briefing_frequency=frequency
    )
    message = "Preferences saved."
    message += (
        " Briefings will also arrive by email." if user.email_opt_in else " Briefings stay on the web (no email)."
    )
    flash(request, message, "success")
    response = _redirect("/me#preferences")
    # Keep the masthead's "Explain for" switch in step with the profile.
    response.set_cookie(AUDIENCE_COOKIE, user.audience, max_age=60 * 60 * 24 * 365, samesite="lax")
    return response


@router.post("/me/interests", dependencies=[CSRF])
def add_interest_route(
    request: Request,
    db: DB,
    user: PostUser,
    kind: Annotated[str, Form()] = "topic",
    value: Annotated[str, Form()] = "",
    vertical: Annotated[str | None, Form()] = None,
):
    before = {i.id for i in user.interests}
    try:
        interest = add_interest(db, user, kind, value, vertical=(vertical or "").strip() or None)
    except ValueError as exc:
        flash(request, str(exc), "error")
        return _redirect("/me#interests")
    if interest.id in before:
        flash(request, f"“{truncate(interest.value, 120)}” is already in your interests.", "info")
    else:
        label = INTEREST_KIND_SINGULAR.get(interest.kind, "Interest")
        flash(
            request,
            f"{label} added: “{truncate(interest.value, 120)}”. Your next briefing will look for it.",
            "success",
        )
    return _redirect("/me#interests")


@router.post("/me/interests/{interest_id}/delete", dependencies=[CSRF])
def remove_interest_route(request: Request, interest_id: int, db: DB, user: PostUser):
    interest = next((i for i in user.interests if i.id == interest_id), None)
    value = interest.value if interest is not None else ""
    if not remove_interest(db, user, interest_id):
        raise HTTPException(status_code=404, detail="We couldn't find that interest.")
    flash(request, f"Removed “{truncate(value, 120)}” from your interests.", "success")
    return _redirect("/me#interests")


# --------------------------------------------------------------------------
# Briefings
# --------------------------------------------------------------------------


@router.get("/me/briefings")
def briefings_page(request: Request, db: DB, user: RequiredUser):
    briefings = list(
        db.scalars(
            select(Briefing)
            .where(Briefing.user_id == user.id)
            .order_by(Briefing.generated_at.desc(), Briefing.id.desc())
            .limit(MAX_BRIEFINGS_LIST)
        )
    )
    rows = []
    for b in briefings:
        content = _briefing_content(b)
        rows.append(
            {
                "briefing": b,
                "intro": content.intro,
                "headings": [s.heading for s in content.sections],
                "is_finance": FINANCE_DISCLAIMER in (content.closing_note or ""),
            }
        )
    ctx = {"rows": rows, "show_finance": any(r["is_finance"] for r in rows)}
    return _page(request, "account/briefings.html", ctx)


@router.get("/me/briefings/{briefing_id}")
def briefing_page(request: Request, briefing_id: int, db: DB, user: RequiredUser):
    briefing = _owned_briefing(db, briefing_id, user)
    first_open = briefing.opened_at is None
    if first_open:
        briefing.opened_at = utcnow()
        db.flush()
    previous = db.scalars(
        select(Briefing)
        .where(
            Briefing.user_id == user.id, Briefing.id != briefing.id, Briefing.generated_at <= briefing.generated_at
        )
        .order_by(Briefing.generated_at.desc(), Briefing.id.desc())
        .limit(1)
    ).first()
    ctx = briefing_view(db, briefing, user)
    ctx.update({"previous": previous, "next_url": f"/me/briefings/{briefing.id}"})
    return _page(request, "account/briefing.html", ctx)


@router.post("/me/briefings/generate", dependencies=[CSRF])
def generate_briefing(request: Request, db: DB, user: PostUser):
    """ "Brief me now": build (force) and deliver a briefing from what changed since the last one."""
    if not user.interests and not user_follows(db, user):
        flash(
            request,
            "Add an interest or follow a question first — the agent builds briefings around what you choose.",
            "info",
        )
        return _redirect("/me#interests")
    try:
        with db.begin_nested():
            briefing = build_briefing(db, user, force=True)
            if briefing is not None:
                deliver_briefing(db, briefing, user)
    except Exception:  # the agent must never take the dashboard down
        log.exception("Briefing for user %s failed", user.id)
        flash(request, "The briefing agent couldn't finish just now. Please try again in a few minutes.", "error")
        return _redirect("/me#briefing")
    if briefing is None:
        flash(
            request,
            "Nothing new since your last briefing. The agent will brief you when published coverage or an alert "
            "matches your interests.",
            "info",
        )
        return _redirect("/me#briefing")
    n = briefing.item_count
    message = f"Your briefing is ready: {n} update{'s' if n != 1 else ''}."
    if briefing.delivery_channel == "email":
        message += " A copy was also emailed to you."
    flash(request, message, "success")
    return _redirect(f"/me/briefings/{briefing.id}")


# --------------------------------------------------------------------------
# Follow This Story
# --------------------------------------------------------------------------


def _explicit_criteria(
    question: str,
    event_types: list[str],
    require_independent: str | None,
    require_primary: str | None,
    include_opinion: str | None,
    custom: str | None,
) -> FollowCriteria | None:
    """Criteria from the form when the reader set any of them; None lets the service derive them from the question.

    Exclusions the form has no control for (blogs, preprints) still come from
    the reader's own words; opinion follows the "Include opinion" box.
    """
    events = list(dict.fromkeys(e.strip() for e in event_types if e and e.strip() in _EVENT_VALUES))
    custom_text = " ".join((custom or "").split())[:MAX_CUSTOM_CRITERION]
    wants_opinion = _truthy(include_opinion)
    if not (events or _truthy(require_independent) or _truthy(require_primary) or wants_opinion or custom_text):
        return None
    derived = derive_criteria(question)
    exclude = [d for d in derived.exclude_doc_types if d != "opinion"]
    if not wants_opinion:
        exclude.insert(0, "opinion")
    elif events:
        # Opinion pieces are classified as opinion/analysis events; keep them eligible as the reader asked.
        events.extend(e for e in (EventType.OPINION.value, EventType.ANALYSIS.value) if e not in events)
    return FollowCriteria(
        event_types=events,
        exclude_doc_types=exclude,
        require_independent=_truthy(require_independent),
        require_primary=_truthy(require_primary),
        custom=custom_text,
    )


@router.post("/follows", dependencies=[CSRF])
def create_follow_route(
    request: Request,
    db: DB,
    user: CurrentUser,
    question: Annotated[str, Form()] = "",
    target_type: Annotated[str, Form()] = "topic",
    story_id: Annotated[str | None, Form()] = None,
    entity_name: Annotated[str | None, Form()] = None,
    vertical: Annotated[str | None, Form()] = None,
    event_types: Annotated[list[str] | None, Form()] = None,
    require_independent: Annotated[str | None, Form()] = None,
    require_primary: Annotated[str | None, Form()] = None,
    include_opinion: Annotated[str | None, Form()] = None,
    custom: Annotated[str | None, Form()] = None,
    next: Annotated[str | None, Form()] = None,
):
    back = safe_next(next, "/me")
    if user is None:
        return _redirect("/login?" + urlencode({"next": back}))

    target = (target_type or "topic").strip().lower() or "topic"
    if target not in FOLLOW_TARGETS:
        flash(request, "Choose what to follow: a topic or question, a story, a company, or a sector.", "error")
        return _redirect(back)
    story: Story | None = None
    raw_story = (story_id or "").strip()
    if raw_story:
        if not raw_story.isdigit():
            raise HTTPException(status_code=404, detail="We couldn't find that story.")
        story = db.get(Story, int(raw_story))
        if story is None:
            raise HTTPException(status_code=404, detail="We couldn't find that story.")
    if target == "story" and story is None:
        flash(request, "Pick the story to follow from its page.", "error")
        return _redirect(back)
    entity = " ".join((entity_name or "").split()) or None
    if target == "company" and not entity:
        flash(request, "Name the company or organization to follow.", "error")
        return _redirect(back)
    vertical_value = (vertical or "").strip().lower() or None
    if vertical_value is not None and vertical_value not in _VERTICAL_VALUES:
        flash(request, "Choose Technology, Education, Industry, or Finance.", "error")
        return _redirect(back)
    if target == "sector" and vertical_value is None and not entity:
        flash(request, "Choose the section (or name the sector) to follow.", "error")
        return _redirect(back)

    text = " ".join((question or "").split())
    if not text and target == "story" and story is not None:
        text = f"Tell me about material developments in: {story.title}"
    criteria = _explicit_criteria(
        text, event_types or [], require_independent, require_primary, include_opinion, custom
    )
    try:
        follow = create_follow(
            db,
            user,
            text,
            target_type=target,
            story=story,
            entity_name=entity,
            vertical=vertical_value,
            criteria=criteria,
        )
    except ValueError as exc:
        flash(request, str(exc), "error")
        return _redirect(back)
    flash(
        request,
        f"Following: “{truncate(follow.question, 140)}”. The agent will alert you only when a new development "
        "meets your criteria.",
        "success",
    )
    return _redirect(back)


@router.post("/follows/{follow_id}/delete", dependencies=[CSRF])
def stop_follow(
    request: Request,
    follow_id: int,
    db: DB,
    user: PostUser,
    next: Annotated[str | None, Form()] = None,
):
    follow = deactivate_follow(db, user, follow_id)
    if follow is None:
        raise HTTPException(status_code=404, detail="We couldn't find that follow.")
    message = f"Stopped following “{truncate(follow.question, 140)}”. Alerts it already found are kept."
    if is_htmx(request):
        return _partial(request, "account/_follow_stopped.html", {"follow": follow, "message": message})
    flash(request, message, "success")
    return _redirect(safe_next(next, "/me#follows"))


# --------------------------------------------------------------------------
# Alerts
# --------------------------------------------------------------------------


@router.post("/alerts/{alert_id}/feedback", dependencies=[CSRF])
def alert_feedback(
    request: Request,
    alert_id: int,
    db: DB,
    user: PostUser,
    feedback: Annotated[str, Form()] = "",
    next: Annotated[str | None, Form()] = None,
):
    alert = _owned_alert(db, alert_id, user)
    verdict = (feedback or "").strip().lower()
    if verdict not in ALERT_FEEDBACK:
        raise HTTPException(status_code=400, detail="Feedback must be “useful” or “unwanted”.")
    mark_alert(db, alert, feedback=verdict, read=True)
    if is_htmx(request):
        ctx = {"alert": alert, "status_message": FEEDBACK_MESSAGES[verdict], "next_url": safe_next(next, "/me")}
        return _partial(request, "account/_alert.html", ctx)
    flash(request, FEEDBACK_MESSAGES[verdict], "success")
    return _redirect(safe_next(next, "/me#alerts"))


@router.post("/alerts/{alert_id}/read", dependencies=[CSRF])
def alert_read(
    request: Request,
    alert_id: int,
    db: DB,
    user: PostUser,
    read: Annotated[str | None, Form()] = None,
    next: Annotated[str | None, Form()] = None,
):
    """Mark an alert read (or unread again with ``read=0``)."""
    alert = _owned_alert(db, alert_id, user)
    mark_read = (read or "1").strip().lower() not in {"0", "false", "no", "off"}
    mark_alert(db, alert, read=mark_read)
    message = "Marked as read." if mark_read else "Marked as unread."
    if is_htmx(request):
        ctx = {"alert": alert, "status_message": message, "next_url": safe_next(next, "/me")}
        return _partial(request, "account/_alert.html", ctx)
    flash(request, message, "success")
    return _redirect(safe_next(next, "/me#alerts"))
