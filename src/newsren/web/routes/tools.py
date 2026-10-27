"""Intelligence tools for an AI Story Brief, the Ask Newsren agent page, feedback, and quizzes.

Reading stays central (plan §5): the brief page embeds these tools in a tab
bar and loads them lazily with htmx (``HX-Request``), while every tool URL
also renders as a full page (wrapped in ``base.html``) so everything works
without JavaScript.

Tools for brief ``{id}`` (visible when the brief is approved, or to editors
as an unpublished preview; otherwise 404):

* ``GET  /briefs/{id}/tools/connected`` — Connected stories: the story's
  timeline merged with related stories' events (cross-vertical storyline) and
  the story-intelligence summary (what happened first / what's new / what is
  disputed / how the evidence changed).
* ``GET  /briefs/{id}/tools/learn?audience=`` — News-to-Learning material and
  quiz (posts to ``POST /learning/{content_id}/quiz``).
* ``GET  /briefs/{id}/tools/evidence`` — Evidence and Bias Inspection.
* ``GET  /briefs/{id}/tools/impact?question=`` — Industry Impact.
* ``GET  /briefs/{id}/tools/follow`` — Follow developments (posts to ``/follows``).
* ``GET  /briefs/{id}/tools/products/{key}`` — one editorial product.
* ``GET  /briefs/{id}/tools/ask`` and ``POST /briefs/{id}/ask`` — Ask about this story.

Plus ``GET/POST /ask`` (the agent entry point), ``GET /questions/{id}``
(answer permalink: the asker's session/account or editors only),
``POST /feedback`` and ``POST /learning/{content_id}/quiz``.

Inline citation markers (``[c3]``) in generated text are rendered by the
``cite_markers`` template filter: the text is escaped first, then markers for
known citation ids become links to that item's source list. Each tool
prefixes its source-list anchors so several lists can share a page.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from markupsafe import Markup, escape
from sqlalchemy import select
from sqlalchemy.orm import Session

from newsren.agents.evidence import inspect_evidence
from newsren.agents.impact import default_question, industry_impact
from newsren.agents.learning import lead_subject, learning_materials, record_quiz_attempt, strip_citation_marks
from newsren.agents.products import PRODUCTS, available_products, run_product
from newsren.agents.research import ask, questions_for
from newsren.editorial.guardrails import FINANCE_DISCLAIMER
from newsren.intelligence.clustering import related_stories
from newsren.intelligence.timeline import story_citations
from newsren.models import (
    AgentRun,
    Alert,
    Article,
    Audience,
    Brief,
    Briefing,
    BriefStatus,
    EventType,
    GeneratedContent,
    Question,
    Story,
    User,
    Vertical,
)
from newsren.schemas import EvidenceReportOut, ImpactOut, LearningOut, ProductOut, ResearchAnswerOut
from newsren.users.service import (
    FEEDBACK_TARGETS,
    derive_criteria,
    describe_criteria,
    record_feedback,
    user_follows,
)
from newsren.web.deps import CSRF, DB, CurrentUser, ReaderAudience, csrf_token, session_key
from newsren.web.routes.auth import safe_next
from newsren.web.templating import badge, flash, render, templates
from newsren.web.tracking import record_view

router = APIRouter()

MAX_QUESTION_CHARS = 1000
MAX_IMPACT_QUESTION_CHARS = 300
MAX_RELATED_STORIES = 5
MAX_SUGGESTED_PROMPTS = 5

# --------------------------------------------------------------------------
# Template helpers: citation markers and paragraph blocks
# --------------------------------------------------------------------------

CITE_MARK = re.compile(r"\s?\[(c\d+)\]")
_BULLET = re.compile(r"^[-•*]\s+")
_SENTENCE_END = re.compile(r"""[.?!;,:"”'’\]]$""")
HEADING_MAX_CHARS = 90


def _field(item: Any, name: str, default: Any = "") -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def cite_markers(text: Any, citations: Iterable[Any] | None = None, prefix: str = "") -> Markup:
    """Escape ``text`` and turn ``[cN]`` markers for known citations into links to ``#{prefix}cite-cN``.

    Markers whose id is not in ``citations`` are dropped (never shown as
    broken references). Safe for untrusted text: everything except the
    generated anchors is escaped.
    """
    if text is None or text == "":
        return Markup("")
    raw = str(text)
    by_id = {str(_field(c, "id")): c for c in citations or []}
    parts: list[Markup] = []
    pos = 0
    for match in CITE_MARK.finditer(raw):
        # The marker (and the space before it) is replaced: superscript refs attach to the word.
        parts.append(escape(raw[pos : match.start()]))
        cid = match.group(1)
        citation = by_id.get(cid)
        if citation is not None:
            title = " — ".join(str(x) for x in (_field(citation, "title"), _field(citation, "publisher")) if x)
            parts.append(
                Markup('<a class="cite" href="#{}cite-{}" title="{}">[{}]</a>').format(prefix, cid, title, cid[1:])
            )
        pos = match.end()
    parts.append(escape(raw[pos:]))
    return Markup("").join(parts)


def text_blocks(text: str | None) -> list[dict[str, Any]]:
    """Split generated text into headings, paragraphs, and bullet lists (blank-line separated blocks)."""
    blocks: list[dict[str, Any]] = []
    chunks = [c for c in re.split(r"\n\s*\n", (text or "").strip()) if c.strip()]
    for index, chunk in enumerate(chunks):
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if (
            len(lines) == 1
            and index < len(chunks) - 1
            and len(lines[0]) <= HEADING_MAX_CHARS
            and not _BULLET.match(lines[0])
            and not _SENTENCE_END.search(lines[0])
            and not CITE_MARK.search(lines[0])
        ):
            blocks.append({"kind": "heading", "text": lines[0]})
            continue
        para: list[str] = []
        items: list[str] = []
        for line in lines:
            if _BULLET.match(line):
                if para:
                    blocks.append({"kind": "paragraph", "text": " ".join(para)})
                    para = []
                items.append(_BULLET.sub("", line))
            else:
                if items:
                    blocks.append({"kind": "list", "entries": items})
                    items = []
                para.append(line)
        if para:
            blocks.append({"kind": "paragraph", "text": " ".join(para)})
        if items:
            blocks.append({"kind": "list", "entries": items})
    return blocks


def label_badges(value: str | None) -> Markup:
    """Badges for a free-text evidence label such as "backtest, simulated" or "independently tested"."""
    parts = [p.strip().lower().replace(" ", "_").replace("-", "_") for p in re.split(r"[,;/]", value or "")]
    return Markup(" ").join(badge(p) for p in dict.fromkeys(parts) if p)


def uncited(ids: Iterable[str] | None, text: str | None) -> list[str]:
    """Citation ids not already referenced inline in ``text`` (avoids printing a reference twice)."""
    inline = set(CITE_MARK.findall(text or ""))
    return [cid for cid in ids or [] if cid not in inline]


def strip_cites(text: str | None) -> str:
    """Text without inline citation markers (for attributes and headings)."""
    return strip_citation_marks(text)


templates.env.filters.update(
    {
        "cite_markers": cite_markers,
        "text_blocks": text_blocks,
        "label_badges": label_badges,
        "uncited": uncited,
        "strip_cites": strip_cites,
    }
)

# --------------------------------------------------------------------------
# Vocabulary shown in the tools
# --------------------------------------------------------------------------

TOOLS: list[dict[str, str]] = [
    {"key": "connected", "label": "Connected stories", "blurb": "Events and evolving timelines"},
    {"key": "learn", "label": "Learn this topic", "blurb": "Explanations matched to readers"},
    {"key": "evidence", "label": "Inspect evidence", "blurb": "Sources, claims, and uncertainty"},
    {"key": "impact", "label": "Industry impact", "blurb": "Who could be affected, and how"},
    {"key": "follow", "label": "Follow developments", "blurb": "Updates only when your criteria are met"},
    {"key": "ask", "label": "Ask about this story", "blurb": "A sourced research answer"},
]
TOOL_LABELS = {t["key"]: t["label"] for t in TOOLS}

SOURCE_ROLE_LABELS = {
    "original": "Original source",
    "secondary": "Secondary reporting",
    "vendor_claim": "Vendor claim",
    "independent_evaluation": "Independent evaluation",
    "commentary": "Commentary",
}

FOLLOW_EVENT_CHOICES: list[tuple[str, str]] = [
    (EventType.INDEPENDENT_EVALUATION.value, "Independent evaluation"),
    (EventType.BENCHMARK.value, "Benchmark result"),
    (EventType.RESEARCH_PUBLICATION.value, "Research publication (e.g. peer-reviewed study)"),
    (EventType.PRODUCT_LAUNCH.value, "Product launch"),
    (EventType.ADOPTION.value, "Adoption (e.g. a school or company starts using it)"),
    (EventType.REGULATION.value, "Regulation"),
    (EventType.POLICY.value, "Policy decision"),
    (EventType.FILING.value, "Filing"),
    (EventType.EARNINGS.value, "Earnings"),
    (EventType.FUNDING.value, "Funding round"),
    (EventType.ENFORCEMENT.value, "Enforcement action"),
    (EventType.CORRECTION.value, "Correction"),
]

FOLLOW_TEMPLATES = {
    Vertical.TECHNOLOGY.value: (
        "Tell me when an independent benchmark or evaluation is published for {subject}, "
        "not whenever someone writes another opinion article about it."
    ),
    Vertical.EDUCATION.value: (
        "Tell me when a peer-reviewed study, a policy decision, or a school adoption involving {subject} "
        "is reported, not opinion pieces."
    ),
    Vertical.INDUSTRY.value: (
        "Tell me when {subject} reports a documented change such as a filing, a supply agreement, "
        "or new capacity, not forecasts or commentary."
    ),
    Vertical.FINANCE.value: (
        "Tell me when {subject} files with a regulator, reports earnings, raises funding, or faces an "
        "enforcement action, not price moves or opinion."
    ),
}

DEFAULT_PROMPTS = {
    Vertical.TECHNOLOGY.value: [
        "Has the claimed performance been independently verified?",
        "What is technically new compared with earlier models?",
    ],
    Vertical.EDUCATION.value: [
        "Who was studied, and which learning outcomes were measured?",
        "What remains uncertain before schools act on this?",
    ],
    Vertical.INDUSTRY.value: [
        "Which suppliers and customers are documented in the sources?",
        "Which of the projected effects are supported by evidence?",
    ],
    Vertical.FINANCE.value: [
        "Is this result a backtest, a simulation, or a live track record?",
        "What did the filing or announcement actually report?",
    ],
}

ASK_EXAMPLES = {
    Vertical.TECHNOLOGY.value: [
        "Has anyone independently verified Lumen-4's math benchmark score?",
        "What is the difference between a vendor benchmark and an independent evaluation?",
    ],
    Vertical.EDUCATION.value: [
        "Did students using AI tutors learn more, or just finish exercises faster?",
        "What does the proposed AI literacy requirement mean for rural districts?",
    ],
    Vertical.INDUSTRY.value: [
        "How could new chip fab capacity affect suppliers and technicians?",
        "Which supply-chain effects of the fab expansion are documented, and which are projections?",
    ],
    Vertical.FINANCE.value: [
        "Is the AI portfolio agents result a backtest or a live track record?",
        "What did regulators find in the recent AI-washing case?",
    ],
}

QUIZ_RESULT_SESSION_KEY = "quiz_result"

# --------------------------------------------------------------------------
# Request helpers
# --------------------------------------------------------------------------


def is_htmx(request: Request) -> bool:
    """True for an htmx partial request (not a history-restore, which needs the full page)."""
    return (
        request.headers.get("HX-Request") == "true" and request.headers.get("HX-History-Restore-Request") != "true"
    )


def can_view_brief(brief: Brief | None, user: User | None) -> bool:
    if brief is None:
        return False
    return brief.status == BriefStatus.APPROVED or (user is not None and user.is_editor)


def visible_brief(db: Session, brief_id: int, user: User | None) -> Brief:
    """The brief if this reader may see it (approved, or any status for editors); otherwise 404."""
    brief = db.get(Brief, brief_id)
    if not can_view_brief(brief, user):
        raise HTTPException(status_code=404, detail="We couldn't find that brief.")
    assert brief is not None
    return brief


def _partial(request: Request, name: str, context: dict[str, Any], status_code: int = 200) -> Response:
    """Render a fragment for htmx without consuming the session's flash message."""
    ctx = dict(context)
    ctx.setdefault("current_user", getattr(request.state, "user", None))
    ctx.setdefault("audience", Audience.PROFESSIONAL.value)
    ctx.setdefault("csrf_token", csrf_token(request))
    ctx.setdefault("flash", None)
    ctx.setdefault("FINANCE_DISCLAIMER", FINANCE_DISCLAIMER)
    response = templates.TemplateResponse(request, name, ctx, status_code=status_code)
    response.headers["Vary"] = "HX-Request"
    return response


def _page(request: Request, name: str, context: dict[str, Any], status_code: int = 200) -> Response:
    ctx = dict(context)
    ctx.setdefault("FINANCE_DISCLAIMER", FINANCE_DISCLAIMER)
    response = render(request, name, ctx, status_code=status_code)
    response.headers["Vary"] = "HX-Request"
    return response


def _citations(rows: Sequence[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in rows or []:
        out.append(c if isinstance(c, dict) else c.model_dump(mode="json"))
    return out


def tool_url(brief: Brief, key: str) -> str:
    return f"/briefs/{brief.id}/tools/{key}"


def _tool_nav(brief: Brief, active: str) -> dict[str, Any]:
    tools = [{**t, "url": tool_url(brief, t["key"]), "active": t["key"] == active} for t in TOOLS]
    products = [
        {
            "key": spec.key,
            "label": spec.name,
            "url": tool_url(brief, f"products/{spec.key}"),
            "active": active == f"product:{spec.key}",
        }
        for spec in available_products(brief)
    ]
    return {"tools": tools, "products": products}


def render_tool(
    request: Request,
    db: Session,
    user: User | None,
    brief: Brief,
    tool: str,
    template: str,
    context: dict[str, Any],
    *,
    title: str,
    finance: bool | None = None,
) -> Response:
    """A tool as an htmx partial (``HX-Request``) or wrapped in the full page layout."""
    full = not is_htmx(request)
    ctx: dict[str, Any] = {
        "brief": brief,
        "tool": tool,
        "tool_title": title,
        "tool_template": template,
        "full_page": full,
        "current_user": user,
        "unpublished": brief.status != BriefStatus.APPROVED,
        "is_finance": brief.vertical == Vertical.FINANCE if finance is None else finance,
        **context,
    }
    if not full:
        return _partial(request, template, ctx)
    record_view(db, user, request.url.path, story_id=brief.story_id, brief_id=brief.id)
    ctx["nav"] = _tool_nav(brief, tool)
    ctx["active_vertical"] = brief.vertical
    return _page(request, "tools/page.html", ctx)


# --------------------------------------------------------------------------
# Connected stories (Story Intelligence, plan §2.2)
# --------------------------------------------------------------------------


def story_intel_visible(story: Story | None, user: User | None) -> bool:
    """The AI story-intelligence summary is public once the story has an approved brief; editors always see it."""
    if story is None:
        return False
    if user is not None and user.is_editor:
        return True
    return any(b.status == BriefStatus.APPROVED for b in story.briefs)


def storyline(db: Session, brief: Brief) -> tuple[Story | None, list[Story], list[dict[str, Any]]]:
    """The brief's story, its related stories, and their timeline events merged in date order."""
    story = brief.story
    if story is None:
        return None, [], []
    related = related_stories(db, story, limit=MAX_RELATED_STORIES)
    items: list[dict[str, Any]] = []
    seen: set[int] = set()
    for member in [story, *related]:
        for event in member.events:
            if event.id in seen:
                continue
            seen.add(event.id)
            article = event.article
            items.append(
                {
                    "event": event,
                    "story": member,
                    "article": article,
                    "own": member.id == story.id,
                    "cross_vertical": member.vertical != brief.vertical,
                }
            )
    items.sort(key=lambda i: (i["event"].occurred_at, i["event"].id))
    return story, [s for s in related if s.events], items


@router.get("/briefs/{brief_id}/tools/connected")
def tool_connected(request: Request, brief_id: int, db: DB, user: CurrentUser):
    brief = visible_brief(db, brief_id, user)
    story, related, items = storyline(db, brief)
    intel = story_intel_visible(story, user)
    products = {spec.key for spec in available_products(brief)}
    run = None
    if story is not None and intel:
        run = db.scalars(
            select(AgentRun)
            .where(
                AgentRun.agent == "story_intelligence",
                AgentRun.subject_type == "story",
                AgentRun.subject_id == story.id,
            )
            .order_by(AgentRun.id.desc())
            .limit(1)
        ).first()
    ctx = {
        "story": story,
        "related": related,
        "items": items,
        "show_intel": intel,
        "intel_generated_by": run.model
        if run is not None and run.provider == "claude" and run.model
        else "offline",
        "intel_generated_at": (run.finished_at or run.started_at) if run is not None else None,
        "story_cites": _citations(story_citations(story)) if (story is not None and intel) else [],
        "verticals_in_storyline": sorted({i["story"].vertical for i in items}),
        "timeline_product": "technology_timeline" in products,
    }
    return render_tool(
        request, db, user, brief, "connected", "tools/connected.html", ctx, title="Connected stories"
    )


# --------------------------------------------------------------------------
# Learn this topic (News-to-Learning, plan §2.3)
# --------------------------------------------------------------------------


def quiz_review(quiz: Sequence[Any], answers: Sequence[int]) -> dict[str, Any]:
    """Per-question review and comprehension/transfer scores for submitted answers."""
    rows = []
    scores = {"comprehension": [0, 0], "transfer": [0, 0]}
    for index, question in enumerate(quiz):
        chosen = answers[index] if index < len(answers) else -1
        correct = chosen == question.answer_index
        bucket = scores.get(question.kind, scores["comprehension"])
        bucket[1] += 1
        bucket[0] += int(correct)
        rows.append(
            {
                "index": index,
                "question": question,
                "chosen": chosen,
                "answered": 0 <= chosen < len(question.options),
                "correct": correct,
                "answer_text": question.options[question.answer_index]
                if 0 <= question.answer_index < len(question.options)
                else "",
            }
        )
    return {
        "rows": rows,
        "comprehension_correct": scores["comprehension"][0],
        "comprehension_total": scores["comprehension"][1],
        "transfer_correct": scores["transfer"][0],
        "transfer_total": scores["transfer"][1],
    }


@router.get("/briefs/{brief_id}/tools/learn")
def tool_learn(request: Request, brief_id: int, db: DB, user: CurrentUser, audience: ReaderAudience):
    brief = visible_brief(db, brief_id, user)
    content = learning_materials(db, brief, audience)
    learning = LearningOut.model_validate(content.content)
    result = None
    stashed = request.session.get(QUIZ_RESULT_SESSION_KEY)
    if isinstance(stashed, dict) and stashed.get("content_id") == content.id:
        request.session.pop(QUIZ_RESULT_SESSION_KEY, None)
        result = quiz_review(learning.quiz, [int(a) for a in stashed.get("answers", [])])
    ctx = {
        "content": content,
        "learning": learning,
        "citations": _citations(content.citations),
        "audience": audience,
        "quiz_result": result,
        "audience_links": [
            {
                "slug": a.value,
                "label": a.value.capitalize(),
                "url": f"{tool_url(brief, 'learn')}?audience={a.value}",
                "active": a.value == audience,
            }
            for a in (Audience.STUDENT, Audience.EDUCATOR, Audience.PROFESSIONAL)
        ],
    }
    return render_tool(request, db, user, brief, "learn", "tools/learn.html", ctx, title="Learn this topic")


async def form_fields(request: Request) -> dict[str, str]:
    """All string fields of a submitted form (for forms with a variable number of inputs)."""
    form = await request.form()
    return {key: value for key, value in form.multi_items() if isinstance(value, str)}


FormFields = Annotated[dict[str, str], Depends(form_fields)]


@router.post("/learning/{content_id}/quiz", dependencies=[CSRF])
def submit_quiz(request: Request, content_id: int, db: DB, user: CurrentUser, fields: FormFields):
    content = db.get(GeneratedContent, content_id)
    if content is None or content.kind != "learning" or content.brief_id is None:
        raise HTTPException(status_code=404, detail="We couldn't find that quiz.")
    brief = visible_brief(db, content.brief_id, user)
    learning = LearningOut.model_validate(content.content)
    answers: list[int] = []
    for index in range(len(learning.quiz)):
        value = fields.get(f"q{index}", "").strip()
        answers.append(int(value) if value.isdigit() else -1)
    record_quiz_attempt(db, content, answers, user=user)
    if is_htmx(request):
        return _partial(
            request,
            "tools/_quiz_result.html",
            {
                "quiz_result": quiz_review(learning.quiz, answers),
                "learning": learning,
                "content": content,
                "citations": _citations(content.citations),
                "brief": brief,
            },
        )
    request.session[QUIZ_RESULT_SESSION_KEY] = {"content_id": content.id, "answers": answers}
    audience = content.audience or Audience.STUDENT.value
    return RedirectResponse(f"{tool_url(brief, 'learn')}?audience={audience}#quiz-{content.id}", status_code=303)


# --------------------------------------------------------------------------
# Inspect evidence (Evidence and Bias Inspection, plan §2.5)
# --------------------------------------------------------------------------


@router.get("/briefs/{brief_id}/tools/evidence")
def tool_evidence(request: Request, brief_id: int, db: DB, user: CurrentUser):
    brief = visible_brief(db, brief_id, user)
    content = inspect_evidence(db, brief)
    report = EvidenceReportOut.model_validate(content.content)
    citations = _citations(content.citations)
    ctx = {
        "content": content,
        "report": report,
        "citations": citations,
        "cite_by_id": {c["id"]: c for c in citations},
        "role_labels": SOURCE_ROLE_LABELS,
    }
    return render_tool(request, db, user, brief, "evidence", "tools/evidence.html", ctx, title="Inspect evidence")


# --------------------------------------------------------------------------
# Industry impact (plan §2.4)
# --------------------------------------------------------------------------


@router.get("/briefs/{brief_id}/tools/impact")
def tool_impact(request: Request, brief_id: int, db: DB, user: CurrentUser, question: str | None = None):
    brief = visible_brief(db, brief_id, user)
    asked = " ".join((question or "").split())[:MAX_IMPACT_QUESTION_CHARS]
    content = industry_impact(db, brief, question=asked or None)
    impact = ImpactOut.model_validate(content.content)
    ctx = {
        "content": content,
        "impact": impact,
        "citations": _citations(content.citations),
        "asked": asked,
        "default_question": default_question(str(brief.vertical)),
    }
    return render_tool(request, db, user, brief, "impact", "tools/impact.html", ctx, title="Industry impact")


# --------------------------------------------------------------------------
# Follow developments (Follow This Story, plan §2.6)
# --------------------------------------------------------------------------


def follow_prompt(brief: Brief) -> str:
    """A suggested "Tell me when ..." question for this brief's story."""
    subject = lead_subject(brief)
    template = FOLLOW_TEMPLATES.get(str(brief.vertical), FOLLOW_TEMPLATES[Vertical.INDUSTRY.value])
    return template.format(subject=subject)


@router.get("/briefs/{brief_id}/tools/follow")
def tool_follow(request: Request, brief_id: int, db: DB, user: CurrentUser):
    brief = visible_brief(db, brief_id, user)
    question = follow_prompt(brief)
    criteria = derive_criteria(question)
    existing = []
    if user is not None and brief.story_id is not None:
        existing = [
            {"follow": f, "criteria": describe_criteria(f.criteria or {})}
            for f in user_follows(db, user)
            if f.story_id == brief.story_id
        ]
    ctx = {
        "question": question,
        "criteria": criteria,
        "event_choices": FOLLOW_EVENT_CHOICES,
        "include_opinion": "opinion" not in criteria.exclude_doc_types,
        "existing": existing,
        "login_url": "/login?" + urlencode({"next": f"/briefs/{brief.id}"}),
    }
    return render_tool(request, db, user, brief, "follow", "tools/follow.html", ctx, title="Follow developments")


# --------------------------------------------------------------------------
# Editorial products (plan §3 table)
# --------------------------------------------------------------------------


def product_display_title(product: ProductOut, name: str, brief: Brief) -> str:
    """The product's own title without a restated product name; empty when it only repeats the headline."""
    title = strip_citation_marks(product.title)
    if title.lower().startswith(name.lower() + ":"):
        title = title[len(name) + 1 :].strip()
    return "" if title.strip().lower() == (brief.headline or "").strip().lower() else title


@router.get("/briefs/{brief_id}/tools/products/{key}")
def tool_product(request: Request, brief_id: int, key: str, db: DB, user: CurrentUser):
    brief = visible_brief(db, brief_id, user)
    if key not in {spec.key for spec in available_products(brief)}:
        raise HTTPException(status_code=404, detail="That editorial product isn't available for this brief.")
    spec = PRODUCTS[key]
    content = run_product(db, key, brief)
    product = ProductOut.model_validate(content.content)
    finance = spec.vertical == Vertical.FINANCE or brief.vertical == Vertical.FINANCE
    ctx = {
        "content": content,
        "product": product,
        "product_title": product_display_title(product, spec.name, brief),
        "spec": spec,
        "citations": _citations(content.citations),
        "other_products": [
            {"key": s.key, "label": s.name, "url": tool_url(brief, f"products/{s.key}")}
            for s in available_products(brief)
            if s.key != key
        ],
    }
    return render_tool(
        request, db, user, brief, f"product:{key}", "tools/product.html", ctx, title=spec.name, finance=finance
    )


# --------------------------------------------------------------------------
# Ask about this story / Ask Newsren (plan §1 reader loop, §7 stage 2)
# --------------------------------------------------------------------------


def _quoted_prompt(lead: str, item: str) -> str:
    text = strip_citation_marks(item).strip().rstrip(".")
    if not text:
        return ""
    if len(text) > 140:
        text = text[:137].rsplit(" ", 1)[0] + "…"
    return f"{lead} “{text}”?"


def suggested_prompts(brief: Brief) -> list[str]:
    """Follow-up prompts from the brief's watch-next and uncertainty items, plus vertical defaults."""
    prompts: list[str] = list(DEFAULT_PROMPTS.get(str(brief.vertical), []))
    for item in (brief.watch_next or [])[:2]:
        prompts.append(_quoted_prompt("Any new evidence on", str(item)))
    for item in (brief.uncertainty or [])[:2]:
        prompts.append(_quoted_prompt("What would resolve this uncertainty:", str(item)))
    return [p for p in dict.fromkeys(prompts) if p][:MAX_SUGGESTED_PROMPTS]


def can_view_question(request: Request, question: Question, user: User | None) -> bool:
    if user is not None and (user.is_editor or (question.user_id is not None and question.user_id == user.id)):
        return True
    sid = request.session.get("sid")
    return bool(sid and question.session_key and question.session_key == sid)


def answer_context(db: Session, question: Question, user: User | None) -> dict[str, Any]:
    """Everything the answer partial needs: parsed result, citations, provenance, finance flag."""
    payload = question.result or {"answer": question.answer or ""}
    result = ResearchAnswerOut.model_validate(payload)
    citations = _citations(question.citations or result.citations)
    brief = db.get(Brief, question.brief_id) if question.brief_id else None
    if not can_view_brief(brief, user):
        brief = None
    run = db.get(AgentRun, question.agent_run_id) if question.agent_run_id else None
    generated_by = run.model if run is not None and run.provider == "claude" and run.model else "offline"
    article_ids = [c["article_id"] for c in citations if c.get("article_id")]
    finance_sources = bool(
        article_ids
        and db.scalar(
            select(Article.id).where(Article.id.in_(article_ids), Article.vertical == Vertical.FINANCE).limit(1)
        )
    )
    finance = bool(
        question.refused or (brief is not None and brief.vertical == Vertical.FINANCE) or finance_sources
    )
    return {
        "question": question,
        "result": result,
        "citations": citations,
        "prefix": f"answer-{question.id}-",
        "generated_by": generated_by,
        "generated_at": question.created_at,
        "answer_brief": brief,
        "is_finance": finance,
        "ask_url": tool_url(brief, "ask") if brief is not None else "/ask",
    }


def _pick_audience(form_value: str | None, reader_audience: str) -> str:
    """The audience a form asked for (the ask form carries the tool's "Explain for" choice), else the reader's."""
    return form_value if form_value in {a.value for a in Audience} else reader_audience


def _clean_question(question: str | None, prompt: str | None) -> str:
    return " ".join((question or "").split()) or " ".join((prompt or "").split())


def _question_error(text: str) -> str | None:
    if not text:
        return "Write a question for the research agent."
    if len(text) > MAX_QUESTION_CHARS:
        return f"Questions can be up to {MAX_QUESTION_CHARS:,} characters."
    return None


def _brief_questions(request: Request, db: Session, brief: Brief, user: User | None) -> list[Question]:
    stmt = select(Question).where(Question.brief_id == brief.id)
    if user is not None:
        stmt = stmt.where(Question.user_id == user.id)
    else:
        sid = request.session.get("sid")
        if not sid:
            return []
        stmt = stmt.where(Question.session_key == sid)
    return list(db.scalars(stmt.order_by(Question.created_at.desc(), Question.id.desc()).limit(5)))


@router.get("/briefs/{brief_id}/tools/ask")
def tool_ask(
    request: Request,
    brief_id: int,
    db: DB,
    user: CurrentUser,
    audience: ReaderAudience,
    question: str | None = None,
):
    brief = visible_brief(db, brief_id, user)
    ctx = {
        "prefill": " ".join((question or "").split())[:MAX_QUESTION_CHARS],
        "prompts": suggested_prompts(brief),
        "earlier": _brief_questions(request, db, brief, user),
        "audience": audience,
    }
    return render_tool(request, db, user, brief, "ask", "tools/ask_brief.html", ctx, title="Ask about this story")


@router.post("/briefs/{brief_id}/ask", dependencies=[CSRF])
def ask_about_brief(
    request: Request,
    brief_id: int,
    db: DB,
    user: CurrentUser,
    audience: ReaderAudience,
    question: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    form_audience: Annotated[str | None, Form(alias="audience")] = None,
):
    brief = visible_brief(db, brief_id, user)
    audience = _pick_audience(form_audience, audience)
    text = _clean_question(question, prompt)
    error = _question_error(text)
    if error:
        if is_htmx(request):
            return _partial(request, "tools/_ask_error.html", {"error": error})
        flash(request, error, "error")
        return RedirectResponse(tool_url(brief, "ask"), status_code=303)
    row = ask(db, text, brief=brief, audience=audience, user=user, session_key=session_key(request))
    if is_htmx(request):
        return _partial(request, "tools/_answer.html", answer_context(db, row, user))
    return RedirectResponse(f"/questions/{row.id}", status_code=303)


def _ask_page(
    request: Request,
    db: Session,
    user: User | None,
    *,
    prefill: str = "",
    error: str | None = None,
    status: int = 200,
) -> Response:
    recent = questions_for(db, user=user, session_key=None if user else request.session.get("sid"), limit=10)
    ctx = {
        "prefill": prefill,
        "error": error,
        "recent": recent,
        "examples": [
            {"vertical": v["slug"], "label": v["label"], "prompts": ASK_EXAMPLES.get(v["slug"], [])}
            for v in templates.env.globals["VERTICALS"]
        ],
        "active_vertical": "ask",
    }
    return _page(request, "tools/ask.html", ctx, status_code=status)


@router.get("/ask")
def ask_page(request: Request, db: DB, user: CurrentUser, question: str | None = None):
    return _ask_page(request, db, user, prefill=" ".join((question or "").split())[:MAX_QUESTION_CHARS])


@router.post("/ask", dependencies=[CSRF])
def ask_submit(
    request: Request,
    db: DB,
    user: CurrentUser,
    audience: ReaderAudience,
    question: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    form_audience: Annotated[str | None, Form(alias="audience")] = None,
):
    audience = _pick_audience(form_audience, audience)
    text = _clean_question(question, prompt)
    error = _question_error(text)
    if error:
        if is_htmx(request):
            return _partial(request, "tools/_ask_error.html", {"error": error})
        return _ask_page(request, db, user, prefill=text[:MAX_QUESTION_CHARS], error=error, status=400)
    row = ask(db, text, audience=audience, user=user, session_key=session_key(request))
    if is_htmx(request):
        return _partial(request, "tools/_answer.html", answer_context(db, row, user))
    return RedirectResponse(f"/questions/{row.id}", status_code=303)


@router.get("/questions/{question_id}")
def question_page(request: Request, question_id: int, db: DB, user: CurrentUser):
    question = db.get(Question, question_id)
    if question is None or not can_view_question(request, question, user):
        raise HTTPException(status_code=404, detail="We couldn't find that answer.")
    ctx = answer_context(db, question, user)
    ctx["active_vertical"] = "ask"
    ctx["answer_heading_tag"] = "h1"
    ctx["permalink"] = True  # no ask form on this page: follow-ups navigate instead of filling a box
    return _page(request, "tools/question.html", ctx)


# --------------------------------------------------------------------------
# Feedback (usefulness metric, plan §9)
# --------------------------------------------------------------------------


def _yes_no(value: str | None) -> bool | None:
    v = (value or "").strip().lower()
    if v in ("yes", "true", "1", "on"):
        return True
    if v in ("no", "false", "0"):
        return False
    return None


def _feedback_target_exists(db: Session, target_type: str, target_id: int, user: User | None) -> bool:
    if target_type == "answer":
        return db.get(Question, target_id) is not None
    if target_type == "brief":
        return can_view_brief(db.get(Brief, target_id), user)
    if target_type == "briefing":
        briefing = db.get(Briefing, target_id)
        return briefing is not None and user is not None and briefing.user_id == user.id
    if target_type == "alert":
        alert = db.get(Alert, target_id)
        return alert is not None and user is not None and alert.user_id == user.id
    content = db.get(GeneratedContent, target_id)
    if content is None:
        return False
    kind_ok = content.kind.startswith("product:") if target_type == "product" else content.kind == target_type
    brief = db.get(Brief, content.brief_id) if content.brief_id else None
    return kind_ok and (brief is None or can_view_brief(brief, user))


@router.post("/feedback", dependencies=[CSRF])
def submit_feedback(
    request: Request,
    db: DB,
    user: CurrentUser,
    target_type: Annotated[str, Form()],
    target_id: Annotated[str, Form()],
    answered_question: Annotated[str | None, Form()] = None,
    saved_time: Annotated[str | None, Form()] = None,
    rating: Annotated[str | None, Form()] = None,
    comment: Annotated[str | None, Form()] = None,
    next: Annotated[str | None, Form()] = None,
):
    if target_type not in FEEDBACK_TARGETS or not target_id.strip().isdigit():
        raise HTTPException(status_code=400, detail="Unknown feedback target.")
    target = int(target_id)
    if not _feedback_target_exists(db, target_type, target, user):
        raise HTTPException(status_code=404, detail="We couldn't find what you're rating.")
    stars: int | None = None
    if (rating or "").strip():
        if not rating.strip().isdigit() or not 1 <= int(rating) <= 5:
            raise HTTPException(status_code=400, detail="Ratings run from 1 to 5.")
        stars = int(rating)
    record_feedback(
        db,
        user,
        target_type,
        target,
        answered_question=_yes_no(answered_question),
        saved_time=_yes_no(saved_time),
        rating=stars,
        comment=(comment or "").strip()[:2000] or None,
    )
    if is_htmx(request):
        return _partial(request, "tools/_feedback_thanks.html", {})
    flash(request, "Thanks for your feedback. It helps us measure whether The Newsren is useful.", "success")
    return RedirectResponse(safe_next(next, "/"), status_code=303)
