"""Editorial workflow: submission, the human review queue, approval, corrections, and claim sampling.

Publication rules (plan §4 and §6):

* Every brief runs the automated quality gate (``run_checks``) on submission.
* Finance briefs always go to a human editor, whatever the settings
  ("human editorial review for any finance story before publication").
* High-impact claims, blockers, and warnings also route a brief to a human.
* A brief is published automatically only when it is not finance, its risk is
  low, no check blocks it, and ``settings.require_human_approval`` is off.
* An editor cannot approve a brief with failing blocker checks unless the
  review notes start with ``OVERRIDE:`` and explain why; the override is
  logged with the checks it overrode.

Every state change is logged as an ``EditorialAction`` (with review duration
for approvals and rejections) so the editorial-efficiency and trust metrics
can be computed. Functions flush but never commit.
"""

from __future__ import annotations

from sqlalchemy import case, select
from sqlalchemy.orm import Session

from newsren.config import Settings, get_settings
from newsren.editorial.checks import BLOCKER, WARNING, assess_risk, failed_checks, run_checks
from newsren.models import (
    Brief,
    BriefClaim,
    BriefStatus,
    Correction,
    EditorialAction,
    RiskLevel,
    User,
    Vertical,
    utcnow,
)
from newsren.schemas import CheckResult

OVERRIDE_PREFIX = "OVERRIDE:"
CLAIM_VERDICTS = frozenset({"accurate", "inaccurate"})


def _require_editor(user: User | None) -> User:
    if user is None or not user.is_editor:
        raise PermissionError("Only editors can review, approve, reject, or correct briefs.")
    return user


def _log(
    session: Session,
    brief: Brief,
    action: str,
    *,
    actor: User | None = None,
    notes: str | None = None,
    duration_seconds: float | None = None,
) -> EditorialAction:
    if brief.id is None:
        session.add(brief)
        session.flush()
    row = EditorialAction(
        brief_id=brief.id,
        actor_id=actor.id if actor is not None else None,
        action=action,
        notes=notes,
        duration_seconds=duration_seconds,
    )
    session.add(row)
    return row


def _is_finance(brief: Brief) -> bool:
    return str(brief.vertical or "").lower() == Vertical.FINANCE


def _store_checks(brief: Brief, checks: list[CheckResult]) -> None:
    brief.check_results = [c.model_dump() for c in checks]
    brief.risk_level = assess_risk(brief, checks)


def _review_seconds(brief: Brief) -> float | None:
    if brief.review_started_at is None:
        return None
    return max(0.0, (utcnow() - brief.review_started_at).total_seconds())


def review_reasons(brief: Brief, checks: list[CheckResult], settings: Settings) -> list[str]:
    """Why a brief needs a human editor (empty when it may be published automatically)."""
    reasons: list[str] = []
    if _is_finance(brief):
        reasons.append("finance story (always reviewed by a human)")
    if any(c.high_impact for c in brief.claims or []):
        reasons.append("contains high-impact claims")
    blockers = failed_checks(checks, BLOCKER)
    if blockers:
        reasons.append("blocking checks failed: " + ", ".join(c.name for c in blockers))
    warnings = failed_checks(checks, WARNING)
    if warnings:
        reasons.append("warnings: " + ", ".join(c.name for c in warnings))
    if settings.require_human_approval:
        reasons.append("human approval is required for publication")
    return reasons


def blocking_checks(brief: Brief) -> list[dict]:
    """The brief's stored check results that block approval."""
    return failed_checks(brief.check_results or [], BLOCKER)


def submit_for_review(session: Session, brief: Brief, settings: Settings | None = None) -> Brief:
    """Run the quality gate and route the brief: to the human review queue, or straight to publication.

    Sets ``check_results``, ``risk_level``, ``requires_human_review`` and the
    status (``in_review``, or ``approved`` with ``published_at`` when no human
    review is needed). Logs an ``auto_check`` action, plus ``submit`` when the
    brief joins the review queue. Resubmitting a brief clears its earlier
    review decision.
    """
    settings = settings or get_settings()
    if brief.id is None:
        session.add(brief)
        session.flush()
    checks = run_checks(session, brief)
    _store_checks(brief, checks)
    reasons = review_reasons(brief, checks, settings)
    blockers = failed_checks(checks, BLOCKER)
    auto_publish = (
        not _is_finance(brief)
        and brief.risk_level == RiskLevel.LOW
        and not blockers
        and not settings.require_human_approval
    )
    brief.requires_human_review = not auto_publish
    brief.reviewer_id = None
    brief.reviewed_at = None
    brief.review_started_at = None
    brief.review_notes = None

    passed = sum(1 for c in checks if c.passed)
    summary = (
        f"Automated checks: {passed}/{len(checks)} passed; {len(blockers)} blocker(s), "
        f"{len(failed_checks(checks, WARNING))} warning(s); risk {brief.risk_level}."
    )
    if auto_publish:
        brief.status = BriefStatus.APPROVED
        brief.published_at = utcnow()
        _log(session, brief, "auto_check", notes=f"{summary} Published automatically: low risk, no blockers.")
    else:
        brief.status = BriefStatus.IN_REVIEW
        brief.published_at = None
        _log(session, brief, "auto_check", notes=summary)
        _log(session, brief, "submit", notes="Sent to human review: " + "; ".join(reasons) + ".")
    session.flush()
    return brief


def start_review(session: Session, brief: Brief, editor: User) -> Brief:
    """Record that ``editor`` opened the brief for review (starts the review-time clock)."""
    _require_editor(editor)
    if brief.review_started_at is None or brief.reviewer_id != editor.id:
        brief.review_started_at = utcnow()
        brief.reviewer_id = editor.id
        _log(session, brief, "start_review", actor=editor)
    if brief.status == BriefStatus.DRAFT:
        brief.status = BriefStatus.IN_REVIEW
    session.flush()
    return brief


def approve(session: Session, brief: Brief, editor: User, notes: str = "") -> Brief:
    """Approve and publish a brief.

    Checks are re-run so the decision reflects the brief as it stands now.
    Raises ``ValueError`` when blocker checks fail, unless ``notes`` starts
    with ``OVERRIDE:`` followed by the editor's justification. Records the
    reviewer, review time (from ``review_started_at``) and notes.
    """
    _require_editor(editor)
    if brief.status == BriefStatus.APPROVED and brief.published_at is not None:
        return brief
    checks = run_checks(session, brief)
    _store_checks(brief, checks)
    blockers = failed_checks(checks, BLOCKER)
    notes = (notes or "").strip()
    action_notes = notes or None
    if blockers:
        names = ", ".join(c.name for c in blockers)
        is_override = notes.upper().startswith(OVERRIDE_PREFIX)
        justification = notes[len(OVERRIDE_PREFIX) :].strip() if is_override else ""
        if not justification:
            raise ValueError(
                f"Brief {brief.id} has {len(blockers)} blocking check(s) ({names}). Fix them, or approve with "
                f"notes starting '{OVERRIDE_PREFIX}' followed by the reason for overriding."
            )
        action_notes = f"{notes} [overrode: {names}]"

    now = utcnow()
    duration = _review_seconds(brief)
    brief.status = BriefStatus.APPROVED
    brief.published_at = now
    brief.reviewer_id = editor.id
    brief.reviewed_at = now
    brief.review_notes = notes or None
    _log(session, brief, "approve", actor=editor, notes=action_notes, duration_seconds=duration)
    session.flush()
    return brief


def reject(session: Session, brief: Brief, editor: User, notes: str) -> Brief:
    """Reject a brief (it leaves the queue and is not published); ``notes`` must say why."""
    _require_editor(editor)
    notes = (notes or "").strip()
    if not notes:
        raise ValueError("Explain why the brief is rejected.")
    now = utcnow()
    duration = _review_seconds(brief)
    brief.status = BriefStatus.REJECTED
    brief.published_at = None
    brief.reviewer_id = editor.id
    brief.reviewed_at = now
    brief.review_notes = notes
    _log(session, brief, "reject", actor=editor, notes=notes, duration_seconds=duration)
    session.flush()
    return brief


def record_correction(
    session: Session, brief: Brief, editor: User, description: str, material: bool = False
) -> Correction:
    """Record a correction to a brief (shown to readers) and bump its version.

    ``material`` corrections change a fact a reader could have relied on; they
    feed the "material corrections per 100 published briefs" trust metric.
    """
    _require_editor(editor)
    description = (description or "").strip()
    if not description:
        raise ValueError("Describe the correction.")
    if brief.id is None:
        session.add(brief)
        session.flush()
    correction = Correction(
        brief_id=brief.id, description=description, material=bool(material), actor_id=editor.id
    )
    session.add(correction)
    brief.version = (brief.version or 1) + 1
    kind = "Material" if material else "Minor"
    _log(session, brief, "correct", actor=editor, notes=f"{kind} correction (v{brief.version}): {description}")
    session.flush()
    return correction


def set_claim_verdict(session: Session, claim: BriefClaim, editor: User, verdict: str) -> BriefClaim:
    """Record an editor's judgement of whether a claim's linked sources support it (trust metric sample)."""
    _require_editor(editor)
    verdict = (verdict or "").strip().lower()
    if verdict not in CLAIM_VERDICTS:
        raise ValueError(f"Verdict must be one of {', '.join(sorted(CLAIM_VERDICTS))}, not {verdict!r}.")
    claim.editor_verdict = verdict
    claim.verdict_at = utcnow()
    brief = claim.brief if claim.brief is not None else session.get(Brief, claim.brief_id)
    if brief is None:
        raise ValueError("The claim does not belong to a stored brief.")
    session.flush()
    _log(
        session,
        brief,
        "claim_verdict",
        actor=editor,
        notes=f"Claim {claim.id} judged {verdict}: {claim.text[:200]}",
    )
    session.flush()
    return claim


def review_queue(session: Session) -> list[Brief]:
    """Briefs awaiting a human decision (in review or draft): high risk first, then oldest first."""
    high_first = case((Brief.risk_level == RiskLevel.HIGH.value, 0), else_=1)
    return list(
        session.scalars(
            select(Brief)
            .where(Brief.status.in_([BriefStatus.IN_REVIEW.value, BriefStatus.DRAFT.value]))
            .order_by(high_first, Brief.generated_at.asc(), Brief.id.asc())
        )
    )


def published_briefs(session: Session, vertical: str | None = None, limit: int = 30) -> list[Brief]:
    """Approved (published) briefs, newest first, optionally for one vertical."""
    query = select(Brief).where(Brief.status == BriefStatus.APPROVED.value)
    if vertical:
        query = query.where(Brief.vertical == str(vertical))
    query = query.order_by(Brief.published_at.desc().nulls_last(), Brief.id.desc()).limit(limit)
    return list(session.scalars(query))


def editorial_history(session: Session, brief: Brief) -> list[EditorialAction]:
    """Every editorial action on a brief, oldest first (audit trail for the review page)."""
    return list(
        session.scalars(
            select(EditorialAction)
            .where(EditorialAction.brief_id == brief.id)
            .order_by(EditorialAction.created_at.asc(), EditorialAction.id.asc())
        )
    )
