"""Editorial workflow: routing, human review, approval with overrides, corrections, and claim verdicts."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from newsren.agents.base import citation_for
from newsren.editorial.workflow import (
    approve,
    blocking_checks,
    editorial_history,
    published_briefs,
    record_correction,
    reject,
    review_queue,
    set_claim_verdict,
    start_review,
    submit_for_review,
)
from newsren.models import Brief, BriefClaim, Correction, EditorialAction, utcnow

DATA = Path(__file__).parent / "data" / "editorial" / "newsroom.json"
LICENSE = "Headline + publisher summary + link; full text not stored."

TECH = {
    "keys": ["vendor_launch", "independent_eval"],
    "fields": {
        "headline": "Lumen-4 launch: independent test finds a lower score than the company's figure",
        "summary": (
            "Lumen Labs released Lumen-4 and says it solves 82% of competition problems. An independent "
            "evaluation of 1,200 held-out problems found it solved 61%."
        ),
        "why_it_matters": "Schools weighing AI math tutors need independent evidence, not only vendor figures.",
    },
    "claims": [
        {"text": "Lumen Labs released the Lumen-4 math tutoring model on Monday.", "citation_ids": ["c1"]},
        {"text": "Lumen Labs says Lumen-4 solves 82% of competition math problems.", "citation_ids": ["c1"]},
        {
            "text": "An independent evaluation of 1,200 held-out problems found Lumen-4 solved 61% of them.",
            "citation_ids": ["c2"],
        },
    ],
}

FINANCE = {
    "keys": ["bank_note", "finance_news"],
    "fields": {
        "headline": "Harbor Bank's AI agents beat a 60/40 portfolio in a backtest, not in live trading",
        "summary": "Harbor Bank reported a 20-year backtest; it has not reported live results for the agents.",
        "why_it_matters": "A backtest shows how a strategy would have done on past data.",
    },
    "claims": [
        {
            "text": (
                "Harbor Bank researchers reported that AI portfolio agents beat a 60/40 portfolio by 1.8 "
                "percentage points a year in a 2005-2025 backtest."
            ),
            "claim_type": "performance",
            "performance_label": "backtest",
            "performance_period": "2005-2025",
            "benchmark": "60/40 stock/bond portfolio",
            "citation_ids": ["c1"],
        },
        {"text": "Harbor Bank has not reported live results for the agents.", "citation_ids": ["c2"]},
    ],
}


class Desk:
    """Builds briefs from the fictional newsroom sample."""

    def __init__(self, factory):
        self.factory = factory
        self.session = factory.session
        self.specs = json.loads(DATA.read_text())["articles"]
        self.sources: dict[str, Any] = {}
        self.articles: dict[str, Any] = {}

    def _article(self, key: str):
        if key not in self.articles:
            spec = self.specs[key]
            if spec["publisher"] not in self.sources:
                src = self.factory.source(
                    vertical=spec["vertical"],
                    trust_tier=spec["trust_tier"],
                    doc_type=spec["doc_type"],
                    name=spec["publisher"],
                )
                src.license_note = LICENSE
                self.sources[spec["publisher"]] = src
            self.articles[key] = self.factory.article(
                spec["title"],
                spec["body"],
                source=self.sources[spec["publisher"]],
                vertical=spec["vertical"],
                doc_type=spec["doc_type"],
                trust_tier=spec["trust_tier"],
            )
        return self.articles[key]

    def brief(
        self, spec: dict[str, Any], *, vertical: str = "technology", extra_claims=(), **fields: Any
    ) -> Brief:
        citations = [citation_for(self._article(k), f"c{i}").model_dump() for i, k in enumerate(spec["keys"], 1)]
        brief = Brief(vertical=vertical, citations=citations, **{**spec["fields"], **fields})
        brief.claims = [
            BriefClaim(position=i, **{"support_status": "supported", **c})
            for i, c in enumerate([*spec["claims"], *extra_claims])
        ]
        self.session.add(brief)
        self.session.flush()
        return brief


@pytest.fixture()
def desk(factory) -> Desk:
    return Desk(factory)


@pytest.fixture()
def editor(factory):
    return factory.user(role="editor", display_name="Desk Editor")


@pytest.fixture()
def auto_settings(settings):
    return settings.with_overrides(require_human_approval=False)


def actions(session, brief: Brief) -> list[str]:
    return [a.action for a in editorial_history(session, brief)]


UNCITED_FACT = {"text": "Lumen-4 is already used in 300 schools.", "citation_ids": []}


# --------------------------------------------------------------------------- routing


def test_low_risk_brief_auto_publishes_when_human_approval_is_off(session, desk, auto_settings) -> None:
    brief = submit_for_review(session, desk.brief(TECH), auto_settings)
    assert brief.status == "approved" and brief.published_at is not None
    assert brief.risk_level == "low" and brief.requires_human_review is False
    assert len(brief.check_results) == 20 and all(c["passed"] for c in brief.check_results)
    assert actions(session, brief) == ["auto_check"]
    history = editorial_history(session, brief)
    assert "Published automatically" in history[0].notes and history[0].actor_id is None
    assert published_briefs(session) == [brief]


def test_human_approval_setting_routes_low_risk_brief_to_queue(session, desk, settings) -> None:
    assert settings.require_human_approval is True
    brief = submit_for_review(session, desk.brief(TECH))  # default settings from get_settings()
    assert brief.status == "in_review" and brief.published_at is None
    assert brief.requires_human_review is True and brief.risk_level == "low"
    assert actions(session, brief) == ["auto_check", "submit"]
    assert "human approval is required" in editorial_history(session, brief)[1].notes
    assert review_queue(session) == [brief]


def test_finance_brief_always_requires_human_review(session, desk, auto_settings) -> None:
    brief = submit_for_review(session, desk.brief(FINANCE, vertical="finance"), auto_settings)
    assert [c["name"] for c in brief.check_results if not c["passed"]] == []
    assert brief.status == "in_review" and brief.published_at is None
    assert brief.requires_human_review is True and brief.risk_level == "high"
    assert "finance story" in editorial_history(session, brief)[-1].notes


def test_warnings_route_to_human_review(session, desk, auto_settings) -> None:
    forecast = {
        "text": "Lumen Labs will release Lumen-5 next year.",
        "claim_type": "forecast",
        "citation_ids": ["c1"],
    }
    brief = submit_for_review(session, desk.brief(TECH, extra_claims=[forecast]), auto_settings)
    assert brief.risk_level == "medium" and brief.status == "in_review"
    assert "forecast_labelling" in editorial_history(session, brief)[-1].notes


def test_high_impact_claim_routes_to_human_review(session, desk, auto_settings) -> None:
    claim = {**TECH["claims"][2], "high_impact": True}
    brief = submit_for_review(session, desk.brief(TECH, extra_claims=[claim]), auto_settings)
    assert brief.risk_level == "high" and brief.status == "in_review"


def test_blockers_prevent_auto_publication(session, desk, auto_settings) -> None:
    brief = submit_for_review(session, desk.brief(TECH, extra_claims=[UNCITED_FACT]), auto_settings)
    assert brief.status == "in_review" and brief.risk_level == "high"
    assert [c["name"] for c in blocking_checks(brief)] == ["citation_coverage"]


# --------------------------------------------------------------------------- approval


def test_approve_records_reviewer_notes_and_review_duration(session, desk, editor) -> None:
    brief = submit_for_review(session, desk.brief(TECH))
    start_review(session, brief, editor)
    brief.review_started_at = utcnow() - timedelta(seconds=90)
    approve(session, brief, editor, notes="Checked both sources.")
    assert brief.status == "approved" and brief.published_at is not None
    assert brief.reviewer_id == editor.id and brief.reviewed_at is not None
    assert brief.review_notes == "Checked both sources."
    approval = editorial_history(session, brief)[-1]
    assert approval.action == "approve" and approval.actor_id == editor.id
    assert 89 <= approval.duration_seconds < 120
    assert actions(session, brief) == ["auto_check", "submit", "start_review", "approve"]
    assert review_queue(session) == []


def test_approve_without_start_review_has_no_duration(session, desk, editor) -> None:
    brief = approve(session, submit_for_review(session, desk.brief(TECH)), editor)
    assert brief.status == "approved"
    assert editorial_history(session, brief)[-1].duration_seconds is None


def test_approve_refuses_blockers_without_override(session, desk, editor) -> None:
    brief = submit_for_review(session, desk.brief(TECH, extra_claims=[UNCITED_FACT]))
    with pytest.raises(ValueError, match="citation_coverage"):
        approve(session, brief, editor, notes="Looks fine.")
    with pytest.raises(ValueError, match="OVERRIDE"):
        approve(session, brief, editor, notes="OVERRIDE:")
    assert brief.status == "in_review" and brief.published_at is None


def test_approve_with_override_publishes_and_logs_what_was_overridden(session, desk, editor) -> None:
    brief = submit_for_review(session, desk.brief(TECH, extra_claims=[UNCITED_FACT]))
    approve(session, brief, editor, notes="OVERRIDE: confirmed with the district by phone; source to follow.")
    assert brief.status == "approved"
    assert brief.review_notes.startswith("OVERRIDE:")
    last = editorial_history(session, brief)[-1]
    assert last.action == "approve" and "[overrode: citation_coverage]" in last.notes


def test_approve_rechecks_the_brief_as_it_stands(session, desk, editor) -> None:
    brief = submit_for_review(session, desk.brief(TECH))
    brief.claims.append(BriefClaim(position=9, text="Lumen-4 is used in 300 schools.", support_status="supported"))
    session.flush()
    with pytest.raises(ValueError):
        approve(session, brief, editor)
    assert any(c["name"] == "citation_coverage" and not c["passed"] for c in brief.check_results)


def test_finance_brief_is_published_only_by_an_editor(session, desk, editor, factory, auto_settings) -> None:
    brief = submit_for_review(session, desk.brief(FINANCE, vertical="finance"), auto_settings)
    reader = factory.user(role="reader")
    with pytest.raises(PermissionError):
        approve(session, brief, reader)
    approve(session, brief, editor, notes="Backtest label, period, and benchmark verified against the note.")
    assert brief.status == "approved" and published_briefs(session, vertical="finance") == [brief]


def test_reject_requires_reason_and_unpublishes(session, desk, editor) -> None:
    brief = submit_for_review(session, desk.brief(TECH))
    with pytest.raises(ValueError):
        reject(session, brief, editor, "  ")
    start_review(session, brief, editor)
    reject(session, brief, editor, "The independent evaluation is not about the same model version.")
    assert brief.status == "rejected" and brief.published_at is None and brief.reviewer_id == editor.id
    last = editorial_history(session, brief)[-1]
    assert last.action == "reject" and last.duration_seconds is not None
    assert review_queue(session) == []


def test_resubmitting_a_rejected_brief_returns_it_to_the_queue(session, desk, editor) -> None:
    brief = submit_for_review(session, desk.brief(TECH))
    reject(session, brief, editor, "Needs a second independent source.")
    submit_for_review(session, brief)
    assert brief.status == "in_review" and brief.reviewer_id is None and brief.review_notes is None


def test_start_review_moves_draft_into_review(session, desk, editor) -> None:
    brief = desk.brief(TECH)
    assert brief.status == "draft"
    start_review(session, brief, editor)
    assert brief.status == "in_review" and brief.review_started_at is not None and brief.reviewer_id == editor.id


# --------------------------------------------------------------------------- corrections and verdicts


def test_record_correction_bumps_version_and_logs(session, desk, editor) -> None:
    brief = approve(session, submit_for_review(session, desk.brief(TECH)), editor)
    correction = record_correction(
        session, brief, editor, "The evaluation covered 1,200 problems, not 1,000.", material=True
    )
    assert isinstance(correction, Correction) and correction.material is True
    assert correction.brief_id == brief.id and correction.actor_id == editor.id
    assert brief.version == 2 and brief.status == "approved"
    last = editorial_history(session, brief)[-1]
    assert last.action == "correct" and last.notes.startswith("Material correction (v2)")
    with pytest.raises(ValueError):
        record_correction(session, brief, editor, "")


def test_set_claim_verdict(session, desk, editor) -> None:
    brief = submit_for_review(session, desk.brief(TECH))
    claim = brief.claims[2]
    set_claim_verdict(session, claim, editor, "Accurate")
    assert claim.editor_verdict == "accurate" and claim.verdict_at is not None
    logged = session.scalars(select(EditorialAction).where(EditorialAction.action == "claim_verdict")).one()
    assert logged.brief_id == brief.id and logged.actor_id == editor.id and str(claim.id) in logged.notes
    with pytest.raises(ValueError):
        set_claim_verdict(session, claim, editor, "mostly true")


# --------------------------------------------------------------------------- queue and front page


def test_review_queue_puts_high_risk_first_then_oldest(session, desk, editor) -> None:
    now = utcnow()
    oldest_low = submit_for_review(session, desk.brief(TECH, generated_at=now - timedelta(hours=5)))
    forecast = {
        "text": "Lumen Labs will release Lumen-5 next year.",
        "claim_type": "forecast",
        "citation_ids": ["c1"],
    }
    medium = submit_for_review(
        session, desk.brief(TECH, extra_claims=[forecast], generated_at=now - timedelta(hours=4))
    )
    newest_high = submit_for_review(
        session, desk.brief(FINANCE, vertical="finance", generated_at=now - timedelta(hours=1))
    )
    older_high = submit_for_review(
        session, desk.brief(TECH, extra_claims=[UNCITED_FACT], generated_at=now - timedelta(hours=2))
    )
    draft = desk.brief(TECH, generated_at=now - timedelta(hours=3))
    published = approve(
        session, submit_for_review(session, desk.brief(TECH, generated_at=now - timedelta(hours=9))), editor
    )
    assert [medium.risk_level, newest_high.risk_level, older_high.risk_level] == ["medium", "high", "high"]

    queue = review_queue(session)
    assert queue == [older_high, newest_high, oldest_low, medium, draft]
    assert published not in queue


def test_published_briefs_newest_first_with_vertical_filter_and_limit(
    session, desk, editor, auto_settings
) -> None:
    first = submit_for_review(session, desk.brief(TECH), auto_settings)
    first.published_at = utcnow() - timedelta(hours=3)
    second = submit_for_review(session, desk.brief(TECH), auto_settings)
    finance = approve(session, submit_for_review(session, desk.brief(FINANCE, vertical="finance")), editor)
    in_review = submit_for_review(session, desk.brief(TECH))
    session.flush()

    assert published_briefs(session) == [finance, second, first]
    assert published_briefs(session, vertical="technology") == [second, first]
    assert published_briefs(session, vertical="finance", limit=1) == [finance]
    assert in_review not in published_briefs(session)
