"""Editor desk (plan §6 editorial quality gate): review queue, review page, decisions, sources, runs, pipeline."""

from __future__ import annotations

import re

import pytest
from markupsafe import escape
from sqlalchemy import func, select

from conftest import _csrf, login
from newsren.editorial.guardrails import FINANCE_DISCLAIMER
from newsren.editorial.workflow import published_briefs
from newsren.models import (
    AgentRun,
    Brief,
    BriefClaim,
    BriefStatus,
    ClaimType,
    Correction,
    EditorialAction,
    FundingRound,
    Source,
    SupportStatus,
    User,
)

HX = {"HX-Request": "true"}
EDITOR_EMAIL = "editor@newsren.example"
DISCLAIMER_HTML = str(escape(FINANCE_DISCLAIMER))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _fresh(session) -> None:
    """End the test session's snapshot so reads see what the app committed (and SQLite isn't locked)."""
    session.rollback()
    session.expire_all()


def _get(session, model, ident):
    _fresh(session)
    return session.get(model, ident)


def _in_review(session, vertical: str | None = None) -> Brief:
    _fresh(session)
    query = select(Brief).where(Brief.status == BriefStatus.IN_REVIEW).order_by(Brief.id)
    if vertical == "finance":
        query = query.where(Brief.vertical == "finance")
    elif vertical == "other":
        query = query.where(Brief.vertical != "finance")
    brief = session.scalars(query).first()
    assert brief is not None, "demo database should have briefs in review"
    return brief


def _approved(session) -> Brief:
    _fresh(session)
    brief = session.scalars(select(Brief).where(Brief.status == BriefStatus.APPROVED).order_by(Brief.id)).first()
    assert brief is not None
    return brief


def _editor(session) -> User:
    _fresh(session)
    return session.scalars(select(User).where(User.email == EDITOR_EMAIL)).one()


def _flash(client) -> str:
    """The flash message shown on the next page (consumes it)."""
    html = client.get("/about").text
    match = re.search(r'<div class="flash flash-(\w+)" role="status">(.*?)</div>', html, re.S)
    return f"{match.group(1)}: {match.group(2).strip()}" if match else ""


def _actions(session, brief_id: int, action: str) -> list[EditorialAction]:
    _fresh(session)
    return list(
        session.scalars(
            select(EditorialAction).where(EditorialAction.brief_id == brief_id, EditorialAction.action == action)
        )
    )


def _add_blocker(session, brief_id: int) -> None:
    """Give a brief a claim citing a source id it does not have: the unknown_citation_ids blocker fails."""
    _fresh(session)
    brief = session.get(Brief, brief_id)
    brief.claims.append(
        BriefClaim(
            position=len(brief.claims),
            text="An unsourced figure slipped into the draft.",
            claim_type=ClaimType.FACT,
            support_status=SupportStatus.SUPPORTED,
            citation_ids=["c99"],
        )
    )
    session.commit()


def _post(client, path: str, data: dict | None = None, **kwargs):
    return client.post(path, data={**(data or {}), "csrf_token": client.csrf}, follow_redirects=False, **kwargs)


# --------------------------------------------------------------------------
# Permissions
# --------------------------------------------------------------------------

GET_PATHS = [
    "/editor",
    "/editor?vertical=finance",
    "/editor/briefs/{brief}",
    "/editor/sources",
    "/editor/runs",
    "/editor/runs/{run}",
    "/editor/pipeline",
    "/editor/funding",
]


def _resolve(path: str, session) -> str:
    _fresh(session)
    brief = session.scalars(select(Brief).order_by(Brief.id)).first()
    run = session.scalars(select(AgentRun).order_by(AgentRun.id)).first()
    return path.format(brief=brief.id, run=run.id)


@pytest.mark.parametrize("path", GET_PATHS)
def test_editor_pages_redirect_anonymous_to_login(client, demo_session, path):
    target = _resolve(path, demo_session)
    resp = client.get(target, follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/login?next=/editor")


@pytest.mark.parametrize("path", GET_PATHS)
def test_editor_pages_forbidden_for_readers(reader_client, demo_session, path):
    resp = reader_client.get(_resolve(path, demo_session), follow_redirects=False)
    assert resp.status_code == 403


@pytest.mark.parametrize("path", GET_PATHS)
def test_editor_pages_open_for_editors(editor_client, demo_session, path):
    resp = editor_client.get(_resolve(path, demo_session))
    assert resp.status_code == 200
    assert 'aria-label="Editor desk"' in resp.text
    assert 'href="/static/css/editor.css"' in resp.text


def test_editor_posts_reject_readers_and_anonymous(client, demo_session):
    brief = _in_review(demo_session)
    token = _csrf(client)
    anon = client.post(
        f"/editor/briefs/{brief.id}/approve", data={"csrf_token": token, "notes": ""}, follow_redirects=False
    )
    assert anon.status_code == 303 and anon.headers["location"].startswith("/login")
    client.csrf = login(client, "reader@newsren.example", "reader-demo-pass")
    for path in (
        f"/editor/briefs/{brief.id}/approve",
        f"/editor/briefs/{brief.id}/reject",
        "/editor/pipeline",
        f"/editor/claims/{brief.claims[0].id}/verdict",
    ):
        assert _post(client, path, {"notes": "x", "verdict": "accurate"}).status_code == 403
    assert _get(demo_session, Brief, brief.id).status == BriefStatus.IN_REVIEW


@pytest.mark.parametrize(
    "path,data",
    [
        ("/editor/briefs/{brief}/approve", {"notes": ""}),
        ("/editor/briefs/{brief}/reject", {"notes": "No"}),
        ("/editor/briefs/{brief}/corrections", {"description": "x"}),
        ("/editor/briefs/{brief}/regenerate", {}),
        ("/editor/claims/{claim}/verdict", {"verdict": "accurate"}),
        ("/editor/sources/{source}/toggle", {"field": "approved"}),
        ("/editor/pipeline", {"max_briefs": "1"}),
        ("/editor/funding/{round}/verify", {}),
    ],
)
def test_editor_posts_require_csrf(editor_client, demo_session, path, data):
    brief = _in_review(demo_session)
    source = demo_session.scalars(select(Source)).first()
    funding = demo_session.scalars(select(FundingRound)).first()
    target = path.format(brief=brief.id, claim=brief.claims[0].id, source=source.id, round=funding.id)
    missing = editor_client.post(target, data=data, follow_redirects=False)
    assert missing.status_code == 403
    wrong = editor_client.post(target, data={**data, "csrf_token": "nope"}, follow_redirects=False)
    assert wrong.status_code == 403
    assert _get(demo_session, Brief, brief.id).status == BriefStatus.IN_REVIEW
    assert _get(demo_session, Source, source.id).approved is True


# --------------------------------------------------------------------------
# Review queue
# --------------------------------------------------------------------------


def test_queue_lists_in_review_briefs_including_finance(editor_client, demo_session):
    _fresh(demo_session)
    queued = demo_session.scalars(select(Brief).where(Brief.status == BriefStatus.IN_REVIEW)).all()
    assert any(b.vertical == "finance" for b in queued)
    html = editor_client.get("/editor").text
    for brief in queued:
        assert f'href="/editor/briefs/{brief.id}"' in html
        assert str(escape(brief.headline)) in html
    assert "finance story (always reviewed by a human)" in html
    assert "High risk" in html
    assert "checks passed" in html
    assert "Needs a human because:" in html
    # The gate is explained, including the finance rule and the override prefix.
    assert "How the quality gate works" in html
    assert "OVERRIDE:" in html
    assert "Recently published" in html
    _fresh(demo_session)
    latest = published_briefs(demo_session, limit=1)[0]
    assert f'href="/briefs/{latest.id}"' in html


def test_queue_filters_by_vertical(editor_client, demo_session):
    finance = _in_review(demo_session, "finance")
    other = _in_review(demo_session, "other")
    html = editor_client.get("/editor?vertical=finance").text
    assert DISCLAIMER_HTML in html
    assert f'href="/editor/briefs/{finance.id}"' in html
    assert f'href="/editor/briefs/{other.id}"' not in html
    # Unknown verticals fall back to the full queue.
    assert f'href="/editor/briefs/{other.id}"' in editor_client.get("/editor?vertical=sports").text


def test_queue_shows_blocker_counts(editor_client, demo_session):
    brief = _in_review(demo_session, "other")
    _fresh(demo_session)
    stored = demo_session.get(Brief, brief.id)
    stored.check_results = [
        *stored.check_results,
        {"name": "approved_sources", "passed": False, "severity": "blocker", "message": "x", "details": []},
    ]
    demo_session.commit()
    html = editor_client.get("/editor").text
    assert "1 blocker" in html
    assert "Blocked by checks" in html


# --------------------------------------------------------------------------
# Review page
# --------------------------------------------------------------------------


def test_review_page_starts_review_once(editor_client, demo_session):
    brief = _in_review(demo_session, "other")
    assert brief.review_started_at is None
    resp = editor_client.get(f"/editor/briefs/{brief.id}")
    assert resp.status_code == 200
    stored = _get(demo_session, Brief, brief.id)
    assert stored.review_started_at is not None
    assert stored.reviewer_id == _editor(demo_session).id
    started = stored.review_started_at
    editor_client.get(f"/editor/briefs/{brief.id}")
    assert _get(demo_session, Brief, brief.id).review_started_at == started
    assert len(_actions(demo_session, brief.id, "start_review")) == 1


def test_review_page_shows_brief_checks_claims_and_forms(editor_client, demo_session):
    brief = _in_review(demo_session, "other")
    html = editor_client.get(f"/editor/briefs/{brief.id}").text
    assert "Unpublished draft — in editorial review" in html
    assert str(escape(brief.headline)) in html
    assert "Automated checks" in html and "check-pass" in html
    assert "Why a human must review this" in html
    # Claims with verdict buttons and their sources (opening in a new tab).
    for claim in brief.claims:
        assert f'action="/editor/claims/{claim.id}/verdict"' in html
    assert 'value="accurate"' in html and 'value="inaccurate"' in html
    assert 'target="_blank" rel="noopener noreferrer"' in html
    assert "retrieved" in html
    # Decision forms, provenance, agent run and audit trail.
    assert f'action="/editor/briefs/{brief.id}/approve"' in html
    assert f'action="/editor/briefs/{brief.id}/reject"' in html
    assert f'action="/editor/briefs/{brief.id}/regenerate"' in html
    assert "corrections" not in re.findall(r'action="(/editor/briefs/\d+/\w+)"', html)
    assert 'name="csrf_token"' in html
    assert "Agent run" in html and "story_brief" in html
    assert "Editorial history" in html and "Auto check" in html
    assert "Drafted by" in html  # generated_note provenance


def test_review_page_for_finance_brief_has_notice(editor_client, demo_session):
    brief = _in_review(demo_session, "finance")
    html = editor_client.get(f"/editor/briefs/{brief.id}").text
    assert DISCLAIMER_HTML in html
    assert "finance story (always reviewed by a human)" in html


def test_review_page_404_for_missing_brief(editor_client):
    assert editor_client.get("/editor/briefs/999999").status_code == 404


def test_review_page_escapes_generated_text(editor_client, demo_session):
    brief = _in_review(demo_session, "other")
    _fresh(demo_session)
    stored = demo_session.get(Brief, brief.id)
    stored.headline = "<script>alert('x')</script> headline"
    stored.summary = "A <b>bold</b> claim [c1]."
    demo_session.commit()
    html = editor_client.get(f"/editor/briefs/{brief.id}").text
    assert "<script>alert('x')</script>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;b&gt;bold&lt;/b&gt;" in html
    assert 'class="cite" href="#cite-c1"' in html


# --------------------------------------------------------------------------
# Claim verdicts (Trust metric sample)
# --------------------------------------------------------------------------


def test_claim_verdict_is_stored_and_logged(editor_client, demo_session):
    brief = _in_review(demo_session)
    claim_id = brief.claims[0].id
    resp = _post(editor_client, f"/editor/claims/{claim_id}/verdict", {"verdict": "accurate"})
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/editor/briefs/{brief.id}#claim-{claim_id}"
    claim = _get(demo_session, BriefClaim, claim_id)
    assert claim.editor_verdict == "accurate" and claim.verdict_at is not None
    assert len(_actions(demo_session, brief.id, "claim_verdict")) == 1
    assert "Trust metric" in _flash(editor_client)
    html = editor_client.get(f"/editor/briefs/{brief.id}").text
    assert "Judged accurate" in html


def test_claim_verdict_htmx_returns_fragment(editor_client, demo_session):
    brief = _in_review(demo_session)
    claim_id = brief.claims[0].id
    resp = _post(editor_client, f"/editor/claims/{claim_id}/verdict", {"verdict": "inaccurate"}, headers=HX)
    assert resp.status_code == 200
    assert resp.text.lstrip().startswith('<li class="claim ed-claim')
    assert f'id="claim-{claim_id}"' in resp.text
    assert "Judged inaccurate" in resp.text
    assert "<html" not in resp.text
    assert _get(demo_session, BriefClaim, claim_id).editor_verdict == "inaccurate"


def test_claim_verdict_rejects_unknown_values(editor_client, demo_session):
    brief = _in_review(demo_session)
    claim_id = brief.claims[0].id
    resp = _post(editor_client, f"/editor/claims/{claim_id}/verdict", {"verdict": "maybe"})
    assert resp.status_code == 303
    assert _flash(editor_client).startswith("error:")
    assert _get(demo_session, BriefClaim, claim_id).editor_verdict is None
    htmx = _post(editor_client, f"/editor/claims/{claim_id}/verdict", {"verdict": ""}, headers=HX)
    assert htmx.status_code == 200 and "Verdict must be one of" in htmx.text and 'role="alert"' in htmx.text
    assert _post(editor_client, "/editor/claims/999999/verdict", {"verdict": "accurate"}).status_code == 404


# --------------------------------------------------------------------------
# Approve / reject / correct / regenerate
# --------------------------------------------------------------------------


def test_approve_publishes_brief_for_anonymous_readers(editor_client, demo_session):
    brief = _in_review(demo_session, "finance")
    editor_client.cookies.clear()
    assert editor_client.get(f"/briefs/{brief.id}").status_code == 404
    editor_client.csrf = login(editor_client, EDITOR_EMAIL, "editor-demo-pass")
    editor_client.get(f"/editor/briefs/{brief.id}")  # opens the review (starts the clock)
    resp = _post(editor_client, f"/editor/briefs/{brief.id}/approve", {"notes": "Checked every figure."})
    assert resp.status_code == 303 and resp.headers["location"] == "/editor"
    assert "Approved and published" in _flash(editor_client)
    stored = _get(demo_session, Brief, brief.id)
    assert stored.status == BriefStatus.APPROVED and stored.published_at is not None
    assert stored.reviewer_id == _editor(demo_session).id
    approvals = _actions(demo_session, brief.id, "approve")
    assert len(approvals) == 1 and approvals[0].duration_seconds is not None
    editor_client.cookies.clear()
    public = editor_client.get(f"/briefs/{brief.id}")
    assert public.status_code == 200
    assert str(escape(stored.headline)) in public.text


def test_approve_with_blockers_requires_override(editor_client, demo_session):
    brief = _in_review(demo_session, "other")
    _add_blocker(demo_session, brief.id)
    page = editor_client.get(f"/editor/briefs/{brief.id}").text
    assert "blocking check" in page and "Unknown citation ids" in page
    assert "Override and publish" in page
    assert "does not match any source in this brief" in page

    refused = _post(editor_client, f"/editor/briefs/{brief.id}/approve", {"notes": "Looks fine to me."})
    assert refused.status_code == 303
    assert refused.headers["location"] == f"/editor/briefs/{brief.id}#decision"
    message = _flash(editor_client)
    assert message.startswith("error:") and "OVERRIDE:" in message and "unknown_citation_ids" in message
    assert _get(demo_session, Brief, brief.id).status == BriefStatus.IN_REVIEW

    bare = _post(editor_client, f"/editor/briefs/{brief.id}/approve", {"notes": "OVERRIDE:"})
    assert bare.status_code == 303
    assert _get(demo_session, Brief, brief.id).status == BriefStatus.IN_REVIEW

    ok = _post(
        editor_client,
        f"/editor/briefs/{brief.id}/approve",
        {"notes": "OVERRIDE: the figure was confirmed with the original publisher by phone."},
    )
    assert ok.status_code == 303 and ok.headers["location"] == "/editor"
    assert _get(demo_session, Brief, brief.id).status == BriefStatus.APPROVED
    (action,) = _actions(demo_session, brief.id, "approve")
    assert "overrode: unknown_citation_ids" in action.notes


def test_reject_requires_notes_and_unpublishes(editor_client, demo_session):
    brief = _in_review(demo_session, "other")
    empty = _post(editor_client, f"/editor/briefs/{brief.id}/reject", {"notes": "   "})
    assert empty.status_code == 303 and empty.headers["location"].endswith("#decision")
    assert _flash(editor_client).startswith("error:")
    assert _get(demo_session, Brief, brief.id).status == BriefStatus.IN_REVIEW

    resp = _post(
        editor_client, f"/editor/briefs/{brief.id}/reject", {"notes": "Overstates what the study measured."}
    )
    assert resp.status_code == 303 and resp.headers["location"] == "/editor"
    stored = _get(demo_session, Brief, brief.id)
    assert stored.status == BriefStatus.REJECTED
    assert stored.review_notes == "Overstates what the study measured."
    queue = editor_client.get("/editor").text
    assert "Recently rejected" in queue and "Overstates what the study measured." in queue
    review = editor_client.get(f"/editor/briefs/{brief.id}").text
    assert "Rejected — not published." in review
    # A rejected draft cannot be approved directly.
    _post(editor_client, f"/editor/briefs/{brief.id}/approve", {"notes": "changed my mind"})
    assert _get(demo_session, Brief, brief.id).status == BriefStatus.REJECTED
    editor_client.cookies.clear()
    assert editor_client.get(f"/briefs/{brief.id}").status_code == 404


def test_published_brief_cannot_be_rejected(editor_client, demo_session):
    brief = _approved(demo_session)
    _post(editor_client, f"/editor/briefs/{brief.id}/reject", {"notes": "no"})
    assert _get(demo_session, Brief, brief.id).status == BriefStatus.APPROVED
    assert "corrected, not rejected" in _flash(editor_client)


def test_correction_increments_version(editor_client, demo_session):
    brief = _approved(demo_session)
    version = brief.version
    page = editor_client.get(f"/editor/briefs/{brief.id}").text
    assert f'action="/editor/briefs/{brief.id}/corrections"' in page
    assert "Published" in page
    resp = _post(
        editor_client,
        f"/editor/briefs/{brief.id}/corrections",
        {"description": "The study had 412 participants, not 4,120.", "material": "1"},
    )
    assert resp.status_code == 303
    stored = _get(demo_session, Brief, brief.id)
    assert stored.version == version + 1
    correction = demo_session.scalars(select(Correction).where(Correction.brief_id == brief.id)).one()
    assert correction.material is True
    assert "412 participants" in editor_client.get(f"/editor/briefs/{brief.id}").text
    # Minor corrections and validation.
    _post(editor_client, f"/editor/briefs/{brief.id}/corrections", {"description": "Fixed a typo."})
    assert _get(demo_session, Brief, brief.id).version == version + 2
    _post(editor_client, f"/editor/briefs/{brief.id}/corrections", {"description": "  "})
    assert _get(demo_session, Brief, brief.id).version == version + 2
    assert _flash(editor_client).startswith("error:")


def test_correction_refused_for_unpublished_brief(editor_client, demo_session):
    brief = _in_review(demo_session)
    _post(editor_client, f"/editor/briefs/{brief.id}/corrections", {"description": "Something"})
    _fresh(demo_session)
    assert demo_session.scalar(select(func.count(Correction.id)).where(Correction.brief_id == brief.id)) == 0
    assert "Corrections are for published briefs" in _flash(editor_client)


def test_regenerate_drafts_a_replacement(editor_client, demo_session):
    brief = _in_review(demo_session, "other")
    resp = _post(editor_client, f"/editor/briefs/{brief.id}/regenerate")
    assert resp.status_code == 303
    match = re.fullmatch(r"/editor/briefs/(\d+)", resp.headers["location"])
    assert match
    new_id = int(match.group(1))
    assert new_id != brief.id
    new = _get(demo_session, Brief, new_id)
    assert new.story_id == brief.story_id and new.status == BriefStatus.IN_REVIEW
    assert _get(demo_session, Brief, brief.id).status == BriefStatus.REJECTED  # superseded
    assert _actions(demo_session, brief.id, "regenerate")
    assert "Regenerated as brief" in _flash(editor_client)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def test_sources_page_lists_registry_and_explains_gate(editor_client, demo_session):
    _fresh(demo_session)
    sources = demo_session.scalars(select(Source)).all()
    html = editor_client.get("/editor/sources").text
    assert "Agents can only use approved, active sources" in html
    for source in sources[:5]:
        assert f'id="source-{source.id}"' in html
    assert 'name="field" value="approved"' in html and 'name="field" value="active"' in html
    assert 'aria-pressed="true"' in html


def test_source_toggle_flips_fields(editor_client, demo_session):
    _fresh(demo_session)
    source = demo_session.scalars(select(Source).order_by(Source.id)).first()
    resp = _post(editor_client, f"/editor/sources/{source.id}/toggle", {"field": "approved"})
    assert resp.status_code == 303 and resp.headers["location"] == f"/editor/sources#source-{source.id}"
    assert _get(demo_session, Source, source.id).approved is False
    _post(editor_client, f"/editor/sources/{source.id}/toggle", {"field": "active"})
    stored = _get(demo_session, Source, source.id)
    assert stored.active is False and stored.approved is False
    _post(editor_client, f"/editor/sources/{source.id}/toggle", {"field": "approved"})
    assert _get(demo_session, Source, source.id).approved is True


def test_source_toggle_htmx_and_validation(editor_client, demo_session):
    _fresh(demo_session)
    source = demo_session.scalars(select(Source).order_by(Source.id)).first()
    resp = _post(editor_client, f"/editor/sources/{source.id}/toggle", {"field": "active"}, headers=HX)
    assert resp.status_code == 200
    assert resp.text.lstrip().startswith(f'<tr id="source-{source.id}"')
    assert "Active is now off" in resp.text and "<html" not in resp.text
    assert _post(editor_client, f"/editor/sources/{source.id}/toggle", {"field": "name"}).status_code == 400
    assert _post(editor_client, "/editor/sources/999999/toggle", {"field": "active"}).status_code == 404


# --------------------------------------------------------------------------
# Agent runs
# --------------------------------------------------------------------------


def test_runs_page_lists_runs(editor_client, demo_session):
    _fresh(demo_session)
    total = demo_session.scalar(select(func.count(AgentRun.id)))
    html = editor_client.get("/editor/runs").text
    assert "story_brief" in html and "story_intelligence" in html
    assert f"of {total:,}" in html
    assert "badge-run-ok" in html
    assert "Fallback (offline engine answered)" in html  # status filter + legend
    assert "Budget exceeded" in html


def test_runs_page_filters_and_paginates(editor_client, demo_session):
    _fresh(demo_session)
    briefs = demo_session.scalar(select(func.count(AgentRun.id)).where(AgentRun.agent == "story_brief"))
    html = editor_client.get("/editor/runs?agent=story_brief&status=ok").text
    assert f"of {briefs:,} for <strong>story_brief</strong>" in html
    assert "Classify AI-in-finance claim" not in html
    page = editor_client.get("/editor/runs?limit=5&offset=5").text
    assert "Showing 6–10" in page
    assert 'rel="prev"' in page and 'rel="next"' in page
    assert "No agent runs match" in editor_client.get("/editor/runs?status=refused").text
    clamped = editor_client.get("/editor/runs?limit=100000&offset=-3")
    assert clamped.status_code == 200 and "Showing 1–" in clamped.text


def test_run_detail(editor_client, demo_session):
    _fresh(demo_session)
    run = demo_session.scalars(select(AgentRun).where(AgentRun.agent == "story_brief")).first()
    html = editor_client.get(f"/editor/runs/{run.id}").text
    assert f"Agent run #{run.id}" in html
    assert "Budget enforced" in html and "Tool trace" in html
    assert editor_client.get("/editor/runs/999999").status_code == 404


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def test_pipeline_page_explains_steps(editor_client):
    html = editor_client.get("/editor/pipeline").text
    for step in ("Discover", "Connect", "Explain", "Verify", "Finance trackers", "Output"):
        assert step in html
    assert 'name="network"' in html and "checked" not in re.search(
        r'<input type="checkbox" name="network"[^>]*>', html
    ).group(0)
    assert "No run from this desk yet" in html


def test_pipeline_post_runs_offline_and_reports(editor_client, demo_session):
    _fresh(demo_session)
    sources_before = demo_session.scalar(select(func.count(Source.id)))
    resp = _post(editor_client, "/editor/pipeline", {"max_briefs": "2"})
    assert resp.status_code == 303 and resp.headers["location"] == "/editor/pipeline#report"
    assert "Pipeline finished" in _flash(editor_client)
    html = editor_client.get("/editor/pipeline").text
    assert "Last run from this desk" in html and "network off" in html
    assert "Briefs drafted" in html and "Finance trackers" in html
    _fresh(demo_session)
    assert demo_session.scalar(select(func.count(Source.id))) == sources_before


def test_pipeline_rejects_bad_numbers(editor_client):
    resp = _post(editor_client, "/editor/pipeline", {"max_briefs": "lots"})
    assert resp.status_code == 303
    assert "up to 10 briefs" in editor_client.get("/editor/pipeline").text


# --------------------------------------------------------------------------
# Funding rounds
# --------------------------------------------------------------------------


def test_funding_page_and_verify(editor_client, demo_session):
    _fresh(demo_session)
    row = demo_session.scalars(select(FundingRound).where(FundingRound.verified.is_(False))).first()
    html = editor_client.get("/editor/funding").text
    assert DISCLAIMER_HTML in html
    assert f'action="/editor/funding/{row.id}/verify"' in html
    assert "never ranks startups" in html
    resp = _post(editor_client, f"/editor/funding/{row.id}/verify", {"next": f"/editor/funding#round-{row.id}"})
    assert resp.status_code == 303 and resp.headers["location"] == f"/editor/funding#round-{row.id}"
    assert _get(demo_session, FundingRound, row.id).verified is True
    assert "Verified by an editor" in editor_client.get("/editor/funding").text


def test_funding_verify_rejects_open_redirect_and_missing(editor_client, demo_session):
    _fresh(demo_session)
    row = demo_session.scalars(select(FundingRound)).first()
    resp = _post(editor_client, f"/editor/funding/{row.id}/verify", {"next": "https://evil.example/"})
    assert resp.headers["location"] == "/editor/funding"
    assert _post(editor_client, "/editor/funding/999999/verify").status_code == 404
