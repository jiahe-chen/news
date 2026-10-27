"""JSON API v1: shapes, visibility rules, permissions, CSRF."""

from __future__ import annotations

from sqlalchemy import func, select

from newsren.editorial.guardrails import FINANCE_DISCLAIMER
from newsren.models import AIFinanceClaim, Brief, BriefStatus, FundingRound, Question, Story


def _fresh(session) -> None:
    session.rollback()
    session.expire_all()


def _brief(session, status: str, vertical: str | None = None) -> Brief:
    _fresh(session)
    query = select(Brief).where(Brief.status == status).order_by(Brief.id)
    if vertical:
        query = query.where(Brief.vertical == vertical)
    brief = session.scalars(query).first()
    assert brief is not None
    return brief


def _api_token(client) -> str:
    resp = client.get("/api/v1/csrf")
    assert resp.status_code == 200
    return resp.json()["csrf_token"]


def _iso(value: str | None) -> bool:
    return value is None or (value.endswith("Z") and "T" in value)


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------


def test_index_documents_endpoints_and_csrf(client):
    data = client.get("/api/v1").json()
    assert data["version"] == "v1"
    paths = {e["path"] for e in data["endpoints"]}
    assert {"/api/v1/briefs", "/api/v1/ask", "/api/v1/metrics", "/api/v1/me/alerts"} <= paths
    assert "X-CSRF-Token" in data["csrf"]


# --------------------------------------------------------------------------
# Briefs
# --------------------------------------------------------------------------


def test_briefs_lists_only_approved_with_claims_and_citations(client, demo_session):
    _fresh(demo_session)
    approved = demo_session.scalar(select(func.count(Brief.id)).where(Brief.status == BriefStatus.APPROVED))
    resp = client.get("/api/v1/briefs?limit=100")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == approved
    assert all(b["status"] == "approved" and b["published"] for b in data["briefs"])
    first = data["briefs"][0]
    for key in ("id", "headline", "summary", "claims", "citations", "published_at", "generated_by", "vertical"):
        assert key in first
    assert first["claims"] and {"text", "claim_type", "support_status", "citation_ids"} <= set(first["claims"][0])
    assert first["citations"] and {"url", "published_at", "retrieved_at"} <= set(first["citations"][0])
    assert _iso(first["published_at"]) and _iso(first["generated_at"])
    assert "editorial" not in first
    assert "editor_verdict" not in first["claims"][0]


def test_briefs_filter_limit_and_validation(client):
    data = client.get("/api/v1/briefs?vertical=education&limit=3").json()
    assert data["vertical"] == "education"
    assert 0 < data["count"] <= 3
    assert all(b["vertical"] == "education" for b in data["briefs"])
    assert data["disclaimer"] is None
    assert client.get("/api/v1/briefs?vertical=sports").status_code == 422
    assert client.get("/api/v1/briefs?limit=0").status_code == 422
    assert client.get("/api/v1/briefs?limit=1000").status_code == 422


def test_brief_detail_visibility(client, demo_session):
    published = _brief(demo_session, BriefStatus.APPROVED)
    unpublished = _brief(demo_session, BriefStatus.IN_REVIEW, "finance")
    ok = client.get(f"/api/v1/briefs/{published.id}")
    assert ok.status_code == 200 and ok.json()["id"] == published.id
    hidden = client.get(f"/api/v1/briefs/{unpublished.id}")
    assert hidden.status_code == 404
    assert hidden.json() == {"detail": "No published brief with that id."}
    assert client.get("/api/v1/briefs/999999").status_code == 404


def test_brief_detail_for_reader_and_editor(reader_client, demo_session):
    unpublished = _brief(demo_session, BriefStatus.IN_REVIEW, "finance")
    assert reader_client.get(f"/api/v1/briefs/{unpublished.id}").status_code == 404
    from conftest import login

    login(reader_client, "editor@newsren.example", "editor-demo-pass")
    data = reader_client.get(f"/api/v1/briefs/{unpublished.id}").json()
    assert data["status"] == "in_review" and data["published"] is False
    assert data["editorial"]["check_results"] and data["editorial"]["risk_level"] == "high"
    assert data["editorial"]["review_url"] == f"/editor/briefs/{unpublished.id}"
    assert data["disclaimer"] == FINANCE_DISCLAIMER
    assert "editor_verdict" in data["claims"][0]


# --------------------------------------------------------------------------
# Stories and search
# --------------------------------------------------------------------------


def _story_with_and_without_brief(session) -> tuple[Story, Story]:
    _fresh(session)
    published_ids = set(
        session.scalars(
            select(Brief.story_id).where(Brief.status == BriefStatus.APPROVED, Brief.story_id.is_not(None))
        )
    )
    stories = session.scalars(select(Story).order_by(Story.id)).all()
    with_brief = next(s for s in stories if s.id in published_ids)
    without = next(s for s in stories if s.id not in published_ids)
    return with_brief, without


def test_story_has_events_related_and_intelligence(client, demo_session):
    story, _ = _story_with_and_without_brief(demo_session)
    data = client.get(f"/api/v1/stories/{story.id}").json()
    assert data["id"] == story.id and data["brief_id"] is not None
    assert data["events"] and {"occurred_at", "event_type", "title", "article"} <= set(data["events"][0])
    assert data["events"][0]["article"]["url"]
    assert isinstance(data["related_story_ids"], list)
    assert all(isinstance(i, int) for i in data["related_story_ids"])
    assert data["intelligence"] is not None and "whats_new" in data["intelligence"]
    assert _iso(data["last_updated_at"])


def test_story_intelligence_hidden_until_published(client, demo_session):
    _, story = _story_with_and_without_brief(demo_session)
    data = client.get(f"/api/v1/stories/{story.id}").json()
    assert data["intelligence"] is None and data["intelligence_note"]
    assert data["events"]  # the sourced timeline stays public
    assert client.get("/api/v1/stories/999999").status_code == 404


def test_story_intelligence_preview_for_editors(editor_client, demo_session):
    _, story = _story_with_and_without_brief(demo_session)
    data = editor_client.get(f"/api/v1/stories/{story.id}").json()
    assert data["intelligence"] is not None and data["intelligence"]["preview"] is True


def test_search_returns_briefs_and_articles(client):
    data = client.get("/api/v1/search?q=AI literacy").json()
    assert data["query"] == "AI literacy"
    assert data["briefs"] or data["articles"]
    for brief in data["briefs"]:
        assert brief["url"].startswith("/briefs/")
    for article in data["articles"]:
        assert article["url"].startswith("http") and _iso(article["retrieved_at"])
    assert client.get("/api/v1/search").status_code == 422
    assert client.get("/api/v1/search?q=").status_code == 422


def test_search_never_returns_unpublished_briefs(client, demo_session):
    unpublished = _brief(demo_session, BriefStatus.IN_REVIEW, "finance")
    words = " ".join(unpublished.headline.split()[:4])
    data = client.get("/api/v1/search", params={"q": words}).json()
    assert unpublished.id not in {b["id"] for b in data["briefs"]}


# --------------------------------------------------------------------------
# Ask Newsren
# --------------------------------------------------------------------------


def test_ask_requires_csrf_header(client, demo_session):
    _fresh(demo_session)
    before = demo_session.scalar(select(func.count(Question.id)))
    resp = client.post("/api/v1/ask", json={"question": "What did the benchmark find?"})
    assert resp.status_code == 403
    bad = client.post("/api/v1/ask", json={"question": "x"}, headers={"X-CSRF-Token": "forged"})
    assert bad.status_code == 403
    _fresh(demo_session)
    assert demo_session.scalar(select(func.count(Question.id))) == before


def test_ask_answers_with_sources(client, demo_session):
    published = _brief(demo_session, BriefStatus.APPROVED)
    token = _api_token(client)
    resp = client.post(
        "/api/v1/ask",
        json={"question": "What evidence supports this story?", "brief_id": published.id, "audience": "student"},
        headers={"X-CSRF-Token": token},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    for key in ("question_id", "answer", "plan", "verified", "could_not_verify", "citations", "refused"):
        assert key in data
    assert data["brief_id"] == published.id and data["audience"] == "student"
    assert data["url"] == f"/questions/{data['question_id']}"
    assert data["answer"]
    _fresh(demo_session)
    stored = demo_session.get(Question, data["question_id"])
    assert stored is not None and stored.brief_id == published.id


def test_ask_refuses_investment_advice_educationally(client):
    token = _api_token(client)
    resp = client.post(
        "/api/v1/ask", json={"question": "Should I buy Corvane stock now?"}, headers={"X-CSRF-Token": token}
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["refused"] is True
    assert data["disclaimer"] == FINANCE_DISCLAIMER


def test_ask_validation(client, demo_session):
    token = _api_token(client)
    headers = {"X-CSRF-Token": token}
    assert client.post("/api/v1/ask", json={"question": ""}, headers=headers).status_code == 422
    assert client.post("/api/v1/ask", json={"question": "x" * 1001}, headers=headers).status_code == 422
    assert client.post("/api/v1/ask", json={"brief_id": 1}, headers=headers).status_code == 422
    assert client.post("/api/v1/ask", content=b"not json", headers=headers).status_code == 400
    unpublished = _brief(demo_session, BriefStatus.IN_REVIEW)
    hidden = client.post("/api/v1/ask", json={"question": "Why?", "brief_id": unpublished.id}, headers=headers)
    assert hidden.status_code == 404


# --------------------------------------------------------------------------
# Finance
# --------------------------------------------------------------------------


def test_finance_funding_trends_and_rounds(client, demo_session):
    _fresh(demo_session)
    total = demo_session.scalar(select(func.count(FundingRound.id)))
    data = client.get("/api/v1/finance/funding").json()
    assert data["disclaimer"] == FINANCE_DISCLAIMER
    assert len(data["rounds"]) == total
    trends = data["trends"]
    assert {"by_sector", "by_stage", "investor_participation", "total_rounds", "note"} <= set(trends)
    assert all({"name", "rounds"} == set(i) for i in trends["investor_participation"])
    first = data["rounds"][0]
    assert {"company_name", "stage", "amount_text", "source_url", "verified_by_editor"} <= set(first)
    assert _iso(first["announced_at"]) and _iso(first["retrieved_at"])
    assert client.get("/api/v1/finance/funding?days=0").status_code == 422


def test_finance_ai_claims(client, demo_session):
    _fresh(demo_session)
    total = demo_session.scalar(select(func.count(AIFinanceClaim.id)))
    data = client.get("/api/v1/finance/ai-claims").json()
    assert data["disclaimer"] == FINANCE_DISCLAIMER
    assert data["count"] == total
    assert "backtest" in data["classifications"]
    backtests = client.get("/api/v1/finance/ai-claims?classification=backtest").json()
    assert backtests["count"] >= 1
    assert all(c["classification"] == "backtest" for c in backtests["claims"])
    assert all(c["performance_label"] for c in backtests["claims"])
    assert client.get("/api/v1/finance/ai-claims?classification=moonshot").status_code == 422


# --------------------------------------------------------------------------
# Reader and editor endpoints
# --------------------------------------------------------------------------


def test_my_alerts_requires_sign_in(client):
    resp = client.get("/api/v1/me/alerts", follow_redirects=False)
    assert resp.status_code == 401
    assert "detail" in resp.json()


def test_my_alerts_for_reader(reader_client):
    data = reader_client.get("/api/v1/me/alerts").json()
    assert data["count"] == len(data["alerts"])
    for alert in data["alerts"]:
        assert {"title", "reason", "citations", "follow", "created_at"} <= set(alert)


def test_metrics_permissions(client, reader_client):
    # reader_client shares the client fixture; check anonymous first on a cleared jar.
    reader_client.cookies.clear()
    assert reader_client.get("/api/v1/metrics").status_code == 401
    from conftest import login

    login(reader_client, "reader@newsren.example", "reader-demo-pass")
    assert reader_client.get("/api/v1/metrics").status_code == 403
    login(reader_client, "editor@newsren.example", "editor-demo-pass")
    data = reader_client.get("/api/v1/metrics").json()
    assert {"trust", "usefulness", "education", "engagement", "editorial", "cost", "generated_at"} <= set(data)
