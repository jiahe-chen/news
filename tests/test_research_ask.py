"""Ask Newsren research agent: offline answers on the demo corpus, context seeding, advice refusal,
citation hygiene, the Claude tool loop (fake client), tool budgets, and the guardrail gate."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import func, select

from newsren.agents import tools
from newsren.agents.base import Orchestrator, citation_for
from newsren.agents.research import AGENT_NAME, SYSTEM_PROMPT, ask, questions_for, sanitize_answer
from newsren.agents.tools import HttpResponse
from newsren.editorial.guardrails import ADVICE_REFUSAL
from newsren.ingest.fixtures import load_demo_data
from newsren.llm.claude import ClaudeProvider
from newsren.models import AgentRun, Article, Brief, Question, Story
from newsren.schemas import CitationRef, ResearchAnswerOut, StatementOut

DATA = Path(__file__).parent / "data" / "research"


@pytest.fixture()
def demo(session) -> dict[str, Article]:
    load_demo_data(session)
    return {a.extra["demo_key"]: a for a in session.scalars(select(Article)) if a.extra.get("demo_key")}


def _article_urls(session) -> set[str]:
    return set(session.scalars(select(Article.url)))


def _assert_well_formed(row: Question) -> ResearchAnswerOut:
    """The stored result validates and every citation id used is listed."""
    result = ResearchAnswerOut.model_validate(row.result)
    ids = {c.id for c in result.citations}
    assert [c.id for c in result.citations] == [f"c{i}" for i in range(1, len(ids) + 1)]
    for statement in result.verified:
        assert statement.citation_ids and set(statement.citation_ids) <= ids
    assert set(re.findall(r"\[(c\d+)\]", result.answer)) <= ids
    assert row.citations == [c.model_dump(mode="json") for c in result.citations]
    return result


# --------------------------------------------------------------------------
# Offline agent on the demo corpus
# --------------------------------------------------------------------------


def test_offline_answer_cites_real_articles(session, demo):
    row = ask(session, "Did an independent benchmark confirm Lumen-4's math claims?")
    result = _assert_well_formed(row)
    assert not row.refused and row.answer
    assert result.citations and {c.url for c in result.citations} <= _article_urls(session)
    assert all(c.article_id and c.published_at and c.snippet for c in result.citations)
    assert "[c1]" in result.answer and len(result.verified) >= 2
    assert 2 <= len(result.plan) <= 5 and result.follow_up_questions

    # The correction to the independent score was read and flagged.
    correction = demo["lumen4-correction"]
    assert correction.url in {c.url for c in result.citations}
    assert "correction" in result.answer.lower()

    run = session.get(AgentRun, row.agent_run_id)
    assert run.agent == AGENT_NAME and run.provider == "offline" and run.status == "ok"
    assert run.subject_type == "question" and run.subject_id == row.id
    assert run.tool_calls >= 3 and run.cost_usd == 0
    tools_used = [t["tool"] for t in row.tool_trace]
    assert tools_used[0] == "search_archive" and "get_article" in tools_used
    assert not any(t in tools_used for t in ("search_research_papers", "search_sec_filings", "fetch_approved_url"))


def test_offline_answer_labels_vendor_claims_and_backtests(session, demo):
    row = ask(session, "Did Northbridge's AI agents beat the market?")
    result = _assert_well_formed(row)
    assert "organization's own account" in result.answer
    assert "backtest" in result.answer.lower()
    assert any("no live track record" in item for item in result.could_not_verify)
    assert "should buy" not in result.answer.lower()


def test_could_not_verify_lists_unsupported_parts(session, demo):
    row = ask(session, "What did Lumen Labs say about Lumen-4 availability in Antarctica?")
    result = _assert_well_formed(row)
    assert any("Antarctica" in item for item in result.could_not_verify)
    assert "do not address" in result.answer and "Antarctica" in result.answer
    assert result.citations, "the parts the archive does cover are still answered with sources"


def test_future_question_is_labelled_as_projection(session, demo):
    row = ask(session, "Will Corvane reach 40,000 wafers a month at Calder Valley?")
    result = _assert_well_formed(row)
    assert any("future" in item for item in result.could_not_verify)
    assert "projection" in result.answer
    assert any("11,500" in s.text for s in result.verified), "reported output is compared with the plan"


def test_no_coverage_is_reported_honestly(session, demo):
    row = ask(session, "How do penguins navigate during polar winter?")
    result = _assert_well_formed(row)
    assert result.citations == [] and result.verified == []
    assert "no coverage" in result.answer and "general knowledge" in result.answer
    assert result.could_not_verify[0].startswith("No retrieved source addresses the question")


def test_research_question_notes_disabled_primary_sources(session, demo):
    row = ask(session, "What did the Halbrook study actually measure?", audience="student")
    result = _assert_well_formed(row)
    assert row.audience == "student"
    assert any("arXiv" in item and "disabled" in item for item in result.could_not_verify)
    assert result.answer.endswith("whether it is reporting a result or making a claim.")
    assert len(result.verified) <= 3


def test_audience_defaults_to_reader_preference(session, demo, factory):
    user = factory.user(audience="educator")
    row = ask(session, "What is Calder's AI literacy requirement?", user=user, session_key="sess-1")
    assert row.audience == "educator" and row.user_id == user.id
    assert "Classroom angle" in row.answer
    assert ask(session, "What is Calder's AI literacy requirement?", audience="robot").audience is None
    assert questions_for(session, user=user) == [row]
    assert questions_for(session, session_key="sess-1") == [row]
    assert questions_for(session) == []


def test_empty_question_is_rejected(session):
    with pytest.raises(ValueError):
        ask(session, "   ")


# --------------------------------------------------------------------------
# Context seeding
# --------------------------------------------------------------------------


@pytest.fixture()
def halbrook_brief(session, demo) -> Brief:
    preprint, release = demo["halbrook-preprint"], demo["halbrook-press-release"]
    brief = Brief(
        vertical="education",
        headline="Halbrook StepWise study: faster practice, no measured transfer gain",
        summary="A university press release said students learned algebra 30% faster with an AI assistant; the "
        "underlying preprint measured exercise completion time and transfer separately.",
        why_it_matters="Schools weighing AI tutors need to know which learning outcome was measured.",
        key_points=["The 30% figure is about completion speed.", "Transfer scores did not differ significantly."],
        citations=[citation_for(preprint, "c1").model_dump(), citation_for(release, "c2").model_dump()],
    )
    session.add(brief)
    session.flush()
    return brief


def test_context_seeding_from_brief_offline(session, demo, halbrook_brief):
    row = ask(session, "Why does this matter for schools?", brief=halbrook_brief)
    result = _assert_well_formed(row)
    assert row.brief_id == halbrook_brief.id
    first_search = row.tool_trace[0]
    assert first_search["tool"] == "search_archive"
    assert "halbrook" in first_search["input"]["query"].lower(), "the search is seeded with the brief's terms"
    read = [t["input"]["article_id"] for t in row.tool_trace if t["tool"] == "get_article"]
    brief_ids = {c["article_id"] for c in halbrook_brief.citations}
    assert brief_ids & set(read), "the agent starts from the brief's own sources"
    assert result.citations and any("Halbrook" in c.title or "StepWise" in c.title for c in result.citations)
    assert any("Causes" in item for item in result.could_not_verify)


def test_context_seeding_from_story_and_article(session, demo):
    members = [demo[k] for k in ("corvane-capacity", "corvane-capacity-analysis", "corvane-output-q3")]
    story = Story(title="Corvane's Calder Valley capacity plan", vertical="industry", keywords=["calder valley"])
    session.add(story)
    session.flush()
    for article in members:
        article.story_id = story.id
    session.flush()
    session.refresh(story)

    row = ask(session, "How far along is the expansion?", story=story)
    result = _assert_well_formed(row)
    assert row.story_id == story.id
    assert {c.url for c in result.citations} & {a.url for a in members}

    syndicated = demo["corvane-output-q3-harbor"]
    row2 = ask(session, "What was the output shortfall?", article=syndicated)
    assert row2.article_id == syndicated.duplicate_of_id, "a syndicated copy resolves to the original"
    read = [t["input"]["article_id"] for t in row2.tool_trace if t["tool"] == "get_article"]
    assert read[0] == syndicated.duplicate_of_id


# --------------------------------------------------------------------------
# Finance advice refusal
# --------------------------------------------------------------------------


def test_advice_request_is_refused_with_education(session, demo):
    row = ask(session, "Should I buy Corvane stock after its earnings?")
    result = _assert_well_formed(row)
    assert row.refused and result.refused
    assert result.answer.startswith(ADVICE_REFUSAL)
    assert result.refusal_reason and "investment advice" in result.refusal_reason
    assert "Concepts that can help" in result.answer
    assert row.agent_run_id is None
    assert session.scalar(select(func.count()).select_from(AgentRun).where(AgentRun.agent == AGENT_NAME)) == 0
    assert result.citations and "Corvane" in result.citations[0].title
    assert {c.url for c in result.citations} <= _article_urls(session)
    assert any("Corvane" in q for q in result.follow_up_questions)
    assert session.scalar(select(func.count()).select_from(Question)) == 1


def test_concept_question_is_not_refused(session, demo):
    row = ask(session, "What is a backtest, and why did Northbridge's result come with a caveat?")
    assert not row.refused and row.agent_run_id is not None


# --------------------------------------------------------------------------
# Citation hygiene
# --------------------------------------------------------------------------


def test_sanitize_drops_unknown_citations_and_moves_unsupported_statements():
    real = CitationRef(
        id="A7", url="https://news.example/a", title="Real title", publisher="Real Pub", article_id=7, snippet="s"
    )
    output = ResearchAnswerOut(
        answer="Supported point [c1]. Invented point [c2]. Both [c1, c2].",
        verified=[
            StatementOut(text="Supported point.", citation_ids=["c1"]),
            StatementOut(text="Invented point.", citation_ids=["c2"]),
        ],
        citations=[
            CitationRef(id="c1", url="https://news.example/a?utm_source=x", title="Wrong title", snippet="nope"),
            CitationRef(id="c2", url="https://made-up.example.org/z", title="Fake"),
        ],
    )
    clean = sanitize_answer(output, [real], {"https://news.example/a": "The real text says supported point."})
    assert [c.id for c in clean.citations] == ["c1"]
    assert clean.citations[0].title == "Real title" and clean.citations[0].publisher == "Real Pub"
    assert clean.answer == "Supported point [c1]. Invented point. Both [c1]."
    assert [s.text for s in clean.verified] == ["Supported point."]
    assert "Not confirmed by a retrieved source: Invented point." in clean.could_not_verify


# --------------------------------------------------------------------------
# Claude tool loop with a fake client
# --------------------------------------------------------------------------


class FakeMessages:
    """Stands in for ``client.beta.messages``: replays a script and records every request."""

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        step = self.script.pop(0)
        return step(kwargs) if callable(step) else step


class FakeClient:
    def __init__(self, script: list[Any]):
        self.beta = SimpleNamespace(messages=FakeMessages(script))


def _response(stop_reason: str, content: list[Any]) -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=1500, output_tokens=400, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    return SimpleNamespace(stop_reason=stop_reason, content=content, model="claude-opus-5-5", usage=usage)


def _tool_use(tool_id: str, name: str, tool_input: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def _final(answer: ResearchAnswerOut) -> SimpleNamespace:
    return _response("end_turn", [SimpleNamespace(type="text", text=answer.model_dump_json())])


def _orchestrator(session, settings, script, **overrides):
    client = FakeClient(script)
    provider = ClaudeProvider(model="claude-opus-5-5", client=client)
    return Orchestrator(session, provider=provider, settings=settings.with_overrides(**overrides)), client


def _tool_results(message: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in message["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]


def test_fake_claude_tool_loop(session, settings, demo):
    def answer_from_search(kwargs: dict[str, Any]) -> SimpleNamespace:
        result_text = _tool_results(kwargs["messages"][-1])[0]["content"]
        url = re.search(r"URL: (https://\S+)", result_text).group(1)
        snippet = re.search(r'Snippet: "([^"]+)"', result_text).group(1)
        return _final(
            ResearchAnswerOut(
                answer="An independent evaluation scored Lumen-4 below the vendor's figure [c1]. "
                "It also won an award [c2].",
                plan=["Search the archive for independent evaluations of Lumen-4.", "Compare with vendor claims."],
                verified=[
                    StatementOut(
                        text="An independent evaluation scored Lumen-4 below the vendor's figure.",
                        citation_ids=["c1"],
                    ),
                    StatementOut(text="Lumen-4 won an industry award.", citation_ids=["c2"]),
                ],
                could_not_verify=["Whether the vendor will update its claims."],
                citations=[
                    CitationRef(id="c1", url=url, title="(model-supplied title)", snippet=snippet),
                    CitationRef(id="c2", url="https://fabricated.example.org/award", title="Award"),
                ],
                follow_up_questions=["Has Lumen Labs responded to the independent result?"],
            )
        )

    script = [
        _response(
            "tool_use",
            [
                SimpleNamespace(type="text", text="I'll search the archive first."),
                _tool_use(
                    "tu1",
                    "search_archive",
                    {"query": "Lumen-4 independent evaluation", "vertical": None, "since_days": None},
                ),
            ],
        ),
        answer_from_search,
    ]
    orch, client = _orchestrator(session, settings, script)
    row = ask(session, "What did independent testing find about Lumen-4?", orchestrator=orch, allow_network=False)
    calls = client.beta.messages.calls

    # Request shape: system prompt, strict tools, structured output, user prompt.
    first = calls[0]
    assert SYSTEM_PROMPT in first["system"][0]["text"]
    assert [t["name"] for t in first["tools"]][0] == "search_archive" and all(t["strict"] for t in first["tools"])
    assert first["output_config"]["format"]["type"] == "json_schema"
    assert "<question>" in first["messages"][0]["content"] and "disabled" in first["messages"][0]["content"]

    # The tool result was sent back for tool_use id tu1.
    results = _tool_results(calls[1]["messages"][-1])
    assert results[0]["tool_use_id"] == "tu1" and results[0]["is_error"] is False
    assert "Archive search for" in results[0]["content"] and "[A" in results[0]["content"]

    run = session.get(AgentRun, row.agent_run_id)
    assert run.provider == "claude" and run.status == "ok" and run.model == "claude-opus-5-5"
    assert run.tool_calls == 1 and run.llm_calls == 2 and run.cost_usd > 0
    assert row.tool_trace[0]["tool"] == "search_archive"

    # Only the retrieved source survives; metadata comes from our records; the fabricated claim moves.
    result = _assert_well_formed(row)
    assert len(result.citations) == 1
    cited = session.scalars(select(Article).where(Article.url == result.citations[0].url)).one()
    assert result.citations[0].title == cited.title and result.citations[0].article_id == cited.id
    assert "[c2]" not in result.answer
    assert [s.text for s in result.verified] == [
        "An independent evaluation scored Lumen-4 below the vendor's figure."
    ]
    assert "Not confirmed by a retrieved source: Lumen-4 won an industry award." in result.could_not_verify
    assert result.citations[0].snippet in results[0]["content"], "a verbatim quote from the tool result is kept"


def test_tool_budget_exhausted(session, settings, demo):
    target = demo["lumen4-independent-eval"]

    def final(kwargs: dict[str, Any]) -> SimpleNamespace:
        return _final(
            ResearchAnswerOut(answer="Budget spent; answering from the search results.", plan=["Search."])
        )

    script = [
        _response(
            "tool_use",
            [_tool_use("tu1", "search_archive", {"query": "Lumen-4", "vertical": None, "since_days": None})],
        ),
        _response("tool_use", [_tool_use("tu2", "get_article", {"article_id": target.id})]),
        final,
    ]
    orch, client = _orchestrator(session, settings, script, agent_max_tool_calls=1)
    row = ask(session, "What did the independent evaluation of Lumen-4 find?", orchestrator=orch)
    calls = client.beta.messages.calls
    assert len(calls) == 3
    first_result = _tool_results(calls[1]["messages"][-1])[0]
    second_result = _tool_results(calls[2]["messages"][-1])[0]
    assert first_result["tool_use_id"] == "tu1" and not first_result["is_error"]
    assert second_result["tool_use_id"] == "tu2" and second_result["is_error"] is True
    assert second_result["content"].startswith("Tool budget exhausted")

    run = session.get(AgentRun, row.agent_run_id)
    assert run.tool_calls == 1 and run.llm_calls == 3
    assert run.budget["max_tool_calls"] == 1
    assert [t["is_error"] for t in row.tool_trace] == [False, True]


def test_claude_prompt_carries_brief_context(session, settings, demo, halbrook_brief):
    script = [
        _final(ResearchAnswerOut(answer="The study measured completion speed and transfer.", plan=["Read."]))
    ]
    orch, client = _orchestrator(session, settings, script)
    ask(session, "What did the study measure?", brief=halbrook_brief, orchestrator=orch, audience="professional")
    prompt = client.beta.messages.calls[0]["messages"][0]["content"]
    assert "<reader_context>" in prompt and halbrook_brief.headline in prompt
    assert halbrook_brief.summary.split(";")[0] in prompt
    for cit in halbrook_brief.citations:
        assert f"[A{cit['article_id']}]" in prompt and cit["url"] in prompt
    assert "Reader audience: professional" in prompt


def test_brief_sources_are_citable_without_rereading(session, settings, demo, halbrook_brief):
    preprint_url = halbrook_brief.citations[0]["url"]
    script = [
        _final(
            ResearchAnswerOut(
                answer="The preprint measured completion time and transfer [c1].",
                verified=[
                    StatementOut(text="The preprint measured completion time and transfer.", citation_ids=["c1"])
                ],
                citations=[CitationRef(id="c1", url=preprint_url, title="x")],
            )
        )
    ]
    orch, _ = _orchestrator(session, settings, script)
    row = ask(session, "What did the study measure?", brief=halbrook_brief, orchestrator=orch)
    result = _assert_well_formed(row)
    assert [c.url for c in result.citations] == [preprint_url]
    assert result.citations[0].doc_type == "research_preprint"


def test_network_defaults_on_for_claude(session, settings, demo, monkeypatch):
    sample = (DATA / "arxiv_tradingagents.xml").read_bytes()
    monkeypatch.setattr(
        tools,
        "http_get",
        lambda url, settings, **kw: HttpResponse(
            url=url, status_code=200, content_type="application/xml", content=sample
        ),
    )
    script = [
        _response("tool_use", [_tool_use("tu1", "search_research_papers", {"query": "TradingAgents"})]),
        _final(ResearchAnswerOut(answer="TradingAgents is a preprint describing a simulated trading firm.")),
    ]
    orch, client = _orchestrator(session, settings, script)
    row = ask(session, "What is the TradingAgents paper about?", orchestrator=orch)
    result = _tool_results(client.beta.messages.calls[1]["messages"][-1])[0]
    assert not result["is_error"] and "arXiv:2412.20138" in result["content"]
    assert "are available" in client.beta.messages.calls[0]["messages"][0]["content"]
    assert row.tool_trace[0]["citations"][0] == "https://arxiv.org/abs/2412.20138"


def test_guardrail_withholds_unsafe_model_answer(session, settings, demo):
    unsafe = "Corvane beat its guidance, so investors should buy Corvane shares now; AI agents reliably beat the market."
    script = [_final(ResearchAnswerOut(answer=unsafe, plan=["Answer."]))]
    orch, _ = _orchestrator(session, settings, script)
    row = ask(session, "What did Corvane report for its third quarter compared with guidance?", orchestrator=orch)
    result = _assert_well_formed(row)
    assert row.answer != unsafe and "should buy" not in row.answer
    assert result.citations, "the extractive fallback answered from approved sources"
    gate = [t for t in row.tool_trace if t["tool"] == "editorial_guardrail"]
    assert gate and gate[0]["is_error"]
    assert result.plan[-1] == "The model's draft was withheld by the editorial quality gate."


def test_claude_refusal_falls_back_to_offline(session, settings, demo):
    refusal = _response("refusal", [])
    refusal.stop_details = SimpleNamespace(category="cyber")
    orch, _ = _orchestrator(session, settings, [refusal])
    row = ask(session, "Which organizations were attacked through the Pallas Gateway flaw?", orchestrator=orch)
    run = session.get(AgentRun, row.agent_run_id)
    assert run.status == "refused" and run.provider == "offline"
    result = _assert_well_formed(row)
    assert result.citations and {c.url for c in result.citations} <= _article_urls(session)
    assert json.loads(json.dumps(row.result)) == row.result


def test_offline_builder_uses_primary_sources_when_network_allowed(session, demo, monkeypatch):
    bodies = {
        tools.ARXIV_API_URL: (DATA / "arxiv_tradingagents.xml").read_bytes(),
        tools.SEC_TICKERS_URL: (DATA / "sec_company_tickers_sample.json").read_bytes(),
        tools.SEC_SUBMISSIONS_URL.format(cik="0001045810"): (
            DATA / "sec_submissions_nvda_sample.json"
        ).read_bytes(),
    }

    def fake(url, settings, **kwargs):
        body = next(v for k, v in bodies.items() if url.startswith(k))
        return HttpResponse(url=url, status_code=200, content_type="application/octet-stream", content=body)

    tools.clear_caches()
    monkeypatch.setattr(tools, "http_get", fake)

    paper = ask(session, "What does the TradingAgents research paper propose?", allow_network=True)
    result = _assert_well_formed(paper)
    assert result.citations[0].url == "https://arxiv.org/abs/2412.20138"
    assert result.answer.startswith("arXiv preprint") and "not peer reviewed" in result.answer
    assert "search_research_papers" in [t["tool"] for t in paper.tool_trace]
    assert not any("disabled" in item for item in result.could_not_verify)

    filings = ask(session, "What did $NVDA file with the SEC recently?", allow_network=True)
    result = _assert_well_formed(filings)
    assert result.answer.startswith("SEC EDGAR lists NVIDIA CORP's most recent material filings")
    assert result.citations[0].doc_type == "filing" and result.citations[0].url.startswith("https://www.sec.gov/")
    assert result.could_not_verify == []
    tools.clear_caches()
