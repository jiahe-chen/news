"""The story-brief agent against a fake Claude client: request shape, citation hygiene, and refusal fallback."""

from __future__ import annotations

import json
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from newsren.agents.base import Orchestrator
from newsren.agents.brief import AGENT_NAME, generate_brief
from newsren.llm.claude import FALLBACK_BETA, ClaudeProvider
from newsren.models import AgentRun, AgentRunStatus, DocType, Story, TrustTier, Vertical, utcnow
from newsren.schemas import BriefOut, ClaimOut, DisagreementOut, DisagreementPosition, SectionOut

MODEL = "claude-opus-5-5"


class FakeMessages:
    def __init__(self, text: str, stop_reason: str = "end_turn"):
        self.text = text
        self.stop_reason = stop_reason
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(
                input_tokens=1200, output_tokens=800, cache_read_input_tokens=0, cache_creation_input_tokens=0
            ),
            model=MODEL,
        )


class FakeClient:
    def __init__(self, text: str, stop_reason: str = "end_turn"):
        self.beta = SimpleNamespace(messages=FakeMessages(text, stop_reason))


@pytest.fixture()
def story(session, factory) -> Story:
    vendor = factory.source(
        trust_tier=TrustTier.VENDOR, doc_type=DocType.PRESS_RELEASE, name="Lumen Labs Newsroom"
    )
    lab = factory.source(trust_tier=TrustTier.INDEPENDENT, doc_type=DocType.RESEARCH_PREPRINT, name="Open Eval")
    now = utcnow()
    launch = factory.article(
        "Lumen Labs launches Lumen-4",
        "Lumen Labs says Lumen-4 scores 92.1% on MathBench-Pro, the best math score it has measured.",
        source=vendor,
        published_at=now - timedelta(days=5),
    )
    evaluation = factory.article(
        "Independent evaluation of Lumen-4 on held-out problems",
        "The Open Eval group found Lumen-4 scored 84.3% on 1,200 held-out problems, below the 92.1% the vendor "
        "reported. This report is a preprint and has not been peer reviewed.",
        source=lab,
        published_at=now - timedelta(days=2),
    )
    story = Story(title="Lumen-4 launch and evaluation", vertical=Vertical.TECHNOLOGY)
    session.add(story)
    launch.story = evaluation.story = story
    session.flush()
    return story


def claude_brief() -> str:
    return BriefOut(
        headline="Lumen-4: vendor reports 92.1%, independent preprint finds 84.3%",
        summary="Lumen Labs says Lumen-4 scores 92.1% [c2]. An independent preprint found 84.3% [c1][c99].",
        why_it_matters="Schools and developers weighing the model need independent evidence [c1].",
        what_changed="An independent evaluation, still a preprint, now tests the company's claim [c1].",
        key_points=["The independent result is lower than the vendor's figure [c1][c2]."],
        sections=[
            SectionOut(title="What happened?", body="Lumen Labs launched Lumen-4 [c2].", citation_ids=["c2"]),
            SectionOut(
                title="What evidence exists?", body="A preprint evaluation [c1].", citation_ids=["c1", "c77"]
            ),
        ],
        claims=[
            ClaimOut(
                text="An independent preprint found Lumen-4 scored 84.3% on 1,200 held-out problems.",
                claim_type="fact",
                support_status="supported",
                citation_ids=["c1", "c99"],
            ),
            ClaimOut(
                text="Lumen Labs says Lumen-4 scores 92.1% on MathBench-Pro.",
                claim_type="fact",
                support_status="contested",
                citation_ids=["c2"],
            ),
            ClaimOut(
                text="Lumen-4 is used by most schools.",
                claim_type="fact",
                support_status="corroborated",
                citation_ids=["c42"],
            ),
        ],
        uncertainty=["The evaluation is a preprint and has not been peer reviewed [c1]."],
        disagreements=[
            DisagreementOut(
                topic="Lumen-4's math score",
                positions=[
                    DisagreementPosition(position="Lumen Labs reports 92.1%.", citation_ids=["c2"]),
                    DisagreementPosition(position="Open Eval measured 84.3%.", citation_ids=["c1", "c99"]),
                ],
            )
        ],
        watch_next=["Peer review of the evaluation [c1]."],
        risk_level="medium",
    ).model_dump_json()


def test_claude_brief_request_shape_and_invalid_citation_is_dropped(session, story):
    client = FakeClient(claude_brief())
    orchestrator = Orchestrator(session, provider=ClaudeProvider(model=MODEL, client=client))

    brief = generate_brief(session, story, orchestrator=orchestrator)

    calls = client.beta.messages.calls
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["model"] == MODEL
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"]["effort"] == "medium"
    fmt = kwargs["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert "claims" in fmt["schema"]["properties"]
    assert FALLBACK_BETA == "server-side-fallback-2026-07-01"
    assert FALLBACK_BETA in kwargs["betas"]
    assert kwargs["fallbacks"] == "default"
    system = kwargs["system"][0]["text"]
    assert "AI Story Brief" in system and "Never invent facts" in system
    prompt = kwargs["messages"][0]["content"]
    assert "<evidence>" in prompt and "Timeline" in prompt
    assert '<source id="c1">' in prompt and '<source id="c2">' in prompt
    assert "What is technically new?" in prompt  # the vertical's expandable sections are requested

    # Output came from Claude and was checked against the evidence pack.
    assert brief.generated_by == MODEL
    run = session.get(AgentRun, brief.agent_run_id)
    assert run.agent == AGENT_NAME and run.provider == "claude" and run.model == MODEL
    assert run.input_tokens == 1200 and run.output_tokens == 800 and run.cost_usd > 0
    assert {c["id"] for c in brief.citations} == {"c1", "c2"}

    first, second, invented = brief.claims
    assert first.citation_ids == ["c1"]  # c99 is not in the pack
    assert first.support_status == "supported"
    assert second.citation_ids == ["c2"]
    assert invented.citation_ids == [] and invented.support_status == "unverified"
    assert "[c99]" not in brief.summary and "[c1]" in brief.summary
    assert brief.sections[1]["citation_ids"] == ["c1"]
    assert brief.disagreements[0]["positions"][1]["citation_ids"] == ["c1"]
    checks = {c["name"]: c for c in brief.check_results}
    assert checks["unknown_citation_ids"]["passed"]
    assert brief.status == "in_review"


def test_claude_refusal_falls_back_to_the_offline_builder(session, story):
    client = FakeClient(json.dumps({"headline": "unused"}), stop_reason="refusal")
    orchestrator = Orchestrator(session, provider=ClaudeProvider(model=MODEL, client=client))

    brief = generate_brief(session, story, orchestrator=orchestrator)

    assert len(client.beta.messages.calls) == 1
    run = session.get(AgentRun, brief.agent_run_id)
    assert run.status == AgentRunStatus.REFUSED
    assert brief.generated_by == "offline"
    assert brief.claims and all(set(c.citation_ids) <= {"c1", "c2"} for c in brief.claims)
    assert "preprint" in " ".join([brief.summary, *brief.uncertainty]).lower()
