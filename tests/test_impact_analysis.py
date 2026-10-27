"""Industry Impact: stakeholders with effects labelled supported (cited), speculative, or scenario (offline mode)."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from newsren.agents.base import Orchestrator
from newsren.agents.impact import (
    AGENT_NAME,
    DEFAULT_QUESTIONS,
    impact_context,
    industry_impact,
    sanitize_impact,
)
from newsren.agents.learning import strip_citation_marks
from newsren.editorial.guardrails import FINANCE_DISCLAIMER, lint_text
from newsren.llm.base import LLMResult
from newsren.models import AgentRun, GeneratedContent
from newsren.schemas import EffectOut, ImpactOut, StakeholderOut


def _load_newsroom():
    name = "explain_newsroom"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).parent / "data" / "explain" / "newsroom.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture()
def newsroom(factory):
    return _load_newsroom().Newsroom(factory)


def _out(row: GeneratedContent) -> ImpactOut:
    return ImpactOut.model_validate(row.content)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", strip_citation_marks(text)).lower()


def _cited_text(row: GeneratedContent, newsroom, cid: str) -> str:
    citation = next(c for c in row.citations if c["id"] == cid)
    article = next(a for a in newsroom.articles.values() if a.id == citation["article_id"])
    return _norm(f"{article.title} {article.summary} {article.content}")


def _check_labelling(row: GeneratedContent, newsroom) -> ImpactOut:
    out = _out(row)
    valid = {c["id"] for c in row.citations}
    assert out.stakeholders
    for stakeholder in out.stakeholders:
        assert stakeholder.name and stakeholder.effects
        for effect in stakeholder.effects:
            assert set(effect.citation_ids) <= valid
            if effect.evidence_status == "supported":
                # A supported effect is stated by the cited source itself.
                assert effect.citation_ids, effect
                statement = _norm(effect.description.split(": ", 1)[-1]).rstrip(".")
                assert any(statement[:60] in _cited_text(row, newsroom, c) for c in effect.citation_ids), effect
            elif effect.evidence_status == "scenario":
                assert effect.description.startswith("Scenario") and not effect.citation_ids
            else:
                assert effect.description.startswith(("Projection", "Opinion", "Gap in the evidence")), effect
    return out


def test_industry_default_question_and_labels(session, newsroom):
    brief = newsroom.brief("semiconductor")
    row = industry_impact(session, brief)
    out = _check_labelling(row, newsroom)

    assert row.kind == "impact" and out.question == DEFAULT_QUESTIONS["industry"]
    statuses = {e.evidence_status for s in out.stakeholders for e in s.effects}
    assert statuses == {"supported", "speculative", "scenario"}
    names = {s.name for s in out.stakeholders}
    assert {"Equipment suppliers", "Chipmakers and foundries"} <= names
    kinds = {s.kind for s in out.stakeholders}
    assert "occupation" in kinds and "industry" in kinds
    # The vendor's capacity target is a projection, not a supported effect.
    projections = [e for s in out.stakeholders for e in s.effects if "40,000" in e.description]
    assert projections and all(e.evidence_status == "speculative" for e in projections)
    assert any("late 2027" in e.time_horizon for e in projections)
    assert out.open_questions and any(q.startswith("Unresolved in the sources") for q in out.open_questions)
    assert any("causation" in c for c in out.caveats)
    assert any("not what will happen" in c for c in out.caveats)


def test_reader_named_groups_become_stakeholders(session, newsroom):
    brief = newsroom.brief("model_launch")
    question = "How could on-device AI affect smartphone makers, chip suppliers, app developers, and privacy?"
    out = _check_labelling(industry_impact(session, brief, question), newsroom)
    names = [s.name for s in out.stakeholders]
    assert out.question == question
    assert names[:4] == [
        "Smartphone and device makers",
        "Chipmakers and chip suppliers",
        "App and software developers",
        "Consumers and privacy",
    ]
    # Groups the evidence never mentions are reported as gaps, not given invented effects.
    smartphone = out.stakeholders[0]
    assert [e.evidence_status for e in smartphone.effects] == ["speculative"]
    assert smartphone.effects[0].description.startswith("Gap in the evidence")
    developers = out.stakeholders[2]
    assert any(e.evidence_status in ("supported", "speculative") and e.citation_ids for e in developers.effects)


def test_education_default_question(session, newsroom):
    brief = newsroom.brief("policy")
    out = _check_labelling(industry_impact(session, brief), newsroom)
    assert out.question == "Who is affected and how?"
    names = {s.name for s in out.stakeholders}
    assert {"Students", "Teachers"} <= names
    assert any("2027" in e.time_horizon or "2028" in e.time_horizon for s in out.stakeholders for e in s.effects)
    assert any("generalize" in c for c in out.caveats)


def test_finance_impact_is_not_advice(session, newsroom):
    brief = newsroom.brief("backtest")
    row = industry_impact(session, brief)
    out = _check_labelling(row, newsroom)
    assert out.question == DEFAULT_QUESTIONS["finance"]
    names = [s.name for s in out.stakeholders]
    assert names[:3] == ["Investors", "Founders and startups", "Finance professionals (advisers and analysts)"]
    assert FINANCE_DISCLAIMER in out.caveats
    text = " ".join(
        [
            out.summary,
            *out.open_questions,
            *out.caveats,
            *(e.description for s in out.stakeholders for e in s.effects),
        ]
    )
    assert lint_text(text, "finance") == []


def test_cached_per_question(session, newsroom):
    brief = newsroom.brief("semiconductor")
    first = industry_impact(session, brief)
    assert industry_impact(session, brief).id == first.id
    assert industry_impact(session, brief, DEFAULT_QUESTIONS["industry"]).id == first.id
    other = industry_impact(session, brief, "Who supplies the tools")
    assert other.id != first.id and _out(other).question == "Who supplies the tools?"
    assert session.query(AgentRun).filter(AgentRun.agent == AGENT_NAME).count() == 2


def test_sanitize_downgrades_unsupported_effects(session, newsroom):
    brief = newsroom.brief("semiconductor")
    ctx = impact_context(session, brief, DEFAULT_QUESTIONS["industry"])
    raw = ImpactOut(
        question="ignored",
        summary="Summary [c99].",
        stakeholders=[
            StakeholderOut(
                name="Chip suppliers",
                effects=[
                    EffectOut(
                        description="Lead times grew [c4].", evidence_status="supported", citation_ids=["c4"]
                    ),
                    EffectOut(
                        description="Prices will double.", evidence_status="supported", citation_ids=["c99"]
                    ),
                    EffectOut(description="If ramps slip, orders could fall.", evidence_status="scenario"),
                    EffectOut(
                        description="Wafer prices will double next year [c2].",
                        evidence_status="supported",
                        citation_ids=["c2"],
                    ),
                ],
            ),
            StakeholderOut(name="Empty", effects=[]),
        ],
        open_questions=["What was ordered? [c42]"],
        caveats=[],
    )
    out = sanitize_impact(raw, ctx)
    effects = out.stakeholders[0].effects
    # A forecast written as fact is not a supported effect, even with a valid citation.
    assert [e.evidence_status for e in effects] == ["supported", "speculative", "scenario", "speculative"]
    assert effects[3].citation_ids == ["c2"]
    assert effects[1].citation_ids == [] and effects[2].description.startswith("Scenario:")
    assert [s.name for s in out.stakeholders] == ["Chip suppliers"]
    assert out.summary == "Summary." and out.open_questions == ["What was ordered?"]
    assert out.question == DEFAULT_QUESTIONS["industry"]
    assert any("causation" in c for c in out.caveats)


class _FakeClaude:
    name = "claude"
    model = "fake-claude"

    def __init__(self, output: ImpactOut):
        self.output = output

    def generate(self, task, budget):
        return LLMResult(output=self.output, provider="claude", model=self.model)

    def run_tools(self, task, budget):  # pragma: no cover - not used by this agent
        raise NotImplementedError


def test_finance_model_output_drops_advice(session, newsroom):
    brief = newsroom.brief("backtest")
    raw = ImpactOut(
        question="q",
        summary="Northbridge ran a backtest [c1].",
        stakeholders=[
            StakeholderOut(
                name="Investors",
                kind="public",
                effects=[
                    EffectOut(
                        description="Investors should buy Northbridge shares now.", evidence_status="speculative"
                    ),
                    EffectOut(
                        description="The agents manage no client money.",
                        evidence_status="supported",
                        citation_ids=["c2"],
                    ),
                ],
            )
        ],
        caveats=["Shares will soar after this."],
    )
    row = industry_impact(session, brief, orchestrator=Orchestrator(session, provider=_FakeClaude(raw)))
    out = _out(row)
    descriptions = [e.description for s in out.stakeholders for e in s.effects]
    assert descriptions == ["The agents manage no client money."]
    assert FINANCE_DISCLAIMER in out.caveats and "Shares will soar after this." not in out.caveats
    assert row.generated_by == "fake-claude"
