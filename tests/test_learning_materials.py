"""News-to-Learning: audience-specific material built only from verified, cited facts (offline mode)."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
from sqlalchemy import select

from newsren.agents.base import Orchestrator
from newsren.agents.learning import (
    AGENT_NAME,
    LESSONS,
    TERMS,
    learning_context,
    learning_materials,
    sanitize_learning,
    strip_citation_marks,
)
from newsren.editorial.guardrails import lint_text
from newsren.finance.concepts import CONCEPTS
from newsren.llm.base import LLMResult
from newsren.models import AgentRun, GeneratedContent
from newsren.schemas import LearningOut, QuizQuestion


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


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", strip_citation_marks(text)).lower()


def _content(row: GeneratedContent) -> LearningOut:
    return LearningOut.model_validate(row.content)


def _pack_ids(row: GeneratedContent) -> set[str]:
    return {c["id"] for c in row.citations}


def test_student_material_structure_and_quiz(session, newsroom):
    brief = newsroom.brief("learning_study")
    row = learning_materials(session, brief, "student")
    out = _content(row)

    assert row.kind == "learning" and row.audience == "student" and row.brief_id == brief.id
    assert out.audience == "student"
    assert out.title.startswith("Explained for students")
    assert 1 <= out.reading_time_minutes <= 6
    assert "What happened" in out.explainer and "Think of it like this" in out.explainer
    # Plain language: the explainer's narrative sentences are short on average.
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", strip_citation_marks(out.explainer)) if s]
    assert sum(len(s.split()) for s in sentences) / len(sentences) < 26

    assert 4 <= len(out.quiz) <= 6
    transfer = [q for q in out.quiz if q.kind == "transfer"]
    comprehension = [q for q in out.quiz if q.kind == "comprehension"]
    assert len(transfer) >= 2 and comprehension
    for q in out.quiz:
        assert 0 <= q.answer_index < len(q.options) and len(set(q.options)) == len(q.options)
        assert q.explanation

    # Comprehension questions are answerable from the explainer (or the cited sources).
    explainer = _norm(out.explainer)
    sources = _norm(" ".join(a.content + " " + (a.summary or "") for a in newsroom.articles.values()))
    for q in comprehension:
        answer = q.options[q.answer_index].lower()
        assert answer in explainer or answer in sources, q.question

    # Speed versus transfer is what this story teaches; a transfer question applies it to a new situation.
    assert any("unfamiliar" in q.options[q.answer_index] for q in transfer)
    assert out.glossary and all(g.definition for g in out.glossary)
    assert {"Transfer", "Preprint"} <= {g.term for g in out.glossary}
    assert out.discussion_questions


def test_only_verified_facts_are_taught(session, newsroom):
    brief = newsroom.brief("learning_study")
    row = learning_materials(session, brief, "student")
    out = _content(row)

    # The pack holds only articles cited by verified claims (the press release backs only a contested claim).
    assert _pack_ids(row) == {"c2", "c3"}
    assert set(out.citation_ids) <= _pack_ids(row)
    assert set(re.findall(r"\[(c\d+)\]", out.explainer)) <= _pack_ids(row)

    facts_block = out.explainer.split("The facts our sources support", 1)[1].split("\n\n")[1]
    assert "22 minutes versus 32 minutes" in facts_block
    assert "learn algebra 30% faster" not in facts_block.lower()
    assert "opinion" not in facts_block.lower()
    # The contested claim is presented only as an open question.
    open_block = out.explainer.split("What we don't know yet", 1)[1]
    assert "Open question, because sources disagree" in open_block
    assert "StepWise helped ninth graders learn algebra 30% faster" in open_block
    # Preprint status reaches the reader.
    assert "preprint" in out.explainer.lower()


def test_glossary_never_invents_definitions(session, newsroom):
    brief = newsroom.brief("model_launch")
    row = learning_materials(session, brief, "student")
    out = _content(row)
    pack_text = " ".join(_norm(a.content) for a in newsroom.articles.values())
    standard = {t.term for t in TERMS} | {c.term for c in CONCEPTS} | {lesson.term for lesson in LESSONS}
    sourced = [item for item in out.glossary if item.term not in standard]
    assert sourced, "expected at least one name defined by a source (e.g. MathBench-Pro)"
    for item in out.glossary:
        if item.term in standard:
            continue
        # Any other entry is a name defined by a source: it cites that source, whose text contains the definition.
        cids = re.findall(r"\[(c\d+)\]", item.definition)
        assert cids and set(cids) <= _pack_ids(row), item
        core = _norm(item.definition).rstrip(".").split(" ", 1)[1]
        assert core[:25] in pack_text, item


def test_educator_guide_has_age_bands_activities_and_misconceptions(session, newsroom):
    brief = newsroom.brief("learning_study")
    out = _content(learning_materials(session, brief, "educator"))

    assert out.audience == "educator" and out.title.startswith("Classroom guide")
    for band in ("Middle school:", "High school:", "University:"):
        assert any(q.startswith(band) for q in out.discussion_questions), band
    assert 1 <= len(out.classroom_activities) <= 2
    for activity in out.classroom_activities:
        assert activity.title and activity.steps and activity.materials and activity.duration_minutes > 0
    assert any("Faster or deeper" in a.title for a in out.classroom_activities)
    assert "Misconceptions to address" in out.explainer
    assert "Finishing work faster means learning more" in out.explainer
    assert "Open questions" in out.explainer


def test_professional_material_has_implications_and_advanced_reading(session, newsroom):
    brief = newsroom.brief("semiconductor")
    row = learning_materials(session, brief, "professional")
    out = _content(row)

    assert out.audience == "professional" and out.professional_implications
    assert any(i.startswith("Documented:") for i in out.professional_implications)
    assert any("reported output" in i for i in out.professional_implications)
    assert out.advanced_reading.startswith("For readers who want the technical detail")
    assert set(re.findall(r"\[(c\d+)\]", out.advanced_reading)) <= _pack_ids(row)
    assert "11,500" in out.explainer
    assert any(g.term == "Wafer" for g in out.glossary)


def test_finance_learning_is_educational_and_lint_clean(session, newsroom):
    brief = newsroom.brief("backtest")
    for audience in ("student", "educator", "professional"):
        out = _content(learning_materials(session, brief, audience))
        assert "not investment advice" in out.explainer
        text = " ".join([out.explainer, out.advanced_reading, *out.professional_implications])
        assert lint_text(text, "finance") == [], audience
    student = _content(learning_materials(session, brief, "student"))
    assert any("past data" in q.options[q.answer_index] for q in student.quiz if q.kind == "transfer")
    assert any(g.term == "Backtesting" for g in student.glossary)


def test_cached_per_audience_and_version(session, newsroom):
    brief = newsroom.brief("policy")
    first = learning_materials(session, brief, "student")
    assert learning_materials(session, brief, "student").id == first.id
    assert learning_materials(session, brief, "educator").id != first.id
    assert learning_materials(session, brief, "student", refresh=True).id != first.id
    brief.version = 2
    assert learning_materials(session, brief, "student").id != first.id
    runs = session.scalars(select(AgentRun).where(AgentRun.agent == AGENT_NAME)).all()
    assert len(runs) == 4 and all(r.subject_type == "brief" and r.subject_id == brief.id for r in runs)


def test_policy_unverified_claim_is_an_open_question(session, newsroom):
    brief = newsroom.brief("policy")
    out = _content(learning_materials(session, brief, "student"))
    assert "Open question, not yet confirmed by the sources" in out.explainer
    assert any(q.kind == "transfer" and "phased timeline" in q.options[q.answer_index] for q in out.quiz)


def test_unknown_audience_rejected(session, newsroom):
    brief = newsroom.brief("policy")
    with pytest.raises(ValueError):
        learning_materials(session, brief, "parent")


class _FakeClaude:
    """Stands in for Claude: returns a fixed structured output (no network)."""

    name = "claude"
    model = "fake-claude"

    def __init__(self, output: LearningOut):
        self.output = output

    def generate(self, task, budget):
        return LLMResult(output=self.output, provider="claude", model=self.model)

    def run_tools(self, task, budget):  # pragma: no cover - not used by this agent
        raise NotImplementedError


def test_model_output_is_sanitized(session, newsroom):
    brief = newsroom.brief("learning_study")
    bad = LearningOut(
        audience="educator",
        title="Learning: StepWise",
        explainer="Students finished faster [c2][c9]. Invented source [c42].",
        quiz=[
            QuizQuestion(question="Broken", options=["a"], answer_index=3),
            QuizQuestion(question="Which was measured?", options=["speed", "mood"], answer_index=0),
        ],
        citation_ids=["c2", "c9"],
    )
    row = learning_materials(
        session, brief, "student", orchestrator=Orchestrator(session, provider=_FakeClaude(bad))
    )
    out = _content(row)
    assert row.generated_by == "fake-claude"
    assert out.audience == "student"
    assert "[c9]" not in out.explainer and "[c42]" not in out.explainer and "[c2]" in out.explainer
    assert out.citation_ids == ["c2"]
    assert all(q.question != "Broken" for q in out.quiz)
    assert sum(q.kind == "transfer" for q in out.quiz) >= 2 and len(out.quiz) >= 4


def test_sanitize_adds_finance_note(session, newsroom):
    brief = newsroom.brief("backtest")
    ctx = learning_context(session, brief, "professional")
    out = sanitize_learning(
        LearningOut(audience="professional", title="t", explainer="Backtests are simulations [c1]."), ctx
    )
    assert out.explainer.endswith("not investment advice.")
