"""Quiz grading: comprehension and transfer scored separately (education metric)."""

from __future__ import annotations

import pytest

from newsren.agents.learning import record_quiz_attempt
from newsren.models import GeneratedContent, QuizAttempt
from newsren.schemas import LearningOut, QuizQuestion


@pytest.fixture()
def quiz_content(session) -> GeneratedContent:
    out = LearningOut(
        audience="student",
        title="Quiz",
        explainer="Explainer.",
        quiz=[
            QuizQuestion(question="C1", options=["a", "b", "c"], answer_index=1),
            QuizQuestion(question="C2", options=["a", "b"], answer_index=0),
            QuizQuestion(question="T1", options=["a", "b", "c", "d"], answer_index=2, kind="transfer"),
            QuizQuestion(question="T2", options=["a", "b", "c"], answer_index=0, kind="transfer"),
        ],
    )
    row = GeneratedContent(kind="learning", cache_key="k", content=out.model_dump(), audience="student")
    session.add(row)
    session.flush()
    return row


def test_all_correct(session, quiz_content, factory):
    user = factory.user()
    attempt = record_quiz_attempt(session, quiz_content, [1, 0, 2, 0], user=user)
    assert isinstance(attempt, QuizAttempt) and attempt.id is not None
    assert (attempt.comprehension_correct, attempt.comprehension_total) == (2, 2)
    assert (attempt.transfer_correct, attempt.transfer_total) == (2, 2)
    assert attempt.user_id == user.id and attempt.content_id == quiz_content.id and attempt.condition == "newsren"


def test_mixed_and_unanswered(session, quiz_content):
    attempt = record_quiz_attempt(session, quiz_content, [1, 1, 2], condition="summary")
    assert (attempt.comprehension_correct, attempt.comprehension_total) == (1, 2)
    # The last transfer question was not answered: counted, but not correct.
    assert (attempt.transfer_correct, attempt.transfer_total) == (1, 2)
    assert attempt.condition == "summary" and attempt.user_id is None


def test_invalid_answers_count_as_wrong(session, quiz_content):
    attempt = record_quiz_attempt(session, quiz_content, [7, True, -1, 0])
    assert attempt.comprehension_correct == 0
    assert attempt.transfer_correct == 1


def test_rejects_bad_input(session, quiz_content):
    with pytest.raises(ValueError):
        record_quiz_attempt(session, quiz_content, [0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        record_quiz_attempt(session, quiz_content, [0], condition="control")
    other = GeneratedContent(kind="impact", cache_key="x", content={})
    session.add(other)
    session.flush()
    with pytest.raises(ValueError):
        record_quiz_attempt(session, other, [0])


def test_grades_generated_quiz(session, factory):
    import importlib.util
    import sys
    from pathlib import Path

    from newsren.agents.learning import learning_materials

    name = "explain_newsroom"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).parent / "data" / "explain" / "newsroom.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    brief = sys.modules[name].Newsroom(factory).brief("model_launch")
    row = learning_materials(session, brief, "student")
    quiz = LearningOut.model_validate(row.content).quiz
    attempt = record_quiz_attempt(session, row, [q.answer_index for q in quiz])
    kinds = [q.kind for q in quiz]
    assert attempt.comprehension_correct == attempt.comprehension_total == kinds.count("comprehension")
    assert attempt.transfer_correct == attempt.transfer_total == kinds.count("transfer") >= 2
