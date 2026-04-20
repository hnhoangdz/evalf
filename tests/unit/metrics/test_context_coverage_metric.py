import asyncio

import pytest

from evalf.metrics.context_coverage import ContextCoverageMetric
from evalf.metrics.context_coverage.schema import ContextCoverageAssessment
from evalf.schemas import EvalCase
from tests.helpers import SequenceLLM

pytestmark = pytest.mark.unit


def _make_case(
    *,
    question: str = "When do FERPA rights transfer?",
    retrieved_contexts: list[str] | None = None,
    reference_contexts: list[str] | None = None,
) -> EvalCase:
    return EvalCase(
        question=question,
        retrieved_contexts=retrieved_contexts
        or ["FERPA rights transfer when a student turns 18 years old."],
        reference_contexts=reference_contexts
        or [
            "FERPA rights transfer when a student turns 18 years old "
            "or enters a postsecondary institution at any age."
        ],
    )


def _make_assessment(
    score: float = 1.0,
    verdict: str = "yes",
    reason: str = "All key info covered.",
) -> ContextCoverageAssessment:
    return ContextCoverageAssessment(score=score, verdict=verdict, reason=reason)


def test_skips_when_required_inputs_are_missing() -> None:
    metric = ContextCoverageMetric(threshold=0.7)

    result = metric.measure(EvalCase(), SequenceLLM([]))

    assert result.status == "skipped"
    assert result.score is None
    assert result.missing_inputs == ["question", "reference_contexts", "retrieved_contexts"]


def test_skips_when_retrieved_contexts_is_empty() -> None:
    metric = ContextCoverageMetric(threshold=0.7)
    case = EvalCase(
        question="Some question",
        retrieved_contexts=[],
        reference_contexts=["Some context"],
    )

    result = metric.measure(case, SequenceLLM([]))

    assert result.status == "skipped"
    assert "retrieved_contexts" in result.missing_inputs


def test_scores_full_coverage() -> None:
    metric = ContextCoverageMetric(threshold=0.7)
    llm = SequenceLLM([_make_assessment(score=1.0, verdict="yes")])

    result = metric.measure(_make_case(), llm)

    assert result.status == "passed"
    assert result.score == 1.0
    assert result.trial_results[0].reason == "All key info covered."


def test_scores_partial_coverage() -> None:
    metric = ContextCoverageMetric(threshold=0.7)
    llm = SequenceLLM(
        [
            _make_assessment(
                score=0.5,
                verdict="no",
                reason="Missing training requirement for service animal.",
            )
        ]
    )

    result = metric.measure(_make_case(), llm)

    assert result.status == "failed"
    assert result.score == pytest.approx(0.5)
    assert result.passed is False


def test_scores_zero_coverage() -> None:
    metric = ContextCoverageMetric(threshold=0.7)
    llm = SequenceLLM([_make_assessment(score=0.0, verdict="no", reason="Completely unrelated.")])

    result = metric.measure(_make_case(), llm)

    assert result.status == "failed"
    assert result.score == 0.0


def test_strict_mode_zeroes_score_below_threshold() -> None:
    metric = ContextCoverageMetric(threshold=0.7, strict_mode=True)
    llm = SequenceLLM([_make_assessment(score=0.5, verdict="no", reason="Partial.")])

    result = metric.measure(_make_case(), llm)

    assert result.score == 0.0
    assert result.passed is False


def test_strict_mode_preserves_score_above_threshold() -> None:
    metric = ContextCoverageMetric(threshold=0.7, strict_mode=True)
    llm = SequenceLLM([_make_assessment(score=0.85, verdict="yes", reason="Good.")])

    result = metric.measure(_make_case(), llm)

    assert result.score == pytest.approx(0.85)
    assert result.passed is True


def test_strict_mode_disabled_keeps_original_score() -> None:
    metric = ContextCoverageMetric(threshold=0.7, strict_mode=False)
    llm = SequenceLLM([_make_assessment(score=0.5, verdict="no", reason="Partial.")])

    result = metric.measure(_make_case(), llm)

    assert result.score == pytest.approx(0.5)
    assert result.passed is False


def test_async_measure_scores_correctly() -> None:
    metric = ContextCoverageMetric(threshold=0.7)
    llm = SequenceLLM([_make_assessment(score=0.75, verdict="yes", reason="Mostly covered.")])

    result = asyncio.run(metric.a_measure(_make_case(), llm))

    assert result.status == "passed"
    assert result.score == pytest.approx(0.75)


def test_clamps_score_above_one() -> None:
    metric = ContextCoverageMetric(threshold=0.7)
    llm = SequenceLLM([_make_assessment(score=1.0, verdict="yes", reason="Perfect.")])

    result = metric.measure(_make_case(), llm)

    assert result.score <= 1.0


def test_metric_name() -> None:
    metric = ContextCoverageMetric()
    assert metric.name == "context_coverage"


def test_required_inputs() -> None:
    metric = ContextCoverageMetric()
    assert set(metric.required_inputs) == {"question", "retrieved_contexts", "reference_contexts"}
