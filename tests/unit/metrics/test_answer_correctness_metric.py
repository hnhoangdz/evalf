import pytest

from evalf.metrics.answer_correctness import AnswerCorrectnessMetric
from evalf.metrics.answer_correctness.schema import AnswerCorrectnessAssessment
from evalf.schemas import EvalCase
from tests.helpers import SequenceLLM

pytestmark = pytest.mark.unit


def test_answer_correctness_metric_skips_when_required_inputs_are_missing() -> None:
    metric = AnswerCorrectnessMetric(threshold=0.7)

    result = metric.measure(EvalCase(), SequenceLLM([]))

    assert result.status == "skipped"
    assert result.score is None
    assert result.missing_inputs == ["actual_output", "expected_output", "question"]


def test_answer_correctness_metric_scores_single_attempt() -> None:
    metric = AnswerCorrectnessMetric(threshold=0.7)
    llm = SequenceLLM(
        [
            AnswerCorrectnessAssessment(
                score=0.75,
                reason="Câu trả lời khớp phần lớn nội dung kỳ vọng nhưng còn thiếu một ý nhỏ.",
            )
        ]
    )

    result = metric.measure(
        EvalCase(
            question="Under FERPA, when do rights transfer from parents to a student?",
            expected_output=(
                "Rights transfer when a student turns 18 or enters a postsecondary institution at any age."
            ),
            actual_output="Rights transfer when a student turns 18.",
        ),
        llm,
    )

    assert result.status == "passed"
    assert result.score == 0.75
    assert result.trial_results[0].reason is not None
    assert "thiếu một ý nhỏ" in result.trial_results[0].reason
