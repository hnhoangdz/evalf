import pytest

from evalf.evaluation import Evaluator, evaluate
from evalf.metrics.answer_relevance import AnswerRelevanceMetric
from evalf.metrics.answer_relevance.schema import AnswerRelevanceAssessment
from evalf.schemas import EvalCase
from tests.helpers import SequenceLLM

pytestmark = pytest.mark.unit


def test_evaluator_rejects_invalid_judge_type() -> None:
    with pytest.raises(TypeError, match="judge must be a BaseLLMModel instance or None"):
        Evaluator(judge="not-a-judge")


def test_evaluate_does_not_close_custom_judge_after_running() -> None:
    judge = SequenceLLM([AnswerRelevanceAssessment(score=1.0, reason="Trả lời trực tiếp.")])

    report = evaluate(
        cases=[EvalCase(id="case-1", question="FERPA là gì?", actual_output="Một luật bảo mật.")],
        metrics=[AnswerRelevanceMetric(threshold=0.7)],
        judge=judge,
        concurrency=1,
    )

    assert report.summary.total_samples == 1
    assert report.samples[0].status == "passed"
    assert judge.closed is False
    assert judge.aclosed is False
