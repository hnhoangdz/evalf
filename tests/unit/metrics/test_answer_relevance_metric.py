import pytest

from evalf.metrics.answer_relevance import AnswerRelevanceMetric
from evalf.metrics.answer_relevance.schema import AnswerRelevanceAssessment
from evalf.schemas import EvalCase
from tests.helpers import SequenceLLM

pytestmark = pytest.mark.unit


def test_answer_relevance_metric_skips_when_required_inputs_are_missing() -> None:
    metric = AnswerRelevanceMetric(threshold=0.7)

    result = metric.measure(EvalCase(), SequenceLLM([]))

    assert result.status == "skipped"
    assert result.score is None
    assert result.missing_inputs == ["actual_output", "question"]


def test_answer_relevance_metric_scores_single_attempt() -> None:
    metric = AnswerRelevanceMetric(threshold=0.7)
    llm = SequenceLLM(
        [
            AnswerRelevanceAssessment(
                score=1.0,
                reason="Câu trả lời đi thẳng vào nội dung người dùng hỏi.",
            )
        ]
    )

    result = metric.measure(
        EvalCase(
            question="What qualifies as a service animal under the ADA?",
            actual_output="A service animal under the ADA is a dog individually trained to perform tasks for a person with a disability.",
        ),
        llm,
    )

    assert result.status == "passed"
    assert result.score == 1.0
    assert result.trial_results[0].reason == "Câu trả lời đi thẳng vào nội dung người dùng hỏi."
