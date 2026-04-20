import pytest

from evalf.evaluation import evaluate
from evalf.metrics.answer_relevance import AnswerRelevanceMetric
from evalf.metrics.answer_relevance.schema import AnswerRelevanceAssessment
from evalf.schemas import EvalCase
from tests.helpers import SequenceLLM

pytestmark = pytest.mark.smoke


def test_evaluate_single_case_smoke() -> None:
    judge = SequenceLLM(
        [
            AnswerRelevanceAssessment(
                score=1.0,
                reason="Trả lời trực tiếp câu hỏi.",
            )
        ]
    )

    report = evaluate(
        cases=[EvalCase(id="case-1", question="FERPA là gì?", actual_output="Một luật bảo mật.")],
        metrics=[AnswerRelevanceMetric(threshold=0.7)],
        judge=judge,
        concurrency=1,
    )

    assert report.summary.total_samples == 1
    assert report.samples[0].status == "passed"
