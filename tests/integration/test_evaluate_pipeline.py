import pytest

from evalf.evaluation import evaluate
from evalf.metrics.answer_correctness import AnswerCorrectnessMetric
from evalf.metrics.answer_correctness.schema import AnswerCorrectnessAssessment
from evalf.metrics.answer_relevance import AnswerRelevanceMetric
from evalf.metrics.answer_relevance.schema import AnswerRelevanceAssessment
from evalf.schemas import EvalCase
from tests.helpers import SequenceLLM

pytestmark = pytest.mark.integration


def test_evaluate_runs_metrics_end_to_end_with_single_case() -> None:
    judge = SequenceLLM(
        [
            AnswerCorrectnessAssessment(
                score=1.0,
                reason="Nội dung khớp với đáp án kỳ vọng.",
            ),
            AnswerRelevanceAssessment(
                score=0.75,
                reason="Câu trả lời nhìn chung đúng trọng tâm nhưng còn hơi dư chi tiết.",
            ),
        ]
    )

    report = evaluate(
        cases=[
            EvalCase(
                id="case-1",
                question="Under FERPA, when do rights transfer from parents to a student?",
                expected_output="Rights transfer when a student turns 18 or enters a postsecondary institution at any age.",
                actual_output="Rights transfer when a student turns 18 or enters a postsecondary institution at any age.",
            )
        ],
        metrics=[
            AnswerCorrectnessMetric(threshold=0.7),
            AnswerRelevanceMetric(threshold=0.7),
        ],
        judge=judge,
        concurrency=1,
    )

    assert report.summary.total_samples == 1
    assert report.summary.passed_samples == 1
    assert report.samples[0].status == "passed"
    assert [metric.name for metric in report.samples[0].metrics] == [
        "answer_correctness",
        "answer_relevance",
    ]
    assert report.samples[0].metrics[0].score == 1.0
    assert report.samples[0].metrics[1].score == 0.75
