import pytest

from evalf.metrics.context_relevance import ContextRelevanceMetric
from evalf.metrics.decomposition import ContextRelevanceVerdict, ContextRelevanceVerdictList
from evalf.schemas import EvalCase
from tests.helpers import SequenceLLM

pytestmark = pytest.mark.unit


def test_context_relevance_metric_skips_when_required_inputs_are_missing() -> None:
    metric = ContextRelevanceMetric(threshold=0.7)

    result = metric.measure(EvalCase(), SequenceLLM([]))

    assert result.status == "skipped"
    assert result.score is None
    assert result.missing_inputs == ["question", "retrieved_contexts"]


def test_context_relevance_metric_scores_context_verdicts() -> None:
    metric = ContextRelevanceMetric(threshold=0.7)
    llm = SequenceLLM(
        [
            ContextRelevanceVerdictList(
                verdicts=[
                    ContextRelevanceVerdict(
                        context_id="ctx_1",
                        verdict="relevant",
                        reason="Trực tiếp hỗ trợ câu hỏi.",
                    ),
                    ContextRelevanceVerdict(
                        context_id="ctx_2",
                        verdict="partially_relevant",
                        reason="Có liên quan nhưng chưa đầy đủ.",
                    ),
                ]
            )
        ]
    )

    result = metric.measure(
        EvalCase(
            question="When do FERPA rights transfer from parents to a student?",
            retrieved_contexts=[
                "FERPA rights transfer when a student turns 18 years old.",
                "FERPA is a federal privacy law.",
            ],
        ),
        llm,
    )

    assert result.status == "passed"
    assert result.score == pytest.approx(0.75)
    assert result.trial_results[0].reason is not None
    assert "Relevant ranks: 1." in result.trial_results[0].reason
