import asyncio

import pytest

from evalf.metrics.context_precision import ContextPrecisionMetric
from evalf.metrics.decomposition import (
    Claim,
    ClaimExtraction,
    ContextCoverageAssessment,
    ContextCoverageVerdict,
)
from evalf.schemas import EvalCase
from tests.helpers import SequenceLLM

pytestmark = pytest.mark.unit


def test_context_precision_metric_skips_when_required_inputs_are_missing() -> None:
    metric = ContextPrecisionMetric(threshold=0.7)

    result = metric.measure(EvalCase(), SequenceLLM([]))

    assert result.status == "skipped"
    assert result.score is None
    assert result.missing_inputs == ["question", "reference_contexts", "retrieved_contexts"]


def test_context_precision_metric_scores_ranked_context_usefulness() -> None:
    metric = ContextPrecisionMetric(threshold=0.7)
    llm = SequenceLLM(
        [
            ClaimExtraction(
                claims=[
                    Claim(claim_id="rc1", text="Rights transfer at age 18."),
                    Claim(
                        claim_id="rc2",
                        text="Rights also transfer upon entering a postsecondary institution.",
                    ),
                ]
            ),
            ContextCoverageAssessment(
                contexts=[
                    ContextCoverageVerdict(
                        context_id="ctx_1",
                        supported_claim_ids=["rc1"],
                        reason="Hỗ trợ claim đầu tiên.",
                    ),
                    ContextCoverageVerdict(
                        context_id="ctx_2",
                        supported_claim_ids=[],
                        reason="Nhiễu, không hỗ trợ claim nào.",
                    ),
                ]
            ),
        ]
    )

    result = metric.measure(
        EvalCase(
            question="When do FERPA rights transfer from parents to a student?",
            retrieved_contexts=[
                "FERPA rights transfer when a student turns 18 years old.",
                "FERPA is a privacy law for student records.",
            ],
            reference_contexts=[
                "FERPA rights transfer when a student turns 18 years old or enters a postsecondary institution at any age."
            ],
        ),
        llm,
    )

    assert result.status == "failed"
    assert result.score == pytest.approx(0.5)
    assert result.trial_results[0].reason is not None
    assert (
        "Retrieved contexts first covered 1/2 reference claim(s)." in result.trial_results[0].reason
    )


def test_context_precision_metric_async_scores_ranked_context_usefulness() -> None:
    metric = ContextPrecisionMetric(threshold=0.7)
    llm = SequenceLLM(
        [
            ClaimExtraction(
                claims=[
                    Claim(claim_id="rc1", text="Rights transfer at age 18."),
                    Claim(
                        claim_id="rc2",
                        text="Rights also transfer upon entering a postsecondary institution.",
                    ),
                ]
            ),
            ContextCoverageAssessment(
                contexts=[
                    ContextCoverageVerdict(
                        context_id="ctx_1",
                        supported_claim_ids=["rc1"],
                        reason="Hỗ trợ claim đầu tiên.",
                    ),
                    ContextCoverageVerdict(
                        context_id="ctx_2",
                        supported_claim_ids=[],
                        reason="Nhiễu, không hỗ trợ claim nào.",
                    ),
                ]
            ),
        ]
    )

    result = asyncio.run(
        metric.a_measure(
            EvalCase(
                question="When do FERPA rights transfer from parents to a student?",
                retrieved_contexts=[
                    "FERPA rights transfer when a student turns 18 years old.",
                    "FERPA is a privacy law for student records.",
                ],
                reference_contexts=[
                    "FERPA rights transfer when a student turns 18 years old or enters a postsecondary institution at any age."
                ],
            ),
            llm,
        )
    )

    assert result.status == "failed"
    assert result.score == pytest.approx(0.5)
    assert "Useful ranks: 1." in (result.trial_results[0].reason or "")
