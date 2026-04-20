import asyncio

import pytest

from evalf.metrics.context_recall import ContextRecallMetric
from evalf.metrics.decomposition import (
    Claim,
    ClaimCoverageAssessment,
    ClaimCoverageVerdict,
    ClaimExtraction,
)
from evalf.schemas import EvalCase
from tests.helpers import SequenceLLM

pytestmark = pytest.mark.unit


def test_context_recall_metric_skips_when_required_inputs_are_missing() -> None:
    metric = ContextRecallMetric(threshold=0.7)

    result = metric.measure(EvalCase(), SequenceLLM([]))

    assert result.status == "skipped"
    assert result.score is None
    assert result.missing_inputs == ["question", "reference_contexts", "retrieved_contexts"]


def test_context_recall_metric_scores_reference_claim_coverage() -> None:
    metric = ContextRecallMetric(threshold=0.7)
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
            ClaimCoverageAssessment(
                verdicts=[
                    ClaimCoverageVerdict(
                        claim_id="rc1",
                        verdict="supported",
                        evidence_context_ids=["ctx_1"],
                        reason="Có trong retrieved contexts.",
                    ),
                    ClaimCoverageVerdict(
                        claim_id="rc2",
                        verdict="unsupported",
                        evidence_context_ids=[],
                        reason="Không có trong retrieved contexts.",
                    ),
                ]
            ),
        ]
    )

    result = metric.measure(
        EvalCase(
            question="When do FERPA rights transfer from parents to a student?",
            retrieved_contexts=["FERPA rights transfer when a student turns 18 years old."],
            reference_contexts=[
                "FERPA rights transfer when a student turns 18 years old or enters a postsecondary institution at any age."
            ],
        ),
        llm,
    )

    assert result.status == "failed"
    assert result.score == pytest.approx(0.5)
    assert result.trial_results[0].reason is not None
    assert "Retrieved contexts cover 1/2 reference claim(s)." in result.trial_results[0].reason


def test_context_recall_metric_async_scores_reference_claim_coverage() -> None:
    metric = ContextRecallMetric(threshold=0.7)
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
            ClaimCoverageAssessment(
                verdicts=[
                    ClaimCoverageVerdict(
                        claim_id="rc1",
                        verdict="supported",
                        evidence_context_ids=["ctx_1"],
                        reason="Có trong retrieved contexts.",
                    ),
                    ClaimCoverageVerdict(
                        claim_id="rc2",
                        verdict="unsupported",
                        evidence_context_ids=[],
                        reason="Không có trong retrieved contexts.",
                    ),
                ]
            ),
        ]
    )

    result = asyncio.run(
        metric.a_measure(
            EvalCase(
                question="When do FERPA rights transfer from parents to a student?",
                retrieved_contexts=["FERPA rights transfer when a student turns 18 years old."],
                reference_contexts=[
                    "FERPA rights transfer when a student turns 18 years old or enters a postsecondary institution at any age."
                ],
            ),
            llm,
        )
    )

    assert result.status == "failed"
    assert result.score == pytest.approx(0.5)
    assert "cover 1/2 reference claim(s)" in (result.trial_results[0].reason or "")
