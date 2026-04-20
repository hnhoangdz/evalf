import asyncio

import pytest

from evalf.metrics.decomposition import (
    Claim,
    ClaimExtraction,
    ClaimSupportAssessment,
    ClaimSupportVerdict,
)
from evalf.metrics.faithfulness import FaithfulnessMetric
from evalf.schemas import EvalCase
from tests.helpers import SequenceLLM

pytestmark = pytest.mark.unit


def test_faithfulness_metric_skips_when_required_inputs_are_missing() -> None:
    metric = FaithfulnessMetric(threshold=0.7)

    result = metric.measure(EvalCase(), SequenceLLM([]))

    assert result.status == "skipped"
    assert result.score is None
    assert result.missing_inputs == ["actual_output", "question", "retrieved_contexts"]


def test_faithfulness_metric_scores_supported_and_contradicted_claims() -> None:
    metric = FaithfulnessMetric(threshold=0.7)
    llm = SequenceLLM(
        [
            ClaimExtraction(
                claims=[
                    Claim(claim_id="c1", text="FERPA rights transfer at age 18."),
                    Claim(claim_id="c2", text="FERPA rights transfer only at age 21."),
                ]
            ),
            ClaimSupportAssessment(
                verdicts=[
                    ClaimSupportVerdict(
                        claim_id="c1",
                        verdict="supported",
                        evidence_context_ids=["ctx_1"],
                        reason="Được hỗ trợ bởi context.",
                    ),
                    ClaimSupportVerdict(
                        claim_id="c2",
                        verdict="contradicted",
                        evidence_context_ids=["ctx_1"],
                        reason="Bị mâu thuẫn với context.",
                    ),
                ]
            ),
        ]
    )

    result = metric.measure(
        EvalCase(
            question="When do FERPA rights transfer from parents to a student?",
            retrieved_contexts=["FERPA rights transfer when a student turns 18 years old."],
            actual_output="FERPA rights transfer at age 18, and only at age 21.",
        ),
        llm,
    )

    assert result.status == "failed"
    assert result.score == pytest.approx(0.25)
    assert result.trial_results[0].reason is not None
    assert "Supported 1/2 material claim(s)." in result.trial_results[0].reason
    assert "extra 0.5 penalty" in result.trial_results[0].reason


def test_faithfulness_metric_async_scores_supported_and_contradicted_claims() -> None:
    metric = FaithfulnessMetric(threshold=0.7)
    llm = SequenceLLM(
        [
            ClaimExtraction(
                claims=[
                    Claim(claim_id="c1", text="FERPA rights transfer at age 18."),
                    Claim(claim_id="c2", text="FERPA rights transfer only at age 21."),
                ]
            ),
            ClaimSupportAssessment(
                verdicts=[
                    ClaimSupportVerdict(
                        claim_id="c1",
                        verdict="supported",
                        evidence_context_ids=["ctx_1"],
                        reason="Được hỗ trợ bởi context.",
                    ),
                    ClaimSupportVerdict(
                        claim_id="c2",
                        verdict="contradicted",
                        evidence_context_ids=["ctx_1"],
                        reason="Bị mâu thuẫn với context.",
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
                actual_output="FERPA rights transfer at age 18, and only at age 21.",
            ),
            llm,
        )
    )

    assert result.status == "failed"
    assert result.score == pytest.approx(0.25)
    assert "extra 0.5 penalty" in (result.trial_results[0].reason or "")
