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

pytestmark = pytest.mark.integration


def test_faithfulness_metric_pipeline_runs_end_to_end() -> None:
    metric = FaithfulnessMetric(threshold=0.7)
    llm = SequenceLLM(
        [
            ClaimExtraction(
                claims=[
                    Claim(claim_id="c1", text="FERPA rights transfer at age 18."),
                    Claim(
                        claim_id="c2",
                        text="FERPA rights transfer only when the student turns 21.",
                    ),
                ]
            ),
            ClaimSupportAssessment(
                verdicts=[
                    ClaimSupportVerdict(
                        claim_id="c1",
                        verdict="supported",
                        evidence_context_ids=["ctx_1"],
                        reason="Supported by the FERPA context.",
                    ),
                    ClaimSupportVerdict(
                        claim_id="c2",
                        verdict="contradicted",
                        evidence_context_ids=["ctx_1"],
                        reason="The FERPA context says age 18, not 21.",
                    ),
                ]
            ),
        ]
    )

    result = metric.measure(
        EvalCase(
            id="case-faithfulness",
            question="When do FERPA rights transfer from parents to a student?",
            retrieved_contexts=["FERPA rights transfer when a student turns 18 years old."],
            actual_output="FERPA rights transfer at age 18, and only when the student turns 21.",
        ),
        llm,
    )

    assert result.status == "failed"
    assert result.score == pytest.approx(0.25)
    assert result.reason is not None
    assert result.trial_results[0].reason is not None
    assert "Supported 1/2 material claim(s)." in result.trial_results[0].reason
