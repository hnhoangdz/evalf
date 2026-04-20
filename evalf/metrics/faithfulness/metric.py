from __future__ import annotations

from evalf.llms.base import BaseLLMModel
from evalf.metrics.base import BaseDecomposedMetric
from evalf.metrics.decomposition import Claim, ClaimExtraction, ClaimSupportAssessment
from evalf.schemas import EvalCase, UsageStats

from .prompt import build_claim_extraction_prompt, build_claim_verification_prompt
from .schema import FaithfulnessAssessment


class FaithfulnessMetric(BaseDecomposedMetric):
    """Score whether answer claims are supported by the retrieved contexts."""

    CONTRADICTION_PENALTY = 0.5

    name = "faithfulness"
    required_inputs = ("question", "retrieved_contexts", "actual_output")
    output_schema = FaithfulnessAssessment

    def _extract_claims(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[list[Claim], UsageStats]:
        """Extract atomic factual claims from the answer under evaluation."""

        system_prompt, user_prompt = build_claim_extraction_prompt(case)
        extraction, usage = self._generate_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ClaimExtraction,
        )
        return extraction.claims, usage

    async def _a_extract_claims(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[list[Claim], UsageStats]:
        """Async variant of answer-claim extraction."""

        system_prompt, user_prompt = build_claim_extraction_prompt(case)
        extraction, usage = await self._a_generate_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ClaimExtraction,
        )
        return extraction.claims, usage

    def _verify_claims(
        self, case: EvalCase, claims: list[Claim], llm: BaseLLMModel
    ) -> tuple[ClaimSupportAssessment, UsageStats]:
        """Verify each extracted claim against the retrieved contexts."""

        system_prompt, user_prompt = build_claim_verification_prompt(case, claims)
        assessment, usage = self._generate_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ClaimSupportAssessment,
        )
        return assessment, usage

    async def _a_verify_claims(
        self, case: EvalCase, claims: list[Claim], llm: BaseLLMModel
    ) -> tuple[ClaimSupportAssessment, UsageStats]:
        """Async variant of claim verification."""

        system_prompt, user_prompt = build_claim_verification_prompt(case, claims)
        assessment, usage = await self._a_generate_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ClaimSupportAssessment,
        )
        return assessment, usage

    def _score_claims(self, claims: list[Claim], assessment: ClaimSupportAssessment) -> tuple[float, str]:
        """Convert claim-level support verdicts into the final faithfulness score."""

        expected_ids = [claim.claim_id for claim in claims]
        observed_ids = [verdict.claim_id for verdict in assessment.verdicts]
        if expected_ids != observed_ids:
            raise ValueError("Faithfulness verdict ids must exactly match the extracted claim ids.")

        claims_by_id = {claim.claim_id: claim for claim in claims}
        supported: list[str] = []
        unsupported: list[str] = []
        contradicted: list[str] = []
        for verdict in assessment.verdicts:
            claim_text = claims_by_id[verdict.claim_id].text
            if verdict.verdict == "supported":
                supported.append(claim_text)
            elif verdict.verdict == "contradicted":
                contradicted.append(claim_text)
            else:
                unsupported.append(claim_text)

        total_claims = len(claims)
        if total_claims == 0:
            return 1.0, "Vacuous pass: the answer does not contain material factual claims to verify."

        score = max(
            0.0,
            (
                len(supported)
                - self.CONTRADICTION_PENALTY * len(contradicted)
            )
            / total_claims,
        )
        reason_parts = [f"Supported {len(supported)}/{total_claims} material claim(s)."]
        if contradicted:
            reason_parts.append(
                f"Contradicted claims apply an extra {self.CONTRADICTION_PENALTY:.1f} penalty each."
            )
            reason_parts.append(
                "Contradicted: " + "; ".join(contradicted[:2]) + "."
            )
        if unsupported:
            reason_parts.append(
                "Unsupported: " + "; ".join(unsupported[:2]) + "."
            )
        return score, " ".join(reason_parts)

    def compute_assessment(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[float, str | None, UsageStats]:
        """Run the sync faithfulness pipeline for one sample."""

        claims, extraction_usage = self._extract_claims(case, llm)
        if not claims:
            return (
                1.0,
                "Vacuous pass: the answer does not contain material factual claims to verify.",
                extraction_usage,
            )

        assessment, verification_usage = self._verify_claims(case, claims, llm)
        score, reason = self._score_claims(claims, assessment)
        return score, reason, UsageStats.combine([extraction_usage, verification_usage])

    async def a_compute_assessment(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[float, str | None, UsageStats]:
        """Run the async faithfulness pipeline for one sample."""

        claims, extraction_usage = await self._a_extract_claims(case, llm)
        if not claims:
            return (
                1.0,
                "Vacuous pass: the answer does not contain material factual claims to verify.",
                extraction_usage,
            )

        assessment, verification_usage = await self._a_verify_claims(case, claims, llm)
        score, reason = self._score_claims(claims, assessment)
        return score, reason, UsageStats.combine([extraction_usage, verification_usage])
