from __future__ import annotations

from evalf.llms.base import BaseLLMModel
from evalf.metrics.base import BaseDecomposedMetric
from evalf.metrics.decomposition import Claim, ClaimCoverageAssessment, ClaimExtraction
from evalf.schemas import EvalCase, UsageStats

from .prompt import build_context_recall_prompt, build_reference_claim_extraction_prompt
from .schema import ContextRecallAssessment


class ContextRecallMetric(BaseDecomposedMetric):
    """Score how fully retrieved contexts cover claims from reference contexts."""

    name = "context_recall"
    required_inputs = ("question", "retrieved_contexts", "reference_contexts")
    output_schema = ContextRecallAssessment

    def _extract_reference_claims(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[list[Claim], UsageStats]:
        """Extract atomic claims from the reference contexts."""

        system_prompt, user_prompt = build_reference_claim_extraction_prompt(case)
        extraction, usage = self._generate_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ClaimExtraction,
        )
        return extraction.claims, usage

    async def _a_extract_reference_claims(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[list[Claim], UsageStats]:
        """Async variant of reference-claim extraction."""

        system_prompt, user_prompt = build_reference_claim_extraction_prompt(case)
        extraction, usage = await self._a_generate_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ClaimExtraction,
        )
        return extraction.claims, usage

    def _verify_coverage(
        self, case: EvalCase, reference_claims: list[Claim], llm: BaseLLMModel
    ) -> tuple[ClaimCoverageAssessment, UsageStats]:
        """Verify whether retrieved contexts support each reference claim."""

        system_prompt, user_prompt = build_context_recall_prompt(case, reference_claims)
        assessment, usage = self._generate_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ClaimCoverageAssessment,
        )
        return assessment, usage

    async def _a_verify_coverage(
        self, case: EvalCase, reference_claims: list[Claim], llm: BaseLLMModel
    ) -> tuple[ClaimCoverageAssessment, UsageStats]:
        """Async variant of claim coverage verification."""

        system_prompt, user_prompt = build_context_recall_prompt(case, reference_claims)
        assessment, usage = await self._a_generate_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ClaimCoverageAssessment,
        )
        return assessment, usage

    def _score_claims(
        self, reference_claims: list[Claim], assessment: ClaimCoverageAssessment
    ) -> tuple[float, str]:
        """Convert claim coverage verdicts into the final recall score."""

        expected_ids = [claim.claim_id for claim in reference_claims]
        observed_ids = [verdict.claim_id for verdict in assessment.verdicts]
        if expected_ids != observed_ids:
            raise ValueError("Context recall verdict ids must exactly match the reference claim ids.")

        claims_by_id = {claim.claim_id: claim for claim in reference_claims}
        supported: list[str] = []
        missing: list[str] = []
        for verdict in assessment.verdicts:
            claim_text = claims_by_id[verdict.claim_id].text
            if verdict.verdict == "supported":
                supported.append(claim_text)
            else:
                missing.append(claim_text)

        total_claims = len(reference_claims)
        if total_claims == 0:
            return (
                1.0,
                "Vacuous pass: no material reference claims were extracted from the reference contexts.",
            )

        score = len(supported) / total_claims
        reason_parts = [f"Retrieved contexts cover {len(supported)}/{total_claims} reference claim(s)."]
        if missing:
            reason_parts.append("Missing: " + "; ".join(missing[:2]) + ".")
        return score, " ".join(reason_parts)

    def compute_assessment(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[float, str | None, UsageStats]:
        """Run the sync context recall pipeline for one sample."""

        reference_claims, extraction_usage = self._extract_reference_claims(case, llm)
        if not reference_claims:
            return (
                1.0,
                "Vacuous pass: no material reference claims were extracted from the reference contexts.",
                extraction_usage,
            )

        assessment, coverage_usage = self._verify_coverage(case, reference_claims, llm)
        score, reason = self._score_claims(reference_claims, assessment)
        return score, reason, UsageStats.combine([extraction_usage, coverage_usage])

    async def a_compute_assessment(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[float, str | None, UsageStats]:
        """Run the async context recall pipeline for one sample."""

        reference_claims, extraction_usage = await self._a_extract_reference_claims(case, llm)
        if not reference_claims:
            return (
                1.0,
                "Vacuous pass: no material reference claims were extracted from the reference contexts.",
                extraction_usage,
            )

        assessment, coverage_usage = await self._a_verify_coverage(case, reference_claims, llm)
        score, reason = self._score_claims(reference_claims, assessment)
        return score, reason, UsageStats.combine([extraction_usage, coverage_usage])
