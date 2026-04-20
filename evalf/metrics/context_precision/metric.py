from __future__ import annotations

from evalf.llms.base import BaseLLMModel
from evalf.metrics.base import BaseDecomposedMetric
from evalf.metrics.context_recall.prompt import build_reference_claim_extraction_prompt
from evalf.metrics.decomposition import (
    Claim,
    ClaimExtraction,
    ContextCoverageAssessment,
    dedupe_ids,
    ensure_complete_id_mapping,
)
from evalf.schemas import EvalCase, UsageStats

from .prompt import build_context_precision_prompt
from .schema import ContextPrecisionAssessment


class ContextPrecisionMetric(BaseDecomposedMetric):
    """Score how efficiently retrieved contexts surface reference claims by rank."""

    name = "context_precision"
    required_inputs = ("question", "retrieved_contexts", "reference_contexts")
    output_schema = ContextPrecisionAssessment

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

    def _assess_contexts(
        self, case: EvalCase, reference_claims: list[Claim], llm: BaseLLMModel
    ) -> tuple[ContextCoverageAssessment, UsageStats]:
        """Assess which reference claims each retrieved context supports."""

        system_prompt, user_prompt = build_context_precision_prompt(case, reference_claims)
        assessment, usage = self._generate_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ContextCoverageAssessment,
        )
        return assessment, usage

    async def _a_assess_contexts(
        self, case: EvalCase, reference_claims: list[Claim], llm: BaseLLMModel
    ) -> tuple[ContextCoverageAssessment, UsageStats]:
        """Async variant of retrieved-context usefulness assessment."""

        system_prompt, user_prompt = build_context_precision_prompt(case, reference_claims)
        assessment, usage = await self._a_generate_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ContextCoverageAssessment,
        )
        return assessment, usage

    def _score_contexts(
        self,
        case: EvalCase,
        reference_claims: list[Claim],
        assessment: ContextCoverageAssessment,
    ) -> tuple[float, str]:
        """Compute rank-weighted precision from context usefulness verdicts."""

        expected_context_ids = [f"ctx_{index}" for index in range(1, len(case.retrieved_contexts or []) + 1)]
        observed_context_ids = [context.context_id for context in assessment.contexts]
        ensure_complete_id_mapping(
            expected_ids=expected_context_ids,
            observed_ids=observed_context_ids,
            entity_name="context",
        )
        if expected_context_ids != observed_context_ids:
            raise ValueError(
                "Context precision verdict ids must exactly match the retrieved context ids in order."
            )

        valid_claim_ids = {claim.claim_id for claim in reference_claims}
        total_claims = len(reference_claims)
        if total_claims == 0:
            return (
                1.0,
                "Vacuous pass: no material reference claims were extracted from the reference contexts.",
            )

        score = 0.0
        seen_claims: set[str] = set()
        discovered_ranks: list[int] = []
        redundant_ranks: list[int] = []
        noisy_ranks: list[int] = []
        useful_context_count = 0

        for rank, context in enumerate(assessment.contexts, start=1):
            supported_claim_ids = dedupe_ids(context.supported_claim_ids)
            unknown_claim_ids = sorted(set(supported_claim_ids) - valid_claim_ids)
            if unknown_claim_ids:
                raise ValueError(
                    "Context precision returned unknown reference claim ids: "
                    + ", ".join(unknown_claim_ids)
                    + "."
                )

            new_claims = [claim_id for claim_id in supported_claim_ids if claim_id not in seen_claims]
            if new_claims:
                useful_context_count += 1
                seen_claims.update(new_claims)
                precision_at_rank = useful_context_count / rank
                score += precision_at_rank * (len(new_claims) / total_claims)
                discovered_ranks.append(rank)
            elif supported_claim_ids:
                redundant_ranks.append(rank)
            else:
                noisy_ranks.append(rank)

        reason_parts = [
            f"Retrieved contexts first covered {len(seen_claims)}/{total_claims} reference claim(s)."
        ]
        if discovered_ranks:
            reason_parts.append(
                "Useful ranks: " + ", ".join(str(rank) for rank in discovered_ranks[:4]) + "."
            )
        if redundant_ranks:
            reason_parts.append(
                "Redundant ranks: " + ", ".join(str(rank) for rank in redundant_ranks[:4]) + "."
            )
        if noisy_ranks:
            reason_parts.append(
                "Noisy ranks: " + ", ".join(str(rank) for rank in noisy_ranks[:4]) + "."
            )
        return score, " ".join(reason_parts)

    def compute_assessment(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[float, str | None, UsageStats]:
        """Run the sync context precision pipeline for one sample."""

        reference_claims, extraction_usage = self._extract_reference_claims(case, llm)
        if not reference_claims:
            return (
                1.0,
                "Vacuous pass: no material reference claims were extracted from the reference contexts.",
                extraction_usage,
            )

        assessment, coverage_usage = self._assess_contexts(case, reference_claims, llm)
        score, reason = self._score_contexts(case, reference_claims, assessment)
        return score, reason, UsageStats.combine([extraction_usage, coverage_usage])

    async def a_compute_assessment(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[float, str | None, UsageStats]:
        """Run the async context precision pipeline for one sample."""

        reference_claims, extraction_usage = await self._a_extract_reference_claims(case, llm)
        if not reference_claims:
            return (
                1.0,
                "Vacuous pass: no material reference claims were extracted from the reference contexts.",
                extraction_usage,
            )

        assessment, coverage_usage = await self._a_assess_contexts(case, reference_claims, llm)
        score, reason = self._score_contexts(case, reference_claims, assessment)
        return score, reason, UsageStats.combine([extraction_usage, coverage_usage])
