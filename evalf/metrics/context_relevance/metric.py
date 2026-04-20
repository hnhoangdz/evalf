from __future__ import annotations

from evalf.llms.base import BaseLLMModel
from evalf.metrics.base import BaseDecomposedMetric
from evalf.metrics.decomposition import ContextRelevanceVerdictList, ensure_complete_id_mapping
from evalf.schemas import EvalCase, UsageStats

from .prompt import build_prompt
from .schema import ContextRelevanceAssessment


class ContextRelevanceMetric(BaseDecomposedMetric):
    """Score the average relevance of retrieved contexts to the user question."""

    name = "context_relevance"
    required_inputs = ("question", "retrieved_contexts")
    output_schema = ContextRelevanceAssessment

    _weights = {
        "relevant": 1.0,
        "partially_relevant": 0.5,
        "irrelevant": 0.0,
    }

    def compute_assessment(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[float, str | None, UsageStats]:
        """Score one sample by averaging relevance verdicts across its contexts."""

        system_prompt, user_prompt = build_prompt(case)
        assessment, usage = self._generate_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ContextRelevanceVerdictList,
        )
        score, reason = self._score_contexts(case, assessment)
        return score, reason, usage

    async def a_compute_assessment(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[float, str | None, UsageStats]:
        """Async variant of context relevance scoring."""

        system_prompt, user_prompt = build_prompt(case)
        assessment, usage = await self._a_generate_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ContextRelevanceVerdictList,
        )
        score, reason = self._score_contexts(case, assessment)
        return score, reason, usage

    def _score_contexts(
        self,
        case: EvalCase,
        assessment: ContextRelevanceVerdictList,
    ) -> tuple[float, str]:
        """Convert per-context verdicts into a normalized relevance score."""

        expected_context_ids = [f"ctx_{index}" for index in range(1, len(case.retrieved_contexts or []) + 1)]
        observed_context_ids = [verdict.context_id for verdict in assessment.verdicts]
        ensure_complete_id_mapping(
            expected_ids=expected_context_ids,
            observed_ids=observed_context_ids,
            entity_name="context",
        )
        if expected_context_ids != observed_context_ids:
            raise ValueError(
                "Context relevance verdict ids must exactly match the retrieved context ids in order."
            )

        weights = [self._weights[verdict.verdict] for verdict in assessment.verdicts]
        score = sum(weights) / len(weights)

        relevant_ranks = [
            str(index)
            for index, verdict in enumerate(assessment.verdicts, start=1)
            if verdict.verdict == "relevant"
        ]
        partial_ranks = [
            str(index)
            for index, verdict in enumerate(assessment.verdicts, start=1)
            if verdict.verdict == "partially_relevant"
        ]
        irrelevant_ranks = [
            str(index)
            for index, verdict in enumerate(assessment.verdicts, start=1)
            if verdict.verdict == "irrelevant"
        ]

        reason_parts = [
            f"Average relevance across {len(assessment.verdicts)} retrieved context(s)."
        ]
        if relevant_ranks:
            reason_parts.append("Relevant ranks: " + ", ".join(relevant_ranks[:4]) + ".")
        if partial_ranks:
            reason_parts.append("Partial ranks: " + ", ".join(partial_ranks[:4]) + ".")
        if irrelevant_ranks:
            reason_parts.append("Irrelevant ranks: " + ", ".join(irrelevant_ranks[:4]) + ".")
        return score, " ".join(reason_parts)
