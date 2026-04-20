from __future__ import annotations

from evalf.llms.base import BaseLLMModel
from evalf.metrics.base import BaseDecomposedMetric
from evalf.schemas import EvalCase, MetricMode, TrialMetricResult, UsageStats

from .prompt import build_prompt as build_c4_prompt
from .prompt import build_reason_prompt
from .schema import C4Assessment, C4SummaryReason

CRITERIA = (
    "alignment_integrity",
    "accuracy_consistency",
    "safety_sovereignty_tone",
    "completeness_coverage",
)


class C4Metric(BaseDecomposedMetric):
    """Composite metric that averages four rubric dimensions in one judge call."""

    name = "c4"
    required_inputs = ("question", "actual_output", "expected_output")
    output_schema = C4Assessment

    def __init__(
        self,
        threshold: float = 0.7,
        mode: MetricMode = "pass@k",
        k: int = 1,
        *,
        include_reason: bool = True,
        need_summary_reason: bool = False,
        strict_mode: bool = False,
    ) -> None:
        """Configure C4 aggregation plus optional reason synthesis behavior."""

        super().__init__(threshold=threshold, mode=mode, k=k)
        self.include_reason = include_reason
        self.need_summary_reason = need_summary_reason
        self.strict_mode = strict_mode

    def _build_breakdown(self, assessment: C4Assessment) -> dict[str, dict[str, str | float]]:
        """Normalize the structured C4 response into a criterion-keyed mapping."""

        return {
            criterion: {
                "score": getattr(assessment, criterion).score,
                "reasoning": getattr(assessment, criterion).reasoning,
            }
            for criterion in CRITERIA
        }

    def _calculate_score(self, breakdown: dict[str, dict[str, str | float]]) -> float:
        """Average criterion scores and optionally zero out failed strict runs."""

        scores = [float(breakdown[criterion]["score"]) for criterion in CRITERIA]
        score = sum(scores) / len(scores)
        if self.strict_mode and score < self.threshold:
            return 0.0
        return round(score, 4)

    def _collapse_reason(self, breakdown: dict[str, dict[str, str | float]]) -> str | None:
        """Serialize criterion-level reasons into a compact single-line summary."""

        if not self.include_reason:
            return None
        parts = [
            f"{criterion}={float(breakdown[criterion]['score']):.2f}: {breakdown[criterion]['reasoning']}"
            for criterion in CRITERIA
            if breakdown[criterion]["reasoning"]
        ]
        return " | ".join(parts) if parts else None

    def _summarize_results(
        self, trial_results: list[TrialMetricResult], passed: bool | None
    ) -> str | None:
        """Prefer the selected scored trial's reason when aggregating attempts."""

        scored_with_reason = [
            result for result in trial_results if result.score is not None and result.reason
        ]
        if scored_with_reason:
            if self.mode == "pass^k":
                chosen = min(scored_with_reason, key=lambda result: result.score or 0.0)
            else:
                chosen = max(scored_with_reason, key=lambda result: result.score or 0.0)
            return chosen.reason
        return super()._summarize_results(trial_results, passed)

    def build_prompt(self, case: EvalCase) -> tuple[str, str]:
        """Build the primary structured C4 evaluation prompt."""

        return build_c4_prompt(case)

    def compute_assessment(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[float, str | None, UsageStats]:
        """Run the sync C4 evaluation flow for one sample."""

        system_prompt, user_prompt = build_c4_prompt(case)
        assessment, usage = self._generate_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=C4Assessment,
        )
        breakdown = self._build_breakdown(assessment)
        score = self._calculate_score(breakdown)

        if not self.include_reason:
            return score, None, usage
        if not self.need_summary_reason:
            return score, self._collapse_reason(breakdown), usage

        reason_system_prompt, reason_user_prompt = build_reason_prompt(
            score=score,
            breakdown=breakdown,
        )
        summary_reason, reason_usage = self._generate_structured(
            llm,
            system_prompt=reason_system_prompt,
            user_prompt=reason_user_prompt,
            output_schema=C4SummaryReason,
        )
        return score, summary_reason.reason, UsageStats.combine([usage, reason_usage])

    async def a_compute_assessment(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[float, str | None, UsageStats]:
        """Run the async C4 evaluation flow for one sample."""

        system_prompt, user_prompt = build_c4_prompt(case)
        assessment, usage = await self._a_generate_structured(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=C4Assessment,
        )
        breakdown = self._build_breakdown(assessment)
        score = self._calculate_score(breakdown)

        if not self.include_reason:
            return score, None, usage
        if not self.need_summary_reason:
            return score, self._collapse_reason(breakdown), usage

        reason_system_prompt, reason_user_prompt = build_reason_prompt(
            score=score,
            breakdown=breakdown,
        )
        summary_reason, reason_usage = await self._a_generate_structured(
            llm,
            system_prompt=reason_system_prompt,
            user_prompt=reason_user_prompt,
            output_schema=C4SummaryReason,
        )
        return score, summary_reason.reason, UsageStats.combine([usage, reason_usage])
