from __future__ import annotations

from evalf.metrics.base import BaseMetric
from evalf.schemas import EvalCase, MetricMode, TrialMetricResult, UsageStats

from .prompt import build_prompt
from .schema import ContextCoverageAssessment


class ContextCoverageMetric(BaseMetric):
    """Score whether retrieved contexts cover enough key information from reference contexts."""

    name = "context_coverage"
    required_inputs = ("question", "retrieved_contexts", "reference_contexts")
    output_schema = ContextCoverageAssessment

    def __init__(
        self,
        threshold: float = 0.7,
        mode: MetricMode = "pass@k",
        k: int = 1,
        *,
        strict_mode: bool = False,
    ) -> None:
        super().__init__(threshold=threshold, mode=mode, k=k)
        self.strict_mode = strict_mode

    def build_prompt(self, case: EvalCase) -> tuple[str, str]:
        return build_prompt(case)

    def _scored_trial_result(
        self,
        *,
        attempt_index: int,
        score: float,
        reason: str | None,
        usage: UsageStats,
    ) -> TrialMetricResult:
        if self.strict_mode and score < self.threshold:
            score = 0.0
        return super()._scored_trial_result(
            attempt_index=attempt_index,
            score=score,
            reason=reason,
            usage=usage,
        )
