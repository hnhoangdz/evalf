from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from statistics import mean

from pydantic import BaseModel

from evalf.llms.base import BaseLLMModel
from evalf.schemas import (
    EvalAttempt,
    EvalCase,
    LLMResponse,
    MetricMode,
    MetricResult,
    TrialMetricResult,
    UsageStats,
)
from evalf.utils import extract_json_payload

MAX_TRIALS = 5


class BaseMetric(ABC):
    """Base class for one-call metrics that may aggregate over multiple attempts."""

    name: str
    required_inputs: tuple[str, ...] = ()
    output_schema: type[BaseModel]

    def __init__(self, threshold: float = 0.7, mode: MetricMode = "pass@k", k: int = 1) -> None:
        """Configure the pass threshold and multi-attempt aggregation mode."""

        if mode not in {"pass@k", "pass^k"}:
            raise ValueError(f"Unsupported metric mode: {mode}")
        if k < 1 or k > MAX_TRIALS:
            raise ValueError(f"k must be between 1 and {MAX_TRIALS}.")
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0.")
        self.threshold = threshold
        self.mode = mode
        self.k = k

    def _is_missing_value(self, value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, tuple, set, frozenset)):
            return len(value) == 0 or all(self._is_missing_value(item) for item in value)
        if isinstance(value, dict):
            return len(value) == 0
        return False

    def get_missing_inputs(self, case: EvalCase | EvalAttempt) -> list[str]:
        """Return required input field names that are absent or blank."""

        missing: list[str] = []
        for field_name in self.required_inputs:
            if self._is_missing_value(getattr(case, field_name, None)):
                missing.append(field_name)
        return missing

    def _materialize_attempts(self, case: EvalCase) -> list[EvalCase]:
        """Expand a case into concrete attempts with inherited sample-level fields."""

        if not case.attempts:
            return [case.model_copy(update={"attempts": None})]

        attempts: list[EvalCase] = []
        for attempt in case.attempts:
            attempts.append(
                EvalCase(
                    id=case.id,
                    question=attempt.question if attempt.question is not None else case.question,
                    retrieved_contexts=(
                        attempt.retrieved_contexts
                        if attempt.retrieved_contexts is not None
                        else case.retrieved_contexts
                    ),
                    reference_contexts=(
                        attempt.reference_contexts
                        if attempt.reference_contexts is not None
                        else case.reference_contexts
                    ),
                    actual_output=(
                        attempt.actual_output
                        if attempt.actual_output is not None
                        else case.actual_output
                    ),
                    expected_output=(
                        attempt.expected_output
                        if attempt.expected_output is not None
                        else case.expected_output
                    ),
                    attempts=None,
                    metadata={**case.metadata, **attempt.metadata},
                )
            )
        return attempts

    @abstractmethod
    def build_prompt(self, case: EvalCase) -> tuple[str, str]:
        """Return the system and user prompts for one structured judge call."""

        raise NotImplementedError

    def _parse_assessment(self, response: LLMResponse, output_schema: type[BaseModel]) -> BaseModel:
        """Coerce a provider response into the expected structured assessment model."""

        if isinstance(response.parsed_output, output_schema):
            return response.parsed_output
        if response.parsed_output is not None:
            return output_schema.model_validate(response.parsed_output)
        if response.text is None:
            raise ValueError("Model response did not include text or parsed output.")
        payload = extract_json_payload(response.text)
        return output_schema.model_validate_json(payload)

    def _generate_structured(
        self,
        llm: BaseLLMModel,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
    ) -> tuple[BaseModel, UsageStats]:
        """Run one sync structured generation and return the parsed assessment."""

        response = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema,
        )
        assessment = self._parse_assessment(response, output_schema)
        return assessment, response.usage

    @staticmethod
    def _skipped_trial_result(
        *, attempt_index: int, missing_inputs: list[str]
    ) -> TrialMetricResult:
        """Return a skipped trial result for attempts missing required inputs."""

        return TrialMetricResult(
            attempt_index=attempt_index,
            status="skipped",
            score=None,
            passed=None,
            missing_inputs=missing_inputs,
        )

    @staticmethod
    def _error_trial_result(*, attempt_index: int, exc: Exception) -> TrialMetricResult:
        """Return an errored trial result for an exception raised during scoring."""

        return TrialMetricResult(
            attempt_index=attempt_index,
            status="error",
            score=None,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    def _scored_trial_result(
        self,
        *,
        attempt_index: int,
        score: float,
        reason: str | None,
        usage: UsageStats,
    ) -> TrialMetricResult:
        """Convert a raw score into the normalized per-trial result object."""

        normalized_score = max(0.0, min(1.0, float(score)))
        passed = normalized_score >= self.threshold
        return TrialMetricResult(
            attempt_index=attempt_index,
            status="passed" if passed else "failed",
            score=normalized_score,
            passed=passed,
            reason=reason,
            usage=usage,
        )

    async def _a_generate_structured(
        self,
        llm: BaseLLMModel,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
    ) -> tuple[BaseModel, UsageStats]:
        """Run one async structured generation and return the parsed assessment."""

        response = await llm.a_generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema,
        )
        assessment = self._parse_assessment(response, output_schema)
        return assessment, response.usage

    def _measure(
        self,
        case: EvalCase,
        llm: BaseLLMModel,
        *,
        attempt_index: int,
    ) -> TrialMetricResult:
        missing_inputs = self.get_missing_inputs(case)
        if missing_inputs:
            return self._skipped_trial_result(
                attempt_index=attempt_index,
                missing_inputs=missing_inputs,
            )

        system_prompt, user_prompt = self.build_prompt(case)
        try:
            assessment, usage = self._generate_structured(
                llm,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=self.output_schema,
            )
            return self._scored_trial_result(
                attempt_index=attempt_index,
                reason=getattr(assessment, "reason", None),
                usage=usage,
                score=float(assessment.score),
            )
        except Exception as exc:
            return self._error_trial_result(
                attempt_index=attempt_index,
                exc=exc,
            )

    async def _ameasure(
        self,
        case: EvalCase,
        llm: BaseLLMModel,
        *,
        attempt_index: int,
    ) -> TrialMetricResult:
        missing_inputs = self.get_missing_inputs(case)
        if missing_inputs:
            return self._skipped_trial_result(
                attempt_index=attempt_index,
                missing_inputs=missing_inputs,
            )

        system_prompt, user_prompt = self.build_prompt(case)
        try:
            assessment, usage = await self._a_generate_structured(
                llm,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=self.output_schema,
            )
            return self._scored_trial_result(
                attempt_index=attempt_index,
                reason=getattr(assessment, "reason", None),
                usage=usage,
                score=float(assessment.score),
            )
        except Exception as exc:
            return self._error_trial_result(
                attempt_index=attempt_index,
                exc=exc,
            )

    def _aggregate_results(
        self, trial_results: list[TrialMetricResult]
    ) -> tuple[float | None, bool | None]:
        """Collapse per-attempt outcomes into a final score/pass tuple."""

        scored_trials = [result.score for result in trial_results if result.score is not None]
        passed_trials = [result for result in trial_results if result.status == "passed"]
        failed_trials = [result for result in trial_results if result.status == "failed"]
        skipped_trials = [result for result in trial_results if result.status == "skipped"]
        error_trials = [result for result in trial_results if result.status == "error"]

        if self.mode == "pass@k":
            if not scored_trials:
                return None, None
            score = max(scored_trials)
            if passed_trials:
                return score, True
            if failed_trials:
                return score, False
            if error_trials:
                return score, False
            return score, None

        if failed_trials:
            return min(scored_trials) if scored_trials else None, False
        if error_trials:
            return None, None
        if skipped_trials:
            return None, None
        if not scored_trials:
            return None, None
        return min(scored_trials), bool(passed_trials) and len(passed_trials) == len(trial_results)

    def _collect_missing_inputs(self, trial_results: list[TrialMetricResult]) -> list[str]:
        missing_inputs = {
            field_name for result in trial_results for field_name in result.missing_inputs
        }
        return sorted(missing_inputs)

    def _summarize_results(
        self, trial_results: list[TrialMetricResult], passed: bool | None
    ) -> str | None:
        """Build a human-readable summary for the aggregated attempt outcomes."""

        if not trial_results:
            return None

        evaluated_results = [
            result for result in trial_results if result.status in {"passed", "failed"}
        ]
        evaluated_count = len(evaluated_results)
        requested_count = len(trial_results)
        success_count = sum(1 for result in evaluated_results if result.passed is True)
        skipped_count = sum(1 for result in trial_results if result.status == "skipped")
        error_count = sum(1 for result in trial_results if result.status == "error")

        if passed is None:
            if error_count:
                return (
                    f"Metric could not be finalized because {error_count}/{requested_count} "
                    "attempt(s) errored."
                )
            if skipped_count and evaluated_count:
                return (
                    f"Only {evaluated_count}/{requested_count} attempt(s) were evaluated and "
                    "all evaluated attempts passed; pass^k requires every requested attempt "
                    "to be evaluated and pass."
                )
            return "Metric was skipped because no attempt could be evaluated."

        if self.mode == "pass@k":
            if passed:
                return f"Passed in {success_count}/{evaluated_count} evaluated attempt(s) under pass@k."
            return f"No attempt met the threshold across {evaluated_count} evaluated attempt(s) under pass@k."

        if passed:
            return (
                f"All {evaluated_count}/{evaluated_count} evaluated attempt(s) passed under pass^k."
            )
        return (
            f"Only {success_count}/{evaluated_count} evaluated attempt(s) passed; "
            "pass^k requires every evaluated attempt to pass."
        )

    def _insufficient_attempts_result(self, *, available_attempts: int) -> MetricResult:
        """Return a skipped metric result when the sample provides fewer than `k` attempts."""

        return MetricResult(
            name=self.name,
            status="skipped",
            score=None,
            threshold=self.threshold,
            passed=None,
            required_inputs=list(self.required_inputs),
            missing_inputs=[],
            reason=f"Requested k={self.k}, but sample only provides {available_attempts} attempt(s).",
            mode=self.mode,
            requested_k=self.k,
            evaluated_k=available_attempts,
        )

    def _build_metric_result(self, trial_results: list[TrialMetricResult]) -> MetricResult:
        """Collapse trial results into the final metric result payload."""

        aggregate_score, passed = self._aggregate_results(trial_results)
        missing_inputs = self._collect_missing_inputs(trial_results)
        usages = UsageStats.combine([result.usage for result in trial_results])
        scored_trials = [result.score for result in trial_results if result.score is not None]
        has_error = any(result.status == "error" for result in trial_results)
        has_non_skipped = any(result.status != "skipped" for result in trial_results)

        if passed is None:
            status = "error" if has_error else "skipped"
        else:
            status = "passed" if passed else "failed"
        if not has_non_skipped and not has_error:
            status = "skipped"

        return MetricResult(
            name=self.name,
            status=status,
            score=aggregate_score,
            threshold=self.threshold,
            passed=passed,
            reason=self._summarize_results(trial_results, passed),
            required_inputs=list(self.required_inputs),
            missing_inputs=missing_inputs,
            usage=usages,
            error="; ".join(result.error for result in trial_results if result.error) or None,
            mode=self.mode,
            requested_k=self.k,
            evaluated_k=len(scored_trials),
            successful_trials=sum(1 for result in trial_results if result.passed is True),
            best_trial_score=max(scored_trials) if scored_trials else None,
            worst_trial_score=min(scored_trials) if scored_trials else None,
            mean_trial_score=round(mean(scored_trials), 4) if scored_trials else None,
            trial_results=trial_results,
        )

    async def a_measure(self, case: EvalCase, llm: BaseLLMModel) -> MetricResult:
        """Evaluate a sample asynchronously and aggregate up to `k` attempts."""

        attempts = self._materialize_attempts(case)
        if len(attempts) < self.k:
            return self._insufficient_attempts_result(available_attempts=len(attempts))

        selected_attempts = attempts[: self.k]
        trial_results = await asyncio.gather(
            *(
                self._ameasure(attempt, llm, attempt_index=index)
                for index, attempt in enumerate(selected_attempts, start=1)
            )
        )
        return self._build_metric_result(list(trial_results))

    def measure(self, case: EvalCase, llm: BaseLLMModel) -> MetricResult:
        """Evaluate a sample synchronously and aggregate up to `k` attempts."""

        attempts = self._materialize_attempts(case)
        if len(attempts) < self.k:
            return self._insufficient_attempts_result(available_attempts=len(attempts))

        selected_attempts = attempts[: self.k]
        trial_results = [
            self._measure(attempt, llm, attempt_index=index)
            for index, attempt in enumerate(selected_attempts, start=1)
        ]
        return self._build_metric_result(trial_results)


class BaseDecomposedMetric(BaseMetric, ABC):
    """Base class for multi-step metrics that orchestrate several judge calls."""

    # Subclasses may still expose a final aggregate schema for metadata and documentation,
    # even though decomposed scoring pipelines do not use `self.output_schema` directly.
    output_schema = BaseModel

    def build_prompt(self, case: EvalCase) -> tuple[str, str]:
        """Reject single-prompt access for metrics implemented as pipelines."""

        raise NotImplementedError(
            f"{type(self).__name__} uses a multi-step metric pipeline and does not expose a single prompt."
        )

    @abstractmethod
    def compute_assessment(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[float, str | None, UsageStats]:
        """Return the score, reason, and usage for one fully materialized attempt."""

        raise NotImplementedError

    async def a_compute_assessment(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[float, str | None, UsageStats]:
        """Async variant of `compute_assessment`, offloading sync-only metrics to a worker thread."""

        return await asyncio.to_thread(self.compute_assessment, case, llm)

    def _measure(
        self,
        case: EvalCase,
        llm: BaseLLMModel,
        *,
        attempt_index: int,
    ) -> TrialMetricResult:
        missing_inputs = self.get_missing_inputs(case)
        if missing_inputs:
            return self._skipped_trial_result(
                attempt_index=attempt_index,
                missing_inputs=missing_inputs,
            )

        try:
            score, reason, usage = self.compute_assessment(case, llm)
            return self._scored_trial_result(
                attempt_index=attempt_index,
                reason=reason,
                usage=usage,
                score=score,
            )
        except Exception as exc:
            return self._error_trial_result(
                attempt_index=attempt_index,
                exc=exc,
            )

    async def _ameasure(
        self,
        case: EvalCase,
        llm: BaseLLMModel,
        *,
        attempt_index: int,
    ) -> TrialMetricResult:
        missing_inputs = self.get_missing_inputs(case)
        if missing_inputs:
            return self._skipped_trial_result(
                attempt_index=attempt_index,
                missing_inputs=missing_inputs,
            )

        try:
            score, reason, usage = await self.a_compute_assessment(case, llm)
            return self._scored_trial_result(
                attempt_index=attempt_index,
                reason=reason,
                usage=usage,
                score=score,
            )
        except Exception as exc:
            return self._error_trial_result(
                attempt_index=attempt_index,
                exc=exc,
            )
