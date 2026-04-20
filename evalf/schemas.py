"""Core data models shared across the CLI, Python API, and reports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MetricStatus = Literal["passed", "failed", "skipped", "error"]
SampleStatus = Literal["passed", "failed", "skipped"]
MetricMode = Literal["pass@k", "pass^k"]


class EvalAttempt(BaseModel):
    """One candidate answer attempt for a sample in a multi-attempt run."""

    model_config = ConfigDict(extra="allow")

    question: str | None = None
    retrieved_contexts: list[str] | None = None
    reference_contexts: list[str] | None = None
    actual_output: str | None = None
    expected_output: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalCase(BaseModel):
    """One evaluation sample consumed by the executor and metrics."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    question: str | None = None
    retrieved_contexts: list[str] | None = None
    reference_contexts: list[str] | None = None
    actual_output: str | None = None
    expected_output: str | None = None
    attempts: list[EvalAttempt] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UsageStats(BaseModel):
    """Token, latency, and estimated-cost metadata emitted by judge calls."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None

    @classmethod
    def empty(cls) -> UsageStats:
        """Return an empty usage object for skipped or unmetered operations."""

        return cls()

    @classmethod
    def combine(cls, usages: list[UsageStats]) -> UsageStats:
        """Aggregate usage records by summing numeric fields when available."""

        if not usages:
            return cls.empty()

        int_fields = ["input_tokens", "output_tokens", "total_tokens"]
        float_fields = ["latency_ms", "cost_usd"]
        data: dict[str, int | float | None] = {}

        for field in int_fields:
            values = [getattr(item, field) for item in usages if getattr(item, field) is not None]
            data[field] = sum(values) if values else None

        for field in float_fields:
            values = [getattr(item, field) for item in usages if getattr(item, field) is not None]
            data[field] = round(sum(values), 8) if values else None

        return cls(**data)


class LLMResponse(BaseModel):
    """Normalized judge response returned by provider model adapters."""

    text: str | None
    model: str
    provider: str
    parsed_output: Any | None = None
    usage: UsageStats = Field(default_factory=UsageStats.empty)


class MetricResult(BaseModel):
    """Aggregated metric outcome for a single sample across up to `k` attempts."""

    name: str
    status: MetricStatus
    score: float | None
    threshold: float
    passed: bool | None
    reason: str | None = None
    required_inputs: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    usage: UsageStats = Field(default_factory=UsageStats.empty)
    error: str | None = None
    mode: MetricMode = "pass@k"
    requested_k: int = 1
    evaluated_k: int = 0
    successful_trials: int = 0
    best_trial_score: float | None = None
    worst_trial_score: float | None = None
    mean_trial_score: float | None = None
    trial_results: list[TrialMetricResult] = Field(default_factory=list)


class TrialMetricResult(BaseModel):
    """Single-attempt metric outcome before pass@k or pass^k aggregation."""

    attempt_index: int
    status: MetricStatus
    score: float | None
    passed: bool | None
    reason: str | None = None
    missing_inputs: list[str] = Field(default_factory=list)
    usage: UsageStats = Field(default_factory=UsageStats.empty)
    error: str | None = None


class SampleResult(BaseModel):
    """Per-sample result including every metric outcome and rolled-up usage."""

    sample_id: str
    status: SampleStatus
    metrics: list[MetricResult]
    usage: UsageStats = Field(default_factory=UsageStats.empty)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel):
    """Run-level totals and pass rates derived from all sample results."""

    total_samples: int
    passed_samples: int
    failed_samples: int
    skipped_samples: int
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    avg_latency_ms_per_sample: float | None = None
    metric_pass_rates: dict[str, float] = Field(default_factory=dict)


class RunReport(BaseModel):
    """Top-level report returned by the CLI and Python API."""

    run_id: str
    summary: RunSummary
    samples: list[SampleResult]


MetricResult.model_rebuild()
