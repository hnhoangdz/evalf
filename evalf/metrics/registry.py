from __future__ import annotations

from typing import Any

from evalf.metrics.answer_correctness import AnswerCorrectnessMetric
from evalf.metrics.answer_relevance import AnswerRelevanceMetric
from evalf.metrics.base import BaseMetric
from evalf.metrics.c4 import C4Metric
from evalf.metrics.context_coverage import ContextCoverageMetric
from evalf.metrics.context_precision import ContextPrecisionMetric
from evalf.metrics.context_recall import ContextRecallMetric
from evalf.metrics.context_relevance import ContextRelevanceMetric
from evalf.metrics.faithfulness import FaithfulnessMetric
from evalf.schemas import MetricMode

METRIC_REGISTRY: dict[str, type[BaseMetric]] = {
    "faithfulness": FaithfulnessMetric,
    "answer_correctness": AnswerCorrectnessMetric,
    "answer_relevance": AnswerRelevanceMetric,
    "c4": C4Metric,
    "context_coverage": ContextCoverageMetric,
    "context_relevance": ContextRelevanceMetric,
    "context_precision": ContextPrecisionMetric,
    "context_recall": ContextRecallMetric,
}


def list_metric_names() -> list[str]:
    """Return the sorted list of registered metric names."""
    return sorted(METRIC_REGISTRY.keys())


def register_metric(name: str, cls: type[BaseMetric]) -> None:
    """Register a custom metric class under a user-facing metric name."""
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("Metric name must not be empty.")
    if not issubclass(cls, BaseMetric):
        raise TypeError("Registered metric classes must inherit from BaseMetric.")
    METRIC_REGISTRY[normalized_name] = cls


def build_metrics(
    names: list[str],
    *,
    default_threshold: float,
    mode: MetricMode = "pass@k",
    k: int = 1,
    threshold_overrides: dict[str, float] | None = None,
    metric_options: dict[str, dict[str, Any]] | None = None,
) -> list[BaseMetric]:
    """Instantiate the requested metrics in the provided order."""
    threshold_overrides = threshold_overrides or {}
    metric_options = metric_options or {}
    metrics: list[BaseMetric] = []
    for name in names:
        normalized_name = name.strip()
        metric_cls = METRIC_REGISTRY.get(normalized_name)
        if metric_cls is None:
            raise ValueError(f"Unsupported metric: {normalized_name}")
        options = dict(metric_options.get(normalized_name, {}))
        metrics.append(
            metric_cls(
                threshold=threshold_overrides.get(normalized_name, default_threshold),
                mode=mode,
                k=k,
                **options,
            )
        )
    return metrics
