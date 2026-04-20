import pytest

from evalf.metrics import build_metrics, list_metric_names, register_metric
from evalf.metrics.c4 import C4Metric
from evalf.metrics.registry import METRIC_REGISTRY

pytestmark = pytest.mark.unit


def test_metric_registry_lists_supported_metrics() -> None:
    metric_names = list_metric_names()

    assert metric_names == sorted(metric_names)
    assert "answer_correctness" in metric_names
    assert "answer_relevance" in metric_names
    assert "c4" in metric_names
    assert "context_coverage" in metric_names
    assert "context_precision" in metric_names
    assert "context_recall" in metric_names
    assert "context_relevance" in metric_names
    assert "faithfulness" in metric_names


def test_build_metrics_constructs_requested_subset_in_order() -> None:
    metrics = build_metrics(["answer_correctness", "c4"], default_threshold=0.7)

    assert [metric.name for metric in metrics] == ["answer_correctness", "c4"]
    assert isinstance(metrics[1], C4Metric)


def test_k_greater_than_five_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_metrics(["faithfulness"], default_threshold=0.7, mode="pass@k", k=6)


def test_register_metric_allows_custom_metric_aliases() -> None:
    register_metric("custom_c4", C4Metric)

    try:
        metric_names = list_metric_names()
        metrics = build_metrics(["custom_c4"], default_threshold=0.7)
    finally:
        METRIC_REGISTRY.pop("custom_c4", None)

    assert "custom_c4" in metric_names
    assert len(metrics) == 1
    assert isinstance(metrics[0], C4Metric)


def test_build_metrics_passes_metric_specific_options() -> None:
    metrics = build_metrics(
        ["c4"],
        default_threshold=0.7,
        metric_options={"c4": {"strict_mode": True, "include_reason": False}},
    )

    assert isinstance(metrics[0], C4Metric)
    assert metrics[0].strict_mode is True
    assert metrics[0].include_reason is False
