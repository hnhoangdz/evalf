import json

import pytest

from evalf.reporting import build_run_summary, report_to_json, report_to_markdown, write_report
from evalf.schemas import MetricResult, RunReport, SampleResult, UsageStats

pytestmark = pytest.mark.unit


def _make_report() -> RunReport:
    samples = [
        SampleResult(
            sample_id="case-1",
            status="passed",
            metrics=[
                MetricResult(
                    name="faithfulness",
                    status="passed",
                    score=0.9,
                    threshold=0.7,
                    passed=True,
                    usage=UsageStats(cost_usd=0.0002),
                )
            ],
            usage=UsageStats(
                input_tokens=100,
                output_tokens=40,
                total_tokens=140,
                latency_ms=10.0,
                cost_usd=0.0002,
            ),
        ),
        SampleResult(
            sample_id="case-2",
            status="failed",
            metrics=[
                MetricResult(
                    name="faithfulness",
                    status="failed",
                    score=0.2,
                    threshold=0.7,
                    passed=False,
                    usage=UsageStats(cost_usd=0.0001),
                )
            ],
            usage=UsageStats(
                input_tokens=50,
                output_tokens=20,
                total_tokens=70,
                latency_ms=20.0,
                cost_usd=0.0001,
            ),
        ),
    ]
    return RunReport(
        run_id="run_test",
        summary=build_run_summary(samples),
        samples=samples,
    )


def test_build_run_summary_aggregates_tokens_costs_and_latency() -> None:
    summary = _make_report().summary

    assert summary.total_samples == 2
    assert summary.passed_samples == 1
    assert summary.failed_samples == 1
    assert summary.total_tokens == 210
    assert summary.total_cost_usd == 0.0003
    assert summary.avg_latency_ms_per_sample == 15.0
    assert summary.metric_pass_rates == {"faithfulness": 0.5}


def test_report_to_json_serializes_summary_and_samples() -> None:
    payload = json.loads(report_to_json(_make_report()))

    assert payload["run_id"] == "run_test"
    assert payload["summary"]["total_samples"] == 2
    assert payload["samples"][0]["sample_id"] == "case-1"


def test_report_to_markdown_and_write_report_support_json_and_markdown(tmp_path) -> None:
    report = _make_report()
    markdown = report_to_markdown(report)

    assert "# evalf Report" in markdown
    assert "### case-1" in markdown

    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"
    write_report(report, markdown_path)
    write_report(report, json_path)

    assert markdown_path.read_text(encoding="utf-8") == markdown
    assert json.loads(json_path.read_text(encoding="utf-8"))["run_id"] == "run_test"


def test_write_report_defaults_extensionless_paths_to_json(tmp_path) -> None:
    output_path = write_report(_make_report(), tmp_path / "report")

    assert output_path == tmp_path / "report.json"
    assert output_path.read_text(encoding="utf-8")
    assert not (tmp_path / "report").exists()


def test_write_report_rejects_unsupported_extensions(tmp_path) -> None:
    with pytest.raises(ValueError, match="Supported output formats are .json and .md."):
        write_report(_make_report(), tmp_path / "report.txt")
