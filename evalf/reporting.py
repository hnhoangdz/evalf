from __future__ import annotations

from pathlib import Path

from evalf.schemas import RunReport, RunSummary, SampleResult, UsageStats


def build_run_summary(samples: list[SampleResult]) -> RunSummary:
    """Aggregate per-sample results into a single run-level summary."""
    total_samples = len(samples)
    passed_samples = sum(1 for sample in samples if sample.status == "passed")
    failed_samples = sum(1 for sample in samples if sample.status == "failed")
    skipped_samples = sum(1 for sample in samples if sample.status == "skipped")

    usages = [sample.usage for sample in samples]
    totals = UsageStats.combine(usages)
    latencies = [
        sample.usage.latency_ms for sample in samples if sample.usage.latency_ms is not None
    ]

    metric_names = sorted({metric.name for sample in samples for metric in sample.metrics})
    metric_pass_rates: dict[str, float] = {}
    for name in metric_names:
        relevant = [
            metric
            for sample in samples
            for metric in sample.metrics
            if metric.name == name and metric.status in {"passed", "failed"}
        ]
        if not relevant:
            continue
        passed = sum(1 for metric in relevant if metric.status == "passed")
        metric_pass_rates[name] = round(passed / len(relevant), 4)

    return RunSummary(
        total_samples=total_samples,
        passed_samples=passed_samples,
        failed_samples=failed_samples,
        skipped_samples=skipped_samples,
        total_input_tokens=totals.input_tokens,
        total_output_tokens=totals.output_tokens,
        total_tokens=totals.total_tokens,
        total_cost_usd=totals.cost_usd,
        avg_latency_ms_per_sample=round(sum(latencies) / len(latencies), 4) if latencies else None,
        metric_pass_rates=metric_pass_rates,
    )


def report_to_json(report: RunReport) -> str:
    """Serialize a run report as pretty-printed JSON."""
    return report.model_dump_json(indent=2)


def report_to_markdown(report: RunReport) -> str:
    """Render a human-readable Markdown summary for a run report."""
    lines = [
        "# evalf Report",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- Total Samples: `{report.summary.total_samples}`",
        f"- Passed Samples: `{report.summary.passed_samples}`",
        f"- Failed Samples: `{report.summary.failed_samples}`",
        f"- Skipped Samples: `{report.summary.skipped_samples}`",
        f"- Total Tokens: `{report.summary.total_tokens}`",
        f"- Total Cost (USD): `{report.summary.total_cost_usd}`",
        "",
        "## Samples",
    ]
    for sample in report.samples:
        lines.extend(
            [
                "",
                f"### {sample.sample_id}",
                f"- Status: `{sample.status}`",
                f"- Total Tokens: `{sample.usage.total_tokens}`",
                f"- Total Cost (USD): `{sample.usage.cost_usd}`",
            ]
        )
        for metric in sample.metrics:
            lines.append(
                f"- `{metric.name}`: status=`{metric.status}`, mode=`{metric.mode}`, k=`{metric.requested_k}`, score=`{metric.score}`, threshold=`{metric.threshold}`, cost_usd=`{metric.usage.cost_usd}`"
            )
    return "\n".join(lines) + "\n"


def write_report(report: RunReport, path: str | None) -> Path | None:
    """Write a report to disk as JSON or Markdown depending on the output suffix."""
    if not path:
        return None
    output_path = Path(path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".md":
        output_path.write_text(report_to_markdown(report), encoding="utf-8")
        return output_path
    if output_path.suffix == ".json":
        output_path.write_text(report_to_json(report), encoding="utf-8")
        return output_path
    raise ValueError("Supported output formats are .json and .md.")
