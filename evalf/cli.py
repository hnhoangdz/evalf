from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Sequence
from contextlib import suppress

import click
from pydantic import ValidationError

from evalf.evaluation import evaluate
from evalf.inputs import build_case_from_values, load_cases_from_json, load_cases_from_path
from evalf.llms.base import BaseLLMModel
from evalf.llms.factory import build_llm
from evalf.metrics import build_metrics, list_metric_names
from evalf.reporting import report_to_json, write_report
from evalf.schemas import RunReport
from evalf.settings import RuntimeSettings, load_runtime_settings
from evalf.utils import split_csv

CORE_METRICS = [
    "faithfulness",
    "answer_correctness",
    "answer_relevance",
    "context_relevance",
]
DEFAULT_C4_INCLUDE_REASON = True
DEFAULT_C4_SUMMARY_REASON = False
DEFAULT_C4_STRICT_MODE = False


def _parse_threshold_overrides(items: Sequence[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid metric threshold override: {item}")
        name, value = item.split("=", 1)
        overrides[name.strip()] = float(value.strip())
    return overrides


def _resolve_cases(
    *,
    input_path: str | None = None,
    sample_json: str | None = None,
    question: str | None = None,
    actual_output: str | None = None,
    expected_output: str | None = None,
    retrieved_contexts: Sequence[str] = (),
    reference_contexts: Sequence[str] = (),
) -> list:
    if input_path:
        return load_cases_from_path(input_path)
    if sample_json:
        return load_cases_from_json(sample_json)

    if any(
        value is not None
        for value in [
            question,
            actual_output,
            expected_output,
            retrieved_contexts or None,
            reference_contexts or None,
        ]
    ):
        return [
            build_case_from_values(
                question=question,
                retrieved_contexts=list(retrieved_contexts) or None,
                reference_contexts=list(reference_contexts) or None,
                actual_output=actual_output,
                expected_output=expected_output,
            )
        ]

    raise ValueError(
        "Provide one of: --input, --sample-json, or direct sample fields such as --question and --actual-output."
    )


def _build_judge_settings(
    *,
    defaults: RuntimeSettings,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    request_timeout_seconds: float | None,
    temperature: float | None,
    max_tokens: int | None,
) -> RuntimeSettings:
    return RuntimeSettings(
        provider=provider or defaults.provider,
        model=model or defaults.model,
        base_url=base_url or defaults.base_url,
        api_key=api_key or defaults.api_key,
        request_timeout_seconds=(
            request_timeout_seconds
            if request_timeout_seconds is not None
            else defaults.request_timeout_seconds
        ),
        max_retries=defaults.max_retries,
        temperature=temperature if temperature is not None else defaults.temperature,
        max_tokens=max_tokens if max_tokens is not None else defaults.max_tokens,
    )


def _log_header(
    *,
    n_cases: int,
    metric_names: list[str],
    provider: str,
    model: str,
    concurrency: int,
) -> None:
    """Print a run summary header to stderr before evaluation starts."""
    metrics_str = ", ".join(metric_names)
    click.echo(
        f"evalf: evaluating {n_cases} sample(s) with {len(metric_names)} metric(s) [{metrics_str}]",
        err=True,
    )
    click.echo(
        f"evalf: judge={provider}/{model} | concurrency={concurrency}",
        err=True,
    )
    click.echo("", err=True)


def _log_footer(report: RunReport, *, elapsed_seconds: float) -> None:
    """Print a run summary footer to stderr after evaluation completes."""
    s = report.summary
    cost_str = f"${s.total_cost_usd:.4f}" if s.total_cost_usd is not None else "-"
    click.echo("", err=True)
    click.echo(
        f"evalf: done — "
        f"{s.passed_samples} passed, "
        f"{s.failed_samples} failed, "
        f"{s.skipped_samples} skipped "
        f"| {elapsed_seconds:.1f}s | {cost_str}",
        err=True,
    )


def _cleanup_cli_judge(judge: BaseLLMModel | None) -> None:
    """Close a CLI-owned judge instance after `evaluate()` returns."""

    if judge is None:
        return

    with suppress(Exception):
        asyncio.run(judge.aclose())
    with suppress(Exception):
        judge.close()


def _run_command(
    *,
    input_path: str | None,
    sample_json: str | None,
    question: str | None,
    actual_output: str | None,
    expected_output: str | None,
    retrieved_contexts: Sequence[str],
    reference_contexts: Sequence[str],
    provider: str | None,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    metrics: str | None,
    metric_mode: str | None,
    k: int | None,
    threshold: float | None,
    metric_threshold: Sequence[str],
    concurrency: int | None,
    request_timeout_seconds: float | None,
    per_sample_timeout_seconds: float | None,
    temperature: float | None,
    max_tokens: int | None,
    output: str | None,
    c4_include_reason: bool | None,
    c4_summary_reason: bool | None,
    c4_strict_mode: bool | None,
) -> None:
    judge: BaseLLMModel | None = None
    try:
        defaults = load_runtime_settings()
        resolved_metrics = split_csv(metrics) if metrics else (defaults.metrics or CORE_METRICS)
        resolved_threshold = threshold if threshold is not None else defaults.threshold
        resolved_metric_mode = metric_mode or defaults.metric_mode
        resolved_k = k if k is not None else defaults.k
        resolved_request_timeout_seconds = (
            request_timeout_seconds
            if request_timeout_seconds is not None
            else defaults.request_timeout_seconds
        )
        resolved_per_sample_timeout_seconds = (
            per_sample_timeout_seconds
            if per_sample_timeout_seconds is not None
            else defaults.per_sample_timeout_seconds
        )
        resolved_c4_include_reason = (
            c4_include_reason if c4_include_reason is not None else DEFAULT_C4_INCLUDE_REASON
        )
        resolved_c4_summary_reason = (
            c4_summary_reason if c4_summary_reason is not None else DEFAULT_C4_SUMMARY_REASON
        )
        resolved_c4_strict_mode = (
            c4_strict_mode if c4_strict_mode is not None else DEFAULT_C4_STRICT_MODE
        )
        threshold_overrides = _parse_threshold_overrides(metric_threshold)
        cases = _resolve_cases(
            input_path=input_path,
            sample_json=sample_json,
            question=question,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieved_contexts=retrieved_contexts,
            reference_contexts=reference_contexts,
        )
        resolved_metric_objects = build_metrics(
            resolved_metrics,
            default_threshold=resolved_threshold,
            mode=resolved_metric_mode,
            k=resolved_k,
            threshold_overrides=threshold_overrides,
            metric_options={
                "c4": {
                    "include_reason": resolved_c4_include_reason,
                    "need_summary_reason": resolved_c4_summary_reason,
                    "strict_mode": resolved_c4_strict_mode,
                }
            },
        )

        judge_settings = _build_judge_settings(
            defaults=defaults,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            request_timeout_seconds=resolved_request_timeout_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        judge = build_llm(
            provider=judge_settings.provider,
            model=judge_settings.model,
            base_url=judge_settings.base_url,
            api_key=judge_settings.api_key,
            timeout_seconds=judge_settings.timeout_seconds,
            max_retries=judge_settings.max_retries,
            temperature=judge_settings.temperature,
            max_tokens=judge_settings.max_tokens,
        )

        resolved_concurrency = concurrency or defaults.concurrency
        _log_header(
            n_cases=len(cases),
            metric_names=resolved_metrics,
            provider=judge_settings.provider,
            model=judge_settings.model,
            concurrency=resolved_concurrency,
        )

        start_time = time.monotonic()
        report = evaluate(
            cases=cases,
            metrics=resolved_metric_objects,
            judge=judge,
            concurrency=resolved_concurrency,
            per_sample_timeout_seconds=resolved_per_sample_timeout_seconds,
        )
        elapsed = time.monotonic() - start_time

        _log_footer(report, elapsed_seconds=elapsed)

        output_path = output or defaults.output_path
        write_report(report, output_path)
    except (FileNotFoundError, OSError, TypeError, ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        _cleanup_cli_judge(judge)

    click.echo(report_to_json(report))


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Evaluate RAG systems with LLM-as-a-Judge."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command("run")
@click.option("--input", "input_path", type=str, help="Path to a .json or .jsonl file.")
@click.option("--sample-json", type=str, help="Inline JSON object or list of JSON objects.")
@click.option("--question", type=str, help="Question associated with the sample being evaluated.")
@click.option("--actual-output", type=str, help="Model output to score.")
@click.option("--expected-output", type=str, help="Expected answer or reference answer.")
@click.option(
    "--retrieved-context",
    "retrieved_contexts",
    multiple=True,
    help="Retrieved context. Repeat to provide multiple values.",
)
@click.option(
    "--reference-context",
    "reference_contexts",
    multiple=True,
    help="Reference context. Repeat to provide multiple values.",
)
@click.option("--provider", type=str, help="Judge provider name, for example openai or gemini.")
@click.option("--model", type=str, help="Judge model name.")
@click.option("--base-url", type=str, help="OpenAI-compatible base URL for the judge provider.")
@click.option("--api-key", type=str, help="API key for the judge provider.")
@click.option("--metrics", type=str, help="Comma-separated metric names.")
@click.option(
    "--metric-mode",
    type=click.Choice(["pass@k", "pass^k"]),
    help="Metric aggregation mode across multiple attempts.",
)
@click.option("--k", type=click.IntRange(1, 5), help="Number of attempts to aggregate, max 5.")
@click.option(
    "--threshold",
    type=float,
    help="Default pass/fail threshold applied to all metrics unless overridden.",
)
@click.option(
    "--metric-threshold",
    multiple=True,
    help="Per-metric threshold override, e.g. faithfulness=0.8",
)
@click.option("--concurrency", type=int, help="Maximum number of samples to evaluate concurrently.")
@click.option(
    "--request-timeout-seconds",
    "--timeout-seconds",
    type=float,
    help="Timeout in seconds for a single LLM request.",
)
@click.option(
    "--per-sample-timeout-seconds",
    type=float,
    help="Timeout in seconds for evaluating one sample across all metrics.",
)
@click.option("--temperature", type=float, help="Sampling temperature passed to the judge model.")
@click.option(
    "--max-tokens", type=int, help="Maximum completion tokens requested from the judge model."
)
@click.option(
    "--output",
    type=str,
    help="Write the report to a .json or .md path. Extensionless paths default to .json.",
)
@click.option(
    "--c4-include-reason/--no-c4-include-reason",
    default=None,
    help="Include criterion-level reasoning in the aggregated C4 metric reason.",
)
@click.option(
    "--c4-summary-reason/--no-c4-summary-reason",
    default=None,
    help="Request a second judge call to synthesize a single summary reason for C4.",
)
@click.option(
    "--c4-strict-mode/--no-c4-strict-mode",
    default=None,
    help="Clamp failed C4 results to 0.0 when the averaged score misses the threshold.",
)
def run(
    input_path: str | None,
    sample_json: str | None,
    question: str | None,
    actual_output: str | None,
    expected_output: str | None,
    retrieved_contexts: tuple[str, ...],
    reference_contexts: tuple[str, ...],
    provider: str | None,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    metrics: str | None,
    metric_mode: str | None,
    k: int | None,
    threshold: float | None,
    metric_threshold: tuple[str, ...],
    concurrency: int | None,
    request_timeout_seconds: float | None,
    per_sample_timeout_seconds: float | None,
    temperature: float | None,
    max_tokens: int | None,
    output: str | None,
    c4_include_reason: bool | None,
    c4_summary_reason: bool | None,
    c4_strict_mode: bool | None,
) -> None:
    """Evaluate one sample or a JSON/JSONL dataset."""
    _run_command(
        input_path=input_path,
        sample_json=sample_json,
        question=question,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieved_contexts=retrieved_contexts,
        reference_contexts=reference_contexts,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        metrics=metrics,
        metric_mode=metric_mode,
        k=k,
        threshold=threshold,
        metric_threshold=metric_threshold,
        concurrency=concurrency,
        request_timeout_seconds=request_timeout_seconds,
        per_sample_timeout_seconds=per_sample_timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
        output=output,
        c4_include_reason=c4_include_reason,
        c4_summary_reason=c4_summary_reason,
        c4_strict_mode=c4_strict_mode,
    )


@cli.command("list-metrics")
def list_metrics_command() -> None:
    """List built-in metrics."""
    click.echo("\n".join(list_metric_names()))


def main(argv: Sequence[str] | None = None) -> None:
    try:
        cli.main(
            args=list(argv) if argv is not None else None, prog_name="evalf", standalone_mode=False
        )
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from exc
    except click.exceptions.Exit as exc:
        if exc.exit_code:
            raise SystemExit(exc.exit_code) from exc
    except click.Abort as exc:
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main(sys.argv[1:])
