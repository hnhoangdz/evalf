from __future__ import annotations

import asyncio
import uuid

from evalf.llms.base import BaseLLMModel
from evalf.metrics.base import BaseMetric
from evalf.reporting import build_run_summary
from evalf.schemas import EvalCase, MetricResult, RunReport, SampleResult, UsageStats


def _sample_id_for_case(case: EvalCase, *, index: int) -> str:
    """Return the stable sample id used for both success and error paths."""
    return case.id or f"sample-{index}"


async def _evaluate_case(
    *,
    case: EvalCase,
    sample_id: str,
    metrics: list[BaseMetric],
    llm: BaseLLMModel,
) -> SampleResult:
    metric_results: list[MetricResult] = []
    for metric in metrics:
        result = await metric.a_measure(case, llm)
        metric_results.append(result)

    executed = [metric for metric in metric_results if metric.status != "skipped"]
    if any(metric.status in {"failed", "error"} for metric in executed):
        status = "failed"
    elif any(metric.status == "passed" for metric in executed):
        status = "passed"
    else:
        status = "skipped"

    usage = UsageStats.combine([metric.usage for metric in metric_results])
    return SampleResult(
        sample_id=sample_id,
        status=status,
        metrics=metric_results,
        usage=usage,
        metadata=case.metadata,
    )


async def execute_cases(
    *,
    cases: list[EvalCase],
    metrics: list[BaseMetric],
    llm: BaseLLMModel,
    concurrency: int = 4,
    per_sample_timeout_seconds: float | None = None,
) -> RunReport:
    """Evaluate a batch of cases concurrently with bounded per-sample timeouts."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_case(case: EvalCase, sample_id: str) -> SampleResult:
        async with semaphore:
            coro = _evaluate_case(case=case, sample_id=sample_id, metrics=metrics, llm=llm)
            if per_sample_timeout_seconds is None:
                return await coro
            return await asyncio.wait_for(coro, timeout=per_sample_timeout_seconds)

    indexed_cases = [
        (index, _sample_id_for_case(case, index=index), case) for index, case in enumerate(cases, start=1)
    ]
    tasks = [
        asyncio.create_task(run_case(case, sample_id), name=f"eval-{sample_id}")
        for _, sample_id, case in indexed_cases
    ]
    sample_results: list[SampleResult] = []

    for (_index, sample_id, case), result in zip(
        indexed_cases,
        await asyncio.gather(*tasks, return_exceptions=True),
        strict=True,
    ):
        if isinstance(result, BaseException):
            sample_results.append(
                SampleResult(
                    sample_id=sample_id,
                    status="failed",
                    metrics=[
                        MetricResult(
                            name="executor",
                            status="error",
                            score=None,
                            threshold=0.0,
                            passed=False,
                            error=f"{type(result).__name__}: {result}",
                        )
                    ],
                    usage=UsageStats.empty(),
                    metadata=case.metadata,
                )
            )
            continue
        sample_results.append(result)

    summary = build_run_summary(sample_results)
    return RunReport(
        run_id=f"run_{uuid.uuid4().hex[:12]}",
        summary=summary,
        samples=sample_results,
    )


def execute_cases_sync(
    *,
    cases: list[EvalCase],
    metrics: list[BaseMetric],
    llm: BaseLLMModel,
) -> RunReport:
    """Evaluate a batch of cases synchronously for tests and local utilities."""
    sample_results: list[SampleResult] = []
    for index, case in enumerate(cases, start=1):
        sample_id = _sample_id_for_case(case, index=index)
        try:
            sample_results.append(
                _evaluate_case_sync(case=case, sample_id=sample_id, metrics=metrics, llm=llm)
            )
        except BaseException as result:
            if isinstance(result, (KeyboardInterrupt, SystemExit)):
                raise
            sample_results.append(
                SampleResult(
                    sample_id=sample_id,
                    status="failed",
                    metrics=[
                        MetricResult(
                            name="executor",
                            status="error",
                            score=None,
                            threshold=0.0,
                            passed=False,
                            error=f"{type(result).__name__}: {result}",
                        )
                    ],
                    usage=UsageStats.empty(),
                    metadata=case.metadata,
                )
            )

    summary = build_run_summary(sample_results)
    return RunReport(
        run_id=f"run_{uuid.uuid4().hex[:12]}",
        summary=summary,
        samples=sample_results,
    )


def _evaluate_case_sync(
    *,
    case: EvalCase,
    sample_id: str,
    metrics: list[BaseMetric],
    llm: BaseLLMModel,
) -> SampleResult:
    metric_results: list[MetricResult] = []
    for metric in metrics:
        result = metric.measure(case, llm)
        metric_results.append(result)

    executed = [metric for metric in metric_results if metric.status != "skipped"]
    if any(metric.status in {"failed", "error"} for metric in executed):
        status = "failed"
    elif any(metric.status == "passed" for metric in executed):
        status = "passed"
    else:
        status = "skipped"

    usage = UsageStats.combine([metric.usage for metric in metric_results])
    return SampleResult(
        sample_id=sample_id,
        status=status,
        metrics=metric_results,
        usage=usage,
        metadata=case.metadata,
    )
