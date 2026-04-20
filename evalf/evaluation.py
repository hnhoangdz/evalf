from __future__ import annotations

import asyncio
from contextlib import suppress

from evalf.executor import execute_cases
from evalf.llms.base import BaseLLMModel
from evalf.llms.factory import build_llm
from evalf.metrics.base import BaseMetric
from evalf.schemas import EvalCase, RunReport


class Evaluator:
    """Coordinate evaluation runs for a collection of cases and metrics."""

    def __init__(
        self,
        *,
        judge: BaseLLMModel | None = None,
        concurrency: int = 4,
        per_sample_timeout_seconds: float | None = None,
    ) -> None:
        if judge is None:
            self.judge: BaseLLMModel | None = None
            self._owns_judge = True
        elif isinstance(judge, BaseLLMModel):
            self.judge = judge
            self._owns_judge = False
        else:
            raise TypeError("judge must be a BaseLLMModel instance or None.")
        self.concurrency = concurrency
        self.per_sample_timeout_seconds = per_sample_timeout_seconds

    def _get_judge(self) -> BaseLLMModel:
        if self.judge is None:
            self.judge = build_llm()
        return self.judge

    async def _cleanup_judge(self) -> None:
        if not self._owns_judge or self.judge is None:
            return

        judge = self.judge
        self.judge = None
        with suppress(Exception):
            await judge.aclose()
        with suppress(Exception):
            judge.close()

    async def a_evaluate(
        self,
        *,
        cases: list[EvalCase],
        metrics: list[BaseMetric],
    ) -> RunReport:
        """Asynchronously evaluate the provided cases with the configured judge."""
        judge = self._get_judge()
        try:
            return await execute_cases(
                cases=cases,
                metrics=metrics,
                llm=judge,
                concurrency=self.concurrency,
                per_sample_timeout_seconds=self.per_sample_timeout_seconds,
            )
        finally:
            await self._cleanup_judge()

    def evaluate(
        self,
        *,
        cases: list[EvalCase],
        metrics: list[BaseMetric],
    ) -> RunReport:
        """Synchronously evaluate the provided cases outside an active event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.a_evaluate(
                    cases=cases,
                    metrics=metrics,
                )
            )
        raise RuntimeError(
            "evaluate() cannot run inside an active event loop. Use a_evaluate() instead."
        )


async def a_evaluate(
    *,
    cases: list[EvalCase],
    metrics: list[BaseMetric],
    judge: BaseLLMModel | None = None,
    concurrency: int = 4,
    per_sample_timeout_seconds: float | None = None,
) -> RunReport:
    """Convenience async helper that builds an `Evaluator` and runs it once."""
    evaluator = Evaluator(
        judge=judge,
        concurrency=concurrency,
        per_sample_timeout_seconds=per_sample_timeout_seconds,
    )
    return await evaluator.a_evaluate(cases=cases, metrics=metrics)


def evaluate(
    *,
    cases: list[EvalCase],
    metrics: list[BaseMetric],
    judge: BaseLLMModel | None = None,
    concurrency: int = 4,
    per_sample_timeout_seconds: float | None = None,
) -> RunReport:
    """Convenience sync helper that builds an `Evaluator` and runs it once."""
    evaluator = Evaluator(
        judge=judge,
        concurrency=concurrency,
        per_sample_timeout_seconds=per_sample_timeout_seconds,
    )
    return evaluator.evaluate(cases=cases, metrics=metrics)
