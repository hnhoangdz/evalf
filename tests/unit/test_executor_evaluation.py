import asyncio

import pytest
from pydantic import BaseModel

from evalf.evaluation import Evaluator, a_evaluate, evaluate
from evalf.executor import execute_cases, execute_cases_sync
from evalf.llms.base import BaseLLMModel
from evalf.metrics.base import BaseDecomposedMetric, BaseMetric
from evalf.schemas import EvalAttempt, EvalCase, LLMResponse, MetricResult, UsageStats

pytestmark = pytest.mark.unit


class DummyAssessment(BaseModel):
    score: float
    reason: str | None = None


class QueueLLM(BaseLLMModel):
    def __init__(self, outputs):
        super().__init__(
            provider="test",
            model="fake-model",
            base_url="https://example.com",
            api_key=None,
        )
        self._outputs = list(outputs)
        self.close_called = False
        self.aclose_called = False

    def generate(self, **kwargs) -> LLMResponse:
        if not self._outputs:
            raise AssertionError("No fake outputs left for generate().")
        parsed_output = self._outputs.pop(0)
        return LLMResponse(
            text=None,
            model=self.model,
            provider=self.provider,
            parsed_output=parsed_output,
            usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15, latency_ms=7.5),
        )

    async def a_generate(self, **kwargs) -> LLMResponse:
        return self.generate(**kwargs)

    def close(self) -> None:
        self.close_called = True

    async def aclose(self) -> None:
        self.aclose_called = True


class DummyMetric(BaseMetric):
    name = "dummy"
    required_inputs = ("question", "actual_output")
    output_schema = DummyAssessment

    def build_prompt(self, case):
        return "system", "user"


class SyncOnlyDecomposedMetric(BaseDecomposedMetric):
    name = "sync_only"
    required_inputs = ("question", "actual_output")

    def compute_assessment(
        self, case: EvalCase, llm: BaseLLMModel
    ) -> tuple[float, str | None, UsageStats]:
        return 1.0, "sync-only", UsageStats.empty()


class SlowMetric:
    name = "slow"

    def measure(self, case, llm) -> MetricResult:
        return MetricResult(
            name=self.name,
            status="passed",
            score=1.0,
            threshold=0.0,
            passed=True,
        )

    async def a_measure(self, case, llm) -> MetricResult:
        await asyncio.sleep(0.05)
        return self.measure(case, llm)


class CancelledMetric:
    name = "cancelled"

    def measure(self, case, llm) -> MetricResult:
        raise asyncio.CancelledError()

    async def a_measure(self, case, llm) -> MetricResult:
        raise asyncio.CancelledError()


class ConcurrencyTrackingLLM(BaseLLMModel):
    def __init__(self) -> None:
        super().__init__(
            provider="test",
            model="fake-model",
            base_url="https://example.com",
            api_key=None,
        )
        self.inflight = 0
        self.max_inflight = 0

    def generate(self, **kwargs) -> LLMResponse:
        return LLMResponse(
            text=None,
            model=self.model,
            provider=self.provider,
            parsed_output=DummyAssessment(score=1.0, reason="ok"),
            usage=UsageStats.empty(),
        )

    async def a_generate(self, **kwargs) -> LLMResponse:
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        await asyncio.sleep(0.01)
        self.inflight -= 1
        return self.generate(**kwargs)


def test_metric_pass_at_k_aggregates_trial_statistics() -> None:
    metric = DummyMetric(threshold=0.7, mode="pass@k", k=2)
    llm = QueueLLM(
        [
            DummyAssessment(score=0.4, reason="Weak answer."),
            DummyAssessment(score=0.9, reason="Strong answer."),
        ]
    )
    case = EvalCase(
        question="What is FERPA?",
        attempts=[
            EvalAttempt(actual_output="Attempt 1"),
            EvalAttempt(actual_output="Attempt 2"),
        ],
    )

    result = metric.measure(case, llm)

    assert result.status == "passed"
    assert result.score == 0.9
    assert result.evaluated_k == 2
    assert result.successful_trials == 1
    assert result.best_trial_score == 0.9
    assert result.worst_trial_score == 0.4
    assert result.mean_trial_score == 0.65
    assert "pass@k" in (result.reason or "")


def test_metric_skips_when_requested_k_exceeds_available_attempts() -> None:
    metric = DummyMetric(threshold=0.7, mode="pass@k", k=2)
    case = EvalCase(
        question="What is FERPA?",
        attempts=[EvalAttempt(actual_output="Only one attempt")],
    )

    result = metric.measure(case, QueueLLM([]))

    assert result.status == "skipped"
    assert result.passed is None
    assert result.evaluated_k == 1
    assert "Requested k=2" in (result.reason or "")


def test_async_metric_trials_run_concurrently_for_pass_at_k() -> None:
    metric = DummyMetric(threshold=0.7, mode="pass@k", k=3)
    llm = ConcurrencyTrackingLLM()
    case = EvalCase(
        question="What is FERPA?",
        attempts=[
            EvalAttempt(actual_output="Attempt 1"),
            EvalAttempt(actual_output="Attempt 2"),
            EvalAttempt(actual_output="Attempt 3"),
        ],
    )

    result = asyncio.run(metric.a_measure(case, llm))

    assert result.status == "passed"
    assert result.evaluated_k == 3
    assert llm.max_inflight >= 2


def test_async_decomposed_metric_offloads_sync_fallback(monkeypatch) -> None:
    calls = {}

    async def fake_to_thread(func, *args):
        calls["func"] = func
        calls["args"] = args
        return func(*args)

    monkeypatch.setattr("evalf.metrics.base.asyncio.to_thread", fake_to_thread)
    metric = SyncOnlyDecomposedMetric()

    result = asyncio.run(
        metric.a_measure(
            EvalCase(question="What is FERPA?", actual_output="A privacy law."),
            QueueLLM([]),
        )
    )

    assert result.status == "passed"
    assert calls["func"] == metric.compute_assessment


def test_evaluator_sync_closes_judge() -> None:
    evaluator = Evaluator(judge=QueueLLM([DummyAssessment(score=0.9, reason="ok")]))

    report = evaluator.evaluate(
        cases=[EvalCase(question="What is FERPA?", actual_output="A privacy law.")],
        metrics=[DummyMetric()],
    )

    assert report.samples[0].status == "passed"
    assert evaluator.judge is not None
    assert evaluator.judge.close_called is False
    assert evaluator.judge.aclose_called is False


def test_evaluator_async_closes_judge() -> None:
    async def run_test() -> None:
        evaluator = Evaluator(judge=QueueLLM([DummyAssessment(score=0.9, reason="ok")]))
        report = await evaluator.a_evaluate(
            cases=[EvalCase(question="What is FERPA?", actual_output="A privacy law.")],
            metrics=[DummyMetric()],
        )

        assert report.samples[0].status == "passed"
        assert evaluator.judge is not None
        assert evaluator.judge.aclose_called is False
        assert evaluator.judge.close_called is False

    asyncio.run(run_test())


def test_evaluator_sync_rejects_active_event_loop() -> None:
    async def run_test() -> None:
        evaluator = Evaluator(judge=QueueLLM([DummyAssessment(score=0.9, reason="ok")]))
        with pytest.raises(RuntimeError, match="Use a_evaluate\\(\\) instead"):
            evaluator.evaluate(
                cases=[EvalCase(question="What is FERPA?", actual_output="A privacy law.")],
                metrics=[DummyMetric()],
            )

    asyncio.run(run_test())


def test_execute_cases_timeout_preserves_case_id_and_metadata() -> None:
    async def run_test():
        return await execute_cases(
            cases=[EvalCase(id="case-timeout", metadata={"source": "fixture"})],
            metrics=[SlowMetric()],
            llm=QueueLLM([]),
            per_sample_timeout_seconds=0.001,
        )

    report = asyncio.run(run_test())

    assert report.samples[0].sample_id == "case-timeout"
    assert report.samples[0].metadata == {"source": "fixture"}
    assert report.samples[0].metrics[0].name == "executor"
    assert "TimeoutError" in (report.samples[0].metrics[0].error or "")


def test_execute_cases_converts_cancelled_error_results_into_executor_errors() -> None:
    async def run_test():
        return await execute_cases(
            cases=[EvalCase(id="case-cancel", question="q", actual_output="a")],
            metrics=[CancelledMetric()],
            llm=QueueLLM([]),
        )

    report = asyncio.run(run_test())

    assert report.samples[0].sample_id == "case-cancel"
    assert report.samples[0].status == "failed"
    assert report.samples[0].metrics[0].name == "executor"
    assert "CancelledError" in (report.samples[0].metrics[0].error or "")


def test_execute_cases_sync_converts_cancelled_error_results_into_executor_errors() -> None:
    report = execute_cases_sync(
        cases=[EvalCase(id="case-cancel", question="q", actual_output="a")],
        metrics=[CancelledMetric()],
        llm=QueueLLM([]),
    )

    assert report.samples[0].sample_id == "case-cancel"
    assert report.samples[0].status == "failed"
    assert report.samples[0].metrics[0].name == "executor"
    assert "CancelledError" in (report.samples[0].metrics[0].error or "")


def test_module_level_evaluate_helpers_delegate_correctly() -> None:
    sync_report = evaluate(
        cases=[EvalCase(question="What is FERPA?", actual_output="A privacy law.")],
        metrics=[DummyMetric()],
        judge=QueueLLM([DummyAssessment(score=0.9, reason="ok")]),
    )

    async_report = asyncio.run(
        a_evaluate(
            cases=[EvalCase(question="What is FERPA?", actual_output="A privacy law.")],
            metrics=[DummyMetric()],
            judge=QueueLLM([DummyAssessment(score=0.9, reason="ok")]),
        )
    )

    assert sync_report.samples[0].status == "passed"
    assert async_report.samples[0].status == "passed"


def test_evaluator_builds_judge_from_environment_when_not_provided(monkeypatch) -> None:
    built_judge = QueueLLM([DummyAssessment(score=0.9, reason="ok")])
    monkeypatch.setattr("evalf.evaluation.build_llm", lambda: built_judge)

    report = Evaluator().evaluate(
        cases=[EvalCase(question="What is FERPA?", actual_output="A privacy law.")],
        metrics=[DummyMetric()],
    )

    assert report.samples[0].status == "passed"
    assert built_judge.close_called is True
    assert built_judge.aclose_called is True


def test_evaluator_rebuilds_owned_judge_for_each_run(monkeypatch) -> None:
    built_judges = [
        QueueLLM([DummyAssessment(score=0.9, reason="run-1")]),
        QueueLLM([DummyAssessment(score=0.95, reason="run-2")]),
    ]
    monkeypatch.setattr("evalf.evaluation.build_llm", lambda: built_judges.pop(0))
    evaluator = Evaluator()

    first_report = evaluator.evaluate(
        cases=[EvalCase(question="What is FERPA?", actual_output="A privacy law.")],
        metrics=[DummyMetric()],
    )
    second_report = evaluator.evaluate(
        cases=[EvalCase(question="What is FERPA?", actual_output="A privacy law.")],
        metrics=[DummyMetric()],
    )

    assert first_report.samples[0].status == "passed"
    assert second_report.samples[0].status == "passed"
    assert evaluator.judge is None
    assert built_judges == []


def test_sync_evaluate_honors_per_sample_timeout() -> None:
    report = Evaluator(
        judge=QueueLLM([]),
        per_sample_timeout_seconds=0.001,
    ).evaluate(
        cases=[EvalCase(id="case-timeout")],
        metrics=[SlowMetric()],
    )

    assert report.samples[0].sample_id == "case-timeout"
    assert report.samples[0].metrics[0].name == "executor"
    assert "TimeoutError" in (report.samples[0].metrics[0].error or "")


def test_execute_cases_uses_stable_generated_sample_ids_for_successful_cases() -> None:
    report = evaluate(
        cases=[EvalCase(question="What is FERPA?", actual_output="A privacy law.")],
        metrics=[DummyMetric()],
        judge=QueueLLM([DummyAssessment(score=0.9, reason="ok")]),
    )

    assert report.samples[0].sample_id == "sample-1"


def test_evaluator_cleanup_runs_when_async_execution_raises(monkeypatch) -> None:
    async def fake_execute_cases(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("evalf.evaluation.execute_cases", fake_execute_cases)
    evaluator = Evaluator()
    built_judge = QueueLLM([])
    monkeypatch.setattr("evalf.evaluation.build_llm", lambda: built_judge)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(
            evaluator.a_evaluate(
                cases=[EvalCase(question="q", actual_output="a")],
                metrics=[DummyMetric()],
            )
        )

    assert evaluator.judge is None
    assert built_judge.close_called is True
    assert built_judge.aclose_called is True
