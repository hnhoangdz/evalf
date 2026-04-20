import pytest
from click.testing import CliRunner

from evalf.cli import (
    _build_judge_settings,
    _parse_threshold_overrides,
    _resolve_cases,
    _run_command,
    cli,
    main,
)
from evalf.schemas import RunReport, RunSummary
from tests.helpers import SequenceLLM

pytestmark = pytest.mark.unit


def test_parse_threshold_overrides_accepts_multiple_metrics() -> None:
    overrides = _parse_threshold_overrides(["faithfulness=0.8", "answer_correctness = 0.9"])

    assert overrides == {
        "faithfulness": 0.8,
        "answer_correctness": 0.9,
    }


def test_parse_threshold_overrides_rejects_invalid_entries() -> None:
    with pytest.raises(ValueError, match="Invalid metric threshold override"):
        _parse_threshold_overrides(["faithfulness"])


def test_resolve_cases_builds_direct_sample_from_cli_values() -> None:
    cases = _resolve_cases(question="q", actual_output="a", expected_output="e")

    assert len(cases) == 1
    assert cases[0].question == "q"
    assert cases[0].expected_output == "e"


def test_cli_lists_metrics() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["list-metrics"])

    assert result.exit_code == 0
    assert "faithfulness" in result.output
    assert "answer_correctness" in result.output


def test_main_run_uses_runtime_defaults_and_writes_report(monkeypatch, capsys) -> None:
    calls = {}
    built_judge = SequenceLLM([])

    def fake_load_runtime_settings():
        return type(
            "Settings",
            (),
            {
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key": "key",
                "metrics": ["faithfulness"],
                "threshold": 0.7,
                "metric_mode": "pass@k",
                "k": 1,
                "request_timeout_seconds": 60.0,
                "per_sample_timeout_seconds": None,
                "max_retries": 3,
                "temperature": 0.0,
                "max_tokens": 800,
                "concurrency": 4,
                "output_path": "report.json",
                "c4_include_reason": True,
                "c4_need_summary_reason": False,
                "c4_strict_mode": False,
            },
        )()

    def fake_build_metrics(names, **kwargs):
        calls["metrics"] = (names, kwargs)
        return ["built-metric"]

    def fake_evaluate(**kwargs):
        calls["evaluate"] = kwargs
        return RunReport(
            run_id="run_test",
            summary=RunSummary(
                total_samples=1,
                passed_samples=1,
                failed_samples=0,
                skipped_samples=0,
            ),
            samples=[],
        )

    def fake_write_report(report, path):
        calls["write_report"] = (report.run_id, path)

    def fake_build_llm(**kwargs):
        built_judge.timeout_seconds = kwargs["timeout_seconds"]
        built_judge.temperature = kwargs["temperature"]
        built_judge.max_tokens = kwargs["max_tokens"]
        built_judge.max_retries = kwargs["max_retries"]
        return built_judge

    monkeypatch.setattr("evalf.cli.load_runtime_settings", fake_load_runtime_settings)
    monkeypatch.setattr("evalf.cli.build_metrics", fake_build_metrics)
    monkeypatch.setattr("evalf.cli.build_llm", fake_build_llm)
    monkeypatch.setattr("evalf.cli.evaluate", fake_evaluate)
    monkeypatch.setattr("evalf.cli.write_report", fake_write_report)
    monkeypatch.setattr("evalf.cli.report_to_json", lambda report: '{"run_id":"run_test"}')

    main(["run", "--question", "q", "--actual-output", "a"])
    output = capsys.readouterr().out

    assert calls["metrics"][0] == ["faithfulness"]
    assert calls["evaluate"]["concurrency"] == 4
    assert calls["evaluate"]["judge"].timeout_seconds == 60.0
    assert calls["evaluate"]["per_sample_timeout_seconds"] is None
    assert calls["write_report"] == ("run_test", "report.json")
    assert '{"run_id":"run_test"}' in output
    assert built_judge.closed is True
    assert built_judge.aclosed is True


def test_main_without_subcommand_prints_help(capsys) -> None:
    main([])
    output = capsys.readouterr().out

    assert "usage:" in output.lower()


def test_main_run_supports_distinct_request_and_per_sample_timeouts(monkeypatch) -> None:
    calls = {}
    built_judge = SequenceLLM([])

    def fake_load_runtime_settings():
        return type(
            "Settings",
            (),
            {
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key": "key",
                "metrics": ["faithfulness"],
                "threshold": 0.7,
                "metric_mode": "pass@k",
                "k": 1,
                "request_timeout_seconds": 60.0,
                "per_sample_timeout_seconds": 90.0,
                "max_retries": 3,
                "temperature": 0.0,
                "max_tokens": 800,
                "concurrency": 4,
                "output_path": None,
                "c4_include_reason": True,
                "c4_need_summary_reason": False,
                "c4_strict_mode": False,
            },
        )()

    monkeypatch.setattr("evalf.cli.load_runtime_settings", fake_load_runtime_settings)
    monkeypatch.setattr("evalf.cli.build_metrics", lambda *args, **kwargs: ["built-metric"])

    def fake_build_llm(**kwargs):
        built_judge.timeout_seconds = kwargs["timeout_seconds"]
        built_judge.temperature = kwargs["temperature"]
        built_judge.max_tokens = kwargs["max_tokens"]
        built_judge.max_retries = kwargs["max_retries"]
        return built_judge

    monkeypatch.setattr("evalf.cli.build_llm", fake_build_llm)

    def fake_evaluate(**kwargs):
        calls["evaluate"] = kwargs
        return RunReport(
            run_id="run_test",
            summary=RunSummary(
                total_samples=1,
                passed_samples=1,
                failed_samples=0,
                skipped_samples=0,
            ),
            samples=[],
        )

    monkeypatch.setattr("evalf.cli.evaluate", fake_evaluate)
    monkeypatch.setattr("evalf.cli.write_report", lambda *args, **kwargs: None)
    monkeypatch.setattr("evalf.cli.report_to_json", lambda report: "{}")

    main(
        [
            "run",
            "--question",
            "q",
            "--actual-output",
            "a",
            "--request-timeout-seconds",
            "12",
            "--per-sample-timeout-seconds",
            "34",
        ]
    )

    assert calls["evaluate"]["judge"].timeout_seconds == 12
    assert calls["evaluate"]["per_sample_timeout_seconds"] == 34
    assert built_judge.closed is True
    assert built_judge.aclosed is True


def test_main_run_passes_c4_options_to_build_metrics(monkeypatch, capsys) -> None:
    calls = {}
    built_judge = SequenceLLM([])

    def fake_load_runtime_settings():
        return type(
            "Settings",
            (),
            {
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key": "key",
                "metrics": ["c4"],
                "threshold": 0.7,
                "metric_mode": "pass@k",
                "k": 1,
                "request_timeout_seconds": 60.0,
                "per_sample_timeout_seconds": None,
                "max_retries": 3,
                "temperature": 0.0,
                "max_tokens": 800,
                "concurrency": 4,
                "output_path": None,
                "c4_include_reason": True,
                "c4_need_summary_reason": False,
                "c4_strict_mode": False,
            },
        )()

    def fake_build_metrics(names, **kwargs):
        calls["metrics"] = (names, kwargs)
        return ["built-metric"]

    def fake_evaluate(**kwargs):
        return RunReport(
            run_id="run_test",
            summary=RunSummary(
                total_samples=1,
                passed_samples=1,
                failed_samples=0,
                skipped_samples=0,
            ),
            samples=[],
        )

    def fake_build_llm(**kwargs):
        built_judge.timeout_seconds = kwargs["timeout_seconds"]
        built_judge.temperature = kwargs["temperature"]
        built_judge.max_tokens = kwargs["max_tokens"]
        built_judge.max_retries = kwargs["max_retries"]
        return built_judge

    monkeypatch.setattr("evalf.cli.load_runtime_settings", fake_load_runtime_settings)
    monkeypatch.setattr("evalf.cli.build_metrics", fake_build_metrics)
    monkeypatch.setattr("evalf.cli.build_llm", fake_build_llm)
    monkeypatch.setattr("evalf.cli.evaluate", fake_evaluate)
    monkeypatch.setattr("evalf.cli.write_report", lambda *args, **kwargs: None)
    monkeypatch.setattr("evalf.cli.report_to_json", lambda report: "{}")

    main(
        [
            "run",
            "--question",
            "q",
            "--actual-output",
            "a",
            "--metrics",
            "c4",
            "--no-c4-include-reason",
            "--c4-summary-reason",
            "--c4-strict-mode",
        ]
    )
    capsys.readouterr()

    assert calls["metrics"][0] == ["c4"]
    assert calls["metrics"][1]["metric_options"]["c4"] == {
        "include_reason": False,
        "need_summary_reason": True,
        "strict_mode": True,
    }
    assert built_judge.closed is True
    assert built_judge.aclosed is True


def test_cli_reports_invalid_metric_without_traceback() -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["run", "--question", "q", "--actual-output", "a", "--metrics", "nope"],
    )

    assert result.exit_code == 1
    assert "Unsupported metric: nope" in result.output
    assert "Traceback" not in result.output


def test_cli_reports_malformed_json_items_without_traceback() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["run", "--sample-json", "[1]"])

    assert result.exit_code == 1
    assert "item 0 has type int" in result.output
    assert "Traceback" not in result.output


def test_run_help_describes_core_options() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--question TEXT" in result.output
    assert "Question associated with the sample being" in result.output
    assert "Judge provider name, for example openai or" in result.output
    assert "--threshold FLOAT" in result.output
    assert "Default pass/fail threshold applied to all" in result.output
    assert "--temperature FLOAT" in result.output
    assert "Sampling temperature passed to the judge" in result.output
    assert "--max-tokens INTEGER" in result.output
    assert "Maximum completion tokens requested from the" in result.output
    assert "Extensionless paths default to .json" in result.output
    assert "Maximum number of samples to evaluate" in result.output
    assert "--c4-strict-mode / --no-c4-strict-mode" in result.output


def test_build_judge_settings_preserves_explicit_zero_max_tokens() -> None:
    defaults = type(
        "Settings",
        (),
        {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "base_url": "https://api.openai.com/v1",
            "api_key": "key",
            "request_timeout_seconds": 60.0,
            "max_retries": 3,
            "temperature": 0.0,
            "max_tokens": 800,
        },
    )()

    settings = _build_judge_settings(
        defaults=defaults,
        provider=None,
        model=None,
        base_url=None,
        api_key=None,
        request_timeout_seconds=None,
        temperature=None,
        max_tokens=0,
    )

    assert settings.max_tokens == 0


def test_run_command_preserves_explicit_zero_k(monkeypatch) -> None:
    calls = {}
    built_judge = SequenceLLM([])

    def fake_load_runtime_settings():
        return type(
            "Settings",
            (),
            {
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key": "key",
                "metrics": ["faithfulness"],
                "threshold": 0.7,
                "metric_mode": "pass@k",
                "k": 1,
                "request_timeout_seconds": 60.0,
                "per_sample_timeout_seconds": None,
                "max_retries": 3,
                "temperature": 0.0,
                "max_tokens": 800,
                "concurrency": 4,
                "output_path": None,
                "c4_include_reason": True,
                "c4_need_summary_reason": False,
                "c4_strict_mode": False,
            },
        )()

    def fake_build_metrics(names, **kwargs):
        calls["k"] = kwargs["k"]
        return ["built-metric"]

    def fake_evaluate(**kwargs):
        return RunReport(
            run_id="run_test",
            summary=RunSummary(
                total_samples=1,
                passed_samples=0,
                failed_samples=0,
                skipped_samples=1,
            ),
            samples=[],
        )

    monkeypatch.setattr("evalf.cli.load_runtime_settings", fake_load_runtime_settings)
    monkeypatch.setattr("evalf.cli.build_metrics", fake_build_metrics)
    monkeypatch.setattr("evalf.cli.build_llm", lambda **kwargs: built_judge)
    monkeypatch.setattr("evalf.cli.evaluate", fake_evaluate)
    monkeypatch.setattr("evalf.cli.write_report", lambda *args, **kwargs: None)
    monkeypatch.setattr("evalf.cli.report_to_json", lambda report: "{}")

    _run_command(
        input_path=None,
        sample_json=None,
        question="q",
        actual_output="a",
        expected_output=None,
        retrieved_contexts=(),
        reference_contexts=(),
        provider=None,
        model=None,
        base_url=None,
        api_key=None,
        metrics=None,
        metric_mode=None,
        k=0,
        threshold=None,
        metric_threshold=(),
        concurrency=None,
        request_timeout_seconds=None,
        per_sample_timeout_seconds=None,
        temperature=None,
        max_tokens=None,
        output=None,
        c4_include_reason=None,
        c4_summary_reason=None,
        c4_strict_mode=None,
    )

    assert calls["k"] == 0
