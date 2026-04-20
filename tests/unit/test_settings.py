import pytest

from evalf.settings import (
    RuntimeSettings,
    load_default_environment,
    load_runtime_settings,
    resolve_api_key,
)

pytestmark = pytest.mark.unit


def test_load_runtime_settings_reads_dotenv_files_with_expected_precedence(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EVALF_MODEL", raising=False)
    monkeypatch.delenv("EVALF_THRESHOLD", raising=False)
    monkeypatch.delenv("EVALF_METRICS", raising=False)
    monkeypatch.delenv("EVALF_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("EVALF_REQUEST_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("EVALF_PER_SAMPLE_TIMEOUT_SECONDS", raising=False)
    (tmp_path / ".env").write_text(
        "EVALF_MODEL=from-dotenv\n"
        "EVALF_THRESHOLD='0.8'\n"
        "EVALF_METRICS=faithfulness, answer_relevance\n"
        "EVALF_TIMEOUT_SECONDS=12\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text(
        'EVALF_MODEL="from-dotenv-local"\nEVALF_PER_SAMPLE_TIMEOUT_SECONDS=45\n',
        encoding="utf-8",
    )

    settings = load_runtime_settings()

    assert settings.model == "from-dotenv-local"
    assert settings.threshold == 0.8
    assert settings.metrics == ["faithfulness", "answer_relevance"]
    assert settings.request_timeout_seconds == 12
    assert settings.per_sample_timeout_seconds == 45


def test_load_default_environment_does_not_override_existing_environment(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVALF_MODEL", "from-shell")
    (tmp_path / ".env").write_text("EVALF_MODEL=from-dotenv\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("EVALF_MODEL=from-dotenv-local\n", encoding="utf-8")

    load_default_environment()

    assert load_runtime_settings().model == "from-shell"


def test_resolve_api_key_prefers_global_and_then_provider_specific(monkeypatch) -> None:
    monkeypatch.setenv("EVALF_API_KEY", "global-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "claude-key")

    assert resolve_api_key("openai") == "global-key"
    assert resolve_api_key("gemini") == "global-key"
    assert resolve_api_key("claude") == "global-key"

    monkeypatch.delenv("EVALF_API_KEY")

    assert resolve_api_key("openai") == "openai-key"
    assert resolve_api_key("gemini") == "gemini-key"
    assert resolve_api_key("claude") == "claude-key"


def test_new_request_timeout_env_overrides_legacy_timeout(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVALF_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("EVALF_REQUEST_TIMEOUT_SECONDS", "34")

    settings = load_runtime_settings()

    assert settings.request_timeout_seconds == 34


def test_unknown_c4_runtime_settings_are_ignored_from_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVALF_C4_INCLUDE_REASON", "false")
    monkeypatch.setenv("EVALF_C4_NEED_SUMMARY_REASON", "true")
    monkeypatch.setenv("EVALF_C4_STRICT_MODE", "true")

    settings = load_runtime_settings()

    assert not hasattr(settings, "c4_include_reason")
    assert not hasattr(settings, "c4_need_summary_reason")
    assert not hasattr(settings, "c4_strict_mode")


def test_load_runtime_settings_supports_utf8_bom_in_dotenv(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EVALF_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "\ufeffEVALF_MODEL=from-bom-dotenv\n",
        encoding="utf-8",
    )

    settings = load_runtime_settings()

    assert settings.model == "from-bom-dotenv"


def test_runtime_settings_direct_construction_normalizes_provider_and_metrics() -> None:
    settings = RuntimeSettings(
        provider=" OpenAI ",
        metrics="faithfulness, answer_relevance",
        api_key="explicit-key",
    )

    assert settings.provider == "openai"
    assert settings.metrics == ["faithfulness", "answer_relevance"]
    assert settings.api_key == "explicit-key"
