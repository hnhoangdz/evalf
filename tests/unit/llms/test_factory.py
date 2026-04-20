import pytest

from evalf.llms.factory import build_llm

pytestmark = pytest.mark.unit


def test_build_llm_uses_environment_defaults(monkeypatch) -> None:
    class FakeOpenAIModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("evalf.llms.factory.OpenAILLMModel", FakeOpenAIModel)
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.setenv("EVALF_PROVIDER", "openai")
    monkeypatch.setenv("EVALF_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("EVALF_REQUEST_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("EVALF_MAX_RETRIES", "2")
    monkeypatch.setenv("EVALF_TEMPERATURE", "0.1")
    monkeypatch.setenv("EVALF_MAX_TOKENS", "200")

    model = build_llm()

    assert model.kwargs["base_url"] == "https://api.openai.com/v1"
    assert model.kwargs["api_key"] == "env-openai-key"
    assert model.kwargs["timeout_seconds"] == 30.0
    assert model.kwargs["max_retries"] == 2
    assert model.kwargs["temperature"] == 0.1
    assert model.kwargs["max_tokens"] == 200


def test_build_llm_prefers_explicit_args_over_environment(monkeypatch) -> None:
    class FakeOpenAIModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("evalf.llms.factory.OpenAILLMModel", FakeOpenAIModel)
    monkeypatch.setenv("EVALF_PROVIDER", "openai")
    monkeypatch.setenv("EVALF_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")

    model = build_llm(
        model="gpt-4.1",
        api_key="explicit-key",
        timeout_seconds=12,
        max_tokens=99,
    )

    assert model.kwargs["model"] == "gpt-4.1"
    assert model.kwargs["api_key"] == "explicit-key"
    assert model.kwargs["timeout_seconds"] == 12
    assert model.kwargs["max_tokens"] == 99


def test_build_llm_does_not_reuse_provider_specific_env_from_other_provider(monkeypatch) -> None:
    class FakeOpenAIModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("evalf.llms.factory.OpenAILLMModel", FakeOpenAIModel)
    monkeypatch.setenv("EVALF_PROVIDER", "gemini")
    monkeypatch.setenv("EVALF_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv(
        "EVALF_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    model = build_llm(provider="openai", model="gpt-4.1-mini")

    assert model.kwargs["base_url"] == "https://api.openai.com/v1"
    assert model.kwargs["api_key"] == "openai-key"


def test_build_llm_rejects_unsupported_providers() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        build_llm(provider="custom", model="gpt-4.1-mini", base_url="https://x")


def test_build_llm_rejects_provider_model_mismatches() -> None:
    with pytest.raises(ValueError, match="Gemini provider requires a Gemini model name"):
        build_llm(provider="gemini", model="gpt-4.1-mini")

    with pytest.raises(ValueError, match="Gemini provider requires a Gemini model name"):
        build_llm(provider="gemini", model="claude-sonnet-4")

    with pytest.raises(ValueError, match="Claude provider requires a Claude model name"):
        build_llm(provider="claude", model="o1-mini")


def test_build_llm_requires_explicit_openai_compatible_base_url_for_claude() -> None:
    with pytest.raises(ValueError, match="Provider 'claude' requires an explicit OpenAI-compatible base URL"):
        build_llm(provider="claude", model="claude-sonnet-4")


def test_build_llm_rejects_native_anthropic_base_url_for_claude() -> None:
    with pytest.raises(ValueError, match="native endpoint 'https://api.anthropic.com/v1' is not supported"):
        build_llm(
            provider="claude",
            model="claude-sonnet-4",
            base_url="https://api.anthropic.com/v1/",
        )
