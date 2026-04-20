import asyncio

import pytest

from evalf.llms.providers import ClaudeLLMModel, GeminiLLMModel, OpenAILLMModel
from tests.llms_helpers import FakeProviderClient

pytestmark = pytest.mark.unit


def test_provider_models_estimate_cost_and_close_client(monkeypatch) -> None:
    monkeypatch.setattr("evalf.llms.providers.OpenAIClient", FakeProviderClient)
    model = OpenAILLMModel(
        model="gpt-4.1-mini",
        base_url="https://example.com/v1",
        api_key="test-key",  # pragma: allowlist secret
    )

    response = model.generate(
        system_prompt="system",
        user_prompt="user",
        output_schema=None,
    )
    async_response = asyncio.run(
        model.a_generate(
            system_prompt="system",
            user_prompt="user",
            output_schema=None,
        )
    )

    assert response.usage.cost_usd == 0.0012
    assert async_response.usage.cost_usd == 0.0012
    model.close()
    asyncio.run(model.aclose())
    assert model.client.closed is True
    assert model.client.aclosed is True


def test_provider_models_reject_empty_prompts(monkeypatch) -> None:
    monkeypatch.setattr("evalf.llms.providers.OpenAIClient", FakeProviderClient)
    model = OpenAILLMModel(
        model="gpt-4.1-mini",
        base_url="https://example.com/v1",
        api_key="test-key",  # pragma: allowlist secret
    )

    with pytest.raises(ValueError, match="system_prompt must be a non-empty string"):
        model.generate(system_prompt=" ", user_prompt="user", output_schema=None)

    with pytest.raises(ValueError, match="user_prompt must be a non-empty string"):
        asyncio.run(model.a_generate(system_prompt="system", user_prompt="  ", output_schema=None))


@pytest.mark.parametrize(
    ("model_cls", "expected_provider"),
    [
        (OpenAILLMModel, "openai"),
        (GeminiLLMModel, "gemini"),
        (ClaudeLLMModel, "claude"),
    ],
)
def test_provider_wrappers_set_expected_provider_name(
    monkeypatch, model_cls, expected_provider
) -> None:
    monkeypatch.setattr("evalf.llms.providers.OpenAIClient", FakeProviderClient)

    model = model_cls(
        model="test-model",
        base_url="https://example.com/v1",
        api_key="test-key",  # pragma: allowlist secret
    )

    assert model.provider == expected_provider
