from __future__ import annotations

from evalf.llms.base import BaseLLMModel, normalize_base_url
from evalf.llms.providers import ClaudeLLMModel, GeminiLLMModel, OpenAILLMModel
from evalf.settings import load_runtime_settings, resolve_api_key

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}

UNSUPPORTED_NATIVE_BASE_URLS = {
    "claude": {"https://api.anthropic.com/v1"},
}
INVALID_PROVIDER_MODEL_PREFIXES = {
    "openai": ("gemini-", "claude-"),
    "gemini": ("gpt-", "o1", "o3", "o4", "claude-"),
    "claude": ("gpt-", "o1", "o3", "o4", "gemini-"),
}
PROVIDER_MODEL_EXAMPLES = {
    "openai": "gpt-4.1-mini",
    "gemini": "gemini-2.0-flash",
    "claude": "claude-sonnet-4",
}
PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "gemini": "Gemini",
    "claude": "Claude",
}


def _validate_provider_model(provider: str, model: str) -> None:
    """Reject obviously mismatched provider/model combinations early."""

    normalized_provider = provider.lower()
    normalized_model = model.lower()
    invalid_prefixes = INVALID_PROVIDER_MODEL_PREFIXES.get(normalized_provider)
    if invalid_prefixes and normalized_model.startswith(invalid_prefixes):
        provider_name = PROVIDER_DISPLAY_NAMES.get(normalized_provider, normalized_provider.title())
        example_model = PROVIDER_MODEL_EXAMPLES.get(normalized_provider, model)
        raise ValueError(
            f"{provider_name} provider requires a {provider_name} model name such as "
            f"'{example_model}'; "
            f"got '{model}'."
        )


def build_llm(
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    headers: dict[str, str] | None = None,
) -> BaseLLMModel:
    """Build an OpenAI-compatible judge client from explicit args or runtime settings."""
    settings = load_runtime_settings()

    resolved_provider = (provider or settings.provider).lower()
    resolved_model = model or settings.model
    provider_overridden = provider is not None and resolved_provider != settings.provider.lower()
    _validate_provider_model(resolved_provider, resolved_model)

    resolved_base_url = base_url
    if resolved_base_url is None and not provider_overridden:
        resolved_base_url = settings.base_url
    if resolved_base_url is None:
        resolved_base_url = DEFAULT_BASE_URLS.get(resolved_provider)
    if not resolved_base_url:
        if resolved_provider == "claude":
            raise ValueError(
                "Provider 'claude' requires an explicit OpenAI-compatible base URL. "
                "Native Anthropic endpoints are not supported by evalf's OpenAI transport."
            )
        raise ValueError(
            f"Provider '{resolved_provider}' requires an explicit base URL for its OpenAI-compatible endpoint."
        )

    normalized_base_url = normalize_base_url(resolved_base_url)
    if normalized_base_url in UNSUPPORTED_NATIVE_BASE_URLS.get(resolved_provider, set()):
        raise ValueError(
            f"Provider '{resolved_provider}' requires an OpenAI-compatible base URL. "
            f"The native endpoint '{normalized_base_url}' is not supported."
        )

    resolved_api_key = api_key or resolve_api_key(resolved_provider)
    if resolved_api_key is None and not provider_overridden:
        resolved_api_key = settings.api_key
    common_kwargs = {
        "model": resolved_model,
        "base_url": normalized_base_url,
        "api_key": resolved_api_key,
        "timeout_seconds": settings.timeout_seconds if timeout_seconds is None else timeout_seconds,
        "max_retries": settings.max_retries if max_retries is None else max_retries,
        "temperature": settings.temperature if temperature is None else temperature,
        "max_tokens": settings.max_tokens if max_tokens is None else max_tokens,
        "default_headers": headers or {},
    }

    if resolved_provider == "openai":
        return OpenAILLMModel(**common_kwargs)
    if resolved_provider == "gemini":
        return GeminiLLMModel(**common_kwargs)
    if resolved_provider == "claude":
        return ClaudeLLMModel(**common_kwargs)
    raise ValueError(f"Unsupported provider: {resolved_provider}")
