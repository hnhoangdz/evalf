from __future__ import annotations

from pydantic import BaseModel

from evalf.llms.base import BaseLLMModel
from evalf.llms.client import OpenAIClient
from evalf.llms.pricing import estimate_cost_usd
from evalf.schemas import LLMResponse


class LLMModel(BaseLLMModel):
    """Shared OpenAI-compatible model implementation used across providers."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 800,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.client = OpenAIClient(
            base_url=self.base_url,
            api_key=self.api_key,
            provider=self.provider,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            default_headers=default_headers,
        )

    @staticmethod
    def _validate_prompt(name: str, value: str) -> None:
        """Reject empty prompts before making a paid provider request."""

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string.")

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a synchronous judge response and attach estimated cost."""
        self._validate_prompt("system_prompt", system_prompt)
        self._validate_prompt("user_prompt", user_prompt)
        response = self.client.create_chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            output_schema=output_schema,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        response.usage.cost_usd = estimate_cost_usd(
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response

    async def a_generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate an asynchronous judge response and attach estimated cost."""
        self._validate_prompt("system_prompt", system_prompt)
        self._validate_prompt("user_prompt", user_prompt)
        response = await self.client.acreate_chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            output_schema=output_schema,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        response.usage.cost_usd = estimate_cost_usd(
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response

    def close(self) -> None:
        """Close the provider client."""
        self.client.close()

    async def aclose(self) -> None:
        """Asynchronously close the provider client."""
        await self.client.aclose()


class _ProviderLLMModel(LLMModel):
    """Provider-specific wrapper that hardcodes the provider name."""

    provider_name: str

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 800,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        provider_name = getattr(type(self), "provider_name", "")
        if not isinstance(provider_name, str) or not provider_name:
            raise TypeError(
                "_ProviderLLMModel subclasses must define a non-empty provider_name string."
            )

        super().__init__(
            provider=provider_name,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            default_headers=default_headers,
        )


class OpenAILLMModel(_ProviderLLMModel):
    """OpenAI provider model backed by the shared OpenAI-compatible transport."""

    provider_name = "openai"


class GeminiLLMModel(_ProviderLLMModel):
    """Gemini provider model backed by the shared OpenAI-compatible transport."""

    provider_name = "gemini"


class ClaudeLLMModel(_ProviderLLMModel):
    """Claude provider model backed by the shared OpenAI-compatible transport."""

    provider_name = "claude"
