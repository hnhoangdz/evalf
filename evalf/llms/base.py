from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from evalf.schemas import LLMResponse

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def normalize_base_url(base_url: str) -> str:
    """Normalize OpenAI-compatible base URLs into a canonical no-trailing-slash form."""

    return base_url.rstrip("/")


class BaseLLMModel(ABC):
    """Abstract interface for judge backends used by metrics and evaluators."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 800,
        max_retries: int = 3,
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_url = normalize_base_url(base_url)
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[SchemaT] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a synchronous model response for the provided prompt pair."""
        raise NotImplementedError

    @abstractmethod
    async def a_generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[SchemaT] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate an asynchronous model response for the provided prompt pair."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release any asynchronous resources held by the model client."""
        return None

    def close(self) -> None:
        """Release any synchronous resources held by the model client."""
        return None
