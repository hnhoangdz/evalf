from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from evalf.llms.base import normalize_base_url
from evalf.schemas import LLMResponse, UsageStats

PARSE_FALLBACK_EXCEPTIONS = (ValidationError, TypeError, ValueError)
RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    httpx.NetworkError,
    httpx.TimeoutException,
)
MAX_RETRY_DELAY_SECONDS = 60.0
logger = logging.getLogger(__name__)


class OpenAIClient:
    """Thin OpenAI-compatible client wrapper with structured-output fallback and retries."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        provider: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = normalize_base_url(base_url)
        self.api_key = api_key
        self.provider = provider
        self.max_retries = max_retries
        self.default_headers = default_headers or {}
        self._sync_closed = False
        self._async_closed = False
        timeout = httpx.Timeout(timeout_seconds)
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        )
        self._sync_http_client = httpx.Client(
            timeout=timeout,
            limits=limits,
            headers=self.default_headers or None,
        )
        self._async_http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            limits=limits,
            headers=self.default_headers or None,
        )
        self._sync_client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout_seconds,
            default_headers=self.default_headers or None,
            max_retries=0,
            http_client=self._sync_http_client,
        )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout_seconds,
            default_headers=self.default_headers or None,
            max_retries=0,
            http_client=self._async_http_client,
        )

    def _build_usage(self, body: Any, latency_ms: float) -> UsageStats:
        """Normalize token usage from OpenAI-compatible responses."""
        usage_data = getattr(body, "usage", None)
        prompt_tokens = getattr(usage_data, "prompt_tokens", None)
        completion_tokens = getattr(usage_data, "completion_tokens", None)
        total_tokens = getattr(usage_data, "total_tokens", None)
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        return UsageStats(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 4),
        )

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        """Return whether the raised exception is considered retryable."""
        return isinstance(exc, RETRYABLE_EXCEPTIONS)

    @staticmethod
    def _retry_after_delay(exc: Exception) -> float | None:
        """Extract a retry delay from a `Retry-After` header when present."""
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None

        retry_after = headers.get("retry-after")
        if retry_after is None:
            return None

        try:
            return min(MAX_RETRY_DELAY_SECONDS, max(0.0, float(retry_after)))
        except ValueError:
            pass

        try:
            retry_at = parsedate_to_datetime(retry_after)
        except (TypeError, ValueError, IndexError):
            return None

        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return min(
            MAX_RETRY_DELAY_SECONDS,
            max(0.0, (retry_at - datetime.now(UTC)).total_seconds()),
        )

    @classmethod
    def _compute_retry_delay(cls, exc: Exception, *, attempt: int) -> float:
        """Compute an exponential backoff delay with jitter or honor `Retry-After`."""
        retry_after_delay = cls._retry_after_delay(exc)
        if retry_after_delay is not None:
            return retry_after_delay

        base_delay = min(8.0, 0.5 * (2**attempt))
        return round(base_delay + random.uniform(0.0, base_delay / 2), 4)

    @staticmethod
    def _extract_first_message(response: Any) -> Any:
        """Return the first assistant message from an OpenAI-compatible response body."""

        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("Model response did not contain any choices.")

        message = getattr(choices[0], "message", None)
        if message is None:
            raise ValueError("Model response did not include a message in the first choice.")
        return message

    def _build_response(
        self,
        *,
        model: str,
        response: Any,
        message: Any,
        parsed_output: Any | None,
        started_at: float,
    ) -> LLMResponse:
        """Build the normalized client response wrapper from a provider payload."""

        latency_ms = (time.perf_counter() - started_at) * 1000.0
        return LLMResponse(
            text=getattr(message, "content", None),
            model=model,
            provider=self.provider,
            parsed_output=parsed_output,
            usage=self._build_usage(response, latency_ms),
        )

    @staticmethod
    def _chat_completion_kwargs(
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Build the shared keyword arguments for OpenAI-compatible chat calls."""

        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def _log_parse_fallback(self, *, model: str, exc: Exception) -> None:
        """Record a structured-output parse fallback for observability."""

        logger.info(
            "Falling back to unstructured completion for provider=%s model=%s "
            "after parse() failed with %s: %s",
            self.provider,
            model,
            type(exc).__name__,
            exc,
        )

    def _log_retry(self, *, model: str, attempt: int, delay: float, exc: Exception) -> None:
        """Record a retry with its backoff delay and root error type."""

        logger.warning(
            "Retrying provider=%s model=%s after failure %s/%s in %.3fs: %s: %s",
            self.provider,
            model,
            attempt + 1,
            self.max_retries + 1,
            delay,
            type(exc).__name__,
            exc,
        )

    def _retry_delay_for_exception(self, exc: Exception, *, attempt: int) -> float | None:
        """Return the retry delay for an exception or `None` when it should be raised."""

        if attempt >= self.max_retries or not self._should_retry(exc):
            return None
        return self._compute_retry_delay(exc, attempt=attempt)

    def _request_sync_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> tuple[Any, Any, Any | None, float]:
        """Execute one synchronous chat request, including structured-output fallback."""

        request_kwargs = self._chat_completion_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        start = time.perf_counter()
        if output_schema is None:
            response = self._sync_client.chat.completions.create(**request_kwargs)
            message = self._extract_first_message(response)
            return response, message, None, start

        try:
            response = self._sync_client.beta.chat.completions.parse(
                **request_kwargs,
                response_format=output_schema,
            )
        except PARSE_FALLBACK_EXCEPTIONS as exc:
            self._log_parse_fallback(model=model, exc=exc)
            start = time.perf_counter()
            response = self._sync_client.chat.completions.create(**request_kwargs)
            message = self._extract_first_message(response)
            return response, message, None, start

        message = self._extract_first_message(response)
        return response, message, getattr(message, "parsed", None), start

    async def _request_async_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> tuple[Any, Any, Any | None, float]:
        """Execute one asynchronous chat request, including structured-output fallback."""

        request_kwargs = self._chat_completion_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        start = time.perf_counter()
        if output_schema is None:
            response = await self._async_client.chat.completions.create(**request_kwargs)
            message = self._extract_first_message(response)
            return response, message, None, start

        try:
            response = await self._async_client.beta.chat.completions.parse(
                **request_kwargs,
                response_format=output_schema,
            )
        except PARSE_FALLBACK_EXCEPTIONS as exc:
            self._log_parse_fallback(model=model, exc=exc)
            start = time.perf_counter()
            response = await self._async_client.chat.completions.create(**request_kwargs)
            message = self._extract_first_message(response)
            return response, message, None, start

        message = self._extract_first_message(response)
        return response, message, getattr(message, "parsed", None), start

    def create_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Perform a sync chat completion with structured-output fallback and retries."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response, message, parsed_output, started_at = self._request_sync_completion(
                    model=model,
                    messages=messages,
                    output_schema=output_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return self._build_response(
                    model=model,
                    response=response,
                    message=message,
                    parsed_output=parsed_output,
                    started_at=started_at,
                )
            except Exception as exc:
                last_error = exc
                delay = self._retry_delay_for_exception(exc, attempt=attempt)
                if delay is None:
                    raise
                self._log_retry(model=model, attempt=attempt, delay=delay, exc=exc)
                time.sleep(delay)

        raise RuntimeError(
            "OpenAIClient exhausted retries without returning a response."
        ) from last_error

    async def acreate_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Perform an async chat completion with structured-output fallback and retries."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response, message, parsed_output, started_at = await self._request_async_completion(
                    model=model,
                    messages=messages,
                    output_schema=output_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return self._build_response(
                    model=model,
                    response=response,
                    message=message,
                    parsed_output=parsed_output,
                    started_at=started_at,
                )
            except Exception as exc:
                last_error = exc
                delay = self._retry_delay_for_exception(exc, attempt=attempt)
                if delay is None:
                    raise
                self._log_retry(model=model, attempt=attempt, delay=delay, exc=exc)
                await asyncio.sleep(delay)

        raise RuntimeError(
            "OpenAIClient exhausted retries without returning a response."
        ) from last_error

    def close(self) -> None:
        """Close the underlying synchronous OpenAI client and HTTP transport."""
        if self._sync_closed:
            return

        self._sync_closed = True
        errors: list[Exception] = []
        for closer in (self._sync_client.close, self._sync_http_client.close):
            try:
                closer()
            except Exception as exc:  # pragma: no cover - defensive close path
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise errors[0] from errors[1]

    async def aclose(self) -> None:
        """Close the underlying asynchronous OpenAI client and HTTP transport."""
        if self._async_closed:
            return

        self._async_closed = True
        errors: list[Exception] = []
        for closer in (self._async_client.close, self._async_http_client.aclose):
            try:
                await closer()
            except Exception as exc:  # pragma: no cover - defensive close path
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise errors[0] from errors[1]
