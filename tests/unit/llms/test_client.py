import asyncio
import logging
from types import SimpleNamespace

import httpx
import pytest
from openai import RateLimitError
from pydantic import ValidationError

from evalf.llms.client import OpenAIClient
from tests.llms_helpers import (
    AsyncFakeCompletions,
    FakeAsyncOpenAI,
    FakeCompletions,
    FakeSyncOpenAI,
    ParsedPayload,
    make_openai_response,
)

pytestmark = pytest.mark.unit


def test_openai_client_returns_parsed_output_when_parse_succeeds(monkeypatch) -> None:
    sync_completions = FakeCompletions(
        parse_responses=[
            make_openai_response(content='{"score": 0.8}', parsed=ParsedPayload(score=0.8))
        ]
    )
    async_completions = AsyncFakeCompletions()
    monkeypatch.setattr(
        "evalf.llms.client.OpenAI",
        lambda **kwargs: FakeSyncOpenAI(sync_completions),
    )
    monkeypatch.setattr(
        "evalf.llms.client.AsyncOpenAI",
        lambda **kwargs: FakeAsyncOpenAI(async_completions),
    )

    client = OpenAIClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        provider="openai",
    )
    response = client.create_chat_completion(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Hi"}],
        output_schema=ParsedPayload,
        temperature=0.0,
        max_tokens=100,
    )

    assert response.parsed_output == ParsedPayload(score=0.8)
    assert response.usage.total_tokens == 15
    assert sync_completions.parse_calls[0]["response_format"] is ParsedPayload


def test_openai_client_falls_back_to_plain_completion_when_parse_fails(monkeypatch) -> None:
    try:
        ParsedPayload.model_validate({"score": "not-a-number"})
    except ValidationError as exc:
        parse_error = exc
    else:
        raise AssertionError("Expected ValidationError when building parse fallback test data.")

    sync_completions = FakeCompletions(
        parse_responses=[parse_error],
        create_responses=[make_openai_response(content='{"score": 0.6}', parsed=None)],
    )
    async_completions = AsyncFakeCompletions()
    monkeypatch.setattr(
        "evalf.llms.client.OpenAI",
        lambda **kwargs: FakeSyncOpenAI(sync_completions),
    )
    monkeypatch.setattr(
        "evalf.llms.client.AsyncOpenAI",
        lambda **kwargs: FakeAsyncOpenAI(async_completions),
    )

    client = OpenAIClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        provider="openai",
    )
    response = client.create_chat_completion(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Hi"}],
        output_schema=ParsedPayload,
        temperature=0.0,
        max_tokens=100,
    )

    assert response.parsed_output is None
    assert response.text == '{"score": 0.6}'
    assert len(sync_completions.create_calls) == 1


def test_openai_client_does_not_fallback_when_parsed_response_has_no_choices(monkeypatch) -> None:
    empty_response = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    sync_completions = FakeCompletions(
        parse_responses=[empty_response],
        create_responses=[make_openai_response(content='{"score": 0.6}', parsed=None)],
    )
    async_completions = AsyncFakeCompletions()
    monkeypatch.setattr("evalf.llms.client.OpenAI", lambda **kwargs: FakeSyncOpenAI(sync_completions))
    monkeypatch.setattr("evalf.llms.client.AsyncOpenAI", lambda **kwargs: FakeAsyncOpenAI(async_completions))

    client = OpenAIClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        provider="openai",
    )

    with pytest.raises(ValueError, match="did not contain any choices"):
        client.create_chat_completion(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hi"}],
            output_schema=ParsedPayload,
            temperature=0.0,
            max_tokens=100,
        )

    assert len(sync_completions.parse_calls) == 1
    assert len(sync_completions.create_calls) == 0


def test_openai_client_logs_parse_fallback_and_resets_latency(monkeypatch, caplog) -> None:
    try:
        ParsedPayload.model_validate({"score": "not-a-number"})
    except ValidationError as exc:
        parse_error = exc
    else:
        raise AssertionError("Expected ValidationError when building parse fallback test data.")

    sync_completions = FakeCompletions(
        parse_responses=[parse_error],
        create_responses=[make_openai_response(content='{"score": 0.6}', parsed=None)],
    )
    async_completions = AsyncFakeCompletions()
    perf_counter_values = iter([1.0, 10.0, 11.0])
    monkeypatch.setattr("evalf.llms.client.OpenAI", lambda **kwargs: FakeSyncOpenAI(sync_completions))
    monkeypatch.setattr("evalf.llms.client.AsyncOpenAI", lambda **kwargs: FakeAsyncOpenAI(async_completions))
    monkeypatch.setattr("evalf.llms.client.time.perf_counter", lambda: next(perf_counter_values))

    client = OpenAIClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        provider="openai",
    )

    with caplog.at_level(logging.INFO):
        response = client.create_chat_completion(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hi"}],
            output_schema=ParsedPayload,
            temperature=0.0,
            max_tokens=100,
        )

    assert "Falling back to unstructured completion" in caplog.text
    assert response.usage.latency_ms == 1000.0


def test_openai_client_does_not_fallback_or_retry_on_non_retryable_parse_errors(
    monkeypatch,
) -> None:
    parse_error = RuntimeError("parser crashed unexpectedly")
    sync_completions = FakeCompletions(
        parse_responses=[parse_error],
        create_responses=[make_openai_response(content='{"score": 0.6}', parsed=None)],
    )
    async_completions = AsyncFakeCompletions()
    monkeypatch.setattr(
        "evalf.llms.client.OpenAI",
        lambda **kwargs: FakeSyncOpenAI(sync_completions),
    )
    monkeypatch.setattr(
        "evalf.llms.client.AsyncOpenAI",
        lambda **kwargs: FakeAsyncOpenAI(async_completions),
    )

    client = OpenAIClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        provider="openai",
        max_retries=3,
    )

    with pytest.raises(RuntimeError, match="parser crashed unexpectedly"):
        client.create_chat_completion(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hi"}],
            output_schema=ParsedPayload,
            temperature=0.0,
            max_tokens=100,
        )

    assert len(sync_completions.parse_calls) == 1
    assert len(sync_completions.create_calls) == 0

    transient_error = httpx.ConnectError(
        "connection failed",
        request=httpx.Request("POST", "https://example.com/v1/chat/completions"),
    )
    retry_sync = FakeCompletions(
        create_responses=[transient_error, make_openai_response(content='{"score": 0.3}')],
    )
    monkeypatch.setattr(
        "evalf.llms.client.OpenAI",
        lambda **kwargs: FakeSyncOpenAI(retry_sync),
    )
    retry_client = OpenAIClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        provider="openai",
        max_retries=1,
    )
    monkeypatch.setattr("evalf.llms.client.time.sleep", lambda *args, **kwargs: None)

    retry_client.create_chat_completion(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Hi"}],
        output_schema=None,
        temperature=0.0,
        max_tokens=100,
    )

    assert len(retry_sync.create_calls) == 2


def test_openai_client_uses_retry_after_header_for_sync_retries(monkeypatch) -> None:
    retry_error = RateLimitError(
        "rate limited",
        response=httpx.Response(
            429,
            headers={"Retry-After": "2"},
            request=httpx.Request("POST", "https://example.com/v1/chat/completions"),
        ),
        body=None,
    )
    sync_completions = FakeCompletions(
        create_responses=[retry_error, make_openai_response(content='{"score": 0.5}')]
    )
    async_completions = AsyncFakeCompletions()
    delays: list[float] = []
    monkeypatch.setattr("evalf.llms.client.OpenAI", lambda **kwargs: FakeSyncOpenAI(sync_completions))
    monkeypatch.setattr("evalf.llms.client.AsyncOpenAI", lambda **kwargs: FakeAsyncOpenAI(async_completions))
    monkeypatch.setattr("evalf.llms.client.time.sleep", delays.append)

    client = OpenAIClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        provider="openai",
        max_retries=1,
    )

    client.create_chat_completion(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Hi"}],
        output_schema=None,
        temperature=0.0,
        max_tokens=100,
    )

    assert delays == [2.0]


def test_openai_client_caps_retry_after_header_for_sync_retries(monkeypatch) -> None:
    retry_error = RateLimitError(
        "rate limited",
        response=httpx.Response(
            429,
            headers={"Retry-After": "3600"},
            request=httpx.Request("POST", "https://example.com/v1/chat/completions"),
        ),
        body=None,
    )
    sync_completions = FakeCompletions(
        create_responses=[retry_error, make_openai_response(content='{"score": 0.5}')]
    )
    async_completions = AsyncFakeCompletions()
    delays: list[float] = []
    monkeypatch.setattr("evalf.llms.client.OpenAI", lambda **kwargs: FakeSyncOpenAI(sync_completions))
    monkeypatch.setattr("evalf.llms.client.AsyncOpenAI", lambda **kwargs: FakeAsyncOpenAI(async_completions))
    monkeypatch.setattr("evalf.llms.client.time.sleep", delays.append)

    client = OpenAIClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        provider="openai",
        max_retries=1,
    )

    client.create_chat_completion(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Hi"}],
        output_schema=None,
        temperature=0.0,
        max_tokens=100,
    )

    assert delays == [60.0]


def test_openai_client_uses_jittered_backoff_when_retry_after_missing(monkeypatch) -> None:
    transient_error = httpx.ConnectError(
        "connection failed",
        request=httpx.Request("POST", "https://example.com/v1/chat/completions"),
    )
    sync_completions = FakeCompletions(
        create_responses=[transient_error, make_openai_response(content='{"score": 0.4}')]
    )
    async_completions = AsyncFakeCompletions()
    delays: list[float] = []
    monkeypatch.setattr("evalf.llms.client.OpenAI", lambda **kwargs: FakeSyncOpenAI(sync_completions))
    monkeypatch.setattr("evalf.llms.client.AsyncOpenAI", lambda **kwargs: FakeAsyncOpenAI(async_completions))
    monkeypatch.setattr("evalf.llms.client.random.uniform", lambda start, end: 0.125)
    monkeypatch.setattr("evalf.llms.client.time.sleep", delays.append)

    client = OpenAIClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        provider="openai",
        max_retries=1,
    )

    client.create_chat_completion(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Hi"}],
        output_schema=None,
        temperature=0.0,
        max_tokens=100,
    )

    assert delays == [0.625]


def test_openai_client_logs_retry_warnings(monkeypatch, caplog) -> None:
    transient_error = httpx.ConnectError(
        "connection failed",
        request=httpx.Request("POST", "https://example.com/v1/chat/completions"),
    )
    sync_completions = FakeCompletions(
        create_responses=[transient_error, make_openai_response(content='{"score": 0.4}')]
    )
    async_completions = AsyncFakeCompletions()
    monkeypatch.setattr("evalf.llms.client.OpenAI", lambda **kwargs: FakeSyncOpenAI(sync_completions))
    monkeypatch.setattr("evalf.llms.client.AsyncOpenAI", lambda **kwargs: FakeAsyncOpenAI(async_completions))
    monkeypatch.setattr("evalf.llms.client.random.uniform", lambda start, end: 0.125)
    monkeypatch.setattr("evalf.llms.client.time.sleep", lambda delay: None)

    client = OpenAIClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        provider="openai",
        max_retries=1,
    )

    with caplog.at_level(logging.WARNING):
        client.create_chat_completion(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hi"}],
            output_schema=None,
            temperature=0.0,
            max_tokens=100,
        )

    assert "Retrying provider=openai model=gpt-4.1-mini" in caplog.text


def test_openai_client_async_retries_then_succeeds(monkeypatch) -> None:
    transient_error = httpx.ConnectError(
        "connection failed",
        request=httpx.Request("POST", "https://example.com/v1/chat/completions"),
    )
    sync_completions = FakeCompletions()
    async_completions = AsyncFakeCompletions(
        create_responses=[transient_error, make_openai_response(content='{"score": 0.7}')]
    )
    monkeypatch.setattr(
        "evalf.llms.client.OpenAI",
        lambda **kwargs: FakeSyncOpenAI(sync_completions),
    )
    monkeypatch.setattr(
        "evalf.llms.client.AsyncOpenAI",
        lambda **kwargs: FakeAsyncOpenAI(async_completions),
    )

    async def fast_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr("evalf.llms.client.asyncio.sleep", fast_sleep)

    client = OpenAIClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        provider="openai",
        max_retries=1,
    )

    async def run_test():
        return await client.acreate_chat_completion(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hi"}],
            output_schema=None,
            temperature=0.0,
            max_tokens=100,
        )

    response = asyncio.run(run_test())

    assert response.text == '{"score": 0.7}'
    assert len(async_completions.create_calls) == 2


def test_openai_client_async_uses_retry_after_header(monkeypatch) -> None:
    retry_error = RateLimitError(
        "rate limited",
        response=httpx.Response(
            429,
            headers={"Retry-After": "3"},
            request=httpx.Request("POST", "https://example.com/v1/chat/completions"),
        ),
        body=None,
    )
    sync_completions = FakeCompletions()
    async_completions = AsyncFakeCompletions(
        create_responses=[retry_error, make_openai_response(content='{"score": 0.9}')]
    )
    delays: list[float] = []

    async def capture_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("evalf.llms.client.OpenAI", lambda **kwargs: FakeSyncOpenAI(sync_completions))
    monkeypatch.setattr("evalf.llms.client.AsyncOpenAI", lambda **kwargs: FakeAsyncOpenAI(async_completions))
    monkeypatch.setattr("evalf.llms.client.asyncio.sleep", capture_sleep)

    client = OpenAIClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        provider="openai",
        max_retries=1,
    )

    async def run_test():
        return await client.acreate_chat_completion(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hi"}],
            output_schema=None,
            temperature=0.0,
            max_tokens=100,
        )

    response = asyncio.run(run_test())

    assert response.text == '{"score": 0.9}'
    assert delays == [3.0]


def test_openai_client_close_methods_close_underlying_clients(monkeypatch) -> None:
    sync_client = FakeSyncOpenAI(FakeCompletions())
    async_client = FakeAsyncOpenAI(AsyncFakeCompletions())
    monkeypatch.setattr("evalf.llms.client.OpenAI", lambda **kwargs: sync_client)
    monkeypatch.setattr("evalf.llms.client.AsyncOpenAI", lambda **kwargs: async_client)

    client = OpenAIClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        provider="openai",
    )
    client.close()
    client.close()
    asyncio.run(client.aclose())
    asyncio.run(client.aclose())

    assert sync_client.closed is True
    assert async_client.closed is True
