from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from evalf.schemas import LLMResponse, UsageStats


class ParsedPayload(BaseModel):
    score: float


def make_openai_response(
    *,
    content: str = "{}",
    parsed=None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int | None = None,
):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, parsed=parsed))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


class FakeCompletions:
    def __init__(self, *, parse_responses=None, create_responses=None):
        self.parse_responses = list(parse_responses or [])
        self.create_responses = list(create_responses or [])
        self.parse_calls = []
        self.create_calls = []

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        response = self.parse_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        response = self.create_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class AsyncFakeCompletions(FakeCompletions):
    async def parse(self, **kwargs):
        return super().parse(**kwargs)

    async def create(self, **kwargs):
        return super().create(**kwargs)


class FakeSyncOpenAI:
    def __init__(self, completions):
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        self.chat = SimpleNamespace(completions=completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeAsyncOpenAI:
    def __init__(self, completions):
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        self.chat = SimpleNamespace(completions=completions)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeProviderClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.aclosed = False

    def create_chat_completion(self, **kwargs) -> LLMResponse:
        return LLMResponse(
            text='{"score": 0.9}',
            model=kwargs["model"],
            provider="openai",
            parsed_output=None,
            usage=UsageStats(input_tokens=1000, output_tokens=500, total_tokens=1500),
        )

    async def acreate_chat_completion(self, **kwargs) -> LLMResponse:
        return self.create_chat_completion(**kwargs)

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.aclosed = True
