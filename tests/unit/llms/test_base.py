import asyncio

import pytest

from evalf.llms.base import BaseLLMModel, normalize_base_url
from evalf.schemas import LLMResponse, UsageStats

pytestmark = pytest.mark.unit


class MinimalLLM(BaseLLMModel):
    def generate(self, **kwargs):
        return LLMResponse(
            text=None,
            model=self.model,
            provider=self.provider,
            parsed_output=None,
            usage=UsageStats.empty(),
        )

    async def a_generate(self, **kwargs):
        return self.generate(**kwargs)


def test_base_llm_close_methods_are_noops() -> None:
    llm = MinimalLLM(
        provider="test",
        model="model",
        base_url="https://example.com",
        api_key=None,
    )

    assert llm.close() is None
    assert asyncio.run(llm.aclose()) is None


def test_normalize_base_url_strips_trailing_slashes() -> None:
    assert normalize_base_url("https://example.com/v1///") == "https://example.com/v1"
