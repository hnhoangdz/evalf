from __future__ import annotations

from evalf.llms.base import BaseLLMModel
from evalf.schemas import LLMResponse, UsageStats


class SequenceLLM(BaseLLMModel):
    def __init__(
        self,
        outputs,
        *,
        provider: str = "test",
        model: str = "fake-model",
        base_url: str = "https://example.com/v1",
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        self._outputs = list(outputs)
        self.closed = False
        self.aclosed = False

    def _next_response(self) -> LLMResponse:
        if not self._outputs:
            raise AssertionError("No fake outputs left for SequenceLLM.")

        output = self._outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        if isinstance(output, LLMResponse):
            return output

        parsed_output = output
        usage = UsageStats.empty()
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], UsageStats):
            parsed_output, usage = output

        return LLMResponse(
            text=None,
            model=self.model,
            provider=self.provider,
            parsed_output=parsed_output,
            usage=usage,
        )

    def generate(self, **kwargs) -> LLMResponse:
        return self._next_response()

    async def a_generate(self, **kwargs) -> LLMResponse:
        return self._next_response()

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.aclosed = True
