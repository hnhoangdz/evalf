import pytest

from evalf.llms.providers import OpenAILLMModel
from evalf.metrics.answer_correctness import AnswerCorrectnessMetric
from evalf.metrics.answer_correctness.schema import AnswerCorrectnessAssessment
from evalf.schemas import EvalCase, LLMResponse, UsageStats

pytestmark = pytest.mark.integration


class _FakeProviderClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def create_chat_completion(self, **kwargs) -> LLMResponse:
        return LLMResponse(
            text=None,
            model=kwargs["model"],
            provider="openai",
            parsed_output=AnswerCorrectnessAssessment(
                score=0.75,
                reason="Câu trả lời đúng phần lớn nội dung kỳ vọng.",
            ),
            usage=UsageStats(input_tokens=1000, output_tokens=500, total_tokens=1500),
        )

    async def acreate_chat_completion(self, **kwargs) -> LLMResponse:
        return self.create_chat_completion(**kwargs)

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def test_provider_model_integrates_with_metric_and_cost_estimation(monkeypatch) -> None:
    monkeypatch.setattr("evalf.llms.providers.OpenAIClient", _FakeProviderClient)

    llm = OpenAILLMModel(
        model="gpt-4.1-mini",
        base_url="https://example.com/v1",
        api_key="test-key",
    )
    metric = AnswerCorrectnessMetric(threshold=0.7)

    result = metric.measure(
        EvalCase(
            question="Under FERPA, when do rights transfer from parents to a student?",
            expected_output="Rights transfer when a student turns 18 or enters a postsecondary institution at any age.",
            actual_output="Rights transfer when a student turns 18 or enters a postsecondary institution.",
        ),
        llm,
    )

    assert result.status == "passed"
    assert result.score == 0.75
    assert result.usage.cost_usd == 0.0012
