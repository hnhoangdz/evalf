from __future__ import annotations

from pydantic import BaseModel

from evalf.metrics.c4.prompt import build_prompt, build_reason_prompt
from evalf.prompt_builder import PydanticPrompt
from evalf.schemas import EvalCase


class DummyPromptInput(BaseModel):
    text: str


class DummyPromptOutput(BaseModel):
    score: float


class DummyPrompt(PydanticPrompt[DummyPromptInput, DummyPromptOutput]):
    input_model = DummyPromptInput
    output_model = DummyPromptOutput
    system_prompt = "System prompt"
    instruction = "Score the input."
    examples = [
        (
            DummyPromptInput(text="hello"),
            DummyPromptOutput(score=1.0),
        )
    ]


def test_prompt_examples_are_copied_per_instance() -> None:
    first = DummyPrompt()
    second = DummyPrompt()

    first.examples.append((DummyPromptInput(text="extra"), DummyPromptOutput(score=0.0)))

    assert len(first.examples) == 2
    assert len(second.examples) == 1


def test_prompt_render_returns_system_and_user_prompt() -> None:
    system_prompt, user_prompt = DummyPrompt().render(DummyPromptInput(text="world"))

    assert system_prompt == "System prompt"
    assert "Score the input." in user_prompt
    assert '"text": "world"' in user_prompt


def test_prompt_generate_examples_returns_empty_string_without_examples() -> None:
    assert DummyPrompt(examples=[])._generate_examples() == ""


def test_c4_prompts_do_not_require_vietnamese_reasoning() -> None:
    system_prompt, user_prompt = build_prompt(
        EvalCase(
            question="Does the VinFast VF 8 support OTA updates?",
            expected_output="Yes, it supports OTA updates.",
            actual_output="Yes, it supports OTA updates.",
        )
    )
    reason_system_prompt, reason_user_prompt = build_reason_prompt(
        score=0.75,
        breakdown={
            "alignment_integrity": {"score": 1.0, "reasoning": "Fully on topic."},
            "accuracy_consistency": {"score": 0.5, "reasoning": "Missing one detail."},
            "safety_sovereignty_tone": {"score": 0.75, "reasoning": "Safe but terse."},
            "completeness_coverage": {"score": 0.75, "reasoning": "Mostly complete."},
        },
    )

    combined_text = "\n".join([system_prompt, user_prompt, reason_system_prompt, reason_user_prompt])

    assert "Vietnamese" not in combined_text
    assert "same language" in combined_text
