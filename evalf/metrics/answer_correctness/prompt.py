from __future__ import annotations

from pydantic import BaseModel

from evalf.prompt_builder import PydanticPrompt
from evalf.schemas import EvalCase

from .schema import AnswerCorrectnessAssessment


class AnswerCorrectnessPromptInput(BaseModel):
    question: str
    expected_output: str
    actual_output: str


class AnswerCorrectnessPrompt(
    PydanticPrompt[AnswerCorrectnessPromptInput, AnswerCorrectnessAssessment]
):
    input_model = AnswerCorrectnessPromptInput
    output_model = AnswerCorrectnessAssessment
    system_prompt = (
        "You are evalf, a strict and reliable judge for answer correctness. "
        "Compare the actual output against the expected output and score factual equivalence. "
        "Use only the provided question, expected output, and actual output."
    )
    instruction = """
Task:
Evaluate how correct the actual output is compared with the expected output.

Rules:
1. Focus on whether the actual output matches the important facts in the expected output.
2. Penalize contradictions, wrong numbers, wrong entities, and missing key facts.
3. Do not penalize harmless paraphrasing when the meaning is materially equivalent.
4. Ignore tone and style unless they change meaning.
5. Never use outside knowledge to upgrade or downgrade the score beyond what the provided expected output supports.

Scoring guidance:
- 1.0: Materially equivalent to the expected output.
- 0.75: Mostly correct, with only a minor omission or imprecision.
- 0.5: Partially correct but missing or weakening important facts.
- 0.25: Mostly incorrect, with only limited overlap with the expected output.
- 0.0: Clearly incorrect or contradictory.

Reason requirements:
- Keep the reason concise and specific.
- Mention the key matched or mismatched fact.
- Use the same language as the input content when possible.
"""
    examples = [
        (
            AnswerCorrectnessPromptInput(
                question="Under FERPA, when do rights transfer from parents to a student?",
                expected_output="Under FERPA, rights transfer when a student turns 18 years old or enters a postsecondary institution at any age.",
                actual_output="Under FERPA, rights transfer only when a student turns 21 years old.",
            ),
            AnswerCorrectnessAssessment(
                score=0.0,
                reason="The actual output contradicts the expected output because FERPA rights do not wait until age 21 and can also transfer upon entering a postsecondary institution at any age.",
            ),
        ),
        (
            AnswerCorrectnessPromptInput(
                question="What qualifies as a service animal under the ADA?",
                expected_output="Under the ADA, a service animal is a dog that has been individually trained to do work or perform tasks for a person with a disability.",
                actual_output="Under the ADA, a service animal is a dog individually trained to perform tasks for a person with a disability.",
            ),
            AnswerCorrectnessAssessment(
                score=1.0,
                reason="The actual output is materially equivalent to the expected output and preserves the key legal definition.",
            ),
        ),
        (
            AnswerCorrectnessPromptInput(
                question="Under FERPA, when do rights transfer from parents to a student?",
                expected_output="Under FERPA, rights transfer when a student turns 18 years old or enters a postsecondary institution at any age.",
                actual_output="Under FERPA, rights usually transfer at age 18.",
            ),
            AnswerCorrectnessAssessment(
                score=0.5,
                reason="The actual output captures the age-18 condition but misses the other key condition that rights also transfer when the student enters a postsecondary institution at any age.",
            ),
        ),
    ]


_ANSWER_CORRECTNESS_PROMPT = AnswerCorrectnessPrompt()


def build_prompt(case: EvalCase) -> tuple[str, str]:
    payload = AnswerCorrectnessPromptInput(
        question=case.question or "",
        expected_output=case.expected_output or "",
        actual_output=case.actual_output or "",
    )
    return _ANSWER_CORRECTNESS_PROMPT.render(payload)
