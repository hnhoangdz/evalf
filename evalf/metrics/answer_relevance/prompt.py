from __future__ import annotations

from pydantic import BaseModel

from evalf.prompt_builder import PydanticPrompt
from evalf.schemas import EvalCase

from .schema import AnswerRelevanceAssessment


class AnswerRelevancePromptInput(BaseModel):
    question: str
    actual_output: str


class AnswerRelevancePrompt(PydanticPrompt[AnswerRelevancePromptInput, AnswerRelevanceAssessment]):
    input_model = AnswerRelevancePromptInput
    output_model = AnswerRelevanceAssessment
    system_prompt = (
        "You are evalf, a strict and reliable judge for answer relevance. "
        "Score how directly the actual output addresses the user's question. "
        "Use only the provided question and answer."
    )
    instruction = """
Task:
Evaluate how directly and usefully the actual output answers the question.

Rules:
1. Reward answers that directly address the user's request.
2. Penalize irrelevant digressions, generic filler, or background content that the user did not ask for.
3. A brief but direct answer can still score high.
4. A long answer can score low if much of it is off-topic.
5. Do not use outside knowledge to fill in missing intent or missing facts.

Scoring guidance:
- 1.0: Fully focused and directly answers the question.
- 0.75: Mostly relevant, with a small amount of unnecessary detail.
- 0.5: Mix of relevant and irrelevant content, or only partially answers the question.
- 0.25: Mostly off-topic or evasive.
- 0.0: Does not answer the question.

Reason requirements:
- Keep the reason concise and specific.
- Mention what part is relevant or irrelevant.
- Use the same language as the input content when possible.
"""
    examples = [
        (
            AnswerRelevancePromptInput(
                question="How long do you generally have to file a charge of employment discrimination with EEOC?",
                actual_output=(
                    "In general, you need to file within 180 calendar days from the day the discrimination took place. "
                    "The deadline can extend to 300 calendar days if a state or local agency enforces a law prohibiting employment discrimination on the same basis. "
                    "EEOC also enforces federal laws against workplace discrimination."
                ),
            ),
            AnswerRelevanceAssessment(
                score=0.75,
                reason="Most of the answer directly addresses the filing deadline, but the final sentence is general background about EEOC rather than part of the requested time limit.",
            ),
        ),
        (
            AnswerRelevancePromptInput(
                question="What qualifies as a service animal under the ADA?",
                actual_output="Under the ADA, a service animal is a dog that has been individually trained to do work or perform tasks for a person with a disability.",
            ),
            AnswerRelevanceAssessment(
                score=1.0,
                reason="The answer is fully focused on the legal definition the user asked for.",
            ),
        ),
        (
            AnswerRelevancePromptInput(
                question="When do FERPA rights transfer from parents to a student?",
                actual_output=(
                    "FERPA is a federal privacy law. Rights transfer when the student turns 18."
                ),
            ),
            AnswerRelevanceAssessment(
                score=0.5,
                reason="The answer partly addresses the transfer timing, but it also adds unnecessary background and omits the postsecondary-institution condition.",
            ),
        ),
    ]


_ANSWER_RELEVANCE_PROMPT = AnswerRelevancePrompt()


def build_prompt(case: EvalCase) -> tuple[str, str]:
    payload = AnswerRelevancePromptInput(
        question=case.question or "",
        actual_output=case.actual_output or "",
    )
    return _ANSWER_RELEVANCE_PROMPT.render(payload)
