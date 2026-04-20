from __future__ import annotations

from pydantic import BaseModel

from evalf.metrics.decomposition import (
    ContextChunk,
    ContextRelevanceVerdict,
    ContextRelevanceVerdictList,
    build_context_chunks,
)
from evalf.prompt_builder import PydanticPrompt
from evalf.schemas import EvalCase


class ContextRelevancePromptInput(BaseModel):
    question: str
    retrieved_contexts: list[ContextChunk]


class ContextRelevancePrompt(
    PydanticPrompt[ContextRelevancePromptInput, ContextRelevanceVerdictList]
):
    input_model = ContextRelevancePromptInput
    output_model = ContextRelevanceVerdictList
    system_prompt = (
        "You are evalf, a strict and reliable judge for retrieval relevance. "
        "Judge each retrieved context independently against the question."
    )
    instruction = """
Task:
For each retrieved context, decide how useful it is for answering the question.

Rules:
1. Use only the provided question and retrieved contexts.
2. relevant: the context directly helps answer the question.
3. partially_relevant: the context is loosely related or provides limited supporting background.
4. irrelevant: the context does not help answer the question.
5. Return one verdict for every context id, in the same order as the input.
6. Keep reasons short and context-specific.
"""
    examples = [
        (
            ContextRelevancePromptInput(
                question="What qualifies as a service animal under the ADA?",
                retrieved_contexts=build_context_chunks(
                    [
                        (
                            "Under the ADA, a service animal is a dog that has been "
                            "individually trained to do work or perform tasks for a "
                            "person with a disability."
                        ),
                        (
                            "Common side effects of injectable flu vaccines include soreness "
                            "and fever."
                        ),
                    ]
                ),
            ),
            ContextRelevanceVerdictList(
                verdicts=[
                    ContextRelevanceVerdict(
                        context_id="ctx_1",
                        verdict="relevant",
                        reason="The context states the ADA service-animal definition.",
                    ),
                    ContextRelevanceVerdict(
                        context_id="ctx_2",
                        verdict="irrelevant",
                        reason="The context is about flu vaccine side effects, not service animals.",
                    ),
                ]
            ),
        ),
        (
            ContextRelevancePromptInput(
                question="When do FERPA rights transfer from parents to a student?",
                retrieved_contexts=build_context_chunks(
                    [
                        ("FERPA is a federal privacy law that protects education records."),
                        (
                            "When a student turns 18 years old, or enters a postsecondary "
                            "institution at any age, FERPA rights transfer from the parents "
                            "to the student."
                        ),
                    ]
                ),
            ),
            ContextRelevanceVerdictList(
                verdicts=[
                    ContextRelevanceVerdict(
                        context_id="ctx_1",
                        verdict="partially_relevant",
                        reason="The context is related to FERPA but does not answer the timing question.",
                    ),
                    ContextRelevanceVerdict(
                        context_id="ctx_2",
                        verdict="relevant",
                        reason="The context directly answers when rights transfer.",
                    ),
                ]
            ),
        ),
    ]


_CONTEXT_RELEVANCE_PROMPT = ContextRelevancePrompt()


def build_prompt(case: EvalCase) -> tuple[str, str]:
    payload = ContextRelevancePromptInput(
        question=case.question or "",
        retrieved_contexts=build_context_chunks(case.retrieved_contexts or []),
    )
    return _CONTEXT_RELEVANCE_PROMPT.render(payload)
