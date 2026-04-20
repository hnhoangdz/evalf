from __future__ import annotations

from pydantic import BaseModel

from evalf.prompt_builder import PydanticPrompt
from evalf.schemas import EvalCase

from .schema import ContextCoverageAssessment


class ContextCoveragePromptInput(BaseModel):
    question: str
    retrieval_contexts: list[str]
    reference_contexts: list[str]


class ContextCoveragePrompt(
    PydanticPrompt[ContextCoveragePromptInput, ContextCoverageAssessment]
):
    input_model = ContextCoveragePromptInput
    output_model = ContextCoverageAssessment
    system_prompt = (
        "You are an *AI Judge*. Your task is to check whether the **Retrieval Contexts** "
        "contain enough information from the **Reference Contexts** to answer the **User Input**.\n\n"
        "## How to evaluate:\n"
        "1. Read the **User Input** to understand the question.\n"
        "2. Read the **Reference Contexts** — this is the ground-truth source containing the correct information.\n"
        "3. Read the **Retrieval Contexts** — this is what the system actually retrieved.\n"
        "4. Check: does the Retrieval Contexts have the key information needed to answer the question?\n"
        "5. Give a **score** (0.0–1.0), a **verdict** (\"yes\" or \"no\"), and a short **reason**.\n\n"
        "## Key Principle:\n"
        "The Reference Contexts can be much LONGER or SHORTER than the Retrieval Contexts. That is fine. "
        "What matters is: does the Retrieval Contexts contain the **important facts** needed to answer the question? "
        "If yes, it passes — even if it is shorter or uses different words.\n\n"
        "## Rules:\n"
        "- Compare **meaning**, not exact words. Same meaning with different wording = OK.\n"
        "- Do NOT reduce score for missing info that is NOT needed to answer the question.\n"
        "- Do NOT reduce score just because the Retrieval Contexts is shorter or SHORTER than the Reference Contexts.\n\n"
        "Output Format: Must be in JSON format"
    )
    instruction = """
Check if the Retrieval Contexts have enough key information from the Reference Contexts to answer the User Input.

**HOW TO EVALUATE:**
1. Read the User Input — what is the question?
2. Look at the Reference Contexts — what key facts are needed to answer the question? (Ignore extra details that don't help answer it.)
3. Look at the Retrieval Contexts — are those key facts present? (Different wording is OK, as long as the meaning is the same.)
4. Give a score based on how much of the needed information is covered.

**SCORING:**
- **1.0** → All key information needed to answer the question is in the Retrieval Contexts (even if shorter or worded differently).
- **0.75** → Most key information is there. Only minor details are missing, but the question can still be fully answered.
- **0.5** → Some key information is there, but important parts are missing. The answer would be incomplete.
- **0.25** → Very little key information is there. The Retrieval Contexts are mostly not helpful.
- **0.0** → None of the key information is there, or the Retrieval Contexts are completely unrelated.

**VERDICT:**
- "yes" if score >= 0.75
- "no" if score < 0.75

**IMPORTANT RULES:**
1. Shorter Retrieval Context with the same core meaning = full credit. Length difference is NEVER a reason to reduce score.
2. Different words / synonyms / rephrased sentences with the same meaning = treat as equivalent.
3. Only reduce score for **actually missing key facts** needed to answer the question.
4. Extra details in Reference Contexts that are not relevant to the question — their absence should NOT reduce the score.
5. Extra correct information in Retrieval Contexts beyond Reference Contexts — this should NOT reduce the score.

**LANGUAGE RULE:**
If there is ANY Vietnamese language in the User Input or contexts, the reason MUST be written in Vietnamese. JSON keys stay in English.
"""
    examples = [
        (
            ContextCoveragePromptInput(
                question="When do FERPA rights transfer from parents to a student?",
                retrieval_contexts=[
                    (
                        "When a student turns 18 years old, or enters a postsecondary "
                        "institution at any age, FERPA rights transfer from the parents "
                        "to the student."
                    )
                ],
                reference_contexts=[
                    (
                        "FERPA is a federal privacy law. When a student turns 18 years old, "
                        "or enters a postsecondary institution at any age, FERPA rights "
                        "transfer from the parents to the student."
                    )
                ],
            ),
            ContextCoverageAssessment(
                score=1.0,
                verdict="yes",
                reason=(
                    "The retrieval context contains both key transfer conditions "
                    "(turning 18 and entering postsecondary institution) which fully "
                    "answer the question. The missing general FERPA description is "
                    "not needed to answer the question."
                ),
            ),
        ),
        (
            ContextCoveragePromptInput(
                question="What qualifies as a service animal under the ADA?",
                retrieval_contexts=[
                    "Under the ADA, a service animal is defined as a dog."
                ],
                reference_contexts=[
                    "Under the ADA, a service animal is a dog.",
                    (
                        "The dog must be individually trained to do work or perform tasks "
                        "for a person with a disability."
                    ),
                ],
            ),
            ContextCoverageAssessment(
                score=0.5,
                verdict="no",
                reason=(
                    "The retrieval context covers the 'dog' requirement but misses "
                    "the critical training requirement — the dog must be individually "
                    "trained to perform tasks for a person with a disability."
                ),
            ),
        ),
    ]


_CONTEXT_COVERAGE_PROMPT = ContextCoveragePrompt()


def build_prompt(case: EvalCase) -> tuple[str, str]:
    payload = ContextCoveragePromptInput(
        question=case.question or "",
        retrieval_contexts=case.retrieved_contexts or [],
        reference_contexts=case.reference_contexts or [],
    )
    return _CONTEXT_COVERAGE_PROMPT.render(payload)
