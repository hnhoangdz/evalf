from __future__ import annotations

from pydantic import BaseModel

from evalf.metrics.context_recall.prompt import build_reference_claim_extraction_prompt
from evalf.metrics.decomposition import (
    Claim,
    ContextChunk,
    ContextCoverageAssessment,
    ContextCoverageVerdict,
    build_context_chunks,
)
from evalf.prompt_builder import PydanticPrompt
from evalf.schemas import EvalCase


class ContextPrecisionCoverageInput(BaseModel):
    question: str
    retrieved_contexts: list[ContextChunk]
    reference_claims: list[Claim]


class ContextPrecisionCoveragePrompt(
    PydanticPrompt[ContextPrecisionCoverageInput, ContextCoverageAssessment]
):
    input_model = ContextPrecisionCoverageInput
    output_model = ContextCoverageAssessment
    system_prompt = (
        "You are evalf, a strict and reliable judge for retrieval precision. "
        "For each retrieved context, identify which reference claims it actually supports."
    )
    instruction = """
Task:
For each retrieved context, list the reference claim ids that are supported by that context.

Rules:
1. Use only the provided question, retrieved contexts, and reference claims.
2. Credit semantically equivalent evidence even if the wording differs from the reference claim.
3. Only include a claim id when the retrieved context itself contains enough evidence for that claim.
4. If a retrieved context supports no reference claim, return an empty list for that context.
5. Return one item for every context id, in the same order as the input contexts.
6. Keep reasons short and context-specific.
"""
    examples = [
        (
            ContextPrecisionCoverageInput(
                question="When do FERPA rights transfer from parents to a student?",
                retrieved_contexts=build_context_chunks(
                    [
                        "FERPA rights transfer when the student turns 18.",
                        "The ADA defines a service animal as a trained dog.",
                        "FERPA rights also transfer when the student enters a postsecondary institution at any age.",
                    ]
                ),
                reference_claims=[
                    Claim(
                        claim_id="rc1",
                        text="FERPA rights transfer when a student turns 18 years old.",
                    ),
                    Claim(
                        claim_id="rc2",
                        text=(
                            "FERPA rights also transfer when a student enters a "
                            "postsecondary institution at any age."
                        ),
                    ),
                ],
            ),
            ContextCoverageAssessment(
                contexts=[
                    ContextCoverageVerdict(
                        context_id="ctx_1",
                        supported_claim_ids=["rc1"],
                        reason="The context covers the age-18 transfer condition.",
                    ),
                    ContextCoverageVerdict(
                        context_id="ctx_2",
                        supported_claim_ids=[],
                        reason="The context is unrelated to FERPA rights transfer.",
                    ),
                    ContextCoverageVerdict(
                        context_id="ctx_3",
                        supported_claim_ids=["rc2"],
                        reason="The context covers the postsecondary-institution condition.",
                    ),
                ]
            ),
        ),
        (
            ContextPrecisionCoverageInput(
                question="What qualifies as a service animal under the ADA?",
                retrieved_contexts=build_context_chunks(
                    [
                        "A service animal under the ADA is a dog that is individually trained to perform tasks.",
                        "A service animal under the ADA is a dog.",
                    ]
                ),
                reference_claims=[
                    Claim(
                        claim_id="rc1",
                        text="Under the ADA, a service animal is a dog.",
                    ),
                    Claim(
                        claim_id="rc2",
                        text=(
                            "Under the ADA, the dog must be individually trained to do work "
                            "or perform tasks for a person with a disability."
                        ),
                    ),
                ],
            ),
            ContextCoverageAssessment(
                contexts=[
                    ContextCoverageVerdict(
                        context_id="ctx_1",
                        supported_claim_ids=["rc1", "rc2"],
                        reason="The context covers both the dog requirement and the training requirement.",
                    ),
                    ContextCoverageVerdict(
                        context_id="ctx_2",
                        supported_claim_ids=["rc1"],
                        reason="The context repeats only the dog requirement.",
                    ),
                ]
            ),
        ),
    ]


_CONTEXT_PRECISION_COVERAGE_PROMPT = ContextPrecisionCoveragePrompt()


def build_context_precision_prompt(
    case: EvalCase, reference_claims: list[Claim]
) -> tuple[str, str]:
    payload = ContextPrecisionCoverageInput(
        question=case.question or "",
        retrieved_contexts=build_context_chunks(case.retrieved_contexts or []),
        reference_claims=reference_claims,
    )
    return _CONTEXT_PRECISION_COVERAGE_PROMPT.render(payload)


def build_prompt(case: EvalCase) -> tuple[str, str]:
    """Return the first-stage reference-claim extraction prompt for precision pipelines."""

    return build_reference_claim_extraction_prompt(case)
