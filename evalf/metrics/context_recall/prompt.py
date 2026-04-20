from __future__ import annotations

from pydantic import BaseModel

from evalf.metrics.decomposition import (
    Claim,
    ClaimCoverageAssessment,
    ClaimCoverageVerdict,
    ClaimExtraction,
    ContextChunk,
    build_context_chunks,
)
from evalf.prompt_builder import PydanticPrompt
from evalf.schemas import EvalCase


class ReferenceClaimExtractionInput(BaseModel):
    question: str
    reference_contexts: list[str]


class ContextRecallCoverageInput(BaseModel):
    question: str
    retrieved_contexts: list[ContextChunk]
    reference_claims: list[Claim]


class ReferenceClaimExtractionPrompt(
    PydanticPrompt[ReferenceClaimExtractionInput, ClaimExtraction]
):
    input_model = ReferenceClaimExtractionInput
    output_model = ClaimExtraction
    system_prompt = (
        "You are evalf, a strict and reliable information extraction judge. "
        "Extract atomic reference claims that capture the material evidence needed to answer the question."
    )
    instruction = """
Task:
Extract material reference claims from the reference contexts.

Rules:
1. Use only the provided reference contexts.
2. Extract only claims that are relevant to answering the question.
3. Keep each claim atomic, factual, and standalone.
4. Preserve qualifiers, dates, numbers, conditions, and negations.
5. Merge duplicate or semantically equivalent evidence into one claim.
6. If the reference contexts contain no relevant factual evidence, return {"claims": []}.
7. Use claim ids exactly as rc1, rc2, rc3 in a sensible logical order.
"""
    examples = [
        (
            ReferenceClaimExtractionInput(
                question="When do FERPA rights transfer from parents to a student?",
                reference_contexts=[
                    (
                        "When a student turns 18 years old, or enters a postsecondary "
                        "institution at any age, FERPA rights transfer from the parents "
                        "to the student."
                    )
                ],
            ),
            ClaimExtraction(
                claims=[
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
                ]
            ),
        ),
        (
            ReferenceClaimExtractionInput(
                question="What qualifies as a service animal under the ADA?",
                reference_contexts=[
                    "Under the ADA, a service animal is a dog.",
                    (
                        "The dog must be individually trained to do work or perform tasks "
                        "for a person with a disability."
                    ),
                ],
            ),
            ClaimExtraction(
                claims=[
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
                ]
            ),
        ),
    ]


class ContextRecallCoveragePrompt(
    PydanticPrompt[ContextRecallCoverageInput, ClaimCoverageAssessment]
):
    input_model = ContextRecallCoverageInput
    output_model = ClaimCoverageAssessment
    system_prompt = (
        "You are evalf, a strict and reliable judge for retrieval recall. "
        "Check whether the retrieved contexts cover each reference claim."
    )
    instruction = """
Task:
For each reference claim, decide whether the retrieved contexts cover it.

Rules:
1. Use only the provided retrieved contexts and reference claims.
2. supported: one or more retrieved contexts contain the evidence for the claim.
3. unsupported: the retrieved contexts do not cover the claim.
4. Credit semantically equivalent evidence even if wording differs from the reference claim.
5. Return one verdict for every reference claim id, in the same order as the input.
6. evidence_context_ids must reference only context ids from the input.
7. Keep reasons short and claim-specific.
"""
    examples = [
        (
            ContextRecallCoverageInput(
                question="What qualifies as a service animal under the ADA?",
                retrieved_contexts=build_context_chunks(
                    [
                        "Under the ADA, a service animal is a dog.",
                        "A service animal must be individually trained to perform tasks.",
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
            ClaimCoverageAssessment(
                verdicts=[
                    ClaimCoverageVerdict(
                        claim_id="rc1",
                        verdict="supported",
                        evidence_context_ids=["ctx_1"],
                        reason="The first retrieved context states that a service animal is a dog.",
                    ),
                    ClaimCoverageVerdict(
                        claim_id="rc2",
                        verdict="supported",
                        evidence_context_ids=["ctx_2"],
                        reason="The second retrieved context covers the training requirement.",
                    ),
                ]
            ),
        ),
        (
            ContextRecallCoverageInput(
                question="What qualifies as a service animal under the ADA?",
                retrieved_contexts=build_context_chunks(
                    ["Under the ADA, a service animal is a dog."]
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
            ClaimCoverageAssessment(
                verdicts=[
                    ClaimCoverageVerdict(
                        claim_id="rc1",
                        verdict="supported",
                        evidence_context_ids=["ctx_1"],
                        reason="The retrieved context covers the dog requirement.",
                    ),
                    ClaimCoverageVerdict(
                        claim_id="rc2",
                        verdict="unsupported",
                        evidence_context_ids=[],
                        reason="The retrieved context does not cover the training requirement.",
                    ),
                ]
            ),
        ),
    ]


_REFERENCE_CLAIM_EXTRACTION_PROMPT = ReferenceClaimExtractionPrompt()
_CONTEXT_RECALL_COVERAGE_PROMPT = ContextRecallCoveragePrompt()


def build_reference_claim_extraction_prompt(case: EvalCase) -> tuple[str, str]:
    payload = ReferenceClaimExtractionInput(
        question=case.question or "",
        reference_contexts=case.reference_contexts or [],
    )
    return _REFERENCE_CLAIM_EXTRACTION_PROMPT.render(payload)


def build_context_recall_prompt(case: EvalCase, reference_claims: list[Claim]) -> tuple[str, str]:
    payload = ContextRecallCoverageInput(
        question=case.question or "",
        retrieved_contexts=build_context_chunks(case.retrieved_contexts or []),
        reference_claims=reference_claims,
    )
    return _CONTEXT_RECALL_COVERAGE_PROMPT.render(payload)


def build_prompt(case: EvalCase) -> tuple[str, str]:
    """Return the first-stage reference-claim extraction prompt for recall pipelines."""

    return build_reference_claim_extraction_prompt(case)
