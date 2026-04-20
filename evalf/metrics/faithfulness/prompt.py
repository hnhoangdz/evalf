from __future__ import annotations

from pydantic import BaseModel

from evalf.metrics.decomposition import (
    Claim,
    ClaimExtraction,
    ClaimSupportAssessment,
    ClaimSupportVerdict,
    ContextChunk,
    build_context_chunks,
)
from evalf.prompt_builder import PydanticPrompt
from evalf.schemas import EvalCase


class FaithfulnessClaimExtractionInput(BaseModel):
    question: str
    actual_output: str


class FaithfulnessClaimVerificationInput(BaseModel):
    question: str
    retrieved_contexts: list[ContextChunk]
    claims: list[Claim]


class FaithfulnessClaimExtractionPrompt(
    PydanticPrompt[FaithfulnessClaimExtractionInput, ClaimExtraction]
):
    input_model = FaithfulnessClaimExtractionInput
    output_model = ClaimExtraction
    system_prompt = (
        "You are evalf, a strict and reliable information extraction judge. "
        "Extract only atomic factual claims that are actually asserted in the answer."
    )
    instruction = """
Task:
Break the actual output into atomic material claims that should later be checked against retrieved contexts.

Rules:
1. Extract only claims that are explicitly asserted in the actual output.
2. Keep each claim atomic, standalone, and factual.
3. Preserve quantities, dates, negations, qualifiers, and conditions.
4. Do not add outside knowledge, implications, or assumptions.
5. Merge repeated paraphrases into one claim.
6. Ignore stylistic filler, greetings, and purely procedural text.
7. If the actual output makes no material factual claim, return {"claims": []}.
8. Use claim ids exactly as c1, c2, c3 in answer order.
"""
    examples = [
        (
            FaithfulnessClaimExtractionInput(
                question="Under FERPA, when do rights transfer from parents to a student?",
                actual_output=(
                    "Under FERPA, rights transfer when a student turns 18 or enters a "
                    "postsecondary institution at any age."
                ),
            ),
            ClaimExtraction(
                claims=[
                    Claim(
                        claim_id="c1",
                        text="Under FERPA, rights transfer when a student turns 18.",
                    ),
                    Claim(
                        claim_id="c2",
                        text=(
                            "Under FERPA, rights also transfer when a student enters a "
                            "postsecondary institution at any age."
                        ),
                    ),
                ]
            ),
        ),
        (
            FaithfulnessClaimExtractionInput(
                question="What did the answer say?",
                actual_output="I do not know based on the provided information.",
            ),
            ClaimExtraction(claims=[]),
        ),
    ]


class FaithfulnessClaimVerificationPrompt(
    PydanticPrompt[FaithfulnessClaimVerificationInput, ClaimSupportAssessment]
):
    input_model = FaithfulnessClaimVerificationInput
    output_model = ClaimSupportAssessment
    system_prompt = (
        "You are evalf, a strict and reliable judge for factual grounding. "
        "Check each claim only against the provided retrieved contexts."
    )
    instruction = """
Task:
For each claim, decide whether it is supported by the retrieved contexts.

Rules:
1. Use only the provided retrieved contexts. Never use outside knowledge.
2. supported: the claim is directly supported or faithfully paraphrased by the contexts.
3. unsupported: the contexts do not establish the claim, or a required detail is missing.
4. contradicted: the contexts directly conflict with the claim.
5. Return one verdict for every claim id, in the same order as the input claims.
6. evidence_context_ids must reference only context ids from the input.
7. Keep reasons short and claim-specific.
"""
    examples = [
        (
            FaithfulnessClaimVerificationInput(
                question="What are common side effects of injectable flu vaccines?",
                retrieved_contexts=build_context_chunks(
                    [
                        (
                            "Common side effects of injectable flu vaccines include soreness, "
                            "redness, and swelling at the injection site, fever, muscle aches, "
                            "headache, and fatigue."
                        )
                    ]
                ),
                claims=[
                    Claim(
                        claim_id="c1",
                        text="Injectable flu vaccines can cause soreness at the injection site.",
                    ),
                    Claim(
                        claim_id="c2",
                        text="Injectable flu vaccines commonly cause seizures.",
                    ),
                ],
            ),
            ClaimSupportAssessment(
                verdicts=[
                    ClaimSupportVerdict(
                        claim_id="c1",
                        verdict="supported",
                        evidence_context_ids=["ctx_1"],
                        reason="The context explicitly lists soreness at the injection site.",
                    ),
                    ClaimSupportVerdict(
                        claim_id="c2",
                        verdict="unsupported",
                        evidence_context_ids=[],
                        reason="The context does not list seizures as a common side effect.",
                    ),
                ]
            ),
        ),
        (
            FaithfulnessClaimVerificationInput(
                question="Under FERPA, when do rights transfer from parents to a student?",
                retrieved_contexts=build_context_chunks(
                    [
                        (
                            "When a student turns 18 years old, or enters a postsecondary "
                            "institution at any age, FERPA rights transfer from the parents "
                            "to the student."
                        )
                    ]
                ),
                claims=[
                    Claim(
                        claim_id="c1",
                        text="Under FERPA, rights transfer only when a student turns 21.",
                    )
                ],
            ),
            ClaimSupportAssessment(
                verdicts=[
                    ClaimSupportVerdict(
                        claim_id="c1",
                        verdict="contradicted",
                        evidence_context_ids=["ctx_1"],
                        reason="The context says rights transfer at age 18, not 21.",
                    )
                ]
            ),
        ),
    ]


_FAITHFULNESS_CLAIM_EXTRACTION_PROMPT = FaithfulnessClaimExtractionPrompt()
_FAITHFULNESS_CLAIM_VERIFICATION_PROMPT = FaithfulnessClaimVerificationPrompt()


def build_claim_extraction_prompt(case: EvalCase) -> tuple[str, str]:
    payload = FaithfulnessClaimExtractionInput(
        question=case.question or "",
        actual_output=case.actual_output or "",
    )
    return _FAITHFULNESS_CLAIM_EXTRACTION_PROMPT.render(payload)


def build_claim_verification_prompt(case: EvalCase, claims: list[Claim]) -> tuple[str, str]:
    payload = FaithfulnessClaimVerificationInput(
        question=case.question or "",
        retrieved_contexts=build_context_chunks(case.retrieved_contexts or []),
        claims=claims,
    )
    return _FAITHFULNESS_CLAIM_VERIFICATION_PROMPT.render(payload)


def build_prompt(case: EvalCase) -> tuple[str, str]:
    """Return the first-stage claim-extraction prompt for faithfulness pipelines."""

    return build_claim_extraction_prompt(case)
