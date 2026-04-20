"""Shared Pydantic models and helpers for multi-step decomposition metrics."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """Atomic factual statement extracted from an answer or reference context."""

    claim_id: str
    text: str


class ContextChunk(BaseModel):
    """Retrieved or reference context chunk labeled with a stable synthetic id."""

    context_id: str
    text: str


class ClaimExtraction(BaseModel):
    """Structured output for prompts that extract a list of atomic claims."""

    claims: list[Claim] = Field(default_factory=list)


class ClaimSupportVerdict(BaseModel):
    """Verification result for whether a claim is grounded in retrieved contexts."""

    claim_id: str
    verdict: Literal["supported", "unsupported", "contradicted"]
    evidence_context_ids: list[str] = Field(default_factory=list)
    reason: str


class ClaimSupportAssessment(BaseModel):
    """Collection of claim support verdicts for faithfulness-style metrics."""

    verdicts: list[ClaimSupportVerdict] = Field(default_factory=list)


class ClaimCoverageVerdict(BaseModel):
    """Coverage result for whether retrieved contexts support a reference claim."""

    claim_id: str
    verdict: Literal["supported", "unsupported"]
    evidence_context_ids: list[str] = Field(default_factory=list)
    reason: str


class ClaimCoverageAssessment(BaseModel):
    """Collection of reference-claim coverage verdicts."""

    verdicts: list[ClaimCoverageVerdict] = Field(default_factory=list)


class ContextCoverageVerdict(BaseModel):
    """Usefulness verdict for a retrieved context against reference claims."""

    context_id: str
    supported_claim_ids: list[str] = Field(default_factory=list)
    reason: str


class ContextCoverageAssessment(BaseModel):
    """Collection of ranked retrieved-context usefulness verdicts."""

    contexts: list[ContextCoverageVerdict] = Field(default_factory=list)


class ContextRelevanceVerdict(BaseModel):
    """Relevance label for one retrieved context relative to a question."""

    context_id: str
    verdict: Literal["relevant", "partially_relevant", "irrelevant"]
    reason: str


class ContextRelevanceVerdictList(BaseModel):
    """Collection of retrieved-context relevance verdicts."""

    verdicts: list[ContextRelevanceVerdict] = Field(default_factory=list)


def build_context_chunks(contexts: list[str]) -> list[ContextChunk]:
    """Attach stable `ctx_N` ids to an ordered list of context strings."""

    return [
        ContextChunk(context_id=f"ctx_{index}", text=context)
        for index, context in enumerate(contexts, start=1)
    ]


def dedupe_ids(ids: list[str]) -> list[str]:
    """Preserve order while removing duplicate ids from a model response."""

    seen: set[str] = set()
    unique_ids: list[str] = []
    for item in ids:
        if item in seen:
            continue
        seen.add(item)
        unique_ids.append(item)
    return unique_ids


def ensure_complete_id_mapping(
    *,
    expected_ids: list[str],
    observed_ids: list[str],
    entity_name: str,
) -> None:
    """Validate that a model response covers each expected id exactly once."""

    duplicate_ids = sorted(item for item, count in Counter(observed_ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"Duplicate {entity_name} ids returned: {', '.join(duplicate_ids)}.")

    missing_ids = sorted(set(expected_ids) - set(observed_ids))
    if missing_ids:
        raise ValueError(f"Missing {entity_name} ids in response: {', '.join(missing_ids)}.")

    unknown_ids = sorted(set(observed_ids) - set(expected_ids))
    if unknown_ids:
        raise ValueError(f"Unknown {entity_name} ids in response: {', '.join(unknown_ids)}.")
