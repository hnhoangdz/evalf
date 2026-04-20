from __future__ import annotations

from pydantic import BaseModel, Field


class CriterionAssessment(BaseModel):
    """One rubric dimension within the composite C4 evaluation."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class C4Assessment(BaseModel):
    """Structured response for the primary four-criterion C4 judge call."""

    alignment_integrity: CriterionAssessment
    accuracy_consistency: CriterionAssessment
    safety_sovereignty_tone: CriterionAssessment
    completeness_coverage: CriterionAssessment


class C4SummaryReason(BaseModel):
    """Optional second-pass summary reason returned for C4 reports."""

    reason: str
