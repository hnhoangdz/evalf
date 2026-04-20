from typing import Literal

from pydantic import BaseModel, Field


class ContextCoverageAssessment(BaseModel):
    """Judge output schema for context coverage scoring."""

    score: float = Field(ge=0.0, le=1.0)
    verdict: Literal["yes", "no"]
    reason: str
