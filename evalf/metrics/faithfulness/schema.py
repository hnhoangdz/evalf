from pydantic import BaseModel, Field


class FaithfulnessAssessment(BaseModel):
    """Judge output schema for aggregated faithfulness scoring."""

    score: float = Field(ge=0.0, le=1.0)
    reason: str
