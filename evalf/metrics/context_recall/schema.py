from pydantic import BaseModel, Field


class ContextRecallAssessment(BaseModel):
    """Judge output schema for aggregated context recall scoring."""

    score: float = Field(ge=0.0, le=1.0)
    reason: str
