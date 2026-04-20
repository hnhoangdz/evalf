from pydantic import BaseModel, Field


class ContextPrecisionAssessment(BaseModel):
    """Judge output schema for aggregated context precision scoring."""

    score: float = Field(ge=0.0, le=1.0)
    reason: str
