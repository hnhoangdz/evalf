from pydantic import BaseModel, Field


class ContextRelevanceAssessment(BaseModel):
    """Judge output schema for aggregated context relevance scoring."""

    score: float = Field(ge=0.0, le=1.0)
    reason: str
