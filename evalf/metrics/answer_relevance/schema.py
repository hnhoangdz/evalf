from pydantic import BaseModel, Field


class AnswerRelevanceAssessment(BaseModel):
    """Judge output schema for answer relevance scoring."""

    score: float = Field(ge=0.0, le=1.0)
    reason: str
