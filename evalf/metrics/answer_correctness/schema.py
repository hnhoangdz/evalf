from pydantic import BaseModel, Field


class AnswerCorrectnessAssessment(BaseModel):
    """Judge output schema for answer correctness scoring."""

    score: float = Field(ge=0.0, le=1.0)
    reason: str
