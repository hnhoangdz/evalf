from evalf.metrics.base import BaseMetric
from evalf.schemas import EvalCase

from .prompt import build_prompt
from .schema import AnswerRelevanceAssessment


class AnswerRelevanceMetric(BaseMetric):
    """Score whether an answer stays on-topic for the given question."""

    name = "answer_relevance"
    required_inputs = ("question", "actual_output")
    output_schema = AnswerRelevanceAssessment

    def build_prompt(self, case: EvalCase) -> tuple[str, str]:
        """Build the structured judge prompt for answer relevance."""

        return build_prompt(case)
