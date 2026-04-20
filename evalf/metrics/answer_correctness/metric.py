from evalf.metrics.base import BaseMetric
from evalf.schemas import EvalCase

from .prompt import build_prompt
from .schema import AnswerCorrectnessAssessment


class AnswerCorrectnessMetric(BaseMetric):
    """Score whether an answer matches the expected answer for a question."""

    name = "answer_correctness"
    required_inputs = ("question", "actual_output", "expected_output")
    output_schema = AnswerCorrectnessAssessment

    def build_prompt(self, case: EvalCase) -> tuple[str, str]:
        """Build the structured judge prompt for answer correctness."""

        return build_prompt(case)
