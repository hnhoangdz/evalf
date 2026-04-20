"""Generic Pydantic-backed prompt builder for structured LLM judge calls."""

from __future__ import annotations

import json
import typing as t

from pydantic import BaseModel

InputModel = t.TypeVar("InputModel", bound=BaseModel)
OutputModel = t.TypeVar("OutputModel", bound=BaseModel)


class PydanticPrompt(t.Generic[InputModel, OutputModel]):
    """Render structured prompts backed by Pydantic input/output schemas."""

    input_model: type[InputModel]
    output_model: type[OutputModel]
    instruction: str = ""
    system_prompt: str = ""
    examples: t.Sequence[tuple[InputModel, OutputModel]] = ()

    def __init__(
        self,
        *,
        examples: t.Sequence[tuple[InputModel, OutputModel]] | None = None,
    ) -> None:
        base_examples = self.__class__.examples if examples is None else examples
        self.examples: list[tuple[InputModel, OutputModel]] = list(base_examples)

    def _generate_output_signature(self) -> str:
        return (
            "Please return the output in JSON format that complies with the "
            "following schema:\n"
            f"{json.dumps(self.output_model.model_json_schema(), indent=2)}\n"
            "Use double quotes, not single quotes."
        )

    def _generate_examples(self) -> str:
        if not self.examples:
            return ""

        example_strings = []
        for index, (input_data, output_data) in enumerate(self.examples, start=1):
            example_strings.append(
                f"Example {index}\n"
                f"Input:\n{input_data.model_dump_json(indent=2, exclude_none=True)}\n\n"
                f"Output:\n{output_data.model_dump_json(indent=2, exclude_none=True)}"
            )

        return "\n--------EXAMPLES-----------\n" + "\n\n".join(example_strings)

    def to_string(self, data: InputModel | None = None) -> str:
        """Render the user prompt body for a structured evaluation call."""
        prompt_parts = [
            self.instruction.strip(),
            self._generate_output_signature(),
            self._generate_examples(),
            "\n-----------------------------\n",
            "\nNow perform the same with the following input:\n",
        ]

        if data is None:
            prompt_parts.append("Input: (None)\n")
        else:
            prompt_parts.append(f"Input:\n{data.model_dump_json(indent=2, exclude_none=True)}\n")

        prompt_parts.append("\nOutput:")
        return "\n".join(prompt_parts)

    def render(self, data: InputModel | None = None) -> tuple[str, str]:
        """Return the `(system_prompt, user_prompt)` pair for the given input."""
        return self.system_prompt, self.to_string(data)
