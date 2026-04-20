from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import EvalAttempt, EvalCase
from .utils import ensure_list, load_json_file, load_jsonl_file


def _coerce_attempts(data: dict[str, Any]) -> list[EvalAttempt] | None:
    """Normalize attempt payloads without mutating the caller's input mapping."""
    attempts_data = data.get("attempts")
    if attempts_data is None:
        actual_outputs = ensure_list(data.get("actual_outputs"))
        if actual_outputs:
            attempts_data = [{"actual_output": output} for output in actual_outputs]

    if attempts_data is None:
        return None

    attempts: list[EvalAttempt] = []
    for attempt in attempts_data:
        payload = dict(attempt)
        if "retrieved_contexts" in payload:
            payload["retrieved_contexts"] = ensure_list(payload["retrieved_contexts"])
        if "reference_contexts" in payload:
            payload["reference_contexts"] = ensure_list(payload["reference_contexts"])
        attempts.append(EvalAttempt.model_validate(payload))
    return attempts


def _coerce_case(data: dict[str, Any], index: int) -> EvalCase:
    """Normalize a raw input object into an `EvalCase` instance."""
    payload = dict(data)
    payload.setdefault("id", f"sample-{index}")
    if "retrieved_contexts" in payload:
        payload["retrieved_contexts"] = ensure_list(payload["retrieved_contexts"])
    if "reference_contexts" in payload:
        payload["reference_contexts"] = ensure_list(payload["reference_contexts"])
    payload["attempts"] = _coerce_attempts(payload)
    payload.pop("actual_outputs", None)
    return EvalCase.model_validate(payload)


def load_cases_from_path(path: str | Path) -> list[EvalCase]:
    """Load evaluation cases from a `.json` or `.jsonl` file."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix == ".jsonl":
        raw_items = load_jsonl_file(input_path)
    elif input_path.suffix == ".json":
        raw_data = load_json_file(input_path)
        if isinstance(raw_data, list):
            raw_items = raw_data
        elif isinstance(raw_data, dict):
            raw_items = [raw_data]
        else:
            raise ValueError("JSON input must be an object or a list of objects.")
    else:
        raise ValueError("Supported input formats are .jsonl and .json.")

    _validate_items(raw_items, f"input file {input_path}")
    return [_coerce_case(item, index + 1) for index, item in enumerate(raw_items)]


def _validate_items(raw_items: list[Any], source: str) -> None:
    """Raise ``TypeError`` when *raw_items* contains non-object entries."""
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise TypeError(
                f"Each item in {source} must be a JSON object, "
                f"but item {idx} has type {type(item).__name__}."
            )


def load_cases_from_json(sample_json: str) -> list[EvalCase]:
    """Load one or more evaluation cases from inline JSON."""
    raw_data = json.loads(sample_json)
    if isinstance(raw_data, dict):
        return [_coerce_case(raw_data, 1)]
    if isinstance(raw_data, list):
        _validate_items(raw_data, "inline JSON")
        return [_coerce_case(item, index + 1) for index, item in enumerate(raw_data)]
    raise ValueError("Inline JSON must be an object or a list of objects.")


def build_case_from_values(
    *,
    question: str | None = None,
    retrieved_contexts: list[str] | None = None,
    reference_contexts: list[str] | None = None,
    actual_output: str | None = None,
    expected_output: str | None = None,
) -> EvalCase:
    """Build a single evaluation case from direct CLI or Python values."""
    return EvalCase(
        id="sample-1",
        question=question,
        retrieved_contexts=retrieved_contexts,
        reference_contexts=reference_contexts,
        actual_output=actual_output,
        expected_output=expected_output,
    )
