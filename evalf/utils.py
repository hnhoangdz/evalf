from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_list(value: Any) -> list[str] | None:
    """Normalize a scalar or list-like input into a list of strings."""
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def split_csv(value: str | None) -> list[str]:
    """Split a comma-separated string into a list of trimmed non-empty parts."""
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def load_json_file(path: Path) -> Any:
    """Load and decode a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_file(path: Path) -> list[dict[str, Any]]:
    """Load a UTF-8 JSONL file into a list of objects."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def strip_code_fences(text: str) -> str:
    """Remove a single surrounding fenced code block when present."""
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return value


def extract_json_payload(text: str) -> str:
    """Extract the first valid JSON payload from a model response string.

    Tries the whole string first (accepts any valid JSON value), then falls
    back to scanning for the first balanced ``{...}`` substring that parses.
    """
    cleaned = strip_code_fences(text)
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    start = None
    depth = 0
    for index, char in enumerate(cleaned):
        if char == "{":
            if start is None:
                start = index
            depth += 1
        elif char == "}":
            if start is None:
                continue
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : index + 1]
                try:
                    json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
                    continue
                return candidate

    raise ValueError("Could not extract a valid JSON object from the model response.")
