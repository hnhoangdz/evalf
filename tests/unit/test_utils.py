from pathlib import Path

import pytest

from evalf.utils import (
    ensure_list,
    extract_json_payload,
    load_json_file,
    load_jsonl_file,
    split_csv,
)

pytestmark = pytest.mark.unit


def test_ensure_list_wraps_scalars_and_preserves_none() -> None:
    assert ensure_list(None) is None
    assert ensure_list("value") == ["value"]
    assert ensure_list([1, "2"]) == ["1", "2"]


def test_split_csv_ignores_empty_segments_and_trims_whitespace() -> None:
    assert split_csv(" faithfulness, , answer_relevance ,,context_recall ") == [
        "faithfulness",
        "answer_relevance",
        "context_recall",
    ]


def test_load_json_file_reads_utf8_json(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    path.write_text('{"score": 0.9, "reason": "ok"}', encoding="utf-8")

    assert load_json_file(path) == {"score": 0.9, "reason": "ok"}


def test_load_jsonl_file_reads_multiple_objects(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")

    assert load_jsonl_file(path) == [{"id": 1}, {"id": 2}]


def test_load_jsonl_file_raises_for_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id": 1}\n{"id": }\n', encoding="utf-8")

    with pytest.raises(ValueError):
        load_jsonl_file(path)


def test_extract_json_payload_handles_code_fences_and_surrounding_text() -> None:
    text = """
    Here is the assessment:

    ```json
    {"score": 0.75, "reason": "Grounded answer."}
    ```
    """

    payload = extract_json_payload(text)

    assert payload == '{"score": 0.75, "reason": "Grounded answer."}'


def test_extract_json_payload_raises_when_no_json_object_exists() -> None:
    with pytest.raises(ValueError, match="Could not extract a valid JSON object"):
        extract_json_payload("No JSON payload here.")


def test_extract_json_payload_skips_invalid_balanced_objects_and_keeps_scanning() -> None:
    text = 'bad {"score": 0.4,} trailing {"score": 0.8, "reason": "ok"}'

    payload = extract_json_payload(text)

    assert payload == '{"score": 0.8, "reason": "ok"}'
