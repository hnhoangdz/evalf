import json

import pytest

from evalf.inputs import build_case_from_values, load_cases_from_json, load_cases_from_path

pytestmark = pytest.mark.unit


def test_load_cases_from_json_object() -> None:
    cases = load_cases_from_json(
        '{"question":"q","retrieved_contexts":["ctx"],"actual_output":"a"}'
    )
    assert len(cases) == 1
    assert cases[0].question == "q"


def test_build_case_from_values() -> None:
    case = build_case_from_values(question="q", actual_output="a")
    assert case.question == "q"
    assert case.actual_output == "a"


def test_load_cases_from_json_list_and_attempt_scalar_contexts() -> None:
    cases = load_cases_from_json(
        json.dumps(
            [
                {
                    "question": "q1",
                    "attempts": [
                        {
                            "actual_output": "a1",
                            "retrieved_contexts": "ctx-1",
                            "reference_contexts": "ref-1",
                        }
                    ],
                },
                {"question": "q2", "actual_output": "a2"},
            ]
        )
    )

    assert len(cases) == 2
    assert cases[0].attempts is not None
    assert cases[0].attempts[0].retrieved_contexts == ["ctx-1"]
    assert cases[0].attempts[0].reference_contexts == ["ref-1"]
    assert cases[1].id == "sample-2"


def test_actual_outputs_are_normalized_into_attempts() -> None:
    cases = load_cases_from_json(
        '{"question":"q","expected_output":"e","actual_outputs":["a1","a2"]}'
    )
    assert cases[0].attempts is not None
    assert len(cases[0].attempts) == 2
    assert cases[0].attempts[1].actual_output == "a2"


def test_load_cases_from_json_does_not_mutate_original_payload() -> None:
    payload = {
        "question": "q",
        "expected_output": "e",
        "actual_outputs": ["a1", "a2"],
    }

    cases = load_cases_from_json(json.dumps(payload))

    assert payload["actual_outputs"] == ["a1", "a2"]
    assert cases[0].attempts is not None
    assert len(cases[0].attempts) == 2


def test_load_cases_from_path_supports_json_and_jsonl(tmp_path) -> None:
    json_path = tmp_path / "cases.json"
    json_path.write_text('[{"question":"q-json","actual_output":"a-json"}]', encoding="utf-8")
    jsonl_path = tmp_path / "cases.jsonl"
    jsonl_path.write_text(
        '{"question":"q-jsonl","actual_output":"a-jsonl"}\n',
        encoding="utf-8",
    )

    json_cases = load_cases_from_path(json_path)
    jsonl_cases = load_cases_from_path(jsonl_path)

    assert json_cases[0].question == "q-json"
    assert jsonl_cases[0].question == "q-jsonl"


def test_load_cases_from_json_rejects_non_object_items() -> None:
    with pytest.raises(TypeError, match="item 0 has type int"):
        load_cases_from_json("[1]")

    with pytest.raises(TypeError, match="item 1 has type str"):
        load_cases_from_json('[{"question":"q"}, "bad"]')


def test_load_cases_from_path_rejects_non_object_items(tmp_path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(TypeError, match="item 0 has type int"):
        load_cases_from_path(bad_json)

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text('"just a string"\n', encoding="utf-8")

    with pytest.raises(TypeError, match="item 0 has type str"):
        load_cases_from_path(bad_jsonl)


def test_load_cases_from_path_rejects_missing_or_unsupported_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cases_from_path(tmp_path / "missing.json")

    unsupported_path = tmp_path / "cases.txt"
    unsupported_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Supported input formats are .jsonl and .json."):
        load_cases_from_path(unsupported_path)
