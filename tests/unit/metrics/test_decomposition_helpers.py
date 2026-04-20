import pytest

from evalf.metrics.decomposition import (
    build_context_chunks,
    dedupe_ids,
    ensure_complete_id_mapping,
)

pytestmark = pytest.mark.unit


def test_build_context_chunks_assigns_stable_ids_in_order() -> None:
    chunks = build_context_chunks(["first context", "second context"])

    assert [(chunk.context_id, chunk.text) for chunk in chunks] == [
        ("ctx_1", "first context"),
        ("ctx_2", "second context"),
    ]


def test_dedupe_ids_preserves_order_of_first_occurrence() -> None:
    assert dedupe_ids(["rc1", "rc2", "rc1", "rc3", "rc2"]) == ["rc1", "rc2", "rc3"]


def test_ensure_complete_id_mapping_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate claim ids returned: rc1"):
        ensure_complete_id_mapping(
            expected_ids=["rc1", "rc2"],
            observed_ids=["rc1", "rc1", "rc2"],
            entity_name="claim",
        )


def test_ensure_complete_id_mapping_rejects_missing_ids() -> None:
    with pytest.raises(ValueError, match="Missing context ids in response: ctx_2"):
        ensure_complete_id_mapping(
            expected_ids=["ctx_1", "ctx_2"],
            observed_ids=["ctx_1"],
            entity_name="context",
        )


def test_ensure_complete_id_mapping_rejects_unknown_ids() -> None:
    with pytest.raises(ValueError, match="Unknown claim ids in response: rc3"):
        ensure_complete_id_mapping(
            expected_ids=["rc1", "rc2"],
            observed_ids=["rc1", "rc2", "rc3"],
            entity_name="claim",
        )
