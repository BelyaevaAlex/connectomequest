from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from connectomequest.confirmatory_evaluation_v2 import (
    evaluate_cell,
    run_direction,
    shard_direction_cells,
)
from connectomequest.confirmatory_v2 import (
    CONFIRMATORY_SEED,
    TEST_QUERIES,
    TOTAL_QUERIES,
    assert_preseal_absence,
    canonical_sha,
    method_matrix,
    query_path,
    verify_artifact_hashes,
    verify_matrix,
)


def _selection() -> dict:
    matrix = method_matrix()
    return {"matrix": matrix, "matrix_sha256": canonical_sha(matrix)}


def test_default_matrix_is_exact_and_includes_b64_audits() -> None:
    cells = method_matrix()
    assert len(cells) == 87
    assert len({(c["held_out"], c["method"], c["budget"], c["seed"]) for c in cells}) == 87
    for held_out in ("h01", "manc", "hemibrain"):
        direction = [cell for cell in cells if cell["held_out"] == held_out]
        assert len(direction) == 29
        assert {cell["method"] for cell in direction if cell["budget"] == 64} == {
            "weight",
            "degree",
            "snapshot_v2",
            "gate_v2",
            "random",
            "minerva",
            "oracle",
        }
        assert all(
            cell["budget"] == 64 for cell in direction if cell["method"] in {"random", "minerva"}
        )

        degree = next(cell for cell in direction if cell["method"] == "degree")
        oracle = next(cell for cell in direction if cell["method"] == "oracle")
        assert degree["access_regime"] == "observable_paid_lookahead"
        assert oracle["access_regime"] == "privileged_ceiling"


def test_protocol_constants_match_preregistered_request() -> None:
    assert CONFIRMATORY_SEED == 2_718_281
    assert TOTAL_QUERIES == 50_000
    assert TEST_QUERIES == 5_000


def test_matrix_rejects_controller_drift() -> None:
    selection = _selection()
    selection["matrix"][0] = {**selection["matrix"][0], "controller": "changed"}
    with pytest.raises(ValueError, match="controller mismatch"):
        verify_matrix(selection)


def test_artifact_verifier_reports_added_removed_and_modified() -> None:
    with pytest.raises(RuntimeError, match="added.py") as error:
        verify_artifact_hashes(
            {"same.py": "a", "removed.py": "b", "modified.py": "c"},
            {"same.py": "a", "added.py": "z", "modified.py": "d"},
        )
    message = str(error.value)
    assert "removed.py" in message
    assert "modified.py" in message


def test_preseal_guard_rejects_even_a_query_manifest(tmp_path: Path) -> None:
    path = query_path(tmp_path, "h01").with_suffix(".manifest.json")
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="before protocol seal"):
        assert_preseal_absence(tmp_path)


def test_direction_dry_run_never_opens_queries(tmp_path: Path, monkeypatch) -> None:
    selection = _selection()
    monkeypatch.setattr(
        "connectomequest.confirmatory_evaluation_v2.verify_seal",
        lambda repo, require_queries=False: {"seal_sha256": "s" * 64},
    )
    monkeypatch.setattr(
        "connectomequest.confirmatory_evaluation_v2.load_method_selection",
        lambda repo: selection,
    )
    monkeypatch.setattr(
        "connectomequest.confirmatory_evaluation_v2.authorize_test_open",
        lambda repo: pytest.fail("dry-run authorized the test open"),
    )
    monkeypatch.setattr(
        "connectomequest.confirmatory_evaluation_v2.read_confirmatory_test_queries",
        lambda *args, **kwargs: pytest.fail("dry-run read test rows"),
    )
    result = run_direction(tmp_path, "manc", dry_run=True)
    assert result["cells"] == 29
    assert result["test_rows_opened"] is False


def test_three_shards_are_an_exact_nonoverlapping_cover() -> None:
    selection = _selection()
    shards = [
        shard_direction_cells(selection, "h01", shard_index=index, num_shards=3)
        for index in range(3)
    ]
    keys = [
        (cell["held_out"], cell["method"], cell["budget"], cell["seed"])
        for shard in shards
        for cell in shard
    ]
    assert [len(shard) for shard in shards] == [10, 10, 9]
    assert len(keys) == len(set(keys)) == 29
    assert set(keys) == {
        (cell["held_out"], cell["method"], cell["budget"], cell["seed"])
        for cell in method_matrix()
        if cell["held_out"] == "h01"
    }


def test_cell_evaluation_reuses_worker_level_seal_verification() -> None:
    source = inspect.getsource(evaluate_cell)
    assert "verify_seal(" not in source
    assert "load_method_selection(" not in source
    assert '"seal_sha256": seal_sha256' in source
    assert '"method_selection_sha256": method_selection_sha256' in source
