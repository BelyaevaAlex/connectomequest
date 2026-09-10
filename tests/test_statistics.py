import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from connectomequest.statistics import compare_active_runs


def _write_run(path, successes) -> None:
    rows = [
        {
            "query_id": f"q{index}",
            "goal_success": success,
            "precision": float(success),
            "recall": float(success),
            "f1": float(success),
            "proof_valid": success,
        }
        for index, success in enumerate(successes)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_compare_active_runs_is_paired_and_reproducible(tmp_path) -> None:
    candidate = tmp_path / "candidate.parquet"
    reference = tmp_path / "reference.parquet"
    _write_run(candidate, [True, True, False, True])
    _write_run(reference, [False, True, False, False])

    report = compare_active_runs(
        candidate, reference, tmp_path / "comparison.json", bootstrap_samples=200, seed=3
    )

    assert report["episodes"] == 4
    assert report["metrics"]["goal_success"]["paired_difference"] == 0.5
    assert report["mcnemar_exact"]["candidate_only_successes"] == 2
    assert report["mcnemar_exact"]["reference_only_successes"] == 0


def test_compare_active_runs_rejects_mismatched_queries(tmp_path) -> None:
    candidate = tmp_path / "candidate.parquet"
    reference = tmp_path / "reference.parquet"
    _write_run(candidate, [True, False])
    _write_run(reference, [True])

    with pytest.raises(ValueError, match="identical query IDs"):
        compare_active_runs(candidate, reference, tmp_path / "comparison.json")
