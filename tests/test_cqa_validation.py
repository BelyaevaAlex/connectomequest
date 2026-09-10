from __future__ import annotations

import pickle

import pytest

from connectomequest.baselines.cqa_validation import TASK_STRUCTURES, validate_cqa_partitions


def test_cqa_partition_validation_allows_an_empty_path_group(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with (bundle / "train-queries.pkl").open("wb") as stream:
        pickle.dump({TASK_STRUCTURES["pi"]: {((1, (2, 3)), (4, (5,)))}}, stream)

    report = validate_cqa_partitions(bundle, ["pi"])

    assert report["path_total"] == 0
    assert report["other_total"] == 1


def test_cqa_partition_validation_rejects_a_missing_requested_task(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with (bundle / "train-queries.pkl").open("wb") as stream:
        pickle.dump({TASK_STRUCTURES["pi"]: {((1, (2, 3)), (4, (5,)))}}, stream)

    with pytest.raises(ValueError, match="empty requested partitions: 1p"):
        validate_cqa_partitions(bundle, ["1p", "pi"])
