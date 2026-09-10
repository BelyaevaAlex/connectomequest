import json

import numpy as np
import pyarrow as pa
import pytest

from connectomequest.adapters.h01 import _h01_avro_inventory, _sample_layer_labels


def test_h01_inventory_excludes_parallel_json_exports(tmp_path) -> None:
    inventory = [
        {"name": "exported/export000", "size": 10},
        {"name": "exported/export000.json", "size": 20},
        {"name": "exported/export001", "size": 30},
        {"name": "exported/export001.json", "size": 40},
    ]
    (tmp_path / "inventory.json").write_text(json.dumps(inventory))

    selected = _h01_avro_inventory(tmp_path)

    assert [item["name"] for item in selected] == [
        "exported/export000",
        "exported/export001",
    ]
    assert sum(item["size"] for item in selected) == 40


def test_h01_inventory_rejects_json_only_release(tmp_path) -> None:
    inventory = [{"name": "exported/export000.json", "size": 20}]
    (tmp_path / "inventory.json").write_text(json.dumps(inventory))

    with pytest.raises(ValueError, match="no Avro"):
        _h01_avro_inventory(tmp_path)


def test_sample_layer_labels_converts_c3_voxels_to_layer_resolution() -> None:
    table = pa.table(
        {
            "x": pa.array([125, 250, 50_000], type=pa.int64()),
            "y": pa.array([125, 250, 50_000], type=pa.int64()),
            "z": pa.array([16, 32, 50_000], type=pa.int64()),
        }
    )
    volume = np.zeros((4, 4, 4), dtype=np.uint64)
    volume[1, 1, 1] = 3
    volume[2, 2, 2] = 5

    labels = _sample_layer_labels(
        table,
        volume,
        np.asarray((1000, 1000, 528), dtype=np.int64),
    )

    assert labels.tolist() == [3, 5, 0]
