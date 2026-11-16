"""Shared, bounded-memory reader for FlyEM aggregate connection tables."""
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa


def membership(values: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_reference, values)
    valid = positions < len(sorted_reference)
    result = np.zeros(len(values), dtype=bool)
    result[valid] = sorted_reference[positions[valid]] == values[valid]
    return result


def iter_connection_batches(
    path: Path, body_ids: np.ndarray, *, minimum_weight: float,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    ids = np.asarray(body_ids)
    if ids.ndim != 1 or not len(ids) or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("body_ids must be nonempty, sorted and unique")
    if not np.isfinite(minimum_weight) or minimum_weight <= 0:
        raise ValueError("minimum_weight must be finite and positive")
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        required = ("body_pre", "body_post", "weight")
        if missing := set(required) - set(reader.schema.names):
            raise ValueError(f"connections lack columns: {sorted(missing)}")
        indices = [reader.schema.get_field_index(name) for name in required]
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            pre, post, weight = [batch.column(i).to_numpy(zero_copy_only=False) for i in indices]
            if not np.isfinite(weight).all() or np.any(weight <= 0):
                raise ValueError(f"invalid aggregate weight in batch {index}")
            if not np.issubdtype(pre.dtype, np.integer) or not np.issubdtype(post.dtype, np.integer):
                raise ValueError("connection body IDs must be non-null integers")
            keep = membership(pre, ids) & membership(post, ids) & (weight >= minimum_weight)
            # Yield empty batches too: checkpoint numbers must follow source batches.
            yield pre[keep], post[keep], weight[keep]
