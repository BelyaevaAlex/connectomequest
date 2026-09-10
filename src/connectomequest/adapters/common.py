"""Shared high-throughput conversion helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pyarrow as pa

from connectomequest.schema import EDGE_SCHEMA, NODE_SCHEMA, NodeType, Relation


def make_neuron_nodes(external_ids: np.ndarray, dataset: str) -> pa.Table:
    external = [str(value) for value in external_ids.tolist()]
    return pa.Table.from_arrays(
        [
            pa.array(np.arange(len(external), dtype=np.int64)),
            pa.array(external, type=pa.string()),
            pa.array([str(NodeType.NEURON)] * len(external), type=pa.string()),
            pa.array([dataset] * len(external), type=pa.string()),
            pa.array(["{}"] * len(external), type=pa.string()),
        ],
        schema=NODE_SCHEMA,
    )


def connection_edges(
    body_ids: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    *,
    source_record: str,
    materialize_inverse: bool = True,
) -> pa.Table:
    src = np.searchsorted(body_ids, pre).astype(np.int64, copy=False)
    dst = np.searchsorted(body_ids, post).astype(np.int64, copy=False)
    weights = weight.astype(np.float32, copy=False)
    sources = [src, dst]
    targets = [dst, src]
    relations = [str(Relation.PRESYNAPTIC_TO), str(Relation.POSTSYNAPTIC_TO)]
    count = 2 if materialize_inverse else 1
    relation_array = pa.concat_arrays(
        [pa.repeat(pa.scalar(relation), len(src)) for relation in relations[:count]]
    )
    total = len(src) * count
    return pa.Table.from_arrays(
        [
            pa.array(np.concatenate(sources[:count])),
            pa.array(np.concatenate(targets[:count])),
            relation_array,
            pa.array(np.tile(weights, count)),
            pa.repeat(pa.scalar(source_record), total),
        ],
        schema=EDGE_SCHEMA,
    )


def first_present(names: Iterable[str], available: Iterable[str]) -> str | None:
    available_set = set(available)
    return next((name for name in names if name in available_set), None)
