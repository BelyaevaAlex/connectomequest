"""Deterministic miniature graph for tests and end-to-end smoke runs."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa

from connectomequest.graph import GraphStore
from connectomequest.schema import EDGE_SCHEMA, NODE_SCHEMA, NodeType, Relation


def make_toy_graph(output: Path) -> GraphStore:
    labels = [
        ("n0", NodeType.NEURON),
        ("n1", NodeType.NEURON),
        ("n2", NodeType.NEURON),
        ("n3", NodeType.NEURON),
        ("n4", NodeType.NEURON),
        ("type_a", NodeType.NEURON_TYPE),
        ("type_b", NodeType.NEURON_TYPE),
    ]
    nodes = pa.Table.from_pylist(
        [
            {
                "node_id": node_id,
                "external_id": label,
                "node_type": str(node_type),
                "dataset": "toy",
                "attrs_json": json.dumps({"label": label}),
            }
            for node_id, (label, node_type) in enumerate(labels)
        ],
        schema=NODE_SCHEMA,
    )
    primary = [(0, 2, 4.0), (0, 3, 2.0), (1, 3, 3.0), (2, 4, 1.0), (3, 4, 5.0)]
    rows: list[dict] = []
    for src, dst, weight in primary:
        rows.append(
            {
                "src": src,
                "dst": dst,
                "relation": str(Relation.PRESYNAPTIC_TO),
                "weight": weight,
                "source_record": "toy",
            }
        )
        rows.append(
            {
                "src": dst,
                "dst": src,
                "relation": str(Relation.POSTSYNAPTIC_TO),
                "weight": weight,
                "source_record": "toy",
            }
        )
    for neuron, neuron_type in ((2, 5), (3, 5), (4, 6)):
        rows.append(
            {
                "src": neuron,
                "dst": neuron_type,
                "relation": str(Relation.HAS_TYPE),
                "weight": 1.0,
                "source_record": "toy",
            }
        )
    edges = pa.Table.from_pylist(rows, schema=EDGE_SCHEMA)
    return GraphStore.create(
        output,
        nodes,
        edges,
        dataset="toy",
        version="1",
        license_name="synthetic",
    )
