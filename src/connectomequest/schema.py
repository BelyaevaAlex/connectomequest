"""Canonical, dataset-independent schema for brain-wiring KGs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pyarrow as pa


class NodeType(StrEnum):
    NEURON = "neuron"
    ROI = "roi"
    NEURON_TYPE = "neuron_type"
    INSTANCE = "instance"
    CELL_BODY_FIBER = "cell_body_fiber"
    CORTICAL_LAYER = "cortical_layer"
    SYNAPSE_CLASS = "synapse_class"


class Relation(StrEnum):
    # src is presynaptic to dst; this is the primary directed wiring edge.
    PRESYNAPTIC_TO = "presynaptic_to"
    # Materialized inverse of PRESYNAPTIC_TO when requested.
    POSTSYNAPTIC_TO = "postsynaptic_to"
    IN_ROI = "in_roi"
    HAS_TYPE = "has_type"
    HAS_INSTANCE = "has_instance"
    HAS_CELL_BODY_FIBER = "has_cell_body_fiber"
    IN_CORTICAL_LAYER = "in_cortical_layer"
    HAS_SYNAPSE_CLASS = "has_synapse_class"


NODE_SCHEMA = pa.schema(
    [
        pa.field("node_id", pa.int64(), nullable=False),
        pa.field("external_id", pa.string(), nullable=False),
        pa.field("node_type", pa.string(), nullable=False),
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("attrs_json", pa.string(), nullable=False),
    ]
)

EDGE_SCHEMA = pa.schema(
    [
        pa.field("src", pa.int64(), nullable=False),
        pa.field("dst", pa.int64(), nullable=False),
        pa.field("relation", pa.string(), nullable=False),
        pa.field("weight", pa.float32(), nullable=False),
        pa.field("source_record", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    version: str
    license: str
    config_path: Path
    metadata: dict[str, Any]


def ensure_schema(table: pa.Table, schema: pa.Schema, label: str) -> pa.Table:
    """Validate required columns and cast to the canonical physical types."""

    missing = set(schema.names) - set(table.column_names)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    return table.select(schema.names).cast(schema, safe=True)
