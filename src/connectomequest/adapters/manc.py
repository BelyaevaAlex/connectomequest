"""MANC/Male CNS v1.0 bulk Feather adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from tqdm import tqdm

from connectomequest.adapters.common import connection_edges, make_neuron_nodes
from connectomequest.config import dataset_config
from connectomequest.graph import GraphStore
from connectomequest.schema import EDGE_SCHEMA, NODE_SCHEMA, NodeType, Relation


def _membership(values: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_reference, values)
    in_bounds = positions < len(sorted_reference)
    result = np.zeros(len(values), dtype=bool)
    result[in_bounds] = sorted_reference[positions[in_bounds]] == values[in_bounds]
    return result


def _curated_annotations(path: Path) -> tuple[pa.Table, np.ndarray]:
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        annotations = reader.read_all()
    required = {"bodyId", "mancType"}
    if missing := required - set(annotations.column_names):
        raise ValueError(f"MANC annotations lack columns: {sorted(missing)}")
    curated = annotations.filter(
        pc.and_(pc.is_valid(annotations["bodyId"]), pc.is_valid(annotations["mancType"]))
    )
    curated = curated.sort_by([("bodyId", "ascending")])
    body_ids = curated["bodyId"].combine_chunks().to_numpy(zero_copy_only=False)
    unique_mask = np.concatenate(([True], body_ids[1:] != body_ids[:-1]))
    if not unique_mask.all():
        curated = curated.filter(pa.array(unique_mask))
        body_ids = body_ids[unique_mask]
    return curated, body_ids


def _filtered_connections(
    path: Path,
    curated_body_ids: np.ndarray,
    minimum_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre_parts: list[np.ndarray] = []
    post_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        names = reader.schema.names
        for required in ("body_pre", "body_post", "weight"):
            if required not in names:
                raise ValueError(f"MANC connections lack column {required!r}")
        indices = {name: names.index(name) for name in ("body_pre", "body_post", "weight")}
        for batch_index in tqdm(range(reader.num_record_batches), desc="filter MANC batches"):
            batch = reader.get_batch(batch_index)
            pre = batch.column(indices["body_pre"]).to_numpy(zero_copy_only=False)
            post = batch.column(indices["body_post"]).to_numpy(zero_copy_only=False)
            weight = batch.column(indices["weight"]).to_numpy(zero_copy_only=False)
            keep = (
                _membership(pre, curated_body_ids)
                & _membership(post, curated_body_ids)
                & (weight >= minimum_weight)
            )
            if keep.any():
                pre_parts.append(pre[keep])
                post_parts.append(post[keep])
                weight_parts.append(weight[keep])
    if not pre_parts:
        raise ValueError("MANC filtering produced no wiring edges")
    return (
        np.concatenate(pre_parts),
        np.concatenate(post_parts),
        np.concatenate(weight_parts),
    )


def _nodes_and_type_edges(
    annotations: pa.Table,
    body_ids: np.ndarray,
) -> tuple[pa.Table, pa.Table]:
    neurons = make_neuron_nodes(body_ids, "manc")
    metadata_columns = [
        name
        for name in ("mancType", "instance", "class", "superclass", "somaSide", "status")
        if name in annotations.column_names
    ]
    metadata = [
        json.dumps(
            {
                name: annotations[name][row].as_py()
                for name in metadata_columns
                if annotations[name][row].as_py() is not None
            },
            sort_keys=True,
        )
        for row in range(len(annotations))
    ]
    neurons = neurons.set_column(
        neurons.schema.get_field_index("attrs_json"),
        NODE_SCHEMA.field("attrs_json"),
        pa.array(metadata, type=pa.string()),
    )
    neuron_types = annotations["mancType"].to_pylist()
    unique_types = sorted(set(neuron_types))
    type_offset = len(body_ids)
    type_to_id = {value: type_offset + index for index, value in enumerate(unique_types)}
    type_nodes = pa.Table.from_pylist(
        [
            {
                "node_id": node_id,
                "external_id": value,
                "node_type": str(NodeType.NEURON_TYPE),
                "dataset": "manc",
                "attrs_json": "{}",
            }
            for value, node_id in type_to_id.items()
        ],
        schema=NODE_SCHEMA,
    )
    type_edges = pa.Table.from_pylist(
        [
            {
                "src": row,
                "dst": type_to_id[neuron_type],
                "relation": str(Relation.HAS_TYPE),
                "weight": 1.0,
                "source_record": "body-annotations:mancType",
            }
            for row, neuron_type in enumerate(neuron_types)
        ],
        schema=EDGE_SCHEMA,
    )
    return pa.concat_tables((neurons, type_nodes)), type_edges


def build_manc(
    raw_dir: Path,
    output_dir: Path,
    *,
    min_weight: float | None = None,
) -> GraphStore:
    config = dataset_config("manc")
    raw_dir = Path(raw_dir)
    connection_file = raw_dir / config["files"]["connections"]["filename"]
    annotation_file = raw_dir / config["files"]["annotations"]["filename"]
    if not connection_file.exists() or not annotation_file.exists():
        raise FileNotFoundError("run cq download manc before building MANC")
    threshold = config["minimum_synapse_weight"] if min_weight is None else min_weight
    annotations, body_ids = _curated_annotations(annotation_file)
    pre, post, weight = _filtered_connections(
        connection_file,
        body_ids,
        float(threshold),
    )
    nodes, type_edges = _nodes_and_type_edges(annotations, body_ids)
    wiring_edges = connection_edges(
        body_ids,
        pre,
        post,
        weight,
        source_record=connection_file.name,
        materialize_inverse=True,
    )
    edges = pa.concat_tables((wiring_edges, type_edges))
    return GraphStore.create(
        output_dir,
        nodes,
        edges,
        dataset="manc",
        version=config["version"],
        license_name=config["license"],
        metadata={
            "minimum_synapse_weight": float(threshold),
            "subset": "non-null mancType at Male CNS v1.0",
            "num_curated_neurons": len(body_ids),
            "num_neuron_types": len(nodes) - len(body_ids),
            "num_directed_wiring_edges": len(pre),
            "annotation_columns": annotations.column_names,
        },
    )
