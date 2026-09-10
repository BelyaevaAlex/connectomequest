"""HemiBrain v1.2.1 neuPrint export and canonical KG adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from connectomequest.adapters.common import connection_edges, make_neuron_nodes
from connectomequest.config import dataset_config
from connectomequest.download import read_neuprint_token
from connectomequest.graph import GraphStore
from connectomequest.manifest import Manifest
from connectomequest.schema import EDGE_SCHEMA, NODE_SCHEMA, NodeType, Relation

NEURON_QUERY = """
MATCH (n:Neuron)
WHERE n.status = "Traced" AND NOT coalesce(n.cropped, false)
RETURN n.bodyId AS bodyId,
       n.type AS type,
       n.instance AS instance,
       n.cellBodyFiber AS cellBodyFiber,
       n.roiInfo AS roiInfo
ORDER BY n.bodyId
"""


def _parse_roi_info(value: object) -> dict[str, object]:
    """Normalize neuPrint roiInfo across JSON and Arrow response formats."""

    parsed = value
    for _ in range(2):
        if not isinstance(parsed, str):
            break
        if not parsed.strip():
            return {}
        parsed = json.loads(parsed)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"neuPrint roiInfo must decode to an object, got {type(parsed).__name__}")
    return parsed


def export_hemibrain(
    raw_dir: Path,
    *,
    chunk_size: int = 256,
) -> Manifest:
    """Export a resumable traced-neuron snapshot from neuPrint."""

    from neuprint import Client

    config = dataset_config("hemibrain")
    raw_dir = Path(raw_dir)
    edge_dir = raw_dir / "edges"
    edge_dir.mkdir(parents=True, exist_ok=True)
    client = Client(
        config["server"],
        dataset=config["neuprint_dataset"],
        token=read_neuprint_token(),
        progress=False,
    )
    frame = client.fetch_custom(NEURON_QUERY)
    frame["roiInfo_json"] = frame["roiInfo"].map(
        lambda value: json.dumps(_parse_roi_info(value), sort_keys=True)
    )
    frame = frame.drop(columns=["roiInfo"])
    nodes_path = raw_dir / "neurons.parquet"
    nodes_temporary = nodes_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), nodes_temporary)
    nodes_temporary.replace(nodes_path)
    body_ids = frame["bodyId"].to_numpy(dtype=np.int64)
    for part, start in enumerate(
        tqdm(range(0, len(body_ids), chunk_size), desc="export HemiBrain edges")
    ):
        destination = edge_dir / f"part-{part:05d}.parquet"
        if destination.exists():
            continue
        source_ids = ",".join(str(int(value)) for value in body_ids[start : start + chunk_size])
        query = f"""
        MATCH (a:Neuron)-[edge:ConnectsTo]->(b:Neuron)
        WHERE a.bodyId IN [{source_ids}]
          AND b.status = "Traced"
          AND NOT coalesce(b.cropped, false)
        RETURN a.bodyId AS body_pre,
               b.bodyId AS body_post,
               edge.weight AS weight
        """
        edges = client.fetch_custom(query)
        temporary = destination.with_suffix(".parquet.tmp")
        pq.write_table(
            pa.Table.from_pandas(edges, preserve_index=False),
            temporary,
            compression="zstd",
        )
        temporary.replace(destination)
    manifest = Manifest(
        "hemibrain",
        config["version"],
        config["license"],
        metadata={
            "neuprint_dataset": config["neuprint_dataset"],
            "subset": "status=Traced and cropped=false",
            "chunk_size": chunk_size,
        },
    )
    manifest.add_file(nodes_path, raw_dir, config["server"])
    for path in sorted(edge_dir.glob("*.parquet")):
        manifest.add_file(path, raw_dir, config["server"])
    manifest.write(raw_dir / "manifest.json")
    return manifest


def _semantic_graph(
    neurons: pa.Table,
    body_ids: np.ndarray,
) -> tuple[pa.Table, pa.Table]:
    base_nodes = make_neuron_nodes(body_ids, "hemibrain")
    metadata = [
        json.dumps(
            {
                key: value
                for key, value in {
                    "type": row.get("type"),
                    "instance": row.get("instance"),
                    "cellBodyFiber": row.get("cellBodyFiber"),
                    "roiInfo": _parse_roi_info(row.get("roiInfo_json")),
                }.items()
                if value is not None
            },
            sort_keys=True,
        )
        for row in neurons.to_pylist()
    ]
    base_nodes = base_nodes.set_column(
        base_nodes.schema.get_field_index("attrs_json"),
        NODE_SCHEMA.field("attrs_json"),
        pa.array(metadata, type=pa.string()),
    )
    category_specs = (
        ("type", NodeType.NEURON_TYPE, Relation.HAS_TYPE),
        ("instance", NodeType.INSTANCE, Relation.HAS_INSTANCE),
        (
            "cellBodyFiber",
            NodeType.CELL_BODY_FIBER,
            Relation.HAS_CELL_BODY_FIBER,
        ),
    )
    node_rows: list[dict] = []
    edge_rows: list[dict] = []
    next_id = len(body_ids)
    for column, node_type, relation in category_specs:
        values = neurons[column].to_pylist()
        unique_values = sorted({str(value) for value in values if value})
        value_to_id = {value: next_id + index for index, value in enumerate(unique_values)}
        next_id += len(unique_values)
        node_rows.extend(
            {
                "node_id": node_id,
                "external_id": value,
                "node_type": str(node_type),
                "dataset": "hemibrain",
                "attrs_json": "{}",
            }
            for value, node_id in value_to_id.items()
        )
        edge_rows.extend(
            {
                "src": neuron_id,
                "dst": value_to_id[str(value)],
                "relation": str(relation),
                "weight": 1.0,
                "source_record": f"neuPrint:{column}",
            }
            for neuron_id, value in enumerate(values)
            if value
        )
    roi_infos = [_parse_roi_info(value) for value in neurons["roiInfo_json"].to_pylist()]
    roi_values = sorted({roi for info in roi_infos for roi in info})
    roi_to_id = {roi: next_id + index for index, roi in enumerate(roi_values)}
    node_rows.extend(
        {
            "node_id": node_id,
            "external_id": roi,
            "node_type": str(NodeType.ROI),
            "dataset": "hemibrain",
            "attrs_json": "{}",
        }
        for roi, node_id in roi_to_id.items()
    )
    edge_rows.extend(
        {
            "src": neuron_id,
            "dst": roi_to_id[roi],
            "relation": str(Relation.IN_ROI),
            "weight": float((counts or {}).get("pre", 0) + (counts or {}).get("post", 0)),
            "source_record": "neuPrint:roiInfo",
        }
        for neuron_id, info in enumerate(roi_infos)
        for roi, counts in info.items()
        if (counts or {}).get("pre", 0) + (counts or {}).get("post", 0) > 0
    )
    semantic_nodes = pa.Table.from_pylist(node_rows, schema=NODE_SCHEMA)
    semantic_edges = pa.Table.from_pylist(edge_rows, schema=EDGE_SCHEMA)
    return pa.concat_tables((base_nodes, semantic_nodes)), semantic_edges


def build_hemibrain(
    raw_dir: Path,
    output_dir: Path,
    *,
    min_weight: float = 1.0,
) -> GraphStore:
    config = dataset_config("hemibrain")
    raw_dir = Path(raw_dir)
    nodes_path = raw_dir / "neurons.parquet"
    edge_dir = raw_dir / "edges"
    if not nodes_path.exists() or not edge_dir.exists():
        raise FileNotFoundError("run cq export-hemibrain with a neuPrint token first")
    neurons = pq.read_table(nodes_path, memory_map=True).sort_by([("bodyId", "ascending")])
    body_ids = neurons["bodyId"].combine_chunks().to_numpy(zero_copy_only=False)
    raw_edges = pq.read_table(edge_dir, memory_map=True)
    weights = raw_edges["weight"].combine_chunks().to_numpy(zero_copy_only=False)
    keep = weights >= min_weight
    pre = raw_edges["body_pre"].combine_chunks().to_numpy(zero_copy_only=False)[keep]
    post = raw_edges["body_post"].combine_chunks().to_numpy(zero_copy_only=False)[keep]
    weights = weights[keep]
    known = np.isin(pre, body_ids) & np.isin(post, body_ids)
    pre, post, weights = pre[known], post[known], weights[known]
    nodes, semantic_edges = _semantic_graph(neurons, body_ids)
    wiring_edges = connection_edges(
        body_ids,
        pre,
        post,
        weights,
        source_record=f"neuPrint:{config['neuprint_dataset']}",
        materialize_inverse=True,
    )
    edges = pa.concat_tables((wiring_edges, semantic_edges))
    return GraphStore.create(
        output_dir,
        nodes,
        edges,
        dataset="hemibrain",
        version=config["version"],
        license_name=config["license"],
        metadata={
            "neuprint_dataset": config["neuprint_dataset"],
            "minimum_synapse_weight": min_weight,
            "num_curated_neurons": len(body_ids),
            "num_directed_wiring_edges": len(pre),
        },
    )
