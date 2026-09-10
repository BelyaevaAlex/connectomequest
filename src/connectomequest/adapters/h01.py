"""Streaming H01 adapter restricted to soma-bearing C3 segments.

The binary Avro export is about 30.6 GiB and contains millions of fragments.
The paper-scale graph is recovered by first mapping the official cell-body mesh
labels to C3 segment IDs, then retaining only synapses whose endpoints are both
soma-bearing segments. Avro shards are downloaded to local scratch, reduced, and
deleted immediately.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fastavro import reader as avro_reader
from tqdm import tqdm

from connectomequest.adapters.common import connection_edges, make_neuron_nodes
from connectomequest.config import dataset_config
from connectomequest.download import (
    h01_avro_objects,
    inventory_h01,
    list_gcs_objects,
    resumable_download,
)
from connectomequest.graph import GraphStore
from connectomequest.manifest import sha256_file
from connectomequest.schema import EDGE_SCHEMA, NODE_SCHEMA, NodeType, Relation

LAYERS_URL = "precomputed://https://storage.googleapis.com/h01-release/data/20210601/layers"
C3_RESOLUTION_NM = np.asarray((8, 8, 33), dtype=np.int64)
LAYER_LABELS = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5", 6: "L6", 7: "WM"}
CELL_BODY_URL = "precomputed://https://storage.googleapis.com/h01-release/data/20210601/cell_bodies"
C3_URL = "precomputed://https://storage.googleapis.com/h01-release/data/20210601/c3"
MESH_PREFIX = "data/20210601/cell_bodies/mesh/"


def _map_soma_mesh_shard(filename: str) -> list[dict]:
    from cloudvolume import CloudVolume

    bodies = CloudVolume(CELL_BODY_URL, progress=False, cache=False, parallel=1)
    labels = bodies.mesh.reader.list_labels(filename, path="mesh")
    meshes = bodies.mesh.get(
        labels.tolist(),
        concat=True,
        progress=False,
        allow_missing=True,
    )
    resolution = np.asarray((8, 8, 33), dtype=np.float64)
    points: dict[int, tuple[int, int, int]] = {}
    for label, mesh in meshes.items():
        physical_point = np.median(mesh.vertices, axis=0)
        points[int(label)] = tuple(int(value) for value in np.floor(physical_point / resolution))
    c3 = CloudVolume(
        C3_URL,
        progress=False,
        cache=False,
        parallel=1,
        fill_missing=True,
    )
    minimum = np.asarray(c3.bounds.minpt)
    maximum = np.asarray(c3.bounds.maxpt)
    in_bounds = [
        point
        for point in points.values()
        if np.all(np.asarray(point) >= minimum) and np.all(np.asarray(point) < maximum)
    ]
    values = c3.scattered_points(in_bounds) if in_bounds else {}
    return [
        {
            "cell_body_label": label,
            "x": point[0],
            "y": point[1],
            "z": point[2],
            "segment_id": int(values.get(point, 0)),
        }
        for label, point in points.items()
    ]


def map_h01_somas(
    output: Path,
    *,
    workers: int = 8,
    max_shards: int | None = None,
) -> pa.Table:
    """Map official cell-body meshes to C3 segment IDs and cache as Parquet."""

    output = Path(output)
    if output.exists():
        return pq.read_table(output, memory_map=True)
    objects = list_gcs_objects("h01-release", MESH_PREFIX)
    filenames = sorted(
        Path(item["name"]).name for item in objects if item["name"].endswith(".shard")
    )
    if max_shards is not None:
        filenames = filenames[:max_shards]
    schema = pa.schema(
        [
            pa.field("cell_body_label", pa.int64(), nullable=False),
            pa.field("x", pa.int64(), nullable=False),
            pa.field("y", pa.int64(), nullable=False),
            pa.field("z", pa.int64(), nullable=False),
            pa.field("segment_id", pa.int64(), nullable=False),
        ]
    )
    parts_dir = output.parent / f"{output.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in filenames if not (parts_dir / f"{name}.parquet").exists()]
    spawn = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=spawn,
        max_tasks_per_child=1,
    ) as pool:
        futures = {pool.submit(_map_soma_mesh_shard, name): name for name in missing}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="map H01 somas",
        ):
            name = futures[future]
            part = pa.Table.from_pylist(future.result(), schema=schema)
            pq.write_table(
                part,
                parts_dir / f"{name}.parquet",
                compression="zstd",
            )
    table = pa.concat_tables(
        [pq.read_table(parts_dir / f"{name}.parquet", memory_map=True) for name in filenames]
    )
    table = table.filter(pa.compute.not_equal(table["segment_id"], 0))
    table = table.sort_by([("segment_id", "ascending")])
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression="zstd")
    return table


def _sample_layer_labels(
    soma_table: pa.Table,
    layer_volume: np.ndarray,
    layer_resolution_nm: np.ndarray,
) -> np.ndarray:
    """Sample an official downsampled cortical-layer volume at soma centroids."""

    coordinates = np.column_stack(
        [
            soma_table[axis].combine_chunks().to_numpy(zero_copy_only=False)
            for axis in ("x", "y", "z")
        ]
    )
    voxels = np.floor(
        coordinates * C3_RESOLUTION_NM[None, :] / layer_resolution_nm[None, :]
    ).astype(np.int64)
    volume = np.asarray(layer_volume)
    if volume.ndim == 4:
        volume = volume[..., 0]
    in_bounds = np.all(voxels >= 0, axis=1) & np.all(voxels < np.asarray(volume.shape), axis=1)
    labels = np.zeros(len(voxels), dtype=np.int8)
    selected = voxels[in_bounds]
    labels[in_bounds] = volume[selected[:, 0], selected[:, 1], selected[:, 2]].astype(
        np.int8,
        copy=False,
    )
    return labels


def annotate_h01_layers(
    soma_map: Path,
    *,
    mip: int = 3,
    workers: int = 8,
) -> pa.Table:
    """Atomically add official cortical-layer labels to a completed soma map."""

    from cloudvolume import CloudVolume

    soma_map = Path(soma_map)
    table = pq.read_table(soma_map, memory_map=True)
    if "cortical_layer" in table.column_names:
        return table
    volume = CloudVolume(
        LAYERS_URL,
        mip=mip,
        progress=True,
        cache=False,
        parallel=workers,
        fill_missing=True,
    )
    layer_data = np.asarray(volume[:])
    resolution = np.asarray(volume.resolution, dtype=np.int64)
    labels = _sample_layer_labels(table, layer_data, resolution)
    table = table.append_column(
        pa.field("cortical_layer", pa.int8(), nullable=False),
        pa.array(labels, type=pa.int8()),
    )
    metadata = dict(table.schema.metadata or {})
    metadata[b"cortical_layer_source"] = LAYERS_URL.encode()
    metadata[b"cortical_layer_mip"] = str(mip).encode()
    metadata[b"cortical_layer_resolution_nm"] = json.dumps(resolution.tolist()).encode()
    table = table.replace_schema_metadata(metadata)
    temporary = soma_map.with_suffix(soma_map.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(soma_map)
    return table


_ALLOWED_H01_SEGMENTS: frozenset[int] = frozenset()


def _init_h01_worker(segment_ids: tuple[int, ...]) -> None:
    global _ALLOWED_H01_SEGMENTS
    _ALLOWED_H01_SEGMENTS = frozenset(segment_ids)


def _reduce_h01_shard(
    item: dict,
    scratch_dir: str,
) -> tuple[Counter[tuple[int, int]], dict]:
    name = item["name"]
    size = int(item["size"])
    url = f"https://storage.googleapis.com/h01-release/{name}"
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    temporary = scratch / Path(name).name
    pair_counts: Counter[tuple[int, int]] = Counter()
    records = valid = 0
    try:
        resumable_download(url, temporary, expected_size=size, show_progress=False)
        with temporary.open("rb") as stream:
            for row in avro_reader(stream):
                records += 1
                pre = row.get("pre_synaptic_site") or {}
                post = row.get("post_synaptic_partner") or {}
                src = pre.get("neuron_id")
                dst = post.get("neuron_id")
                if (
                    src is None
                    or dst is None
                    or src == dst
                    or src not in _ALLOWED_H01_SEGMENTS
                    or dst not in _ALLOWED_H01_SEGMENTS
                ):
                    continue
                valid += 1
                pair_counts[(int(src), int(dst))] += 1
    finally:
        if temporary.exists():
            temporary.unlink()
        partial = temporary.with_suffix(temporary.suffix + ".part")
        if partial.exists():
            partial.unlink()
    return pair_counts, {
        "object": name,
        "records": records,
        "retained_synapses": valid,
        "retained_pairs": len(pair_counts),
    }


def _read_h01_inventory(raw_dir: Path) -> list[dict]:
    inventory_path = Path(raw_dir) / "inventory.json"
    if not inventory_path.exists():
        inventory_h01(raw_dir)
    return json.loads(inventory_path.read_text())


def _h01_avro_inventory(raw_dir: Path) -> list[dict]:
    """Select Avro objects and exclude parallel JSON exports."""

    return h01_avro_objects(_read_h01_inventory(raw_dir))


def _h01_part_paths(parts_dir: Path, item: dict) -> tuple[Path, Path]:
    stem = Path(item["name"]).name
    return parts_dir / f"{stem}.parquet", parts_dir / f"{stem}.json"


def _write_h01_part(
    part_path: Path,
    stats_path: Path,
    counts: Counter[tuple[int, int]],
    stats: dict,
) -> None:
    pairs = list(counts.items())
    table = pa.table(
        {
            "pre": pa.array((pair[0][0] for pair in pairs), type=pa.int64()),
            "post": pa.array((pair[0][1] for pair in pairs), type=pa.int64()),
            "weight": pa.array((pair[1] for pair in pairs), type=pa.int64()),
        }
    )
    temporary_part = part_path.with_suffix(part_path.suffix + ".tmp")
    pq.write_table(table, temporary_part, compression="zstd")
    temporary_part.replace(part_path)
    temporary_stats = stats_path.with_suffix(stats_path.suffix + ".tmp")
    temporary_stats.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    temporary_stats.replace(stats_path)


def _read_h01_part(part_path: Path, stats_path: Path) -> tuple[Counter, dict]:
    table = pq.read_table(part_path, memory_map=True)
    pre = table["pre"].to_numpy(zero_copy_only=False)
    post = table["post"].to_numpy(zero_copy_only=False)
    weight = table["weight"].to_numpy(zero_copy_only=False)
    counts = Counter(
        {
            (int(src), int(dst)): int(value)
            for src, dst, value in zip(pre, post, weight, strict=True)
        }
    )
    return counts, json.loads(stats_path.read_text())


def _h01_part_complete(parts_dir: Path, item: dict) -> bool:
    part_path, stats_path = _h01_part_paths(parts_dir, item)
    return part_path.exists() and stats_path.exists()


def build_h01_streaming(
    raw_dir: Path,
    soma_map: Path,
    output_dir: Path,
    *,
    scratch_dir: Path,
    workers: int = 8,
    minimum_pair_weight: int = 1,
    max_shards: int | None = None,
) -> GraphStore:
    """Build H01-KG without persisting the 30.6 GiB binary Avro export."""

    config = dataset_config("h01")
    soma_table = pq.read_table(soma_map, memory_map=True)
    if "cortical_layer" not in soma_table.column_names:
        raise ValueError("soma map lacks cortical_layer; run cq annotate-h01-layers")

    segment_ids = tuple(
        int(value)
        for value in np.unique(
            soma_table["segment_id"].combine_chunks().to_numpy(zero_copy_only=False)
        )
        if value != 0
    )
    inventory = _h01_avro_inventory(raw_dir)
    if max_shards is not None:
        inventory = inventory[:max_shards]
    output_dir = Path(output_dir)
    parts_dir = output_dir.parent / f"{output_dir.name}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    missing = [item for item in inventory if not _h01_part_complete(parts_dir, item)]
    spawn = get_context("spawn")
    batch_size = workers * 2
    with (
        ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_h01_worker,
            initargs=(segment_ids,),
            mp_context=spawn,
        ) as pool,
        tqdm(total=len(missing), desc="reduce H01 Avro") as progress,
    ):
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            futures = {
                pool.submit(_reduce_h01_shard, item, str(scratch_dir)): item for item in batch
            }
            for future in as_completed(futures):
                item = futures[future]
                counts, stats = future.result()
                _write_h01_part(*_h01_part_paths(parts_dir, item), counts, stats)
                progress.update()
    total_counts: Counter[tuple[int, int]] = Counter()
    shard_stats: list[dict] = []
    for item in inventory:
        counts, stats = _read_h01_part(*_h01_part_paths(parts_dir, item))
        total_counts.update(counts)
        shard_stats.append(stats)
    retained = [
        (src, dst, weight)
        for (src, dst), weight in total_counts.items()
        if weight >= minimum_pair_weight
    ]
    if not retained:
        raise ValueError("H01 streaming reduction produced no soma-to-soma edges")
    connected_ids = np.unique(
        np.asarray(
            [value for src, dst, _ in retained for value in (src, dst)],
            dtype=np.int64,
        )
    )
    nodes = make_neuron_nodes(connected_ids, "h01")
    soma_by_segment: dict[int, dict] = {}
    layer_by_segment: dict[int, int] = {}
    for row in soma_table.to_pylist():
        segment_id = int(row["segment_id"])
        layer = int(row["cortical_layer"])
        soma_by_segment.setdefault(
            segment_id,
            {
                "soma_voxel": [row["x"], row["y"], row["z"]],
                "cell_body_label": row["cell_body_label"],
                "cortical_layer": LAYER_LABELS.get(layer),
            },
        )
        if layer in LAYER_LABELS:
            layer_by_segment.setdefault(segment_id, layer)
    observed_layers = sorted(
        {
            layer_by_segment[int(segment_id)]
            for segment_id in connected_ids
            if int(segment_id) in layer_by_segment
        }
    )
    attrs = [
        json.dumps(soma_by_segment.get(int(segment_id), {}), sort_keys=True)
        for segment_id in connected_ids
    ]
    nodes = nodes.set_column(
        nodes.schema.get_field_index("attrs_json"),
        NODE_SCHEMA.field("attrs_json"),
        pa.array(attrs),
    )
    layer_offset = len(connected_ids)
    layer_to_node = {layer: layer_offset + index for index, layer in enumerate(observed_layers)}
    layer_nodes = pa.Table.from_pylist(
        [
            {
                "node_id": node_id,
                "external_id": f"H01:{LAYER_LABELS[layer]}",
                "node_type": str(NodeType.CORTICAL_LAYER),
                "dataset": "h01",
                "attrs_json": json.dumps(
                    {"label": LAYER_LABELS[layer], "official_segment_id": layer},
                    sort_keys=True,
                ),
            }
            for layer, node_id in layer_to_node.items()
        ],
        schema=NODE_SCHEMA,
    )
    nodes = pa.concat_tables((nodes, layer_nodes))
    layer_edges = pa.Table.from_pylist(
        [
            {
                "src": neuron_id,
                "dst": layer_to_node[layer_by_segment[int(segment_id)]],
                "relation": str(Relation.IN_CORTICAL_LAYER),
                "weight": 1.0,
                "source_record": "H01 official cortical layers 20210601",
            }
            for neuron_id, segment_id in enumerate(connected_ids)
            if int(segment_id) in layer_by_segment
        ],
        schema=EDGE_SCHEMA,
    )
    pre = np.asarray([row[0] for row in retained], dtype=np.int64)
    post = np.asarray([row[1] for row in retained], dtype=np.int64)
    weights = np.asarray([row[2] for row in retained], dtype=np.float32)
    edges = connection_edges(
        connected_ids,
        pre,
        post,
        weights,
        source_record="H01 C3 Avro 20210729",
        materialize_inverse=True,
    )
    edges = pa.concat_tables((edges, layer_edges))
    return GraphStore.create(
        output_dir,
        nodes,
        edges,
        dataset="h01",
        version=config["version"],
        license_name=config["license"],
        metadata={
            "subset": "C3 segments sampled at official cell-body mesh centroids",
            "num_mapped_somas": len(segment_ids),
            "num_directed_wiring_edges": len(retained),
            "minimum_pair_weight": minimum_pair_weight,
            "source_objects": len(inventory),
            "source_bytes": sum(int(item["size"]) for item in inventory),
            "records_scanned": sum(item["records"] for item in shard_stats),
            "synapses_retained": sum(item["retained_synapses"] for item in shard_stats),
            "streamed_raw_deleted": True,
            "reduction_checkpoints": len(inventory),
            "source_inventory_sha256": sha256_file(Path(raw_dir) / "inventory.json"),
            "soma_map_sha256": sha256_file(Path(soma_map)),
            "checkpoint_directory": str(parts_dir),
            "num_cortical_layers": len(observed_layers),
            "num_layer_annotated_neurons": len(layer_edges),
            "cortical_layer_source": LAYERS_URL,
            "cortical_layer_semantics": "official volume sampled at soma centroid",
        },
    )
