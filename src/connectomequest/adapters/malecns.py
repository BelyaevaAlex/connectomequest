"""MaleCNS neuronal wiring view; shares raw files with the historical MANC subset."""
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def select_neuronal_annotations(
    path: Path, *, allowed_superclasses: tuple[str, ...],
) -> tuple[pa.Table, dict]:
    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_file(source).read_all()
    if missing := {"bodyId", "superclass", "status"} - set(table.column_names):
        raise ValueError(f"annotations lack columns: {sorted(missing)}")
    if not pa.types.is_integer(table["bodyId"].type) or table["bodyId"].null_count:
        raise ValueError("bodyId must contain non-null integers")
    table = table.sort_by([("bodyId", "ascending")])
    ids = table["bodyId"].to_numpy()
    if np.any(ids <= 0):
        raise ValueError("bodyId must be positive")
    repeats = np.flatnonzero(ids[1:] == ids[:-1]) + 1
    for i in repeats:
        if table.slice(int(i), 1).to_pylist() != table.slice(int(i - 1), 1).to_pylist():
            raise ValueError(f"conflicting annotation for bodyId {ids[i]}")
    unique = np.ones(len(table), dtype=bool)
    unique[repeats] = False
    table = table.filter(pa.array(unique))
    classes = set(pc.unique(table["superclass"]).to_pylist()) - {None}
    if unknown := classes - set(allowed_superclasses):
        raise ValueError(f"unknown neuronal superclass: {sorted(unknown)}")
    superclasses = table["superclass"].to_pylist()
    statuses = table["status"].to_pylist()
    keep = [sc is not None and status in (None, "Traced", "Anchor")
            for sc, status in zip(superclasses, statuses)]
    counts = {
        "annotation_rows": len(ids), "duplicate_annotations": len(repeats),
        "excluded_null_superclass": sum(sc is None for sc in superclasses),
        "excluded_status": dict(Counter(str(status) for sc, status in zip(superclasses, statuses)
                                        if sc is not None and status not in (None, "Traced", "Anchor"))),
        "included_neurons": sum(keep),
    }
    selected = table.filter(pa.array(keep, type=pa.bool_()))
    if not len(selected):
        raise ValueError("neuronal inclusion predicate selected no bodies")
    return selected, counts


def _sources(raw_dir: Path) -> tuple[dict, dict]:
    from connectomequest.config import dataset_config
    from connectomequest.manifest import Manifest, sha256_file
    config = dataset_config("malecns")
    files = dataset_config(config["raw_source_config"])["files"]
    sources = {key: {**spec, "sha256": sha256_file(raw_dir / spec["filename"])}
               for key, spec in files.items()}
    if (raw_dir / "manifest.json").exists():
        manifest = Manifest.read(raw_dir / "manifest.json")
        recorded = {item.path: item.sha256 for item in manifest.artifacts}
        for spec in sources.values():
            if spec["filename"] in recorded and recorded[spec["filename"]] != spec["sha256"]:
                raise ValueError(f"raw manifest hash mismatch: {spec['filename']}")
    return config, sources


def malecns_inventory(raw_dir: Path) -> dict:
    config, sources = _sources(Path(raw_dir))
    annotations, counts = select_neuronal_annotations(
        Path(raw_dir) / sources["annotations"]["filename"],
        allowed_superclasses=tuple(config["allowed_superclasses"]),
    )
    with pa.memory_map(str(Path(raw_dir) / sources["connections"]["filename"]), "r") as stream:
        reader = pa.ipc.open_file(stream)
        source_rows = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
    return {"sources": sources, "counts": counts, "source_rows": source_rows,
            "csr_upper_bound_bytes": source_rows * 8 + (len(annotations) + 1) * 8,
            "view": config["view"], "independent_of_historical_manc": False}


def build_malecns_topology(
    raw_dir: Path, output: Path, *, min_weight: float = 1.0, scratch_dir: Path,
    max_rss_bytes: int = 8 * 2**30, max_output_bytes: int = 8 * 2**30,
) -> Path:
    """Stream source batches into resumable partitions, dedup, and publish one CSR.

    Partitions use compact source index // 4096. Groups of 32 source batches are
    committed atomically, bounding memory and avoiding hundreds of thousands of
    tiny NFS files. Interrupted groups are discarded, completed groups verified.
    """
    import json
    import os
    import resource
    import shutil
    import time

    import pyarrow.parquet as pq
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    from connectomequest.adapters.flyem import iter_connection_batches
    from connectomequest.manifest import sha256_file
    from connectomequest.transfer.artifacts import digest, write_json
    from connectomequest.transfer.topology import Topology, load_topology, save_topology

    raw_dir, output, scratch_dir = map(Path, (raw_dir, output, scratch_dir))
    resolved = [path.resolve() for path in (raw_dir, output, scratch_dir)]
    for index, left in enumerate(resolved):
        for right in resolved[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("raw, output and scratch paths must be disjoint")
    if scratch_dir.is_symlink():
        raise ValueError("scratch must be an owned directory, not a symlink")
    started = time.perf_counter()
    config, sources = _sources(raw_dir)
    identity = digest({"schema": "malecns-neuronal-v1", "sources": sources,
                       "config": config, "min_weight": min_weight})
    if output.exists():
        old = load_topology(output)
        if old.metadata.get("export_identity") != identity:
            raise ValueError("output belongs to a different source/configuration identity")
        return output
    state_path = scratch_dir / "state.json"
    owner = "malecns-export-scratch-v1"
    if scratch_dir.exists():
        if not state_path.is_file():
            if any(scratch_dir.iterdir()):
                raise ValueError("nonempty scratch is not owned by this exporter")
        else:
            previous = json.loads(state_path.read_text())
            if previous.get("schema") != owner or previous.get("identity") != identity:
                raise ValueError("scratch is not owned by this export identity")
            import re
            allowed = re.compile(r"(?:state\.json|indices\.bin|weights\.bin|unique|group-\d{5}(?:\.tmp)?)")
            if any(not allowed.fullmatch(path.name) for path in scratch_dir.iterdir()):
                raise ValueError("scratch contains files not owned by this exporter")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    state = {"schema": owner, "identity": identity, "status": "incomplete"}
    write_json(state_path, state)

    def check_caps():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        disk = sum(p.stat().st_size for p in scratch_dir.rglob("*") if p.is_file())
        if rss > max_rss_bytes or disk > max_output_bytes:
            write_json(state_path, {**state, "status": "resource_limit", "peak_rss_bytes": rss,
                                    "scratch_bytes": disk})
            raise MemoryError("MaleCNS export resource cap exceeded; resumable state retained")

    check_caps()
    annotations, counts = select_neuronal_annotations(
        raw_dir / sources["annotations"]["filename"],
        allowed_superclasses=tuple(config["allowed_superclasses"]),
    )
    ids = annotations["bodyId"].to_numpy()
    counts["overlap_historical_manc"] = (
        sum(v is not None for v in annotations["mancType"].to_pylist())
        if "mancType" in annotations.column_names else 0)
    del annotations
    connection_path = raw_dir / sources["connections"]["filename"]
    group_parts = []
    removed_loops = 0
    groups = []

    def commit_group(group_number, parts, loops):
        final = scratch_dir / f"group-{group_number:05d}"
        if final.exists():
            record = json.loads((final / "complete.json").read_text())
            for name, sha in record["files"].items():
                if sha256_file(final / name) != sha:
                    raise ValueError("corrupt completed export shard")
            return final, record["removed_loops"]
        temporary = scratch_dir / f"group-{group_number:05d}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        paths = {}
        if parts:
            src, dst, weight = (np.concatenate([part[i] for part in parts]) for i in range(3))
            partition = src // 4096
            order = np.argsort(partition, kind="stable")
            boundaries = np.r_[0, np.flatnonzero(np.diff(partition[order])) + 1, len(src)]
            for begin, end in zip(boundaries[:-1], boundaries[1:]):
                take = order[begin:end]
                name = f"part-{int(partition[take[0]]):05d}.parquet"
                pq.write_table(pa.table({"src": src[take], "dst": dst[take], "weight": weight[take]}),
                               temporary / name, compression="zstd")
                paths[name] = sha256_file(temporary / name)
        write_json(temporary / "complete.json", {"files": paths, "removed_loops": loops})
        os.replace(temporary, final)
        return final, loops

    group_loops = 0
    batch_index = -1
    for batch_index, (pre, post, weight) in enumerate(iter_connection_batches(
        connection_path, ids, minimum_weight=min_weight,
    )):
        # Reading remains sequential on resume; completed groups are not re-written.
        keep = pre != post
        group_loops += int((~keep).sum())
        if keep.any() and not (scratch_dir / f"group-{batch_index // 32:05d}").exists():
            group_parts.append((np.searchsorted(ids, pre[keep]).astype(np.int32),
                                np.searchsorted(ids, post[keep]).astype(np.int32),
                                weight[keep]))
        if batch_index % 32 == 31:
            group, loops = commit_group(batch_index // 32, group_parts, group_loops)
            groups.append(group)
            removed_loops += loops
            group_parts, group_loops = [], 0
            check_caps()
    if batch_index % 32 != 31:
        group, loops = commit_group(batch_index // 32, group_parts, group_loops)
        groups.append(group)
        removed_loops += loops
    counts.update(removed_self_loops=removed_loops, duplicate_edges=0)
    degrees = np.zeros(len(ids), dtype=np.int64)
    unique_dir = scratch_dir / "unique"
    unique_dir.mkdir(exist_ok=True)
    histogram = np.zeros(8, dtype=np.int64)
    for partition in range((len(ids) + 4095) // 4096):
        name = f"part-{partition:05d}.parquet"
        tables = [pq.read_table(group / name) for group in groups if (group / name).exists()]
        if not tables:
            continue
        table = pa.concat_tables(tables)
        src, dst, weight = (table[col].to_numpy() for col in ("src", "dst", "weight"))
        order = np.lexsort((dst, src))
        src, dst, weight = src[order], dst[order], weight[order]
        duplicate = np.r_[False, (src[1:] == src[:-1]) & (dst[1:] == dst[:-1])]
        if np.any(duplicate[1:] & (weight[1:] != weight[:-1])):
            raise ValueError("conflicting weights for an aggregate neuronal pair")
        counts["duplicate_edges"] += int(duplicate.sum())
        src, dst, weight = src[~duplicate], dst[~duplicate], weight[~duplicate]
        degrees += np.bincount(src, minlength=len(ids))
        histogram += np.histogram(weight, bins=[0, 1, 2, 4, 8, 16, 32, 64, np.inf])[0]
        pq.write_table(pa.table({"src": src, "dst": dst, "weight": weight}), unique_dir / name,
                       compression="zstd")
        del table, tables, src, dst, weight, order, duplicate
        check_caps()
    indptr = np.r_[0, np.cumsum(degrees)]
    edge_count = int(indptr[-1])
    if not edge_count:
        raise ValueError("neuronal filter produced no edges")
    if edge_count * 8 + indptr.nbytes > max_output_bytes:
        write_json(state_path, {**state, "status": "resource_limit", "reason": "CSR size"})
        raise MemoryError("CSR exceeds output cap")
    indices = np.memmap(scratch_dir / "indices.bin", dtype=np.int32, mode="w+", shape=edge_count)
    weights = np.memmap(scratch_dir / "weights.bin", dtype=np.float32, mode="w+", shape=edge_count)
    for path in sorted(unique_dir.glob("*.parquet")):
        table = pq.read_table(path)
        src = table["src"].to_numpy()
        offset = int(indptr[src[0]])
        indices[offset:offset + len(table)] = table["dst"].to_numpy()
        weights[offset:offset + len(table)] = table["weight"].to_numpy()
    indices.flush()
    weights.flush()
    matrix = sparse.csr_matrix((weights, indices, indptr), shape=(len(ids), len(ids)), copy=False)
    counts.update(edges=edge_count, weak_components=int(connected_components(matrix, directed=True,
                                                                          connection="weak")[0]))
    check_caps()
    if _sources(raw_dir)[1] != sources:
        raise ValueError("source hashes changed during export")
    metadata = {"export_identity": identity, "dataset": config["version"], "view": config["view"],
                "sources": sources, "minimum_weight": min_weight, "counts": counts,
                "weight_histogram_edges": [0, 1, 2, 4, 8, 16, 32, 64, "inf"],
                "weight_histogram_counts": histogram.tolist(),
                "independent_of_historical_manc": False, "remove_self_loops": True,
                "inclusion": {"superclasses": config["allowed_superclasses"],
                              "statuses": ["Traced", None, "Anchor"]}}
    save_topology(Topology(matrix, ids, metadata), output)
    # A separate measurement sidecar keeps topology identity independent of timing.
    write_json(output / "export_measurements.json", {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "output_bytes": sum(p.stat().st_size for p in output.iterdir()),
    })
    load_topology(output)
    # NFS retains unlinked open mmap files as .nfs entries: close before cleanup.
    del matrix
    indices._mmap.close()
    weights._mmap.close()
    shutil.rmtree(scratch_dir)
    return output
