"""Export canonical snapshots into official graph-reasoning baseline formats."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from connectomequest.graph import GraphStore
from connectomequest.manifest import sha256_file
from connectomequest.query import Query, QueryType
from connectomequest.query_io import read_queries, validate_query_snapshot

ENTITY = "e"
RELATION = "r"
NEGATION = "n"
STRUCTURES = {
    QueryType.ONE_HOP: (ENTITY, (RELATION,)),
    QueryType.TWO_HOP: (ENTITY, (RELATION, RELATION)),
    QueryType.INTERSECTION: ((ENTITY, (RELATION,)), (ENTITY, (RELATION,))),
    QueryType.INTERSECTION_NEGATION: (
        (ENTITY, (RELATION,)),
        (ENTITY, (RELATION, NEGATION)),
    ),
    QueryType.INTERSECTION_TYPE: (
        ((ENTITY, (RELATION,)), (ENTITY, (RELATION,))),
        (RELATION,),
    ),
    # 2pt is exported as the equivalent 2p ∩ inverse(type) "pi" structure.
    QueryType.TWO_HOP_TYPE: (
        (ENTITY, (RELATION, RELATION)),
        (ENTITY, (RELATION,)),
    ),
}


def relation_vocabulary(graph: GraphStore) -> tuple[dict[str, int], dict[str, int]]:
    """Return stable BetaE-style adjacent forward / inverse relation IDs."""

    relations = sorted(graph.relations)
    forward = {relation: 2 * index for index, relation in enumerate(relations)}
    inverse = {relation: 2 * index + 1 for index, relation in enumerate(relations)}
    return forward, inverse


def _query_instance(
    query: Query,
    forward: dict[str, int],
    inverse: dict[str, int],
) -> tuple[Any, tuple]:
    wiring = forward["presynaptic_to"]
    structure = STRUCTURES[query.query_type]
    if query.query_type == QueryType.ONE_HOP:
        instance = (query.anchors[0], (wiring,))
    elif query.query_type == QueryType.TWO_HOP:
        instance = (query.anchors[0], (wiring, wiring))
    elif query.query_type == QueryType.INTERSECTION:
        instance = (
            (query.anchors[0], (wiring,)),
            (query.anchors[1], (wiring,)),
        )
    elif query.query_type == QueryType.INTERSECTION_NEGATION:
        instance = (
            (query.anchors[0], (wiring,)),
            (query.anchors[1], (wiring, -2)),
        )
    elif query.query_type == QueryType.INTERSECTION_TYPE:
        semantic = query.semantic_relation or "has_type"
        instance = (
            (
                (query.anchors[0], (wiring,)),
                (query.anchors[1], (wiring,)),
            ),
            (forward[semantic],),
        )
    elif query.query_type == QueryType.TWO_HOP_TYPE:
        semantic = query.semantic_relation or "has_type"
        instance = (
            (query.anchors[0], (wiring, wiring)),
            (query.anchors[1], (inverse[semantic],)),
        )
    else:
        raise ValueError(query.query_type)
    return instance, structure


def _atomic_pickle(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _write_betae_queries(
    queries: list[Query],
    output: Path,
    forward: dict[str, int],
    inverse: dict[str, int],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    index_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        selected = [query for query in queries if query.split == split]
        grouped: defaultdict[tuple, set] = defaultdict(set)
        answers: dict[tuple, set[int]] = {}
        for query in selected:
            instance, structure = _query_instance(query, forward, inverse)
            grouped[structure].add(instance)
            answers[instance] = set(query.answers)
            index_rows.append(
                {
                    "query_id": query.query_id,
                    "split": split,
                    "query_type": str(query.query_type),
                    "baseline_structure": repr(structure),
                    "baseline_query": repr(instance),
                }
            )
        baseline_split = "valid" if split == "validation" else split
        _atomic_pickle(dict(grouped), output / f"{baseline_split}-queries.pkl")
        if split == "train":
            _atomic_pickle(answers, output / "train-answers.pkl")
        else:
            _atomic_pickle(answers, output / f"{baseline_split}-hard-answers.pkl")
            _atomic_pickle(
                {instance: set() for instance in answers},
                output / f"{baseline_split}-easy-answers.pkl",
            )
        counts[baseline_split] = len(selected)
    (output / "query-index.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    return counts


def _write_triples(
    graph: GraphStore,
    queries: list[Query],
    output: Path,
    forward: dict[str, int],
    inverse: dict[str, int],
    *,
    include_inverse: bool,
) -> int:
    train_path = output / "train.txt"
    temporary = train_path.with_suffix(".txt.tmp")
    count = 0
    parquet = pq.ParquetFile(graph.edges_path)
    with temporary.open("w", encoding="utf-8", buffering=1024 * 1024) as stream:
        for batch in parquet.iter_batches(
            batch_size=262_144,
            columns=["src", "dst", "relation"],
        ):
            rows = batch.to_pydict()
            for src, dst, relation in zip(
                rows["src"],
                rows["dst"],
                rows["relation"],
                strict=True,
            ):
                stream.write(f"{src}\t{forward[relation]}\t{dst}\n")
                count += 1
                if include_inverse:
                    stream.write(f"{dst}\t{inverse[relation]}\t{src}\n")
                    count += 1
    temporary.replace(train_path)
    graph_path = output / "graph.txt"
    graph_path.unlink(missing_ok=True)
    try:
        os.link(train_path, graph_path)
    except OSError:
        shutil.copyfile(train_path, graph_path)

    for split, name in (("validation", "valid"), ("test", "test")):
        with (output / f"{name}.txt").open("w", encoding="utf-8") as stream:
            for query in queries:
                if query.split != split or query.query_type != QueryType.ONE_HOP:
                    continue
                for answer in sorted(query.answers):
                    stream.write(f"{query.anchors[0]}\t{forward['presynaptic_to']}\t{answer}\n")
    return count


def export_baseline_bundle(
    graph: GraphStore,
    query_path: Path,
    output: Path,
    *,
    include_inverse: bool = True,
) -> dict[str, Any]:
    """Write one immutable bundle consumable by all pinned baseline families."""

    if not include_inverse:
        raise ValueError("baseline bundles require adjacent inverse relations")
    validate_query_snapshot(query_path, graph)
    queries = read_queries(query_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    forward, inverse = relation_vocabulary(graph)
    relation_map = {
        **forward,
        **(
            {f"inverse:{name}": value for name, value in inverse.items()} if include_inverse else {}
        ),
    }
    (output / "relations.json").write_text(
        json.dumps(relation_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    id2relation = {value: name for name, value in relation_map.items()}
    _atomic_pickle(id2relation, output / "id2rel.pkl")
    nodes = graph.nodes(columns=["node_id", "external_id", "node_type"])
    id2entity = {
        int(row["node_id"]): str(row["external_id"])
        for row in nodes.select(["node_id", "external_id"]).to_pylist()
    }
    _atomic_pickle(id2entity, output / "id2ent.pkl")
    (output / "entities.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in nodes.to_pylist()),
        encoding="utf-8",
    )
    triple_count = _write_triples(
        graph,
        queries,
        output,
        forward,
        inverse,
        include_inverse=include_inverse,
    )
    query_counts = _write_betae_queries(queries, output, forward, inverse)
    num_relations = len(forward) * (2 if include_inverse else 1)
    (output / "stats.txt").write_text(
        f"numentity: {graph.num_nodes}\nnumrelations: {num_relations}\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "dataset": graph.manifest.dataset,
        "dataset_version": graph.manifest.version,
        "graph_manifest_sha256": sha256_file(graph.manifest_path),
        "query_sha256": sha256_file(Path(query_path)),
        "num_entities": graph.num_nodes,
        "num_relations": num_relations,
        "num_triples": triple_count,
        "query_counts": query_counts,
        "include_inverse": include_inverse,
        "protocol": "full_graph_complex_query_answering",
        "formats": {
            "betae_pickle": ["Query2Box", "BetaE", "ConE", "GNN-QE", "UltraQuery"],
            "graph_message_passing": ["GNN-QE", "UltraQuery"],
        },
        "semantic_rewrite": {
            "2pt": "2p(anchor) intersect inverse(semantic_relation)(type_anchor)",
        },
    }
    manifest_path = output / "bundle-manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def edge_split(
    src: int,
    relation: str,
    dst: int,
    *,
    seed: int = 17,
) -> str:
    """Assign a canonical edge to a stable 80/10/10 holdout."""

    payload = f"{seed}:{src}:{relation}:{dst}".encode()
    bucket = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % 10
    if bucket < 8:
        return "train"
    return "valid" if bucket == 8 else "test"


def export_link_prediction_bundle(
    graph: GraphStore,
    output: Path,
    *,
    seed: int = 17,
) -> dict[str, Any]:
    """Export a leakage-safe 1p bundle for path-search/link-prediction models."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    forward, inverse = relation_vocabulary(graph)
    relation_map = {
        **forward,
        **{f"inverse:{name}": value for name, value in inverse.items()},
    }
    (output / "relations.json").write_text(
        json.dumps(relation_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _atomic_pickle({value: name for name, value in relation_map.items()}, output / "id2rel.pkl")
    nodes = graph.nodes(columns=["node_id", "external_id", "node_type"])
    _atomic_pickle(
        {
            int(row["node_id"]): str(row["external_id"])
            for row in nodes.select(["node_id", "external_id"]).to_pylist()
        },
        output / "id2ent.pkl",
    )
    (output / "entities.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in nodes.to_pylist()),
        encoding="utf-8",
    )

    temporary = {split: output / f"{split}.txt.tmp" for split in ("train", "valid", "test")}
    counts = {split: 0 for split in temporary}
    with ExitStack() as stack:
        writers = {
            split: stack.enter_context(path.open("w", encoding="utf-8", buffering=1024 * 1024))
            for split, path in temporary.items()
        }
        parquet = pq.ParquetFile(graph.edges_path)
        for batch in parquet.iter_batches(batch_size=262_144, columns=["src", "dst", "relation"]):
            rows = batch.to_pydict()
            for src, dst, relation in zip(rows["src"], rows["dst"], rows["relation"], strict=True):
                split = edge_split(src, relation, dst, seed=seed)
                writers[split].write(f"{src}\t{forward[relation]}\t{dst}\n")
                counts[split] += 1
                if split == "train":
                    writers[split].write(f"{dst}\t{inverse[relation]}\t{src}\n")
    for split, path in temporary.items():
        path.replace(output / f"{split}.txt")
    graph_path = output / "graph.txt"
    graph_path.unlink(missing_ok=True)
    try:
        os.link(output / "train.txt", graph_path)
    except OSError:
        shutil.copyfile(output / "train.txt", graph_path)

    num_relations = 2 * len(forward)
    (output / "stats.txt").write_text(
        f"numentity: {graph.num_nodes}\nnumrelations: {num_relations}\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "protocol": "deterministic_edge_holdout_link_prediction",
        "query_scope": "1p_only",
        "dataset": graph.manifest.dataset,
        "dataset_version": graph.manifest.version,
        "graph_manifest_sha256": sha256_file(graph.manifest_path),
        "seed": seed,
        "split": "80/10/10_blake2b",
        "num_entities": graph.num_nodes,
        "num_relations": num_relations,
        "canonical_edge_counts": counts,
        "train_triples_with_inverse": 2 * counts["train"],
        "consumers": ["AStarNet", "NBFNet", "MINERVA", "ULTRA-1p"],
        "leakage_control": "forward and inverse forms share one canonical split",
    }
    (output / "bundle-manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
