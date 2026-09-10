"""Columnar query-set serialization and deterministic generation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from connectomequest.graph import GraphStore
from connectomequest.manifest import sha256_file
from connectomequest.query import Query, QueryType, sample_queries

QUERY_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("query_type", pa.string(), nullable=False),
        pa.field("anchors", pa.list_(pa.int64()), nullable=False),
        pa.field("answers", pa.list_(pa.int64()), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("semantic_relation", pa.string()),
    ]
)


def write_queries(queries: list[Query], output: Path, graph: GraphStore, seed: int) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "query_id": query.query_id,
                "query_type": str(query.query_type),
                "anchors": list(query.anchors),
                "answers": sorted(query.answers),
                "split": query.split,
                "semantic_relation": query.semantic_relation,
            }
            for query in queries
        ],
        schema=QUERY_SCHEMA,
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(output)
    split_counts = Counter(query.split for query in queries)
    type_counts = Counter(str(query.query_type) for query in queries)
    semantic_counts = Counter(
        query.semantic_relation for query in queries if query.semantic_relation is not None
    )
    manifest = {
        "graph_manifest_sha256": sha256_file(graph.manifest_path),
        "query_file_sha256": sha256_file(output),
        "seed": seed,
        "split_policy": (
            "fixed neuron-entity-hash 80/10/10 pools shared by every query type; "
            "neuron anchors are split-disjoint, ontology/type anchors are shared; "
            "exact per-type quotas"
        ),
        "num_queries": len(queries),
        "split_counts": dict(sorted(split_counts.items())),
        "query_type_counts": dict(sorted(type_counts.items())),
        "semantic_relation_counts": dict(sorted(semantic_counts.items())),
        **(
            {
                "semantic_query_protocol": {
                    "query_type": "2pt",
                    "feasibility": "at least one weight-sorted V32 observable proof path",
                }
            }
            if "2pt" in type_counts
            else {}
        ),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_temporary.replace(manifest_path)


def validate_query_snapshot(path: Path, graph: GraphStore) -> dict:
    """Reject a modified query file or one generated from another graph."""

    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"query manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("query_file_sha256") != sha256_file(path):
        raise ValueError("query file SHA-256 does not match its manifest")
    if manifest.get("graph_manifest_sha256") != sha256_file(graph.manifest_path):
        raise ValueError("query snapshot was generated from a different graph manifest")
    return manifest


def read_queries(path: Path) -> list[Query]:
    table = pq.read_table(path, memory_map=True)
    rows = table.to_pylist()
    return [
        Query(
            query_id=row["query_id"],
            query_type=QueryType(row["query_type"]),
            anchors=tuple(row["anchors"]),
            answers=frozenset(row["answers"]),
            split=row["split"],
            semantic_relation=row.get("semantic_relation"),
        )
        for row in rows
    ]


def generate_query_set(
    graph: GraphStore,
    output: Path,
    *,
    count_per_type: int,
    seed: int = 17,
    query_types: tuple[QueryType, ...] | None = None,
) -> list[Query]:
    queries: list[Query] = []
    selected_types = tuple(QueryType) if query_types is None else query_types
    for offset, query_type in enumerate(selected_types):
        queries.extend(
            sample_queries(
                graph,
                query_type,
                count=count_per_type,
                seed=seed,
                sampling_seed=seed + offset,
            )
        )
    write_queries(queries, output, graph, seed)
    return queries
