"""Compact Parquet-on-disk and CSR-in-memory graph representation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse

from connectomequest.manifest import Manifest
from connectomequest.schema import EDGE_SCHEMA, NODE_SCHEMA, Relation, ensure_schema


@dataclass(slots=True)
class NeighborBatch:
    node_ids: np.ndarray
    weights: np.ndarray


class GraphStore:
    """A graph store optimized for immutable benchmark snapshots.

    Parquet remains the source of truth. Relation-specific CSR matrices are built
    lazily and can be cached in RAM by CPU workers without copying Arrow buffers.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.nodes_path = self.root / "nodes.parquet"
        self.edges_path = self.root / "edges.parquet"
        self.manifest_path = self.root / "manifest.json"
        for required in (self.nodes_path, self.edges_path, self.manifest_path):
            if not required.exists():
                raise FileNotFoundError(required)
        self.manifest = Manifest.read(self.manifest_path)
        self._relations: dict[str, sparse.csr_matrix] = {}
        self._num_nodes: int | None = None

    @property
    def num_nodes(self) -> int:
        if self._num_nodes is None:
            metadata = pq.read_metadata(self.nodes_path)
            self._num_nodes = metadata.num_rows
        return self._num_nodes

    @property
    def relations(self) -> tuple[str, ...]:
        table = pq.read_table(self.edges_path, columns=["relation"])
        return tuple(sorted(pc.unique(table["relation"]).to_pylist()))

    def nodes(self, columns: list[str] | None = None) -> pa.Table:
        return pq.read_table(self.nodes_path, columns=columns, memory_map=True)

    def edges(self, relation: str | Relation | None = None) -> pa.Table:
        filters = None if relation is None else [("relation", "=", str(relation))]
        return pq.read_table(self.edges_path, filters=filters, memory_map=True)

    def csr(self, relation: str | Relation) -> sparse.csr_matrix:
        key = str(relation)
        if key not in self._relations:
            edges = self.edges(key)
            src = edges["src"].to_numpy(zero_copy_only=False)
            dst = edges["dst"].to_numpy(zero_copy_only=False)
            weight = edges["weight"].to_numpy(zero_copy_only=False)
            matrix = sparse.coo_matrix(
                (weight, (src, dst)), shape=(self.num_nodes, self.num_nodes), dtype=np.float32
            ).tocsr()
            matrix.sum_duplicates()
            matrix.sort_indices()
            self._relations[key] = matrix
        return self._relations[key]

    def neighbors(
        self,
        node_id: int,
        relation: str | Relation,
        *,
        offset: int = 0,
        limit: int | None = None,
        by_weight: bool = True,
    ) -> NeighborBatch:
        matrix = self.csr(relation)
        start, end = matrix.indptr[node_id : node_id + 2]
        ids = matrix.indices[start:end]
        weights = matrix.data[start:end]
        if by_weight and len(ids):
            order = np.lexsort((ids, -weights))
            ids, weights = ids[order], weights[order]
        stop = None if limit is None else offset + limit
        return NeighborBatch(ids[offset:stop].copy(), weights[offset:stop].copy())

    def has_edge(self, src: int, dst: int, relation: str | Relation) -> bool:
        return bool(self.csr(relation)[src, dst] != 0)

    @classmethod
    def create(
        cls,
        root: Path,
        nodes: pa.Table,
        edges: pa.Table,
        *,
        dataset: str,
        version: str,
        license_name: str,
        metadata: dict | None = None,
    ) -> GraphStore:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        nodes = ensure_schema(nodes, NODE_SCHEMA, "nodes")
        edges = ensure_schema(edges, EDGE_SCHEMA, "edges")
        ids = nodes["node_id"].to_numpy(zero_copy_only=False)
        if len(ids) and (
            ids.min() != 0 or ids.max() != len(ids) - 1 or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("node_id must be unique and contiguous in [0, num_nodes)")
        for column in ("src", "dst"):
            values = edges[column].to_numpy(zero_copy_only=False)
            if len(values) and (values.min() < 0 or values.max() >= len(ids)):
                raise ValueError(f"edge {column} contains an unknown node_id")
        nodes_path = root / "nodes.parquet"
        edges_path = root / "edges.parquet"
        nodes_tmp = nodes_path.with_suffix(".parquet.tmp")
        edges_tmp = edges_path.with_suffix(".parquet.tmp")
        pq.write_table(nodes, nodes_tmp, compression="zstd", row_group_size=1_000_000)
        pq.write_table(edges, edges_tmp, compression="zstd", row_group_size=2_000_000)
        nodes_tmp.replace(nodes_path)
        edges_tmp.replace(edges_path)
        manifest = Manifest(
            dataset=dataset,
            version=version,
            license=license_name,
            metadata={
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "relations": sorted(pc.unique(edges["relation"]).to_pylist()),
                **(metadata or {}),
            },
        )
        manifest.add_file(nodes_path, root)
        manifest.add_file(edges_path, root)
        manifest.write(root / "manifest.json")
        return cls(root)


def rows_to_table(rows: Iterable[dict], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=schema)
