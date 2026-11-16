"""Immutable directed neuronal topology, independent of the task graph."""

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from connectomequest.manifest import sha256_file
from connectomequest.transfer.artifacts import canonical, write_json


@dataclass(frozen=True)
class Topology:
    matrix: sparse.csr_matrix
    node_ids: np.ndarray
    metadata: dict

    def __post_init__(self):
        matrix = self.matrix.astype(np.float32).tocsr(copy=True)
        ids = np.asarray(self.node_ids)
        if ids.dtype.kind not in "iu" or ids.ndim != 1 or (ids < 0).any():
            raise ValueError("node IDs must be nonnegative integer identifiers")
        ids = ids.astype(np.uint64, copy=True)
        n = len(ids)
        if not n or matrix.shape != (n, n) or len(np.unique(ids)) != n:
            raise ValueError("topology shape and unique node IDs must agree")
        if not matrix.has_canonical_format or matrix.diagonal().any():
            raise ValueError("topology must be simple, sorted and without self loops")
        if not np.isfinite(matrix.data).all() or (matrix.data <= 0).any():
            raise ValueError("weights must be finite and positive")
        metadata = json.loads(canonical(self.metadata))
        for array in (ids, matrix.data, matrix.indices, matrix.indptr):
            array.flags.writeable = False
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "node_ids", ids)
        object.__setattr__(self, "metadata", metadata)

    @property
    def identity(self) -> str:
        h = hashlib.sha256(canonical(self.metadata))
        for a in (self.node_ids, self.matrix.indptr.astype("<i8"),
                  self.matrix.indices.astype("<i8"), self.matrix.data.astype("<f4")):
            h.update(a.tobytes())
        return h.hexdigest()


def save_topology(topology: Topology, root: Path) -> Path:
    root = Path(root)
    if root.exists():
        if load_topology(root).identity != topology.identity:
            raise ValueError("output already contains a different topology")
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        sparse.save_npz(stage / "matrix.npz", topology.matrix)
        np.save(stage / "node_ids.npy", topology.node_ids, allow_pickle=False)
        write_json(stage / "manifest.json", {
            "schema": "policy-topology-v1", "identity": topology.identity,
            "metadata": topology.metadata, "nodes": len(topology.node_ids),
            "edges": topology.matrix.nnz,
            "files": {n: sha256_file(stage / n) for n in ("matrix.npz", "node_ids.npy")},
        })
        stage.rename(root)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return root


def load_topology(root: Path) -> Topology:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "policy-topology-v1":
        raise ValueError("unsupported topology schema")
    if set(manifest["files"]) != {"matrix.npz", "node_ids.npy"}:
        raise ValueError("invalid topology artifact set")
    for name, expected in manifest["files"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"topology hash mismatch: {name}")
    topology = Topology(sparse.load_npz(root / "matrix.npz"),
                        np.load(root / "node_ids.npy", allow_pickle=False), manifest["metadata"])
    if topology.identity != manifest["identity"]:
        raise ValueError("topology identity hash mismatch")
    return topology


def select_anatomical_topology(full: Topology, annotations: Path, *, class_name: str) -> Topology:
    """Induce an anatomically defined mask; selection has no policy/mapping seed."""
    import pyarrow.feather as feather
    from scipy.sparse.csgraph import breadth_first_order, connected_components

    table = feather.read_table(annotations, columns=['bodyId', 'class'])
    wanted = np.asarray([body for body, label in zip(table['bodyId'].to_pylist(),
                         table['class'].to_pylist()) if label == class_name], dtype=np.uint64)
    indices = np.flatnonzero(np.isin(full.node_ids, wanted))
    if not len(indices):
        raise ValueError('empty anatomical selection')
    matrix = full.matrix[indices, :][:, indices].tocsr()
    incoming = int(full.matrix[:, indices].nnz - matrix.nnz)
    outgoing = int(full.matrix[indices, :].nnz - matrix.nnz)
    boundary = incoming + outgoing
    starts = np.unique(np.linspace(0, len(indices)-1, min(64, len(indices)), dtype=int))
    reached = [len(breadth_first_order(matrix, int(i), directed=True,
                                     return_predecessors=False))-1 for i in starts]
    audit = {
        'nodes': len(indices), 'edges': matrix.nnz, 'boundary_edges': boundary,
        'incoming_boundary_edges': incoming, 'outgoing_boundary_edges': outgoing,
        'external_edge_fraction': boundary / max(1, boundary + matrix.nnz),
        'weak_components': int(connected_components(matrix, connection='weak')[0]),
        'strong_components': int(connected_components(matrix, connection='strong')[0]),
        'isolated_nodes': int(np.sum((matrix.getnnz(0) + matrix.getnnz(1)) == 0)),
        'csr_bytes': matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
        'directed_reachability': {'mean_fraction': float(np.mean(reached) / max(1, len(indices)-1)),
                                  'starts': starts.tolist(), 'exact': len(starts) == len(indices),
                                  'sampling': 'up to 64 equally spaced indices; descriptive'},
    }
    return Topology(matrix, full.node_ids[indices], {
        'parent_identity': full.identity, 'annotation_sha256': sha256_file(annotations),
        'selection': {'field': 'class', 'value': class_name, 'method': 'induced intersection'},
        'anatomical_audit': audit, 'signs': 'unknown; positive synapse counts are not transmitter signs',
    })
