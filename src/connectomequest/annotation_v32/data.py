"""Versioned annotation acquisition data, with an explicit public/private boundary.

Old annotations are fallible hints. Only separately loaded private labels may
back an oracle. No source/validation snapshot is masqueraded as historical hints.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence
from functools import cached_property

import numpy as np
from scipy import sparse

OTHER = "__OTHER__"
# Explicit nomenclature aliases, never prefix rules. Hulse et al. eLife66039
# identifies PEN_a/PEN1 and PEN_b/PEN2; hemibrain stores both names in one string.
TYPE_ALIASES = {"PEN_a(PEN1)": "PEN_a", "PEN_b(PEN2)": "PEN_b"}

# Versioned forward groups, approved before outcomes. These are broad declared
# anatomical categories, not claims that fine cell types are interchangeable.
COMMON_GROUP_MAPPINGS = {
    "hemibrain:v1.2.1": {
        "PAM01_a":"PAM01","PAM01_b":"PAM01","PAM04_a":"PAM04","PAM04_b":"PAM04",
        "KCab-m":"group:KCab-mcs","KCab-c":"group:KCab-mcs","KCab-s":"group:KCab-mcs",
        "LHAV7b1":"group:LHAV7b1","LHAV2a3":"group:LHAV2a3",
    },
    "male-cns:v1.0": {
        "KCab-m":"group:KCab-mcs","KCab-c":"group:KCab-mcs","KCab-s":"group:KCab-mcs",
        "LHAV7b1":"group:LHAV7b1","LHAV2a3":"group:LHAV2a3",
    },
    "flywire:v2.1.0": {
        "PAM01_a":"PAM01","PAM01_b":"PAM01","PAM04_a":"PAM04","PAM04_b":"PAM04",
        "KCab":"group:KCab-mcs","KCab-m":"group:KCab-mcs","KCab-c":"group:KCab-mcs","KCab-s":"group:KCab-mcs",
        "LHAV7b1":"group:LHAV7b1","CB1640":"group:LHAV7b1",
        "LHAV2a3":"group:LHAV2a3","LHAV2a3a":"group:LHAV2a3","LHAV2a3b":"group:LHAV2a3","LHAV2a3c":"group:LHAV2a3",
    },
    "flywire:v3.1.0": {
        "KCab":"group:KCab-mcs","LHAV7b1":"group:LHAV7b1","CB1640":"group:LHAV7b1",
        "LHAV2a3a":"group:LHAV2a3","LHAV2a3b":"group:LHAV2a3","LHAV2a3c":"group:LHAV2a3",
    },
}
ONTOLOGY_SCHEMA = "cq-v32-author-coarsening-v1"


def ontology_manifest():
    payload={"schema":ONTOLOGY_SCHEMA,"per_release":COMMON_GROUP_MAPPINGS,"aliases":TYPE_ALIASES,
             "unsupported_intents":{"olfactory-3":"LHPD2a2 ambiguous crosswalk","olfactory-4":"LHPD2a2 ambiguous crosswalk"},
             "provenance":{"hemibrain_file5_sha256":"c9ca8421a5b6a5382348e06a7f62ef452bc78562983deff9ff479e30be2bd2a2",
             "hemibrain_file5_url":"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/8587524c1748ce5ef2080822a2fc890fc03bf597/supplemental_files/Supplemental_file5_hemibrain_meta.csv",
             "flywire_file1_sha256":"9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be",
             "malecns_annotations_sha256":"2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2"}}
    payload["sha256"]=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return payload


def ontology_type(name, release):
    name=canonical_type(name)
    return COMMON_GROUP_MAPPINGS.get(release,{}).get(name,name)


def canonical_type(name):
    if name is None or name == "":
        return None
    return TYPE_ALIASES.get(str(name), str(name))


def freeze_vocabulary(source_names: Sequence[str | None], *, release="hemibrain:v1.2.1") -> tuple[str, ...]:
    return (OTHER, *sorted({ontology_type(x,release) for x in source_names if ontology_type(x,release)} - {OTHER}))


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class PublicAnnotationGraph:
    node_ids: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    provisional: Mapping[int, int | None]
    vocabulary: tuple[str, ...]
    metadata: Mapping = field(default_factory=dict)
    old_label_names: Mapping[int, str | None] = field(default_factory=dict)

    def __post_init__(self):
        ids = np.asarray(self.node_ids)
        if ids.dtype.kind not in "iu" or ids.ndim != 1 or np.any(ids <= 0) or np.any(ids[1:] <= ids[:-1]):
            raise ValueError("node IDs must be positive sorted unique integers")
        n = len(ids)
        if self.vocabulary[:1] != (OTHER,) or len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("vocabulary must be unique, with OTHER at code zero")
        if len(self.indptr) != n + 1 or self.indptr[0] != 0 or self.indptr[-1] != len(self.indices):
            raise ValueError("invalid CSR indptr")
        if np.any(np.diff(self.indptr) < 0) or len(self.weights) != len(self.indices):
            raise ValueError("invalid CSR lengths")
        if np.any(self.indices < 0) or np.any(self.indices >= n) or np.any(self.weights <= 0):
            raise ValueError("invalid CSR edge")
        if set(self.provisional) != set(map(int, ids)):
            raise ValueError("provisional hints must cover exactly the public node IDs")
        if any(x is not None and not 0 <= x < len(self.vocabulary) for x in self.provisional.values()):
            raise ValueError("provisional label outside vocabulary")
        for array in (self.node_ids, self.indptr, self.indices, self.weights):
            array.flags.writeable = False

    def neighbors(self, cell: int) -> tuple[int, ...]:
        i = int(np.searchsorted(self.node_ids, cell))
        if i == len(self.node_ids) or int(self.node_ids[i]) != cell:
            raise KeyError(cell)
        return tuple(map(int, self.node_ids[self.indices[self.indptr[i]:self.indptr[i + 1]]]))

    def label_code(self, name: str) -> int:
        name = canonical_type(name)
        try:
            return self.vocabulary.index(name)
        except ValueError as exc:
            raise ValueError(f"type {name!r} absent from source-frozen vocabulary") from exc

    @cached_property
    def _incoming(self):
        n=len(self.node_ids)
        return sparse.csr_matrix((np.ones(len(self.indices),dtype=np.int8),self.indices,self.indptr),shape=(n,n)).tocsc()

    def predecessors(self, cell: int) -> tuple[int, ...]:
        i=int(np.searchsorted(self.node_ids,cell))
        if i==len(self.node_ids) or int(self.node_ids[i])!=cell:
            raise KeyError(cell)
        matrix=self._incoming
        return tuple(map(int,self.node_ids[matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]]))


@dataclass(frozen=True)
class PrivateAnnotations:
    labels: Mapping[int, int | None]
    metadata: Mapping = field(default_factory=dict)


def _codes(names, vocabulary, release=None):
    lookup = {name: i for i, name in enumerate(vocabulary)}
    return {int(cell): None if canonical_type(name) is None else lookup.get(ontology_type(name,release), 0)
            for cell, name in names.items()}


def _from_matrix(ids, matrix, hints, vocabulary, metadata=None):
    matrix = matrix.tocsr()
    matrix.sum_duplicates()
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return PublicAnnotationGraph(np.asarray(ids, dtype=np.int64), matrix.indptr.astype(np.int64),
                                 matrix.indices.astype(np.int32), matrix.data.astype(np.int64),
                                 _codes(hints, vocabulary,(metadata or {}).get("hint_release")), tuple(vocabulary), metadata or {}, dict(hints))


def build_public_graph(node_ids, edges, provisional_names, vocabulary, *, min_synapses=5, metadata=None):
    ids = np.asarray(sorted(node_ids), dtype=np.int64)
    if len(np.unique(ids)) != len(ids) or np.any(ids <= 0):
        raise ValueError("node IDs must be unique positive integers")
    if min_synapses < 1:
        raise ValueError("min_synapses must be positive")
    values = np.fromiter(edges, dtype=np.dtype([("pre","<i8"),("post","<i8"),("weight","<i8")])).view(np.int64).reshape(-1,3)
    if len(values) and np.any(values[:, 2] <= 0):
        raise ValueError("weights must be positive")
    if len(values) and not len(ids):
        raise ValueError("cannot attach edges to an empty node universe")
    if len(values):
        pre, post = np.searchsorted(ids, values[:, 0]), np.searchsorted(ids, values[:, 1])
        valid = (pre < len(ids)) & (post < len(ids))
        safe_pre, safe_post = np.minimum(pre, len(ids)-1), np.minimum(post, len(ids)-1)
        valid &= (ids[safe_pre] == values[:, 0]) & (ids[safe_post] == values[:, 1])
        matrix = sparse.coo_matrix((values[valid, 2], (pre[valid], post[valid])), shape=(len(ids), len(ids))).tocsr()
    else:
        matrix = sparse.csr_matrix((len(ids), len(ids)), dtype=np.int64)
    matrix.data[matrix.data < min_synapses] = 0
    matrix.eliminate_zeros()
    hints = {int(i): provisional_names.get(int(i)) for i in ids}
    return _from_matrix(ids, matrix, hints, vocabulary, metadata)


def save_prepared_dataset(path, public, private):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite prepared data: {path}")
    if set(private.labels) != set(map(int, public.node_ids)):
        raise ValueError("oracle and public node universes differ")
    if any(x is not None and not 0 <= x < len(public.vocabulary) for x in private.labels.values()):
        raise ValueError("oracle labels outside frozen vocabulary")
    pub, pri = path / "public", path / "private"
    pub.mkdir(parents=True, exist_ok=True)
    pri.mkdir()
    np.savez_compressed(pub / "topology.npz", node_ids=public.node_ids, indptr=public.indptr,
                        indices=public.indices, weights=public.weights)
    ids = list(map(int, public.node_ids))
    pq.write_table(pa.table({"cell_id": pa.array(ids, type=pa.int64()),
                            "provisional": pa.array([public.provisional[i] for i in ids], type=pa.int32()),
                            "old_label_name": pa.array([public.old_label_names.get(i) for i in ids], type=pa.string())}), pub / "nodes.parquet")
    pq.write_table(pa.table({"cell_id": pa.array(ids, type=pa.int64()),
                            "label": pa.array([private.labels[i] for i in ids], type=pa.int32())}), pri / "labels.parquet")
    (pub / "manifest.json").write_text(json.dumps({"schema": "annotation-v32-public-v1", "vocabulary": public.vocabulary,
        "metadata": dict(public.metadata), "files": {n: hash_file(pub/n) for n in ["nodes.parquet", "topology.npz"]}}, sort_keys=True, indent=2))
    (pri / "manifest.json").write_text(json.dumps({"schema": "annotation-v32-private-v1", "metadata": dict(private.metadata),
        "files": {"labels.parquet": hash_file(pri/"labels.parquet")}}, sort_keys=True, indent=2))
    return path


def _manifest(root):
    m = json.loads((root / "manifest.json").read_text())
    for name, expected in m["files"].items():
        if Path(name).name != name or hash_file(root / name) != expected:
            raise ValueError(f"prepared file hash mismatch: {name}")
    return m


def load_public_graph(path) -> PublicAnnotationGraph:
    import pyarrow.parquet as pq
    root = Path(path) / "public"
    m = _manifest(root)
    if m["schema"] != "annotation-v32-public-v1":
        raise ValueError("unsupported public data schema")
    nodes = pq.read_table(root/"nodes.parquet").to_pylist()
    with np.load(root/"topology.npz", allow_pickle=False) as a:
        return PublicAnnotationGraph(a["node_ids"],a["indptr"],a["indices"],a["weights"],
            {r["cell_id"]:r["provisional"] for r in nodes},tuple(m["vocabulary"]),m["metadata"],
            {r["cell_id"]:r["old_label_name"] for r in nodes})


def load_prepared_dataset(path):
    import pyarrow.parquet as pq
    public = load_public_graph(path)
    root = Path(path) / "private"
    m = _manifest(root)
    if m["schema"] != "annotation-v32-private-v1":
        raise ValueError("unsupported private data schema")
    private = PrivateAnnotations({r["cell_id"]:r["label"] for r in pq.read_table(root/"labels.parquet").to_pylist()},m["metadata"])
    if set(private.labels) != set(public.provisional):
        raise ValueError("private universe differs from public graph")
    return public, private


def _read_annotations(path):
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    keys = [r["supervoxel_id"] for r in rows]
    roots = [r["root_id"] for r in rows]
    if len(set(keys)) != len(keys) or len(set(roots)) != len(roots):
        raise ValueError("annotation supervoxels/root IDs must be unique")
    return {r["supervoxel_id"]:r for r in rows}


def prepare_flywire(old_tsv, new_tsv, connections_gz, output, *, vocabulary, min_synapses=5):
    old, new = _read_annotations(old_tsv), _read_annotations(new_tsv)
    common = sorted(old.keys() & new.keys())
    if any(old[k]["root_id"] != new[k]["root_id"] for k in common):
        raise ValueError("root ID changed across historical snapshots")
    hints = {int(old[k]["root_id"]):old[k]["cell_type"] or None for k in common}
    final = {int(new[k]["root_id"]):new[k]["cell_type"] or None for k in common}
    def edges():
        with gzip.open(connections_gz, "rt", newline="") as stream:
            for row in csv.DictReader(stream):
                a,b = int(row["pre_root_id"]),int(row["post_root_id"])
                if a in hints and b in hints:
                    yield a,b,int(row["syn_count"])
    metadata = {"dataset":"flywire783", "old_version":"v2.1.0", "oracle_version":"v3.1.0", "min_synapses":min_synapses,
                "hint_release":"flywire:v2.1.0","ontology":ontology_manifest(),
                "historical_hints":True, "files":{Path(p).name:hash_file(p) for p in [old_tsv,new_tsv,connections_gz]},
                "identity_rule":"shared supervoxel_id with identical root_id", "aliases":TYPE_ALIASES}
    public = build_public_graph(hints, edges(), hints, vocabulary, min_synapses=min_synapses, metadata=metadata)
    transitions = Counter("null_to_label" if not hints[i] and final[i] else "label_to_null" if hints[i] and not final[i]
        else "revision" if hints[i] and final[i] and hints[i] != final[i] else "same_nonempty" if final[i] else "both_empty" for i in hints)
    grouped_transitions=Counter("null_to_label" if not hints[i] and final[i] else "label_to_null" if hints[i] and not final[i]
        else "revision" if hints[i] and final[i] and ontology_type(hints[i],"flywire:v2.1.0")!=ontology_type(final[i],"flywire:v3.1.0")
        else "same_nonempty" if final[i] else "both_empty" for i in hints)
    private = PrivateAnnotations(_codes(final,vocabulary,"flywire:v3.1.0"), {"oracle_version":"v3.1.0", "transitions":dict(transitions),
        "grouped_transitions":dict(grouped_transitions),"ontology_sha256":ontology_manifest()["sha256"],
        "unseen_final_types":sorted({ontology_type(x,"flywire:v3.1.0") for x in final.values() if x and ontology_type(x,"flywire:v3.1.0") not in vocabulary}),
        "quality":"versioned curator reference, not independently validated biological truth"})
    return save_prepared_dataset(output,public,private)


def prepare_hemibrain(raw_dir, output, *, vocabulary=None, min_synapses=5):
    import pyarrow.parquet as pq
    raw = Path(raw_dir)
    rows = pq.read_table(raw/"neurons.parquet",columns=["bodyId","type"]).to_pylist()
    names = {r["bodyId"]:r["type"] for r in rows}
    vocabulary = tuple(vocabulary) if vocabulary is not None else freeze_vocabulary(list(names.values()))
    def edges():
        for path in sorted((raw/"edges").glob("*.parquet")):
            for r in pq.read_table(path).to_pylist():
                yield r["body_pre"],r["body_post"],r["weight"]
    public = build_public_graph(names,edges(),{},vocabulary,min_synapses=min_synapses,
        metadata={"dataset":"hemibrain:v1.2.1","historical_hints":False,"min_synapses":min_synapses,
                  "annotation_sha256":hash_file(raw/"neurons.parquet"),"aliases":TYPE_ALIASES,"ontology":ontology_manifest()})
    return save_prepared_dataset(output, public, PrivateAnnotations(_codes(names,vocabulary,"hemibrain:v1.2.1"),{"role":"source","ontology_sha256":ontology_manifest()["sha256"]}))


def prepare_malecns(raw_dir, output, *, vocabulary, min_synapses=5, topology_dir=None):
    from connectomequest.adapters.malecns import select_neuronal_annotations
    from connectomequest.config import dataset_config
    raw = Path(raw_dir)
    annotation = raw/"body-annotations-male-cns-v1.0-minconf-0.5.feather"
    table,_ = select_neuronal_annotations(annotation,allowed_superclasses=tuple(dataset_config("malecns")["allowed_superclasses"]))
    names = {r["bodyId"]:r["type"] for r in table.select(["bodyId","type"]).to_pylist()}
    metadata={"dataset":"male-cns:v1.0","historical_hints":False,"min_synapses":min_synapses,
              "annotation_sha256":hash_file(annotation),"independent_of_historical_manc":False,"aliases":TYPE_ALIASES,"ontology":ontology_manifest()}
    if topology_dir is not None:
        from connectomequest.transfer.topology import load_topology
        topology=load_topology(Path(topology_dir))
        metadata["derived_topology_manifest_sha256"]=hash_file(Path(topology_dir)/"manifest.json")
        if set(map(int,topology.node_ids)) != set(names):
            raise ValueError("existing topology must have the full neuronal annotation universe")
        matrix=topology.matrix.copy()
        matrix.data[matrix.data<min_synapses]=0
        public=_from_matrix(topology.node_ids,matrix,{i:None for i in names},vocabulary,metadata)
    else:
        from connectomequest.adapters.flyem import iter_connection_batches
        batches=list(iter_connection_batches(raw/"connectome-weights-male-cns-v1.0-minconf-0.5.feather",np.asarray(sorted(names)),minimum_weight=1))
        ids=np.asarray(sorted(names),dtype=np.int64)
        pre=np.concatenate([np.searchsorted(ids,x[0]) for x in batches]);post=np.concatenate([np.searchsorted(ids,x[1]) for x in batches])
        weights=np.concatenate([x[2] for x in batches]);matrix=sparse.coo_matrix((weights,(pre,post)),shape=(len(ids),len(ids))).tocsr()
        matrix.data[matrix.data<min_synapses]=0
        public=_from_matrix(ids,matrix,{i:None for i in names},vocabulary,metadata)
    return save_prepared_dataset(output,public,PrivateAnnotations(_codes(names,vocabulary,"male-cns:v1.0"),{"role":"validation","ontology_sha256":ontology_manifest()["sha256"]}))
