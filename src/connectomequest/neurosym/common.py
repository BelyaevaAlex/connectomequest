"""Shared data for the neurosymbolic pilots (development split only).

Everything that more than one pilot needs is computed once and cached under
``outputs/neurosym/cache``: the graphs of the hemibrain and of the two evaluation
connectomes (pre x post synapse counts), the reference hemibrain type of every
neuron under the declared crosswalk, release-specific type names, sides and
cross-release correspondences, and the hemibrain type-to-type connectivity.

The held-out split is never touched here: ``development_families`` returns the
question families of development types only, and ``halves`` splits those types
into a fitting half and a validation half by a hash of the type name.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "outputs/neurosym/cache"
BENCH = ROOT / "outputs/v35/benchmark"
SOURCE = ROOT / "inputs/v32/prepared-common-v1/hemibrain"
VIEW_ROOTS = dict(malecns=ROOT / "inputs/v32/prepared-common-v1/malecns", flywire=ROOT / "inputs/v32/prepared-common-v1-hints/flywire")
RELEASE = dict(malecns="male-cns:v1.0", flywire="flywire:v3.1.0")
ANNOTATIONS = dict(malecns=ROOT / "data/raw/manc/body-annotations-male-cns-v1.0-minconf-0.5.feather", flywire=ROOT / "inputs/v32/flywire/cq-flywire-v3.1.0.tsv")
RHOS = ((1, 4), (1, 2), (3, 4), (1, 1))


def load_graph(root):
    """Node ids and a CSC matrix with ``matrix[pre, post]`` = synapses (self-loops removed)."""
    with np.load(root / "public/topology.npz", allow_pickle=False) as z:
        ids, indptr, indices, weights = z["node_ids"], z["indptr"], z["indices"], z["weights"]
    n = len(ids)
    rows = np.repeat(np.arange(n), np.diff(indptr))
    keep = rows != indices
    matrix = sparse.csc_matrix((weights[keep], (rows[keep], indices[keep])), shape=(n, n))
    matrix.sum_duplicates()
    matrix.sort_indices()
    return ids, matrix


@lru_cache(maxsize=1)
def vocabulary():
    return json.loads((SOURCE / "public/manifest.json").read_text())["vocabulary"]


def _save(path, **arrays):
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")  # per-process name: concurrent builders never share a temp file
    np.savez(tmp, **arrays)
    tmp.replace(path)


def _csc_arrays(prefix, m):
    m = m.tocsc()
    return {prefix + "_indptr": m.indptr, prefix + "_indices": m.indices, prefix + "_data": m.data.astype(np.int64), prefix + "_shape": np.asarray(m.shape)}


def _csc_from(z, prefix):
    return sparse.csc_matrix((z[prefix + "_data"], z[prefix + "_indices"], z[prefix + "_indptr"]), shape=tuple(z[prefix + "_shape"]))


def build_hemibrain_cache():
    import pyarrow.parquet as pq

    ids, matrix = load_graph(SOURCE)
    table = {int(r["cell_id"]): r["label"] for r in pq.read_table(SOURCE / "private/labels.parquet").to_pylist()}
    labels = np.array([-1 if table.get(c) in (None, 0) else table[c] for c in ids.tolist()], dtype=np.int32)
    v = len(vocabulary())
    typed = labels >= 0
    onehot = sparse.csr_matrix((np.ones(int(typed.sum())), (np.flatnonzero(typed), labels[typed])), shape=(len(ids), v))
    by_type = (onehot.T @ matrix.tocsr() @ onehot).tocsc()  # by_type[pre_type, post_type]
    _save(CACHE / "hemibrain.npz", ids=ids, labels=labels, **_csc_arrays("graph", matrix), **_csc_arrays("by_type", by_type))


def build_view_cache(view):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from v35_build import view_labels  # declared crosswalk, checked against the frozen reference

    ids, matrix = load_graph(VIEW_ROOTS[view])
    lookup = {name: i for i, name in enumerate(vocabulary()) if i}
    labels, _ = view_labels(ROOT, view, ids, lookup)
    if view == "flywire":
        with ANNOTATIONS[view].open(newline="") as stream:
            rows = {int(r["root_id"]): r for r in csv.DictReader(stream, delimiter="\t")}
        get = lambda r, k: (r or {}).get(k) or None
        release = [get(rows.get(c), "cell_type") for c in ids.tolist()]
        declared = [get(rows.get(c), "hemibrain_type") for c in ids.tolist()]
        side = [get(rows.get(c), "side") for c in ids.tolist()]
        other = [None] * len(ids)
        nt = [get(rows.get(c), "top_nt") for c in ids.tolist()]
        group = [None] * len(ids)
    else:
        import pyarrow.feather as feather

        cols = ["bodyId", "type", "hemibrainType", "somaSide", "flywireType", "group", "class"]
        table = {int(r["bodyId"]): r for r in feather.read_table(ANNOTATIONS[view], columns=cols).to_pylist()}
        get = lambda r, k: (r or {}).get(k) or None
        release = [get(table.get(c), "type") for c in ids.tolist()]
        declared = [get(table.get(c), "hemibrainType") for c in ids.tolist()]
        side = [get(table.get(c), "somaSide") for c in ids.tolist()]
        other = [get(table.get(c), "flywireType") for c in ids.tolist()]
        nt = [None] * len(ids)
        group = [None if get(table.get(c), "group") is None else str(get(table.get(c), "group")) for c in ids.tolist()]
    strings = dict(release=release, declared=declared, side=side, other_release=other, top_nt=nt, group=group)
    codes, names = {}, {}
    for key, values in strings.items():
        uniq = sorted({x for x in values if x is not None})
        index = {x: i for i, x in enumerate(uniq)}
        codes[key] = np.array([-1 if x is None else index[x] for x in values], dtype=np.int32)
        names[key] = uniq
    _save(CACHE / f"{view}.npz", ids=ids, labels=labels, **_csc_arrays("graph", matrix), **{k + "_code": v for k, v in codes.items()})
    (CACHE / f"{view}-names.json").write_text(json.dumps(names))


@lru_cache(maxsize=None)
def hemibrain():
    """ids, graph (CSC pre x post), labels (hemibrain codes, -1 untyped), by_type (CSC pre_type x post_type)."""
    if not (CACHE / "hemibrain.npz").exists():
        build_hemibrain_cache()
    with np.load(CACHE / "hemibrain.npz") as z:
        return dict(ids=z["ids"], graph=_csc_from(z, "graph"), labels=z["labels"], by_type=_csc_from(z, "by_type"))


@lru_cache(maxsize=None)
def view_data(view):
    """ids, graph (CSC pre x post), labels (reference hemibrain codes, -1 unresolved) and *_code arrays with names."""
    if not (CACHE / f"{view}.npz").exists():
        build_view_cache(view)
    with np.load(CACHE / f"{view}.npz") as z:
        out = {k: z[k] for k in z.files if not k.startswith("graph_")}
        out["graph"] = _csc_from(z, "graph")
    out["names"] = json.loads((CACHE / f"{view}-names.json").read_text())
    return out


TOPK = 64


def build_mlp_cache(view):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from v35_build import MLP, VIEWS, full_probabilities, rows_of
    from connectomequest.annotation_v32.training import StructuralMLP

    ids = view_data(view)["ids"] if view != "hemibrain" else hemibrain()["ids"]
    features_dir = ROOT / (VIEWS[view]["features"] if view != "hemibrain" else _hemibrain_features())
    with np.load(features_dir / "features.npz", allow_pickle=False) as z:
        feature_ids, features = z["node_ids"], z["features"]
    features = features[rows_of(feature_ids, ids)]
    model = StructuralMLP.load(ROOT / MLP / "model.npz")
    temperature = json.loads((ROOT / MLP / "report.json").read_text())["temperature"]
    full = full_probabilities(model, temperature, features)
    other = full[:, 0].copy()
    body = full[:, 1:]
    top = np.argpartition(-body, TOPK, axis=1)[:, :TOPK]
    probs = np.take_along_axis(body, top, 1)
    order = np.argsort(-probs, axis=1)
    top, probs = np.take_along_axis(top, order, 1) + 1, np.take_along_axis(probs, order, 1)
    _save(CACHE / f"mlp-{view}.npz", ids=ids, top=top.astype(np.int32), probs=probs.astype(np.float32), other=other.astype(np.float32),
          rest=np.clip(1 - other - probs.sum(1), 0, 1).astype(np.float32))


def _hemibrain_features():
    raise NotImplementedError("hemibrain features are the MLP's training data; not used by the pilots")


@lru_cache(maxsize=None)
def mlp_topk(view):
    """Frozen mlp_s0 per node: top (n, 64) codes, probs (n, 64), other = P(__OTHER__), rest = remaining mass."""
    if not (CACHE / f"mlp-{view}.npz").exists():
        build_mlp_cache(view)
    with np.load(CACHE / f"mlp-{view}.npz") as z:
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=None)
def benchmark(view):
    with np.load(BENCH / f"{view}.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    lengths = np.diff(d["target_offset"])[d["fam_target"]]
    d["fam_plus_offset"] = np.concatenate(([0], np.cumsum(lengths)))
    return d


@lru_cache(maxsize=1)
def manifest():
    return json.loads((BENCH / "manifest.json").read_text())


def type_half(name):
    """0 = fitting half, 1 = validation half of the development types (hash of the name)."""
    return int(sha256(f"neurosym-half:{name}".encode()).hexdigest(), 16) % 2


def split_families(view, split, half=None):
    """Question families of the types of ``split`` (development: optionally of one half)."""
    if split not in ("development", "test") or (half is not None and split != "development"):
        raise ValueError("halves exist only on the development split")
    d = benchmark(view)
    of = {t["code"]: t["split"] for t in manifest()["types"]}
    names = vocabulary()
    types = d["target_type"][d["fam_target"]]
    keep = np.array([of[int(t)] == split and (half is None or type_half(names[int(t)]) == half) for t in types])
    return np.flatnonzero(keep)


def development_families(view, half=None):
    """Question families of development types (optionally of one half)."""
    return split_families(view, "development", half)


def family(view, f):
    """Cells, weights, outcomes and stored priors of one question family."""
    from connectomequest.annotation_v35 import engine as e

    d = benchmark(view)
    t = int(d["fam_target"][f])
    lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
    labels = d["pre_label"][lo:hi]
    codes = d["fam_codes"][d["fam_code_offset"][f] : d["fam_code_offset"][f + 1]]
    outcomes = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
    start = d["fam_plus_offset"][f]
    priors = {}
    for key in d:
        if key.startswith("plus_"):
            name = key[5:]
            plus = np.clip(d[key][start : start + hi - lo].astype(float), 0, 1)
            unknown = np.clip(d["unknown_" + name][lo:hi].astype(float), 0, 1 - plus)
            priors[name] = np.column_stack((plus, 1 - plus - unknown, unknown))
    return dict(family=int(f), target=t, target_cell=int(d["target_cell"][t]), target_type=int(d["target_type"][t]), cells=d["pre_cell"][lo:hi],
                weights=d["pre_weight"][lo:hi], labels=labels, codes=codes, outcomes=outcomes, priors=priors, kind=int(d["fam_kind"][f]),
                source_weight=int(d["fam_source_weight"][f]), source_total=int(d["fam_source_total"][f]))


def thresholds(fam):
    """(rho index, threshold) for the four retention levels, as in the frozen protocol."""
    from connectomequest.annotation_v35 import engine as e

    total = int(fam["weights"].sum())
    out = []
    for i, (num, den) in enumerate(RHOS):
        threshold, _ = e.threshold_goals(total, num * fam["source_weight"], den * fam["source_total"])
        out.append((i, threshold))
    return out
