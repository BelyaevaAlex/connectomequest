"""BANC as a third evaluation view (round 5, workstream ``banc``).

BANC is the female brain-and-nerve-cord connectome of Bates, Phelps, Kim, Yang et al.
(bioRxiv 2025, Nature 2026), public release on Harvard Dataverse
(doi:10.7910/DVN/7WTH1N, CC BY 4.0, materialization v888). The view is built by
``scripts/r5_banc_build.py`` with the rules of ``scripts/v35_build.py``:

* graph: neurons of the v888 meta table with a neuronal ``super_class`` (glia,
  ``not_a_neuron``, trachea and segments without a super class are not neurons);
  edges from the v2 neuron-to-neuron edgelist (the synapse set of the BANC paper),
  duplicate records summed, pairs with at least five synapses, self pairs removed;
* crosswalk (declared before any question was built): the neuron's own ``cell_type``
  if it is a hemibrain type, else the author-declared ``hemibrain_cell_type`` if it
  names exactly one hemibrain type, else unresolved. Values of the form ``auto:X``
  are automatic NBLAST top hits that the authors did not review; read literally
  they do not name a hemibrain type and stay unresolved. Type names pass the same
  aliases as the other views and the union of the declared per-release coarsening
  maps (every release map of ``annotation_v32.data`` is a subset of it);
* questions: the 3,277 types, partner sets, hemibrain shares and splits of
  ``outputs/v35/benchmark/manifest.json``; up to two targets per type among BANC
  neurons of the type with at least ten input cells, ordered by the v35 endpoint
  hash with view name ``banc``.

What other workstreams get from here:

* ``manifest()``, ``benchmark()``: the arrays in the layout of
  ``outputs/v35/benchmark/{view}.npz`` (plus ``fam_plus_offset``), checked against the
  manifest hash; priors ``mlp_s0`` (topology MLP, transferred unchanged),
  ``const_half`` and ``const_share`` (no prediction);
* ``family(f)``, ``thresholds(fam)``, ``split_families(split, half=None)``: as in
  ``neurosym.common``;
* ``view_data()``, ``profiles()``, ``rows_of(cells)``, ``cosine(nodes)``,
  ``question_kernel(fam, gamma, tau)``: graph, reference labels and connectivity
  profiles for answer propagation (as ``neurosym.kernel.profiles`` builds them);
* ``frozen(split)``: per-question costs of this workstream's runs (weight order, AG,
  recalibrated AG, ...) in the layout of ``outputs/v35/runs``;
* ``mlp_probabilities(rows)``: the full frozen ``mlp_s0`` distribution of graph rows (stored features);
* ``install_neurosym_view()``: makes ``neurosym.common`` / ``kernel`` / ``harness``
  accept the view name ``"banc"`` at run time (no file is modified);
* ``guard(split, script)``: the round-5 protocol check for held-out runs.
"""

from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy import sparse

from connectomequest.annotation_v32.data import COMMON_GROUP_MAPPINGS, canonical_type
from connectomequest.annotation_v32.runtime import hash_file

ROOT = Path(__file__).resolve().parents[3]
VIEW = "banc"
RELEASE = "banc:v888"
RAW = ROOT / "inputs/round5/banc/raw"
PREPARED = ROOT / "inputs/round5/banc/prepared-v888"
OUT = ROOT / "outputs/round5/banc"
BENCH = OUT / "benchmark.npz"
MANIFEST = OUT / "manifest.json"
VIEWFILE = OUT / "view.npz"
VIEWNAMES = OUT / "view-names.json"
RUNS = OUT / "runs"
RECORD = ROOT / "outputs/round5/record.json"
V35_MANIFEST = ROOT / "outputs/v35/benchmark/manifest.json"
RHOS = ((1, 4), (1, 2), (3, 4), (1, 1))
SPLITS = ("development", "test")
MIN_SYNAPSES = 5

# Dataverse doi:10.7910/DVN/7WTH1N, version 3.0 (file id, md5 published by Dataverse).
RAW_FILES = {
    "banc_888_meta.feather": (14033740, "6275eda42f98c49539d1ab513d979d09"),
    "banc_888_edgelist_simple_v2.feather": (13992792, "394406f8a9bdf093c895f95aff4f6c49"),
    "banc_hemibrain_reviewed_matches.csv.gz": (13994487, "8ab3ee75866fa3efcc91ddbc76d9eb7c"),
}
DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile/{}"

NEURONAL_SUPER_CLASSES = (
    "ascending",
    "ascending_visceral_circulatory",
    "central_brain_intrinsic",
    "descending",
    "motor",
    "optic_lobe_intrinsic",
    "sensory",
    "sensory_ascending",
    "sensory_descending",
    "ventral_nerve_cord_intrinsic",
    "visceral_circulatory",
    "visual_centrifugal",
    "visual_projection",
)


def _union_groups():
    out = {}
    for mapping in COMMON_GROUP_MAPPINGS.values():
        for name, group in mapping.items():
            if out.get(name, group) != group:
                raise ValueError(f"declared coarsening maps disagree on {name}")
            out[name] = group
    return out


GROUPS = _union_groups()


# --------------------------------------------------------------------------- crosswalk


def banc_type(name):
    """Canonical name of a BANC (or declared hemibrain) type string: aliases, then the coarsening map."""
    name = canonical_type(name)
    return None if name is None else GROUPS.get(name, name)


def crosswalk(primary, secondary, lookup):
    """``(code, rule)``: own type if hemibrain, else one declared hemibrain type, else unresolved (-1).

    Same rule as ``v35_build.crosswalk``; ``auto:X`` strings are not hemibrain type names.
    """
    name = banc_type(primary or None)
    if name in lookup:
        return lookup[name], "release-type"
    names = [banc_type(x.strip()) for x in (secondary or "").split(",") if x.strip()]
    if len(names) == 1 and names[0] in lookup:
        return lookup[names[0]], "hemibrain-field"
    return -1, "unresolved"


def endpoint_key(type_name, cell):
    """Pseudo-random order of candidate targets (the v35 endpoint hash, view name ``banc``)."""
    return sha256("|".join(("cq-v35-endpoint", VIEW, str(type_name), str(int(cell)))).encode()).hexdigest()


# --------------------------------------------------------------------------- benchmark arrays


@lru_cache(maxsize=1)
def manifest():
    return json.loads(MANIFEST.read_text())


@lru_cache(maxsize=1)
def vocabulary():
    return json.loads((ROOT / "inputs/v32/prepared-common-v1/hemibrain/public/manifest.json").read_text())["vocabulary"]


@lru_cache(maxsize=1)
def benchmark():
    """Arrays of ``benchmark.npz`` (layout of ``outputs/v35/benchmark/{view}.npz``) plus ``fam_plus_offset``."""
    view = manifest()["view"]
    if hash_file(BENCH) != view["sha256"]:
        raise ValueError("benchmark.npz differs from its manifest")
    with np.load(BENCH, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    lengths = np.diff(d["target_offset"])[d["fam_target"]]
    d["fam_plus_offset"] = np.concatenate(([0], np.cumsum(lengths)))
    return d


def priors():
    return list(manifest()["view"]["priors"])


@lru_cache(maxsize=1)
def split_of():
    """Hemibrain type code -> split, from the v35 manifest (the BANC manifest repeats it)."""
    return {int(t["code"]): t["split"] for t in manifest()["types"]}


def type_half(name):
    """0 = fitting half, 1 = validation half of the development types (as ``neurosym.common.type_half``)."""
    return int(sha256(f"neurosym-half:{name}".encode()).hexdigest(), 16) % 2


def split_families(split, half=None):
    """Question families of the types of ``split`` (development: optionally of one half)."""
    if split not in SPLITS or (half is not None and split != "development"):
        raise ValueError("halves exist only on the development split")
    d, of, names = benchmark(), split_of(), vocabulary()
    types = d["target_type"][d["fam_target"]]
    keep = np.array([of[int(t)] == split and (half is None or type_half(names[int(t)]) == half) for t in types], dtype=bool)
    return np.flatnonzero(keep)


def probabilities(plus, unknown):
    """(member, nonmember, unresolved) as in ``v35_run.probabilities``."""
    plus = np.clip(np.asarray(plus, dtype=float), 0.0, 1.0)
    unknown = np.clip(np.asarray(unknown, dtype=float), 0.0, 1.0 - plus)
    return np.column_stack((plus, 1.0 - plus - unknown, unknown))


def family(f):
    """Cells, weights, outcomes and stored priors of one question family (keys of ``neurosym.common.family``)."""
    from connectomequest.annotation_v35 import engine as e

    d = benchmark()
    t = int(d["fam_target"][f])
    lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
    labels = d["pre_label"][lo:hi]
    codes = d["fam_codes"][d["fam_code_offset"][f] : d["fam_code_offset"][f + 1]]
    outcomes = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
    start = int(d["fam_plus_offset"][f])
    out_priors = {}
    for key in d:
        if key.startswith("plus_"):
            name = key[5:]
            out_priors[name] = probabilities(d[key][start : start + hi - lo], d["unknown_" + name][lo:hi])
    return dict(family=int(f), target=t, target_cell=int(d["target_cell"][t]), target_type=int(d["target_type"][t]),
                cells=d["pre_cell"][lo:hi], weights=d["pre_weight"][lo:hi], labels=labels, codes=codes, outcomes=outcomes,
                priors=out_priors, kind=int(d["fam_kind"][f]), source_weight=int(d["fam_source_weight"][f]),
                source_total=int(d["fam_source_total"][f]))


def thresholds(fam):
    """(rho index, threshold) for the four retention levels, as in the frozen protocol."""
    from connectomequest.annotation_v35 import engine as e

    total = int(np.asarray(fam["weights"]).sum())
    return [(i, e.threshold_goals(total, num * fam["source_weight"], den * fam["source_total"])[0]) for i, (num, den) in enumerate(RHOS)]


# --------------------------------------------------------------------------- graph and propagation profiles


@lru_cache(maxsize=1)
def view_data():
    """ids, graph (CSC pre x post, prepared five-synapse graph), labels (crosswalk codes, -1 unresolved),
    ``*_code`` arrays with ``names`` (release, declared, side, fafb, malecns, super_class, region, top_nt; plus
    ``other_release`` = fafb and an empty ``group`` for the keys of ``neurosym.common.view_data``)."""
    from connectomequest.neurosym.common import load_graph

    ids, graph = load_graph(PREPARED)
    with np.load(VIEWFILE, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    if not np.array_equal(out["ids"], ids):
        raise ValueError("view.npz and the prepared graph disagree on the node universe")
    out["graph"] = graph
    out["names"] = json.loads(VIEWNAMES.read_text())
    # keys of neurosym.common.view_data: the cross-release type is the FlyWire (FAFB) correspondence; BANC has no group
    out["other_release_code"], out["names"]["other_release"] = out["fafb_code"], out["names"]["fafb"]
    out["group_code"], out["names"]["group"] = np.full(len(ids), -1, dtype=np.int32), []
    return out


def rows_of(cells):
    """Node indices of external cell ids (raises if a cell is absent)."""
    ids, cells = view_data()["ids"], np.asarray(cells)
    rows = np.searchsorted(ids, cells)
    if np.any(rows >= len(ids)) or not np.array_equal(ids[np.minimum(rows, len(ids) - 1)], cells):
        raise ValueError("cell absent from the BANC graph")
    return rows


@lru_cache(maxsize=1)
def profiles():
    """CSR (N, 2N): row j = [sqrt outputs of j, sqrt inputs of j], L2-normalized (as ``neurosym.kernel.profiles``)."""
    g = view_data()["graph"].tocsr().astype(np.float64)
    prof = sparse.hstack([g.sqrt(), g.T.tocsr().sqrt()]).tocsr()
    norm = np.sqrt(np.asarray(prof.multiply(prof).sum(1)).ravel())
    prof = sparse.diags(np.where(norm > 0, 1.0 / np.where(norm > 0, norm, 1.0), 0.0)) @ prof
    return prof.tocsr()


def cosine(nodes):
    p = profiles()[np.asarray(nodes)]
    return np.asarray((p @ p.T).todense(), dtype=np.float64)


def question_kernel(fam, gamma, tau):
    """Thresholded powered cosine kernel of the input cells of ``fam`` (as ``neurosym.kernel.question_kernel``)."""
    from connectomequest.neurosym.kernel import kernel

    return kernel(cosine(rows_of(fam["cells"])), gamma, tau)


# --------------------------------------------------------------------------- runs of this workstream


@lru_cache(maxsize=None)
def frozen(split="development"):
    """Per-question costs of ``outputs/round5/banc/runs/banc-{split}.npz`` with ``arms`` and ``key`` (as ``harness.frozen``)."""
    meta = json.loads((RUNS / f"{VIEW}-{split}.json").read_text())
    if hash_file(RUNS / f"{VIEW}-{split}.npz") != meta["result_sha256"]:
        raise ValueError("run file differs from its record")
    with np.load(RUNS / f"{VIEW}-{split}.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["arms"] = {n: i for i, n in enumerate(meta["arms"])}
    d["key"] = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(d["family"], d["rho"]))}
    return d


def install_neurosym_view():
    """Let ``neurosym.common``/``kernel``/``harness`` accept ``view="banc"`` (run-time dispatch; files untouched).

    After the call, ``common.benchmark("banc")``, ``common.view_data("banc")``, ``common.family("banc", f)``,
    ``common.split_families("banc", split)``, ``kernel.profiles("banc")``, ``kernel.question_kernel("banc", ...)``,
    ``constraints.rows_of("banc", cells)`` and ``harness.frozen("banc", split)`` / ``baseline`` / ``ratio`` / ``run``
    use the BANC files; every other view is passed through unchanged.
    """
    from connectomequest.neurosym import common as c
    from connectomequest.neurosym import harness as h
    from connectomequest.neurosym import kernel as kk

    if getattr(c, "_banc_installed", False):
        return

    def dispatch(original, mine):
        def call(view, *args, **kwargs):
            return mine(*args, **kwargs) if view == VIEW else original(view, *args, **kwargs)

        call.__wrapped__ = original
        call.cache_clear = getattr(original, "cache_clear", lambda: None)
        return call

    c.benchmark = dispatch(c.benchmark, benchmark)
    c.view_data = dispatch(c.view_data, view_data)
    h.frozen = dispatch(h.frozen, frozen)
    kk.SIDES.setdefault(VIEW, ("left", "right"))
    c._banc_installed = True


def mlp_probabilities(rows):
    """Full frozen ``mlp_s0`` distributions (5,551 classes, class 0 = outside the vocabulary) of graph rows."""
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    from v35_build import MLP, full_probabilities
    from connectomequest.annotation_v32.training import StructuralMLP

    with np.load(OUT / "features.npz", allow_pickle=False) as z:
        features = z["features"][np.asarray(rows)]
    model = StructuralMLP.load(ROOT / MLP / "model.npz")
    temperature = json.loads((ROOT / MLP / "report.json").read_text())["temperature"]
    return full_probabilities(model, temperature, features)


# --------------------------------------------------------------------------- protocol


_HEX = re.compile(r"[0-9a-f]{64}")


def _hashes_in(obj, found):
    """Every ``{relative path: sha256}`` entry of the record (any nesting; also ``{"path": .., "sha256": ..}``)."""
    if isinstance(obj, dict):
        if isinstance(obj.get("path"), str) and isinstance(obj.get("sha256"), str):
            found.setdefault(obj["path"], set()).add(obj["sha256"])
        for key, value in obj.items():
            if isinstance(value, str) and _HEX.fullmatch(value):
                found.setdefault(key, set()).add(value)
            _hashes_in(value, found)
    elif isinstance(obj, list):
        for value in obj:
            _hashes_in(value, found)
    return found


def imported_round5_files(script):
    """The running script and every loaded ``connectomequest.round5`` module, as repository paths."""
    files = {Path(script).resolve()}
    for name, module in list(sys.modules.items()):
        if name == "connectomequest.round5" or name.startswith("connectomequest.round5."):
            path = getattr(module, "__file__", None)
            if path:
                files.add(Path(path).resolve())
    return sorted(str(p.relative_to(ROOT)) for p in files)


def guard(split, script, extra=(), record=RECORD):
    """Refuse ``test`` unless ``record`` lists the sha256 of the script, of every loaded round-5 module and of
    ``extra`` files (repository paths, e.g. the benchmark); ``development`` needs nothing."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split != "test":
        return None
    record = Path(record)
    if not record.exists():
        raise PermissionError(f"the held-out split needs {record.relative_to(ROOT) if record.is_relative_to(ROOT) else record}")
    listed = _hashes_in(json.loads(record.read_text()), {})
    files = imported_round5_files(script) + [str(Path(x).resolve().relative_to(ROOT)) for x in extra]
    missing = [rel for rel in files if hash_file(ROOT / rel) not in listed.get(rel, set())]
    if missing:
        raise PermissionError(f"record lacks matching sha256 for: {missing}")
    return listed


def save_new(path, writer):
    """Write via ``writer(stream)`` to a temporary file, then link it into place; never overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as stream:
            writer(stream)
        os.link(tmp, path)  # fails if the path appeared meanwhile
    finally:
        if tmp.exists():
            tmp.unlink()
    return path
