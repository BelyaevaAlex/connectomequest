"""Shared plumbing of the round-5 workstream ``costs`` (cost models and biological readout).

* ``guard``: a held-out run needs ``outputs/round5/record.json`` listing the sha256 of the
  running script and of every round-5 module it has imported; development runs need nothing.
* ``write_new`` / ``save_npz_new``: outputs are written to a temporary file and linked into
  place, so an existing output is never overwritten (``FileExistsError``).
* ``cached_npz``: shared caches under ``outputs/round5/cache`` (temp file + atomic rename,
  reused when present).
* ``Bootstrap``: the type-clustered percentile bootstrap of the paper (2,000 resamples of
  cell types, seed 20261112), for ratios of totals, means and differences.
* data access: the frozen benchmark arrays, the frozen per-question runs and the release
  type of every input cell (from the neurosym cache, built by the declared crosswalk code).
"""

from __future__ import annotations

import json
import os
import sys
from functools import cache
from pathlib import Path

import numpy as np

from connectomequest.annotation_v32.runtime import hash_file
from connectomequest.neurosym import common as nc

ROOT = nc.ROOT
OUT = ROOT / "outputs/round5/costs"
CACHE = ROOT / "outputs/round5/cache"
RECORD = ROOT / "outputs/round5/record.json"
RUNS = ROOT / "outputs/v35/runs"
BENCH = ROOT / "outputs/v35/benchmark"
VIEWS = ("malecns", "flywire")
RHOS = nc.RHOS
SPLITS = ("development", "test")
BOOTSTRAP = 2000
SEED = 20261112


# --------------------------------------------------------------------------- protocol


def _hashes_in(obj, found):
    """Collect every ``{relative path: sha256}`` entry found under keys named ``code``/``files``."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("code", "files", "sha256", "hashes") and isinstance(value, dict):
                for path, digest in value.items():
                    if isinstance(digest, str):
                        found.setdefault(path, set()).add(digest)
            _hashes_in(value, found)
    elif isinstance(obj, list):
        for value in obj:
            _hashes_in(value, found)
    return found


def imported_round5_files(script):
    """The running script and every loaded ``connectomequest.round5`` module, as repo paths."""
    files = {Path(script).resolve()}
    for name, module in list(sys.modules.items()):
        if name == "connectomequest.round5" or name.startswith("connectomequest.round5."):
            path = getattr(module, "__file__", None)
            if path:
                files.add(Path(path).resolve())
    return sorted(str(p.relative_to(ROOT)) for p in files)


def guard(split, script, record=RECORD, extra=()):
    """Refuse the held-out split unless the record lists matching hashes of all code in use
    (and of every file in ``extra``, e.g. an expectations file)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split != "test":
        return None
    if not Path(record).exists():
        raise PermissionError("the held-out split needs outputs/round5/record.json")
    listed = _hashes_in(json.loads(Path(record).read_text()), {})
    missing = []
    needed = imported_round5_files(script) + [str(Path(x).resolve().relative_to(ROOT)) for x in extra]
    for rel in needed:
        if hash_file(ROOT / rel) not in listed.get(rel, set()):
            missing.append(rel)
    if missing:
        raise PermissionError(f"record lacks matching sha256 for: {missing}")
    return listed


def write_new(path, data):
    """Write bytes or text to ``path`` atomically; never overwrite an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as stream:
        stream.write(data.encode() if isinstance(data, str) else data)
    try:
        os.link(tmp, path)  # fails if path appeared meanwhile
    finally:
        tmp.unlink()


def save_npz_new(path, **arrays):
    import io

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    write_new(path, buffer.getvalue())


def write_json_new(path, obj):
    write_new(path, json.dumps(obj, indent=1, sort_keys=True, default=_json_default) + "\n")


def _json_default(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def cached_npz(name, builder):
    """Load ``outputs/round5/cache/<name>`` or build it (temp file + atomic rename)."""
    path = CACHE / name
    if not path.exists():
        arrays = builder()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- data


@cache
def frozen(view, split):
    """Frozen per-question costs of the pre-specified arms (never recomputed here)."""
    meta = json.loads((RUNS / f"{view}-{split}.json").read_text())
    with np.load(RUNS / f"{view}-{split}.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["arms"] = {n: i for i, n in enumerate(meta["arms"])}
    return data


def split_families(view, split):
    return nc.split_families(view, split)


@cache
def input_rows(view):
    """Row of every benchmark input cell (``pre_cell``) in the neurosym view cache."""
    d = nc.benchmark(view)
    ids = nc.view_data(view)["ids"]
    rows = np.searchsorted(ids, d["pre_cell"])
    if np.any(rows >= len(ids)) or not np.array_equal(ids[rows], d["pre_cell"]):
        raise ValueError("input cell missing from the view cache")
    return rows


@cache
def input_release(view):
    """Release type code of every benchmark input cell (-1: no release type)."""
    return nc.view_data(view)["release_code"][input_rows(view)]


@cache
def node_strength(view):
    """In- plus out-strength of every neuron of the thresholded view graph."""
    g = nc.view_data(view)["graph"]
    return np.asarray(g.sum(0)).ravel() + np.asarray(g.sum(1)).ravel()


def cluster_of_rows(view, families):
    d = nc.benchmark(view)
    return d["target_type"][d["fam_target"][families]]


# --------------------------------------------------------------------------- statistics


class Bootstrap:
    """Type-clustered percentile bootstrap, as in ``scripts/v35_analyze.py``."""

    def __init__(self, clusters, draws=BOOTSTRAP, seed=SEED):
        self.labels, self.index = np.unique(np.asarray(clusters), return_inverse=True)
        self.n = len(self.labels)
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, self.n, size=(draws, self.n))
        self.counts = np.stack([np.bincount(row, minlength=self.n) for row in picks]).astype(float)

    def sums(self, x, mask):
        mask = np.asarray(mask, dtype=bool)
        return np.bincount(self.index[mask], weights=np.asarray(x, dtype=float)[mask], minlength=self.n)

    @staticmethod
    def _summary(value, boot):
        boot = boot[np.isfinite(boot)]
        return dict(value=float(value), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))

    def ratio(self, x, y, mask):
        sx, sy = self.sums(x, mask), self.sums(y, mask)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (self.counts @ sx) / (self.counts @ sy)
        return self._summary(sx.sum() / sy.sum(), boot)

    def mean(self, x, mask):
        return self.ratio(x, np.ones(len(self.index)), mask)

    def mean_difference(self, x, mask_a, mask_b):
        """Mean of ``x`` over ``mask_a`` minus over ``mask_b`` (types resampled jointly)."""
        one = np.ones(len(self.index))
        sa, na = self.sums(x, mask_a), self.sums(one, mask_a)
        sb, nb = self.sums(x, mask_b), self.sums(one, mask_b)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (self.counts @ sa) / (self.counts @ na) - (self.counts @ sb) / (self.counts @ nb)
        return self._summary(sa.sum() / na.sum() - sb.sum() / nb.sum(), boot)


# --------------------------------------------------------------------------- expectations


def lookup(result, path):
    node = result
    for key in path:
        node = node[key]
    return node["value"] if isinstance(node, dict) and "value" in node else node


def check_expectations(result, expectations):
    """Evaluate ``[{"name", "path", "op", "value", "minus"?}]``; ``minus`` is a second path
    whose value is subtracted; ops: <, <=, >, >=, ==, abs< (absolute value below)."""
    verdicts = []
    for item in expectations:
        observed = lookup(result, item["path"])
        if "minus" in item:
            observed = observed - lookup(result, item["minus"])
        t = item["value"]
        holds = {"<": observed < t, "<=": observed <= t, ">": observed > t, ">=": observed >= t, "==": observed == t,
                 "abs<": abs(observed) < t}[item["op"]]
        verdicts.append(dict(item, observed=observed, holds=bool(holds)))
    return verdicts
