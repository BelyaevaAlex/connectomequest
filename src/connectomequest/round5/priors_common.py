"""Shared plumbing of the round-5 workstream ``priors``.

* ``guard``: a held-out run needs ``outputs/round5/record.json`` listing, under a key named
  ``code``/``files``/``sha256``/``hashes`` at any depth, ``{repo path: sha256}`` for the running
  script and for every ``connectomequest.round5`` module it has imported. Development runs
  need nothing.
* ``save_npz_new`` / ``write_text_new``: outputs go to a temporary file and are linked into
  place, so an existing output is never overwritten (``FileExistsError``).
* ``cached``: shared caches under ``outputs/round5/cache`` (temp file + atomic rename; reused).
* ``seed_uniform``: the fixed pseudo-random number in [0, 1) of a neuron that decides whether a
  curator has confirmed it (seed set of fraction f = neurons with a number below f; nested in f).
* ``Clusters``: the type-clustered percentile bootstrap of the paper (2,000 resamples, seed
  20261112) for ratios of totals.
"""

from __future__ import annotations

import json
import os
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np

from connectomequest.annotation_v32.runtime import hash_file
from connectomequest.neurosym import common as nc

ROOT = nc.ROOT
OUT = ROOT / "outputs/round5/priors"
CACHE = ROOT / "outputs/round5/cache"
RECORD = ROOT / "outputs/round5/record.json"
VIEWS = ("malecns", "flywire")
SPLITS = ("development", "test")
FRACTIONS = (0.1, 0.3, 0.5)
BOOTSTRAP = 2000
BOOT_SEED = 20261112


# --------------------------------------------------------------------------- protocol


def _hashes_in(obj, found):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("code", "files", "sha256", "hashes") and isinstance(value, dict):
                for path, digest in value.items():
                    if isinstance(digest, str):
                        found.setdefault(str(path), set()).add(digest)
            _hashes_in(value, found)
    elif isinstance(obj, list):
        for value in obj:
            _hashes_in(value, found)
    return found


def imported_round5_files(script):
    """The running script and every loaded ``connectomequest.round5`` module, as repo-relative paths."""
    files = {Path(script).resolve()}
    for name, module in list(sys.modules.items()):
        if name == "connectomequest.round5" or name.startswith("connectomequest.round5."):
            path = getattr(module, "__file__", None)
            if path:
                files.add(Path(path).resolve())
    return sorted(str(p.relative_to(ROOT)) for p in files)


def guard(split, script, record=RECORD):
    """Refuse the held-out split unless the record lists matching hashes of all round-5 code in use."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split != "test":
        return None
    record = Path(record)
    if not record.exists():
        raise PermissionError("the held-out split needs outputs/round5/record.json")
    listed = _hashes_in(json.loads(record.read_text()), {})
    bad = [rel for rel in imported_round5_files(script) if hash_file(ROOT / rel) not in listed.get(rel, set())]
    if bad:
        raise PermissionError(f"code not recorded (or changed since the record): {bad}")
    return {rel: hash_file(ROOT / rel) for rel in imported_round5_files(script)}


def guard_data(split, paths, record=RECORD):
    """For the held-out split, every data file a run depends on (e.g. a prior) must be listed with its sha256 too."""
    if split != "test":
        return None
    listed = _hashes_in(json.loads(Path(record).read_text()), {})
    rels = [str(Path(p).resolve().relative_to(ROOT)) for p in paths]
    bad = [rel for rel in rels if hash_file(ROOT / rel) not in listed.get(rel, set())]
    if bad:
        raise PermissionError(f"data not recorded (or changed since the record): {bad}")
    return {rel: hash_file(ROOT / rel) for rel in rels}


def code_identity(script):
    return {rel: hash_file(ROOT / rel) for rel in imported_round5_files(script)}


def _tmp(path):
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def save_npz_new(path, **arrays):
    """Write an npz that must not exist yet (temporary file, then a hard link that fails if it exists)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = _tmp(path)
    with tmp.open("wb") as stream:
        np.savez(stream, **arrays)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def write_text_new(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = _tmp(path)
    tmp.write_text(text)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def write_json_new(path, obj):
    write_text_new(path, json.dumps(obj, indent=1, sort_keys=True, default=_json_default) + "\n")


def _json_default(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def cached(name, build):
    """Load ``outputs/round5/cache/<name>.npz`` or build it (dict of arrays), atomically."""
    path = CACHE / f"{name}.npz"
    if not path.exists():
        arrays = build()
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = _tmp(path)
        with tmp.open("wb") as stream:
            np.savez(stream, **arrays)
        tmp.replace(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- scenario


def _uniform(tag, view, ids):
    return np.array([int(sha256(f"{tag}:{view}:{int(i)}".encode()).hexdigest()[:15], 16) / 16.0**15 for i in ids], dtype=np.float64)


def seed_uniform(view, ids):
    """Number in [0, 1) per neuron; seeds of fraction f are the neurons below f (independent of every question)."""
    return _uniform("cq-r5-seed", view, ids)


def calibration_uniform(view, ids):
    """Second number in [0, 1) per neuron; seeds below ``CALIBRATION_SHARE`` are held aside for calibration."""
    return _uniform("cq-r5-calibration", view, ids)


CALIBRATION_SHARE = 0.2


def fraction_tag(f):
    return f"{int(round(100 * f)):02d}"


# --------------------------------------------------------------------------- statistics


class Clusters:
    """Type-clustered bootstrap over sessions (or any rows) with a cell-type label each."""

    def __init__(self, cluster):
        clusters, self.index = np.unique(np.asarray(cluster), return_inverse=True)
        self.k = len(clusters)
        rng = np.random.default_rng(BOOT_SEED)
        self.counts = np.stack([np.bincount(row, minlength=self.k) for row in rng.integers(0, self.k, size=(BOOTSTRAP, self.k))]).astype(float)

    def sums(self, x, mask=None):
        x = np.asarray(x, dtype=float)
        mask = np.ones(len(x), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        return np.bincount(self.index[mask], weights=x[mask], minlength=self.k)

    def ratio(self, x, y, mask=None):
        sx, sy = self.sums(x, mask), self.sums(y, mask)
        boot = (self.counts @ sx) / (self.counts @ sy)
        return dict(value=float(sx.sum() / sy.sum()), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))

    def mean(self, x, mask=None):
        return self.ratio(x, np.ones(len(x)), mask)


def family_types(view, families):
    d = nc.benchmark(view)
    return d["target_type"][d["fam_target"][np.asarray(families)]]
