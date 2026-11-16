"""Round 5, workstream ``policies``: question data, priors, bootstrap and the protocol guard.

Views are generic: ``malecns`` and ``flywire`` read the frozen benchmark
(``outputs/v35/benchmark``); any other view (e.g. ``banc``) is read from
``outputs/round5/<view>/`` or through ``connectomequest.round5.<view>`` if that
module provides ``benchmark_path()`` / ``manifest_path()``.

Priors are named on the command line: ``const_half`` and ``const_share`` (the
no-prediction priors of the first post hoc round), every ``plus_<name>`` array of the
benchmark (``mlp_s0``, ``provisional``, ...) and every file
``outputs/round5/priors/<name>-<view>.npz`` with the benchmark layout
(``plus_<name>`` aligned with the family-major cell rows, ``unknown_<name>`` aligned
with the target-major cell rows).

Nothing here reads or writes held-out results; the guard ``require_record`` is what
every round-5 runner of this workstream calls before it touches the test split.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np

from connectomequest.annotation_v35 import engine as e

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "outputs/v35/benchmark"
ROUND5 = ROOT / "outputs/round5"
OUT = ROUND5 / "policies"
PRIORS = ROUND5 / "priors"
RECORD = ROUND5 / "record.json"
RHOS = ((1, 4), (1, 2), (3, 4), (1, 1))
STATUS = {"unresolved": 0, "positive": 1, "negative": 2}
CONSTANT = ("const_half", "const_share")
BOOTSTRAP = 2000
BOOT_SEED = 20261112


# --------------------------------------------------------------------------- files


def hash_file(path):
    h = sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _view_module(view):
    try:
        return importlib.import_module(f"connectomequest.round5.{view}")
    except ImportError:
        return None


def preload(view):
    """Import the view's round-5 module (if any) before the code identity is taken."""
    return _view_module(view) if not (BENCH / f"{view}.npz").exists() else None


def question_kernel(view, fam, gamma, tau):
    """Answer-propagation kernel of the input cells of ``fam`` (neurosym for the v35 views, the view module otherwise)."""
    mod = preload(view)
    if mod is not None and hasattr(mod, "question_kernel"):
        return mod.question_kernel(fam, gamma, tau)
    from connectomequest.neurosym import kernel as kk

    return kk.question_kernel(view, fam, gamma, tau)


def preload_kernel(view):
    mod = preload(view)
    if mod is not None and hasattr(mod, "profiles"):
        mod.profiles()
        return
    from connectomequest.neurosym import constraints as k
    from connectomequest.neurosym import kernel as kk

    kk.profiles(view)
    k.rows_of(view, benchmark(view)["pre_cell"][:1])


def benchmark_path(view):
    if (BENCH / f"{view}.npz").exists():
        return BENCH / f"{view}.npz"
    mod = _view_module(view)
    if mod is not None and hasattr(mod, "benchmark_path"):
        return Path(mod.benchmark_path())
    for candidate in (ROUND5 / view / f"{view}.npz", ROUND5 / view / "benchmark" / f"{view}.npz", ROUND5 / view / "benchmark.npz"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no benchmark for view {view}")


def manifest_path(view):
    if (BENCH / f"{view}.npz").exists():
        return BENCH / "manifest.json"
    mod = _view_module(view)
    if mod is not None and hasattr(mod, "manifest_path"):
        return Path(mod.manifest_path())
    for candidate in (ROUND5 / view / "manifest.json", ROUND5 / view / "benchmark" / "manifest.json"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no manifest for view {view}")


@lru_cache(maxsize=None)
def benchmark(view):
    with np.load(benchmark_path(view), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    lengths = np.diff(d["target_offset"])[d["fam_target"]]
    d["fam_plus_offset"] = np.concatenate(([0], np.cumsum(lengths)))
    return d


@lru_cache(maxsize=None)
def manifest(view):
    return json.loads(manifest_path(view).read_text())


def split_families(view, split):
    """Question families whose target type belongs to ``split`` (``development`` or ``test``)."""
    if split not in ("development", "test"):
        raise ValueError(split)
    d = benchmark(view)
    of = {int(t["code"]): t["split"] for t in manifest(view)["types"]}
    types = d["target_type"][d["fam_target"]]
    return np.flatnonzero(np.array([of[int(t)] == split for t in types], dtype=bool))


def family(view, f):
    d = benchmark(view)
    t = int(d["fam_target"][f])
    lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
    labels = d["pre_label"][lo:hi]
    codes = d["fam_codes"][d["fam_code_offset"][f] : d["fam_code_offset"][f + 1]]
    outcomes = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER)).astype(np.intp)
    return dict(view=view, family=int(f), target=t, target_cell=int(d["target_cell"][t]), target_type=int(d["target_type"][t]),
                cells=d["pre_cell"][lo:hi], weights=d["pre_weight"][lo:hi], labels=labels, codes=codes, outcomes=outcomes,
                kind=int(d["fam_kind"][f]), source_weight=int(d["fam_source_weight"][f]), source_total=int(d["fam_source_total"][f]),
                lo=lo, hi=hi, plus_start=int(d["fam_plus_offset"][f]))


def thresholds(fam):
    total = int(np.asarray(fam["weights"]).sum())
    return [(i, e.threshold_goals(total, num * fam["source_weight"], den * fam["source_total"])[0]) for i, (num, den) in enumerate(RHOS)]


# --------------------------------------------------------------------------- priors


def probabilities(plus, unknown):
    """(member, nonmember, unresolved) rows exactly as the frozen runners build them."""
    plus = np.clip(np.asarray(plus).astype(float), 0.0, 1.0)
    unknown = np.clip(np.asarray(unknown).astype(float), 0.0, 1.0 - plus)
    return np.column_stack((plus, 1.0 - plus - unknown, unknown))


@lru_cache(maxsize=None)
def _prior_file(view, name):
    path = PRIORS / f"{name}-{view}.npz"
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        plus_key = f"plus_{name}" if f"plus_{name}" in keys else next(k for k in keys if k.startswith("plus"))
        unknown_key = f"unknown_{name}" if f"unknown_{name}" in keys else next(k for k in keys if k.startswith("unknown"))
        plus, unknown = z[plus_key], z[unknown_key]
    d = benchmark(view)
    if len(plus) != int(d["fam_plus_offset"][-1]) or len(unknown) != len(d["pre_cell"]):
        raise ValueError(f"{path} is not aligned with the {view} benchmark")
    return plus, unknown


def available_priors(view):
    d = benchmark(view)
    names = list(CONSTANT) + [k[5:] for k in d if k.startswith("plus_")]
    if PRIORS.exists():
        names += sorted(p.name[: -len(f"-{view}.npz")] for p in PRIORS.glob(f"*-{view}.npz"))
    return list(dict.fromkeys(names))


def prior(view, fam, name):
    """Prior of one question family: ``(n, 3)`` member / nonmember / unresolved probabilities."""
    n = len(fam["cells"])
    if name == "const_half":
        return np.tile((0.5, 0.5, 0.0), (n, 1))
    if name == "const_share":
        share = float(np.clip(fam["source_weight"] / fam["source_total"], 0.01, 0.99))
        return np.tile((share, 1.0 - share, 0.0), (n, 1))
    d = benchmark(view)
    if f"plus_{name}" in d:
        plus, unknown = d[f"plus_{name}"], d[f"unknown_{name}"]
    else:
        plus, unknown = _prior_file(view, name)
    s = fam["plus_start"]
    return probabilities(plus[s : s + n], unknown[fam["lo"] : fam["hi"]])


def pin_prior(view, name):
    """The prior's identity before a run reads it; call ``.check()`` before saving: a prior file that changed
    during the run (the run may have read either version) refuses the save."""
    ident = prior_identity(view, name)

    class Pin(dict):
        def check(self):
            if prior_identity(view, name) != ident:
                raise SystemExit(f"refusing to save: the prior {name} of {view} changed during the run")
            return dict(ident)

    return Pin(ident)


def prior_identity(view, name):
    """What the prior is computed from (for the run metadata)."""
    if name in CONSTANT:
        return dict(kind="constant")
    if f"plus_{name}" in benchmark(view):
        return dict(kind="benchmark", sha256=hash_file(benchmark_path(view)))
    return dict(kind="file", path=str((PRIORS / f"{name}-{view}.npz").relative_to(ROOT)), sha256=hash_file(PRIORS / f"{name}-{view}.npz"))


# --------------------------------------------------------------------------- frozen results (optional)


@lru_cache(maxsize=None)
def frozen(view, split):
    """Frozen per-question costs (``outputs/v35/runs`` or ``outputs/round5/<view>/runs``) if they exist, else ``None``."""
    meta = ROOT / f"outputs/v35/runs/{view}-{split}.json"
    if not meta.exists():
        meta = ROUND5 / view / "runs" / f"{view}-{split}.json"
    if not meta.exists():
        return None
    names = json.loads(meta.read_text())["arms"]
    with np.load(meta.with_suffix(".npz"), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["arms"] = {n: i for i, n in enumerate(names)}
    d["key"] = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(d["family"], d["rho"]))}
    return d


@lru_cache(maxsize=None)
def frozen_posthoc(view, split):
    meta = ROOT / f"outputs/v35/posthoc/{view}-{split}.json"
    if not meta.exists():
        return None
    names = json.loads(meta.read_text())["arms"]
    with np.load(meta.with_suffix(".npz"), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["arms"] = {n: i for i, n in enumerate(names)}
    fr = frozen(view, split)  # its rows are indexed with the frozen run's rows
    if fr is None or not (np.array_equal(d["family"], fr["family"]) and np.array_equal(d["rho"], fr["rho"])):
        raise AssertionError(f"post hoc results of {view}-{split} are not row-aligned with the frozen run")
    return d


@lru_cache(maxsize=None)
def frozen_propagation(view, split):
    path = ROOT / f"outputs/neurosym/B/heldout/{view}-{split}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def stored_cost(view, split, arm):
    """Cost of an arm that already exists in a frozen result set (``None`` if it does not).

    ``arm`` is ``prior:policy``; frozen runs, the first post hoc round and the stored
    answer-propagation run (``mlp_s0:prop-recal``, ``mlp_s0:prop-ag``) are searched.
    """
    for source in (frozen(view, split), frozen_posthoc(view, split)):
        if source is not None and arm in source["arms"]:
            return source["cost"][:, source["arms"][arm]].astype(np.int64)
    prop = frozen_propagation(view, split)
    if prop is not None and arm in ("mlp_s0:prop-recal", "mlp_s0:prop-ag"):
        return prop["b4__" + arm.split("-")[1]].astype(np.int64)
    return None


# --------------------------------------------------------------------------- bootstrap


class Clusters:
    """Type-cluster bootstrap of ratios of totals (2,000 resamples of target cell types)."""

    def __init__(self, view, family_of_session):
        d = benchmark(view)
        types = d["target_type"][d["fam_target"][np.asarray(family_of_session)]]
        _, self.index = np.unique(types, return_inverse=True)
        self.k = int(self.index.max()) + 1
        rng = np.random.default_rng(BOOT_SEED)
        self.counts = np.stack([np.bincount(row, minlength=self.k) for row in rng.integers(0, self.k, size=(BOOTSTRAP, self.k))]).astype(float)

    def sums(self, x, mask):
        return np.bincount(self.index[mask], weights=np.asarray(x, float)[mask], minlength=self.k)

    def ratio(self, x, y, mask):
        mask = np.asarray(mask, dtype=bool)
        sx, sy = self.sums(x, mask), self.sums(y, mask)
        if sy.sum() <= 0:
            return dict(value=float("nan"), low=float("nan"), high=float("nan"), n=int(mask.sum()))
        den = self.counts @ sy
        boot = (self.counts @ sx)[den > 0] / den[den > 0]
        return dict(value=float(sx.sum() / sy.sum()), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)), n=int(mask.sum()))

    def mean(self, x, mask):
        mask = np.asarray(mask, dtype=bool)
        sx = self.sums(x, mask)
        sn = np.bincount(self.index[mask], minlength=self.k).astype(float)
        den = self.counts @ sn
        boot = (self.counts @ sx)[den > 0] / den[den > 0]
        return dict(value=float(sx.sum() / max(sn.sum(), 1)), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)), n=int(mask.sum()))


# --------------------------------------------------------------------------- protocol guard


def round5_modules():
    """Files of every ``connectomequest.round5`` module imported so far (paths relative to the repository)."""
    base = (ROOT / "src/connectomequest/round5").resolve()
    out = []
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).resolve().is_relative_to(base):
            out.append(str(Path(path).resolve().relative_to(ROOT)))
    return sorted(set(out))


def code_identity(runner):
    files = [str(Path(runner).resolve().relative_to(ROOT))] + round5_modules()
    return {name: hash_file(ROOT / name) for name in files}


def _path_maps(node):
    if isinstance(node, dict):
        if node and all(isinstance(k, str) and isinstance(v, str) and len(v) == 64 for k, v in node.items()):
            yield node
        for v in node.values():
            yield from _path_maps(v)
    elif isinstance(node, list):
        for v in node:
            yield from _path_maps(v)


_GUARD = {}  # set by require_record on the held-out split: the files it checked


def require_record(runner, split):
    """Refuse the held-out split unless ``outputs/round5/record.json`` lists the sha256 of the runner and of every
    round-5 module it imports.

    The nested ``{relative path: sha256}`` maps under the record's key ``policies`` count (the whole record if
    it has no such key); a path listed more than once must carry the current hash everywhere. The checked
    files are remembered: ``save_*`` refuse to write if a round-5 module was imported afterwards."""
    if split != "test":
        return None
    if not RECORD.exists():
        raise SystemExit(f"refusing --split test: {RECORD} does not exist")
    record = json.loads(RECORD.read_text())
    from connectomequest.round5 import heldout  # noqa: F401  (the shared gate the runner claims next; checked here too)
    scope = record["policies"] if isinstance(record.get("policies"), dict) else record
    listed = {}
    for m in _path_maps(scope):
        for k, v in m.items():
            listed.setdefault(k, set()).add(v)
    code = code_identity(runner)
    missing = [k for k in code if k not in listed]
    changed = [k for k in code if k in listed and listed[k] != {code[k]}]
    if missing or changed:
        raise SystemExit(f"refusing --split test: record lacks {missing} or lists other hashes for {changed}")
    _GUARD["files"] = set(code)
    return code


def default_output(output, default, split):
    """The held-out split writes only to the default place, so ``fresh`` blocks a second held-out run."""
    if split == "test" and Path(output).resolve() != Path(default).resolve():
        raise SystemExit(f"refusing --output {output} on the held-out split (default: {default})")
    return Path(output)


def _recheck():
    """On the held-out split: no round-5 module may have been imported after the record was checked."""
    if _GUARD:
        late = sorted(set(round5_modules()) - _GUARD["files"])
        if late:
            raise SystemExit(f"refusing to save: round-5 modules imported after the record check: {late}")


def fresh(path):
    """Refuse to overwrite an existing output."""
    path = Path(path)
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _publish(tmp, path):
    """Atomic and never overwriting: a hard link fails if ``path`` exists (also when a concurrent job made it)."""
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise SystemExit(f"refusing to overwrite {path}") from None
    finally:
        tmp.unlink()
    return path


def save_npz(path, **arrays):
    """Write to a temporary file, then publish it atomically; never overwrite."""
    _recheck()
    path = fresh(path)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    with tmp.open("xb") as stream:
        np.savez(stream, **arrays)
    return _publish(tmp, path)


def save_text(path, text):
    _recheck()
    path = fresh(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x") as stream:
        stream.write(text)
    return _publish(tmp, path)


def save_json(path, payload):
    return save_text(path, json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n")


def workers(default=6):
    return max(1, min(int(os.environ.get("R5_WORKERS", default)), len(os.sched_getaffinity(0))))
