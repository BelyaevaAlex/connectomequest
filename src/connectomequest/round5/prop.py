"""Round 5, workstream ``prop``: does answer propagation hold beyond the weak predictor?

Answer propagation (paper, Eq. 4; ``connectomequest.neurosym.kernel``) passes every
verified answer to unchecked cells with similar connectivity through
``K = cos ** gamma`` for ``cos >= tau`` (cosine of square-root synapse profiles,
outputs and inputs, in the graph of the connectome) with pseudo-count weight ``lam``.
The setting is the recorded one (``outputs/v35/posthoc-record-4.json``: lam = 8,
gamma = 3, tau = 0.3 in both connectomes) and is never re-tuned here.

This module adds

* priors: the stored benchmark priors (``mlp_s0`` = topology prior, ``provisional``
  = earlier-release prior, FlyWire only), the two no-prediction priors of the
  supplement (``const_half``: p+ = 1/2; ``const_share``: p+ = hemibrain share of the
  partner set clipped to [0.01, 0.99]; p_unresolved = 0 in both) and external
  priors with the layout of the benchmark ``plus_*`` / ``unknown_*`` arrays;
* policies with propagation: AG and recalibrated AG (the recorded
  ``KernelGreedy`` / ``KernelRecalibrated``) and member-first / direction-fixed
  whose beliefs follow propagation after every check, so that the remaining cells
  are re-sorted. The latter is an active-search-like rule (Wang, Garnett and
  Schneider, KDD 2013: query the node most likely to be positive given propagated
  labels), here with weights: the cell with the largest expected confirmed member
  weight ``w_v p'_v+``. Direction-fixed commits to its direction before the first
  check from the prior, exactly as the frozen rule, and re-sorts within it. With
  ``lam = 0`` every policy reproduces its frozen counterpart order for order;
* a resumable session runner (chunks written atomically, finished chunks never
  redone) whose every cost is checked against the frozen exact verifier.

Probabilities only order checks; every certificate is the frozen integer arithmetic.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse

from connectomequest.annotation_v35 import engine as e
from connectomequest.neurosym import common as c
from connectomequest.neurosym import constraints as k
from connectomequest.neurosym import harness as h
from connectomequest.neurosym import kernel as kk

ROOT = c.ROOT
OUT = ROOT / "outputs/round5/prop"
PRIORS_DIR = ROOT / "outputs/round5/priors"
CACHE = ROOT / "outputs/round5/cache"
SETTING = dict(lam=8.0, gamma=3.0, tau=0.3)  # recorded (posthoc-record-4); not a parameter of this workstream
POLICIES = ("ag", "recal", "member", "direction")
NO_PREDICTION = ("const_half", "const_share")
BOOTSTRAP_SEED, BOOTSTRAP_DRAWS = 20261112, 2000  # as connectomequest.neurosym.harness
STATUS = h.STATUS


# --------------------------------------------------------------------------- policies


class PropagatedOrder(kk._KernelCoupling, e._Adaptive):
    """Member-first / nonmember-first / direction-fixed re-sorted by propagated beliefs.

    Before every check the unchecked cell with the largest ``w_v p'_v`` of the
    committed answer (member, or nonmember) is chosen, ties by ascending neuron id.
    ``p'`` is the static prior updated by the recorded kernel coupling (as in
    ``KernelGreedy``). ``rule="direction"`` commits to the direction of the frozen
    direction-fixed rule (forecasts of the prior; member on a tie) before any check.
    """

    def __init__(self, instance, probabilities, kernel_matrix, lam, rule):
        e._Adaptive.__init__(self, instance, probabilities)
        self._setup(kernel_matrix, lam)
        self._base = self.p.copy()
        if rule == "direction":
            (_, forecast_positive), (_, forecast_negative) = e.direction_forecasts(instance, probabilities)
            rule = "member" if forecast_positive <= forecast_negative else "nonmember"
        if rule not in ("member", "nonmember"):
            raise ValueError(rule)
        self.rule = rule
        self._column = e.MEMBER if rule == "member" else e.NONMEMBER

    def choose(self, a, b, available):
        return self._best(self._w * self.p[:, self._column], available)

    def observe(self, cell, outcome):
        if self._lam == 0:
            return
        self._couple(cell, outcome, self._base)


def make_policy(policy, instance, probabilities, kernel_matrix, lam):
    """Policy object for ``run_adaptive``; ``lam = 0`` is the frozen rule."""
    if policy == "ag":
        return kk.KernelGreedy(instance, probabilities, kernel_matrix, lam)
    if policy == "recal":
        return kk.KernelRecalibrated(instance, probabilities, kernel_matrix, lam)
    if policy in ("member", "direction"):
        return PropagatedOrder(instance, probabilities, kernel_matrix, lam, policy)
    raise ValueError(policy)


@dataclass(frozen=True)
class Arm:
    prior: str
    policy: str
    propagate: bool

    @property
    def name(self):
        if self.policy == "weight":
            return "weight"
        return f"{self.prior}:{self.policy}" + ("+prop" if self.propagate else "")

    @property
    def lam(self):
        return SETTING["lam"] if self.propagate else 0.0

    @property
    def static(self):
        return Arm(self.prior, self.policy, False)

    @classmethod
    def parse(cls, name):
        if name == "weight":
            return WEIGHT
        prior, rest = name.split(":", 1)
        propagate = rest.endswith("+prop")
        policy = rest[: -len("+prop")] if propagate else rest
        if policy not in POLICIES:
            raise ValueError(name)
        return cls(prior, policy, propagate)


WEIGHT = Arm("", "weight", False)  # prediction-free baseline, needed only for views without frozen runs


# --------------------------------------------------------------------------- data sources


def profiles_from_graph(graph):
    """CSR (N, 2N), row j = [sqrt outputs of j, sqrt inputs of j], L2-normalized: ``kernel.profiles`` for any graph."""
    g = sparse.csr_matrix(graph).astype(np.float64)
    prof = sparse.hstack([g.sqrt(), g.T.tocsr().sqrt()]).tocsr()
    norm = np.sqrt(np.asarray(prof.multiply(prof).sum(1)).ravel())
    prof = sparse.diags(np.where(norm > 0, 1.0 / np.where(norm > 0, norm, 1.0), 0.0)) @ prof
    return prof.tocsr()


def external_prior_files(view):
    """Prior files of another workstream for ``view`` (same layout as the benchmark plus_/unknown_ arrays)."""
    if not PRIORS_DIR.is_dir():
        return []
    out = []
    for p in sorted(PRIORS_DIR.glob("*.npz")):
        if ".tmp" in p.name or p.name.startswith("."):
            continue
        sidecar = p.with_suffix(".json")
        owner = json.loads(sidecar.read_text()).get("view") if sidecar.exists() else None
        if owner == view or (owner is None and (p.stem.endswith(f"-{view}") or p.stem.startswith(f"{view}-"))):
            out.append(p)
    return out


def load_external_priors(view, bench):
    """{name: (plus, unknown)} from ``outputs/round5/priors``; every array is checked against the benchmark layout."""
    out, files = {}, {}
    for path in external_prior_files(view):
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        for key in arrays:
            if not key.startswith("plus_"):
                continue
            name = key[5:]
            if "unknown_" + name not in arrays:
                raise ValueError(f"{path.name}: plus_{name} without unknown_{name}")
            plus, unknown = arrays[key], arrays["unknown_" + name]
            if plus.shape != (int(bench["fam_plus_offset"][-1]),) or unknown.shape != bench["pre_cell"].shape:
                raise ValueError(f"{path.name}: {name} does not have the layout of the benchmark arrays")
            if not (np.isfinite(plus).all() and np.isfinite(unknown).all()):
                raise ValueError(f"{path.name}: {name} has non-finite values")
            for ref in ("pre_cell", "fam_target", "target_offset"):
                if ref in arrays and not np.array_equal(arrays[ref], bench[ref]):
                    raise ValueError(f"{path.name}: {ref} differs from the benchmark")
            if name in out or ("plus_" + name) in bench or name in NO_PREDICTION:
                raise ValueError(f"prior name {name} is not unique")
            out[name], files[name] = (plus, unknown), path
    return out, files


@dataclass
class Source:
    """Benchmark arrays, connectivity profiles and node lookup of one view."""

    view: str
    bench: dict
    profiles: sparse.csr_matrix
    ids: np.ndarray
    external: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)  # role -> path of every data file the results depend on

    def rows(self, cells):
        return k._v35_build().rows_of(self.ids, np.asarray(cells))

    def priors(self):
        return sorted({key[5:] for key in self.bench if key.startswith("plus_")} | set(NO_PREDICTION) | set(self.external))


@lru_cache(maxsize=None)
def source(view, external=True):
    if view in c.VIEW_ROOTS:
        bench = c.benchmark(view)
        files = {"benchmark": c.BENCH / f"{view}.npz", "graph": c.CACHE / f"{view}.npz"}
        ext, ext_files = load_external_priors(view, bench) if external else ({}, {})
        files.update({f"prior:{n}": p for n, p in ext_files.items()})
        return Source(view, bench, kk.profiles(view), c.view_data(view)["ids"], ext, files)
    if view == "banc":
        from connectomequest.round5 import banc  # another workstream's loader; imported only for this view

        return banc_source(banc, external)
    raise ValueError(view)


def banc_source(banc, external):
    """BANC through ``connectomequest.round5.banc``: its benchmark arrays and profiles of its five-synapse graph."""
    bench = banc.benchmark()
    files = {"benchmark": banc.BENCH, "manifest": banc.MANIFEST, "view": banc.VIEWFILE, "view-names": banc.VIEWNAMES,
             "graph": banc.PREPARED / "public/topology.npz"}
    ext, ext_files = load_external_priors("banc", bench) if external else ({}, {})
    files.update({f"prior:{n}": p for n, p in ext_files.items()})
    return Source("banc", bench, banc.profiles(), banc.view_data()["ids"], ext, files)


def family(src, f, names):
    """Cells, weights, outcomes and the priors ``names`` of one question family (as ``common.family``)."""
    d = src.bench
    t = int(d["fam_target"][f])
    lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
    n = hi - lo
    labels = d["pre_label"][lo:hi]
    codes = d["fam_codes"][d["fam_code_offset"][f] : d["fam_code_offset"][f + 1]]
    outcomes = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
    start = int(d["fam_plus_offset"][f])
    source_weight, source_total = int(d["fam_source_weight"][f]), int(d["fam_source_total"][f])
    priors = {}
    for name in names:
        if name == "const_half":
            priors[name] = np.tile((0.5, 0.5, 0.0), (n, 1))
        elif name == "const_share":
            share = float(np.clip(source_weight / source_total, 0.01, 0.99))
            priors[name] = np.tile((share, 1.0 - share, 0.0), (n, 1))
        else:
            plus_all, unknown_all = src.external[name] if name in src.external else (d["plus_" + name], d["unknown_" + name])
            plus = np.clip(plus_all[start : start + n].astype(float), 0, 1)
            unknown = np.clip(unknown_all[lo:hi].astype(float), 0, 1 - plus)
            priors[name] = np.column_stack((plus, 1 - plus - unknown, unknown))
    return dict(family=int(f), target=t, target_cell=int(d["target_cell"][t]), target_type=int(d["target_type"][t]), cells=d["pre_cell"][lo:hi],
                weights=d["pre_weight"][lo:hi], labels=labels, codes=codes, outcomes=outcomes, priors=priors, kind=int(d["fam_kind"][f]),
                source_weight=source_weight, source_total=source_total)


def question_kernel(src, cells):
    """Recorded kernel of the input cells of one target (rows aligned with ``cells``)."""
    p = src.profiles[src.rows(cells)]
    similarity = np.asarray((p @ p.T).todense(), dtype=np.float64)
    return kk.kernel(similarity, SETTING["gamma"], SETTING["tau"])


def split_families(src, split):
    if src.view in c.VIEW_ROOTS:
        return c.split_families(src.view, split)
    if src.view == "banc":
        from connectomequest.round5 import banc

        return banc.split_families(split)
    raise ValueError(src.view)


# --------------------------------------------------------------------------- session runner


_STATE = {}


def _run_chunk(job):
    """Run every arm on the sessions of one chunk of families and write the chunk atomically."""
    index, families, path = job
    src, arms = _STATE["source"], _STATE["arms"]
    names = sorted({a.prior for a in arms if a != WEIGHT})
    out = dict(family=[], rho=[], truth=[], size=[], clairvoyant=[], cost=[])
    target, kern = None, None
    for f in families:
        fam = family(src, int(f), names)
        if fam["target"] != target:
            target, kern = fam["target"], question_kernel(src, fam["cells"])
        for rho_index, threshold in c.thresholds(fam):
            inst = e.threshold_instance(fam["cells"], fam["weights"], threshold)
            truth = e.full_information_status(inst, fam["outcomes"])
            bound = e.clairvoyant_cost(inst, fam["outcomes"])
            costs = []
            for arm in arms:
                if arm == WEIGHT:
                    order = e.weight_order(inst)
                else:
                    order = e.run_adaptive(inst, fam["outcomes"], make_policy(arm.policy, inst, fam["priors"][arm.prior], kern, arm.lam))
                cost, status = e.evaluate(inst, fam["outcomes"], order)
                if status != truth or (truth == "unresolved" and cost != inst.size) or (bound is not None and cost < bound):
                    raise AssertionError(f"{arm.name}: order contradicts the exact full-information status")
                costs.append(cost)
            out["family"].append(int(f))
            out["rho"].append(rho_index)
            out["truth"].append(STATUS[truth])
            out["size"].append(inst.size)
            out["clairvoyant"].append(-1 if bound is None else bound)
            out["cost"].append(costs)
    arrays = {key: np.asarray(v, dtype=np.int64).reshape(len(out["family"]), len(arms)) if key == "cost" else np.asarray(v, dtype=np.int64)
              for key, v in out.items()}
    atomic_savez(path, **arrays)
    return index


def atomic_savez(path, **arrays):
    """Write ``path`` through a per-process temporary file and an atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    with tmp.open("wb") as stream:
        np.savez(stream, **arrays)
    tmp.replace(path)


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def run_sessions(src, families, arms, parts_dir, workers=6, chunk=24, signature=None, log=print):
    """Costs (sessions x arms) of every arm on every session of ``families``; resumes from ``parts_dir``.

    ``signature`` (any JSON) is stored with the parts; a parts directory written with
    another signature is never mixed in. Chunk ``i`` covers ``families[i*chunk:(i+1)*chunk]``.
    """
    from multiprocessing import get_context

    parts_dir = Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    sig_path = parts_dir / "signature.json"
    sig = json.dumps(dict(signature=signature, arms=[a.name for a in arms], chunk=chunk, families=[int(f) for f in families]), sort_keys=True)
    if sig_path.exists():
        if sig_path.read_text() != sig:
            raise ValueError(f"{parts_dir} holds parts of another run; remove it or choose another --name")
    else:
        atomic_write_text(sig_path, sig)
    families = np.asarray(families, dtype=np.int64)
    jobs = [(i, families[s : s + chunk], parts_dir / f"chunk-{i:05d}.npz") for i, s in enumerate(range(0, len(families), chunk))]
    todo = [j for j in jobs if not j[2].exists()]
    log(f"{src.view}: {len(jobs)} chunks, {len(jobs) - len(todo)} already on disk, {len(todo)} to run on {workers} workers")
    _STATE.update(source=src, arms=list(arms))
    if todo:
        # largest targets first so that the tail of the pool is short
        size = np.diff(src.bench["target_offset"])[src.bench["fam_target"]]
        todo.sort(key=lambda j: -int((size[j[1]].astype(np.int64) ** 2).sum()))
        if workers <= 1:
            for done, j in enumerate(todo, 1):
                _run_chunk(j)
        else:
            with get_context("fork").Pool(workers) as pool:
                for done, _ in enumerate(pool.imap_unordered(_run_chunk, todo), 1):
                    if done % max(1, len(todo) // 10) == 0 or done == len(todo):
                        log(f"{src.view}: {done}/{len(todo)} chunks")
    parts = []
    for _, fams, path in jobs:
        with np.load(path, allow_pickle=False) as z:
            part = {key: z[key] for key in z.files}
        expected = sorted((int(f), r) for f in fams for r in range(len(c.RHOS)))
        if sorted(zip(part["family"].tolist(), part["rho"].tolist())) != expected or part["cost"].shape != (len(expected), len(arms)):
            raise AssertionError(f"{path.name} does not hold exactly the sessions and arms of its chunk")
        parts.append(part)
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


# --------------------------------------------------------------------------- results computed elsewhere (never recomputed)


B4_DIR = ROOT / "outputs/neurosym/B/heldout"
POSTHOC_DIR = ROOT / "outputs/v35/posthoc"


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def _align(view, split, families, family, rho):
    key = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(family, rho))}
    return np.array([key[(int(f), r)] for f in families for r in range(len(c.RHOS))])


def existing_arms(view, split):
    """{arm name: cost} for arms already computed by earlier runs, aligned with ``harness.rows(view, families, split)``.

    * frozen runs (``outputs/v35/runs``): ``{prior}:{policy}`` without propagation and ``weight``;
    * first post hoc round (``outputs/v35/posthoc``): ``const_half`` / ``const_share`` with AG and recal;
    * fourth post hoc round (``outputs/neurosym/B/heldout``): ``mlp_s0:{ag,recal}+prop`` at the recorded setting.
    """
    if view == "banc":
        return _banc_existing(split)
    families = c.split_families(view, split)
    d = h.frozen(view, split)
    idx = h.rows(view, families, split)
    out = {name: d["cost"][idx, col].astype(np.int64) for name, col in d["arms"].items()}
    out["clairvoyant"] = d["clairvoyant"][idx].astype(np.int64)
    meta = json.loads((POSTHOC_DIR / f"{view}-{split}.json").read_text())
    z = _load(POSTHOC_DIR / f"{view}-{split}.npz")
    rows = _align(view, split, families, z["family"], z["rho"])
    for col, name in enumerate(meta["arms"]):
        if name.split(":")[0] in NO_PREDICTION:
            out[name] = z["cost"][rows, col].astype(np.int64)
    b4_meta = B4_DIR / f"{view}-{split}.json"
    if b4_meta.exists():
        picks = json.loads(b4_meta.read_text())["picks"]["config"]
        if {key: float(picks[key]) for key in SETTING} != SETTING:
            raise AssertionError("the recorded propagation run does not use the recorded setting")
        z = _load(B4_DIR / f"{view}-{split}.npz")
        rows = _align(view, split, families, z["family"], z["rho"])
        for base in ("ag", "recal"):
            out[f"mlp_s0:{base}+prop"] = z[f"b4__{base}"][rows].astype(np.int64)
    return out


def _banc_existing(split):
    """Costs of the banc workstream's own runs (``banc.frozen``), when they exist."""
    from connectomequest.round5 import banc

    if not (banc.RUNS / f"banc-{split}.json").exists():
        return {}
    d = banc.frozen(split)
    families = banc.split_families(split)
    idx = _align("banc", split, families, d["family"], d["rho"])
    out = {name: d["cost"][idx, col].astype(np.int64) for name, col in d["arms"].items()}
    if "clairvoyant" in d:
        out["clairvoyant"] = d["clairvoyant"][idx].astype(np.int64)
    return out


def reference(view, split):
    """family, rho, truth and clairvoyant of the stored run of the split (frozen v35 run, or the banc run), or None."""
    if view in c.VIEW_ROOTS:
        families = c.split_families(view, split)
        d, idx = h.frozen(view, split), h.rows(view, families, split)
    elif view == "banc":
        from connectomequest.round5 import banc

        if not (banc.RUNS / f"banc-{split}.json").exists():
            return None
        d = banc.frozen(split)
        idx = _align("banc", split, banc.split_families(split), d["family"], d["rho"])
    else:
        raise ValueError(view)
    return {key: d[key][idx].astype(np.int64) for key in ("family", "rho", "truth", "clairvoyant") if key in d}


def existing_files(view, split):
    """Stored result files whose arms are reused (hashed into every analysis)."""
    if view == "banc":
        from connectomequest.round5 import banc

        return [p for p in (banc.RUNS / f"banc-{split}.npz", banc.RUNS / f"banc-{split}.json") if p.exists()]
    files = [ROOT / f"outputs/v35/runs/{view}-{split}.{x}" for x in ("npz", "json")]
    files += [POSTHOC_DIR / f"{view}-{split}.{x}" for x in ("npz", "json")]
    files += [p for p in (B4_DIR / f"{view}-{split}.npz", B4_DIR / f"{view}-{split}.json") if p.exists()]
    return files


# --------------------------------------------------------------------------- statistics


class Clusters:
    """Type-cluster bootstrap of ``connectomequest.neurosym.harness`` (same seed and draws) for any session list."""

    def __init__(self, session_types):
        _, self.index = np.unique(np.asarray(session_types), return_inverse=True)
        self.k = int(self.index.max()) + 1
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        self.counts = np.stack([np.bincount(row, minlength=self.k) for row in rng.integers(0, self.k, size=(BOOTSTRAP_DRAWS, self.k))]).astype(float)

    def sums(self, x, mask):
        return np.bincount(self.index[mask], weights=np.asarray(x, float)[mask], minlength=self.k)

    def ratio(self, x, y, mask):
        sx, sy = self.sums(x, mask), self.sums(y, mask)
        if sy.sum() <= 0:
            raise ValueError("no reference cost in the selected sessions")
        den = self.counts @ sy
        boot = (self.counts @ sx)[den > 0] / den[den > 0]
        return dict(value=float(sx.sum() / sy.sum()), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))

    def shares(self, x, y, mask):
        """Per-question shares (point and 95% interval) of x < y (better), x == y (tied), x > y (worse)."""
        n = self.sums(np.ones(len(x)), mask)
        dn = self.counts @ n
        out = {}
        for key, flag in (("better", x < y), ("tied", x == y), ("worse", x > y)):
            s = self.sums(flag.astype(float), mask)
            boot = (self.counts @ s)[dn > 0] / dn[dn > 0]
            out[key] = dict(value=float(s.sum() / n.sum()), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))
        return out

    def gap_closed(self, prop, static, clair, mask):
        """Share of the gap between ``static`` and the clairvoyant bound that ``prop`` closes (ratio of totals)."""
        s = [self.sums(v, mask) for v in (prop, static, clair)]
        if s[1].sum() == s[2].sum():
            return dict(value=float("nan"), low=float("nan"), high=float("nan"))  # no gap to close
        point = (s[1].sum() - s[0].sum()) / (s[1].sum() - s[2].sum())
        b = [self.counts @ v for v in s]
        ok = b[1] != b[2]
        boot = (b[1][ok] - b[0][ok]) / (b[1][ok] - b[2][ok])
        return dict(value=float(point), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))


def session_types(view, families):
    d = c.benchmark(view) if view in c.VIEW_ROOTS else source(view).bench
    fam = np.repeat(np.asarray(families), len(c.RHOS))
    return d["target_type"][d["fam_target"][fam]]


def session_sizes(view, families):
    """Number of input cells of the target of every session (four per family)."""
    d = c.benchmark(view) if view in c.VIEW_ROOTS else source(view).bench
    fam = np.repeat(np.asarray(families), len(c.RHOS))
    return np.diff(d["target_offset"])[d["fam_target"][fam]]
