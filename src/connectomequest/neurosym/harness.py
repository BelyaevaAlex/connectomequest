"""Session runner shared by the neurosymbolic pilots (development split only).

``run(view, families, arms)`` replays every (family, rho) session of the given
development families with the orders produced by ``arms(fam, instance, outcomes,
rho_index)`` (a dict name -> payment order), checks each order with the frozen exact
verifier, and returns costs aligned with the frozen run of the split (development by default), so that the
baseline policies (weight order, AG, recal, direction-fixed, ...) are read from
``outputs/v35/runs/{view}-development.npz`` instead of being recomputed.
"""

from __future__ import annotations

import json
from functools import lru_cache
from multiprocessing import get_context

import numpy as np

from connectomequest.annotation_v35 import engine as e
from connectomequest.neurosym import common as c

STATUS = {"unresolved": 0, "positive": 1, "negative": 2}
_ARMS = {}


@lru_cache(maxsize=None)
def frozen(view, split="development"):
    """The frozen run of ``split``: arrays per session plus ``arms`` (name -> column)."""
    meta = json.loads((c.ROOT / f"outputs/v35/runs/{view}-{split}.json").read_text())
    with np.load(c.ROOT / f"outputs/v35/runs/{view}-{split}.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["arms"] = {n: i for i, n in enumerate(meta["arms"])}
    d["key"] = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(d["family"], d["rho"]))}
    return d


def rows(view, families, split="development"):
    """Indices of the frozen sessions of ``families`` (all four rho levels)."""
    key = frozen(view, split)["key"]
    return np.array([key[(int(f), r)] for f in families for r in range(len(c.RHOS))])


def baseline(view, families, name, split="development"):
    """Frozen costs of arm ``name`` (e.g. ``weight``, ``mlp_s0:ag``) for the sessions of ``families``."""
    d = frozen(view, split)
    return d["cost"][rows(view, families, split), d["arms"][name]].astype(np.int64)


def _family_job(f):
    view, arms = _ARMS["view"], _ARMS["arms"]
    fam = c.family(view, f)
    out = []
    for rho_index, threshold in c.thresholds(fam):
        inst = e.threshold_instance(fam["cells"], fam["weights"], threshold)
        truth = e.full_information_status(inst, fam["outcomes"])
        bound = e.clairvoyant_cost(inst, fam["outcomes"])
        costs = {}
        for name, order in arms(fam, inst, fam["outcomes"], rho_index).items():
            cost, status = e.evaluate(inst, fam["outcomes"], order)
            if status != truth or (truth == "unresolved" and cost != inst.size) or (bound is not None and cost < bound):
                raise AssertionError(f"{name}: order contradicts the exact full-information status")
            costs[name] = cost
        out.append((int(f), rho_index, STATUS[truth], costs))
    return out


def run(view, families, arms, workers=None, chunksize=4, split="development"):
    """Costs of every arm on every session of ``families``, aligned with ``rows(view, families)``."""
    import os

    workers = workers or min(32, len(os.sched_getaffinity(0)))
    _ARMS.update(view=view, arms=arms)
    c.benchmark(view)  # load before forking
    with get_context("fork").Pool(workers) as pool:
        results = pool.map(_family_job, list(map(int, families)), chunksize=chunksize)
    flat = [r for group in results for r in group]
    d = frozen(view, split)
    index = rows(view, families, split)
    for i, (f, r, truth, _) in zip(index, flat):
        if (int(d["family"][i]), int(d["rho"][i]), int(d["truth"][i])) != (f, r, truth):
            raise AssertionError("session misaligned with the frozen development run")
    names = sorted({n for *_, costs in flat for n in costs})
    return {n: np.array([costs[n] for *_, costs in flat], dtype=np.int64) for n in names}


@lru_cache(maxsize=None)
def _clusters(view, families):
    d = c.benchmark(view)
    fam = np.repeat(np.asarray(families), len(c.RHOS))
    _, index = np.unique(d["target_type"][d["fam_target"][fam]], return_inverse=True)
    k = int(index.max()) + 1
    rng = np.random.default_rng(20261112)
    counts = np.stack([np.bincount(row, minlength=k) for row in rng.integers(0, k, size=(2000, k))]).astype(float)
    return index, counts, k


def ratio(view, families, x, y, mask=None, split="development"):
    """Ratio of total costs with a type-cluster bootstrap 95 % interval (default mask: certifiable sessions)."""
    families = tuple(int(f) for f in families)
    index, counts, k = _clusters(view, families)
    if mask is None:
        mask = frozen(view, split)["truth"][rows(view, families, split)] > 0
    sx = np.bincount(index[mask], weights=np.asarray(x, float)[mask], minlength=k)
    sy = np.bincount(index[mask], weights=np.asarray(y, float)[mask], minlength=k)
    denominator = counts @ sy
    if sy.sum() <= 0:
        raise ValueError("no reference cost in the selected sessions")
    boot = (counts @ sx)[denominator > 0] / denominator[denominator > 0]  # drop the rare draws without any selected session
    return dict(value=float(sx.sum() / sy.sum()), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))


def truth(view, families, split="development"):
    return frozen(view, split)["truth"][rows(view, families, split)]
