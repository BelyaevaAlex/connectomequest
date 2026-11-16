"""Pilot B round 2: kernel coupling (B4) and pooled prior (B3')."""

import numpy as np

from connectomequest.annotation_v35 import engine as e
from connectomequest.neurosym import common as c
from connectomequest.neurosym import harness as h
from connectomequest.neurosym import kernel as kk

import pytest as _pytest

from connectomequest.neurosym import common as _common

if not all((_common.VIEW_ROOTS[v] / "public/topology.npz").exists() and (_common.BENCH / f"{v}.npz").exists() for v in _common.VIEW_ROOTS):
    _pytest.skip("needs the connectome graphs and the benchmark arrays", allow_module_level=True)

VIEW = "flywire"


def _families(n):
    return c.development_families(VIEW, 0)[:n]


def test_lambda_zero_is_bit_for_bit_frozen():
    d = h.frozen(VIEW)
    fams = _families(10)
    pooled = kk.family_priors(VIEW, fams, gamma=1.0, tau=0.5, lam=0.0)
    for f in fams:
        fam = c.family(VIEW, f)
        assert np.array_equal(pooled[int(f)], fam["priors"]["mlp_s0"])
        kern = kk.question_kernel(VIEW, fam, 1.0, 0.3)
        for rho, threshold in c.thresholds(fam):
            inst = e.threshold_instance(fam["cells"], fam["weights"], threshold)
            for base, arm in (("ag", "mlp_s0:ag"), ("recal", "mlp_s0:recal")):
                order = e.run_adaptive(inst, fam["outcomes"], kk.policy(base, inst, fam["priors"]["mlp_s0"], kern, 0.0))
                frozen = e.run_adaptive(inst, fam["outcomes"], (e.AdaptiveGreedy if base == "ag" else e.RecalibratedGreedy)(inst, fam["priors"]["mlp_s0"]))
                assert np.array_equal(order, frozen)
                assert e.evaluate(inst, fam["outcomes"], order)[0] == d["cost"][d["key"][(int(f), rho)], d["arms"][arm]]


def _coupled(fam, inst, outcomes, rho):
    kern = kk.question_kernel(VIEW, fam, 1.0, 0.3)
    p = fam["priors"]["mlp_s0"]
    return {b: e.run_adaptive(inst, outcomes, kk.policy(b, inst, p, kern, 8.0)) for b in ("ag", "recal")}


def test_coupled_orders_are_sound_under_the_exact_verifier():
    fams = _families(24)
    out = h.run(VIEW, fams, _coupled, workers=4)  # raises if any order contradicts the full-information status
    assert set(out) == {"ag", "recal"} and all(len(v) == 4 * len(fams) for v in out.values())


def test_control_kernels_keep_what_they_claim_to_keep():
    from connectomequest.neurosym import constraints as k

    fam = c.family(VIEW, int(_families(1)[0]))
    cfg = dict(kind="cos", gamma=3.0, tau=0.3, lam=8.0)
    base = kk.control_kernel(VIEW, fam, cfg)
    shuffled = kk.control_kernel(VIEW, fam, dict(cfg, shuffle=0))
    assert np.array_equal(np.sort(base.ravel()), np.sort(shuffled.ravel())) and np.allclose(shuffled, shuffled.T)
    nodes = k.rows_of(VIEW, fam["cells"])
    target = int(k.rows_of(VIEW, np.array([fam["target_cell"]]))[0])
    p = kk.profiles(VIEW)[nodes]
    assert np.all(np.asarray(p[:, target].todense()) > 0)  # every presynaptic cell has the shared output to the target
    nt = kk.cosine_without(VIEW, nodes, target)
    assert np.allclose(np.diag(nt)[np.asarray(p.getnnz(1)) > 1], 1.0)
    weight = kk.control_kernel(VIEW, fam, dict(kind="weight", sigma=0.5, tau=0.3, lam=8.0))
    assert (weight > 0).sum() == (base > 0).sum() and np.allclose(weight, weight.T)
    oracle = kk.control_kernel(VIEW, fam, dict(kind="oracle", lam=8.0))
    j, m = np.nonzero(oracle)
    assert np.all(fam["labels"][j] == fam["labels"][m]) and np.all(fam["labels"][j] >= 0)


def test_beliefs_and_pooled_priors_stay_normalized():
    fam = c.family(VIEW, int(_families(1)[0]))
    inst = e.threshold_instance(fam["cells"], fam["weights"], c.thresholds(fam)[1][1])
    kern = kk.question_kernel(VIEW, fam, 1.0, 0.3)
    assert np.all(kern >= 0) and np.all(np.diag(kern) == 0) and kern.max() <= 1
    for base in ("ag", "recal"):
        pol = kk.policy(base, inst, fam["priors"]["mlp_s0"], kern, 8.0)
        for j in range(min(15, inst.size)):
            pol.observe(j, int(fam["outcomes"][j]))
            e.check_probabilities(inst, pol.p)
    prior = kk.family_priors(VIEW, [fam["family"]], gamma=1.0, tau=0.3, lam=1.0)[fam["family"]]
    e.check_probabilities(inst, prior)
    assert not np.array_equal(prior, fam["priors"]["mlp_s0"])
