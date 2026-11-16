"""Pilot B constraints: normalization, kept unresolved mass, and identity with mlp_s0 at beta = 0."""

import numpy as np

from connectomequest.annotation_v35 import engine as e
from connectomequest.neurosym import common as c
from connectomequest.neurosym import constraints as k

import pytest as _pytest

from connectomequest.neurosym import common as _common

if not all((_common.VIEW_ROOTS[v] / "public/topology.npz").exists() and (_common.BENCH / f"{v}.npz").exists() for v in _common.VIEW_ROOTS):
    _pytest.skip("needs the connectome graphs and the benchmark arrays", allow_module_level=True)

VIEW = "flywire"


def _q0(n=6, v=40, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.dirichlet(np.full(v, 0.3), size=n)
    q[0, 1:] = 0.0  # a cell that is all __OTHER__
    q[0, 0] = 1.0
    return q


def test_posterior_is_normalized_and_keeps_other_mass():
    q0 = _q0()
    phi = np.random.default_rng(1).normal(scale=5.0, size=q0.shape)
    q = k.posterior(q0, phi)
    assert np.all(q >= 0)
    assert np.allclose(q.sum(1), 1.0, atol=1e-12)
    assert np.array_equal(q[:, 0], q0[:, 0])  # unresolved forecast never collapses
    assert np.allclose(q[1:, 1:].sum(1), q0[1:, 1:].sum(1))


def test_shares_are_distributions():
    for direction in ("in", "out"):
        r, marginal = k.shares(direction)
        assert np.allclose(r[:, 1:].sum(0), 1.0)
        assert np.all(r[0] == 0) and marginal[0] == 0
        assert np.isclose(marginal.sum(), 1.0)


def test_full_cache_reproduces_stored_mlp_s0_priors():
    fams = c.development_families(VIEW, 0)[:200]
    assert k.check_full(VIEW, fams) == (0.0, 0.0)


def test_beta_zero_equals_mlp_s0_and_priors_are_valid():
    fams = c.development_families(VIEW, 0)[:40]
    n = len(c.view_data(VIEW)["ids"])
    field = np.random.default_rng(2).normal(size=(1, 5551)).astype(np.float32)
    rows = np.zeros(n, dtype=np.int64)  # every node reads the same random field row
    zero_b1 = k.family_priors(VIEW, fams, b1=dict(beta=0.0, eps=1e-3, form="share"))
    zero_b2 = k.family_priors(VIEW, fams, b2=dict(beta=0.0), field=field, field_rows=rows)
    on = k.family_priors(VIEW, fams, b1=dict(beta=1.0, eps=1e-3, form="share"), b2=dict(beta=0.5), field=field, field_rows=rows)
    for f in fams:
        fam = c.family(VIEW, f)
        assert np.array_equal(zero_b1[int(f)], fam["priors"]["mlp_s0"])
        assert np.array_equal(zero_b2[int(f)], fam["priors"]["mlp_s0"])
        p = on[int(f)]
        inst = e.threshold_instance(fam["cells"], fam["weights"], c.thresholds(fam)[0][1])
        e.check_probabilities(inst, p)
        assert np.array_equal(p[:, 2], fam["priors"]["mlp_s0"][:, 2])
        assert not np.array_equal(p[:, 0], fam["priors"]["mlp_s0"][:, 0])


def test_beta_zero_reproduces_frozen_costs():
    from connectomequest.neurosym import harness as h

    fams = c.development_families(VIEW, 0)[:12]
    priors = k.family_priors(VIEW, fams, b1=dict(beta=0.0, eps=1e-3, form="share"))
    for f in fams:
        fam = c.family(VIEW, f)
        for rho, threshold in c.thresholds(fam):
            inst = e.threshold_instance(fam["cells"], fam["weights"], threshold)
            for policy, arm in ((e.AdaptiveGreedy, "mlp_s0:ag"), (e.RecalibratedGreedy, "mlp_s0:recal")):
                order = e.run_adaptive(inst, fam["outcomes"], policy(inst, priors[int(f)]))
                cost, _ = e.evaluate(inst, fam["outcomes"], order)
                d = h.frozen(VIEW)
                assert cost == d["cost"][d["key"][(int(f), rho)], d["arms"][arm]]
