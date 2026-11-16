import os

import numpy as np
import pytest

from connectomequest.annotation_v35 import engine as e
from connectomequest.neurosym import policy as P

import pytest as _pytest

from connectomequest.neurosym import common as _common

if not all((_common.VIEW_ROOTS[v] / "public/topology.npz").exists() and (_common.BENCH / f"{v}.npz").exists() for v in _common.VIEW_ROOTS):
    _pytest.skip("needs the connectome graphs and the benchmark arrays", allow_module_level=True)

ENGINE = {
    "ag": lambda inst, y, p: e.run_adaptive(inst, y, e.AdaptiveGreedy(inst, p)),
    "recal": lambda inst, y, p: e.run_adaptive(inst, y, e.RecalibratedGreedy(inst, p)),
    "adir": lambda inst, y, p: e.run_adaptive(inst, y, e.AdaptiveDirection(inst, p)),
    "adir-recal": lambda inst, y, p: e.run_adaptive(inst, y, e.RecalibratedDirection(inst, p)),
}


def random_question(rng, n):
    weights = rng.integers(1, 20, n)
    inst = e.threshold_instance(rng.permutation(np.arange(1, 10 * n + 1))[:n], weights, int(rng.integers(1, int(weights.sum()) + 1)))
    y = rng.choice(3, size=n, p=(0.3, 0.45, 0.25))
    p = rng.dirichlet((1.0, 1.5, 0.6), size=n)
    p[rng.random(n) < 0.2, 0] = 0.0  # exact zeros exercise the fallback
    return inst, y, p / p.sum(1, keepdims=True)


def test_engine_policies_are_special_cases_on_random_questions():
    rng = np.random.default_rng(0)
    for _ in range(300):
        inst, y, p = random_question(rng, int(rng.integers(2, 40)))
        for name, run in ENGINE.items():
            assert np.array_equal(P.run_session(inst, y, p, P.special(name)), run(inst, y, p)), name
        # the static direction order, cut at its certificate
        cost, _ = e.evaluate(inst, y, e.direction_order(inst, p))
        assert np.array_equal(P.run_session(inst, y, p, P.special("direction")), e.direction_order(inst, p)[:cost])


def test_random_parameters_certify_correctly_and_ignore_scale():
    rng = np.random.default_rng(1)
    for _ in range(150):
        inst, y, p = random_question(rng, int(rng.integers(2, 30)))
        x = rng.normal(size=P.DIM)
        params = P.decode(x)
        assert np.isclose(np.abs(params["theta"]).sum(), 1.0) and 0 <= params["recal"] <= 1
        order = P.run_session(inst, y, p, params)
        cost, status = e.evaluate(inst, y, order)
        assert status == e.full_information_status(inst, y)
        assert cost == len(order) and (status == "unresolved" or cost >= e.clairvoyant_cost(inst, y))
        scaled = dict(params, theta=3.0 * params["theta"])
        assert np.array_equal(P.run_session(inst, y, p, scaled), order)
        assert np.allclose(P.encode(params), P.encode(P.decode(P.encode(params))))


def _frozen_arms(fam, inst, y, rho):
    out = {}
    for prior in ("mlp_s0", "provisional"):
        if prior in fam["priors"]:
            for name in ("ag", "recal", "direction"):
                out[f"{prior}:{name}"] = P.run_session(inst, y, fam["priors"][prior], P.special(name))
    return out


@pytest.mark.parametrize("view", ["flywire", "malecns"])
def test_special_cases_reproduce_frozen_costs_bit_for_bit(view):
    from connectomequest.neurosym import common as c, harness as h

    families = c.development_families(view, 0)[:100]
    out = h.run(view, families, _frozen_arms, workers=min(10, len(os.sched_getaffinity(0))))
    assert {"mlp_s0:ag", "mlp_s0:recal"} <= set(out)
    for name, costs in out.items():
        assert np.array_equal(costs, h.baseline(view, families, name)), name
