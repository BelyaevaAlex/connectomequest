import numpy as np
import pytest

from connectomequest.annotation_v35 import ablation as ab
from connectomequest.annotation_v35 import engine as e


def random_question(rng, n):
    weights = rng.integers(1, 20, n)
    inst = e.threshold_instance(np.arange(1, n + 1), weights, int(rng.integers(1, int(weights.sum()) + 1)))
    y = rng.choice(3, size=n, p=(0.3, 0.45, 0.25))
    p = rng.dirichlet((1.0, 1.5, 0.6), size=n)
    return inst, y, p


def test_frozen_choices_reproduce_the_frozen_recalibration():
    rng = np.random.default_rng(0)
    for _ in range(300):
        inst, y, p = random_question(rng, int(rng.integers(2, 40)))
        frozen = e.run_adaptive(inst, y, e.RecalibratedGreedy(inst, p))
        ablated = e.run_adaptive(inst, y, ab.AblatedRecalibration(inst, p, **ab.RECAL_VARIANTS["frozen"]))
        assert np.array_equal(frozen, ablated)


def test_every_variant_keeps_distributions_and_certifies_correctly():
    rng = np.random.default_rng(1)
    for _ in range(100):
        inst, y, p = random_question(rng, int(rng.integers(2, 30)))
        truth = e.full_information_status(inst, y)
        for settings in ab.RECAL_VARIANTS.values():
            policy = ab.AblatedRecalibration(inst, p, **settings)
            order = e.run_adaptive(inst, y, policy)
            assert len(set(order.tolist())) == len(order)
            assert np.allclose(policy.p.sum(1), 1) and np.all(policy.p >= -1e-12)
            assert e.evaluate(inst, y, order)[1] == truth


def test_infinite_strength_with_prior_unresolved_rate_is_adaptive_greedy():
    rng = np.random.default_rng(2)
    for _ in range(200):
        inst, y, p = random_question(rng, int(rng.integers(2, 30)))
        greedy = e.run_adaptive(inst, y, e.AdaptiveGreedy(inst, p))
        frozen_prior = e.run_adaptive(inst, y, ab.AblatedRecalibration(inst, p, kappa=1e15, unresolved="prior"))
        assert np.array_equal(greedy, frozen_prior)


def test_group_edges_define_the_groups():
    inst, _, _ = random_question(np.random.default_rng(3), 6)
    p = np.array([[0.01, 0.99, 0], [0.1, 0.9, 0], [0.3, 0.7, 0], [0.6, 0.4, 0], [0.9, 0.1, 0], [0.99, 0.01, 0]])
    assert ab.AblatedRecalibration(inst, p, edges=())._group.tolist() == [0] * 6
    assert ab.AblatedRecalibration(inst, p, edges=(0.25, 0.75))._group.tolist() == [0, 0, 1, 1, 2, 2]
    assert ab.AblatedRecalibration(inst, p)._group.tolist() == [0, 1, 2, 2, 3, 4]


def mixture(rng, n, alpha=0.9):
    topology = rng.dirichlet((1.0, 4.0, 0.5), size=n)
    labelled = rng.random(n) < 0.6
    cls = rng.choice(3, size=n)
    stored = ab.earlier_release_prior(topology, labelled, cls, alpha=alpha).astype(np.float32)
    return topology.astype(np.float32), stored, labelled, cls


def test_earlier_labels_are_recovered_from_the_stored_mixture():
    rng = np.random.default_rng(4)
    topology, stored, labelled, cls = mixture(rng, 5000)
    found, found_cls = ab.earlier_release_parts(stored[:, 0], stored[:, 2], topology[:, 0], topology[:, 2])
    assert np.array_equal(found, labelled)
    assert np.array_equal(found_cls[labelled], cls[labelled])
    rebuilt = ab.earlier_release_prior(topology, found, found_cls, alpha=0.9)
    assert np.max(np.abs(rebuilt - stored)) < 1e-6


def test_a_prior_that_is_not_the_mixture_is_rejected():
    rng = np.random.default_rng(5)
    topology, stored, _, _ = mixture(rng, 200)
    with pytest.raises(ValueError):
        ab.earlier_release_parts(stored[:, 0], stored[:, 2], topology[:, 0], topology[:, 2], alpha=0.8)


def test_rebuilt_priors_are_distributions_and_class_rates_are_shares():
    rng = np.random.default_rng(6)
    topology, _, labelled, cls = mixture(rng, 1000)
    for alpha in (0.8, 0.95, 0.99):
        q = ab.earlier_release_prior(topology, labelled, cls, alpha=alpha)
        assert np.allclose(q.sum(1), 1) and np.allclose(q[~labelled], topology[~labelled])
    y = rng.choice(3, size=1000)
    rates = ab.class_rates(cls, y)
    assert np.allclose(rates.sum(1), 1)
    assert np.allclose(rates[1], np.bincount(y[cls == 1], minlength=3) / (cls == 1).sum())
    q = ab.earlier_release_prior(topology, labelled, cls, rates=rates)
    assert np.allclose(q[labelled], rates[cls[labelled]])
    with pytest.raises(ValueError):
        ab.earlier_release_prior(topology, labelled, cls)


def test_first_step_of_the_worked_example():
    inst = e.threshold_instance([1, 2, 3, 4], [1, 2, 3, 4], 5)
    p = np.array([[0.5, 0.5, 0], [0.8, 0.2, 0], [0.05, 0.95, 0], [0.5, 0.5, 0]])
    policy = e.AdaptiveGreedy(inst, p)
    assert (inst.a0, inst.b0) == (5, 6)
    assert np.allclose(policy._gain(5, 6)[[3, 2, 1, 0]], (22, 15.15, 11.6, 5.5))
    assert policy.choose(5, 6, np.ones(4, dtype=bool)) == 3
