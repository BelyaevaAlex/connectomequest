import numpy as np
import pytest

from connectomequest.annotation_v35 import engine as e
from connectomequest.annotation_v35 import sweep as s


def population(rng, n, member_rate=0.2, unresolved_rate=0.35):
    member = rng.random(n) < member_rate
    unresolved = ~member & (rng.random(n) < unresolved_rate)
    return np.where(member, e.MEMBER, np.where(unresolved, e.UNRESOLVED, e.NONMEMBER))


@pytest.mark.parametrize("target", [0.6, 0.7, 0.8, 0.9, 0.95])
def test_synthetic_prior_reaches_the_requested_auroc(target):
    rng = np.random.default_rng(0)
    y = population(rng, 200_000)
    p = s.synthetic_prior(y, rng.standard_normal(len(y)), target, 0.2)
    assert abs(s.member_auroc(p[:, e.MEMBER], y) - target) < 0.005


def test_synthetic_prior_is_calibrated_when_the_base_rates_are_right():
    rng = np.random.default_rng(1)
    y = population(rng, 400_000, member_rate=0.15, unresolved_rate=0.4)
    p = s.synthetic_prior(y, rng.standard_normal(len(y)), 0.85, 0.15, unresolved_rate=0.4)
    bins = np.digitize(p[:, e.MEMBER], np.linspace(0.1, 0.9, 9))
    for b in np.unique(bins):
        m = bins == b
        if m.sum() > 2000:
            assert abs((y[m] == e.MEMBER).mean() - p[m, e.MEMBER].mean()) < 0.02
            assert abs((y[m] == e.UNRESOLVED).mean() - p[m, e.UNRESOLVED].mean()) < 0.02


def test_rows_are_distributions_and_variants_share_the_membership_probability():
    rng = np.random.default_rng(2)
    y = population(rng, 1000)
    noise = rng.standard_normal(len(y))
    plain = s.synthetic_prior(y, noise, 0.8, 0.3)
    based = s.synthetic_prior(y, noise, 0.8, 0.3, unresolved_rate=0.25)
    for p in (plain, based):
        assert np.all(p >= 0) and np.allclose(p.sum(1), 1)
    assert np.allclose(plain[:, e.MEMBER], based[:, e.MEMBER])
    assert np.all(plain[:, e.UNRESOLVED] == 0)
    assert np.allclose(based[:, e.UNRESOLVED], 0.25 * (1 - based[:, e.MEMBER]))
    inst = e.threshold_instance(np.arange(1, 1001), rng.integers(1, 9, 1000), 500)
    e.check_probabilities(inst, plain)
    e.check_probabilities(inst, based)


def test_without_separation_the_prior_is_constant_and_adaptive_greedy_is_weight_order():
    rng = np.random.default_rng(3)
    for _ in range(100):
        n = int(rng.integers(2, 30))
        y = population(rng, n, member_rate=0.4)
        p = s.synthetic_prior(y, rng.standard_normal(n), 0.5, 0.4)
        assert np.allclose(p[:, e.MEMBER], 0.4)
        inst = e.threshold_instance(np.arange(1, n + 1), rng.integers(1, 9, n), 1)
        inst = e.threshold_instance(inst.cells, inst.weights, int(rng.integers(1, int(inst.weights.sum()) + 1)))
        if e.full_information_status(inst, y) == "unresolved":
            continue
        greedy = e.evaluate(inst, y, e.run_adaptive(inst, y, e.AdaptiveGreedy(inst, p)))
        assert greedy == e.evaluate(inst, y, e.weight_order(inst))


def test_unresolved_rate_keeps_the_membership_probabilities_of_an_existing_prior():
    rng = np.random.default_rng(4)
    p = rng.dirichlet((2, 3, 0.2), size=50)
    q = s.with_unresolved_rate(p, 0.37)
    assert np.allclose(q[:, e.MEMBER], p[:, e.MEMBER])
    assert np.allclose(q[:, e.UNRESOLVED], 0.37 * (1 - p[:, e.MEMBER]))
    assert np.allclose(q.sum(1), 1)


def test_member_auroc_matches_pairwise_counting_with_ties():
    rng = np.random.default_rng(5)
    for _ in range(50):
        n = int(rng.integers(2, 40))
        y = rng.choice(3, size=n)
        plus = rng.integers(0, 5, size=n) / 4.0
        positive = y == e.MEMBER
        if positive.all() or not positive.any():
            assert s.member_auroc(plus, y) is None
            continue
        pairs = [(1.0 if a > b else 0.5 if a == b else 0.0) for a in plus[positive] for b in plus[~positive]]
        assert abs(s.member_auroc(plus, y) - np.mean(pairs)) < 1e-12


@pytest.mark.parametrize("target", [0.6, 0.7, 0.8, 0.9, 0.95])
def test_noisy_oracle_reaches_the_requested_auroc_and_is_confident(target):
    rng = np.random.default_rng(6)
    y = population(rng, 200_000)
    p = s.oracle_prior(y, rng.random(len(y)), rng.random(len(y)), target)
    assert abs(s.member_auroc(p[:, e.MEMBER], y) - target) < 0.005
    assert np.allclose(p.sum(1), 1) and np.allclose(p.max(1), 0.9)
    named = p.argmax(1)
    assert abs((named != y).mean() - s.flip_probability(target)) < 0.005
    wrong = named != y
    for k in range(3):
        assert abs((named[wrong & (y == k)] == (k + 1) % 3).mean() - 0.5) < 0.02


def test_noisy_oracle_flips_are_nested_across_levels():
    rng = np.random.default_rng(7)
    y = population(rng, 5000)
    u, v = rng.random(len(y)), rng.random(len(y))
    wrong = [s.oracle_prior(y, u, v, a).argmax(1) != y for a in (0.6, 0.7, 0.8, 0.9, 0.95)]
    for lower, higher in zip(wrong, wrong[1:]):
        assert np.all(higher <= lower)
