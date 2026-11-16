import importlib
import sys
from fractions import Fraction
from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np
import pytest

from connectomequest.annotation_v35 import engine as e
from connectomequest.annotation_v35.statistical import sequential_certificate

try:  # The two pilot implementations are compared cell by cell when they are present.
    from connectomequest.annotation_v33 import policies as v33
    from connectomequest.annotation_v33.session import run_session as v33_run_session
    from connectomequest.annotation_v33.weighted_query import WeightedQuery
    from connectomequest.annotation_v34 import policies as v34

    PILOT = True
except ImportError:
    PILOT = False
needs_pilot = pytest.mark.skipif(not PILOT, reason="pilot implementations are not part of this archive")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def random_threshold(rng, n, num=1, den=2):
    cells = rng.choice(10_000, size=n, replace=False) + 1
    weights = rng.integers(1, 9, size=n)
    threshold, _ = e.threshold_goals(int(weights.sum()), num, den)
    raw = rng.dirichlet((1, 1, 0.3), size=n)
    outcomes = np.array([rng.choice(3, p=row) for row in raw])
    return e.threshold_instance(cells, weights, threshold), raw, outcomes


def test_threshold_goals_are_exact_and_reject_unsupported_questions():
    assert e.threshold_goals(10, 1, 2) == (5, 6)
    assert e.threshold_goals(11, 1, 2) == (6, 6)
    assert e.threshold_goals(7, 3, 10) == (3, 5)
    assert e.threshold_goals(10**15 + 1, 1, 3) == ((10**15 + 1 + 2) // 3, 10**15 + 1 - (10**15 + 3) // 3 + 1)
    for bad in ((0, 1, 2), (5, 0, 2), (5, 3, 2), (5, 1, 0)):
        with pytest.raises(ValueError):
            e.threshold_goals(*bad)
    with pytest.raises(ValueError):
        e.threshold_goals(5, True, 2)


def test_verifier_matches_bounds_definition_on_every_prefix():
    rng = np.random.default_rng(0)
    for _ in range(200):
        inst, _, y = random_threshold(rng, int(rng.integers(1, 9)), int(rng.integers(1, 4)), 4)
        order = rng.permutation(inst.size)
        total, threshold = int(inst.weights.sum()), inst.a0
        expected = (inst.size, "unresolved")
        lower, upper = 0, total
        for step, j in enumerate(order, 1):
            lower += int(inst.weights[j]) * (y[j] == e.MEMBER)
            upper -= int(inst.weights[j]) * (y[j] == e.NONMEMBER)
            if lower >= threshold or upper < threshold:
                expected = (step, "positive" if lower >= threshold else "negative")
                break
        assert e.evaluate(inst, y, order) == expected
        assert e.full_information_status(inst, y) == e.evaluate(inst, y, np.arange(inst.size))[1]


def test_verifier_rejects_repeated_or_foreign_payments():
    inst = e.threshold_instance([5, 6, 7], [1, 2, 3], 3)
    with pytest.raises(ValueError):
        e.evaluate(inst, [0, 0, 0], [0, 0])
    with pytest.raises(ValueError):
        e.evaluate(inst, [0, 0, 0], [0, 3])
    with pytest.raises(ValueError):
        e.evaluate(inst, [0, 0, 3], [0])


def test_clairvoyant_cost_equals_smallest_certifying_subset():
    rng = np.random.default_rng(1)
    for _ in range(150):
        inst, _, y = random_threshold(rng, int(rng.integers(1, 8)), int(rng.integers(1, 4)), 4)
        best = None
        for size in range(1, inst.size + 1):
            if any(e.evaluate(inst, y, subset)[1] != "unresolved" for subset in combinations(range(inst.size), size)):
                best = size
                break
        assert e.clairvoyant_cost(inst, y) == best
        if best is None:
            assert e.full_information_status(inst, y) == "unresolved"


def test_worked_example_of_the_manuscript():
    inst = e.threshold_instance([1, 2, 3, 4], [1, 2, 3, 4], 5)
    p = np.array([[0.5, 0.5, 0], [0.8, 0.2, 0], [0.05, 0.95, 0], [0.5, 0.5, 0]])
    policy = e.AdaptiveGreedy(inst, p)
    available = np.array([True, True, True, False])
    assert np.allclose(policy._gain(1, 6)[[2, 1, 0]], (3.15, 5.20, 3.50))
    assert policy.choose(1, 6, available) == 1
    assert np.allclose(policy._gain(5, 2)[[2, 1, 0]], (9.80, 5.20, 3.50))
    assert policy.choose(5, 2, available) == 2


class _RowPrior:
    def __init__(self, cells, rows):
        self.rows = {int(c): r for c, r in zip(cells, rows)}

    def outcome_probabilities(self, query, cells, observed):
        return np.array([self.rows[int(c)] for c in cells])


def _v33_order(inst, raw, y, policy, num, den):
    labels = {int(c): (7 if k == 0 else 9 if k == 1 else None) for c, k in zip(inst.cells, y)}
    query = WeightedQuery(
        query_id="q",
        weights=tuple(sorted((int(c), int(w)) for c, w in zip(inst.cells, inst.weights))),
        members=(7,),
        threshold_num=num,
        threshold_den=den,
        metadata={},
    )
    journal = v33_run_session((query,), policy, labels.__getitem__, _RowPrior(inst.cells, raw), budget=inst.size)
    index = {int(c): i for i, c in enumerate(inst.cells)}
    return [index[ev["cell"]] for ev in journal["events"]], journal


@needs_pilot
def test_threshold_policies_reproduce_the_frozen_v33_implementation():
    control = importlib.import_module("v33_direction_control")
    rng = np.random.default_rng(2)
    for trial in range(60):
        num = int(rng.integers(1, 4))
        inst, raw, y = random_threshold(rng, int(rng.integers(2, 14)), num, 4)
        mine = {
            "weight": e.weight_order(inst),
            "member": e.positive_order(inst, raw),
            "nonmember": e.negative_order(inst, raw),
            "direction": e.direction_order(inst, raw),
            "ag": e.run_adaptive(inst, y, e.AdaptiveGreedy(inst, raw)),
        }
        theirs = {
            "weight": v33.DecreasingWeightPolicy(),
            "member": v33.FixedPriorPolicy("member", unknown_codes=(0,)),
            "nonmember": v33.FixedPriorPolicy("nonmember", unknown_codes=(0,)),
            "direction": control.DirectionFixedPolicy(),
            "ag": v33.AdaptiveGoalPolicy(unknown_codes=(0,)),
        }
        for name, order in mine.items():
            cost, status = e.evaluate(inst, y, order)
            paid, journal = _v33_order(inst, raw, y, theirs[name], num, 4)
            assert paid == list(order[:cost]), (trial, name)
            assert journal["cost"] == cost
            assert journal["resolved_queries"] == int(status != "unresolved")


def _comparison_instance(cells, weights):
    w = np.asarray(weights, dtype=np.int64)
    zero = np.zeros_like(w)
    total = int(w.sum())
    return e.Instance(cells, w, np.column_stack((2 * w, zero, w, zero)), np.column_stack((zero, 2 * w, w, zero)), total + 1, total)


class _Wrapped:
    def __init__(self, policy):
        self.policy = policy

    def choose(self, a, b, available):
        return self.policy.choose(int(a), int(b), available)

    def observe(self, cell, outcome):
        pass


@needs_pilot
def test_generic_engine_reproduces_frozen_v34_comparison_policies():
    ggn = importlib.import_module("v34_ggn_control")
    rng = np.random.default_rng(3)
    for trial in range(40):
        n = int(rng.integers(2, 90 if trial % 8 == 0 else 16))
        cells = rng.choice(10_000, size=n, replace=False) + 1
        weights = rng.integers(1, 9, size=n)
        p = rng.dirichlet((1, 1, 1, 0.3), size=n)
        y = np.array([rng.choice(4, p=row) for row in p])
        inst = _comparison_instance(cells, weights)
        args = (weights, tuple(int(c) for c in cells), p)
        pairs = [
            (e.run_adaptive(inst, y, e.AdaptiveGreedy(inst, p)), v34.AdaptiveGoalPolicy(*args)),
            (e.run_adaptive(inst, y, e.AdaptiveDualGreedy(inst, p)), v34.AdaptiveDualGreedyPolicy(*args)),
            (e.weight_order(inst), v34.FixedWeightPolicy(*args)),
            (e.positive_order(inst, p), v34.FixedExpectedPolicy(*args, direction="positive")),
            (e.negative_order(inst, p), v34.FixedExpectedPolicy(*args, direction="negative")),
            (e.direction_order(inst, p), v34.DirectionFixedPolicy(*args)),
            (e.ggn_static_order(inst, p), ggn.GGNStaticPolicy(*args)),
        ]
        for mine, theirs in pairs:
            cost, _ = e.evaluate(inst, y, mine)
            reference = e.run_adaptive(inst, y, _Wrapped(theirs))
            assert list(mine[:cost]) == list(reference), (trial, type(theirs).__name__)


def test_adaptive_dual_greedy_matches_rational_full_history_reference():
    rng = np.random.default_rng(4)
    for _ in range(40):
        n = int(rng.integers(2, 7))
        cells = [int(c) for c in rng.choice(100, size=n, replace=False) + 1]
        weights = [int(w) for w in rng.integers(1, 6, size=n)]
        raw = [[Fraction(int(x), 10) for x in row] for row in rng.multinomial(10, (0.45, 0.45, 0.1), size=n)]
        threshold, z = e.threshold_goals(sum(weights), 1, 2)
        inst = e.threshold_instance(cells, weights, threshold)
        p = np.array([[float(x) for x in row] for row in raw])
        y = np.array([int(rng.integers(0, 3)) for _ in range(n)])

        def gain(a, b, j):
            return raw[j][0] * min(weights[j], a) * b + raw[j][1] * a * min(weights[j], b)

        a, b, remaining, past, trace = threshold, z, set(range(n)), [], []
        while remaining and a > 0 and b > 0:
            m = {j: gain(a, b, j) for j in remaining}
            positive = [j for j in remaining if m[j] > 0]
            if positive:
                r = {j: 1 - sum(step * gain(aa, bb, j) for aa, bb, step in past) for j in positive}
                j = min(positive, key=lambda k: (r[k] / m[k], cells[k]))
                past.append((a, b, r[j] / m[j]))
            else:
                j = min(remaining, key=lambda k: (-weights[k], cells[k]))
            trace.append(j)
            remaining.remove(j)
            a -= weights[j] * (y[j] == 0)
            b -= weights[j] * (y[j] == 1)
        assert list(e.run_adaptive(inst, y, e.AdaptiveDualGreedy(inst, p))) == trace


def test_best_fixed_order_cost_on_the_small_disjunction_example():
    # a OR (b AND c): weights (2,1,1), threshold 2; adaptive optimum starts with b.
    inst = e.threshold_instance([1, 2, 3], [2, 1, 1], 2)
    p = np.array([[0.1, 0.9, 0], [0.5, 0.5, 0], [0.9, 0.1, 0]])

    def expected(order_fn):
        value = Fraction(0)
        for y in product((0, 1), repeat=3):
            prob = np.prod([p[j, y[j]] for j in range(3)])
            value += Fraction(float(prob)).limit_denominator(10**6) * e.evaluate(inst, np.array(y), order_fn(np.array(y)))[0]
        return value

    best_fixed = min(expected(lambda y, o=o: np.array(o)) for o in permutations(range(3)))
    assert best_fixed == Fraction(47, 20)


def test_recalibrated_greedy_starts_as_greedy_and_stays_normalized():
    rng = np.random.default_rng(5)
    inst, raw, y = random_threshold(rng, 30)
    static, online = e.AdaptiveGreedy(inst, raw), e.RecalibratedGreedy(inst, raw)
    available = np.ones(inst.size, dtype=bool)
    assert static.choose(inst.a0, inst.b0, available) == online.choose(inst.a0, inst.b0, available)
    order = e.run_adaptive(inst, y, online)
    assert np.allclose(online.p.sum(1), 1) and np.all(online.p >= 0)
    assert len(set(order.tolist())) == len(order)
    # Paid members in a group raise the member belief of its unread cells.
    inst2 = e.threshold_instance(np.arange(1, 41), np.ones(40, dtype=int), 30)
    prior = np.tile([0.5, 0.5, 0.0], (40, 1))
    policy = e.RecalibratedGreedy(inst2, prior)
    for j in range(10):
        policy.observe(j, e.MEMBER)
    assert policy.p[20, e.MEMBER] > 0.75
    for j in range(10, 20):
        policy.observe(j, e.UNRESOLVED)
    assert 0.3 < policy.p[30, e.UNRESOLVED] < 0.6


def test_every_order_and_policy_pays_distinct_cells_and_agrees_with_full_information():
    rng = np.random.default_rng(6)
    for _ in range(40):
        inst, raw, y = random_threshold(rng, int(rng.integers(1, 80)), int(rng.integers(1, 4)), 4)
        truth = e.full_information_status(inst, y)
        orders = [
            e.weight_order(inst),
            e.random_order(inst, rng),
            e.positive_order(inst, raw),
            e.negative_order(inst, raw),
            e.confident_order(inst, raw),
            e.direction_order(inst, raw),
            e.ggn_static_order(inst, raw),
            e.run_adaptive(inst, y, e.AdaptiveGreedy(inst, raw)),
            e.run_adaptive(inst, y, e.AdaptiveDualGreedy(inst, raw)),
            e.run_adaptive(inst, y, e.RecalibratedGreedy(inst, raw)),
            e.run_adaptive(inst, y, e.AdaptiveDirection(inst, raw)),
            e.run_adaptive(inst, y, e.RecalibratedDirection(inst, raw)),
        ]
        bound = e.clairvoyant_cost(inst, y)
        for order in orders:
            cost, status = e.evaluate(inst, y, order)
            assert status == truth
            assert cost == inst.size if truth == "unresolved" else cost >= bound


def test_sequential_certificate_controls_wrong_certificates_under_both_nulls():
    rng = np.random.default_rng(7)
    weights = np.ones(400, dtype=np.int64)
    wrong = 0
    # Member weight just below the threshold: a positive certificate is an error.
    y = np.array([e.MEMBER] * 199 + [e.NONMEMBER] * 100 + [e.UNRESOLVED] * 101)
    for _ in range(400):
        cost, status, exact = sequential_certificate(weights, y, 200, alpha=0.05, rng=rng)
        wrong += status == "positive"
        assert status != "negative" or exact is False
    assert wrong / 400 <= 0.05 + 0.03
    # Upper bound exactly at the threshold: a negative certificate is an error.
    wrong = 0
    y = np.array([e.MEMBER] * 150 + [e.UNRESOLVED] * 50 + [e.NONMEMBER] * 200)
    for _ in range(400):
        cost, status, exact = sequential_certificate(weights, y, 200, alpha=0.05, rng=rng)
        wrong += status == "negative"
    assert wrong / 400 <= 0.05 + 0.03


def test_sequential_certificate_is_cheap_far_from_the_threshold_and_exact_at_extremes():
    rng = np.random.default_rng(8)
    weights = rng.integers(1, 20, size=600)
    y = np.where(rng.random(600) < 0.8, e.MEMBER, e.NONMEMBER)
    threshold, _ = e.threshold_goals(int(weights.sum()), 1, 2)
    costs = []
    for _ in range(50):
        cost, status, _ = sequential_certificate(weights, y, threshold, alpha=0.05, rng=rng)
        assert status == "positive"
        costs.append(cost)
    inst = e.threshold_instance(np.arange(1, 601), weights, threshold)
    assert np.mean(costs) < 0.25 * e.clairvoyant_cost(inst, y)
    # Threshold of one synapse: the first member is an exact certificate.
    cost, status, exact = sequential_certificate(weights, y, 1, alpha=0.05, rng=rng)
    assert status == "positive" and exact
    # Unresolvable at full information: every cell is paid and nothing is certified.
    y3 = np.full(50, e.UNRESOLVED)
    cost, status, exact = sequential_certificate(np.ones(50, dtype=np.int64), y3, 25, alpha=0.05, rng=rng)
    assert (cost, status, exact) == (50, "unresolved", True)


def test_adaptive_direction_pursues_the_reachable_goal_only():
    # Threshold 3 of W=40: the positive goal needs one likely member; the negative goal is far.
    weights = np.array([10, 10, 10, 3, 3, 4])
    inst = e.threshold_instance(np.arange(1, 7), weights, 3)
    p = np.array([[0.0, 1.0, 0.0]] * 3 + [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.1, 0.9, 0.0]])
    available = np.ones(6, dtype=bool)
    # The two-goal utility is drawn to heavy sure nonmembers; the direction policy is not.
    assert e.AdaptiveGreedy(inst, p).choose(inst.a0, inst.b0, available) == 3
    assert e.AdaptiveDirection(inst, p).choose(inst.a0, inst.b0, available) == 3
    far = e.threshold_instance(np.arange(1, 7), weights, 14)
    assert e.AdaptiveGreedy(far, p).choose(far.a0, far.b0, available) in (0, 1, 2)
    assert e.AdaptiveDirection(far, p).choose(far.a0, far.b0, available) in (0, 1, 2)
    # Neither goal is reachable in expectation at threshold 6 (5.5 < 6 and 34.5 < 35): tie, positive.
    near = e.threshold_instance(np.arange(1, 7), weights, 6)
    policy = e.AdaptiveDirection(near, p)
    positive, negative = policy._directional(near.a0, near.b0)
    assert policy._needed(positive, near.a0) == 7 and policy._needed(negative, near.b0) == 7
    assert policy.choose(near.a0, near.b0, available) == 3
    # With no positive mass anywhere the policy falls back to the negative goal.
    none = np.tile([0.0, 1.0, 0.0], (6, 1))
    assert e.AdaptiveDirection(near, none).choose(near.a0, near.b0, available) == 0


def test_adaptive_greedy_hedges_even_with_a_perfect_prior():
    """m unit members with T = m and m - 1 nonmembers of weight 2: "no" is unreachable,
    yet the two-goal utility ranks every nonmember first, so AG pays 2m - 1 cells
    where m suffice. The manuscript states this example."""
    for m in (3, 5, 12, 40):
        weights = np.array([1] * m + [2] * (m - 1))
        cells = np.arange(1, len(weights) + 1)
        outcomes = np.array([e.MEMBER] * m + [e.NONMEMBER] * (m - 1))
        inst = e.threshold_instance(cells, weights, m)
        perfect = np.zeros((len(weights), 3))
        perfect[np.arange(len(weights)), outcomes] = 1.0
        assert inst.b0 == 2 * m - 1 > int(weights[m:].sum())
        assert e.clairvoyant_cost(inst, outcomes) == m
        order = e.run_adaptive(inst, outcomes, e.AdaptiveGreedy(inst, perfect))
        assert e.evaluate(inst, outcomes, order) == (2 * m - 1, "positive")
        assert set(order[: m - 1].tolist()) == set(range(m, 2 * m - 1))
        assert e.evaluate(inst, outcomes, e.direction_order(inst, perfect)) == (m, "positive")
