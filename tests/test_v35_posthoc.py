from itertools import combinations

import numpy as np

from connectomequest.annotation_v35 import engine as e
from connectomequest.annotation_v35 import posthoc as p


def random_question(rng, n):
    cells = rng.choice(10_000, size=n, replace=False) + 1
    weights = rng.integers(1, 9, size=n)
    threshold = int(rng.integers(1, int(weights.sum()) + 1))
    outcomes = rng.choice(3, size=n, p=(0.35, 0.35, 0.3))
    return e.threshold_instance(cells, weights, threshold), outcomes


def test_three_way_stop_agrees_with_the_frozen_verifier_on_certifiable_questions():
    rng = np.random.default_rng(0)
    seen = {"positive": 0, "negative": 0, "undecidable": 0}
    for _ in range(600):
        inst, y = random_question(rng, int(rng.integers(1, 12)))
        order = rng.permutation(inst.size)
        cost, status = e.evaluate(inst, y, order)
        cost3, status3 = p.evaluate_three_way(inst, y, order)
        if status == "unresolved":
            assert status3 == "undecidable" and cost3 <= cost
        else:
            assert (cost3, status3) == (cost, status)
        seen[status3] += 1
    assert min(seen.values()) > 20


def test_undecidability_is_proven_exactly_when_no_completion_can_certify():
    """Brute force over every typing of the unchecked cells."""
    rng = np.random.default_rng(1)
    for _ in range(300):
        inst, y = random_question(rng, int(rng.integers(1, 8)))
        order = rng.permutation(inst.size)
        total = int(inst.weights.sum())
        for k in range(1, inst.size + 1):
            checked = order[:k]
            member = int(inst.weights[checked][y[checked] == e.MEMBER].sum())
            nonmember = int(inst.weights[checked][y[checked] == e.NONMEMBER].sum())
            unchecked = total - int(inst.weights[checked].sum())
            # A completion may type every unchecked cell freely; the best cases bound both goals.
            can_yes = member + unchecked >= inst.a0
            can_no = nonmember + unchecked >= inst.b0
            cost3, status3 = p.evaluate_three_way(inst, y, order[:k])
            done = member >= inst.a0 or nonmember >= inst.b0 or not (can_yes or can_no)
            assert (status3 != "open") == any(
                int(inst.weights[order[:i]][y[order[:i]] == e.MEMBER].sum()) >= inst.a0
                or int(inst.weights[order[:i]][y[order[:i]] == e.NONMEMBER].sum()) >= inst.b0
                or not (
                    int(inst.weights[order[:i]][y[order[:i]] == e.MEMBER].sum()) + total - int(inst.weights[order[:i]].sum()) >= inst.a0
                    or int(inst.weights[order[:i]][y[order[:i]] == e.NONMEMBER].sum()) + total - int(inst.weights[order[:i]].sum()) >= inst.b0
                )
                for i in range(1, k + 1)
            )
            if done:
                assert status3 != "open" and cost3 <= k


def test_the_example_of_one_heavy_unresolved_cell():
    inst = e.threshold_instance(np.array([1, 2, 3]), np.array([6, 2, 2]), 5)
    y = np.array([e.UNRESOLVED, e.MEMBER, e.NONMEMBER])
    assert p.evaluate_three_way(inst, y, np.array([0, 1, 2])) == (1, "undecidable")
    assert e.evaluate(inst, y, np.array([0, 1, 2])) == (3, "unresolved")
    assert p.undecidability_lower_bound(inst, y) == 1


def test_lower_bound_never_exceeds_the_best_proof_found_by_exhaustive_search():
    rng = np.random.default_rng(2)
    checked = 0
    for _ in range(400):
        inst, y = random_question(rng, int(rng.integers(1, 9)))
        bound = p.undecidability_lower_bound(inst, y)
        if bound is None:
            assert e.full_information_status(inst, y) != "unresolved"
            continue
        best = None
        for k in range(1, inst.size + 1):
            if any(p.evaluate_three_way(inst, y, np.array(subset))[1] == "undecidable" for subset in combinations(range(inst.size), k)):
                best = k
                break
        assert best is not None and bound <= best
        checked += 1
    assert checked > 50


def test_round_robin_alternates_and_lists_every_cell_once():
    rng = np.random.default_rng(3)
    for _ in range(100):
        inst, _ = random_question(rng, int(rng.integers(1, 15)))
        prob = rng.dirichlet((1, 1, 0.3), size=inst.size)
        order = p.round_robin_order(inst, prob)
        assert sorted(order.tolist()) == list(range(inst.size))
        assert order[0] == e.positive_order(inst, prob)[0]
        if inst.size > 1:
            negative = e.negative_order(inst, prob)
            assert order[1] == next(c for c in negative if c != order[0])


def test_round_robin_costs_at_most_twice_the_cheaper_fixed_order():
    rng = np.random.default_rng(4)
    for _ in range(300):
        inst, y = random_question(rng, int(rng.integers(1, 15)))
        prob = rng.dirichlet((1, 1, 0.3), size=inst.size)
        if e.full_information_status(inst, y) == "unresolved":
            continue
        cheaper = min(e.evaluate(inst, y, e.positive_order(inst, prob))[0], e.evaluate(inst, y, e.negative_order(inst, prob))[0])
        assert e.evaluate(inst, y, p.round_robin_order(inst, prob))[0] <= 2 * cheaper


def test_hybrid_never_contradicts_an_exact_certificate_more_often_than_its_level():
    rng = np.random.default_rng(5)
    wrong = audits = 0
    for trial in range(400):
        n = int(rng.integers(20, 60))
        cells = np.arange(1, n + 1)
        weights = rng.integers(1, 6, size=n)
        outcomes = rng.choice(3, size=n, p=(0.4, 0.4, 0.2))
        total = int(weights.sum())
        member = int(weights[outcomes == e.MEMBER].sum())
        upper = total - int(weights[outcomes == e.NONMEMBER].sum())
        # Thresholds just beyond the truth: every statistical certificate would be wrong.
        for threshold in {min(total, member + 1), max(1, min(total, upper))}:
            inst = e.threshold_instance(cells, weights, threshold)
            truth = e.full_information_status(inst, outcomes)
            for repeat in range(3):
                cost, status, exact = p.hybrid_certificate(inst, outcomes, alpha=0.05, rng=np.random.default_rng([trial, threshold, repeat]))
                assert 1 <= cost <= n
                if exact:
                    assert status == truth
                audits += 1
                wrong += status not in ("unresolved", truth)
    assert audits > 1500 and wrong / audits <= 0.05


def test_hybrid_equals_weight_order_when_the_heavy_half_decides():
    rng = np.random.default_rng(6)
    for _ in range(200):
        inst, y = random_question(rng, int(rng.integers(4, 30)))
        cost, status = e.evaluate(inst, y, e.weight_order(inst))
        order = e.weight_order(inst)
        half = int(np.searchsorted(np.cumsum(inst.weights[order]) * 2, int(inst.weights.sum()), side="left")) + 1
        if status != "unresolved" and cost <= half:
            assert p.hybrid_certificate(inst, y, rng=np.random.default_rng(0)) == (cost, status, True)
