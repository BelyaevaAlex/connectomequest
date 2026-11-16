"""Tests of the analysis of missing type correspondences (scripts/correspondences.py); no benchmark data needed."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import correspondences as c  # noqa: E402
import correspondences_readings as readings  # noqa: E402
from connectomequest.annotation_v35 import engine as e  # noqa: E402


def random_question(rng):
    """Input cells with outcomes, a threshold, and a release-specific type for some unresolved cells."""
    n = int(rng.integers(3, 9))
    weights = rng.integers(1, 12, size=n)
    outcomes = rng.choice([e.MEMBER, e.NONMEMBER, e.UNRESOLVED], size=n, p=[0.3, 0.2, 0.5])
    kind = np.where(outcomes == e.UNRESOLVED, rng.integers(-1, 3, size=n), -1)
    threshold = int(rng.integers(1, weights.sum() + 1))
    return weights, outcomes, kind, threshold


def status(weights, outcomes, threshold):
    return e.full_information_status(e.threshold_instance(np.arange(len(weights)), weights, threshold), outcomes)


def test_single_type_criteria_agree_with_the_engine():
    rng = np.random.default_rng(0)
    seen = dict(alone=0, decided=0, questions=0)
    for _ in range(3000):
        weights, outcomes, kind, threshold = random_question(rng)
        if status(weights, outcomes, threshold) != "unresolved":
            continue
        seen["questions"] += 1
        lower = int(weights[outcomes == e.MEMBER].sum())
        upper = lower + int(weights[outcomes == e.UNRESOLVED].sum())
        for k in range(3):
            w = int(weights[kind == k].sum())
            if not w:
                continue
            as_nonmember = status(weights, np.where(kind == k, e.NONMEMBER, outcomes), threshold)
            as_member = status(weights, np.where(kind == k, e.MEMBER, outcomes), threshold)
            alone = c.certified_alone(lower, upper, threshold, w)
            decided = c.decided_alone(lower, upper, threshold, w)
            assert alone == (as_nonmember == "negative")
            assert decided == (as_nonmember == "negative" and as_member == "positive")
            assert as_nonmember != "positive" and as_member != "negative"  # a declaration never certifies the other answer
            seen["alone"] += alone
            seen["decided"] += decided
    assert seen["questions"] > 500 and seen["alone"] > 50 and seen["decided"] > 10


def test_both_readings_of_a_type_agree_with_the_engine():
    rng = np.random.default_rng(5)
    seen = dict(inside=0, outside=0)
    for _ in range(2000):
        weights, outcomes, kind, threshold = random_question(rng)
        if status(weights, outcomes, threshold) != "unresolved":
            continue
        lower = int(weights[outcomes == e.MEMBER].sum())
        upper = lower + int(weights[outcomes == e.UNRESOLVED].sum())
        for k in range(3):
            w = int(weights[kind == k].sum())
            if not w:
                continue
            inside = readings.certified_inside(lower, threshold, w)
            outside = readings.certified_outside(upper, threshold, w)
            assert inside == (status(weights, np.where(kind == k, e.MEMBER, outcomes), threshold) == "positive")
            assert outside == (status(weights, np.where(kind == k, e.NONMEMBER, outcomes), threshold) == "negative")
            assert (inside and outside) == c.decided_alone(lower, upper, threshold, w)
            seen["inside"] += inside
            seen["outside"] += outside
    assert seen["inside"] > 50 and seen["outside"] > 50


def test_split_types_are_named_after_a_hemibrain_type_with_a_suffix():
    vocabulary = {"LC10": 1, "LT1": 2, "LC1": 3, "CL085": 4}
    assert readings.split_parent("LC10a", vocabulary) == "LC10"  # the longest hemibrain name, not LC1
    assert readings.split_parent("LC10c-2", vocabulary) == "LC10"
    assert readings.split_parent("CL085_b", vocabulary) == "CL085"
    assert readings.split_parent("LC10", vocabulary) is None  # a hemibrain type itself is not a split type
    assert readings.split_parent("LT1B", vocabulary) is None and readings.split_parent("Tm12", vocabulary) is None


def test_types_needed_is_the_smallest_certifying_set():
    rng = np.random.default_rng(1)
    for _ in range(300):
        weights = rng.integers(1, 9, size=int(rng.integers(1, 6))).tolist()
        deficit = int(rng.integers(1, 30))
        sizes = [k for k in range(1, len(weights) + 1) for chosen in itertools.combinations(weights, k) if sum(chosen) >= deficit]
        assert c.types_needed(deficit, weights) == (min(sizes) if sizes else 0)


def pairs_of(per_question):
    question = np.array([q for q, row in enumerate(per_question) for _ in row], dtype=np.int64)
    kind = np.array([k for row in per_question for k in row], dtype=np.int64)
    weight = np.array([w for row in per_question for w in row.values()], dtype=np.int64)
    return question, kind, weight


def test_priority_list_on_a_worked_example():
    # deficits 5, 4, 6, 3; type 0 certifies questions 0 and 1 alone, type 1 certifies question 2, question 3 is out of reach
    per_question = [{0: 5, 1: 1}, {0: 4}, {1: 6, 2: 2}, {2: 2}]
    deficit = np.array([5, 4, 6, 3])
    question, kind, weight = pairs_of(per_question)
    order, certified = c.greedy_list(question, kind, weight, deficit, 3, ["a", "b", "c"])
    assert order.tolist() == [0, 1] and certified.tolist() == [2, 3]  # type 2 only helps the question out of reach
    step = c.apply_list(order, question, kind, weight, deficit, 3)
    assert step.tolist() == [1, 1, 2, 0]


def test_ties_go_to_the_larger_deficit_fraction_and_then_to_the_name():
    # no type certifies a question alone; type 1 removes 3/4 of a deficit, types 0 and 2 remove 1/4 each
    per_question = [{0: 1, 1: 3}, {2: 1, 1: 3}]
    question, kind, weight = pairs_of(per_question)
    order, certified = c.greedy_list(question, kind, weight, np.array([4, 4]), 3, ["z", "m", "a"])
    assert order.tolist() == [1, 2, 0] and certified.tolist() == [0, 1, 2]


def test_list_applied_to_its_questions_reproduces_its_construction_and_never_beats_brute_force():
    rng = np.random.default_rng(2)
    for _ in range(40):
        n_types = 5
        per_question = [{int(k): int(rng.integers(1, 8)) for k in rng.choice(n_types, size=int(rng.integers(1, 4)), replace=False)}
                        for _ in range(8)]
        deficit = rng.integers(1, 12, size=8)
        question, kind, weight = pairs_of(per_question)
        names = [f"t{k}" for k in range(n_types)]
        order, certified = c.greedy_list(question, kind, weight, deficit, n_types, names)
        step = c.apply_list(order, question, kind, weight, deficit, n_types)
        reach = np.array([sum(row.values()) >= d for row, d in zip(per_question, deficit)])
        assert np.array_equal(np.bincount(step[step > 0], minlength=len(order) + 1)[1:].cumsum(), certified)
        assert ((step > 0) == reach).all()  # the whole list certifies exactly the reachable questions
        for k in range(1, len(order) + 1):  # the first k types of the list never certify more than the best set of k types
            best = max(sum(reach[q] and sum(w for t, w in row.items() if t in chosen) >= deficit[q] for q, row in enumerate(per_question))
                       for chosen in itertools.combinations(range(n_types), k))
            assert certified[k - 1] <= best


def test_shares_are_ratios_of_sums_with_intervals_over_target_types():
    cluster = np.repeat(np.arange(30), 4)
    rng = np.random.default_rng(3)
    numerator = rng.random(120) < 0.3
    share = c.resampler(cluster, np.random.default_rng(4))
    got = share(numerator, np.ones(120))
    assert np.isclose(got["value"], numerator.mean()) and got["low"] <= got["value"] <= got["high"]
    assert got["high"] - got["low"] > 0.05


def test_expectations_are_evaluated_on_point_estimates():
    result = dict(view=dict(x=dict(value=0.3, low=0.1, high=0.5), n=7))
    verdicts = c.evaluate([dict(quantity="view|x", max=0.4), dict(quantity="view|n", min=8), dict(quantity="view|x", min=0.2, max=0.25)], result)
    assert [v["holds"] for v in verdicts] == [True, False, False]
