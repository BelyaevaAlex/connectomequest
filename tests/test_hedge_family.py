"""Proposition 1: Adaptive Greedy with a perfect prior checks k + n cells where k suffice, and n grows like ln Z."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import hedge_family as h  # noqa: E402
from connectomequest.annotation_v35 import engine as e  # noqa: E402


@pytest.mark.parametrize("k", (2, 3, 4, 7, 12))
def test_weights_follow_the_recurrence_and_the_bound_on_z(k):
    for n in range(1, 30):
        weights = h.nonmember_weights(k, n)
        assert weights == sorted(weights, reverse=True) and len(weights) == n
        missing = 1  # synapses still missing for a "no" after the nonmembers that follow
        for w in reversed(weights):
            assert (k - 1) * w > missing >= (k - 1) * (w - 1)  # w = floor(missing / (k - 1)) + 1
            assert k * w > missing + w  # the nonmember gains more than a member while no member is checked
            missing += w
        z = sum(weights) + 1
        assert missing == z
        assert z + k - 1 <= k * (k / (k - 1)) ** n * (1 + 1e-12)
        assert n >= (k - 1) * math.log(z / k)
    assert h.nonmember_weights(2, 4) == [16, 8, 4, 2]


@pytest.mark.parametrize("k,n", [(2, 1), (2, 7), (2, 30), (3, 1), (3, 12), (4, 25), (7, 3), (7, 40), (12, 60)])
def test_adaptive_greedy_checks_every_nonmember_first(k, n):
    instance, outcomes, prior = h.question(k, n)
    assert instance.a0 == k and instance.b0 == int(instance.weights[k:].sum()) + 1
    assert e.full_information_status(instance, outcomes) == "positive"
    assert e.clairvoyant_cost(instance, outcomes) == k
    order = e.run_adaptive(instance, outcomes, e.AdaptiveGreedy(instance, prior))
    assert e.evaluate(instance, outcomes, order) == (k + n, "positive")
    assert order[:n].tolist() == list(range(k, k + n))  # the nonmembers, heaviest first, before any member
    assert e.evaluate(instance, outcomes, e.direction_order(instance, prior)) == (k, "positive")
    row = h.checks(k, n)
    h.verify(row)
    assert row["ag"] == k + n and row["direction"] == row["optimum"] == k and row["ag"] >= row["bound"]


def test_recalibration_does_not_remove_the_hedge_of_a_perfect_prior():
    for k, n in ((2, 20), (3, 20), (5, 20)):
        row = h.checks(k, n)
        assert row["recalibrated"] > row["optimum"]
    assert h.checks(2, 20)["recalibrated"] == 22


def test_the_family_rejects_degenerate_parameters():
    for k, n in ((1, 3), (2, 0)):
        with pytest.raises(ValueError):
            h.nonmember_weights(k, n)
    assert np.array_equal(h.question(2, 3)[0].weights, [1, 1, 8, 4, 2])
