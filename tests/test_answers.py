"""Tests of the analysis of questions of both answers (scripts/answers.py); no benchmark data needed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")  # imported by the diagnostics module that answers.py reuses
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import answers  # noqa: E402
from connectomequest.annotation_v35 import ablation  # noqa: E402
from connectomequest.annotation_v35 import engine as e  # noqa: E402


def test_earlier_labels_read_leniently_move_only_outside_vocabulary_mass_to_nonmember():
    rng = np.random.default_rng(0)
    topology = rng.dirichlet(np.ones(3), size=40)
    labelled = rng.random(40) < 0.7
    cls = rng.integers(0, 3, size=40)
    built = ablation.earlier_release_prior(topology, labelled, cls, alpha=ablation.FROZEN_ALPHA)
    lenient = ablation.earlier_release_prior(topology, labelled, np.where(cls == 1, 2, cls), alpha=ablation.FROZEN_ALPHA)
    assert np.allclose(lenient.sum(1), 1)
    outside = labelled & (cls == 1)
    assert np.allclose(lenient[~outside], built[~outside])
    assert np.allclose(lenient[outside, e.MEMBER], built[outside, e.MEMBER])  # member-first is unchanged
    assert np.allclose(lenient[outside, e.NONMEMBER] - built[outside, e.NONMEMBER], ablation.FROZEN_ALPHA)
    assert np.allclose(built[outside, e.UNRESOLVED] - lenient[outside, e.UNRESOLVED], ablation.FROZEN_ALPHA)


def toy(monkeypatch):
    """Six questions of three types: answers yes, yes, no, no, yes, uncertifiable."""
    monkeypatch.setattr(answers, "clusters_of", lambda view, family: np.array([0, 0, 1, 1, 2, 2]))
    truth = np.array([1, 1, 2, 2, 1, 0])
    weight = np.array([10, 10, 20, 20, 10, 50])
    return truth, weight


def test_balanced_ratio_is_the_mean_of_the_yes_and_the_no_ratio(monkeypatch):
    truth, weight = toy(monkeypatch)
    ratios = answers.Ratios("view", np.arange(6), truth, weight, np.random.default_rng(1))
    cost = np.array([5, 5, 30, 30, 10, 50])
    m = ratios.measure(cost)
    assert m["yes"]["value"] == pytest.approx(20 / 30)
    assert m["no"]["value"] == pytest.approx(60 / 40)
    assert m["balanced"]["value"] == pytest.approx((20 / 30 + 60 / 40) / 2)
    assert m["all"]["value"] == pytest.approx(80 / 70)  # uncertifiable questions are excluded
    assert m["all"]["low"] <= m["all"]["value"] <= m["all"]["high"]
    assert ratios.measure(cost, base=cost)["balanced"]["value"] == pytest.approx(1)


def test_regret_is_relative_to_the_cheapest_rule_per_answer(monkeypatch):
    truth, weight = toy(monkeypatch)
    names = ["weight"] + [f"p:{q}" for q in answers.POLICIES]
    cost = np.tile(weight[:, None], (1, len(names))).astype(float)
    cost[truth == 1, names.index("p:member")] *= 0.5  # cheapest on yes
    cost[truth == 2, names.index("p:nonmember")] *= 0.5  # cheapest on no
    cost[truth == 1, names.index("p:recal")] *= 0.55
    cost[truth == 2, names.index("p:recal")] *= 0.55
    out = answers.analyse_part("view", np.arange(6), truth, cost, names, np.full(6, -1), np.random.default_rng(2), [])
    regret = out["regret"]["p"]
    assert regret["recal"]["worst"] == pytest.approx(1.1)
    assert regret["member"]["no"] == pytest.approx(2.0) and regret["nonmember"]["yes"] == pytest.approx(2.0)
    assert regret["others_best_worst"] == pytest.approx(2.0)
    assert out["share_no_of_certifiable"] == pytest.approx(2 / 5)


def test_expectations_read_paths_with_contrast_names_and_both_bounds():
    result = {"v": {"part": {"contrast": {"p:a/p:b": {"no": {"value": 0.9}}}, "x": 1.2}}}
    expectations = [
        dict(id=1, quantity="v|part|contrast|p:a/p:b|no", max=0.92),
        dict(id=2, quantity="v|part|x", min=1.25),
        dict(id=3, quantity="v|part|x", min=1.0, max=1.3),
    ]
    verdicts = answers.evaluate(expectations, result)
    assert [v["holds"] for v in verdicts] == [True, False, True]
    assert verdicts[0]["value"] == pytest.approx(0.9)
