"""The third protocol (scripts/newtypes.py): partial coverage, arms, simultaneous intervals, stored models and gates."""
import json
import pickle
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

newtypes = pytest.importorskip("newtypes")
from connectomequest.annotation_v35 import engine as e  # noqa: E402


def priors(n, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name in newtypes.PRIORS:
        p = rng.random((n, 3)) + 0.05
        out[name] = (p / p.sum(1, keepdims=True)).astype(np.float32)
    return out


def test_partial_coverage_mixes_the_two_priors_on_nested_sets_of_neurons():
    cells = np.arange(10_000, 14_000, dtype=np.int64) * 7919
    p = priors(len(cells))
    covered = {}
    for level in newtypes.COVERAGE:
        mixed = newtypes.at_coverage(p, cells, level)
        from_earlier = (mixed == p["provisional"]).all(1)
        from_topology = (mixed == p["mlp_s0"]).all(1)
        assert (from_earlier | from_topology).all()
        covered[level] = from_earlier
        assert abs(from_earlier.mean() - level) < 0.03  # a pseudo-random share of the neurons
    assert (covered[0.25] <= covered[0.5]).all() and (covered[0.5] <= covered[0.75]).all()
    # the same neuron is covered or not wherever it occurs
    again = newtypes.at_coverage({k: v[::-1] for k, v in p.items()}, cells[::-1], 0.5)
    assert np.array_equal((again == p["provisional"][::-1]).all(1)[::-1], covered[0.5])
    assert np.array_equal(newtypes.prior_variant(p, cells, "cov0.5"), newtypes.at_coverage(p, cells, 0.5))
    assert newtypes.prior_variant(p, cells, "provisional") is p["provisional"]


def test_arms_of_each_set_and_the_models_they_follow():
    held = newtypes.arm_names("flywire", "heldout")
    assert held[0] == "weight" and not any(name.startswith(("mlp_s0:", "provisional:")) for name in held)  # run before on these targets
    assert "provisional:recal" in newtypes.arm_names("malecns", "heldout")  # MaleCNS had no earlier-release prior before
    new = newtypes.arm_names("flywire", "new")
    for level in newtypes.COVERAGE:
        assert {f"cov{level}:{policy}" for policy in newtypes.AT_COVERAGE} <= set(new) and {f"cov{level}:{policy}" for policy in newtypes.AT_COVERAGE} <= set(held)
    assert {"none:prop", "synthetic0.7:ag", "mlp_s0:ec2", "provisional:imitation"} <= set(new)
    arms = newtypes.imitation_arms("malecns", new)
    assert "weight" not in arms and "mlp_s0:recal" not in arms
    assert arms["provisional:imitation"] == ("malecns-provisional", "provisional") and arms["mlp_s0:imitation"] == ("malecns-mlp_s0", "mlp_s0")
    assert arms["cov0.25:imitation-full"] == ("malecns-provisional", "cov0.25")  # the order fitted at full coverage, on the mixed prior
    assert arms["cov0.5:imitation-mixed"] == ("malecns-mixed", "cov0.5")
    assert arms["cov0.0:imitation-mixed"] == ("malecns-mixed", "mlp_s0") and arms["cov1.0:imitation-mixed"] == ("malecns-mixed", "provisional")


def test_ratio_of_totals_with_a_simultaneous_interval_that_contains_the_95_percent_one():
    rng = np.random.default_rng(1)
    types = np.repeat(np.arange(60), 5)
    truth = rng.choice([0, 1, 2], size=len(types), p=[0.2, 0.6, 0.2])
    base = rng.integers(2, 30, len(types))
    cost = np.maximum(1, base - rng.integers(0, 4, len(types)))
    boot = newtypes.Bootstrap(types, truth, np.random.default_rng(2), tail=0.05 / (2 * 43))
    out = boot.measure(cost, base, np.ones(len(types), dtype=bool))
    for name, answer in (("all", truth > 0), ("yes", truth == 1), ("no", truth == 2)):
        d = out[name]
        assert d["value"] == pytest.approx(cost[answer].sum() / base[answer].sum())
        assert d["low_simultaneous"] <= d["low"] <= d["value"] <= d["high"] <= d["high_simultaneous"]
        assert d["high_simultaneous"] - d["low_simultaneous"] > d["high"] - d["low"]
    assert boot.measure(cost, base, np.zeros(len(types), dtype=bool))["all"] is None
    names = ["weight", "cov0.5:recal", "cov0.5:direction", "provisional:recal"]
    assert newtypes.contrasts(names) == [("cov0.5:recal", "cov0.5:direction")]  # only pairs whose two arms were run


def test_models_fitted_for_this_protocol_are_the_recorded_ones_and_reach_the_exact_status():
    if not (newtypes.OUT / "models.npz").exists():
        pytest.skip("fitted models are not available")
    pytest.importorskip("sklearn")
    import imitation

    fit = json.loads((newtypes.OUT / "fit.json").read_text())
    assert set(fit["models"]) == {"malecns-mixed", "malecns-provisional", "flywire-mixed"}
    rng = np.random.default_rng(3)
    for key in fit["models"]:
        # written by ``newtypes.py fit`` of this project; ``model_bytes`` stops unless the hash equals the one in fit.json,
        # which the record pins, so pickle only loads bytes this pipeline produced
        model = pickle.loads(newtypes.model_bytes(key))
        for _ in range(4):
            n = 14
            weights, outcomes = rng.integers(1, 40, n), rng.choice(3, n, p=[0.3, 0.5, 0.2])
            belief = priors(n, int(rng.integers(1 << 30)))["provisional"].astype(float)
            threshold = max(1, int(weights.sum()) // 3)
            inst = e.threshold_instance(np.arange(100, 100 + n), weights, threshold)
            score = model.predict_proba(imitation.features(weights, belief, threshold))[:, 1]
            order = np.lexsort((inst.ranks, -score))
            assert sorted(order.tolist()) == list(range(n))
            assert e.evaluate(inst, outcomes, order)[1] == e.full_information_status(inst, outcomes)


def test_recorded_expectations_resolve_and_the_sets_cannot_be_run_again():
    if not newtypes.RECORD.exists():
        pytest.skip("the record is not available")
    record = json.loads(newtypes.RECORD.read_text())
    assert record["simultaneous"]["expectations"] == len(record["expectations"]) and {x["split"] for x in record["expectations"]} == {"heldout", "new"}
    for split in ("heldout", "new"):
        analysis = json.loads((newtypes.OUT / f"analysis-{split}.json").read_text())
        verdicts = json.loads((newtypes.OUT / f"verdicts-{split}.json").read_text())
        assert [v["id"] for v in verdicts] == [x["id"] for x in record["expectations"] if x["split"] == split]
        assert analysis["simultaneous"]["tail"] == pytest.approx(newtypes.ALPHA / (2 * len(record["expectations"])))
        for v in verdicts:
            assert newtypes.rep.lookup(analysis, v["quantity"]) == v["value"]
            assert v["low_simultaneous"] <= v["value"] <= v["high_simultaneous"]
            assert v["holds"] or not v["holds_simultaneous"]  # the whole interval within the threshold implies the estimate
        for view in newtypes.VIEWS:
            # a second run stops: the result exists, or (in an anonymized copy) the code differs from the record
            before = sorted(p.name for p in newtypes.OUT.iterdir())
            with pytest.raises(SystemExit):
                newtypes.run(Namespace(view=view, split=split, workers=1, limit=None))
            assert sorted(p.name for p in newtypes.OUT.iterdir()) == before
