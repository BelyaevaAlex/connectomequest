"""Generic policies (edge cutting, information gain) and the helpers of the replication on new target neurons."""
import sys
from pathlib import Path

import numpy as np
import pytest

from connectomequest.annotation_v35 import decision
from connectomequest.annotation_v35 import engine as e

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def question(seed, n=14):
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 40, n)
    outcomes = rng.choice(3, n, p=[0.3, 0.5, 0.2])
    prior = np.full((n, 3), 0.15)
    prior[np.arange(n), rng.choice(3, n)] = 0.7
    return np.arange(100, 100 + n), weights, outcomes, prior


@pytest.mark.parametrize("name", sorted(decision.POLICIES))
def test_generic_policies_check_distinct_cells_and_reach_the_exact_status(name):
    for seed in range(12):
        cells, weights, outcomes, prior = question(seed)
        for threshold in (max(1, int(weights.sum()) // 5), max(1, int(weights.sum()) // 2)):
            inst = e.threshold_instance(cells, weights, threshold)
            order = e.run_adaptive(inst, outcomes, decision.policy(name, inst, prior, np.random.default_rng([seed, threshold])))
            assert len(set(order.tolist())) == len(order)
            cost, status = e.evaluate(inst, outcomes, order)
            assert status == e.full_information_status(inst, outcomes)
            bound = e.clairvoyant_cost(inst, outcomes)
            assert bound is None or cost >= bound


def test_recalibrated_variants_use_the_recalibration_of_the_engine():
    cells, weights, outcomes, prior = question(3)
    inst = e.threshold_instance(cells, weights, 5)
    reference = e.RecalibratedGreedy(inst, prior)
    for name in ("ec2-recal", "ig-recal"):
        policy = decision.policy(name, inst, prior, np.random.default_rng(1))
        for cell in (0, 4, 9):
            policy.observe(cell, int(outcomes[cell]))
        for cell in (0, 4, 9):
            reference.observe(cell, int(outcomes[cell]))
        assert np.array_equal(policy.p, reference.p)
        reference = e.RecalibratedGreedy(inst, prior)


def test_decision_probabilities_match_exhaustive_enumeration():
    cells = np.arange(4)
    weights = np.array([5, 3, 2, 1])
    prior = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.5, 0.25, 0.25], [0.1, 0.1, 0.8]])
    inst = e.threshold_instance(cells, weights, 6)
    policy = decision.policy("ig", inst, prior, np.random.default_rng(0), samples=200_000)
    _, _, now, after = policy.decisions(inst.a0, inst.b0, np.ones(4, dtype=bool))
    yes = no = 0.0
    yes_if_member_0 = 0.0
    for combo in np.ndindex(3, 3, 3, 3):
        p = float(np.prod([prior[i, o] for i, o in enumerate(combo)]))
        member = sum(int(weights[i]) for i, o in enumerate(combo) if o == e.MEMBER)
        nonmember = sum(int(weights[i]) for i, o in enumerate(combo) if o == e.NONMEMBER)
        yes += p * (member >= inst.a0)
        no += p * (nonmember >= inst.b0)
        if combo[0] == e.MEMBER:
            yes_if_member_0 += p / prior[0, e.MEMBER] * (member >= inst.a0)
    assert abs(now[0] - yes) < 0.01 and abs(now[1] - no) < 0.01
    assert abs(after[e.MEMBER][0][0] - yes_if_member_0) < 0.01


def test_information_gain_prefers_the_cell_that_decides():
    cells = np.arange(3)
    weights = np.array([10, 1, 1])
    prior = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    inst = e.threshold_instance(cells, weights, 10)  # yes iff the heavy cell is a member
    policy = decision.policy("ig", inst, prior, np.random.default_rng(0), samples=4096)
    assert policy.choose(inst.a0, inst.b0, np.ones(3, dtype=bool)) == 0


def test_own_stopping_rule_is_recorded_when_all_realizations_agree():
    cells = np.arange(3)
    weights = np.array([4, 3, 2])
    prior = np.tile((0.0, 1.0, 0.0), (3, 1))  # the prior is certain of nonmembers
    outcomes = np.array([e.MEMBER, e.MEMBER, e.NONMEMBER])
    inst = e.threshold_instance(cells, weights, 7)
    policy = decision.policy("ig", inst, prior, np.random.default_rng(0))
    order = e.run_adaptive(inst, outcomes, policy)
    assert e.evaluate(inst, outcomes, order)[1] == "positive"
    assert policy.unanimous == (0, 2)  # before any check, on the answer no


def test_thresholds_of_retention_and_share_questions():
    replication = pytest.importorskip("replication")
    assert replication.threshold_of("share", 1, 2, 101, 7, 100) == 51
    assert replication.threshold_of("retention", 1, 2, 100, 30, 100) == 15
    assert replication.threshold_of("retention", 1, 1, 10, 1, 3) == 4


def test_ratio_of_totals_by_answer():
    replication = pytest.importorskip("replication")
    types = np.array([1, 1, 2, 2, 3, 3])
    truth = np.array([1, 2, 1, 2, 0, 1])
    ratios = replication.Ratios(types, truth, np.random.default_rng(0))
    cost, base = np.array([2, 4, 2, 4, 9, 2]), np.array([4, 4, 4, 4, 9, 4])
    out = ratios.measure(cost, base, np.ones(6, dtype=bool))
    assert out["yes"]["value"] == pytest.approx(0.5) and out["no"]["value"] == pytest.approx(1.0)
    assert out["all"]["value"] == pytest.approx(14 / 20) and out["balanced"]["value"] == pytest.approx(0.75)
    assert replication.lookup(dict(a=dict(b=dict(value=3))), "a|b|value") == 3


def test_learned_policy_uses_the_stored_parameters_and_reaches_the_exact_status():
    import learned
    from connectomequest.neurosym import policy as P

    fits = Path(__file__).resolve().parents[1] / "outputs/neurosym/A"
    if not (fits / "fit-flywire-provisional.json").exists():
        pytest.skip("fitted parameters are not available")
    for view, priors in learned.rep.PRIORS.items():
        for prior in priors:
            params = learned.fitted(view, prior)
            assert params["theta"].shape == (len(P.FEATURES),) and np.isfinite(params["theta"]).all()
            assert abs(np.abs(params["theta"]).sum() - 1) < 1e-9 and 0 <= params["recal"] <= 1 and params["strength"] > 0
            for seed in range(8):
                cells, weights, outcomes, belief = question(seed)
                inst = e.threshold_instance(cells, weights, max(1, int(weights.sum()) // 3))
                order = e.run_adaptive(inst, outcomes, P.make(inst, belief, params))
                assert len(set(order.tolist())) == len(order)
                assert e.evaluate(inst, outcomes, order)[1] == e.full_information_status(inst, outcomes)
    assert learned.arm_names("flywire") == ["weight", "mlp_s0:recal", "mlp_s0:learned", "provisional:recal", "provisional:learned"]


def test_learned_policy_with_the_parameters_of_recalibrated_ag_reproduces_it():
    from connectomequest.neurosym import policy as P

    for seed in range(8):
        cells, weights, outcomes, belief = question(seed)
        inst = e.threshold_instance(cells, weights, max(1, int(weights.sum()) // 3))
        expected = e.run_adaptive(inst, outcomes, e.RecalibratedGreedy(inst, belief))
        assert np.array_equal(e.run_adaptive(inst, outcomes, P.make(inst, belief, P.special("recal"))), expected)


def test_imitated_certificate_is_the_smallest_and_certifies():
    import imitation

    seen = 0
    for seed in range(40):
        cells, weights, outcomes, _ = question(seed)
        for threshold in (max(1, int(weights.sum()) // 4), max(1, int(weights.sum()) // 2)):
            inst = e.threshold_instance(cells, weights, threshold)
            mask = imitation.certificate(inst, weights, outcomes, threshold)
            if e.full_information_status(inst, outcomes) == "unresolved":
                assert mask is None
                continue
            seen += 1
            chosen = np.flatnonzero(mask)
            assert len(chosen) == e.clairvoyant_cost(inst, outcomes)
            assert len(set(outcomes[chosen].tolist())) == 1  # members only, or nonmembers only
            rest = np.flatnonzero(~mask)
            cost, status = e.evaluate(inst, outcomes, np.concatenate((chosen, rest)))
            assert cost == len(chosen) and status == e.full_information_status(inst, outcomes)
    assert seen >= 20


def test_imitation_features_use_weights_prior_and_threshold_only():
    import imitation

    cells, weights, outcomes, belief = question(3)
    threshold = int(weights.sum()) // 3
    x = imitation.features(weights, belief, threshold)
    assert x.shape == (len(weights), len(imitation.FEATURES)) and np.isfinite(x).all()
    assert "outcomes" not in imitation.features.__code__.co_varnames
    heaviest = int(np.argmax(weights))
    assert x[heaviest, 1] == 0 and x[heaviest, 2] == pytest.approx(weights[heaviest] / weights.sum())
    assert np.allclose(x[:, 3:6], belief) and np.allclose(x[:, 6], threshold / weights.sum())
    # cells of distinct weight keep their features when the cells are listed in another order
    distinct = np.flatnonzero(np.bincount(weights)[weights] == 1)
    perm = np.random.default_rng(0).permutation(len(weights))
    y = imitation.features(weights[perm], belief[perm], threshold)
    back = np.empty(len(perm), dtype=int)
    back[perm] = np.arange(len(perm))
    assert np.allclose(y[back][distinct], x[distinct])


def test_imitation_order_from_the_stored_models_reaches_the_exact_status():
    import pickle

    import imitation
    import imitation_models

    if not imitation_models.PACKED.exists():
        pytest.skip("fitted models are not available")
    pytest.importorskip("sklearn")
    for view, priors in imitation.rep.PRIORS.items():
        for prior in priors:
            data = imitation_models.stored(f"{view}:{prior}")  # bytes whose hash equals the one in fit.json
            path = imitation.OUT / f"model-{view}-{prior}.pkl"
            assert not path.exists() or path.read_bytes() == data
            model = pickle.loads(data)  # written by scripts/imitation.py fit; hash checked by ``stored``
            for seed in range(6):
                cells, weights, outcomes, belief = question(seed)
                threshold = max(1, int(weights.sum()) // 3)
                inst = e.threshold_instance(cells, weights, threshold)
                score = model.predict_proba(imitation.features(weights, belief, threshold))[:, 1]
                order = np.lexsort((inst.ranks, -score))
                assert sorted(order.tolist()) == list(range(len(weights)))
                assert e.evaluate(inst, outcomes, order)[1] == e.full_information_status(inst, outcomes)
    assert imitation.arm_names("flywire") == ["weight", "mlp_s0:imitation", "provisional:imitation"]
