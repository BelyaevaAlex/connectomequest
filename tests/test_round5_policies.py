"""Tests of the round-5 ``policies`` workstream (baselines, optimal DP, VOI, safe interleave, WoR tests)."""

from functools import lru_cache
from itertools import product

import sys

import numpy as np
import pytest

from connectomequest.annotation_v35 import engine as e
from connectomequest.annotation_v35 import posthoc as ph
from connectomequest.round5 import heldout  # noqa: F401  (loaded on the held-out split, so part of every code identity)
from connectomequest.round5 import policies_baselines as bl
from connectomequest.round5 import policies_data as pdata
from connectomequest.round5 import policies_dp as dp
from connectomequest.round5 import policies_interleave as il
from connectomequest.round5 import policies_voi as voi
from connectomequest.round5 import policies_wor as wor


def random_question(rng, n, unresolved=0.3, wmax=9):
    cells = rng.choice(10_000, size=n, replace=False) + 1
    weights = rng.integers(1, wmax, size=n)
    threshold = int(rng.integers(1, int(weights.sum()) + 1))
    member = (1 - unresolved) / 2
    outcomes = rng.choice(3, size=n, p=(member, member, unresolved))
    return e.threshold_instance(cells, weights, threshold), outcomes


def random_prior(rng, n):
    return rng.dirichlet((1, 1, 0.3), size=n)


# --------------------------------------------------------------------------- baselines


def test_entropy_and_uncertainty_orders():
    p = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]])
    assert np.allclose(bl.entropy(p), [0.0, np.log(2), np.log(3)])
    inst = e.threshold_instance(np.array([30, 20, 10]), np.array([1, 5, 2]), 3)
    assert bl.uncertainty_order(inst, p).tolist() == [2, 1, 0]
    assert bl.weighted_uncertainty_order(inst, p).tolist() == [1, 2, 0]


def test_weighted_uncertainty_with_a_constant_prior_is_weight_order():
    rng = np.random.default_rng(10)
    for _ in range(200):
        inst, y = random_question(rng, int(rng.integers(1, 30)))
        share = float(rng.uniform(0.01, 0.99))
        p = np.tile((share, 1 - share, 0.0), (inst.size, 1))
        assert np.array_equal(bl.weighted_uncertainty_order(inst, p), e.weight_order(inst))
        # constant beliefs stay constant under recalibration only in one bin; the order is then weight order too
        order = bl.run("wus-recal", inst, y, p)
        assert e.evaluate(inst, y, order)[1] == e.full_information_status(inst, y)


def test_adaptive_baselines_check_distinct_cells_and_reach_the_exact_status():
    rng = np.random.default_rng(11)
    for _ in range(200):
        inst, y = random_question(rng, int(rng.integers(1, 20)))
        p = random_prior(rng, inst.size)
        for name in ("us", "wus", "us-recal", "wus-recal"):
            order = bl.run(name, inst, y, p)
            assert len(set(order.tolist())) == len(order)
            cost, status = e.evaluate(inst, y, order)
            assert status == e.full_information_status(inst, y)
            assert cost >= e.clairvoyant_cost(inst, y) if status != "unresolved" else cost == inst.size


# --------------------------------------------------------------------------- safe interleave


def _policy(kind, inst, p):
    return {"ag": e.AdaptiveGreedy, "adg": e.AdaptiveDualGreedy, "recal": e.RecalibratedGreedy}[kind](inst, p)


def test_safe_interleave_is_at_most_twice_the_cheaper_policy():
    rng = np.random.default_rng(12)
    checked = tight = 0
    for _ in range(1500):
        inst, y = random_question(rng, int(rng.integers(1, 25)), unresolved=float(rng.uniform(0, 0.5)))
        p = random_prior(rng, inst.size)
        kind = ("ag", "adg", "recal")[int(rng.integers(3))]
        alone = e.run_adaptive(inst, y, _policy(kind, inst, p))
        k_p, status = e.evaluate(inst, y, alone)
        k_w, _ = e.evaluate(inst, y, e.weight_order(inst))
        order = il.safe_interleave(inst, y, alone)
        assert sorted(order.tolist()) == sorted(set(order.tolist()))
        cost, s = e.evaluate(inst, y, order)
        assert s == status == e.full_information_status(inst, y)
        assert cost <= min(2 * k_p - 1, 2 * k_w) or (status == "unresolved" and cost == inst.size)
        assert cost <= 2 * min(k_p, k_w)
        checked += 1
        tight += cost == 2 * min(k_p, k_w) - 1 or cost == 2 * min(k_p, k_w)
    assert checked == 1500 and tight > 0


def test_safe_interleave_alternates_and_follows_the_stand_alone_trajectory():
    rng = np.random.default_rng(13)
    for _ in range(300):
        inst, y = random_question(rng, int(rng.integers(2, 20)))
        p = random_prior(rng, inst.size)
        alone = e.run_adaptive(inst, y, e.RecalibratedGreedy(inst, p))
        order = il.safe_interleave(inst, y, alone)
        weight = e.weight_order(inst)
        assert order[0] == alone[0]
        # Turn t checks the first cell of its own sequence that is not yet checked.
        seen = set()
        for t, cell in enumerate(order.tolist()):
            own, other = (alone, weight) if t % 2 == 0 else (weight, alone)
            first = next((int(c) for c in own if int(c) not in seen), None)
            if first is None:  # that sequence is exhausted: the other one continues
                first = next(int(c) for c in other if int(c) not in seen)
            assert cell == first
            seen.add(cell)


def test_informed_interleave_reaches_the_exact_status():
    rng = np.random.default_rng(14)
    for _ in range(300):
        inst, y = random_question(rng, int(rng.integers(1, 20)))
        p = random_prior(rng, inst.size)
        order = il.informed_interleave(inst, y, e.RecalibratedGreedy(inst, p))
        assert e.evaluate(inst, y, order)[1] == e.full_information_status(inst, y)


# --------------------------------------------------------------------------- exact dynamic programming


def brute_force_optimum(inst, p):
    """Minimal expected checks over all adaptive policies, by plain recursion (tiny questions)."""
    w = inst.weights

    @lru_cache(maxsize=None)
    def value(mask, wm, wn):
        if wm >= inst.a0 or wn >= inst.b0:
            return 0.0
        free = [v for v in range(inst.size) if not (mask >> v) & 1]
        if not free:
            return 0.0
        best = np.inf
        for v in free:
            q = 1.0
            for o in range(3):
                if p[v, o] > 0:
                    q += p[v, o] * value(mask | (1 << v), wm + (w[v] if o == 0 else 0), wn + (w[v] if o == 1 else 0))
            best = min(best, q)
        return best

    return value(0, 0, 0)


def test_dp_matches_brute_force_and_bounds_every_policy():
    rng = np.random.default_rng(15)
    for _ in range(150):
        inst, y = random_question(rng, int(rng.integers(1, 8)), unresolved=float(rng.uniform(0, 0.5)))
        p = random_prior(rng, inst.size)
        if rng.random() < 0.3:
            p[:, 2] = 0.0
            p /= p.sum(1, keepdims=True)
        sol = dp.Solution(inst, p, capacity=1 << 4)
        assert sol.value == pytest.approx(brute_force_optimum(inst, p), abs=1e-9)
        ag = e.AdaptiveGreedy(inst, p)
        v_ag, _ = dp.evaluate_stateful(inst, p, lambda av, a, b: ag.choose(a, b, av))
        weight = e.weight_order(inst)
        v_w, _ = dp.evaluate_stateful(inst, p, lambda av, a, b: next(int(j) for j in weight if av[j]))
        v_r, _ = dp.evaluate_tree(inst, p, e.RecalibratedGreedy(inst, p))
        for v in (v_ag, v_w, v_r):
            assert sol.value <= v + 1e-9
        order = sol.order(y)
        cost, status = e.evaluate(inst, y, order)
        assert status == e.full_information_status(inst, y)


def test_dp_resolves_states_the_prior_excludes_optimally():
    """An unresolved answer under a prior with p_unresolved = 0 leads to a state of probability zero;
    the action there must be optimal for the rest of the question (brute force from that state)."""
    rng = np.random.default_rng(23)
    resolved = 0
    for _ in range(120):
        n = int(rng.integers(2, 8))
        inst, y = random_question(rng, n, unresolved=0.4)
        p = random_prior(rng, inst.size)
        p[:, 2] = 0.0
        p /= p.sum(1, keepdims=True)
        sol = dp.Solution(inst, p, capacity=1 << 4)
        w = inst.weights

        @lru_cache(maxsize=None)
        def value(mask, wm, wn):
            if wm >= inst.a0 or wn >= inst.b0:
                return 0.0
            free = [v for v in range(inst.size) if not (mask >> v) & 1]
            if not free or (wm + sum(int(w[v]) for v in free) < inst.a0 and wn + sum(int(w[v]) for v in free) < inst.b0):
                return float(len(free)) if free else 0.0
            return min(1.0 + sum(p[v, o] * value(mask | (1 << v), wm + (w[v] if o == 0 else 0), wn + (w[v] if o == 1 else 0))
                                 for o in range(3) if p[v, o] > 0) for v in free)

        mask = wm = wn = 0
        rem = int(w.sum())
        while wm < inst.a0 and wn < inst.b0 and mask != (1 << n) - 1 and not (wm + rem < inst.a0 and wn + rem < inst.b0):
            before = sol.resolves
            v, val = sol.action(mask, wm, wn)
            assert val == pytest.approx(value(mask, wm, wn), abs=1e-9)
            resolved += sol.resolves > before
            mask |= 1 << v
            rem -= int(w[v])
            wm += int(w[v]) if y[v] == e.MEMBER else 0
            wn += int(w[v]) if y[v] == e.NONMEMBER else 0
    assert resolved > 10  # the branch is exercised


def test_policy_evaluation_agrees_with_enumeration_over_all_answers():
    rng = np.random.default_rng(16)
    for _ in range(60):
        inst, _ = random_question(rng, int(rng.integers(1, 7)))
        p = random_prior(rng, inst.size)
        expected = {"ag": 0.0, "recal": 0.0, "opt": 0.0}
        sol = dp.Solution(inst, p)
        for answers in product(range(3), repeat=inst.size):
            y = np.array(answers)
            prob = float(np.prod(p[np.arange(inst.size), y]))
            expected["ag"] += prob * e.evaluate(inst, y, e.run_adaptive(inst, y, e.AdaptiveGreedy(inst, p)))[0]
            expected["recal"] += prob * e.evaluate(inst, y, e.run_adaptive(inst, y, e.RecalibratedGreedy(inst, p)))[0]
            expected["opt"] += prob * e.evaluate(inst, y, sol.order(y))[0]
        ag = e.AdaptiveGreedy(inst, p)
        assert dp.evaluate_stateful(inst, p, lambda av, a, b: ag.choose(a, b, av))[0] == pytest.approx(expected["ag"])
        assert dp.evaluate_tree(inst, p, e.RecalibratedGreedy(inst, p))[0] == pytest.approx(expected["recal"])
        assert sol.value == pytest.approx(expected["opt"])


def test_dp_with_a_constant_prior_and_unit_weights_needs_no_more_than_weight_order():
    rng = np.random.default_rng(17)
    for _ in range(40):
        n = int(rng.integers(2, 10))
        inst = e.threshold_instance(np.arange(1, n + 1), np.ones(n, dtype=int), int(rng.integers(1, n + 1)))
        p = np.tile((0.5, 0.5, 0.0), (n, 1))
        weight = e.weight_order(inst)
        v_w, _ = dp.evaluate_stateful(inst, p, lambda av, a, b: next(int(j) for j in weight if av[j]))
        assert dp.Solution(inst, p).value == pytest.approx(v_w)  # all cells are exchangeable


# --------------------------------------------------------------------------- one-step VOI


def test_rollout_cost_is_the_weight_order_cost_when_the_beliefs_are_certain():
    rng = np.random.default_rng(18)
    for _ in range(300):
        inst, y = random_question(rng, int(rng.integers(1, 25)))
        q = np.eye(3)[y]
        order = e.weight_order(inst)
        cost, status = e.evaluate(inst, y, order)
        h = voi.rollout_cost(inst.weights[order].astype(float), q[order], inst.a0, inst.b0)
        assert h == pytest.approx(cost)


def test_rollout_cost_tracks_the_exact_expectation_of_weight_order():
    rng = np.random.default_rng(26)
    errors = []
    for _ in range(40):
        inst, _ = random_question(rng, int(rng.integers(3, 9)))
        p = random_prior(rng, inst.size)
        weight = e.weight_order(inst)
        exact, _ = dp.evaluate_stateful(inst, p, lambda av, a, b: next(int(j) for j in weight if av[j]))
        approx = voi.rollout_cost(inst.weights[weight].astype(float), p[weight], inst.a0, inst.b0)
        errors.append(abs(approx - exact) / exact)
    assert np.median(errors) < 0.15


def test_voi_policies_reach_the_exact_status():
    rng = np.random.default_rng(19)
    for _ in range(200):
        inst, y = random_question(rng, int(rng.integers(1, 25)))
        p = random_prior(rng, inst.size)
        for cls in voi.POLICIES.values():
            order = e.run_adaptive(inst, y, cls(inst, p))
            assert e.evaluate(inst, y, order)[1] == e.full_information_status(inst, y)


# --------------------------------------------------------------------------- without-replacement tests


def test_pps_order_draws_in_proportion_to_weight():
    rng = np.random.default_rng(20)
    w = np.array([1, 2, 3, 4])
    first = np.bincount([wor.pps_order(w, rng)[0] for _ in range(20000)], minlength=4) / 20000
    assert np.allclose(first, w / w.sum(), atol=0.015)
    # second draw given the first: proportional to the remaining weight
    second = np.zeros(4)
    count = 0
    for _ in range(20000):
        o = wor.pps_order(w, rng)
        if o[0] == 3:
            second[o[1]] += 1
            count += 1
    assert np.allclose(second[:3] / count, w[:3] / w[:3].sum(), atol=0.03)


@pytest.mark.parametrize("hybrid", [False, True])
def test_wor_tests_control_wrong_certificates_at_adversarial_thresholds(hybrid):
    rng = np.random.default_rng(21 + hybrid)
    wrong = audits = 0
    for trial in range(300):
        n = int(rng.integers(20, 80))
        cells = np.arange(1, n + 1)
        weights = rng.integers(1, 12, size=n)
        outcomes = rng.choice(3, size=n, p=(0.4, 0.4, 0.2))
        total = int(weights.sum())
        member = int(weights[outcomes == e.MEMBER].sum())
        upper = total - int(weights[outcomes == e.NONMEMBER].sum())
        for threshold in {min(total, member + 1), max(1, min(total, upper))}:
            inst = e.threshold_instance(cells, weights, threshold)
            truth = e.full_information_status(inst, outcomes)
            for repeat in range(4):
                fn = wor.wor_hybrid_certificate if hybrid else wor.wor_certificate
                cost, status, exact = fn(inst, outcomes, alpha=0.05, rng=np.random.default_rng([trial, threshold, repeat]))
                assert 1 <= cost <= n
                if exact:
                    assert status == truth
                audits += 1
                wrong += status not in ("unresolved", truth)
    assert audits > 1500 and wrong / audits <= 0.05


def test_wor_test_controls_each_claim_under_its_null():
    rng = np.random.default_rng(23)
    n = 400
    inst = e.threshold_instance(np.arange(1, n + 1), np.ones(n, dtype=int), 200)
    y = np.array([e.MEMBER] * 199 + [e.NONMEMBER] * 100 + [e.UNRESOLVED] * 101)
    wrong = sum(wor.wor_certificate(inst, y, rng=rng)[1] == "positive" for _ in range(400))
    assert wrong / 400 <= 0.05 + 0.03
    y = np.array([e.MEMBER] * 150 + [e.UNRESOLVED] * 50 + [e.NONMEMBER] * 200)
    wrong = sum(wor.wor_certificate(inst, y, rng=rng)[1] == "negative" for _ in range(400))
    assert wrong / 400 <= 0.05 + 0.03


def test_wor_test_is_cheap_far_from_the_threshold():
    rng = np.random.default_rng(24)
    weights = rng.integers(1, 20, size=600)
    y = np.where(rng.random(600) < 0.8, e.MEMBER, e.NONMEMBER)
    threshold, _ = e.threshold_goals(int(weights.sum()), 1, 2)
    inst = e.threshold_instance(np.arange(1, 601), weights, threshold)
    costs = [wor.wor_certificate(inst, y, rng=rng) for _ in range(50)]
    assert all(s == "positive" for _, s, _ in costs)
    assert np.mean([c for c, _, _ in costs]) < 0.25 * e.clairvoyant_cost(inst, y)


def test_wor_hybrid_equals_weight_order_when_the_heavy_half_decides():
    rng = np.random.default_rng(25)
    for _ in range(200):
        inst, y = random_question(rng, int(rng.integers(4, 30)))
        cost, status = e.evaluate(inst, y, e.weight_order(inst))
        order = e.weight_order(inst)
        half = int(np.searchsorted(np.cumsum(inst.weights[order]) * 2, int(inst.weights.sum()), side="left")) + 1
        if status != "unresolved" and cost <= half:
            assert wor.wor_hybrid_certificate(inst, y, rng=np.random.default_rng(0)) == (cost, status, True)
            assert ph.hybrid_certificate(inst, y, rng=np.random.default_rng(0)) == (cost, status, True)


def test_wor_test_checks_every_cell_when_nothing_can_be_certified():
    inst = e.threshold_instance(np.array([1, 2, 3]), np.array([6, 2, 2]), 5)
    y = np.array([e.UNRESOLVED, e.MEMBER, e.NONMEMBER])
    for seed in range(20):
        cost, status, exact = wor.wor_certificate(inst, y, rng=np.random.default_rng(seed))
        assert status in ("unresolved", "positive", "negative")
        if exact:
            assert (cost, status) == (3, "unresolved")


# --------------------------------------------------------------------------- protocol guard


def test_guard_refuses_the_held_out_split_without_a_record(tmp_path, monkeypatch):
    monkeypatch.setattr(pdata, "RECORD", tmp_path / "record.json")
    monkeypatch.setattr(pdata, "_GUARD", {})
    runner = pdata.ROOT / "src/connectomequest/round5/policies_data.py"
    assert pdata.require_record(runner, "development") is None
    with pytest.raises(SystemExit):
        pdata.require_record(runner, "test")
    code = pdata.code_identity(runner)
    (tmp_path / "record.json").write_text(__import__("json").dumps({"policies": {"code": code}}))
    assert pdata.require_record(runner, "test") == code
    bad = dict(code)
    bad[next(iter(bad))] = "0" * 64
    (tmp_path / "record.json").write_text(__import__("json").dumps({"code": bad}))
    with pytest.raises(SystemExit):
        pdata.require_record(runner, "test")


def test_guard_rejects_conflicting_hashes_late_imports_and_other_outputs(tmp_path, monkeypatch):
    import json
    import sys

    monkeypatch.setattr(pdata, "RECORD", tmp_path / "record.json")
    monkeypatch.setattr(pdata, "_GUARD", {})
    runner = pdata.ROOT / "src/connectomequest/round5/policies_data.py"
    code = pdata.code_identity(runner)
    stale = {k: "0" * 64 for k in code}
    # a path listed twice must carry the current hash everywhere (no last-wins merge)
    (tmp_path / "record.json").write_text(json.dumps({"policies": {"code": code, "old": stale}}))
    with pytest.raises(SystemExit):
        pdata.require_record(runner, "test")
    # only the record's "policies" entry counts when there is one
    (tmp_path / "record.json").write_text(json.dumps({"policies": {"code": stale}, "other": {"code": code}}))
    with pytest.raises(SystemExit):
        pdata.require_record(runner, "test")
    (tmp_path / "record.json").write_text(json.dumps({"policies": {"code": code}, "other": {"code": stale}}))
    assert pdata.require_record(runner, "test") == code
    # a round-5 module imported after the check blocks every save
    fake = type(sys)("connectomequest.round5._late_for_test")
    fake.__file__ = str(pdata.ROOT / "src/connectomequest/round5/_late_for_test.py")
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    with pytest.raises(SystemExit):
        pdata.save_json(tmp_path / "x.json", {})
    assert not (tmp_path / "x.json").exists()
    # the held-out split writes only to the default place
    with pytest.raises(SystemExit):
        pdata.default_output(tmp_path / "elsewhere", pdata.OUT / "runs", "test")
    assert pdata.default_output(tmp_path / "elsewhere", pdata.OUT / "runs", "development") == tmp_path / "elsewhere"


def test_own_guard_and_shared_gate_accept_one_record(tmp_path, monkeypatch):
    """A record in the layout of round5.heldout passes both checks, and the gate's own import is no late import."""
    import json

    from connectomequest.round5 import heldout

    record = tmp_path / "record.json"
    for mod, name in ((pdata, "RECORD"), (heldout, "RECORD")):
        monkeypatch.setattr(mod, name, record)
    monkeypatch.setattr(heldout, "CLAIMS", tmp_path / "claims")
    monkeypatch.setattr(pdata, "_GUARD", {})
    exits = []
    monkeypatch.setattr(heldout.atexit, "register", exits.append)
    runner = pdata.ROOT / "scripts/r5_policies_wor.py"
    expect = "outputs/round5/policies/safe_interleave.md"  # any recorded repository file stands in for the expectations
    names = [heldout.relative(runner), expect] + heldout.loaded_repository_files()
    monkeypatch.setattr(heldout.sys, "addaudithook", lambda hook: None)
    monkeypatch.setattr(sys, "argv", [str(runner), "--view", "x", "--split", "test", "--workers", "16"])
    argv = {"wor-x": ["scripts/r5_policies_wor.py", "--view", "x", "--split", "test"]}
    record.write_text(json.dumps(dict(files={n: heldout.sha256(pdata.ROOT / n) for n in names}, expectations={"policies": expect},
                                      data={"policies": []}, allowed={"policies": ["wor-x"]}, argv={"policies": argv},
                                      environment=heldout.environment(),
                                      launch={"PYTHONPATH": "src:.v32-deps:scripts", "threads_env": {"OMP_NUM_THREADS": "1"}})))
    monkeypatch.setenv("PYTHONPATH", "src:.v32-deps:scripts")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.chdir(pdata.ROOT)
    assert pdata.require_record(runner, "test")
    heldout.claim(runner, "policies", "wor-x", tmp_path / "out.npz")
    pdata.save_json(tmp_path / "out.json", {})  # the gate imported before the check: no late round-5 module
    with pytest.raises(SystemExit):
        heldout.claim(runner, "policies", "wor-x", tmp_path / "out.npz")  # one claim per computation
    with pytest.raises(SystemExit):
        heldout.claim(runner, "policies", "wor-y", tmp_path / "out2.npz")  # key not recorded
    exits[0]()
    assert json.loads((tmp_path / "claims" / "r5_policies_wor--wor-x.json").read_text())["status"] == "failed"  # out.npz absent


def test_outputs_are_never_overwritten(tmp_path):
    path = tmp_path / "x.npz"
    pdata.save_npz(path, a=np.arange(3))
    with pytest.raises(SystemExit):
        pdata.save_npz(path, a=np.arange(3))
    # a file created between the check and the publication is not overwritten either
    tmp = tmp_path / ".y.tmp"
    tmp.write_text("new")
    (tmp_path / "y.json").write_text("old")
    with pytest.raises(SystemExit):
        pdata._publish(tmp, tmp_path / "y.json")
    assert (tmp_path / "y.json").read_text() == "old" and not tmp.exists()
    pdata.save_text(tmp_path / "z.md", "text")
    with pytest.raises(SystemExit):
        pdata.save_text(tmp_path / "z.md", "other")
    assert (tmp_path / "z.md").read_text() == "text"


def test_vectorized_rollout_lookahead_matches_one_candidate_at_a_time():
    rng = np.random.default_rng(27)
    for _ in range(200):
        inst, _ = random_question(rng, int(rng.integers(1, 30)), wmax=40)
        p = random_prior(rng, inst.size)
        order = e.weight_order(inst)
        cand = rng.choice(inst.size, size=min(inst.size, 5), replace=False)
        a, b = int(rng.integers(1, inst.a0 + 1)), int(rng.integers(1, inst.b0 + 1))
        w = inst.weights.astype(float)
        got = voi.rollout_lookahead(w[order], p[order], order, cand, w[cand], p[cand], a, b)
        for i, v in enumerate(cand):
            keep = order[order != v]
            want = 1.0
            for o, (da, db) in ((0, (w[v], 0)), (1, (0, w[v])), (2, (0, 0))):
                want += p[v, o] * voi.rollout_cost(w[keep], p[keep], a - da, b - db)
            assert got[i] == pytest.approx(want, rel=1e-9, abs=1e-9)
