"""Round 5, workstream ``costs``: batch and heterogeneous cost models, protocol guard."""

from itertools import combinations, product

import numpy as np
import pytest

from connectomequest.annotation_v35 import engine as e
from connectomequest.round5 import costs_batch as cb
from connectomequest.round5 import costs_common as cc
from connectomequest.round5 import costs_hetero as ch

HAVE_DATA = all((cc.RUNS / f"{v}-development.npz").exists() for v in cc.VIEWS) and all(
    (cc.BENCH / f"{v}.npz").exists() for v in cc.VIEWS
)
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="needs the benchmark arrays and frozen runs")
RAW = ("data/raw/manc/body-annotations-male-cns-v1.0-minconf-0.5.feather", "inputs/v32/flywire/cq-flywire-v3.1.0.tsv",
       "inputs/v32/ontology/Supplemental_file5_hemibrain_meta.csv")
HAVE_RAW = all((cc.ROOT / path).exists() for path in RAW)
needs_raw = pytest.mark.skipif(not HAVE_RAW, reason="needs the raw annotation releases and the hemibrain metadata")


def random_question(rng, n, n_types):
    cells = np.sort(rng.choice(10_000, size=n, replace=False) + 1)
    weights = rng.integers(1, 9, size=n)
    threshold = int(rng.integers(1, int(weights.sum()) + 1))
    release = rng.integers(-1, n_types, size=n)
    # outcomes mostly shared within a release type, sometimes mixed
    type_outcome = rng.choice(3, size=n_types + 1, p=(0.35, 0.35, 0.3))
    outcomes = np.where(rng.random(n) < 0.85, type_outcome[release + 1], rng.choice(3, size=n))
    p = rng.dirichlet(np.ones(3), size=n)
    return cells, weights, outcomes, threshold, release, p


# --------------------------------------------------------------------------- batch model


def test_batch_clairvoyant_is_the_fewest_batches_of_any_certificate():
    rng = np.random.default_rng(0)
    for _ in range(300):
        cells, w, y, t, rel, _ = random_question(rng, int(rng.integers(1, 9)), 4)
        bq = cb.BatchQuestion(cells, w, y, t, cb.batch_keys(rel))
        best = None
        for k in range(1, bq.units + 1):
            for subset in combinations(range(bq.units), k):
                s = np.asarray(subset)
                if bq.member[s].sum() >= t or bq.nonmember[s].sum() >= bq.instance.b0:
                    best = k
                    break
            if best is not None:
                break
        assert bq.clairvoyant() == best


def test_batch_policies_reach_the_full_information_status_of_the_cells():
    rng = np.random.default_rng(1)
    for _ in range(300):
        cells, w, y, t, rel, p = random_question(rng, int(rng.integers(1, 14)), 5)
        inst = e.threshold_instance(cells, w, t)
        truth = e.full_information_status(inst, y)
        bq = cb.BatchQuestion(cells, w, y, t, cb.batch_keys(rel))
        assert bq.status() == truth
        bound = bq.clairvoyant()
        for order in (cb.weight_order(bq), cb.heaviest_cell_order(bq), cb.member_order(bq, p), cb.direction_order(bq, p),
                      cb.run_greedy(bq, p), cb.run_sum_greedy(bq, p), cb.run_recalibrated(bq, p)):
            assert len(set(order.tolist())) == len(order)
            c, s = bq.evaluate(order)
            assert s == truth
            if bound is not None:
                assert c >= bound
            else:
                assert c == bq.units


def test_batch_certificate_is_the_cell_certificate_of_the_union():
    """Checking batches in an order certifies exactly when its cells, checked together, do."""
    rng = np.random.default_rng(2)
    for _ in range(200):
        cells, w, y, t, rel, _ = random_question(rng, int(rng.integers(1, 12)), 4)
        bq = cb.BatchQuestion(cells, w, y, t, cb.batch_keys(rel))
        order = rng.permutation(bq.units)
        c, s = bq.evaluate(order)
        inst = e.threshold_instance(cells, w, t)
        for k in range(1, len(order) + 1):
            paid = np.flatnonzero(np.isin(bq.batch_of, order[:k]))
            L = int(w[paid][y[paid] == e.MEMBER].sum())
            N = int(w[paid][y[paid] == e.NONMEMBER].sum())
            certified = L >= t or N >= inst.b0
            if certified:
                assert k == c and s == ("positive" if L >= t else "negative")
                break
        else:
            assert s == "unresolved"


def test_singleton_batches_are_the_per_cell_policies_on_random_questions():
    rng = np.random.default_rng(3)
    for _ in range(300):
        cells, w, y, t, _, p = random_question(rng, int(rng.integers(1, 14)), 3)
        p = np.round(p, 3)
        p /= p.sum(1, keepdims=True)
        inst = e.threshold_instance(cells, w, t)
        bq = cb.BatchQuestion(cells, w, y, t, cb.batch_keys(-np.ones(len(cells), dtype=int)))
        assert bq.units == len(cells)
        pairs = (
            (e.weight_order(inst), cb.weight_order(bq)),
            (e.positive_order(inst, p), cb.member_order(bq, p)),
            (e.direction_order(inst, p), cb.direction_order(bq, p)),
            (e.run_adaptive(inst, y, e.AdaptiveGreedy(inst, p)), cb.run_greedy(bq, p)),
            (e.run_adaptive(inst, y, e.AdaptiveGreedy(inst, p)), cb.run_sum_greedy(bq, p)),
            (e.run_adaptive(inst, y, e.RecalibratedGreedy(inst, p)), cb.run_recalibrated(bq, p)),
        )
        for cell_order, batch_order in pairs:
            assert e.evaluate(inst, y, cell_order) == bq.evaluate(batch_order)


def test_batch_prior_is_the_weight_average_and_exact_for_single_cells():
    cells = np.array([1, 2, 3, 4])
    w = np.array([1, 3, 2, 5])
    y = np.array([0, 0, 1, 2])
    bq = cb.BatchQuestion(cells, w, y, 4, np.array([7, 7, -1, 9]))
    p = np.array([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]])
    q = bq.probabilities(p)
    assert q.shape == (3, 4) and np.allclose(q[:, 3], 0)
    assert np.allclose(q[0, :3], (1 * p[0] + 3 * p[1]) / 4)
    assert np.array_equal(q[1, :3], p[2]) and np.array_equal(q[2, :3], p[3])
    assert np.array_equal(bq.weights, [4, 2, 5]) and np.array_equal(bq.member, [4, 0, 0])


@needs_data
@pytest.mark.parametrize("view", cc.VIEWS)
def test_singleton_batches_reproduce_frozen_costs_on_development_questions(view):
    from connectomequest.neurosym import common as nc

    frozen = cc.frozen(view, "development")
    row = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(frozen["family"], frozen["rho"]))}
    arm = frozen["arms"]
    priors = ("mlp_s0", "provisional") if view == "flywire" else ("mlp_s0",)
    for f in nc.development_families(view)[::40][:40].tolist():
        fam = nc.family(view, f)
        keys = cb.batch_keys(-np.ones(len(fam["cells"]), dtype=int))
        for rho, t in nc.thresholds(fam):
            bq = cb.BatchQuestion(fam["cells"], fam["weights"], fam["outcomes"], t, keys)
            i = row[(f, rho)]
            assert bq.evaluate(cb.weight_order(bq))[0] == frozen["cost"][i, arm["weight"]]
            for prior in priors:
                p = fam["priors"][prior]
                assert bq.evaluate(cb.member_order(bq, p))[0] == frozen["cost"][i, arm[f"{prior}:member"]]
                assert bq.evaluate(cb.direction_order(bq, p))[0] == frozen["cost"][i, arm[f"{prior}:direction"]]
                assert bq.evaluate(cb.run_greedy(bq, p))[0] == frozen["cost"][i, arm[f"{prior}:ag"]]
                assert bq.evaluate(cb.run_recalibrated(bq, p))[0] == frozen["cost"][i, arm[f"{prior}:recal"]]


# --------------------------------------------------------------------------- heterogeneous costs


def test_cost_aware_rules_with_unit_costs_are_the_plain_rules():
    rng = np.random.default_rng(4)
    for _ in range(300):
        cells, w, y, t, _, p = random_question(rng, int(rng.integers(1, 14)), 3)
        inst = e.threshold_instance(cells, w, t)
        one = np.ones(inst.size)
        assert np.array_equal(ch.weight_per_cost_order(inst, one), e.weight_order(inst))
        assert np.array_equal(ch.run_cost_aware(inst, y, p, one), e.run_adaptive(inst, y, e.AdaptiveGreedy(inst, p)))
        assert np.array_equal(
            ch.run_cost_aware(inst, y, p, one, recalibrated=True), e.run_adaptive(inst, y, e.RecalibratedGreedy(inst, p))
        )


def test_two_level_clairvoyant_cost_is_the_cheapest_certificate():
    rng = np.random.default_rng(5)
    for _ in range(300):
        cells, w, y, t, _, p = random_question(rng, int(rng.integers(1, 9)), 3)
        inst = e.threshold_instance(cells, w, t)
        cost = np.where(rng.random(inst.size) < 0.4, int(rng.integers(2, 4)), 1)
        best = None
        for mask in product((0, 1), repeat=inst.size):
            s = np.flatnonzero(mask)
            if w[s][y[s] == e.MEMBER].sum() >= t or w[s][y[s] == e.NONMEMBER].sum() >= inst.b0:
                c = int(cost[s].sum())
                best = c if best is None else min(best, c)
        assert ch.clairvoyant_cost(inst, y, cost) == best
        if best is not None:
            for order in (ch.weight_per_cost_order(inst, cost), ch.run_cost_aware(inst, y, p, cost),
                          ch.run_cost_aware(inst, y, p, cost, recalibrated=True)):
                n, s = e.evaluate(inst, y, order)
                assert s == e.full_information_status(inst, y)
                assert ch.paid_cost(order, cost, n) >= best


def test_cost_aware_greedy_prefers_cheap_cells_of_equal_gain():
    inst = e.threshold_instance(np.array([1, 2]), np.array([5, 5]), 5)
    p = np.array([[0.9, 0.1, 0.0], [0.9, 0.1, 0.0]])
    y = np.array([0, 0])
    assert e.run_adaptive(inst, y, e.AdaptiveGreedy(inst, p))[0] == 0
    assert ch.run_cost_aware(inst, y, p, np.array([3.0, 1.0]))[0] == 1


# --------------------------------------------------------------------------- protocol


def test_guard_refuses_the_held_out_split_without_a_matching_record(tmp_path):
    assert cc.guard("development", __file__) is None
    with pytest.raises(PermissionError):
        cc.guard("test", __file__, record=tmp_path / "absent.json")
    record = tmp_path / "record.json"
    files = cc.imported_round5_files(__file__)
    record.write_text('{"code": {}}')
    with pytest.raises(PermissionError):
        cc.guard("test", __file__, record=record)
    import json

    record.write_text(json.dumps({"workstreams": {"costs": {"code": {f: cc.hash_file(cc.ROOT / f) for f in files}}}}))
    assert cc.guard("test", __file__, record=record)
    record.write_text(json.dumps({"code": {f: "0" * 64 for f in files}}))
    with pytest.raises(PermissionError):
        cc.guard("test", __file__, record=record)


def test_outputs_are_never_overwritten(tmp_path):
    path = tmp_path / "x.json"
    cc.write_new(path, "a")
    with pytest.raises(FileExistsError):
        cc.write_new(path, "b")
    assert path.read_text() == "a"
    assert not list(tmp_path.glob(".*tmp"))


# --------------------------------------------------------------------------- biological readout


def test_permutation_test_and_holm_on_toy_tables():
    from connectomequest.round5 import costs_bio as cbio

    N = np.full(40, 20.0)
    flag = np.arange(40) < 10
    D = np.where(flag, 10.0, 1.0)  # strong enrichment among flagged units
    res = cbio.permutation_test(D, N, flag, permutations=2000)
    assert res["p_value"] < 0.01 and res["rate_difference"] == pytest.approx(0.5 - 0.05)
    assert res["rate_difference_95"][0] > 0.3
    D = np.full(40, 2.0)
    res = cbio.permutation_test(D, N, flag, permutations=500)
    assert res["rate_difference"] == 0 and res["p_value"] == 1.0
    adj = cbio.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


@needs_data
@needs_raw
def test_lenient_statuses_reproduce_the_frozen_lenient_analysis():
    import json

    from connectomequest.round5 import costs_bio as cbio

    ref = json.loads((cc.ROOT / "outputs/v35/analysis/lenient-development.json").read_text())
    for view in cc.VIEWS:
        fr = cc.frozen(view, "development")
        lenient = cbio.lenient_status(view, "development")
        for mine, theirs in ((fr["truth"], "strict"), (lenient, "lenient")):
            for k, v in (("positive", 1), ("negative", 2), ("unresolved", 0)):
                assert float((mine == v).mean()) == pytest.approx(ref[view]["answers"][theirs][k], abs=1e-12)
        decided = fr["truth"] > 0
        assert np.array_equal(lenient[decided], fr["truth"][decided])  # lenient only adds 'no'
        assert np.all(lenient[fr["truth"] == 0] != 1)


@needs_data
@needs_raw
def test_truncation_shares_are_bounded_and_cropped_neurons_absent_from_the_source_graph():
    from connectomequest.round5 import costs_bio as cbio

    tr = cbio.truncation_by_type()
    for key in ("pre_share", "post_share"):
        assert np.all((tr[key] >= 0) & (tr[key] <= 1))
    assert tr["cropped_total"] > 100 and tr["cropped_in_graph"] < 0.1 * tr["cropped_total"]
