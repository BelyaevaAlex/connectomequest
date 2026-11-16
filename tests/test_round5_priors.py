"""Tests of the round-5 workstream ``priors`` (seeded type predictor, correlated-error priors, runners)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from connectomequest.annotation_v35 import engine as e
from connectomequest.annotation_v35 import sweep as sw
from connectomequest.round5 import priors_common as pc
from connectomequest.round5 import priors_corr as cr
from connectomequest.round5 import priors_model as pm

ROOT = pc.ROOT
HAVE_BENCH = (ROOT / "outputs/v35/benchmark/flywire.npz").exists() and (ROOT / "outputs/neurosym/cache/flywire.npz").exists()
needs_bench = pytest.mark.skipif(not HAVE_BENCH, reason="benchmark arrays or neurosym cache absent")


def population(rng, n, member_rate=0.2, unresolved_rate=0.35):
    member = rng.random(n) < member_rate
    unresolved = ~member & (rng.random(n) < unresolved_rate)
    return np.where(member, e.MEMBER, np.where(unresolved, e.UNRESOLVED, e.NONMEMBER))


# --------------------------------------------------------------------------- correlated-error priors


def test_constant_modulation_reproduces_the_sweep_priors_exactly():
    rng = np.random.default_rng(0)
    y = population(rng, 500)
    noise = rng.standard_normal(len(y))
    for level in (0.6, 0.8, 0.95):
        a = cr.calibrated(y, noise, np.full(len(y), sw.separation(level)), 0.2, 0.3)
        b = sw.synthetic_prior(y, noise, level, 0.2, 0.3)
        assert np.array_equal(a, b)
        flip, alt = rng.random(len(y)), rng.random(len(y))
        assert np.array_equal(cr.confident(y, flip, alt, np.full(len(y), sw.flip_probability(level))), sw.oracle_prior(y, flip, alt, level))


def test_weight_streams_are_the_sweep_streams_and_type_streams_are_shared():
    n, family, draw = 7, 123, 2
    noise, flip, alt = cr.random_numbers(family, draw, n, np.arange(n), "wplus")
    assert np.array_equal(noise, np.random.default_rng([29, family, draw]).standard_normal(n))
    u = np.random.default_rng([31, family, draw]).random((2, n))
    assert np.array_equal(flip, u[0]) and np.array_equal(alt, u[1])
    groups = np.array([5, 5, -1, 9, 5, -1, 9])
    noise, flip, alt = cr.random_numbers(family, draw, n, groups, "rtype")
    for x in (noise, flip, alt):
        assert x[0] == x[1] == x[4] and x[3] == x[6]
        assert len({x[0], x[2], x[3], x[5]}) == 4  # cells without a release type are groups of their own


def test_weight_percentile_and_modulation_direction():
    r = cr.weight_percentile(np.array([1, 5, 5, 10]))
    assert np.allclose(r, [0, 0.5, 0.5, 1])
    assert np.allclose(cr.weight_percentile(np.array([3])), [0.5])
    up, down = cr.modulation(r, "wplus"), cr.modulation(r, "wminus")
    assert up[3] / up[0] == pytest.approx(4.0) and down[0] / down[3] == pytest.approx(4.0)
    assert np.allclose(cr.modulation(r, "rtype"), 1.0)


def test_heavy_cells_are_more_accurate_under_wplus_and_less_under_wminus():
    rng = np.random.default_rng(3)
    n = 200_000
    y = population(rng, n)
    pct = rng.random(n)
    numbers = (rng.standard_normal(n), rng.random(n), rng.random(n))
    heavy = pct > 0.8
    light = pct < 0.2
    for form, scale in (("base", 1.2), ("oracle", 0.25)):
        for corr, sign in (("wplus", 1), ("wminus", -1)):
            q = cr.prior(form, corr, scale, y, pct, numbers, 0.2, 0.35)
            a_heavy = sw.member_auroc(q[heavy, e.MEMBER], y[heavy])
            a_light = sw.member_auroc(q[light, e.MEMBER], y[light])
            assert sign * (a_heavy - a_light) > 0.05
            assert np.allclose(q.sum(1), 1.0)


def test_heterogeneous_calibrated_prior_stays_calibrated():
    rng = np.random.default_rng(4)
    n = 400_000
    y = population(rng, n, member_rate=0.15, unresolved_rate=0.4)
    pct = rng.random(n)
    q = cr.prior("base", "wplus", 1.0, y, pct, (rng.standard_normal(n), None, None), 0.15, 0.4)
    bins = np.digitize(q[:, e.MEMBER], np.linspace(0.1, 0.9, 9))
    for b in np.unique(bins):
        m = bins == b
        if m.sum() > 2000:
            assert abs((y[m] == e.MEMBER).mean() - q[m, e.MEMBER].mean()) < 0.02


def test_grouped_auroc_matches_the_sweep_definition_with_ties():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(30), rng.integers(1, 40, size=30))
    rng.shuffle(groups)
    y = population(rng, len(groups))
    score = rng.integers(0, 4, size=len(groups)).astype(float) / 3  # many ties
    got = cr.grouped_auroc(score, y == e.MEMBER, groups)
    for g in range(30):
        m = groups == g
        want = sw.member_auroc(score[m], y[m])
        assert (want is None and np.isnan(got[g])) or got[g] == pytest.approx(want)


def test_fit_scale_inverts_a_monotone_function():
    assert cr.fit_scale(lambda c: 1 - np.exp(-c), 0.5, 0.0, 10.0, True) == pytest.approx(np.log(2), abs=1e-6)
    assert cr.fit_scale(lambda c: np.exp(-c), 0.25, 0.0, 10.0, False) == pytest.approx(np.log(4), abs=1e-6)


# --------------------------------------------------------------------------- protocol helpers


def test_seed_sets_are_deterministic_nested_and_of_the_right_size():
    ids = np.arange(100_000, 120_000)
    u = pc.seed_uniform("flywire", ids)
    assert np.array_equal(u, pc.seed_uniform("flywire", ids))
    assert abs((u < 0.3).mean() - 0.3) < 0.02
    assert np.all((u < 0.1) <= (u < 0.3)) and np.all((u < 0.3) <= (u < 0.5))
    assert not np.array_equal(u, pc.seed_uniform("malecns", ids))
    assert abs(np.corrcoef(u, pc.calibration_uniform("flywire", ids))[0, 1]) < 0.05


def test_guard_refuses_the_held_out_split_without_a_matching_record(tmp_path):
    script = ROOT / "scripts/r5_priors_run.py"
    assert pc.guard("development", script) is None
    with pytest.raises(PermissionError):
        pc.guard("test", script, record=tmp_path / "missing.json")
    files = pc.imported_round5_files(script)
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"code": {f: "0" * 64 for f in files}}))
    with pytest.raises(PermissionError):
        pc.guard("test", script, record=wrong)
    right = tmp_path / "right.json"
    right.write_text(json.dumps({"round5": {"priors": {"code": {f: pc.hash_file(ROOT / f) for f in files}}}}))
    assert set(pc.guard("test", script, record=right)) == set(files)
    with pytest.raises(ValueError):
        pc.guard("validation", script)


def test_held_out_runs_need_recorded_data_files(tmp_path):
    data = pc.OUT / "README.md"
    if not data.exists():
        pytest.skip("no data file to test with")
    assert pc.guard_data("development", [data]) is None
    rel = str(data.relative_to(ROOT))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"files": {rel: "0" * 64}}))
    with pytest.raises(PermissionError):
        pc.guard_data("test", [data], record=bad)
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"files": {rel: pc.hash_file(data)}}))
    assert pc.guard_data("test", [data], record=good) == {rel: pc.hash_file(data)}


def test_outputs_are_never_overwritten(tmp_path):
    path = tmp_path / "x.npz"
    pc.save_npz_new(path, a=np.arange(3))
    with pytest.raises(FileExistsError):
        pc.save_npz_new(path, a=np.arange(4))
    assert np.array_equal(np.load(path)["a"], np.arange(3))
    pc.write_json_new(tmp_path / "x.json", {"a": 1})
    with pytest.raises(FileExistsError):
        pc.write_json_new(tmp_path / "x.json", {"a": 2})


@pytest.mark.parametrize("script", ["r5_priors_run.py", "r5_priors_corrsweep.py"])
def test_runners_refuse_the_held_out_split_without_the_record(script, tmp_path):
    if (ROOT / "outputs/round5/record.json").exists():
        pytest.skip("a record exists; the refusal depends on its content")
    env = {"PYTHONPATH": "src:.v32-deps:scripts", "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--view", "flywire", "--split", "test", "--output", str(tmp_path)],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode != 0 and "PermissionError" in out.stderr
    assert not any(tmp_path.iterdir())


# --------------------------------------------------------------------------- predictor pieces


def test_first_distinct_and_candidate_stats():
    labels = np.array([[3, 5, 3, 7, -1], [2, 2, 2, 2, 2]])
    weights = np.array([[1.0, 0.9, 0.8, 0.1, 5.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    top = pm.first_distinct(labels, weights, 3)
    assert top.tolist() == [[3, 5, 7], [2, -1, -1]]
    cand = np.array([[0, 3, 5, -1], [0, 2, 9, -1]])
    st = pm.candidate_stats(cand, labels, weights)
    assert st[0, 1, 0] == pytest.approx(1.8 / 7.8) and st[0, 1, 2] == 1 and st[0, 2, 1] == pytest.approx(np.log(2))
    assert st[1, 1, 0] == pytest.approx(1.0) and st[1, 2, 0] == 0 and st[1, 2, 1] == pytest.approx(np.log1p(5))


def test_scorer_learns_a_toy_ranking_and_returns_distributions():
    rng = np.random.default_rng(6)
    n, c = 3000, 6
    cf = rng.standard_normal((n, c, 3)).astype(np.float32)
    target = rng.integers(0, c, size=n)
    cf[np.arange(n), target, 0] += 3.0
    qf = rng.standard_normal((n, 2)).astype(np.float32)
    mask = np.ones((n, c), dtype=bool)
    mask[:, -1] = False
    target = np.where(target == c - 1, c, target)  # the masked slot's class counts as a miss
    sc = pm.Scorer(epochs=15, threads=1).fit(cf[:2000], qf[:2000], mask[:2000], target[:2000], log=lambda *a: None)
    lg = sc.logits(cf[2000:], qf[2000:], mask[2000:])
    sc.fit_temperature(lg, target[2000:])
    p = sc.probabilities(lg)
    assert np.allclose(p.sum(1), 1.0) and np.all(p[:, c - 1] < 1e-6)
    hit = target[2000:] < c
    assert (p[hit].argmax(1) == target[2000:][hit]).mean() > 0.8


def test_class_mass_is_a_distribution_over_the_vocabulary():
    rng = np.random.default_rng(7)
    n, v, k = 20, 12, 4
    mlp = rng.random((n, v))
    mlp /= mlp.sum(1, keepdims=True)
    cand = np.stack([np.r_[0, rng.choice(np.arange(1, v), k - 1, replace=False)] for _ in range(n)])
    cand[3, 2:] = -1
    prob = rng.random((n, k)) * (cand >= 0)
    miss = rng.random(n)
    total = prob.sum(1) + miss
    pred = pm.Prediction(cand=cand, prob=prob / total[:, None], miss=miss / total, temperature=1.0, report={})
    rows = np.arange(n)
    full = pm.class_mass(pred, rows, np.arange(1, v), mlp)
    other = (pred.prob * (cand == 0)).sum(1)
    assert np.allclose(full + other, 1.0)
    singles = sum(pm.class_mass(pred, rows, [c], mlp) for c in range(1, v))
    assert np.allclose(singles, full)


# --------------------------------------------------------------------------- data-dependent checks


@needs_bench
def test_layout_reproduces_the_frozen_topology_prior_and_seed_mixing():
    from connectomequest.neurosym import common as nc
    from connectomequest.round5 import priors_layout as pl

    view = "flywire"
    d = nc.benchmark(view)
    mlp = np.load(ROOT / f"outputs/neurosym/cache/mlp-full-{view}.npy", mmap_mode="r")
    ids = nc.view_data(view)["ids"]
    mass = lambda rows, codes: np.asarray(mlp[rows])[:, codes].sum(1)
    other = np.asarray(mlp[:, 0])
    fams = nc.development_families(view)[:40]
    no_seed = np.zeros(len(ids), dtype=bool)
    plus, unknown, _ = pl.benchmark_arrays(view, mass, other, no_seed, np.zeros(len(ids), dtype=np.int64), 0.99, families=fams)
    done = ~np.isnan(plus)
    assert done.sum() > 1000 and np.allclose(plus[done], d["plus_mlp_s0"][done], atol=1e-5)
    assert np.allclose(unknown, d["unknown_mlp_s0"], atol=1e-6)
    label = nc.view_data(view)["labels"].clip(min=0)
    all_seed = np.ones(len(ids), dtype=bool)
    plus2, unknown2, seed = pl.benchmark_arrays(view, mass, other, all_seed, label, 0.99, families=fams)
    assert seed.all()
    for f in fams[:10]:
        lo, hi, start, y = pl.family_cells(view, f)
        want = 0.99 * (y == e.MEMBER) + 0.01 * d["plus_mlp_s0"][start : start + hi - lo]
        assert np.allclose(plus2[start : start + hi - lo], want, atol=1e-5)
        assert np.allclose(unknown2[lo:hi], 0.99 * (y == e.UNRESOLVED) + 0.01 * d["unknown_mlp_s0"][lo:hi], atol=1e-5)


@needs_bench
def test_quality_reproduces_the_frozen_calibration_table():
    from connectomequest.neurosym import common as nc
    from connectomequest.round5 import priors_layout as pl

    frozen = json.loads((ROOT / "outputs/v35/analysis/development.json").read_text())
    for view in ("flywire",):
        d = nc.benchmark(view)
        q = pl.quality(view, d["plus_mlp_s0"], d["unknown_mlp_s0"], nc.development_families(view))
        want = frozen[view]["calibration"]["mlp_s0"]
        for key in ("member_auroc_within_question", "member_brier", "member_ece", "mean_predicted_unresolved", "observed_unresolved_rate"):
            assert q[key] == pytest.approx(want[key], rel=1e-9)


@needs_bench
def test_runner_reproduces_frozen_costs_with_a_frozen_prior():
    sys.path.insert(0, str(ROOT / "scripts"))
    import r5_priors_run as run
    from connectomequest.neurosym import common as nc

    view = "flywire"
    bench = nc.benchmark(view)
    run._D.clear()
    run._D.update(bench)
    run._D["priors"] = {"mlp_s0": (bench["plus_mlp_s0"], bench["unknown_mlp_s0"])}
    names = run.arm_names(["mlp_s0"])
    run._D["arms"] = {n: i for i, n in enumerate(names)}
    meta = json.loads((ROOT / f"outputs/v35/runs/{view}-development.json").read_text())
    with np.load(ROOT / f"outputs/v35/runs/{view}-development.npz") as z:
        frozen = {k: z[k] for k in ("family", "rho", "cost")}
    arm = {n: i for i, n in enumerate(meta["arms"])}
    key = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(frozen["family"], frozen["rho"]))}
    checked = 0
    for f in nc.development_families(view)[:25]:
        for row in run.run_family(int(f)):
            i = key[row["family"], row["rho"]]
            for name in names:
                assert row["cost"][run._D["arms"][name]] == frozen["cost"][i, arm[name]]
            checked += 1
    assert checked > 20


@needs_bench
def test_corrsweep_without_correlation_reproduces_the_frozen_sweep(monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import r5_priors_corrsweep as cs
    from connectomequest.neurosym import common as nc

    view = "flywire"
    monkeypatch.setattr(cr, "BETA", 0.0)
    cs._D.clear()
    cs._D.update(nc.benchmark(view))
    vd = nc.view_data(view)
    cs._D["release"] = vd["release_code"][np.searchsorted(vd["ids"], cs._D["pre_cell"])].astype(np.int64)
    meta = json.loads((ROOT / f"outputs/v35/sweep/{view}-development.json").read_text())
    cs._D["rates"] = meta["rates"]
    cs._D["scales"] = {f"{form}/{c}/{lvl:g}": dict(scale=sw.separation(lvl) if form == "base" else sw.flip_probability(lvl))
                       for form in cr.FORMS for c in cr.CORRELATIONS for lvl in cs.LEVELS}
    with np.load(ROOT / f"outputs/v35/sweep/{view}-development.npz") as z:
        frozen = {k: z[k] for k in ("family", "rho", "cost")}
    key = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(frozen["family"], frozen["rho"]))}
    policy_index = [meta["policies"].index(p) for p in cs.POLICIES]
    checked = 0
    for f in np.unique(frozen["family"])[:6]:
        for row in cs.run_family(int(f)):
            i = key[row["family"], row["rho"]]
            for k, corr in enumerate(cr.CORRELATIONS):
                if corr == "rtype":
                    continue  # other random streams by design
                for m, form in enumerate(cr.FORMS):
                    want = frozen["cost"][i][:, :, cs.SWEEP_VARIANT[form]][:, :, policy_index]
                    assert np.array_equal(row["cost"][:, :, k, m, :], want)
            checked += 1
    assert checked >= 6


@needs_bench
def test_built_priors_have_the_benchmark_layout():
    from connectomequest.neurosym import common as nc
    from connectomequest.round5 import priors_layout as pl

    files = sorted(pc.OUT.glob("seeded*-*.npz"))
    if not files:
        pytest.skip("no seeded priors built yet")
    for path in files:
        name, view = path.stem.rsplit("-", 1)
        d = nc.benchmark(view)
        with np.load(path) as z:
            plus, unknown, seed = z["plus_" + name], z["unknown_" + name], z["seed_" + name]
            assert np.array_equal(z["pre_cell"], d["pre_cell"])
        assert plus.shape == (int(d["fam_plus_offset"][-1]),) and unknown.shape == d["pre_cell"].shape
        assert np.isfinite(plus).all() and plus.min() >= 0 and unknown.min() >= 0
        for f in nc.development_families(view)[:50]:
            lo, hi, start, y = pl.family_cells(view, f)
            p = plus[start : start + hi - lo]
            assert np.all(p + unknown[lo:hi] <= 1 + 1e-5)
            s = seed[lo:hi]
            assert np.all(p[s & (y == e.MEMBER)] >= 0.99 - 1e-6) and np.all(p[s & (y != e.MEMBER)] <= 0.01 + 1e-6)


def test_training_rows_never_count_themselves_in_class_features():
    n, vocab, k, codes = 2, 5, 3, 9
    ev = pm.Evidence("toy", 0.1)
    for name in pm.KNN_SOURCES:
        ev.sources[name] = (np.full((n, k), 2), np.ones((n, k)))
    ev.hemi = (np.full((n, 4), -1), np.zeros((n, 4)))
    table = np.zeros((vocab, codes))
    table[2, 3] = 1.0  # the only training seed of class 2 is row 0, transmitter 3
    ev.nt = (np.array([3, 3]), table, np.log(np.full(codes, 1.0 / codes)))
    for key in pm.QUERY_CONN + pm.QUERY_NONCONN:
        ev.query[key] = np.zeros(n)
    ev.query["_seed_counts"] = np.bincount([2], minlength=vocab).astype(float)
    ev.query["_own"] = np.array([2, -1])  # row 0 is a training seed of class 2, row 1 is not a training row
    cand = np.array([[0, 2, -1], [0, 2, -1]])
    mlp = np.full((n, vocab), 1.0 / vocab)
    cf, _ = pm.features(ev, np.arange(n), cand, mlp, conn_only=False)
    count_col, nt_col = cf.shape[-1] - 3, cf.shape[-1] - 1
    assert cf[0, 1, count_col] == pytest.approx(0.0) and cf[1, 1, count_col] == pytest.approx(np.log(2.0))
    assert cf[0, 1, nt_col] == pytest.approx(np.log(1.0 / codes) - np.log(1.0 / codes))  # as if class 2 had no seed
    assert cf[1, 1, nt_col] == pytest.approx(np.log(2.0 / (1 + codes)) - np.log(1.0 / codes))
