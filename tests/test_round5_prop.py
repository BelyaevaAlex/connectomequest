"""Round 5, workstream prop: answer propagation with other priors and other policies."""

import json

import numpy as np
import pytest

from connectomequest.annotation_v35 import engine as e
from connectomequest.round5 import prop as P

ROOT = P.ROOT


def _random_instance(rng, n):
    w = rng.integers(1, 30, size=n)
    inst = e.threshold_instance(np.arange(100, 100 + n), w, int(rng.integers(1, w.sum() + 1)))
    y = rng.integers(0, 3, size=n)
    p = rng.dirichlet(np.ones(3), size=n)
    s = rng.random((n, n))
    kern = P.kk.kernel((s + s.T) / 2, P.SETTING["gamma"], P.SETTING["tau"])
    return inst, y, p, kern


def test_lambda_zero_fixed_orders_are_the_static_orders():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 25))
        inst, y, p, kern = _random_instance(rng, n)
        for rule, static in (("member", e.positive_order(inst, p)), ("direction", e.direction_order(inst, p))):
            order = e.run_adaptive(inst, y, P.make_policy(rule, inst, p, kern, 0.0))
            assert e.evaluate(inst, y, order) == e.evaluate(inst, y, static)
            assert np.array_equal(order, static[: len(order)])


def test_propagated_orders_are_sound_and_keep_normalized_beliefs():
    rng = np.random.default_rng(1)
    for _ in range(100):
        n = int(rng.integers(1, 30))
        inst, y, p, kern = _random_instance(rng, n)
        truth = e.full_information_status(inst, y)
        for policy in P.POLICIES:
            pol = P.make_policy(policy, inst, p, kern, P.SETTING["lam"])
            order = e.run_adaptive(inst, y, pol)
            e.check_probabilities(inst, pol.p)
            assert len(set(order.tolist())) == len(order)
            cost, status = e.evaluate(inst, y, order)
            assert status == truth and cost >= (e.clairvoyant_cost(inst, y) or 0)


def test_direction_commits_before_the_first_check():
    rng = np.random.default_rng(2)
    for _ in range(50):
        inst, y, p, kern = _random_instance(rng, 12)
        pol = P.make_policy("direction", inst, p, kern, P.SETTING["lam"])
        (_, fp), (_, fn) = e.direction_forecasts(inst, p)
        assert pol.rule == ("member" if fp <= fn else "nonmember")


def test_propagation_moves_beliefs_only_through_the_kernel():
    inst = e.threshold_instance(np.arange(4), np.array([5, 4, 3, 2]), 6)
    p = np.tile((0.5, 0.5, 0.0), (4, 1))
    kern = np.zeros((4, 4))
    kern[0, 2] = kern[2, 0] = 1.0
    pol = P.make_policy("member", inst, p, kern, 8.0)
    pol.observe(0, e.MEMBER)
    assert np.allclose(pol.p[2], ((0.5 + 8) / 9, 1 - (0.5 + 8) / 9, 0.0))
    assert np.array_equal(pol.p[[1, 3]], p[[1, 3]])


def test_arm_names_round_trip():
    for name in ("mlp_s0:recal+prop", "const_half:ag", "provisional:direction+prop"):
        assert P.Arm.parse(name).name == name
    assert P.Arm.parse("mlp_s0:member+prop").static.name == "mlp_s0:member"
    with pytest.raises(ValueError):
        P.Arm.parse("mlp_s0:adg+prop")


def test_cluster_bootstrap_shares_partition():
    rng = np.random.default_rng(3)
    types = np.repeat(rng.integers(0, 30, size=50), 4)
    x, y = rng.integers(1, 9, size=200), rng.integers(1, 9, size=200)
    mask = rng.random(200) < 0.7
    s = P.Clusters(types).shares(x, y, mask)
    assert abs(sum(s[k]["value"] for k in s) - 1) < 1e-12
    r = P.Clusters(types).ratio(x, y, mask)
    assert r["low"] <= r["value"] <= r["high"]


# --------------------------------------------------------------------------- runner protocol (no data needed)


def runner_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("r5_prop_run", ROOT / "scripts/r5_prop_run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_runner_refuses_the_held_out_split_without_a_record():
    m = runner_module()
    record = m.RECORD
    if record.exists():
        pytest.skip("a round5 record exists; refusal without a record cannot be tested here")
    with pytest.raises(SystemExit, match="refusing --split test"):
        m.require_record("test", [])
    assert m.require_record("development", []) is None


def test_runner_refuses_a_record_with_other_hashes(tmp_path, monkeypatch):
    m = runner_module()
    fake = tmp_path / "record.json"
    need = m.required_files([])
    assert m.rel(m.RUNNER) in need and "src/connectomequest/round5/prop.py" in need
    wrong = {name: "0" * 64 for name in need}
    fake.write_text(json.dumps({"code": wrong}))
    monkeypatch.setattr(m, "RECORD", fake)
    with pytest.raises(SystemExit, match="hash differs"):
        m.require_record("test", [])
    partial = dict(need)
    partial.pop("src/connectomequest/round5/prop.py")
    fake.write_text(json.dumps({"workstreams": {"prop": {"code": partial}}}))
    with pytest.raises(SystemExit, match="not listed"):
        m.require_record("test", [])
    fake.write_text(json.dumps({"workstreams": {"prop": {"code": need}}}))
    assert m.require_record("test", []) == need
    other = {"result_sha256": "1" * 64, "x": {"result_sha256": "2" * 64}}  # non-path hashes are ignored, not conflicts
    fake.write_text(json.dumps({"files": [{"path": k, "sha256": v} for k, v in need.items()], **other}))
    assert m.require_record("test", []) == need


# --------------------------------------------------------------------------- against the stored runs (needs the benchmark)


needs_data = pytest.mark.skipif(
    not all((P.c.VIEW_ROOTS[v] / "public/topology.npz").exists() and (P.c.BENCH / f"{v}.npz").exists() for v in P.c.VIEW_ROOTS),
    reason="needs the connectome graphs and the benchmark arrays",
)


@needs_data
def test_family_and_kernel_match_the_recorded_pilot():
    from connectomequest.neurosym import common as c
    from connectomequest.neurosym import kernel as kk

    src = P.source("flywire", external=False)
    for f in c.development_families("flywire")[:6]:
        ref = c.family("flywire", int(f))
        fam = P.family(src, int(f), ["mlp_s0", "provisional"])
        for key in ("cells", "weights", "labels", "codes", "outcomes"):
            assert np.array_equal(fam[key], ref[key])
        for name in ("mlp_s0", "provisional"):
            assert np.array_equal(fam["priors"][name], ref["priors"][name])
        assert np.array_equal(P.question_kernel(src, fam["cells"]), kk.question_kernel("flywire", ref, P.SETTING["gamma"], P.SETTING["tau"]))


@needs_data
@pytest.mark.parametrize("view", ["flywire", "malecns"])
def test_runner_reproduces_frozen_and_recorded_costs(view, tmp_path):
    """lam = 0 gives the frozen member / direction / AG / recal costs; lam = 8 the recorded propagation costs."""
    from connectomequest.neurosym import common as c
    from connectomequest.neurosym import harness as h

    src = P.source(view, external=False)
    fams = c.development_families(view)[:: max(1, len(c.development_families(view)) // 40)][:40]
    arms = [P.Arm("mlp_s0", pol, False) for pol in P.POLICIES] + [P.Arm("mlp_s0", pol, True) for pol in ("ag", "recal")]
    arms += [P.Arm("const_half", "ag", False), P.Arm("const_share", "recal", False)]
    res = P.run_sessions(src, fams, arms, tmp_path / "parts", workers=1, chunk=8, signature="test", log=lambda *_: None)
    existing = P.existing_arms(view, "development")
    rows = h.rows(view, c.development_families(view), "development")
    key = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(h.frozen(view)["family"][rows], h.frozen(view)["rho"][rows]))}
    at = np.array([key[(int(f), int(r))] for f, r in zip(res["family"], res["rho"])])
    names = {"mlp_s0:member": "mlp_s0:member", "mlp_s0:direction": "mlp_s0:direction", "mlp_s0:ag": "mlp_s0:ag", "mlp_s0:recal": "mlp_s0:recal",
             "mlp_s0:ag+prop": "mlp_s0:ag+prop", "mlp_s0:recal+prop": "mlp_s0:recal+prop", "const_half:ag": "weight", "const_share:recal": "const_share:recal"}
    for col, arm in enumerate(arms):
        assert np.array_equal(res["cost"][:, col], existing[names[arm.name]][at]), arm.name
    assert np.array_equal(existing["const_share:recal"], existing["weight"])
    # resuming from the same parts directory recomputes nothing and gives the same costs
    again = P.run_sessions(src, fams, arms, tmp_path / "parts", workers=1, chunk=8, signature="test", log=lambda *_: None)
    assert all(np.array_equal(again[k], res[k]) for k in res)
    with pytest.raises(ValueError, match="another run"):
        P.run_sessions(src, fams, arms[:2], tmp_path / "parts", workers=1, chunk=8, signature="test", log=lambda *_: None)


@needs_data
def test_cluster_bootstrap_is_the_harness_bootstrap():
    from connectomequest.neurosym import common as c
    from connectomequest.neurosym import harness as h

    fams = c.development_families("flywire")
    a = P.existing_arms("flywire", "development")
    mask = h.truth("flywire", fams) > 0
    ours = P.Clusters(P.session_types("flywire", fams)).ratio(a["mlp_s0:recal"], a["weight"], mask)
    assert ours == h.ratio("flywire", fams, a["mlp_s0:recal"], a["weight"], mask)


@needs_data
def test_profiles_from_graph_is_the_kernel_profile():
    from connectomequest.neurosym import common as c
    from connectomequest.neurosym import kernel as kk

    a, b = P.profiles_from_graph(c.view_data("flywire")["graph"]), kk.profiles("flywire")
    assert a.shape == b.shape and abs(a - b).max() == 0


@needs_data
def test_runner_never_overwrites_a_result(tmp_path, monkeypatch):
    m = runner_module()
    monkeypatch.setattr(m, "OUT", tmp_path)
    (tmp_path / "flywire-development-x.json").write_text("{}")
    with pytest.raises(FileExistsError):
        m.run("flywire", "development", ["const_half"], ["member"], "x", 1, 8, False)


@needs_data
@pytest.mark.parametrize("view", ["flywire", "malecns", "banc"])
def test_current_code_reproduces_the_stored_development_results(view, tmp_path):
    """A sample of families recomputed with the current code gives the stored development costs of this workstream."""
    m = runner_module()
    if view == "banc" and not (ROOT / "outputs/round5/banc/benchmark.npz").exists():
        pytest.skip("the banc view is not built")
    stored = m.result_files(view, "development")
    if not stored:
        pytest.skip("no development results of this workstream yet")
    src = P.source(view)
    fams = P.split_families(src, "development")
    sample = fams[:: max(1, len(fams) // 30)][:30]
    for path in stored:
        meta = json.loads(path.with_suffix(".json").read_text())
        arms = [P.Arm.parse(n) for n in meta["arms"] if n not in m.stale_arms(meta)]
        if not arms:
            continue
        z = P._load(path)
        res = P.run_sessions(src, sample, arms, tmp_path / path.stem, workers=1, chunk=10, signature="check", log=lambda *_: None)
        key = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(z["family"], z["rho"]))}
        at = np.array([key[(int(f), int(r))] for f, r in zip(res["family"], res["rho"])])
        for col, arm in enumerate(arms):
            assert np.array_equal(res["cost"][:, col], z["cost"][at, meta["arms"].index(arm.name)]), arm.name


def test_half_written_results_are_ignored_and_never_overwritten(tmp_path, monkeypatch):
    m = runner_module()
    monkeypatch.setattr(m, "OUT", tmp_path)
    np.savez(tmp_path / "flywire-development-a.npz", x=np.arange(3))
    assert m.result_files("flywire", "development") == []
    (tmp_path / "flywire-development-a.json").write_text("{}")
    assert [p.name for p in m.result_files("flywire", "development")] == ["flywire-development-a.npz"]
    m.write_new(tmp_path / "new.txt", lambda s: s.write(b"1"))
    with pytest.raises(FileExistsError):
        m.write_new(tmp_path / "new.txt", lambda s: s.write(b"2"))
    assert (tmp_path / "new.txt").read_text() == "1" and not list(tmp_path.glob(".*.tmp"))


@needs_data
def test_a_chunk_with_other_sessions_is_refused(tmp_path):
    from connectomequest.neurosym import common as c

    src = P.source("flywire", external=False)
    fams = c.development_families("flywire")[:4]
    arms = [P.Arm("mlp_s0", "member", True)]
    P.run_sessions(src, fams, arms, tmp_path / "parts", workers=1, chunk=2, signature="s", log=lambda *_: None)
    chunk = tmp_path / "parts" / "chunk-00000.npz"
    z = dict(np.load(chunk))
    P.atomic_savez(chunk, **{k: v[:-1] for k, v in z.items()})  # one session missing
    with pytest.raises(AssertionError, match="exactly the sessions"):
        P.run_sessions(src, fams, arms, tmp_path / "parts", workers=1, chunk=2, signature="s", log=lambda *_: None)


def test_arms_on_a_changed_prior_file_are_stale(tmp_path, monkeypatch):
    m = runner_module()
    f = tmp_path / "seededX-flywire.npz"
    f.write_bytes(b"new")
    meta = {"arms": ["seededX:recal", "mlp_s0:recal+prop", "weight"],
            "data": {"prior:seededX": {"path": str(f), "sha256": "0" * 64}}}
    assert m.stale_arms(meta) == {"seededX:recal"}
    meta["data"]["prior:seededX"]["sha256"] = m.hash_file(f)
    assert m.stale_arms(meta) == set()
    f.unlink()
    assert m.stale_arms(meta) == {"seededX:recal"}
