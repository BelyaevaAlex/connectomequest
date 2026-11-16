"""Round 5, workstream banc: BANC as a third evaluation view.

Pure tests (crosswalk, endpoint hash, edge rule, protocol guard) always run; tests of the built
files are skipped when the BANC benchmark has not been built.
"""

import json
import sys

import numpy as np
import pytest

from connectomequest.annotation_v32.runtime import hash_file
from connectomequest.round5 import banc as b

sys.path.insert(0, str(b.ROOT / "scripts"))
import r5_banc_build as rb  # noqa: E402
import v35_build  # noqa: E402

BUILT = all(p.exists() for p in (b.BENCH, b.MANIFEST, b.VIEWFILE, b.PREPARED / "public/topology.npz"))
RUN = (b.RUNS / "banc-development.npz").exists()
needs_build = pytest.mark.skipif(not BUILT, reason="needs the built BANC view (scripts/r5_banc_build.py)")
needs_run = pytest.mark.skipif(not (BUILT and RUN), reason="needs the BANC development run (scripts/r5_banc_run.py)")

LOOKUP = {"SMP001": 1, "PEN_a": 2, "group:KCab-mcs": 3, "KCg-m": 4, "PAM01": 5}


# --------------------------------------------------------------------------- crosswalk and construction rules


def test_crosswalk_declared_rule():
    assert b.crosswalk("SMP001", "KCg-m", LOOKUP) == (1, "release-type")  # own name first
    assert b.crosswalk("CB9999", "KCg-m", LOOKUP) == (4, "hemibrain-field")
    assert b.crosswalk(None, " KCg-m ", LOOKUP) == (4, "hemibrain-field")
    assert b.crosswalk("CB9999", "auto:KCg-m", LOOKUP) == (-1, "unresolved")  # unreviewed NBLAST hit
    assert b.crosswalk("CB9999", "KCg-m,SMP001", LOOKUP) == (-1, "unresolved")  # two declared types
    assert b.crosswalk("CB9999", "CB8888", LOOKUP) == (-1, "unresolved")
    assert b.crosswalk("SMP001,PEN_a", None, LOOKUP) == (-1, "unresolved")  # multi-valued own type
    assert b.crosswalk("PEN_a(PEN1)", None, LOOKUP) == (2, "release-type")  # declared alias
    assert b.crosswalk("KCab", None, LOOKUP) == (3, "release-type")  # FlyWire-style coarse name
    assert b.crosswalk("KC", "KCab-m", LOOKUP) == (3, "hemibrain-field")  # hemibrain-style fine name
    assert b.crosswalk("PAM01", "PAM01_b", LOOKUP) == (5, "release-type")


def test_groups_are_the_consistent_union_of_declared_maps():
    from connectomequest.annotation_v32.data import COMMON_GROUP_MAPPINGS

    for mapping in COMMON_GROUP_MAPPINGS.values():
        for name, group in mapping.items():
            assert b.GROUPS[name] == group
    assert set(b.GROUPS) == set().union(*COMMON_GROUP_MAPPINGS.values())


@pytest.mark.skipif(not (b.ROOT / v35_build.SOURCE / "public/manifest.json").exists(),
                    reason="needs the prepared hemibrain manifest")
def test_crosswalk_agrees_with_v35_where_the_maps_agree():
    vocabulary = json.loads((b.ROOT / v35_build.SOURCE / "public/manifest.json").read_text())["vocabulary"]
    lookup = {n: i for i, n in enumerate(vocabulary) if i}
    rng = np.random.default_rng(0)
    names = [vocabulary[i] for i in rng.integers(1, len(vocabulary), 300)] + ["CB9999", "auto:SMP001", None]
    for own in names[:150]:
        for declared in names[150:]:
            # no name of this sample is in a coarsening map, so the BANC rule equals the frozen rule
            if own in b.GROUPS or declared in b.GROUPS:
                continue
            assert b.crosswalk(own, declared, lookup) == v35_build.crosswalk(own, declared, "male-cns:v1.0", lookup)


def test_endpoint_key_is_the_v35_hash_with_view_banc():
    assert b.endpoint_key("MBON01", 720575941633499884) == v35_build.digest("cq-v35-endpoint", "banc", "MBON01", 720575941633499884)


def test_edge_matrix_sums_duplicates_before_the_five_synapse_rule():
    ids = np.array([10, 20, 30, 40])
    pre = np.array([10, 10, 20, 30, 30, 40, 99, 10])
    post = np.array([20, 20, 30, 10, 10, 40, 10, 30])
    count = np.array([3, 2, 4, 5, 1, 9, 50, 7])  # 10->20 twice (3+2), 20->30 below 5, 30->10 (5+1), 40->40 self, 99 outside
    matrix, valid = rb.edge_matrix(ids, pre, post, count)
    dense = matrix.toarray()
    assert dense[0, 1] == 5 and dense[1, 2] == 0 and dense[2, 0] == 6 and dense[0, 2] == 7
    assert dense[3, 3] == 9  # self pair still present here; _from_matrix removes it
    assert valid.tolist() == [True] * 6 + [False, True]
    from connectomequest.annotation_v32.data import _from_matrix

    graph = _from_matrix(ids, matrix, {int(i): None for i in ids}, ("__OTHER__", "A"), {})
    rows = np.repeat(np.arange(len(ids)), np.diff(graph.indptr))
    assert not np.any(rows == graph.indices) and graph.weights.min() >= 5


def test_guard_refuses_test_without_matching_record(tmp_path):
    script = b.ROOT / "scripts/r5_banc_run.py"
    assert b.guard("development", script) is None
    with pytest.raises(ValueError):
        b.guard("holdout", script)
    with pytest.raises(PermissionError):
        b.guard("test", script, record=tmp_path / "missing.json")
    files = b.imported_round5_files(script)
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"code": {f: "0" * 64 for f in files}}))
    with pytest.raises(PermissionError):
        b.guard("test", script, record=record)
    record.write_text(json.dumps({"code": {f: hash_file(b.ROOT / f) for f in files}}))
    assert b.guard("test", script, record=record) is not None
    extra = b.ROOT / "scripts/r5_banc_report.py"
    with pytest.raises(PermissionError):
        b.guard("test", script, extra=(extra,), record=record)
    record.write_text(json.dumps({"files": [{"path": f, "sha256": hash_file(b.ROOT / f)} for f in files + ["scripts/r5_banc_report.py"]]}))
    assert b.guard("test", script, extra=(extra,), record=record) is not None


def test_save_new_never_overwrites(tmp_path):
    path = b.save_new(tmp_path / "x.bin", lambda s: s.write(b"1"))
    with pytest.raises(FileExistsError):
        b.save_new(path, lambda s: s.write(b"2"))
    assert path.read_bytes() == b"1" and not list(tmp_path.glob(".x.bin.*"))


# --------------------------------------------------------------------------- the built view


@needs_build
def test_prepared_graph_follows_the_rules():
    from connectomequest.neurosym.common import load_graph

    with np.load(b.PREPARED / "public/topology.npz") as z:
        indptr, indices, weights = z["indptr"], z["indices"], z["weights"]
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    assert weights.min() >= 5 and not np.any(rows == indices)
    ids, graph = load_graph(b.PREPARED)
    assert np.array_equal(ids, b.view_data()["ids"]) and graph.nnz == len(weights)
    names = b.view_data()["names"]["super_class"]
    assert set(names) <= set(b.NEURONAL_SUPER_CLASSES)


@needs_build
def test_prepared_graph_matches_the_edgelist_on_sampled_pairs():
    import pyarrow.compute as pc
    import pyarrow.feather as feather

    table = feather.read_table(b.RAW / "banc_888_edgelist_simple_v2.feather", columns=["pre", "post", "count"])
    pre, post = pc.cast(table["pre"], "int64").to_numpy(), pc.cast(table["post"], "int64").to_numpy()
    count = table["count"].to_numpy()
    v = b.view_data()
    ids, graph = v["ids"], v["graph"].tocsr()
    pick = np.random.default_rng(1).choice(len(pre), 20000, replace=False)
    for i, j, n in zip(pre[pick], post[pick], count[pick]):
        a, c = np.searchsorted(ids, i), np.searchsorted(ids, j)
        inside = a < len(ids) and c < len(ids) and ids[a] == i and ids[c] == j
        if inside:  # the release has one record per pair
            assert graph[a, c] == (n if n >= 5 and i != j else 0)


@needs_build
def test_features_are_the_frozen_feature_code_on_the_banc_graph():
    from connectomequest.annotation_v32.data import load_public_graph
    from connectomequest.annotation_v32.predictors import structural_features

    graph = load_public_graph(b.PREPARED)
    with np.load(b.OUT / "features.npz") as z:
        assert np.array_equal(z["node_ids"], graph.node_ids)
        assert np.array_equal(z["features"], structural_features(graph))


@needs_build
def test_benchmark_layout_and_question_construction():
    d, m = b.benchmark(), b.manifest()
    frozen_keys = set(np.load(b.ROOT / "outputs/v35/benchmark/flywire.npz").files)
    structural = {k for k in frozen_keys if not k.startswith(("plus_", "unknown_"))}
    assert structural | {f"{s}_{p}" for p in b.priors() for s in ("plus", "unknown")} | {"fam_plus_offset"} == set(d)
    assert m["view"]["priors"] == ["mlp_s0", "const_half", "const_share"]
    n_pre = int(d["target_offset"][-1])
    sizes = np.diff(d["target_offset"])
    assert len(d["pre_cell"]) == n_pre and all(len(d["unknown_" + p]) == n_pre for p in b.priors())
    assert all(len(d["plus_" + p]) == int(sizes[d["fam_target"]].sum()) for p in b.priors())
    v35 = json.loads(b.V35_MANIFEST.read_text())
    types = {t["code"]: t for t in v35["types"]}
    assert [t["code"] for t in m["types"]] == [t["code"] for t in v35["types"]]
    assert all(t["split"] == types[t["code"]]["split"] for t in m["types"])
    v = b.view_data()
    graph = v["graph"]
    indegree = np.diff(graph.indptr)
    by_type = {}
    for t, code in enumerate(d["target_type"]):
        by_type.setdefault(int(code), []).append(int(d["target_cell"][t]))
    names = b.vocabulary()
    for code, cells in by_type.items():
        candidates = np.flatnonzero((v["labels"] == code) & (indegree >= 10))
        expected = sorted((int(v["ids"][i]) for i in candidates), key=lambda c: b.endpoint_key(names[code], c))[:2]
        assert cells == expected
    assert sum(len(t["banc_targets"]) for t in m["types"]) == len(d["target_cell"])
    for f in range(0, len(d["fam_target"]), 97):
        fam = b.family(f)
        t = types[fam["target_type"]]
        spec = t["families"][0] if fam["kind"] == 0 else t["families"][1]
        assert fam["codes"].tolist() == spec["codes"] and fam["source_weight"] == spec["source_weight"]
        assert fam["source_total"] == t["source_total_weight"]
        node = b.rows_of([fam["target_cell"]])[0]
        lo, hi = graph.indptr[node], graph.indptr[node + 1]
        pre = v["ids"][graph.indices[lo:hi]]
        order = np.argsort(pre)
        assert np.array_equal(fam["cells"], pre[order]) and np.array_equal(fam["weights"], graph.data[lo:hi][order])
        assert np.array_equal(fam["labels"], v["labels"][b.rows_of(fam["cells"])])
        assert len(fam["cells"]) >= 10


@needs_build
def test_stored_priors():
    from connectomequest.annotation_v32.training import StructuralMLP

    d = b.benchmark()
    model = StructuralMLP.load(b.ROOT / v35_build.MLP / "model.npz")
    temperature = json.loads((b.ROOT / v35_build.MLP / "report.json").read_text())["temperature"]
    with np.load(b.OUT / "features.npz") as z:
        features = z["features"]
    for f in range(0, len(d["fam_target"]), 211):
        fam = b.family(f)
        rows = b.rows_of(fam["cells"])
        full = v35_build.full_probabilities(model, temperature, features[rows])
        assert np.array_equal(full, b.mlp_probabilities(rows))
        plus = full[:, fam["codes"]].astype(np.float64).sum(1)
        p = fam["priors"]["mlp_s0"]
        assert np.allclose(p[:, 0], np.clip(plus, 0, 1), atol=2e-6) and np.allclose(p[:, 2], full[:, 0], atol=2e-6)
        assert np.all(fam["priors"]["const_half"] == [0.5, 0.5, 0.0])
        share = np.float32(np.clip(fam["source_weight"] / fam["source_total"], 0.01, 0.99))
        assert np.allclose(fam["priors"]["const_share"][:, 0], share) and np.all(fam["priors"]["const_share"][:, 2] == 0)


@needs_build
def test_profiles_and_neurosym_dispatch():
    from connectomequest.neurosym import common as c
    from connectomequest.neurosym import constraints as k
    from connectomequest.neurosym import kernel as kk

    b.install_neurosym_view()
    assert (kk.profiles("banc") != b.profiles()).nnz == 0
    norms = np.sqrt(np.asarray(b.profiles().multiply(b.profiles()).sum(1)).ravel())
    assert np.allclose(norms[norms > 0], 1.0)
    fams = c.split_families("banc", "development")
    assert np.array_equal(fams, b.split_families("development"))
    fam = c.family("banc", int(fams[0]))
    mine = b.family(int(fams[0]))
    assert all(np.array_equal(fam[x], mine[x]) for x in ("cells", "weights", "outcomes", "codes"))
    assert np.array_equal(k.rows_of("banc", fam["cells"]), b.rows_of(fam["cells"]))
    assert np.allclose(kk.question_kernel("banc", fam, 3.0, 0.3), b.question_kernel(fam, 3.0, 0.3))
    assert kk.contralateral("banc", b.rows_of(fam["cells"])).shape == (len(fam["cells"]),) * 2
    assert c.benchmark("flywire")["target_cell"].shape[0] == 6460  # other views pass through


@needs_run
def test_harness_runs_on_banc_and_reproduces_weight_order():
    from connectomequest.annotation_v35 import engine as e
    from connectomequest.neurosym import harness as h

    b.install_neurosym_view()
    fams = b.split_families("development")[:6]
    costs = h.run("banc", fams, lambda fam, inst, outcomes, rho: {"weight": e.weight_order(inst)}, workers=2)
    assert np.array_equal(costs["weight"], h.baseline("banc", fams, "weight"))


@needs_run
def test_development_run_identities():
    d = b.frozen("development")
    w = d["cost"][:, d["arms"]["weight"]]
    for name in ("const_half:recal", "const_share:ag"):
        assert np.array_equal(d["cost"][:, d["arms"][name]], w)  # no prediction = weight order
    cert = d["truth"] > 0
    assert np.all(d["clairvoyant"][cert] <= d["cost"][cert].min(1)) and np.all(d["clairvoyant"][~cert] == -1)
    assert set(np.unique(b.benchmark()["target_type"][b.benchmark()["fam_target"][d["family"]]])) <= {
        c for c, s in b.split_of().items() if s == "development"
    }
