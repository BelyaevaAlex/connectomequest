"""Tests on cell types that no earlier analysis used, with earlier labels in both connectomes and at partial coverage.

The benchmark asks about hemibrain cell types that have target neurons in both MaleCNS and FlyWire. The hemibrain
types with target neurons in exactly one of the two connectomes were never used, neither for development nor for a
held-out run. This protocol asks the questions of the replication (retention and share questions) about the first
two target neurons of each such type and evaluates, once, the policies whose exploratory results the paper reports.

It also adds two things that no earlier run had:

* an earlier-release prior for MaleCNS, built from the public release v0.9 of its annotation by the rule used for
  FlyWire (probability 0.9 on the earlier type name if it is a hemibrain type, 0.9 on the extra class if it is another
  name, the topology prior otherwise), with the release v1.0 as the reference;
* partial coverage: a fixed pseudo-random fraction of the neurons keeps the earlier-release prior and the others the
  topology prior (the function of ``scripts/v35_coverage.py``), with direction-fixed, AG, recalibrated AG and the
  imitation order, the latter with the model trained at full coverage and with one model per connectome trained
  across coverage levels.

Sets of questions (``prepare``), each an array file like those of the benchmark:
  fit          development types of the benchmark, their benchmark targets (first two neurons): training questions
  development  development types, the third and fourth neuron (the development targets of the replication)
  heldout      held-out types of the benchmark, the third and fourth neuron (the held-out targets of the replication);
               only the arms that no earlier run evaluated on them are run here
  new          the cell types that no analysis used, first two neurons

``prepare`` reproduces the stored arrays of the benchmark and of the replication for every target it shares with them.
``fit`` trains the imitation models that did not exist. ``run`` evaluates the development set, where every arm shared
with the replication and with the imitation order must reproduce the stored costs question by question; after
``record`` has pinned code, inputs, models and expected outcomes it evaluates the held-out and the new set once per
connectome. ``analyze`` reports ratios of total checks with 95% intervals from resampling cell types and, for the
recorded expectations, simultaneous intervals (Bonferroni over the number of expectations).

Usage:
  newtypes.py prepare
  newtypes.py fit
  newtypes.py run --view flywire --split development
  newtypes.py analyze --split development
  newtypes.py record                                   # needs outputs/newtypes/expectations.json
  newtypes.py run --view flywire --split heldout       # once per connectome
  newtypes.py run --view flywire --split new           # once per connectome
  newtypes.py analyze --split heldout
  newtypes.py analyze --split new
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import replication as rep  # noqa: E402  (sets one numerical thread per worker before numpy loads)
import numpy as np  # noqa: E402
from connectomequest.annotation_v35 import decision, engine as e, sweep  # noqa: E402

OUT = ROOT / "outputs/newtypes"
ARRAYS = OUT / "arrays"
RECORD = OUT / "record.json"
EARLIER = "inputs/earlier/malecns-v0.9/body-annotations-male-cns-v0.9-minconf-0.5.feather"
VIEWS = ("malecns", "flywire")
PRIORS = ("mlp_s0", "provisional")
SETS = dict(fit=("development", 0, 2), development=("development", 2, 4), heldout=("test", 2, 4), new=("new", 0, 2))
COVERAGE = (0.25, 0.5, 0.75)
MIXED = (0.0, 0.25, 0.5, 0.75, 1.0)
ORDERS = ("member", "nonmember", "direction", "ag", "adg", "recal", "prop")
GENERIC = ("ec2", "ig")
AT_COVERAGE = ("direction", "ag", "recal", "imitation-full", "imitation-mixed")
DRAWS, SEED, ALPHA = 20000, 20261117, 0.05
CODE = ("scripts/newtypes.py", "scripts/replication.py", "scripts/imitation.py", "scripts/imitation_models.py", "scripts/v35_coverage.py",
        "scripts/v35_build.py", "src/connectomequest/annotation_v35/engine.py", "src/connectomequest/annotation_v35/decision.py",
        "src/connectomequest/annotation_v35/sweep.py", "src/connectomequest/neurosym/kernel.py", "src/connectomequest/neurosym/common.py",
        "src/connectomequest/neurosym/constraints.py")
DATA = (EARLIER, "outputs/newtypes/arrays/manifest.json", "outputs/newtypes/models.npz", "outputs/newtypes/fit.json",
        "outputs/imitation/models.npz", "outputs/imitation/fit.json", "outputs/replication/record.json", "outputs/imitation/record.json",
        "outputs/v35/sweep/malecns-development.json", "outputs/v35/sweep/flywire-development.json") + tuple(
            f"outputs/newtypes/arrays/{view}-{name}.npz" for view in VIEWS for name in SETS)
_D = {}


# --------------------------------------------------------------------------- question arrays


def prepare(args):
    import pyarrow.feather as feather
    import pyarrow.parquet as pq

    import v35_build as b
    from connectomequest.annotation_v32 import data as names_of
    from connectomequest.neurosym import common as c
    from connectomequest.neurosym import constraints as k

    ARRAYS.mkdir(parents=True, exist_ok=False)
    benchmark = json.loads((rep.OLD / "manifest.json").read_text())
    vocabulary = json.loads((ROOT / b.SOURCE / "public/manifest.json").read_text())["vocabulary"]
    lookup = {name: i for i, name in enumerate(vocabulary) if i}
    by_type, source_total, source_counts = b.source_statistics(ROOT, len(vocabulary))
    alpha = b.PROTOCOL["provisional_alpha"]
    minimum = b.PROTOCOL["minimum_partners"]
    graphs = {}
    for view, cfg in b.VIEWS.items():
        ids, matrix = b.load_graph(ROOT / cfg["root"])
        if not np.array_equal(ids, c.view_data(view)["ids"]):
            raise ValueError("node order differs from the cached graph")
        labels, _ = b.view_labels(ROOT, view, ids, lookup)
        endpoint = (labels >= 0) & (np.diff(matrix.indptr) >= minimum)
        graphs[view] = dict(ids=ids, matrix=matrix, labels=labels, endpoint=endpoint,
                            any=np.bincount(labels[labels >= 0], minlength=len(vocabulary)) > 0,
                            has=np.bincount(labels[endpoint], minlength=len(vocabulary)) > 0)
    candidate = (source_counts > 0) & graphs["malecns"]["any"] & graphs["flywire"]["any"]
    candidate[0] = False

    # earlier labels: FlyWire as prepared for the benchmark, MaleCNS from release v0.9 by the same rule
    table = pq.read_table(ROOT / b.VIEWS["flywire"]["root"] / "public/nodes.parquet", columns=["cell_id", "provisional"]).to_pylist()
    found = {int(r["cell_id"]): r["provisional"] for r in table}
    hints = dict(flywire=np.array([-1 if found[x] is None else found[x] for x in graphs["flywire"]["ids"].tolist()], dtype=np.int32))
    earlier = {int(r["bodyId"]): r["type"] for r in feather.read_table(ROOT / EARLIER, columns=["bodyId", "type"]).to_pylist()}
    code_of = {name: i for i, name in enumerate(vocabulary)}
    release = b.VIEWS["malecns"]["release"]  # the name aliases of the reference release
    hints["malecns"] = np.array([-1 if names_of.canonical_type(earlier.get(x)) is None else code_of.get(names_of.ontology_type(earlier.get(x), release), 0)
                                 for x in graphs["malecns"]["ids"].tolist()], dtype=np.int32)

    # cell types
    used = {t["code"] for t in benchmark["types"]}
    types = dict(development=[t for t in benchmark["types"] if t["split"] == "development"],
                 test=[t for t in benchmark["types"] if t["split"] == "test"])
    new = {}
    for view, other in (("malecns", "flywire"), ("flywire", "malecns")):
        rows = []
        for code in np.flatnonzero((source_counts > 0) & graphs[view]["has"] & ~graphs[other]["has"]).tolist():
            name = vocabulary[code]
            if code == 0 or source_total[code] < b.PROTOCOL["minimum_source_weight"]:
                continue
            if any(name.startswith(p) for p in b.PROTOCOL["forced_development_prefixes"]):
                continue  # the types of the preliminary studies
            if code in used:
                raise AssertionError("a type of the benchmark has targets in one connectome only")
            families, typed_total = b.families_for(code, by_type, candidate)
            if families:
                rows.append(dict(code=int(code), name=name, source_cells=int(source_counts[code]), source_total_weight=int(source_total[code]),
                                 source_typed_weight=int(typed_total), families=families))
        rows.sort(key=lambda t: b.digest("cq-v35-roster", t["name"]))
        new[view] = rows

    def collect(view, full, first, last, rows):
        g, hint = graphs[view], hints[view]
        ids, matrix, labels = g["ids"], g["matrix"], g["labels"]
        target_cell, target_type, target_offset = [], [], [0]
        pre_cell, pre_weight, pre_label, pre_hint = [], [], [], []
        fam_target, fam_kind, fam_source, fam_total, fam_code_offset, fam_codes = [], [], [], [], [0], []
        plus = {name: [] for name in PRIORS}
        unknown = {name: [] for name in PRIORS}
        for t in rows:
            code = t["code"]
            cells = sorted(np.flatnonzero(g["endpoint"] & (labels == code)).tolist(), key=lambda i: b.digest("cq-v35-endpoint", view, vocabulary[code], int(ids[i])))
            for i in cells[first:last]:
                start, stop = matrix.indptr[i], matrix.indptr[i + 1]
                pre, weight = matrix.indices[start:stop], np.asarray(matrix.data[start:stop], dtype=np.int64)
                order = np.argsort(ids[pre], kind="stable")
                pre, weight = pre[order], weight[order]
                q = np.asarray(full[pre])
                mixed = q.copy()
                hinted = np.flatnonzero(hint[pre] >= 0)
                mixed[hinted] *= np.float32(1 - alpha)
                mixed[hinted, hint[pre][hinted]] += np.float32(alpha)
                target = len(target_cell)
                target_cell.append(int(ids[i]))
                target_type.append(code)
                target_offset.append(target_offset[-1] + len(pre))
                pre_cell.append(ids[pre])
                pre_weight.append(weight)
                pre_label.append(labels[pre])
                pre_hint.append(hint[pre])
                for family in t["families"]:
                    fam_target.append(target)
                    fam_kind.append(0 if family["kind"] == "top1" else 1)
                    fam_source.append(family["source_weight"])
                    fam_total.append(t["source_total_weight"])
                    fam_codes.extend(family["codes"])
                    fam_code_offset.append(len(fam_codes))
                    plus["mlp_s0"].append(q[:, family["codes"]].astype(np.float64).sum(1).astype(np.float32))
                    plus["provisional"].append(mixed[:, family["codes"]].astype(np.float64).sum(1).astype(np.float32))
                unknown["mlp_s0"].append(q[:, 0].copy())
                unknown["provisional"].append(mixed[:, 0].copy())
        arrays = dict(
            target_cell=np.asarray(target_cell, dtype=np.int64), target_type=np.asarray(target_type, dtype=np.int32),
            target_offset=np.asarray(target_offset, dtype=np.int64), pre_cell=np.concatenate(pre_cell),
            pre_weight=np.concatenate(pre_weight), pre_label=np.concatenate(pre_label).astype(np.int32),
            pre_hint=np.concatenate(pre_hint).astype(np.int32),
            fam_target=np.asarray(fam_target, dtype=np.int32), fam_kind=np.asarray(fam_kind, dtype=np.int8),
            fam_source_weight=np.asarray(fam_source, dtype=np.int64), fam_source_total=np.asarray(fam_total, dtype=np.int64),
            fam_code_offset=np.asarray(fam_code_offset, dtype=np.int64), fam_codes=np.asarray(fam_codes, dtype=np.int32))
        for name in PRIORS:
            arrays["plus_" + name] = np.concatenate(plus[name])
            arrays["unknown_" + name] = np.concatenate(unknown[name])
        return arrays

    report = dict(purpose=__doc__.split("\n\n")[0], earlier_release=dict(malecns="male-cns:v0.9", flywire="flywire:v2.1.0"),
                  earlier_sha256=rep.sha256(ROOT / EARLIER), benchmark_manifest_sha256=rep.sha256(rep.OLD / "manifest.json"), views=[], new_types=new)
    for view in VIEWS:
        entry = dict(view=view, sets={})
        full = np.load(k.full_path(view))  # the stored distribution of the topology model over all types, in memory
        for name, (group, first, last) in SETS.items():
            arrays = collect(view, full, first, last, new[view] if group == "new" else types[group])
            stored = None if group == "new" else (rep.OLD if name == "fit" else rep.BENCH) / f"{view}.npz"
            arrays["fam_origin"] = np.arange(len(arrays["fam_target"]), dtype=np.int64) if stored is None else same_as(arrays, stored, view)
            with (ARRAYS / f"{view}-{name}.npz").open("xb") as stream:
                np.savez(stream, **arrays)
            entry["sets"][name] = dict(types=int(len(set(arrays["target_type"].tolist()))), targets=int(len(arrays["target_cell"])),
                                       families=int(len(arrays["fam_target"])), input_cells=int(arrays["target_offset"][-1]),
                                       reproduces=None if stored is None else str(stored.relative_to(ROOT)), sha256=rep.sha256(ARRAYS / f"{view}-{name}.npz"))
            print(view, name, json.dumps(entry["sets"][name]), flush=True)
        report["views"].append(entry)
    (ARRAYS / "manifest.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")


def same_as(mine, path, view):
    """Every target must equal the stored target of the same neuron; returns the stored index of every partner set."""
    with np.load(path, allow_pickle=False) as z:
        priors = [p for p in PRIORS if "plus_" + p in z.files]
        if view == "flywire" and priors != list(PRIORS):
            raise AssertionError("stored FlyWire arrays lack a prior")
        keys = ["target_cell", "target_offset", "pre_cell", "pre_weight", "pre_label", "fam_target", "fam_kind", "fam_source_weight", "fam_source_total",
                "fam_code_offset", "fam_codes"] + [x + p for p in priors for x in ("plus_", "unknown_")]
        stored = {key: z[key] for key in keys}
    position = {int(cell): i for i, cell in enumerate(stored["target_cell"].tolist())}
    first = {"mine": np.searchsorted(mine["fam_target"], np.arange(len(mine["target_cell"]) + 1)),
             "stored": np.searchsorted(stored["fam_target"], np.arange(len(stored["target_cell"]) + 1))}
    offset = {"mine": np.concatenate(([0], np.cumsum(np.diff(mine["target_offset"])[mine["fam_target"]]))),
              "stored": np.concatenate(([0], np.cumsum(np.diff(stored["target_offset"])[stored["fam_target"]])))}
    origin = np.empty(len(mine["fam_target"]), dtype=np.int64)
    for t, cell in enumerate(mine["target_cell"].tolist()):
        s = position[cell]
        a, b = slice(*mine["target_offset"][t : t + 2]), slice(*stored["target_offset"][s : s + 2])
        for key in ["pre_cell", "pre_weight", "pre_label"] + ["unknown_" + p for p in priors]:
            if not np.array_equal(mine[key][a], stored[key][b]):
                raise AssertionError(f"{view}: {key} of a stored target is not reproduced")
        i0, i1, j0, j1 = first["mine"][t], first["mine"][t + 1], first["stored"][s], first["stored"][s + 1]
        if i1 - i0 != j1 - j0:
            raise AssertionError(f"{view}: partner sets of a stored target are not reproduced")
        for i, j in zip(range(i0, i1), range(j0, j1)):
            origin[i] = j
            same = all(mine[key][i] == stored[key][j] for key in ("fam_kind", "fam_source_weight", "fam_source_total"))
            same &= np.array_equal(mine["fam_codes"][mine["fam_code_offset"][i] : mine["fam_code_offset"][i + 1]],
                                   stored["fam_codes"][stored["fam_code_offset"][j] : stored["fam_code_offset"][j + 1]])
            for p in priors:
                same &= np.array_equal(mine["plus_" + p][offset["mine"][i] : offset["mine"][i + 1]], stored["plus_" + p][offset["stored"][j] : offset["stored"][j + 1]])
            if not same:
                raise AssertionError(f"{view}: a partner set of a stored target is not reproduced")
    return origin


def load(view, name):
    with np.load(ARRAYS / f"{view}-{name}.npz", allow_pickle=False) as z:
        d = {key: z[key] for key in z.files}
    d["fam_plus_offset"] = np.concatenate(([0], np.cumsum(np.diff(d["target_offset"])[d["fam_target"]])))
    return d


def family_cells(d, family):
    target = int(d["fam_target"][family])
    lo, hi = int(d["target_offset"][target]), int(d["target_offset"][target + 1])
    cells, weights, labels = d["pre_cell"][lo:hi], d["pre_weight"][lo:hi], d["pre_label"][lo:hi]
    codes = d["fam_codes"][d["fam_code_offset"][family] : d["fam_code_offset"][family + 1]]
    outcomes = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
    start = int(d["fam_plus_offset"][family])
    priors = {p: rep.probabilities(d["plus_" + p][start : start + hi - lo], d["unknown_" + p][lo:hi]) for p in PRIORS}
    return target, cells, weights, outcomes, priors


def thresholds(d, family, total):
    return [rep.threshold_of(kind, num, den, total, int(d["fam_source_weight"][family]), int(d["fam_source_total"][family])) for kind, num, den in rep.LEVELS]


def at_coverage(priors, cells, level):
    """The earlier-release prior for the covered neurons and the topology prior for the others."""
    from v35_coverage import covered

    return np.where(covered(cells, level)[:, None], priors["provisional"], priors["mlp_s0"])


# --------------------------------------------------------------------------- imitation models


def model_bytes(key):
    """A fitted model by ``view-variant``: the three of ``scripts/imitation.py`` or one of this script."""
    import imitation_models

    view, variant = key.split("-", 1)
    if variant == "mlp_s0" or key == "flywire-provisional":
        return imitation_models.stored(f"{view}:{variant}")
    with np.load(OUT / "models.npz", allow_pickle=False) as z:
        data = z[key].tobytes()
    if hashlib.sha256(data).hexdigest() != json.loads((OUT / "fit.json").read_text())["models"][key]["sha256"]:
        raise SystemExit(f"the stored model {key} differs from fit.json")
    return data


def fit(args):
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    import imitation

    if (OUT / "fit.json").exists():
        raise SystemExit("the models exist")
    report = dict(features=list(imitation.FEATURES), model=imitation.MODEL, sklearn=sklearn.__version__, coverage_levels=list(MIXED), models={})
    packed = {}
    for view in VIEWS:
        d = load(view, "fit")
        blocks = {"provisional": [], "mixed": []}
        labels, questions = [], 0
        for family in range(len(d["fam_target"])):
            _, cells, weights, outcomes, priors = family_cells(d, family)
            variants = [at_coverage(priors, cells, level) for level in MIXED]
            for threshold in thresholds(d, family, int(weights.sum())):
                inst = e.threshold_instance(cells, weights, threshold)
                mask = imitation.certificate(inst, weights, outcomes, threshold)
                if mask is None:
                    continue
                labels.append(mask)
                questions += 1
                blocks["provisional"].append(imitation.features(weights, priors["provisional"], threshold))
                blocks["mixed"].append([imitation.features(weights, p, threshold) for p in variants])
        y = np.concatenate(labels)
        tasks = {"mixed": (np.concatenate([np.concatenate([b[i] for b in blocks["mixed"]]) for i in range(len(MIXED))]), np.tile(y, len(MIXED)))}
        if view == "malecns":  # the FlyWire model with the earlier-release prior is the one of scripts/imitation.py
            tasks["provisional"] = (np.concatenate(blocks["provisional"]), y)
        for variant, (x, target) in tasks.items():
            started = time.time()
            model = HistGradientBoostingClassifier(**imitation.MODEL).fit(x, target)
            data = pickle.dumps(model)
            key = f"{view}-{variant}"
            packed[key] = np.frombuffer(data, dtype=np.uint8)
            report["models"][key] = dict(questions=questions, rows=int(len(target)), positives=float(target.mean()),
                                         training_auroc=float(roc_auc_score(target, model.predict_proba(x)[:, 1])),
                                         seconds=round(time.time() - started, 1), sha256=hashlib.sha256(data).hexdigest())
            print(key, report["models"][key], flush=True)
    with (OUT / "models.npz").open("xb") as stream:
        np.savez(stream, **packed)
    (OUT / "fit.json").write_text(json.dumps(report, indent=1) + "\n")


# --------------------------------------------------------------------------- runs


def arm_names(view, split):
    def standard(prior):
        return [f"{prior}:{policy}" for policy in ORDERS + GENERIC + ("imitation",)]

    coverage = [f"cov{level}:{policy}" for level in COVERAGE for policy in AT_COVERAGE] + ["cov0.0:imitation-mixed", "cov1.0:imitation-mixed"]
    if split == "heldout":  # the other arms were run on these targets by the replication and the imitation order
        return ["weight"] + (standard("provisional") if view == "malecns" else []) + coverage
    return ["weight"] + standard("mlp_s0") + standard("provisional") + ["none:prop"] + [f"synthetic{a}:ag" for a in rep.AUROCS] + coverage


def imitation_arms(view, names):
    """arm -> (model, prior variant) for the arms that follow a fitted order."""
    out = {}
    for name in names:
        prior, _, policy = name.partition(":")
        if policy == "imitation":
            out[name] = (f"{view}-{prior}", prior)
        elif policy == "imitation-full":
            out[name] = (f"{view}-provisional", prior)
        elif policy == "imitation-mixed":
            out[name] = (f"{view}-mixed", {"cov0.0": "mlp_s0", "cov1.0": "provisional"}.get(prior, prior))
    return out


def prior_variant(priors, cells, variant):
    return priors[variant] if variant in priors else at_coverage(priors, cells, float(variant[3:]))


def run_family(family):
    from connectomequest.neurosym import constraints as k
    from connectomequest.neurosym import kernel as kk

    d, view, arms, rates = _D["data"], _D["view"], _D["arms"], _D["rates"]
    target, cells, weights, outcomes, priors = family_cells(d, family)
    origin, n, total = int(d["fam_origin"][family]), len(weights), int(weights.sum())
    half = np.tile((0.5, 0.5, 0.0), (n, 1))
    synthetic = {level: sweep.synthetic_prior(outcomes, np.random.default_rng([41, origin, i]).standard_normal(n), level,
                                              rates["member"][str(int(d["fam_kind"][family]))], rates["unresolved"])
                 for i, level in enumerate(rep.AUROCS) if f"synthetic{level}:ag" in arms}
    covered = {level: at_coverage(priors, cells, level) for level in COVERAGE}
    kern = None
    if any(name.endswith(":prop") for name in arms):
        if _D.get("kernel_target") != target:
            _D["kernel_target"] = target
            _D["kernel"] = kk.kernel(kk.cosine(view, k.rows_of(view, cells)), rep.SETTING["gamma"], rep.SETTING["tau"])
        kern = _D["kernel"]
    rows = []
    for level, threshold in enumerate(thresholds(d, family, total)):
        inst = e.threshold_instance(cells, weights, threshold)
        truth = e.full_information_status(inst, outcomes)
        bound = e.clairvoyant_cost(inst, outcomes)
        cost = np.zeros(len(arms), dtype=np.int32)

        def record(name, order):
            c, s = e.evaluate(inst, outcomes, order)
            if s != truth or (bound is not None and c < bound):
                raise AssertionError(f"{name} contradicts the exact full-information status")
            cost[arms[name]] = c

        def adaptive(policy):
            return e.run_adaptive(inst, outcomes, policy)

        def fixed_and_greedy(prefix, p, policies):
            for policy in policies:
                name = f"{prefix}:{policy}"
                if name not in arms:
                    continue
                if policy == "member":
                    record(name, e.positive_order(inst, p))
                elif policy == "nonmember":
                    record(name, e.negative_order(inst, p))
                elif policy == "direction":
                    record(name, e.direction_order(inst, p))
                elif policy == "ag":
                    record(name, adaptive(e.AdaptiveGreedy(inst, p)))
                elif policy == "adg":
                    record(name, adaptive(e.AdaptiveDualGreedy(inst, p)))
                elif policy == "recal":
                    record(name, adaptive(e.RecalibratedGreedy(inst, p)))
                elif policy == "prop":
                    record(name, adaptive(kk.policy("recal", inst, p, kern, rep.SETTING["lam"])))
                elif policy in GENERIC:
                    rng = np.random.default_rng([43, origin, level, rep.GENERIC.index(policy)])
                    record(name, adaptive(decision.policy(policy, inst, p, rng)))

        if truth == "unresolved":  # every policy checks all cells of an uncertifiable question
            cost[:] = inst.size
        else:
            record("weight", e.weight_order(inst))
            for prior in PRIORS:
                fixed_and_greedy(prior, priors[prior], ORDERS + GENERIC)
            if "none:prop" in arms:
                record("none:prop", adaptive(kk.policy("recal", inst, half, kern, rep.SETTING["lam"])))
            for value, p in synthetic.items():
                record(f"synthetic{value}:ag", adaptive(e.AdaptiveGreedy(inst, p)))
            for value, p in covered.items():
                fixed_and_greedy(f"cov{value}", p, ("direction", "ag", "recal"))
        rows.append(dict(family=family, origin=origin, level=level, size=inst.size, total=total, threshold=threshold,
                         member=int(weights[outcomes == e.MEMBER].sum()), unresolved=int(weights[outcomes == e.UNRESOLVED].sum()),
                         truth=rep.STATUS[truth], clairvoyant=-1 if bound is None else bound, cost=cost))
    return rows


def imitation_costs(view, d, arrays, names):
    """Fill the columns of the fitted orders: one prediction per model and prior variant over all certifiable questions."""
    import imitation

    wanted = imitation_arms(view, names)
    variants = sorted({variant for _, variant in wanted.values()})
    arm = {n: i for i, n in enumerate(names)}
    questions, blocks = [], {variant: [] for variant in variants}
    for row in np.flatnonzero(arrays["truth"] > 0).tolist():
        family, level = int(arrays["family"][row]), int(arrays["level"][row])
        _, cells, weights, outcomes, priors = family_cells(d, family)
        threshold = int(arrays["threshold"][row])
        questions.append((row, e.threshold_instance(cells, weights, threshold), outcomes))
        for variant in variants:
            blocks[variant].append(imitation.features(weights, prior_variant(priors, cells, variant), threshold))
    offsets = np.concatenate(([0], np.cumsum([q[1].size for q in questions])))
    stacked = {variant: np.concatenate(blocks[variant]) for variant in variants}
    for name, (model_key, variant) in wanted.items():
        # the models are written by ``fit`` of this script or of scripts/imitation.py; ``model_bytes`` compares their
        # hashes with the fit reports, which the record pins, so pickle only loads files these pipelines produced
        score = pickle.loads(model_bytes(model_key)).predict_proba(stacked[variant])[:, 1]
        for i, (row, inst, outcomes) in enumerate(questions):
            order = np.lexsort((inst.ranks, -score[offsets[i] : offsets[i + 1]]))
            cost, status = e.evaluate(inst, outcomes, order)
            if rep.STATUS[status] != arrays["truth"][row] or cost < arrays["clairvoyant"][row]:
                raise AssertionError(f"{name} contradicts the exact full-information status")
            arrays["cost"][row, arm[name]] = cost


def stored_costs(view, split):
    """Costs that earlier runs stored for the targets of a set: (family, level) -> {arm: cost}."""
    out = {}
    for folder, label in ((rep.OUT, "replication"), (ROOT / "outputs/imitation", "imitation")):
        meta = json.loads((folder / f"{view}-{split}.json").read_text())
        with np.load(folder / f"{view}-{split}.npz", allow_pickle=False) as z:
            family, level, cost, truth, clairvoyant = z["family"], z["level"], z["cost"], z["truth"], z["clairvoyant"]
        for i in range(len(family)):
            entry = out.setdefault((int(family[i]), int(level[i])), dict(truth=int(truth[i]), clairvoyant=int(clairvoyant[i]), cost={}))
            if entry["truth"] != int(truth[i]) or entry["clairvoyant"] != int(clairvoyant[i]):
                raise AssertionError("stored runs disagree")
            for j, name in enumerate(meta["arms"]):
                if label == "replication" or name != "weight":
                    entry["cost"][name] = int(cost[i, j])
    return out


def check_record():
    if not RECORD.exists():
        raise SystemExit("this set needs the record")
    record = json.loads(RECORD.read_text())
    for group in ("code", "data"):
        for path, digest in record[group].items():
            if rep.sha256(ROOT / path) != digest:
                raise SystemExit(f"{path} differs from the record")
    return record


def run(args):
    from connectomequest.neurosym import kernel as kk

    view, split = args.view, args.split
    manifest = json.loads((ARRAYS / "manifest.json").read_text())
    entry = next(v for v in manifest["views"] if v["view"] == view)["sets"][split]
    if rep.sha256(ARRAYS / f"{view}-{split}.npz") != entry["sha256"]:
        raise SystemExit("question arrays differ from their manifest")
    out = OUT / f"{view}-{split}.npz"
    if split != "development":
        check_record()
        if out.exists():
            raise SystemExit("the result exists")
        with (OUT / f"claim-{view}-{split}").open("x") as stream:
            stream.write(datetime.now(timezone.utc).isoformat() + "\n")
    d = load(view, split)
    names = arm_names(view, split)
    rates = json.loads((ROOT / f"outputs/v35/sweep/{view}-development.json").read_text())["rates"]
    _D.update(data=d, view=view, arms={n: i for i, n in enumerate(names)}, rates=rates)
    families = list(range(len(d["fam_target"])))
    if args.limit is not None:
        if split != "development":
            raise SystemExit("this set runs completely")
        families = families[: args.limit]
    kk.profiles(view)  # build before forking
    started = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        results = pool.map(run_family, families, chunksize=2)
    rows = [row for group in results for row in group]
    fields = ("family", "origin", "level", "size", "total", "threshold", "member", "unresolved", "truth", "clairvoyant")
    arrays = {key: np.asarray([r[key] for r in rows], dtype=np.int64) for key in fields}
    arrays["cost"] = np.stack([r["cost"] for r in rows])
    imitation_costs(view, d, arrays, names)
    reproduced = None
    if split != "new":  # the targets of these sets were run before: shared arms must reproduce the stored costs
        stored = stored_costs(view, "development" if split == "development" else "test")
        reproduced = set()
        for i in range(len(rows)):
            entry_ = stored[int(arrays["origin"][i]), int(arrays["level"][i])]
            if entry_["truth"] != arrays["truth"][i] or entry_["clairvoyant"] != arrays["clairvoyant"][i]:
                raise SystemExit("answers differ from the stored runs")
            for j, name in enumerate(names):
                if name in entry_["cost"]:
                    if entry_["cost"][name] != arrays["cost"][i, j]:
                        raise SystemExit(f"the cost of {name} differs from the stored runs")
                    reproduced.add(name)
        reproduced = sorted(reproduced)
    path = out if args.limit is None else OUT / f"{view}-{split}-limit{args.limit}.npz"
    with path.open("wb" if split == "development" else "xb") as stream:
        np.savez(stream, **arrays)
    meta = dict(view=view, split=split, arms=names, questions=len(rows), families=len(families), limit=args.limit, reproduced_arms=reproduced,
                code={p: rep.sha256(ROOT / p) for p in CODE}, arrays_sha256=entry["sha256"], wall_seconds=time.time() - started)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in meta.items() if key not in ("arms", "code")}))


# --------------------------------------------------------------------------- analysis


class Bootstrap:
    """Ratios of total checks over certifiable questions; cell types are resampled, with the same draws for all ratios."""

    def __init__(self, types, truth, rng, tail):
        self.truth, self.tail = truth, tail
        self.cluster = np.unique(types, return_inverse=True)[1]
        self.n = int(self.cluster.max()) + 1
        self.counts = rng.multinomial(self.n, np.full(self.n, 1.0 / self.n), size=DRAWS).astype(np.float64)

    def measure(self, cost, base, mask):
        """Ratio of totals over ``mask`` for all certifiable questions and by answer, with 95% and simultaneous intervals."""
        out = {}
        for name, answer in (("all", self.truth > 0), ("yes", self.truth == 1), ("no", self.truth == 2)):
            m = mask & answer
            sx = np.bincount(self.cluster[m], weights=cost[m].astype(float), minlength=self.n)
            sy = np.bincount(self.cluster[m], weights=base[m].astype(float), minlength=self.n)
            if sy.sum() <= 0:
                out[name] = None
                continue
            denominator = self.counts @ sy
            boot = (self.counts @ sx)[denominator > 0] / denominator[denominator > 0]
            low, high, low_s, high_s = np.quantile(boot, [0.025, 0.975, self.tail, 1 - self.tail])
            out[name] = dict(value=float(sx.sum() / sy.sum()), low=float(low), high=float(high), low_simultaneous=float(low_s), high_simultaneous=float(high_s))
        return out


def contrasts(names):
    pairs = [("none:prop", "mlp_s0:prop"), ("cov1.0:imitation-mixed", "provisional:recal"), ("cov1.0:imitation-mixed", "provisional:imitation"),
             ("cov0.0:imitation-mixed", "mlp_s0:recal"), ("cov0.0:imitation-mixed", "mlp_s0:imitation")]
    for prior in PRIORS:
        pairs += [(f"{prior}:{a}", f"{prior}:{b}") for a, b in (("prop", "recal"), ("ag", "direction"), ("recal", "direction"), ("recal", "ag"), ("ec2", "ag"),
                                                                 ("ig", "ag"), ("ec2", "recal"), ("ig", "recal"), ("imitation", "recal"), ("imitation", "direction"),
                                                                 ("member", "direction"), ("adg", "ag"))]
    for level in COVERAGE:
        pairs += [(f"cov{level}:{a}", f"cov{level}:{b}") for a, b in (("recal", "direction"), ("ag", "direction"), ("recal", "ag"), ("imitation-mixed", "recal"),
                                                                       ("imitation-full", "recal"), ("imitation-mixed", "direction"), ("imitation-full", "direction"),
                                                                       ("imitation-mixed", "imitation-full"))]
    return [(a, b) for a, b in pairs if a in names and b in names]


def analyse_view(view, split, tail):
    d = load(view, split)
    meta = json.loads((OUT / f"{view}-{split}.json").read_text())
    with np.load(OUT / f"{view}-{split}.npz", allow_pickle=False) as z:
        r = {key: z[key] for key in z.files}
    cost = {name: r["cost"][:, i] for i, name in enumerate(meta["arms"])}
    if split == "heldout":  # the arms that the replication and the imitation order ran on these targets
        stored = stored_costs(view, "test")
        for name in next(iter(stored.values()))["cost"]:
            if name not in cost:
                cost[name] = np.array([stored[int(o), int(lv)]["cost"][name] for o, lv in zip(r["origin"], r["level"])])
    targets = d["fam_target"][r["family"]]
    types = d["target_type"][targets]
    kinds = np.array([rep.LEVELS[i][0] for i in r["level"]])
    boot = Bootstrap(types, r["truth"], np.random.default_rng([SEED, VIEWS.index(view), list(SETS).index(split)]), tail)
    weight = cost["weight"]
    clairvoyant = np.where(r["clairvoyant"] >= 0, r["clairvoyant"], 0)
    cells = np.concatenate([np.arange(d["target_offset"][t], d["target_offset"][t + 1]) for t in sorted(set(targets.tolist()))])
    hint, label = d["pre_hint"][cells], d["pre_label"][cells]
    both = (hint > 0) & (label > 0)
    result = dict(types=int(len(set(types.tolist()))), targets=int(len(set(targets.tolist()))), questions=int(len(types)),
                  earlier=dict(hemibrain_type=float((hint > 0).mean()), other_name=float((hint == 0).mean()), none=float((hint < 0).mean()),
                               agreement=float((hint[both] == label[both]).mean()), earlier_type_given_resolved=float((hint[label > 0] > 0).mean()),
                               unresolved=float((label < 0).mean())))
    for part in ("retention", "share"):
        mask = kinds == part
        certifiable = mask & (r["truth"] > 0)
        out = dict(questions=int(mask.sum()), answers=dict(yes=int((mask & (r["truth"] == 1)).sum()), no=int((mask & (r["truth"] == 2)).sum()),
                                                            uncertifiable=int((mask & (r["truth"] == 0)).sum())),
                   share_uncertifiable=float((r["truth"][mask] == 0).mean()), share_no_of_certifiable=float((r["truth"][certifiable] == 2).mean()),
                   weight_median=float(np.median(weight[certifiable])),
                   ratio={name: boot.measure(x, weight, mask) for name, x in cost.items() if name != "weight"},
                   contrast={f"{a}/{b}": boot.measure(cost[a], cost[b], mask) for a, b in contrasts(cost)})
        out["ratio"]["clairvoyant"] = boot.measure(clairvoyant, weight, mask)
        result[part] = out
    return result


def analyze(args):
    source = RECORD if RECORD.exists() else OUT / "expectations.json"
    expectations = json.loads(source.read_text())["expectations"] if source.exists() else []
    count = max(len(expectations), 1)
    tail = ALPHA / (2 * count)
    result = dict(simultaneous=dict(expectations=count, tail=tail, draws=DRAWS), **{view: analyse_view(view, args.split, tail) for view in VIEWS})
    (OUT / f"analysis-{args.split}.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    if args.split != "development":
        record = check_record()
        mine = [x for x in record["expectations"] if x["split"] == args.split]
        verdicts = rep.evaluate(mine, result)
        for v in verdicts:
            base = v["quantity"].rsplit("|", 1)[0]
            low, high = rep.lookup(result, base + "|low_simultaneous"), rep.lookup(result, base + "|high_simultaneous")
            v["low_simultaneous"], v["high_simultaneous"] = low, high
            v["holds_simultaneous"] = bool((v.get("min") is None or low >= v["min"]) and (v.get("max") is None or high <= v["max"]))
        (OUT / f"verdicts-{args.split}.json").write_text(json.dumps(verdicts, indent=1) + "\n")
        print(f"{sum(v['holds'] for v in verdicts)} of {len(verdicts)} recorded expectations hold; "
              f"{sum(v['holds_simultaneous'] for v in verdicts)} with their whole simultaneous interval")
        for v in verdicts:
            if not v["holds"]:
                print("  fails:", v["text"], "->", round(v["value"], 4))
    for view in VIEWS:
        for part in ("retention", "share"):
            x = result[view][part]
            show = {k: [round(v["all"][q], 3) for q in ("value", "low", "high")] for k, v in x["contrast"].items() if v["all"]}
            print(view, part, "questions", x["questions"], "uncertifiable", round(x["share_uncertifiable"], 3), "no", round(x["share_no_of_certifiable"], 3))
            for k, v in show.items():
                print("   ", k, v)


def write_record(args):
    if RECORD.exists():
        raise SystemExit("the record exists")
    expectations = json.loads((OUT / "expectations.json").read_text())
    development = [f"{view}-development.{ext}" for view in VIEWS for ext in ("npz", "json")] + ["analysis-development.json"]
    record = dict(written_utc=datetime.now(timezone.utc).isoformat(), purpose=__doc__.split("\n\n")[0],
                  code={p: rep.sha256(ROOT / p) for p in CODE}, data={p: rep.sha256(ROOT / p) for p in DATA},
                  development={p: rep.sha256(OUT / p) for p in development},
                  simultaneous=dict(alpha=ALPHA, draws=DRAWS, expectations=len(expectations["expectations"])),
                  expectations=expectations["expectations"], notes=expectations.get("notes", []))
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    RECORD.chmod(0o444)
    print(f"recorded {len(record['expectations'])} expectations")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("fit")
    r = sub.add_parser("run")
    r.add_argument("--view", required=True, choices=VIEWS)
    r.add_argument("--split", required=True, choices=("development", "heldout", "new"))
    r.add_argument("--workers", type=int, default=28)
    r.add_argument("--limit", type=int)
    a = sub.add_parser("analyze")
    a.add_argument("--split", required=True, choices=("development", "heldout", "new"))
    sub.add_parser("record")
    args = parser.parse_args()
    dict(prepare=prepare, fit=fit, run=run, analyze=analyze, record=write_record)[args.command](args)


if __name__ == "__main__":
    main()
