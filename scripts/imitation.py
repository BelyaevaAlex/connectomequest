"""A ranker that imitates the smallest certificate: a learned order outside the greedy family, one held-out evaluation.

For every certifiable development question of the benchmark, the cells of the smallest certificate (the clairvoyant
bound: the heaviest members until the threshold, or the heaviest nonmembers until its complement) are positives and
the other input cells negatives. A gradient-boosted classifier predicts membership in the smallest certificate from
twelve features that are known before any check: the weight of the cell, its rank and the weight of the cells at
least as heavy, the three probabilities of the prior, the threshold, the number of input cells, the weight capped
at each goal, and the expected member and nonmember weight of the question relative to the goals. The *imitation
order* checks cells by decreasing predicted probability. It uses neither the gain of Adaptive Greedy nor
recalibration and does not adapt to outcomes.

``fit`` trains one model per connectome and prior on retention and share questions of the development types of the
benchmark. ``run`` evaluates the order on the question arrays of the replication (new target neurons), on the
development types first and, after ``record`` has pinned code, models, inputs and expected outcomes, once per
connectome on the held-out types. ``analyze`` compares it with the costs that the replication stored.

Usage:
  imitation.py fit
  imitation.py run --view flywire --split development
  imitation.py analyze --split development
  imitation.py record                                   # needs outputs/imitation/expectations.json
  imitation.py run --view flywire --split test          # once per connectome
  imitation.py analyze --split test
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import replication as rep  # noqa: E402
from connectomequest.annotation_v35 import engine as e  # noqa: E402

OUT = ROOT / "outputs/imitation"
RECORD = OUT / "record.json"
SEED = 20261116
FEATURES = ("weight share", "weight rank", "weight of cells at least as heavy", "member probability", "nonmember probability",
            "unresolved probability", "threshold share", "log number of cells", "weight capped at the yes goal",
            "weight capped at the no goal", "expected member weight over the yes goal", "expected nonmember weight over the no goal")
MODEL = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, early_stopping=False, random_state=0)
BASELINES = ("recal", "direction", "member", "ag")
CODE = ("scripts/imitation.py", "scripts/replication.py", "src/connectomequest/annotation_v35/engine.py")
DATA = ("outputs/replication/benchmark/manifest.json", "outputs/replication/benchmark/malecns.npz", "outputs/replication/benchmark/flywire.npz",
        "outputs/replication/record.json", "outputs/imitation/model-malecns-mlp_s0.pkl", "outputs/imitation/model-flywire-mlp_s0.pkl",
        "outputs/imitation/model-flywire-provisional.pkl", "outputs/imitation/fit.json")


def load(path):
    with np.load(path, allow_pickle=False) as z:
        d = {key: z[key] for key in z.files if not key.startswith(("plus_", "unknown_")) or key.split("_", 1)[1] in ("mlp_s0", "provisional")}
    lengths = np.diff(d["target_offset"])[d["fam_target"]]
    d["fam_plus_offset"] = np.concatenate(([0], np.cumsum(lengths)))
    return d


def family_cells(d, view, family):
    target = int(d["fam_target"][family])
    lo, hi = int(d["target_offset"][target]), int(d["target_offset"][target + 1])
    cells, weights, labels = d["pre_cell"][lo:hi], d["pre_weight"][lo:hi], d["pre_label"][lo:hi]
    codes = d["fam_codes"][d["fam_code_offset"][family] : d["fam_code_offset"][family + 1]]
    outcomes = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
    start = int(d["fam_plus_offset"][family])
    priors = {p: rep.probabilities(d["plus_" + p][start : start + hi - lo], d["unknown_" + p][lo:hi]) for p in rep.PRIORS[view]}
    return cells, weights, outcomes, priors


def thresholds(d, family, total):
    return [rep.threshold_of(kind, num, den, total, int(d["fam_source_weight"][family]), int(d["fam_source_total"][family]))
            for kind, num, den in rep.LEVELS]


def features(weights, p, threshold):
    """Twelve features per input cell, all known before any check."""
    w = weights.astype(float)
    n, total = len(w), float(w.sum())
    goal_yes, goal_no = float(threshold), total - threshold + 1.0
    order = np.argsort(-w, kind="stable")
    rank, heavier = np.empty(n), np.empty(n)
    rank[order] = np.arange(n)
    heavier[order] = np.cumsum(w[order])
    return np.column_stack((w / total, rank / n, heavier / total, p[:, e.MEMBER], p[:, e.NONMEMBER], p[:, e.UNRESOLVED],
                            np.full(n, goal_yes / total), np.full(n, np.log(n)), np.minimum(w, goal_yes) / goal_yes,
                            np.minimum(w, goal_no) / goal_no, np.full(n, (p[:, e.MEMBER] * w).sum() / goal_yes),
                            np.full(n, (p[:, e.NONMEMBER] * w).sum() / goal_no)))


def certificate(inst, weights, outcomes, threshold):
    """Cells of the smallest certificate of a certifiable question (ties by ascending external ID)."""
    for outcome, goal in ((e.MEMBER, threshold), (e.NONMEMBER, int(weights.sum()) - threshold + 1)):
        idx = np.flatnonzero(outcomes == outcome)
        idx = idx[np.lexsort((inst.ranks[idx], -weights[idx]))]
        reached = int(np.searchsorted(np.cumsum(weights[idx]), goal, side="left"))
        if reached < len(idx):
            mask = np.zeros(len(weights), dtype=bool)
            mask[idx[: reached + 1]] = True
            return mask
    return None


def fit(args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    import sklearn

    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "fit.json").exists():
        raise SystemExit("the models exist")
    report = dict(features=list(FEATURES), model=MODEL, sklearn=sklearn.__version__, levels=[list(x) for x in rep.LEVELS], models={})
    for view in rep.PRIORS:
        d = load(rep.OLD / f"{view}.npz")
        families = rep.split_families(view, d, "development")
        rows = {prior: [] for prior in rep.PRIORS[view]}
        labels, questions = [], 0
        for family in families:
            cells, weights, outcomes, priors = family_cells(d, view, family)
            for threshold in thresholds(d, family, int(weights.sum())):
                inst = e.threshold_instance(cells, weights, threshold)
                mask = certificate(inst, weights, outcomes, threshold)
                if mask is None:
                    continue
                if int(mask.sum()) != e.clairvoyant_cost(inst, outcomes):
                    raise AssertionError("the certificate is not the smallest")
                labels.append(mask)
                questions += 1
                for prior, p in priors.items():
                    rows[prior].append(features(weights, p, threshold))
        y = np.concatenate(labels)
        for prior in rep.PRIORS[view]:
            x = np.concatenate(rows[prior])
            started = time.time()
            model = HistGradientBoostingClassifier(**MODEL).fit(x, y)
            path = OUT / f"model-{view}-{prior}.pkl"
            path.write_bytes(pickle.dumps(model))
            report["models"][f"{view}:{prior}"] = dict(questions=questions, rows=int(len(y)), positives=float(y.mean()),
                                                       training_auroc=float(roc_auc_score(y, model.predict_proba(x)[:, 1])),
                                                       seconds=round(time.time() - started, 1), sha256=rep.sha256(path))
            print(view, prior, report["models"][f"{view}:{prior}"], flush=True)
    (OUT / "fit.json").write_text(json.dumps(report, indent=1) + "\n")


def check_record():
    if not RECORD.exists():
        raise SystemExit("the held-out split needs the record")
    record = json.loads(RECORD.read_text())
    for group in ("code", "data"):
        for path, digest in record[group].items():
            if rep.sha256(ROOT / path) != digest:
                raise SystemExit(f"{path} differs from the record")
    return record


def arm_names(view):
    return ["weight"] + [f"{prior}:imitation" for prior in rep.PRIORS[view]]


def run(args):
    view, split = args.view, args.split
    manifest = json.loads((rep.BENCH / "manifest.json").read_text())
    entry = next(v for v in manifest["views"] if v["view"] == view)
    if rep.sha256(rep.BENCH / f"{view}.npz") != entry["sha256"]:
        raise SystemExit("question arrays differ from their manifest")
    out = OUT / f"{view}-{split}.npz"
    if split == "test":
        check_record()
        if out.exists():
            raise SystemExit("the held-out result exists")
        with (OUT / f"claim-{view}-test").open("x") as stream:
            stream.write(datetime.now(timezone.utc).isoformat() + "\n")
    started = time.time()
    d = load(rep.BENCH / f"{view}.npz")
    # The models are written by ``fit`` of this script; ``record`` pins their hashes and ``check_record`` compares them
    # before the held-out run, so pickle only ever loads files this pipeline produced.
    models = {prior: pickle.loads((OUT / f"model-{view}-{prior}.pkl").read_bytes()) for prior in rep.PRIORS[view]}
    names = arm_names(view)
    arm = {n: i for i, n in enumerate(names)}
    questions, blocks = [], {prior: [] for prior in rep.PRIORS[view]}
    for family in rep.split_families(view, d, split):
        cells, weights, outcomes, priors = family_cells(d, view, family)
        for level, threshold in enumerate(thresholds(d, family, int(weights.sum()))):
            questions.append((family, level, e.threshold_instance(cells, weights, threshold), outcomes))
            for prior, p in priors.items():
                blocks[prior].append(features(weights, p, threshold))
    sizes = np.array([q[2].size for q in questions])
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    scores = {prior: models[prior].predict_proba(np.concatenate(blocks[prior]))[:, 1] for prior in rep.PRIORS[view]}
    rows = dict(family=[], level=[], truth=[], clairvoyant=[], cost=[])
    for i, (family, level, inst, outcomes) in enumerate(questions):
        truth = e.full_information_status(inst, outcomes)
        bound = e.clairvoyant_cost(inst, outcomes)
        cost = np.zeros(len(names), dtype=np.int32)
        orders = dict(weight=e.weight_order(inst))
        for prior in rep.PRIORS[view]:
            orders[f"{prior}:imitation"] = np.lexsort((inst.ranks, -scores[prior][offsets[i] : offsets[i + 1]]))
        for name, order in orders.items():
            c, s = e.evaluate(inst, outcomes, order)
            if s != truth or (truth == "unresolved" and c != inst.size) or (bound is not None and c < bound):
                raise AssertionError(f"{name} contradicts the exact full-information status")
            cost[arm[name]] = c
        for key, value in (("family", family), ("level", level), ("truth", rep.STATUS[truth]), ("clairvoyant", -1 if bound is None else bound), ("cost", cost)):
            rows[key].append(value)
    arrays = {key: np.asarray(rows[key], dtype=np.int64) for key in ("family", "level", "truth", "clairvoyant")}
    arrays["cost"] = np.stack(rows["cost"])
    stored, meta = stored_costs(view, split)
    for key in ("family", "level", "truth", "clairvoyant"):
        if not np.array_equal(stored[key], arrays[key]):
            raise SystemExit(f"{key} differs from the replication")
    if not np.array_equal(stored["cost"][:, meta["arms"].index("weight")], arrays["cost"][:, arm["weight"]]):
        raise SystemExit("the cost of weight order differs from the replication")
    with out.open("wb" if split == "development" else "xb") as stream:
        np.savez(stream, **arrays)
    info = dict(view=view, split=split, arms=names, questions=len(questions), code={p: rep.sha256(ROOT / p) for p in CODE},
                models={prior: rep.sha256(OUT / f"model-{view}-{prior}.pkl") for prior in rep.PRIORS[view]},
                arrays_sha256=entry["sha256"], wall_seconds=time.time() - started)
    out.with_suffix(".json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in info.items() if key not in ("arms", "code", "models")}))


def stored_costs(view, split):
    meta = json.loads((rep.OUT / f"{view}-{split}.json").read_text())
    with np.load(rep.OUT / f"{view}-{split}.npz", allow_pickle=False) as z:
        return {key: z[key] for key in ("family", "level", "truth", "clairvoyant", "cost")}, meta


def analyse_view(view, split):
    d = load(rep.BENCH / f"{view}.npz")
    meta = json.loads((OUT / f"{view}-{split}.json").read_text())
    with np.load(OUT / f"{view}-{split}.npz", allow_pickle=False) as z:
        r = {key: z[key] for key in z.files}
    stored, stored_meta = stored_costs(view, split)
    arm = {n: i for i, n in enumerate(meta["arms"])}
    types = d["target_type"][d["fam_target"][r["family"]]]
    kinds = np.array([rep.LEVELS[i][0] for i in r["level"]])
    ratios = rep.Ratios(types, r["truth"], np.random.default_rng([SEED, sorted(rep.PRIORS).index(view)]))
    weight = r["cost"][:, arm["weight"]]
    result = dict(types=int(len(set(types.tolist()))), questions=int(len(types)))
    for part in ("retention", "share"):
        mask = kinds == part
        out = {}
        for prior in rep.PRIORS[view]:
            imitation = r["cost"][:, arm[f"{prior}:imitation"]]
            out[prior] = dict(imitation=ratios.measure(imitation, weight, mask))
            for name in BASELINES:
                out[prior][f"imitation_vs_{name}"] = ratios.measure(imitation, stored["cost"][:, stored_meta["arms"].index(f"{prior}:{name}")], mask)
        result[part] = out
    return result


def analyze(args):
    result = {view: analyse_view(view, args.split) for view in rep.PRIORS}
    (OUT / f"analysis-{args.split}.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    if args.split == "test":
        record = check_record()
        verdicts = rep.evaluate(record["expectations"], result)
        (OUT / "verdicts-test.json").write_text(json.dumps(verdicts, indent=1) + "\n")
        print(f"{sum(v['holds'] for v in verdicts)} of {len(verdicts)} recorded expectations hold")
        for v in verdicts:
            if not v["holds"]:
                print("  fails:", v["text"], "->", round(v["value"], 4))
    for view, r in result.items():
        for part in ("retention", "share"):
            for prior, m in r[part].items():
                print(view, part, prior, {name: {q: [round(v[x], 3) for x in ("value", "low", "high")] for q, v in x.items() if v and q != "balanced"}
                                          for name, x in m.items() if name in ("imitation", "imitation_vs_recal", "imitation_vs_direction")})


def write_record(args):
    if RECORD.exists():
        raise SystemExit("the record exists")
    expectations = json.loads((OUT / "expectations.json").read_text())
    development = ("malecns-development.npz", "flywire-development.npz", "analysis-development.json")
    record = dict(written_utc=datetime.now(timezone.utc).isoformat(), purpose=__doc__.split("\n\n")[0],
                  code={p: rep.sha256(ROOT / p) for p in CODE}, data={p: rep.sha256(ROOT / p) for p in DATA},
                  development={p: rep.sha256(OUT / p) for p in development},
                  expectations=expectations["expectations"], notes=expectations.get("notes", []))
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    RECORD.chmod(0o444)
    print(f"recorded {len(record['expectations'])} expectations")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fit")
    r = sub.add_parser("run")
    r.add_argument("--view", required=True, choices=tuple(rep.PRIORS))
    r.add_argument("--split", required=True, choices=("development", "test"))
    a = sub.add_parser("analyze")
    a.add_argument("--split", required=True, choices=("development", "test"))
    sub.add_parser("record")
    args = parser.parse_args()
    dict(fit=fit, run=run, analyze=analyze, record=write_record)[args.command](args)


if __name__ == "__main__":
    main()
