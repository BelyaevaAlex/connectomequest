#!/usr/bin/env python3
"""Policies on questions of both answers: costs split by the answer, under the declared and a lenient crosswalk.

Under the declared crosswalk nine in ten certifiable questions have the answer yes, because a certified no needs
confirmed nonmembers carrying all but T-1 input synapses and a third of the input is unresolved. This analysis asks
whether the conclusions about the hedge of Adaptive Greedy and its recalibration hold where no is reached.

Part A splits the pre-specified held-out runs (outputs/v35/runs) by the answer; it reads stored costs only.
Part B re-runs the policies under the lenient crosswalk (cells of release-specific types without a declared hemibrain
correspondence count as nonmembers), where no is reached in a quarter to two fifths of the certifiable questions. Its
priors are the topology prior, the earlier-release prior and, in FlyWire, the earlier-release prior read in the same
lenient way (an earlier name outside the vocabulary counts as a nonmember instead of an unresolved outcome). Part B
also re-runs four pre-specified arms under the declared crosswalk and requires the stored cost of every question.

Measures: cells checked relative to weight order on certifiable questions, over all of them, over yes questions, over
no questions, and answer-balanced (the mean of the yes and the no ratio, i.e. as if both answers took equally many
checks of weight order); 95% intervals from 2,000 resamples of cell types.

Usage:
  answers.py run --view flywire --split development
  answers.py analyze --split development
  answers.py record                       # writes outputs/answers/record.json (expectations from development)
  answers.py run --view flywire --split test    # once; needs the record
  answers.py analyze --split test               # evaluates the recorded expectations
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from connectomequest.annotation_v35 import ablation  # noqa: E402
from connectomequest.annotation_v35 import engine as e  # noqa: E402
from v35_diagnostics import VIEWS, annotation_rows, unresolved_reason  # noqa: E402

BENCH = ROOT / "outputs/v35/benchmark"
RUNS = ROOT / "outputs/v35/runs"
OUT = ROOT / "outputs/answers"
RECORD = OUT / "record.json"
RHOS = ((1, 4), (1, 2), (3, 4), (1, 1))
POLICIES = ("member", "nonmember", "direction", "ag", "adg", "recal")
REPRODUCE = ("weight", "mlp_s0:ag", "mlp_s0:recal", "mlp_s0:direction")  # declared crosswalk, against outputs/v35/runs
BOOTSTRAP, SEED = 2000, 20261113
STATUS = {"unresolved": 0, "positive": 1, "negative": 2}
CODE = ("scripts/answers.py", "src/connectomequest/annotation_v35/engine.py", "src/connectomequest/annotation_v35/ablation.py",
        "scripts/v35_diagnostics.py", "src/connectomequest/annotation_v32/data.py")
DATA = ("outputs/v35/benchmark/manifest.json", "outputs/v35/benchmark/malecns.npz", "outputs/v35/benchmark/flywire.npz",
        "inputs/v32/prepared-common-v1/hemibrain/public/manifest.json", VIEWS["malecns"]["annotations"], VIEWS["flywire"]["annotations"],
        "outputs/v35/runs/malecns-test.npz", "outputs/v35/runs/flywire-test.npz", "outputs/v35/runs/malecns-test.json", "outputs/v35/runs/flywire-test.json")
_D = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def priors_of(view):
    return ["mlp_s0"] + (["provisional", "provisional_lenient"] if view == "flywire" else [])


def arm_names(view):
    return ["weight"] + [f"{p}:{q}" for p in priors_of(view) for q in POLICIES]


def probabilities(plus, unknown):
    plus = np.clip(plus.astype(float), 0.0, 1.0)
    unknown = np.clip(unknown.astype(float), 0.0, 1.0 - plus)
    return np.column_stack((plus, 1.0 - plus - unknown, unknown))


def order_of(name, inst, outcomes, p):
    if name == "member":
        return e.positive_order(inst, p)
    if name == "nonmember":
        return e.negative_order(inst, p)
    if name == "direction":
        return e.direction_order(inst, p)
    rule = dict(ag=e.AdaptiveGreedy, adg=e.AdaptiveDualGreedy, recal=e.RecalibratedGreedy)[name]
    return e.run_adaptive(inst, outcomes, rule(inst, p))


def run_family(family):
    d = _D
    target = int(d["fam_target"][family])
    lo, hi = int(d["target_offset"][target]), int(d["target_offset"][target + 1])
    cells, weights, labels = d["pre_cell"][lo:hi], d["pre_weight"][lo:hi], d["pre_label"][lo:hi]
    codes = d["fam_codes"][d["fam_code_offset"][family] : d["fam_code_offset"][family + 1]]
    strict = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
    lenient = np.where((labels < 0) & d["release_specific"][lo:hi], e.NONMEMBER, strict)
    start = int(d["plus_offset"][family])
    prob = {p: probabilities(d["plus_" + p][start : start + hi - lo], d["unknown_" + p][lo:hi]) for p in ("mlp_s0", "provisional") if "plus_" + p in d}
    if "provisional" in prob:
        labelled, cls = ablation.earlier_release_parts(d["plus_provisional"][start : start + hi - lo], d["unknown_provisional"][lo:hi],
                                                       d["plus_mlp_s0"][start : start + hi - lo], d["unknown_mlp_s0"][lo:hi])
        prob["provisional_lenient"] = ablation.earlier_release_prior(prob["mlp_s0"], labelled, np.where(cls == 1, 2, cls), alpha=ablation.FROZEN_ALPHA)
    arms = d["arms"]
    rows = []
    total = int(weights.sum())
    for rho, (num, den) in enumerate(RHOS):
        threshold, _ = e.threshold_goals(total, num * int(d["fam_source_weight"][family]), den * int(d["fam_source_total"][family]))
        inst = e.threshold_instance(cells, weights, threshold)
        truth = {name: e.full_information_status(inst, y) for name, y in (("strict", strict), ("lenient", lenient))}
        bound = e.clairvoyant_cost(inst, lenient)
        cost = np.zeros(len(arms), dtype=np.int32)

        def record(name, order):
            c, s = e.evaluate(inst, lenient, order)
            if s != truth["lenient"] or (truth["lenient"] == "unresolved" and c != inst.size) or (bound is not None and c < bound):
                raise AssertionError("arm contradicts the exact full-information status")
            cost[arms[name]] = c

        record("weight", e.weight_order(inst))
        for prior, p in prob.items():
            for policy in POLICIES:
                record(f"{prior}:{policy}", order_of(policy, inst, lenient, p))
        strict_cost = [e.evaluate(inst, strict, e.weight_order(inst))[0]]
        strict_cost += [e.evaluate(inst, strict, order_of(name.split(":")[1], inst, strict, prob["mlp_s0"]))[0] for name in REPRODUCE[1:]]
        rows.append(dict(family=family, rho=rho, size=inst.size, threshold=threshold, strict=STATUS[truth["strict"]], truth=STATUS[truth["lenient"]],
                         clairvoyant=-1 if bound is None else bound, cost=cost, strict_cost=np.asarray(strict_cost, dtype=np.int32)))
    return rows


def load(view, split):
    manifest = json.loads((BENCH / "manifest.json").read_text())
    info = next(v for v in manifest["views"] if v["view"] == view)
    if sha256(BENCH / f"{view}.npz") != info["sha256"]:
        raise SystemExit("benchmark file differs from its manifest")
    with np.load(BENCH / f"{view}.npz", allow_pickle=False) as z:
        _D.clear()
        _D.update({k: z[k] for k in z.files if not k.startswith(("plus_mlp_s1", "plus_mlp_s2", "plus_mlp_s3", "plus_mlp_s4", "plus_linear", "plus_ntac",
                                                                   "unknown_mlp_s1", "unknown_mlp_s2", "unknown_mlp_s3", "unknown_mlp_s4", "unknown_linear", "unknown_ntac"))})
    vocabulary = json.loads((ROOT / "inputs/v32/prepared-common-v1/hemibrain/public/manifest.json").read_text())["vocabulary"]
    lookup = {name: i for i, name in enumerate(vocabulary) if i}
    rows = annotation_rows(view)
    reason, flags = {}, np.zeros(len(_D["pre_cell"]), dtype=bool)
    for i, (cell, label) in enumerate(zip(_D["pre_cell"].tolist(), _D["pre_label"].tolist())):
        if label >= 0:
            continue
        if cell not in reason:
            reason[cell] = unresolved_reason(*rows.get(cell, (None, None)), VIEWS[view]["release"], lookup) == "release-specific type without correspondence"
        flags[i] = reason[cell]
    _D["release_specific"] = flags
    lengths = np.diff(_D["target_offset"])[_D["fam_target"]]
    _D["plus_offset"] = np.concatenate(([0], np.cumsum(lengths)))
    _D["arms"] = {n: i for i, n in enumerate(arm_names(view))}
    split_of = {t["code"]: t["split"] for t in manifest["types"]}
    return [f for f in range(len(_D["fam_target"])) if split_of[int(_D["target_type"][_D["fam_target"][f]])] == split]


def check_record():
    record = json.loads(RECORD.read_text())
    changed = [p for p, h in {**record["code"], **record["data"]}.items() if sha256(ROOT / p) != h]
    if changed:
        raise SystemExit(f"files differ from the record: {changed}")
    return record


def run(args):
    out = OUT / f"{args.view}-{args.split}.npz"
    if args.split == "test":
        check_record()
        claim = OUT / f"claim-{args.view}-test"
        with claim.open("x") as stream:  # one run per connectome: refuses if the claim exists
            stream.write(datetime.now(timezone.utc).isoformat() + "\n")
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    families = load(args.view, args.split)
    started = time.time()
    with Pool(args.workers) as pool:
        rows = [r for group in pool.map(run_family, families, chunksize=4) for r in group]
    fields = ("family", "rho", "size", "threshold", "strict", "truth", "clairvoyant")
    arrays = {k: np.asarray([r[k] for r in rows], dtype=np.int64) for k in fields}
    arrays.update(cost=np.stack([r["cost"] for r in rows]), strict_cost=np.stack([r["strict_cost"] for r in rows]))
    # the declared-crosswalk arms must reproduce the pre-specified runs question by question
    stored = np.load(RUNS / f"{args.view}-{args.split}.npz")
    names = json.loads((RUNS / f"{args.view}-{args.split}.json").read_text())["arms"]
    key = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(stored["family"], stored["rho"]))}
    index = np.array([key[(int(f), int(r))] for f, r in zip(arrays["family"], arrays["rho"])])
    if not np.array_equal(arrays["strict"], stored["truth"][index]):
        raise SystemExit("answers under the declared crosswalk differ from the pre-specified runs")
    for j, name in enumerate(REPRODUCE):
        if not np.array_equal(arrays["strict_cost"][:, j], stored["cost"][index, names.index(name)]):
            raise SystemExit(f"{name} under the declared crosswalk differs from the pre-specified run")
    OUT.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        np.savez(stream, **arrays)
    meta = dict(view=args.view, split=args.split, arms=arm_names(args.view), reproduced=list(REPRODUCE), sessions=len(rows),
                families=len(families), code={p: sha256(ROOT / p) for p in CODE}, result_sha256=sha256(out), wall_seconds=round(time.time() - started, 1))
    (OUT / f"{args.view}-{args.split}.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("arms", "code")}))


def clusters_of(view, family):
    with np.load(BENCH / f"{view}.npz", allow_pickle=False) as z:
        return z["target_type"][z["fam_target"][family]]


class Ratios:
    """Ratios of totals over certifiable questions, with a type bootstrap shared by every measure of one table."""

    def __init__(self, view, family, truth, weight, rng):
        self.truth, self.weight = truth, weight.astype(float)
        cluster = np.unique(clusters_of(view, family), return_inverse=True)[1]
        self.cluster, self.n = cluster, int(cluster.max()) + 1
        self.counts = np.stack([np.bincount(row, minlength=self.n) for row in rng.integers(0, self.n, size=(BOOTSTRAP, self.n))]).astype(float)

    def _parts(self, cost, base, mask):
        sx = np.bincount(self.cluster[mask], weights=cost[mask].astype(float), minlength=self.n)
        sy = np.bincount(self.cluster[mask], weights=base[mask], minlength=self.n)
        return sx.sum() / sy.sum(), (self.counts @ sx) / (self.counts @ sy)

    def measure(self, cost, base=None):
        base = self.weight if base is None else base.astype(float)
        out, parts = {}, {}
        for name, mask in (("all", self.truth > 0), ("yes", self.truth == 1), ("no", self.truth == 2)):
            parts[name] = self._parts(cost, base, mask)
            value, boot = parts[name]
            out[name] = dict(value=float(value), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))
        value = (parts["yes"][0] + parts["no"][0]) / 2
        boot = (parts["yes"][1] + parts["no"][1]) / 2
        out["balanced"] = dict(value=float(value), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))
        return out


def analyse_part(view, family, truth, cost, names, clairvoyant, rng, contrasts):
    weight = cost[:, names.index("weight")]
    ratios = Ratios(view, family, truth, weight, rng)
    result = dict(questions=int(len(truth)), answers={k: int((truth == v).sum()) for k, v in STATUS.items()})
    result["share_no_of_certifiable"] = float((truth == 2).sum() / max(1, (truth > 0).sum()))
    result["share_no_checks"] = float(weight[truth == 2].sum() / weight[truth > 0].sum())
    result["ratio"] = {n: ratios.measure(cost[:, names.index(n)]) for n in names if n != "weight"}
    result["ratio"]["clairvoyant"] = ratios.measure(np.where(clairvoyant < 0, weight, clairvoyant))
    result["contrast"] = {f"{a}/{b}": ratios.measure(cost[:, names.index(a)], base=cost[:, names.index(b)]) for a, b in contrasts if a in names and b in names}
    # regret: cells checked by a rule relative to the cheapest of the six rules with the same prior, per answer
    result["regret"] = {}
    for prior in sorted({n.split(":")[0] for n in names if ":" in n}):
        rules = [f"{prior}:{q}" for q in POLICIES]
        table = {}
        for answer, mask in (("yes", truth == 1), ("no", truth == 2)):
            totals = {r: cost[mask, names.index(r)].sum() / weight[mask].sum() for r in rules}
            best = min(totals.values())
            table[answer] = {r.split(":")[1]: float(v / best) for r, v in totals.items()}
        result["regret"][prior] = {q: dict(yes=table["yes"][q], no=table["no"][q], worst=max(table["yes"][q], table["no"][q])) for q in POLICIES}
        result["regret"][prior]["others_best_worst"] = min(result["regret"][prior][q]["worst"] for q in POLICIES if q != "recal")
    return result


def contrasts_for(prior):
    pairs = [(f"{prior}:ag", f"{prior}:direction"), (f"{prior}:recal", f"{prior}:direction"), (f"{prior}:recal", f"{prior}:ag"),
             (f"{prior}:recal", f"{prior}:nonmember"), (f"{prior}:recal", f"{prior}:member"), (f"{prior}:ag", f"{prior}:nonmember")]
    return pairs


def analyze(args):
    rng = np.random.default_rng(SEED)
    result = {}
    for view in ("malecns", "flywire"):
        priors = ["mlp_s0"] + (["provisional"] if view == "flywire" else [])
        contrasts = [c for p in priors_of(view) for c in contrasts_for(p)]
        # Part A: pre-specified runs, declared crosswalk
        z = np.load(RUNS / f"{view}-{args.split}.npz")
        names = json.loads((RUNS / f"{view}-{args.split}.json").read_text())["arms"]
        keep = ["weight"] + [f"{p}:{q}" for p in priors for q in POLICIES]
        cost = np.stack([z["cost"][:, names.index(n)] for n in keep], axis=1)
        declared = analyse_part(view, z["family"], z["truth"], cost, keep, z["clairvoyant"], rng, contrasts)
        # Part B: lenient crosswalk
        path = OUT / f"{view}-{args.split}.npz"
        lenient = None
        if path.exists():
            b = np.load(path)
            lenient = analyse_part(view, b["family"], b["truth"], b["cost"], arm_names(view), b["clairvoyant"], rng, contrasts)
            lenient["moved"] = dict(unresolved_to_no=int(((b["strict"] == 0) & (b["truth"] == 2)).sum()),
                                    unresolved_to_yes=int(((b["strict"] == 0) & (b["truth"] == 1)).sum()),
                                    contradictions=int(((b["strict"] > 0) & (b["strict"] != b["truth"])).sum()))
        result[view] = dict(declared=declared, lenient=lenient)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"analysis-{args.split}.json"
    if args.split == "test":
        check_record()
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    if args.split == "test":
        verdicts = evaluate(json.loads(RECORD.read_text())["expectations"], result)
        (OUT / "verdicts-test.json").write_text(json.dumps(verdicts, indent=1) + "\n")
        print(f"{sum(v['holds'] for v in verdicts)} of {len(verdicts)} expectations hold")
    else:
        print(json.dumps(summary(result), indent=1))


def lookup(result, path):
    x = result
    for part in path.split("|"):
        x = x[part]
    return x["value"] if isinstance(x, dict) and "value" in x else x


def evaluate(expectations, result):
    verdicts = []
    for x in expectations:
        value = lookup(result, x["quantity"])
        holds = (x.get("min") is None or value >= x["min"]) and (x.get("max") is None or value <= x["max"])
        verdicts.append(dict(x, value=value, holds=bool(holds)))
    return verdicts


def summary(result):
    out = {}
    for view, parts in result.items():
        for part, r in parts.items():
            if r is None:
                continue
            out[f"{view}/{part}/share_no"] = round(r["share_no_of_certifiable"], 3)
            for n, m in r["ratio"].items():
                out[f"{view}/{part}/{n}"] = {k: round(v["value"], 3) for k, v in m.items()}
            for n, m in r["contrast"].items():
                out[f"{view}/{part}/{n}"] = {k: (round(v["value"], 3), round(v["low"], 3), round(v["high"], 3)) for k, v in m.items()}
            for prior, rules in r["regret"].items():
                out[f"{view}/{part}/regret/{prior}"] = {q: round(v["worst"], 3) if isinstance(v, dict) else round(v, 3) for q, v in rules.items()}
    return out


def write_record(args):
    if RECORD.exists():
        raise SystemExit("the record exists")
    expectations = json.loads((OUT / "expectations.json").read_text())
    record = dict(written_utc=datetime.now(timezone.utc).isoformat(), purpose=__doc__.split("\n\n")[0],
                  code={p: sha256(ROOT / p) for p in CODE}, data={p: sha256(ROOT / p) for p in DATA},
                  development={p: sha256(OUT / p) for p in ("malecns-development.npz", "flywire-development.npz", "analysis-development.json")},
                  expectations=expectations["expectations"], notes=expectations.get("notes", []))
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    RECORD.chmod(0o444)
    print(f"recorded {len(record['expectations'])} expectations")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--view", required=True, choices=("malecns", "flywire"))
    r.add_argument("--split", required=True, choices=("development", "test"))
    r.add_argument("--workers", type=int, default=28)
    a = sub.add_parser("analyze")
    a.add_argument("--split", required=True, choices=("development", "test"))
    sub.add_parser("record")
    args = parser.parse_args()
    dict(run=run, analyze=analyze, record=write_record)[args.command](args)


if __name__ == "__main__":
    main()
