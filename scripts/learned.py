"""A learned policy on the new target neurons of the replication: one held-out evaluation.

The policy of ``neurosym/policy.py`` checks the cell with the largest ``theta . phi`` over nine features of the
current beliefs and bounds (among them the gain of Adaptive Greedy, the progress toward each answer and the weight),
with beliefs mixed between the prior and its recalibration. Adaptive Greedy, recalibrated AG, the single-goal policy
and direction-fixed are special cases of it. Its parameters were fitted earlier, by the cross-entropy method on the
total number of checks, to benchmark questions of one half of the development cell types, separately for each
connectome and prior (``outputs/neurosym/A/fit-<view>-<prior>.json``, entry ``final``). Nothing is fitted here.

The script evaluates the fitted policy on the question arrays of the replication (``outputs/replication/benchmark``:
new target neurons, retention and share questions), first on the development types and, after ``record`` has pinned
code, inputs and expected outcomes, once per connectome on the held-out types. Every question must reproduce the
costs of weight order and of recalibrated AG that the replication stored.

Usage:
  learned.py run --view flywire --split development
  learned.py analyze --split development
  learned.py record                                   # needs outputs/learned/expectations.json
  learned.py run --view flywire --split test          # once per connectome
  learned.py analyze --split test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import replication as rep  # noqa: E402
from connectomequest.annotation_v35 import engine as e  # noqa: E402
from connectomequest.neurosym import policy as P  # noqa: E402

OUT = ROOT / "outputs/learned"
RECORD = OUT / "record.json"
FITS = ROOT / "outputs/neurosym/A"
SEED = 20261115
CODE = ("scripts/learned.py", "scripts/replication.py", "src/connectomequest/neurosym/policy.py", "src/connectomequest/annotation_v35/engine.py")
DATA = ("outputs/replication/benchmark/manifest.json", "outputs/replication/benchmark/malecns.npz", "outputs/replication/benchmark/flywire.npz",
        "outputs/replication/record.json", "outputs/neurosym/A/fit-malecns-mlp_s0.json", "outputs/neurosym/A/fit-flywire-mlp_s0.json",
        "outputs/neurosym/A/fit-flywire-provisional.json")
_D = {}


def fitted(view, prior):
    js = json.loads((FITS / f"fit-{view}-{prior}.json").read_text())["final"]["params"]
    return dict(theta=np.array([js["theta"][k] for k in P.FEATURES]), recal=js["recal"], strength=js["strength"], fallback=js["fallback"])


def arm_names(view):
    return ["weight"] + [f"{prior}:{policy}" for prior in rep.PRIORS[view] for policy in ("recal", "learned")]


def run_family(family):
    d, view, arms, params = _D["data"], _D["view"], _D["arms"], _D["params"]
    target = int(d["fam_target"][family])
    lo, hi = int(d["target_offset"][target]), int(d["target_offset"][target + 1])
    cells, weights, labels = d["pre_cell"][lo:hi], d["pre_weight"][lo:hi], d["pre_label"][lo:hi]
    n = hi - lo
    codes = d["fam_codes"][d["fam_code_offset"][family] : d["fam_code_offset"][family + 1]]
    outcomes = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
    total = int(weights.sum())
    start = int(d["fam_plus_offset"][family])
    priors = {p: rep.probabilities(d["plus_" + p][start : start + n], d["unknown_" + p][lo:hi]) for p in rep.PRIORS[view]}
    rows = []
    for level, (question, num, den) in enumerate(rep.LEVELS):
        threshold = rep.threshold_of(question, num, den, total, int(d["fam_source_weight"][family]), int(d["fam_source_total"][family]))
        inst = e.threshold_instance(cells, weights, threshold)
        truth = e.full_information_status(inst, outcomes)
        bound = e.clairvoyant_cost(inst, outcomes)
        cost = np.zeros(len(arms), dtype=np.int32)

        def record(name, order):
            c, s = e.evaluate(inst, outcomes, order)
            if s != truth or (truth == "unresolved" and c != inst.size) or (bound is not None and c < bound):
                raise AssertionError(f"{name} contradicts the exact full-information status")
            cost[arms[name]] = c

        record("weight", e.weight_order(inst))
        for prior, p in priors.items():
            record(f"{prior}:recal", e.run_adaptive(inst, outcomes, e.RecalibratedGreedy(inst, p)))
            record(f"{prior}:learned", e.run_adaptive(inst, outcomes, P.make(inst, p, params[prior])))
        rows.append(dict(family=family, level=level, truth=rep.STATUS[truth], clairvoyant=-1 if bound is None else bound, cost=cost))
    return rows


def check_record():
    if not RECORD.exists():
        raise SystemExit("the held-out split needs the record")
    record = json.loads(RECORD.read_text())
    for group in ("code", "data"):
        for path, digest in record[group].items():
            if rep.sha256(ROOT / path) != digest:
                raise SystemExit(f"{path} differs from the record")
    return record


def reproduce(view, split, arrays, names):
    """The questions and the costs of weight order and recalibrated AG must be those of the replication."""
    meta = json.loads((rep.OUT / f"{view}-{split}.json").read_text())
    with np.load(rep.OUT / f"{view}-{split}.npz", allow_pickle=False) as z:
        stored = {key: z[key] for key in ("family", "level", "truth", "clairvoyant", "cost")}
    for key in ("family", "level", "truth", "clairvoyant"):
        if not np.array_equal(stored[key], arrays[key]):
            raise SystemExit(f"{key} differs from the replication")
    for i, name in enumerate(names):
        if not name.endswith(":learned") and not np.array_equal(stored["cost"][:, meta["arms"].index(name)], arrays["cost"][:, i]):
            raise SystemExit(f"the cost of {name} differs from the replication")


def run(args):
    view, split = args.view, args.split
    manifest = json.loads((rep.BENCH / "manifest.json").read_text())
    entry = next(v for v in manifest["views"] if v["view"] == view)
    if rep.sha256(rep.BENCH / f"{view}.npz") != entry["sha256"]:
        raise SystemExit("question arrays differ from their manifest")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{view}-{split}.npz"
    if split == "test":
        check_record()
        if out.exists():
            raise SystemExit("the held-out result exists")
        with (OUT / f"claim-{view}-test").open("x") as stream:
            stream.write(datetime.now(timezone.utc).isoformat() + "\n")
    d = rep.load(view)
    names = arm_names(view)
    _D.update(data=d, view=view, arms={n: i for i, n in enumerate(names)}, params={prior: fitted(view, prior) for prior in rep.PRIORS[view]})
    families = rep.split_families(view, d, split)
    started = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        results = pool.map(run_family, families, chunksize=4)
    rows = [row for group in results for row in group]
    arrays = {key: np.asarray([r[key] for r in rows], dtype=np.int64) for key in ("family", "level", "truth", "clairvoyant")}
    arrays["cost"] = np.stack([r["cost"] for r in rows])
    reproduce(view, split, arrays, names)
    with out.open("wb" if split == "development" else "xb") as stream:
        np.savez(stream, **arrays)
    meta = dict(view=view, split=split, arms=names, questions=len(rows), families=len(families),
                parameters={prior: json.loads((FITS / f"fit-{view}-{prior}.json").read_text())["final"]["params"] for prior in rep.PRIORS[view]},
                code={p: rep.sha256(ROOT / p) for p in CODE}, arrays_sha256=entry["sha256"], wall_seconds=time.time() - started)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in meta.items() if key not in ("arms", "code", "parameters")}))


def analyse_view(view, split):
    d = rep.load(view)
    meta = json.loads((OUT / f"{view}-{split}.json").read_text())
    with np.load(OUT / f"{view}-{split}.npz", allow_pickle=False) as z:
        r = {key: z[key] for key in z.files}
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
            learned, recal = r["cost"][:, arm[f"{prior}:learned"]], r["cost"][:, arm[f"{prior}:recal"]]
            out[prior] = dict(learned=ratios.measure(learned, weight, mask), recal=ratios.measure(recal, weight, mask),
                              learned_vs_recal=ratios.measure(learned, recal, mask))
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
                                          for name, x in m.items() if name != "recal"})


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
    r = sub.add_parser("run")
    r.add_argument("--view", required=True, choices=tuple(rep.PRIORS))
    r.add_argument("--split", required=True, choices=("development", "test"))
    r.add_argument("--workers", type=int, default=28)
    a = sub.add_parser("analyze")
    a.add_argument("--split", required=True, choices=("development", "test"))
    sub.add_parser("record")
    args = parser.parse_args()
    dict(run=run, analyze=analyze, record=write_record)[args.command](args)


if __name__ == "__main__":
    main()
