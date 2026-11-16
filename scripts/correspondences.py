#!/usr/bin/env python3
"""Which missing type correspondences keep questions uncertifiable (exploratory, descriptive; no policy is run).

A question asks whether the member weight of the input cells of a target neuron reaches a threshold T. With every
cell checked, L is the member weight and U adds the weight of the unresolved cells; the question is uncertifiable
when L < T <= U. Most unresolved cells carry a type that exists only in the newer release and has no declared
hemibrain correspondence ("release-specific type"). A declaration for such a type would place all its cells inside
the partner set (members) or outside it (nonmembers). For an uncertifiable question and a release-specific type
with weight w among its input cells:

  certified alone   the question is certified "no" if the type alone is declared to have no counterpart in the
                    partner set:  U - w < T
  decided alone     the question is certified whatever the declaration says:  w >= max(T - L, U - T + 1)

The reading of the paper that takes every release-specific type as a nonmember certifies a question when the
weights of all its release-specific types reach the deficit U - T + 1 ("reachable" question). A priority list
orders the types: each step takes the type whose declaration as nonmember certifies the most reachable questions
not certified yet; ties go to the type that removes the largest total fraction of the remaining deficits, then to
the name. The list is evaluated on the questions it was built from and on other questions (on the development
split: built from half of the target types and applied to the other half; on the held-out split: the development
list applied to the held-out questions).

Stages
  run --view V --split development     per-question arrays and the type list; free to repeat (--force)
  analyze --split development          summary with intervals over target types
  record                               pins code, inputs, development outputs and outputs/correspondences/expectations.json
  run --view V --split test            once per connectome, only after the record (claim file)
  analyze --split test                 summary and verdicts on the recorded expectations
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "outputs/v35/benchmark"
RUNS = ROOT / "outputs/v35/runs"
LENIENT = ROOT / "outputs/v35/analysis"
OUT = ROOT / "outputs/correspondences"
RECORD = OUT / "record.json"
RHOS = ((1, 4), (1, 2), (3, 4), (1, 1))
VIEWS = ("malecns", "flywire")
RELEASE_SPECIFIC = "release-specific type without correspondence"
SUPERCLASS = dict(malecns="superclass", flywire="super_class")
FIRST = (10, 100, 500, 1000)
SHARES = (0.25, 0.5, 0.75, 0.9)
BOOTSTRAP = 2000
SEED = 20261113
CODE = ("scripts/correspondences.py", "scripts/v35_diagnostics.py", "src/connectomequest/annotation_v35/engine.py",
        "src/connectomequest/annotation_v32/data.py")
DATA = ("outputs/v35/benchmark/manifest.json", "outputs/v35/benchmark/malecns.npz", "outputs/v35/benchmark/flywire.npz",
        "data/raw/manc/body-annotations-male-cns-v1.0-minconf-0.5.feather", "inputs/v32/flywire/cq-flywire-v3.1.0.tsv",
        "inputs/v32/prepared-common-v1/hemibrain/public/manifest.json")
DEVELOPMENT = tuple(f"{view}-development.npz" for view in VIEWS) + tuple(f"types-{view}-development.csv" for view in VIEWS) + (
    "analysis-development.json",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------- arithmetic of one question
def certified_alone(lower, upper, threshold, weight):
    """Declaring the type a nonmember certifies "no"."""
    return upper - weight < threshold


def decided_alone(lower, upper, threshold, weight):
    """Either declaration certifies the question: member gives L + w >= T, nonmember gives U - w < T."""
    return weight >= max(threshold - lower, upper - threshold + 1)


def types_needed(deficit, weights):
    """Smallest number of types whose declaration as nonmembers certifies "no"; 0 if all of them do not."""
    total = 0
    for k, w in enumerate(sorted(weights, reverse=True), 1):
        total += w
        if total >= deficit:
            return k
    return 0


# ---------------------------------------------------------------------------------------------- priority list
def greedy_list(question, kind, weight, deficit, n_types, names):
    """Order of the types and the number of questions certified after each step.

    question, kind, weight: one entry per (question, release-specific type) pair; deficit: U - T + 1 per question.
    Only questions that all their types together certify take part.
    """
    reach = (np.bincount(question, minlength=len(deficit)) > 0) & (np.bincount(question, weights=weight, minlength=len(deficit)) >= deficit)
    keep = reach[question]
    question, kind, weight = question[keep], kind[keep], weight[keep].astype(float)
    remaining = deficit.astype(float).copy()
    initial = remaining.copy()
    done = ~reach
    order, certified = [], []
    rank_of_name = np.argsort(np.argsort(np.array(names)))  # ties go to the first name in sorted order
    while len(question):
        left = remaining[question]
        newly = np.bincount(kind, weights=weight >= left, minlength=n_types)
        gain = np.bincount(kind, weights=np.minimum(weight, left) / initial[question], minlength=n_types)
        if gain.max() <= 0:
            break
        best = np.flatnonzero(newly == newly.max())
        best = best[np.isclose(gain[best], gain[best].max(), rtol=1e-12, atol=0)]
        chosen = int(best[np.argmin(rank_of_name[best])])
        hit = kind == chosen
        remaining[question[hit]] -= weight[hit]
        done |= remaining <= 0
        live = ~hit & ~done[question]
        question, kind, weight = question[live], kind[live], weight[live]
        order.append(chosen)
        certified.append(int(done.sum() - (~reach).sum()))
    return np.array(order, dtype=np.int64), np.array(certified, dtype=np.int64)


def apply_list(order, question, kind, weight, deficit, n_types):
    """Step of the list at which each question is certified (1 = after the first type; 0 = never)."""
    rank = np.full(n_types, -1, dtype=np.int64)
    rank[order] = np.arange(1, len(order) + 1)
    r = rank[kind]
    keep = r > 0
    question, r, weight = question[keep], r[keep], weight[keep]
    index = np.lexsort((r, question))
    question, r, weight = question[index], r[index], weight[index]
    total = np.cumsum(weight)
    start = np.flatnonzero(np.r_[True, question[1:] != question[:-1]]) if len(question) else np.array([], dtype=np.int64)
    before = np.repeat(np.r_[0, total][start], np.diff(np.r_[start, len(question)]))
    reached = (total - before) >= deficit[question]
    step = np.zeros(len(deficit), dtype=np.int64)
    first = np.flatnonzero(reached)
    first = first[np.r_[True, question[first][1:] != question[first][:-1]]] if len(first) else first
    step[question[first]] = r[first]
    return step


# ---------------------------------------------------------------------------------------------- data
def annotation_table(view):
    """cell id -> (type of the release, hemibrain type field, superclass of the release)."""
    from v35_diagnostics import VIEWS as SOURCES

    path = ROOT / SOURCES[view]["annotations"]
    if view == "flywire":
        with path.open(newline="") as stream:
            return {int(r["root_id"]): (r["cell_type"], r["hemibrain_type"], r[SUPERCLASS[view]] or "none")
                    for r in csv.DictReader(stream, delimiter="\t")}
    import pyarrow.ipc as ipc

    with path.open("rb") as stream:
        table = ipc.open_file(stream).read_all().select(["bodyId", "type", "hemibrainType", SUPERCLASS[view]]).to_pylist()
    return {int(r["bodyId"]): (r["type"], r["hemibrainType"], r[SUPERCLASS[view]] or "none") for r in table}


def release_specific_types(view, cells, labels):
    """Per input cell: index of its release-specific type (-1 otherwise), with type names and superclasses."""
    from v35_diagnostics import VIEWS as SOURCES, unresolved_reason

    vocabulary = json.loads((ROOT / DATA[5]).read_text())["vocabulary"]
    lookup = {name: i for i, name in enumerate(vocabulary) if i}
    rows = annotation_table(view)
    cache, index_of, votes = {}, {}, {}
    kind = np.full(len(cells), -1, dtype=np.int64)
    for i in np.flatnonzero(labels < 0).tolist():
        cell = int(cells[i])
        if cell not in cache:
            primary, secondary, superclass = rows.get(cell, (None, None, "none"))
            if unresolved_reason(primary, secondary, SOURCES[view]["release"], lookup) == RELEASE_SPECIFIC:
                cache[cell] = index_of.setdefault(primary, len(index_of))
                votes.setdefault(primary, {}).setdefault(superclass, set()).add(cell)
            else:
                cache[cell] = -1
        kind[i] = cache[cell]
    names = sorted(index_of, key=index_of.get)
    superclass = [min(votes[name], key=lambda s: (-len(votes[name][s]), s)) for name in names]  # majority of the cells
    return kind, names, superclass


def split_families(bench, split):
    manifest = json.loads((BENCH / "manifest.json").read_text())
    codes = [t["code"] for t in manifest["types"] if t["split"] == split]
    return np.flatnonzero(np.isin(bench["target_type"], codes)[bench["fam_target"]])


# ---------------------------------------------------------------------------------------------- run
def check_record():
    record = json.loads(RECORD.read_text())
    changed = [p for p, h in {**record["code"], **record["data"]}.items() if sha256(ROOT / p) != h]
    changed += [p for p, h in record["development"].items() if sha256(OUT / p) != h]
    if changed:
        raise SystemExit(f"files differ from the record: {changed}")
    return record


def run(args):
    from connectomequest.annotation_v35 import engine as e

    out = OUT / f"{args.view}-{args.split}.npz"
    if args.split == "test":
        check_record()
        with (OUT / f"claim-{args.view}-test").open("x") as stream:  # one run per connectome: refuses if the claim exists
            stream.write(datetime.now(timezone.utc).isoformat() + "\n")
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
    elif out.exists() and not args.force:
        raise SystemExit(f"{out} exists (use --force on the development split)")
    with np.load(BENCH / f"{args.view}.npz", allow_pickle=False) as z:
        bench = {k: z[k] for k in ("target_type", "target_offset", "pre_cell", "pre_weight", "pre_label", "fam_target", "fam_codes",
                                   "fam_code_offset", "fam_source_weight", "fam_source_total")}
    kind_of_cell, names, superclass = release_specific_types(args.view, bench["pre_cell"], bench["pre_label"])
    classes = sorted(set(superclass))
    class_of = np.array([classes.index(s) for s in superclass], dtype=np.int64)
    rows, pairs = [], []
    for f in split_families(bench, args.split).tolist():
        t = int(bench["fam_target"][f])
        lo, hi = int(bench["target_offset"][t]), int(bench["target_offset"][t + 1])
        weights, labels, kinds = bench["pre_weight"][lo:hi], bench["pre_label"][lo:hi], kind_of_cell[lo:hi]
        codes = bench["fam_codes"][bench["fam_code_offset"][f]:bench["fam_code_offset"][f + 1]]
        lower = int(weights[(labels >= 0) & np.isin(labels, codes)].sum())
        upper = lower + int(weights[labels < 0].sum())
        total = int(weights.sum())
        mine = kinds >= 0
        per_type = np.bincount(kinds[mine], weights=weights[mine], minlength=len(names)).astype(np.int64)
        present = np.flatnonzero(per_type)
        for rho, (num, den) in enumerate(RHOS):
            threshold, _ = e.threshold_goals(total, num * int(bench["fam_source_weight"][f]), den * int(bench["fam_source_total"][f]))
            status = 1 if lower >= threshold else 2 if upper < threshold else 0
            row = dict(family=f, rho=rho, cluster=int(bench["target_type"][t]), lower=lower, upper=upper, threshold=threshold,
                       total=total, status=status)
            if status == 0:
                w = per_type[present]
                row.update(alone=bool(certified_alone(lower, upper, threshold, w).any()),
                           decided=bool((w >= max(threshold - lower, upper - threshold + 1)).any()),
                           needed=types_needed(upper - threshold + 1, w.tolist()))
                pairs.append((np.full(len(present), len(rows)), present, w))
            rows.append(row)
    fields = ("family", "rho", "cluster", "lower", "upper", "threshold", "total", "status")
    arrays = {k: np.array([r[k] for r in rows], dtype=np.int64) for k in fields}
    arrays["alone"] = np.array([r.get("alone", False) for r in rows])
    arrays["decided"] = np.array([r.get("decided", False) for r in rows])
    arrays["needed"] = np.array([r.get("needed", 0) for r in rows], dtype=np.int64)
    arrays["pair_question"] = np.concatenate([p[0] for p in pairs]).astype(np.int64)
    arrays["pair_type"] = np.concatenate([p[1] for p in pairs]).astype(np.int64)
    arrays["pair_weight"] = np.concatenate([p[2] for p in pairs]).astype(np.int64)
    arrays["type_class"] = class_of

    # the answers must reproduce the pre-specified runs question by question
    stored = np.load(RUNS / f"{args.view}-{args.split}.npz")
    key = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(stored["family"], stored["rho"]))}
    index = np.array([key[(int(f), int(r))] for f, r in zip(arrays["family"], arrays["rho"])])
    for mine_, theirs in (("status", "truth"), ("threshold", "threshold"), ("lower", "member"), ("total", "total")):
        if len(index) != len(stored["family"]) or not np.array_equal(arrays[mine_], stored[theirs][index]):
            raise SystemExit(f"{mine_} differs from the pre-specified runs")
    if not np.array_equal(arrays["upper"] - arrays["lower"], stored["unresolved"][index]):
        raise SystemExit("unresolved weight differs from the pre-specified runs")

    deficit = arrays["upper"] - arrays["threshold"] + 1
    order, certified = greedy_list(arrays["pair_question"], arrays["pair_type"], arrays["pair_weight"], deficit, len(names), names)
    arrays["step"] = apply_list(order, arrays["pair_question"], arrays["pair_type"], arrays["pair_weight"], deficit, len(names))
    arrays["order"] = order
    if not np.array_equal(np.bincount(arrays["step"][arrays["step"] > 0], minlength=len(order) + 1)[1:].cumsum(), certified):
        raise SystemExit("the list applied to its own questions does not reproduce its construction")

    OUT.mkdir(parents=True, exist_ok=True)
    with out.open("wb" if args.split == "development" else "xb") as stream:
        np.savez(stream, **arrays)
    blocked = np.bincount(arrays["pair_type"], minlength=len(names))
    w, q = arrays["pair_weight"], arrays["pair_question"]
    alone = np.bincount(arrays["pair_type"], weights=certified_alone(arrays["lower"][q], arrays["upper"][q], arrays["threshold"][q], w),
                        minlength=len(names)).astype(int)
    decided = np.bincount(arrays["pair_type"], minlength=len(names), weights=w >= np.maximum(
        arrays["threshold"][q] - arrays["lower"][q], arrays["upper"][q] - arrays["threshold"][q] + 1)).astype(int)
    rank = {int(k): i + 1 for i, k in enumerate(order)}
    listed = sorted(range(len(names)), key=lambda k: (rank.get(k, len(names) + 1), -blocked[k], names[k]))
    with (OUT / f"types-{args.view}-{args.split}.csv").open("w", newline="") as stream:
        table = csv.writer(stream, lineterminator="\n")
        table.writerow(["rank", "type", "superclass", "uncertifiable_questions_with_type", "certified_alone", "decided_alone",
                        "certified_by_list_so_far"])
        for k in listed:
            table.writerow([rank.get(k, ""), names[k], superclass[k], int(blocked[k]), alone[k], decided[k],
                            int(certified[rank[k] - 1]) if k in rank else ""])
    meta = dict(view=args.view, split=args.split, questions=len(rows), types=len(names), classes=classes, type_names=names,
                list_length=len(order), code={p: sha256(ROOT / p) for p in CODE}, result_sha256=sha256(out))
    (OUT / f"{args.view}-{args.split}.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("code", "classes", "type_names")}))


# ---------------------------------------------------------------------------------------------- analysis
def resampler(cluster, rng):
    index = np.unique(cluster, return_inverse=True)[1]
    n = int(index.max()) + 1
    counts = np.stack([np.bincount(row, minlength=n) for row in rng.integers(0, n, size=(BOOTSTRAP, n))]).astype(float)

    def share(numerator, denominator):
        a = np.bincount(index, weights=np.asarray(numerator, dtype=float), minlength=n)
        b = np.bincount(index, weights=np.asarray(denominator, dtype=float), minlength=n)
        boot = (counts @ a) / (counts @ b)
        return dict(value=float(a.sum() / b.sum()), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))

    return share


def list_summary(step, reach, share, length):
    out = dict(length=int(length), first={str(k): share((step > 0) & (step <= k), reach) for k in FIRST},
               whole=share(step > 0, reach))
    curve = np.bincount(step[step > 0], minlength=length + 1)[1:].cumsum()
    out["types_for"] = {f"{s:.2f}": (int(np.searchsorted(curve, s * reach.sum()) + 1) if len(curve) and curve[-1] >= s * reach.sum() else None)
                        for s in SHARES}
    return out


def analyse_view(view, split, development_order=None):
    z = np.load(OUT / f"{view}-{split}.npz")
    meta = json.loads((OUT / f"{view}-{split}.json").read_text())
    with (OUT / f"types-{view}-{split}.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    status, cluster = z["status"], z["cluster"]
    uncertifiable = status == 0
    question, kind, weight = z["pair_question"], z["pair_type"], z["pair_weight"]
    deficit = z["upper"] - z["threshold"] + 1
    names = meta["type_names"]
    reach = uncertifiable & (np.bincount(question, weights=weight, minlength=len(status)) >= deficit)
    share = resampler(cluster, np.random.default_rng(SEED))
    out = dict(questions=len(status), target_types=len(np.unique(cluster)), uncertifiable=share(uncertifiable, np.ones(len(status))),
               reachable=share(reach, uncertifiable), types=int(len(np.unique(kind))),
               types_in_reachable=int(len(np.unique(kind[reach[question]]))),
               certified_alone=share(z["alone"], uncertifiable), decided_alone=share(z["decided"], uncertifiable))
    needed = z["needed"][reach]
    out["types_needed"] = dict(q25=float(np.quantile(needed, 0.25)), median=float(np.median(needed)), q75=float(np.quantile(needed, 0.75)),
                               one=share(reach & (z["needed"] == 1), reach), over_five=share(reach & (z["needed"] > 5), reach))
    # superclasses of the release: share of the release-specific unresolved weight, and questions that the class alone certifies
    class_weight = np.zeros((len(status), len(meta["classes"])))
    np.add.at(class_weight, (question, z["type_class"][kind]), weight)
    all_weight = class_weight.sum(1)
    present = np.unique(kind)
    out["superclass"] = {}
    for j, name in enumerate(meta["classes"]):
        if not class_weight[:, j].any():
            continue
        out["superclass"][name] = dict(types=int((z["type_class"][present] == j).sum()), weight=share(class_weight[:, j], all_weight),
                                       certifies_alone=share(uncertifiable & (class_weight[:, j] >= deficit), uncertifiable))
    out["list"] = list_summary(z["step"], reach, share, len(z["order"]))
    out["top"] = [dict(type=r["type"], superclass=r["superclass"], questions=int(r["uncertifiable_questions_with_type"]),
                       alone=int(r["certified_alone"]), decided=int(r["decided_alone"]), so_far=int(r["certified_by_list_so_far"]))
                  for r in table[:10]]

    # a list applied to questions it was not built from
    if split == "development":  # built from half of the target types, applied to the other half
        codes = np.unique(cluster)
        half = np.isin(cluster, np.random.default_rng(SEED + 1).permutation(codes)[: len(codes) // 2])
        a = half[question]
        order, _ = greedy_list(question[a], kind[a], weight[a], deficit, len(names), names)
        step = apply_list(order, question[~a], kind[~a], weight[~a], deficit, len(names))
        other = reach & ~half
        out["transfer"] = dict(source="half of the development target types", **list_summary(np.where(other, step, 0), other, share, len(order)))
    else:
        index = {name: k for k, name in enumerate(names)}
        order = np.array([index[name] for name in development_order if name in index], dtype=np.int64)
        step = apply_list(order, question, kind, weight, deficit, len(names))
        out["transfer"] = dict(source="development list", development_length=len(development_order), **list_summary(
            np.where(reach, step, 0), reach, share, len(order)))
    return out


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


def analyze(args):
    if args.split == "test":
        record = check_record()
    result = {}
    for view in VIEWS:
        development_order = None
        if args.split == "test":
            with (OUT / f"types-{view}-development.csv").open(newline="") as stream:
                development_order = [r["type"] for r in csv.DictReader(stream) if r["rank"]]
        result[view] = analyse_view(view, args.split, development_order)
        # the shares must agree with the analysis of the lenient reading
        lenient = json.loads((LENIENT / f"lenient-{args.split}.json").read_text())[view]
        if abs(result[view]["uncertifiable"]["value"] - lenient["answers"]["strict"]["unresolved"]) > 1e-12 or abs(
                result[view]["reachable"]["value"] - lenient["moved_from_unresolved"]["to_negative"]) > 1e-12:
            raise SystemExit(f"{view}: shares differ from the analysis of the lenient reading")
    (OUT / f"analysis-{args.split}.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    if args.split == "test":
        verdicts = evaluate(record["expectations"], result)
        (OUT / "verdicts-test.json").write_text(json.dumps(verdicts, indent=1) + "\n")
        print(f"{sum(v['holds'] for v in verdicts)} of {len(verdicts)} expectations hold")
        for v in verdicts:
            if not v["holds"]:
                print("  does not hold:", v["text"], v["value"])
    for view in VIEWS:
        r = result[view]
        print(view, json.dumps(dict(uncertifiable=round(r["uncertifiable"]["value"], 4), reachable=round(r["reachable"]["value"], 4),
                                    types=r["types"], alone=round(r["certified_alone"]["value"], 4),
                                    decided=round(r["decided_alone"]["value"], 4), needed=r["types_needed"]["median"],
                                    types_for=r["list"]["types_for"], first={k: round(v["value"], 3) for k, v in r["list"]["first"].items()},
                                    transfer={k: round(v["value"], 3) for k, v in r["transfer"]["first"].items()},
                                    transfer_whole=round(r["transfer"]["whole"]["value"], 3))))
        for name, c in sorted(r["superclass"].items(), key=lambda kv: -kv[1]["weight"]["value"])[:5]:
            print(f"   {name:20s} types {c['types']:5d} weight {c['weight']['value']:.3f} certifies alone {c['certifies_alone']['value']:.3f}")


def write_record(args):
    if RECORD.exists():
        raise SystemExit("the record exists")
    expectations = json.loads((OUT / "expectations.json").read_text())
    record = dict(written_utc=datetime.now(timezone.utc).isoformat(), purpose=__doc__.split("\n\n")[0],
                  code={p: sha256(ROOT / p) for p in CODE}, data={p: sha256(ROOT / p) for p in DATA},
                  development={p: sha256(OUT / p) for p in DEVELOPMENT},
                  expectations=expectations["expectations"], notes=expectations.get("notes", []))
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    RECORD.chmod(0o444)
    print(f"recorded {len(record['expectations'])} expectations")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("run")
    p.add_argument("--view", required=True, choices=VIEWS)
    p.add_argument("--split", required=True, choices=("development", "test"))
    p.add_argument("--force", action="store_true")
    p.set_defaults(call=run)
    p = sub.add_parser("analyze")
    p.add_argument("--split", required=True, choices=("development", "test"))
    p.set_defaults(call=analyze)
    p = sub.add_parser("record")
    p.set_defaults(call=write_record)
    args = parser.parse_args()
    args.call(args)


if __name__ == "__main__":
    main()
