#!/usr/bin/env python3
"""Replication on new target neurons, with share questions and two generic policies.

The benchmark asks its questions about the first two neurons of every cell type in a pseudo-random order. This
script asks them about the next two neurons of the same order, which no earlier run has touched, so that findings
first obtained as exploratory analyses can be tested as hypotheses recorded in advance. For every new target neuron
and partner set it asks

* the four *retention questions* of the benchmark, ``T = ceil(rho * s * W)``, and
* three *share questions*, ``T = ceil(tau * W)`` with ``tau`` in 1/4, 1/2, 3/4: does the partner set supply at
  least that share of the input synapses? Under the declared crosswalk many of them are answered *no*.

Besides the policies of the benchmark it runs outcome propagation (recorded setting), propagation without a
predictor, the single-goal policy with recalibration, calibrated synthetic priors of a chosen AUROC, and two generic
policies: edge cutting (decision region determination) and information gain (active feature acquisition), each with
the prior as it is and with recalibrated beliefs (``connectomequest.annotation_v35.decision``).

Order of events: ``build`` (question arrays for the new targets of both splits, checked against the benchmark
arrays of the old targets), ``run --split development``, ``analyze --split development``, ``record`` (code, inputs
and the expected outcomes of ``outputs/replication/expectations.json``), then ``run --split test`` once per
connectome and ``analyze --split test``, which evaluates the expectations.

Usage:
  PYTHONPATH=src:.v32-deps:scripts python scripts/replication.py build
  PYTHONPATH=src:.v32-deps:scripts python scripts/replication.py run --view flywire --split development
  PYTHONPATH=src:.v32-deps:scripts python scripts/replication.py analyze --split development
  PYTHONPATH=src:.v32-deps:scripts python scripts/replication.py record
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from connectomequest.annotation_v35 import decision  # noqa: E402
from connectomequest.annotation_v35 import engine as e  # noqa: E402
from connectomequest.annotation_v35 import sweep  # noqa: E402

OUT = ROOT / "outputs/replication"
BENCH = OUT / "benchmark"
RECORD = OUT / "record.json"
OLD = ROOT / "outputs/v35/benchmark"
FIRST, LAST = 2, 4  # the third and fourth neuron of a type in the order of the benchmark, which used the first two
RETENTION = ((1, 4), (1, 2), (3, 4), (1, 1))
SHARE = ((1, 4), (1, 2), (3, 4))
LEVELS = tuple(("retention", n, d) for n, d in RETENTION) + tuple(("share", n, d) for n, d in SHARE)
SETTING = dict(lam=8.0, gamma=3.0, tau=0.3)  # outcome propagation, as recorded for the benchmark
AUROCS = (0.7, 0.8, 0.9)
ORDERS = ("member", "nonmember", "direction", "ag", "adg", "recal", "single-recal", "prop")
GENERIC = tuple(decision.POLICIES)
SIX = ("member", "nonmember", "direction", "ag", "adg", "recal")
PRIORS = dict(malecns=("mlp_s0",), flywire=("mlp_s0", "provisional"))
BOOTSTRAP, SEED = 2000, 20261114
STATUS = {"unresolved": 0, "positive": 1, "negative": 2}
CODE = ("scripts/replication.py", "src/connectomequest/annotation_v35/decision.py", "src/connectomequest/annotation_v35/engine.py",
        "src/connectomequest/annotation_v35/sweep.py", "src/connectomequest/neurosym/kernel.py", "src/connectomequest/neurosym/common.py",
        "src/connectomequest/neurosym/constraints.py", "scripts/v35_build.py")
DATA = ("outputs/v35/benchmark/manifest.json", "outputs/replication/benchmark/manifest.json", "outputs/replication/benchmark/malecns.npz",
        "outputs/replication/benchmark/flywire.npz", "outputs/v35/sweep/malecns-development.json", "outputs/v35/sweep/flywire-development.json",
        "outputs/neurosym/cache/malecns.npz", "outputs/neurosym/cache/flywire.npz")
_D = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def arm_names(view):
    names = ["weight"]
    for prior in PRIORS[view]:
        names += [f"{prior}:{policy}" for policy in ORDERS + GENERIC]
    names.append("none:prop")
    for level in AUROCS:
        names += [f"synthetic{level}:ag", f"synthetic{level}:recal"]
    return names


# --------------------------------------------------------------------------- question arrays for the new targets


def build(args):
    import pyarrow.parquet as pq

    import v35_build as b
    from connectomequest.neurosym import common as c
    from connectomequest.neurosym import constraints as k

    BENCH.mkdir(parents=True, exist_ok=False)
    old_manifest = json.loads((OLD / "manifest.json").read_text())
    vocabulary = json.loads((ROOT / b.SOURCE / "public/manifest.json").read_text())["vocabulary"]
    lookup = {name: i for i, name in enumerate(vocabulary) if i}
    alpha = b.PROTOCOL["provisional_alpha"]
    views = []
    for view, cfg in b.VIEWS.items():
        ids, matrix = b.load_graph(ROOT / cfg["root"])
        if not np.array_equal(ids, c.view_data(view)["ids"]):
            raise ValueError("node order differs from the cached graph")
        labels, _ = b.view_labels(ROOT, view, ids, lookup)
        full = k.full_mlp(view)
        provisional = None
        if view == "flywire":
            table = pq.read_table(ROOT / cfg["root"] / "public/nodes.parquet", columns=["cell_id", "provisional"]).to_pylist()
            hints = {int(r["cell_id"]): r["provisional"] for r in table}
            provisional = np.array([-1 if hints[x] is None else hints[x] for x in ids.tolist()], dtype=np.int32)
        indegree = np.diff(matrix.indptr)

        def priors_of(pre, codes):
            """(plus, unknown) per prior for the input cells ``pre`` and one partner set, as v35_build stores them."""
            q = np.asarray(full[pre])
            out = {"mlp_s0": (q[:, codes].astype(np.float64).sum(1).astype(np.float32), q[:, 0].copy())}
            if provisional is not None:
                mixed = q.copy()
                hinted = np.flatnonzero(provisional[pre] >= 0)
                mixed[hinted] *= np.float32(1 - alpha)
                mixed[hinted, provisional[pre][hinted]] += np.float32(alpha)
                out["provisional"] = (mixed[:, codes].astype(np.float64).sum(1).astype(np.float32), mixed[:, 0].copy())
            return out

        def collect(first, last, types):
            target_cell, target_type, target_offset = [], [], [0]
            pre_cell, pre_weight, pre_label = [], [], []
            fam_target, fam_kind, fam_source, fam_total, fam_code_offset, fam_codes = [], [], [], [], [0], []
            plus = {name: [] for name in PRIORS[view]}
            unknown = {name: [] for name in PRIORS[view]}
            for t in types:
                code = t["code"]
                cells = np.flatnonzero((labels == code) & (indegree >= b.PROTOCOL["minimum_partners"]))
                cells = sorted(cells.tolist(), key=lambda i: b.digest("cq-v35-endpoint", view, vocabulary[code], int(ids[i])))
                for i in cells[first:last]:
                    start, stop = matrix.indptr[i], matrix.indptr[i + 1]
                    pre, weight = matrix.indices[start:stop], np.asarray(matrix.data[start:stop], dtype=np.int64)
                    order = np.argsort(ids[pre], kind="stable")
                    pre, weight = pre[order], weight[order]
                    target = len(target_cell)
                    target_cell.append(int(ids[i]))
                    target_type.append(code)
                    target_offset.append(target_offset[-1] + len(pre))
                    pre_cell.append(ids[pre])
                    pre_weight.append(weight)
                    pre_label.append(labels[pre])
                    stored = None
                    for family in t["families"]:
                        fam_target.append(target)
                        fam_kind.append(0 if family["kind"] == "top1" else 1)
                        fam_source.append(family["source_weight"])
                        fam_total.append(t["source_total_weight"])
                        fam_codes.extend(family["codes"])
                        fam_code_offset.append(len(fam_codes))
                        stored = priors_of(pre, family["codes"])
                        for name in PRIORS[view]:
                            plus[name].append(stored[name][0])
                    for name in PRIORS[view]:
                        unknown[name].append(stored[name][1])
            arrays = dict(
                target_cell=np.asarray(target_cell, dtype=np.int64), target_type=np.asarray(target_type, dtype=np.int32),
                target_offset=np.asarray(target_offset, dtype=np.int64), pre_cell=np.concatenate(pre_cell),
                pre_weight=np.concatenate(pre_weight), pre_label=np.concatenate(pre_label).astype(np.int32),
                fam_target=np.asarray(fam_target, dtype=np.int32), fam_kind=np.asarray(fam_kind, dtype=np.int8),
                fam_source_weight=np.asarray(fam_source, dtype=np.int64), fam_source_total=np.asarray(fam_total, dtype=np.int64),
                fam_code_offset=np.asarray(fam_code_offset, dtype=np.int64), fam_codes=np.asarray(fam_codes, dtype=np.int32))
            for name in PRIORS[view]:
                arrays["plus_" + name] = np.concatenate(plus[name])
                arrays["unknown_" + name] = np.concatenate(unknown[name])
            return arrays

        # the same code must reproduce the stored arrays of the old targets (first 300 types of the roster)
        probe = old_manifest["types"][:300]
        again = collect(0, b.PROTOCOL["endpoints_per_type"], probe)
        with np.load(OLD / f"{view}.npz", allow_pickle=False) as z:
            n_targets, n_cells, n_families = len(again["target_cell"]), len(again["pre_cell"]), len(again["fam_target"])
            n_plus = len(again["plus_mlp_s0"])
            for key, value in again.items():
                stop = dict(target_cell=n_targets, target_type=n_targets, target_offset=n_targets + 1, pre_cell=n_cells, pre_weight=n_cells,
                            pre_label=n_cells, fam_target=n_families, fam_kind=n_families, fam_source_weight=n_families,
                            fam_source_total=n_families, fam_code_offset=n_families + 1, fam_codes=len(again["fam_codes"]))
                stop = stop.get(key, n_plus if key.startswith("plus_") else n_cells)
                if not np.array_equal(z[key][:stop], value):
                    raise AssertionError(f"{view}: {key} of the old targets is not reproduced")
        arrays = collect(FIRST, LAST, old_manifest["types"])
        with (BENCH / f"{view}.npz").open("xb") as stream:
            np.savez(stream, **arrays)
        split_of = {t["code"]: t["split"] for t in old_manifest["types"]}
        splits = np.array([split_of[int(x)] for x in arrays["target_type"]])
        views.append(dict(view=view, targets=len(arrays["target_cell"]), families=len(arrays["fam_target"]),
                          input_cells=int(arrays["target_offset"][-1]), priors=list(PRIORS[view]),
                          targets_by_split={s: int((splits == s).sum()) for s in ("development", "test")},
                          types_by_split={s: int(len(set(arrays["target_type"][splits == s].tolist()))) for s in ("development", "test")},
                          reproduced_old_targets=n_targets, sha256=sha256(BENCH / f"{view}.npz")))
        print(json.dumps(views[-1]), flush=True)
    manifest = dict(purpose="question arrays for the third and fourth neuron of every cell type in the order of the benchmark",
                    first=FIRST, last=LAST, benchmark_manifest_sha256=sha256(OLD / "manifest.json"), views=views)
    (BENCH / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- runs


def load(view):
    with np.load(BENCH / f"{view}.npz", allow_pickle=False) as z:
        d = {key: z[key] for key in z.files}
    lengths = np.diff(d["target_offset"])[d["fam_target"]]
    d["fam_plus_offset"] = np.concatenate(([0], np.cumsum(lengths)))
    return d


def split_families(view, d, split):
    manifest = json.loads((OLD / "manifest.json").read_text())
    split_of = {t["code"]: t["split"] for t in manifest["types"]}
    return [f for f in range(len(d["fam_target"])) if split_of[int(d["target_type"][d["fam_target"][f]])] == split]


def probabilities(plus, unknown):
    plus = np.clip(plus.astype(float), 0.0, 1.0)
    unknown = np.clip(unknown.astype(float), 0.0, 1.0 - plus)
    return np.column_stack((plus, 1.0 - plus - unknown, unknown))


def threshold_of(kind, num, den, total, source_weight, source_total):
    if kind == "retention":
        return e.threshold_goals(total, num * source_weight, den * source_total)[0]
    return e.threshold_goals(total, num, den)[0]


def run_family(family):
    from connectomequest.neurosym import constraints as k
    from connectomequest.neurosym import kernel as kk

    d, view = _D["data"], _D["view"]
    arms, rates = _D["arms"], _D["rates"]
    target = int(d["fam_target"][family])
    lo, hi = int(d["target_offset"][target]), int(d["target_offset"][target + 1])
    cells, weights, labels = d["pre_cell"][lo:hi], d["pre_weight"][lo:hi], d["pre_label"][lo:hi]
    n = hi - lo
    codes = d["fam_codes"][d["fam_code_offset"][family] : d["fam_code_offset"][family + 1]]
    outcomes = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
    total = int(weights.sum())
    start = int(d["fam_plus_offset"][family])
    priors = {p: probabilities(d["plus_" + p][start : start + n], d["unknown_" + p][lo:hi]) for p in PRIORS[view]}
    half = np.tile((0.5, 0.5, 0.0), (n, 1))
    kind = int(d["fam_kind"][family])
    synthetic = {level: sweep.synthetic_prior(outcomes, np.random.default_rng([41, family, i]).standard_normal(n), level,
                                              rates["member"][str(kind)], rates["unresolved"]) for i, level in enumerate(AUROCS)}
    if _D.get("kernel_target") != target:
        similarity = kk.cosine(view, k.rows_of(view, cells))
        _D["kernel_target"], _D["kernel"] = target, kk.kernel(similarity, SETTING["gamma"], SETTING["tau"])
    kern = _D["kernel"]
    rows = []
    for level, (question, num, den) in enumerate(LEVELS):
        threshold = threshold_of(question, num, den, total, int(d["fam_source_weight"][family]), int(d["fam_source_total"][family]))
        inst = e.threshold_instance(cells, weights, threshold)
        truth = e.full_information_status(inst, outcomes)
        bound = e.clairvoyant_cost(inst, outcomes)
        cost = np.zeros(len(arms), dtype=np.int32)
        stop_checks = np.full(len(PRIORS[view]), -1, dtype=np.int32)  # own stopping rule of information gain, per prior
        stop_decision = np.full(len(PRIORS[view]), -1, dtype=np.int8)

        def record(name, order):
            c, s = e.evaluate(inst, outcomes, order)
            if s != truth or (truth == "unresolved" and c != inst.size) or (bound is not None and c < bound):
                raise AssertionError(f"{name} contradicts the exact full-information status")
            cost[arms[name]] = c

        def adaptive(policy):
            return e.run_adaptive(inst, outcomes, policy)

        record("weight", e.weight_order(inst))
        for prior, p in priors.items():
            record(f"{prior}:member", e.positive_order(inst, p))
            record(f"{prior}:nonmember", e.negative_order(inst, p))
            record(f"{prior}:direction", e.direction_order(inst, p))
            record(f"{prior}:ag", adaptive(e.AdaptiveGreedy(inst, p)))
            record(f"{prior}:adg", adaptive(e.AdaptiveDualGreedy(inst, p)))
            record(f"{prior}:recal", adaptive(e.RecalibratedGreedy(inst, p)))
            record(f"{prior}:single-recal", adaptive(e.RecalibratedDirection(inst, p)))
            record(f"{prior}:prop", adaptive(kk.policy("recal", inst, p, kern, SETTING["lam"])))
            for g, name in enumerate(GENERIC):
                if truth == "unresolved":  # every policy checks all cells of an uncertifiable question
                    cost[arms[f"{prior}:{name}"]] = inst.size
                else:
                    generic = decision.policy(name, inst, p, np.random.default_rng([43, family, level, g]))
                    record(f"{prior}:{name}", adaptive(generic))
                    if name == "ig" and generic.unanimous is not None:
                        i = PRIORS[view].index(prior)
                        stop_checks[i], stop_decision[i] = generic.unanimous
        record("none:prop", adaptive(kk.policy("recal", inst, half, kern, SETTING["lam"])))
        for value, p in synthetic.items():
            record(f"synthetic{value}:ag", adaptive(e.AdaptiveGreedy(inst, p)))
            record(f"synthetic{value}:recal", adaptive(e.RecalibratedGreedy(inst, p)))
        rows.append(dict(family=family, level=level, size=inst.size, total=total, threshold=threshold,
                         member=int(weights[outcomes == e.MEMBER].sum()), unresolved=int(weights[outcomes == e.UNRESOLVED].sum()),
                         truth=STATUS[truth], clairvoyant=-1 if bound is None else bound, cost=cost,
                         stop_checks=stop_checks, stop_decision=stop_decision))
    return rows


def check_record():
    if not RECORD.exists():
        raise SystemExit("the held-out split needs the record")
    record = json.loads(RECORD.read_text())
    for group, base in (("code", ROOT), ("data", ROOT)):
        for path, digest in record[group].items():
            if sha256(base / path) != digest:
                raise SystemExit(f"{path} differs from the record")
    return record


def run(args):
    from connectomequest.neurosym import kernel as kk

    view, split = args.view, args.split
    manifest = json.loads((BENCH / "manifest.json").read_text())
    entry = next(v for v in manifest["views"] if v["view"] == view)
    if sha256(BENCH / f"{view}.npz") != entry["sha256"]:
        raise SystemExit("question arrays differ from their manifest")
    out = OUT / f"{view}-{split}.npz"
    if split == "test":
        if args.limit is not None:
            raise SystemExit("the held-out split runs completely")
        check_record()
        if out.exists():
            raise SystemExit("the held-out result exists")
        with (OUT / f"claim-{view}-test").open("x") as stream:
            stream.write(datetime.now(timezone.utc).isoformat() + "\n")
    d = load(view)
    names = arm_names(view)
    rates = json.loads((ROOT / f"outputs/v35/sweep/{view}-development.json").read_text())["rates"]
    _D.update(data=d, view=view, arms={n: i for i, n in enumerate(names)}, rates=rates)
    families = split_families(view, d, split)
    if args.limit is not None:
        families = families[: args.limit]
    kk.profiles(view)  # build before forking
    started = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        results = pool.map(run_family, families, chunksize=2)
    rows = [row for group in results for row in group]
    fields = ("family", "level", "size", "total", "threshold", "member", "unresolved", "truth", "clairvoyant")
    arrays = {key: np.asarray([r[key] for r in rows], dtype=np.int64) for key in fields}
    for key in ("cost", "stop_checks", "stop_decision"):
        arrays[key] = np.stack([r[key] for r in rows])
    path = out if args.limit is None else OUT / f"{view}-{split}-limit{args.limit}.npz"
    with path.open("wb" if split == "development" else "xb") as stream:
        np.savez(stream, **arrays)
    meta = dict(view=view, split=split, arms=names, levels=[list(x) for x in LEVELS], setting=SETTING, samples=decision.SAMPLES,
                aurocs=list(AUROCS), questions=len(rows), families=len(families), limit=args.limit,
                code={p: sha256(ROOT / p) for p in CODE}, arrays_sha256=entry["sha256"], wall_seconds=time.time() - started)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in meta.items() if key not in ("arms", "code")}))


# --------------------------------------------------------------------------- analysis


class Ratios:
    """Ratios of total checks over certifiable questions with a bootstrap over cell types shared by all measures."""

    def __init__(self, types, truth, rng):
        self.truth = truth
        self.cluster = np.unique(types, return_inverse=True)[1]
        self.n = int(self.cluster.max()) + 1
        draws = rng.integers(0, self.n, size=(BOOTSTRAP, self.n))
        self.counts = np.stack([np.bincount(row, minlength=self.n) for row in draws]).astype(float)

    def measure(self, cost, base, mask):
        """Ratio of totals of ``cost`` to ``base`` over ``mask``; all, yes, no and the mean of yes and no."""
        out, parts = {}, {}
        for name, answer in (("all", self.truth > 0), ("yes", self.truth == 1), ("no", self.truth == 2)):
            m = mask & answer
            sx = np.bincount(self.cluster[m], weights=cost[m].astype(float), minlength=self.n)
            sy = np.bincount(self.cluster[m], weights=base[m].astype(float), minlength=self.n)
            if sy.sum() <= 0:
                out[name] = None
                continue
            denominator = self.counts @ sy
            keep = denominator > 0
            boot = np.full(BOOTSTRAP, np.nan)
            boot[keep] = (self.counts @ sx)[keep] / denominator[keep]
            parts[name] = (sx.sum() / sy.sum(), boot)
            out[name] = dict(value=float(parts[name][0]), low=float(np.nanquantile(boot, 0.025)), high=float(np.nanquantile(boot, 0.975)))
        if "yes" in parts and "no" in parts:
            value, boot = (parts["yes"][0] + parts["no"][0]) / 2, (parts["yes"][1] + parts["no"][1]) / 2
            out["balanced"] = dict(value=float(value), low=float(np.nanquantile(boot, 0.025)), high=float(np.nanquantile(boot, 0.975)))
        return out


CONTRASTS = (("prop", "recal"), ("ag", "direction"), ("recal", "direction"), ("recal", "ag"), ("single-recal", "recal"), ("ec2", "ag"),
             ("ig", "ag"), ("ec2-recal", "recal"), ("ig-recal", "recal"), ("ig", "recal"), ("ec2", "recal"), ("adg", "ag"),
             ("single-recal", "direction"), ("member", "direction"), ("prop", "direction"), ("ec2", "ig"))
OTHER_CONTRASTS = (("none:prop", "mlp_s0:prop"), ("none:prop", "mlp_s0:recal")) + tuple((f"synthetic{a}:recal", f"synthetic{a}:ag") for a in AUROCS)


def analyse_view(view, split):
    d = load(view)
    meta = json.loads((OUT / f"{view}-{split}.json").read_text())
    with np.load(OUT / f"{view}-{split}.npz", allow_pickle=False) as z:
        r = {key: z[key] for key in z.files}
    arm = {n: i for i, n in enumerate(meta["arms"])}
    types = d["target_type"][d["fam_target"][r["family"]]]
    kinds = np.array([LEVELS[i][0] for i in r["level"]])
    ratios = Ratios(types, r["truth"], np.random.default_rng([SEED, sorted(PRIORS).index(view)]))
    weight = r["cost"][:, arm["weight"]]
    result = dict(types=int(len(set(types.tolist()))), targets=int(len(set(d["fam_target"][r["family"]].tolist()))),
                  questions=int(len(types)), wall_seconds=meta["wall_seconds"])
    for part in ("retention", "share"):
        mask = kinds == part
        truth = r["truth"][mask]
        certifiable = int((truth > 0).sum())
        out = dict(questions=int(mask.sum()), answers=dict(uncertifiable=int((truth == 0).sum()), yes=int((truth == 1).sum()), no=int((truth == 2).sum())),
                   share_uncertifiable=float((truth == 0).mean()), share_no_of_certifiable=float((truth == 2).sum() / certifiable),
                   share_no_checks=float(weight[mask & (r["truth"] == 2)].sum() / weight[mask & (r["truth"] > 0)].sum()),
                   weight_median=float(np.median(weight[mask & (r["truth"] > 0)])), ratio={}, contrast={}, regret={}, own_stop={})
        out["ratio"]["clairvoyant"] = ratios.measure(r["clairvoyant"], weight, mask)
        for name, column in arm.items():
            if name != "weight":
                out["ratio"][name] = ratios.measure(r["cost"][:, column], weight, mask)
        for prior in PRIORS[view]:
            for a, b in CONTRASTS:
                out["contrast"][f"{prior}:{a}/{prior}:{b}"] = ratios.measure(r["cost"][:, arm[f"{prior}:{a}"]], r["cost"][:, arm[f"{prior}:{b}"]], mask)
            totals = {q: {p: float(r["cost"][mask & (r["truth"] == code), arm[f"{prior}:{p}"]].sum()) for p in SIX} for q, code in (("yes", 1), ("no", 2))}
            regret = {}
            for p in SIX:
                per = {q: totals[q][p] / min(totals[q].values()) for q in totals if min(totals[q].values()) > 0}
                regret[p] = dict(per, worst=max(per.values()))
            regret["others_best_worst"] = min(regret[p]["worst"] for p in SIX if p != "recal")
            out["regret"][prior] = regret
            # the stopping rule of the generic policies (all sampled realizations agree), judged on certifiable questions
            i = PRIORS[view].index(prior)
            certifiable_mask = mask & (r["truth"] > 0)
            early = certifiable_mask & (r["stop_decision"][:, i] >= 0) & (r["stop_checks"][:, i] < r["cost"][:, arm[f"{prior}:ig"]])
            wrong = early & (r["stop_decision"][:, i] != r["truth"])
            out["own_stop"][prior] = dict(before_certificate=float(early.sum() / certifiable_mask.sum()),
                                          wrong_decision=float(wrong.sum() / certifiable_mask.sum()),
                                          at_first_check=float((early & (r["stop_checks"][:, i] == 0)).sum() / certifiable_mask.sum()))
        for a, b in OTHER_CONTRASTS:
            out["contrast"][f"{a}/{b}"] = ratios.measure(r["cost"][:, arm[a]], r["cost"][:, arm[b]], mask)
        out["by_level"] = {}
        for index, (question, num, den) in enumerate(LEVELS):
            if question != part:
                continue
            level_mask = r["level"] == index
            truth = r["truth"][level_mask]
            out["by_level"][f"{num}/{den}"] = dict(
                answers=dict(uncertifiable=int((truth == 0).sum()), yes=int((truth == 1).sum()), no=int((truth == 2).sum())),
                ratio={name: ratios.measure(r["cost"][:, column], weight, level_mask)["all"] for name, column in arm.items() if name != "weight"})
        result[part] = out
    return result


def lookup(result, path):
    x = result
    for part in path.split("|"):
        x = x[part]
    return x


def evaluate(expectations, result):
    verdicts = []
    for x in expectations:
        value = lookup(result, x["quantity"])
        holds = (x.get("min") is None or value >= x["min"]) and (x.get("max") is None or value <= x["max"])
        verdicts.append(dict(x, value=value, holds=bool(holds)))
    return verdicts


def analyze(args):
    result = {view: analyse_view(view, args.split) for view in PRIORS}
    (OUT / f"analysis-{args.split}.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    if args.split == "test":
        record = check_record()
        verdicts = evaluate(record["expectations"], result)
        (OUT / "verdicts-test.json").write_text(json.dumps(verdicts, indent=1) + "\n")
        print(f"{sum(v['holds'] for v in verdicts)} of {len(verdicts)} recorded expectations hold")
        for v in verdicts:
            if not v["holds"]:
                print("  fails:", v["text"], "->", round(v["value"], 4))
    else:
        print(json.dumps(summary(result), indent=1))


def summary(result):
    out = {}
    for view, r in result.items():
        for part in ("retention", "share"):
            p = r[part]
            out[f"{view}/{part}/answers"] = dict(p["answers"], no_of_certifiable=round(p["share_no_of_certifiable"], 3))
            for name, m in p["ratio"].items():
                out[f"{view}/{part}/{name}"] = {q: round(v["value"], 3) for q, v in m.items() if v}
            for name, m in p["contrast"].items():
                out[f"{view}/{part}/{name}"] = {q: [round(v[x], 3) for x in ("value", "low", "high")] for q, v in m.items() if v}
            for prior, rules in p["regret"].items():
                out[f"{view}/{part}/regret/{prior}"] = {q: round(v["worst"], 3) if isinstance(v, dict) else round(v, 3) for q, v in rules.items()}
            out[f"{view}/{part}/own_stop"] = {prior: {q: round(v, 3) for q, v in x.items()} for prior, x in p["own_stop"].items()}
    return out


def write_record(args):
    if RECORD.exists():
        raise SystemExit("the record exists")
    expectations = json.loads((OUT / "expectations.json").read_text())
    development = ("malecns-development.npz", "flywire-development.npz", "analysis-development.json")
    record = dict(written_utc=datetime.now(timezone.utc).isoformat(), purpose=__doc__.split("\n\n")[0],
                  code={p: sha256(ROOT / p) for p in CODE}, data={p: sha256(ROOT / p) for p in DATA},
                  development={p: sha256(OUT / p) for p in development},
                  expectations=expectations["expectations"], notes=expectations.get("notes", []))
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    RECORD.chmod(0o444)
    print(f"recorded {len(record['expectations'])} expectations")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--view", required=True, choices=tuple(PRIORS))
    r.add_argument("--split", required=True, choices=("development", "test"))
    r.add_argument("--workers", type=int, default=28)
    r.add_argument("--limit", type=int)
    a = sub.add_parser("analyze")
    a.add_argument("--split", required=True, choices=("development", "test"))
    sub.add_parser("record")
    args = parser.parse_args()
    dict(build=build, run=run, analyze=analyze, record=write_record)[args.command](args)


if __name__ == "__main__":
    main()
