#!/usr/bin/env python3
"""Two-way coupling (B4) on the held-out split, under the fourth post hoc record.

B4 is Adaptive Greedy / Recalibrated Greedy on the frozen mlp_s0 prior whose beliefs
about unpaid cells follow similar paid cells (``connectomequest.neurosym.kernel``):
every paid response is spread through ``K(j, k) = cos(j, k) ** gamma`` (``cos >= tau``)
of the connectivity profiles in the view graph, with pseudo-count weight ``lam``.

Stages
------
``pick``     development only: per view, the B4 setting (lam, gamma, tau, base) with the
             smallest total cost relative to the better of frozen mlp_s0:recal / mlp_s0:ag
             on all certifiable development sessions. It combines the two cross-fitting
             halves of round 2 exactly (sum of half costs), so nothing is recomputed.
``run``      one split (``development`` first; ``test`` needs the fourth record, whose code
             hashes must match, runs completely and never overwrites): B4 with both bases
             and four controls with the same parameters - profiles without the target's
             column, a permuted kernel (seed 0), the kernel without left-right pairs, and an
             ORACLE same-type kernel built from reference labels (a reference, no claim).
``analyze``  ratios with type-cluster bootstrap intervals, shares of B4's gain kept by the
             controls and, on the held-out split, the record's expectations.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from hashlib import sha256
from itertools import product

import numpy as np

from connectomequest.annotation_v32.runtime import hash_file
from connectomequest.neurosym import common as c
from connectomequest.neurosym import constraints as k
from connectomequest.neurosym import harness as h
from connectomequest.neurosym import kernel as kk

ROOT = c.ROOT
ROUND2 = ROOT / "outputs/neurosym/B/round2"
OUT = ROOT / "outputs/neurosym/B/heldout"
RECORD = ROOT / "outputs/v35/posthoc-record-4.json"
VIEWS = ("flywire", "malecns")
GRID = dict(lam=(0.5, 2.0, 8.0), gamma=(1.0, 3.0), tau=(0.3, 0.5, 0.7))
BASES = ("ag", "recal")
CODE_FILES = (
    "src/connectomequest/annotation_v35/engine.py",
    "src/connectomequest/neurosym/common.py",
    "src/connectomequest/neurosym/harness.py",
    "src/connectomequest/neurosym/constraints.py",
    "src/connectomequest/neurosym/kernel.py",
    "scripts/v35_build.py",
    "scripts/neurosym_heldout.py",
)
WORKERS = int(os.environ.get("NEUROSYM_WORKERS", str(min(32, len(os.sched_getaffinity(0))))))
_RUN = dict(view=None, arms={})
_CACHE = {}


def code_identity():
    return {name: hash_file(ROOT / name) for name in CODE_FILES}


def better_baseline(view, families, split):
    mask = h.truth(view, families, split) > 0
    base = {b: h.baseline(view, families, f"mlp_s0:{b}", split) for b in BASES}
    name = min(base, key=lambda b: (base[b][mask].sum(), b))
    return name, base, mask


# --------------------------------------------------------------------------- pick (development only)


def pick():
    out = {}
    for view in VIEWS:
        parts = []
        for half in (0, 1):
            fit = json.loads((ROUND2 / f"fit-{view}-h{half}.json").read_text())
            families = c.development_families(view, half)
            if fit["families"] != len(families):
                raise AssertionError("round 2 fit does not match the development half")
            name, base, mask = better_baseline(view, families, "development")
            if name != fit["best_baseline"]:
                raise AssertionError("baseline differs from the round 2 fit")
            parts.append((fit, float(base[name][mask].sum()), name))
        if len({p[2] for p in parts}) != 1:
            raise AssertionError("the better baseline differs between halves; halves cannot be combined exactly")
        name_all, base_all, mask_all = better_baseline(view, c.development_families(view), "development")
        if name_all != parts[0][2]:
            raise AssertionError("the better baseline on all development sessions differs from the halves")
        rows = {}
        for fit, denominator, _ in parts:
            for row in fit["rows"]:
                if row["variant"] != "B4":
                    continue
                cfg = row["config"]["b4"]
                key = json.dumps(cfg, sort_keys=True)
                for b in BASES:
                    num, den = rows.get((key, b), (0.0, 0.0))
                    rows[(key, b)] = (num + row[b] * denominator, den + denominator)
        expected = len(list(product(*GRID.values()))) * len(BASES)
        if len(rows) != expected or any(den != sum(p[1] for p in parts) for _, den in rows.values()):
            raise AssertionError("incomplete round 2 grid")
        ranked = sorted(((num / den, b, key) for (key, b), (num, den) in rows.items()), key=lambda x: (round(x[0], 9), x[1], x[2]))
        value, b, key = ranked[0]
        out[view] = dict(config=json.loads(key), base=b, baseline=name_all, development_ratio=value,
                         runner_up=dict(config=json.loads(ranked[1][2]), base=ranked[1][1], development_ratio=ranked[1][0]))
    OUT.mkdir(parents=True, exist_ok=True)
    payload = dict(rule="B4 setting and base with the smallest total cost relative to the better frozen mlp_s0 policy on all certifiable "
                        "development sessions; the two round 2 halves are combined exactly (costs = ratio x the half's baseline total)",
                   inputs={f"fit-{v}-h{hf}.json": hash_file(ROUND2 / f"fit-{v}-h{hf}.json") for v in VIEWS for hf in (0, 1)}, picks=out)
    (OUT / "picks.json").write_text(json.dumps(payload, indent=2))
    return payload


# --------------------------------------------------------------------------- run


def arm_configs(p):
    cfg = dict(lam=p["config"]["lam"], gamma=p["config"]["gamma"], tau=p["config"]["tau"])
    return {
        "b4": dict(kind="cos", **cfg),
        "no_target": dict(kind="cos", no_target=True, **cfg),
        "shuffled": dict(kind="cos", shuffle=0, **cfg),
        "ipsilateral": dict(kind="cos", ipsilateral_only=True, **cfg),
        "oracle": dict(kind="oracle", lam=cfg["lam"]),
    }


def arms(fam, inst, outcomes, rho_index):
    view = _RUN["view"]
    if _CACHE.get("target") != fam["target"]:
        _CACHE.clear()
        _CACHE["target"] = fam["target"]
    prior = fam["priors"]["mlp_s0"]
    out = {}
    for name, cfg in _RUN["arms"].items():
        if name not in _CACHE:
            _CACHE[name] = kk.control_kernel(view, fam, cfg)
        for base in BASES:
            out[f"{name}:{base}"] = kk.e.run_adaptive(inst, outcomes, kk.policy(base, inst, prior, _CACHE[name], cfg["lam"]))
    return out


def run(view, split, record):
    picks = json.loads((OUT / "picks.json").read_text())
    code = code_identity()
    if split == "test":
        if record is None or not record.exists():
            raise ValueError("the held-out split needs the fourth post hoc record")
        rec = json.loads(record.read_text())
        if rec["code"] != code or rec["picks_sha256"] != hash_file(OUT / "picks.json"):
            raise ValueError("code or picks changed after the fourth post hoc record")
    out = OUT / f"{view}-{split}.npz"
    if out.exists():
        raise FileExistsError(out)
    families = c.split_families(view, split)
    _RUN.update(view=view, arms=arm_configs(picks["picks"][view]))
    kk.profiles(view), k.rows_of(view, c.benchmark(view)["pre_cell"][:1]), c.view_data(view)  # load before forking
    started = time.time()
    costs = h.run(view, families, arms, workers=WORKERS, split=split)
    rows = h.rows(view, families, split)
    d = h.frozen(view, split)
    arrays = dict(family=d["family"][rows], rho=d["rho"][rows], truth=d["truth"][rows], **{n.replace(":", "__"): v for n, v in costs.items()})
    for b in BASES:
        arrays[f"frozen_{b}"] = h.baseline(view, families, f"mlp_s0:{b}", split)
    arrays["frozen_weight"] = h.baseline(view, families, "weight", split)
    tmp = out.with_name(f"{out.stem}.{os.getpid()}.tmp.npz")
    with tmp.open("xb") as stream:
        np.savez(stream, **arrays)
    tmp.replace(out)
    meta = dict(view=view, split=split, families=len(families), sessions=len(rows), arms=sorted(costs), code=code,
                picks=picks["picks"][view], picks_sha256=hash_file(OUT / "picks.json"), seconds=round(time.time() - started, 1),
                result_sha256=hash_file(out))
    (OUT / f"{view}-{split}.json").write_text(json.dumps(meta, indent=2))
    return meta


# --------------------------------------------------------------------------- analyze


def retained(index, counts, k_, x, y, z, mask):
    """Share of B4's gain kept by a control: (1 - x/z) / (1 - y/z), paired type-cluster bootstrap."""
    s = [np.bincount(index[mask], weights=np.asarray(v, float)[mask], minlength=k_) for v in (x, y, z)]
    point = (1 - s[0].sum() / s[2].sum()) / (1 - s[1].sum() / s[2].sum())
    b = [counts @ v for v in s]
    ok = (b[2] > 0) & (b[1] != b[2])
    boot = (1 - b[0][ok] / b[2][ok]) / (1 - b[1][ok] / b[2][ok])
    return dict(value=float(point), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))


def analyze(split):
    picks = json.loads((OUT / "picks.json").read_text())["picks"]
    result = dict(split=split, views={})
    for view in VIEWS:
        with np.load(OUT / f"{view}-{split}.npz", allow_pickle=False) as z:
            a = {key: z[key] for key in z.files}
        families = c.split_families(view, split)
        if not np.array_equal(a["family"], h.frozen(view, split)["family"][h.rows(view, families, split)]):
            raise AssertionError("stored sessions differ from the frozen run")
        mask = a["truth"] > 0
        better = min(BASES, key=lambda b: (a[f"frozen_{b}"][mask].sum(), b))
        ref = a[f"frozen_{better}"]
        base = picks[view]["base"]
        ratio = lambda x, y: h.ratio(view, families, x, y, mask, split)
        index, counts, k_ = h._clusters(view, tuple(int(f) for f in families))
        v = dict(families=len(families), sessions=int(len(mask)), certifiable=int(mask.sum()), better_frozen=better, pick=picks[view],
                 frozen_to_weight={b: ratio(a[f"frozen_{b}"], a["frozen_weight"]) for b in BASES})
        v["b4"] = {b: dict(to_better=ratio(a[f"b4__{b}"], ref), to_same_base=ratio(a[f"b4__{b}"], a[f"frozen_{b}"]),
                           to_weight=ratio(a[f"b4__{b}"], a["frozen_weight"])) for b in BASES}
        v["controls"] = {}
        for name in ("no_target", "shuffled", "ipsilateral", "oracle"):
            x = a[f"{name}__{base}"]
            v["controls"][name] = dict(to_better=ratio(x, ref), retained=retained(index, counts, k_, x, a[f"b4__{base}"], ref, mask))
        v["paired_b4_vs_better"] = dict(better=float((a[f"b4__{base}"][mask] < ref[mask]).mean()), tied=float((a[f"b4__{base}"][mask] == ref[mask]).mean()),
                                         worse=float((a[f"b4__{base}"][mask] > ref[mask]).mean()))
        result["views"][view] = v
    if split == "test":
        result["expectations"] = expectations(result["views"])
    (OUT / f"analysis-{split}.json").write_text(json.dumps(result, indent=2))
    return result


def expectations(views):
    checks = {}
    for view, v in views.items():
        base = v["pick"]["base"]
        other = "ag" if base == "recal" else "recal"
        b4 = v["b4"][base]["to_better"]
        ctl = v["controls"]
        checks[f"K1:{view}"] = dict(rule="B4 (picked base) / better frozen <= 0.92 with CI high < 1", value=b4, holds=b4["value"] <= 0.92 and b4["high"] < 1)
        o = v["b4"][other]["to_same_base"]
        checks[f"K2:{view}"] = dict(rule="B4 with the other base / the same frozen base: CI high < 1", value=o, holds=o["high"] < 1)
        s = ctl["shuffled"]["to_better"]
        checks[f"K3:{view}"] = dict(rule="permuted kernel / better frozen >= 1.00", value=s, holds=s["value"] >= 1.0)
        n = ctl["no_target"]["retained"]
        checks[f"K4:{view}"] = dict(rule="profiles without the target column keep >= 2/3 of B4's gain", value=n, holds=n["value"] >= 2 / 3)
        i = ctl["ipsilateral"]["retained"]
        checks[f"K5:{view}"] = dict(rule="kernel without left-right pairs keeps between 0.70 and 1.00 of B4's gain", value=i, holds=0.70 <= i["value"] <= 1.0)
        r = ctl["oracle"]["retained"]
        checks[f"K6:{view}"] = dict(rule="ORACLE same-type kernel keeps >= 1.00 of B4's gain (reference)", value=r, holds=r["value"] >= 1.0)
    checks["summary"] = dict(held=sum(x["holds"] for key, x in checks.items() if key != "summary"), of=len(checks))
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("pick", "run", "analyze"))
    p.add_argument("--view", nargs="+", default=list(VIEWS))
    p.add_argument("--split", choices=("development", "test"), default="development")
    p.add_argument("--record", type=lambda s: __import__("pathlib").Path(s))
    a = p.parse_args()
    if a.stage == "pick":
        print(json.dumps(pick()["picks"], indent=1))
    elif a.stage == "run":
        for view in a.view:
            print(json.dumps({k_: v for k_, v in run(view, a.split, a.record).items() if k_ != "code"}, indent=1))
    else:
        r = analyze(a.split)
        for view, v in r["views"].items():
            print(view, {b: round(x["to_better"]["value"], 3) for b, x in v["b4"].items()},
                  {n: (round(x["to_better"]["value"], 3), round(x["retained"]["value"], 2)) for n, x in v["controls"].items()})
        if "expectations" in r:
            print(r["expectations"]["summary"])


if __name__ == "__main__":
    main()
