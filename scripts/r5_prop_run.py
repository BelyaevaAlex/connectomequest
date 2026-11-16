#!/usr/bin/env python3
"""Round 5, workstream ``prop``: answer propagation beyond the weak predictor.

Answer propagation at the recorded setting (lam = 8, gamma = 3, tau = 0.3) with
* the earlier-release prior (FlyWire ``provisional``) on AG and recalibrated AG,
* the no-prediction priors (``const_half``: p+ = 1/2; ``const_share``: p+ = hemibrain
  share), on AG and recalibrated AG,
* member-first and direction-fixed re-sorted after every check by propagated
  beliefs (active-search-like), with the topology (``mlp_s0``) and earlier-release
  priors,
* external priors (``outputs/round5/priors``) on recalibrated AG, when present.

Arms that exist already are read, never recomputed: frozen runs (``outputs/v35/runs``,
rules without propagation), the first post hoc round (``outputs/v35/posthoc``,
no-prediction priors without propagation) and the fourth (``outputs/neurosym/B/heldout``,
``mlp_s0`` with propagation on AG and recalibrated AG).

Stages
------
``run``             compute the missing arms of one split; resumable (chunks under
                    ``outputs/round5/prop/parts``); never overwrites a result.
``analyze``         ratios to weight order and to the same rule without propagation,
                    paired per-question better / tied / worse shares, contrasts; 95%
                    type-cluster bootstrap intervals.
``record-entries``  the files (with sha256) that ``outputs/round5/record.json`` must list
                    before this script accepts ``--split test``.

The held-out split (``--split test``) is refused unless ``outputs/round5/record.json``
lists the sha256 of this script, of every round5 module it imports and of every
round5 data file it reads, and they match.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import re
import sys
import time
from hashlib import sha256
from pathlib import Path

import numpy as np

from connectomequest.annotation_v32.runtime import hash_file
from connectomequest.neurosym import common as c
from connectomequest.round5 import prop as P

ROOT = P.ROOT
OUT = P.OUT
RECORD = ROOT / "outputs/round5/record.json"
RUNNER = Path(__file__).resolve()
FROZEN_CODE = (
    "src/connectomequest/annotation_v35/engine.py",
    "src/connectomequest/neurosym/common.py",
    "src/connectomequest/neurosym/harness.py",
    "src/connectomequest/neurosym/constraints.py",
    "src/connectomequest/neurosym/kernel.py",
    "scripts/v35_build.py",
)
VIEWS = ("flywire", "malecns")
REAL = ("mlp_s0", "provisional")
DEFAULT_WORKERS = 6  # local cap shared by five workstreams


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def rel(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------- protocol


def round5_modules():
    """Relative paths of every imported module that lives under src/connectomequest/round5."""
    base = (ROOT / "src/connectomequest/round5").resolve()
    out = set()
    for module in list(sys.modules.values()):
        f = getattr(module, "__file__", None)
        if f and Path(f).resolve().is_relative_to(base):
            out.add(rel(f))
    return sorted(out)


def required_files(sources):
    """Runner, imported round5 modules and round5 data files that a held-out run depends on."""
    files = {rel(RUNNER), *round5_modules()}
    for src in sources:
        files |= {rel(p) for p in src.files.values() if rel(p).startswith(("outputs/round5/", "inputs/round5/"))}
    return {name: hash_file(ROOT / name) for name in sorted(files)}


_HEX = re.compile(r"[0-9a-f]{64}")


def _listed(out, key, value):
    if "/" not in key and "." not in key:  # not a file path (e.g. "result_sha256")
        return
    name = rel(ROOT / key) if not key.startswith("/") else rel(key)
    out.setdefault(name, set()).add(value)


def listed_hashes(obj, out=None):
    """{path: set of sha256} from every ``"path": "<sha256>"`` pair and ``{"path": .., "sha256": ..}`` entry of the record."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        if isinstance(obj.get("path"), str) and isinstance(obj.get("sha256"), str) and _HEX.fullmatch(obj["sha256"]):
            _listed(out, obj["path"], obj["sha256"])
        for key, value in obj.items():
            if isinstance(key, str) and isinstance(value, str) and _HEX.fullmatch(value):
                _listed(out, key, value)
            else:
                listed_hashes(value, out)
    elif isinstance(obj, list):
        for value in obj:
            listed_hashes(value, out)
    return out


def require_record(split, sources):
    if split != "test":
        return None
    if not RECORD.exists():
        raise SystemExit(f"refusing --split test: {rel(RECORD)} does not exist (record the design first)")
    listed = listed_hashes(json.loads(RECORD.read_text()))
    need = required_files(sources)
    bad = [f"{name} ({'not listed' if name not in listed else 'hash differs'})" for name, sha in need.items() if sha not in listed.get(name, ())]
    if bad:
        raise SystemExit("refusing --split test: the record does not match\n  " + "\n  ".join(bad))
    return need


# --------------------------------------------------------------------------- run


def result_files(view, split):
    """Complete result pairs (an .npz without its .json is a run that did not finish writing)."""
    return sorted(p for p in OUT.glob(f"{view}-{split}-*.npz") if ".tmp" not in p.name and not p.name.startswith(".") and p.with_suffix(".json").exists())


def write_new(path, writer):
    """Write through a temporary file and link it into place; never overwrite (FileExistsError)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as stream:
            writer(stream)
        os.link(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


_HASHES = {}


def current_hash(path):
    path = Path(path)
    key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    if key not in _HASHES:
        _HASHES[key] = hash_file(path)
    return _HASHES[key]


def stale_arms(meta):
    """Arms of a result file whose external prior has changed since: the sha256 recorded for the prior file
    differs from the file now at that path (or the file is gone). Such arms are ignored, never deleted."""
    out = set()
    for name in meta["arms"]:
        arm = P.Arm.parse(name)
        rec = meta.get("data", {}).get(f"prior:{arm.prior}")
        if rec is None:
            continue
        path = ROOT / rec["path"]
        if not path.exists() or current_hash(path) != rec["sha256"]:
            out.add(name)
    return out


def computed_arms(view, split, quiet=False):
    """{arm: cost} from earlier runs of this workstream (aligned with harness rows); arms on superseded priors are skipped."""
    out = {}
    families = P.split_families(P.source(view), split)
    for path in result_files(view, split):
        meta = json.loads(path.with_suffix(".json").read_text())
        stale = stale_arms(meta)
        if stale and not quiet:
            log(f"{rel(path)}: ignoring {len(stale)} arms on superseded priors")
        z = P._load(path)
        rows = P._align(view, split, families, z["family"], z["rho"])
        for col, name in enumerate(meta["arms"]):
            if name in stale:
                continue
            if name in out and not np.array_equal(out[name], z["cost"][rows, col]):
                raise AssertionError(f"{name} differs between result files")
            out[name] = z["cost"][rows, col].astype(np.int64)
    return out


def requested_arms(view, src, priors, policies):
    available = set(src.priors())
    if priors is None:
        priors = [p for p in ("mlp_s0", "provisional", "const_half", "const_share") if p in available]
    priors = [x for p in priors for x in (sorted(src.external) if p == "external" else [p])]  # "external": every prior of outputs/round5/priors
    arms = []
    for prior in priors:
        if prior not in available:
            log(f"{view}: prior {prior} does not exist here; skipped")
            continue
        pols = policies or (("ag", "recal") if prior in P.NO_PREDICTION else ("recal",) if prior not in REAL else P.POLICIES)
        arms += [P.Arm(prior, pol, True) for pol in pols]
    return arms + [P.WEIGHT]  # weight order is the denominator of every ratio


def run(view, split, priors, policies, name, workers, chunk, dry):
    src = P.source(view)
    need = require_record(split, [src])
    out = OUT / f"{view}-{split}-{name}.npz"
    if split == "test":  # claim before any held-out result is read (--dry-run is refused on the held-out split)
        if out.with_suffix(".json").exists():
            raise FileExistsError(out)
        from connectomequest.round5 import heldout
        heldout.claim(RUNNER, "prop", f"{view}-{name}", out)
    have = set(P.existing_arms(view, split)) | set(computed_arms(view, split))
    wanted = []
    for arm in requested_arms(view, src, priors, policies):
        for a in (arm, arm.static) if arm.propagate else (arm,):  # the rule without propagation is needed for every comparison
            if a.name not in have and a not in wanted:
                wanted.append(a)
    skipped = [a.name for arm in requested_arms(view, src, priors, policies) for a in {arm, arm.static} if a.name in have]
    log(f"{view}/{split}: reuse {sorted(set(skipped))}; compute {[a.name for a in wanted]}")
    if not wanted or dry:
        return None
    if out.with_suffix(".json").exists():
        raise FileExistsError(out)
    families = P.split_families(src, split)
    started = time.time()
    code = {rel(RUNNER): hash_file(RUNNER), **{m: hash_file(ROOT / m) for m in round5_modules()}}  # the code that runs, hashed before it runs
    frozen_code = {f: hash_file(ROOT / f) for f in FROZEN_CODE}
    data = {role: dict(path=rel(p), sha256=hash_file(p)) for role, p in src.files.items()}
    signature = dict(view=view, split=split, setting=P.SETTING, code={**code, **frozen_code}, data={role: x["sha256"] for role, x in data.items()})
    res = P.run_sessions(src, families, wanted, OUT / "parts" / f"{view}-{split}-{name}", workers=workers, chunk=chunk, signature=signature, log=log)
    order = P._align(view, split, families, res["family"], res["rho"])
    res = {key: v[order] for key, v in res.items()}
    ref = P.reference(view, split)
    if ref is None:
        log(f"{view}/{split}: no stored run of this split to check sessions against")
    for key, value in (ref or {}).items():
        if not np.array_equal(res[key], value):
            raise AssertionError(f"{key} differs from the stored run of the split")
    if out.exists():  # an unfinished earlier attempt left the .npz without its .json
        z = P._load(out)
        if set(z) != set(res) or not all(np.array_equal(z[k], res[k]) for k in res):
            raise FileExistsError(f"{out} exists with other content")
    else:
        write_new(out, lambda stream: np.savez(stream, **res))
    meta = dict(view=view, split=split, name=name, families=len(families), sessions=int(len(order)), arms=[a.name for a in wanted], setting=P.SETTING,
                code={**code, **frozen_code}, data=data, record=need, workers=workers,
                seconds=round(time.time() - started, 1), result_sha256=hash_file(out))
    write_new(out.with_suffix(".json"), lambda stream: stream.write(json.dumps(meta, indent=2).encode()))
    log(f"{view}/{split}: wrote {rel(out)} in {meta['seconds']} s")
    return meta


# --------------------------------------------------------------------------- analyze


def contrasts(arms):
    """(label, numerator, denominator) for every comparison whose arms exist."""
    out = []
    for prior in P.NO_PREDICTION:
        for pol in ("ag", "recal"):
            out.append((f"need the MLP? {prior}:{pol}+prop / mlp_s0:{pol}+prop", f"{prior}:{pol}+prop", f"mlp_s0:{pol}+prop"))
            out.append((f"need the MLP? {prior}:{pol}+prop / mlp_s0:{pol}", f"{prior}:{pol}+prop", f"mlp_s0:{pol}"))
    out.append(("earlier release: provisional:recal+prop / provisional:direction", "provisional:recal+prop", "provisional:direction"))
    out.append(("earlier release: provisional:ag+prop / provisional:direction", "provisional:ag+prop", "provisional:direction"))
    out.append(("earlier release: provisional:recal+prop / mlp_s0:recal+prop", "provisional:recal+prop", "mlp_s0:recal+prop"))
    for prior in REAL:
        for pol in ("member", "direction"):
            out.append((f"active search: {prior}:{pol}+prop / {prior}:recal+prop", f"{prior}:{pol}+prop", f"{prior}:recal+prop"))
            out.append((f"active search: {prior}:{pol}+prop / {prior}:recal", f"{prior}:{pol}+prop", f"{prior}:recal"))
    for name in sorted(arms):
        arm = P.Arm.parse(name) if ":" in name and name.split(":", 1)[1].replace("+prop", "") in P.POLICIES else None
        if arm and arm.propagate and arm.prior not in (*REAL, *P.NO_PREDICTION) and arm.policy == "recal":
            out.append((f"external prior: {name} / mlp_s0:recal+prop", name, "mlp_s0:recal+prop"))
            out.append((f"external prior: {name} / mlp_s0:recal", name, "mlp_s0:recal"))
    return [x for x in out if x[1] in arms and x[2] in arms]


def analyze_view(view, split):
    families = P.split_families(P.source(view), split)
    arms = P.existing_arms(view, split)
    ours = computed_arms(view, split)
    clash = [n for n in ours if n in arms and not np.array_equal(ours[n], arms[n])]
    if clash:
        raise AssertionError(f"recomputed arms differ from stored ones: {clash}")
    arms.update(ours)
    ref = P.reference(view, split)
    if ref is None:  # sessions of this workstream's own result files (all are aligned and carry the truth)
        if not result_files(view, split):
            raise ValueError(f"{view}/{split}: neither a stored run nor a result of this workstream")
        z = P._load(result_files(view, split)[0])
        rows = P._align(view, split, families, z["family"], z["rho"])
        ref = {key: z[key][rows] for key in ("truth", "clairvoyant")}
    arms.setdefault("clairvoyant", ref["clairvoyant"])
    truth = ref["truth"]
    mask = truth > 0
    cl = P.Clusters(P.session_types(view, families))
    weight = arms["weight"]
    size = P.session_sizes(view, families)
    strata = {"<500 input cells": size < 500, ">=500 input cells": size >= 500}
    v = dict(families=len(families), sessions=int(len(mask)), certifiable=int(mask.sum()), setting=P.SETTING,
             no_prediction_equals_weight={n: bool(np.array_equal(arms[n], weight)) for n in arms if n.split(":")[0] in P.NO_PREDICTION and "+prop" not in n},
             strata={label: dict(certifiable=int((mask & m).sum()), weight_checks_share=float(weight[mask & m].sum() / weight[mask].sum())) for label, m in strata.items()},
             propagation={}, contrasts={})
    for name in sorted(n for n in arms if n.endswith("+prop")):
        arm = P.Arm.parse(name)
        static = arms.get(arm.static.name)
        x = arms[name]
        row = dict(to_weight=cl.ratio(x, weight, mask), paired_vs_weight=cl.shares(x, weight, mask), source="this workstream" if name in ours else "earlier run")
        if static is not None:
            row.update(static_to_weight=cl.ratio(static, weight, mask), to_static=cl.ratio(x, static, mask), paired_vs_static=cl.shares(x, static, mask),
                       gap_closed=cl.gap_closed(x, static, arms["clairvoyant"], mask),
                       to_static_by_size={label: cl.ratio(x, static, mask & m) for label, m in strata.items() if (mask & m).any()},
                       to_weight_by_size={label: cl.ratio(x, weight, mask & m) for label, m in strata.items() if (mask & m).any()})
        v["propagation"][name] = row
    for label, num, den in contrasts(arms):
        v["contrasts"][label] = dict(ratio=cl.ratio(arms[num], arms[den], mask), paired=cl.shares(arms[num], arms[den], mask),
                                     by_size={s_: cl.ratio(arms[num], arms[den], mask & m) for s_, m in strata.items() if (mask & m).any()})
    v["clairvoyant_to_weight"] = cl.ratio(arms["clairvoyant"], weight, mask)
    return v


def analyze(split, views):
    sources = [P.source(v) for v in views]
    need = require_record(split, sources)
    inputs = {rel(p): hash_file(p) for v in views for p in [*result_files(v, split), *P.existing_files(v, split)]}
    ignored = {rel(p): sorted(stale_arms(json.loads(p.with_suffix(".json").read_text()))) for v in views for p in result_files(v, split)}
    tag = sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:12]
    path = OUT / f"analysis-{split}-{tag}.json"
    if split == "test":  # claim before any held-out answer is analysed
        from connectomequest.round5 import heldout
        heldout.claim(RUNNER, "prop", "analyze", path)
    result = dict(split=split, inputs=inputs, ignored_superseded_arms={k: x for k, x in ignored.items() if x}, record=need,
                  views={v: analyze_view(v, split) for v in views})
    text = json.dumps(result, indent=2)
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"{path} exists with other content")
    else:
        write_new(path, lambda stream: stream.write(text.encode()))
    log(f"wrote {rel(path)}")
    return result


def fmt(x):
    return f"{x['value']:.3f} [{x['low']:.3f}, {x['high']:.3f}]"


def show(result):
    for view, v in result["views"].items():
        print(f"\n== {view} {result['split']}: {v['certifiable']} certifiable of {v['sessions']} sessions; clairvoyant/weight {fmt(v['clairvoyant_to_weight'])}")
        print(f"   no-prediction arms equal weight order: {v['no_prediction_equals_weight']}")
        print(f"   {'arm':28s} {'/ weight':26s} {'/ same rule w/o prop':26s} {'better/tied/worse vs w/o prop':32s} {'gap closed'}")
        for name, r in v["propagation"].items():
            if "to_static" in r:
                p = r["paired_vs_static"]
                pw = f"{p['better']['value']:.3f}/{p['tied']['value']:.3f}/{p['worse']['value']:.3f}"
                small = r["to_static_by_size"].get("<500 input cells")
                print(f"   {name:28s} {fmt(r['to_weight']):26s} {fmt(r['to_static']):26s} {pw:32s} {r['gap_closed']['value']:.2f}"
                      + (f"   <500: {fmt(small)}" if small else ""))
            else:
                print(f"   {name:28s} {fmt(r['to_weight']):26s}")
        for label, r in v["contrasts"].items():
            p = r["paired"]
            print(f"   {label:62s} {fmt(r['ratio'])}  b/t/w {p['better']['value']:.3f}/{p['tied']['value']:.3f}/{p['worse']['value']:.3f}")


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("run", "analyze", "record-entries"))
    ap.add_argument("--view", nargs="+", default=list(VIEWS), choices=(*VIEWS, "banc"))
    ap.add_argument("--split", choices=("development", "test"), default="development")
    ap.add_argument("--prior", nargs="+", help="priors (default: mlp_s0, provisional, const_half, const_share; 'external' = all priors in outputs/round5/priors)")
    ap.add_argument("--policy", nargs="+", choices=P.POLICIES, help="policies with propagation (default per prior)")
    ap.add_argument("--name", default="main", help="name of the result file of this run")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.workers > DEFAULT_WORKERS:
        log(f"--workers {a.workers} capped at {DEFAULT_WORKERS} (local machine shared by five workstreams)")
        a.workers = DEFAULT_WORKERS
    if a.stage == "record-entries":
        print(json.dumps(required_files([P.source(v) for v in a.view]), indent=2))
    elif a.stage == "run":
        if a.split == "test":
            if a.dry_run:
                raise SystemExit("--dry-run reads stored held-out costs without a claim; not on the held-out split")
            require_record("test", [P.source(v) for v in a.view])  # refuse before any view is run
        for view in a.view:
            run(view, a.split, a.prior, a.policy, a.name, a.workers, a.chunk, a.dry_run)
    else:
        show(analyze(a.split, a.view))


if __name__ == "__main__":
    main()
