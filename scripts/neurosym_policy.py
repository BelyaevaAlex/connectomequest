"""Pilot A: fit the index policy to certificate cost by CEM (half 0), evaluate on half 1.

  fit       --view V --prior P   CEM over policy.DIM parameters on development half 0;
                                 state is saved after every generation and resumed.
  evaluate                       half-1 comparison of the fitted policies with the
                                 frozen baselines -> outputs/neurosym/A/results.json

Objective: total cost of the certifiable sessions of half 0 (uncertifiable sessions
cost N under every order and are skipped while fitting; never while evaluating).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from connectomequest.annotation_v35 import engine as e
from connectomequest.neurosym import common as c
from connectomequest.neurosym import harness as h
from connectomequest.neurosym import policy as P

OUT = c.ROOT / "outputs/neurosym/A"
SETTINGS = (("flywire", "mlp_s0"), ("flywire", "provisional"), ("malecns", "mlp_s0"))
CRITERION = {"mlp_s0": ("recal", "ag", "direction"), "provisional": ("recal", "direction")}
EXTRA = ("ag", "recal", "direction", "adir", "adir-recal", "adg")
_POP = {}


def _workers():
    return int(os.environ.get("NEUROSYM_WORKERS", len(os.sched_getaffinity(0))))


def _cem_arms(fam, inst, y, rho_index):
    cands = _POP["cands"]
    if e.full_information_status(inst, y) == "unresolved":
        order = e.weight_order(inst)  # every complete order costs N here
        return {f"c{i:03d}": order for i in range(len(cands))}
    p = fam["priors"][_POP["prior"]]
    return {f"c{i:03d}": P.run_session(inst, y, p, params) for i, params in enumerate(cands)}


def _costs(view, prior, families, cands, workers):
    _POP.update(prior=prior, cands=cands)
    out = h.run(view, families, _cem_arms, workers=workers)
    mask = h.truth(view, families) > 0
    return np.array([int(out[f"c{i:03d}"][mask].sum()) for i in range(len(cands))])


def _jsonable(params):
    return dict(theta=dict(zip(P.FEATURES, map(float, params["theta"]))), recal=float(params["recal"]),
                strength=float(params["strength"]), fallback=bool(params["fallback"]))


def _params(js):
    return dict(theta=np.array([js["theta"][k] for k in P.FEATURES]), recal=js["recal"], strength=js["strength"], fallback=js["fallback"])


def fit(view, prior, pop=40, iters=20, elite=8, seed=0, smooth=0.7, workers=None):
    workers = workers or _workers()
    path = OUT / f"fit-{view}-{prior}.json"
    families = c.development_families(view, 0)
    weight_total = int(h.baseline(view, families, "weight")[h.truth(view, families) > 0].sum())
    if path.exists():
        state = json.loads(path.read_text())
    else:
        init = P.encode(P.special("recal"))
        std = np.concatenate((np.full(len(P.FEATURES), 0.3), [0.4, 1.5]))
        state = dict(view=view, prior=prior, half=0, families=len(families), weight_total=weight_total, pop=pop, iters=iters, elite=elite,
                     seed=seed, smooth=smooth, mean=init.tolist(), std=std.tolist(), generation=0, history=[], best=None)
    rng = np.random.default_rng([seed, state["generation"], sum(map(ord, view + prior))])
    while state["generation"] < state["iters"]:
        g, t0 = state["generation"], time.time()
        mean, std = np.array(state["mean"]), np.array(state["std"])
        xs = [mean] + [mean + std * rng.standard_normal(P.DIM) for _ in range(state["pop"] - 1)]
        cands = [P.decode(x) for x in xs]
        if g == 0:  # engine special cases enter the first generation
            cands[1 : 1 + 4] = [P.special(n) for n in ("ag", "recal", "adir", "adir-recal")]
        xs = [P.encode(p) for p in cands]
        costs = _costs(view, prior, families, cands, workers)
        top = np.argsort(costs, kind="stable")[: state["elite"]]
        elite_x = np.array([xs[i] for i in top])
        new_mean = (1 - smooth) * mean + smooth * elite_x.mean(0)
        new_std = np.maximum((1 - smooth) * std + smooth * elite_x.std(0), 0.02)
        i = int(top[0])
        if state["best"] is None or costs[i] < state["best"]["cost"]:
            state["best"] = dict(cost=int(costs[i]), ratio=costs[i] / weight_total, generation=g, params=_jsonable(cands[i]))
        state["history"].append(dict(generation=g, mean_cost=int(costs[0]), mean_ratio=costs[0] / weight_total, best_ratio=costs[i] / weight_total,
                                     elite_ratio=float(costs[top].mean() / weight_total), seconds=round(time.time() - t0, 1),
                                     **({"special": {n: costs[1 + k] / weight_total for k, n in enumerate(("ag", "recal", "adir", "adir-recal"))}} if g == 0 else {})))
        state.update(mean=new_mean.tolist(), std=new_std.tolist(), generation=g + 1)
        print(f"[{view}/{prior}] gen {g}: mean {costs[0] / weight_total:.4f} best {costs[i] / weight_total:.4f} "
              f"elite {costs[top].mean() / weight_total:.4f} ({time.time() - t0:.0f}s)", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(path)
    if "final" not in state:  # pre-declared output: the final CEM mean
        final = P.decode(np.array(state["mean"]))
        cost = _costs(view, prior, families, [final], workers)[0]
        state["final"] = dict(cost=int(cost), ratio=cost / weight_total, params=_jsonable(final))
        path.write_text(json.dumps(state, indent=1))
    print(f"[{view}/{prior}] final mean ratio on half 0: {state['final']['ratio']:.4f}", flush=True)
    return state


def _eval_arms(fam, inst, y, rho_index):
    return {"learned": P.run_session(inst, y, fam["priors"][_POP["prior"]], _POP["params"])}


def evaluate(workers=None, half=1):
    workers = workers or _workers()
    results = dict(half=half, criterion="learned / best criterion baseline <= 0.98 with 95% upper < 1.00 on >=1 setting, others <= 1.01", settings={})
    for view, prior in SETTINGS:
        state = json.loads((OUT / f"fit-{view}-{prior}.json").read_text())
        params = _params(state["final"]["params"])
        rec = dict(params=state["final"]["params"], half0_ratio_to_weight=state["final"]["ratio"])
        for hf in (0, half):
            families = c.development_families(view, hf)
            _POP.update(prior=prior, params=params)
            learned = h.run(view, families, _eval_arms, workers=workers)["learned"]
            mask = h.truth(view, families) > 0
            weight = h.baseline(view, families, "weight")
            base = {n: h.baseline(view, families, f"{prior}:{n}") for n in EXTRA}
            crit = min(CRITERION[prior], key=lambda n: int(base[n][mask].sum()))
            allbest = min(EXTRA, key=lambda n: int(base[n][mask].sum()))
            rec[f"half{hf}"] = dict(
                families=len(families), certifiable_sessions=int(mask.sum()),
                learned_vs_weight=h.ratio(view, families, learned, weight),
                baselines_vs_weight={n: h.ratio(view, families, base[n], weight) for n in EXTRA},
                best_criterion_baseline=crit, learned_vs_best_criterion=h.ratio(view, families, learned, base[crit]),
                best_extra_baseline=allbest, learned_vs_best_extra=h.ratio(view, families, learned, base[allbest]))
            print(view, prior, f"half{hf}", "learned/weight", round(rec[f"half{hf}"]["learned_vs_weight"]["value"], 4),
                  f"learned/{crit}", {k: round(v, 4) for k, v in rec[f"half{hf}"]["learned_vs_best_criterion"].items()}, flush=True)
        results["settings"][f"{view}:{prior}"] = rec
    pairs = [r[f"half{half}"]["learned_vs_best_criterion"] for r in results["settings"].values()]
    results["verdict"] = "DO" if any(p["value"] <= 0.98 and p["high"] < 1.0 for p in pairs) and all(p["value"] <= 1.01 for p in pairs) else "DO NOT"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print("verdict:", results["verdict"], flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--view", required=True)
    f.add_argument("--prior", required=True)
    f.add_argument("--pop", type=int, default=40)
    f.add_argument("--iters", type=int, default=20)
    f.add_argument("--elite", type=int, default=8)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--workers", type=int, default=None)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a.view, a.prior, pop=a.pop, iters=a.iters, elite=a.elite, seed=a.seed, workers=a.workers)
    else:
        evaluate(workers=a.workers)


if __name__ == "__main__":
    main()
