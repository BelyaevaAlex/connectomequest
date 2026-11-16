#!/usr/bin/env python3
"""Controls of the B4 mechanism ("same type => same connectivity"), cross-fitted as in round 2.

For each view and direction (fit half H -> evaluate half 1-H), with the B4 pick of
round 2 on half H:

* shuffled     K[pi, pi] within the question (3 seeds), same B4 parameters;
* no-target    profiles without the output-to-target coordinate (i) same parameters,
               (ii) refitted on half H over the B4 grid;
* weight-only  exp(-|log w_j - log w_k| / sigma) on as many pairs as B4's kernel keeps;
               sigma (i) refitted with B4's lam and base, (ii) refitted with lam and base;
* oracle       ORACLE, uses reference labels: K = 1 for resolved cells of one hemibrain type;
               (i) B4's lam and base, (ii) lam/base refitted; no verdict;
* ipsilateral  B4 kernel with left-right pairs removed (side annotation; diagnostic);
* headroom     lam / gamma above the grid, fitting half only.

``run --view V --half H`` writes ``outputs/neurosym/B/round2-controls/{V}-h{H}.json``;
``assemble`` writes ``outputs/neurosym/B/round2-controls.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import product

import numpy as np

from connectomequest.neurosym import common as c
from connectomequest.neurosym import constraints as k
from connectomequest.neurosym import harness as h
from connectomequest.neurosym import kernel as kk

ROUND2 = c.ROOT / "outputs/neurosym/B/round2"
OUT = c.ROOT / "outputs/neurosym/B/round2-controls"
WORKERS = int(os.environ.get("NEUROSYM_WORKERS", "10"))
B4_GRID = dict(lam=(0.5, 2.0, 8.0), gamma=(1.0, 3.0), tau=(0.3, 0.5, 0.7))
SIGMAS = (0.25, 0.5, 1.0, 2.0)
LAMS = (0.5, 2.0, 8.0)
HEADROOM = [(g, l) for g, l in product((3.0, 4.0, 6.0), (8.0, 16.0, 32.0)) if (g, l) != (3.0, 8.0)]
SEEDS = (0, 1, 2)
_RUN = dict(view=None, configs={})
_KCACHE = {}


def key_of(cfg):
    return json.dumps(cfg, sort_keys=True)


def arms(fam, inst, outcomes, rho_index):
    view, out = _RUN["view"], {}
    if _KCACHE.get("target") != fam["target"]:
        _KCACHE.clear()
        _KCACHE["target"] = fam["target"]
    prior = fam["priors"]["mlp_s0"]
    for name, cfg in _RUN["configs"].items():
        if name not in _KCACHE:
            _KCACHE[name] = kk.control_kernel(view, fam, cfg)
        for base in ("ag", "recal"):
            out[f"{name}:{base}"] = kk.e.run_adaptive(inst, outcomes, kk.policy(base, inst, prior, _KCACHE[name], cfg["lam"]))
    return out


def costs(view, families, configs):
    _RUN.update(view=view, configs={key_of(cfg): cfg for cfg in configs})
    kk.profiles(view), k.rows_of(view, c.benchmark(view)["pre_cell"][:1]), c.view_data(view)  # before the fork
    out = h.run(view, families, arms, workers=WORKERS)
    _RUN["configs"] = {}
    return {key_of(cfg): {b: out[f"{key_of(cfg)}:{b}"] for b in ("ag", "recal")} for cfg in configs}


def baseline(view, families):
    mask = h.truth(view, families) > 0
    base = {b: h.baseline(view, families, f"mlp_s0:{b}") for b in ("ag", "recal")}
    return base[min(base, key=lambda b: base[b][mask].sum())], mask


def ratio_value(view, families, x, ref):
    return h.ratio(view, families, x, ref)["value"]


def pick(view, families, rows, ref, bases=("ag", "recal")):
    """(cfg, base) with the smallest ratio to the better baseline among ``rows`` (dict key -> costs)."""
    cand = [(ratio_value(view, families, r[b], ref), b, cfg) for cfg, r in rows for b in bases]
    value, base, cfg = min(cand, key=lambda x: (round(x[0], 6), x[1], key_of(x[2])))
    return dict(config=cfg, base=base, fit_ratio=value)


def retained(view, families, x, y, z, mask):
    """Share of B4's gain kept by the control: (1 - x/z) / (1 - y/z), paired type-cluster bootstrap."""
    index, counts, kk_ = h._clusters(view, tuple(int(f) for f in families))
    s = [np.bincount(index[mask], weights=np.asarray(v, float)[mask], minlength=kk_) for v in (x, y, z)]
    point = (1 - s[0].sum() / s[2].sum()) / (1 - s[1].sum() / s[2].sum())
    b = [counts @ v for v in s]
    boot = (1 - b[0] / b[2]) / (1 - b[1] / b[2])
    return dict(value=float(point), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))


def lr_diagnostic(view, families, tau):
    """Pairs with cos >= tau among the presynaptic cells of each evaluated target (side annotation; no verdict)."""
    d = c.benchmark(view)
    vd = c.view_data(view)
    names = vd["names"]["side"]
    left, right = (names.index(x) for x in kk.SIDES[view])
    tally = dict(pairs=0, contra=0, ipsi=0, other_side=0, contra_resolved=0, contra_same=0, ipsi_resolved=0, ipsi_same=0,
                 all_resolved_pairs=0, all_same=0)
    for t in np.unique(d["fam_target"][np.asarray(families)]):
        lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
        nodes = k.rows_of(view, d["pre_cell"][lo:hi])
        if len(nodes) < 2:
            continue
        labels = d["pre_label"][lo:hi]
        iu = np.triu_indices(len(nodes), 1)
        sim = kk.cosine(view, nodes)[iu]
        side = vd["side_code"][nodes]
        a, b = side[iu[0]], side[iu[1]]
        la, lb = labels[iu[0]], labels[iu[1]]
        resolved = (la >= 0) & (lb >= 0)
        same = resolved & (la == lb)
        lr = np.isin(a, (left, right)) & np.isin(b, (left, right))
        contra, ipsi = lr & (a != b), lr & (a == b)
        keep = sim >= tau
        tally["pairs"] += int(keep.sum())
        tally["contra"] += int((keep & contra).sum())
        tally["ipsi"] += int((keep & ipsi).sum())
        tally["other_side"] += int((keep & ~lr).sum())
        tally["contra_resolved"] += int((keep & contra & resolved).sum())
        tally["contra_same"] += int((keep & contra & same).sum())
        tally["ipsi_resolved"] += int((keep & ipsi & resolved).sum())
        tally["ipsi_same"] += int((keep & ipsi & same).sum())
        tally["all_resolved_pairs"] += int(resolved.sum())
        tally["all_same"] += int(same.sum())
    share = lambda x, y: float(x / y) if y else None
    return dict(tally, share_contralateral=share(tally["contra"], tally["pairs"]), share_ipsilateral=share(tally["ipsi"], tally["pairs"]),
                same_type_contralateral=share(tally["contra_same"], tally["contra_resolved"]),
                same_type_ipsilateral=share(tally["ipsi_same"], tally["ipsi_resolved"]),
                same_type_any_pair_in_question=share(tally["all_same"], tally["all_resolved_pairs"]))


def run(view, half):
    path = OUT / f"{view}-h{half}.json"
    if path.exists() and json.loads(path.read_text()).get("complete"):
        print(view, half, "already done", flush=True)
        return
    fitted = json.loads((ROUND2 / f"fit-{view}-h{half}.json").read_text())["picks"]["B4"]
    b4, b4_base = fitted["config"]["b4"], fitted["base"]
    fit_f, eval_f = c.development_families(view, half), c.development_families(view, 1 - half)
    out = dict(view=view, fit_half=half, eval_half=1 - half, b4=dict(b4, base=b4_base, fit_ratio=fitted["fit_ratio"]))

    def save():
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(out, indent=1, default=float) + "\n")
        tmp.replace(path)

    # ---- fitting half: refits and headroom (selection only)
    ref_fit, _ = baseline(view, fit_f)
    nt = [dict(kind="cos", no_target=True, lam=l, gamma=g, tau=t) for l, g, t in product(*B4_GRID.values())]
    wt = [dict(kind="weight", sigma=s, lam=l, tau=b4["tau"]) for s, l in product(SIGMAS, LAMS)]
    orc = [dict(kind="oracle", lam=l) for l in LAMS]
    head = [dict(kind="cos", lam=l, gamma=g, tau=b4["tau"]) for g, l in HEADROOM]
    fit_rows = costs(view, fit_f, nt + wt + orc + head)
    rows = lambda cfgs: [(cfg, fit_rows[key_of(cfg)]) for cfg in cfgs]
    picks = {
        "no_target_ii": pick(view, fit_f, rows(nt), ref_fit),
        "weight_i": pick(view, fit_f, rows([x for x in wt if x["lam"] == b4["lam"]]), ref_fit, bases=(b4_base,)),
        "weight_ii": pick(view, fit_f, rows(wt), ref_fit),
        "oracle_ii": pick(view, fit_f, rows(orc), ref_fit),
    }
    out["fit_picks"] = picks
    out["headroom_fit_half_only"] = dict(
        b4_fit_ratio=fitted["fit_ratio"],
        rows=[dict(gamma=cfg["gamma"], lam=cfg["lam"], tau=cfg["tau"], **{b: ratio_value(view, fit_f, r[b], ref_fit) for b in ("ag", "recal")})
              for cfg, r in rows(head)])
    save()
    print(view, half, "fit picks", json.dumps(picks), flush=True)

    # ---- evaluation half: each control once
    same = dict(lam=b4["lam"], gamma=b4["gamma"], tau=b4["tau"])
    evals = {
        "B4": (dict(kind="cos", **same), b4_base),
        **{f"shuffled_s{s}": (dict(kind="cos", shuffle=s, **same), b4_base) for s in SEEDS},
        "no_target_i": (dict(kind="cos", no_target=True, **same), b4_base),
        "no_target_ii": (picks["no_target_ii"]["config"], picks["no_target_ii"]["base"]),
        "weight_i": (picks["weight_i"]["config"], picks["weight_i"]["base"]),
        "weight_ii": (picks["weight_ii"]["config"], picks["weight_ii"]["base"]),
        "oracle_i": (dict(kind="oracle", lam=b4["lam"]), b4_base),
        "oracle_ii": (picks["oracle_ii"]["config"], picks["oracle_ii"]["base"]),
        "ipsilateral_only": (dict(kind="cos", ipsilateral_only=True, **same), b4_base),
    }
    res = costs(view, eval_f, [cfg for cfg, _ in evals.values()])
    ref, mask = baseline(view, eval_f)
    olds = {b: h.baseline(view, eval_f, f"mlp_s0:{b}") for b in ("ag", "recal")}
    b4_cost = res[key_of(evals["B4"][0])][b4_base]
    r2 = json.loads((c.ROOT / "outputs/neurosym/B/round2.json").read_text())["results"][f"{view}:fit{half}->eval{1 - half}"]["B4"]["primary"]["value"]
    assert abs(ratio_value(view, eval_f, b4_cost, ref) - r2) < 1e-12, "B4 reference must reproduce round2.json"
    out["eval"] = {}
    for name, (cfg, base) in evals.items():
        r = res[key_of(cfg)]
        out["eval"][name] = dict(config=cfg, base=base, primary=h.ratio(view, eval_f, r[base], ref),
                                 retained_of_b4=retained(view, eval_f, r[base], b4_cost, ref, mask),
                                 ag_new_over_old=h.ratio(view, eval_f, r["ag"], olds["ag"]),
                                 recal_new_over_old=h.ratio(view, eval_f, r["recal"], olds["recal"]))
        print(view, half, name, round(out["eval"][name]["primary"]["value"], 4), "kept", round(out["eval"][name]["retained_of_b4"]["value"], 3), flush=True)
    out["lr_diagnostic"] = lr_diagnostic(view, eval_f, b4["tau"])
    out["complete"] = True
    save()


def criterion(e):
    """Pre-declared: shuffled (every seed) and weight-only (ii) keep <= 1/3; no-target (ii) keeps >= 2/3; oracle (ii) gain >= B4."""
    shuffled = all(e[f"shuffled_s{s}"]["retained_of_b4"]["value"] <= 1 / 3 for s in SEEDS)
    weight = e["weight_ii"]["retained_of_b4"]["value"] <= 1 / 3
    no_target = e["no_target_ii"]["retained_of_b4"]["value"] >= 2 / 3
    oracle = e["oracle_ii"]["retained_of_b4"]["value"] >= 1.0
    return dict(shuffled_le_third=shuffled, weight_only_le_third=weight, no_target_ge_two_thirds=no_target, oracle_ge_b4=oracle,
                all=bool(shuffled and weight and no_target and oracle))


def assemble():
    out = dict(protocol=dict(
        cross_fit="B4 picks of round 2 (fit half H) evaluated on half 1-H; refits use half H only; both views, both directions",
        primary="cost ratio to the better of mlp_s0:recal/ag on the evaluation half, paired type-cluster bootstrap",
        retained="(1 - control/baseline) / (1 - B4/baseline) on the same bootstrap resamples",
        mechanism_supported_if="shuffled and weight-only keep <= 1/3 of B4's gain, no-target keeps >= 2/3, oracle gains >= B4 (in every view x direction)",
        oracle_note="ORACLE: uses reference hemibrain labels of unpaid cells; reference only, no verdict",
        lr_note="side annotation of the release, diagnostics only"), runs={}, criterion={})
    for view in ("flywire", "malecns"):
        for half in (0, 1):
            r = json.loads((OUT / f"{view}-h{half}.json").read_text())
            key = f"{view}:fit{half}->eval{1 - half}"
            out["runs"][key] = r
            out["criterion"][key] = criterion(r["eval"])
    out["mechanism_supported"] = all(v["all"] for v in out["criterion"].values())
    (OUT.parent / "round2-controls.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False) + "\n")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("run", "assemble"))
    p.add_argument("--view", nargs="+", default=["flywire", "malecns"])
    p.add_argument("--half", nargs="+", type=int, default=[0, 1])
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "run":
        for view in a.view:
            for half in a.half:
                run(view, half)
    else:
        assemble()


if __name__ == "__main__":
    main()
