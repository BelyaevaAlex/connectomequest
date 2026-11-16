"""Biological readout of the frozen answers (descriptive; one split at a time).

B3  Partial identification of conservation. For each kind of partner set and retention level
    the share of target neurons whose partner set keeps a fraction rho of its hemibrain share
    is identified only up to [certified yes, 1 - certified no]; unresolved questions could go
    either way. Computed under the declared crosswalk (frozen ``truth``) and under the
    lenient crosswalk of ``scripts/v35_lenient.py`` (release-specific types without a declared
    hemibrain correspondence read as nonmembers), recomputed per question with the same code.
B4  Sexual dimorphism. Certified cross-connectome disagreements (one target certified yes, a
    target of the same type in the other connectome certified no, same partner set and rho)
    against sexually dimorphic / fru-dsx annotations of the target type or of the partner
    types (MaleCNS ``dimorphism``/``fruDsx``, FlyWire v3.1.0 ``dimorphism``/``fru_dsx``,
    mapped to hemibrain types by the declared crosswalk); permutation tests over types.
B5  Hemibrain truncation. The local hemibrain export keeps only neurons with
    ``status=Traced and cropped=false`` (data/raw/hemibrain/manifest.json); the hemibrain
    metadata of Schlegel et al. (inputs/v32/ontology, column ``cropped``) lists the cropped
    neurons that were therefore left out. Per hemibrain type we compute the share of its
    input (postsynapses) and of its output (presynapses) on cropped neurons and stratify
    shares, answers and costs by the truncation of the target type and of its partner set.
"""

from __future__ import annotations

import sys
from functools import cache, lru_cache

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc

from connectomequest.annotation_v35 import engine as e
from connectomequest.neurosym import common as nc
from connectomequest.round5 import costs_common as cc

STATUS = {"unresolved": 0, "positive": 1, "negative": 2}
KINDS = ("top1", "coalition")
RHO_NAMES = ("1/4", "1/2", "3/4", "1/1")
PERMUTATIONS = 10000
HEMIBRAIN_META = cc.ROOT / "inputs/v32/ontology/Supplemental_file5_hemibrain_meta.csv"


def _scripts():
    path = str(cc.ROOT / "scripts")
    if path not in sys.path:
        sys.path.insert(0, path)


# --------------------------------------------------------------------------- B3


def release_specific_flags(view, split):
    """Input cells (of targets of ``split``) that are unresolved because their release type
    has no declared hemibrain correspondence: the cells the lenient crosswalk reads as
    nonmembers. Same rule as ``scripts/v35_lenient.py``."""
    _scripts()
    from v35_diagnostics import VIEWS, annotation_rows, unresolved_reason

    def build():
        d = nc.benchmark(view)
        targets = np.unique(d["fam_target"][cc.split_families(view, split)])
        lookup = {name: i for i, name in enumerate(nc.vocabulary()) if i}
        rows = annotation_rows(view)
        flags = np.zeros(len(d["pre_cell"]), dtype=bool)
        memo = {}
        for t in targets.tolist():
            for i in range(int(d["target_offset"][t]), int(d["target_offset"][t + 1])):
                if d["pre_label"][i] >= 0:
                    continue
                cell = int(d["pre_cell"][i])
                if cell not in memo:
                    reason = unresolved_reason(*rows.get(cell, (None, None)), VIEWS[view]["release"], lookup)
                    memo[cell] = reason == "release-specific type without correspondence"
                flags[i] = memo[cell]
        return dict(flags=flags, targets=targets)

    return cc.cached_npz(f"costs-release-specific-{view}-{split}.npz", build)["flags"]


def lenient_status(view, split):
    """Per frozen row: full-information status under the lenient crosswalk."""
    fr = cc.frozen(view, split)
    d = nc.benchmark(view)
    flags = release_specific_flags(view, split)
    out = np.full(len(fr["family"]), -1, dtype=np.int64)
    for f in np.unique(fr["family"]).tolist():
        fam = nc.family(view, f)
        t = fam["target"]
        lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
        lenient = np.where((fam["labels"] < 0) & flags[lo:hi], e.NONMEMBER, fam["outcomes"])
        rows = np.flatnonzero(fr["family"] == f)
        for rho, threshold in nc.thresholds(fam):
            inst = e.threshold_instance(fam["cells"], fam["weights"], threshold)
            row = rows[fr["rho"][rows] == rho]
            out[row] = STATUS[e.full_information_status(inst, lenient)]
            strict = STATUS[e.full_information_status(inst, fam["outcomes"])]
            if strict != fr["truth"][row[0]]:
                raise AssertionError("declared status differs from the frozen truth")
    if np.any(out < 0):
        raise AssertionError("frozen rows without a lenient status")
    return out


def conservation(view, split):
    fr = cc.frozen(view, split)
    d = nc.benchmark(view)
    kind = d["fam_kind"][fr["family"]]
    boot = cc.Bootstrap(cc.cluster_of_rows(view, fr["family"]))
    statuses = dict(declared=fr["truth"], lenient=lenient_status(view, split))
    out = {}
    for name, status in statuses.items():
        yes, no, unres = (status == 1).astype(float), (status == 2).astype(float), (status == 0).astype(float)
        for k, kname in enumerate(KINDS):
            for r, rname in enumerate(RHO_NAMES):
                m = (kind == k) & (fr["rho"] == r)
                lower = boot.mean(yes, m)
                upper_no = boot.mean(no, m)
                upper = dict(value=1 - upper_no["value"], low=1 - upper_no["high"], high=1 - upper_no["low"])
                out[f"{name}:{kname}@rho={rname}"] = dict(
                    targets=int(m.sum()), types=int(len(np.unique(boot.index[m]))), lower=lower, upper=upper,
                    width=boot.mean(unres, m), outer_95=[lower["low"], upper["high"]],
                )
    reproduction = {}
    for name, status in statuses.items():
        reproduction[name] = {k: float((status == v).mean()) for k, v in (("positive", 1), ("negative", 2), ("unresolved", 0))}
    moved = statuses["declared"] != statuses["lenient"]
    contradictions = int((moved & (statuses["declared"] != 0)).sum())
    return dict(intervals=out, answers=reproduction, contradictions=contradictions, statuses=statuses)


# --------------------------------------------------------------------------- B4


@cache
def neuron_flags(view):
    """Per neuron of the view cache: dimorphic (any annotation but isomorphic), strict
    dimorphic (no 'potentially'), fru/dsx-expressing."""
    ids = nc.view_data(view)["ids"]
    if view == "malecns":
        table = ipc.open_file(cc.ROOT / "data/raw/manc/body-annotations-male-cns-v1.0-minconf-0.5.feather").read_all()
        table = table.select(["bodyId", "dimorphism", "fruDsx"]).to_pandas().set_index("bodyId")
        dim, fru = table["dimorphism"].reindex(ids), table["fruDsx"].reindex(ids)
    else:
        table = pd.read_csv(nc.ANNOTATIONS["flywire"], sep="\t", usecols=["root_id", "dimorphism", "fru_dsx"], low_memory=False)
        table = table.set_index("root_id")
        dim, fru = table["dimorphism"].reindex(ids), table["fru_dsx"].reindex(ids)
    dim = dim.astype("object").where(dim.notna(), None).to_numpy()
    fru = fru.astype("object").where(fru.notna(), None).to_numpy()
    dimorphic = np.array([x is not None and x != "isomorphic" for x in dim])
    strict = np.array([x is not None and x != "isomorphic" and not str(x).startswith("potentially") for x in dim])
    return dict(dimorphic=dimorphic, dimorphic_strict=strict, fru_dsx=np.array([x is not None for x in fru]))


@cache
def type_flags():
    """Per hemibrain type code: flagged if any neuron carrying it (declared crosswalk) in
    MaleCNS or FlyWire carries the annotation; also the neuron counts."""
    v = len(nc.vocabulary())
    out = {}
    for key in ("dimorphic", "dimorphic_strict", "fru_dsx"):
        flag = np.zeros(v, dtype=bool)
        for view in cc.VIEWS:
            labels = nc.view_data(view)["labels"]
            ok = labels >= 0
            flag |= np.bincount(labels[ok], weights=neuron_flags(view)[key][ok], minlength=v) > 0
        out[key] = flag
    return out


def decided_pairs(split):
    """Cross-connectome and within-connectome pairs of decided answers per (type, kind, rho)."""
    table = {}
    for view in cc.VIEWS:
        fr = cc.frozen(view, split)
        d = nc.benchmark(view)
        target = d["fam_target"][fr["family"]]
        for t, code, kind, rho, truth in zip(target.tolist(), d["target_type"][target].tolist(), d["fam_kind"][fr["family"]].tolist(), fr["rho"].tolist(), fr["truth"].tolist()):
            table.setdefault((code, kind, rho), {}).setdefault(view, {})[t] = truth
    rows = []
    for (code, kind, rho), by_view in table.items():
        male, fly = by_view.get("malecns", {}), by_view.get("flywire", {})
        for a in male.values():
            for b in fly.values():
                if a > 0 and b > 0:
                    rows.append(("across", code, kind, rho, a, b))
        for view, group in by_view.items():
            vals = sorted(group.items())
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    if vals[i][1] > 0 and vals[j][1] > 0:
                        rows.append((f"within:{view}", code, kind, rho, vals[i][1], vals[j][1]))
    return pd.DataFrame(rows, columns=["scope", "code", "kind", "rho", "a", "b"])


def _rate_difference(D, N, flag):
    if N[flag].sum() == 0 or N[~flag].sum() == 0:
        return np.nan
    return D[flag].sum() / N[flag].sum() - D[~flag].sum() / N[~flag].sum()


def bootstrap_difference(D, N, flag, clusters, draws=cc.BOOTSTRAP, seed=cc.SEED):
    """Percentile interval of the pooled rate difference, resampling types (clusters)."""
    labels, index = np.unique(np.asarray(clusters), return_inverse=True)
    n = len(labels)
    sums = np.zeros((4, n))
    for row, (x, m) in enumerate(((D, flag), (N, flag), (D, ~flag), (N, ~flag))):
        sums[row] = np.bincount(index[m], weights=x[m], minlength=n)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(r, minlength=n) for r in rng.integers(0, n, size=(draws, n))]).astype(float)
    fd, fn, ud, un = (counts @ sums.T).T
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = fd / fn - ud / un
    diff = diff[np.isfinite(diff)]
    return [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]


def permutation_test(D, N, flag, strata=None, clusters=None, permutations=PERMUTATIONS, seed=cc.SEED):
    """Two-sided permutation test of the pooled discordance-rate difference.

    ``D``, ``N``: discordant and decided pairs per unit (a type, or a type and kind of
    partner set); ``flag``: annotation per unit; flags are permuted among units with N > 0,
    within ``strata`` if given (kind of partner set). ``clusters`` (default: the units) are
    the types resampled for the bootstrap interval of the difference."""
    D, N, flag = np.asarray(D, float), np.asarray(N, float), np.asarray(flag, bool)
    keep = N > 0
    clusters = np.arange(len(D)) if clusters is None else np.asarray(clusters)
    D, N, flag, clusters = D[keep], N[keep], flag[keep], clusters[keep]
    strata = np.zeros(len(D), dtype=int) if strata is None else np.asarray(strata)[keep]
    observed = _rate_difference(D, N, flag)
    rng = np.random.default_rng(seed)
    stats = np.empty(permutations)
    for b in range(permutations):
        perm = flag.copy()
        for s in np.unique(strata):
            idx = np.flatnonzero(strata == s)
            perm[idx] = perm[rng.permutation(idx)]
        stats[b] = _rate_difference(D, N, perm)
    p = (1 + np.sum(np.abs(stats) >= abs(observed) - 1e-12)) / (1 + permutations)
    fd, fn, ud, un = D[flag].sum(), N[flag].sum(), D[~flag].sum(), N[~flag].sum()
    odds = ((fd + 0.5) * (un - ud + 0.5)) / ((fn - fd + 0.5) * (ud + 0.5))
    return dict(
        units=int(len(D)), flagged_units=int(flag.sum()), flagged_pairs=int(fn), flagged_discordant=int(fd),
        other_pairs=int(un), other_discordant=int(ud), flagged_rate=float(fd / fn) if fn else None,
        other_rate=float(ud / un) if un else None, rate_difference=float(observed),
        rate_ratio=float((fd / fn) / (ud / un)) if fn and un and ud else None, odds_ratio_haldane=float(odds), p_value=float(p),
        rate_difference_95=bootstrap_difference(D, N, flag, clusters) if fn and un else None,
    )


def holm(pvalues):
    names = sorted(pvalues, key=lambda k: pvalues[k])
    m, running, out = len(names), 0.0, {}
    for i, name in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvalues[name]))
        out[name] = running
    return out


def dimorphism(split):
    pairs = decided_pairs(split)
    flags = type_flags()
    manifest = nc.manifest()
    types = {t["code"]: t for t in manifest["types"] if t["split"] == split}
    partner = {}
    for code, t in types.items():
        for fam in t["families"]:
            k = 0 if fam["kind"] == "top1" else 1
            partner[(code, k)] = {key: bool(flags[key][fam["codes"]].any()) for key in flags}
    pairs["discordant"] = pairs["a"] != pairs["b"]
    pairs["male_no_fly_yes"] = (pairs["a"] == 2) & (pairs["b"] == 1)
    pairs["male_yes_fly_no"] = (pairs["a"] == 1) & (pairs["b"] == 2)
    results, primary = {}, {}
    for scope in ("across", "within"):
        sub = pairs[pairs["scope"] == "across"] if scope == "across" else pairs[pairs["scope"].str.startswith("within")]
        by_type = sub.groupby("code").agg(D=("discordant", "sum"), N=("discordant", "size"))
        by_tk = sub.groupby(["code", "kind"]).agg(D=("discordant", "sum"), N=("discordant", "size")).reset_index()
        for key in ("dimorphic", "fru_dsx", "dimorphic_strict"):
            target_flag = flags[key][by_type.index.to_numpy()]
            res = permutation_test(by_type["D"], by_type["N"], target_flag)
            name = f"{scope}:target:{key}"
            results[name] = res
            pflag = np.array([partner[(c, k)][key] for c, k in zip(by_tk["code"], by_tk["kind"])])
            res = permutation_test(by_tk["D"], by_tk["N"], pflag, strata=by_tk["kind"].to_numpy(), clusters=by_tk["code"].to_numpy())
            results[f"{scope}:partner:{key}"] = res
            if scope == "across" and key != "dimorphic_strict":
                primary[f"{scope}:target:{key}"] = results[name]["p_value"]
                primary[f"{scope}:partner:{key}"] = res["p_value"]
    adjusted = holm(primary)
    for name, p in adjusted.items():
        results[name]["p_holm_primary"] = p
    across = pairs[pairs["scope"] == "across"]
    direction = {}
    for key in ("dimorphic", "fru_dsx"):
        f = flags[key][across["code"].to_numpy()]
        for label, m in (("flagged", f), ("other", ~f)):
            direction[f"target:{key}:{label}"] = dict(
                pairs=int(m.sum()), male_no_fly_yes=int(across["male_no_fly_yes"][m].sum()), male_yes_fly_no=int(across["male_yes_fly_no"][m].sum())
            )
    by_rho = {}
    for (k, r), g in across.groupby(["kind", "rho"]):
        by_rho[f"{KINDS[k]}@rho={RHO_NAMES[r]}"] = dict(pairs=int(len(g)), discordant=int(g["discordant"].sum()))
    counts = dict(
        split_types=len(types), flagged_types={key: int(flags[key][list(types)].sum()) for key in flags},
        flagged_types_names={key: sorted(types[c]["name"] for c in types if flags[key][c]) for key in ("dimorphic", "fru_dsx")},
    )
    return dict(tests=results, primary_family=sorted(primary), direction=direction, across_by_kind_rho=by_rho, counts=counts,
                discordant_across_types=sorted({types[c]["name"] for c in across["code"][across["discordant"]].tolist()}))


# --------------------------------------------------------------------------- B5


@lru_cache(maxsize=1)
def truncation_by_type():
    """Per hemibrain type code: share of its postsynapses (input) and presynapses (output)
    carried by cropped neurons, and the numbers of its neurons (all / cropped)."""
    meta = pd.read_csv(HEMIBRAIN_META, usecols=["bodyId", "type", "cropped", "pre", "post"], low_memory=False)
    lookup = {name: i for i, name in enumerate(nc.vocabulary()) if i}
    meta = meta[meta["type"].isin(lookup)].copy()
    meta["code"] = meta["type"].map(lookup).astype(int)
    meta["cropped"] = meta["cropped"].astype(bool)
    v = len(nc.vocabulary())
    out = {}
    for col in ("pre", "post"):
        x = meta[col].fillna(0).to_numpy(float)
        total = np.bincount(meta["code"], weights=x, minlength=v)
        lost = np.bincount(meta["code"], weights=x * meta["cropped"].to_numpy(), minlength=v)
        out[f"{col}_share"] = np.divide(lost, total, out=np.zeros(v), where=total > 0)
    out["neurons"] = np.bincount(meta["code"], minlength=v)
    out["cropped_neurons"] = np.bincount(meta["code"], weights=meta["cropped"].to_numpy(float), minlength=v).astype(int)
    hb = nc.hemibrain()
    in_graph = np.isin(meta["bodyId"].to_numpy(), hb["ids"])
    out["cropped_in_graph"] = int((meta["cropped"].to_numpy() & in_graph).sum())
    out["cropped_total"] = int(meta["cropped"].sum())
    return out


def partner_truncation(code, partner_codes):
    """Synapse-weighted output share on cropped neurons of the partner types of ``code``:
    sum_c S(c) phi_c / sum_c S(c), with S(c) the hemibrain synapses from type c onto type code."""
    tr = truncation_by_type()
    by_type = nc.hemibrain()["by_type"]
    s = np.array([by_type[c, code] for c in partner_codes], dtype=float)
    phi = tr["pre_share"][np.asarray(partner_codes)]
    return float((s * phi).sum() / s.sum()) if s.sum() > 0 else float(phi.mean())


TRUNCATION_ARMS = dict(malecns=("mlp_s0:ag", "mlp_s0:recal"), flywire=("mlp_s0:ag", "mlp_s0:recal", "provisional:direction", "provisional:recal"))
SUBSTANTIAL = 0.1


def truncation(view, split):
    """Shares, answers and costs by truncation of the target type and of the partner set."""
    fr = cc.frozen(view, split)
    d = nc.benchmark(view)
    tr = truncation_by_type()
    manifest = {t["code"]: t for t in nc.manifest()["types"]}
    fam = fr["family"]
    code = d["target_type"][d["fam_target"][fam]]
    kind = d["fam_kind"][fam]
    memo = {}
    for f in np.unique(fam).tolist():
        codes = d["fam_codes"][d["fam_code_offset"][f] : d["fam_code_offset"][f + 1]]
        memo[f] = partner_truncation(int(d["target_type"][d["fam_target"][f]]), codes.tolist())
    partner = np.array([memo[int(f)] for f in fam])
    target = tr["post_share"][code]
    share = d["fam_source_weight"][fam] / d["fam_source_total"][fam]
    boot = cc.Bootstrap(code)
    truth = fr["truth"]
    certifiable = truth > 0
    arm = fr["arms"]
    weight = fr["cost"][:, arm["weight"]].astype(float)
    clair = np.where(fr["clairvoyant"] < 0, fr["size"], fr["clairvoyant"]).astype(float)
    costs = {"clairvoyant": clair, **{a: fr["cost"][:, arm[a]].astype(float) for a in TRUNCATION_ARMS[view]}}
    strata = {
        "target:none": target == 0,
        "target:some": target > 0,
        f"target:>={SUBSTANTIAL}": target >= SUBSTANTIAL,
        "partner:none": partner == 0,
        "partner:some": partner > 0,
        f"partner:>={SUBSTANTIAL}": partner >= SUBSTANTIAL,
    }
    out = {}
    for kname_i, kname in ((None, "all"), (0, "top1"), (1, "coalition")):
        km = np.ones(len(fam), dtype=bool) if kname_i is None else kind == kname_i
        for sname, sm in strata.items():
            m = km & sm
            if not m.any():
                continue
            mc = m & certifiable
            out[f"{kname}:{sname}"] = dict(
                questions=int(m.sum()), types=int(len(np.unique(code[m]))),
                yes=boot.mean((truth == 1).astype(float), m), no=boot.mean((truth == 2).astype(float), m),
                unresolved=boot.mean((truth == 0).astype(float), m), hemibrain_share=boot.mean(share, m),
                inputs=boot.mean(fr["size"].astype(float), m),
                weight_fraction_read=boot.mean(weight / fr["size"], mc) if mc.any() else None,
                ratio_to_weight={a: boot.ratio(x, weight, mc) for a, x in costs.items()} if mc.any() else None,
            )
        for which, none, some in (("target", "target:none", "target:some"), ("partner", "partner:none", "partner:some")):
            a, b = km & strata[some], km & strata[none]
            if a.any() and b.any():
                out[f"{kname}:{which}:some-minus-none"] = dict(
                    yes=boot.mean_difference((truth == 1).astype(float), a, b), no=boot.mean_difference((truth == 2).astype(float), a, b),
                    unresolved=boot.mean_difference((truth == 0).astype(float), a, b), hemibrain_share=boot.mean_difference(share, a, b),
                )
    split_codes = [c for c, t in manifest.items() if t["split"] == split]
    summary = dict(
        cropped_neurons_in_metadata=tr["cropped_total"], cropped_neurons_in_source_graph=tr["cropped_in_graph"],
        split_types_with_cropped_neurons=int((tr["cropped_neurons"][split_codes] > 0).sum()), split_types=len(split_codes),
        target_types_with_input_loss=int(len(np.unique(code[target > 0]))),
        families_with_partner_loss={KINDS[k]: int(len(np.unique(fam[(partner > 0) & (kind == k)]))) for k in (0, 1)},
        families={KINDS[k]: int(len(np.unique(fam[kind == k]))) for k in (0, 1)},
    )
    return dict(summary=summary, strata=out)
