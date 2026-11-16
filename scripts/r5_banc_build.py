#!/usr/bin/env python3
"""Build the BANC view of the V35 conserved-partner benchmark (round 5, workstream ``banc``).

Steps (each refuses to overwrite an existing output):

1. check the public Dataverse files (md5 published by Dataverse) under ``inputs/round5/banc/raw``;
2. prepared graph ``inputs/round5/banc/prepared-v888`` in the layout of ``inputs/v32/prepared-common-v1``:
   neurons with a neuronal super class, v2 edgelist, duplicate records summed, pairs with at least
   five synapses, self pairs removed (``annotation_v32.data`` functions);
3. the twelve structural features of ``annotation_v32.predictors.structural_features`` on that graph;
4. reference labels by the declared crosswalk (``round5.banc.crosswalk``);
5. questions with the construction of ``v35_build.build_view``: the types, partner sets, hemibrain shares and
   splits of ``outputs/v35/benchmark/manifest.json``; priors ``mlp_s0`` (frozen model, unchanged) and the two
   no-prediction priors ``const_half`` / ``const_share`` of the first post hoc round;
6. ``outputs/round5/banc/{benchmark.npz, manifest.json, view.npz, view-names.json, features.npz, build.json}``.

Nothing here runs a question. Answer-dependent summaries (coverage of input cells) are computed on
development targets only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import inspect
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.feather as feather
from scipy import sparse

from connectomequest.annotation_v32.data import PrivateAnnotations, TYPE_ALIASES, _from_matrix, save_prepared_dataset
from connectomequest.annotation_v32.predictors import structural_features
from connectomequest.annotation_v32.runtime import hash_file
from connectomequest.annotation_v32.training import StructuralMLP
from connectomequest.round5 import banc as b

import v35_build  # frozen; imported, never modified

ROOT = b.ROOT
PROTOCOL = dict(
    schema="cq-r5-banc-benchmark-v1",
    base_protocol="cq-v35-conserved-partner-benchmark-v1 (outputs/v35/benchmark/manifest.json)",
    release=b.RELEASE,
    source="Harvard Dataverse doi:10.7910/DVN/7WTH1N v3.0 (CC BY 4.0); Bates, Phelps, Kim, Yang et al.",
    universe="rows of banc_888_meta.feather whose super_class is neuronal: " + ", ".join(b.NEURONAL_SUPER_CLASSES),
    graph_rule="v2 neuron-to-neuron edgelist (banc_888_edgelist_simple_v2.feather, size>=5 voxel synapses); duplicate records summed; pairs with at least five synapses; self pairs excluded",
    crosswalk="cell_type if in the hemibrain vocabulary, else one single-valued author hemibrain correspondence (hemibrain_cell_type), else unresolved; 'auto:' values (unreviewed NBLAST top hits) are not hemibrain type names",
    name_rules=dict(aliases=TYPE_ALIASES, coarsening="union of annotation_v32.data.COMMON_GROUP_MAPPINGS", groups=b.GROUPS),
    endpoints="per type, BANC neurons of the type with >= minimum_partners input cells ordered by sha256('cq-v35-endpoint|banc|<type>|<id>'); first endpoints_per_type",
    minimum_partners=v35_build.PROTOCOL["minimum_partners"],
    endpoints_per_type=v35_build.PROTOCOL["endpoints_per_type"],
    min_synapses=b.MIN_SYNAPSES,
    priors=dict(mlp_s0="v35_build.MLP model and temperature, features of structural_features on the BANC graph",
                const_half="p+ = 1/2, p_unresolved = 0", const_share="p+ = clip(hemibrain share, 0.01, 0.99), p_unresolved = 0"),
    partner_sets="the v35 manifest families unchanged, including partner types absent from BANC",
)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_raw():
    out = {}
    for name, (file_id, expected) in b.RAW_FILES.items():
        path = b.RAW / name
        if md5(path) != expected:
            raise ValueError(f"{name}: md5 differs from Dataverse")
        out[name] = dict(dataverse_file_id=file_id, md5=expected, sha256=hash_file(path), url=b.DATAVERSE.format(file_id))
    return out


META_COLUMNS = ["banc_888_id", "super_class", "cell_type", "hemibrain_cell_type", "fafb_cell_type", "malecns_cell_type",
                "side", "region", "neurotransmitter_predicted", "proofread"]


def load_meta():
    table = feather.read_table(b.RAW / "banc_888_meta.feather", columns=META_COLUMNS)
    rows = table.to_pylist()
    ids = [int(r["banc_888_id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("banc_888_id is not unique")
    keep = [r for r in rows if r["super_class"] in b.NEURONAL_SUPER_CLASSES]
    meta = {int(r["banc_888_id"]): r for r in keep}
    excluded = Counter(r["super_class"] or "<none>" for r in rows if r["super_class"] not in b.NEURONAL_SUPER_CLASSES)
    return meta, dict(meta_rows=len(rows), neurons=len(meta), excluded=dict(excluded))


def edge_matrix(ids, pre, post, count, min_synapses=b.MIN_SYNAPSES):
    """CSR ``[pre, post]`` over ``ids``: records between listed neurons, duplicates summed, pairs below
    ``min_synapses`` dropped (self pairs are removed afterwards by ``annotation_v32.data._from_matrix``)."""
    ids, pre, post, count = (np.asarray(x, dtype=np.int64) for x in (ids, pre, post, count))
    if np.any(count <= 0):
        raise ValueError("nonpositive synapse count")
    i, j = np.searchsorted(ids, pre), np.searchsorted(ids, post)
    valid = (i < len(ids)) & (j < len(ids))
    valid &= (ids[np.minimum(i, len(ids) - 1)] == pre) & (ids[np.minimum(j, len(ids) - 1)] == post)
    matrix = sparse.coo_matrix((count[valid], (i[valid], j[valid])), shape=(len(ids), len(ids))).tocsr()  # sums duplicate records
    matrix.data[matrix.data < min_synapses] = 0
    matrix.eliminate_zeros()
    return matrix, valid


def build_graph(meta, vocabulary, lookup):
    ids = np.asarray(sorted(meta), dtype=np.int64)
    table = feather.read_table(b.RAW / "banc_888_edgelist_simple_v2.feather", columns=["pre", "post", "count"])
    pre = pc.cast(table["pre"], "int64").to_numpy()
    post = pc.cast(table["post"], "int64").to_numpy()
    count = table["count"].to_numpy().astype(np.int64)
    matrix, valid = edge_matrix(ids, pre, post, count)
    o = np.lexsort((post, pre))
    duplicates = int(((pre[o][1:] == pre[o][:-1]) & (post[o][1:] == post[o][:-1])).sum())
    stats = dict(edgelist_rows=int(len(pre)), rows_between_neurons=int(valid.sum()), self_rows=int((pre == post).sum()),
                 duplicate_pair_rows=duplicates)
    metadata = dict(dataset=b.RELEASE, historical_hints=False, min_synapses=b.MIN_SYNAPSES, aliases=TYPE_ALIASES,
                    annotation_sha256=hash_file(b.RAW / "banc_888_meta.feather"),
                    edgelist_sha256=hash_file(b.RAW / "banc_888_edgelist_simple_v2.feather"),
                    universe=PROTOCOL["universe"], groups=b.GROUPS)
    public = _from_matrix(ids, matrix, {int(x): None for x in ids}, vocabulary, metadata)  # also removes self pairs
    own = {int(c): None if b.banc_type(meta[int(c)]["cell_type"]) is None else lookup.get(b.banc_type(meta[int(c)]["cell_type"]), 0)
           for c in ids}
    private = PrivateAnnotations(own, {"role": "validation", "release": b.RELEASE, "label_rule": "own cell_type name (rule 1 only)"})
    save_prepared_dataset(b.PREPARED, public, private)
    stats.update(pairs_at_least_5=int(len(public.indices)), nodes=int(len(ids)))
    return public, own, stats


def labels_of(ids, meta, lookup, own):
    labels = np.full(len(ids), -1, dtype=np.int32)
    rule = np.full(len(ids), 2, dtype=np.int8)
    counts = Counter()
    for k, cell in enumerate(ids.tolist()):
        r = meta[cell]
        labels[k], kind = b.crosswalk(r["cell_type"], r["hemibrain_cell_type"], lookup)
        rule[k] = ("release-type", "hemibrain-field", "unresolved").index(kind)
        counts[kind] += 1
        if own[cell] is not None and own[cell] > 0 and own[cell] != labels[k]:
            raise ValueError("crosswalk contradicts the own-name reference")
    return labels, rule, dict(counts)


def view_arrays(ids, meta, labels, rule):
    fields = dict(release="cell_type", declared="hemibrain_cell_type", side="side", fafb="fafb_cell_type",
                  malecns="malecns_cell_type", super_class="super_class", region="region", top_nt="neurotransmitter_predicted")
    arrays, names = dict(ids=ids, labels=labels, rule=rule), {}
    for key, column in fields.items():
        values = [meta[c][column] or None for c in ids.tolist()]
        uniq = sorted({x for x in values if x is not None})
        index = {x: i for i, x in enumerate(uniq)}
        arrays[key + "_code"] = np.array([-1 if x is None else index[x] for x in values], dtype=np.int32)
        names[key] = uniq
    arrays["proofread"] = np.array([meta[c]["proofread"] == "TRUE" for c in ids.tolist()], dtype=bool)
    names["rule"] = ["release-type", "hemibrain-field", "unresolved"]
    return arrays, names


def chained_codes(ids, meta, lookup):
    """Diagnostic only: hemibrain code reachable by chaining fafb_cell_type through the FlyWire v3.1.0 table."""
    per_type = {}
    with (ROOT / "inputs/v32/flywire/cq-flywire-v3.1.0.tsv").open(newline="") as stream:
        for r in csv.DictReader(stream, delimiter="\t"):
            code, _ = v35_build.crosswalk(r["cell_type"], r["hemibrain_type"], "flywire:v3.1.0", lookup)
            if r["cell_type"]:
                per_type.setdefault(r["cell_type"], set()).add(code)
    out = np.full(len(ids), -1, dtype=np.int32)
    for k, cell in enumerate(ids.tolist()):
        name = meta[cell]["fafb_cell_type"]
        codes = per_type.get(name or "", set())
        if name and not name.startswith("auto:") and len(codes) == 1 and min(codes) >= 0:
            out[k] = min(codes)
    return out


def build_questions(vocabulary, ids, matrix, labels, full, v35):
    indegree = np.diff(matrix.indptr)
    present = np.bincount(labels[labels >= 0], minlength=len(vocabulary)) > 0
    prior_names = ["mlp_s0", "const_half", "const_share"]
    target_cell, target_type, target_offset = [], [], [0]
    pre_cell, pre_weight, pre_label = [], [], []
    fam_target, fam_kind, fam_source, fam_total, fam_code_offset, fam_codes = [], [], [], [], [0], []
    plus = {p: [] for p in prior_names}
    unknown = {p: [] for p in prior_names}
    types = []
    for t in v35["types"]:  # the v35 roster order
        code = int(t["code"])
        cells = np.flatnonzero((labels == code) & (indegree >= v35_build.PROTOCOL["minimum_partners"]))
        cells = sorted(cells.tolist(), key=lambda i: b.endpoint_key(vocabulary[code], int(ids[i])))
        chosen = cells[: v35_build.PROTOCOL["endpoints_per_type"]]
        families = [dict(f, absent_in_banc=[vocabulary[x] for x in f["codes"] if not present[x]]) for f in t["families"]]
        types.append(dict(t, families=families, banc_neurons=int((labels == code).sum()), banc_candidates=len(cells),
                          banc_targets=[int(ids[i]) for i in chosen]))
        for i in chosen:
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
            for f in t["families"]:
                codes = [int(x) for x in f["codes"]]
                fam_target.append(target)
                fam_kind.append(0 if f["kind"] == "top1" else 1)
                fam_source.append(int(f["source_weight"]))
                fam_total.append(int(t["source_total_weight"]))
                fam_codes.extend(codes)
                fam_code_offset.append(len(fam_codes))
                block = full[np.ix_(pre, codes)].astype(np.float64)
                plus["mlp_s0"].append(block.sum(1).astype(np.float32))
                plus["const_half"].append(np.full(len(pre), 0.5, dtype=np.float32))
                share = float(np.clip(int(f["source_weight"]) / int(t["source_total_weight"]), 0.01, 0.99))
                plus["const_share"].append(np.full(len(pre), share, dtype=np.float32))
            unknown["mlp_s0"].append(full[pre, 0])
            unknown["const_half"].append(np.zeros(len(pre), dtype=np.float32))
            unknown["const_share"].append(np.zeros(len(pre), dtype=np.float32))
    arrays = dict(
        target_cell=np.asarray(target_cell, dtype=np.int64),
        target_type=np.asarray(target_type, dtype=np.int32),
        target_offset=np.asarray(target_offset, dtype=np.int64),
        pre_cell=np.concatenate(pre_cell),
        pre_weight=np.concatenate(pre_weight),
        pre_label=np.concatenate(pre_label).astype(np.int32),
        fam_target=np.asarray(fam_target, dtype=np.int32),
        fam_kind=np.asarray(fam_kind, dtype=np.int8),
        fam_source_weight=np.asarray(fam_source, dtype=np.int64),
        fam_source_total=np.asarray(fam_total, dtype=np.int64),
        fam_code_offset=np.asarray(fam_code_offset, dtype=np.int64),
        fam_codes=np.asarray(fam_codes, dtype=np.int32),
    )
    for p in prior_names:
        arrays["plus_" + p] = np.concatenate(plus[p]).astype(np.float32)
        arrays["unknown_" + p] = np.concatenate(unknown[p]).astype(np.float32)
    return arrays, prior_names, types


def quantiles(x):
    x = np.asarray(x, dtype=float)
    return {q: float(np.quantile(x, p)) for q, p in (("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9), ("max", 1.0))} | dict(mean=float(x.mean()), n=int(len(x)))


def summaries(arrays, types, ids, labels, rule, chained, vocabulary):
    """Answer-blind sizes for every target; reference coverage of input cells for development targets only."""
    from connectomequest.neurosym import common as nc

    v35m = json.loads(b.V35_MANIFEST.read_text())
    split = {int(t["code"]): t["split"] for t in v35m["types"]}
    sizes = np.diff(arrays["target_offset"])
    tsplit = np.array([split[int(t)] for t in arrays["target_type"]])
    totals = np.add.reduceat(arrays["pre_weight"], arrays["target_offset"][:-1]) if len(sizes) else np.array([])
    out = dict(
        neurons=dict(total=int(len(ids)), resolved=int((labels >= 0).sum()), by_rule={n: int((rule == k).sum()) for k, n in enumerate(("release-type", "hemibrain-field", "unresolved"))},
                     chained_flywire_extra=int(((labels < 0) & (chained >= 0)).sum())),
        types=dict(manifest=len(types), with_targets=sum(bool(t["banc_targets"]) for t in types),
                   with_targets_by_split=dict(Counter(t["split"] for t in types if t["banc_targets"])),
                   without_targets_by_split=dict(Counter(t["split"] for t in types if not t["banc_targets"])),
                   families_with_absent_partner=sum(bool(f["absent_in_banc"]) for t in types for f in t["families"]),
                   absent_partner_names=sorted({x for t in types for f in t["families"] for x in f["absent_in_banc"]})),
        targets=dict(total=int(len(sizes)), by_split=dict(Counter(tsplit.tolist())), input_cells=quantiles(sizes), input_synapses=quantiles(totals),
                     input_cells_development=quantiles(sizes[tsplit == "development"]), input_cells_at_least_500=int((sizes >= 500).sum())),
    )
    # same types in the other views (answer-blind): input cells of their targets
    for view in ("malecns", "flywire"):
        d = nc.benchmark(view)
        sz = np.diff(d["target_offset"])
        out["targets"][f"input_cells_{view}"] = quantiles(sz)
    # development targets: reference coverage of input cells
    dev = np.flatnonzero(tsplit == "development")
    lab, w = arrays["pre_label"], arrays["pre_weight"]
    cells = np.concatenate([np.arange(arrays["target_offset"][t], arrays["target_offset"][t + 1]) for t in dev])
    rows = np.searchsorted(ids, arrays["pre_cell"][cells])
    gain = (lab[cells] < 0) & (chained[rows] >= 0)
    cover = dict(banc=dict(targets=int(len(dev)), input_cells=int(len(cells)), resolved_cells=float((lab[cells] >= 0).mean()),
                           resolved_synapses=float(w[cells][lab[cells] >= 0].sum() / w[cells].sum()),
                           by_rule_cells={n: float((rule[rows] == k).mean()) for k, n in enumerate(("release-type", "hemibrain-field", "unresolved"))},
                           chained_flywire_extra_cells=float(gain.mean()),
                           chained_flywire_extra_synapses=float(w[cells][gain].sum() / w[cells].sum())))
    for view in ("malecns", "flywire"):
        d = nc.benchmark(view)
        tt = d["target_type"]
        devt = np.flatnonzero([split[int(t)] == "development" for t in tt])
        c2 = np.concatenate([np.arange(d["target_offset"][t], d["target_offset"][t + 1]) for t in devt])
        cover[view] = dict(targets=int(len(devt)), input_cells=int(len(c2)), resolved_cells=float((d["pre_label"][c2] >= 0).mean()),
                           resolved_synapses=float(d["pre_weight"][c2][d["pre_label"][c2] >= 0].sum() / d["pre_weight"][c2].sum()))
    out["coverage_development"] = cover
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    started = time.time()
    for path in (b.PREPARED, b.BENCH, b.MANIFEST, b.VIEWFILE, b.VIEWNAMES, b.OUT / "features.npz", b.OUT / "features-identity.json", b.OUT / "build.json"):
        if path.exists():
            raise FileExistsError(path)
    raw = check_raw()
    vocabulary = json.loads((ROOT / v35_build.SOURCE / "public/manifest.json").read_text())["vocabulary"]
    lookup = {name: i for i, name in enumerate(vocabulary) if i}
    v35 = json.loads(b.V35_MANIFEST.read_text())
    meta, meta_stats = load_meta()
    public, own, graph_stats = build_graph(meta, vocabulary, lookup)
    print("graph", graph_stats, flush=True)
    ids, matrix = v35_build.load_graph(b.PREPARED)
    if not np.array_equal(ids, public.node_ids):
        raise AssertionError("prepared graph reloads differently")
    features = structural_features(public)
    identity = dict(implementation=inspect.getsource(structural_features),
                    public={f"public/{n}": hash_file(b.PREPARED / "public" / n) for n in ("manifest.json", "nodes.parquet", "topology.npz")})
    b.save_new(b.OUT / "features.npz", lambda s: np.savez(s, node_ids=ids, features=features))
    b.save_new(b.OUT / "features-identity.json", lambda s: s.write((json.dumps(identity, indent=1, sort_keys=True) + "\n").encode()))
    labels, rule, crosswalk_counts = labels_of(ids, meta, lookup, own)
    chained = chained_codes(ids, meta, lookup)
    arrays_view, names = view_arrays(ids, meta, labels, rule)
    arrays_view["chained_flywire"] = chained
    b.save_new(b.VIEWFILE, lambda s: np.savez(s, **arrays_view))
    b.save_new(b.VIEWNAMES, lambda s: s.write(json.dumps(names).encode()))
    model = StructuralMLP.load(ROOT / v35_build.MLP / "model.npz")
    temperature = json.loads((ROOT / v35_build.MLP / "report.json").read_text())["temperature"]
    full = v35_build.full_probabilities(model, temperature, features)
    print("priors computed", round(time.time() - started), "s", flush=True)
    arrays, prior_names, types = build_questions(vocabulary, ids, matrix, labels, full, v35)
    del full
    b.save_new(b.BENCH, lambda s: np.savez(s, **arrays))
    view = dict(view=b.VIEW, nodes=int(len(ids)), crosswalk=crosswalk_counts, targets=int(len(arrays["target_cell"])),
                families=int(len(arrays["fam_target"])), presynaptic_pairs=int(arrays["target_offset"][-1]), priors=prior_names,
                sha256=hash_file(b.BENCH),
                inputs=dict(topology=hash_file(b.PREPARED / "public/topology.npz"), labels=hash_file(b.PREPARED / "private/labels.parquet"),
                            annotations=raw["banc_888_meta.feather"]["sha256"], edgelist=raw["banc_888_edgelist_simple_v2.feather"]["sha256"],
                            features=hash_file(b.OUT / "features.npz"), view=hash_file(b.VIEWFILE), mlp_s0_model=hash_file(ROOT / v35_build.MLP / "model.npz")))
    manifest = dict(protocol=PROTOCOL, code_sha256={str(Path(__file__).resolve().relative_to(ROOT)): hash_file(Path(__file__)),
                                                    "src/connectomequest/round5/banc.py": hash_file(ROOT / "src/connectomequest/round5/banc.py")},
                    v35_manifest_sha256=hash_file(b.V35_MANIFEST), raw=raw, view=view, types=types,
                    counts=dict(types=len(types), development=sum(t["split"] == "development" for t in types), test=sum(t["split"] == "test" for t in types)))
    b.save_new(b.MANIFEST, lambda s: s.write((json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()))
    report = dict(meta=meta_stats, graph=graph_stats, crosswalk=crosswalk_counts,
                  summaries=summaries(arrays, types, ids, labels, rule, chained, vocabulary), wall_seconds=time.time() - started)
    b.save_new(b.OUT / "build.json", lambda s: s.write((json.dumps(report, indent=1, sort_keys=True) + "\n").encode()))
    print(json.dumps({k: v for k, v in report.items() if k != "summaries"}, indent=1))
    print(json.dumps(report["summaries"], indent=1))


if __name__ == "__main__":
    main()
