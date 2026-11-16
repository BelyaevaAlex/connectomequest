#!/usr/bin/env python3
"""Build the V35 conserved-partner benchmark from public graphs and frozen predictors.

Everything that defines a question is answer-blind with respect to the evaluation
views: target types, the partner categories and the thresholds come from the
hemibrain source connectome; endpoints are chosen by a salted hash among cells of
the target type. All three graphs use the same five-synapse pair rule. Reference
labels follow one declared crosswalk for both views: the release's own type name
when it is in the hemibrain vocabulary, otherwise a single author-provided
hemibrain correspondence, otherwise unresolved. No fitting happens here.
"""

from __future__ import annotations

import argparse
import csv
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import pyarrow.parquet as pq
from scipy import sparse

from connectomequest.annotation_v32.data import ontology_type
from connectomequest.annotation_v32.predictors import score_probabilities
from connectomequest.annotation_v32.runtime import hash_file
from connectomequest.annotation_v32.training import StructuralMLP
from connectomequest.annotation_v34.linear import StructuralLinear

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "inputs/v32/prepared-common-v1/hemibrain"
MLP = "outputs/v32/predictors/59cd92072106eb705ce5ae288aa891a79ee47de74bbba3cc5dce30ae4fc38ea3"
LINEAR = "outputs/v34/predictors/506e892832babb70d1b737371958ae6a3d04580cde64913648e7cdcd8abaf7ba"
SEEDS = "outputs/v35/mlp-seeds"
VIEWS = dict(
    malecns=dict(
        root="inputs/v32/prepared-common-v1/malecns",
        release="male-cns:v1.0",
        annotations="data/raw/manc/body-annotations-male-cns-v1.0-minconf-0.5.feather",
        features="outputs/v32/predictors/_features/25433f45ef58a7c7dbefc05e63d6ec6d8ea42bb1bc3f8dea4d1ef5c3cfe283d9",
        ntac="outputs/v32/ntac-transfer/malecns/predictions/58ce9b3903e36438d7c59277b7c61f4eec6fbd39c981fd71b36b1b7bcffa74f2",
    ),
    flywire=dict(
        root="inputs/v32/prepared-common-v1-hints/flywire",
        release="flywire:v3.1.0",
        annotations="inputs/v32/flywire/cq-flywire-v3.1.0.tsv",
        features="outputs/v32/predictors/_features/f84b836f1c031fe13eab4379fcf47130c375501dfc69b1081bdd70de94ab05cb",
        ntac="outputs/v32/ntac-transfer/flywire-hints/predictions/b23e6ca59d418851996773ab8cd7bdc2d0afebd01070490737b2f0d42b043f60",
    ),
)
PROTOCOL = dict(
    schema="cq-v35-conserved-partner-benchmark-v1",
    minimum_source_weight=100,
    minimum_partners=10,
    endpoints_per_type=2,
    coalition_share=[1, 2],
    coalition_max_types=20,
    development_percent=25,
    forced_development_prefixes=["MBON"],
    provisional_alpha=0.9,
    mlp_seeds=[0, 1, 2, 3, 4],
    graph_rule="pairs with at least five synapses in all three connectomes; self pairs excluded",
    crosswalk="release type if in the hemibrain vocabulary, else one single-valued author hemibrain correspondence, else unresolved",
)


def digest(*parts):
    return sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def load_graph(root):
    with np.load(root / "public/topology.npz", allow_pickle=False) as z:
        ids, indptr, indices, weights = z["node_ids"], z["indptr"], z["indices"], z["weights"]
    n = len(ids)
    rows = np.repeat(np.arange(n), np.diff(indptr))
    keep = rows != indices
    matrix = sparse.csc_matrix((weights[keep], (rows[keep], indices[keep])), shape=(n, n))
    matrix.sum_duplicates()
    matrix.sort_indices()
    return ids, matrix


def crosswalk(primary, secondary, release, lookup):
    name = ontology_type(primary or None, release)
    if name in lookup:
        return lookup[name], "release-type"
    names = [ontology_type(x.strip(), release) for x in (secondary or "").split(",") if x.strip()]
    if len(names) == 1 and names[0] in lookup:
        return lookup[names[0]], "hemibrain-field"
    return -1, "unresolved"


def view_labels(root, view, ids, lookup):
    cfg = VIEWS[view]
    if view == "flywire":
        with (root / cfg["annotations"]).open(newline="") as stream:
            rows = {
                int(r["root_id"]): (r["cell_type"], r["hemibrain_type"])
                for r in csv.DictReader(stream, delimiter="\t")
            }
    else:
        table = feather.read_table(root / cfg["annotations"], columns=["bodyId", "type", "hemibrainType"])
        rows = {int(r["bodyId"]): (r["type"], r["hemibrainType"]) for r in table.to_pylist()}
    labels = np.full(len(ids), -1, dtype=np.int32)
    counts = {"release-type": 0, "hemibrain-field": 0, "unresolved": 0}
    for i, cell in enumerate(ids.tolist()):
        primary, secondary = rows.get(cell, (None, None))
        labels[i], kind = crosswalk(primary, secondary, cfg["release"], lookup)
        counts[kind] += 1
    frozen = pq.read_table(root / cfg["root"] / "private/labels.parquet").to_pylist()
    frozen = {int(r["cell_id"]): r["label"] for r in frozen}
    for cell, mine in zip(ids.tolist(), labels.tolist()):
        old = frozen[cell]
        if old is not None and old > 0 and old != mine:
            raise ValueError("declared crosswalk contradicts the frozen reference")
    return labels, counts


def source_statistics(root, lookup_size):
    ids, matrix = load_graph(root / SOURCE)
    rows = pq.read_table(root / SOURCE / "private/labels.parquet").to_pylist()
    table = {int(r["cell_id"]): r["label"] for r in rows}
    labels = np.array([-1 if table[c] is None or table[c] == 0 else table[c] for c in ids.tolist()])
    typed = labels >= 0
    onehot = sparse.csr_matrix(
        (np.ones(int(typed.sum())), (np.flatnonzero(typed), labels[typed])), shape=(len(ids), lookup_size)
    )
    by_type = (onehot.T @ matrix.tocsr() @ onehot).tocsc()
    total = np.asarray((matrix.tocsr() @ onehot).sum(0)).ravel()
    counts = np.bincount(labels[typed], minlength=lookup_size)
    return by_type, total, counts


def families_for(code, by_type, candidate):
    column = by_type.getcol(code)
    partners, weights = column.indices, np.asarray(column.data, dtype=np.int64)
    typed_total = int(weights.sum())
    keep = candidate[partners] & (weights > 0)
    partners, weights = partners[keep], weights[keep]
    if not len(partners):
        return [], typed_total
    order = np.lexsort((partners, -weights))
    partners, weights = partners[order], weights[order]
    num, den = PROTOCOL["coalition_share"]
    cumulative = np.cumsum(weights)
    size = int(np.searchsorted(cumulative * den, typed_total * num, side="left")) + 1
    size = min(size, len(partners), PROTOCOL["coalition_max_types"])
    families = [dict(kind="top1", codes=[int(partners[0])], source_weight=int(weights[0]))]
    if size > 1:
        families.append(
            dict(kind="coalition", codes=[int(c) for c in partners[:size]], source_weight=int(cumulative[size - 1]))
        )
    return families, typed_total


def full_probabilities(model, temperature, features, chunk=8192):
    """Exact frozen probabilities for every node, computed once per cell."""
    out = np.empty((len(features), model.label_count), dtype=np.float32)
    for start in range(0, len(features), chunk):
        out[start : start + chunk] = score_probabilities(model.logits(features[start : start + chunk]), temperature)
    return out


def predictors(root, view, ids):
    cfg = VIEWS[view]
    with np.load(root / cfg["features"] / "features.npz", allow_pickle=False) as z:
        feature_ids, features = z["node_ids"], z["features"]
    features = features[rows_of(feature_ids, ids)]
    models = {"mlp_s0": (StructuralMLP.load(root / MLP / "model.npz"), json.loads((root / MLP / "report.json").read_text())["temperature"])}
    for seed in PROTOCOL["mlp_seeds"][1:]:
        report = json.loads((root / SEEDS / f"seed{seed}" / "report.json").read_text())
        models[f"mlp_s{seed}"] = (StructuralMLP.load(root / SEEDS / f"seed{seed}" / "model.npz"), report["summary"]["temperature"])
    models["linear"] = (StructuralLinear.load(root / LINEAR / "model.npz"), json.loads((root / LINEAR / "report.json").read_text())["temperature"])
    full = {name: full_probabilities(model, temperature, features) for name, (model, temperature) in models.items()}
    ntac_ids = np.load(root / cfg["ntac"] / "node_ids.npy", allow_pickle=False)
    ntac = np.load(root / cfg["ntac"] / "probabilities.npy", allow_pickle=False)
    full["ntac"] = ntac[rows_of(ntac_ids, ids)]
    return full


def rows_of(ids, cells):
    rows = np.searchsorted(ids, cells)
    if np.any(rows >= len(ids)) or not np.array_equal(ids[rows], cells):
        raise ValueError("cell absent from a frozen predictor input")
    return rows


def build_view(root, view, roster, families, source_total, vocabulary, output):
    cfg = VIEWS[view]
    ids, matrix = load_graph(root / cfg["root"])
    lookup = {name: i for i, name in enumerate(vocabulary) if i}
    labels, crosswalk_counts = view_labels(root, view, ids, lookup)
    full = predictors(root, view, ids)
    provisional = None
    if view == "flywire":
        table = pq.read_table(root / cfg["root"] / "public/nodes.parquet", columns=["cell_id", "provisional"]).to_pylist()
        hints = {int(r["cell_id"]): r["provisional"] for r in table}
        provisional = np.array([-1 if hints[c] is None else hints[c] for c in ids.tolist()], dtype=np.int32)
        alpha = PROTOCOL["provisional_alpha"]
        mixed = full["mlp_s0"].copy()
        hinted = np.flatnonzero(provisional >= 0)
        mixed[hinted] *= np.float32(1 - alpha)
        mixed[hinted, provisional[hinted]] += np.float32(alpha)
        full["provisional"] = mixed
    prior_names = list(full)
    indegree = np.diff(matrix.indptr)
    target_cell, target_type, target_offset = [], [], [0]
    pre_cell, pre_weight, pre_label = [], [], []
    fam_target, fam_kind, fam_source, fam_code_offset, fam_codes = [], [], [], [0], []
    p_plus = {name: [] for name in prior_names}
    p_unknown = {name: [] for name in prior_names}
    for code in roster:
        cells = np.flatnonzero((labels == code) & (indegree >= PROTOCOL["minimum_partners"]))
        cells = sorted(cells.tolist(), key=lambda i: digest("cq-v35-endpoint", view, vocabulary[code], int(ids[i])))
        for i in cells[: PROTOCOL["endpoints_per_type"]]:
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
            for family in families[code]:
                fam_target.append(target)
                fam_kind.append(0 if family["kind"] == "top1" else 1)
                fam_source.append(family["source_weight"])
                fam_codes.extend(family["codes"])
                fam_code_offset.append(len(fam_codes))
                for name in prior_names:
                    block = full[name][np.ix_(pre, family["codes"])].astype(np.float64)
                    p_plus[name].append(block.sum(1).astype(np.float32))
            for name in prior_names:
                p_unknown[name].append(full[name][pre, 0])
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
        fam_source_total=np.asarray([source_total[target_type[t]] for t in fam_target], dtype=np.int64),
        fam_code_offset=np.asarray(fam_code_offset, dtype=np.int64),
        fam_codes=np.asarray(fam_codes, dtype=np.int32),
    )
    for name in prior_names:
        arrays["plus_" + name] = np.concatenate(p_plus[name])
        arrays["unknown_" + name] = np.concatenate(p_unknown[name])
    with (output / f"{view}.npz").open("xb") as stream:
        np.savez(stream, **arrays)
    return dict(
        view=view,
        nodes=len(ids),
        crosswalk=crosswalk_counts,
        targets=len(target_cell),
        families=len(fam_target),
        presynaptic_pairs=int(target_offset[-1]),
        priors=prior_names,
        sha256=hash_file(output / f"{view}.npz"),
        inputs={
            "topology": hash_file(root / cfg["root"] / "public/topology.npz"),
            "annotations": hash_file(root / cfg["annotations"]),
            "features": hash_file(root / cfg["features"] / "features.npz"),
            "ntac": hash_file(root / cfg["ntac"] / "manifest.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v35/benchmark")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    vocabulary = json.loads((ROOT / SOURCE / "public/manifest.json").read_text())["vocabulary"]
    lookup = {name: i for i, name in enumerate(vocabulary) if i}
    by_type, source_total, source_counts = source_statistics(ROOT, len(vocabulary))
    present = {}
    for view, cfg in VIEWS.items():
        ids, matrix = load_graph(ROOT / cfg["root"])
        labels, _ = view_labels(ROOT, view, ids, lookup)
        indegree = np.diff(matrix.indptr)
        present[view] = dict(
            any=np.bincount(labels[labels >= 0], minlength=len(vocabulary)) > 0,
            endpoint=np.bincount(
                labels[(labels >= 0) & (indegree >= PROTOCOL["minimum_partners"])], minlength=len(vocabulary)
            )
            > 0,
        )
    candidate = (source_counts > 0) & present["malecns"]["any"] & present["flywire"]["any"]
    candidate[0] = False
    roster, families, typed_totals = [], {}, {}
    for code in np.flatnonzero(candidate & present["malecns"]["endpoint"] & present["flywire"]["endpoint"]).tolist():
        if source_total[code] < PROTOCOL["minimum_source_weight"]:
            continue
        found, typed_total = families_for(code, by_type, candidate)
        if found:
            roster.append(code)
            families[code] = found
            typed_totals[code] = typed_total
    roster.sort(key=lambda c: digest("cq-v35-roster", vocabulary[c]))

    def split(name):
        forced = any(name.startswith(p) for p in PROTOCOL["forced_development_prefixes"])
        bucket = int(digest("cq-v35-split", name)[:8], 16) % 100
        return "development" if forced or bucket < PROTOCOL["development_percent"] else "test"

    types = [
        dict(
            code=c,
            name=vocabulary[c],
            split=split(vocabulary[c]),
            source_cells=int(source_counts[c]),
            source_total_weight=int(source_total[c]),
            source_typed_weight=int(typed_totals[c]),
            families=[dict(f, names=[vocabulary[x] for x in f["codes"]]) for f in families[c]],
        )
        for c in roster
    ]
    views = [build_view(ROOT, view, roster, families, source_total, vocabulary, args.output) for view in VIEWS]
    manifest = dict(
        protocol=PROTOCOL,
        code_sha256=hash_file(Path(__file__)),
        source={"topology": hash_file(ROOT / SOURCE / "public/topology.npz"), "labels": hash_file(ROOT / SOURCE / "private/labels.parquet")},
        models={"mlp_s0": hash_file(ROOT / MLP / "model.npz"), "linear": hash_file(ROOT / LINEAR / "model.npz"),
                **{f"mlp_s{s}": hash_file(ROOT / SEEDS / f"seed{s}" / "model.npz") for s in PROTOCOL["mlp_seeds"][1:]}},
        types=types,
        views=views,
        counts=dict(types=len(types), development=sum(t["split"] == "development" for t in types), test=sum(t["split"] == "test" for t in types)),
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(counts=manifest["counts"], views=[{k: v for k, v in x.items() if k != "inputs"} for x in views])))


if __name__ == "__main__":
    main()
