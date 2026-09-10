#!/usr/bin/env python3
"""Single-source paper tables with protocol, row, arithmetic and primary checks."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from audit_paper_integrity import validate_cells, validate_queries, validate_summary
from run_graph_evaluation import METHODS, PRIMARY, ROOT

from connectomequest.confirmatory_v2 import query_path
from connectomequest.embodied.protocol import verify_archived_source_hashes
from connectomequest.manifest import sha256_file
from connectomequest.query import entity_split

LABEL = {
    "weight": "Weight",
    "random": "Random",
    "minerva": r"\shortstack{MINERVA-\\inspired}",
    "snapshot_v2": "ID-free",
    "gate_v2": "Gated",
    "random_snapshot_v2": "Random+ID-free",
}
GRAPHS = {"h01": "H01", "manc": "MANC", "hemibrain": "HemiBrain"}


def table(caption, label, cols, header, rows):
    return "\n".join(
        [
            r"\begin{table}[h]",
            r"\centering\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\caption{" + caption + "}",
            r"\label{" + label + "}",
            r"\begin{tabular}{" + cols + "}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def paired_ci(rows, left_key, right_key, seed=172903):
    by_query = {}
    for r in rows:
        by_query.setdefault(r["query_id"], []).append(float(r[right_key]) - float(r[left_key]))
    diffs = np.array([np.mean(v) for v in by_query.values()])
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(diffs, len(diffs)).mean() for _ in range(5000)])
    return (
        float(diffs.mean()),
        np.quantile(boot, [0.025, 0.975]).tolist(),
        np.quantile(boot, [0.0125, 0.9875]).tolist(),
    )


def generate():
    outputs = {}
    taxonomy = []
    coarse_main = []
    coarse_extra = []
    audit = {"primary_mismatches": 0, "cells": 0}
    manifest_seeds = {
        g: json.loads(query_path(REPO, g).with_suffix(".manifest.json").read_text())["seed"]
        for g in GRAPHS
    }
    assert len(set(manifest_seeds.values())) == 1
    outputs["protocol_numbers.tex"] = (
        r"\newcommand{\PrimaryQuerySeed}{" + str(next(iter(manifest_seeds.values()))) + "}\n"
    )
    audit["primary_query_manifest_seeds"] = manifest_seeds
    for graph, label in GRAPHS.items():
        directory = ROOT / "full" / graph / "taxonomy"
        result = json.loads((directory / "aggregate.json").read_text())
        assert result["protocol"] == json.loads((directory / "PROTOCOL.json").read_text())
        assert result["rows_sha256"] == sha256_file(directory / "rows.parquet")
        rows = pq.read_table(directory / "rows.parquet").to_pylist()
        ids = (
            pq.read_table(
                query_path(REPO, graph),
                columns=["query_id"],
                filters=[("split", "=", "test"), ("query_type", "=", "2pt")],
            )
            .column("query_id")
            .to_pylist()
        )
        cells = validate_cells(rows, METHODS, [17, 29, 43], ids, 64, 64)
        validate_summary(cells, result["summary"])
        for m in METHODS:
            subset = [r for r in rows if r["method"] == m]
            for seed in sorted({r["seed"] for r in subset}):
                primary = {
                    r["query_id"]: r
                    for r in pq.read_table(
                        PRIMARY / graph / f"{m}-b64-seed{seed}.parquet"
                    ).to_pylist()
                }
                cell = [r for r in subset if r["seed"] == seed]
                assert len(cell) == len(primary) == 5000
                assert len({r["query_id"] for r in cell}) == 5000
                for r in cell:
                    assert r["goal_success"] == primary[r["query_id"]]["goal_success"]
                    assert r["budget_used"] == primary[r["query_id"]]["budget_used"]
                audit["cells"] += 1
            d = result["summary"][m]
            rates = d["failure_rates"]
            assert abs(sum(rates.values()) - 1) < 1e-12
            assert abs(rates.get("success", 0) - d["success"]) < 1e-12
            values = [
                rates.get(k, 0)
                for k in [
                    "a_no_first_page_witness",
                    "b_no_useful_middle_traversed",
                    "c_selected_no_certificate",
                    "d_resource_reserve",
                ]
            ]
            assert abs(sum(values) + d["success"] - 1) < 1e-12
            taxonomy.append(
                f"{label} & {LABEL[m]} & {d['success']:.4f} & "
                + " & ".join(f"{v:.4f}" for v in values)
                + f" & {d['budget']:.2f} & {d['distinct_middles']:.2f} "
                + r"\\"
            )
        if graph == "h01":
            continue
        directory = ROOT / "full" / graph / "coarse"
        result = json.loads((directory / "aggregate.json").read_text())
        assert result["protocol"] == json.loads((directory / "PROTOCOL.json").read_text())
        assert result["rows_sha256"] == sha256_file(directory / "rows.parquet")
        rows = pq.read_table(directory / "rows.parquet").to_pylist()
        cells = validate_cells(rows, METHODS, [17, 29, 43], ids, 64, 64)
        validate_summary(cells, result["summary"])
        d = result["summary"]
        sample = [r for r in rows if r["method"] == "weight"]
        assert len(sample) == 5000
        density = np.mean([r["native_relevance_fraction"] for r in sample])
        coarse_density = np.mean([r["evaluated_relevance_fraction"] for r in sample])
        coarse_main.append(
            f"{label} & {density:.4f}/{coarse_density:.4f} & "
            + " & ".join(
                f"{d[m]['primary_success']:.4f}/{d[m]['success']:.4f}"
                for m in ["weight", "random", "snapshot_v2"]
            )
            + r" \\"
        )
        for method in METHODS:
            selected = [r for r in rows if r["method"] == method]
            delta, ci, _ = paired_ci(selected, "primary_success", "goal_success")
            coarse_extra.append(
                f"{label} & {LABEL[method]} & {d[method]['primary_success']:.4f} & {d[method]['success']:.4f} & {delta:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] "
                + r"\\"
            )
    outputs["taxonomy_v2.tex"] = table(
        "Controller-matched primary telemetry: 5,000 queries per graph, seeds 17/29/43 "
        "(Weight once). Categories plus Success sum to one. Middle count is based on actual traversals, not inferred from evidence.",
        "tab:failure-taxonomy",
        "llrrrrrrr",
        "Graph & Policy & Success & (a) & (b) & (c) & (d) & Budget & Middles",
        taxonomy,
    )
    outputs["paired_coarse_main.tex"] = table(
        "Paired label-coarsening audit, 5,000 original native queries per fly graph. "
        "Every entry is native/coarse7. Visible target prevalence is the relevant fraction among first-page two-hop candidate nodes; "
        "other columns are Success$(64)$, averaged over the original action/model seeds. No new model fitting.",
        "tab:ladder",
        "lrrrr",
        r"Graph & \shortstack{Visible target\\prevalence} & Weight & Random & ID-free",
        coarse_main,
    )
    outputs["paired_coarse_extra.tex"] = table(
        "Paired coarsening on the same anchors and pairing identifiers: all policies, "
        "paired bootstrap intervals over matched instances, conditional on three frozen model seeds. Frequency-balanced bins are "
        "synthetic unions, not ontological equivalence classes.",
        "tab:coarse-paired",
        "llrrl",
        "Graph & Policy & Native & Coarse7 & Difference [95\% CI]",
        coarse_extra,
    )
    fresh = REPO / "outputs/nemo/strong-accept/fresh-contract-test-v1"
    fresh_rows = []
    contrasts = []
    cfg = json.loads((fresh / "PROTOCOL.json").read_text())
    source_inputs = {name: digest for name, digest in cfg["inputs"].items() if name.endswith(".py")}
    verify_archived_source_hashes(REPO / "paper/artifact.zip", source_inputs)
    for name, digest in cfg["inputs"].items():
        if name.endswith(".py"):
            continue
        if sha256_file(REPO / name) != digest:
            raise ValueError(f"fresh input drift: {name}")
    for graph, label in GRAPHS.items():
        d = json.loads((fresh / graph / "aggregate.json").read_text())
        assert d["protocol_sha256"] == sha256_file(fresh / "PROTOCOL.json")
        assert d["rows_sha256"] == sha256_file(fresh / graph / "rows.parquet")
        assert d["query_sha256"] == sha256_file(fresh / graph / "QUERIES.json")
        all_rows = pq.read_table(fresh / graph / "rows.parquet").to_pylist()
        query_data = json.loads((fresh / graph / "QUERIES.json").read_text())
        query_seal = json.loads((fresh / graph / "QUERY-SEAL.json").read_text())
        if query_seal["sha256"] != sha256_file(fresh / graph / "QUERIES.json"):
            raise ValueError("fresh query seal mismatch")
        prior_pairs = set()
        for name, digest in cfg["prior_query_files"][graph].items():
            if sha256_file(REPO / name) != digest:
                raise ValueError(f"prior-query drift: {name}")
            for r in pq.read_table(REPO / name, columns=["query_type", "anchors"]).to_pylist():
                if r["query_type"] == "2pt":
                    prior_pairs.add(tuple(r["anchors"]))
        if (
            len(prior_pairs) != query_data["excluded_pairs"]
            or query_seal["overlap_prior_pairs"] != 0
        ):
            raise ValueError("fresh exclusion metadata mismatch")
        ids = validate_queries(
            query_data["queries"],
            cfg["n_per_graph"],
            prior_pairs,
            lambda anchor: entity_split(anchor, cfg["anchor_split_seed"]),
        )
        cells = validate_cells(all_rows, cfg["methods"], cfg["seeds"], ids, cfg["B"], cfg["H"])
        validate_summary(cells, d["summary"])
        fresh_rows.append(
            label
            + " & "
            + " & ".join(
                f"{d['summary'][m]:.4f}/{np.mean([r['budget_used'] for r in all_rows if r['method'] == m]):.2f}"
                for m in METHODS
            )
            + r" \\"
        )
        weight = {r["query_id"]: r["goal_success"] for r in all_rows if r["method"] == "weight"}
        rows = [
            {**r, "weight": weight[r["query_id"]]} for r in all_rows if r["method"] == "gate_v2"
        ]
        delta, ci, adjusted = paired_ci(rows, "weight", "goal_success")
        contrasts.append(
            {
                "graph": graph,
                "difference": delta,
                "ci95": ci,
                "bonferroni2": adjusted,
                "planned": graph != "h01",
            }
        )
    outputs["fresh_contract_test.tex"] = table(
        "Sealed new-pair follow-up evaluation: 1,000 previously unlisted "
        "query pairs per graph and frozen controller/checkpoints. Each entry is Success/charged actions, averaged across seeds 17/29/43 for stochastic policies; Weight is deterministic. All specified policies are reported. Excluding old pairs "
        "changes the sampling population; absolute rates are not directly comparable with the original test.",
        "tab:fresh-test",
        "lrrrrrr",
        r"Graph & Weight & Random & \shortstack{MINERVA-\\inspired} & ID-free & Gated & Random+ID-free",
        fresh_rows,
    )
    outputs["fresh_contract_ci.tex"] = (
        "\n".join(
            f"{GRAPHS[d['graph']]}: Gated minus Weight {d['difference']:+.4f}, nominal 95\\% CI "
            f"[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}]"
            + (
                f", two-contrast Bonferroni interval [{d['bonferroni2'][0]:+.4f},{d['bonferroni2'][1]:+.4f}]."
                if d["planned"]
                else " (descriptive)."
            )
            for d in contrasts
        )
        + "\n"
    )
    by_graph = {d["graph"]: d for d in contrasts}
    main_lines = []
    for g in ("manc", "hemibrain", "h01"):
        d = by_graph[g]
        main_lines.append(
            f"{GRAPHS[g]}: {d['difference']:+.4f} (95\\% CI [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}])."
        )
    positive = [GRAPHS[d["graph"]] for d in contrasts if d["planned"] and d["bonferroni2"][0] > 0]
    main_lines.append(
        "Among the two planned fly contrasts, "
        + (", ".join(positive) if positive else "neither graph")
        + " excludes zero after correction."
    )
    outputs["fresh_contract_main.tex"] = "\n".join(main_lines) + "\n"
    thresholds = []
    for graph, label in GRAPHS.items():
        vals = [
            torch.load(
                REPO / f"outputs/nemo/v2/gates/heldout-{graph}-seed{s}.pt",
                map_location="cpu",
                weights_only=True,
            )["confidence_threshold"]
            for s in (17, 29, 43)
        ]
        thresholds.append(label + ": " + ", ".join(f"{v:.2f}" for v in vals))
    outputs["gate_thresholds.tex"] = (
        "For seeds 17/29/43, frozen confidence thresholds are " + "; ".join(thresholds) + ".\n"
    )
    audit["fresh_contrasts"] = contrasts
    return outputs, audit


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    files, audit = generate()
    for name, text in files.items():
        path = REPO / "paper/generated" / name
        if a.check:
            assert path.read_text() == text, f"stale paper table: {name}"
        else:
            path.write_text(text)
    if not a.check:
        (ROOT / "PAPER-AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit))
