"""Hash-verified aggregation of the sealed NEmo v2 same-query matrix."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import pyarrow.parquet as pq

from connectomequest.confirmatory_evaluation_v2 import cell_stem
from connectomequest.confirmatory_v2 import (
    TEST_QUERIES,
    canonical_sha,
    confirmatory_root,
    load_method_selection,
    test_open_path,
    verify_matrix,
    verify_seal,
)
from connectomequest.manifest import sha256_file
from connectomequest.validation_report_v2 import paired_bootstrap_ci

PAIR_METRICS = {
    "success_at_budget": "goal_success",
    "proof_validity": "proof_valid",
    "mean_budget_used": "budget_used",
    "first_proof_mrr": "first_proof_reciprocal_rank",
}


def _summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected summary object: {path}")
    return payload


def _test_rows(path: Path) -> dict[str, dict[str, float]]:
    columns = ["query_id", *PAIR_METRICS.values(), "invalid_actions", "split", "query_type"]
    rows: dict[str, dict[str, float]] = {}
    for row in pq.read_table(path, columns=columns, memory_map=True).to_pylist():
        if row["split"] != "test" or row["query_type"] != "2pt":
            raise ValueError(f"non-confirmatory-test row in {path}")
        query_id = str(row["query_id"])
        if query_id in rows:
            raise ValueError(f"duplicate query id {query_id!r} in {path}")
        rows[query_id] = {
            report_name: float(row[column]) for report_name, column in PAIR_METRICS.items()
        }
        rows[query_id]["invalid_actions"] = float(row["invalid_actions"])
    if len(rows) != TEST_QUERIES:
        raise ValueError(f"expected {TEST_QUERIES} rows in {path}; found {len(rows)}")
    return rows


def _seed_mean(records: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    keys = records[0].keys()
    if any(record.keys() != keys for record in records[1:]):
        raise ValueError("seed runs do not contain identical same-query ids")
    return {
        query_id: {
            metric: mean(record[query_id][metric] for record in records)
            for metric in records[0][query_id]
        }
        for query_id in keys
    }


def aggregate_confirmatory(
    repo: Path,
    *,
    bootstrap_seed: int = 2_718_281,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    repo = repo.resolve()
    seal = verify_seal(repo, require_queries=True)
    if not test_open_path(repo).is_file():
        raise PermissionError("confirmatory results cannot exist before TEST-OPENED marker")
    selection = load_method_selection(repo)
    cells = verify_matrix(selection)
    indexed: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    query_ids_by_direction: dict[str, set[str]] = {}
    input_hashes = []
    for cell in cells:
        stem = cell_stem(confirmatory_root(repo), cell)
        parquet = stem.with_suffix(".parquet")
        summary_path = stem.with_suffix(".json")
        if not parquet.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"missing frozen confirmatory cell: {stem}")
        summary = _summary(summary_path)
        if summary.get("result_sha256") != sha256_file(parquet):
            raise ValueError(f"confirmatory result hash mismatch: {parquet}")
        protocol = summary.get("protocol", {})
        expected = {
            "partition": "confirmatory-test",
            "query_type": "2pt",
            "held_out": cell["held_out"],
            "method": cell["method"],
            "budget": cell["budget"],
            "seed": cell["seed"],
            "controller": cell["controller"],
            "seal_sha256": seal["seal_sha256"],
            "method_selection_sha256": selection["selection_sha256"],
        }
        wrong = {
            key: (protocol.get(key), value)
            for key, value in expected.items()
            if protocol.get(key) != value
        }
        if wrong:
            raise ValueError(f"confirmatory protocol mismatch in {summary_path}: {wrong}")
        run_spec = dict(protocol)
        run_digest = run_spec.pop("run_sha256", None)
        if run_digest != canonical_sha(run_spec):
            raise ValueError(f"invalid run digest: {summary_path}")
        rows = _test_rows(parquet)
        held_out = cell["held_out"]
        query_ids = set(rows)
        if held_out in query_ids_by_direction and query_ids_by_direction[held_out] != query_ids:
            raise ValueError(f"same-query alignment failed for {held_out}")
        query_ids_by_direction[held_out] = query_ids
        key = (held_out, int(cell["budget"]), cell["method"], int(cell["seed"]))
        indexed[key] = {"cell": cell, "summary": summary, "rows": rows}
        input_hashes.append((str(parquet.relative_to(repo)), sha256_file(parquet)))
        input_hashes.append((str(summary_path.relative_to(repo)), sha256_file(summary_path)))

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for (held_out, budget, method, _seed), record in indexed.items():
        grouped[(held_out, budget, method)].append(record)
    aggregates = []
    for (held_out, budget, method), records in sorted(grouped.items()):
        records.sort(key=lambda record: int(record["cell"]["seed"]))
        metric_names = tuple(records[0]["summary"]["overall"])
        metrics = {}
        for metric in metric_names:
            values = [float(record["summary"]["overall"][metric]) for record in records]
            metrics[metric] = {
                "mean": mean(values),
                "sample_sd": stdev(values) if len(values) > 1 else None,
                "n_seeds": len(values),
            }
        aggregates.append(
            {
                "held_out": held_out,
                "budget": budget,
                "method": method,
                "seeds": [int(record["cell"]["seed"]) for record in records],
                "metrics": metrics,
                "total_illegal_actions": int(
                    sum(
                        row["invalid_actions"]
                        for record in records
                        for row in record["rows"].values()
                    )
                ),
            }
        )

    comparisons = []
    comparison_index = 0
    for (held_out, budget, method), records in sorted(grouped.items()):
        if method == "weight":
            continue
        weight = indexed.get((held_out, budget, "weight", 17))
        if weight is None:
            raise ValueError(f"no same-budget Weight reference for {(held_out, budget, method)}")
        candidate = _seed_mean([record["rows"] for record in records])
        reference = weight["rows"]
        cis = {}
        for metric_index, metric in enumerate(PAIR_METRICS):
            cis[metric] = paired_bootstrap_ci(
                {query_id: values[metric] for query_id, values in candidate.items()},
                {query_id: values[metric] for query_id, values in reference.items()},
                seed=bootstrap_seed + comparison_index * 10 + metric_index,
                resamples=bootstrap_resamples,
            )
        comparisons.append(
            {
                "held_out": held_out,
                "budget": budget,
                "candidate": method,
                "reference": "weight",
                "candidate_seeds_averaged_within_query": [
                    int(record["cell"]["seed"]) for record in records
                ],
                "reference_seed": 17,
                "query_alignment": "exact 5000-query id equality",
                "paired_bootstrap": cis,
            }
        )
        comparison_index += 1
    return {
        "campaign": "nemo-2026-v2-confirmatory",
        "partition": "confirmatory-test",
        "protocol_version": 1,
        "tuning_authorization": False,
        "open_test_once": True,
        "same_query_evaluation": True,
        "seal_sha256": seal["seal_sha256"],
        "method_selection_sha256": selection["selection_sha256"],
        "matrix_cells": len(cells),
        "queries_per_connectome": TEST_QUERIES,
        "input_fingerprint_sha256": canonical_sha(sorted(input_hashes)),
        "bootstrap": {
            "method": "paired percentile bootstrap after within-query seed averaging",
            "confidence": 0.95,
            "resamples": bootstrap_resamples,
            "base_seed": bootstrap_seed,
        },
        "aggregates": aggregates,
        "paired_comparisons": comparisons,
    }
