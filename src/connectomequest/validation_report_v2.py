"""Deterministic, development-only aggregation for NEmo v2 validation.

The aggregator intentionally accepts only replication-validation artifacts.  It
never discovers or reads confirmatory/test files, and it verifies every reused
baseline against the frozen reuse manifest before computing statistics.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from connectomequest.manifest import sha256_file

DATASETS = ("h01", "manc", "hemibrain")
SEEDS = (17, 29, 43)
BUDGETS = (16, 32, 64)
BASELINES = ("weight", "random", "minerva")
METRICS = (
    "success_at_budget",
    "proof_validity",
    "mean_budget_used",
    "mean_steps",
    "invalid_action_rate",
    "mean_first_proof_reciprocal_rank",
    "precision",
    "recall",
    "f1",
    "provenance_completeness",
)
BOOTSTRAP_SEED = 20260811
BOOTSTRAP_RESAMPLES = 10_000


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _assert_development_summary(summary: dict[str, Any], path: Path) -> None:
    protocol = summary.get("protocol", {})
    partition = protocol.get("partition", protocol.get("split"))
    if partition not in {"replication-validation", "validation"}:
        raise ValueError(f"non-validation artifact rejected: {path}")
    if protocol.get("query_type") != "2pt":
        raise ValueError(f"non-2pt artifact rejected: {path}")


def _seed_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("cannot summarize an empty seed set")
    return {
        "mean": mean(numbers),
        "sample_sd": stdev(numbers) if len(numbers) > 1 else None,
        "n_seeds": len(numbers),
    }


def paired_bootstrap_ci(
    candidate: dict[str, float],
    reference: dict[str, float],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float | int]:
    """Percentile CI for a mean paired difference, aligned by query id."""

    if candidate.keys() != reference.keys():
        missing_candidate = sorted(reference.keys() - candidate.keys())
        missing_reference = sorted(candidate.keys() - reference.keys())
        raise ValueError(
            "query_id mismatch: "
            f"candidate_missing={missing_candidate[:3]}, "
            f"reference_missing={missing_reference[:3]}"
        )
    if not candidate:
        raise ValueError("paired bootstrap requires at least one query")
    if resamples < 2:
        raise ValueError("paired bootstrap requires at least two resamples")
    query_ids = sorted(candidate)
    differences = np.asarray(
        [candidate[query_id] - reference[query_id] for query_id in query_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    # Bound temporary allocations while keeping the exact RNG stream stable.
    chunk = max(1, min(1_000, resamples))
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, (0.025, 0.975))
    return {
        "estimate": float(differences.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "queries": len(query_ids),
        "resamples": resamples,
        "bootstrap_seed": seed,
    }


def _per_query(path: Path) -> dict[str, dict[str, float]]:
    columns = [
        "query_id",
        "goal_success",
        "proof_valid",
        "budget_used",
        "steps",
        "invalid_actions",
        "split",
        "query_type",
    ]
    table = pq.read_table(path, columns=columns, memory_map=True)
    rows: dict[str, dict[str, float]] = {}
    for row in table.to_pylist():
        if row["split"] != "validation" or row["query_type"] != "2pt":
            raise ValueError(f"non-validation/2pt row rejected: {path}")
        query_id = str(row["query_id"])
        if query_id in rows:
            raise ValueError(f"duplicate query_id {query_id!r}: {path}")
        rows[query_id] = {
            "success_at_budget": float(row["goal_success"]),
            "proof_validity": float(row["proof_valid"]),
            "mean_budget_used": float(row["budget_used"]),
            "mean_steps": float(row["steps"]),
            "invalid_actions_per_episode": float(row["invalid_actions"]),
        }
    if not rows:
        raise ValueError(f"empty result: {path}")
    return rows


def _mean_candidate_by_query(paths: list[Path]) -> dict[str, dict[str, float]]:
    per_seed = [_per_query(path) for path in paths]
    query_ids = per_seed[0].keys()
    if any(rows.keys() != query_ids for rows in per_seed[1:]):
        raise ValueError("candidate seed files have different query_id sets")
    metrics = tuple(next(iter(per_seed[0].values())))
    return {
        query_id: {metric: mean(rows[query_id][metric] for rows in per_seed) for metric in metrics}
        for query_id in query_ids
    }


def _validate_reuse_manifest(path: Path) -> tuple[dict[str, Any], dict[tuple, dict]]:
    manifest = _json(path)
    records = manifest.get("records")
    if not isinstance(records, list) or manifest.get("cells") != len(records):
        raise ValueError("malformed baseline reuse manifest")
    indexed: dict[tuple, dict] = {}
    for record in records:
        key = (
            record["held_out"],
            int(record["budget"]),
            record["method"],
            int(record["seed"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate baseline reuse cell: {key}")
        parquet = Path(record["parquet"])
        summary_path = Path(record["summary"])
        if sha256_file(parquet) != record["parquet_sha256"]:
            raise ValueError(f"baseline parquet SHA-256 mismatch: {parquet}")
        if sha256_file(summary_path) != record["summary_sha256"]:
            raise ValueError(f"baseline summary SHA-256 mismatch: {summary_path}")
        summary = _json(summary_path)
        _assert_development_summary(summary, summary_path)
        indexed[key] = {**record, "summary_payload": summary}
    return manifest, indexed


def _load_v2_cells(validation_root: Path) -> dict[tuple, dict]:
    indexed: dict[tuple, dict] = {}
    methods = {"gate-v2": (16, 32, 64), "snapshot-v2": (64,)}
    for dataset in DATASETS:
        for method, budgets in methods.items():
            for budget in budgets:
                for seed in SEEDS:
                    stem = validation_root / dataset / f"{method}-b{budget}-seed{seed}"
                    parquet = stem.with_suffix(".parquet")
                    summary_path = stem.with_suffix(".json")
                    if not parquet.is_file() or not summary_path.is_file():
                        raise FileNotFoundError(f"missing full v2 validation cell: {stem}")
                    summary = _json(summary_path)
                    _assert_development_summary(summary, summary_path)
                    if summary.get("result_sha256") != sha256_file(parquet):
                        raise ValueError(f"v2 result SHA-256 mismatch: {parquet}")
                    protocol = summary["protocol"]
                    expected_method = (
                        "contextual-gate-v2" if method == "gate-v2" else "snapshot-v2-ablation"
                    )
                    expected = {
                        "held_out": dataset,
                        "budget": budget,
                        "seed": seed,
                        "method": expected_method,
                    }
                    if any(protocol.get(key) != value for key, value in expected.items()):
                        raise ValueError(f"v2 protocol mismatch: {summary_path}")
                    indexed[(dataset, budget, method, seed)] = {
                        "parquet": str(parquet.resolve()),
                        "parquet_sha256": sha256_file(parquet),
                        "summary": str(summary_path.resolve()),
                        "summary_sha256": sha256_file(summary_path),
                        "summary_payload": summary,
                    }
    return indexed


def _aggregate_cells(
    records: dict[tuple, dict],
    *,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    budgets: tuple[int, ...],
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for dataset in datasets:
        for budget in budgets:
            for method in methods:
                matching = [
                    (key, record)
                    for key, record in records.items()
                    if key[0] == dataset and key[1] == budget and key[2] == method
                ]
                if not matching:
                    continue
                matching.sort(key=lambda item: item[0][3])
                summaries = [record["summary_payload"] for _, record in matching]
                metric_summary = {
                    metric: _seed_summary(summary["overall"][metric] for summary in summaries)
                    for metric in METRICS
                }
                total_illegal = 0
                total_episodes = 0
                fallback_values: list[float] = []
                selections: Counter[str] = Counter()
                for (_, record), summary in zip(matching, summaries, strict=True):
                    parquet_rows = _per_query(Path(record["parquet"]))
                    total_episodes += len(parquet_rows)
                    total_illegal += int(
                        sum(row["invalid_actions_per_episode"] for row in parquet_rows.values())
                    )
                    stats = summary.get("policy_stats")
                    if isinstance(stats, dict) and "fallback_fraction" in stats:
                        fallback_values.append(float(stats["fallback_fraction"]))
                        selections.update(stats.get("selected", {}))
                cell: dict[str, Any] = {
                    "held_out": dataset,
                    "budget": budget,
                    "method": method,
                    "seeds": [key[3] for key, _ in matching],
                    "metrics": metric_summary,
                    "proof_and_safety": {
                        "total_episodes": total_episodes,
                        "total_illegal_actions": total_illegal,
                    },
                    "probe_summary": {
                        "status": "not_instrumented_in_saved_episode_schema",
                        "exact_probe_count": None,
                        "configured_fixed_subgoal_probes": summaries[0]
                        .get("protocol", {})
                        .get("subgoal_probes"),
                        "max_subgoal_probes": summaries[0]
                        .get("protocol", {})
                        .get("max_subgoal_probes"),
                        "adaptive_probing": summaries[0]
                        .get("protocol", {})
                        .get("adaptive_probing"),
                        "cost_proxy": "metrics.mean_budget_used",
                    },
                }
                if fallback_values:
                    cell["fallback"] = {
                        "fraction": _seed_summary(fallback_values),
                        "selected_counts_across_seeds": dict(sorted(selections.items())),
                    }
                cells.append(cell)
    return cells


def aggregate_validation(
    validation_root: Path,
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Aggregate frozen development artifacts without evaluating any policies."""

    validation_root = validation_root.resolve()
    reuse_path = validation_root / "baseline-reuse.json"
    reuse_manifest, baseline_records = _validate_reuse_manifest(reuse_path)
    v2_records = _load_v2_cells(validation_root)
    baseline_cells = _aggregate_cells(
        baseline_records,
        datasets=DATASETS,
        methods=BASELINES,
        budgets=BUDGETS,
    )
    v2_cells = _aggregate_cells(
        v2_records,
        datasets=DATASETS,
        methods=("gate-v2", "snapshot-v2"),
        budgets=BUDGETS,
    )

    comparisons: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        weight = baseline_records[(dataset, 64, "weight", 17)]
        weight_rows = _per_query(Path(weight["parquet"]))
        for method_index, method in enumerate(("gate-v2", "snapshot-v2")):
            candidate_paths = [
                Path(v2_records[(dataset, 64, method, seed)]["parquet"]) for seed in SEEDS
            ]
            candidate_rows = _mean_candidate_by_query(candidate_paths)
            metric_cis = {}
            for metric_index, metric in enumerate(
                (
                    "success_at_budget",
                    "proof_validity",
                    "mean_budget_used",
                    "mean_steps",
                    "invalid_actions_per_episode",
                )
            ):
                metric_cis[metric] = paired_bootstrap_ci(
                    {query_id: values[metric] for query_id, values in candidate_rows.items()},
                    {query_id: values[metric] for query_id, values in weight_rows.items()},
                    seed=bootstrap_seed + dataset_index * 100 + method_index * 10 + metric_index,
                    resamples=bootstrap_resamples,
                )
            comparisons.append(
                {
                    "held_out": dataset,
                    "budget": 64,
                    "candidate": method,
                    "reference": "weight",
                    "candidate_seeds_averaged_within_query": list(SEEDS),
                    "reference_seed": 17,
                    "query_alignment": "exact query_id set equality",
                    "primary_metric": "success_at_budget",
                    "paired_bootstrap": metric_cis,
                }
            )

    secondary_comparisons: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        gate_rows = _mean_candidate_by_query(
            [Path(v2_records[(dataset, 64, "gate-v2", seed)]["parquet"]) for seed in SEEDS]
        )
        snapshot_rows = _mean_candidate_by_query(
            [Path(v2_records[(dataset, 64, "snapshot-v2", seed)]["parquet"]) for seed in SEEDS]
        )
        metric_cis = {}
        for metric_index, metric in enumerate(
            (
                "success_at_budget",
                "proof_validity",
                "mean_budget_used",
                "mean_steps",
                "invalid_actions_per_episode",
            )
        ):
            metric_cis[metric] = paired_bootstrap_ci(
                {query_id: values[metric] for query_id, values in gate_rows.items()},
                {query_id: values[metric] for query_id, values in snapshot_rows.items()},
                seed=bootstrap_seed + 1_000 + dataset_index * 10 + metric_index,
                resamples=bootstrap_resamples,
            )
        secondary_comparisons.append(
            {
                "held_out": dataset,
                "budget": 64,
                "candidate": "gate-v2",
                "reference": "snapshot-v2",
                "candidate_seeds_averaged_within_query": list(SEEDS),
                "reference_seeds_averaged_within_query": list(SEEDS),
                "query_alignment": "exact query_id set equality",
                "metric": "success_at_budget",
                "paired_bootstrap": metric_cis,
            }
        )

    input_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "reuse_manifest_sha256": sha256_file(reuse_path),
                "v2": sorted(
                    (str(key), record["parquet_sha256"], record["summary_sha256"])
                    for key, record in v2_records.items()
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "development_only": True,
        "partition": "replication-validation",
        "confirmatory_or_test_access": False,
        "tuning_authorization": False,
        "report_version": 1,
        "input_fingerprint_sha256": input_fingerprint,
        "baseline_reuse_manifest_sha256": sha256_file(reuse_path),
        "baseline_reuse_cells": reuse_manifest["cells"],
        "v2_cells": len(v2_records),
        "bootstrap": {
            "method": "paired percentile bootstrap over query ids after within-query seed averaging",
            "confidence": 0.95,
            "resamples": bootstrap_resamples,
            "base_seed": bootstrap_seed,
        },
        "baseline_aggregates": baseline_cells,
        "v2_aggregates": v2_cells,
        "primary_paired_comparisons": comparisons,
        "secondary_gate_vs_snapshot_comparisons": secondary_comparisons,
        "measurement_notes": {
            "sample_sd": "Bessel-corrected across run seeds; null for deterministic n=1 Weight",
            "probes": (
                "Exact PROBE actions were not stored in the episode Parquet schema. "
                "No probe count is inferred from total cost."
            ),
            "fallback": "Exact gate policy_stats from each immutable summary JSON",
        },
    }


def _pm(summary: dict[str, Any], digits: int = 3) -> str:
    value = float(summary["mean"])
    sd = summary["sample_sd"]
    if sd is None:
        return f"{value:.{digits}f} (exact)"
    return f"{value:.{digits}f} ± {float(sd):.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact, deterministic human-readable development report."""

    lines = [
        "# NEmo v2 development-validation report",
        "",
        "> **Development only.** Replication-validation/2pt data only; no "
        "confirmatory/test access and no tuning authorization.",
        "",
        f"Input fingerprint: `{report['input_fingerprint_sha256']}`.",
        "",
        "## Active-agent validation",
        "",
        "| Held out | Budget | Method | Success | Proof valid | Cost | Illegal actions | Fallback |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    order = {"weight": 0, "random": 1, "minerva": 2, "gate-v2": 3, "snapshot-v2": 4}
    cells = report["baseline_aggregates"] + report["v2_aggregates"]
    for cell in sorted(
        cells,
        key=lambda row: (DATASETS.index(row["held_out"]), row["budget"], order[row["method"]]),
    ):
        metrics = cell["metrics"]
        fallback = cell.get("fallback")
        fallback_text = _pm(fallback["fraction"]) if fallback else "—"
        lines.append(
            "| {held} | {budget} | {method} | {success} | {proof} | {cost} | {illegal} | {fallback} |".format(
                held=cell["held_out"],
                budget=cell["budget"],
                method=cell["method"],
                success=_pm(metrics["success_at_budget"]),
                proof=_pm(metrics["proof_validity"]),
                cost=_pm(metrics["mean_budget_used"], 2),
                illegal=cell["proof_and_safety"]["total_illegal_actions"],
                fallback=fallback_text,
            )
        )
    lines.extend(
        [
            "",
            "Weight is deterministic and represented by its exact seed-17 artifact; "
            "all ± values are Bessel-corrected sample SD over seeds 17/29/43.",
            "",
            "## Primary paired B64 comparisons",
            "",
            "Candidate seed outcomes are averaged within each query, then query IDs are "
            "paired exactly to the immutable Weight result and bootstrapped.",
            "",
            "| Held out | Candidate − Weight Success@64 | 95% paired CI | Δ proof | Δ cost |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for comparison in report["primary_paired_comparisons"]:
        metrics = comparison["paired_bootstrap"]
        primary = metrics["success_at_budget"]
        proof = metrics["proof_validity"]
        cost = metrics["mean_budget_used"]
        lines.append(
            f"| {comparison['held_out']} | {comparison['candidate']} | "
            f"{primary['estimate']:+.3f} [{primary['ci95_low']:+.3f}, {primary['ci95_high']:+.3f}] | "
            f"{proof['estimate']:+.3f} | {cost['estimate']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Secondary paired B64: gate versus SnapshotV2",
            "",
            "| Held out | Gate − SnapshotV2 Success@64 | 95% paired CI |",
            "|---|---:|---:|",
        ]
    )
    for comparison in report.get("secondary_gate_vs_snapshot_comparisons", []):
        primary = comparison["paired_bootstrap"]["success_at_budget"]
        lines.append(
            f"| {comparison['held_out']} | {primary['estimate']:+.3f} | "
            f"[{primary['ci95_low']:+.3f}, {primary['ci95_high']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Probe and fallback accounting",
            "",
            "Exact fallback fractions are reported above from each gate summary. Exact "
            "PROBE counts are **not instrumented** in the saved per-episode schema; the "
            "JSON report records the adaptive-probing protocol/cap and does not infer "
            "probes from total cost. Mean budget used is therefore the honest available "
            "cost measure.",
            "",
            "This report is descriptive development analysis only and must not be used "
            "as confirmatory evidence.",
            "",
        ]
    )
    return "\n".join(lines)
