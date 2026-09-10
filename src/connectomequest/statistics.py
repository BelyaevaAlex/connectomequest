"""Paired statistical comparisons for active-agent episode traces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import binomtest

from connectomequest.manifest import sha256_file

PAIRED_METRICS = ("goal_success", "precision", "recall", "f1", "proof_valid")


def _indexed_rows(path: Path) -> dict[str, dict]:
    columns = ["query_id", *PAIRED_METRICS]
    rows = pq.read_table(path, columns=columns).to_pylist()
    indexed = {row["query_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate query_id values in {path}")
    return indexed


def compare_active_runs(
    candidate: Path,
    reference: Path,
    output: Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 17,
) -> dict:
    """Compare two runs query-by-query with paired bootstrap and McNemar tests."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    candidate = Path(candidate)
    reference = Path(reference)
    candidate_rows = _indexed_rows(candidate)
    reference_rows = _indexed_rows(reference)
    if candidate_rows.keys() != reference_rows.keys():
        only_candidate = len(candidate_rows.keys() - reference_rows.keys())
        only_reference = len(reference_rows.keys() - candidate_rows.keys())
        raise ValueError(
            "runs do not contain identical query IDs "
            f"({only_candidate} candidate-only, {only_reference} reference-only)"
        )

    query_ids = sorted(candidate_rows)
    rng = np.random.default_rng(seed)
    report_metrics: dict[str, dict[str, float]] = {}
    for metric in PAIRED_METRICS:
        candidate_values = np.asarray(
            [candidate_rows[query_id][metric] for query_id in query_ids], dtype=np.float64
        )
        reference_values = np.asarray(
            [reference_rows[query_id][metric] for query_id in query_ids], dtype=np.float64
        )
        differences = candidate_values - reference_values
        sample_means = np.empty(bootstrap_samples, dtype=np.float64)
        for start in range(0, bootstrap_samples, 512):
            stop = min(start + 512, bootstrap_samples)
            indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
            sample_means[start:stop] = differences[indices].mean(axis=1)
        report_metrics[metric] = {
            "candidate_mean": float(candidate_values.mean()),
            "reference_mean": float(reference_values.mean()),
            "paired_difference": float(differences.mean()),
            "paired_bootstrap_ci95_low": float(np.quantile(sample_means, 0.025)),
            "paired_bootstrap_ci95_high": float(np.quantile(sample_means, 0.975)),
        }

    candidate_success = np.asarray(
        [candidate_rows[query_id]["goal_success"] for query_id in query_ids], dtype=bool
    )
    reference_success = np.asarray(
        [reference_rows[query_id]["goal_success"] for query_id in query_ids], dtype=bool
    )
    candidate_only = int(np.sum(candidate_success & ~reference_success))
    reference_only = int(np.sum(~candidate_success & reference_success))
    discordant = candidate_only + reference_only
    pvalue = (
        float(binomtest(min(candidate_only, reference_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )

    report = {
        "candidate": str(candidate),
        "candidate_sha256": sha256_file(candidate),
        "reference": str(reference),
        "reference_sha256": sha256_file(reference),
        "episodes": len(query_ids),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "metrics": report_metrics,
        "mcnemar_exact": {
            "candidate_only_successes": candidate_only,
            "reference_only_successes": reference_only,
            "two_sided_pvalue": pvalue,
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
