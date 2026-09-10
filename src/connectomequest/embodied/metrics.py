"""Outcome-conditioned metrics for paired embodied evaluations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def _optional_mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def summarize_embodied_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(rows)
    oracle_steps = {
        (int(row["size"]), int(row["task_seed"])): int(row["steps"])
        for row in records
        if row["condition"] == "partial_oracle" and row["success"]
    }
    conditions: dict[str, dict[str, Any]] = {}
    for size in sorted({int(row["size"]) for row in records}):
        by_size: dict[str, Any] = {}
        for condition in sorted(
            {str(row["condition"]) for row in records if int(row["size"]) == size}
        ):
            selected = [
                row for row in records if int(row["size"]) == size and row["condition"] == condition
            ]
            successes = [row for row in selected if bool(row["success"])]
            failures = [row for row in selected if not bool(row["success"])]
            excess = [
                float(row["steps"] - oracle_steps[(size, int(row["task_seed"]))])
                for row in successes
                if (size, int(row["task_seed"])) in oracle_steps
            ]
            by_size[condition] = {
                "episodes": len(selected),
                "success": float(mean(bool(row["success"]) for row in selected)),
                "mean_actions_unconditional": float(mean(float(row["steps"]) for row in selected)),
                "mean_actions_given_success": _optional_mean(
                    [float(row["steps"]) for row in successes]
                ),
                "mean_actions_given_failure": _optional_mean(
                    [float(row["steps"]) for row in failures]
                ),
                "mean_excess_actions_given_success_vs_oracle": _optional_mean(excess),
                "mean_replans": float(mean(float(row.get("replans", 0)) for row in selected)),
                "mean_map_resets": float(mean(float(row.get("map_resets", 0)) for row in selected)),
                "mean_failed_preconditions": float(
                    mean(float(row.get("failed_preconditions", 0)) for row in selected)
                ),
                "goal_claim_attempts": sum(
                    int(row.get("goal_claim_attempts", int(bool(row["success"]))))
                    for row in selected
                ),
                "accepted_proofs": sum(
                    int(row.get("proof_accepted", row.get("proof_valid", False)))
                    for row in selected
                ),
                "organic_malformed_proofs": sum(
                    int(row.get("organic_malformed_proof", False)) for row in selected
                ),
                "illegal_actions": sum(int(row.get("illegal_actions", 0)) for row in selected),
            }
        conditions[str(size)] = by_size
    return {"conditions": conditions, "rows": len(records)}


def paired_success_difference(
    rows: Iterable[dict[str, Any]],
    *,
    left: str,
    right: str,
    seed: int,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    paired: dict[tuple[int, int, int], dict[str, float]] = {}
    for row in rows:
        if row["condition"] not in {left, right}:
            continue
        key = int(row["size"]), int(row["task_seed"]), int(row["model_seed"])
        paired.setdefault(key, {})[str(row["condition"])] = float(row["success"])
    seed_level = {
        key: values[right] - values[left]
        for key, values in paired.items()
        if left in values and right in values
    }
    by_task: dict[tuple[int, int], list[float]] = {}
    for (size, task_seed, _model_seed), difference in seed_level.items():
        by_task.setdefault((size, task_seed), []).append(difference)
    differences = np.asarray(
        [mean(values) for _, values in sorted(by_task.items())], dtype=np.float64
    )
    if not len(differences):
        raise ValueError(f"no paired episodes for {left} and {right}")
    rng = np.random.default_rng(seed)
    samples = np.mean(
        rng.choice(
            differences,
            size=(bootstrap_samples, len(differences)),
            replace=True,
        ),
        axis=1,
    )
    return {
        "left": left,
        "right": right,
        "pairs": int(len(differences)),
        "seed_level_pairs": int(len(seed_level)),
        "difference": float(differences.mean()),
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def write_once(path: Path, payload: dict[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite mismatched artifact: {path}")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return digest
