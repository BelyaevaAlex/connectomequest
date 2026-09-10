"""Independent paired analysis for amortized retained-plan validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

COMPONENTS = (
    "index_build_ns",
    "index_maintenance_ns",
    "validation_ns",
    "planning_ns",
)
METHODS = ("complete_validation", "dependency_indexed")


def primary_rows(rows: Iterable[dict]) -> list[dict]:
    return [
        row for row in rows if int(row["plan_length"]) >= 64 and int(row["update_count"]) in (4, 8)
    ]


def _cluster_interval(
    values: dict[int, list[float]], *, seed: int, draws: int
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("no paired layouts")
    means = np.asarray([np.mean(values[key]) for key in sorted(values)], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = means[rng.integers(0, len(means), size=(draws, len(means)))].mean(axis=1)
    return (
        float(means.mean()),
        float(np.quantile(sampled, 0.025)),
        float(np.quantile(sampled, 0.975)),
    )


def summarize_fully_accounted(
    rows: Iterable[dict], *, seed: int = 40_040, draws: int = 10_000
) -> dict:
    rows = list(rows)
    required = {
        "layout_seed",
        "update_seed",
        "update_count",
        "plan_length",
        "method",
        "total_reasoning_ns",
        "decision_sha256",
        "predicate_evaluations",
        "plan_position_visits",
        *COMPONENTS,
    }
    for row in rows:
        if not required <= set(row):
            raise ValueError("fully accounted timing fields are required")
        if int(row["total_reasoning_ns"]) != sum(int(row[name]) for name in COMPONENTS):
            raise ValueError("fully accounted total does not equal its components")
        if not row.get("raw_repetitions_ns"):
            raise ValueError("fully accounted timing requires raw repetitions")

    selected = primary_rows(rows)
    pairs: dict[tuple[int, int, int], dict[str, dict]] = defaultdict(dict)
    for row in selected:
        key = (
            int(row["layout_seed"]),
            int(row["update_seed"]),
            int(row["update_count"]),
        )
        method = str(row["method"])
        if method in pairs[key]:
            raise ValueError("duplicate paired cell")
        pairs[key][method] = row
    if not pairs or any(set(pair) != set(METHODS) for pair in pairs.values()):
        raise ValueError("unpaired timing cells")

    time_by_layout: dict[int, list[float]] = defaultdict(list)
    work_by_layout: dict[int, list[float]] = defaultdict(list)
    decisions_match = True
    for key, pair in pairs.items():
        scan = pair["complete_validation"]
        index = pair["dependency_indexed"]
        scan_time = float(scan["total_reasoning_ns"])
        index_time = float(index["total_reasoning_ns"])
        if scan_time <= 0:
            raise ValueError("non-positive timing denominator")
        time_by_layout[key[0]].append(1.0 - index_time / scan_time)
        scan_work = float(scan["predicate_evaluations"] + scan["plan_position_visits"])
        index_work = float(index["predicate_evaluations"] + index["plan_position_visits"])
        if scan_work <= 0:
            raise ValueError("non-positive work denominator")
        work_by_layout[key[0]].append(1.0 - index_work / scan_work)
        decisions_match &= scan["decision_sha256"] == index["decision_sha256"]

    time_estimate, time_low, time_high = _cluster_interval(time_by_layout, seed=seed, draws=draws)
    work_estimate, work_low, work_high = _cluster_interval(
        work_by_layout, seed=seed + 1, draws=draws
    )
    return {
        "eligible_long": len(time_by_layout),
        "paired_cells": len(pairs),
        "decision_agreement": float(decisions_match),
        "time_reduction": time_estimate,
        "time_reduction_ci": [time_low, time_high],
        "work_reduction": work_estimate,
        "work_reduction_ci": [work_low, work_high],
    }


def evaluate_v40_gates(
    *,
    reduction: float,
    lower: float,
    decision_agreement: float,
    work_reduction: float,
    eligible_long: int,
) -> dict:
    correctness_criteria = {
        "decision_agreement": decision_agreement == 1.0,
        "eligible_long": eligible_long >= 64,
    }
    efficiency_criteria = {
        "fully_accounted_reduction": reduction >= 0.20,
        "cluster_interval_lower": lower > 0.10,
        "instrumented_work_reduction": work_reduction >= 0.20,
        "eligible_long": eligible_long >= 64,
    }
    return {
        "correctness": {
            "passed": all(correctness_criteria.values()),
            "criteria": correctness_criteria,
        },
        "efficiency": {
            "passed": all(efficiency_criteria.values()),
            "criteria": efficiency_criteria,
        },
    }
