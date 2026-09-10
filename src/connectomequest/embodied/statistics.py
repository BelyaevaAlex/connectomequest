"""Paired clustered inference and prospective V39 gate evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

_PAIR_FIELDS = (
    "layout_seed",
    "update_seed",
    "condition",
    "update_count",
    "relevant_fraction",
)


@dataclass(frozen=True)
class PairedEstimate:
    mean: float
    lower: float
    upper: float
    n_layouts: int
    n_pairs: int


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    criteria: tuple[tuple[str, bool], ...]


def _value(row: Any, name: str):
    if isinstance(row, Mapping):
        if name in row:
            return row[name]
        counters = row.get("counters")
    else:
        if hasattr(row, name):
            return getattr(row, name)
        counters = getattr(row, "counters", None)
    if counters is not None:
        if isinstance(counters, Mapping) and name in counters:
            return counters[name]
        if hasattr(counters, name):
            return getattr(counters, name)
    raise KeyError(name)


def _pair_key(row: Any) -> tuple:
    return tuple(_value(row, field) for field in _PAIR_FIELDS)


def paired_layout_bootstrap(
    rows: Iterable[Any],
    method_a: str,
    method_b: str,
    metric: str,
    *,
    draws: int = 10_000,
    seed: int = 39_039,
) -> PairedEstimate:
    if draws <= 0:
        raise ValueError("draws must be positive")
    cells: dict[tuple, dict[str, float]] = {}
    for row in rows:
        method = str(_value(row, "method"))
        if method not in (method_a, method_b):
            continue
        key = _pair_key(row)
        bucket = cells.setdefault(key, {})
        if method in bucket:
            raise ValueError(f"duplicate method cell: {key}, {method}")
        bucket[method] = float(_value(row, metric))
    if not cells or any(set(bucket) != {method_a, method_b} for bucket in cells.values()):
        raise ValueError("unpaired method cells")
    by_layout: dict[int, list[float]] = {}
    for key, bucket in cells.items():
        difference = bucket[method_a] - bucket[method_b]
        by_layout.setdefault(int(key[0]), []).append(difference)
    layout_ids = tuple(sorted(by_layout))
    cluster_means = np.asarray(
        [np.mean(by_layout[layout]) for layout in layout_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(cluster_means),
        size=(draws, len(cluster_means)),
    )
    samples = cluster_means[indices].mean(axis=1)
    return PairedEstimate(
        float(cluster_means.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
        len(layout_ids),
        len(cells),
    )


def _gate(name: str, criteria: Mapping[str, bool]) -> GateResult:
    frozen = tuple((key, bool(value)) for key, value in criteria.items())
    return GateResult(name, all(value for _, value in frozen), frozen)


def evaluate_integrity_gate(summary: Mapping[str, Any]) -> GateResult:
    return _gate(
        "integrity",
        {
            "source_hashes_match": bool(summary["source_hashes_match"]),
            "pre_update_access_match": bool(summary["pre_update_access_match"]),
            "pre_update_action_prefix_match": bool(summary["pre_update_action_prefix_match"]),
            "update_schedules_match": bool(summary["update_schedules_match"]),
            "all_traces_replay": float(summary["trace_replay_rate"]) == 1.0,
            "no_oracle_fields_exposed": not bool(summary["oracle_fields_exposed"]),
        },
    )


def evaluate_correctness_gate(summary: Mapping[str, Any]) -> GateResult:
    return _gate(
        "correctness",
        {
            "suffix_stable_next_action_matches": bool(summary["suffix_stable_next_action_matches"]),
            "competent_methods_never_unsupported": int(
                summary["competent_unsupported_authorizations"]
            )
            == 0,
            "task_success_noninferiority": float(summary["task_success_difference_ci_low"]) > -0.01,
        },
    )


def evaluate_practical_gate(summary: Mapping[str, Any]) -> GateResult:
    return _gate(
        "practical_significance",
        {
            "checking_planning_time_reduction": float(summary["time_reduction"]) >= 0.20,
            "time_reduction_lower_ci": float(summary["time_reduction_ci_low"]) > 0.10,
            "instrumented_work_reduction": float(summary["work_reduction"]) >= 0.25,
            "total_reasoning_lower": float(summary["total_reasoning_reduction"]) > 0.0,
            "episode_wall_clock_not_worse": float(summary["wall_clock_increase"]) <= 0.01,
        },
    )


def evaluate_safety_gate(summary: Mapping[str, Any]) -> GateResult:
    return _gate(
        "safety",
        {
            "competent_methods_never_unsupported": int(
                summary["competent_unsupported_authorizations"]
            )
            == 0,
            "receipt_index_never_unsupported": float(summary["receipt_index_unsupported_rate"])
            == 0.0,
            "unchecked_reuse_strictly_worse": float(summary["unchecked_reuse_unsupported_rate"])
            > float(summary["receipt_index_unsupported_rate"]),
        },
    )


def _derived_rows(rows: Iterable[Any]) -> list[dict[str, Any]]:
    derived = []
    for row in rows:
        predicate = float(_value(row, "predicate_evaluations"))
        successors = float(_value(row, "successor_expansions"))
        try:
            checking_planning = float(_value(row, "checking_planning_ns"))
        except KeyError:
            checking_planning = (
                float(_value(row, "revalidation_ns"))
                + float(_value(row, "planning_ns"))
                + float(_value(row, "index_build_ns"))
            )
        item = {field: _value(row, field) for field in _PAIR_FIELDS}
        item["plan_length_stratum"] = _value(row, "plan_length_stratum")
        item.update(
            method=_value(row, "method"),
            task_success=float(bool(_value(row, "task_success"))),
            exposed_update_count=int(_value(row, "exposed_update_count")),
            predicate_evaluations=predicate,
            successor_expansions=successors,
            instrumented_work=predicate + successors,
            checking_planning_ns=checking_planning,
            total_reasoning_ns=float(_value(row, "total_reasoning_ns")),
            total_wall_clock_ns=float(_value(row, "total_wall_clock_ns")),
        )
        derived.append(item)
    return derived


def _ratio_rows(
    rows: list[dict[str, Any]],
    numerator_method: str,
    denominator_method: str,
    source_metric: str,
    target_metric: str,
    *,
    one_minus: bool,
) -> list[dict[str, Any]]:
    cells = {}
    for row in rows:
        if row["method"] not in (numerator_method, denominator_method):
            continue
        cells.setdefault(_pair_key(row), {})[row["method"]] = row
    result = []
    for key, pair in cells.items():
        if set(pair) != {numerator_method, denominator_method}:
            raise ValueError("unpaired method cells")
        denominator = float(pair[denominator_method][source_metric])
        numerator = float(pair[numerator_method][source_metric])
        if denominator <= 0:
            raise ValueError(f"non-positive denominator for {source_metric}")
        value = 1.0 - numerator / denominator if one_minus else numerator / denominator - 1.0
        common = {field: key[index] for index, field in enumerate(_PAIR_FIELDS)}
        result.extend(
            (
                {**common, "method": numerator_method, target_metric: value},
                {**common, "method": denominator_method, target_metric: 0.0},
            )
        )
    return result


def summarize_confirmation(
    rows: Iterable[Any],
    *,
    draws: int = 10_000,
    seed: int = 39_039,
) -> dict[str, Any]:
    rows = list(rows)
    derived = _derived_rows(rows)
    primary = [
        row
        for row in derived
        if row["plan_length_stratum"] in ("long", "very_long")
        and 1 <= row["update_count"] <= 2
        and row["relevant_fraction"] <= 0.20
        and row["method"] in ("receipt_index", "full_scan")
    ]
    if not primary:
        raise ValueError("primary stratum is empty")

    def reduction(metric, target, *, one_minus=True):
        ratio = _ratio_rows(
            primary,
            "receipt_index",
            "full_scan",
            metric,
            target,
            one_minus=one_minus,
        )
        return paired_layout_bootstrap(
            ratio,
            "receipt_index",
            "full_scan",
            target,
            draws=draws,
            seed=seed,
        )

    time = reduction("checking_planning_ns", "reduction")
    work = reduction("instrumented_work", "reduction")
    reasoning = reduction("total_reasoning_ns", "reduction")
    wall = reduction("total_wall_clock_ns", "increase", one_minus=False)
    success = paired_layout_bootstrap(
        primary,
        "receipt_index",
        "full_scan",
        "task_success",
        draws=draws,
        seed=seed,
    )
    return {
        "row_counts": {
            "all": len(derived),
            "successful": sum(row["task_success"] == 1.0 for row in derived),
            "failed": sum(row["task_success"] == 0.0 for row in derived),
            "unexposed": sum(row["exposed_update_count"] == 0 for row in derived),
        },
        "primary_stratum": {
            "paired_cells": time.n_pairs,
            "layouts": time.n_layouts,
            "time_reduction": time.mean,
            "time_reduction_ci": [time.lower, time.upper],
            "work_reduction": work.mean,
            "work_reduction_ci": [work.lower, work.upper],
            "total_reasoning_reduction": reasoning.mean,
            "total_reasoning_reduction_ci": [reasoning.lower, reasoning.upper],
            "wall_clock_increase": wall.mean,
            "wall_clock_increase_ci": [wall.lower, wall.upper],
            "task_success_difference": success.mean,
            "task_success_difference_ci": [success.lower, success.upper],
        },
    }
