#!/usr/bin/env python3
"""Run the frozen amortized receipt-validation experiment on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from connectomequest.embodied.dynamic_task import FrozenPerceptionCache
from connectomequest.embodied.environment import generate_layout
from connectomequest.embodied.episode_runner import prepare_episode
from connectomequest.embodied.perception import Perceptor
from connectomequest.embodied.protocol import (
    CHECKPOINT,
    atomic_case_write,
    sha256_file,
    verify_source_hashes,
)
from connectomequest.embodied.revalidation_benchmark import (
    AmortizedProtocol,
    TimingRow,
    prepare_case_stream,
    time_case_stream,
)
from connectomequest.embodied.symbolic_controller import ColorPerceptor
from connectomequest.embodied.types import MethodInput, canonical_json_bytes
from scripts.freeze_revalidation_protocol import OUTPUT_ROOT

REPO = ROOT
METHODS = ("complete_validation", "dependency_indexed")
_CACHE = None


def _digest(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def pin_serial_timing_process() -> dict:
    """Pin only the serial timing phase; preparation workers remain parallel."""

    metadata = {
        "platform": platform.platform(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "selected_cpu": None,
        "affinity_before": None,
        "affinity_after": None,
    }
    before = sorted(os.sched_getaffinity(0))
    selected = before[0]
    os.sched_setaffinity(0, {selected})
    metadata.update(
        selected_cpu=selected,
        affinity_before=before,
        affinity_after=sorted(os.sched_getaffinity(0)),
    )
    return metadata


def _initialize_worker() -> None:
    global _CACHE
    import torch

    torch.set_num_threads(1)
    _CACHE = FrozenPerceptionCache(ColorPerceptor(Perceptor(CHECKPOINT)), sha256_file(CHECKPOINT))


def _prepare_inner(layout_seed: int):
    if _CACHE is None:
        raise RuntimeError("worker is not initialized")
    layout = generate_layout(int(layout_seed), "very_long")
    prepared = prepare_episode(
        layout,
        _CACHE,
        update_seed=17,
        condition="irrelevant_receipt_withdrawal",
        update_count=1,
        relevant_fraction=0.0,
    )
    if prepared.exclusion_reason is not None:
        return None, {"layout_seed": layout_seed, "reason": prepared.exclusion_reason}
    if len(prepared.plan) < 32:
        return None, {"layout_seed": layout_seed, "reason": "plan_shorter_than_32"}
    required = set().union(*(row.required_receipts for row in prepared.dependencies))
    if len(set(prepared.active_receipt_ids) - required) < 8:
        return None, {"layout_seed": layout_seed, "reason": "fewer_than_8_irrelevant_receipts"}
    return prepared, None


class _EnrollmentTimeout(Exception):
    pass


def _prepare(layout_seed: int):
    def timeout_handler(_signum, _frame):
        raise _EnrollmentTimeout()

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(180)
    try:
        return _prepare_inner(layout_seed)
    except _EnrollmentTimeout:
        return None, {"layout_seed": layout_seed, "reason": "enrollment_timeout"}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _belief_snapshot(prepared) -> tuple[tuple[str, str], ...]:
    cells = {}
    for dependency in prepared.dependencies:
        for assertion in dependency.required_assertions:
            left, separator, label = assertion.rpartition("=")
            if not separator or not left.startswith("cell:"):
                raise ValueError(f"invalid assertion: {assertion}")
            cells[left.removeprefix("cell:")] = label
    return tuple(sorted(cells.items()))


def _inputs(prepared, update_count: int, update_seed: int) -> tuple[MethodInput, ...]:
    events = prepare_case_stream(
        dependencies=prepared.dependencies,
        active_receipts=prepared.active_receipt_ids,
        update_count=update_count,
        update_seed=update_seed,
    )
    active = set(prepared.active_receipt_ids)
    first = prepared.plan[0]
    result = []
    for offset, raw_event in enumerate(events):
        active.difference_update(raw_event.withdrawn_receipt_ids)
        event = replace(raw_event, action_index=len(prepared.prefix_actions) + offset)
        result.append(
            MethodInput(
                pose=tuple(first.position),
                direction=int(first.direction),
                inventory=None,
                mission="pick up the target box",
                belief_snapshot=_belief_snapshot(prepared),
                active_receipt_ids=tuple(sorted(active)),
                assertion_to_receipt_bindings=prepared.assertion_to_receipt_bindings,
                remaining_plan=prepared.plan,
                cursor=0,
                update=event,
                public_action_set=("left", "right", "forward", "toggle", "pickup", "drop"),
            )
        )
    return tuple(result)


def _combine(rows: list[TimingRow]) -> TimingRow:
    first = rows[0]
    scalar = (
        "index_build_ns",
        "index_maintenance_ns",
        "validation_ns",
        "planning_ns",
        "predicate_evaluations",
        "plan_position_visits",
        "index_lookups",
        "posting_entries",
    )
    values = {name: int(median(getattr(row, name) for row in rows)) for name in scalar}
    values["total_reasoning_ns"] = sum(
        values[name]
        for name in ("index_build_ns", "index_maintenance_ns", "validation_ns", "planning_ns")
    )
    return TimingRow(
        **{
            **asdict(first),
            **values,
            "raw_repetitions_ns": tuple(row.raw_repetitions_ns[0] for row in rows),
        }
    )


def _time_pair(prepared, update_count: int, update_seed: int, protocol: AmortizedProtocol):
    inputs = _inputs(prepared, update_count, update_seed)
    layout_seed = prepared.layout.spec.layout_seed
    for repetition in range(protocol.warmups):
        order = METHODS if (layout_seed + update_seed + repetition) % 2 == 0 else METHODS[::-1]
        for method in order:
            time_case_stream(
                method=method,
                plan=prepared.plan,
                dependencies=prepared.dependencies,
                inputs=inputs,
                layout_seed=layout_seed,
                update_seed=update_seed,
                warmups=0,
                repetitions=1,
            )
    retained = {method: [] for method in METHODS}
    for repetition in range(protocol.repetitions):
        order = METHODS if (layout_seed + update_seed + repetition) % 2 == 0 else METHODS[::-1]
        for method in order:
            row, _ = time_case_stream(
                method=method,
                plan=prepared.plan,
                dependencies=prepared.dependencies,
                inputs=inputs,
                layout_seed=layout_seed,
                update_seed=update_seed,
                warmups=0,
                repetitions=1,
            )
            retained[method].append(row)
    combined = [_combine(retained[method]) for method in METHODS]
    if len({row.method_input_sha256 for row in combined}) != 1:
        raise ValueError("method-visible stream mismatch")
    if len({row.decision_sha256 for row in combined}) != 1:
        raise ValueError("behavioral decision mismatch")
    return combined


def _bootstrap(differences: dict[int, list[float]], seed: int, draws: int):
    layout_means = np.asarray([np.mean(differences[key]) for key in sorted(differences)])
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(layout_means), size=(draws, len(layout_means)))
    samples = layout_means[choices].mean(axis=1)
    return float(layout_means.mean()), [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def _summarize(rows: list[dict], protocol: AmortizedProtocol) -> dict:
    cells = {}
    for row in rows:
        key = (row["layout_seed"], row["update_seed"], row["update_count"])
        cells.setdefault(key, {})[row["method"]] = row
    if not cells or any(set(pair) != set(METHODS) for pair in cells.values()):
        raise ValueError("unpaired cells")
    reductions = {}
    work_reductions = {}
    for key, pair in cells.items():
        scan = pair["complete_validation"]
        index = pair["dependency_indexed"]
        reductions.setdefault(key[0], []).append(
            1 - index["total_reasoning_ns"] / scan["total_reasoning_ns"]
        )
        scan_work = scan["plan_position_visits"] + scan["predicate_evaluations"]
        index_work = index["plan_position_visits"] + index["predicate_evaluations"]
        work_reductions.setdefault(key[0], []).append(1 - index_work / scan_work)
    estimate, interval = _bootstrap(reductions, protocol.bootstrap_seed, protocol.bootstrap_draws)
    work, work_interval = _bootstrap(
        work_reductions, protocol.bootstrap_seed + 1, protocol.bootstrap_draws
    )
    agreement = float(
        all(
            pair[METHODS[0]]["decision_sha256"] == pair[METHODS[1]]["decision_sha256"]
            for pair in cells.values()
        )
    )
    return {
        "eligible_layouts": len({key[0] for key in cells}),
        "paired_cells": len(cells),
        "decision_agreement": agreement,
        "time_reduction": estimate,
        "time_reduction_ci": interval,
        "work_reduction": work,
        "work_reduction_ci": work_interval,
        "by_update_count": {
            str(count): {
                method: {
                    "median_total_reasoning_ns": float(
                        median(
                            row["total_reasoning_ns"]
                            for row in rows
                            if row["update_count"] == count and row["method"] == method
                        )
                    ),
                    "median_plan_length": float(
                        median(row["plan_length"] for row in rows if row["update_count"] == count)
                    ),
                }
                for method in METHODS
            }
            for count in protocol.update_counts
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "confirmation"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    stage = OUTPUT_ROOT / args.stage
    protocol_path = stage / "protocol.json"
    if not protocol_path.exists():
        raise SystemExit("freeze the stage first")
    frozen = json.loads(protocol_path.read_text())
    verify_source_hashes(REPO, frozen["source_hashes"])
    if sha256_file(CHECKPOINT) != frozen["checkpoint_sha256"]:
        raise ValueError("perception checkpoint drift")
    protocol = AmortizedProtocol(
        **{
            **frozen["protocol"],
            "layout_seeds": tuple(frozen["protocol"]["layout_seeds"]),
            "update_counts": tuple(frozen["protocol"]["update_counts"]),
            "update_seeds": tuple(frozen["protocol"]["update_seeds"]),
            "methods": tuple(frozen["protocol"]["methods"]),
        }
    )
    started = time.perf_counter()
    prepared_cases = []
    exclusions = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(max(1, args.workers), len(protocol.layout_seeds)),
        mp_context=context,
        initializer=_initialize_worker,
    ) as pool:
        futures = {pool.submit(_prepare, seed): seed for seed in protocol.layout_seeds}
        for index, future in enumerate(as_completed(futures), start=1):
            prepared, exclusion = future.result()
            if prepared is not None:
                prepared_cases.append(prepared)
            if exclusion is not None:
                exclusions.append(exclusion)
            heartbeat = {
                "status": "preparing",
                "completed": index,
                "total": len(futures),
                "eligible": len(prepared_cases),
                "elapsed_seconds": time.perf_counter() - started,
            }
            stage.mkdir(parents=True, exist_ok=True)
            (stage / "heartbeat.json").write_text(json.dumps(heartbeat, sort_keys=True) + "\n")

    cpu_metadata = pin_serial_timing_process()
    rows = []
    total_cells = len(prepared_cases) * len(protocol.update_counts) * len(protocol.update_seeds)
    cell_index = 0
    for prepared in sorted(prepared_cases, key=lambda item: item.layout.spec.layout_seed):
        for update_count in protocol.update_counts:
            for update_seed in protocol.update_seeds:
                paired = _time_pair(prepared, update_count, update_seed, protocol)
                for row in paired:
                    payload = asdict(row)
                    rows.append(payload)
                    path = (
                        stage
                        / "cases"
                        / (
                            f"{row.layout_seed}-{row.update_seed}-{row.update_count}-{row.method}.json"
                        )
                    )
                    atomic_case_write(
                        path,
                        {
                            "protocol_sha256": sha256_file(protocol_path),
                            "payload_sha256": _digest(payload),
                            "result": payload,
                        },
                    )
                cell_index += 1
                (stage / "heartbeat.json").write_text(
                    json.dumps(
                        {
                            "status": "timing",
                            "completed": cell_index,
                            "total": total_cells,
                            "eligible": len(prepared_cases),
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    statistics = _summarize(rows, protocol)
    design_gate = bool(
        statistics["eligible_layouts"] >= (8 if args.stage == "development" else 64)
        and statistics["decision_agreement"] == 1.0
        and all(row["plan_length"] >= protocol.minimum_plan_length for row in rows)
    )
    verification = {
        "stage": args.stage,
        "protocol_sha256": sha256_file(protocol_path),
        "design_gate_passed": design_gate,
        "source_hashes_match": True,
        "excluded": sorted(exclusions, key=lambda row: row["layout_seed"]),
        "cpu_metadata": cpu_metadata,
        "statistics": statistics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_case_write(stage / "verification.json", verification)
    (stage / "heartbeat.json").write_text(
        json.dumps({"status": "completed"}, sort_keys=True) + "\n"
    )
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
