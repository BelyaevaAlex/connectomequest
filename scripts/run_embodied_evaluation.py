#!/usr/bin/env python3
"""Run one sealed local CPU V39 stage without scientific overrides."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from connectomequest.embodied.dynamic_task import FrozenPerceptionCache
from connectomequest.embodied.environment import generate_layout
from connectomequest.embodied.episode_runner import (
    prepare_episode,
    reschedule_episode,
    run_episode,
)
from connectomequest.embodied.perception import Perceptor
from connectomequest.embodied.protocol import (
    CHECKPOINT,
    OUTPUT_ROOT,
    ROOT,
    atomic_case_write,
    verify_source_hashes,
)
from connectomequest.embodied.symbolic_controller import ColorPerceptor
from connectomequest.embodied.types import EpisodeProtocol

_STRATA = ("short", "medium", "long", "very_long")


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _episode_protocol(payload) -> EpisodeProtocol:
    cells = payload["schedule_cells"]
    return EpisodeProtocol(
        protocol_version=payload["protocol_version"],
        stage=payload["stage"],
        horizon=payload["horizon"],
        methods=tuple(payload["methods"]),
        public_action_set=("left", "right", "forward", "toggle", "pickup", "drop"),
        plan_length_strata=_STRATA,
        update_conditions=tuple(cell["condition"] for cell in cells),
        update_counts=tuple(sorted({cell["update_count"] for cell in cells})),
        relevant_fractions=tuple(sorted({cell["relevant_fraction"] for cell in cells})),
        layout_seeds=tuple(
            seed
            for stratum in _STRATA
            for seeds in (payload["layout_seeds_by_stratum"][stratum],)
            for seed in seeds
        ),
        update_seeds=tuple(payload["update_seeds"]),
        bootstrap_seed=payload["bootstrap_seed"],
        bootstrap_draws=payload["bootstrap_draws"],
        common_planner_id=payload["common_planner"],
        action_model_id=payload["action_model"],
        perception_sha256=payload["checkpoint_sha256"],
    )


def _layout_jobs(payload) -> tuple[tuple[str, int], ...]:
    return tuple(
        (stratum, int(seed))
        for stratum in _STRATA
        for seed in payload["layout_seeds_by_stratum"][stratum]
    )


def _method_order(methods: tuple[str, ...], layout_seed: int, cell_index: int) -> tuple[str, ...]:
    if not methods:
        return ()
    offset = (int(layout_seed) + int(cell_index)) % len(methods)
    return methods[offset:] + methods[:offset]


_WORKER_CACHE = None
_WORKER_PROTOCOL = None
_WORKER_PAYLOAD = None


def _initialize_worker(payload) -> None:
    import torch

    global _WORKER_CACHE, _WORKER_PROTOCOL, _WORKER_PAYLOAD
    torch.set_num_threads(1)
    _WORKER_PAYLOAD = payload
    _WORKER_PROTOCOL = _episode_protocol(payload)
    _WORKER_CACHE = FrozenPerceptionCache(
        ColorPerceptor(Perceptor(CHECKPOINT)),
        payload["checkpoint_sha256"],
    )


def _run_layout_job(job: tuple[str, int]):
    if _WORKER_CACHE is None or _WORKER_PROTOCOL is None or _WORKER_PAYLOAD is None:
        raise RuntimeError("layout worker is not initialized")
    stratum, layout_seed = job
    payload = _WORKER_PAYLOAD
    generated = generate_layout(layout_seed, stratum)
    base = prepare_episode(
        generated,
        _WORKER_CACHE,
        update_seed=payload["update_seeds"][0],
        condition="irrelevant_receipt_withdrawal",
        update_count=1,
        relevant_fraction=0.0,
    )
    exclusions = []
    rows = []
    if base.exclusion_reason is not None:
        exclusions.append(
            {
                "layout_seed": layout_seed,
                "stratum": stratum,
                "condition": "base",
                "reason": base.exclusion_reason,
            }
        )
        return tuple(rows), tuple(exclusions)
    for cell_index, cell in enumerate(payload["schedule_cells"]):
        update_seed = payload["update_seeds"][cell_index % len(payload["update_seeds"])]
        scheduled = reschedule_episode(
            base,
            update_seed=update_seed,
            condition=cell["condition"],
            update_count=cell["update_count"],
            relevant_fraction=cell["relevant_fraction"],
        )
        if scheduled.exclusion_reason is not None:
            exclusions.append(
                {
                    "layout_seed": layout_seed,
                    "stratum": stratum,
                    "condition": cell["condition"],
                    "reason": scheduled.exclusion_reason,
                }
            )
            continue
        for method in _method_order(_WORKER_PROTOCOL.methods, layout_seed, cell_index):
            rows.append(asdict(run_episode(_WORKER_PROTOCOL, scheduled, method, _WORKER_CACHE)))
    return tuple(rows), tuple(exclusions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "pilot", "confirmation"), required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("workers must be in [1, 32]")
    stage_name = args.stage
    stage = OUTPUT_ROOT / stage_name
    protocol_path = stage / "protocol.json"
    if not protocol_path.exists():
        raise SystemExit(f"freeze {stage_name} protocol first")
    payload = json.loads(protocol_path.read_text())
    verify_source_hashes(ROOT, payload["source_hashes"])
    protocol = _episode_protocol(payload)
    cases = stage / "cases"
    exclusions = []
    rows = []
    started = time.perf_counter()
    jobs = _layout_jobs(payload)
    total_layouts = len(jobs)
    completed_layouts = 0
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()

    def record_layout(output) -> None:
        nonlocal completed_layouts
        layout_rows, layout_exclusions = output
        exclusions.extend(layout_exclusions)
        for row in layout_rows:
            path = cases / (
                f"{row['plan_length_stratum']}-{row['layout_seed']}-"
                f"{row['update_seed']}-{row['condition']}-{row['method']}.json"
            )
            wrapped = {
                "protocol_sha256": protocol_sha256,
                "payload_sha256": _digest(row),
                "result": row,
            }
            atomic_case_write(path, wrapped)
            rows.append(row)
        completed_layouts += 1
        heartbeat = {
            "status": "running",
            "completed_layouts": completed_layouts,
            "total_layouts": total_layouts,
            "workers": args.workers,
            "elapsed_seconds": time.perf_counter() - started,
        }
        temporary = stage / "heartbeat.tmp"
        temporary.write_text(json.dumps(heartbeat, sort_keys=True) + "\n")
        temporary.replace(stage / "heartbeat.json")

    if args.workers == 1:
        _initialize_worker(payload)
        for job in jobs:
            record_layout(_run_layout_job(job))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(jobs)),
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(payload,),
        ) as pool:
            futures = [pool.submit(_run_layout_job, job) for job in jobs]
            for future in as_completed(futures):
                record_layout(future.result())

    competent = {"full_replan", "full_scan", "receipt_index"}
    grouped = {}
    for row in rows:
        key = (
            row["layout_seed"],
            row["update_seed"],
            row["condition"],
            row["update_count"],
            row["relevant_fraction"],
        )
        grouped.setdefault(key, []).append(row)
    access_match = all(
        len({row["access_digest"] for row in group}) == 1
        and len({row["pre_update_action_digest"] for row in group}) == 1
        for group in grouped.values()
        if len(group) == len(protocol.methods)
    )
    verification = {
        "stage": stage_name,
        "integrity_passed": bool(
            rows
            and access_match
            and all(row["trace_replay_ok"] for row in rows)
            and all(
                row["unsupported_authorizations"] == 0 for row in rows if row["method"] in competent
            )
        ),
        "rows": len(rows),
        "paired_cells": len(grouped),
        "generated_layouts": total_layouts,
        "eligible_layouts": len({row["layout_seed"] for row in rows}),
        "primary_attainable_layouts": len(
            {
                row["layout_seed"]
                for row in rows
                if row["plan_length_stratum"] in ("long", "very_long")
            }
        ),
        "access_match": access_match,
        "trace_replay_rate": (
            sum(row["trace_replay_ok"] for row in rows) / len(rows) if rows else 0.0
        ),
        "competent_unsupported_authorizations": sum(
            row["unsupported_authorizations"] for row in rows if row["method"] in competent
        ),
        "exclusions": exclusions,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_case_write(stage / "verification.json", verification)
    atomic_case_write(stage / "exit.json", {"exit_code": 0})
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
