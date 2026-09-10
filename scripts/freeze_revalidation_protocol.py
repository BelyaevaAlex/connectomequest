#!/usr/bin/env python3
"""Freeze the write-once development or confirmation timing protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from connectomequest.embodied.protocol import CHECKPOINT, freeze_payload, sha256_file, source_hashes
from connectomequest.embodied.revalidation_benchmark import AmortizedProtocol

REPO = ROOT
OUTPUT_ROOT = REPO / "outputs/nemo/strong-accept/amortized-revalidation-v40"
SOURCES = (
    "src/connectomequest/embodied/revalidation_benchmark.py",
    "src/connectomequest/embodied/revalidation_statistics.py",
    "src/connectomequest/embodied/environment.py",
    "src/connectomequest/embodied/revalidation.py",
    "src/connectomequest/embodied/task_planner.py",
    "src/connectomequest/embodied/episode_runner.py",
    "src/connectomequest/embodied/types.py",
    "src/connectomequest/embodied/evidence_index.py",
    "scripts/report_revalidation_benchmark.py",
    "scripts/freeze_revalidation_protocol.py",
    "scripts/run_revalidation_benchmark.py",
)


def payload(stage: str) -> dict:
    protocol = (
        AmortizedProtocol.development()
        if stage == "development"
        else AmortizedProtocol.confirmation()
    )
    return {
        "protocol": asdict(protocol),
        "layout_stratum": "very_long",
        "enrollment_timeout_seconds": 180,
        "public_case_enrollment": "frozen RGB perceptor and symbolic planner",
        "intervention": "distinct withdrawals of receipts absent from every remaining-plan dependency",
        "timing": "serial, one thread, cyclic method order, 10 warmups, 40 retained repetitions",
        "accounting": [
            "index_build_ns",
            "index_maintenance_ns",
            "validation_ns",
            "planning_ns",
        ],
        "primary_contrast": "paired relative reduction of dependency-indexed versus complete validation",
        "primary_population": "eligible plans of length at least 64 with 4 or 8 distinct irrelevant receipt withdrawals",
        "gates": {
            "decision_agreement": 1.0,
            "eligible_long_min": 64,
            "time_reduction_min": 0.20,
            "time_reduction_ci_low": 0.10,
            "work_reduction_min": 0.20,
        },
        "bootstrap_unit": "layout_seed",
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "source_hashes": source_hashes(REPO, SOURCES),
        "oracle_fields_in_method_input": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--development", action="store_true")
    group.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    stage = "development" if args.development else "confirmation"
    if stage == "confirmation":
        verification = OUTPUT_ROOT / "development/verification.json"
        if not verification.exists():
            raise SystemExit("development verification is required")
        result = json.loads(verification.read_text())
        if not result.get("design_gate_passed"):
            raise SystemExit("development design gate did not pass")
    path = OUTPUT_ROOT / stage / "protocol.json"
    freeze_payload(path, payload(stage))
    print(path, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
