"""Write-once protocol and shard utilities for V39 full episodes."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "outputs/nemo/strong-accept/full-episode-v39-attempt-08"
SOURCE_FILES = (
    "src/connectomequest/embodied/types.py",
    "src/connectomequest/embodied/environment.py",
    "src/connectomequest/embodied/updates.py",
    "src/connectomequest/embodied/revalidation.py",
    "src/connectomequest/embodied/task_planner.py",
    "src/connectomequest/embodied/episode_runner.py",
    "src/connectomequest/embodied/statistics.py",
    "src/connectomequest/embodied/evidence_index.py",
    "src/connectomequest/embodied/search.py",
    "src/connectomequest/embodied/dynamic_task.py",
    "src/connectomequest/embodied/symbolic_controller.py",
    "src/connectomequest/embodied/protocol.py",
    "scripts/freeze_embodied_protocol.py",
    "scripts/run_embodied_evaluation.py",
    "scripts/report_embodied_evaluation.py",
)
CHECKPOINT = (
    ROOT / "outputs/nemo/strong-accept/dependent-plan-v1/development-attempt-02/perceptor.pt"
)


def _canonical(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(root: Path, files: Iterable[str]) -> dict[str, str]:
    return {str(name): sha256_file(root / name) for name in sorted(str(item) for item in files)}


def verify_source_hashes(root: Path, expected: Mapping[str, str]) -> None:
    actual = source_hashes(root, expected)
    if actual != dict(expected):
        changed = sorted(
            name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name)
        )
        raise ValueError(f"source hash mismatch: {', '.join(changed)}")


def verify_archived_source_hashes(archive_path: Path, expected: Mapping[str, str]) -> None:
    """Verify sealed source bytes stored in the publication provenance archive."""

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        index = (
            json.loads(archive.read("provenance/SOURCE_INDEX.json"))
            if "provenance/SOURCE_INDEX.json" in names
            else {}
        )
        changed: list[str] = []
        for historical_path, digest in expected.items():
            indexed_digest = index.get(historical_path)
            indexed_name = (
                f"provenance/source/{indexed_digest}.txt" if indexed_digest is not None else None
            )
            archived_name = (
                indexed_name
                if indexed_name is not None and indexed_name in names
                else historical_path
            )
            if archived_name not in names:
                changed.append(historical_path)
                continue
            actual = hashlib.sha256(archive.read(archived_name)).hexdigest()
            if actual != digest:
                changed.append(historical_path)
    if changed:
        raise ValueError(f"archived source hash mismatch: {', '.join(sorted(changed))}")


def freeze_payload(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical(payload) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"write-once payload mismatch: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def atomic_case_write(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical(payload) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"duplicate shard mismatch: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_text() != encoded:
            raise ValueError(f"duplicate shard mismatch: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _seed_block(stage: str, per_stratum: int) -> dict[str, list[int]]:
    bases = {
        "smoke": 392_000_000,
        "pilot": 393_000_000,
        "confirmation": 394_000_000,
    }
    base = bases[stage]
    return {
        stratum: list(range(base + index * 100_000, base + index * 100_000 + per_stratum))
        for index, stratum in enumerate(("short", "medium", "long", "very_long"))
    }


def protocol_payload(stage: str) -> dict:
    if stage not in ("smoke", "pilot", "confirmation"):
        raise ValueError(f"unknown stage: {stage}")
    per_stratum = {"smoke": 5, "pilot": 200, "confirmation": 300}[stage]
    files = source_hashes(ROOT, SOURCE_FILES)
    return {
        "protocol_version": "v39-attempt-08",
        "stage": stage,
        "layout_seeds_by_stratum": _seed_block(stage, per_stratum),
        "update_seeds": [17, 29, 43],
        "methods": [
            "full_replan",
            "full_scan",
            "receipt_index",
            "unchecked_reuse",
        ],
        "schedule_cells": [
            {
                "condition": "none",
                "update_count": 0,
                "relevant_fraction": 0.0,
            },
            {
                "condition": "irrelevant_receipt_withdrawal",
                "update_count": 1,
                "relevant_fraction": 0.0,
            },
            {
                "condition": "alternative_support",
                "update_count": 1,
                "relevant_fraction": 1.0,
            },
            {
                "condition": "final_support_loss",
                "update_count": 1,
                "relevant_fraction": 1.0,
            },
            {
                "condition": "semantic_correction",
                "update_count": 1,
                "relevant_fraction": 1.0,
            },
            {
                "condition": "sparse_burst",
                "update_count": 8,
                "relevant_fraction": 0.1,
            },
        ],
        "horizon": 8192,
        "common_planner": "full-mission-astar-v2",
        "tie_break": "action-order-v1",
        "action_model": "unlockpickupdist-v1",
        "public_observation": (
            "egocentric RGB, relative pose, direction, inventory type, "
            "mission, acknowledgement, reward, termination"
        ),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "source_hashes": files,
        "bootstrap_seed": 39_039,
        "bootstrap_draws": 10_000,
        "bootstrap_cluster": "layout_seed",
        "primary_contrast": (
            "receipt_index minus full_scan; pooled long/very_long; "
            "one or two updates; at least 80 percent irrelevant"
        ),
        "gates": {
            "task_success_ci_low": -0.01,
            "time_reduction": 0.20,
            "time_reduction_ci_low": 0.10,
            "work_reduction": 0.25,
            "wall_clock_increase_max": 0.01,
        },
        "oracle_fields_in_method_input": [],
    }


def assert_confirmation_can_open(output_root: Path = OUTPUT_ROOT) -> None:
    path = Path(output_root) / "pilot" / "verification.json"
    if not path.exists():
        raise ValueError("pilot verification is missing")
    verification = json.loads(path.read_text())
    if not verification.get("integrity_passed"):
        raise ValueError("pilot verification did not pass integrity")
    attainable = int(verification.get("primary_attainable_layouts", 0))
    if attainable < 300:
        raise ValueError(f"pilot established {attainable} primary layouts; 300 are required")


def freeze_stage(stage: str, output_root: Path = OUTPUT_ROOT) -> Path:
    if stage == "confirmation":
        assert_confirmation_can_open(output_root)
    payload = protocol_payload(stage)
    path = Path(output_root) / stage / "protocol.json"
    freeze_payload(path, payload)
    return path
