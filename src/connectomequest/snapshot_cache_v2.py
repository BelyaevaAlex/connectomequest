"""SHA-keyed feature cache for non-test DecisionSnapshot training views."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import pyarrow.parquet as pq
import torch

from connectomequest.decision_benchmark import DecisionSnapshot, DecisionStage
from connectomequest.manifest import sha256_file
from connectomequest.models import inductive_v5
from connectomequest.models.inductive import QUERY_TO_ID
from connectomequest.models.inductive_v5 import FEATURE_DIM_V5, candidate_features_v5
from connectomequest.snapshot_training import SnapshotLists


def _feature_code_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(inductive_v5.__file__)):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _verified_snapshot_sha(path: Path) -> str:
    if "confirmatory" in path.parts:
        raise ValueError("v2 training must not access frozen confirmatory snapshots")
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = sha256_file(path)
    if manifest["snapshot_sha256"] != actual:
        raise ValueError(f"snapshot SHA mismatch: {path}")
    return actual


def _read_stage(path: Path, stage: DecisionStage) -> list[DecisionSnapshot]:
    """Push the stage predicate into Arrow before decoding JSON observations."""

    table = pq.read_table(
        path,
        memory_map=True,
        use_threads=False,
        filters=[("stage", "=", str(stage))],
    )
    rows = [DecisionSnapshot.from_record(row) for row in table.to_pylist()]
    if any(snapshot.split == "test" for snapshot in rows):
        raise ValueError("v2 feature cache accepts only train/source-validation snapshots")
    return rows


def _build_lists(
    snapshots: list[DecisionSnapshot],
    *,
    stage: DecisionStage,
    maximum: int,
    seed: int,
) -> SnapshotLists:
    selected = [
        snapshot
        for snapshot in snapshots
        if snapshot.covered
        and len(snapshot.candidates) >= 2
        and len(snapshot.relevant_candidates) < len(snapshot.candidates)
    ]
    random.Random(seed).shuffle(selected)
    selected = selected[:maximum]
    if not selected:
        raise ValueError(f"no informative {stage} snapshots")
    width = max(len(snapshot.candidates) for snapshot in selected)
    rows = len(selected)
    features = torch.zeros(rows, width, FEATURE_DIM_V5)
    relevant = torch.zeros(rows, width)
    mask = torch.zeros(rows, width, dtype=torch.bool)
    query_ids = torch.empty(rows, dtype=torch.long)
    semantic = torch.empty(rows)
    stages = torch.full((rows,), 0 if stage == DecisionStage.FIRST_HOP else 1, dtype=torch.long)
    domains = torch.zeros(rows, dtype=torch.long)
    for index, snapshot in enumerate(selected):
        candidates = list(snapshot.candidates)
        size = len(candidates)
        features[index, :size] = candidate_features_v5(
            snapshot.observation,
            snapshot.query,
            candidates,
            frontier=stage == DecisionStage.FIRST_HOP,
        )
        relevant[index, :size] = torch.tensor(
            [float(candidate in snapshot.relevant_candidates) for candidate in candidates]
        )
        mask[index, :size] = True
        query_ids[index] = QUERY_TO_ID[snapshot.query.query_type]
        semantic[index] = float(snapshot.query.semantic_relation is not None)
    return SnapshotLists(features, relevant, mask, query_ids, semantic, stages, domains)


def cached_snapshot_lists_v2(
    path: Path,
    *,
    stage: DecisionStage,
    maximum: int,
    seed: int,
    cache_dir: Path,
) -> SnapshotLists:
    """Load/build one cache entry keyed by data, feature code, and sampling spec."""

    path = Path(path)
    snapshot_sha = _verified_snapshot_sha(path)
    key_payload = {
        "snapshot_sha256": snapshot_sha,
        "feature_code_sha256": _feature_code_sha256(),
        "stage": str(stage),
        "maximum": maximum,
        "seed": seed,
        "utility": "binary_relevance",
    }
    key = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload["key"] != key_payload:
            raise ValueError(f"feature-cache key collision: {cache_path}")
        return SnapshotLists(**payload["tensors"])
    lists = _build_lists(_read_stage(path, stage), stage=stage, maximum=maximum, seed=seed)
    payload = {
        "key": key_payload,
        "tensors": {name: getattr(lists, name) for name in SnapshotLists.__dataclass_fields__},
    }
    temporary = cache_path.with_suffix(f".{os.getpid()}.pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(cache_path)
    return lists
