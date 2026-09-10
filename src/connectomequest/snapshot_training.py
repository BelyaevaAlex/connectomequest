"""Balanced listwise training data derived from immutable DecisionSnapshots."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from connectomequest.decision_benchmark import (
    DecisionSnapshot,
    DecisionStage,
    read_decision_snapshots,
)
from connectomequest.models.inductive import QUERY_TO_ID
from connectomequest.models.inductive_v5 import FEATURE_DIM_V5, candidate_features_v5


@dataclass(frozen=True, slots=True)
class SnapshotLists:
    features: Tensor
    relevant: Tensor
    mask: Tensor
    query_ids: Tensor
    semantic_flags: Tensor
    stages: Tensor
    domains: Tensor

    def __len__(self) -> int:
        return len(self.stages)

    def select(self, indices: Tensor) -> SnapshotLists:
        return SnapshotLists(
            self.features[indices],
            self.relevant[indices],
            self.mask[indices],
            self.query_ids[indices],
            self.semantic_flags[indices],
            self.stages[indices],
            self.domains[indices],
        )


def _sample_stage(
    snapshots: list[DecisionSnapshot],
    stage: DecisionStage,
    maximum: int,
    seed: int,
) -> list[DecisionSnapshot]:
    selected = [
        snapshot
        for snapshot in snapshots
        if snapshot.stage == stage
        and snapshot.covered
        and len(snapshot.candidates) >= 2
        and len(snapshot.relevant_candidates) < len(snapshot.candidates)
    ]
    random.Random(seed).shuffle(selected)
    return selected[:maximum]


def prepare_snapshot_lists(
    paths: list[Path],
    *,
    max_per_stage_source: int,
    sampling_seed: int = 1729,
) -> SnapshotLists:
    rows: list[tuple[DecisionSnapshot, int]] = []
    for domain, path in enumerate(paths):
        snapshots = read_decision_snapshots(path)
        for stage_index, stage in enumerate(DecisionStage):
            selected = _sample_stage(
                snapshots,
                stage,
                max_per_stage_source,
                sampling_seed + 101 * domain + stage_index,
            )
            rows.extend((snapshot, domain) for snapshot in selected)
    if not rows:
        raise ValueError("no informative covered DecisionSnapshots")
    width = max(len(snapshot.candidates) for snapshot, _ in rows)
    features = torch.zeros((len(rows), width, FEATURE_DIM_V5), dtype=torch.float32)
    relevant = torch.zeros((len(rows), width), dtype=torch.float32)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    query_ids = torch.empty(len(rows), dtype=torch.long)
    semantic = torch.empty(len(rows), dtype=torch.float32)
    stages = torch.empty(len(rows), dtype=torch.long)
    domains = torch.empty(len(rows), dtype=torch.long)
    for index, (snapshot, domain) in enumerate(rows):
        candidates = list(snapshot.candidates)
        size = len(candidates)
        frontier = snapshot.stage == DecisionStage.FIRST_HOP
        features[index, :size] = candidate_features_v5(
            snapshot.observation,
            snapshot.query,
            candidates,
            frontier=frontier,
        )
        relevant[index, :size] = torch.tensor(
            [float(candidate in snapshot.relevant_candidates) for candidate in candidates]
        )
        mask[index, :size] = True
        query_ids[index] = QUERY_TO_ID[snapshot.query.query_type]
        semantic[index] = float(snapshot.query.semantic_relation is not None)
        stages[index] = 0 if frontier else 1
        domains[index] = domain
    return SnapshotLists(features, relevant, mask, query_ids, semantic, stages, domains)


def listwise_snapshot_loss(scores: Tensor, relevant: Tensor, mask: Tensor) -> Tensor:
    scores = scores.masked_fill(~mask, -torch.inf)
    targets = relevant / relevant.sum(dim=1, keepdim=True).clamp_min(1.0)
    return -(targets * torch.log_softmax(scores, dim=1)).masked_fill(~mask, 0.0).sum(1).mean()


def batch_first_proof_mrr(scores: Tensor, relevant: Tensor, mask: Tensor) -> Tensor:
    scores = scores.masked_fill(~mask, -torch.inf)
    order = torch.argsort(scores, dim=1, descending=True)
    ordered_relevant = torch.gather(relevant.bool(), 1, order)
    first = torch.argmax(ordered_relevant.to(torch.int64), dim=1) + 1
    covered = ordered_relevant.any(dim=1)
    return torch.where(covered, first.float().reciprocal(), torch.zeros_like(first.float()))
