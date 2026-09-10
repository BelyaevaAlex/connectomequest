"""Balanced, stage-aware objectives for SnapshotPolicyV2."""

from __future__ import annotations

import torch
from torch import Tensor

from connectomequest.snapshot_training import SnapshotLists


def concatenate_snapshot_lists(parts: list[SnapshotLists]) -> SnapshotLists:
    """Concatenate listwise datasets after padding their candidate dimension."""

    if not parts:
        raise ValueError("at least one SnapshotLists object is required")
    width = max(part.features.shape[1] for part in parts)
    feature_dimension = parts[0].features.shape[2]
    if any(part.features.shape[2] != feature_dimension for part in parts):
        raise ValueError("feature dimensions must match")

    def pad_features(value: Tensor) -> Tensor:
        padding = width - value.shape[1]
        return torch.nn.functional.pad(value, (0, 0, 0, padding))

    def pad_candidates(value: Tensor) -> Tensor:
        padding = width - value.shape[1]
        return torch.nn.functional.pad(value, (0, padding))

    return SnapshotLists(
        features=torch.cat([pad_features(part.features) for part in parts]),
        relevant=torch.cat([pad_candidates(part.relevant) for part in parts]),
        mask=torch.cat([pad_candidates(part.mask) for part in parts]),
        query_ids=torch.cat([part.query_ids for part in parts]),
        semantic_flags=torch.cat([part.semantic_flags for part in parts]),
        stages=torch.cat([part.stages for part in parts]),
        domains=torch.cat([part.domains for part in parts]),
    )


def stage_aware_listwise_utility_loss(
    scores: Tensor,
    utility: Tensor,
    mask: Tensor,
    stages: Tensor,
    *,
    first_hop_weight: float = 2.0,
    second_hop_weight: float = 1.0,
) -> Tensor:
    """Optimize non-negative proof utility with explicit stage emphasis.

    ``utility`` can be binary relevance today and proof-completion counts in a
    future train-only snapshot schema. Padding is excluded before both softmax
    and target normalization.
    """

    if scores.shape != utility.shape or scores.shape != mask.shape:
        raise ValueError("scores, utility, and mask must have identical shapes")
    if stages.ndim != 1 or len(stages) != len(scores):
        raise ValueError("stages must contain one entry per snapshot")
    if first_hop_weight <= 0.0 or second_hop_weight <= 0.0:
        raise ValueError("stage weights must be positive")
    masked_scores = scores.masked_fill(~mask, -torch.inf)
    positive = utility.clamp_min(0.0).masked_fill(~mask, 0.0)
    targets = positive / positive.sum(dim=1, keepdim=True).clamp_min(1.0)
    per_row = (
        -(targets * torch.log_softmax(masked_scores, dim=1)).masked_fill(~mask, 0.0).sum(dim=1)
    )
    weights = torch.where(
        stages.to(scores.device) == 0,
        torch.as_tensor(first_hop_weight, dtype=scores.dtype, device=scores.device),
        torch.as_tensor(second_hop_weight, dtype=scores.dtype, device=scores.device),
    )
    return torch.sum(per_row * weights) / weights.sum().clamp_min(1e-12)


def balanced_domain_stage_indices(
    data: SnapshotLists,
    *,
    samples_per_stratum: int,
    generator: torch.Generator,
) -> Tensor:
    """Draw equal domain-stage strata, with replacement only for small strata."""

    if samples_per_stratum <= 0:
        raise ValueError("samples_per_stratum must be positive")
    strata = sorted(set(zip(data.domains.tolist(), data.stages.tolist(), strict=True)))
    if not strata:
        raise ValueError("cannot sample empty SnapshotLists")
    selected: list[Tensor] = []
    for domain, stage in strata:
        pool = torch.nonzero(
            (data.domains == domain) & (data.stages == stage), as_tuple=False
        ).flatten()
        if len(pool) >= samples_per_stratum:
            order = torch.randperm(len(pool), generator=generator)
            draw = pool[order[:samples_per_stratum]]
        else:
            draw = pool[torch.randint(len(pool), (samples_per_stratum,), generator=generator)]
        selected.append(draw)
    indices = torch.cat(selected)
    return indices[torch.randperm(len(indices), generator=generator)]
