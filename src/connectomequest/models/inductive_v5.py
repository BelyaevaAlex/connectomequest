"""Domain-robust, entity-ID-free frontier policy for ConnectomeQuest v5."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from connectomequest.env import Observation
from connectomequest.models.inductive import QUERY_TO_ID, candidate_features
from connectomequest.query import QuerySpec

BASE_FEATURE_DIM = 22
PROBE_STAT_NAMES = ("outgoing_degree", "total_weight", "max_weight", "mean_weight")
PROBE_FEATURE_DIM = 1 + 2 * len(PROBE_STAT_NAMES)
FEATURE_DIM_V5 = BASE_FEATURE_DIM + PROBE_FEATURE_DIM
WEIGHT_RANK_INDEX = 19
OOD_FEATURE_INDICES = (17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30)


def _percentile_and_robust_z(
    values: list[float], present: list[bool]
) -> tuple[list[float], list[float]]:
    """Return relabeling-invariant within-observation ranks and robust z scores."""

    selected = [value for value, keep in zip(values, present, strict=True) if keep]
    if not selected:
        return [0.0] * len(values), [0.0] * len(values)
    ordered = sorted(selected)
    denominator = max(1, len(ordered) - 1)
    percentiles: list[float] = []
    for value, keep in zip(values, present, strict=True):
        if not keep:
            percentiles.append(0.0)
            continue
        below = sum(candidate < value for candidate in ordered)
        tied = sum(candidate == value for candidate in ordered)
        percentiles.append((below + 0.5 * max(0, tied - 1)) / denominator)
    median = ordered[len(ordered) // 2]
    deviations = sorted(abs(value - median) for value in ordered)
    mad = deviations[len(deviations) // 2]
    scale = max(1e-6, 1.4826 * mad)
    robust = [
        max(-5.0, min(5.0, (value - median) / scale)) / 5.0 if keep else 0.0
        for value, keep in zip(values, present, strict=True)
    ]
    return percentiles, robust


def candidate_features_v5(
    observation: Observation,
    query: QuerySpec,
    candidates: list[int],
    *,
    frontier: bool,
) -> Tensor:
    """Build relative two-tower features without absolute entity or dataset IDs."""

    if not candidates:
        return torch.empty((0, FEATURE_DIM_V5), dtype=torch.float32)
    legacy = candidate_features(observation, query, candidates, frontier=frontier)
    base = legacy[:, :BASE_FEATURE_DIM]
    sketches = {sketch.node_id: sketch for sketch in observation.node_sketches}
    present = [candidate in sketches for candidate in candidates]
    columns: list[list[float]] = []
    for name in PROBE_STAT_NAMES:
        values = [
            math.log1p(max(0.0, float(getattr(sketches[candidate], name))))
            if candidate in sketches
            else 0.0
            for candidate in candidates
        ]
        percentile, robust = _percentile_and_robust_z(values, present)
        columns.extend((percentile, robust))
    probe = torch.tensor(
        [
            [float(present[index]), *(column[index] for column in columns)]
            for index in range(len(candidates))
        ],
        dtype=torch.float32,
    )
    return torch.cat((base, probe), dim=1)


@dataclass(frozen=True, slots=True)
class PolicyUncertainty:
    margin: float
    normalized_entropy: float
    ood_score: float


class InductiveQueryPolicyV5(nn.Module):
    """Two-tower scorer with a relative weight prior and observable OOD statistics."""

    def __init__(
        self,
        hidden_dim: int = 128,
        query_dim: int = 32,
        dropout: float = 0.1,
        *,
        use_weight_prior: bool = True,
        use_probe_features: bool = True,
        use_query_embedding: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.dropout = dropout
        self.use_weight_prior = use_weight_prior
        self.use_probe_features = use_probe_features
        self.use_query_embedding = use_query_embedding
        self.feature_dim = FEATURE_DIM_V5
        self.query_embedding = nn.Embedding(len(QUERY_TO_ID), query_dim)
        common = query_dim + 1
        self.base_scorer = nn.Sequential(
            nn.Linear(BASE_FEATURE_DIM + common, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        probe_hidden = max(32, hidden_dim // 2)
        self.probe_scorer = nn.Sequential(
            nn.Linear(PROBE_FEATURE_DIM + common, probe_hidden),
            nn.LayerNorm(probe_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(probe_hidden, 1),
        )
        nn.init.zeros_(self.base_scorer[-1].weight)
        nn.init.zeros_(self.base_scorer[-1].bias)
        nn.init.zeros_(self.probe_scorer[-1].weight)
        nn.init.zeros_(self.probe_scorer[-1].bias)
        self.relative_weight_scale_raw = nn.Parameter(torch.tensor(-1.0))
        self.register_buffer("feature_reference_mean", torch.zeros(FEATURE_DIM_V5))
        self.register_buffer("feature_reference_std", torch.ones(FEATURE_DIM_V5))

    def set_feature_reference(self, mean: Tensor, std: Tensor) -> None:
        if mean.shape != (FEATURE_DIM_V5,) or std.shape != (FEATURE_DIM_V5,):
            raise ValueError("feature reference has an unexpected shape")
        self.feature_reference_mean.copy_(mean)
        self.feature_reference_std.copy_(std.clamp_min(1e-4))

    def forward(
        self,
        features: Tensor,
        query_type_ids: Tensor,
        semantic_flags: Tensor,
    ) -> Tensor:
        embedded_query = self.query_embedding(query_type_ids)
        query_vector = (
            embedded_query if self.use_query_embedding else torch.zeros_like(embedded_query)
        )
        common = torch.cat((query_vector, semantic_flags[:, None]), dim=-1)
        base = features[:, :BASE_FEATURE_DIM]
        probe = features[:, BASE_FEATURE_DIM:]
        base_score = self.base_scorer(torch.cat((base, common), dim=-1)).squeeze(-1)
        if self.use_probe_features:
            probe_score = self.probe_scorer(torch.cat((probe, common), dim=-1)).squeeze(-1)
            probe_contribution = probe[:, 0] * probe_score
        else:
            probe_contribution = torch.zeros_like(base_score)
        if self.use_weight_prior:
            relative_weight_scale = torch.nn.functional.softplus(self.relative_weight_scale_raw)
            prior_contribution = relative_weight_scale * base[:, WEIGHT_RANK_INDEX]
        else:
            prior_contribution = torch.zeros_like(base_score)
        return base_score + probe_contribution + prior_contribution

    def _features_and_scores(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
        *,
        frontier: bool,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        features = candidate_features_v5(observation, query, candidates, frontier=frontier).to(
            device
        )
        query_ids = torch.full(
            (len(candidates),), QUERY_TO_ID[query.query_type], dtype=torch.long, device=device
        )
        semantic_flags = torch.full(
            (len(candidates),),
            float(query.semantic_relation is not None),
            dtype=torch.float32,
            device=device,
        )
        return features, self(features, query_ids, semantic_flags)

    @torch.inference_mode()
    def score_candidates(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
        *,
        frontier: bool,
        device: torch.device,
    ) -> dict[int, float]:
        if not candidates:
            return {}
        self.eval()
        _, scores = self._features_and_scores(
            query, observation, candidates, frontier=frontier, device=device
        )
        return {
            candidate: float(score)
            for candidate, score in zip(candidates, scores.float().cpu(), strict=True)
        }

    @torch.inference_mode()
    def uncertainty(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
        *,
        frontier: bool,
        device: torch.device,
    ) -> PolicyUncertainty:
        if len(candidates) < 2:
            return PolicyUncertainty(float("inf"), 0.0, 0.0)
        self.eval()
        features, scores = self._features_and_scores(
            query, observation, candidates, frontier=frontier, device=device
        )
        ordered = torch.sort(scores.float(), descending=True).values
        margin = float((ordered[0] - ordered[1]).cpu())
        probabilities = torch.softmax(scores.float(), dim=0)
        entropy = -torch.sum(probabilities * torch.log(probabilities.clamp_min(1e-12)))
        normalized_entropy = float((entropy / math.log(len(candidates))).cpu())
        indices = torch.tensor(OOD_FEATURE_INDICES, device=device)
        selected = torch.index_select(features.float(), 1, indices)
        mean = torch.index_select(self.feature_reference_mean.to(device), 0, indices)
        std = torch.index_select(self.feature_reference_std.to(device), 0, indices)
        ood = float(torch.mean(torch.abs((selected - mean) / std.clamp_min(1e-4))).cpu())
        return PolicyUncertainty(margin, normalized_entropy, ood)

    @torch.inference_mode()
    def rank(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
        *,
        frontier: bool,
        device: torch.device,
        chunk_size: int = 65_536,
    ) -> list[int]:
        del chunk_size
        scores = self.score_candidates(
            query, observation, candidates, frontier=frontier, device=device
        )
        return sorted(candidates, key=lambda node: (-scores[node], node))
