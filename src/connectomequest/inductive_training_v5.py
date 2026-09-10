"""Domain-balanced listwise training for the v5 observable frontier policy."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from connectomequest.inductive_training import ObservableIndex, _observation
from connectomequest.models.inductive import QUERY_TO_ID
from connectomequest.models.inductive_v5 import FEATURE_DIM_V5, candidate_features_v5
from connectomequest.query import Query, QueryType


@dataclass(frozen=True, slots=True)
class ListBatch:
    features: Tensor
    utilities: Tensor
    mask: Tensor
    query_type_ids: Tensor
    semantic_flags: Tensor

    def __len__(self) -> int:
        return len(self.query_type_ids)


@dataclass(slots=True)
class FeatureMoments:
    """Streaming feature moments; padded candidates never enter the estimate."""

    total: Tensor
    squared_total: Tensor
    count: int = 0

    @classmethod
    def empty(cls) -> FeatureMoments:
        return cls(
            torch.zeros(FEATURE_DIM_V5, dtype=torch.float64),
            torch.zeros(FEATURE_DIM_V5, dtype=torch.float64),
        )

    def update(self, batch: ListBatch) -> None:
        values = batch.features[batch.mask].double().cpu()
        if not len(values):
            return
        self.total += values.sum(dim=0)
        self.squared_total += values.square().sum(dim=0)
        self.count += len(values)

    def mean_std(self) -> tuple[Tensor, Tensor]:
        if self.count == 0:
            return torch.zeros(FEATURE_DIM_V5), torch.ones(FEATURE_DIM_V5)
        mean = self.total / self.count
        variance = (self.squared_total / self.count - mean.square()).clamp_min(1e-8)
        return mean.float(), variance.sqrt().float()


def _probe_count(rng: random.Random, choices: tuple[int, ...]) -> int:
    # Deliberately over-sample zero/low-probe observations to match deployment.
    weights = tuple(2 ** (-index) for index in range(len(choices)))
    return rng.choices(choices, weights=weights, k=1)[0]


def _append_list(
    rows: list[Tensor],
    utilities: list[Tensor],
    query_ids: list[int],
    semantic_flags: list[float],
    *,
    features: Tensor,
    values: list[float],
    query: Query,
) -> None:
    if len(values) < 2 or max(values) <= min(values):
        return
    rows.append(features)
    utilities.append(torch.tensor(values, dtype=torch.float32))
    query_ids.append(QUERY_TO_ID[query.query_type])
    semantic_flags.append(float(query.semantic_relation is not None))


def query_list_batch(
    index: ObservableIndex,
    queries: list[Query],
    *,
    probe_choices: tuple[int, ...] = (0, 1, 2, 4),
    seed: int = 17,
) -> ListBatch | None:
    """Build complete candidate lists using only protocol-observable inputs."""

    rng = random.Random(seed)
    rows: list[Tensor] = []
    utility_rows: list[Tensor] = []
    query_ids: list[int] = []
    semantic_flags: list[float] = []
    for query in queries:
        if query.query_type not in {
            QueryType.TWO_HOP,
            QueryType.TWO_HOP_TYPE,
            QueryType.INTERSECTION_NEGATION,
        }:
            continue
        first_ids, first_weights = index.neighbors(query.anchors[0])
        candidates = [int(node) for node in first_ids]
        if len(candidates) < 2:
            continue
        edges = [
            (query.anchors[0], int(dst), float(weight))
            for dst, weight in zip(first_ids, first_weights, strict=True)
        ]
        probe_count = min(len(candidates), _probe_count(rng, probe_choices))
        # Probe order is observable weight order, exactly as in the deployed controller.
        sketches = tuple(index.sketch(node) for node in candidates[:probe_count])
        observation = _observation(query, query.anchors[0], edges, node_sketches=sketches)
        if query.query_type == QueryType.INTERSECTION_NEGATION:
            values = [float(node in query.answers) for node in candidates]
        else:
            values = []
            for middle in candidates:
                successors, _ = index.neighbors(middle)
                values.append(float(sum(int(node) in query.answers for node in successors)))
        _append_list(
            rows,
            utility_rows,
            query_ids,
            semantic_flags,
            features=candidate_features_v5(observation, query.spec, candidates, frontier=True),
            values=values,
            query=query,
        )

        if query.query_type != QueryType.TWO_HOP_TYPE:
            continue
        useful = [node for node, value in zip(candidates, values, strict=True) if value > 0]
        rng.shuffle(useful)
        for middle in useful[:2]:
            second_ids, second_weights = index.neighbors(middle)
            second = [int(node) for node in second_ids]
            if len(second) < 2:
                continue
            second_observation = _observation(
                query,
                middle,
                edges
                + [
                    (middle, int(dst), float(weight))
                    for dst, weight in zip(second_ids, second_weights, strict=True)
                ],
                remaining_budget=56.0,
                step=8,
            )
            _append_list(
                rows,
                utility_rows,
                query_ids,
                semantic_flags,
                features=candidate_features_v5(
                    second_observation, query.spec, second, frontier=False
                ),
                values=[float(node in query.answers) for node in second],
                query=query,
            )

    if not rows:
        return None
    width = max(len(row) for row in rows)
    features = torch.zeros((len(rows), width, FEATURE_DIM_V5), dtype=torch.float32)
    values = torch.zeros((len(rows), width), dtype=torch.float32)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for index_row, (feature_row, utility_row) in enumerate(zip(rows, utility_rows, strict=True)):
        size = len(feature_row)
        features[index_row, :size] = feature_row
        values[index_row, :size] = utility_row
        mask[index_row, :size] = True
    return ListBatch(
        features,
        values,
        mask,
        torch.tensor(query_ids, dtype=torch.long),
        torch.tensor(semantic_flags, dtype=torch.float32),
    )


def listwise_loss(
    model: torch.nn.Module,
    batch: ListBatch,
    *,
    device: torch.device,
    temperature: float = 0.5,
) -> Tensor:
    shape = batch.features.shape
    flat_features = batch.features.reshape(-1, shape[-1]).to(device, non_blocking=True)
    query_ids = batch.query_type_ids[:, None].expand(-1, shape[1]).reshape(-1).to(device)
    semantic = batch.semantic_flags[:, None].expand(-1, shape[1]).reshape(-1).to(device)
    scores = model(flat_features, query_ids, semantic).reshape(shape[:2])
    mask = batch.mask.to(device)
    scores = scores.masked_fill(~mask, -torch.inf)
    utilities = torch.log1p(batch.utilities.to(device)) / temperature
    utilities = utilities.masked_fill(~mask, -torch.inf)
    targets = torch.softmax(utilities, dim=1)
    return -(targets * torch.log_softmax(scores, dim=1)).masked_fill(~mask, 0.0).sum(1).mean()


def train_domain_balanced_epoch(
    model: torch.nn.Module,
    indices: list[ObservableIndex],
    source_queries: list[list[Query]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    batch_queries: int = 128,
    probe_choices: tuple[int, ...] = (0, 1, 2, 4),
    temperature: float = 0.5,
    domain_variance_weight: float = 0.1,
    seed: int = 17,
    use_bf16: bool = True,
) -> tuple[dict[str, float | int], FeatureMoments]:
    """Interleave every domain in each optimizer step and penalize domain imbalance."""

    orders: list[list[int]] = []
    for source_index, queries in enumerate(source_queries):
        order = list(range(len(queries)))
        random.Random(seed + source_index).shuffle(order)
        orders.append(order)
    steps = max((len(order) + batch_queries - 1) // batch_queries for order in orders)
    losses: list[float] = []
    informative_lists = 0
    moments = FeatureMoments.empty()
    model.train()
    for step in range(steps):
        batches: list[ListBatch] = []
        for source_index, order in enumerate(orders):
            if not order:
                continue
            start = (step * batch_queries) % len(order)
            selected_indices = (order + order)[start : start + min(batch_queries, len(order))]
            selected = [source_queries[source_index][item] for item in selected_indices]
            batch = query_list_batch(
                indices[source_index],
                selected,
                probe_choices=probe_choices,
                seed=seed + step * len(orders) + source_index,
            )
            if batch is not None:
                batches.append(batch)
                moments.update(batch)
        if not batches:
            continue
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16 and device.type == "cuda",
        ):
            domain_losses = torch.stack(
                [
                    listwise_loss(model, batch, device=device, temperature=temperature)
                    for batch in batches
                ]
            )
            loss = domain_losses.mean()
            if len(domain_losses) > 1:
                loss = loss + domain_variance_weight * domain_losses.var(unbiased=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        informative_lists += sum(len(batch) for batch in batches)
    if not losses:
        raise ValueError("no informative observable ranking lists were generated")
    return {
        "loss": float(np.mean(losses)),
        "optimizer_steps": len(losses),
        "informative_lists": informative_lists,
    }, moments
