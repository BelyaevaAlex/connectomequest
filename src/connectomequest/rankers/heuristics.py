"""Cheap rankers using only frozen observations, plus a labeled oracle ceiling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor

from connectomequest.decision_benchmark import DecisionSnapshot
from connectomequest.env import Observation
from connectomequest.proof import EvidenceKind
from connectomequest.query import QuerySpec
from connectomequest.rankers.base import AccessRegime


@dataclass(frozen=True, slots=True)
class WeightRanker:
    name: str = "weight"
    access_regime: AccessRegime = AccessRegime.OBSERVABLE

    def score(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
    ) -> Tensor:
        del query
        candidate_set = set(candidates)
        weights: dict[int, float] = {}
        for edge in observation.discovered_edges:
            if (
                edge.kind != EvidenceKind.NON_EDGE
                and edge.src == observation.current_node
                and edge.dst in candidate_set
            ):
                weights[edge.dst] = max(weights.get(edge.dst, 0.0), float(edge.weight))
        return torch.tensor(
            [weights.get(candidate, 0.0) for candidate in candidates],
            dtype=torch.float32,
        )


@dataclass(frozen=True, slots=True)
class DegreeRanker:
    name: str = "degree"
    access_regime: AccessRegime = AccessRegime.OBSERVABLE_PAID_LOOKAHEAD

    def score(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
    ) -> Tensor:
        del query
        sketches = {sketch.node_id: sketch for sketch in observation.node_sketches}
        values = []
        for candidate in candidates:
            sketch = sketches.get(candidate)
            if sketch is None:
                values.append(-1.0)
            else:
                values.append(float(sketch.outgoing_degree) + 1e-9 * float(sketch.total_weight))
        return torch.tensor(values, dtype=torch.float32)


@dataclass(frozen=True, slots=True)
class RandomRanker:
    seed: int = 17
    name: str = "random"
    access_regime: AccessRegime = AccessRegime.OBSERVABLE

    def score(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
    ) -> Tensor:
        values = []
        for candidate in candidates:
            payload = (
                f"{self.seed}:{query.query_id}:{observation.current_node}:{candidate}"
            ).encode()
            digest = hashlib.blake2b(payload, digest_size=8).digest()
            values.append(int.from_bytes(digest, "big") / float(2**64))
        return torch.tensor(values, dtype=torch.float32)


@dataclass(frozen=True, slots=True)
class ObservableOracleRanker:
    relevance: dict[tuple[str, int], frozenset[int]]
    name: str = "oracle"
    access_regime: AccessRegime = AccessRegime.PRIVILEGED_CEILING

    @classmethod
    def from_snapshots(
        cls,
        snapshots: list[DecisionSnapshot],
    ) -> ObservableOracleRanker:
        relevance = {
            (
                snapshot.query.query_id,
                snapshot.observation.current_node,
            ): snapshot.relevant_candidates
            for snapshot in snapshots
        }
        return cls(relevance)

    def score(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
    ) -> Tensor:
        relevant = self.relevance.get(
            (query.query_id, observation.current_node),
            frozenset(),
        )
        return torch.tensor(
            [float(candidate in relevant) for candidate in candidates],
            dtype=torch.float32,
        )
