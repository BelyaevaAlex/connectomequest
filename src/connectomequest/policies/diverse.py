"""Observable diversity ensemble over independently trained LOCO policies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor

from connectomequest.env import Observation
from connectomequest.policies.base import FrontierPolicy
from connectomequest.policies.minerva import load_minerva_policy
from connectomequest.policies.snapshot import load_snapshot_policy
from connectomequest.proof import EvidenceKind
from connectomequest.query import QuerySpec
from connectomequest.rankers.base import AccessRegime


@dataclass(slots=True)
class DiversePolicy:
    """Round-robin component rankings to spend search budget on diverse branches."""

    snapshot: FrontierPolicy
    minerva: FrontierPolicy
    components: tuple[str, ...]
    seed: int = 17
    name: str = "diverse"
    access_regime: AccessRegime = AccessRegime.OBSERVABLE

    def _weight_order(self, observation: Observation, candidates: list[int]) -> list[int]:
        candidate_set = set(candidates)
        weights: dict[int, float] = {}
        for edge in observation.discovered_edges:
            if edge.kind != EvidenceKind.NON_EDGE and edge.dst in candidate_set:
                weights[edge.dst] = max(weights.get(edge.dst, 0.0), float(edge.weight))
        return sorted(candidates, key=lambda node: (-weights.get(node, 0.0), node))

    def _random_order(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
    ) -> list[int]:
        def key(node: int) -> tuple[int, int]:
            payload = f"{self.seed}:{query.query_id}:{observation.current_node}:{node}".encode()
            return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big"), node

        return sorted(candidates, key=key)

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
        if not candidates:
            return []
        orders = []
        for component in self.components:
            if component == "snapshot":
                orders.append(
                    self.snapshot.rank(
                        query,
                        observation,
                        candidates,
                        frontier=frontier,
                        device=device,
                        chunk_size=chunk_size,
                    )
                )
            elif component == "minerva":
                orders.append(
                    self.minerva.rank(
                        query,
                        observation,
                        candidates,
                        frontier=frontier,
                        device=device,
                        chunk_size=chunk_size,
                    )
                )
            elif component == "weight":
                orders.append(self._weight_order(observation, candidates))
            elif component == "random":
                orders.append(self._random_order(query, observation, candidates))
            else:
                raise ValueError(f"unknown diverse component: {component}")
        result: list[int] = []
        seen: set[int] = set()
        for rank in range(len(candidates)):
            for order in orders:
                if rank < len(order) and order[rank] not in seen:
                    result.append(order[rank])
                    seen.add(order[rank])
        if len(result) != len(candidates):
            raise RuntimeError("diverse ensemble returned an incomplete candidate order")
        return result

    @torch.inference_mode()
    def score(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
    ) -> Tensor:
        device = next(self.snapshot.parameters()).device
        frontier = observation.current_node == query.anchors[0]
        order = self.rank(
            query,
            observation,
            candidates,
            frontier=frontier,
            device=device,
        )
        ranks = {candidate: rank for rank, candidate in enumerate(order)}
        return torch.tensor(
            [len(candidates) - ranks[candidate] for candidate in candidates]
        ).float()


def load_diverse_policy(payload: dict, device: torch.device) -> DiversePolicy:
    if payload.get("model_kind") != "observable_diverse_policy_v1":
        raise ValueError("checkpoint is not a diverse policy")
    return DiversePolicy(
        snapshot=load_snapshot_policy(payload["snapshot_payload"], device),
        minerva=load_minerva_policy(payload["minerva_payload"], device),
        components=tuple(payload["components"]),
        seed=int(payload["seed"]),
        name=f"diverse_{'_'.join(payload['components'])}",
    )
