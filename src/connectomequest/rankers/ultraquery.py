"""Candidate-only adapter for the pinned official UltraQuery checkpoint."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from connectomequest.decision_benchmark import DecisionSnapshot
from connectomequest.env import Observation
from connectomequest.manifest import sha256_file
from connectomequest.query import QuerySpec, QueryType
from connectomequest.rankers.base import AccessRegime


@dataclass(slots=True)
class UltraQueryRanker:
    """Run a full-graph UltraQuery encoder and expose only legal candidates."""

    bundle: Path
    upstream_root: Path
    checkpoint: Path
    device: torch.device
    batch_size: int = 4
    name: str = "ultraquery"
    access_regime: AccessRegime = AccessRegime.FULL_GRAPH_ENCODER_VISIBLE_OUTPUT
    model: torch.nn.Module = field(init=False, repr=False)
    graph: object = field(init=False, repr=False)
    relations: dict[str, int] = field(init=False, repr=False)
    query_class: type = field(init=False, repr=False)
    checkpoint_sha256: str = field(init=False)
    _score_cache: dict[tuple, Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.bundle = Path(self.bundle).resolve()
        self.upstream_root = Path(self.upstream_root).resolve()
        self.checkpoint = Path(self.checkpoint).resolve()
        for required in (
            self.bundle / "bundle-manifest.json",
            self.bundle / "relations.json",
            self.checkpoint,
        ):
            if not required.exists():
                raise FileNotFoundError(required)

        sys.path.insert(0, str(self.upstream_root))
        try:
            from ultra import datasets_query
            from ultra.models import Ultra
            from ultra.query_utils import Query
            from ultra.ultraquery import UltraQuery

            bundle = self.bundle

            class ConnectomeQuestLogicalQuery(datasets_query.LogicalQueryDataset):
                name = "connectomequest"

                @property
                def raw_dir(self) -> str:
                    return str(bundle)

                @property
                def processed_dir(self) -> str:
                    return str(bundle / ".ultra-cache")

                def download(self) -> None:
                    raise RuntimeError("the local bundle is incomplete")

            dataset = ConnectomeQuestLogicalQuery(
                str(bundle), query_types=["pi"], force_reload=False
            )
            self.graph = dataset.test_graph.to(self.device)
            self.query_class = Query
            model = UltraQuery(
                model=Ultra(
                    rel_model_cfg={
                        "class": "RelNBFNet",
                        "input_dim": 64,
                        "hidden_dims": [64] * 6,
                        "message_func": "distmult",
                        "aggregate_func": "sum",
                        "short_cut": True,
                        "layer_norm": True,
                    },
                    entity_model_cfg={
                        "class": "QueryNBFNet",
                        "input_dim": 64,
                        "hidden_dims": [64] * 6,
                        "message_func": "distmult",
                        "aggregate_func": "sum",
                        "short_cut": True,
                        "layer_norm": True,
                    },
                ),
                logic="product",
                dropout_ratio=0.25,
                threshold=0.0,
                more_dropout=0.0,
            )
        finally:
            sys.path.remove(str(self.upstream_root))

        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        self.model = model.to(self.device).eval()
        self.relations = json.loads((self.bundle / "relations.json").read_text(encoding="utf-8"))
        self.checkpoint_sha256 = sha256_file(self.checkpoint)

    def _query(self, query: QuerySpec) -> Tensor:
        if query.query_type != QueryType.TWO_HOP_TYPE or len(query.anchors) < 2:
            raise ValueError("UltraQuery decision adapter currently supports only 2pt")
        semantic = query.semantic_relation or "has_type"
        try:
            wiring = self.relations["presynaptic_to"]
            inverse_semantic = self.relations[f"inverse:{semantic}"]
        except KeyError as error:
            raise ValueError(f"bundle lacks relation required by 2pt: {error}") from error
        nested = (
            (query.anchors[0], (wiring, wiring)),
            (query.anchors[1], (inverse_semantic,)),
        )
        return self.query_class.from_nested(nested).unsqueeze(0).to(self.device)

    @staticmethod
    def _score_key(
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
    ) -> tuple:
        return (query.query_id, observation.current_node, tuple(candidates))

    @torch.inference_mode()
    def prepare(self, snapshots: list[DecisionSnapshot]) -> None:
        """Batch full-graph encodings once, then cache candidate-only outputs."""

        grouped: dict[str, list[DecisionSnapshot]] = defaultdict(list)
        for snapshot in snapshots:
            grouped[snapshot.query.query_id].append(snapshot)
        query_groups = list(grouped.values())
        for start in range(0, len(query_groups), self.batch_size):
            selected = query_groups[start : start + self.batch_size]
            query_tensor = self.query_class(
                torch.cat([self._query(rows[0].query) for rows in selected], dim=0)
            )
            logits = self.model(self.graph, query_tensor, symbolic_traversal=False)
            for row_index, rows in enumerate(selected):
                for snapshot in rows:
                    candidates = list(snapshot.candidates)
                    candidate_index = torch.tensor(
                        candidates,
                        dtype=torch.long,
                        device=self.device,
                    )
                    key = self._score_key(snapshot.query, snapshot.observation, candidates)
                    self._score_cache[key] = (
                        logits[row_index].index_select(0, candidate_index).float().cpu()
                    )

    @torch.inference_mode()
    def score(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
    ) -> Tensor:
        if not candidates:
            return torch.empty(0, dtype=torch.float32)
        key = self._score_key(query, observation, candidates)
        if key in self._score_cache:
            return self._score_cache[key]
        logits = self.model(self.graph, self._query(query), symbolic_traversal=False)
        candidate_index = torch.tensor(candidates, dtype=torch.long, device=self.device)
        scores = logits[0].index_select(0, candidate_index).float().cpu()
        self._score_cache[key] = scores
        return scores
