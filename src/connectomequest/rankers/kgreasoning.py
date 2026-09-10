"""Candidate-only adapters for pinned Query2Box, BetaE, and ConE checkpoints."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import torch
from torch import Tensor

from connectomequest.env import Observation
from connectomequest.manifest import sha256_file
from connectomequest.query import QuerySpec, QueryType
from connectomequest.rankers.base import AccessRegime

PI_STRUCTURE = (("e", ("r", "r")), ("e", ("r",)))
QUERY_NAME_DICT = {
    ("e", ("r",)): "1p",
    ("e", ("r", "r")): "2p",
    (("e", ("r",)), ("e", ("r",))): "2i",
    ((("e", ("r",)), ("e", ("r",))), ("r",)): "ip",
    PI_STRUCTURE: "pi",
    (("e", ("r",)), ("e", ("r", "n"))): "2in",
}


def _load_module(path: Path, name: str) -> ModuleType:
    upstream = str(path.parent)
    sys.path.insert(0, upstream)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import upstream module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(upstream)


@dataclass(slots=True)
class KGReasoningRanker:
    """Score only supplied candidates with a full-graph CQA query encoder."""

    model_kind: str
    bundle: Path
    checkpoint_dir: Path
    upstream_root: Path
    device: torch.device
    name: str | None = None
    access_regime: AccessRegime = AccessRegime.FULL_GRAPH_ENCODER_VISIBLE_OUTPUT
    model: torch.nn.Module = field(init=False, repr=False)
    relations: dict[str, int] = field(init=False, repr=False)
    checkpoint_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.model_kind not in {"query2box", "betae", "cone"}:
            raise ValueError(f"unsupported KGReasoning model: {self.model_kind}")
        self.bundle = Path(self.bundle)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.upstream_root = Path(self.upstream_root)
        self.name = self.name or self.model_kind
        checkpoint_path = self.checkpoint_dir / "checkpoint"
        config_path = self.checkpoint_dir / "config.json"
        for required in (
            checkpoint_path,
            config_path,
            self.bundle / "relations.json",
            self.bundle / "bundle-manifest.json",
        ):
            if not required.exists():
                raise FileNotFoundError(required)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        bundle_manifest = json.loads(
            (self.bundle / "bundle-manifest.json").read_text(encoding="utf-8")
        )
        if int(config["nentity"]) != int(bundle_manifest["num_entities"]):
            raise ValueError("checkpoint and bundle entity vocabularies differ")
        if int(config["nrelation"]) != int(bundle_manifest["num_relations"]):
            raise ValueError("checkpoint and bundle relation vocabularies differ")
        self.relations = json.loads((self.bundle / "relations.json").read_text(encoding="utf-8"))
        if self.model_kind in {"query2box", "betae"}:
            module = _load_module(
                self.upstream_root / "kgreasoning" / "models.py",
                f"connectomequest_{self.model_kind}_models",
            )
            geo = "box" if self.model_kind == "query2box" else "beta"
            box_mode = ("none", 0.02)
            beta_mode = (1600, 2)
            model = module.KGReasoning(
                nentity=int(config["nentity"]),
                nrelation=int(config["nrelation"]),
                hidden_dim=int(config["hidden_dim"]),
                gamma=float(config["gamma"]),
                geo=geo,
                test_batch_size=1,
                box_mode=box_mode,
                beta_mode=beta_mode,
                use_cuda=self.device.type == "cuda",
                query_name_dict=QUERY_NAME_DICT,
            )
        else:
            module = _load_module(
                self.upstream_root / "cone" / "models.py",
                "connectomequest_cone_models",
            )
            model = module.KGReasoning(
                nentity=int(config["nentity"]),
                nrelation=int(config["nrelation"]),
                hidden_dim=int(config["hidden_dim"]),
                gamma=float(config["gamma"]),
                test_batch_size=1,
                use_cuda=self.device.type == "cuda",
                query_name_dict=QUERY_NAME_DICT,
                center_reg=float(config.get("center_reg", 0.02)),
                drop=float(config.get("drop", 0.0)),
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        self.model = model.to(self.device).eval()
        self.checkpoint_sha256 = sha256_file(checkpoint_path)

    def _flat_pi_query(self, query: QuerySpec) -> Tensor:
        if query.query_type != QueryType.TWO_HOP_TYPE or len(query.anchors) < 2:
            raise ValueError("KGReasoning decision adapter currently supports only 2pt")
        semantic = query.semantic_relation or "has_type"
        try:
            wiring = self.relations["presynaptic_to"]
            inverse_semantic = self.relations[f"inverse:{semantic}"]
        except KeyError as error:
            raise ValueError(f"bundle lacks relation required by 2pt: {error}") from error
        return torch.tensor(
            [
                [
                    query.anchors[0],
                    wiring,
                    wiring,
                    query.anchors[1],
                    inverse_semantic,
                ]
            ],
            dtype=torch.long,
            device=self.device,
        )

    @torch.inference_mode()
    def score(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
    ) -> Tensor:
        del observation
        if not candidates:
            return torch.empty(0, dtype=torch.float32)
        negative = torch.tensor(
            [candidates],
            dtype=torch.long,
            device=self.device,
        )
        query_tensor = self._flat_pi_query(query)
        _, logits, _, indexes = self.model(
            None,
            negative,
            None,
            {PI_STRUCTURE: query_tensor},
            {PI_STRUCTURE: [0]},
        )
        if indexes != [0] or logits.shape != (1, len(candidates)):
            raise RuntimeError(
                f"unexpected upstream output: indexes={indexes}, shape={tuple(logits.shape)}"
            )
        return logits[0].float().cpu()
