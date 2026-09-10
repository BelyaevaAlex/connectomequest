"""Bounded, source-only calibration for the observable contextual expert gate."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch import Tensor

from connectomequest.decision_benchmark import DecisionSnapshot, DecisionStage
from connectomequest.manifest import sha256_file
from connectomequest.policies.base import FrontierPolicy
from connectomequest.policies.expert_gate_v2 import (
    ContextualExpertGateV2,
    ExpertGateNetwork,
    context_features_v2,
)


@dataclass(frozen=True, slots=True)
class GateExamplesV2:
    features: Tensor
    utilities: Tensor
    domains: tuple[str, ...]
    probe_regimes: tuple[int, ...]
    snapshot_ids: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.snapshot_ids)


@dataclass(frozen=True, slots=True)
class ThresholdCalibrationV2:
    threshold: float
    mean_utility: float
    fallback_fraction: float
    selected_counts: tuple[int, ...]
    fallback_mean_utility: float
    paired_gain: float
    paired_gain_lcb: float
    bootstrap_samples: int
    confidence: float


def _domain_key(value: str) -> str:
    normalized = value.lower().replace("_", "-")
    for canonical in ("h01", "manc", "hemibrain"):
        if normalized == canonical or normalized.startswith(canonical + "-"):
            return canonical
    return normalized.replace("-kg", "").replace("-", "")


def verified_source_validation_sha_v2(
    path: Path,
    *,
    source_domain: str,
    held_out: str,
) -> str:
    """Fail closed on target, test, confirmatory, or mislabeled source data."""

    path = Path(path)
    if "confirmatory" in path.parts:
        raise ValueError("gate calibration cannot access confirmatory snapshots")
    if _domain_key(source_domain) == _domain_key(held_out):
        raise ValueError("held-out domain cannot be a gate-calibration source")
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = sha256_file(path)
    if manifest.get("snapshot_sha256") != actual:
        raise ValueError(f"snapshot SHA mismatch: {path}")
    table = pq.read_table(path, columns=["held_out", "split", "stage"], memory_map=True)
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"empty source validation snapshots: {path}")
    if any(_domain_key(row["held_out"]) != _domain_key(source_domain) for row in rows):
        raise ValueError(f"source domain label mismatch: {path}")
    if any(row["split"] != "validation" for row in rows):
        raise ValueError("gate calibration accepts validation snapshots only")
    return actual


def source_probe_matrix_v2(
    sources: list[tuple[str, Path]],
    *,
    held_out: str,
) -> dict[str, dict[str, str | int]]:
    """Validate two source domains with one immutable file per probe regime."""

    if len(sources) != 8 or len({_domain_key(domain) for domain, _ in sources}) != 2:
        raise ValueError("gate calibration requires 8 files: 2 source domains x 4 probes")
    if len({Path(path).resolve() for _, path in sources}) != 8:
        raise ValueError("source validation snapshot paths must be unique")
    metadata: dict[str, dict[str, str | int]] = {}
    grouped: dict[str, set[int]] = {}
    for domain, raw_path in sources:
        path = Path(raw_path)
        snapshot_sha = verified_source_validation_sha_v2(
            path,
            source_domain=domain,
            held_out=held_out,
        )
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        try:
            probe_regime = int(manifest["protocol"]["subgoal_probes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"missing probe regime in manifest: {path}") from error
        if probe_regime not in {0, 1, 2, 4}:
            raise ValueError(f"unsupported probe regime {probe_regime}: {path}")
        key = _domain_key(domain)
        if probe_regime in grouped.setdefault(key, set()):
            raise ValueError(f"duplicate probe regime {probe_regime} for {domain}")
        grouped[key].add(probe_regime)
        metadata[str(path.resolve())] = {
            "sha256": snapshot_sha,
            "source_domain": domain,
            "probe_regime": probe_regime,
        }
    if any(regimes != {0, 1, 2, 4} for regimes in grouped.values()):
        raise ValueError("each source domain must provide probe regimes 0, 1, 2, and 4")
    return metadata


def _source_first_hop_rows(
    path: Path,
    *,
    source_domain: str,
    held_out: str,
    maximum: int,
    seed: int,
) -> list[DecisionSnapshot]:
    verified_source_validation_sha_v2(
        path,
        source_domain=source_domain,
        held_out=held_out,
    )
    table = pq.read_table(
        path,
        memory_map=True,
        filters=[("stage", "=", str(DecisionStage.FIRST_HOP))],
    )
    rows = [
        DecisionSnapshot.from_record(row)
        for row in table.to_pylist()
        if row["covered"] and len(row["candidates"]) >= 2
    ]
    random.Random(seed).shuffle(rows)
    rows = rows[:maximum]
    if not rows:
        raise ValueError(f"no covered first-hop source validation snapshots: {path}")
    return rows


def first_proof_reciprocal_rank_v2(
    order: list[int],
    relevant_candidates: frozenset[int],
) -> float:
    for rank, candidate in enumerate(order, start=1):
        if candidate in relevant_candidates:
            return 1.0 / rank
    return 0.0


def build_gate_examples_v2(
    sources: list[tuple[str, Path]],
    *,
    held_out: str,
    experts: dict[str, FrontierPolicy],
    fallback_expert: str,
    second_hop_expert: str,
    maximum_per_source: int = 1000,
    seed: int = 1729,
) -> GateExamplesV2:
    """Materialize tiny observable features and evaluator-only utilities on CPU."""

    source_metadata = source_probe_matrix_v2(sources, held_out=held_out)
    if maximum_per_source < 2:
        raise ValueError("maximum_per_source must be at least 2")
    expert_names = tuple(experts)
    if not expert_names or expert_names[0] != fallback_expert:
        raise ValueError("fallback expert must be first for deterministic utility ties")
    template = ContextualExpertGateV2(
        experts=experts,
        expert_names=expert_names,
        gate=ExpertGateNetwork(len(expert_names)),
        confidence_threshold=0.0,
        fallback_expert=fallback_expert,
        second_hop_expert=second_hop_expert,
    )
    prepared: list[tuple[str, int, list[DecisionSnapshot]]] = []
    for source_index, (domain, path) in enumerate(sources):
        metadata = source_metadata[str(Path(path).resolve())]
        snapshots = _source_first_hop_rows(
            path,
            source_domain=domain,
            held_out=held_out,
            maximum=maximum_per_source,
            seed=seed + 101 * source_index,
        )
        prepared.append((_domain_key(domain), int(metadata["probe_regime"]), snapshots))
    balanced_count = min(len(snapshots) for _, _, snapshots in prepared)
    if balanced_count < 2:
        raise ValueError("each domain-by-probe stratum requires at least two examples")

    device = torch.device("cpu")
    feature_rows: list[Tensor] = []
    utility_rows: list[list[float]] = []
    domains: list[str] = []
    probe_regimes: list[int] = []
    snapshot_ids: list[str] = []
    for domain, probe_regime, snapshots in prepared:
        for snapshot in snapshots[:balanced_count]:
            candidates = list(snapshot.candidates)
            base = context_features_v2(
                snapshot.query,
                snapshot.observation,
                candidates,
                frontier=True,
            )
            statistics = template._expert_statistics(
                snapshot.query,
                snapshot.observation,
                candidates,
                frontier=True,
                device=device,
            )
            feature_rows.append(torch.cat((base, statistics)))
            utilities = []
            for expert in experts.values():
                order = expert.rank(
                    snapshot.query,
                    snapshot.observation,
                    candidates,
                    frontier=True,
                    device=device,
                )
                utilities.append(
                    first_proof_reciprocal_rank_v2(order, snapshot.relevant_candidates)
                )
            utility_rows.append(utilities)
            domains.append(domain)
            probe_regimes.append(probe_regime)
            snapshot_ids.append(f"{domain}:p{probe_regime}:{snapshot.snapshot_id}")
    return GateExamplesV2(
        torch.stack(feature_rows),
        torch.tensor(utility_rows, dtype=torch.float32),
        tuple(domains),
        tuple(probe_regimes),
        tuple(snapshot_ids),
    )


def deterministic_source_split_v2(
    examples: GateExamplesV2,
    *,
    calibration_fraction: float = 0.25,
    seed: int = 2718,
) -> tuple[Tensor, Tensor]:
    if not 0.05 <= calibration_fraction <= 0.5:
        raise ValueError("calibration_fraction must be in [0.05, 0.5]")
    train: list[int] = []
    calibration: list[int] = []
    strata = sorted(set(zip(examples.domains, examples.probe_regimes)))
    for domain, probe_regime in strata:
        indices = [
            index
            for index, values in enumerate(zip(examples.domains, examples.probe_regimes))
            if values == (domain, probe_regime)
        ]
        label = f"{domain}:p{probe_regime}"
        stable = int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")
        random.Random(seed + stable).shuffle(indices)
        count = max(1, min(len(indices) - 1, round(len(indices) * calibration_fraction)))
        calibration.extend(indices[:count])
        train.extend(indices[count:])
    if not train or not calibration:
        raise ValueError("each domain-by-probe stratum requires at least two examples")
    return torch.tensor(sorted(train)), torch.tensor(sorted(calibration))


def train_gate_network_v2(
    examples: GateExamplesV2,
    train_indices: Tensor,
    *,
    hidden_dim: int = 32,
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 17,
) -> tuple[ExpertGateNetwork, list[float]]:
    if not 1 <= epochs <= 1000:
        raise ValueError("epochs must be in [1, 1000]")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    torch.manual_seed(seed)
    network = ExpertGateNetwork(examples.utilities.shape[1], hidden_dim).cpu()
    if examples.features.ndim != 2 or examples.features.shape[1] != network.input.in_features:
        raise ValueError(
            f"gate features must have width {network.input.in_features}, "
            f"got {tuple(examples.features.shape)}"
        )
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=1e-4)
    features = examples.features[train_indices]
    labels = torch.argmax(examples.utilities[train_indices], dim=1)
    generator = torch.Generator().manual_seed(seed)
    history: list[float] = []
    for _ in range(epochs):
        order = torch.randperm(len(train_indices), generator=generator)
        losses = []
        network.train()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(network(features[indices]), labels[indices])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(sum(losses) / len(losses))
    return network.eval(), history


@torch.inference_mode()
def gate_probabilities_v2(network: ExpertGateNetwork, features: Tensor) -> Tensor:
    return torch.softmax(network(features.cpu()), dim=1)


def calibrate_confidence_threshold_v2(
    probabilities: Tensor,
    utilities: Tensor,
    *,
    fallback_index: int,
    grid_size: int = 101,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 314159,
    confidence_level: float = 0.95,
) -> ThresholdCalibrationV2:
    """Choose a utility-optimal threshold whose paired gain has nonnegative LCB."""

    if probabilities.shape != utilities.shape or probabilities.ndim != 2:
        raise ValueError("probabilities and utilities must have matching [state, expert] shape")
    if not 0 <= fallback_index < probabilities.shape[1]:
        raise ValueError("invalid fallback_index")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0.5, 1.0)")
    confidence, predicted = torch.max(probabilities, dim=1)
    fallback_values = utilities[:, fallback_index]
    generator = torch.Generator().manual_seed(bootstrap_seed)
    bootstrap_indices = torch.randint(
        len(utilities),
        (bootstrap_samples, len(utilities)),
        generator=generator,
    )
    best: ThresholdCalibrationV2 | None = None
    for threshold in torch.linspace(0.0, 1.0, grid_size).tolist():
        use_fallback = confidence < threshold
        deployed = torch.where(
            use_fallback,
            torch.full_like(predicted, fallback_index),
            predicted,
        )
        values = utilities.gather(1, deployed[:, None]).squeeze(1)
        paired = values - fallback_values
        bootstrapped = paired[bootstrap_indices].mean(dim=1)
        lower_bound = float(torch.quantile(bootstrapped, 1.0 - confidence_level))
        counts = tuple(int((deployed == index).sum()) for index in range(utilities.shape[1]))
        result = ThresholdCalibrationV2(
            threshold=float(threshold),
            mean_utility=float(values.mean()),
            fallback_fraction=float(use_fallback.float().mean()),
            selected_counts=counts,
            fallback_mean_utility=float(fallback_values.mean()),
            paired_gain=float(paired.mean()),
            paired_gain_lcb=lower_bound,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence_level,
        )
        if lower_bound < -1e-12:
            continue
        if (
            best is None
            or result.mean_utility > best.mean_utility + 1e-12
            or (
                abs(result.mean_utility - best.mean_utility) <= 1e-12
                and result.threshold > best.threshold
            )
        ):
            best = result
    if best is None:
        raise RuntimeError("no confidence threshold passed paired fallback safety")
    return best
