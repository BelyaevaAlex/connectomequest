from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch

from connectomequest.decision_benchmark import (
    DecisionSnapshot,
    DecisionStage,
    write_decision_snapshots,
)
from connectomequest.env import Observation
from connectomequest.expert_gate_training_v2 import (
    GateExamplesV2,
    calibrate_confidence_threshold_v2,
    deterministic_source_split_v2,
    first_proof_reciprocal_rank_v2,
    source_probe_matrix_v2,
    train_gate_network_v2,
    verified_source_validation_sha_v2,
)
from connectomequest.policies.expert_gate_v2 import (
    ExpertGateNetwork,
    ObservableRandomPolicy,
    load_contextual_expert_gate_v2,
)
from connectomequest.query import QuerySpec, QueryType


@dataclass
class FixedPolicy:
    order: tuple[int, ...]

    def rank(self, query, observation, candidates, **kwargs):
        del query, observation, kwargs
        return [candidate for candidate in self.order if candidate in candidates]


def _snapshot(split: str = "validation", held_out: str = "h01") -> DecisionSnapshot:
    return DecisionSnapshot(
        snapshot_id=f"{held_out}-{split}",
        held_out=held_out,
        split=split,
        stage=DecisionStage.FIRST_HOP,
        query=QuerySpec("q", QueryType.TWO_HOP_TYPE, (0, 9), "has_type"),
        observation=Observation(0, (), (), 63.0, 1),
        candidates=(1, 2),
        relevant_candidates=frozenset({2}),
        probes_used=0,
        graph_manifest_sha256="graph",
        query_sha256="query",
    )


def test_first_proof_utility_uses_best_relevant_rank() -> None:
    assert first_proof_reciprocal_rank_v2([3, 2, 1], frozenset({1, 2})) == 0.5
    assert first_proof_reciprocal_rank_v2([3, 2, 1], frozenset({4})) == 0.0


def test_threshold_calibration_uses_exact_weight_fallback() -> None:
    probabilities = torch.tensor([[0.4, 0.6], [0.1, 0.9]])
    utilities = torch.tensor([[1.0, 0.5], [0.2, 1.0]])
    result = calibrate_confidence_threshold_v2(
        probabilities,
        utilities,
        fallback_index=0,
    )
    assert result.threshold == pytest.approx(0.9)
    assert result.mean_utility == 1.0
    assert result.fallback_fraction == 0.5
    assert result.selected_counts == (1, 1)


def test_source_split_is_deterministic_and_stratified() -> None:
    examples = GateExamplesV2(
        features=torch.zeros(8, 4),
        utilities=torch.zeros(8, 2),
        domains=("h01",) * 4 + ("manc",) * 4,
        probe_regimes=(0, 0, 1, 1, 0, 0, 1, 1),
        snapshot_ids=tuple(f"s{index}" for index in range(8)),
    )
    first = deterministic_source_split_v2(examples, calibration_fraction=0.25, seed=7)
    second = deterministic_source_split_v2(examples, calibration_fraction=0.25, seed=7)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    train, calibration = first
    assert len(train) == 4
    assert {
        (examples.domains[int(index)], examples.probe_regimes[int(index)]) for index in calibration
    } == {("h01", 0), ("h01", 1), ("manc", 0), ("manc", 1)}


def test_gate_training_is_bounded_cpu_classifier() -> None:
    examples = GateExamplesV2(
        features=torch.zeros(4, ExpertGateNetwork(2, hidden_dim=4).input.in_features),
        utilities=torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        domains=("a", "a", "b", "b"),
        probe_regimes=(0, 0, 0, 0),
        snapshot_ids=("0", "1", "2", "3"),
    )
    model, history = train_gate_network_v2(
        examples,
        torch.arange(4),
        hidden_dim=4,
        epochs=3,
        batch_size=2,
        seed=3,
    )
    assert next(model.parameters()).device.type == "cpu"
    assert len(history) == 3
    with pytest.raises(ValueError, match="epochs"):
        train_gate_network_v2(examples, torch.arange(4), epochs=1001)


def test_source_validation_guard_rejects_target_and_test(tmp_path) -> None:
    validation = tmp_path / "h01-validation.parquet"
    write_decision_snapshots([_snapshot()], validation)
    verified_source_validation_sha_v2(
        validation,
        source_domain="h01",
        held_out="manc",
    )
    with pytest.raises(ValueError, match="held-out"):
        verified_source_validation_sha_v2(
            validation,
            source_domain="h01",
            held_out="h01",
        )
    with pytest.raises(ValueError, match="held-out"):
        verified_source_validation_sha_v2(
            validation,
            source_domain="h01-20210729-c3",
            held_out="h01",
        )

    test_path = tmp_path / "h01-test.parquet"
    write_decision_snapshots([replace(_snapshot(), split="test")], test_path)
    with pytest.raises(ValueError, match="validation snapshots only"):
        verified_source_validation_sha_v2(
            test_path,
            source_domain="h01",
            held_out="manc",
        )


def test_gate_loader_validates_schema_and_preserves_fixed_second_hop() -> None:
    network = ExpertGateNetwork(2, hidden_dim=4)
    payload = {
        "model_kind": "contextual_expert_gate_v2",
        "gate_config": {"num_experts": 2, "hidden_dim": 4},
        "gate_state": network.state_dict(),
        "expert_names": ("weight", "snapshot_v2"),
        "confidence_threshold": 0.75,
        "fallback_expert": "weight",
        "second_hop_expert": "snapshot_v2",
    }
    experts = {
        "weight": FixedPolicy((1, 2)),
        "snapshot_v2": FixedPolicy((2, 1)),
    }
    gate = load_contextual_expert_gate_v2(payload, experts)
    assert gate.second_hop_expert == "snapshot_v2"
    broken = dict(payload)
    broken["gate_config"] = {"num_experts": 3, "hidden_dim": 4}
    with pytest.raises(ValueError, match="num_experts"):
        load_contextual_expert_gate_v2(broken, experts)


def test_observable_random_is_query_seeded_and_complete() -> None:
    policy = ObservableRandomPolicy(seed=29)
    query = QuerySpec("q", QueryType.TWO_HOP_TYPE, (0, 9), "has_type")
    observation = Observation(0, (), (), 63.0, 1)
    first = policy.rank(
        query,
        observation,
        [1, 2, 3, 4],
        frontier=True,
        device=torch.device("cpu"),
    )
    second = policy.rank(
        query,
        observation,
        [1, 2, 3, 4],
        frontier=True,
        device=torch.device("cpu"),
    )
    assert first == second
    assert set(first) == {1, 2, 3, 4}


def test_source_probe_matrix_requires_all_eight_unique_files(tmp_path) -> None:
    sources = []
    for domain in ("h01-20210729-c3", "manc-v1.0"):
        for probe in (0, 1, 2, 4):
            path = tmp_path / f"{domain}-p{probe}.parquet"
            write_decision_snapshots(
                [replace(_snapshot(held_out=domain), snapshot_id=f"{domain}-p{probe}")],
                path,
                protocol={"subgoal_probes": probe},
            )
            sources.append((domain, path))
    metadata = source_probe_matrix_v2(sources, held_out="hemibrain-v1.2.1")
    assert len(metadata) == 8
    assert {row["probe_regime"] for row in metadata.values()} == {0, 1, 2, 4}
    with pytest.raises(ValueError, match="unique"):
        source_probe_matrix_v2(sources[:-1] + [sources[0]], held_out="hemibrain")
