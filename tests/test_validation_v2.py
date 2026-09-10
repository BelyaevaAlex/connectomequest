from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from connectomequest.manifest import sha256_file
from connectomequest.policies.expert_gate_v2 import ExpertGateNetwork
from connectomequest.policies.snapshot_v2 import SnapshotPolicyV2
from connectomequest.validation_v2 import (
    ObservableWeightPolicy,
    _existing_result,
    _validate_gate_payload,
    load_v2_policy,
)


def test_weight_policy_matches_observable_tie_break() -> None:
    from connectomequest.env import Observation
    from connectomequest.proof import Evidence, EvidenceKind
    from connectomequest.query import QuerySpec, QueryType

    observation = Observation(
        current_node=0,
        discovered_edges=(
            Evidence(EvidenceKind.EDGE, 0, "presynaptic_to", 2, "x", 3.0),
            Evidence(EvidenceKind.EDGE, 0, "presynaptic_to", 1, "x", 3.0),
        ),
        memory=(),
        remaining_budget=10.0,
        step=1,
    )
    query = QuerySpec("q", QueryType.TWO_HOP_TYPE, (0, 9), "has_type")
    assert ObservableWeightPolicy().rank(
        query, observation, [3, 2, 1], frontier=True, device=torch.device("cpu")
    ) == [1, 2, 3]


def test_gate_validation_fails_closed_on_non_source_protocol() -> None:
    payload = {
        "model_kind": "contextual_expert_gate_v2",
        "held_out": "manc",
        "seed": 17,
        "expert_names": ["weight", "snapshot_v2"],
        "expert_checkpoint_sha256": {
            "weight": {"observable_builtin": "weight"},
            "snapshot_v2": {"sha256": "a" * 64},
        },
        "protocol": {
            "source_only": False,
            "target_connectome_validation": False,
            "test_access": False,
        },
    }
    with pytest.raises(ValueError, match="source-only"):
        _validate_gate_payload(
            payload,
            held_out="manc",
            seed=17,
            expert_digests={"weight": "builtin:weight", "snapshot_v2": "a" * 64},
        )


def test_gate_validation_pins_direction_seed_and_expert_hashes() -> None:
    payload = {
        "model_kind": "contextual_expert_gate_v2",
        "held_out": "manc",
        "seed": 17,
        "expert_names": ["weight", "snapshot_v2"],
        "expert_checkpoint_sha256": {
            "weight": {"observable_builtin": "weight"},
            "snapshot_v2": {"checkpoint_sha256": "a" * 64},
        },
        "protocol": {
            "source_only": True,
            "target_connectome_validation": False,
            "test_access": False,
        },
    }
    _validate_gate_payload(
        payload,
        held_out="manc",
        seed=17,
        expert_digests={"weight": "builtin:weight", "snapshot_v2": "a" * 64},
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _validate_gate_payload(
            payload,
            held_out="manc",
            seed=17,
            expert_digests={"weight": "builtin:weight", "snapshot_v2": "b" * 64},
        )


def test_skip_guard_requires_matching_run_and_result_hash(tmp_path: Path) -> None:
    output = tmp_path / "cell.parquet"
    pq.write_table(pa.table({"goal_success": [True]}), output)
    summary = {
        "protocol": {"run_sha256": "run"},
        "result_sha256": sha256_file(output),
    }
    output.with_suffix(".json").write_text(json.dumps(summary))
    reused = _existing_result(output, "run")
    assert reused is not None and reused["reused"] is True
    with pytest.raises(RuntimeError, match="mismatched"):
        _existing_result(output, "different")


def test_loader_pins_gate_to_exact_snapshot_checkpoint(tmp_path: Path, monkeypatch) -> None:
    snapshot_path = tmp_path / "snapshot.pt"
    snapshot = SnapshotPolicyV2(hidden_dim=8, query_dim=4, dropout=0.0)
    torch.save(
        {
            "model_kind": "entity_id_free_snapshot_policy_v2",
            "model_config": {"hidden_dim": 8, "query_dim": 4, "dropout": 0.0},
            "model_state": snapshot.state_dict(),
            "seed": 17,
            "protocol": {
                "held_out": "manc",
                "target_connectome_validation": False,
                "immutable_test_access": False,
            },
        },
        snapshot_path,
    )
    snapshot_sha = sha256_file(snapshot_path)
    diverse_path = tmp_path / "diverse.pt"
    torch.save({"model_kind": "fake_diverse"}, diverse_path)
    diverse_sha = sha256_file(diverse_path)
    monkeypatch.setattr(
        "connectomequest.validation_v2.load_diverse_policy",
        lambda payload, device: ObservableWeightPolicy(),
    )
    network = ExpertGateNetwork(num_experts=4, hidden_dim=4)
    gate_path = tmp_path / "gate.pt"
    torch.save(
        {
            "model_kind": "contextual_expert_gate_v2",
            "gate_config": {"num_experts": 4, "hidden_dim": 4},
            "gate_state": network.state_dict(),
            "expert_names": ("weight", "random", "snapshot_v2", "diverse"),
            "fallback_expert": "weight",
            "second_hop_expert": "snapshot_v2",
            "confidence_threshold": 0.75,
            "held_out": "manc",
            "seed": 17,
            "protocol": {
                "source_only": True,
                "target_connectome_validation": False,
                "test_access": False,
            },
            "expert_checkpoint_sha256": {
                "weight": {"observable_builtin": "weight"},
                "random": {
                    "observable_builtin": "query_seeded_random",
                    "seed": 17,
                },
                "snapshot_v2": {"sha256": snapshot_sha},
                "diverse": {"sha256": diverse_sha},
            },
        },
        gate_path,
    )
    gate, provenance = load_v2_policy(
        snapshot_path,
        held_out="manc",
        seed=17,
        device=torch.device("cpu"),
        gate_checkpoint=gate_path,
        diverse_checkpoint=diverse_path,
    )
    assert gate.expert_names == ("weight", "random", "snapshot_v2", "diverse")
    assert provenance["snapshot_checkpoint_sha256"] == snapshot_sha
    assert provenance["gate_checkpoint_sha256"] == sha256_file(gate_path)
    assert provenance["diverse_checkpoint_sha256"] == diverse_sha
