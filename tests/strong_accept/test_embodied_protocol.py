from __future__ import annotations

import json

import pytest


def test_stage_seed_blocks_are_disjoint_and_scientific_fields_fixed():
    from connectomequest.embodied.protocol import protocol_payload

    payloads = {stage: protocol_payload(stage) for stage in ("smoke", "pilot", "confirmation")}
    blocks = [
        {seed for seeds in payload["layout_seeds_by_stratum"].values() for seed in seeds}
        for payload in payloads.values()
    ]
    assert not (blocks[0] & blocks[1] or blocks[0] & blocks[2] or blocks[1] & blocks[2])
    for payload in payloads.values():
        assert payload["methods"] == [
            "full_replan",
            "full_scan",
            "receipt_index",
            "unchecked_reuse",
        ]
        assert payload["bootstrap_draws"] == 10_000
        assert payload["gates"]["time_reduction"] == 0.20
        assert payload["gates"]["work_reduction"] == 0.25


def test_freeze_is_write_once_and_detects_payload_drift(tmp_path):
    from connectomequest.embodied.protocol import freeze_payload

    path = tmp_path / "protocol.json"
    freeze_payload(path, {"stage": "smoke", "x": 1})
    freeze_payload(path, {"x": 1, "stage": "smoke"})
    with pytest.raises(ValueError, match="mismatch"):
        freeze_payload(path, {"stage": "smoke", "x": 2})


def test_atomic_case_write_reuses_match_and_refuses_duplicate_drift(tmp_path):
    from connectomequest.embodied.protocol import atomic_case_write

    path = tmp_path / "case.json"
    atomic_case_write(path, {"a": 1})
    atomic_case_write(path, {"a": 1})
    with pytest.raises(ValueError, match="duplicate"):
        atomic_case_write(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


def test_source_hash_verification_detects_drift(tmp_path):
    from connectomequest.embodied.protocol import source_hashes, verify_source_hashes

    source = tmp_path / "module.py"
    source.write_text("x = 1\n")
    expected = source_hashes(tmp_path, ("module.py",))
    verify_source_hashes(tmp_path, expected)
    source.write_text("x = 2\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        verify_source_hashes(tmp_path, expected)


def test_confirmation_requires_passing_pilot_gate(tmp_path):
    from connectomequest.embodied.protocol import assert_confirmation_can_open

    with pytest.raises(ValueError, match="pilot verification"):
        assert_confirmation_can_open(tmp_path)
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    (pilot / "verification.json").write_text(
        json.dumps({"integrity_passed": True, "primary_attainable_layouts": 299})
    )
    with pytest.raises(ValueError, match="300"):
        assert_confirmation_can_open(tmp_path)
    (pilot / "verification.json").write_text(
        json.dumps({"integrity_passed": True, "primary_attainable_layouts": 300})
    )
    assert_confirmation_can_open(tmp_path)


def test_stage_runner_uses_scientific_stratum_order_not_json_key_order():
    from scripts.run_embodied_evaluation import _episode_protocol

    payload = {
        "protocol_version": "test",
        "stage": "smoke",
        "horizon": 8192,
        "methods": ["full_scan"],
        "layout_seeds_by_stratum": {
            "long": [3],
            "medium": [2],
            "short": [1],
            "very_long": [4],
        },
        "schedule_cells": [{"condition": "none", "update_count": 0, "relevant_fraction": 0.0}],
        "update_seeds": [17],
        "bootstrap_seed": 39,
        "bootstrap_draws": 100,
        "common_planner": "full-mission-astar-v2",
        "action_model": "unlockpickupdist-v1",
        "checkpoint_sha256": "0" * 64,
    }
    protocol = _episode_protocol(payload)
    assert protocol.plan_length_strata == ("short", "medium", "long", "very_long")
    assert protocol.layout_seeds == (1, 2, 3, 4)


def test_current_attempt_uses_a_fresh_write_once_namespace():
    from connectomequest.embodied.protocol import OUTPUT_ROOT, protocol_payload

    assert OUTPUT_ROOT.name == "full-episode-v39-attempt-08"
    for stage in ("smoke", "pilot", "confirmation"):
        assert protocol_payload(stage)["protocol_version"] == "v39-attempt-08"


def test_layout_jobs_keep_all_methods_of_a_layout_in_one_worker():
    from scripts.run_embodied_evaluation import _layout_jobs

    payload = {
        "layout_seeds_by_stratum": {
            "short": [10, 11],
            "medium": [20],
            "long": [30],
            "very_long": [40],
        }
    }

    jobs = _layout_jobs(payload)

    assert jobs == (
        ("short", 10),
        ("short", 11),
        ("medium", 20),
        ("long", 30),
        ("very_long", 40),
    )


def test_method_order_is_cyclically_balanced_without_splitting_methods():
    from scripts.run_embodied_evaluation import _method_order

    methods = ("a", "b", "c", "d")
    orders = [_method_order(methods, 10, cell) for cell in range(4)]
    assert orders == [
        ("c", "d", "a", "b"),
        ("d", "a", "b", "c"),
        ("a", "b", "c", "d"),
        ("b", "c", "d", "a"),
    ]
