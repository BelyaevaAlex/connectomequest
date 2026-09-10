from __future__ import annotations

import dataclasses
import importlib
from types import MappingProxyType


def _api():
    return importlib.import_module("connectomequest.embodied.types")


def _input(api):
    event = api.EvidenceEvent(
        action_index=4,
        withdrawn_receipt_ids=("rgb-1",),
        superseded_receipt_ids=(),
        changed_semantic_keys=("cell:2,3",),
        added_assertion_bindings=(("cell:2,3=door_open:red", ("rgb-2",)),),
        condition="semantic_correction",
    )
    return api.MethodInput(
        pose=(1, 2),
        direction=3,
        inventory="key:red",
        mission="pick up the green ball",
        belief_snapshot=(("cell:2,3", "door_open:red"),),
        active_receipt_ids=("rgb-2",),
        assertion_to_receipt_bindings=(("cell:2,3=door_open:red", ("rgb-2",)),),
        remaining_plan=(("forward", (1, 2), 3),),
        cursor=0,
        update=event,
        public_action_set=("left", "right", "forward", "pickup"),
    )


def test_method_input_contains_no_oracle_fields():
    api = _api()
    names = {field.name for field in dataclasses.fields(api.MethodInput)}
    assert names == {
        "pose",
        "direction",
        "inventory",
        "mission",
        "belief_snapshot",
        "active_receipt_ids",
        "assertion_to_receipt_bindings",
        "remaining_plan",
        "cursor",
        "update",
        "public_action_set",
    }
    assert not names & {
        "oracle_grid",
        "true_labels",
        "oracle_plan",
        "oracle_affected_steps",
        "generator_audit_label",
    }


def test_counter_components_are_not_collapsed():
    api = _api()
    row = api.EpisodeCounters()
    assert dataclasses.asdict(row) == {
        "environment_actions": 0,
        "predicate_evaluations": 0,
        "successor_expansions": 0,
        "plan_position_visits": 0,
        "index_lookups": 0,
        "posting_entries": 0,
        "planner_calls": 0,
        "revalidation_ns": 0,
        "planning_ns": 0,
    }


def test_protocol_records_prospective_axes_and_is_frozen():
    api = _api()
    protocol = api.EpisodeProtocol(
        protocol_version="v39",
        stage="smoke",
        horizon=512,
        methods=("full_replan", "full_scan", "receipt_index", "unchecked_reuse"),
        public_action_set=("left", "right", "forward", "toggle", "pickup", "drop"),
        plan_length_strata=("short", "medium", "long", "very_long"),
        update_conditions=("none", "irrelevant_withdrawal"),
        update_counts=(0, 1, 2, 4, 8),
        relevant_fractions=(0.0, 0.1, 0.5),
        layout_seeds=(39000001,),
        update_seeds=(17, 29, 43),
        bootstrap_seed=39039,
        bootstrap_draws=10_000,
        common_planner_id="counted-ucs-v1",
        action_model_id="unlockpickupdist-v1",
        perception_sha256="0" * 64,
    )
    assert protocol.bootstrap_draws == 10_000
    try:
        protocol.horizon = 1
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("EpisodeProtocol must be frozen")


def test_observation_and_evidence_records_are_frozen():
    api = _api()
    observation = api.ObservationPacket(
        action_index=0,
        receipt_id="rgb-0",
        rgb_sha256="1" * 64,
        pose=(1, 1),
        direction=0,
        inventory=None,
        mission="pick up the green ball",
        action_outcome=(("terminated", False), ("reward", 0.0)),
    )
    event = api.EvidenceEvent(
        action_index=0,
        withdrawn_receipt_ids=(),
        superseded_receipt_ids=(),
        changed_semantic_keys=(),
        added_assertion_bindings=(),
        condition="none",
    )
    for record in (observation, event):
        try:
            record.action_index = 2
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError(f"{type(record).__name__} must be frozen")


def test_access_digest_is_canonical_and_accepts_read_only_mapping():
    api = _api()
    first = _input(api)
    second = dataclasses.replace(
        first,
        belief_snapshot=MappingProxyType({"cell:2,3": "door_open:red"}),
    )
    assert api.digest_public_access(first) == api.digest_public_access(second)
    assert len(api.digest_public_access(first)) == 64


def test_access_digest_changes_for_every_public_field():
    api = _api()
    row = _input(api)
    alternatives = {
        "pose": (2, 2),
        "direction": 0,
        "inventory": None,
        "mission": "different mission",
        "belief_snapshot": (("cell:2,3", "wall"),),
        "active_receipt_ids": ("rgb-3",),
        "assertion_to_receipt_bindings": (("cell:2,3=wall", ("rgb-3",)),),
        "remaining_plan": (("left", (1, 2), 3),),
        "cursor": 1,
        "update": dataclasses.replace(row.update, action_index=5),
        "public_action_set": ("left", "right"),
    }
    original = api.digest_public_access(row)
    assert set(alternatives) == {field.name for field in dataclasses.fields(row)}
    for name, value in alternatives.items():
        changed = dataclasses.replace(row, **{name: value})
        assert api.digest_public_access(changed) != original, name


def test_episode_result_separates_safety_work_and_timing():
    api = _api()
    names = {field.name for field in dataclasses.fields(api.EpisodeResult)}
    required = {
        "task_success",
        "milestones",
        "unsupported_authorizations",
        "illegal_environment_actions",
        "counters",
        "index_build_ns",
        "index_peak_bytes",
        "total_reasoning_ns",
        "total_wall_clock_ns",
        "trace_replay_ok",
        "access_digest",
        "action_trace",
        "observation_hashes",
        "update_digests",
    }
    assert required <= names
