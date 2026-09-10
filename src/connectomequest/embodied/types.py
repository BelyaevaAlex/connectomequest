"""Immutable records for the prospective V39 full-episode experiment.

This module is the serialization boundary between the method-independent runner
and each evaluated revalidation strategy. MethodInput contains no generator-only
or oracle information.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

FrozenPairs = tuple[tuple[str, Any], ...]


def _pairs(value: Mapping[str, Any] | tuple[tuple[str, Any], ...]) -> FrozenPairs:
    items = value.items() if isinstance(value, Mapping) else value
    return tuple(sorted(((str(key), item) for key, item in items), key=lambda row: row[0]))


def _bindings(
    value: Mapping[str, tuple[str, ...]] | tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    return tuple(
        sorted(
            (
                (str(key), tuple(sorted(str(receipt) for receipt in receipts)))
                for key, receipts in items
            ),
            key=lambda row: row[0],
        )
    )


@dataclass(frozen=True)
class EpisodeProtocol:
    protocol_version: str
    stage: str
    horizon: int
    methods: tuple[str, ...]
    public_action_set: tuple[str, ...]
    plan_length_strata: tuple[str, ...]
    update_conditions: tuple[str, ...]
    update_counts: tuple[int, ...]
    relevant_fractions: tuple[float, ...]
    layout_seeds: tuple[int, ...]
    update_seeds: tuple[int, ...]
    bootstrap_seed: int
    bootstrap_draws: int
    common_planner_id: str
    action_model_id: str
    perception_sha256: str

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.bootstrap_draws <= 0:
            raise ValueError("bootstrap_draws must be positive")
        if len(self.perception_sha256) != 64:
            raise ValueError("perception_sha256 must be a SHA-256 hexadecimal digest")
        int(self.perception_sha256, 16)
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("method names must be unique")


@dataclass(frozen=True)
class ObservationPacket:
    action_index: int
    receipt_id: str
    rgb_sha256: str
    pose: tuple[int, int]
    direction: int
    inventory: str | None
    mission: str
    action_outcome: FrozenPairs

    def __post_init__(self) -> None:
        if self.action_index < 0:
            raise ValueError("action_index must be non-negative")
        object.__setattr__(self, "pose", tuple(int(x) for x in self.pose))
        object.__setattr__(self, "action_outcome", _pairs(self.action_outcome))


@dataclass(frozen=True)
class EvidenceEvent:
    action_index: int
    withdrawn_receipt_ids: tuple[str, ...]
    superseded_receipt_ids: tuple[str, ...]
    changed_semantic_keys: tuple[str, ...]
    added_assertion_bindings: tuple[tuple[str, tuple[str, ...]], ...]
    condition: str

    def __post_init__(self) -> None:
        if self.action_index < 0:
            raise ValueError("action_index must be non-negative")
        object.__setattr__(
            self, "withdrawn_receipt_ids", tuple(sorted(set(self.withdrawn_receipt_ids)))
        )
        object.__setattr__(
            self, "superseded_receipt_ids", tuple(sorted(set(self.superseded_receipt_ids)))
        )
        object.__setattr__(
            self, "changed_semantic_keys", tuple(sorted(set(self.changed_semantic_keys)))
        )
        object.__setattr__(
            self, "added_assertion_bindings", _bindings(self.added_assertion_bindings)
        )


@dataclass(frozen=True)
class MethodInput:
    pose: tuple[int, int]
    direction: int
    inventory: str | None
    mission: str
    belief_snapshot: FrozenPairs
    active_receipt_ids: tuple[str, ...]
    assertion_to_receipt_bindings: tuple[tuple[str, tuple[str, ...]], ...]
    remaining_plan: tuple[Any, ...]
    cursor: int
    update: EvidenceEvent
    public_action_set: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.cursor < 0:
            raise ValueError("cursor must be non-negative")
        object.__setattr__(self, "pose", tuple(int(x) for x in self.pose))
        object.__setattr__(self, "belief_snapshot", _pairs(self.belief_snapshot))
        object.__setattr__(self, "active_receipt_ids", tuple(sorted(set(self.active_receipt_ids))))
        object.__setattr__(
            self,
            "assertion_to_receipt_bindings",
            _bindings(self.assertion_to_receipt_bindings),
        )
        object.__setattr__(self, "remaining_plan", tuple(self.remaining_plan))
        object.__setattr__(self, "public_action_set", tuple(self.public_action_set))


@dataclass(frozen=True)
class EpisodeCounters:
    environment_actions: int = 0
    predicate_evaluations: int = 0
    successor_expansions: int = 0
    plan_position_visits: int = 0
    index_lookups: int = 0
    posting_entries: int = 0
    planner_calls: int = 0
    revalidation_ns: int = 0
    planning_ns: int = 0

    def __add__(self, other: EpisodeCounters) -> EpisodeCounters:
        return EpisodeCounters(
            **{
                field.name: getattr(self, field.name) + getattr(other, field.name)
                for field in fields(self)
            }
        )


@dataclass(frozen=True)
class EpisodeResult:
    protocol_version: str
    stage: str
    method: str
    layout_seed: int
    environment_seed: int
    room_size: int
    distractor_count: int
    horizon: int
    update_seed: int
    condition: str
    update_count: int
    relevant_fraction: float
    plan_length_stratum: str
    pre_update_remaining_plan_length: int
    task_success: bool
    terminated: bool
    truncated: bool
    exposed_update_count: int
    milestones: tuple[tuple[str, bool], ...]
    unsupported_authorizations: int
    illegal_environment_actions: int
    counters: EpisodeCounters
    index_build_ns: int
    index_peak_bytes: int
    total_reasoning_ns: int
    total_wall_clock_ns: int
    trace_replay_ok: bool
    access_digest: str
    pre_update_action_digest: str
    action_trace: tuple[str, ...]
    observation_hashes: tuple[str, ...]
    update_digests: tuple[str, ...]
    plan_digests: tuple[str, ...]
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "milestones", _pairs(self.milestones))
        if self.total_reasoning_ns != (
            self.counters.revalidation_ns + self.counters.planning_ns + self.index_build_ns
        ):
            raise ValueError(
                "total_reasoning_ns must include revalidation, planning, and index build"
            )
        nonnegative = (
            self.update_count,
            self.horizon,
            self.pre_update_remaining_plan_length,
            self.exposed_update_count,
            self.unsupported_authorizations,
            self.illegal_environment_actions,
            self.index_build_ns,
            self.index_peak_bytes,
            self.total_wall_clock_ns,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("episode counts and timings must be non-negative")


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
        )
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite floats are not canonical JSON")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest_public_access(method_input: MethodInput) -> str:
    if not isinstance(method_input, MethodInput):
        raise TypeError("digest_public_access accepts only MethodInput")
    return hashlib.sha256(canonical_json_bytes(method_input)).hexdigest()
