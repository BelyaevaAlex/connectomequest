"""Method-independent long-horizon UnlockPickupDist layout generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import numpy as np
from minigrid.core.actions import Actions
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.unlock import ObjDesc, PickupInstr, UnlockPickup
from minigrid.utils.baby_ai_bot import BabyAIBot
from minigrid.wrappers import RGBImgPartialObsWrapper

from connectomequest.embodied.observations import PlanObservation

ALLOWED_GENERATOR_EXCLUSIONS: Final = frozenset(
    {
        "unsolved_common_planner",
        "length_outside_stratum",
        "instrumentation_failure",
    }
)
_STRATUM_SIZES: Final = {
    "short": (4, 5, 6),
    "medium": (7, 8, 9),
    "long": (22, 23, 24),
    "very_long": (34, 35, 36),
}
_PUBLIC_ACTIONS: Final = ("left", "right", "forward", "pickup", "drop", "toggle")


@dataclass(frozen=True)
class LayoutSpec:
    layout_seed: int
    environment_seed: int
    room_size: int
    distractor_count: int
    horizon: int = 8192


@dataclass(frozen=True)
class GeneratedLayout:
    spec: LayoutSpec
    requested_stratum: str
    reference_solution_length: int
    enrollment_action_trace: tuple[str, ...]
    exclusion_reason: str | None
    generator_attempt: int


class _LongUnlockPickup(UnlockPickup):
    """UnlockPickupDist semantics with only size and distractor count varied."""

    def __init__(
        self,
        room_size: int,
        distractor_count: int,
        *,
        max_steps: int,
        render_mode: str | None = None,
    ) -> None:
        self.distractors = distractor_count > 0
        self._distractor_count = int(distractor_count)
        RoomGridLevel.__init__(
            self,
            num_rows=1,
            num_cols=2,
            room_size=int(room_size),
            max_steps=int(max_steps),
            render_mode=render_mode,
        )

    def gen_mission(self) -> None:
        target, _ = self.add_object(1, 0, kind="box")
        door, _ = self.add_door(0, 0, 0, locked=True)
        self.add_object(0, 0, "key", door.color)
        if self._distractor_count:
            self.add_distractors(num_distractors=self._distractor_count)
        self.place_agent(0, 0)
        self.instrs = PickupInstr(ObjDesc(target.type, target.color))


class PublicLongUnlockPickup:
    """Public RGB boundary; latent grid is retained only for audit functions."""

    def __init__(self, spec: LayoutSpec) -> None:
        self.spec = spec
        self._base = _LongUnlockPickup(
            spec.room_size,
            spec.distractor_count,
            max_steps=spec.horizon,
            render_mode="rgb_array",
        )
        self._env = RGBImgPartialObsWrapper(self._base, tile_size=8)
        raw, _ = self._env.reset(seed=spec.environment_seed)
        self._origin = tuple(map(int, self._base.agent_pos))
        match = re.fullmatch(
            r"pick up the (red|green|blue|purple|yellow|grey) box",
            raw["mission"],
        )
        if not match:
            raise ValueError(f"unsupported public mission: {raw['mission']}")
        self._goals = (match.group(1),)
        self._obs = self._public(raw, None, True, False, 0.0)

    def _public(self, raw, action, acknowledged, terminal, reward) -> PlanObservation:
        rgb = np.asarray(raw["image"]).copy()
        rgb.setflags(write=False)
        position = tuple(
            int(value) - origin for value, origin in zip(self._base.agent_pos, self._origin)
        )
        carrying = None if self._base.carrying is None else self._base.carrying.type
        step = int(self._base.step_count)
        stage = int(float(reward) > 0)
        payload = [
            self.spec.environment_seed,
            position,
            int(self._base.agent_dir),
            carrying,
            raw["mission"],
            step,
            stage,
            action,
            bool(acknowledged),
            bool(terminal),
            float(reward),
            hashlib.sha256(rgb.tobytes()).hexdigest(),
        ]
        receipt_id = hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        return PlanObservation(
            rgb,
            position,
            int(self._base.agent_dir),
            carrying,
            raw["mission"],
            self._goals,
            stage,
            action,
            bool(acknowledged),
            step,
            int(self._base.max_steps),
            bool(terminal),
            float(reward),
            receipt_id,
        )

    def observation(self) -> PlanObservation:
        return self._obs

    def step(self, action: str) -> PlanObservation:
        if self._obs.terminated:
            raise RuntimeError("episode terminated")
        if action not in _PUBLIC_ACTIONS:
            raise ValueError(f"unsupported action: {action}")
        before_position = tuple(self._base.agent_pos)
        before_carrying = self._base.carrying
        front = self._base.grid.get(*self._base.front_pos)
        before_front = None if front is None else tuple(front.encode())
        raw, reward, terminated, truncated, _ = self._env.step(Actions[action])
        acknowledged = True
        if action == "forward":
            acknowledged = tuple(self._base.agent_pos) != before_position
        elif action == "pickup":
            acknowledged = before_carrying is None and self._base.carrying is not None
        elif action == "drop":
            acknowledged = before_carrying is not None and self._base.carrying is None
        elif action == "toggle":
            after = self._base.grid.get(*self._base.front_pos)
            after_front = None if after is None else tuple(after.encode())
            acknowledged = before_front != after_front
        self._obs = self._public(
            raw,
            action,
            acknowledged,
            terminated or truncated,
            reward,
        )
        return self._obs

    def close(self) -> None:
        self._env.close()


def classify_remaining_length(length: int) -> str | None:
    if length < 8:
        return None
    if length <= 15:
        return "short"
    if length <= 31:
        return "medium"
    if length <= 63:
        return "long"
    return "very_long"


def _distractor_count(room_size: int, attempt: int) -> int:
    if room_size <= 4:
        return 0
    capacity_guard = max(1, 2 * (room_size - 3) - 2)
    return min(capacity_guard, 2 + attempt % min(7, room_size - 2))


def _environment_seed(layout_seed: int, stratum: str, attempt: int) -> int:
    payload = f"v39:{layout_seed}:{stratum}:{attempt}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFF_FFFF


def _reference_solution(spec: LayoutSpec) -> tuple[tuple[str, ...], str | None]:
    env = _LongUnlockPickup(
        spec.room_size,
        spec.distractor_count,
        max_steps=spec.horizon,
    )
    try:
        env.reset(seed=spec.environment_seed)
        bot = BabyAIBot(env)
        trace: list[str] = []
        previous = None
        for _ in range(spec.horizon):
            action = bot.replan(previous)
            action_name = Actions(action).name
            if action_name == "done":
                break
            _, reward, terminated, truncated, _ = env.step(action)
            trace.append(action_name)
            previous = action
            if terminated or truncated:
                return (
                    tuple(trace),
                    None if terminated and float(reward) > 0 else "unsolved_common_planner",
                )
        return tuple(trace), "unsolved_common_planner"
    except Exception:
        return (), "instrumentation_failure"
    finally:
        env.close()


@lru_cache(maxsize=4096)
def generate_layout(layout_seed: int, stratum: str) -> GeneratedLayout:
    if stratum not in _STRATUM_SIZES:
        raise ValueError(f"unknown plan-length stratum: {stratum}")
    saw_solution = False
    saw_instrumentation_failure = False
    last_spec = LayoutSpec(layout_seed, layout_seed, _STRATUM_SIZES[stratum][0], 0)
    last_trace: tuple[str, ...] = ()
    attempt = 0
    for room_size in _STRATUM_SIZES[stratum]:
        for local_attempt in range(16):
            environment_seed = _environment_seed(layout_seed, stratum, attempt)
            spec = LayoutSpec(
                layout_seed=int(layout_seed),
                environment_seed=environment_seed,
                room_size=room_size,
                distractor_count=_distractor_count(room_size, local_attempt),
            )
            trace, error = _reference_solution(spec)
            last_spec, last_trace = spec, trace
            if error == "instrumentation_failure":
                saw_instrumentation_failure = True
            elif error is None:
                saw_solution = True
                return GeneratedLayout(
                    spec,
                    stratum,
                    len(trace),
                    trace,
                    None,
                    attempt,
                )
            attempt += 1
    if not saw_solution:
        reason = (
            "instrumentation_failure" if saw_instrumentation_failure else "unsolved_common_planner"
        )
    else:
        reason = "length_outside_stratum"
    return GeneratedLayout(
        last_spec,
        stratum,
        len(last_trace),
        last_trace,
        reason,
        max(0, attempt - 1),
    )


def reset_layout(spec: LayoutSpec) -> PublicLongUnlockPickup:
    return PublicLongUnlockPickup(spec)


def audit_environment_hash(env: PublicLongUnlockPickup) -> str:
    """Generator/replay audit only; this digest is never included in MethodInput."""

    carrying = None
    if env._base.carrying is not None:
        carrying = tuple(env._base.carrying.encode())
    payload = {
        "grid": env._base.grid.encode().tolist(),
        "agent_position": list(map(int, env._base.agent_pos)),
        "agent_direction": int(env._base.agent_dir),
        "carrying": carrying,
        "step": int(env._base.step_count),
        "mission": env.observation().mission,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-seeds", type=int, default=0)
    args = parser.parse_args()
    rows = []
    for offset in range(args.probe_seeds):
        for stratum in _STRATUM_SIZES:
            layout = generate_layout(39_100_000 + offset, stratum)
            rows.append(
                {
                    "layout_seed": layout.spec.layout_seed,
                    "stratum": stratum,
                    "reference_solution_length": layout.reference_solution_length,
                    "exclusion_reason": layout.exclusion_reason,
                    "attempt": layout.generator_attempt,
                }
            )
    print(json.dumps(rows, sort_keys=True, indent=2))


if __name__ == "__main__":
    _main()
