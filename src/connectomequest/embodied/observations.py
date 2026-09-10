"""Trusted MiniGrid two-room sequential-pickup task and public sensor boundary."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from minigrid.core.actions import Actions
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box, Door, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgPartialObsWrapper

COLORS = ("red", "green", "purple", "yellow")
CONDITIONS = ("clean", "rgb_replay", "door_change")
CONFIGS = {"short": (6, 2), "long": (6, 4), "wide": (8, 4)}


class TwoRoomSequence(MiniGridEnv):
    def __init__(self, room_size=6, goals=2, budget=256):
        self.room_size = room_size
        self.goal_colors = COLORS[:goals]
        self.stage = 0
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: "pick up the red box"),
            width=2 * room_size - 1,
            height=room_size,
            max_steps=budget,
            see_through_walls=False,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        divider = self.room_size - 1
        for y in range(1, height - 1):
            self.grid.set(divider, y, Wall())
        self.grid.set(divider, self._rand_int(1, height - 1), Door("blue", is_locked=True))
        self.place_obj(Key("blue"), top=(1, 1), size=(divider - 1, height - 2))
        for i, color in enumerate(self.goal_colors):
            # First goal right; later goals require crossings back and forth.
            top = (divider + 1, 1) if i % 2 == 0 else (1, 1)
            self.place_obj(Box(color), top=top, size=(divider - 1, height - 2))
        self.place_agent(top=(1, 1), size=(divider - 1, height - 2))
        self.stage = 0
        self.mission = f"pick up the {self.goal_colors[0]} box"

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        reward = 0.0
        if action == Actions.pickup and self.carrying is not None:
            if self.carrying.type == "box" and self.carrying.color == self.goal_colors[self.stage]:
                self.stage += 1
                if self.stage == len(self.goal_colors):
                    terminated = True
                    reward = 1.0
                else:
                    self.mission = f"pick up the {self.goal_colors[self.stage]} box"
                    obs["mission"] = self.mission
        return obs, reward, terminated, truncated, info


@dataclass(frozen=True)
class PlanObservation:
    rgb: np.ndarray
    position: tuple
    direction: int
    carrying: str | None
    mission: str
    goals: tuple
    stage: int
    last_action: str | None
    acknowledged: bool
    step: int
    budget: int
    terminated: bool
    reward: float
    receipt_id: str


class PublicSequence:
    def __init__(self, seed, config="short", condition="clean", budget=256):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        size, goals = CONFIGS[config]
        self._base = TwoRoomSequence(size, goals, budget)
        self._env = RGBImgPartialObsWrapper(self._base, tile_size=8)
        obs, _ = self._env.reset(seed=seed)
        self._origin = tuple(map(int, self._base.agent_pos))
        self._seed = seed
        self._condition = condition
        self._rgb = None
        self.exposure = {"replayed_frames": 0, "door_closures": 0}
        self._observation = self._public(obs, None, True, False, 0.0)

    def _public(self, obs, action, ack, terminal, reward):
        rgb = np.asarray(obs["image"]).copy()
        t = self._base.step_count
        if (
            self._condition == "rgb_replay"
            and (8 <= t <= 11 or 40 <= t <= 43)
            and self._rgb is not None
        ):
            rgb = self._rgb.copy()
            self.exposure["replayed_frames"] += 1
        else:
            self._rgb = rgb.copy()
        rgb.setflags(write=False)
        pos = tuple(int(a) - b for a, b in zip(self._base.agent_pos, self._origin))
        carried = None if self._base.carrying is None else self._base.carrying.type
        payload = [
            int(self._seed),
            pos,
            int(self._base.agent_dir),
            carried,
            self._base.mission,
            self._base.goal_colors,
            self._base.stage,
            action,
            bool(ack),
            t,
            self._base.max_steps,
            bool(terminal),
            float(reward),
            hashlib.sha256(rgb.tobytes()).hexdigest(),
        ]
        rid = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
        return PlanObservation(
            rgb,
            pos,
            int(self._base.agent_dir),
            carried,
            self._base.mission,
            self._base.goal_colors,
            self._base.stage,
            action,
            bool(ack),
            t,
            self._base.max_steps,
            bool(terminal),
            float(reward),
            rid,
        )

    def observation(self):
        return self._observation

    def step(self, action):
        if self._observation.terminated:
            raise RuntimeError("episode ended")
        a = Actions[action]
        if action not in ("left", "right", "forward", "pickup", "drop", "toggle"):
            raise ValueError(action)
        before = tuple(self._base.agent_pos)
        carried = self._base.carrying
        front = self._base.grid.get(*self._base.front_pos)
        encoded = None if front is None else front.encode()
        obs, reward, term, trunc, _ = self._env.step(a)
        ack = True
        if action == "forward":
            ack = tuple(self._base.agent_pos) != before
        elif action == "pickup":
            ack = carried is None and self._base.carrying is not None
        elif action == "drop":
            ack = carried is not None and self._base.carrying is None
        elif action == "toggle":
            front = self._base.grid.get(*self._base.front_pos)
            ack = encoded != (None if front is None else front.encode())
        if (
            self._condition == "door_change"
            and self._base.step_count in (20, 60)
            and not (term or trunc)
        ):
            for cell in self._base.grid.grid:
                if getattr(cell, "type", None) == "door" and cell.is_open:
                    cell.is_open = False
                    self.exposure["door_closures"] += 1
            obs = self._env.observation(self._base.gen_obs())
        self._observation = self._public(obs, action, ack, term or trunc, reward)
        return self._observation

    def close(self):
        self._env.close()
