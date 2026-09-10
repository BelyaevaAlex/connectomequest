"""Standard BabyAI task, RGB public boundary, no private grid in policy inputs."""

import hashlib
import json
import re

import gymnasium as gym
import numpy as np
from minigrid.core.actions import Actions
from minigrid.wrappers import RGBImgPartialObsWrapper

from connectomequest.embodied.full_task import MatchedController
from connectomequest.embodied.navigation import PlanStep
from connectomequest.embodied.observations import PlanObservation
from connectomequest.embodied.recheck_policy import supported_plan
from connectomequest.embodied.sequence_recovery import RevisitAgent

METHODS = ("current", "full_replan", "indexed")


class PublicUnlockPickup:
    def __init__(self, seed, budget=256):
        self._base = gym.make("BabyAI-UnlockPickup-v0", max_steps=budget).unwrapped
        self._env = RGBImgPartialObsWrapper(self._base, tile_size=8)
        raw, _ = self._env.reset(seed=seed)
        self._seed = seed
        self._origin = tuple(map(int, self._base.agent_pos))
        match = re.fullmatch(r"pick up the (red|green|blue|purple|yellow|grey) box", raw["mission"])
        if not match:
            raise ValueError(f"unsupported public mission: {raw['mission']}")
        self._goals = (match.group(1),)
        self._obs = self._public(raw, None, True, False, 0.0)

    def _public(self, raw, action, ack, terminal, reward):
        rgb = np.asarray(raw["image"]).copy()
        rgb.setflags(write=False)
        position = tuple(int(a) - b for a, b in zip(self._base.agent_pos, self._origin))
        carrying = None if self._base.carrying is None else self._base.carrying.type
        step = int(self._base.step_count)
        stage = int(reward > 0)
        payload = [
            self._seed,
            position,
            int(self._base.agent_dir),
            carrying,
            raw["mission"],
            step,
            stage,
            action,
            ack,
            terminal,
            float(reward),
            hashlib.sha256(rgb.tobytes()).hexdigest(),
        ]
        rid = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
        return PlanObservation(
            rgb,
            position,
            int(self._base.agent_dir),
            carrying,
            raw["mission"],
            self._goals,
            stage,
            action,
            bool(ack),
            step,
            int(self._base.max_steps),
            bool(terminal),
            float(reward),
            rid,
        )

    def observation(self):
        return self._obs

    def step(self, action):
        if self._obs.terminated:
            raise RuntimeError("episode terminated")
        if action not in ("left", "right", "forward", "pickup", "drop", "toggle"):
            raise ValueError(action)
        before = tuple(self._base.agent_pos)
        carried = self._base.carrying
        # Trusted acknowledgement only. Neither this cell nor its label is sent to policy.
        front = self._base.grid.get(*self._base.front_pos)
        encoded = None if front is None else front.encode()
        raw, reward, term, trunc, _ = self._env.step(Actions[action])
        ack = True
        if action == "forward":
            ack = tuple(self._base.agent_pos) != before
        elif action == "pickup":
            ack = carried is None and self._base.carrying is not None
        elif action == "drop":
            ack = carried is not None and self._base.carrying is None
        elif action == "toggle":
            after = self._base.grid.get(*self._base.front_pos)
            ack = encoded != (None if after is None else after.encode())
        self._obs = self._public(raw, action, ack, term or trunc, reward)
        return self._obs

    def close(self):
        self._env.close()


class PublicDistractorTask(PublicUnlockPickup):
    """Public-observation adapter for BabyAI UnlockPickup with distractors."""

    def __init__(self, seed: int, budget: int = 256):
        self._base = gym.make("BabyAI-UnlockPickupDist-v0", max_steps=budget).unwrapped
        self._env = RGBImgPartialObsWrapper(self._base, tile_size=8)
        raw, _ = self._env.reset(seed=seed)
        self._seed = seed
        self._origin = tuple(map(int, self._base.agent_pos))
        match = re.fullmatch(
            r"pick up the (red|green|blue|purple|yellow|grey) box",
            raw["mission"],
        )
        if not match:
            raise ValueError(f"unsupported public mission: {raw['mission']}")
        self._goals = (match.group(1),)
        self._obs = self._public(raw, None, True, False, 0.0)


class StandardController:
    def __init__(self, perceptor, method):
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        self.guard_rejections = 0
        self.requirements = ()
        if method == "indexed":
            self.agent = RevisitAgent(perceptor, "indexed")
            self.wrapped = None
        else:
            self.wrapped = MatchedController(perceptor, method, "clean")
            self.agent = self.wrapped.agent

    def act(self, obs):
        if self.wrapped:
            action = self.wrapped.act(obs)
            self.requirements = self.wrapped.last_requirements
            assert self.agent.map.verify(self.requirements, self.wrapped.last_certificate)
            self.guard_rejections = self.wrapped.guard_rejections
            return action
        action = self.agent.act(obs)
        pending = self.agent.plan[self.agent.plan_index - 1 :]
        step = (
            pending[0]
            if pending and pending[0].action == action
            else PlanStep(action, obs.position, obs.direction, ())
        )
        legal = supported_plan(self.agent.map, obs, [step])
        if legal is None:
            self.guard_rejections += 1
            legal = [PlanStep("right", obs.position, obs.direction, ())]
            self.agent.plan = legal
            self.agent.plan_index = 1
            self.agent.commitment = None
        self.requirements = legal[0].requirements
        assert all(
            self.agent.map.graph.supported(self.agent.map.fact(p, l)) for p, l in self.requirements
        )
        return legal[0].action
