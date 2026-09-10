"""Deterministic sensor and pose stressors for the embodied evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Position = tuple[int, int]
DIR_VECTORS: tuple[Position, ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))


def view_to_world(
    position: Position,
    direction: int,
    view_x: int,
    view_y: int,
    *,
    view_size: int = 7,
) -> Position:
    """Invert MiniGrid's egocentric view transform using only reported pose."""

    dx, dy = DIR_VECTORS[direction]
    rx, ry = -dy, dx
    half = view_size // 2
    forward = view_size - 1 - view_y
    lateral = view_x - half
    return (
        int(position[0] + dx * forward + rx * lateral),
        int(position[1] + dy * forward + ry * lateral),
    )


def observation_coordinate_is_addressable(
    position: Position, *, pose_mode: str, width: int, height: int
) -> bool:
    """Clip only absolute coordinates; relative odometry has no hidden origin."""

    if pose_mode == "odometry":
        return True
    if pose_mode != "global":
        raise ValueError(f"unknown pose mode: {pose_mode}")
    return 0 <= position[0] < width and 0 <= position[1] < height


@dataclass(frozen=True, slots=True)
class PaletteShift:
    """A fixed mild color-space shift not used by perception training."""

    matrix: tuple[tuple[float, float, float], ...] = (
        (0.90, 0.08, 0.02),
        (0.04, 0.91, 0.05),
        (0.07, 0.03, 0.90),
    )
    bias: tuple[float, float, float] = (3.0, -2.0, 2.0)

    def __call__(self, image: np.ndarray, *, seed: int, step: int) -> np.ndarray:
        del seed, step
        transformed = (
            np.asarray(image, dtype=np.float32) @ np.asarray(self.matrix, dtype=np.float32).T
        )
        transformed += np.asarray(self.bias, dtype=np.float32)
        return np.clip(np.rint(transformed), 0, 255).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class TileOcclusion:
    rate: float = 0.10
    tile_size: int = 8

    def __call__(self, image: np.ndarray, *, seed: int, step: int) -> np.ndarray:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError("occlusion rate must lie in [0, 1]")
        result = image.copy()
        width = result.shape[0] // self.tile_size
        height = result.shape[1] // self.tile_size
        rng = np.random.default_rng(seed * 1_000_003 + step * 97 + 31)
        mask = rng.random((width, height)) < self.rate
        for x, y in zip(*np.nonzero(mask), strict=True):
            result[
                y * self.tile_size : (y + 1) * self.tile_size,
                x * self.tile_size : (x + 1) * self.tile_size,
            ] = 0
        return result


@dataclass(slots=True)
class ActionOdometry:
    position: Position
    direction: int
    corruption_rate: float = 0.0
    corruptions: int = 0

    def update(self, action: str, *, moved: bool, seed: int, step: int) -> None:
        if not 0.0 <= self.corruption_rate <= 1.0:
            raise ValueError("odometry corruption rate must lie in [0, 1]")
        rng = np.random.default_rng(seed * 1_000_003 + step * 193 + 47)
        corrupted = bool(rng.random() < self.corruption_rate)
        self.corruptions += int(corrupted)
        if action in {"left", "right"}:
            turn = -1 if action == "left" else 1
            if corrupted:
                turn *= -1
            self.direction = (self.direction + turn) % 4
            return
        if action != "forward" or not moved:
            return
        move_direction = self.direction
        if corrupted:
            move_direction = (move_direction + (1 if step % 2 else -1)) % 4
        dx, dy = DIR_VECTORS[move_direction]
        self.position = self.position[0] + dx, self.position[1] + dy
