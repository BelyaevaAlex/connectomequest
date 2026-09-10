"""Learned RGB tile perception for the MiniGrid semantic-map adapter."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from minigrid.core.constants import IDX_TO_OBJECT, STATE_TO_IDX
from minigrid.envs import DoorKeyEnv
from minigrid.wrappers import RGBImgPartialObsWrapper
from torch import Tensor, nn

CLASS_NAMES = (
    "unseen",
    "empty",
    "wall",
    "door_open",
    "door_closed",
    "door_locked",
    "key",
    "goal",
    "other",
)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
IDX_TO_STATE = {index: name for name, index in STATE_TO_IDX.items()}


def semantic_label(object_index: int, state_index: int) -> str:
    name = IDX_TO_OBJECT[int(object_index)]
    if name == "door":
        return f"door_{IDX_TO_STATE[int(state_index)]}"
    return name if name in CLASS_TO_ID else "other"


def split_rgb_tiles(image: np.ndarray, *, view_size: int = 7) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != image.shape[1]:
        raise ValueError("expected a square RGB partial observation")
    tile_size = image.shape[0] // view_size
    if tile_size * view_size != image.shape[0]:
        raise ValueError("RGB observation is not divisible into view tiles")
    return np.stack(
        [
            image[y * tile_size : (y + 1) * tile_size, x * tile_size : (x + 1) * tile_size]
            for x in range(view_size)
            for y in range(view_size)
        ]
    )


def encoded_tile_labels(encoded: np.ndarray) -> np.ndarray:
    width, height, _ = encoded.shape
    return np.asarray(
        [
            CLASS_TO_ID[semantic_label(encoded[x, y, 0], encoded[x, y, 2])]
            for x in range(width)
            for y in range(height)
        ],
        dtype=np.int64,
    )


class TilePerceptionNet(nn.Module):
    def __init__(self, classes: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, classes)

    def forward(self, tiles: Tensor) -> Tensor:
        return self.classifier(self.encoder(tiles).flatten(1))


@dataclass(frozen=True, slots=True)
class PerceptionConfig:
    tile_size: int = 8
    view_size: int = 7
    classes: tuple[str, ...] = CLASS_NAMES


def collect_doorkey_tiles(
    seeds: Iterable[int],
    *,
    size: int = 8,
    tile_size: int = 8,
    max_per_class: int = 4_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Render supervised partial views; simulator state is used only for labels."""

    by_class: dict[int, list[np.ndarray]] = {index: [] for index in range(len(CLASS_NAMES))}
    for seed in seeds:
        base = DoorKeyEnv(size=size, render_mode="rgb_array", max_steps=4 * size * size)
        env = RGBImgPartialObsWrapper(base, tile_size=tile_size)
        env.reset(seed=int(seed))
        positions = [
            (x, y)
            for x in range(base.width)
            for y in range(base.height)
            if (cell := base.grid.get(x, y)) is None or cell.can_overlap()
        ]
        doors = [
            base.grid.get(x, y)
            for x in range(base.width)
            for y in range(base.height)
            if getattr(base.grid.get(x, y), "type", None) == "door"
        ]
        for door_state in ("locked", "closed", "open"):
            for door in doors:
                door.is_locked = door_state == "locked"
                door.is_open = door_state == "open"
            for position in positions:
                base.agent_pos = np.asarray(position)
                for direction in range(4):
                    base.agent_dir = direction
                    encoded = base.gen_obs()["image"]
                    rgb = base.get_frame(tile_size=tile_size, agent_pov=True)
                    tiles = split_rgb_tiles(rgb)
                    labels = encoded_tile_labels(encoded)
                    for tile, label in zip(tiles, labels, strict=True):
                        if len(by_class[int(label)]) < max_per_class:
                            by_class[int(label)].append(tile.copy())
        env.close()
        if all(
            len(by_class[index]) >= max_per_class
            for index in range(len(CLASS_NAMES))
            if CLASS_NAMES[index] not in {"door_open", "door_closed", "other"}
        ):
            break
    nonempty = [(index, values) for index, values in by_class.items() if values]
    if not nonempty:
        raise RuntimeError("no perception tiles were collected")
    images = np.concatenate([np.stack(values) for _, values in nonempty], axis=0).astype(np.uint8)
    labels = np.concatenate(
        [np.full(len(values), index, dtype=np.int64) for index, values in nonempty]
    )
    return images, labels


def _augment(batch: Tensor, generator: torch.Generator) -> Tensor:
    scale = torch.empty((len(batch), 1, 1, 1), device=batch.device).uniform_(
        0.85, 1.15, generator=generator
    )
    noise = torch.randn(batch.shape, device=batch.device, generator=generator) * 0.025
    return (batch * scale + noise).clamp(0.0, 1.0)


def train_perception(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    epochs: int = 12,
    batch_size: int = 512,
) -> tuple[TilePerceptionNet, dict[str, float]]:
    torch.manual_seed(seed)
    model = TilePerceptionNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)
    targets = torch.from_numpy(labels)
    split_mask = torch.arange(len(targets)) % 5 != 0
    train_indices = torch.nonzero(split_mask, as_tuple=False).flatten()
    validation_indices = torch.nonzero(~split_mask, as_tuple=False).flatten()
    generator = torch.Generator(device=device).manual_seed(seed)
    for epoch in range(epochs):
        epoch_generator = torch.Generator().manual_seed(seed + epoch)
        permutation = torch.randperm(len(train_indices), generator=epoch_generator)
        order = train_indices[permutation]
        model.train()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch = tensor[indices].to(device)
            target = targets[indices].to(device)
            logits = model(_augment(batch, generator))
            loss = nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        logits = model(tensor[validation_indices].to(device))
        predictions = logits.argmax(dim=-1).cpu()
    accuracy = float((predictions == targets[validation_indices]).float().mean())
    macro = []
    for class_id in sorted(set(targets[validation_indices].tolist())):
        mask = targets[validation_indices] == class_id
        macro.append(float((predictions[mask] == class_id).float().mean()))
    return model, {
        "validation_accuracy": accuracy,
        "validation_macro_accuracy": float(np.mean(macro)),
        "train_tiles": int(len(train_indices)),
        "validation_tiles": int(len(validation_indices)),
    }


def save_perception(
    path: Path,
    model: TilePerceptionNet,
    *,
    seed: int,
    metrics: dict[str, float],
) -> str:
    payload = {
        "model_kind": "minigrid_tile_perception_v1",
        "config": asdict(PerceptionConfig()),
        "seed": seed,
        "metrics": metrics,
        "model_state": model.cpu().state_dict(),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    digest = hashlib.sha256(buffer.getvalue()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)
    return digest


def load_perception(path: Path, device: torch.device) -> tuple[TilePerceptionNet, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("model_kind") != "minigrid_tile_perception_v1":
        raise ValueError("not a MiniGrid perception checkpoint")
    if tuple(payload["config"]["classes"]) != CLASS_NAMES:
        raise ValueError("perception class vocabulary mismatch")
    model = TilePerceptionNet().to(device)
    model.load_state_dict(payload["model_state"])
    return model.eval(), payload


@torch.inference_mode()
def predict_tiles(
    model: TilePerceptionNet,
    image: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tiles = split_rgb_tiles(image)
    tensor = torch.from_numpy(tiles).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    probabilities = model(tensor).softmax(dim=-1)
    confidence, labels = probabilities.max(dim=-1)
    size = int(np.sqrt(len(labels)))
    return (
        labels.cpu().numpy().reshape(size, size),
        confidence.cpu().numpy().reshape(size, size),
    )
