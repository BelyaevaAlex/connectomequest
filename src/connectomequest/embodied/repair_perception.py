"""RGB-only inference; simulator supervision is confined to dataset collection."""

import numpy as np
import torch
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT

from connectomequest.embodied.tile_perception import TilePerceptionNet, split_rgb_tiles

CLASSES = (
    "unseen",
    "empty",
    "wall",
    "door_open",
    "door_closed",
    "door_locked",
    "goal",
    "floor",
    "lava",
) + tuple(f"{kind}:{color}" for kind in ("key", "box", "ball") for color in IDX_TO_COLOR.values())


def label(encoded):
    kind = IDX_TO_OBJECT[int(encoded[0])]
    if kind == "door":
        return ("door_open", "door_closed", "door_locked")[int(encoded[2])]
    if kind in {"key", "box", "ball"}:
        return f"{kind}:{IDX_TO_COLOR[int(encoded[1])]}"
    return kind if kind in CLASSES else "empty"


def collect(seeds, cap=256):
    import gymnasium as gym
    from minigrid.wrappers import RGBImgPartialObsWrapper

    pools = {c: [] for c in CLASSES}
    for seed in seeds:
        base = gym.make("BabyAI-UnlockPickup-v0").unwrapped
        env = RGBImgPartialObsWrapper(base, tile_size=8)
        env.reset(seed=int(seed))
        places = [
            (x, y)
            for x in range(base.width)
            for y in range(base.height)
            if (obj := base.grid.get(x, y)) is None or obj.can_overlap()
        ]
        # Seeded pose subsampling bounds CPU/render cost; not evaluation tasks.
        rng = np.random.default_rng(seed)
        rng.shuffle(places)
        doors = [obj for obj in base.grid.grid if getattr(obj, "type", None) == "door"]
        for state in range(3):
            for door in doors:
                door.is_open = state == 0
                door.is_locked = state == 2
            for pos in places[:16]:
                base.agent_pos = np.array(pos)
                for direction in range(4):
                    base.agent_dir = direction
                    encoded = base.gen_obs()["image"]
                    rgb = base.get_frame(tile_size=8, agent_pov=True)
                    for tile, cell in zip(split_rgb_tiles(rgb), encoded.reshape(-1, 3)):
                        name = label(cell)
                        if len(pools[name]) < cap:
                            pools[name].append(tile.copy())
        env.close()
    images = []
    targets = []
    for i, c in enumerate(CLASSES):
        images.extend(pools[c])
        targets.extend([i] * len(pools[c]))
    return np.stack(images), np.array(targets, dtype=np.int64)


class Perceptor:
    def __init__(self, path):
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(blob["classes"]) != CLASSES:
            raise ValueError("class schema mismatch")
        self.model = TilePerceptionNet(len(CLASSES))
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()

    def predict(self, rgb):
        x = torch.from_numpy(split_rgb_tiles(rgb)).permute(0, 3, 1, 2).float() / 255
        with torch.inference_mode():
            prob, index = self.model(x).softmax(-1).max(-1)
        return [
            (i // 7, i % 7, CLASSES[int(k)], float(p)) for i, (k, p) in enumerate(zip(index, prob))
        ]
