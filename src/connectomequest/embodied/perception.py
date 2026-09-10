"""Checkpoint-dispatched RGB perceptor retaining spatial tile features."""

import torch
from torch import nn

from connectomequest.embodied.repair_perception import CLASSES
from connectomequest.embodied.tile_perception import TilePerceptionNet, split_rgb_tiles


class SpatialTileNet(TilePerceptionNet):
    def __init__(self):
        super().__init__(len(CLASSES))
        self.encoder[-1] = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Linear(64 * 16, len(CLASSES))


class Perceptor:
    def __init__(self, path):
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(blob["classes"]) != CLASSES:
            raise ValueError("class schema")
        self.model = (
            SpatialTileNet()
            if blob.get("architecture") == "spatial4"
            else TilePerceptionNet(len(CLASSES))
        )
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()

    def predict(self, rgb):
        x = torch.from_numpy(split_rgb_tiles(rgb)).permute(0, 3, 1, 2).float() / 255
        with torch.inference_mode():
            p, k = self.model(x).softmax(-1).max(-1)
        return [
            (i // 7, i % 7, CLASSES[int(label)], float(prob))
            for i, (label, prob) in enumerate(zip(k, p))
        ]
