"""Matched linear-softmax control with portable NumPy inference.

The fixed schedule matches the frozen MLP. This control measures the value of
its hidden nonlinearity under that schedule; it cannot prove neural necessity.
"""

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StructuralLinear:
    mean: np.ndarray
    scale: np.ndarray
    w: np.ndarray
    b: np.ndarray

    def __post_init__(self):
        for name in ("mean", "scale", "w", "b"):
            array = np.array(getattr(self, name), dtype=np.float32, copy=True)
            if not np.isfinite(array).all():
                raise ValueError("finite model arrays required")
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        if (
            self.mean.shape != (12,)
            or self.scale.shape != (12,)
            or self.b.ndim != 1
            or len(self.b) < 1
            or self.w.shape != (12, len(self.b))
            or np.any(self.scale <= 0)
        ):
            raise ValueError("invalid 12-feature linear model")

    @property
    def label_count(self):
        return len(self.b)

    def logits(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 12 or not np.isfinite(x).all():
            raise ValueError("finite N by 12 features required")
        return ((x - self.mean) / self.scale) @ self.w + self.b

    def save(self, path):
        with open(path, "xb") as stream:
            np.savez(stream, **{k: getattr(self, k) for k in ("mean", "scale", "w", "b")})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"mean", "scale", "w", "b"}:
                raise ValueError("unexpected linear model arrays")
            return cls(**{k: arrays[k] for k in arrays.files})


def train_linear(
    x_train, y_train, x_calibration, y_calibration, *, label_count, device="cpu", wall_seconds=300
):
    """One fixed candidate: seed 0, 40 Adam(.01) epochs, batch size 256.

    Keep the earliest minimum untempered calibration-NLL checkpoint. Synthetic
    arrays may use a smaller vocabulary; the production CLI pins all 5551 codes.
    """
    x = np.asarray(x_train, dtype=np.float32)
    xc = np.asarray(x_calibration, dtype=np.float32)
    raw_y, raw_yc = np.asarray(y_train), np.asarray(y_calibration)
    if (
        x.ndim != 2
        or xc.ndim != 2
        or x.shape[1] != 12
        or xc.shape[1] != 12
        or raw_y.shape != (len(x),)
        or raw_yc.shape != (len(xc),)
        or not len(x)
        or not len(xc)
        or len(x) > 50000
        or len(xc) > 50000
        or not isinstance(label_count, int)
        or not 1 <= label_count <= 20000
        or not np.isfinite(x).all()
        or not np.isfinite(xc).all()
        or not np.isfinite(raw_y).all()
        or not np.isfinite(raw_yc).all()
        or np.any(raw_y != np.floor(raw_y))
        or np.any(raw_yc != np.floor(raw_yc))
        or np.any(raw_y < 0)
        or np.any(raw_y >= label_count)
        or np.any(raw_yc < 0)
        or np.any(raw_yc >= label_count)
        or device not in ("cpu", "cuda")
        or not np.isfinite(wall_seconds)
        or not 0 < wall_seconds <= 300
    ):
        raise ValueError("invalid matched linear training inputs")
    if len(xc) * label_count > 150000000:
        raise MemoryError("calibration logit cap exceeded")
    import torch

    y, yc = raw_y.astype(np.int64), raw_yc.astype(np.int64)
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    torch.manual_seed(0)
    net = torch.nn.Linear(12, label_count).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    tx = torch.as_tensor((x - mean) / scale, device=device)
    ty = torch.as_tensor(y, device=device)
    tc = torch.as_tensor((xc - mean) / scale, device=device)
    cy = torch.as_tensor(yc, device=device)
    rng = np.random.default_rng(0)
    losses, calibration = [], []
    best_loss, best_state, best_epoch, updates = float("inf"), None, 0, 0
    started = time.monotonic()
    for epoch in range(40):
        total = 0.0
        for indices in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x) / 256)))):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(tx[indices]), ty[indices])
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(indices)
            updates += 1
        losses.append(total / len(x))
        with torch.no_grad():
            current = float(torch.nn.functional.cross_entropy(net(tc), cy).cpu())
        if not np.isfinite(current) or not np.isfinite(losses[-1]):
            raise ValueError("nonfinite training or calibration loss")
        calibration.append(current)
        if current < best_loss:
            best_loss, best_epoch = current, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if time.monotonic() - started > wall_seconds:
            raise TimeoutError("linear training exceeded fixed wall cap")
    net.load_state_dict(best_state)
    model = StructuralLinear(
        mean, scale, net.weight.detach().cpu().numpy().T, net.bias.detach().cpu().numpy()
    )
    return model, dict(
        architecture="linear_softmax",
        train_rows=len(x),
        calibration_rows=len(xc),
        seed=0,
        epochs=40,
        batch_size=256,
        device=device,
        learning_rate=0.01,
        optimizer_updates=updates,
        training_loss=losses,
        calibration_nll_trajectory=calibration,
        best_epoch=best_epoch,
        training_wall_seconds=time.monotonic() - started,
    )
