"""Small configuration helpers with no global mutable state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def dataset_config(name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "configs" / "data" / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"unknown dataset {name!r}; expected {path}")
    return load_yaml(path)
