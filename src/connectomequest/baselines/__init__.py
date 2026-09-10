"""Adapters for pinned learned graph-reasoning baselines."""

from typing import Any

__all__ = ["export_baseline_bundle"]


def __getattr__(name: str) -> Any:
    """Keep lightweight baseline utilities independent of export-time PyArrow."""

    if name == "export_baseline_bundle":
        from connectomequest.baselines.export import export_baseline_bundle

        return export_baseline_bundle
    raise AttributeError(name)
