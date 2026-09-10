"""Preflight validation for BetaE-style complex-query bundles."""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from pathlib import Path

TASK_STRUCTURES = {
    "1p": ("e", ("r",)),
    "2p": ("e", ("r", "r")),
    "2i": (("e", ("r",)), ("e", ("r",))),
    "ip": ((("e", ("r",)), ("e", ("r",))), ("r",)),
    "pi": (("e", ("r", "r")), ("e", ("r",))),
    "2in": (("e", ("r",)), ("e", ("r", "n"))),
}
PATH_TASKS = frozenset({"1p", "2p"})


def validate_cqa_partitions(
    bundle: Path,
    tasks: Iterable[str],
    *,
    split: str = "train",
) -> dict[str, object]:
    """Return query counts and reject missing requested structures before training."""

    selected = tuple(tasks)
    unknown = sorted(set(selected) - TASK_STRUCTURES.keys())
    if unknown:
        raise ValueError(f"unsupported CQA tasks: {', '.join(unknown)}")
    query_path = Path(bundle) / f"{split}-queries.pkl"
    with query_path.open("rb") as stream:
        queries = pickle.load(stream)  # noqa: S301 - bundle is generated locally
    counts = {task: len(queries.get(TASK_STRUCTURES[task], ())) for task in selected}
    missing = [task for task, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"{query_path} has empty requested partitions: {', '.join(missing)}")
    path_total = sum(count for task, count in counts.items() if task in PATH_TASKS)
    return {
        "bundle": str(Path(bundle).resolve()),
        "split": split,
        "counts": counts,
        "path_total": path_total,
        "other_total": sum(counts.values()) - path_total,
    }
