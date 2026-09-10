"""Outcome-conditioned cost summaries for the held-out graph evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def _value(row: dict[str, Any], primary: str, fallback: str) -> Any:
    if primary in row:
        return row[primary]
    if fallback in row:
        return row[fallback]
    raise ValueError(f"row is missing {primary!r}/{fallback!r}")


def summarize_policy_costs(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, float | int | None]]:
    """Aggregate actions without conditioning the overall denominator on success.

    The function accepts both the paper-neutral names used by tests
    (``policy``, ``success``, ``actions``, ``budget``) and the immutable raw
    schema (``method``, ``goal_success``, ``budget_used``).  Exhaustion means
    that the charged action budget was reached, irrespective of task outcome.
    """

    groups: dict[tuple[str, str], list[tuple[bool, float, float]]] = defaultdict(list)
    seen: set[tuple[str, str, str, int]] = set()
    for row in rows:
        graph = str(row["graph"])
        policy = str(_value(row, "policy", "method"))
        success = bool(_value(row, "success", "goal_success"))
        actions = float(_value(row, "actions", "budget_used"))
        budget = float(row.get("budget", 64.0))
        exhaustion_at = float(row.get("exhaustion_at", budget))
        if actions < 0 or budget <= 0 or actions > budget:
            raise ValueError("invalid charged-action count")
        if "query_id" in row and "seed" in row:
            identity = (graph, policy, str(row["query_id"]), int(row["seed"]))
            if identity in seen:
                raise ValueError(f"duplicate episode-policy row: {identity}")
            seen.add(identity)
        groups[(graph, policy)].append((success, actions, exhaustion_at))

    result: dict[tuple[str, str], dict[str, float | int | None]] = {}
    for key, values in groups.items():
        successful = [actions for success, actions, _ in values if success]
        failed = [actions for success, actions, _ in values if not success]
        result[key] = {
            "n": len(values),
            "actions": sum(v[1] for v in values) / len(values),
            "actions_success": (sum(successful) / len(successful) if successful else None),
            "actions_failure": sum(failed) / len(failed) if failed else None,
            "budget_exhaustion": (
                sum(actions >= budget for _, actions, budget in values) / len(values)
            ),
        }
    return result


def format_optional(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"
