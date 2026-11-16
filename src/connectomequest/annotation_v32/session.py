"""Sequential query execution with one paid annotation cache per policy."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict
from operator import index
from types import MappingProxyType

import numpy as np

from .core import Query, evaluate, replay_certificate


def _cache_digest(observed):
    payload = json.dumps(sorted(observed.items()), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run_session(
    queries: Iterable[Query],
    policy,
    oracle: Callable[[int], int | None],
    prior,
    *,
    budget: int,
    seed: int = 0,
) -> dict:
    """Expose one query at a time; only this runner can call the oracle.

    The budget is a session-wide count of unique attempted annotations. Missing
    annotations cost one attempt and cannot be purchased repeatedly. Existing
    provisional annotations are predictor inputs, never free proof facts.
    """
    if isinstance(budget, bool) or index(budget) < 0:
        raise ValueError("budget must be a nonnegative integer")
    observed: dict[int, int | None] = {}
    events = []
    outcomes = []
    rng = np.random.default_rng(seed)
    total_policy_cpu = 0.0
    seen_query_ids = set()
    for query in queries:
        if query.query_id in seen_query_ids:
            raise ValueError("query_id must be unique within a session")
        seen_query_ids.add(query.query_id)
        if query.metadata.get("computationally_unsupported"):
            raise ValueError("unsupported incomplete query enumeration")
        start_cost = len(observed)
        start_digest = _cache_digest(observed)
        begin_cpu = time.process_time()
        policy.begin_query(query, MappingProxyType(dict(observed)))
        total_policy_cpu += time.process_time() - begin_cpu
        query_cells = frozenset(query.cells)
        while True:
            verification = evaluate(query, observed)
            if verification.status != "unresolved":
                reason = "certificate"
                break
            if not (query_cells - observed.keys()):
                reason = "no_readable_cells"
                break
            if len(observed) >= budget:
                reason = "budget_exhausted"
                break
            visible = MappingProxyType(dict(observed))
            before_digest = _cache_digest(observed)
            decision_start = time.process_time()
            cell = policy.choose_next(query, visible, prior, rng)
            decision_cpu = time.process_time() - decision_start
            total_policy_cpu += decision_cpu
            if cell is None:
                reason = "policy_stopped"
                break
            try:
                cell = index(cell)
            except TypeError as exc:
                raise ValueError("policy must select a legal integer cell") from exc
            if cell not in query_cells or cell in observed:
                raise ValueError("policy selected a cell outside the legal unread query cells")
            label = oracle(cell)
            if label is not None:
                try:
                    label = index(label)
                except TypeError as exc:
                    raise ValueError("oracle label must be categorical integer or None") from exc
                if label < 0:
                    raise ValueError("oracle label must be nonnegative")
            observed[cell] = label
            events.append({
                "query_id": query.query_id, "cell": cell, "label": label,
                "cache_before_sha256": before_digest,
                "decision_cpu_seconds": decision_cpu,
            })
        valid = verification.status != "unresolved" and replay_certificate(query, verification)
        if verification.status != "unresolved" and not valid:
            raise RuntimeError("verifier returned a certificate that does not replay")
        outcomes.append({
            "query_id": query.query_id, "status": verification.status,
            "stop_reason": reason, "reads": len(observed) - start_cost,
            "cache_start_sha256": start_digest, "cache_end_sha256": _cache_digest(observed),
            "certificate": asdict(verification), "certificate_valid": valid,
            "policy_diagnostics": copy.deepcopy(getattr(policy, "diagnostics", {})),
        })
    return {
        "schema": "cq-annotation-session-v32", "seed": int(seed), "budget": int(budget),
        "cost": len(observed), "resolved_queries": sum(q["certificate_valid"] for q in outcomes),
        "query_count": len(outcomes), "queries": outcomes, "events": events,
        "observed": {str(v): t for v, t in sorted(observed.items())},
        "policy_cpu_seconds": total_policy_cpu,
    }
