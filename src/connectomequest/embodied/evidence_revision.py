"""Versioned frame interpretations: identical latest-active semantics, two implementations.

Receipts stay immutable. Withdrawing a frame interpretation makes its predicted
cell bindings inactive; it does not erase acquisition history or assert truth.
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter_ns


@dataclass(frozen=True)
class FrameWithdrawal:
    interpretations: tuple[str, ...]
    issued_step: int


class RevisionMap:
    def __init__(self, method):
        if method not in ("local", "reset", "history_scan", "indexed"):
            raise ValueError(method)
        self.method = method
        self.cells = {}
        self.sources = {}
        self.receipts = {}
        self.frames = {}
        self.inactive = set()
        self.by_cell = defaultdict(list) if method == "indexed" else None
        self.serial = 0
        self.invalidations = 0
        self.stats = {
            "revision_events": 0,
            "revision_ns": 0,
            "maintenance_ns": 0,
            "revision_records_examined": 0,
            "map_resets": 0,
            "revision_changed_cells": 0,
            "revision_retained_cells": 0,
        }

    def label(self, position):
        return self.cells.get(position)

    def _assign(self, position, value, source):
        if value is None:
            self.cells.pop(position, None)
        else:
            self.cells[position] = value
        self.sources[position] = source

    def record(self, identity, digest, items):
        items = dict(items)
        binding = hashlib.sha256(repr(sorted(items.items())).encode()).hexdigest()
        payload = (digest, binding)
        if identity in self.receipts:
            if self.receipts[identity] != payload:
                raise ValueError("immutable interpretation identity changed")
            return
        self.receipts[identity] = payload
        self.serial += 1
        if self.method in ("history_scan", "indexed"):
            self.frames[identity] = (self.serial, items)
        for position, value in items.items():
            if self.by_cell is not None:
                self.by_cell[position].append((self.serial, identity, value))
            self._assign(position, value, identity)

    def observe(self, receipt, digest, items):
        started = perf_counter_ns()
        items = list(items)
        # Agent appends exact current-position occupancy after RGB predictions.
        # Keep that assertion separate: withdrawing RGB projection is not withdrawing odometry.
        before = {p: self.cells.get(p) for p, _ in items}
        self.record(receipt, digest, items[:-1])
        self.record("pose:" + receipt, digest, items[-1:])
        self.stats["maintenance_ns"] += perf_counter_ns() - started
        return {p for p, value in before.items() if value != self.cells.get(p)}

    def invalidate(self, positions):
        # Public failed-action evidence makes the cell unknown until a newer view.
        self.invalidations += 1
        self.record(
            f"action-failure:{self.invalidations}",
            "public-action-failure",
            [(p, None) for p in positions],
        )

    def withdraw(self, event, current_step):
        if event.issued_step > current_step or event.issued_step < 0:
            raise ValueError("future/invalid revision step")
        ids = set(event.interpretations)
        if not ids <= self.receipts.keys() or any(
            i.startswith(("pose:", "action-failure:")) for i in ids
        ):
            raise ValueError("unknown or non-projection interpretation")
        ids -= self.inactive
        if not ids:
            return
        start = perf_counter_ns()
        before_count = len(self.cells)
        self.inactive.update(ids)
        self.stats["revision_events"] += 1
        if self.method == "history_scan":
            before = dict(self.cells)
            self.cells = {}
            self.sources = {}
            seen = set()
            # Strong scan control: scan newest first and resolve each cell once.
            for identity, (_, items) in reversed(list(self.frames.items())):
                if identity in self.inactive:
                    continue
                for position, value in items.items():
                    self.stats["revision_records_examined"] += 1
                    if position not in seen:
                        self._assign(position, value, identity)
                        seen.add(position)
        elif self.method == "indexed":
            affected = {p for identity in ids for p in self.frames[identity][1]}
            before = {p: self.cells.get(p) for p in affected}
            for position in affected:
                self.cells.pop(position, None)
                self.sources.pop(position, None)
                for _, identity, value in reversed(self.by_cell[position]):
                    self.stats["revision_records_examined"] += 1
                    if identity not in self.inactive:
                        self._assign(position, value, identity)
                        break
        else:
            affected = [p for p, identity in self.sources.items() if identity in ids]
            before = (
                dict(self.cells)
                if affected and self.method == "reset"
                else {p: self.cells.get(p) for p in affected}
            )
            self.stats["revision_records_examined"] += len(self.sources)
            if affected and self.method == "reset":
                self.cells.clear()
                self.sources.clear()
                self.stats["map_resets"] += 1
            else:
                for p in affected:
                    self.cells.pop(p, None)
                    self.sources.pop(p, None)
        touched = (
            before.keys() | self.cells.keys() if self.method == "history_scan" else before.keys()
        )
        self.stats["revision_changed_cells"] += sum(
            before.get(p) != self.cells.get(p) for p in touched
        )
        self.stats["revision_retained_cells"] += before_count - sum(
            v is not None and self.cells.get(p) != v for p, v in before.items()
        )
        self.stats["revision_ns"] += perf_counter_ns() - start

    def supports(self, requirements):
        return all(self.label(p) == label for p, label in requirements)

    def metrics(self):
        entries = sum(len(items) for _, items in self.frames.values())
        indexes = (
            sum(len(values) for values in self.by_cell.values()) if self.by_cell is not None else 0
        )
        return {
            **self.stats,
            "receipts": len(self.receipts),
            "history_entries": entries,
            "index_entries": indexes,
            "active_cells": len(self.cells),
            "dependency_work": 0,
            "dependency_ms": 0.0,
            "assumptions": entries,
            "active_assumptions": len(self.cells),
            "facts": len(self.cells),
            "rule_edges": 0,
            "reverse_index_edges": indexes,
        }
