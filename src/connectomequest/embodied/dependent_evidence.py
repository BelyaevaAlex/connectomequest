"""Finite acyclic AND/OR support graph, with immutable acquisition receipts."""

from collections import defaultdict, deque
from contextlib import contextmanager
from time import perf_counter_ns


class EvidenceGraph:
    def __init__(self, algorithm="indexed"):
        if algorithm not in {"indexed", "full"}:
            raise ValueError(algorithm)
        self.algorithm = algorithm
        self.receipts = {}
        self.assumptions = {}  # assumption id -> (receipt, fact)
        self.active = set()
        self.roots = defaultdict(set)
        self.rules = defaultdict(list)
        self.users = defaultdict(set)
        self.order = []
        self.truth = {}
        self.work = 0
        self.update_ns = 0
        self.pending = None

    @contextmanager
    def batch(self):
        if self.pending is not None:
            raise RuntimeError("nested batch")
        self.pending = set()
        try:
            yield
        finally:
            changed = self.pending
            self.pending = None
            self._update(changed)

    def _ensure(self, fact):
        if fact not in self.truth:
            self.truth[fact] = False
            self.order.append(fact)

    def record(self, receipt, payload_digest):
        if receipt in self.receipts and self.receipts[receipt] != payload_digest:
            raise ValueError("receipt payload cannot be overwritten")
        self.receipts[receipt] = payload_digest

    def assert_fact(self, assumption, receipt, fact):
        if receipt not in self.receipts:
            raise ValueError("unregistered acquisition")
        if assumption in self.assumptions:
            raise ValueError("duplicate assumption identifier")
        self._ensure(fact)
        self.assumptions[assumption] = (receipt, fact)
        self.roots[fact].add(assumption)
        self.active.add(assumption)
        self._update([fact])

    def justify(self, head, premises):
        premises = tuple(premises)
        if not premises or any(p not in self.truth for p in premises):
            raise ValueError("premises must be existing facts")
        self._ensure(head)
        rank = {f: i for i, f in enumerate(self.order)}
        if any(rank[p] >= rank[head] for p in premises):
            raise ValueError("cyclic/non-topological rule")
        if premises not in self.rules[head]:
            self.rules[head].append(premises)
            for fact in premises:
                self.users[fact].add(head)
            self._update([head])

    def withdraw(self, assumptions):
        changed = set()
        for assumption in assumptions:
            if assumption in self.active:
                self.active.remove(assumption)
                changed.add(self.assumptions[assumption][1])
        self._update(changed)

    def _value(self, fact):
        self.work += len(self.roots[fact])
        root = bool(self.roots[fact] & self.active)
        supported = False
        for rule in self.rules[fact]:
            self.work += len(rule)
            supported |= all(self.truth[p] for p in rule)
        return root or supported

    def _update(self, changed):
        if self.pending is not None:
            self.pending.update(changed)
            return
        start = perf_counter_ns()
        if self.algorithm == "full":
            for fact in self.order:
                self.truth[fact] = self._value(fact)
        else:
            queue = deque(sorted(changed, key=repr))
            while queue:
                fact = queue.popleft()
                old = self.truth[fact]
                self.truth[fact] = self._value(fact)
                if old != self.truth[fact]:
                    queue.extend(sorted(self.users[fact], key=repr))
        self.update_ns += perf_counter_ns() - start

    def supported(self, fact):
        return self.truth.get(fact, False)

    def metrics(self):
        return {
            "dependency_work": self.work,
            "dependency_ms": self.update_ns / 1e6,
            "receipts": len(self.receipts),
            "assumptions": len(self.assumptions),
            "active_assumptions": len(self.active),
            "facts": len(self.truth),
            "rule_edges": sum(len(r) for rs in self.rules.values() for r in rs),
            "reverse_index_edges": sum(len(v) for v in self.users.values()),
        }


class SupportedMap:
    def __init__(self, algorithm="indexed"):
        self.graph = EvidenceGraph(algorithm)
        self.cells = {}
        self.cell_assumptions = defaultdict(set)
        self.last_receipt = {}
        self.counter = 0

    @staticmethod
    def fact(position, label):
        return ("cell", tuple(position), label)

    def observe(self, receipt, digest, predictions):
        with self.graph.batch():
            return self._observe(receipt, digest, predictions)

    def _observe(self, receipt, digest, predictions):
        self.graph.record(receipt, digest)
        changed = set()
        for pos, label in dict(predictions).items():
            pos = tuple(pos)
            if self.cells.get(pos) != label:
                changed.add(pos)
                self.graph.withdraw(self.cell_assumptions[pos])
                self.cell_assumptions[pos].clear()
            self.counter += 1
            aid = f"{receipt}:{self.counter}"
            self.graph.assert_fact(aid, receipt, self.fact(pos, label))
            self.cell_assumptions[pos].add(aid)
            self.cells[pos] = label
            self.last_receipt[pos] = receipt
        return changed

    def invalidate(self, positions):
        for pos in positions:
            self.graph.withdraw(self.cell_assumptions[pos])
            self.cell_assumptions[pos].clear()
            self.cells.pop(pos, None)

    def label(self, pos):
        return self.cells.get(pos)
