"""Lean latest-observation map for baselines that do not need support indexes."""


class PlainMap:
    def __init__(self):
        self.cells = {}

    def observe(self, receipt, digest, items):
        items = dict(items)
        changed = {p for p, label in items.items() if self.cells.get(p) != label}
        self.cells.update(items)
        return changed

    def invalidate(self, positions):
        for p in positions:
            self.cells.pop(p, None)

    def label(self, pos):
        return self.cells.get(pos)

    def metrics(self):
        return {
            "dependency_work": 0,
            "dependency_ms": 0.0,
            "receipts": 0,
            "assumptions": 0,
            "active_assumptions": 0,
            "facts": len(self.cells),
            "rule_edges": 0,
            "reverse_index_edges": 0,
        }
