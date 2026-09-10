"""Read-only structural checks for experiment reports, independent of inference."""

from collections import Counter, defaultdict
from math import isclose, isfinite
from statistics import mean


def validate_cells(rows, methods, seeds, query_ids, budget, horizon):
    """Require the exact method x seed x query matrix, not just observed cells."""
    ids = set(query_ids)
    if not ids or len(ids) != len(query_ids):
        raise ValueError("expected query IDs must be nonempty and unique")
    if not seeds or len(set(seeds)) != len(seeds) or len(set(methods)) != len(methods):
        raise ValueError("invalid method/seed specification")
    expected = {(m, s) for m in methods for s in (seeds[:1] if m == "weight" else seeds)}
    cells = defaultdict(list)
    seen = set()
    for row in rows:
        key = row["method"], row["seed"]
        identity = (*key, row["query_id"])
        if key not in expected or row["query_id"] not in ids or identity in seen:
            raise ValueError(f"unexpected or duplicate episode: {identity}")
        seen.add(identity)
        for field, cap in [("goal_success", 1), ("budget_used", budget), ("steps", horizon)]:
            value = float(row[field])
            if not isfinite(value) or not 0 <= value <= cap or not value.is_integer():
                raise ValueError(f"invalid {field} in {identity}")
        cells[key].append(row)
    if set(cells) != expected:
        raise ValueError(f"missing cells: {sorted(expected - set(cells))}")
    for key, cell in cells.items():
        if {r["query_id"] for r in cell} != ids:
            raise ValueError(f"incomplete query coverage: {key}")
    return dict(cells)


def validate_summary(cells, summary):
    """Recompute published means and failure frequencies from checked rows."""
    methods = {m for m, _ in cells}
    if set(summary) != methods:
        raise ValueError("summary method set differs from the rows")

    def equal(actual, expected, name):
        if not isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"stale summary: {name}")

    for method in methods:
        rows = [r for (m, _), cell in cells.items() if m == method for r in cell]
        saved = summary[method]
        if not isinstance(saved, dict):
            equal(saved, mean(r["goal_success"] for r in rows), method)
            continue
        equal(saved["episodes"], len(rows), f"{method}.episodes")
        for published, field in [
            ("success", "goal_success"),
            ("budget", "budget_used"),
            ("primary_success", "primary_success"),
            ("distinct_middles", "distinct_middles"),
        ]:
            equal(saved[published], mean(r[field] for r in rows), f"{method}.{published}")
        counts = Counter(r["failure_category"] for r in rows)
        if set(counts) != set(saved["failure_rates"]):
            raise ValueError(f"stale summary failure categories: {method}")
        for category, count in counts.items():
            equal(saved["failure_rates"][category], count / len(rows), f"{method}.{category}")
        for r in rows:
            if r["failure_category"] != "not_applicable":
                if (r["failure_category"] == "success") != bool(r["goal_success"]):
                    raise ValueError(f"failure label disagrees with success: {method}")


def validate_queries(records, n, prior_pairs, anchor_split):
    """Verify fresh-pair uniqueness, prior-pair exclusion and anchor partition."""
    if len(records) != n:
        raise ValueError("wrong fresh query count")
    ids, pairs = set(), set()
    for record in records:
        pair = tuple(record["anchors"])
        if len(pair) != 2 or pair in pairs or pair in prior_pairs:
            raise ValueError("duplicate, previously observed or malformed query pair")
        qid = record["query_id"]
        if not qid or qid in ids or not record["answers"]:
            raise ValueError("duplicate query ID or empty answer set")
        if anchor_split(pair[0]) != "test":
            raise ValueError("anchor not in frozen test partition")
        ids.add(qid)
        pairs.add(pair)
    return [r["query_id"] for r in records]
