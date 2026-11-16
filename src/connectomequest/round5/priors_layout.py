"""Benchmark layout of a prior and its quality on question families.

The frozen benchmark (``outputs/v35/benchmark/<view>.npz``) stores a prior as two arrays:
``plus_<name>`` with one value per (question family, input cell of its target), families in
order and cells in the order of ``pre_cell`` within the target, and ``unknown_<name>`` with one
value per entry of ``pre_cell`` (per target, independent of the partner set).
"""

from __future__ import annotations

import numpy as np

from connectomequest.annotation_v35 import engine as e
from connectomequest.annotation_v35 import sweep as sw
from connectomequest.neurosym import common as nc


def node_rows(view, cells):
    ids = nc.view_data(view)["ids"]
    rows = np.searchsorted(ids, cells)
    if np.any(rows >= len(ids)) or not np.array_equal(ids[rows], cells):
        raise ValueError("cell absent from the view")
    return rows


def benchmark_arrays(view, mass, other, is_seed, label, confidence, families=None):
    """plus/unknown arrays in the benchmark layout.

    ``mass(rows, codes)`` gives the model mass on a class set, ``other[row]`` the model mass on the
    extra class; seeds (``is_seed[row]``) get ``confidence`` on their known ``label[row]`` and
    ``1 - confidence`` times the model. ``families`` restricts the work to some families (tests);
    the plus entries of the others are left at NaN.
    """
    d = nc.benchmark(view)
    rows = node_rows(view, d["pre_cell"])
    seed = is_seed[rows]
    known = label[rows]
    unknown = (1 - confidence) * other[rows] + confidence * (seed & (known == 0))
    unknown = np.where(seed, unknown, other[rows])
    plus = np.full(int(d["fam_plus_offset"][-1]), np.nan)
    todo = range(len(d["fam_target"])) if families is None else [int(f) for f in families]
    for f in todo:
        t = int(d["fam_target"][f])
        lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
        codes = d["fam_codes"][d["fam_code_offset"][f] : d["fam_code_offset"][f + 1]]
        r = rows[lo:hi]
        m = mass(r, codes)
        s = seed[lo:hi]
        m = np.where(s, (1 - confidence) * m + confidence * np.isin(known[lo:hi], codes), m)
        start = int(d["fam_plus_offset"][f])
        plus[start : start + hi - lo] = m
    plus = np.clip(plus, 0.0, 1.0)
    unknown = np.clip(unknown, 0.0, 1.0)
    # rows must stay normalisable: plus + unknown <= 1
    for f in todo:
        t = int(d["fam_target"][f])
        lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
        start = int(d["fam_plus_offset"][f])
        over = plus[start : start + hi - lo] + unknown[lo:hi] - 1.0
        if np.any(over > 1e-5):
            raise ArithmeticError("plus + unknown exceeds one")
    return plus.astype(np.float32), unknown.astype(np.float32), seed


def family_cells(view, f):
    d = nc.benchmark(view)
    t = int(d["fam_target"][f])
    lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
    codes = d["fam_codes"][d["fam_code_offset"][f] : d["fam_code_offset"][f + 1]]
    labels = d["pre_label"][lo:hi]
    outcomes = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
    start = int(d["fam_plus_offset"][f])
    return lo, hi, start, outcomes


def quality(view, plus, unknown, families, keep=None):
    """Within-question AUROC (members vs others, questions with both classes), pooled Brier and ECE (10 bins).

    ``keep``: optional mask over ``pre_cell`` entries (e.g. non-seed cells) restricting every statistic.
    Mirrors ``scripts/v35_analyze.py:calibration`` when ``keep`` is None.
    """
    aurocs, brier, count = [], 0.0, 0
    bins = np.zeros((10, 3))
    unknown_sum, unresolved_sum, top_cells = 0.0, 0.0, 0
    d = nc.benchmark(view)
    for f in families:
        lo, hi, start, y = family_cells(view, f)
        p = plus[start : start + hi - lo].astype(float)
        u = unknown[lo:hi].astype(float)
        sel = np.ones(hi - lo, dtype=bool) if keep is None else keep[lo:hi]
        if not sel.any():
            continue
        p, y, u = p[sel], y[sel], u[sel]
        member = (y == e.MEMBER).astype(float)
        brier += float(((p - member) ** 2).sum())
        count += len(p)
        idx = np.minimum((p * 10).astype(int), 9)
        np.add.at(bins, (idx, 0), 1)
        np.add.at(bins, (idx, 1), p)
        np.add.at(bins, (idx, 2), member)
        a = sw.member_auroc(p, y)
        if a is not None:
            aurocs.append(a)
        if d["fam_kind"][f] == 0:
            unknown_sum += float(u.sum())
            unresolved_sum += float((y == e.UNRESOLVED).sum())
            top_cells += len(u)
    nonempty = bins[:, 0] > 0
    return dict(
        member_auroc_within_question=float(np.mean(aurocs)) if aurocs else None,
        questions_with_both_classes=len(aurocs),
        member_brier=brier / max(count, 1),
        member_ece=float(np.abs(bins[nonempty, 1] - bins[nonempty, 2]).sum() / max(count, 1)),
        mean_predicted_unresolved=unknown_sum / max(top_cells, 1),
        observed_unresolved_rate=unresolved_sum / max(top_cells, 1),
        cells=count,
    )


def question_aurocs(view, plus, families):
    """Within-question AUROC per family (NaN without both classes)."""
    out = np.full(len(families), np.nan)
    for i, f in enumerate(families):
        lo, hi, start, y = family_cells(view, f)
        a = sw.member_auroc(plus[start : start + hi - lo].astype(float), y)
        out[i] = np.nan if a is None else a
    return out
