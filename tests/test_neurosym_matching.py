"""Pilot C: candidate-set answers checked by the frozen V35 verifier."""

import numpy as np
import pytest

from connectomequest.annotation_v35 import engine as e
from connectomequest.neurosym import common as c
from connectomequest.neurosym import matching as m

import pytest as _pytest

from connectomequest.neurosym import common as _common

if not all((_common.VIEW_ROOTS[v] / "public/topology.npz").exists() and (_common.BENCH / f"{v}.npz").exists() for v in _common.VIEW_ROOTS):
    _pytest.skip("needs the connectome graphs and the benchmark arrays", allow_module_level=True)

B = m.BOTTOM


def _truth_answer(label, codes):
    """True response: unresolved cells (None) stay unresolved; BOTTOM is a nonmember of every set."""
    if label is None:
        return e.UNRESOLVED
    return e.MEMBER if label in codes else e.NONMEMBER


def _prefix_statuses(inst, y, order):
    """Status after every prefix of ``order`` (exact verifier)."""
    return [e.evaluate(inst, y, order[: i + 1])[1] for i in range(len(order))]


def test_answer_semantics():
    a = {3, 5}
    assert m.answer(frozenset({3}), a) == e.MEMBER
    assert m.answer(frozenset({3, 5}), a) == e.MEMBER
    assert m.answer(frozenset({3, B}), a) == e.UNRESOLVED  # BOTTOM blocks a member answer
    assert m.answer(frozenset({4, B}), a) == e.NONMEMBER  # BOTTOM is a nonmember of every set
    assert m.answer(frozenset({B}), a) == e.NONMEMBER
    assert m.answer(frozenset({3, 4}), a) == e.UNRESOLVED
    assert m.answer(None, a) == e.UNRESOLVED
    codes, sims = np.array([7, 8, 9]), np.array([0.9, 0.88, 0.5])
    assert m.connectivity_set(codes, sims, 2, 0.05, 0.6) == frozenset({7, 8})
    assert m.connectivity_set(codes, sims, 1, 1.0, 0.6) == frozenset({7})
    assert m.connectivity_set(codes, sims * 0.5, 2, 0.05, 0.6) == frozenset({B})
    assert m.connectivity_set(codes, sims * 0.5, 2, 0.05, 0.6, None) is None


def test_soundness_when_true_type_in_candidates():
    """If the true type is in S, no prefix of any order certifies against the true certificate."""
    rng = np.random.default_rng(0)
    universe = list(range(1, 13))
    for _ in range(400):
        n = int(rng.integers(1, 14))
        weights = rng.integers(1, 30, size=n)
        cells = rng.permutation(10**6)[:n] + 1
        labels = [None if rng.random() < 0.2 else (B if rng.random() < 0.2 else int(rng.choice(universe))) for _ in range(n)]
        codes = set(rng.choice(universe, size=int(rng.integers(1, 5)), replace=False).tolist())
        sets = []
        for lab in labels:
            if lab is None or rng.random() < 0.15:
                sets.append(None)  # no constraint: always unresolved
                continue
            extra = set(rng.choice(universe + [B], size=int(rng.integers(0, 4)), replace=False).tolist())
            sets.append(frozenset({lab} | extra))
        assert all(s is None or m.contains(s, l) for s, l in zip(sets, labels))
        true = np.array([_truth_answer(l, codes) for l in labels], dtype=np.intp)
        cand = m.answers(sets, codes)
        threshold = int(rng.integers(1, int(weights.sum()) + 1))
        inst = e.threshold_instance(cells, weights, threshold)
        truth = e.full_information_status(inst, true)
        assert e.full_information_status(inst, cand) in ("unresolved", truth)
        for order in (e.weight_order(inst), rng.permutation(n)):
            for status in _prefix_statuses(inst, cand, order):
                assert status in ("unresolved", truth)
            cost_c, status_c = e.evaluate(inst, cand, order)
            cost_t, _ = e.evaluate(inst, true, order)
            if status_c != "unresolved":
                assert cost_c >= cost_t  # a coarser reading never certifies sooner


def test_true_type_outside_candidates_can_contradict():
    """The guarantee needs the true type in S: a wrong singleton flips the certificate."""
    inst = e.threshold_instance(np.array([1, 2]), np.array([5, 5]), 6)
    true = np.array([_truth_answer(l, {1}) for l in (1, 2)], dtype=np.intp)  # member, nonmember -> negative
    cand = m.answers([frozenset({1}), frozenset({1})], {1})  # second cell wrongly read as type 1
    assert e.full_information_status(inst, true) == "negative"
    assert e.full_information_status(inst, cand) == "positive"


def test_declared_field_parsing():
    voc = c.vocabulary()
    lookup = {n: i for i, n in enumerate(voc) if i}
    release = c.RELEASE["flywire"]
    s, kind = m.parse_declared("AOTU055,AOTU056", release)
    assert kind == "multi-valued" and s == frozenset({lookup["AOTU055"], lookup["AOTU056"]})
    s, kind = m.parse_declared("PAM06", release)
    assert kind == "subtypes" and s == frozenset({lookup["PAM06_a"], lookup["PAM06_b"]})
    s, kind = m.parse_declared("ORN_DC1", release)
    assert kind == "outside-vocabulary" and s == frozenset({B})
    s, kind = m.parse_declared("KCab-m", release)  # hemibrain ontology coarsening applies to declared hemibrain names
    assert kind == "in-vocabulary" and s == frozenset({lookup["group:KCab-mcs"]})
    hb = c.hemibrain()
    typed = int(np.flatnonzero(hb["labels"] > 0)[0])
    body, label = int(hb["ids"][typed]), int(hb["labels"][typed])
    assert m.parse_declared(f"hb{body}", release) == (frozenset({label}), "hemibrain-body")
    assert m.parse_declared(f"({body})", release) == (frozenset({label}), "hemibrain-body")
    assert m.parse_declared("hb999999999999", release) == (None, "hemibrain-body-absent")


def test_soundness_on_development_families_with_connectivity_distractors():
    """Real half-1 families: every resolved cell read through {true type} + connectivity top-3 (+ sometimes BOTTOM)."""
    view = "flywire"
    rng = np.random.default_rng(1)
    fams = rng.choice(c.development_families(view, 1), size=25, replace=False)
    d = c.view_data(view)
    rows = m.rows_of(view)
    for f in fams.tolist():
        fam = c.family(view, f)
        rel = d["release_code"][[rows[int(x)] for x in fam["cells"]]]
        types = tuple(sorted(set(int(r) for r in rel if r >= 0)))
        top, _ = m.similarity_top(view, types)
        pos = {t: i for i, t in enumerate(types)}
        sets = []
        for lab, r in zip(fam["labels"].tolist(), rel.tolist()):
            if lab < 0 or r < 0:
                sets.append(None)
                continue
            extra = set(top[pos[int(r)], :3].tolist()) | ({B} if rng.random() < 0.3 else set())
            sets.append(frozenset({lab} | extra))
        cand = m.answers(sets, fam["codes"])
        for _, threshold in c.thresholds(fam):
            inst = e.threshold_instance(fam["cells"], fam["weights"], threshold)
            truth = e.full_information_status(inst, fam["outcomes"])
            assert e.full_information_status(inst, cand) in ("unresolved", truth)
            for status in _prefix_statuses(inst, cand, e.weight_order(inst)):
                assert status in ("unresolved", truth)
