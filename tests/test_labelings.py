"""Tests of scripts/labelings.py: the labellings, the bounds they give and the smallest certificate."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import labelings as lab  # noqa: E402
from connectomequest.annotation_v35 import engine as e  # noqa: E402

LOOKUP = {"LC10": 1, "MBON01": 2, "PAM01": 3, "group:KCab-mcs": 4, "LT1": 5}


def test_crosswalk_follows_the_benchmark_rule():
    assert lab.crosswalk("LC10", None, "male-cns:v1.0", LOOKUP) == (1, "release-type")
    assert lab.crosswalk("LC10a", "LC10", "male-cns:v1.0", LOOKUP) == (1, "hemibrain-field")
    assert lab.crosswalk("KCab-m", None, "male-cns:v1.0", LOOKUP) == (4, "release-type")  # declared group of the vocabulary
    assert lab.crosswalk("LC10a", "LC10, LT1", "male-cns:v1.0", LOOKUP) == (lab.NONE, "unresolved")  # several candidates
    assert lab.crosswalk("LC10a", "LC99", "male-cns:v1.0", LOOKUP) == (lab.NONE, "unresolved")  # outside the vocabulary
    assert lab.crosswalk("LC10a", None, "male-cns:v1.0", LOOKUP) == (lab.NONE, "unresolved")
    assert lab.crosswalk(None, None, "male-cns:v1.0", LOOKUP) == (lab.NONE, "unresolved")


def test_suffix_rule_needs_a_hemibrain_name_and_a_suffix():
    assert lab.suffix_rule("LC10a", LOOKUP) == 1
    assert lab.suffix_rule("LC10c-1", LOOKUP) == 1
    assert lab.suffix_rule("LT1_b", LOOKUP) == 5
    assert lab.suffix_rule("LC101", LOOKUP) == lab.NONE  # a digit continues the name of another type
    assert lab.suffix_rule("LC10", LOOKUP) == lab.NONE  # no suffix
    assert lab.suffix_rule("Tm5f", LOOKUP) == lab.NONE


def test_linked_rule_needs_one_label_from_all_linked_types():
    other = {"LC10a": 1, "LT1": 5, "X": lab.SEVERAL, "MBON01": 2}
    assert lab.linked_rule("LC10a", other, {}) == 1  # the type of the same name
    assert lab.linked_rule("A", other, {"A": {"LT1"}}) == 5  # a linked type
    assert lab.linked_rule("A", other, {"A": {"LT1", "unknown"}}) == 5  # a linked type without correspondence is ignored
    assert lab.linked_rule("A", other, {"A": {"LT1", "MBON01"}}) == lab.NONE  # linked types disagree
    assert lab.linked_rule("LC10a", other, {"LC10a": {"LT1"}}) == lab.NONE  # same name and link disagree
    assert lab.linked_rule("A", other, {"A": {"X"}}) == lab.NONE  # the linked type has several correspondences
    assert lab.linked_rule("A", other, {"A": {"X", "LT1"}}) == lab.NONE
    assert lab.linked_rule("A", other, {}) == lab.NONE


def test_combine_keeps_agreement_and_drops_conflict():
    assert lab.combine(1, lab.NONE) == 1
    assert lab.combine(lab.NONE, 5) == 5
    assert lab.combine(5, 5) == 5
    assert lab.combine(1, 5) == lab.NONE
    assert lab.combine(lab.NONE, lab.NONE) == lab.NONE


def test_type_links_run_both_ways():
    rows = {1: ("A", None, "X, Y"), 2: ("A", None, "X"), 3: ("B", None, None), 4: (None, None, "Z"), 5: ("C", None, "X")}
    links = lab.type_links(rows)
    assert links["malecns"] == {"A": {"X", "Y"}, "C": {"X"}}
    assert links["flywire"] == {"X": {"A", "C"}, "Y": {"A"}}


def test_declared_labels_are_unanimous_or_several():
    rows = {1: ("LC10a", "LC10", None), 2: ("LC10a", None, None), 3: ("Q", "LC10", None), 4: ("Q", "LT1", None), 5: ("LT1", None, None),
            6: ("R", None, None)}
    declared, by_field = lab.declared_labels(rows, "male-cns:v1.0", LOOKUP)
    assert declared == {"LC10a": 1, "Q": lab.SEVERAL, "LT1": 5}
    assert by_field == {"LC10a", "Q"}


def tables(rows, other_rows):
    t = lab.Tables.__new__(lab.Tables)
    from v35_diagnostics import unresolved_reason

    t.vocabulary = ["__OTHER__"] + sorted(LOOKUP, key=LOOKUP.get)
    t.lookup = LOOKUP
    t.rows = dict(malecns=rows, flywire=other_rows)
    t.declared, t.by_field = {}, {}
    for view in lab.VIEWS:
        t.declared[view], t.by_field[view] = lab.declared_labels(t.rows[view], lab.RELEASE[view], LOOKUP)
    t.links = lab.type_links(rows)
    t._reason = unresolved_reason
    t._rules = {view: {} for view in lab.VIEWS}
    return t


def test_cell_labels_under_every_labelling():
    male = {
        1: ("LC10a", None, None),        # release-specific, suffix gives LC10
        2: ("Tm5f", None, "Tm5f_fly"),   # release-specific, linked through the FlyWire type
        3: ("LT1b", "MBON01", None),     # declared by the field; the suffix rule would say LT1
        4: ("MBON01", None, "M1"),       # declared by its name; the linked FlyWire type says PAM01
        5: ("Zz9", None, None),          # release-specific, no rule
        6: ("LC10b", "LC10, LT1", None),  # several candidates: not release-specific
        7: (None, None, None),
    }
    fly = {10: ("Tm5f_fly", "LT1", None), 11: ("M1", "PAM01", None)}
    t = tables(male, fly)
    earlier = {1: ("LC10", None, None), 3: ("LT1b", None, None), 5: ("MBON01", None, None)}
    got = {cell: t.cell_labels("malecns", cell, earlier) for cell in male}
    #            reference suffix linked both suffix-check linked-check earlier
    assert got[1] == ((-1, 1, -1, 1, -1, -1, 1), "LC10a")
    assert got[2] == ((-1, -1, 5, 5, -1, -1, -1), "Tm5f")
    assert got[3] == ((2, 2, 2, 2, 5, 2, -1), None)
    assert got[4] == ((2, 2, 2, 2, 2, 3, -1), None)
    assert got[5] == ((-1, -1, -1, -1, -1, -1, 2), "Zz9")
    assert got[6] == ((-1, -1, -1, -1, -1, -1, -1), None)
    assert got[7] == ((-1, -1, -1, -1, -1, -1, -1), None)
    assert len(got[1][0]) == len(lab.LABELINGS)


def test_labelings_reports_the_types_each_rule_labels():
    male = {1: ("LC10a", None, None), 2: ("Tm5f", None, "Tm5f_fly"), 5: ("Zz9", None, None), 8: ("LC10a", None, None)}
    fly = {10: ("Tm5f_fly", "LT1", None)}
    t = tables(male, fly)
    original = lab.read_annotations
    lab.read_annotations = lambda view, path: {}
    try:
        labels, found = lab.labelings("malecns", np.array([5, 1, 2, 8, 1, 99]), t)
    finally:
        lab.read_annotations = original
    assert labels.shape == (len(lab.LABELINGS), 6)
    assert labels[lab.LABELINGS.index("both")].tolist() == [-1, 1, 5, 1, 1, -1]
    assert labels[lab.LABELINGS.index("reference")].tolist() == [-1] * 6
    assert found == dict(release_specific_types=3, covered=dict(suffix=["LC10a"], linked=["Tm5f"], both=["LC10a", "Tm5f"]))


@pytest.mark.parametrize("seed", range(40))
def test_bounds_and_answer_agree_with_the_engine(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 12))
    weights = rng.integers(1, 30, size=n).astype(np.int64)
    labels = rng.integers(-1, 6, size=(3, n))
    codes = rng.choice(np.arange(1, 6), size=int(rng.integers(1, 4)), replace=False)
    threshold = int(rng.integers(1, int(weights.sum()) + 1))
    outcome = lab.outcomes(labels, codes)
    lower, upper = lab.bounds(outcome, weights)
    status = lab.answer(lower, upper, threshold)
    instance = e.threshold_instance(np.arange(n), weights, threshold)
    for k in range(3):
        engine_outcome = np.where(outcome[k] == 1, e.MEMBER, np.where(outcome[k] == 2, e.NONMEMBER, e.UNRESOLVED))
        assert ("unresolved", "positive", "negative")[int(status[k])] == e.full_information_status(instance, engine_outcome)
        if status[k]:
            certificate = lab.smallest_certificate(outcome[k], weights, int(status[k]), threshold)
            assert len(certificate) == e.clairvoyant_cost(instance, engine_outcome)
            assert len(set(certificate.tolist())) == len(certificate) and (outcome[k][certificate] == status[k]).all()
            need = threshold if status[k] == 1 else int(weights.sum()) - threshold + 1
            assert weights[certificate].sum() >= need > weights[certificate[:-1]].sum()


@pytest.mark.parametrize("seed", range(40))
def test_added_labels_never_change_a_certified_answer(seed):
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(2, 12))
    weights = rng.integers(1, 30, size=n).astype(np.int64)
    reference = rng.integers(-1, 6, size=n)
    more = np.where((reference < 0) & (rng.random(n) < 0.6), rng.integers(1, 6, size=n), reference)
    codes = rng.choice(np.arange(1, 6), size=2, replace=False)
    threshold = int(rng.integers(1, int(weights.sum()) + 1))
    lower, upper = lab.bounds(lab.outcomes(np.stack([reference, more]), codes), weights)
    before, after = lab.answer(lower, upper, threshold)
    assert before == 0 or after == before
    assert lower[1] >= lower[0] and upper[1] <= upper[0]


def test_circuits_by_the_name_of_the_hemibrain_type():
    for name in ("MBON01", "KCg-m", "group:KCab-mcs", "PAM04", "PPL101", "APL", "DPM"):
        assert lab.circuit_of(name) == "mushroom body", name
    for name in ("EPG", "EL", "PEN_a", "PFNa", "Delta7", "FB4H", "ER3m", "ExR1", "hDeltaA", "vDeltaK", "LNO1", "SA1_a", "P6-8P9"):
        assert lab.circuit_of(name) == "central complex", name
    for name in ("LC10", "ELF1", "PPL201", "APL2", "SAD001", "FB", "ERx", "LAL001", "MeVP10", "CL366"):
        assert lab.circuit_of(name) == "other", name


def test_expectations_file_is_evaluated_like_the_base_analysis():
    result = dict(malecns=dict(earlier=dict(kept=dict(value=0.99, low=0.98, high=1.0), reversed_questions=0)))
    verdicts = lab.base.evaluate([dict(text="a", quantity="malecns|earlier|kept", min=0.95),
                                  dict(text="b", quantity="malecns|earlier|reversed_questions", max=0),
                                  dict(text="c", quantity="malecns|earlier|kept", max=0.9)], result)
    assert [v["holds"] for v in verdicts] == [True, True, False]
