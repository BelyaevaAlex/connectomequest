#!/usr/bin/env python3
"""Answers under other labels: two rules that complete the declared correspondences, the labels of an earlier release, and the questions of two well-described circuits.

Every benchmark question is answered, after all checks, by the reference labels of its connectome. This script answers
the same questions under other labels of the same input cells. One pass per connectome computes the bounds of every
question under seven labellings, so that no labelling repeats the work of another:

  reference      the labels of the benchmark, derived again from the annotation release and compared with the benchmark
  suffix         reference; a release-specific type whose name extends the name of a hemibrain type by a suffix
                 (LC10a extends LC10) receives that hemibrain type
  linked         reference; a release-specific type receives the hemibrain type that the release of the other
                 connectome declares for the types linked to it: the type of the same name and the types that the
                 field flywireType of the MaleCNS release links. The rule applies when all linked types agree
  both           reference with both rules; a type on which the rules disagree stays unresolved
  suffix-check   the correspondence that a release declares in its hemibrain field is replaced by the suffix rule
                 wherever the rule applies: the rule is tested on types whose correspondence is known
  linked-check   the same test of the linked rule, on every type with a known correspondence
  earlier        the labels that the same crosswalk gives for the earlier annotation release of the connectome
                 (FlyWire v2.1.0, MaleCNS v0.9), with the synaptic weights of the benchmark unchanged

The two rules add labels only to cells that the reference leaves unresolved, so they can certify uncertifiable
questions and cannot change a certified answer. The two checks measure how often a rule would have given another
answer than the declared correspondence. The earlier labels measure what trusting an annotation release costs: of the
answers that the earlier release certifies, how many the later release certifies again, leaves open or reverses, and
whether a smallest certificate (heaviest cells first) is still valid cell by cell.

The analysis also splits the questions by the circuit of the target type (mushroom body, central complex, other; by
the name of the hemibrain type) and builds, for each circuit, the priority list of scripts/correspondences.py from the
stored arrays of that analysis. The same arrays give the list that remains after both rules: the release-specific
types that neither rule labels, ranked for the questions that stay uncertifiable, each with the questions it would
certify alone when its cells are read as nonmembers (no) or as members (yes).

Nothing is fitted and no policy runs. The development questions were analysed first; expected outcomes were then
recorded with the hashes of this code, the inputs and the development outputs, and the held-out questions are
analysed once, guarded by a claim file.

Stages
  types                        agreement of both rules with declared correspondences, type by type (no questions)
  run --view V --split S       bounds of every question under every labelling
  analyze --split S            summary with intervals from a cluster bootstrap over target types
  record                       pins code, inputs, development outputs and expected outcomes before the held-out run
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import correspondences as base
from correspondences_readings import split_parent

ROOT, BENCH, RUNS = base.ROOT, base.BENCH, base.RUNS
OUT = ROOT / "outputs/labelings"
RECORD = OUT / "record.json"
VIEWS, RHOS = base.VIEWS, base.RHOS
OTHER_VIEW = dict(malecns="flywire", flywire="malecns")
ANNOTATIONS = dict(malecns="data/raw/manc/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                   flywire="inputs/v32/flywire/cq-flywire-v3.1.0.tsv")
EARLIER = dict(malecns="inputs/earlier/malecns-v0.9/body-annotations-male-cns-v0.9-minconf-0.5.feather",
               flywire="inputs/v32/flywire/cq-flywire-v2.1.0.tsv")
RELEASE = dict(malecns="male-cns:v1.0", flywire="flywire:v3.1.0")
EARLIER_RELEASE = dict(malecns="male-cns:v1.0", flywire="flywire:v2.1.0")  # name aliases; MaleCNS v0.9 uses those of v1.0
VOCABULARY = "inputs/v32/prepared-common-v1/hemibrain/public/manifest.json"
LABELINGS = ("reference", "suffix", "linked", "both", "suffix-check", "linked-check", "earlier")
RULES = ("suffix", "linked", "both")
CHECKS = ("suffix-check", "linked-check")
NONE, SEVERAL = -1, -2
CIRCUITS = ("mushroom body", "central complex", "other")
MUSHROOM_BODY = re.compile(r"(KC|group:KC|MBON|PAM|PPL1|APL$|DPM$)")
CENTRAL_COMPLEX = re.compile(r"(EPG|EL$|PEG|PEN_|PFG|PFL|PFN|PFR|Delta7|P1-9|P6-8P9|LPsP|SpsP|IbSpsP|ER\d|ExR|FB\d|FC\d|FR\d|FS\d|hDelta|vDelta"
                             r"|LNO|LCNO|GLNO|SA\d|SAF)")
CODE = ("scripts/labelings.py", "scripts/correspondences.py", "scripts/correspondences_readings.py", "scripts/v35_diagnostics.py",
        "src/connectomequest/annotation_v35/engine.py", "src/connectomequest/annotation_v32/data.py")
DATA = ("outputs/v35/benchmark/manifest.json", "outputs/v35/benchmark/malecns.npz", "outputs/v35/benchmark/flywire.npz",
        ANNOTATIONS["malecns"], ANNOTATIONS["flywire"], EARLIER["malecns"], EARLIER["flywire"], VOCABULARY) + tuple(
    f"outputs/correspondences/{view}-{split}.{kind}" for view in VIEWS for split in ("development", "test") for kind in ("npz", "json"))
DEVELOPMENT = ("types.json", "analysis-development.json") + tuple(f"{view}-development.npz" for view in VIEWS)


# ---------------------------------------------------------------------------------------------- labels
def crosswalk(primary, secondary, release, lookup):
    """The crosswalk of the benchmark: the release type if it is a hemibrain type, else one declared hemibrain type."""
    from connectomequest.annotation_v32.data import ontology_type

    name = ontology_type(primary or None, release)
    if name in lookup:
        return lookup[name], "release-type"
    names = [ontology_type(x.strip(), release) for x in (secondary or "").split(",") if x.strip()]
    if len(names) == 1 and names[0] in lookup:
        return lookup[names[0]], "hemibrain-field"
    return NONE, "unresolved"


def read_annotations(view, path):
    """cell id -> (type of the release, hemibrain field, FlyWire type field of MaleCNS or None)."""
    if view == "flywire":
        with (ROOT / path).open(newline="") as stream:
            return {int(r["root_id"]): (r["cell_type"] or None, r["hemibrain_type"] or None, None)
                    for r in csv.DictReader(stream, delimiter="\t")}
    import pyarrow.ipc as ipc

    with (ROOT / path).open("rb") as stream:
        table = ipc.open_file(stream).read_all().select(["bodyId", "type", "hemibrainType", "flywireType"]).to_pylist()
    return {int(r["bodyId"]): (r["type"] or None, r["hemibrainType"] or None, r["flywireType"] or None) for r in table}


def declared_labels(rows, release, lookup):
    """Per type name of a release: (label its resolved cells agree on, or SEVERAL; whether a cell is labelled by the field)."""
    seen, by_field = {}, set()
    for primary, secondary, _ in rows.values():
        if primary:
            label, how = crosswalk(primary, secondary, release, lookup)
            if label >= 0:
                seen.setdefault(primary, set()).add(label)
                if how == "hemibrain-field":
                    by_field.add(primary)
    return {name: next(iter(labels)) if len(labels) == 1 else SEVERAL for name, labels in seen.items()}, by_field


def type_links(male_rows):
    """Type names linked across the connectomes by the field flywireType of the MaleCNS release, in both directions."""
    to_fly, to_male = {}, {}
    for primary, _, field in male_rows.values():
        if primary and field:
            for name in (x.strip() for x in field.split(",")):
                if name:
                    to_fly.setdefault(primary, set()).add(name)
                    to_male.setdefault(name, set()).add(primary)
    return dict(malecns=to_fly, flywire=to_male)


def suffix_rule(name, lookup):
    parent = split_parent(name, lookup)
    return lookup[parent] if parent else NONE


def linked_rule(name, other_declared, links):
    """The one hemibrain type that the other release declares for all linked types that have a correspondence."""
    labels = {other_declared.get(n, NONE) for n in links.get(name, set()) | {name}} - {NONE}
    return next(iter(labels)) if len(labels) == 1 and SEVERAL not in labels else NONE


def combine(first, second):
    """Both rules: the label that one gives or on which both agree; unresolved if they disagree."""
    labels = {first, second} - {NONE}
    return next(iter(labels)) if len(labels) == 1 else NONE


class Tables:
    """Annotation rows of both reference releases with the two rules, shared by every stage."""

    def __init__(self):
        from v35_diagnostics import unresolved_reason

        vocabulary = json.loads((ROOT / VOCABULARY).read_text())["vocabulary"]
        self.vocabulary = vocabulary
        self.lookup = {name: i for i, name in enumerate(vocabulary) if i}
        self.rows = {view: read_annotations(view, ANNOTATIONS[view]) for view in VIEWS}
        self.declared, self.by_field = {}, {}
        for view in VIEWS:
            self.declared[view], self.by_field[view] = declared_labels(self.rows[view], RELEASE[view], self.lookup)
        self.links = type_links(self.rows["malecns"])
        self._reason = unresolved_reason
        self._rules = {view: {} for view in VIEWS}

    def rules(self, view, name):
        """(suffix label, linked label) of a type name."""
        cache = self._rules[view]
        if name not in cache:
            cache[name] = (suffix_rule(name, self.lookup), linked_rule(name, self.declared[OTHER_VIEW[view]], self.links[view]))
        return cache[name]

    def release_specific(self, view, primary, secondary):
        return self._reason(primary, secondary, RELEASE[view], self.lookup) == base.RELEASE_SPECIFIC

    def cell_labels(self, view, cell, earlier):
        """Labels of one cell under every labelling, in the order of LABELINGS, and its release-specific type or None."""
        primary, secondary, _ = self.rows[view].get(cell, (None, None, None))
        reference, how = crosswalk(primary, secondary, RELEASE[view], self.lookup)
        suffix, linked = self.rules(view, primary) if primary else (NONE, NONE)
        specific = reference < 0 and self.release_specific(view, primary, secondary)
        old_primary, old_secondary, _ = earlier.get(cell, (None, None, None))
        old = crosswalk(old_primary, old_secondary, EARLIER_RELEASE[view], self.lookup)[0]
        return (reference,
                suffix if specific and suffix >= 0 else reference,
                linked if specific and linked >= 0 else reference,
                combine(suffix, linked) if specific else reference,
                suffix if how == "hemibrain-field" and suffix >= 0 else reference,
                linked if reference >= 0 and linked >= 0 else reference,
                old), (primary if specific else None)


def labelings(view, cells, tables):
    """Labels of the input cells (one row per labelling) and the release-specific types among them that each rule labels."""
    earlier = read_annotations(view, EARLIER[view])
    unique, inverse = np.unique(cells, return_inverse=True)
    labels = np.empty((len(LABELINGS), len(unique)), dtype=np.int32)
    specific = {}
    for j, cell in enumerate(unique.tolist()):
        row, name = tables.cell_labels(view, cell, earlier)
        labels[:, j] = row
        if name is not None:
            specific.setdefault(name, []).append(j)
    covered = {rule: sorted(name for name, js in specific.items() if labels[LABELINGS.index(rule), js[0]] >= 0) for rule in RULES}
    return labels[:, inverse], dict(release_specific_types=len(specific), covered=covered)


def outcomes(labels, codes):
    """1 member, 2 nonmember, 0 unresolved: the same codes as the answers yes and no."""
    return np.where(np.isin(labels, codes), 1, np.where(labels >= 0, 2, 0))


def bounds(outcome, weights):
    """Lower and upper bound of the member weight after all checks, per labelling."""
    lower = ((outcome == 1) * weights).sum(-1)
    return lower, int(weights.sum()) - ((outcome == 2) * weights).sum(-1)


def answer(lower, upper, threshold):
    """1 yes, 2 no, 0 uncertifiable."""
    return np.where(lower >= threshold, 1, np.where(upper < threshold, 2, 0))


def smallest_certificate(outcome, weights, status, threshold):
    """Positions of a smallest certificate of a certified answer: the heaviest cells with the certifying outcome."""
    order = np.argsort(-weights, kind="stable")
    pick = order[outcome[order] == status]
    need = threshold if status == 1 else int(weights.sum()) - threshold + 1
    return pick[: int(np.searchsorted(np.cumsum(weights[pick]), need)) + 1]


def circuit_of(name):
    return CIRCUITS[0] if MUSHROOM_BODY.match(name) else CIRCUITS[1] if CENTRAL_COMPLEX.match(name) else CIRCUITS[2]


# ---------------------------------------------------------------------------------------------- types
def types(args):
    """Agreement of both rules with the correspondences that the releases declare, over all types of each release."""
    t = Tables()
    result = {}
    for view in VIEWS:
        declared = {name: label for name, label in t.declared[view].items() if label >= 0}
        specific, cells_of = set(), {}
        for primary, secondary, _ in t.rows[view].values():
            if primary:
                cells_of[primary] = cells_of.get(primary, 0) + 1
                if primary not in t.declared[view] and t.release_specific(view, primary, secondary):
                    specific.add(primary)
        out = dict(types=len(cells_of), declared=len(declared), declared_by_field=len(set(declared) & t.by_field[view]),
                   release_specific=len(specific), release_specific_cells=sum(cells_of[n] for n in specific))
        for k, rule in enumerate(("suffix", "linked")):
            pool = sorted(set(declared) & t.by_field[view]) if rule == "suffix" else sorted(declared)
            tested = [(name, t.rules(view, name)[k]) for name in pool]
            tested = [(name, label) for name, label in tested if label >= 0]
            wrong = [(name, label) for name, label in tested if label != declared[name]]
            labelled = sorted(name for name in specific if t.rules(view, name)[k] >= 0)
            out[rule] = dict(tested=len(tested), agree=len(tested) - len(wrong), agreement=(len(tested) - len(wrong)) / len(tested),
                             tested_cells=sum(cells_of[n] for n, _ in tested), wrong_cells=sum(cells_of[n] for n, _ in wrong),
                             disagreements=[dict(type=name, rule=t.vocabulary[label], declared=t.vocabulary[declared[name]])
                                            for name, label in wrong],
                             labels=len(labelled), labels_cells=sum(cells_of[n] for n in labelled),
                             examples=[dict(type=name, hemibrain=t.vocabulary[t.rules(view, name)[k]]) for name in labelled[:12]])
        pairs = [t.rules(view, name) for name in sorted(specific)]
        out["both"] = dict(labels=sum(combine(*p) >= 0 for p in pairs), conflicts=sum(p[0] >= 0 and p[1] >= 0 and p[0] != p[1] for p in pairs),
                           by_both=sum(p[0] >= 0 and p[0] == p[1] for p in pairs))
        result[view] = out
        print(view, json.dumps({k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a not in ("disagreements", "examples")})
                                for k, v in out.items()}))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "types.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------------------------- run
def check_record():
    record = json.loads(RECORD.read_text())
    changed = [p for p, h in {**record["code"], **record["data"]}.items() if base.sha256(ROOT / p) != h]
    changed += [p for p, h in record["development"].items() if base.sha256(OUT / p) != h]
    if changed:
        raise SystemExit(f"files differ from the record: {changed}")
    return record


def run(args):
    from connectomequest.annotation_v35 import engine as e

    out = OUT / f"{args.view}-{args.split}.npz"
    if args.split == "test":
        check_record()
        with (OUT / f"claim-{args.view}-test").open("x") as stream:  # one run per connectome: refuses if the claim exists
            stream.write(datetime.now(timezone.utc).isoformat() + "\n")
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
    elif out.exists() and not args.force:
        raise SystemExit(f"{out} exists (use --force on the development split)")
    with np.load(BENCH / f"{args.view}.npz", allow_pickle=False) as z:
        bench = {k: z[k] for k in ("target_type", "target_offset", "pre_cell", "pre_weight", "pre_label", "fam_target", "fam_codes",
                                   "fam_code_offset", "fam_source_weight", "fam_source_total")}
    labels, found = labelings(args.view, bench["pre_cell"], Tables())
    if not np.array_equal(labels[0], bench["pre_label"]):
        raise SystemExit("the reference labels differ from the benchmark")
    earlier = LABELINGS.index("earlier")
    rows = {k: [] for k in ("family", "rho", "cluster", "threshold", "total", "cells", "lower", "upper", "changed_cells", "changed_weight",
                            "certificate_size", "certificate_changed", "certificate_valid")}
    for f in base.split_families(bench, args.split).tolist():
        t = int(bench["fam_target"][f])
        lo, hi = int(bench["target_offset"][t]), int(bench["target_offset"][t + 1])
        weights = bench["pre_weight"][lo:hi].astype(np.int64)
        codes = bench["fam_codes"][bench["fam_code_offset"][f]:bench["fam_code_offset"][f + 1]]
        outcome = outcomes(labels[:, lo:hi], codes)
        lower, upper = bounds(outcome, weights)
        changed = outcome != outcome[0]
        total = int(weights.sum())
        for rho, (num, den) in enumerate(RHOS):
            threshold, _ = e.threshold_goals(total, num * int(bench["fam_source_weight"][f]), den * int(bench["fam_source_total"][f]))
            status = int(answer(lower[earlier], upper[earlier], threshold))
            size, other = 0, 0
            if status:
                certificate = smallest_certificate(outcome[earlier], weights, status, threshold)
                size, other = len(certificate), int((outcome[0][certificate] != status).sum())
            for key, value in (("family", f), ("rho", rho), ("cluster", int(bench["target_type"][t])), ("threshold", threshold),
                               ("total", total), ("cells", hi - lo), ("lower", lower), ("upper", upper),
                               ("changed_cells", changed.sum(1)), ("changed_weight", (changed * weights).sum(1)),
                               ("certificate_size", size), ("certificate_changed", other), ("certificate_valid", bool(status) and not other)):
                rows[key].append(value)
    arrays = {k: np.array(v, dtype=bool if k == "certificate_valid" else np.int64) for k, v in rows.items()}

    # the reference answers must reproduce the pre-specified runs question by question
    stored = np.load(RUNS / f"{args.view}-{args.split}.npz")
    key = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(stored["family"], stored["rho"]))}
    index = np.array([key[(int(f), int(r))] for f, r in zip(arrays["family"], arrays["rho"])])
    mine = dict(status=answer(arrays["lower"][:, 0], arrays["upper"][:, 0], arrays["threshold"]), threshold=arrays["threshold"],
                lower=arrays["lower"][:, 0], total=arrays["total"], unresolved=arrays["upper"][:, 0] - arrays["lower"][:, 0])
    for name, theirs in (("status", "truth"), ("threshold", "threshold"), ("lower", "member"), ("total", "total"), ("unresolved", "unresolved")):
        if len(index) != len(stored["family"]) or not np.array_equal(mine[name], stored[theirs][index]):
            raise SystemExit(f"{name} differs from the pre-specified runs")

    OUT.mkdir(parents=True, exist_ok=True)
    with out.open("wb" if args.split == "development" else "xb") as stream:
        np.savez_compressed(stream, **arrays)
    meta = dict(view=args.view, split=args.split, questions=len(arrays["family"]), labelings=list(LABELINGS),
                release_specific_types=found["release_specific_types"], covered={k: len(v) for k, v in found["covered"].items()},
                covered_types=found["covered"], code={p: base.sha256(ROOT / p) for p in CODE}, result_sha256=base.sha256(out))
    (OUT / f"{args.view}-{args.split}.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("code", "covered_types")}))


# ---------------------------------------------------------------------------------------------- analysis
def circuit_lists(view, split, z, circuit, share):
    """Priority lists of scripts/correspondences.py built for the questions of each circuit, from its stored arrays."""
    stored = np.load(base.OUT / f"{view}-{split}.npz")
    names = json.loads((base.OUT / f"{view}-{split}.json").read_text())["type_names"]
    if not (np.array_equal(stored["family"], z["family"]) and np.array_equal(stored["rho"], z["rho"])):
        raise SystemExit("the questions differ from those of scripts/correspondences.py")
    question, kind, weight = stored["pair_question"], stored["pair_type"], stored["pair_weight"]
    deficit = stored["upper"] - stored["threshold"] + 1
    uncertifiable = stored["status"] == 0
    reach = uncertifiable & (np.bincount(question, weights=weight, minlength=len(uncertifiable)) >= deficit)
    out = {}
    for j, name in enumerate(CIRCUITS):
        mine = circuit == j
        a = mine[question]
        order, _ = base.greedy_list(question[a], kind[a], weight[a], deficit, len(names), names)
        step = base.apply_list(order, question[a], kind[a], weight[a], deficit, len(names))
        target = reach & mine
        summary = base.list_summary(np.where(target, step, 0), target, share, len(order))
        out[name] = dict(reachable=share(target, uncertifiable & mine), uncertifiable_questions=int((uncertifiable & mine).sum()),
                         reachable_questions=int(target.sum()), types=int(len(np.unique(kind[a]))), length=summary["length"],
                         first=summary["first"], types_for=summary["types_for"], top=[names[k] for k in order[:10].tolist()])
    return out


def residual_list(view, split, z, share):
    """The priority list for the questions that stay uncertifiable after both rules, over the types that no rule labels."""
    stored = np.load(base.OUT / f"{view}-{split}.npz")
    names = json.loads((base.OUT / f"{view}-{split}.json").read_text())["type_names"]
    labelled = set(json.loads((OUT / f"{view}-{split}.json").read_text())["covered_types"]["both"])
    k = LABELINGS.index("both")
    lower, upper, threshold = z["lower"][:, k], z["upper"][:, k], z["threshold"]
    question, kind, weight = stored["pair_question"], stored["pair_type"], stored["pair_weight"]
    by_rule = np.isin(kind, [j for j, name in enumerate(names) if name in labelled])
    removed = np.bincount(question[by_rule], weights=weight[by_rule], minlength=len(threshold)).astype(np.int64)
    if not np.array_equal(removed[stored["status"] == 0], z["changed_weight"][:, k][stored["status"] == 0]):
        raise SystemExit("the cells that the rules label differ from the types of scripts/correspondences.py")
    open_ = answer(lower, upper, threshold) == 0
    keep = ~by_rule & open_[question]
    question, kind, weight = question[keep], kind[keep], weight[keep]
    deficit = upper - threshold + 1
    order, _ = base.greedy_list(question, kind, weight, deficit, len(names), names)
    step = base.apply_list(order, question, kind, weight, deficit, len(names))
    reach = open_ & (np.bincount(question, weights=weight, minlength=len(threshold)) >= deficit)
    summary = base.list_summary(np.where(reach, step, 0), reach, share, len(order))
    no, yes = upper[question] - weight < threshold[question], lower[question] + weight >= threshold[question]
    count = lambda mask: np.bincount(kind[mask], minlength=len(names))
    with_type, alone_no, alone_yes = np.bincount(kind, minlength=len(names)), count(no), count(yes)
    one = lambda mask: share(np.isin(np.arange(len(threshold)), question[mask]), open_)
    return dict(open=share(open_, np.ones(len(threshold))), open_questions=int(open_.sum()), reachable=share(reach, open_),
                types=int(len(np.unique(kind))), length=summary["length"], first=summary["first"], types_for=summary["types_for"],
                one_type_no=one(no), one_type_yes=one(yes),
                top=[dict(type=names[j], questions=int(with_type[j]), no=int(alone_no[j]), yes=int(alone_yes[j])) for j in order[:10].tolist()])


def analyse_view(view, split, vocabulary):
    z = np.load(OUT / f"{view}-{split}.npz")
    status = answer(z["lower"], z["upper"], z["threshold"][:, None])
    reference = status[:, 0]
    ones = np.ones(len(reference))
    share = base.resampler(z["cluster"], np.random.default_rng(base.SEED))
    unresolved = z["upper"][:, 0] - z["lower"][:, 0]
    out = dict(questions=len(reference), target_types=int(len(np.unique(z["cluster"]))), uncertifiable=share(reference == 0, ones))

    for name in RULES:  # labels are only added: a certified answer cannot change
        k = LABELINGS.index(name)
        if not ((reference == 0) | (status[:, k] == reference)).all():
            raise SystemExit(f"{name} changed a certified answer")
        open_ = reference == 0
        out[name] = dict(certified=share(open_ & (status[:, k] > 0), open_), yes=share(open_ & (status[:, k] == 1), open_),
                         no=share(open_ & (status[:, k] == 2), open_), certified_questions=int((open_ & (status[:, k] > 0)).sum()),
                         with_labelled_cells=share(open_ & (z["changed_cells"][:, k] > 0), open_),
                         unresolved_weight_labelled=share(open_ * z["changed_weight"][:, k], open_ * unresolved))
    for name in CHECKS:
        k = LABELINGS.index(name)
        certified, touched = reference > 0, z["changed_cells"][:, k] > 0
        reversed_ = certified & (status[:, k] > 0) & (status[:, k] != reference)
        out[name] = dict(questions_with_other_labels=share(touched, ones), same_answer=share(status[:, k] == reference, ones),
                         reversed=share(reversed_, certified), reversed_questions=int(reversed_.sum()),
                         lost=share(certified & (status[:, k] == 0), certified), lost_questions=int((certified & (status[:, k] == 0)).sum()),
                         gained=share(~certified & (status[:, k] > 0), ~certified), gained_questions=int((~certified & (status[:, k] > 0)).sum()))

    k = LABELINGS.index("earlier")
    old = status[:, k]
    certified = old > 0
    kept = certified & (reference == old)
    words = ("uncertifiable", "yes", "no")
    out["earlier"] = dict(
        certified=share(certified, ones), uncertifiable=share(~certified, ones),
        kept=share(kept, certified), lost=share(certified & (reference == 0), certified),
        reversed=share(certified & (reference > 0) & (reference != old), certified),
        reversed_questions=int((certified & (reference > 0) & (reference != old)).sum()),
        kept_yes=share(kept & (old == 1), old == 1), kept_no=share(kept & (old == 2), old == 2),
        gained=share(~certified & (reference > 0), ~certified), same_answer=share(reference == old, ones),
        gained_yes=share(~certified & (reference == 1), ~certified & (reference > 0)), gained_questions=int((~certified & (reference > 0)).sum()),
        lost_questions=int((certified & (reference == 0)).sum()),
        certificate_valid=share(z["certificate_valid"], certified), certificate_valid_when_kept=share(z["certificate_valid"] & kept, kept),
        certificate_cells=float(np.median(z["certificate_size"][certified])), certificate_cells_mean=float(z["certificate_size"][certified].mean()),
        certificate_cells_changed=share(z["certificate_changed"], z["certificate_size"]),
        certified_before=share(kept, reference > 0),
        cells_with_other_outcome=share(z["changed_cells"][:, k], z["cells"]), weight_with_other_outcome=share(z["changed_weight"][:, k], z["total"]),
        questions_with_other_outcome=share(z["changed_cells"][:, k] > 0, ones),
        transitions={f"{words[a]} to {words[b]}": int(((old == a) & (reference == b)).sum()) for a in range(3) for b in range(3)})

    out["residual"] = residual_list(view, split, z, share)
    circuit = np.array([CIRCUITS.index(circuit_of(vocabulary[c])) for c in z["cluster"]])
    lists = circuit_lists(view, split, z, circuit, share)
    out["circuit"] = {}
    for j, name in enumerate(CIRCUITS):
        mine = circuit == j
        out["circuit"][name] = dict(questions=int(mine.sum()), target_types=int(len(np.unique(z["cluster"][mine]))),
                                    uncertifiable=share(mine & (reference == 0), mine), yes=share(mine & (reference == 1), mine),
                                    no=share(mine & (reference == 2), mine),
                                    certified_by_both_rules=share(mine & (reference == 0) & (status[:, LABELINGS.index("both")] > 0),
                                                                  mine & (reference == 0)),
                                    earlier_kept=share(mine & kept, mine & certified), list=lists[name])
    return out


def analyze(args):
    record = check_record() if args.split == "test" else None
    vocabulary = json.loads((ROOT / VOCABULARY).read_text())["vocabulary"]
    result = {view: analyse_view(view, args.split, vocabulary) for view in VIEWS}
    (OUT / f"analysis-{args.split}.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    if record:
        verdicts = base.evaluate(record["expectations"], result)
        (OUT / "verdicts-test.json").write_text(json.dumps(verdicts, indent=1) + "\n")
        print(f"{sum(v['holds'] for v in verdicts)} of {len(verdicts)} expectations hold")
        for v in verdicts:
            if not v["holds"]:
                print("  does not hold:", v["text"], v["value"])
    r3 = lambda x: round(x["value"], 4)
    for view in VIEWS:
        r = result[view]
        print(view, "uncertifiable", r3(r["uncertifiable"]))
        for name in RULES:
            print("  ", name, {k: (r3(v) if isinstance(v, dict) else v) for k, v in r[name].items()})
        for name in CHECKS:
            print("  ", name, {k: (r3(v) if isinstance(v, dict) else v) for k, v in r[name].items()})
        print("   earlier", {k: (r3(v) if isinstance(v, dict) and "value" in v else v) for k, v in r["earlier"].items()})
        print("   residual", {k: (r3(v) if isinstance(v, dict) and "value" in v else v) for k, v in r["residual"].items() if k not in ("first", "top")},
              "first", {k: r3(v) for k, v in r["residual"]["first"].items()})
        for row in r["residual"]["top"][:6]:
            print("      ", row)
        for name, c in r["circuit"].items():
            print("  ", name, {k: (r3(v) if isinstance(v, dict) and "value" in v else v) for k, v in c.items() if k != "list"},
                  "list", dict(reachable=r3(c["list"]["reachable"]), types=c["list"]["types"], types_for=c["list"]["types_for"],
                               first={k: r3(v) for k, v in c["list"]["first"].items()}))


def write_record(args):
    if RECORD.exists():
        raise SystemExit("the record exists")
    expectations = json.loads((OUT / "expectations.json").read_text())
    record = dict(written_utc=datetime.now(timezone.utc).isoformat(), purpose=__doc__.split("\n\n")[0],
                  code={p: base.sha256(ROOT / p) for p in CODE}, data={p: base.sha256(ROOT / p) for p in DATA},
                  development={p: base.sha256(OUT / p) for p in DEVELOPMENT},
                  expectations=expectations["expectations"], notes=expectations.get("notes", []))
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    RECORD.chmod(0o444)
    print(f"recorded {len(record['expectations'])} expectations")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("types")
    p.set_defaults(call=types)
    p = sub.add_parser("run")
    p.add_argument("--view", required=True, choices=VIEWS)
    p.add_argument("--split", required=True, choices=("development", "test"))
    p.add_argument("--force", action="store_true")
    p.set_defaults(call=run)
    p = sub.add_parser("analyze")
    p.add_argument("--split", required=True, choices=("development", "test"))
    p.set_defaults(call=analyze)
    p = sub.add_parser("record")
    p.set_defaults(call=write_record)
    args = parser.parse_args()
    args.call(args)


if __name__ == "__main__":
    main()
