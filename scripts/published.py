#!/usr/bin/env python3
"""Published typed-input statements, asked end to end in MaleCNS and FlyWire.

Each statement is taken from a paper that analyses the hemibrain: named presynaptic cell types supply a stated share
of the synaptic input of a named postsynaptic cell type. For every neuron of the postsynaptic type in an evaluation
connectome the script asks two questions, with the engine, the reference and the earlier-release prior of the
benchmark: does the partner set supply at least the published share of the neuron's input synapses, and does it
supply at least half of that share?

For every question it stores the answer under the reference (certified yes, certified no, uncertifiable), the exact
bounds L and U after all checks, the clairvoyant bound and the number of checks of weight order and, with the
earlier-release prior, of direction-fixed, Adaptive Greedy and recalibrated Adaptive Greedy. For the hemibrain it
stores the share that the partner set supplies in the graph of the benchmark, which keeps neuron pairs joined by at
least five synapses; the published share counts all synapses, so the two need not coincide.

The statements, with the sentences quoted from the papers, are in inputs/published/statements.json. The record pins
that file, this script, the engine and the input data before the run. The run is made once. The table and the numbers
of the paper are written from its output by published_assets.py.

Stages
  published.py record     # before the run
  published.py run        # once; writes outputs/published/questions.npz and results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from connectomequest.annotation_v35 import engine as e  # noqa: E402

OUT = ROOT / "outputs/published"
RECORD = OUT / "record.json"
STATEMENTS = "inputs/published/statements.json"
EARLIER = "inputs/earlier/malecns-v0.9/body-annotations-male-cns-v0.9-minconf-0.5.feather"
LEVELS = (("published", 1, 1), ("half", 1, 2))
POLICIES = ("weight", "direction", "ag", "recal")
STATUS = {"unresolved": 0, "positive": 1, "negative": 2}
CODE = ("scripts/published.py", "scripts/v35_build.py", "src/connectomequest/annotation_v35/engine.py",
        "src/connectomequest/annotation_v32/data.py", "src/connectomequest/neurosym/common.py",
        "src/connectomequest/neurosym/constraints.py")
DATA = (STATEMENTS, EARLIER, "inputs/v32/prepared-common-v1/hemibrain/public/topology.npz",
        "inputs/v32/prepared-common-v1/hemibrain/public/manifest.json",
        "inputs/v32/prepared-common-v1/hemibrain/private/labels.parquet",
        "inputs/v32/prepared-common-v1/malecns/public/topology.npz",
        "inputs/v32/prepared-common-v1-hints/flywire/public/topology.npz",
        "inputs/v32/prepared-common-v1-hints/flywire/public/nodes.parquet",
        "data/raw/manc/body-annotations-male-cns-v1.0-minconf-0.5.feather", "inputs/v32/flywire/cq-flywire-v3.1.0.tsv",
        "outputs/neurosym/cache/malecns.npz", "outputs/neurosym/cache/flywire.npz")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- statements


def read_statements(path: Path):
    """The statements of the file, validated: quoted sentence, source, types and a share in (0, 100]."""
    statements = json.loads(path.read_text())["statements"]
    seen = set()
    for s in statements:
        for key in ("id", "source", "doi", "where", "quote", "partner", "target", "percent"):
            if key not in s:
                raise ValueError(f"statement without {key}: {s.get('id')}")
        if s["id"] in seen:
            raise ValueError(f"statement listed twice: {s['id']}")
        seen.add(s["id"])
        if not s["quote"].strip() or not s["partner"] or not s["target"]:
            raise ValueError(f"empty quote or types: {s['id']}")
        share = Fraction(str(s["percent"])) / 100
        if not 0 < share <= 1:
            raise ValueError(f"share outside (0, 1]: {s['id']}")
    return statements


def codes_of(patterns, vocabulary):
    """Vocabulary codes of the named types; a name that ends with ``*`` matches every type that starts with it.

    Code 0 is the class outside the vocabulary and is never matched. A name that matches no type is an error.
    """
    codes = set()
    for pattern in patterns:
        if pattern.endswith("*"):
            found = [i for i, name in enumerate(vocabulary) if i and name.startswith(pattern[:-1])]
        else:
            found = [i for i, name in enumerate(vocabulary) if i and name == pattern]
        if not found:
            raise ValueError(f"no hemibrain type matches {pattern!r}")
        codes.update(found)
    return np.array(sorted(codes), dtype=np.int32)


def threshold(total, percent, numerator, denominator):
    """Exact T = ceil(level * share * W) for a share given in percent as a decimal string or number."""
    share = Fraction(str(percent)) / 100 * Fraction(numerator, denominator)
    return e.threshold_goals(int(total), share.numerator, share.denominator)[0]


# --------------------------------------------------------------------------- one question


def probabilities(plus, unknown):
    plus = np.clip(np.asarray(plus, dtype=float), 0.0, 1.0)
    unknown = np.clip(np.asarray(unknown, dtype=float), 0.0, 1.0 - plus)
    return np.column_stack((plus, 1.0 - plus - unknown, unknown))


def earlier_prior(topology, hint, codes, alpha):
    """Probabilities of member, nonmember and unresolved: weight ``alpha`` on the earlier label, the rest on topology.

    ``topology`` holds one row per input cell with the distribution of the topology prior over the vocabulary
    (column 0 is the class outside it); ``hint`` holds the code of the earlier label of the cell or -1.
    """
    mixed = np.array(topology, dtype=np.float32, copy=True)
    hinted = np.flatnonzero(hint >= 0)
    mixed[hinted] *= np.float32(1 - alpha)
    mixed[hinted, hint[hinted]] += np.float32(alpha)
    return probabilities(mixed[:, codes].astype(np.float64).sum(1), mixed[:, 0])


def ask(cells, weights, outcomes, prior, goal):
    """Answer, bounds and costs of one question; every order is verified against the status of the full question."""
    inst = e.threshold_instance(cells, weights, goal)
    truth = e.full_information_status(inst, outcomes)
    bound = e.clairvoyant_cost(inst, outcomes)
    orders = dict(weight=e.weight_order(inst), direction=e.direction_order(inst, prior),
                  ag=e.run_adaptive(inst, outcomes, e.AdaptiveGreedy(inst, prior)),
                  recal=e.run_adaptive(inst, outcomes, e.RecalibratedGreedy(inst, prior)))
    costs = {}
    for name, order in orders.items():
        cost, status = e.evaluate(inst, outcomes, order)
        if status != truth or (truth == "unresolved" and cost != inst.size) or (bound is not None and cost < bound):
            raise AssertionError(f"{name} contradicts the exact full-information status")
        costs[name] = cost
    return truth, bound, costs


def summarize(rows):
    """Counts of answers and totals of checks for the questions of one statement, connectome and level."""
    truth = np.array([r["truth"] for r in rows])
    certifiable = truth != STATUS["unresolved"]
    out = dict(neurons=len(rows), yes=int((truth == STATUS["positive"]).sum()), no=int((truth == STATUS["negative"]).sum()),
               uncertifiable=int((truth == STATUS["unresolved"]).sum()))
    out["input_cells_median"] = float(np.median([r["size"] for r in rows])) if rows else None
    if certifiable.any():
        chosen = [r for r, ok in zip(rows, certifiable) if ok]
        out["input_cells_certifiable"] = int(sum(r["size"] for r in chosen))
        out["clairvoyant"] = int(sum(r["clairvoyant"] for r in chosen))
        out["weight_median"] = float(np.median([r["weight"] for r in chosen]))
        for name in POLICIES:
            out[name] = int(sum(r[name] for r in chosen))
    return out


# --------------------------------------------------------------------------- data


def load_views():
    """Graphs, reference labels, topology priors and earlier labels of the two evaluation connectomes."""
    import pyarrow.feather as feather
    import pyarrow.parquet as pq

    import v35_build as b
    from connectomequest.annotation_v32 import data as names_of
    from connectomequest.neurosym import common as c
    from connectomequest.neurosym import constraints as k

    vocabulary = json.loads((ROOT / b.SOURCE / "public/manifest.json").read_text())["vocabulary"]
    lookup = {name: i for i, name in enumerate(vocabulary) if i}
    views = {}
    for view, cfg in b.VIEWS.items():
        ids, matrix = b.load_graph(ROOT / cfg["root"])
        if not np.array_equal(ids, c.view_data(view)["ids"]):
            raise ValueError("node order differs from the cached graph")
        labels, _ = b.view_labels(ROOT, view, ids, lookup)
        views[view] = dict(ids=ids, matrix=matrix, labels=labels, topology=k.full_mlp(view))
    table = pq.read_table(ROOT / b.VIEWS["flywire"]["root"] / "public/nodes.parquet", columns=["cell_id", "provisional"]).to_pylist()
    found = {int(r["cell_id"]): r["provisional"] for r in table}
    views["flywire"]["hint"] = np.array([-1 if found[x] is None else found[x] for x in views["flywire"]["ids"].tolist()], dtype=np.int32)
    earlier = {int(r["bodyId"]): r["type"] for r in feather.read_table(ROOT / EARLIER, columns=["bodyId", "type"]).to_pylist()}
    code_of = {name: i for i, name in enumerate(vocabulary)}
    release = b.VIEWS["malecns"]["release"]
    views["malecns"]["hint"] = np.array(
        [-1 if names_of.canonical_type(earlier.get(x)) is None else code_of.get(names_of.ontology_type(earlier.get(x), release), 0)
         for x in views["malecns"]["ids"].tolist()], dtype=np.int32)
    return vocabulary, views, b.PROTOCOL["provisional_alpha"], b.PROTOCOL["minimum_partners"]


def hemibrain_share(partner, target):
    """Share of the input synapses of the hemibrain neurons of the target types that the partner types supply."""
    import pyarrow.parquet as pq

    import v35_build as b

    ids, matrix = b.load_graph(ROOT / b.SOURCE)
    rows = pq.read_table(ROOT / b.SOURCE / "private/labels.parquet").to_pylist()
    table = {int(r["cell_id"]): r["label"] for r in rows}
    labels = np.array([-1 if table[x] is None or table[x] == 0 else table[x] for x in ids.tolist()])
    columns = np.flatnonzero(np.isin(labels, target))
    block = matrix[:, columns]
    total = int(block.sum())
    member = int(block[np.isin(labels, partner)].sum())
    return dict(neurons=int(len(columns)), synapses=total, partner_synapses=member, share=member / total if total else None)


# --------------------------------------------------------------------------- stages


def check_record():
    if not RECORD.exists():
        raise SystemExit("the run needs the record")
    record = json.loads(RECORD.read_text())
    for group in ("code", "data"):
        for path, digest in record[group].items():
            if sha256(ROOT / path) != digest:
                raise SystemExit(f"{path} differs from the record")
    return record


def write_record(args):
    if RECORD.exists():
        raise SystemExit("the record exists")
    statements = read_statements(ROOT / STATEMENTS)
    OUT.mkdir(parents=True, exist_ok=True)
    record = dict(written_utc=datetime.now(timezone.utc).isoformat(), purpose=__doc__.split("\n\n")[0],
                  code={p: sha256(ROOT / p) for p in CODE}, data={p: sha256(ROOT / p) for p in DATA},
                  statements=[s["id"] for s in statements], levels=[name for name, _, _ in LEVELS], policies=list(POLICIES),
                  notes=["no expected outcome is recorded: the run reports every statement of the file, whatever its answer",
                         "the statements file lists what was looked at in the evaluation connectomes before this record"])
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    RECORD.chmod(0o444)
    print(f"recorded {len(statements)} statements")


def run(args):
    check_record()
    if (OUT / "results.json").exists():
        raise SystemExit("the result exists")
    with (OUT / "claim").open("x") as stream:
        stream.write(datetime.now(timezone.utc).isoformat() + "\n")
    statements = read_statements(ROOT / STATEMENTS)
    vocabulary, views, alpha, minimum = load_views()
    results, columns = [], {key: [] for key in ("statement", "view", "level", "cell", "size", "total", "threshold", "member", "unresolved",
                                                 "truth", "clairvoyant") + POLICIES}
    for index, s in enumerate(statements):
        partner, target = codes_of(s["partner"], vocabulary), codes_of(s["target"], vocabulary)
        entry = dict(id=s["id"], percent=s["percent"], partner_types=[vocabulary[i] for i in partner],
                     target_types=[vocabulary[i] for i in target], hemibrain=hemibrain_share(partner, target), views={})
        for v, (view, d) in enumerate(views.items()):
            matrix, labels = d["matrix"], d["labels"]
            neurons = np.flatnonzero(np.isin(labels, target) & (np.diff(matrix.indptr) >= minimum))
            rows = {name: [] for name, _, _ in LEVELS}
            member_sum = unresolved_sum = total_sum = 0
            for i in neurons.tolist():
                start, stop = matrix.indptr[i], matrix.indptr[i + 1]
                pre, weights = matrix.indices[start:stop], np.asarray(matrix.data[start:stop], dtype=np.int64)
                order = np.argsort(d["ids"][pre], kind="stable")
                pre, weights = pre[order], weights[order]
                outcomes = np.where(labels[pre] < 0, e.UNRESOLVED, np.where(np.isin(labels[pre], partner), e.MEMBER, e.NONMEMBER))
                prior = earlier_prior(d["topology"][pre], d["hint"][pre], partner, alpha)
                total = int(weights.sum())
                member, unresolved = int(weights[outcomes == e.MEMBER].sum()), int(weights[outcomes == e.UNRESOLVED].sum())
                member_sum, unresolved_sum, total_sum = member_sum + member, unresolved_sum + unresolved, total_sum + total
                for level, (name, numerator, denominator) in enumerate(LEVELS):
                    goal = threshold(total, s["percent"], numerator, denominator)
                    truth, bound, costs = ask(d["ids"][pre], weights, outcomes, prior, goal)
                    row = dict(size=len(pre), total=total, threshold=goal, member=member, unresolved=unresolved, truth=STATUS[truth],
                               clairvoyant=-1 if bound is None else bound, **costs)
                    rows[name].append(row)
                    for key, value in dict(row, statement=index, view=v, level=level, cell=int(d["ids"][i])).items():
                        columns[key].append(value)
            entry["views"][view] = dict(
                neurons=int(len(neurons)), synapses=total_sum,
                lower=member_sum / total_sum if total_sum else None,
                upper=(member_sum + unresolved_sum) / total_sum if total_sum else None,
                levels={name: summarize(rows[name]) for name, _, _ in LEVELS})
            print(json.dumps(dict(id=s["id"], view=view, **{k: entry["views"][view][k] for k in ("neurons", "lower", "upper")})), flush=True)
        results.append(entry)
    with (OUT / "questions.npz").open("xb") as stream:
        np.savez_compressed(stream, **{key: np.asarray(value, dtype=np.int64) for key, value in columns.items()})
    (OUT / "results.json").write_text(json.dumps(dict(statements=results, views=list(views), levels=[name for name, _, _ in LEVELS],
                                                      questions_sha256=sha256(OUT / "questions.npz")), indent=1) + "\n")
    print(f"{len(columns['truth'])} questions of {len(results)} statements")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("record", "run"):
        sub.add_parser(name)
    args = parser.parse_args()
    dict(record=write_record, run=run)[args.command](args)


if __name__ == "__main__":
    main()
