#!/usr/bin/env python3
"""Published statements in the source connectome: is each statement read as its authors meant it?

For every statement of inputs/published/statements.json the script computes, in the hemibrain itself, the share of
the input synapses of the neurons of the target type that the partner types supply, twice: over all synapses between
traced neurons (the raw edge list of the public release), and over the neuron pairs joined by at least five
synapses, the restriction of every graph of the benchmark. The first share is the one a published statement reports;
the second must equal the share that scripts/published.py stored for the hemibrain graph of the benchmark, and the
script stops if it does not.

It reads no evaluation connectome and asks no question. Types are those of the benchmark's hemibrain labels.

Usage: python scripts/published_source.py        # writes outputs/published/source.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import published as p  # noqa: E402

RAW = "data/raw/hemibrain"
MINIMUM = 5  # synapses per neuron pair kept by the graphs of the benchmark
OUTPUT = p.OUT / "source.json"


def shares(pre, post, weight, labels_of, partner, target, minimum):
    """Partner and total input synapses of the target neurons, over all pairs and over pairs with at least ``minimum``."""
    targets = np.array(sorted(body for body, label in labels_of.items() if label in target), dtype=np.int64)
    members = np.array(sorted(body for body, label in labels_of.items() if label in partner), dtype=np.int64)
    into = np.isin(post, targets) & (pre != post)
    member = np.isin(pre[into], members)
    w = weight[into]
    kept = w >= minimum
    return dict(neurons=int(len(targets)), synapses=int(w.sum()), partner_synapses=int(w[member].sum()),
                synapses_kept=int(w[kept].sum()), partner_synapses_kept=int(w[member & kept].sum()))


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq

    import v35_build as b

    statements = p.read_statements(ROOT / p.STATEMENTS)
    stored = {s["id"]: s["hemibrain"] for s in json.loads((p.OUT / "results.json").read_text())["statements"]}
    vocabulary = json.loads((ROOT / b.SOURCE / "public/manifest.json").read_text())["vocabulary"]
    rows = pq.read_table(ROOT / b.SOURCE / "private/labels.parquet").to_pylist()
    labels_of = {int(r["cell_id"]): int(r["label"]) for r in rows if r["label"]}
    edges = pa.concat_tables([pq.read_table(f) for f in sorted((ROOT / RAW / "edges").glob("*.parquet"))])
    pre, post, weight = (edges.column(name).to_numpy() for name in ("body_pre", "body_post", "weight"))
    out = []
    for s in statements:
        partner = set(p.codes_of(s["partner"], vocabulary).tolist())
        target = set(p.codes_of(s["target"], vocabulary).tolist())
        r = shares(pre, post, weight, labels_of, partner, target, MINIMUM)
        kept = stored[s["id"]]
        if (r["neurons"], r["synapses_kept"], r["partner_synapses_kept"]) != (kept["neurons"], kept["synapses"], kept["partner_synapses"]):
            raise SystemExit(f"{s['id']}: the raw edges with at least {MINIMUM} synapses differ from the hemibrain graph of the benchmark")
        out.append(dict(id=s["id"], percent=s["percent"], share_all=r["partner_synapses"] / r["synapses"],
                        share_kept=r["partner_synapses_kept"] / r["synapses_kept"], **r))
        print(f"{s['id']}: published {s['percent']}%, all synapses {100 * out[-1]['share_all']:.1f}%, pairs of at least {MINIMUM} {100 * out[-1]['share_kept']:.1f}%")
    OUTPUT.write_text(json.dumps(dict(source=RAW, release=json.loads((ROOT / RAW / "manifest.json").read_text())["metadata"],
                                      minimum=MINIMUM, statements=out), indent=1) + "\n")


if __name__ == "__main__":
    main()
