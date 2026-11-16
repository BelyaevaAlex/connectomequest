#!/usr/bin/env python3
"""Published statements: how does MaleCNS name the cells of a partner type that none of its neurons carries?

One statement names the partner type pC1d, and under the reference no MaleCNS neuron is a pC1d. The answers alone
cannot tell a connectome without such cells from a release that names them otherwise. This script reads the raw
annotation release of MaleCNS and counts, for every partner type of a statement that no neuron of the release
carries, the neurons whose release type equals the name, the release types that share its stem (the name without
its last letter), their neurons, and the hemibrain types that the release declares for those neurons.

It asks no question and reads no connectome graph. It was added after the run of the statements, without a record.

Usage: python scripts/published_names.py        # writes outputs/published/names.json
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import published as p  # noqa: E402

RELEASE = "data/raw/manc/body-annotations-male-cns-v1.0-minconf-0.5.feather"
OUTPUT = p.OUT / "names.json"


def naming(types, declared, name):
    """Counts for one type name: its neurons, the release types that share its stem, and what the release declares.

    ``declared_as_type`` counts the neurons of the whole release, whatever their type, for which the release names
    the type among the hemibrain types it declares.
    """
    stem = name[:-1]
    sharing = [(t, d) for t, d in zip(types, declared) if t and t != name and t.startswith(stem)]
    counts = collections.Counter(d for _, d in sharing if d)
    return dict(type=name, stem=stem, neurons=sum(t == name for t in types),
                types_sharing_stem=len({t for t, _ in sharing}), neurons_sharing_stem=len(sharing),
                declared_hemibrain_types={key: counts[key] for key in sorted(counts)},
                declared_as_type=sum(name in d.split(",") for d in declared if d))


def main():
    import pyarrow.feather as feather

    table = feather.read_table(ROOT / RELEASE, columns=["type", "hemibrainType"])
    types, declared = table.column("type").to_pylist(), table.column("hemibrainType").to_pylist()
    carried = set(filter(None, types))
    partners = sorted({name for s in p.read_statements(ROOT / p.STATEMENTS) for name in s["partner"]})
    names = [name for name in partners if not name.endswith("*") and ":" not in name]  # type names, not patterns or groups
    out = [naming(types, declared, name) for name in names if name not in carried]
    for row in out:
        print(f"{row['type']}: {row['neurons']} neurons, {row['types_sharing_stem']} release types begin with {row['stem']} "
              f"({row['neurons_sharing_stem']} neurons), declared as {row['type']}: {row['declared_as_type']}")
    OUTPUT.write_text(json.dumps(dict(release=RELEASE.rsplit("/", 1)[1], neurons=len(types), types=out), indent=1) + "\n")


if __name__ == "__main__":
    main()
