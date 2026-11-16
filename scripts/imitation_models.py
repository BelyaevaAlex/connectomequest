"""The fitted models of the imitation order as one array file.

``scripts/imitation.py fit`` writes one pickled scikit-learn model per connectome and prior
(``outputs/imitation/model-<view>-<prior>.pkl``); ``outputs/imitation/fit.json`` lists their hashes and the record
pins them. The archive carries binary data as ``.npz`` files, so ``pack`` stores the bytes of the three files,
unchanged, as unsigned 8-bit arrays in ``outputs/imitation/models.npz``. ``unpack`` writes the files back from the
array file and stops unless each hash equals the one in ``fit.json``; it never replaces an existing model.

Usage:
  imitation_models.py pack
  imitation_models.py unpack        # before re-running scripts/imitation.py from the archive
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "outputs/imitation"
PACKED = OUT / "models.npz"


def names():
    return {key: f"model-{key.replace(':', '-')}.pkl" for key in json.loads((OUT / "fit.json").read_text())["models"]}


def digests():
    return {key: entry["sha256"] for key, entry in json.loads((OUT / "fit.json").read_text())["models"].items()}


def stored(key):
    """Bytes of one model from the array file, checked against the hash of ``fit.json``."""
    with np.load(PACKED, allow_pickle=False) as z:
        data = z[key.replace(":", "-")].tobytes()
    if hashlib.sha256(data).hexdigest() != digests()[key]:
        raise SystemExit(f"the stored model {key} differs from fit.json")
    return data


def pack(_):
    arrays = {}
    for key, name in names().items():
        data = (OUT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digests()[key]:
            raise SystemExit(f"{name} differs from fit.json")
        arrays[key.replace(":", "-")] = np.frombuffer(data, dtype=np.uint8)
    np.savez(PACKED, **arrays)
    print(f"packed {len(arrays)} models")


def unpack(_):
    for key, name in names().items():
        path = OUT / name
        data = stored(key)
        if path.exists():
            if path.read_bytes() != data:
                raise SystemExit(f"{name} exists and differs from the stored model")
            continue
        with path.open("xb") as stream:
            stream.write(data)
    print(f"unpacked {len(names())} models")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pack")
    sub.add_parser("unpack")
    args = parser.parse_args()
    dict(pack=pack, unpack=unpack)[args.command](args)


if __name__ == "__main__":
    main()
