"""Inputs of the seeded type predictor that do not depend on the seed fraction (cached once).

* ``modalities(view)``: what a release offers without the cell type: soma position (FlyWire
  v3.1.0 ``soma_x/y/z``; MaleCNS v1.0 ``somaLocation``) and the per-neuron predicted
  neurotransmitter with its confidence (FlyWire ``top_nt``/``top_nt_conf``; MaleCNS
  ``predicted_nt``/``predicted_nt_confidence`` of the public per-body file, never the
  type-level or consensus columns). Curated fields (known transmitter, class, hemilineage,
  side) are not used.
* ``degree_features(view)``: the twelve degree/strength features of the frozen topology prior.
* ``raw_neighbours(view)``: label-free connectivity similarity. Every neuron is described by the
  square roots of its synapse counts to and from every neuron (as the propagation kernel of the
  paper), and the K most similar other neurons by cosine are kept.
"""

from __future__ import annotations

import csv
import json
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy import sparse

from connectomequest.neurosym import common as nc

from . import priors_common as pc

MALE_NT = pc.CACHE / "raw-malecns-nt/body-neurotransmitters-male-cns-v1.0.feather"
MALE_NT_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather"
MALE_NT_SHA256 = "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621"
NT_NAMES = ("acetylcholine", "dopamine", "gaba", "glutamate", "histamine", "octopamine", "serotonin", "tyramine")
FEATURES = dict(
    malecns="outputs/v32/predictors/_features/25433f45ef58a7c7dbefc05e63d6ec6d8ea42bb1bc3f8dea4d1ef5c3cfe283d9",
    flywire="outputs/v32/predictors/_features/f84b836f1c031fe13eab4379fcf47130c375501dfc69b1081bdd70de94ab05cb",
)
RAW_K = 200


def _nt_code(name):
    name = (name or "").strip().lower()
    return NT_NAMES.index(name) if name in NT_NAMES else -1


def _build_modalities(view):
    ids = nc.view_data(view)["ids"]
    position = {i: k for k, i in enumerate(ids.tolist())}
    soma = np.full((len(ids), 3), np.nan)
    nt = np.full(len(ids), -1, dtype=np.int32)
    conf = np.full(len(ids), np.nan)
    if view == "flywire":
        with nc.ANNOTATIONS[view].open(newline="") as stream:
            for r in csv.DictReader(stream, delimiter="\t"):
                k = position.get(int(r["root_id"]))
                if k is None:
                    continue
                if r["soma_x"] and r["soma_y"] and r["soma_z"]:
                    soma[k] = float(r["soma_x"]), float(r["soma_y"]), float(r["soma_z"])
                nt[k] = _nt_code(r["top_nt"])
                if r["top_nt_conf"]:
                    conf[k] = float(r["top_nt_conf"])
        sources = {"annotations": pc.hash_file(nc.ANNOTATIONS[view])}
    else:
        import pyarrow.ipc as ipc

        table = ipc.open_file(str(nc.ANNOTATIONS[view])).read_all().select(["bodyId", "somaLocation"]).to_pylist()
        for r in table:
            k = position.get(int(r["bodyId"]))
            if k is not None and r["somaLocation"] is not None and len(r["somaLocation"]) == 3:
                soma[k] = r["somaLocation"]
        if pc.hash_file(MALE_NT) != MALE_NT_SHA256:
            raise ValueError(f"{MALE_NT} differs from the downloaded public file ({MALE_NT_URL})")
        table = ipc.open_file(str(MALE_NT)).read_all().select(["body", "predicted_nt", "predicted_nt_confidence"]).to_pylist()
        for r in table:
            k = position.get(int(r["body"]))
            if k is not None:
                nt[k] = _nt_code(r["predicted_nt"])
                if r["predicted_nt_confidence"] is not None:
                    conf[k] = float(r["predicted_nt_confidence"])
        sources = {"annotations": pc.hash_file(nc.ANNOTATIONS[view]), "neurotransmitters": MALE_NT_SHA256}
    return dict(ids=ids, soma=soma, nt=nt, nt_conf=conf, sources=np.asarray(json.dumps(sources)))


def modalities(view):
    out = pc.cached(f"priors-modalities-{view}", lambda: _build_modalities(view))
    if not np.array_equal(out["ids"], nc.view_data(view)["ids"]):
        raise ValueError("modality cache is not aligned with the view")
    return out


VOXEL_NM = dict(flywire=(4.0, 4.0, 40.0), malecns=(8.0, 8.0, 8.0))  # soma coordinates are voxel indices


def soma_nm(view):
    """Soma position in nanometres (NaN where the release has none)."""
    return modalities(view)["soma"] * np.asarray(VOXEL_NM[view])[None]


def degree_features(view):
    root = pc.ROOT / FEATURES[view]
    with np.load(root / "features.npz", allow_pickle=False) as z:
        ids, x = z["node_ids"], z["features"]
    want = nc.view_data(view)["ids"]
    rows = np.searchsorted(ids, want)
    if np.any(rows >= len(ids)) or not np.array_equal(ids[rows], want):
        raise ValueError("degree features do not cover the view")
    return x[rows].astype(np.float64)


def connectivity_profiles(graph):
    """Rows: sqrt synapse counts to (outputs) and from (inputs) every neuron, L2-normalised."""
    g = graph.tocsr().astype(np.float64)
    g.data = np.sqrt(g.data)
    x = sparse.hstack([g, g.T.tocsr()]).tocsr()
    norm = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
    return (sparse.diags(np.where(norm > 0, 1.0 / np.where(norm > 0, norm, 1.0), 0.0)) @ x).tocsr().astype(np.float32)


def _build_raw(view, k=RAW_K, chunk=500, log=print):
    graph = nc.view_data(view)["graph"]
    x = connectivity_profiles(graph)
    xt = x.T.tocsr()
    n = x.shape[0]
    index = np.zeros((n, k), dtype=np.int32)
    sim = np.zeros((n, k), dtype=np.float32)
    started = time.time()
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        s = (x[start:stop] @ xt).toarray()
        s[np.arange(stop - start), np.arange(start, stop)] = -1.0  # never one's own neighbour
        top = np.argpartition(-s, k, axis=1)[:, :k]
        val = np.take_along_axis(s, top, 1)
        order = np.argsort(-val, axis=1, kind="stable")
        index[start:stop] = np.take_along_axis(top, order, 1)
        sim[start:stop] = np.take_along_axis(val, order, 1)
        if start // chunk % 50 == 0:
            log(f"raw neighbours {view}: {stop}/{n} ({time.time() - started:.0f} s)")
    return dict(ids=nc.view_data(view)["ids"], index=index, sim=np.maximum(sim, 0.0))


def raw_neighbours(view):
    out = pc.cached(f"priors-rawknn{RAW_K}-{view}", lambda: _build_raw(view))
    if not np.array_equal(out["ids"], nc.view_data(view)["ids"]):
        raise ValueError("neighbour cache is not aligned with the view")
    return out


def view_labels(view):
    """Reference class per neuron: hemibrain code, or 0 (outside the vocabulary) when unresolved."""
    labels = nc.view_data(view)["labels"].astype(np.int64)
    return np.where(labels < 0, 0, labels)
