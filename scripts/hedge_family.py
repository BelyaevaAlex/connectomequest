#!/usr/bin/env python3
"""Questions on which Adaptive Greedy with a perfect prior checks about ln Z times the cells that suffice (Proposition 1).

For integers k >= 2 and n >= 1 the question has k members of weight 1 with threshold T = k and n nonmembers whose
weights are defined from the last to the first: with b = 1 synapse still missing for a "no" after the last nonmember,

    w = floor(b / (k - 1)) + 1,    b <- b + w .

The nonmembers carry Z - 1 synapses, so the only certificate is "yes" through the k members. A perfect prior puts
probability one on the outcome of every cell. Adaptive Greedy then checks every nonmember before the first member,
k + n cells where k suffice, and Z <= k (k / (k - 1))^n, so n >= (k - 1) ln(Z / k).

The script runs the policies of the engine on members of the family and writes a table of these runs
(paper/tables/hedge.tex) and the numbers of the text; the build stops if a run contradicts the proposition.

Usage: python scripts/hedge_family.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from connectomequest.annotation_v35 import engine as e  # noqa: E402

OUT = ROOT / "outputs/hedge"
PAPER = ROOT / "paper"
GRID = ((2, 5), (2, 10), (2, 20), (2, 40), (3, 20), (5, 20), (10, 20), (10, 100))
EXAMPLE = (2, 20)


def nonmember_weights(k, n):
    """Weights of the n nonmembers, heaviest first."""
    if k < 2 or n < 1:
        raise ValueError("the family needs k >= 2 and n >= 1")
    b, weights = 1, []
    for _ in range(n):
        w = b // (k - 1) + 1
        weights.append(w)
        b += w
    return weights[::-1]


def question(k, n):
    """Instance, outcomes and perfect prior: cells 1..k are the members, the others the nonmembers by decreasing weight."""
    weights = np.array([1] * k + nonmember_weights(k, n), dtype=np.int64)
    outcomes = np.array([e.MEMBER] * k + [e.NONMEMBER] * n)
    instance = e.threshold_instance(np.arange(1, k + n + 1), weights, k)
    prior = np.zeros((k + n, 3))
    prior[np.arange(k + n), outcomes] = 1.0
    return instance, outcomes, prior


def checks(k, n):
    """Cells checked by each policy with the perfect prior, and the quantities of the proposition."""
    instance, outcomes, prior = question(k, n)
    cost = lambda order: e.evaluate(instance, outcomes, order)
    adaptive = lambda policy: cost(e.run_adaptive(instance, outcomes, policy(instance, prior)))
    out = dict(k=k, n=n, cells=k + n, Z=int(instance.b0), heaviest=int(instance.weights.max()),
               optimum=e.clairvoyant_cost(instance, outcomes), bound=k + (k - 1) * math.log(instance.b0 / k))
    for name, (count, status) in (("ag", adaptive(e.AdaptiveGreedy)), ("recalibrated", adaptive(e.RecalibratedGreedy)),
                                  ("direction", cost(e.direction_order(instance, prior))), ("weight", cost(e.weight_order(instance)))):
        if status != "positive":
            raise SystemExit(f"k={k}, n={n}: {name} does not certify yes")
        out[name] = count
    return out


def verify(row):
    k, n = row["k"], row["n"]
    claims = {
        "the nonmembers carry Z - 1 synapses": sum(nonmember_weights(k, n)) == row["Z"] - 1,
        "k checks suffice": row["optimum"] == k,
        "direction-fixed makes k checks": row["direction"] == k,
        "AG checks every cell": row["ag"] == k + n,
        "Z <= k (k / (k - 1))^n": row["Z"] <= k * (k / (k - 1)) ** n * (1 + 1e-12),
        "AG makes at least k + (k - 1) ln(Z / k) checks": row["ag"] >= row["bound"],
    }
    failed = [claim for claim, ok in claims.items() if not ok]
    if failed:
        raise SystemExit(f"k={k}, n={n}: " + "; ".join(failed))


def main():
    rows = [checks(k, n) for k, n in GRID]
    for row in rows:
        verify(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "family.json").write_text(json.dumps(rows, indent=1) + "\n")
    number = lambda x: f"{x:,}".replace(",", "{,}")
    power = lambda z: f"$2^{{{int(math.log2(z + 1))}}}-1$" if z + 1 == 2 ** int(math.log2(z + 1)) and z > 10 ** 6 else (
        number(z) if z < 10 ** 7 else f"${z / 10 ** int(math.log10(z)):.1f}\\cdot10^{{{int(math.log10(z))}}}$")
    lines = [f"{r['k']} & {r['n']} & {power(r['Z'])} & {r['optimum']} & {r['direction']} & {r['ag']} & {r['recalibrated']} & {r['weight']} & "
             f"{r['ag'] / r['optimum']:.1f} & {1 + (1 - 1 / r['k']) * math.log(r['Z'] / r['k']):.1f}\\\\" for r in rows]
    (PAPER / "tables/hedge.tex").write_text("\n".join(lines) + "\n")
    x = next(r for r in rows if (r["k"], r["n"]) == EXAMPLE)
    if x["recalibrated"] != x["ag"]:
        raise SystemExit("the text says that recalibration does not shorten the example")
    words = {2: "two", 3: "three"}
    macros = dict(HedgeExampleK=words[x["k"]], HedgeExampleN=x["n"], HedgeExampleAG=x["ag"], HedgeExampleOptimum=words[x["optimum"]],
                  HedgeExampleZPower=int(math.log2(x["Z"] + 1)), HedgeExampleRatio=f"{x['ag'] / x['optimum']:.0f}")
    (PAPER / "hedge_numbers.tex").write_text("% Generated by scripts/hedge_family.py. Do not edit.\n" + "".join(
        f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in macros.items()))
    for r in rows:
        print({key: (round(v, 2) if isinstance(v, float) else v) for key, v in r.items()})


if __name__ == "__main__":
    main()
