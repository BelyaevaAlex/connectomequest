#!/usr/bin/env python3
"""Render the sealed V40 result for publication without changing its analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.freeze_revalidation_protocol import OUTPUT_ROOT
from scripts.render_embodied_pilot import build_summary as build_rgb_summary
from scripts.report_revalidation_benchmark import (
    _binned_medians,
    _layout_points,
    exact_binomial_interval,
    sha256_file,
    verify_stage,
)

OUTPUT = ROOT / "paper/generated/revalidation_tradeoff.pdf"


def timing_label(method: str, update_count: int) -> str:
    names = {
        "complete_validation": "Complete validation",
        "dependency_indexed": "Dependency-indexed",
    }
    noun = "change" if update_count == 1 else "changes"
    return f"{names[method]}, {update_count} {noun}"


def bar_label(method: str) -> str:
    return {
        "full_replan": "Full\nreplanning",
        "complete_validation": "Complete\nvalidation",
        "dependency_indexed": "Indexed\nvalidation",
        "unchecked_reuse": "Unchecked\nreuse",
    }[method]


def render_publication_figure(rows: list[dict], path: Path) -> None:
    rgb = build_rgb_summary()
    methods = ("complete_validation", "dependency_indexed")
    colors = {"complete_validation": "#C44E52", "dependency_indexed": "#4C72B0"}
    markers = {1: "o", 4: "s", 8: "^"}
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75), gridspec_kw={"width_ratios": [1.7, 1.0]})
    ax = axes[0]
    for count in (1, 4, 8):
        for method in methods:
            x, y = _layout_points(rows, count, method)
            centers, medians, lows, highs = _binned_medians(
                x, y, seed=40_040 + count + (0 if method == methods[0] else 100)
            )
            ax.errorbar(
                centers,
                medians,
                yerr=(medians - lows, highs - medians),
                color=colors[method],
                marker=markers[count],
                linestyle="-" if method == methods[0] else "--",
                linewidth=1.2,
                markersize=4,
                capsize=2,
                label=timing_label(method, count),
            )
    ax.set_xlabel("Remaining-plan length")
    ax.set_ylabel("Validation/replanning CPU time (ms, log scale)")
    ax.set_yscale("log")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(
        fontsize=5.5,
        ncol=2,
        loc="center left",
        bbox_to_anchor=(0.015, 0.51),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.94,
        borderpad=0.25,
        handlelength=1.8,
        columnspacing=0.8,
    )
    ax.set_title("(a) Irrelevant receipt withdrawals", loc="left", fontsize=9)

    ax = axes[1]
    data_keys = (
        "full_replan",
        "full_scan",
        "receipt_index",
        "unchecked_reuse",
    )
    public_keys = (
        "full_replan",
        "complete_validation",
        "dependency_indexed",
        "unchecked_reuse",
    )
    names = tuple(bar_label(key) for key in public_keys)
    n = int(rgb["safety"][data_keys[0]]["n"])
    counts = np.asarray([int(rgb["safety"][key]["unsupported_episodes"]) for key in data_keys])
    rates = counts / n
    intervals = [exact_binomial_interval(int(value), n) for value in counts]
    lower = rates - np.asarray([value[0] for value in intervals])
    upper = np.asarray([value[1] for value in intervals]) - rates
    bars = ax.bar(
        range(len(data_keys)),
        rates,
        color=["#55A868", "#55A868", "#4C72B0", "#C44E52"],
        width=0.68,
    )
    ax.errorbar(
        range(len(data_keys)),
        rates,
        yerr=(lower, upper),
        fmt="none",
        color="black",
        capsize=2,
        linewidth=0.8,
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.04, bar.get_height() + 0.035),
            f"{count}/{n}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax.set_xticks(range(len(data_keys)), names, fontsize=6.0)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Unsupported-authorization rate")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.set_title("(b) Final-support loss", loc="left", fontsize=9)

    fig.tight_layout(pad=0.5, w_pad=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={"Creator": "anonymous artifact", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    stage = OUTPUT_ROOT / "confirmation"
    protocol, rows = verify_stage(stage, current_sources=True)
    sealed = json.loads((stage / "sealed_report.json").read_text())
    if sealed["protocol_sha256"] != sha256_file(stage / "protocol.json"):
        raise AssertionError("sealed report is not bound to the confirmation protocol")
    if not sealed["gates"]["correctness"]["passed"] or not sealed["gates"]["efficiency"]["passed"]:
        raise AssertionError("publication renderer requires passed frozen gates")

    if args.check:
        temporary = stage / ".selective-validation-publication-check.pdf"
        render_publication_figure(rows, temporary)
        if sha256_file(temporary) != sha256_file(OUTPUT):
            raise AssertionError("stale publication figure")
        temporary.unlink()
    else:
        render_publication_figure(rows, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
