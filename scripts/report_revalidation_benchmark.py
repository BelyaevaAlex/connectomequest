#!/usr/bin/env python3
"""Independently verify, aggregate, and render the amortized CPU experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

from connectomequest.embodied.protocol import (
    atomic_case_write,
    sha256_file,
    verify_archived_source_hashes,
)
from connectomequest.embodied.revalidation_benchmark import TimingRow
from connectomequest.embodied.revalidation_statistics import (
    evaluate_v40_gates,
    summarize_fully_accounted,
)
from connectomequest.embodied.types import canonical_json_bytes
from scripts.freeze_revalidation_protocol import OUTPUT_ROOT
from scripts.render_embodied_pilot import build_summary as build_rgb_summary

PAPER_GENERATED = ROOT / "paper/generated"


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def exact_binomial_interval(
    successes: int, trials: int, *, alpha: float = 0.05
) -> tuple[float, float]:
    """Two-sided Clopper--Pearson interval."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    high = (
        1.0
        if successes == trials
        else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    )
    return low, high


def load_verified_rows(
    stage: Path, *, validate_timing_rows: bool = True
) -> tuple[dict, list[dict]]:
    """Load only case shards bound to the frozen protocol and payload digests."""

    protocol_path = stage / "protocol.json"
    if not protocol_path.is_file():
        raise ValueError(f"missing protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text())
    protocol_digest = sha256_file(protocol_path)
    rows: list[dict] = []
    seen: set[tuple[int, int, int, str]] = set()
    for path in sorted((stage / "cases").glob("*.json")):
        wrapper = json.loads(path.read_text())
        if wrapper.get("protocol_sha256") != protocol_digest:
            raise ValueError(f"protocol digest mismatch: {path}")
        result = wrapper.get("result")
        if wrapper.get("payload_sha256") != _digest(result):
            raise ValueError(f"payload digest mismatch: {path}")
        if validate_timing_rows:
            typed_result = {
                **result,
                "raw_repetitions_ns": tuple(result["raw_repetitions_ns"]),
            }
            TimingRow(**typed_result).validate()
            key = (
                int(result["layout_seed"]),
                int(result["update_seed"]),
                int(result["update_count"]),
                str(result["method"]),
            )
            if key in seen:
                raise ValueError(f"duplicate timing cell: {key}")
            seen.add(key)
        rows.append(result)
    if not rows:
        raise ValueError("no case shards")
    return protocol, rows


def verify_stage(stage: Path, *, current_sources: bool) -> tuple[dict, list[dict]]:
    protocol, rows = load_verified_rows(stage)
    if current_sources:
        verify_archived_source_hashes(ROOT / "paper/artifact.zip", protocol["source_hashes"])
    verification_path = stage / "verification.json"
    if not verification_path.is_file():
        raise ValueError("missing runner verification")
    verification = json.loads(verification_path.read_text())
    if verification.get("protocol_sha256") != sha256_file(stage / "protocol.json"):
        raise ValueError("runner verification is not bound to the protocol")
    expected_stage = protocol["protocol"]["stage"]
    if verification.get("stage") != expected_stage:
        raise ValueError("stage mismatch")
    excluded = {int(row["layout_seed"]) for row in verification.get("excluded", [])}
    observed = {int(row["layout_seed"]) for row in rows}
    declared = {int(seed) for seed in protocol["protocol"]["layout_seeds"]}
    if observed & excluded or observed | excluded != declared:
        raise ValueError("layout accounting is incomplete")
    update_seeds = set(map(int, protocol["protocol"]["update_seeds"]))
    update_counts = set(map(int, protocol["protocol"]["update_counts"]))
    methods = set(protocol["protocol"]["methods"])
    expected_n = len(observed) * len(update_seeds) * len(update_counts) * len(methods)
    if len(rows) != expected_n:
        raise ValueError(f"incomplete cell grid: {len(rows)} != {expected_n}")
    return protocol, rows


def _layout_points(
    rows: Iterable[dict], update_count: int, method: str
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        if int(row["update_count"]) == update_count and row["method"] == method:
            grouped.setdefault(int(row["layout_seed"]), []).append(row)
    xs, ys = [], []
    for layout in sorted(grouped):
        group = grouped[layout]
        xs.append(float(np.median([row["plan_length"] for row in group])))
        ys.append(float(np.median([row["total_reasoning_ns"] for row in group])) / 1e6)
    return np.asarray(xs), np.asarray(ys)


def _binned_medians(
    x: np.ndarray, y: np.ndarray, *, seed: int, bins: int = 4, draws: int = 2000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    centers, medians, lows, highs = [], [], [], []
    rng = np.random.default_rng(seed)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (x >= left) & ((x <= right) if index == len(edges) - 2 else (x < right))
        bx, by = x[mask], y[mask]
        if len(by) < 2:
            continue
        boot = np.median(by[rng.integers(0, len(by), size=(draws, len(by)))], axis=1)
        centers.append(float(np.median(bx)))
        medians.append(float(np.median(by)))
        low, high = np.quantile(boot, (0.025, 0.975))
        lows.append(float(low))
        highs.append(float(high))
    return map(np.asarray, (centers, medians, lows, highs))


def render_figure(rows: list[dict], path: Path) -> None:
    rgb = build_rgb_summary()
    methods = ("complete_validation", "dependency_indexed")
    colors = {"complete_validation": "#C44E52", "dependency_indexed": "#4C72B0"}
    labels = {
        "complete_validation": "Complete validation",
        "dependency_indexed": "Dependency-indexed",
    }
    markers = {1: "o", 4: "s", 8: "^"}
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75), gridspec_kw={"width_ratios": [1.7, 1.0]})
    ax = axes[0]
    for count in (1, 4, 8):
        for method in methods:
            x, y = _layout_points(rows, count, method)
            centers, medians, lows, highs = _binned_medians(
                x, y, seed=40_040 + count + (0 if method == methods[0] else 100)
            )
            label = f"{labels[method]}, {count} update{'s' if count != 1 else ''}"
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
                label=label,
            )
    ax.set_xlabel("Remaining-plan length")
    ax.set_ylabel("Total reasoning time (ms, log scale)")
    ax.set_yscale("log")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=5.7, ncol=2, loc="upper left", frameon=False)
    ax.set_title("(a) Irrelevant receipt withdrawals", loc="left", fontsize=9)

    ax = axes[1]
    names = ("Full\nreplanning", "Complete\nvalidation", "Dependency-\nindexed", "Unchecked\nreuse")
    keys = ("full_replan", "full_scan", "receipt_index", "unchecked_reuse")
    n = int(rgb["safety"][keys[0]]["n"])
    counts = np.asarray([int(rgb["safety"][key]["unsupported_episodes"]) for key in keys])
    rates = counts / n
    intervals = [exact_binomial_interval(int(value), n) for value in counts]
    lower = rates - np.asarray([value[0] for value in intervals])
    upper = np.asarray([value[1] for value in intervals]) - rates
    bars = ax.bar(
        range(len(keys)), rates, color=["#55A868", "#55A868", "#4C72B0", "#C44E52"], width=0.72
    )
    ax.errorbar(
        range(len(keys)),
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
    ax.set_xticks(range(len(keys)), names, fontsize=6.2)
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


def render_numbers(report: dict) -> str:
    low, high = report["time_reduction_ci"]
    return "\n".join(
        [
            rf"\newcommand{{\AmortizedEligible}}{{{report['eligible_long']}}}",
            rf"\newcommand{{\AmortizedPairedCells}}{{{report['paired_cells']}}}",
            rf"\newcommand{{\AmortizedAgreement}}{{{100 * report['decision_agreement']:.1f}\%}}",
            rf"\newcommand{{\AmortizedReduction}}{{{100 * report['time_reduction']:.1f}\%}}",
            rf"\newcommand{{\AmortizedReductionLow}}{{{100 * low:.1f}\%}}",
            rf"\newcommand{{\AmortizedReductionHigh}}{{{100 * high:.1f}\%}}",
            rf"\newcommand{{\AmortizedWorkReduction}}{{{100 * report['work_reduction']:.1f}\%}}",
            "",
        ]
    )


def render_figure_tex(report: dict) -> str:
    return "\n".join(
        [
            r"\begin{figure*}[t]",
            r"\centering",
            r"\includegraphics[width=\textwidth]{generated/selective_validation_v40.pdf}",
            rf"\caption{{Two distinct interventions under matched RGB access. (a) Under irrelevant receipt withdrawals, dependency indexing and complete retained-plan validation make identical decisions on {report['eligible_long']} eligible long-plan layouts, while the index avoids work unrelated to the retained plan. Points show binned medians; bars are layout-bootstrap 95\% intervals. (b) Under final-support loss, every checking method prevents unsupported authorization, whereas unchecked reuse does not. Binomial bars are exact 95\% intervals.}}",
            r"\label{fig:selective-validation}",
            r"\end{figure*}",
            "",
        ]
    )


def _write_or_check(path: Path, data: bytes, *, check: bool) -> None:
    if check:
        if not path.is_file() or path.read_bytes() != data:
            raise AssertionError(f"stale generated artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "confirmation"), required=True)
    parser.add_argument("--check-integrity", action="store_true")
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stage = OUTPUT_ROOT / args.stage
    protocol, rows = verify_stage(stage, current_sources=False)
    report = summarize_fully_accounted(
        rows,
        seed=int(protocol["protocol"]["bootstrap_seed"]),
        draws=int(protocol["protocol"]["bootstrap_draws"]),
    )
    report["gates"] = evaluate_v40_gates(
        reduction=report["time_reduction"],
        lower=report["time_reduction_ci"][0],
        decision_agreement=report["decision_agreement"],
        work_reduction=report["work_reduction"],
        eligible_long=report["eligible_long"],
    )
    report["protocol_sha256"] = sha256_file(stage / "protocol.json")
    report["analysis_sha256"] = sha256_file(Path(__file__))
    report["primary_population"] = (
        "remaining-plan length at least 64 and 4 or 8 distinct irrelevant receipt withdrawals"
    )
    encoded = canonical_json_bytes(report) + b"\n"
    if args.seal:
        atomic_case_write(stage / "sealed_report.json", report)
    elif not args.check_integrity:
        _write_or_check(stage / "report.json", encoded, check=args.check)
    if args.stage == "confirmation" and not args.check_integrity:
        if not report["gates"]["correctness"]["passed"]:
            raise SystemExit("correctness gate failed; refusing publication artifacts")
        numbers = render_numbers(report).encode()
        _write_or_check(PAPER_GENERATED / "revalidation_numbers.tex", numbers, check=args.check)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
