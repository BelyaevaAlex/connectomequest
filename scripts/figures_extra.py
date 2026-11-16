#!/usr/bin/env python3
"""Six explanatory figures of the main text, drawn from results that already exist.

    figures/pipeline.pdf   the certification loop, the bounds and one worked question
    figures/hedge.pdf      the family of Proposition 1: one question, and checks against Z
    figures/releases.pdf   what changes between two annotation releases, and how rarely a certificate breaks
    figures/missing.pdf    uncertifiable questions certified as correspondences are declared
    figures/coverage_rule.pdf  cells checked against coverage of the earlier release (outputs/newtypes/analysis-heldout.json)
    figures/choice.pdf     the rule of the Discussion as a decision diagram

No policy is run on held-out questions here. The worked question is the development example of the text
(v35_paper_assets.example_orders), the points of the second figure are read from outputs/hedge/family.json and its
lines follow the recurrence of the proposition, the third figure reads the recorded analysis
outputs/labelings/analysis-test.json, and the fourth the complete lists outputs/labelings/residual-{view}-test.csv,
which are checked here against the recorded analysis.

The script also asserts one statement that the main text makes in words: share questions are mostly answered "no"
in both replications.

Usage: python scripts/figures_extra.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import v35_paper_assets as assets  # noqa: E402  (plot style, colours and the worked question)
from hedge_family import nonmember_weights  # noqa: E402

plt = assets.plt
Rectangle = assets.Rectangle
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BLUE, ORANGE, AQUA = assets.BLUE, assets.ORANGE, assets.AQUA
INK, MUTED, GRID = assets.INK, assets.MUTED, assets.GRID
PAPER = ROOT / "paper"
SOFT = "#efeee9"
plt.rcParams.update({"mathtext.fontset": "stix"})  # mathematics in the typeface of the text
VIEW = (("malecns", "MaleCNS", BLUE, "o"), ("flywire", "FlyWire", ORANGE, "s"))


def box(ax, x, y, w, h, head, body):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", facecolor="white", edgecolor=MUTED, linewidth=0.6,
                                mutation_aspect=ax.bbox.width / ax.bbox.height))
    ax.text(x + 0.009, y + h - 0.06, head, ha="left", va="top", fontsize=7.2, color=INK, fontweight="bold")
    ax.text(x + 0.009, y + h - 0.27, body, ha="left", va="top", fontsize=6.0, color=INK, linespacing=1.2)


def arrow(ax, start, end, style="-|>", rad=0.0, color=INK):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=7, linewidth=0.7, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def figure_pipeline():
    curves, tau, size, _ = assets.example_orders()
    fig = plt.figure(figsize=(7.0, 2.66))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.36], width_ratios=[1.0, 1.22], hspace=0.22, wspace=0.20)

    ax = fig.add_subplot(grid[0, :])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    width, gap, low, height = 0.184, 0.020, 0.17, 0.80
    steps = (
        ("1  Statement", "“Type $A$ supplies at\nleast 20% of the input\nof neuron $n$.”"),
        ("2  Question", "Do the input cells of\ntype $A$ carry at least\n$T$ of the $W$ synapses?"),
        ("3  Policy", "picks the next cell:\nby synapse count, or\nguided by a prior"),
        ("4  Check", "An expert confirms\nits type: member, non-\nmember or unresolved."),
        ("5  Bounds", "$L\\geq T$: certified yes\n$U<T$: certified no\nelse: next check"),
    )
    left = [i * (width + gap) for i in range(len(steps))]
    for x, (head, body) in zip(left, steps):
        box(ax, x, low, width, height, head, body)
    for x in left[:-1]:
        arrow(ax, (x + width + 0.003, low + height / 2), (x + width + gap - 0.003, low + height / 2))
    back_from, back_to = left[4] + width / 2, left[2] + width / 2
    ax.plot([back_from, back_from, back_to], [low, 0.04, 0.04], color=INK, linewidth=0.7, solid_capstyle="butt")
    arrow(ax, (back_to, 0.04), (back_to, low - 0.01))
    ax.text((back_from + back_to) / 2, 0.075, "no certificate yet", ha="center", va="bottom", fontsize=6.4, color=INK)
    ax.text(0.0, 0.04, "(a) from a statement to a certificate", ha="left", va="center", fontsize=7.2, color=INK)

    ax = fig.add_subplot(grid[1, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y, h = 0.36, 0.17
    segments = ((0.0, 0.28, BLUE, None), (0.28, 0.46, "white", "////"), (0.46, 0.72, SOFT, None), (0.72, 1.0, MUTED, None))
    for x0, x1, color, hatch in segments:
        ax.add_patch(Rectangle((x0, y), x1 - x0 - 0.004, h, facecolor=color, edgecolor=MUTED if hatch or color == SOFT else "none",
                               hatch=hatch, linewidth=0.5))
    for x, label in ((0.14, "checked:\nmember"), (0.37, "checked:\nunresolved"), (0.59, "not yet\nchecked"), (0.86, "checked:\nnonmember")):
        ax.text(x, y - 0.04, label, ha="center", va="top", fontsize=6.5, color=INK, linespacing=0.95)
    ax.plot([0.53, 0.53], [y - 0.02, y + h + 0.12], color=ORANGE, linewidth=1.2)
    ax.text(0.55, y + h + 0.09, "threshold $T$", ha="left", va="center", fontsize=6.8, color=INK)
    ax.annotate("", xy=(0.28, y + h + 0.05), xytext=(0.0, y + h + 0.05), arrowprops=dict(arrowstyle="<->", color=INK, linewidth=0.6))
    ax.text(0.14, y + h + 0.065, "$L$", ha="center", va="bottom", fontsize=7.5)
    ax.annotate("", xy=(0.72, y + h + 0.22), xytext=(0.0, y + h + 0.22), arrowprops=dict(arrowstyle="<->", color=INK, linewidth=0.6))
    ax.text(0.36, y + h + 0.235, "$U$", ha="center", va="bottom", fontsize=7.5)
    ax.text(0.0, 1.0, "(b) the $W$ input synapses after some checks", ha="left", va="top", fontsize=7.2, color=INK)
    ax.text(0.0, 0.0, "here $L<T\\leq U$: the answer is still open", ha="left", va="bottom", fontsize=6.6, color=INK)

    ax = fig.add_subplot(grid[1, 1])
    styles = {"weight order": (MUTED, "weight order"), "earlier release, member-first": (ORANGE, "earlier release")}
    for name, (color, label) in styles.items():
        c = curves[name]
        x = np.arange(len(c["lower"]))
        ax.plot(x, c["lower"], color=color, linewidth=1.3, label=f"{label} ({c['cost']})")
        ax.plot([c["cost"]], [c["lower"][-1]], marker="o", markersize=3.5, color=color, markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(tau, color=INK, linewidth=0.6, linestyle=(0, (4, 3)))
    ax.text(3, tau - 0.006, "threshold $T/W$", ha="left", va="top", fontsize=6.5, color=MUTED)
    ax.set_xlabel("cells checked", labelpad=1.5)
    ax.set_ylabel("lower bound $L/W$")
    ax.set_xlim(0, curves["weight order"]["cost"] * 1.03)
    ax.set_ylim(0, tau * 1.08)
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.5, borderaxespad=0.2, title="order of checks (checks needed)", title_fontsize=6.5,
              frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    ax.set_title(f"(c) one large question: {size} input cells, $T$ = {100 * tau:.0f}% of $W$", fontsize=7.2, loc="left", pad=3)
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    assets.clean(ax)
    fig.savefig(PAPER / "figures/pipeline.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def family_curve(k, n_max):
    """Z and the checks of Adaptive Greedy, k + n, for n = 1..n_max (Proposition 1)."""
    z, cells = [], []
    for n in range(1, n_max + 1):
        z.append(sum(nonmember_weights(k, n)) + 1)
        cells.append(k + n)
    return np.array(z, dtype=float), np.array(cells)


def figure_hedge():
    rows = json.loads((ROOT / "outputs/hedge/family.json").read_text())
    fig = plt.figure(figsize=(3.3, 1.66))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.86, 1.0], wspace=0.50)

    k, n = 2, 5
    weights = nonmember_weights(k, n) + [1] * k
    row = next(r for r in rows if (r["k"], r["n"]) == (k, n))
    if row["ag"] != k + n or row["direction"] != k or row["Z"] != sum(weights[:n]) + 1:
        raise SystemExit("the drawn question contradicts outputs/hedge/family.json")
    ax = fig.add_subplot(grid[0])
    x = np.arange(k + n)
    ax.bar(x[:n], np.log2(weights[:n]) + 1, width=0.72, color=MUTED, linewidth=0)
    ax.bar(x[n:], np.log2(weights[n:]) + 1, width=0.72, color=BLUE, linewidth=0)
    for i, w in enumerate(weights):
        ax.text(i, np.log2(w) + 1.12, str(w), ha="center", va="bottom", fontsize=6.3, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(k + n)], fontsize=6.3)
    ax.tick_params(axis="x", length=0, pad=1.5)
    for i in range(n, n + k):
        ax.text(i, -1.42, str(i - n + 1), ha="center", va="top", fontsize=6.3, color=ORANGE, fontweight="bold")
    ax.text(-0.85, -0.36, "AG", ha="right", va="top", fontsize=6.3, color=INK, fontweight="bold")
    ax.text(-0.85, -1.42, "DF", ha="right", va="top", fontsize=6.3, color=ORANGE, fontweight="bold")
    for label in ax.get_xticklabels():
        label.set_color(INK)
        label.set_fontweight("bold")
    ax.set_yticks([])
    ax.set_ylim(0, 7.4)
    ax.set_xlim(-0.6, k + n - 0.4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.text(1.9, 6.9, "nonmembers", ha="left", va="center", fontsize=6.3, color=MUTED)
    ax.text(1.9, 6.0, "members", ha="left", va="center", fontsize=6.3, color=BLUE)
    ax.set_title(f"(a) $T$ = {k}, $Z$ = {row['Z']}", fontsize=7.0, loc="left", pad=3, x=-0.2)
    ax.text(0.42, -0.36, "order of checks", transform=ax.transAxes, ha="center", va="top", fontsize=6.6, color=INK)

    ax = fig.add_subplot(grid[1])
    for k, n_max, style in ((2, 43, "-"), (10, 100, (0, (4, 2)))):
        z, cells = family_curve(k, n_max)
        ax.plot(z, cells, color=BLUE, linewidth=1.2, linestyle=style)
        ax.plot(z, np.full_like(z, k), color=ORANGE, linewidth=1.2, linestyle=style)
    for r in rows:
        if r["k"] in (2, 10):
            ax.plot([r["Z"]], [r["ag"]], marker="o", markersize=3.0, color=BLUE, markeredgecolor="white", markeredgewidth=0.5)
            ax.plot([r["Z"]], [r["direction"]], marker="s", markersize=2.8, color=ORANGE, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlim(8, 4e12)
    ax.set_ylim(0, 118)
    ax.set_xticks([1e2, 1e5, 1e8, 1e11])
    ax.set_xlabel("$Z$ (synapses for a no)", labelpad=1.5)
    ax.set_ylabel("cells checked", labelpad=1.5)
    ax.text(1.5e6, 104, "AG, $k$ = 10", ha="left", va="center", fontsize=6.3, color=BLUE)
    ax.text(3e7, 40, "AG, $k$ = 2", ha="right", va="center", fontsize=6.3, color=BLUE)
    ax.text(3e12, 4.5, "DF: $k$ checks", ha="right", va="bottom", fontsize=6.3, color=ORANGE)
    ax.set_title("(b) perfect prior", fontsize=7.0, loc="left", pad=3)
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    assets.clean(ax)
    fig.savefig(PAPER / "figures/hedge.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def residual_curve(view):
    """Share of the questions that all listed types certify together, after the first r types of the ranked list."""
    with open(ROOT / f"outputs/labelings/residual-{view}-test.csv", newline="") as handle:
        ranked = [int(row["certified_by_list_so_far"]) for row in csv.DictReader(handle) if row["rank"]]
    return np.arange(1, len(ranked) + 1), np.array(ranked) / ranked[-1]


def figure_missing():
    analysis = json.loads((ROOT / "outputs/labelings/analysis-test.json").read_text())
    fig, ax = plt.subplots(figsize=(3.3, 1.72))
    hundred = []
    for view, name, color, marker in VIEW:
        rank, share = residual_curve(view)
        recorded = analysis[view]["residual"]
        half = int(np.flatnonzero(share >= 0.5)[0]) + 1
        if abs(share[99] - recorded["first"]["100"]["value"]) > 1e-9 or half != recorded["types_for"]["0.50"] or len(rank) != recorded["length"]:
            raise SystemExit(f"{view}: the complete list contradicts the recorded analysis")
        ax.plot(rank, 100 * share, color=color, linewidth=1.3)
        ax.plot([100, half], [100 * share[99], 100 * share[half - 1]], linestyle="none", marker=marker, markersize=3.6, color=color,
                markeredgecolor="white", markeredgewidth=0.6)
        hundred.append(100 * share[99])
        if view == "malecns":
            ax.text(half * 1.15, 100 * share[half - 1] - 4, f"{name}: half\nneeds {half} types", ha="left", va="top", fontsize=6.4, color=color,
                    linespacing=0.95)
        else:
            ax.text(half * 0.82, 100 * share[half - 1] + 5, f"{name}: half\nneeds {half} types", ha="right", va="bottom", fontsize=6.4, color=color,
                    linespacing=0.95)
    ax.text(80, 30, f"first hundred types:\n{min(hundred):.0f}–{max(hundred):.0f}%", ha="right", va="bottom", fontsize=6.4, color=INK, linespacing=0.95)
    ax.set_xscale("log")
    ax.set_xlim(1, 3200)
    ax.set_ylim(0, 100)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.set_xlabel("release-specific types given a correspondence", labelpad=1.5)
    ax.set_ylabel("reachable questions\ncertified (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    assets.clean(ax)
    fig.savefig(PAPER / "figures/missing.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def figure_releases():
    """Shares of held-out questions (first row) and of answers certified under the earlier release (other rows)."""
    analysis = json.loads((ROOT / "outputs/labelings/analysis-test.json").read_text())
    rows = (
        ("questions with an input cell\nwhose outcome changed", lambda e: e["questions_with_other_outcome"]),
        ("smallest certificates\nno longer valid", lambda e: {k: 1 - e["certificate_valid"][j] for k, j in (("value", "value"), ("low", "high"), ("high", "low"))}),
        ("certified answers\nnot certified again", lambda e: {k: 1 - e["kept"][j] for k, j in (("value", "value"), ("low", "high"), ("high", "low"))}),
    )
    floor = 0.004
    fig, ax = plt.subplots(figsize=(3.3, 1.18))
    for y, (label, pick) in enumerate(rows):
        for (view, name, color, marker), shift in zip(VIEW, (0.14, -0.14)):
            e = analysis[view]["earlier"]
            if abs((1 - e["kept"]["value"]) - (e["lost"]["value"] + e["reversed"]["value"])) > 1e-12:
                raise SystemExit(f"{view}: answers not certified again are not the lost and the reversed answers")
            q = {k: 100 * v for k, v in pick(e).items()}
            at = len(rows) - 1 - y + shift
            ax.plot([max(q["low"], floor), q["high"]], [at, at], color=color, linewidth=1.0, solid_capstyle="butt")
            ax.plot([q["value"]], [at], marker=marker, markersize=3.8, color=color, markeredgecolor="white", markeredgewidth=0.6, linestyle="none",
                    label=name if y == 0 else None)
            ax.text(q["high"] * 1.25, at, f"{q['value']:.2g}%" if q["value"] < 10 else f"{q['value']:.0f}%", ha="left", va="center", fontsize=6.2, color=color)
    ax.set_xscale("log")
    ax.set_xlim(floor, 400)
    ax.set_xticks([0.01, 0.1, 1, 10, 100])
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows][::-1], fontsize=6.5, linespacing=0.95)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlabel("share (%, logarithmic scale)", labelpad=1.5)
    ax.grid(axis="x", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=6.4, handlelength=0.8, borderaxespad=0.1, handletextpad=0.3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.savefig(PAPER / "figures/releases.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def figure_coverage():
    """Cells checked relative to weight order as the coverage of the earlier release grows (held-out types, retention questions).

    Coverage 0 is the topology prior alone, coverage 1 the earlier-release prior on every neuron; the dotted line is
    outcome propagation without a predictor, which does not depend on coverage. All values are read from the
    recorded analysis outputs/newtypes/analysis-heldout.json.
    """
    analysis = json.loads((ROOT / "outputs/newtypes/analysis-heldout.json").read_text())
    levels = (0.0, 0.25, 0.5, 0.75, 1.0)

    def arm(level, policy):
        return "mlp_s0:" + policy if level == 0.0 else "provisional:" + policy if level == 1.0 else f"cov{level}:{policy}"

    policies = (("direction", "direction-fixed", ORANGE, "s"), ("recal", "recalibrated AG", BLUE, "o"))
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.45), sharey=True)
    for ax, (view, name, _, _) in zip(axes, VIEW):
        ratios = analysis[view]["retention"]["ratio"]
        ax.axhline(1.0, color=MUTED, linewidth=0.7)
        ax.text(1.02, 1.0, "weight\norder", fontsize=5.6, color=MUTED, va="center", ha="left", transform=ax.get_yaxis_transform(), linespacing=0.9) if view == "flywire" else None
        prop = ratios["none:prop"]["all"]
        ax.axhspan(prop["low"], prop["high"], color=AQUA, alpha=0.18, linewidth=0)
        ax.axhline(prop["value"], color=AQUA, linewidth=0.9, linestyle=":")
        for key, label, color, marker in policies:
            e = [ratios[arm(level, key)]["all"] for level in levels]
            ax.fill_between(levels, [x["low"] for x in e], [x["high"] for x in e], color=color, alpha=0.15, linewidth=0)
            ax.plot(levels, [x["value"] for x in e], color=color, marker=marker, markersize=3.0, linewidth=1.0,
                    markeredgecolor="white", markeredgewidth=0.4, label=label)
        ax.set_title(name, fontsize=7, pad=2)
        ax.set_xticks(levels)
        ax.set_xticklabels(["0", "¼", "½", "¾", "1"])
        ax.set_xlabel("coverage", labelpad=1)
        ax.grid(axis="y", color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("cells checked\n/ weight order", labelpad=2, linespacing=0.9, fontsize=6.5)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=AQUA, linestyle=":", linewidth=0.9))
    labels.append("propagation, no predictor")
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=5.8, frameon=False, handlelength=1.4,
               columnspacing=0.8, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(left=0.17, right=0.9, top=0.88, bottom=0.36, wspace=0.12)
    fig.savefig(PAPER / "figures/coverage_rule.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def figure_choice():
    """Which policy to use: the rule of the Discussion as a decision diagram."""
    fig, ax = plt.subplots(figsize=(3.3, 1.12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def node(x, y, w, h, text, fill="white", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02", facecolor=fill, edgecolor=MUTED,
                                    linewidth=0.6, mutation_aspect=ax.bbox.width / ax.bbox.height))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.1, color=INK, linespacing=1.05,
                fontweight="bold" if bold else "normal")

    node(0.0, 0.62, 0.29, 0.3, "earlier labels\nfor the connectome?")
    node(0.37, 0.62, 0.29, 0.3, "does it cover nearly\nevery neuron?")
    node(0.0, 0.05, 0.29, 0.36, "weight order with\noutcome propagation", fill=SOFT)
    node(0.37, 0.05, 0.29, 0.36, "recalibrated AG;\nat half or less:\npropagation", fill=SOFT)
    node(0.73, 0.62, 0.27, 0.3, "direction-fixed\nor recalibrated AG", fill=SOFT)
    arrow(ax, (0.29, 0.77), (0.37, 0.77))
    ax.text(0.33, 0.8, "yes", fontsize=5.8, ha="center", va="bottom", color=MUTED)
    arrow(ax, (0.145, 0.62), (0.145, 0.41))
    ax.text(0.155, 0.515, "no", fontsize=5.8, ha="left", va="center", color=MUTED)
    arrow(ax, (0.66, 0.77), (0.73, 0.77))
    ax.text(0.695, 0.8, "yes", fontsize=5.8, ha="center", va="bottom", color=MUTED)
    arrow(ax, (0.515, 0.62), (0.515, 0.41))
    ax.text(0.525, 0.515, "no", fontsize=5.8, ha="left", va="center", color=MUTED)
    ax.text(0.865, 0.42, "both save a third;\ndirection-fixed\nis simpler", fontsize=5.6, ha="center", va="top", color=MUTED, linespacing=1.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    fig.savefig(PAPER / "figures/choice.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def worded_claims():
    """Share questions are mostly answered "no": more than half of the certifiable ones, in every connectome of both replications."""
    shares = {}
    for name in ("replication_numbers.tex", "newtypes_numbers.tex"):
        for macro, value in re.findall(r"\\newcommand\{\\((?:Rep|NT)ShareNo(?:Male|Fly))\}\{(\d+)\}", (PAPER / name).read_text()):
            shares[macro] = int(value)
    if sorted(shares) != ["NTShareNoFly", "NTShareNoMale", "RepShareNoFly", "RepShareNoMale"] or min(shares.values()) <= 50:
        raise SystemExit(f"a claim made in words no longer holds: share questions are mostly answered no ({shares})")


def main():
    worded_claims()
    figure_pipeline()
    figure_hedge()
    figure_releases()
    figure_missing()
    figure_coverage()
    figure_choice()
    print("figures: pipeline, hedge, releases, missing, coverage rule, choice")


if __name__ == "__main__":
    main()
