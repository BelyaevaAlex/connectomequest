"""Condition-separated publication rendering for existing full-episode outcomes."""

from __future__ import annotations

from collections.abc import Mapping

METHODS = ("full_replan", "full_scan", "receipt_index", "unchecked_reuse")
LABELS = {
    "full_replan": "Full replanning",
    "full_scan": "Complete validation",
    "receipt_index": r"\textbf{Dependency-indexed}",
    "unchecked_reuse": r"Unchecked reuse$^{\dagger}$",
}


def _common_n(rows: Mapping[str, Mapping[str, float]], panel: str) -> int:
    if set(rows) != set(METHODS):
        raise ValueError(f"{panel} does not contain all methods")
    sizes = {int(rows[method]["n"]) for method in METHODS}
    if len(sizes) != 1:
        raise ValueError(f"{panel} methods have different denominators")
    return sizes.pop()


def render_condition_panels(summary: Mapping[str, object]) -> str:
    """Render efficiency and safety interventions without joining their outcomes."""

    primary = summary["primary"]
    safety = summary["safety"]
    if not isinstance(primary, Mapping) or not isinstance(safety, Mapping):
        raise TypeError("summary panels must be mappings")
    efficiency_n = _common_n(primary, "efficiency")
    safety_n = _common_n(safety, "safety")
    lines = [
        r"\begin{table}[t]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\caption{Full-episode mechanism audit. Panels are distinct interventions and must not be compared row-wise across panels.}",
        r"\label{tab:full-episode-primary}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        rf"\multicolumn{{8}}{{l}}{{\textit{{Panel A: irrelevant evidence withdrawal ({efficiency_n} paired schedule cells)}}}} \\",
        r"Method & Success & Actions & Pred. & Succ. & Positions & Lookups & Unsupported \\",
        r"\midrule",
    ]
    for method in METHODS:
        row = primary[method]
        lines.append(
            f"{LABELS[method]} & {float(row['success']):.3f} & "
            f"{float(row['actions']):.1f} & {float(row['predicates']):.1f} & "
            f"{float(row['successors']):.1f} & {float(row['positions']):.1f} & "
            f"{float(row['lookups']):.1f} & -- \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            rf"\multicolumn{{8}}{{l}}{{\textit{{Panel B: final-support loss ({safety_n} paired episodes)}}}} \\",
            r"Method & \multicolumn{6}{l}{Unsupported authorization episodes} & Rate \\",
            r"\midrule",
        ]
    )
    for method in METHODS:
        unsupported = int(safety[method]["unsupported_episodes"])
        rate = unsupported / safety_n if safety_n else float("nan")
        lines.append(
            rf"{LABELS[method]} & \multicolumn{{6}}{{l}}{{{unsupported}/{safety_n}}} "
            f"& {rate:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{1mm}",
            r"\parbox{0.97\linewidth}{\scriptsize Pred./Succ. are predicate evaluations and planner-successor expansions. Positions are retained-suffix positions visited. $^{\dagger}$Unchecked reuse is a safety ablation that performs no validation after an update.}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)
