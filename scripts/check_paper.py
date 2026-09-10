#!/usr/bin/env python3
"""Check the transitive active manuscript for V40 publication-contract errors."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_INPUT = re.compile(r"\\input\{([^}]+)\}")
_BANNED = (
    (re.compile(r"\bGated\b"), "Gated"),
    (re.compile(r"MINERVA-inspired", re.IGNORECASE), "MINERVA-inspired"),
    (
        re.compile(r"\b(?:FullReplan|FullScan|ReceiptIndex|UncheckedReuse)\b"),
        "internal-method-alias",
    ),
    (re.compile(r"legacy base", re.IGNORECASE), "legacy-base"),
    (re.compile(r"\bSnapshotV\d*\b"), "SnapshotV"),
    (re.compile(r"\bGateV\d*\b"), "GateV"),
    (re.compile(r"\bPDPS\b"), "PDPS"),
    (re.compile(r"\bID-free\b", re.IGNORECASE), "ID-free"),
    (re.compile(r"REINFORCE--\\LSTM", re.IGNORECASE), "REINFORCE--LSTM"),
    (re.compile(r"\battribute-based scorer\b", re.IGNORECASE), "attribute-based scorer"),
    (re.compile(r"\bheld-out-once\b", re.IGNORECASE), "held-out-once"),
    (re.compile(r"\bevidence token\b", re.IGNORECASE), "evidence token"),
    (re.compile(r"\bsupport checker\b", re.IGNORECASE), "support checker"),
    (re.compile(r"\bpost-seal\b", re.IGNORECASE), "post-seal"),
    (re.compile(r"\breviewer-facing\b", re.IGNORECASE), "reviewer-facing"),
    (re.compile(r"\bgraph-free verifier\b", re.IGNORECASE), "graph-free verifier"),
    (re.compile(r"(?<![A-Za-z0-9_])V\d{1,3}(?![A-Za-z0-9_.])"), "internal-version"),
)
_CONTRACT_ERRORS = (
    (re.compile(r"holding\s+wiring\s+and\s+queries\s+fixed", re.I), "query-target-contradiction"),
    (re.compile(r"fully\s+accounted\s+reasoning\s+time", re.I), "undefined-timing-endpoint"),
)


def _resolve_input(paper: Path, name: str) -> Path:
    candidate = paper / name
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".tex")
    return candidate


def _active_sources(root: Path) -> tuple[Path, ...]:
    paper = root / "paper"
    pending = [paper / "main.tex", paper / "supplement.tex"]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        path = path.resolve()
        if path in visited:
            continue
        if not path.is_file():
            continue
        visited.add(path)
        text = path.read_text()
        for name in _INPUT.findall(text):
            child = _resolve_input(paper, name).resolve()
            if child.name == "audit_archive.tex":
                continue
            pending.append(child)
    return tuple(sorted(visited))


def active_text(root: Path) -> str:
    """Return the concatenated text of the transitive active TeX sources."""

    return "\n".join(path.read_text() for path in _active_sources(root))


def scan_active_sources(root: Path) -> list[str]:
    """Return deterministic publication-contract errors for active TeX inputs."""

    errors: list[str] = []
    paper = root / "paper"
    for path in _active_sources(root):
        text = path.read_text()
        relative = path.relative_to(paper.resolve())
        for pattern, label in _BANNED:
            if pattern.search(text):
                errors.append(f"banned-term:{label}:{relative}")
        for pattern, label in _CONTRACT_ERRORS:
            if pattern.search(text):
                errors.append(f"{label}:{relative}")
        if re.search(r"correlated\s+diagnostics", text, re.I) and re.search(
            r"Clopper--Pearson", text, re.I
        ):
            errors.append(f"correlated-binomial-bound:{relative}")
        if re.search(r"final column uses\s+\d+", text, re.IGNORECASE):
            errors.append(f"mixed-condition-table:{relative}")
    return sorted(errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    errors = scan_active_sources(args.root.resolve())
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
