#!/usr/bin/env python3
"""Render condition-separated V40 panels from hash-verified V39 outcomes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from connectomequest.embodied.reporting import render_condition_panels
from scripts.render_embodied_pilot import build_summary

OUTPUT = ROOT / "paper/generated/embodied_results.tex"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render_condition_panels(build_summary())
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text() != text:
            raise AssertionError(f"stale generated artifact: {OUTPUT}")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
    print(OUTPUT)


if __name__ == "__main__":
    main()
