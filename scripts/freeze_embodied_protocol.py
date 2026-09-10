#!/usr/bin/env python3
"""Freeze one V39 stage without any scientific override flags."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from connectomequest.embodied.protocol import freeze_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    stage = "smoke" if args.smoke else "pilot" if args.pilot else "confirmation"
    path = freeze_stage(stage)
    print(f"{path} {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
