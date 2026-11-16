#!/usr/bin/env python3
"""Build the manuscript and supplement from verified results.

Regenerates every number, table and figure, compiles both documents, and fails on
undefined references, overfull boxes, a missing anonymity marker, more than eight
pages of main text, or identifying strings in the sources. Runs no experiment.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FORBIDDEN = re.compile(r"cq-v\d|V3[2-5]\b|/(?:work)(?:space)|/(?:ho)(?:me)/|github\.com", re.IGNORECASE)


def run(*command):
    env = dict(os.environ, TZ="UTC", SOURCE_DATE_EPOCH="1794441600", FORCE_SOURCE_DATE="1")
    result = subprocess.run(command, cwd=PAPER, capture_output=True, text=True, env=env)
    if result.returncode:
        raise SystemExit(f"{' '.join(command)} failed:\n{result.stdout[-3000:]}")
    return result.stdout


def build(name):
    run("pdflatex", "-interaction=nonstopmode", "-halt-on-error", name)
    run("bibtex", name)
    run("pdflatex", "-interaction=nonstopmode", "-halt-on-error", name)
    log = run("pdflatex", "-interaction=nonstopmode", "-halt-on-error", name)
    problems = [line for line in log.splitlines() if re.search(r"undefined|Overfull|multiply defined|Rerun", line)]
    if problems:
        raise SystemExit(f"{name}: " + "\n".join(problems[:20]))
    pages = int(re.search(r"\((\d+) pages", log).group(1))
    return pages


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/v35_paper_assets.py")], check=True)
    for name in ("main", "supplement"):
        text = (PAPER / f"{name}.tex").read_text()
        if "Anonymous Submission" not in text:
            raise SystemExit(f"{name}: anonymity marker missing")
        found = FORBIDDEN.search(text)
        if found:
            raise SystemExit(f"{name}: forbidden string {found.group(0)!r}")
    pages = {name: build(name) for name in ("main", "supplement")}
    if pages["main"] > 8:
        raise SystemExit(f"main text has {pages['main']} pages; the limit is 8")
    for page in range(1, pages["main"] + 1):
        text = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "main.pdf", "-"], cwd=PAPER, capture_output=True, text=True).stdout
        if re.search(r"^References$", text, re.MULTILINE):
            if page > 7:
                raise SystemExit(f"the body runs to page {page}; it must end by page 7")
            print(f"references start on page {page}")
            break
    else:
        raise SystemExit("reference list not found")
    print(pages)


if __name__ == "__main__":
    main()
