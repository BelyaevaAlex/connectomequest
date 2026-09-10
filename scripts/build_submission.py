#!/usr/bin/env python3
"""Rebuild, audit, and package the anonymous manuscript in ``paper/``."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
ARTIFACT = PAPER / "artifact.zip"
SOURCES = PAPER / "sources.zip"
VERIFICATION = PAPER / "verification.json"
CONFIRMATION = ROOT / "outputs/nemo/strong-accept/amortized-revalidation-v40/confirmation"
IDENTITY = re.compile(
    r"/(?:home|workspace-[^/]+)/[^/\s]+|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
    re.I,
)
BAD_LOG = re.compile(
    r"Overfull|Undefined control sequence|(?:Citation|Reference).*undefined|multiply defined", re.I
)
INPUT = re.compile(r"\\input\{([^}]+)\}")
GRAPHIC = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
STAMP = (2026, 9, 10, 0, 0, 0)


def run(command: list[str], cwd: Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_text(path: Path) -> str:
    return subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)


def pdf_pages(path: Path) -> int:
    information = subprocess.check_output(["pdfinfo", str(path)], text=True)
    match = re.search(r"^Pages:\s+(\d+)$", information, re.M)
    if not match:
        raise ValueError(f"cannot read page count: {path}")
    return int(match.group(1))


def body_pages(path: Path) -> int:
    for page in range(1, pdf_pages(path) + 1):
        text = subprocess.check_output(
            ["pdftotext", "-f", str(page), "-l", str(page), "-layout", str(path), "-"], text=True
        )
        if re.search(r"^\s*(?:\d+\s+)?References\s*$", text, re.M):
            return page - 1
    raise AssertionError("References heading not found")


def source_files() -> dict[str, Path]:
    selected: set[Path] = {
        PAPER / "main.tex",
        PAPER / "supplement.tex",
        PAPER / "references.bib",
        PAPER / "neurips_2026.sty",
    }
    pending = [PAPER / "main.tex", PAPER / "supplement.tex"]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        text = path.read_text(encoding="utf-8")
        if IDENTITY.search(text):
            raise AssertionError(f"identity in {path.relative_to(ROOT)}")
        for name in INPUT.findall(text):
            child = PAPER / name
            if child.suffix == "":
                child = child.with_suffix(".tex")
            selected.add(child)
            if child.suffix == ".tex":
                pending.append(child)
        for name in GRAPHIC.findall(text):
            selected.add(PAPER / name)
    missing = [str(path) for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return {str(path.relative_to(PAPER)): path for path in sorted(selected)}


def write_source_archive(files: dict[str, Path]) -> None:
    with zipfile.ZipFile(SOURCES, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            info = zipfile.ZipInfo(name, STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())


def clean_rebuild() -> dict[str, object]:
    expected = {stem: pdf_text(PAPER / f"{stem}.pdf") for stem in ("main", "supplement")}
    with tempfile.TemporaryDirectory(prefix="connectomequest-clean-build-") as directory:
        target = Path(directory)
        with zipfile.ZipFile(SOURCES) as archive:
            archive.extractall(target)
        for stem, expected_text in expected.items():
            run(
                ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"],
                target,
            )
            if pdf_text(target / f"{stem}.pdf") != expected_text:
                raise AssertionError(f"clean rebuild differs: {stem}")
    return {"passed": True, "sources_sha256": sha256(SOURCES)}


def test_counts(path: Path) -> dict[str, int]:
    counts = {name: 0 for name in ("tests", "failures", "errors", "skipped")}
    for suite in ET.parse(path).getroot().iter("testsuite"):
        for name in counts:
            counts[name] += int(suite.attrib.get(name, 0))
    return counts


def validate_bundle(stored: dict[str, object]) -> None:
    for name, expected in stored["bundle_sha256"].items():
        path = PAPER / name
        if not path.is_file() or sha256(path) != expected:
            raise AssertionError(f"bundle drift: {name}")


def build(check: bool) -> dict[str, object]:
    python = str(ROOT / ".venv/bin/python")
    run([python, "scripts/check_paper.py"])
    for script, extra in (
        ("render_embodied_results.py", []),
        ("render_graph_costs.py", []),
        ("render_graph_results.py", []),
        ("render_certificate_audit.py", []),
        ("report_revalidation_benchmark.py", ["--stage", "confirmation"]),
        ("render_revalidation_figure.py", []),
    ):
        mode = ["--check"] if check else []
        run([python, f"scripts/{script}", *extra, *mode])
    if not check:
        run([python, "scripts/build_artifact.py"])
    if not ARTIFACT.is_file():
        raise AssertionError("missing artifact.zip")

    junit = Path("/tmp/connectomequest-publication-tests.xml")
    run([python, "-m", "pytest", "-q", f"--junitxml={junit}"])
    tests = test_counts(junit)
    if tests["failures"] or tests["errors"]:
        raise AssertionError(tests)

    for stem in ("main", "supplement"):
        run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"], PAPER)
        log = (PAPER / f"{stem}.log").read_text(errors="replace")
        match = BAD_LOG.search(log)
        if match:
            raise AssertionError(f"{stem}.log contains {match.group(0)}")
        if IDENTITY.search(pdf_text(PAPER / f"{stem}.pdf")):
            raise AssertionError(f"identity in {stem}.pdf")
    pages = body_pages(PAPER / "main.pdf")
    if pages != 8:
        raise AssertionError(f"main paper has {pages} body pages, expected 8")

    files = source_files()
    if not check:
        write_source_archive(files)
    if not SOURCES.is_file():
        raise AssertionError("missing sources.zip")
    if check:
        stored = json.loads(VERIFICATION.read_text())
        validate_bundle(stored)
        return stored

    sealed = json.loads((CONFIRMATION / "sealed_report.json").read_text())
    if not sealed["gates"]["correctness"]["passed"]:
        raise AssertionError("confirmation correctness gate failed")
    clean = clean_rebuild()
    manifest = {
        "status": "verified",
        "body_pages": pages,
        "tests": tests,
        "gates": sealed["gates"],
        "paper_sources": {name: sha256(path) for name, path in sorted(files.items())},
        "clean_rebuild": clean,
        "pdfs": {
            stem: {
                "pages": pdf_pages(PAPER / f"{stem}.pdf"),
                "sha256": sha256(PAPER / f"{stem}.pdf"),
            }
            for stem in ("main", "supplement")
        },
        "identity_scan": {"passed": True},
        "terminology_scan": {"passed": True},
    }
    manifest["bundle_sha256"] = {
        name: sha256(PAPER / name)
        for name in ("main.pdf", "supplement.pdf", "artifact.zip", "sources.zip")
    }
    VERIFICATION.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.check), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
