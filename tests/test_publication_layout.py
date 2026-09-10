"""Repository-level invariants for the publication release."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_python_api_has_no_workshop_named_identifiers() -> None:
    roots = (ROOT / "src", ROOT / "scripts", ROOT / "tests")
    violations: list[str] = []
    for root in roots:
        for path in root.rglob("*.py"):
            if "nemo" in path.name.lower():
                violations.append(str(path.relative_to(ROOT)))
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if "nemo" in node.name.lower():
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
                if isinstance(node, ast.ImportFrom) and node.module:
                    if "nemo" in node.module.lower():
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.module}")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if "nemo" in alias.name.lower():
                            violations.append(
                                f"{path.relative_to(ROOT)}:{node.lineno}:{alias.name}"
                            )
    assert violations == []


def test_paper_has_one_flat_publication_bundle() -> None:
    paper = ROOT / "paper"
    required = {
        "main.tex",
        "main.pdf",
        "supplement.tex",
        "supplement.pdf",
        "artifact.zip",
        "sources.zip",
        "verification.json",
    }
    assert required <= {path.name for path in paper.iterdir() if path.is_file()}
    assert not list(paper.glob("submission*"))
    assert not list(paper.glob("*.orig"))
    assert not (paper / "main_no_checklist.tex").exists()


def test_repository_has_no_backup_sources() -> None:
    assert not list(ROOT.rglob("*.orig"))
    assert not list(ROOT.rglob("*.rej"))
