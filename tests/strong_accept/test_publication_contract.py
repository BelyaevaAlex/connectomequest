from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_scanner_follows_active_inputs_but_ignores_unreferenced_archive(tmp_path: Path):
    from scripts.check_paper import scan_active_sources

    _write(tmp_path / "paper/main.tex", r"\input{active}" + "\n")
    _write(tmp_path / "paper/supplement.tex", "Connected supplement.\n")
    _write(tmp_path / "paper/active.tex", "The Gated policy is active.\n")
    _write(tmp_path / "paper/audit_archive.tex", "SnapshotV2 and PDPS.\n")

    errors = scan_active_sources(tmp_path)

    assert len(errors) == 1
    assert "banned-term:Gated" in errors[0]
    assert "SnapshotV2" not in "\n".join(errors)
    assert "PDPS" not in "\n".join(errors)


def test_scanner_rejects_a_table_that_combines_two_conditions(tmp_path: Path):
    from scripts.check_paper import scan_active_sources

    _write(
        tmp_path / "paper/main.tex",
        r"\input{generated/embodied_results}" + "\n",
    )
    _write(tmp_path / "paper/supplement.tex", "Connected supplement.\n")
    _write(
        tmp_path / "paper/generated/embodied_results.tex",
        "Means use 381 efficiency episodes; final column uses 513 safety episodes.\n",
    )

    errors = scan_active_sources(tmp_path)

    assert any("mixed-condition-table" in error for error in errors)


def test_scanner_rejects_fixed_query_when_target_is_replaced(tmp_path: Path):
    """Catches the bug where q=(a,y) is called fixed while y is coarsened."""
    from scripts.check_paper import scan_active_sources

    _write(
        tmp_path / "paper/main.tex",
        "Holding wiring and queries fixed while coarsening semantic targets.\n",
    )
    _write(tmp_path / "paper/supplement.tex", "Connected supplement.\n")

    errors = scan_active_sources(tmp_path)

    assert any("query-target-contradiction" in error for error in errors)


def test_scanner_rejects_undefined_timing_endpoint(tmp_path: Path):
    """Catches reintroduction of the nonstandard headline timing name."""
    from scripts.check_paper import scan_active_sources

    _write(
        tmp_path / "paper/main.tex",
        "Dependency indexing reduces fully accounted reasoning time.\n",
    )
    _write(tmp_path / "paper/supplement.tex", "Connected supplement.\n")

    errors = scan_active_sources(tmp_path)

    assert any("undefined-timing-endpoint" in error for error in errors)


def test_scanner_rejects_binomial_bound_for_correlated_mutations(tmp_path: Path):
    """Catches an invalid independent-Bernoulli interpretation of fault cases."""
    from scripts.check_paper import scan_active_sources

    _write(
        tmp_path / "paper/main.tex",
        "The rows are correlated diagnostics. The Clopper--Pearson bound is small.\n",
    )
    _write(tmp_path / "paper/supplement.tex", "Connected supplement.\n")

    errors = scan_active_sources(tmp_path)

    assert any("correlated-binomial-bound" in error for error in errors)


def test_current_active_manuscript_satisfies_the_contract():
    from scripts.check_paper import scan_active_sources

    assert scan_active_sources(ROOT) == []
