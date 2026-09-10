from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def text(name: str) -> str:
    return (ROOT / "paper" / name).read_text()


def test_theory_defines_dependency_index_before_its_propositions() -> None:
    theory = text("main.tex")
    for phrase in (
        "receipt-identifier set",
        "semantic update keys",
        "receipt-complete",
        "key-complete",
        "affected positions",
        "authorizes",
    ):
        assert phrase in theory.lower()
    assert theory.index("The recorded set $D_i") < theory.index("Coverage and authorization")
    assert "trusted issuer" in theory
    assert "symbolic action model" in " ".join(theory.split())


def test_main_does_not_claim_universal_task_superiority() -> None:
    manuscript = text("main.tex") + text("supplement.tex")
    forbidden = (
        "universally superior",
        "physical-world soundness",
        "outperforms all",
        "task-level advantage over complete",
    )
    assert not any(phrase in manuscript.lower() for phrase in forbidden)
    assert "authorization" in manuscript.lower()


def test_public_sources_contain_no_internal_attempt_labels() -> None:
    manuscript = text("main.tex") + text("supplement.tex")
    for label in ("attempt-01", "attempt-02", "attempt-03", "V36", "V39", "V40"):
        assert label not in manuscript


def test_generated_numbers_are_traceable_and_not_manually_duplicated() -> None:
    generated = text("generated/revalidation_numbers.tex")
    assert r"\newcommand{\AmortizedEligible}{221}" in generated
    assert r"\newcommand{\AmortizedAgreement}{100.0\%}" in generated
    assert r"\newcommand{\AmortizedReduction}{96.7\%}" in generated
