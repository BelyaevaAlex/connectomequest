from itertools import combinations

import pytest

from connectomequest.embodied.evidence_index import EvidenceUpdate, PlanDependency, ReceiptIndex


def dependency_rows() -> tuple[PlanDependency, ...]:
    return (
        PlanDependency(0, frozenset({"r0"}), frozenset({"door:red"})),
        PlanDependency(1, frozenset(), frozenset()),
        PlanDependency(2, frozenset({"r1"}), frozenset({"goal:green"})),
        PlanDependency(3, frozenset({"r1", "r2"}), frozenset({"door:blue"})),
    )


def test_withdrawn_receipts_return_sorted_unique_future_steps() -> None:
    index = ReceiptIndex.from_dependencies(dependency_rows())
    update = EvidenceUpdate(
        withdrawn_receipts=frozenset({"r0", "r1"}),
        added_keys=frozenset(),
        superseded_receipts=frozenset(),
    )

    assert index.affected(update, cursor=1) == (2, 3)


def test_added_optimization_key_marks_only_dependent_steps() -> None:
    index = ReceiptIndex.from_dependencies(dependency_rows())
    update = EvidenceUpdate(
        withdrawn_receipts=frozenset(),
        added_keys=frozenset({"goal:green"}),
        superseded_receipts=frozenset(),
    )

    assert index.affected(update, cursor=0) == (2,)


def test_superseded_receipt_is_treated_as_changed_support() -> None:
    index = ReceiptIndex.from_dependencies(dependency_rows())
    update = EvidenceUpdate(
        withdrawn_receipts=frozenset(),
        added_keys=frozenset(),
        superseded_receipts=frozenset({"r2"}),
    )

    assert index.affected(update, cursor=0) == (3,)


@pytest.mark.parametrize(
    "rows, message",
    [
        ((PlanDependency(-1, frozenset(), frozenset()),), "non-negative"),
        (
            (
                PlanDependency(0, frozenset(), frozenset()),
                PlanDependency(0, frozenset(), frozenset()),
            ),
            "unique",
        ),
        (
            (
                PlanDependency(0, frozenset(), frozenset()),
                PlanDependency(2, frozenset(), frozenset()),
            ),
            "cover every plan step",
        ),
    ],
)
def test_malformed_dependency_rows_are_rejected(
    rows: tuple[PlanDependency, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ReceiptIndex.from_dependencies(rows)


@pytest.mark.parametrize("cursor", [-1, 5])
def test_cursor_outside_plan_is_rejected(cursor: int) -> None:
    index = ReceiptIndex.from_dependencies(dependency_rows())
    update = EvidenceUpdate(frozenset(), frozenset(), frozenset())

    with pytest.raises(ValueError, match="cursor outside plan"):
        index.affected(update, cursor)


def test_sparse_lookup_matches_hand_scanned_receipt_subsets() -> None:
    rows = dependency_rows()
    index = ReceiptIndex.from_dependencies(rows)
    receipts = ("r0", "r1", "r2", "absent")
    for size in range(len(receipts) + 1):
        for subset in combinations(receipts, size):
            for cursor in range(5):
                changed = frozenset(subset)
                expected = tuple(
                    row.step_index
                    for row in rows
                    if row.step_index >= cursor and row.required_receipts & changed
                )
                update = EvidenceUpdate(changed, frozenset(), frozenset())
                assert index.affected(update, cursor) == expected


def test_validate_complete_rejects_a_different_plan_length() -> None:
    index = ReceiptIndex.from_dependencies(dependency_rows())
    index.validate_complete(4)

    with pytest.raises(ValueError, match="incomplete dependency index"):
        index.validate_complete(5)
