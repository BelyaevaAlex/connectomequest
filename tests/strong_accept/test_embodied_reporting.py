from __future__ import annotations


def _summary():
    primary = {}
    safety = {}
    for method, unsupported in (
        ("full_replan", 0),
        ("full_scan", 0),
        ("receipt_index", 0),
        ("unchecked_reuse", 11),
    ):
        primary[method] = {
            "n": 9,
            "success": 1.0,
            "actions": 50.0,
            "predicates": 6.0,
            "successors": 8.0,
            "positions": 7.0,
            "lookups": 1.0 if method == "receipt_index" else 0.0,
        }
        safety[method] = {"n": 12, "unsupported_episodes": unsupported}
    return {"primary": primary, "safety": safety}


def test_renderer_separates_efficiency_and_safety_denominators():
    from connectomequest.embodied.reporting import render_condition_panels

    text = render_condition_panels(_summary())
    panel_a, panel_b = text.split(r"\multicolumn{8}{l}{\textit{Panel B")
    assert "Panel A" in panel_a
    assert "9 paired schedule cells" in panel_a
    assert "12" not in panel_a
    assert "12 paired episodes" in panel_b
    assert "11/12" in panel_b
    assert "9 paired schedule cells" not in panel_b


def test_renderer_labels_conditions_instead_of_joining_columns():
    from connectomequest.embodied.reporting import render_condition_panels

    text = render_condition_panels(_summary())
    assert "irrelevant evidence withdrawal" in text
    assert "final-support loss" in text
    assert "final column uses" not in text
