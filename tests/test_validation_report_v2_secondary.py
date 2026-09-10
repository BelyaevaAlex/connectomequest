from connectomequest.validation_report_v2 import render_markdown


def test_markdown_renders_secondary_gate_vs_snapshot_ci() -> None:
    report = {
        "input_fingerprint_sha256": "a" * 64,
        "baseline_aggregates": [],
        "v2_aggregates": [],
        "primary_paired_comparisons": [],
        "secondary_gate_vs_snapshot_comparisons": [
            {
                "held_out": "manc",
                "paired_bootstrap": {
                    "success_at_budget": {
                        "estimate": 0.0163,
                        "ci95_low": 0.0040,
                        "ci95_high": 0.0290,
                    }
                },
            }
        ],
    }
    rendered = render_markdown(report)
    assert "Secondary paired B64" in rendered
    assert "| manc | +0.016 | [+0.004, +0.029] |" in rendered
