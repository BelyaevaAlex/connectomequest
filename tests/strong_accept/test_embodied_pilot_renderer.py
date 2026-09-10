from __future__ import annotations


def test_full_episode_pilot_renderer_recomputes_paired_audit():
    from scripts.render_embodied_pilot import build_summary, render_tex

    summary = build_summary()
    assert summary["rows"] == 12_312
    assert summary["paired_cells"] == 3_078
    assert summary["exact_replay_rate"] == 1.0
    assert summary["primary"]["receipt_index"]["n"] == 381
    assert summary["primary"]["receipt_index"]["success"] == 1.0
    assert summary["primary"]["full_scan"]["success"] == 1.0
    assert summary["primary"]["receipt_index"]["predicates"] == 0.0
    assert summary["primary"]["full_scan"]["predicates"] > 60.0
    assert summary["safety"]["receipt_index"]["unsupported_episodes"] == 0
    assert summary["safety"]["unchecked_reuse"]["unsupported_episodes"] == 511
    tex = render_tex(summary)
    assert "381 paired" in tex
    assert "Full-episode mechanism audit" in tex
    assert "Timing is excluded" in tex
    assert "511/513" in tex
