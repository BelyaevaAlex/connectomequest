from scripts.render_revalidation_figure import (
    bar_label,
    timing_label,
)


def test_publication_labels_are_short_and_define_both_methods():
    assert timing_label("complete_validation", 4) == "Complete validation, 4 changes"
    assert timing_label("dependency_indexed", 8) == "Dependency-indexed, 8 changes"
    assert bar_label("complete_validation") == "Complete\nvalidation"
    assert bar_label("dependency_indexed") == "Indexed\nvalidation"
