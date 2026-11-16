"""Skip tests of tests/test_round5_policies.py whose requirements are absent, as in the submitted archive.

That file is pinned by outputs/round5/record.json and is therefore not edited. Its exact-solver tests need numba,
and its gate test records the environment, which reads the pinned dependency directory .v32-deps; neither ships
with the archive. Every other test declares its own requirements.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NEEDS_NUMBA = {
    "test_dp_matches_brute_force_and_bounds_every_policy",
    "test_dp_resolves_states_the_prior_excludes_optimally",
    "test_policy_evaluation_agrees_with_enumeration_over_all_answers",
    "test_dp_with_a_constant_prior_and_unit_weights_needs_no_more_than_weight_order",
}
NEEDS_DEPS = {"test_own_guard_and_shared_gate_accept_one_record"}


def pytest_collection_modifyitems(config, items):
    have_numba = importlib.util.find_spec("numba") is not None
    have_deps = (ROOT / ".v32-deps/NTAC-COMMIT").exists()
    for item in items:
        if item.path.name != "test_round5_policies.py":
            continue
        if item.name in NEEDS_NUMBA and not have_numba:
            item.add_marker(pytest.mark.skip(reason="the exact solver needs numba"))
        if item.name in NEEDS_DEPS and not have_deps:
            item.add_marker(pytest.mark.skip(reason="needs the pinned dependency directory"))
