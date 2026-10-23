"""DDFIX-1023/INTEG: diagnostics distinguish discovery from acquisition. Pre-fetch R-ACQ-IRRELEVANT drops (REL rev2,
outcome "dropped" -> discovered_not_selected) are NOT acquisition attempts; a full-text R-ACQ-IRRELEVANT rejection is.
OFFLINE, synthetic records."""
from tests.test_teval_1023 import _steps


def test_all_candidates_prefetch_irrelevant_search_ran_acquisition_not_attempted():
    c = [{"doi": f"10.9999/irr{i}", "outcome": "dropped", "reason": "R-ACQ-IRRELEVANT"} for i in range(5)]
    s = _steps("none", c)
    assert s["search"] == "ran" and s["acquisition"] == "not_attempted" and s["fetch_requests"] == 0, s
    assert s["discovered"] == 5


def test_full_text_irrelevant_rejection_is_an_attempt():
    s = _steps("none", [{"doi": "10.9999/ft", "outcome": "rejected", "reason": "R-ACQ-IRRELEVANT"},
                        {"doi": "10.9999/pre", "outcome": "dropped", "reason": "R-ACQ-IRRELEVANT"}])
    assert s["acquisition"] == "attempted" and s["fetch_requests"] == 1, s
