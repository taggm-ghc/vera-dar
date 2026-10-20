"""PLAN-FDA POL tests: pure functions, no I/O."""
import pytest

from vera.fulldoc_policy import DEFAULTS, pointer_only, require_supported_claim


def rec(*verdicts):
    return [{"id": i, "verdict": v} for i, v in enumerate(verdicts)]


def test_require_supported_claim():
    assert require_supported_claim(rec("supported"))
    assert require_supported_claim(rec("partial", "supported", "unsupported"))
    assert not require_supported_claim(rec("partial"))
    assert not require_supported_claim(rec("partial", "partial", "unsupported"))
    assert not require_supported_claim([]) and not require_supported_claim(None)
    assert not require_supported_claim([None, "supported", {"verdict": None}])


@pytest.mark.parametrize("t", [
    "See Box 2.", "see Table 3 and Fig. 4.", "As described in Section 5.", "Details are shown in Figure 2B.",
    "For details see [12].", "(see Supporting Information)", "Cf. Table S1.",
])
def test_pointer_only_true(t):
    assert pointer_only(t, {})


def test_reference_list_is_pointer_only():
    refs = ("1. Smith J, Doe A. Title of a paper. J Open Res. 2019;12(3):45-60.\n"
            "2. Brown B et al. Another paper. Nature. 2020;580:100-104. doi:10.1038/xyz\n"
            "3. Lee C, Kim D. Third paper. PLoS One. 2018;13(2):e0192.\n")
    assert pointer_only(refs, {})


@pytest.mark.parametrize("t", [
    "Reproducible workflows reduce analysis errors because every step is recorded and can be rerun by others (see Table 2).",
    "Alpha systems reduce latency by forty percent in field trials.",
    "As described in Section 3, the estimator reweights each sample by the inverse probability of inclusion in the cohort.",
    "", "   ", None,
])
def test_pointer_only_false(t):
    assert not pointer_only(t, {})


def test_cfg_thresholds_and_defaults():
    t = "Results are shown in Table 2 for the larger cohort."
    assert pointer_only(t, {})  # prior: < 6 content tokens
    assert not pointer_only(t, {"pointer_filter": {"min_content_tokens": 1}})
    assert not pointer_only(t, {"min_content_tokens": 1})
    assert pointer_only(t, {"pointer_filter": {"min_content_tokens": "bad"}})  # bad value -> default
    assert pointer_only(t, None) == pointer_only(t, {})  # tolerant of a missing cfg
    assert DEFAULTS["min_content_tokens"] == 6
