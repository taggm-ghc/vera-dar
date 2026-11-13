"""REASON-TEXT-1113: each R-ACQ-RATELIMITED-<why> limiter-skip code gets a reading-list sentence true for its cause.
Offline; fixtures use example.org only."""
import pytest

from tests.test_discovery_panel import render, texts
from tests.test_discovery_reasons import INTERNAL, data_for
from vera import discovery_panel as dp
from vera import research_outcome as ro

BASE = "R-ACQ-RATELIMITED"
SUFFIXES = ["DEADLINE_PASSED", "SEED_EXCLUDED", "WAIT_TOO_LONG", "DEADLINE", "BREAKER_OPEN", "HALF_OPEN_BUSY",
            "PACING", "CONCURRENCY", "DAILY_CAP"]
EXPECTED = {
    "DEADLINE_PASSED": "Not fetched: the time available for this question had already run out before another request to that site could be made.",
    "SEED_EXCLUDED": "Not fetched: background seeding does not use that service.",
    "WAIT_TOO_LONG": "Not fetched: that site's rate limit required a longer wait than the time left for this question.",
    "DEADLINE": "Not fetched: that site's rate limit required a longer wait than the time left for this question.",
    "BREAKER_OPEN": "Left for later: the site failed repeatedly, so VERA-DAR paused requests to it.",
    "HALF_OPEN_BUSY": "Left for later: the site failed repeatedly, so VERA-DAR paused requests to it.",
    "PACING": "Not fetched: VERA-DAR's own rate limit for that site was reached.",
    "CONCURRENCY": "Not fetched: VERA-DAR's own rate limit for that site was reached.",
    "DAILY_CAP": "Not fetched: VERA-DAR's own rate limit for that site was reached.",
}
CLASS_OF = {
    "DEADLINE_PASSED": "unreachable_deadline_passed", "SEED_EXCLUDED": "unreachable_seed",
    "WAIT_TOO_LONG": "unreachable_paced", "DEADLINE": "unreachable_paced",
    "BREAKER_OPEN": "unreachable_breaker", "HALF_OPEN_BUSY": "unreachable_breaker",
    "PACING": "unreachable_ownlimit", "CONCURRENCY": "unreachable_ownlimit", "DAILY_CAP": "unreachable_ownlimit",
}
CLASS_SET = {"unreachable_ownlimit", "unreachable_paced", "unreachable_deadline_passed", "unreachable_seed",
             "unreachable_breaker"}
LEGACY = dp.REASON_TEXT["R-ACQ-RATELIMITED"]


def line(code):
    return dp._reason_line({"outcome": "rejected", "reason": code})


@pytest.mark.parametrize("sfx", SUFFIXES)
def test_i_suffix_gets_its_table_sentence(sfx):
    t = line(f"{BASE}-{sfx}")
    assert t == EXPECTED[sfx]
    assert "too many requests" not in t


def test_ii_distinct_causes_distinct_sentences():
    ts = [line(f"{BASE}-{s}") for s in ("DEADLINE_PASSED", "SEED_EXCLUDED", "WAIT_TOO_LONG", "BREAKER_OPEN", "PACING")]
    assert len(set(ts)) == 5
    dpass = line(f"{BASE}-DEADLINE_PASSED").lower()
    assert "rate limit" not in dpass and "wait" not in dpass
    seed = line(f"{BASE}-SEED_EXCLUDED").lower()
    assert "limit" not in seed and "wait" not in seed


@pytest.mark.parametrize("code", [f"{BASE}-COOLDOWN", f"{BASE}-THROTTLED", BASE, f"{BASE}-ZZZ",
                                  "r-acq-ratelimited-pacing"])
def test_iii_lookalikes_keep_legacy_text(code):
    assert line(code) == LEGACY


def test_iv_parity_with_research_outcome():
    assert set(SUFFIXES) == set(CLASS_OF) == set(dp._LIMITER_TEXT)
    for sfx in SUFFIXES:
        code = f"{BASE}-{sfx}"
        assert ro.unreachable_class(code) == CLASS_OF[sfx] and CLASS_OF[sfx] in CLASS_SET
        # same cause -> same sentence in DP
        same = [s for s in SUFFIXES if CLASS_OF[s] == CLASS_OF[sfx]]
        assert len({line(f"{BASE}-{s}") for s in same}) == 1
    assert len({line(f"{BASE}-{s}") for s in SUFFIXES}) == len(CLASS_SET)


def test_v_end_to_end_reading_list_row():
    data = data_for([{"outcome": "rejected", "reason": f"{BASE}-DEADLINE_PASSED"}])
    t = " | ".join(texts(render(data)))
    assert EXPECTED["DEADLINE_PASSED"] in t
    assert "too many requests" not in t
    assert not INTERNAL.search(t.replace("not_acquired", ""))


def test_vi_known_row_has_exactly_one_membership_label():
    data = data_for([{"outcome": "already_stored", "reason": "R-ACQ-KNOWN"}])
    ts = texts(render(data))
    assert not any("not in VERA-DAR's collection" in t for t in ts)
    assert not any("Already in the collection" in t for t in ts)
    assert sum("in VERA-DAR's collection: full document stored" in t for t in ts) == 1


@pytest.mark.parametrize("sfx", ["DEADLINE_PASSED", "SEED_EXCLUDED"])
def test_vii_research_outcome_pins(sfx):
    code = f"{BASE}-{sfx}"
    assert ro.unreachable_class(code) == CLASS_OF[sfx]
    c = ro.CandidateOutcome("c1", terminal_reason="queued_background", detail=code)
    res = ro.summarize_outcome([ro.PortalAttempt("openalex", "completed", 2)], [c], budget_exhausted=False,
                               older_evidence_available=False, new_evidence_admitted=False, source_unreachable=True)
    assert "rate limit" not in res.user_text
