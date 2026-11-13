"""REASON-TEXT2-1113: COOLDOWN gets its own true sentence; the internal report (live_report._research) reuses the
reading-list reason lookup. Offline; fixtures use synthetic codes only."""
import inspect
import re

import pytest

from tests.test_discovery_reasons import INTERNAL
from vera import discovery_panel as dp
from vera import research_outcome as ro
from vera.m7 import live_report as lr

BASE = "R-ACQ-RATELIMITED"
COOLDOWN = f"{BASE}-COOLDOWN"
EXPECTED = "Left for later: that site recently limited requests from VERA-DAR, so VERA-DAR was waiting before sending another request."


def line(code):
    return dp._reason_line({"outcome": "rejected", "reason": code})


def lr_reason(code):
    out = lr._research({"research": {"candidates": [{"reason": code, "outcome": "rejected"}]}})
    return out["candidates"][0]["reason"]


def test_1_ro_class_and_text_for_cooldown():
    assert ro.unreachable_class(COOLDOWN) == "unreachable_cooldown"
    c = ro.CandidateOutcome("c1", terminal_reason="queued_background", detail=COOLDOWN)
    res = ro.summarize_outcome([ro.PortalAttempt("openalex", "completed", 2)], [c], budget_exhausted=False,
                               older_evidence_available=False, new_evidence_admitted=False, source_unreachable=True)
    assert "recently limited requests" in res.user_text
    assert "own rate limit" not in res.user_text


def test_2_dp_line_for_cooldown():
    t = line(COOLDOWN)
    assert t == EXPECTED
    assert "too many requests" not in t
    for other in (f"{BASE}-THROTTLED", f"{BASE}-PACING", f"{BASE}-ZZZ"):
        assert line(other) != t


@pytest.mark.parametrize("code", [f"{BASE}-{s}" for s in dp._LIMITER_TEXT] + [
    f"{BASE}-THROTTLED", BASE, f"{BASE}-ZZZ", "r-acq-ratelimited-pacing", "r-acq-ratelimited-cooldown",
    "R-ACQ-ROBOTS_UNREADABLE-TIMEOUT", "R-ACQ-ROBOTS-STATUS_403", "R-ACQ-NOLIC"])
def test_3_lr_reason_equals_panel_line(code):
    assert lr_reason(code) == line(code)
    assert lr_reason(code)


def test_3b_cooldown_in_table_and_empty_code():
    assert "COOLDOWN" in dp._LIMITER_TEXT
    assert lr_reason("") == ""
    assert dp.code_text("") is None and dp.code_text(None) is None and dp.code_text(5) is None


def test_4_lr_source_uses_code_text():
    src = inspect.getsource(lr._research)
    assert "code_text" in src
    assert "REASON_TEXT.get(" not in src
    whole = inspect.getsource(lr)
    assert "Not fetched:" not in whole and "Left for later:" not in whole


@pytest.mark.parametrize("code", [COOLDOWN, f"{BASE}-ZZZ", "R-ACQ-ROBOTS-STATUS_403", "R-ACQ-NOLIC"])
def test_5_no_internal_leak_in_lr_reason(code):
    t = lr_reason(code)
    assert not INTERNAL.search(t)
    assert not re.search(r"https?://|\b[a-z0-9-]+\.(org|com|net)\b", t, re.I)
