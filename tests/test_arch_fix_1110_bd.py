"""ARCH-FIX-1110 B/D: own_cap names the service; fetch-failure wording split by code."""
import pytest

from vera import research_outcome as ro


def test_own_cap_names_service():
    a = [ro.PortalAttempt("duckduckgo_lite", "own_cap", 0)]
    assert "allowance for DuckDuckGo had" in ro._cause_sentences(["own_cap"], a)
    a2 = a + [ro.PortalAttempt("openalex", "own_cap", 0), ro.PortalAttempt("doaj", "completed", 3)]
    assert "allowance for DuckDuckGo and OpenAlex had" in ro._cause_sentences(["own_cap"], a2)
    assert "one search service" in ro._cause_sentences(["own_cap"], [ro.PortalAttempt("zzz", "own_cap", 0)])
    res = ro.summarize_outcome(a, [], budget_exhausted=False, older_evidence_available=False, new_evidence_admitted=False)
    assert "DuckDuckGo" in res.user_text


@pytest.mark.parametrize("code,key,words", [
    ("R-ACQ-ERR-ReadTimeout", "unreachable_timeout", "did not respond in time"),
    ("R-ACQ-ERR-ConnectTimeout", "unreachable_timeout", "did not respond in time"),
    ("R-ACQ-UNREACHABLE", "unreachable", "could not be reached"),
    ("R-ACQ-ERR-ConnectionError", "unreachable", "could not be reached"),
    ("R-ACQ-RATELIMITED-PACING", "unreachable_ownlimit", "not fetched because VERA-DAR's own rate limit for that site was reached"),
    ("R-ACQ-RATELIMITED-COOLDOWN", "unreachable_ownlimit", "not fetched because VERA-DAR's own rate limit"),
    ("R-ACQ-RATELIMITED", "unreachable_throttled", "asked VERA-DAR to slow down"),
    ("R-ACQ-RATELIMITED-BREAKER_OPEN", "unreachable_breaker", "the site failed repeatedly"),
    ("R-ACQ-RATELIMITED-HALF_OPEN_BUSY", "unreachable_breaker", "the site failed repeatedly"),
    ("R-ACQ-RATELIMITED-DEADLINE", "unreachable_paced", "the time available for this question ran out before the site's rate limit allowed another request"),
    ("R-ACQ-RATELIMITED-SEED_EXCLUDED", "unreachable_seed", "background seeding does not use that service"),
    ("R-ACQ-RATELIMITED-THROTTLED", "unreachable_throttled", "asked VERA-DAR to slow down"),
])
def test_fetch_failure_wording_by_code(code, key, words):
    assert ro.unreachable_class(code) == key
    c = ro.CandidateOutcome("c1", terminal_reason="queued_background", detail=code)
    res = ro.summarize_outcome([ro.PortalAttempt("openalex", "completed", 2)], [c], budget_exhausted=False,
                               older_evidence_available=False, new_evidence_admitted=False, source_unreachable=True)
    assert key in res.report_detail["incomplete_causes"]
    assert words in res.user_text
    assert "in time" not in res.user_text.replace("did not respond in time", "")
