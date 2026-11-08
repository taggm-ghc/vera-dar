"""OUTCOME-LABELS-1108: VERA-DAR's own decisions are not reported as "a search service had a failure"."""
from vera import rate_limit, research_outcome as ro, research_record as rr, portal_recorder
from vera import search_providers as sp
from vera.search_and_fetch import SearchError


def _dec(msg):
    return sp._own_decision(SearchError("search failed after bounded retries across providers: " + msg))


def test_ddg_limiter_skip_reasons():
    assert _dec("attempt 1: rate_limited (pacing; not an attempt)") == "paced"
    assert _dec("attempt 1: rate_limited (concurrency; not an attempt)") == "paced"
    assert _dec("attempt 1: rate_limited (daily_cap; not an attempt)") == "own_cap"
    # cooldown / breaker follow a real provider refusal: stays a service failure
    assert _dec("attempt 1: rate_limited (cooldown; not an attempt)") is None
    assert _dec("attempt 1: rate_limited (breaker_open; not an attempt)") is None


def test_own_daily_caps_and_robots():
    assert _dec("attempt 1: provider unavailable: duckduckgo_lite daily query cap (quota) of 8 spent for d") == "own_cap"
    assert sp._own_decision(sp.ProviderUnavailable("openalex daily search budget of 90 spent for d")) == "own_cap"
    assert sp._own_decision(sp.ProviderUnavailable("R-ACQ-ROBOTS: robots.txt of the DuckDuckGo host disallows")) == "robots"
    assert _dec("attempt 1: provider unavailable: R-ACQ-ROBOTS: robots.txt disallows") == "robots"
    assert sp._own_decision(rate_limit.RateLimited("openalex", "daily_cap", 5)) == "own_cap"


def test_real_failures_stay_failures():
    assert sp._own_decision(RuntimeError("HTTP 500 boom")) is None
    assert sp._own_decision(RuntimeError("duckduckgo challenge page; not evaded")) is None
    assert sp._own_decision(RuntimeError("HTTP 429 too many requests")) is None


def test_statuses_flow_to_distinct_sentences():
    for st, words in (("own_cap", "daily search allowance"), ("robots", "robots rules do not permit")):
        assert rr._PORTAL_STATUS[st] == st and rr._PUBLIC_STATUS[st] == "skipped"
        assert portal_recorder._STATUS[st] == "skipped"
        assert ro._causes([ro.PortalAttempt("ddg", st, 0)], False) == [st]
        t = ro._CAUSE_TEXT[st]
        assert t.startswith("VERA-DAR") and words in t and "failure" not in t and "a search service" not in t
    assert ro._CAUSE_TEXT["error"] == "a search service had a failure"
    assert ro._causes([ro.PortalAttempt("ddg", "error", 0)], False) == ["error"]
