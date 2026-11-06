"""NEWS-THROTTLE-NT12-1106: DDG 202/challenge -> trace status throttled + why; no spacing sleep during a cooldown."""
import pytest

import vera.rate_limit as rl
import vera.request_trace as rt
import vera.search_and_fetch as sf
import vera.search_providers as sp
from tests.config_support import load_pair  # noqa: F401  (keeps the config helpers importable like sibling tests)
from vera.eval_config import load_eval_config

DDG = load_eval_config().search["duckduckgo_lite"]
Q = "What does published research show about AI coding assistants?"


class R:
    def __init__(self, status=200, text=""):
        self.status_code, self.text, self.headers = status, text, {}


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def __call__(self):
        return self.t

    def wall(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def _prov(resps, clock):
    seq = list(resps)
    p = sp.DuckDuckGoLiteProvider(DDG, transport=lambda u, d, h, t: seq.pop(0), query_terms=load_eval_config().search["query_terms"],
                                  sleep=clock.sleep)
    return p


def _run(p, clock):
    return sf._attempts_once(p, Q, 5, ledger=None, max_retries=1, backoff_s=0.0, deadline_at=clock.t + 1e6,
                             timeout_s=5.0, sleep=clock.sleep, clock=clock)


def _notes(p, clock):
    with pytest.raises(sf._ProviderFailed) as ei:
        _run(p, clock)
    return ei.value.notes


def _captured(monkeypatch, notes):
    got = []
    monkeypatch.setattr(rt, "event", lambda stage, code, counts=None, detail=None: got.append(detail))
    sf._trace_provider("duckduckgo_lite", None, notes=notes, ms=3)
    return got[0]


@pytest.mark.parametrize("resp,why", [(R(202, ""), "http_202"),
                                      (R(200, "<html><div class='anomaly-modal'>x</div></html>"), "challenge")])
def test_anti_bot_is_throttled_with_why(monkeypatch, resp, why):
    clock = Clock()
    d = _captured(monkeypatch, _notes(_prov([resp], clock), clock))
    assert d["status"] == "throttled" and d["why"] == why


def test_other_why_tokens_keep_their_status():
    assert sf._failure_why(["attempt 1: rate_limited (cooldown; not an attempt)"]) == "cooldown"
    assert sf._failure_why(["attempt 1: rate_limited (daily_cap; not an attempt)"]) == "daily_cap"
    assert sf._failure_why(["provider unavailable: R-ACQ-ROBOTS: x"]) == "robots"
    assert sf._failure_why(["attempt 1: HTTP 429 'x'"]) == "http_429"
    assert sf._failure_why(["attempt 1: network error"]) is None
    assert set(sf.WHY_TOKENS) >= {"http_202", "challenge", "cooldown", "daily_cap", "robots", "http_429"}


def test_why_dropped_by_deny_all_allowlist_is_safe():
    # the trace gate drops unlisted detail keys; the event call itself must not raise either way
    rt.event("search", "provider", counts={"results": 0}, detail={"provider": "x", "status": "throttled", "why": "http_202"})


def _registry(clock, base_cooldown=900.0):
    dflt = {"increase_frac": 0.1, "decrease_factor": 0.5, "min_rate_factor": 0.1, "breaker_failures": 5,
            "breaker_open_s": 10.0, "breaker_open_max_s": 40.0, "half_open_max": 1, "jitter_frac": 0.0,
            "max_wait_s": 2.0, "min_call_s": 1.0, "retry_after_max_s": 600.0, "base_cooldown_s": base_cooldown,
            "max_cooldown_s": 3600.0, "poll_s": 0.01}
    raw = {"defaults": dflt,
           "portals": {"duckduckgo": {"hosts": ["duckduckgo.com"], "rate_per_s": 1.0, "burst": 1, "max_concurrency": 1,
                                      "throttle_statuses": [202], "sources": [{"url": "https://example.invalid/x", "read": "2026-11-06"}]}}}
    return rl.Registry(rl.parse_config(raw, env={}), clock=clock, wall=clock.wall, sleep=clock.sleep, rand=lambda: 0.0)


@pytest.fixture
def reg(monkeypatch):
    clock = Clock()
    r = _registry(clock)
    rl.set_registry_for_tests(r)
    yield r, clock
    rl.set_registry_for_tests(None)


def test_cooldown_skips_spacing_sleep(reg):
    r, clock = reg
    ad = r.portal("duckduckgo")
    g = ad.acquire()
    assert g.ok
    g.record(status=202)  # throttle -> cooldown
    assert ad.cooling_down()
    p2 = sp.DuckDuckGoLiteProvider(DDG, query_terms=load_eval_config().search["query_terms"], sleep=clock.sleep,
                                    robots_check=lambda *a: True)
    p2.gate.min_interval_s = 5.0
    p2.gate.stamp(clock())
    # drive the real transport path through the adapter: requests.post must never be reached
    notes = _notes(p2, clock)
    assert clock.slept == []
    assert any("rate_limited (cooldown" in n for n in notes)


def test_spacing_honoured_when_not_cooling(reg):
    r, clock = reg
    assert not r.portal("duckduckgo").cooling_down()
    p = _prov([R(202, "")], clock)
    p.gate.min_interval_s = 5.0
    p.gate.stamp(clock())
    _notes(p, clock)
    assert clock.slept and abs(clock.slept[0] - 5.0) < 1e-6
