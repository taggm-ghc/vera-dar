"""RATE-1022: offline tests for vera/rate_limit.py. Fake clock, fake responses; no network, no DB, no files written."""
import asyncio
import json
import threading
from pathlib import Path

import pytest

from vera import rate_limit as rl

ROOT = Path(__file__).resolve().parent.parent


class Clock:
    def __init__(self, t=1000.0, wall=1_760_000_000.0):
        self.t, self.w = t, wall
        self.slept = []

    def __call__(self):
        return self.t

    def wall(self):
        return self.w

    def sleep(self, s):
        self.slept.append(s)
        self.t += s
        self.w += s

    def advance(self, s):
        self.t += s
        self.w += s


class Resp:
    def __init__(self, status=200, headers=None, text=""):
        self.status_code, self.headers, self.text = status, dict(headers or {}), text


DEFAULTS = {"increase_frac": 0.1, "decrease_factor": 0.5, "min_rate_factor": 0.1, "breaker_failures": 3,
            "breaker_open_s": 10.0, "breaker_open_max_s": 40.0, "half_open_max": 1, "jitter_frac": 0.0,
            "max_wait_s": 2.0, "min_call_s": 1.0, "retry_after_max_s": 600.0, "base_cooldown_s": 30.0,
            "max_cooldown_s": 120.0, "poll_s": 0.01}
SRC = [{"url": "https://example.invalid/docs", "read": "2026-10-21"}]


def make(portal=None, clock=None, rand=lambda: 0.0, env=None, **defaults):
    p = {"hosts": ["api.example.invalid"], "rate_per_s": 1.0, "burst": 1, "max_concurrency": 2, "sources": SRC}
    p.update(portal or {})
    cfg = rl.parse_config({"defaults": {**DEFAULTS, **defaults}, "portals": {"pt": p}}, env=env or {})
    clock = clock or Clock()
    reg = rl.Registry(cfg, clock=clock, wall=clock.wall, sleep=clock.sleep, rand=rand)
    return reg, reg.portal("pt"), clock


def ok(ad, status=200, headers=None, throttled=None):
    g = ad.acquire()
    assert g.ok, g
    return g.record(status=status, headers=headers, throttled=throttled)


# ---------------------------------------------------------------- config
def test_shipped_config_loads_and_every_portal_has_a_dated_source():
    cfg = rl.load_config(ROOT / "config" / "rate_limits.json", env={})
    raw = json.loads((ROOT / "config" / "rate_limits.json").read_text())
    for name in ("openalex", "europepmc", "duckduckgo", "arxiv", "doaj", "semantic_scholar", "wikipedia", "groq"):
        assert name in cfg["portals"]
        assert all(s.get("url", "").startswith("https://") and s.get("read") for s in raw["portals"][name]["sources"])
    assert cfg["portals"]["duckduckgo"]["daily_cap"] == 8 and 202 in cfg["portals"]["duckduckgo"]["throttle_statuses"]
    assert cfg["portals"]["openalex"]["base_cooldown_s"] == 900.0
    assert all(p["contact_param"] is None for p in cfg["portals"].values())  # no email by default anywhere
    assert "@" not in (ROOT / "config" / "rate_limits.json").read_text()


def test_key_overrides_apply_only_when_env_key_present():
    base = rl.load_config(ROOT / "config" / "rate_limits.json", env={})["portals"]["openalex"]["rate_per_s"]
    keyed = rl.load_config(ROOT / "config" / "rate_limits.json", env={"OPENALEX_API_KEY": "k" * 12})
    assert keyed["portals"]["openalex"]["rate_per_s"] > base


@pytest.mark.parametrize("bad", [{"rate_per_s": 0}, {"burst": "x"}, {"sources": []}, {"throttle_statuses": [42]},
                                 {"remaining_headers": [{"reset": "x"}]}])
def test_config_validation_rejects(bad):
    with pytest.raises(rl.RateLimitConfigError):
        make(bad)


def test_contact_email_only_from_env():
    reg, _ad, _c = make({"contact_param": "mailto"})
    assert rl.contact_params("pt", env={}, registry=reg) == {}
    assert rl.contact_params("pt", env={"VERA_CONTACT_EMAIL": "not-an-email"}, registry=reg) == {}
    assert rl.contact_params("pt", env={"VERA_CONTACT_EMAIL": "a@b.org"}, registry=reg) == {"mailto": "a@b.org"}


# ---------------------------------------------------------------- token bucket / deadline
def test_token_bucket_paces_within_max_wait():
    _reg, ad, c = make({"rate_per_s": 1.0, "burst": 1})
    ok(ad)
    g = ad.acquire()  # needs 1 s, max_wait 2 s: waits via injected sleep
    assert g.ok and c.slept == [pytest.approx(1.0)]
    g.record(status=200)


def test_pacing_skip_when_wait_exceeds_max_wait():
    _reg, ad, c = make({"rate_per_s": 0.1, "burst": 1})
    ok(ad)
    s = ad.acquire()
    assert not s.ok and s.reason == "pacing" and s.retry_in_s == pytest.approx(10.0) and c.slept == []


def test_deadline_skip_never_sleeps_past_caller_budget():
    _reg, ad, c = make({"rate_per_s": 1.0, "burst": 1})
    ok(ad)
    s = ad.acquire(deadline=c.t + 1.5)  # 1.5 s left - min_call 1 s = 0.5 s budget < 1 s wait
    assert not s.ok and s.reason == "deadline" and c.slept == []
    s2 = ad.acquire(deadline=c.t + 0.5)  # less than min_call_s left: skip at once
    assert not s2.ok and s2.reason == "deadline"


def test_deadline_scope_contextvar():
    _reg, ad, c = make({"rate_per_s": 1.0, "burst": 1})
    ok(ad)
    with rl.deadline_scope(c.t + 1.2):
        assert ad.acquire().reason == "deadline"
    g = ad.acquire()
    assert g.ok
    g.release()


# ---------------------------------------------------------------- Retry-After / backoff
def test_retry_after_seconds_sets_cooldown_and_skip():
    _reg, ad, c = make({"burst": 5})
    assert ok(ad, 429, {"Retry-After": "120"}) == rl.Outcome.RATE_LIMITED
    s = ad.acquire()
    assert not s.ok and s.reason == "cooldown" and s.retry_in_s == pytest.approx(120.0)
    c.advance(121)
    assert ok(ad) == rl.Outcome.OK


def test_retry_after_http_date_and_clamp():
    _reg, ad, c = make({"burst": 5})
    from email.utils import formatdate
    ok(ad, 503, {"retry-after": formatdate(c.w + 60, usegmt=True)})
    assert ad.acquire().retry_in_s == pytest.approx(60.0, abs=1.0)
    _reg, ad2, c2 = make({"burst": 5})
    ok(ad2, 429, {"Retry-After": "999999"})
    assert ad2.acquire().retry_in_s == pytest.approx(600.0)  # retry_after_max_s


def test_backoff_doubles_without_retry_after_and_jitter_is_upward_only():
    _reg, ad, c = make({"burst": 5}, rand=lambda: 1.0, jitter_frac=0.2, breaker_failures=10)
    ok(ad, 429)
    assert ad.acquire().retry_in_s == pytest.approx(36.0)  # 30 * (1 + 0.2)
    c.advance(40)
    ok(ad, 429)
    assert ad.acquire().retry_in_s == pytest.approx(72.0)  # 60 * 1.2
    c.advance(80)
    ok(ad, 429)
    assert ad.acquire().retry_in_s == pytest.approx(144.0)  # min(120 cap,..)*1.2


# ---------------------------------------------------------------- AIMD
def test_aimd_halves_on_throttle_and_recovers_additively_up_to_base():
    _reg, ad, c = make({"rate_per_s": 4.0, "burst": 10}, breaker_failures=10)
    ok(ad, 429)
    assert ad.rate == pytest.approx(2.0)
    c.advance(31)
    ok(ad, 429, throttled=True)
    assert ad.rate == pytest.approx(1.0)
    c.advance(61)
    for _ in range(3):
        ok(ad)
    assert ad.rate == pytest.approx(1.0 + 3 * 0.4)
    for _ in range(50):
        c.advance(1)
        ok(ad)
    assert ad.rate == pytest.approx(4.0)  # never above the documented base


def test_aimd_floor_is_min_rate():
    _reg, ad, c = make({"rate_per_s": 1.0, "burst": 10}, breaker_failures=100)
    for _ in range(20):
        ok(ad, 429)
        c.advance(10_000)
    assert ad.rate == pytest.approx(0.1)


def test_caller_flagged_antibot_page_counts_as_throttle():
    reg, ad, c = make({"throttle_statuses": [429, 503, 202], "burst": 5})
    assert ok(ad, 202) == rl.Outcome.RATE_LIMITED
    reg2, ad2, _c2 = make({"burst": 5})
    w = rl.limited("pt", lambda: Resp(200, text="anomaly challenge"), classify=lambda r: "challenge" in r.text,
                   registry=reg2)
    w()
    assert ad2.n["rate_limited"] == 1 and ad2.acquire().reason == "cooldown"


# ---------------------------------------------------------------- header-driven limits
def test_remaining_zero_holds_until_reset():
    _reg, ad, c = make({"burst": 5, "remaining_headers": [
        {"remaining": "X-RateLimit-Remaining", "reset": "X-RateLimit-Reset", "reset_format": "seconds", "pace": False}]})
    ok(ad, 200, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "300"})
    s = ad.acquire()
    assert s.reason == "cooldown" and s.retry_in_s == pytest.approx(300.0)


def test_remaining_paces_rate_and_duration_reset_format():
    _reg, ad, c = make({"rate_per_s": 10.0, "burst": 5, "remaining_headers": [
        {"remaining": "x-ratelimit-remaining-requests", "reset": "x-ratelimit-reset-requests",
         "reset_format": "duration", "reserve": 0}]})
    ok(ad, 200, {"x-ratelimit-remaining-requests": "30", "x-ratelimit-reset-requests": "1m0s"})
    assert ad.rate == pytest.approx(0.5)  # 30 left over 60 s
    assert rl.parse_duration("2m59.56s") == pytest.approx(179.56)
    assert rl.parse_duration("250ms") == pytest.approx(0.25) and rl.parse_duration("garbage") is None


def test_fractional_budget_header_without_pacing():
    _reg, ad, c = make({"burst": 5, "remaining_headers": [
        {"remaining": "X-RateLimit-Remaining", "reset": "X-RateLimit-Reset", "reserve": 0, "pace": False}]})
    ok(ad, 200, {"X-RateLimit-Remaining": "0.05", "X-RateLimit-Reset": "3600"})
    assert ad.rate == pytest.approx(1.0) and ad.acquire().ok


def test_interval_header_sets_ceiling():
    _reg, ad, c = make({"rate_per_s": 50.0, "burst": 5, "interval_header":
                        {"limit": "X-Rate-Limit-Limit", "interval": "X-Rate-Limit-Interval"}})
    ok(ad, 200, {"X-Rate-Limit-Limit": "5", "X-Rate-Limit-Interval": "1s"})
    assert ad.ceiling == pytest.approx(5.0) and ad.rate == pytest.approx(5.0)
    for _ in range(5):
        c.advance(1)
        ok(ad)
    assert ad.rate == pytest.approx(5.0)


# ---------------------------------------------------------------- breaker
def test_breaker_open_half_open_probe_close():
    _reg, ad, c = make({"burst": 10, "base_cooldown_s": 0.0})
    for _ in range(3):
        ok(ad, 500)
    assert ad.snapshot()["breaker"] == "open"
    s = ad.acquire()
    assert s.reason == "breaker_open" and s.retry_in_s == pytest.approx(10.0)
    c.advance(10.5)
    probe = ad.acquire()
    assert probe.ok and probe.probe
    assert ad.acquire().reason == "half_open_busy"  # only one probe at a time
    probe.record(status=200)
    assert ad.breaker == "closed" and ad.acquire().ok


def test_breaker_failed_probe_reopens_with_doubled_window():
    _reg, ad, c = make({"burst": 10, "base_cooldown_s": 0.0})
    for _ in range(3):
        ad.acquire().record(error=OSError("boom"))
    c.advance(11)
    ad.acquire().record(status=502)
    s = ad.acquire()
    assert s.reason == "breaker_open" and s.retry_in_s == pytest.approx(20.0)
    for _ in range(5):
        c.advance(100)
        ad.acquire().record(status=502)
    c.advance(1)
    assert ad.open_s == pytest.approx(40.0)  # capped at breaker_open_max_s


# ---------------------------------------------------------------- caps, concurrency, accounting
def test_daily_cap_counts_sent_requests_including_throttled_and_rolls_over():
    _reg, ad, c = make({"burst": 10, "daily_cap": 2, "base_cooldown_s": 0.0})
    ok(ad, 429)
    ok(ad)
    s = ad.acquire()
    assert s.reason == "daily_cap" and s.retry_in_s > 0
    c.advance(86400)
    assert ad.acquire().ok


def test_concurrency_cap_skips_without_blocking_when_no_budget():
    _reg, ad, c = make({"burst": 10, "max_concurrency": 1}, max_wait_s=0.0)
    g = ad.acquire()
    s = ad.acquire()
    assert g.ok and not s.ok and s.reason == "concurrency"
    g.release()
    assert ad.acquire().ok


def test_concurrency_waiter_proceeds_when_slot_frees_threads():
    _reg, ad, c = make({"burst": 100, "rate_per_s": 100.0, "max_concurrency": 1}, poll_s=0.01)
    g = ad.acquire()
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", ad.acquire()))
    t.start()
    threading.Timer(0.05, g.release).start()
    t.join(timeout=5)
    assert out["r"].ok


def test_threaded_inflight_never_exceeds_cap():
    _reg, ad, c = make({"burst": 1000, "rate_per_s": 1000.0, "max_concurrency": 3}, max_wait_s=0.0)
    seen, lock = [], threading.Lock()
    barrier = threading.Barrier(12)

    def worker():
        barrier.wait()
        g = ad.acquire()
        with lock:
            seen.append(ad.inflight)
        if g.ok:
            g.record(status=200)

    ts = [threading.Thread(target=worker) for _ in range(12)]
    [t.start() for t in ts]
    [t.join(5) for t in ts]
    assert max(seen) <= 3 and ad.inflight == 0


def test_async_acquire_and_skip():
    _reg, ad, c = make({"burst": 1, "rate_per_s": 1.0, "max_concurrency": 1}, max_wait_s=0.0)

    async def go():
        a = await ad.acquire_async()
        b = await ad.acquire_async()
        return a, b
    a, b = asyncio.run(go())
    assert a.ok and not b.ok


def test_outcomes_and_attempt_accounting():
    assert rl.Outcome.counts_as_attempt(rl.Outcome.OK) and rl.Outcome.counts_as_attempt(rl.Outcome.FAILED)
    assert not rl.Outcome.counts_as_attempt(rl.Outcome.RATE_LIMITED)
    assert not rl.Outcome.counts_as_attempt(rl.Outcome.SKIPPED)
    _reg, ad, c = make({"burst": 5})
    ok(ad, 429, {"Retry-After": "5"})
    ad.acquire()  # skip (cooldown): not a request
    snap = ad.snapshot()
    assert snap["requests"] == 1 and snap["rate_limited"] == 1 and snap["skips"] == {"cooldown": 1}


# ---------------------------------------------------------------- wrappers and telemetry
def test_limited_wrapper_raises_rate_limited_and_does_not_call():
    reg, ad, c = make({"burst": 5})
    calls = []
    w = rl.limited("pt", lambda *a, **k: calls.append(1) or Resp(429, {"Retry-After": "60"}), registry=reg)
    assert w("u").status_code == 429
    with pytest.raises(rl.RateLimited) as ei:
        w("u")
    assert len(calls) == 1 and "rate limit" in str(ei.value) and ei.value.reason == "cooldown"


def test_wrapper_records_transport_errors_and_reraises():
    reg, ad, c = make({"burst": 5})

    def boom(*a, **k):
        raise ConnectionError("x")
    with pytest.raises(ConnectionError):
        rl.limited("pt", boom, registry=reg)()
    assert ad.n["failed"] == 1 and ad.inflight == 0


def test_limited_by_url_routes_by_host_and_passes_unknown_hosts():
    reg, ad, c = make({"burst": 5})
    w = rl.limited_by_url(lambda url, **k: Resp(200), registry=reg)
    w("https://api.example.invalid/x", timeout=1)
    w("https://other.example.invalid/y", timeout=1)
    assert ad.n["requests"] == 1


def test_snapshot_has_no_hosts_or_secrets():
    reg, ad, c = make({"burst": 5}, env={})
    ok(ad, 429, {"Retry-After": "1"})
    text = json.dumps(reg.snapshot()) + reg.log_line()
    assert "example.invalid" not in text and "http" not in text
    assert reg.log_line().startswith("ratelimit pt=req:1,429:1")


def test_httpx_transport_wraps_inner_and_reads_headers():
    httpx = pytest.importorskip("httpx")
    reg, ad, c = make({"burst": 5, "throttle_statuses": [429]})

    class Inner(httpx.BaseTransport):
        def handle_request(self, request):
            return httpx.Response(429, headers={"retry-after": "7"}, request=request)
    t = rl.httpx_transport("pt", Inner(), registry=reg)
    with httpx.Client(transport=t) as cl:
        assert cl.get("https://api.example.invalid/v1").status_code == 429
        with pytest.raises(rl.RateLimited):
            cl.get("https://api.example.invalid/v1")
    assert ad.n["rate_limited"] == 1
