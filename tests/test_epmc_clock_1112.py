"""EPMC-CLOCK-1112: our own rate-limit wait is not per-fetch time and is never cached as a robots TIMEOUT.

Plan: work/EPMC-CLOCK-1112/PLAN.md (Verify (a)-(g)). Offline: fake clock and fake sleep shared by the limiter registry,
acq_discovered._sleep (crawl_wait) and the per-fetch clocks; real config/rate_limits.json europepmc portal (0.1/s,
burst 1, concurrency 1) and the real api_permissions.limited_transport; an instant (or deliberately slow) fake server.
Promoted from the ASK-SUPERSTR-1112 repro, where the first www.ebi.ac.uk robots.txt read waited 10 s in the limiter
against a 5 s clock, was cached as TIMEOUT, and every later candidate got CACHED_TIMEOUT with nothing sent."""
from __future__ import annotations

import pytest

from tests import acq_support as A
from vera import acq_config, acq_discovered, api_permissions, auto_acquire, rate_limit

HOST = "www.ebi.ac.uk"
SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.9999/epmc.clock{}"
ROBOTS_PLAIN = b"User-agent: *\nAllow: /\n"
ROBOTS_CRAWL = b"User-agent: *\nCrawl-delay: 10\nAllow: /\n"


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps: list[float] = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(float(s))
        self.t += max(0.0, s)


class World:
    """One test's clock, registry, transport and server log. `slow` = seconds the server takes per matching URL."""

    def __init__(self, monkeypatch, robots=ROBOTS_PLAIN, slow=None):
        acq_discovered.reset_for_tests()
        self.clk = Clock()
        self.reg = rate_limit.Registry(rate_limit.load_config(), clock=self.clk, wall=lambda: 1.7e9,
                                       sleep=self.clk.sleep, rand=lambda: 0.5)
        rate_limit.set_registry_for_tests(self.reg)
        monkeypatch.setattr(acq_discovered, "_sleep", self.clk.sleep, raising=False)
        self.robots, self.slow = robots, dict(slow or {})
        self.sent: list[tuple[float, str]] = []
        self.http = api_permissions.limited_transport(self.server, strict=True)

    def server(self, url, headers=None, timeout=None, allow_redirects=None, stream=None, **kw):
        self.sent.append((self.clk.t, url))
        for key, secs in self.slow.items():
            if key in url:
                self.clk.t += secs  # server time: spent after the limiter granted the slot
        if url.endswith("/robots.txt"):
            return A.Resp(200, self.robots, "text/plain")
        if "tdmrep" in url:
            return A.Resp(404, b"", "application/json")
        return A.Resp(200, b'{"resultList": {"result": []}}', "application/json")

    def use_token(self):
        """As the /ask Europe PMC search does just before acquisition: the portal's one token is spent now."""
        p = self.reg.portal("europepmc").acquire(None)
        assert isinstance(p, rate_limit.Permit)
        p.record(status=200, headers={})

    def get(self, n, cfg, queue_s=25.0, deadline_s=45.0, timeout_s=20.0):
        deps = A.deps([], self.http)
        with rate_limit.queue_scope(self.clk() + queue_s):
            return auto_acquire._guarded_get(SEARCH.format(n), cfg, deps, 100000, timeout_s, True, clock=self.clk,
                                             deadline=self.clk() + deadline_s, route_hop=True)

    def code(self, n, cfg, **kw):
        with pytest.raises(auto_acquire._Stop) as e:
            self.get(n, cfg, **kw)
        return e.value.code

    def urls(self):
        return [u for _t, u in self.sent]


@pytest.fixture(autouse=True)
def _reset():
    yield
    rate_limit.set_registry_for_tests(None)
    acq_discovered.reset_for_tests()


def _not_negative_cached():
    hit = acq_discovered._ROBOTS.get(HOST)
    assert hit is not None and hit[1] is not False, hit
    tdm = acq_discovered._TDMREP.get(HOST)
    assert tdm is None or tdm[1] != "unreadable", tdm


# (a) + (b, shipped budget) --------------------------------------------------------------------------------------
def test_limiter_wait_on_robots_read_is_not_a_timeout_and_is_not_cached(monkeypatch):
    w = World(monkeypatch)
    cfg = A.cfg()
    w.use_token()
    first = w.code(184, cfg)  # robots.txt at +10 s, tdmrep.json at +20 s, the search's slot (+30 s) misses the 25 s queue
    assert any(abs(s - 10.0) < 0.01 for s in w.clk.sleeps), w.clk.sleeps  # non-vacuous: the limiter really waited
    assert "TIMEOUT" not in first, first
    assert first.startswith(acq_config.reason(cfg, "rate_limited")), first  # our pacing: a rate-limit skip
    assert w.urls() == [f"https://{HOST}/robots.txt", f"https://{HOST}/.well-known/tdmrep.json"]
    _not_negative_cached()
    # a second candidate within robots_fail_ttl_s: not CACHED_TIMEOUT, and its request IS sent
    n = len(w.sent)
    raw, final = w.get(185, cfg)
    assert final == SEARCH.format(185) and b"resultList" in raw
    assert len(w.sent) == n + 1 and w.sent[-1][1] == SEARCH.format(185)


# (b) budget that fits the three paced requests ------------------------------------------------------------------
def test_first_candidate_is_fetched_when_the_budget_fits_three_paced_requests(monkeypatch):
    w = World(monkeypatch)
    w.use_token()
    raw, final = w.get(184, A.cfg(), queue_s=45.0, deadline_s=60.0, timeout_s=5.0)
    assert final == SEARCH.format(184) and b"resultList" in raw
    assert [u for u in w.urls()] == [f"https://{HOST}/robots.txt", f"https://{HOST}/.well-known/tdmrep.json",
                                     SEARCH.format(184)]
    _not_negative_cached()


# (c) spacing is unchanged and comes from the limiter alone -------------------------------------------------------
@pytest.mark.parametrize("crawl_on", [True, False])
def test_limiter_still_spaces_requests_ten_seconds(monkeypatch, crawl_on):
    w = World(monkeypatch, robots=ROBOTS_CRAWL)
    cfg = A.cfg()
    cfg["crawl_delay"]["enforce_on_acquisition"] = crawl_on
    w.use_token()
    t0 = w.clk()
    for n in (1, 2, 3):
        w.get(n, cfg, queue_s=45.0, deadline_s=60.0, timeout_s=5.0)
    times = [t0] + [t for t, _u in w.sent]  # t0 = the search's token
    assert len(w.sent) == 5  # robots, tdmrep, then one request per candidate
    assert all(b - a >= 10.0 - 1e-9 for a, b in zip(times, times[1:])), times


def test_send_times_identical_with_and_without_a_tally(monkeypatch):
    def run(tally):
        clk = Clock()
        reg = rate_limit.Registry(rate_limit.load_config(), clock=clk, wall=lambda: 1.7e9, sleep=clk.sleep,
                                  rand=lambda: 0.5)
        rate_limit.set_registry_for_tests(reg)
        sent = []
        http = api_permissions.limited_transport(lambda u, **kw: sent.append(clk.t) or A.Resp(200, b"{}"),
                                                 strict=True)
        for n in range(4):
            with rate_limit.deadline_scope(clk() + 60), rate_limit.queue_scope(clk() + 60):
                if tally:
                    with rate_limit.wait_tally():
                        http(SEARCH.format(n))
                else:
                    http(SEARCH.format(n))
        return sent
    assert run(True) == run(False) == [1000.0, 1010.0, 1020.0, 1030.0]


# (d) a genuinely slow server still times out and is cached ------------------------------------------------------
@pytest.mark.parametrize("limiter_waits", [False, True])
def test_slow_server_on_robots_read_still_times_out_and_is_cached(monkeypatch, limiter_waits):
    w = World(monkeypatch, slow={"/robots.txt": 6.0})
    cfg = A.cfg()
    if limiter_waits:
        w.use_token()
    first = w.code(184, cfg)
    assert first.endswith("-TIMEOUT") and "CACHED" not in first, first
    assert acq_discovered._ROBOTS[HOST][1] is False and acq_discovered._ROBOTS[HOST][2] == "TIMEOUT"
    n = len(w.sent)
    second = w.code(185, cfg)
    assert second.endswith("-CACHED_TIMEOUT"), second
    assert len(w.sent) == n  # nothing sent: the real failure is still remembered


def test_nested_gate_tallies_do_not_hide_server_time(monkeypatch):
    # robots.txt and tdmrep.json each wait 10 s in the limiter (their own tallies, credited as gate time), the search
    # itself waits 10 s too (credited), then the server takes 6 s against a 5 s clock: still a fetch timeout.
    w = World(monkeypatch, slow={"/rest/search": 6.0})
    cfg = A.cfg()
    w.use_token()
    code = w.code(184, cfg, queue_s=45.0, deadline_s=60.0, timeout_s=5.0)
    assert code == acq_config.reason(cfg, "fetch_timeout"), code
    assert w.urls()[-1] == SEARCH.format(184)
    assert [round(s, 3) for s in w.clk.sleeps].count(10.0) == 3, w.clk.sleeps
    _not_negative_cached()  # the gate reads succeeded; only the document fetch was slow


# (e) the document clock of _guarded_get ----------------------------------------------------------------------------
def test_guarded_get_credits_the_limiter_wait(monkeypatch):
    w = World(monkeypatch)
    A.seed_access(HOST, w.clk())
    w.use_token()
    raw, final = w.get(184, A.cfg(), timeout_s=5.0)
    assert final == SEARCH.format(184) and b"resultList" in raw
    assert w.clk.sleeps == [10.0]


# (f) rate_limit unit ---------------------------------------------------------------------------------------------
def test_wait_tally_unit():
    clk = Clock()
    reg = rate_limit.Registry(rate_limit.load_config(), clock=clk, wall=lambda: 1.7e9, sleep=clk.sleep,
                              rand=lambda: 0.5)
    ad = reg.portal("europepmc")
    p = ad.acquire(None)  # no tally open: no effect, no error
    p.record(status=200, headers={})
    with rate_limit.queue_scope(clk() + 3600), rate_limit.request_scope() as led, rate_limit.wait_tally() as outer:
        with rate_limit.wait_tally() as inner:
            p = ad.acquire(clk() + 30)
            assert isinstance(p, rate_limit.Permit)
            p.record(status=200, headers={})
        assert inner[0] == pytest.approx(10.0) and outer[0] == 0.0  # nested tallies are separate
        p = ad.acquire(clk() + 30)
        p.record(status=200, headers={})
        assert outer[0] == pytest.approx(10.0)
        assert led.summary()["waited_s"] == pytest.approx(20.0)  # ledger unchanged: both permits' waits
        with rate_limit.wait_tally() as skipped:
            s = ad.acquire(clk() + 2)  # needs 10 s, has 2: a skip
            assert not s.ok
        assert skipped[0] == 0.0


# (g) a deadline shorter than the limiter wait ----------------------------------------------------------------------
def test_deadline_shorter_than_limiter_wait_is_a_passthrough_not_cached(monkeypatch):
    w = World(monkeypatch)
    cfg = A.cfg()
    w.use_token()
    code = w.code(184, cfg, queue_s=4.0, deadline_s=4.0)
    assert code.startswith(acq_config.reason(cfg, "rate_limited")) or code == acq_config.reason(cfg, "deadline"), code
    assert "TIMEOUT" not in code
    assert HOST not in acq_discovered._ROBOTS and w.sent == []
