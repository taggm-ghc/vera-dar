"""REFPORTAL-CLOCK-1112: our own rate-limit wait is not per-fetch time in the reference portals.

Plan: work/REFPORTAL-CLOCK-1112/PLAN.md (Verify (a)-(g)). Sibling of EPMC-CLOCK-1112 (tests/test_epmc_clock_1112.py).
Offline: one fake clock and fake sleep shared by a limiter Registry built from the real config/rate_limits.json and by
the per-fetch clocks; the real rate_limit.limited (reference_portal) and api_permissions.limited_transport
(wiki_ref_portal); a fake server that answers at once or deliberately slowly. SYNTHETIC values only.
Defect: reference_portal._tdmrep_ok started its clock before the limiter's own wait, so a paced tdmrep.json read could
end as "deadline", be cached as "unreadable" for 3600 s and refuse the host (tdmrep_unreadable_is_reserved)."""
from __future__ import annotations

import copy
import json
import pathlib

import pytest

from vera import api_permissions, rate_limit
from vera import reference_portal as rp
from vera import wiki_ref_portal as w

ROOT = pathlib.Path(__file__).resolve().parents[1]
HOST = "en.wikipedia.org"
PAGE_URL = f"https://{HOST}/wiki/Zorblats"
TDM_URL = f"https://{HOST}/.well-known/tdmrep.json"
PAGE = b"<html><head><title>SYNTHETIC</title></head><body><p>synthetic zorblat page</p></body></html>"
TDM_RULES = json.dumps([{"location": "/private/*", "tdm-reservation": 1}]).encode()  # /wiki/ not reserved
OK = lambda u: (True, "")  # noqa: E731


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps: list[float] = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(float(s))
        self.t += max(0.0, s)


class Resp:
    def __init__(self, status=200, body=b"", ctype="text/html; charset=utf-8"):
        self.status_code = status
        self._body = body
        self.headers = {"Content-Type": ctype}

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]

    def close(self):
        pass


class World:
    """One test's clock, registry and server log. rate = wikipedia portal rate (None = shipped 1/s); slow = server
    seconds per URL substring (spent after the limiter granted the slot, inside the call)."""

    def __init__(self, rate=None, slow=None):
        rp.reset_for_tests()
        self.clk = Clock()
        cfg = copy.deepcopy(rate_limit.load_config())
        if rate is not None:
            cfg["portals"]["wikipedia"]["rate_per_s"] = float(rate)
        self.reg = rate_limit.Registry(cfg, clock=self.clk, wall=lambda: 1.7e9, sleep=self.clk.sleep,
                                       rand=lambda: 0.5)
        rate_limit.set_registry_for_tests(self.reg)
        self.slow = dict(slow or {})
        self.sent: list[tuple[float, str]] = []
        self.order: list[str] = []

    def server(self, url, headers=None, timeout=None, allow_redirects=None, stream=None, **kw):
        assert allow_redirects is False
        self.sent.append((self.clk.t, url))
        self.order.append(url)
        for key, secs in self.slow.items():
            if key in url:
                self.clk.t += secs
        if url == TDM_URL:
            return Resp(200, TDM_RULES, "application/json")
        return Resp(200, PAGE)

    def use_token(self):
        """Another request (e.g. the previous portal page) has just spent the portal's one token."""
        p = self.reg.portal("wikipedia").acquire(None)
        assert isinstance(p, rate_limit.Permit)
        p.record(status=200, headers={})

    def waited(self, about):
        return any(abs(s - about) < 0.6 for s in self.clk.sleeps)

    def tdm(self, cfg):
        lim = lambda portal, fn: rate_limit.limited(portal, fn)  # noqa: E731  (production default)
        return rp._tdmrep_ok(PAGE_URL, cfg, self.server, OK, clock=self.clk, limited=lim, rate_portal="wikipedia")

    def deps(self, cfg):
        def robots(u, ua):
            self.order.append("robots:" + u)
            return True
        return rp.Deps(http_get=self.server, url_check=OK, robots_ok=robots, no_ai_input=lambda u, ua: False,
                       tdmrep_ok=lambda u, portal: rp._tdmrep_ok(u, cfg, self.server, OK, clock=self.clk,
                                                                limited=lambda p, fn: rate_limit.limited(p, fn),
                                                                rate_portal=portal),
                       crawl_ok=lambda u, ua: True, limited=lambda p, fn: rate_limit.limited(p, fn))

    def page(self, cfg, deadline=None):
        return rp.fetch_page(PAGE_URL, {"rate_portal": "wikipedia"}, cfg, self.deps(cfg), self.clk, deadline)

    def article(self, deadline=None):
        get = api_permissions.limited_transport(self.server, fixed_portal="wikipedia")
        c = copy.deepcopy(w.load_config())
        c["enabled"] = True  # shipped off; tests opt in on a copy
        return w.get_article(PAGE_URL, c, http_get=get, url_check=OK, clock=self.clk, deadline=deadline)


@pytest.fixture(autouse=True)
def _reset():
    yield
    rate_limit.set_registry_for_tests(None)
    rp.reset_for_tests()


def rcfg(**caps):
    c = json.loads((ROOT / "config" / "reference_portals.json").read_text(encoding="utf-8"))
    c["caps"].update(caps)
    return c


def _cached_rules():
    hit = rp._TDM.get(HOST)
    assert hit is not None
    return hit[1]


# (a) _tdmrep_ok: the cached negative --------------------------------------------------------------------------------
def test_tdmrep_after_a_long_limiter_wait_is_read_and_not_cached_unreadable():
    wd = World(rate=0.1)
    wd.use_token()
    with rate_limit.stream_wait_scope():
        assert wd.tdm(rcfg()) is True
    assert wd.waited(10.0), wd.clk.sleeps  # non-vacuous: the limiter really waited longer than caps.timeout_s 5
    assert _cached_rules() != "unreadable"
    n = len(wd.sent)
    assert wd.tdm(rcfg()) is True  # within _TDM_TTL_S: cached, nothing sent
    assert len(wd.sent) == n


def test_tdmrep_shipped_portal_limiter_wait_plus_server_time_within_timeout():
    """Shipped 1/s portal and 5 s timeout: ~1 s of our pacing plus 4.5 s of server time is within the host's 5 s."""
    wd = World(slow={"tdmrep": 4.5})
    wd.use_token()
    assert wd.tdm(rcfg()) is True
    assert wd.waited(1.0), wd.clk.sleeps
    assert _cached_rules() != "unreadable"


# (b) fetch_page / (c) get_article --------------------------------------------------------------------------------
def test_fetch_page_after_a_long_limiter_wait_returns_the_page():
    wd = World(rate=0.1)
    wd.use_token()
    with rate_limit.stream_wait_scope():
        html = wd.page(rcfg())
    assert "synthetic zorblat page" in html
    assert sum(1 for s in wd.clk.sleeps if abs(s - 10.0) < 0.6) == 2, wd.clk.sleeps  # tdmrep slot, then page slot


def test_get_article_after_a_long_limiter_wait_returns_the_page():
    wd = World(rate=0.1)
    wd.use_token()
    with rate_limit.stream_wait_scope():
        html = wd.article()
    assert "synthetic zorblat page" in html and wd.waited(10.0)


# (d) a genuinely slow host still fails (and tdmrep is still cached unreadable) ---------------------------------------
@pytest.mark.parametrize("rate,token", [(None, False), (0.1, True)])
def test_slow_tdmrep_host_still_fails_closed_and_is_cached(rate, token):
    wd = World(rate=rate, slow={"tdmrep": 5.5})
    if token:
        wd.use_token()
    with rate_limit.stream_wait_scope():
        assert wd.tdm(rcfg()) is False
    assert _cached_rules() == "unreadable"


@pytest.mark.parametrize("rate,token", [(None, False), (0.1, True)])
def test_slow_page_host_still_times_out(rate, token):
    wd = World(rate=rate, slow={"/wiki/": 5.5})
    if token:
        wd.use_token()
    with rate_limit.stream_wait_scope():
        with pytest.raises(rp.Skip) as e:
            wd.page(rcfg())
        assert str(e.value) == "deadline"
        with pytest.raises(w.PortalError) as e2:
            wd.article()
        assert str(e2.value) == "deadline"


def test_tdmrep_wait_is_not_credited_to_the_page_clock():
    """fetch_page's tdmrep read waited 10 s in the limiter and the page slot waited 10 s too: 5.5 s of page server
    time still times out (the tdmrep tally does not leak into the page clock)."""
    wd = World(rate=0.1, slow={"/wiki/": 5.5})
    wd.use_token()
    with rate_limit.stream_wait_scope():
        with pytest.raises(rp.Skip) as e:
            wd.page(rcfg())
    assert str(e.value) == "deadline"
    assert sum(1 for s in wd.clk.sleeps if abs(s - 10.0) < 0.6) == 2


# (e) the caller deadline still caps the credit -----------------------------------------------------------------------
@pytest.mark.parametrize("server_s,ok", [(1.5, True), (2.5, False)])
def test_caller_deadline_caps_the_credit(server_s, ok):
    for which in ("page", "article"):
        wd = World(rate=0.1, slow={"/wiki/": server_s})
        cfg = rcfg()
        with rate_limit.stream_wait_scope():
            if which == "page":
                assert wd.tdm(cfg) is True  # tdmrep read first (and cached) so the page slot is the only wait below
            wd.use_token()
            deadline = wd.clk() + 12.0  # 10 s wait (stream budget 30 s); end capped at deadline: 10 + server_s vs 12
            call = (lambda: wd.page(cfg, deadline)) if which == "page" else (lambda: wd.article(deadline))
            if ok:
                assert "synthetic zorblat page" in call()
            else:
                with pytest.raises((rp.Skip, w.PortalError)) as e:
                    call()
                assert str(e.value) == "deadline"
        assert wd.waited(10.0)


# (f) spacing: no request faster than config ----------------------------------------------------------------------
def test_requests_to_the_host_stay_spaced_by_the_portal_rate():
    wd = World(rate=0.1)
    wd.use_token()
    t0 = wd.clk()
    with rate_limit.stream_wait_scope():
        wd.page(rcfg())
        wd.article()
    times = [t0] + [t for t, _u in wd.sent]
    assert len(wd.sent) == 3
    assert all(b - a >= 10.0 - 1e-9 for a, b in zip(times, times[1:])), times  # token spacing is exactly 1/rate


# (g) robots order -------------------------------------------------------------------------------------------------
def test_robots_is_checked_before_any_request_to_the_host():
    wd = World()
    wd.page(rcfg())
    assert wd.order[0] == "robots:" + PAGE_URL
    assert wd.order[1:] == [TDM_URL, PAGE_URL]
