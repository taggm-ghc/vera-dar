"""ROBOTS-LABEL-1111 (WP-L) tests 20, 23, 24: the web route (public_fetch) and the portal gate (selective_fetch) apply
R1 (OD-1111-ROBOTS-4XX, owner 2026-11-11): only a robots.txt 404 or 410 is "no restrictions"; any other 4xx stops. An
unreadable robots.txt is its own refusal ("robots_unreadable"), not "robots". Offline: DNS, transport and http_get are
fakes."""
from __future__ import annotations

import pytest

from tests.test_public_fetch_1103 import CFG, Net, Resp, html, robots
from vera import public_fetch as pf
from vera import selective_fetch as sf
from vera import web_route as wr

DNS = {"a.example": ["93.184.216.34"]}


@pytest.fixture(autouse=True)
def _clean_sf_cache():
    sf.robots_cache_clear()  # selective_fetch._ROBOTS is a process cache the conftest does not reset
    yield
    sf.robots_cache_clear()


def _fetch(pages):
    net = Net(DNS, pages)
    try:
        return net, pf.fetch("https://a.example/p", CFG, resolve=net.resolve, transport=net.transport)
    except pf.FetchRefused as e:
        return net, e


def _robots_calls(net):
    return sum(1 for c in net.calls if c[3] == "/robots.txt")


def test_t20_unreadable_is_its_own_refusal_and_a_rule_is_still_robots():
    net, e = _fetch({("a.example", "/robots.txt"): Resp(500, {"Content-Type": "text/plain"}, b""),
                     ("a.example", "/p"): html()})
    assert isinstance(e, pf.FetchRefused) and e.reason == "robots_unreadable" and not any(c[3] == "/p" for c in net.calls)
    net, e = _fetch({("a.example", "/robots.txt"): robots(b"User-agent: *\nDisallow: /\n"), ("a.example", "/p"): html()})
    assert e.reason == "robots"
    net, r = _fetch({("a.example", "/robots.txt"): Resp(404, {}, b""), ("a.example", "/p"): html()})
    assert r["text"] == "hello"  # 404 = no restrictions (unchanged)


def test_t20b_robots_403_is_now_refused_R1():
    # R1 / OD-1111-ROBOTS-4XX: FLIPPED from "robots 403 = allowed" (RFC 9309 s2.3.1.3): a recorded deviation.
    net, e = _fetch({("a.example", "/robots.txt"): Resp(403, {"Content-Type": "text/html"}, b""),
                     ("a.example", "/p"): html()})
    assert isinstance(e, pf.FetchRefused) and e.reason == "robots_unreadable"
    assert not any(c[3] == "/p" for c in net.calls)


@pytest.mark.parametrize("bad", [None, "", "maybe", 0, "deny"])
def test_t20c_unexpected_robots_check_values_are_refused(bad):
    net = Net(DNS, {("a.example", "/p"): html()})
    with pytest.raises(pf.FetchRefused) as e:
        pf._get("https://a.example/p", CFG, net.resolve, net.transport, lambda: 0.0, 100.0, ("text/html",),
                lambda u, h, p: bad, 5000, 20000)
    assert e.value.reason == "robots" and net.calls == []


def test_t20d_allow_values_pass():
    for ok in (True, "allow"):
        net = Net(DNS, {("a.example", "/p"): html()})
        body, _u, st, _c = pf._get("https://a.example/p", CFG, net.resolve, net.transport, lambda: 0.0, 100.0,
                                   ("text/html",), lambda u, h, p: ok, 5000, 20000)
        assert st == 200 and body


@pytest.mark.parametrize("status", [400, 401, 403, 405, 451])
def test_t23_public_fetch_other_4xx_is_unreadable(status):
    net, e = _fetch({("a.example", "/robots.txt"): Resp(status, {"Content-Type": "text/html"}, b""),
                     ("a.example", "/p"): html()})
    assert isinstance(e, pf.FetchRefused) and e.reason == "robots_unreadable"


@pytest.mark.parametrize("status", [404, 410])
def test_t23b_404_410_allow_and_one_robots_request_per_fetch(status):
    net, r = _fetch({("a.example", "/robots.txt"): Resp(status, {}, b""), ("a.example", "/p"): html()})
    assert r["text"] == "hello" and _robots_calls(net) == 1


def test_t23c_html_robots_file_is_unreadable_Q3():
    # Q-3: a 200 text/html answer is a web page, not a rules file; "non-text still stops" (previous: parsed as rules)
    net, e = _fetch({("a.example", "/robots.txt"): Resp(200, {"Content-Type": "text/html"}, b"<html>x</html>"),
                     ("a.example", "/p"): html()})
    assert isinstance(e, pf.FetchRefused) and e.reason == "robots_unreadable"


def test_t23d_a_4xx_stop_is_cached_for_later_hops_of_one_request():
    cache: dict = {}
    net = Net(DNS, {("a.example", "/robots.txt"): Resp(403, {}, b""), ("a.example", "/p"): html()})
    for _ in range(2):
        with pytest.raises(pf.FetchRefused):
            pf.fetch("https://a.example/p", CFG, resolve=net.resolve, transport=net.transport, robots_cache=cache)
    assert _robots_calls(net) == 1  # the same request's later hop sends no second robots.txt request (RC-1)


def test_t23e_web_route_robots_unreadable_phrase_and_gate():
    net = Net(DNS, {("a.example", "/robots.txt"): Resp(500, {}, b"")})

    def f(url, **kw):
        return pf.fetch(url, CFG, resolve=net.resolve, transport=net.transport, **kw)
    r = wr.read_web("q", [{"url": "https://a.example/p", "title": "T", "snippet": "s", "acq_source": "ddg"}], fetch=f,
                    registry=[], deny_list=[])
    assert r.dev[0]["gate"]["read"] == "read_robots_unreadable"
    assert r.listed[0]["phrase"] == "not read: the site's robots rules could not be read"


class _R:
    def __init__(self, status, body=b"", ctype="text/plain"):
        self.status_code, self.headers, self._b = status, {"Content-Type": ctype}, body

    def iter_content(self, n):
        yield self._b

    def close(self):
        pass


class _NoWait:
    def wait(self, url):
        return 0.0


def _sf(status, body=b"", clock=lambda: 1000.0, url="https://p.example/page", calls=None):
    def get(cur, **kw):
        if calls is not None:
            calls.append(cur)
        return _R(status, body)
    return sf.robots_allowed(url, http_get=get, url_check=lambda u: (True, ""), pacer=_NoWait(), clock=clock)


@pytest.mark.parametrize("status,allowed", [(404, True), (410, True), (400, False), (401, False), (403, False),
                                            (405, False), (451, False), (429, False), (500, False), (503, False)])
def test_t24_selective_fetch_robots_status_policy(status, allowed):
    # R1 / OD-1111-ROBOTS-4XX: FLIPPED for 400/401/403/405/451 (before: any non-429 4xx = allowed)
    assert _sf(status) is allowed


def test_t24b_4xx_stop_held_for_robots_ttl_and_5xx_for_fail_ttl():
    # The real TTL here is sf.ROBOTS_TTL_S (300 s), not 3600 s: the plan's "3599 s for selective_fetch" in test 18 was a slip
    # (3600 s is acq_discovered's robots_ttl_s, asserted in the gate file); this test asserts the real constants.
    assert sf.ROBOTS_TTL_S == 300.0
    now = [1000.0]
    calls: list = []
    assert _sf(403, clock=lambda: now[0], calls=calls) is False
    now[0] = 1000.0 + sf.ROBOTS_FAIL_TTL_S + 1  # past the 60 s fail TTL: still held (no extra request)
    assert _sf(403, clock=lambda: now[0], calls=calls) is False and len(calls) == 1
    now[0] = 1000.0 + sf.ROBOTS_TTL_S + 1
    assert _sf(403, clock=lambda: now[0], calls=calls) is False and len(calls) == 2
    sf.robots_cache_clear()
    calls2: list = []
    now[0] = 2000.0
    assert _sf(500, clock=lambda: now[0], calls=calls2) is False
    now[0] = 2000.0 + sf.ROBOTS_FAIL_TTL_S + 1
    assert _sf(500, clock=lambda: now[0], calls=calls2) is False and len(calls2) == 2


def test_t24c_two_tuple_false_entry_keeps_the_fail_ttl_without_error():
    sf._ROBOTS[("https", "p.example", sf.USER_AGENT)] = (1000.0, False)  # an older fixture shape
    calls: list = []
    assert _sf(404, clock=lambda: 1000.0 + 10, calls=calls) is False and calls == []
    assert _sf(404, clock=lambda: 1000.0 + 61, calls=calls) is True and len(calls) == 1


def test_t24d_consumers_refuse_on_a_robots_403():
    # the DuckDuckGo gate, reference_portal and wiki_ref_portal all call selective_fetch.robots_allowed
    import inspect

    from vera import reference_portal, search_providers, wiki_ref_portal
    for mod in (reference_portal, search_providers, wiki_ref_portal):
        assert "sf.robots_allowed(" in inspect.getsource(mod)
    assert _sf(403, url="https://duckduckgo.example/html") is False


def test_t24e_robots_429_and_url_check_and_redirect_exhaustion_have_their_own_cause_lines(capsys):
    # D-3 (verifier): log text only; the stop itself is unchanged
    assert _sf(429) is False
    sf.robots_cache_clear()
    assert sf.robots_allowed("https://p.example/page", http_get=lambda *a, **k: _R(200), url_check=lambda u: (False, ""),
                             pacer=_NoWait(), clock=lambda: 1000.0) is False
    sf.robots_cache_clear()

    def loop(cur, **kw):
        r = _R(302)
        r.headers["Location"] = "/robots2"
        return r
    assert sf.robots_allowed("https://p.example/page", http_get=loop, url_check=lambda u: (True, ""), pacer=_NoWait(),
                             clock=lambda: 1000.0) is False
    out = capsys.readouterr().out
    assert "cause=RATELIMITED" in out and "cause=HOST" in out and "cause=REDIRECTS" in out


def test_t24f_public_fetch_robots_429_is_ratelimited_in_the_cause_line(capsys):
    net, e = _fetch({("a.example", "/robots.txt"): Resp(429, {}, b""), ("a.example", "/p"): html()})
    assert e.reason == "robots_unreadable" and "cause=RATELIMITED" in capsys.readouterr().out


def test_t23f_robots_without_a_content_type_is_unreadable():
    # deviation 6b: beyond the plan's wording, a 200 robots.txt with NO Content-Type is refused (only removes requests)
    net, e = _fetch({("a.example", "/robots.txt"): Resp(200, {}, b"User-agent: *\nAllow: /\n"), ("a.example", "/p"): html()})
    assert isinstance(e, pf.FetchRefused) and e.reason == "robots_unreadable"
