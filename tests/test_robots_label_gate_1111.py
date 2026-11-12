"""ROBOTS-LABEL-1111 (WP-L) gate tests 1-9, 14, 17-18: robots.txt read failures are "unreadable" (own code and cause),
parsed rules stay R-ACQ-ROBOTS, and the owner rulings hold: R1 (OD-1111-ROBOTS-4XX) only 404/410 are unrestricted,
any other robots.txt 4xx fails closed; no request is added by the change (RC-1: a 4xx stop is cached as long as the old
"allow" was). Fake HTTP only (tests/acq_support.Web); no network, no DB."""
from __future__ import annotations

import pytest

from tests import acq_support as A
from vera import acq_discovered, auto_acquire

EBI = "www.ebi.ac.uk"
JATS = f"https://{EBI}/europepmc/webservices/rest/PMC1/fullTextXML"
ROBOTS = "/robots.txt"
PUB = "pub.example"
Resp = A.Resp


def _txt(body: str = "", status: int = 200, ctype: str = "text/plain", headers=None):
    return Resp(status=status, body=body.encode(), ctype=ctype, headers=headers)


def _web(over):
    return A.Web({}, {"PMC1": b"<article/>"}, over)


def _gg(web, clock=None, route_hop=False):
    return auto_acquire._guarded_get(JATS, A.cfg(), A.deps([], web), 100000, 5.0, False,
                                     clock=clock or A.clock_at(), route_hop=route_hop)


def _code(web, **kw):
    with pytest.raises(auto_acquire._Stop) as e:
        _gg(web, **kw)
    return e.value.code


def _verdict(web, host=PUB, clock=None):
    return acq_discovered._robots_verdict(f"https://{host}/article/1", {host}, A.deps([], web), A.cfg(),
                                          clock or A.clock_at(), None)


def test_t00_cache_is_empty_at_start_so_no_probe_passes_vacuously():
    assert acq_discovered._ROBOTS == {} and acq_discovered._DENIED == {}


@pytest.mark.parametrize("status,cause", [(500, "STATUS_5XX"), (502, "STATUS_5XX")])
def test_t01_5xx_is_unreadable_with_cause(status, cause):
    web = _web({ROBOTS: _txt("", status)})
    assert _code(web) == f"R-ACQ-ROBOTS_UNREADABLE-{cause}" and A.doc_urls(web.urls) == []


def test_t01b_ebi_503_is_rate_limited_not_robots():
    # 503 is in the europepmc throttle_statuses (config/rate_limits.json): a rate limit, never a robots code
    assert _code(_web({ROBOTS: _txt("", 503)})) == "R-ACQ-RATELIMITED"


@pytest.mark.parametrize("exc", [ConnectionError("down")])
def test_t02_network_error_is_unreadable_network(exc):
    assert _code(_web({ROBOTS: exc})) == "R-ACQ-ROBOTS_UNREADABLE-NETWORK"


def test_t03_per_fetch_clock_is_timeout_and_job_deadline_is_deadline():
    t = [0.0]

    def slow(url, **kw):
        t[0] += 10.0
        return _web({})(url, **kw)
    with pytest.raises(auto_acquire._Stop) as e:  # job deadline: passthrough, unchanged
        auto_acquire._guarded_get(JATS, A.cfg(), A.deps([], slow), 100000, 50.0, False, clock=lambda: t[0], deadline=5.0)
    assert e.value.code == "R-ACQ-DEADLINE"
    acq_discovered.reset_for_tests()
    t[0] = 0.0
    body_web = _web({ROBOTS: _txt("User-agent: *\nAllow: /\n")})  # a body is streamed, so the clock is read after the wait

    def slow_body(url, **kw):
        t[0] += 10.0
        return body_web(url, **kw)
    v = acq_discovered._robots_verdict(f"https://{PUB}/a", {PUB}, A.deps([], slow_body), A.cfg(), lambda: t[0], None)
    assert v == ("unreadable", "TIMEOUT")  # robots.txt read took longer than the per-fetch clock (5 s)


def test_t04_html_page_is_ctype():
    assert _verdict(_web({ROBOTS: _txt("<html></html>", 200, "text/html")})) == ("unreadable", "CTYPE")


def test_t05_redirect_cap_is_redirects_and_cross_host_stays_host():
    loc = {"Location": "/robots2"}
    assert _verdict(_web({ROBOTS: _txt("", 302, headers=loc), "/robots2": _txt("", 302, headers={"Location": "/robots3"}),
                          "/robots3": _txt("", 302, headers={"Location": "/robots4"})})) == ("unreadable", "REDIRECTS")
    acq_discovered.reset_for_tests()
    with pytest.raises(auto_acquire._Stop) as e:
        _verdict(_web({ROBOTS: _txt("", 302, headers={"Location": "https://other.example/robots.txt"})}))
    assert e.value.code == "R-ACQ-HOST"


def test_t06_encoding_oversize_and_loop():
    assert _verdict(_web({ROBOTS: _txt("x", 200, headers={"Content-Encoding": "gzip"})})) == ("unreadable", "ENCODING")
    acq_discovered.reset_for_tests()
    assert _verdict(_web({ROBOTS: _txt("x" * 600000)})) == ("unreadable", "SIZE")
    acq_discovered.reset_for_tests()
    assert _verdict(_web({ROBOTS: _txt("", 302, headers={"Location": "/robots.txt"})})) == ("unreadable", "CYCLE")


def test_t07_evaluation_exception_is_eval_not_a_rule(monkeypatch):
    from vera import robots_ai

    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(robots_ai, "allowed", boom)
    assert _verdict(_web({ROBOTS: _txt("User-agent: *\nAllow: /\n")})) == ("unreadable", "EVAL")


def test_t08_parsed_rules_keep_r_acq_robots():
    assert _code(_web({ROBOTS: _txt("User-agent: *\nDisallow: /\n")})) == "R-ACQ-ROBOTS"
    acq_discovered.reset_for_tests()
    assert _code(_web({ROBOTS: _txt("User-agent: *\nContent-Signal: ai-input=no\nAllow: /\n")})) == "R-ACQ-ROBOTS"
    acq_discovered.reset_for_tests()
    v, why = _verdict(_web({ROBOTS: _txt("User-agent: *\nDisallow: /\n")}))
    assert (v, why) == ("disallow", "verdict:disallow")


@pytest.mark.parametrize("status,suffix", [(400, "STATUS_4XX"), (401, "STATUS_401"), (403, "STATUS_403"),
                                           (405, "STATUS_4XX"), (451, "STATUS_4XX")])
def test_t09_robots_4xx_other_than_404_410_fails_closed_R1(status, suffix):
    # R1 / OD-1111-ROBOTS-4XX (owner 2026-11-11): FLIPPED from "robots 4xx = allowed" (RFC 9309 s2.3.1.3). A recorded
    # deviation: only 404 and 410 are unrestricted; the page is never requested.
    web = _web({ROBOTS: _txt("", status)})
    assert _code(web) == f"R-ACQ-ROBOTS_UNREADABLE-{suffix}" and A.doc_urls(web.urls) == []


@pytest.mark.parametrize("status", [404, 410])
def test_t09b_404_410_are_unrestricted_and_429_is_rate_limited(status):
    web = _web({ROBOTS: _txt("", status)})
    assert _gg(web)[0] == b"<article/>" and A.doc_urls(web.urls) != []
    acq_discovered.reset_for_tests()
    assert _code(_web({ROBOTS: _txt("", 429)})) == "R-ACQ-RATELIMITED"


def test_t09c_4xx_stop_is_cached_for_robots_ttl_not_fail_ttl():
    # RC-1: the stop is cached as long as the old "allow" was, so no robots.txt request is added (60 s would re-read)
    web = _web({ROBOTS: _txt("", 403)})
    now = [100.0]
    _code(web, clock=lambda: now[0])
    n = sum(1 for u in web.urls if u.endswith(ROBOTS))
    for t in (161.0, 3699.0):  # t0+61 s and t0+3599 s: a cache hit, no new robots.txt request
        now[0] = t
        assert _code(web, clock=lambda: now[0]).startswith("R-ACQ-ROBOTS_UNREADABLE-CACHED_STATUS_403")
        assert sum(1 for u in web.urls if u.endswith(ROBOTS)) == n == 1
    now[0] = 3701.0  # t0+3601 s: one new request
    _code(web, clock=lambda: now[0])
    assert sum(1 for u in web.urls if u.endswith(ROBOTS)) == 2


def test_t09d_5xx_keeps_the_60s_fail_ttl():
    web = _web({ROBOTS: _txt("", 500)})
    now = [100.0]
    _code(web, clock=lambda: now[0])
    now[0] = 159.0
    assert _code(web, clock=lambda: now[0]).startswith("R-ACQ-ROBOTS_UNREADABLE-CACHED_STATUS_5XX")
    now[0] = 161.0
    _code(web, clock=lambda: now[0])
    assert sum(1 for u in web.urls if u.endswith(ROBOTS)) == 2


def test_t14_cache_is_per_host_and_2tuple_false_entry_is_tolerated():
    acq_discovered._ROBOTS[EBI] = (1.0, False)  # an older fixture shape: no cause element
    v = _verdict(_web({}), host=PUB)  # another host: its own robots.txt is read (404 = allow)
    assert v == ("allow", None)
    assert acq_discovered._robots_verdict(JATS, {EBI}, A.deps([], _web({})), A.cfg(), A.clock_at(2.0), None) == \
        ("unreadable", "CACHED")  # no cause: fail TTL, suffix CACHED
    assert acq_discovered.hop_access(JATS, {EBI}, A.deps([], _web({})), A.cfg(), A.clock_at(2.0), None) == \
        "robots_unreadable-CACHED"


def test_t17_no_request_is_added_across_the_cases():
    # per-URL request counts (a multiset) for a 5xx failure: one robots.txt request, the document never
    web = _web({ROBOTS: _txt("", 500)})
    _code(web)
    assert sorted(web.urls) == [f"https://{EBI}/robots.txt"]


def test_t18_second_hop_of_one_request_reads_robots_once():
    web = _web({ROBOTS: _txt("", 403)})
    for _ in range(3):
        _code(web)
    assert sum(1 for u in web.urls if u.endswith(ROBOTS)) == 1
