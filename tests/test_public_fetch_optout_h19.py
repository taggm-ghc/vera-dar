"""H19-T3a: licence_href / meta_author / TDM Reservation signals from public_fetch.fetch, and the A9 evaluate path.
Offline: DNS and transport are fakes (helpers from the S6 fetch tests)."""
import json

import pytest

from tests.test_public_fetch_1103 import CFG, GOOD, Net, Resp, html, robots
from vera import public_fetch as pf, source_permissions as sp, web_route

TD = ("a.example", "/.well-known/tdmrep.json")
PAGE = ("a.example", "/p")
HEAD = (b'<html><head><link rel="license" href="https://creativecommons.org/licenses/by-nc-nd/3.0/us/">'
        b'<meta name="author" content="Jane Roe"></head><body><p>hello</p></body></html>')


def jresp(obj, status=200, ctype="application/json"):
    return Resp(status, {"Content-Type": ctype}, json.dumps(obj).encode())


def go(pages, tdm=True, headers=None, body=HEAD, cache=None):
    pg = {("a.example", "/robots.txt"): robots(), PAGE: Resp(200, {"Content-Type": "text/html", **(headers or {})}, body),
          **pages}
    n = Net(GOOD, pg)
    kw = {"robots_cache": cache} if cache is not None else {}
    return n, pf.fetch("https://a.example/p", CFG, resolve=n.resolve, transport=n.transport, tdm_check=tdm, **kw)


def test_signal_strings_licence_author_and_default_no_tdm_request():
    n, r = go({}, tdm=False)
    assert r["licence_href"] == "https://creativecommons.org/licenses/by-nc-nd/3.0/us/"
    assert r["meta_author"] == "jane roe" and r["tdm_reservation"] == ""
    assert [c[3] for c in n.calls] == ["/robots.txt", "/p"]  # tdmrep.json is opt-in: existing callers unchanged


def test_absent_signals_are_empty_strings():
    _n, r = go({}, tdm=False, body=b"<p>x</p>")
    assert (r["licence_href"], r["meta_author"], r["tdm_reservation"]) == ("", "", "")


def test_tdm_meta_and_header_reserve_without_fetching_tdmrep():
    n, r = go({}, body=b'<meta name="tdm-reservation" content="1"><p>x</p>')
    assert r["tdm_reservation"] == "1" and [c[3] for c in n.calls] == ["/robots.txt", "/p"]
    n, r = go({}, headers={"TDM-Reservation": "1"})
    assert r["tdm_reservation"] == "1"
    _n, r = go({TD: jresp([])}, body=b'<meta name="tdm-reservation" content="0">')
    assert r["tdm_reservation"] == ""


@pytest.mark.parametrize("status", [404, 410])
def test_tdmrep_404_410_means_no_reservation(status):
    n, r = go({TD: Resp(status)})
    assert r["tdm_reservation"] == "" and n.calls[-1][3] == "/.well-known/tdmrep.json"


@pytest.mark.parametrize("rule", [Resp(500), Resp(503), Resp(403), Resp(429), RuntimeError("down"),
                                  Resp(200, {"Content-Type": "application/json"}, b"{not json"),
                                  jresp({"location": "/"}), jresp([{"nolocation": 1}]),
                                  Resp(302, {"Location": "https://10.0.0.1/x"})])
def test_tdmrep_any_other_failure_is_unreadable(rule):
    _n, r = go({TD: rule})
    assert r["tdm_reservation"] == "unreadable"


def test_tdmrep_location_match():
    rules = [{"location": "/private/", "tdm-reservation": 1}, {"location": "/p", "tdm-reservation": 0}]
    assert go({TD: jresp(rules)})[1]["tdm_reservation"] == ""
    assert go({TD: jresp([{"location": "/", "tdm-reservation": 1}])})[1]["tdm_reservation"] == "1"
    assert go({TD: jresp([{"location": "/p*", "tdm-reservation": "1"}])})[1]["tdm_reservation"] == "1"
    # the stricter of several matching rules applies
    both = [{"location": "/", "tdm-reservation": 0}, {"location": "/p", "tdm-reservation": 1}]
    assert go({TD: jresp(both)})[1]["tdm_reservation"] == "1"


def test_tdmrep_cached_once_per_host_per_request():
    cache = {}
    n, _ = go({TD: jresp([])}, cache=cache)
    n.calls.clear()
    n.pages[("a.example", "/p")] = html(HEAD)
    pf.fetch("https://a.example/p", CFG, resolve=n.resolve, transport=n.transport, tdm_check=True, robots_cache=cache)
    assert [c[3] for c in n.calls] == ["/p"]


def test_tdmrep_hop_gate_refusal_is_unreadable_and_nothing_sent():
    seen = []

    def gate(u):
        seen.append(u)
        return "read_lane_hop_refused" if u.endswith("tdmrep.json") else None
    n = Net(GOOD, {("a.example", "/robots.txt"): robots(), PAGE: html(HEAD)})
    r = pf.fetch("https://a.example/p", CFG, resolve=n.resolve, transport=n.transport, hop_gate=gate, tdm_check=True)
    assert r["tdm_reservation"] == "unreadable" and not any(c[3].endswith("tdmrep.json") for c in n.calls)
    assert any(u.endswith("tdmrep.json") for u in seen)


def test_evaluate_fetched_passes_tdm_signal_and_bars_use():
    got = {"text": "t", "url": "https://x.example/a", "meta_robots": "", "x_robots_tag": "", "content_signal": "",
           "tdm_reservation": "1"}
    rec = web_route.evaluate_fetched("https://x.example/a", got)
    assert rec.use_reason == "use_tdm_reserved" and rec.use_for_answer is False
    got["tdm_reservation"] = "unreadable"
    assert web_route.evaluate_fetched("https://x.example/a", got).use_reason == "use_tdm_reserved"
    got["tdm_reservation"] = ""
    assert web_route.evaluate_fetched("https://x.example/a", got).use_for_answer is True


def test_noai_xrt_and_tdm_each_bar_use_alone():
    base = dict(licence="CC-BY-4.0")
    assert sp.evaluate("https://x.example/a", headers={"X-Robots-Tag": "noai"}, **base).use_for_answer is False
    assert sp.evaluate("https://x.example/a", html_head='<meta name="robots" content="noimageai">', **base).use_for_answer is False
    assert sp.evaluate("https://x.example/a", headers={"TDM-Reservation": "1"}, **base).use_for_answer is False


def test_gated_fetch_tdm_opt_in_only_for_fetchers_that_take_it():
    seen = {}

    def f_new(url, hop_gate=None, robots_cache=None, tdm_check=False):
        seen["tdm"] = tdm_check
        return {"text": "", "url": url}

    def f_old(url, hop_gate=None, robots_cache=None):
        seen["old"] = True
        return {"text": "", "url": url}
    web_route.gated_fetch("https://x.example/a", fetch=f_new, tdm_check=True)
    assert seen["tdm"] is True
    web_route.gated_fetch("https://x.example/a", fetch=f_new)
    assert seen["tdm"] is False
    web_route.gated_fetch("https://x.example/a", fetch=f_old, tdm_check=True)
    assert seen["old"] is True


def test_gated_fetch_tdm_default_on_for_nd_access_outlets():
    seen = {}

    def f(url, hop_gate=None, robots_cache=None, tdm_check=False):
        seen["t"] = tdm_check
        return {"text": "", "url": url}
    web_route.gated_fetch("https://www.commondreams.org/a", fetch=f)
    assert seen["t"] is True
    web_route.gated_fetch("https://globalvoices.org/a", fetch=f)
    assert seen["t"] is False
    web_route.gated_fetch("https://www.commondreams.org/a", fetch=f, tdm_check=False)
    assert seen["t"] is False


def test_real_lane_fetch_path_threads_tdm_check_and_fails_closed(monkeypatch):
    """R1: web_route.gated_fetch -> lane_evidence._public_fetch (the lane's real default fetcher) -> public_fetch.fetch."""
    from vera import lane_evidence
    dns = {"commondreams.org": ["93.184.216.34"], "www.commondreams.org": ["93.184.216.34"]}
    pages = {("www.commondreams.org", "/robots.txt"): robots(), ("www.commondreams.org", "/a"): html(HEAD),
             ("www.commondreams.org", "/.well-known/tdmrep.json"): jresp([{"location": "/", "tdm-reservation": 1}])}
    n = Net(dns, pages)
    real = pf.fetch

    def fetch(url, hop_gate=None, robots_cache=None, tdm_check=False):
        return real(url, CFG, resolve=n.resolve, transport=n.transport, hop_gate=hop_gate, robots_cache=robots_cache,
                    tdm_check=tdm_check)
    monkeypatch.setattr(pf, "fetch", fetch)
    got = web_route.gated_fetch("https://www.commondreams.org/a", fetch=lane_evidence._public_fetch)
    assert got["tdm_reservation"] == "1" and any(c[3] == "/.well-known/tdmrep.json" for c in n.calls)
    # a non-ND host: no tdmrep request
    n2 = Net({"a.example": ["93.184.216.34"]}, {("a.example", "/robots.txt"): robots(), PAGE: html(HEAD)})
    monkeypatch.setattr(pf, "fetch", lambda url, hop_gate=None, robots_cache=None, tdm_check=False: real(
        url, CFG, resolve=n2.resolve, transport=n2.transport, hop_gate=hop_gate, robots_cache=robots_cache, tdm_check=tdm_check))
    web_route.gated_fetch("https://a.example/p", fetch=lane_evidence._public_fetch)
    assert not any(c[3].endswith("tdmrep.json") for c in n2.calls)
    # fetcher that cannot take tdm_check + ND outlet = refused, never skipped
    monkeypatch.setattr(pf, "fetch", lambda url, hop_gate=None, robots_cache=None: {"text": "", "url": url})
    with pytest.raises(pf.FetchRefused):
        web_route.gated_fetch("https://www.commondreams.org/a", fetch=lane_evidence._public_fetch)
