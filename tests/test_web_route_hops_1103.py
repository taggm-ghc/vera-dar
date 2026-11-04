"""NONASK-1103 WP6-P2b (security-push2 S1, S2, M2): per-hop READ gate, per-request bounds of the web route, robots.txt
cache per request, abandoned DNS workers. Offline: DNS and transport are fakes; no socket, DB or env file."""
import socket
import threading

import pytest

from tests.test_public_fetch_1103 import CFG, Net, Resp, html, robots
from vera import inline_acquire
from vera import public_fetch as pf
from vera import web_route as wr

_REAL_READ_WEB = wr.read_web
REG = [{"entity": "AWS", "domains": ["docs.aws.amazon.com"], "state": "active", "access_verdict": "forbid"},
       {"entity": "Links", "domains": ["links.example"], "state": "active", "terms_verdict": "links_only"}]
DNS = {h: ["93.184.216.34"] for h in ("a.example", "b.example", "c.example", "docs.aws.amazon.com", "links.example",
                                      "denied2.example")}


def redirect(to):
    return Resp(302, {"Location": to})


def real_fetch(net):
    def f(url, **kw):
        return pf.fetch(url, CFG, resolve=net.resolve, transport=net.transport, **kw)
    return f


def cand(url):
    return {"url": url, "title": "T", "snippet": "s", "acq_source": "ddg"}


def run(cands, pages, **kw):
    net = Net(DNS, pages)
    kw.setdefault("registry", REG)
    kw.setdefault("deny_list", ["denied2.example"])
    return wr.read_web("q", cands, fetch=real_fetch(net), **kw), net


def sent_hosts(net):
    return [c[0] for c in net.calls]


@pytest.mark.parametrize("target,code", [("https://docs.aws.amazon.com/q", "read_registry_forbid"),
                                         ("https://links.example/q", "read_registry_forbid"),
                                         ("https://denied2.example/q", "read_deny_list"),
                                         ("https://b.example/login?next=/x", "read_login_paywall")])
def test_redirect_hop_refused_before_any_request(target, code):
    pages = {("a.example", "/robots.txt"): robots(), ("a.example", "/p"): redirect(target),
             ("b.example", "/robots.txt"): robots()}
    r, net = run([cand("https://a.example/p")], pages)
    assert r.pages == [] and r.dev[0]["gate"]["read"] == code and r.listed[0]["phrase"]
    host = target.split("/")[2]
    assert host not in sent_hosts(net), "a request (robots.txt or page) was sent to a read-restricted hop"


def test_hop_gate_in_public_fetch_also_guards_robots_redirect():
    pages = {("a.example", "/robots.txt"): redirect("https://docs.aws.amazon.com/robots.txt"), ("a.example", "/p"): html()}
    net = Net(DNS, pages)
    gate = lambda u: "read_registry_forbid" if "amazon" in u else None  # noqa: E731
    with pytest.raises(pf.FetchRefused) as e:
        pf.fetch("https://a.example/p", CFG, resolve=net.resolve, transport=net.transport, hop_gate=gate)
    assert e.value.reason == "robots" and "docs.aws.amazon.com" not in sent_hosts(net)


def test_redirect_to_host_already_read_this_round_is_not_sent():
    pages = {("a.example", "/robots.txt"): robots(), ("a.example", "/p"): html(),
             ("b.example", "/robots.txt"): robots(), ("b.example", "/p"): redirect("https://a.example/other")}
    r, net = run([cand("https://a.example/p"), cand("https://b.example/p")], pages)
    assert len(r.pages) == 1 and r.dev[1]["gate"]["read"] == "read_host_round_cap"
    assert ("a.example", "93.184.216.34", 443, "/other") not in [c[:4] for c in net.calls]


def test_fetch_budget_bounds_fetches_and_lists_the_rest():
    pages = {(h, t): (robots() if t == "/robots.txt" else html()) for h in ("a.example", "b.example", "c.example")
             for t in ("/robots.txt", "/p")}
    r, net = run([cand(f"https://{h}/p") for h in ("a.example", "b.example", "c.example")], pages, max_fetches=2)
    assert r.fetched == 2 and len(r.pages) == 2 and r.dev[2]["gate"]["read"] == "read_fetch_budget"
    assert "c.example" not in sent_hosts(net) and r.listed[0]["phrase"]


def test_stop_signal_checked_before_each_fetch():
    pages = {(h, t): (robots() if t == "/robots.txt" else html()) for h in ("a.example", "b.example")
             for t in ("/robots.txt", "/p")}
    calls = iter([None, "deadline"])
    r, net = run([cand("https://a.example/p"), cand("https://b.example/p")], pages, should_stop=lambda: next(calls))
    assert r.fetched == 1 and r.dev[1]["gate"]["read"] == "read_deadline" and "b.example" not in sent_hosts(net)


def test_robots_cached_per_host_for_the_request():
    # one call reads robots.txt once per host even when two fetches reach that host (redirect from another host)
    pages = {("a.example", "/robots.txt"): robots(), ("a.example", "/p"): html(),
             ("b.example", "/robots.txt"): robots(), ("b.example", "/p"): html()}
    net = Net(DNS, pages)
    cache: dict = {}
    for u in ("https://a.example/p", "https://a.example/p", "https://b.example/p"):
        pf.fetch(u, CFG, resolve=net.resolve, transport=net.transport, robots_cache=cache)
    assert [c[3] for c in net.calls].count("/robots.txt") == 2


def test_abandoned_dns_threads_count_against_cap(monkeypatch):
    old = list(pf._ABANDONED)
    ev = threading.Event()
    try:
        pf._ABANDONED[:] = []

        def slow(host, port):
            ev.wait(30)
            return [(socket.AF_INET, 0, 0, "", ("93.184.216.34", port))]
        for _ in range(pf.MAX_ABANDONED):
            with pytest.raises(socket.gaierror):
                pf._bounded_resolve(slow, "a.example", 443, 0.05)
        assert len(pf._ABANDONED) == pf.MAX_ABANDONED
        with pytest.raises(pf.FetchRefused) as e:
            pf._bounded_resolve(slow, "a.example", 443, 0.05)
        assert e.value.reason == "busy"
    finally:
        ev.set()
        pf._ABANDONED[:] = old


class _Ctx:
    def __init__(self, stop):
        self.stop = stop

    def stop_reason(self):
        return self.stop

    def touch(self):
        pass


def _inline(monkeypatch, ctx, deadline, clock_v, max_fetches):
    seen = {}
    real = _REAL_READ_WEB

    def fake_read(question, web, **kw):
        seen.update(kw)
        return real(question, web, fetch=lambda u, **k: {"url": u, "text": "body"}, registry=[], deny_list=[],
                           **kw)
    monkeypatch.setattr(wr, "read_web", fake_read)
    st: dict = {}
    acq = {"inline": {"web_read_enabled": True}, "bounds": {"per_doc_timeout_s": 25}}
    inline_acquire._web_read("q", acq, [cand("https://a.example/p"), cand("https://b.example/p")], st, ctx,
                             lambda *_: None, clock=lambda: clock_v, deadline=deadline, max_fetches=max_fetches)
    return st, seen


def test_inline_passes_deadline_reserve_and_budget(monkeypatch):
    st, seen = _inline(monkeypatch, None, deadline=100.0, clock_v=80.0, max_fetches=5)  # 20 s left < 25 s reserve
    assert seen["max_fetches"] == 5 and seen["should_stop"]() == "deadline" and st["n_web_fetched"] == 0
    st, _ = _inline(monkeypatch, None, deadline=100.0, clock_v=0.0, max_fetches=1)
    assert st["n_web_fetched"] == 1 and st["n_web"] == 1


def test_inline_stream_stop_signal(monkeypatch):
    ctx = _Ctx(None)
    st, seen = _inline(monkeypatch, ctx, deadline=None, clock_v=0.0, max_fetches=5)
    ctx.stop = "disconnect"
    assert seen["should_stop"]() == "stopped"


def test_inline_web_read_runs_once_per_question(monkeypatch):
    st, _ = _inline(monkeypatch, None, deadline=100.0, clock_v=0.0, max_fetches=5)
    first = st["web"]
    inline_acquire._web_read("q", {"inline": {"web_read_enabled": True}}, [cand("https://c.example/p")], st, None,
                             lambda *_: None, clock=lambda: 0.0, deadline=100.0, max_fetches=5)
    assert st["web"] is first
