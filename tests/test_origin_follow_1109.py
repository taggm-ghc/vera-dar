"""OD-1109-ORIGIN (owner 2026-11-09, "Follow to any original host (gated)") and OD-1109-QUOTA. A reference on a
registered portal page may lead to its ORIGINAL host even when unregistered; the page is read only through the existing
gates (vera.web_route -> vera.public_fetch). SYNTHETIC fixtures; DNS and transport are fakes; no socket, DB or model."""
from __future__ import annotations

import json
import pathlib
import re
import socket
from html import escape
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

from vera import public_fetch as pf
from vera import reference_portal as rp
from vera import web_route as wr
from vera.ask_service import _six_line
from vera import research_record as rr

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = {"user_agent": "UA-test", "deny_hosts": [], "max_redirects": 2, "max_compressed_bytes": 50000,
       "max_decompressed_bytes": 200000, "max_robots_bytes": 2000, "timeout_s": 5,
       "content_types": ["text/html", "application/xhtml+xml", "text/plain"]}
ORIGIN = "https://origin.example.net/papers/zorb"
BODY = b"<html><body><p>" + b"Zorblats are synthetic test organisms. " * 20 + b"</p></body></html>"


class Resp:
    def __init__(self, status=200, headers=None, body=b""):
        self.status_code, self.headers, self.body = status, headers or {}, body

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]

    def close(self):
        pass


class Net:
    def __init__(self, dns, pages):
        self.dns, self.pages, self.calls, self.lookups = dns, pages, [], []

    def resolve(self, host, port):
        self.lookups.append(host)
        ans = self.dns[host]
        ans = ans.pop(0) if isinstance(ans, list) and ans and isinstance(ans[0], list) else ans
        return [(socket.AF_INET, 0, 0, "", (a, port)) for a in ans]

    def transport(self, host, ip, port, target, headers, timeout):
        self.calls.append((host, ip, target))
        r = self.pages.get((host, target))
        return r if r is not None else Resp(404, {"Content-Type": "text/html"}, b"")

    def fetch(self, url, **kw):
        return pf.fetch(url, CFG, resolve=self.resolve, transport=self.transport, **kw)


def page(headers=None, body=BODY):
    return Resp(200, {"Content-Type": "text/html; charset=utf-8", **(headers or {})}, body)


ROBOTS_OK = Resp(200, {"Content-Type": "text/plain"}, b"User-agent: *\nAllow: /\n")
PUBLIC = ["93.184.216.34"]


def cand(url=ORIGIN, origin=True):
    c = {"url": url, "title": "origin.example.net", "snippet": "", "doi": None, "discovered_url": url}
    if origin:
        c["portal_origin"] = True
    return c


def read(net, c):
    return wr.read_web("What are zorblats?", [c], fetch=net.fetch, registry={}, deny_list=[], cfg=wr.load_config())


def test_origin_page_on_a_public_host_is_read_answer_only():
    net = Net({"origin.example.net": PUBLIC}, {("origin.example.net", "/robots.txt"): ROBOTS_OK,
                                               ("origin.example.net", "/papers/zorb"): page()})
    out = read(net, cand())
    assert len(out.pages) == 1 and out.pages[0].outlet == "origin.example.net"
    assert all(d.get("ingest") in (False, None) or d["gate"].get("ingest") == "ingest_answer_only_wp" for d in out.dev)


# ---------------------------------------------------------------------------------------------------------- SSRF
@pytest.mark.parametrize("ip", ["10.0.0.5", "127.0.0.1", "169.254.169.254", "192.168.1.9", "100.64.1.1"])
def test_private_loopback_link_local_metadata_ips_are_refused(ip):
    net = Net({"origin.example.net": [ip]}, {})
    out = read(net, cand())
    assert not out.pages and not net.calls  # nothing sent to a non-public address


def test_dns_rebinding_second_resolution_private_is_refused():
    # robots.txt resolves to a public address, the page request re-resolves to a private one: refused, never sent there
    net = Net({"origin.example.net": [PUBLIC, ["10.1.2.3"]]}, {("origin.example.net", "/robots.txt"): ROBOTS_OK,
                                                                ("origin.example.net", "/papers/zorb"): page()})
    out = read(net, cand())
    assert not out.pages
    assert all(ip != "10.1.2.3" for _h, ip, _t in net.calls)


def test_redirect_to_another_host_is_refused_for_origin_links():
    redirect = Resp(302, {"Location": "https://other.example.org/x", "Content-Type": "text/html"}, b"")
    pages = {("origin.example.net", "/robots.txt"): ROBOTS_OK, ("origin.example.net", "/papers/zorb"): redirect,
             ("other.example.org", "/robots.txt"): ROBOTS_OK, ("other.example.org", "/x"): page()}
    net = Net({"origin.example.net": PUBLIC, "other.example.org": PUBLIC}, dict(pages))
    out = read(net, cand())
    assert not out.pages and not any(h == "other.example.org" for h, _i, _t in net.calls)
    assert any(d["gate"].get("read") == "read_redirect_off_host" for d in out.dev)


def test_same_host_redirect_is_followed():
    redirect = Resp(301, {"Location": "https://origin.example.net/papers/zorb-v2", "Content-Type": "text/html"}, b"")
    net = Net({"origin.example.net": PUBLIC}, {("origin.example.net", "/robots.txt"): ROBOTS_OK,
                                               ("origin.example.net", "/papers/zorb"): redirect,
                                               ("origin.example.net", "/papers/zorb-v2"): page()})
    assert len(read(net, cand()).pages) == 1


# ------------------------------------------------------------------------------------------- robots, signals, licence
def test_robots_disallow_for_our_tokens_is_refused():
    deny = Resp(200, {"Content-Type": "text/plain"}, b"User-agent: *\nDisallow: /\n")
    net = Net({"origin.example.net": PUBLIC}, {("origin.example.net", "/robots.txt"): deny,
                                               ("origin.example.net", "/papers/zorb"): page()})
    out = read(net, cand())
    assert not out.pages and not any(t == "/papers/zorb" for _h, _i, t in net.calls)


@pytest.mark.parametrize("hdr", [{"Content-Usage": "ai-use=n"}, {"Content-Signal": "ai-input=no"},
                                 {"X-Robots-Tag": "noai"}])
def test_ai_use_signals_are_honoured(hdr):
    net = Net({"origin.example.net": PUBLIC}, {("origin.example.net", "/robots.txt"): ROBOTS_OK,
                                               ("origin.example.net", "/papers/zorb"): page(hdr)})
    assert not read(net, cand()).pages


@pytest.mark.parametrize("lic", ["https://creativecommons.org/licenses/by-nd/4.0/",
                                 "https://creativecommons.org/licenses/by-nc-nd/4.0/"])
def test_licence_reject_is_never_used(lic):
    body = f'<html><head><link rel="license" href="{lic}"></head><body><p>'.encode() + b"Zorblats text. " * 30 + b"</p></body></html>"
    net = Net({"origin.example.net": PUBLIC}, {("origin.example.net", "/robots.txt"): ROBOTS_OK,
                                               ("origin.example.net", "/papers/zorb"): page(body=body)})
    out = read(net, cand())
    assert not out.pages


# ------------------------------------------------------------------------------------- portal side: origin + hop limit
P1 = "https://en.wikipedia.org/wiki/Zorblats"
P2 = "https://portal-two.example.org/records/zorb"


def coins(atitle, url):
    return f'<span class="Z3988" title="{escape(urlencode([("ctx_ver", "Z39.88-2004"), ("rft.atitle", atitle), ("rft_id", url)]))}"></span>'


class PResp:
    def __init__(self, body):
        self.status_code, self._b = 200, body.encode()
        self.headers = {"Content-Type": "text/html; charset=utf-8"}

    def iter_content(self, n):
        for i in range(0, len(self._b), n):
            yield self._b[i:i + n]

    def close(self):
        pass


def _portal_cfg(follow=True, max_hops=2):
    c = json.loads((ROOT / "config" / "reference_portals.json").read_text())
    c["mode"] = "on"
    for a in c["adapters"].values():
        if isinstance(a, dict):
            a["enabled"] = False
    c["adapters"]["wikipedia_title"]["enabled"] = True
    c["caps"]["max_hops"] = max_hops
    c["classifier"]["follow_origin"] = follow
    return c


def _collect(pages, monkeypatch, **kw):
    calls = []
    deps = rp.Deps(http_get=lambda u, **k: (calls.append(u), PResp(pages[u]))[1], url_check=lambda u: (True, ""),
                   robots_ok=lambda u, ua: True, no_ai_input=lambda u, ua: False, tdmrep_ok=lambda u, p=None: True,
                   crawl_ok=lambda u, ua: True, limited=lambda portal, fn: fn, openalex_lookup=lambda s, n, t: [])
    monkeypatch.setattr(rp, "wikipedia_title_urls", lambda q, c: [P1])
    rx = re.compile(r"/(wiki|records)/[A-Za-z_]+")
    hosts = {h: {"host": h, "paths": rx, "rate_portal": "wikipedia"} for h in ("en.wikipedia.org", "portal-two.example.org")}
    rp.reset_for_tests()
    cands, counts = rp.collect("What are zorblats?", cfg=_portal_cfg(**kw), deps=deps, hosts=hosts, clock=lambda: 0.0)
    return cands, calls


def test_on_question_reference_link_becomes_a_gated_origin_candidate(monkeypatch):
    pages = {P1: coins("Zorblats in the wild", ORIGIN) + coins("Unrelated weather", "https://x.example.com/w")}
    cands, calls = _collect(pages, monkeypatch)
    urls = {c["url"]: c for c in cands}
    assert ORIGIN in urls and urls[ORIGIN]["portal_origin"] is True
    assert "https://x.example.com/w" not in urls  # off-question reference: not followed
    assert calls == [P1]  # the portal never fetches the origin itself; only web_route does, through its gates
    assert wr.is_web_candidate(urls[ORIGIN])


def test_follow_origin_off_is_the_previous_rule(monkeypatch):
    cands, _ = _collect({P1: coins("Zorblats in the wild", ORIGIN)}, monkeypatch, follow=False)
    assert not any(c.get("url") == ORIGIN for c in cands)


def test_hop_limit_no_origin_from_a_second_hop_portal(monkeypatch):
    deep = "https://deep.example.net/zorb"
    pages = {P1: f'<a href="{P2}">portal two</a>', P2: coins("Zorblats deeper", deep)}
    cands, calls = _collect(pages, monkeypatch)
    assert P2 in calls  # hop 2 portal read
    assert not any(c.get("url") == deep for c in cands)  # its origin would be hop 3: never followed
    cands1, calls1 = _collect(pages, monkeypatch, max_hops=1)
    assert calls1 == [P1]


def test_origin_candidate_is_on_topic_for_the_web_topic_gate():
    c = cand()
    assert wr.topic_order("What are zorblats?", [c], {"topic_order": {"enabled": True}}) == [c]


# ------------------------------------------------------------------------------------------------------ quota
def test_quota_raised_and_exhaustion_is_stated_plainly():
    acq = json.loads((ROOT / "config" / "auto_acquire.json").read_text())
    assert acq["bounds"]["jobs_per_hour"] == 120
    assert acq["bounds"]["acq_rounds_one_job"] is True
    texts = acq["freshness"]["texts"]
    line = _six_line(texts, [rr.TRIGGER_NEW], SimpleNamespace(status="rate", admitted=0, web=None), acq)
    assert line == texts["new_documents_not_run"]["rate"] and "this hour" in line
    assert "obtained 2 new" in _six_line(texts, [rr.TRIGGER_NEW], SimpleNamespace(status="ok", admitted=2, web=None), acq)
