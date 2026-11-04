"""NONASK-1103 WP6-P2b commit 9 (verification-push2 D2, D3): robots.txt Content-Signal ai-input=no reaches
source_permissions.evaluate on every public-page path; both lanes use the per-hop READ gate and evaluate(); the
Official lane follows R-050 for unknown terms. Offline: DNS and transport are fakes; no socket, DB or env file."""
import json
from pathlib import Path

import pytest

from tests.test_public_fetch_1103 import CFG, Net, Resp, html
from vera import lane_evidence as le
from vera import official_lane as ol
from vera import public_fetch as pf
from vera import selective_fetch as sf
from vera import source_permissions as sp
from vera import web_route as wr

NOAI_ROBOTS = b"User-agent: *\nContent-Signal: search=yes, ai-input=no\nAllow: /\n"
OK_ROBOTS = b"User-agent: *\nContent-Signal: search=yes, ai-train=no\nAllow: /\n"
DNS = {h: ["93.184.216.34"] for h in ("a.example", "docs.aws.amazon.com", "docs.cloud.google.com")}


def rob(txt):
    return Resp(200, {"Content-Type": "text/plain"}, txt)


def net_fetch(net):
    def f(url, **kw):
        return pf.fetch(url, CFG, resolve=net.resolve, transport=net.transport, **kw)
    return f


# ---- D2: robots.txt Content-Signal -----------------------------------------------------------------
def test_robots_content_signal_parsed_signal_only():
    assert sp.robots_content_signal("# c\nUser-agent: *\nContent-Signal: ai-input=no # x\nDisallow:\n") == "ai-input=no"
    assert sp.robots_no_ai(sp.robots_entries("search=yes, ai-input=no"))
    assert not sp.robots_no_ai(sp.robots_entries("search=yes, ai-train=no")) and sp.robots_entries("") == []


def test_public_fetch_returns_robots_signal_and_web_route_bars_use():
    net = Net(DNS, {("a.example", "/robots.txt"): rob(NOAI_ROBOTS), ("a.example", "/p"): html(b"<p>Readable text.</p>")})
    got = net_fetch(net)("https://a.example/p")
    assert got["content_signal"] == "search=yes, ai-input=no"  # robots.txt value joins the header key
    r = wr.read_web("q", [{"url": "https://a.example/p", "title": "T"}], fetch=net_fetch(Net(DNS, {
        ("a.example", "/robots.txt"): rob(NOAI_ROBOTS), ("a.example", "/p"): html(b"<p>Readable text.</p>")})),
        registry=[], deny_list=[])
    assert r.pages == [] and r.dev[0]["gate"]["use"] == "use_no_ai_signal"


def test_robots_without_no_ai_keeps_page():
    net = Net(DNS, {("a.example", "/robots.txt"): rob(OK_ROBOTS), ("a.example", "/p"): html(b"<p>Readable text.</p>")})
    r = wr.read_web("q", [{"url": "https://a.example/p", "title": "T"}], fetch=net_fetch(net), registry=[], deny_list=[])
    assert len(r.pages) == 1


class _HResp:
    def __init__(self, status=200, body=b"", ctype="text/plain"):
        self.status_code, self.headers, self.body, self.encoding = status, {"Content-Type": ctype}, body, "utf-8"

    def iter_content(self, n):
        yield self.body


def test_selective_fetch_robots_no_ai_never_requests_page():
    sf.robots_cache_clear()
    sent = []

    def http_get(url, **kw):
        sent.append(url)
        return _HResp(200, NOAI_ROBOTS) if url.endswith("/robots.txt") else _HResp(200, b"<p>x</p>", "text/html")
    body, final, err = sf.fetch_html("https://a.example/p", http_get=http_get, url_check=lambda u: (True, ""),
                                     pacer=type("P", (), {"wait": lambda self, u: None})())
    sf.robots_cache_clear()
    assert body is None and err == "blocked: robots.txt no-AI signal" and sent == ["https://a.example/robots.txt"]


# ---- D3: lanes ---------------------------------------------------------------------------------------
GC = {"entity": "Google Cloud", "domains": ["docs.cloud.google.com"], "state": "active", "terms_verdict": "unknown",
      "docs_entry_urls": ["https://docs.cloud.google.com/docs"]}
AWS = {"entity": "AWS", "domains": ["docs.aws.amazon.com"], "state": "active", "access_verdict": "forbid"}


def test_official_lane_unknown_terms_read_and_redirect_hop_gated():
    pages = {("docs.cloud.google.com", "/robots.txt"): rob(OK_ROBOTS),
             ("docs.cloud.google.com", "/docs"): Resp(302, {"Location": "https://docs.aws.amazon.com/x"})}
    net = Net(DNS, pages)
    out = ol.OfficialLane({"enabled": True}, registry=[GC, AWS], fetch=net_fetch(net)).run(["Google Cloud"])
    assert out["pages"] == [] and out["links"][0]["reason"] == "read_registry_forbid"
    assert "docs.aws.amazon.com" not in [c[0] for c in net.calls]
    ok = Net(DNS, {("docs.cloud.google.com", "/robots.txt"): rob(OK_ROBOTS),
                   ("docs.cloud.google.com", "/docs"): html(b"<p>Official text.</p>")})
    out = ol.OfficialLane({"enabled": True}, registry=[GC], fetch=net_fetch(ok)).run(["Google Cloud"])
    assert len(out["pages"]) == 1 and out["links"][0]["read"] is True  # R-050: unknown terms are read


@pytest.mark.parametrize("extra,code", [({"meta_robots": "noai"}, "use_no_ai_signal"),
                                        ({"content_signal": "ai-input=no"}, "use_no_ai_signal"),
                                        ({"content_signal": "search=yes, ai-input=no"}, "use_no_ai_signal")])
def test_official_lane_evaluates_signals(extra, code):
    f = lambda u, **k: {"url": u, "text": "Official text.", **extra}  # noqa: E731
    out = ol.OfficialLane({"enabled": True}, registry=[GC], fetch=f).run(["Google Cloud"])
    assert out["pages"] == [] and out["links"][0]["reason"] == code


def test_official_lane_derivative_forbid_and_login_path():
    row = dict(GC, derivative_verdict="forbid")
    out = ol.OfficialLane({"enabled": True}, registry=[row], fetch=lambda u, **k: {"url": u, "text": "t"}).run(
        ["Google Cloud"])
    assert out["pages"] == [] and out["links"][0]["reason"] == "use_registry_forbid"
    row = dict(GC, docs_entry_urls=["https://docs.cloud.google.com/login"])
    calls = []
    out = ol.OfficialLane({"enabled": True}, registry=[row],
                          fetch=lambda u, **k: calls.append(u) or {"url": u, "text": "t"}).run(["Google Cloud"])
    assert calls == [] and out["links"][0]["reason"] == "read_login_paywall"


def _collect(monkeypatch, fetch):
    from tests.test_lane_evidence_1103 import FakeReporting, rep_on
    rep_on(monkeypatch)
    item = {"title": "Outage", "url": "https://a.example/news", "published": "2026-11-01", "outlet": "a.example"}
    return le.collect("What is the latest news on kubernetes outages?", reporting=FakeReporting([item]), fetch=fetch)


def test_lane_evidence_outlet_redirect_hop_gated(monkeypatch):
    net = Net(DNS, {("a.example", "/robots.txt"): rob(OK_ROBOTS),
                    ("a.example", "/news"): Resp(302, {"Location": "https://docs.aws.amazon.com/x"})})
    res = _collect(monkeypatch, net_fetch(net))
    assert res.pages == [] and res.links[0]["read"] is False
    assert "docs.aws.amazon.com" not in [c[0] for c in net.calls]


@pytest.mark.parametrize("extra", [{"meta_robots": "noai"}, {"content_signal": "search=yes, ai-input=no"},
                                   {"x_robots_tag": "noai"}])
def test_lane_evidence_outlet_page_evaluated(monkeypatch, extra):
    res = _collect(monkeypatch, lambda u, **k: {"url": u, "text": "Reported text about kubernetes outages.", **extra})
    assert res.pages == [] and res.links[0]["read"] is False
    ok = _collect(monkeypatch, lambda u, **k: {"url": u, "text": "Reported text about kubernetes outages."})
    assert len(ok.pages) == 1


def test_shared_hop_gate_codes():
    gate = wr.make_hop_gate("https://a.example/p", registry=[AWS], deny_list=["denied.example"])
    assert gate("https://docs.aws.amazon.com/x") == "read_registry_forbid"
    assert gate("https://denied.example/x") == "read_deny_list"
    assert gate("https://a.example/login") == "read_login_paywall" and gate("https://a.example/news") is None


def test_arxiv_acknowledgement_on_about_and_readme():
    ack = "Thank you to arXiv for use of its open access interoperability."
    root = Path(__file__).resolve().parent.parent
    assert ack in json.dumps(json.loads((root / "config/ui_help_pages.json").read_text())["about"])
    readme = (root / "README.md").read_text()
    assert ack in readme.split("## Credit", 1)[1]
