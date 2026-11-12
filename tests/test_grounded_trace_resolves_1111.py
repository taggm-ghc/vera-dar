"""DENIAL-COVERAGE-1111 WP-D / T12: scripts/grounded_trace_run._resolves is pinned, guarded and robots-first. Offline:
requests.Session.get is patched; requests per (host, path) are counted."""
import importlib.util
import socket
from pathlib import Path

import pytest
import requests

from vera import acq_discovered as ad
from vera import selective_fetch as sf

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "grounded_trace_run.py"
_spec = importlib.util.spec_from_file_location("grounded_trace_run_1111", SCRIPT)
gtr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gtr)


class Resp:
    def __init__(self, status=200, headers=None, body=b""):
        self.status_code, self.headers, self.body = status, headers or {}, body

    def iter_content(self, n):
        yield self.body

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    ad.reset_for_tests()
    with sf._ROBOTS_LOCK:
        sf._ROBOTS.clear()
    monkeypatch.setattr(socket, "getaddrinfo", lambda h, p, *a, **k: [(socket.AF_INET, 0, 0, "", ("93.184.216.34", p))])
    yield
    ad.reset_for_tests()


def net(monkeypatch, script):
    calls = []

    def get(self, url, headers=None, **kw):
        rest = url.split("://", 1)[1]
        key = ((headers or {}).get("Host", "?"), "/" + (rest.split("/", 1)[1] if "/" in rest else "").split("?")[0])
        calls.append(key)
        seq = script.get(key) or [Resp(404 if key[1] == "/robots.txt" else 200)]
        return seq.pop(0) if len(seq) > 1 else seq[0]
    monkeypatch.setattr(requests.Session, "get", get)
    monkeypatch.setattr(requests, "head", lambda *a, **k: pytest.fail("plain requests.head used"))
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("plain requests.get used"))
    return calls


def test_t12_403_means_exists_and_second_call_sends_nothing(monkeypatch):
    calls = net(monkeypatch, {("p.example", "/a"): [Resp(403)]})
    assert gtr._resolves("https://p.example/a") is True
    assert gtr._resolves("https://p.example/a") is True
    assert calls.count(("p.example", "/a")) == 1


def test_t12_404_and_200(monkeypatch):
    net(monkeypatch, {("p.example", "/gone"): [Resp(404)]})
    assert gtr._resolves("https://p.example/gone") is False
    assert gtr._resolves("https://p.example/ok") is True


def test_t12_follows_at_most_three_redirects(monkeypatch):
    script = {("p.example", f"/r{i}"): [Resp(302, {"Location": f"https://p.example/r{i + 1}"})] for i in range(6)}
    calls = net(monkeypatch, script)
    assert gtr._resolves("https://p.example/r0") is False
    assert [c for c in calls if c[1] != "/robots.txt"] == [("p.example", f"/r{i}") for i in range(4)]
    script2 = {("p.example", "/r0"): [Resp(302, {"Location": "https://p.example/r1"})],
               ("p.example", "/r1"): [Resp(200)]}
    net(monkeypatch, script2)
    assert gtr._resolves("https://p.example/r0") is True


def test_t12_redirect_target_already_denied_is_not_requested(monkeypatch):
    ad.deny(ad.denial_ids(urls=("https://q.example/secret",)))
    calls = net(monkeypatch, {("p.example", "/go"): [Resp(302, {"Location": "https://q.example/secret"})]})
    assert gtr._resolves("https://p.example/go") is True  # a remembered 401/403 means the page exists
    assert ("q.example", "/secret") not in calls


def test_t12_robots_disallow_sends_no_document_request(monkeypatch):
    robots = Resp(200, {"Content-Type": "text/plain"}, b"User-agent: *\nDisallow: /\n")
    calls = net(monkeypatch, {("p.example", "/robots.txt"): [robots]})
    assert gtr._resolves("https://p.example/a") is None  # not checked
    assert ("p.example", "/a") not in calls
    # a robots-disallowed redirect target
    robots2 = Resp(200, {"Content-Type": "text/plain"}, b"User-agent: *\nDisallow: /\n")
    calls = net(monkeypatch, {("p.example", "/go"): [Resp(302, {"Location": "https://z.example/x"})],
                              ("z.example", "/robots.txt"): [robots2]})
    assert gtr._resolves("https://p.example/go") is None
    assert ("z.example", "/x") not in calls


def test_t12_script_has_no_plain_requests_calls():
    src = SCRIPT.read_text()
    assert "requests.get(" not in src and "requests.head(" not in src
