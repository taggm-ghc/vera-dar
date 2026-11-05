"""ASK-PARALLEL-FIX-1105 (owner rule feedback_respect_access_and_compliance_terms; S-K defect 2). OFFLINE: the DDG
landing-page DOI lookups honour robots.txt Crawl-delay per host (selective_fetch.crawl_reserve), waiting within the hop
budget and skipping (never sending early) when the delay does not fit."""
import threading
import time
from collections import defaultdict

import pytest
import requests

from vera import acq_discovered
from vera import search_providers as sp
from vera import selective_fetch

PAGE = b'<html><head><meta name="citation_doi" content="10.1234/abc"></head></html>'


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real network transport called in a test")
    monkeypatch.setattr(requests, "get", boom)
    for _ in range(2):
        selective_fetch.reset_for_tests()
        selective_fetch.robots_cache_clear()
        acq_discovered.reset_for_tests()
        if _ == 0:
            yield


class R:
    def __init__(self, status, headers=None, body=b""):
        self.status_code, self.headers, self._b = status, headers or {}, body

    def iter_content(self, n):
        yield self._b

    def close(self):
        pass

    @property
    def text(self):
        return self._b.decode()


class Net:
    def __init__(self, robots):
        self.robots, self.lock, self.starts = robots, threading.Lock(), defaultdict(list)

    def __call__(self, url, headers=None, timeout=None, allow_redirects=False, stream=False, **kw):
        host, path = url.split("/")[2], "/" + "/".join(url.split("/")[3:])
        with self.lock:
            self.starts[host].append((path, time.monotonic()))
        if path == "/robots.txt":
            return R(200, {"Content-Type": "text/plain"}, self.robots.get(host, b"User-agent: *\nAllow: /\n"))
        time.sleep(0.05)
        return R(200, {"Content-Type": "text/html"}, PAGE)


def _pages(net, host):
    return [t for p, t in net.starts[host] if p != "/robots.txt"]


def test_crawl_delay_spaces_page_requests_to_one_host():
    net = Net({"slow-site.example": b"User-agent: *\nCrawl-delay: 2\nAllow: /\n"})
    out, skipped = sp._landing_dois_concurrently(["https://slow-site.example/a", "https://slow-site.example/b"], 5.0,
                                                 100000, get=net, url_check=lambda u: True, deadline=None,
                                                 clock=time.monotonic, acq={}, robots_ok=None)
    p = _pages(net, "slow-site.example")
    assert out == ["10.1234/abc", "10.1234/abc"] and not skipped and len(p) == 2
    assert p[1] - p[0] >= 2.0 - 0.02


def test_crawl_delay_that_does_not_fit_the_budget_skips_and_never_sends_early():
    net = Net({"slower.example": b"User-agent: *\nCrawl-delay: 30\nAllow: /\n"})
    a = sp.landing_doi("https://slower.example/a", 5.0, 100000, get=net, url_check=lambda u: True, acq={})
    t0 = time.monotonic()
    b = sp.landing_doi("https://slower.example/b", 5.0, 100000, get=net, url_check=lambda u: True, acq={})
    assert a == "10.1234/abc" and b is None and time.monotonic() - t0 < 1.0
    assert len(_pages(net, "slower.example")) == 1


def test_no_crawl_delay_means_no_wait():
    net = Net({})
    t0 = time.monotonic()
    for s in "abc":
        assert sp.landing_doi(f"https://fast.example/{s}", 5.0, 100000, get=net, url_check=lambda u: True,
                              acq={}) == "10.1234/abc"
    assert time.monotonic() - t0 < 1.0


def test_crawl_delay_shared_with_selective_fetch_gate():
    """A request selective_fetch just reserved for the host counts: the landing waits for the same per-host gate."""
    net = Net({"shared.example": b"User-agent: *\nCrawl-delay: 1\nAllow: /\n"})
    assert sp.landing_doi("https://shared.example/a", 5.0, 100000, get=net, url_check=lambda u: True, acq={})
    assert selective_fetch.crawl_wait_s("shared.example", 1.0) > 0.5
    assert not selective_fetch.crawl_reserve("shared.example", 1.0)


def test_crawl_wait_s():
    t = [100.0]
    clock = lambda: t[0]  # noqa: E731
    assert selective_fetch.crawl_wait_s("h.example", 2.0, clock) == 0.0
    assert selective_fetch.crawl_reserve("h.example", 2.0, clock)
    t[0] = 100.5
    assert selective_fetch.crawl_wait_s("H.example", 2.0, clock) == pytest.approx(1.5)
    assert selective_fetch.crawl_wait_s("h.example", 0, clock) == 0.0
    t[0] = 103.0
    assert selective_fetch.crawl_wait_s("h.example", 2.0, clock) == 0.0


def _two(host, robots):
    net = Net({host: robots})
    out, skipped = sp._landing_dois_concurrently([f"https://{host}/a", f"https://{host}/b"], 5.0, 100000, get=net,
                                                 url_check=lambda u: True, deadline=None, clock=time.monotonic,
                                                 acq={}, robots_ok=None)
    return net, out


def test_fractional_crawl_delay_is_honoured():
    net, out = _two("frac.example", b"User-agent: *\nCrawl-delay: 1.5\nAllow: /\n")
    p = _pages(net, "frac.example")
    assert out == ["10.1234/abc", "10.1234/abc"] and len(p) == 2
    assert p[1] - p[0] >= 1.5 - 0.02


@pytest.mark.parametrize("value", [b"abc", b"", b"-1", b"nan", b"inf"])
def test_malformed_or_empty_crawl_delay_skips_the_host(value):
    net, out = _two("bad.example", b"User-agent: *\nCrawl-delay: " + value + b"\nAllow: /\n")
    assert out == [None, None] and _pages(net, "bad.example") == []  # robots.txt read; no page fetched at all


def test_absent_crawl_delay_line_means_no_wait():
    t0 = time.monotonic()
    net, out = _two("none.example", b"User-agent: *\nAllow: /\n")
    assert out == ["10.1234/abc", "10.1234/abc"] and time.monotonic() - t0 < 1.0


def test_parse_crawl_delay_group_matching():
    import math
    ua = selective_fetch.USER_AGENT.split("/")[0]
    txt = f"User-agent: *\nCrawl-delay: 9\n\nUser-agent: {ua}\nCrawl-delay: 0.5\n"
    assert selective_fetch.parse_crawl_delay(txt) == 0.5  # our own group wins over *
    assert selective_fetch.parse_crawl_delay("User-agent: other\nCrawl-delay: 3\n") is None
    assert selective_fetch.parse_crawl_delay("User-agent: *\nCrawl-delay: 1 # c\nCrawl-delay: 2.5\n") == 2.5
    assert selective_fetch.parse_crawl_delay("User-agent: *\nCrawl-delay: abc\n") == math.inf
    assert selective_fetch.parse_crawl_delay("User-agent: *\nCrawl-delay:\n") == math.inf
    assert selective_fetch.parse_crawl_delay("User-agent: *\nAllow: /\n") is None


def test_selective_fetch_robots_crawl_delay_reads_fractions():
    selective_fetch.remember_robots_text("sf.example", "User-agent: *\nCrawl-delay: 1.25\n")
    import urllib.robotparser as rpm
    p = rpm.RobotFileParser()
    p.parse(["User-agent: *", "Crawl-delay: 1.25"])
    with selective_fetch._ROBOTS_LOCK:
        selective_fetch._ROBOTS[("https", "sf.example", selective_fetch.USER_AGENT)] = (time.monotonic(), p)
    assert selective_fetch.robots_crawl_delay("https://sf.example/x") == 1.25
