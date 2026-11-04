"""NONASK-1103 WP6-P2: the Crawl-delay parsed by selective_fetch's robots check feeds the same per-host
minimum-interval gate (selective_fetch.crawl_reserve) as the web route, so the scholarly fetch path is never early."""
import pytest

from tests.test_nonask_1103_compliance import Resp, _get, ok
from vera import selective_fetch as sf


@pytest.fixture(autouse=True)
def _clear():
    sf.robots_cache_clear()
    sf.reset_for_tests()
    yield
    sf.robots_cache_clear()
    sf.reset_for_tests()


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_robots_crawl_delay_refuses_an_early_second_request_then_allows():
    log, clk = [], Clock()
    g = _get(Resp(200, b"User-agent: *\nCrawl-delay: 5\nAllow: /\n"), log=log)
    kw = dict(http_get=g, url_check=ok, pacer=sf.HostPacer({}), crawl_clock=clk)
    assert sf.fetch_candidate("https://x.example/a", **kw).ok
    assert sf.robots_crawl_delay("https://x.example/b") == 5.0
    clk.t += 2
    r = sf.fetch_candidate("https://x.example/b", **kw)
    assert not r.ok and r.error == "crawl_delay"
    assert log.count("https://x.example/b") == 0  # never sent early
    clk.t += 4
    assert sf.fetch_candidate("https://x.example/b", **kw).ok


def test_no_crawl_delay_no_gate_and_other_hosts_independent():
    clk = Clock()
    g = _get(Resp(200, b"User-agent: *\nAllow: /\n"))
    kw = dict(http_get=g, url_check=ok, pacer=sf.HostPacer({}), crawl_clock=clk)
    assert sf.fetch_candidate("https://y.example/a", **kw).ok
    assert sf.fetch_candidate("https://y.example/b", **kw).ok
    assert sf.robots_crawl_delay("https://y.example/a") == 0.0
    assert sf.robots_crawl_delay("https://never-seen.example/a") == 0.0  # not cached: no fetch, 0
