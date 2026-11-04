"""NONASK-1103 I-A6 C5: selective_fetch Crawl-delay gate. Offline; fake clock and transport."""
from vera import selective_fetch as sf


class R:
    status_code, headers, encoding = 200, {"Content-Type": "text/html"}, "utf-8"

    def iter_content(self, n):
        yield b"<p>hello world</p>"


def test_reserve_blocks_early_allows_after_and_zero_is_free():
    sf.reset_for_tests()
    t = [0.0]
    c = lambda: t[0]  # noqa: E731
    assert sf.crawl_reserve("h.example", 10, c)
    t[0] = 9
    assert not sf.crawl_reserve("h.example", 10, c)
    t[0] = 10
    assert sf.crawl_reserve("h.example", 10, c)
    assert sf.crawl_reserve("h.example", 0, c) and sf.crawl_reserve("other.example", 10, c)
    sf.reset_for_tests()


def test_table_bounded_oldest_evicted():
    sf.reset_for_tests()
    for i in range(sf._CRAWL_MAX_HOSTS + 20):
        sf.crawl_reserve(f"h{i}.example", 5, lambda: 1.0)
    assert len(sf._CRAWL_LAST) == sf._CRAWL_MAX_HOSTS and "h0.example" not in sf._CRAWL_LAST
    sf.reset_for_tests()


def test_fetch_candidate_refuses_early_and_does_not_send():
    sf.reset_for_tests()
    sent, t = [], [100.0]

    def get(url, **kw):
        sent.append(url)
        return R()
    kw = dict(http_get=get, url_check=lambda u: (True, ""), crawl_delays={"h.example": 10}, crawl_clock=lambda: t[0])
    assert sf.fetch_candidate("https://h.example/a", **kw).ok
    t[0] += 3
    r = sf.fetch_candidate("https://h.example/b", **kw)
    pages = lambda: [u for u in sent if not u.endswith("/robots.txt")]  # COMP: robots.txt is fetched first
    assert not r.ok and r.error == "crawl_delay" and pages() == ["https://h.example/a"]
    t[0] += 8
    assert sf.fetch_candidate("https://h.example/b", **kw).ok and len(pages()) == 2
    sf.reset_for_tests()
