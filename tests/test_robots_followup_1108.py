"""ROBOTS-FOLLOWUP-1108: acquisition hops read page-level no-AI signals; selective_fetch reads Content-Usage; Crawl-delay
from any own-token group (strictest); Content-Usage train-ai=n is informational. Fake HTTP only."""
from __future__ import annotations

import math
from urllib import robotparser

import pytest

from tests import acq_support as A
from vera import acq_discovered, auto_acquire, robots_ai, source_permissions as sp
from vera import selective_fetch as sf

JATS = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML"
ROBOTS = "/robots.txt"


def _web(over=None):
    return A.Web({}, {"PMC1": b"<article/>"}, over or {})


def _gg(web, cfg=None):
    return auto_acquire._guarded_get(JATS, cfg or A.cfg(), A.deps([], web), 100000, 5.0, False, clock=A.clock_at())


def _code(web, cfg=None):
    with pytest.raises(auto_acquire._Stop) as e:
        _gg(web, cfg)
    return e.value.code


# ---- (1) acquisition: header signals
@pytest.mark.parametrize("hdr", [
    {"X-Robots-Tag": "noai"}, {"X-Robots-Tag": "noimageai"}, {"X-Robots-Tag": "bingbot: ai-grounding:disallow"},
    {"X-Robots-Tag": "ai-grounding: disallowextended"}, {"Content-Signal": "ai-train=no, ai-input=no"},
    {"Content-Usage": "ai-use=n"}, {"Content-Usage": "train-ai=y, ai-use = n"}])
def test_guarded_get_refuses_header_no_ai(hdr):
    web = _web({JATS: A.Resp(body=b"<article/>", ctype="application/xml", headers=hdr)})
    # WPL-1111 (ROBOTS-LABEL-1111): the page's own no-AI signal is R-ACQ-ROBOTS-PAGE_NOAI (same text as R-ACQ-ROBOTS)
    assert _code(web) == "R-ACQ-ROBOTS-PAGE_NOAI"


@pytest.mark.parametrize("hdr", [{"X-Robots-Tag": "ai-grounding:allow"}, {"Content-Signal": "ai-train=no"},
                                 {"Content-Usage": "train-ai=n"}, {"Content-Usage": "ai-use=y"}])
def test_guarded_get_allows_non_barring_headers(hdr):
    web = _web({JATS: A.Resp(body=b"<article/>", ctype="application/xml", headers=hdr)})
    assert _gg(web)[0] == b"<article/>"


@pytest.mark.parametrize("meta", ['<meta name="robots" content="noindex, noai">',
                                  '<meta name="Bingbot" content="ai-grounding:disallow">',
                                  "<meta name='bingbot' content='ai-grounding:DisallowExtended'>"])
def test_guarded_get_refuses_meta_no_ai_in_html(meta):
    cfg = A.cfg()
    cfg["fetch"]["content_types"] = list(cfg["fetch"]["content_types"]) + ["text/html"]
    page = f"<html><head>{meta}</head><body>x</body></html>".encode()
    assert _code(_web({JATS: A.Resp(body=page, ctype="text/html")}), cfg) == "R-ACQ-ROBOTS-PAGE_NOAI"  # WPL-1111


def test_page_no_ai_unit_and_fail_closed(monkeypatch):
    assert acq_discovered.page_no_ai({"content-usage": "ai-use=n"}) is True
    assert acq_discovered.page_no_ai({}, b"<meta name=robots content=noai>", "text/html") is True
    assert acq_discovered.page_no_ai({}, b"<meta name=robots content=noai>", "application/xml") is False  # not HTML
    assert acq_discovered.page_no_ai({"x-robots-tag": "noarchive"}, b"<p>hi</p>", "text/html") is False
    monkeypatch.setattr(sp, "_lower_headers", lambda h: (_ for _ in ()).throw(RuntimeError("x")))
    assert acq_discovered.page_no_ai({}) is True


def test_b_get_refuses_header_no_ai_on_the_discovered_route():
    web = _web({JATS: A.Resp(body=b"<article/>", ctype="application/xml", headers={"Content-Usage": "ai-use=n"})})
    with pytest.raises(auto_acquire._Stop) as e:
        acq_discovered._b_get(JATS, {"www.ebi.ac.uk"}, {"application/xml"}, 100000, 2, A.deps([], web), A.cfg(),
                              A.clock_at(), None, None)
    assert e.value.code == "R-ACQ-ROBOTS-PAGE_NOAI"  # WPL-1111 (ROBOTS-LABEL-1111): the page's signal, same text


@pytest.mark.parametrize("txt", ["Content-Signal: ai-input=no\nUser-agent: *\nAllow: /\n",
                                 "User-agent: *\nAllow: /\nContent-Usage: /x ai-use=n\n"])
def test_acquisition_robots_txt_use_signal_refuses(txt):
    acq_discovered.reset_for_tests()
    assert _code(_web({ROBOTS: A.Resp(body=txt.encode(), ctype="text/plain")})) == "R-ACQ-ROBOTS"


# ---- (2) selective_fetch page headers
class _R:
    def __init__(self, headers, body=b"hello", status=200):
        self.status_code, self.headers, self._b = status, {"Content-Type": "text/plain", **headers}, body
        self.encoding = "utf-8"

    def iter_content(self, n):
        yield self._b

    def close(self):
        pass


def _dl(headers):
    return sf._download("https://a.example/p", timeout_s=1, http_get=lambda u, **k: _R(headers),
                        url_check=lambda u: (True, ""), pacer=type("P", (), {"wait": lambda s, u: None})(),
                        robots_ok=lambda *a, **k: True)


def test_selective_fetch_reads_content_usage_header():
    assert _dl({"Content-Usage": "ai-use=n"})[5] == "blocked: Content-Usage ai-use=n"
    assert _dl({"Content-Usage": "train-ai=n"})[5] is None  # informational only
    assert _dl({})[5] is None


# ---- (3) Crawl-delay from any own-token group, strictest
def _rp(txt):
    p = robotparser.RobotFileParser()
    p.parse(txt.splitlines())
    return robots_ai.annotate(p, txt)


TXT = ("User-agent: *\nCrawl-delay: 2\n\nUser-agent: vera-dar\nCrawl-delay: 7\n\n"
       "User-agent: VERA-research-pipeline\nUser-agent: other\nCrawl-delay: 11\n\nUser-agent: somebot\nCrawl-delay: 99\n")


def test_robots_ai_crawl_delay_takes_strictest_own_group():
    assert robots_ai.crawl_delay(_rp(TXT), "VERA-research-fetcher/1.0") == 11.0


def test_robots_ai_crawl_delay_malformed_own_is_inf_and_star_only_unchanged():
    assert robots_ai.crawl_delay(_rp("User-agent: vera-dar\nCrawl-delay: abc\n"), "VERA-research-fetcher") == math.inf
    assert robots_ai.crawl_delay(_rp("User-agent: *\nCrawl-delay: 3\n"), "VERA-research-fetcher") == 3.0


def test_selective_parse_crawl_delay_takes_strictest_own_group():
    assert sf.parse_crawl_delay(TXT, "VERA-research-fetcher/1.0") == 11.0
    assert sf.parse_crawl_delay("User-agent: *\nCrawl-delay: 2\nUser-agent: vera-dar\nCrawl-delay: 1\n",
                                "VERA-research-fetcher") == 2.0  # never lower than the group that applies


# ---- (4) train-ai=n informational
def test_content_usage_train_ai_is_info_not_a_bar():
    assert sp.content_usage_signal("train-ai=n") == "ai-train=no"
    assert sp.content_usage_signal("ai-use=n, train-ai=n") == "ai-input=no"
    assert sp._content_signal_no_ai(sp.content_usage_signal("train-ai=n")) is False
    assert sp.robots_no_ai(sp.robots_entries(sp.robots_content_signal("Content-Usage: train-ai=n"))) is False
    assert sp.robots_content_signal("Content-Usage: train-ai=n") == "ai-train=no"
