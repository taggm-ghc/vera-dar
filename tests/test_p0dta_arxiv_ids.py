"""P0DTA-ARXFIX regression: every arXiv URL shape for 2605.23989v1 is parsed by the one shared parser, routed to the
versioned abs/HTML route and admitted under the REAL licence gate (faked network). arxiv_id is a string."""
import pytest

from tests import arxiv_synth as X
from tests.test_acq_discovered import Router, _direct, cfg_on, disc, run  # noqa: F401  (_direct: autouse)
from vera import acq_discovered, arxiv_fulltext, inline_acquire

BASE = "arxiv.org"
SHAPES = [
    "https://arxiv.org/abs/2605.23989v1",
    "http://arxiv.org/abs/2605.23989v1",
    "https://arxiv.org/html/2605.23989v1",
    "http://arxiv.org/html/2605.23989v1",
    "https://arxiv.org/abs/2605.23989v1?context=cs",
    "https://arxiv.org/abs/2605.23989v1#abstract",
    "https://arxiv.org/pdf/2605.23989v1",
    "https://arxiv.org/pdf/2605.23989v1.pdf",
    "https://www.arxiv.org/abs/2605.23989v1",
    "https://export.arxiv.org/abs/2605.23989v1",
]


@pytest.fixture(autouse=True)
def _nowait(monkeypatch):
    monkeypatch.setattr(arxiv_fulltext, "_sleep", lambda s: None)


@pytest.mark.parametrize("u", SHAPES)
def test_shared_parser_accepts_all_shapes(u):
    assert arxiv_fulltext.parse_arxiv_url(u) == ("2605.23989", "v1")
    assert acq_discovered._arxiv_id({"url": u}) == "2605.23989"
    assert arxiv_fulltext.arxiv_ident({"url": u}) == ("2605.23989", "v1")


def test_parser_unversioned_old_style_and_rejects():
    assert arxiv_fulltext.parse_arxiv_url("http://arxiv.org/abs/2605.23989") == ("2605.23989", None)
    assert arxiv_fulltext.parse_arxiv_url("https://arxiv.org/abs/hep-th/9901001v2") == ("hep-th/9901001", "v2")
    for bad in ("https://evilarxiv.org/abs/2605.23989v1", "https://arxiv.org.evil.com/abs/2605.23989v1",
                "ftp://arxiv.org/abs/2605.23989v1", "https://arxiv.org/list/cs/2605", None, ""):
        assert arxiv_fulltext.parse_arxiv_url(bad) is None


@pytest.mark.parametrize("u", SHAPES)
def test_openalex_no_doi_shape_admitted_via_versioned_route(u):
    src = {"provider": "openalex", "url": u, "doi": None, "title": X.TITLE,
           "snippet": "transmission incidence outbreak pathogen", "openalex_id": "https://openalex.org/W1",
           "licence": {"metadata": {"id": "CC0-1.0"}}, "stored_text_kind": "abstract_metadata"}
    web = Router({}, {}, X.routes())
    o, store = run(cfg_on(), [src], web)
    assert disc(o)["outcome"] == "admitted", o.public()
    assert len(store.plans) == 1
    urls = [c[0] for c in web.calls]
    assert X.ABS_URL in urls and X.HTML_URL in urls
    assert store.plans[0]["source"]["provenance"]["acquisition"]["arxiv"]["html_url"] == X.HTML_URL


@pytest.mark.parametrize("u", ["https://arxiv.org/html/2605.23989v1", "http://arxiv.org/html/2605.23989v1",
                               "https://arxiv.org/html/2605.23989v1?x=1"])
def test_ddg_html_url_admitted(u):
    src = {"provider": "duckduckgo_lite", "url": u, "doi": None, "title": X.TITLE,
           "snippet": "transmission incidence outbreak pathogen"}
    o, store = run(cfg_on(), [src], Router({}, {}, X.routes()))
    assert disc(o)["outcome"] == "admitted", o.public()
    assert len(store.plans) == 1


def test_arxiv_id_candidate_field_is_string():
    o, _ = run(cfg_on(), [X.discovery_src()], Router({}, {}, X.routes()))
    ids = [c["arxiv_id"] for c in o.candidates]
    assert ids == ["2605.23989v1"] and all(isinstance(i, str) for i in ids)
    assert inline_acquire._arxiv_id_str({"id": "2605.23989", "version": None}) == "2605.23989"
    assert inline_acquire._arxiv_id_str(None) is None


# ---- defect C: robots Crawl-delay: 15, cold robots cache, fake clock (live case arXiv:2605.23989v1)
from tests import acq_support as A  # noqa: E402

ROBOTS = "https://arxiv.org/robots.txt"


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _crawl_run(clk, last=None, inline_s=None):
    web = Router({}, {}, {**X.routes(), ROBOTS: A.Resp(body=b"User-agent: *\nCrawl-delay: 15\n", ctype="text/plain")})
    d = A.deps([], web, store=A.Store())
    cfg = cfg_on()
    if inline_s is not None:
        cfg["inline"]["deadline_s"] = inline_s
    o = inline_acquire.acquire(A.Q_EPI, cfg, env=A.ENV_ON, deps_factory=lambda a, e: d,
                               discovered=[X.discovery_src()], clock=clk)
    return o, web, d.store if hasattr(d, "store") else None


@pytest.fixture
def clk(monkeypatch):
    from vera import acq_discovered
    acq_discovered._ROBOTS.clear()
    arxiv_fulltext._HOST_LAST.clear()
    c = _Clock()
    waits = []
    monkeypatch.setattr(arxiv_fulltext, "_sleep", lambda s: (waits.append(s), setattr(c, "t", c.t + float(s))))
    c.waits = waits
    yield c
    arxiv_fulltext._HOST_LAST.clear()
    acq_discovered._ROBOTS.clear()


def test_cold_robots_crawl_delay_15_now_admitted_without_shortening_delay(clk):
    o, web, _ = _crawl_run(clk)
    assert disc(o)["outcome"] == "admitted", o.public()
    assert [round(w) for w in clk.waits] == [15, 15]  # abs and html each wait the FULL crawl-delay after the prior request
    urls = [c[0] for c in web.calls]
    assert X.ABS_URL in urls and X.HTML_URL in urls


def test_warm_host_old_last_request_goes_out_immediately(clk):
    _crawl_run(clk)  # warms robots + records the last request time
    clk.waits.clear()
    clk.t += 60
    inline_acquire.reset_for_tests()
    o, _web, _ = _crawl_run(clk)
    assert disc(o)["outcome"] == "admitted", o.public()
    assert [round(w) for w in clk.waits] == [15]  # only abs->html spacing; first request not delayed


def test_crawl_delay_not_fitting_budget_is_queued_with_distinct_code(clk):
    o, web, _ = _crawl_run(clk, inline_s=12)
    d = disc(o)
    assert d["reason"] == "R-ACQ-CRAWL_DELAY" and d["reason"] != "R-ACQ-DEADLINE", o.public()
    assert X.HTML_URL not in [c[0] for c in web.calls]
    assert o.candidates[0]["outcome"] == "queued_background"


# ---- verifier fixes: interleaved records keep the host spacing; strict ASCII / anchored parsing
def test_interleaved_records_keep_crawl_delay_gap(clk, monkeypatch):
    from vera import auto_acquire as aa
    sent = []
    monkeypatch.setattr(aa, "_guarded_get", lambda url, *a, **k: (sent.append((clk.t, url)), (b"", url))[1])
    acq = {"arxiv_route": {"enabled": True, "robots": False, "min_interval_s": 15, "hosts": ["arxiv.org"]},
           "bounds": {"fetch_timeout_s": 10}}
    pA, pB = {}, {}
    for url, pace in (("https://arxiv.org/abs/A", pA), ("https://arxiv.org/abs/B", pB), ("https://arxiv.org/html/A", pA),
                      ("https://arxiv.org/html/B", pB)):
        arxiv_fulltext._get(url, acq, None, 1000, clk, None, [], pace)
    gaps = [b[0] - a[0] for a, b in zip(sent, sent[1:])]
    assert len(sent) == 4 and min(gaps) >= 15, gaps


@pytest.mark.parametrize("u", [
    "https://arxiv.org/abs/２６０５.２３９８９v1",          # fullwidth digits
    "https://arxıv.org/abs/2605.23989v1",                   # dotless i
    "https://arxiv.org/abs/2605.23989v1K",             # Kelvin sign
    "https://arxiv.org/abs/hep-th/99010０1",
    "https://evilarxiv.org/abs/2605.23989v1", "https://user@arxiv.org/abs/2605.23989v1",
    "https://evil.com/arxiv.org/abs/2605.23989v1", "ftp://arxiv.org/abs/2605.23989v1",
    "//arxiv.org/abs/2605.23989v1", "https://arxiv.org/abs/2401.123456",
])
def test_strict_parse_rejects(u):
    assert arxiv_fulltext.parse_arxiv_url(u) is None
    assert acq_discovered._arxiv_id({"url": u}) is None
    assert arxiv_fulltext.arxiv_ident({"url": u}) is None


def test_ascii_only_id_text_and_doi():
    assert arxiv_fulltext.arxiv_ident({"identifier": "arXiv:２６０５.２３９８９v1"}) is None
    assert arxiv_fulltext.arxiv_ident({"doi": "10.48550/arxiv.２６０５.２３９８９"}) is None
    assert arxiv_fulltext.arxiv_ident({"doi": "10.48550/arXiv.2605.23989"}) == ("2605.23989", None)
