"""NONASK-1103 COMP: robots.txt (R-050), attribution, acknowledgements, BY-SA notice, arXiv cache. OFFLINE, synthetic."""
import json
from types import SimpleNamespace

import pytest

from tests.test_item71_ddg import DDG, QT, R, HTML, page
from vera import auto_acquire, disclosure, selective_fetch as sf, ui_evidence
from vera.m7 import live_report as lr
from vera.search_providers import ArxivProvider, DuckDuckGoLiteProvider, ProviderUnavailable

UA = DDG["user_agent"]


class Resp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code, self._b, self.headers = status, body, headers or {}
        self.encoding = "utf-8"

    def iter_content(self, n):
        yield self._b


@pytest.fixture(autouse=True)
def _clear():
    sf.robots_cache_clear()
    yield
    sf.robots_cache_clear()


def _get(robots, page_body=b"<html><body>hello text</body></html>", log=None):
    def g(url, **kw):
        log.append(url) if log is not None else None
        if url.endswith("/robots.txt"):
            return robots() if callable(robots) else robots
        return Resp(200, page_body, {"Content-Type": "text/html"})
    return g


ok = lambda u: (True, "")


def test_selective_fetch_blocks_when_robots_disallows_and_fetches_nothing():
    log = []
    r = sf.fetch_candidate("https://x.example/a", http_get=_get(Resp(200, b"User-agent: *\nDisallow: /\n"), log=log),
                           url_check=ok, pacer=sf.HostPacer({}))
    assert not r.ok and "robots" in r.error and log == ["https://x.example/robots.txt"]


def test_selective_fetch_allows_and_caches_robots_per_host():
    log = []
    g = _get(Resp(200, b"User-agent: *\nDisallow: /private\n"), log=log)
    for p in ("a", "b"):
        assert sf.fetch_candidate(f"https://x.example/{p}", http_get=g, url_check=ok, pacer=sf.HostPacer({})).ok
    assert log.count("https://x.example/robots.txt") == 1
    assert not sf.fetch_candidate("https://x.example/private", http_get=g, url_check=ok, pacer=sf.HostPacer({})).ok


@pytest.mark.parametrize("resp", [Resp(500), Resp(429), "boom"])
def test_selective_fetch_fails_closed_on_robots_error(resp):
    def rob():
        if resp == "boom":
            raise TimeoutError("x")
        return resp
    assert not sf.fetch_candidate("https://x.example/a", http_get=_get(rob), url_check=ok, pacer=sf.HostPacer({})).ok


def test_selective_fetch_404_robots_means_allowed():
    assert sf.fetch_candidate("https://x.example/a", http_get=_get(Resp(404)), url_check=ok, pacer=sf.HostPacer({})).ok


def test_ddg_skipped_with_cause_when_robots_disallows():
    calls = []
    p = DuckDuckGoLiteProvider(DDG, transport=lambda *a: calls.append(a) or R(text=HTML), query_terms=QT,
                               sleep=lambda s: None, robots_check=lambda url, ua: (seen.append((url, ua)) or False))
    seen = []
    with pytest.raises(ProviderUnavailable, match="R-ACQ-ROBOTS"):
        p.call("agent frameworks", 5, 5.0)
    assert calls == [] and p.last_query["cause"] == "R-ACQ-ROBOTS" and seen == [(DDG["endpoint"], UA)]


def test_ddg_robots_check_uses_the_queried_host_with_cache(monkeypatch):
    seen = []
    def g(url, headers=None, **kw):
        seen.append((url, headers["User-Agent"]))
        return Resp(200, b"User-agent: *\nDisallow: /lite\n" if "//duckduckgo.com" in url else b"User-agent: *\nAllow: /\n")
    monkeypatch.setattr(sf, "pinned_get", g)
    monkeypatch.setattr(sf, "is_public_url", ok)
    assert DuckDuckGoLiteProvider._default_robots("https://lite.duckduckgo.com/lite/", UA)
    assert DuckDuckGoLiteProvider._default_robots("https://lite.duckduckgo.com/lite/", UA)
    assert not DuckDuckGoLiteProvider._default_robots("https://duckduckgo.com/lite/", UA)
    assert seen == [("https://lite.duckduckgo.com/robots.txt", UA), ("https://duckduckgo.com/robots.txt", UA)]


def test_tasl_attribution_for_europe_pmc_has_all_cc_by_parts():
    a = auto_acquire.tasl_attribution("T", "A B; C D", "https://doi.org/10.1/x", "cc by", "https://creativecommons.org/licenses/by/4.0/")
    assert a == {"creator": "A B; C D", "title": "T", "source": "https://doi.org/10.1/x", "licence": "cc by",
                 "licence_url": "https://creativecommons.org/licenses/by/4.0/", "changes": "excerpted"}


class St:
    def __init__(self):
        self.out = []

    def __getattr__(self, name):
        return lambda *a, **k: self.out.append(str(a[0]) if a else "")


def test_dict_attribution_renders_with_licence_link_and_excerpt_mark():
    at = auto_acquire.tasl_attribution("Paper (arXiv:1)", "Au", "https://arxiv.org/abs/1", "CC BY", "https://creativecommons.org/licenses/by/4.0/", changes="")
    rep = {"groups": [{"n": 1, "ns": [1], "passages": [], "title": "Paper", "url": "https://arxiv.org/abs/1", "published": "",
                       "identifier": "", "licence": "cc by", "abstract_source": "", "attribution": at,
                       "excerpts": [{"claim": 1, "verdict": "supported", "quote": "q", "passage": ""}]}]}
    st = St()
    lr._render_sources(st, rep, lambda x: x)
    text = "\n".join(st.out)
    assert "Au" in text and "creativecommons.org/licenses/by/4.0" in text and "Changes: excerpted" in text


def test_semantic_scholar_credit_shown_with_its_metadata_and_on_about():
    rep = {"groups": [{"n": 1, "ns": [1], "passages": [], "title": "P", "url": "", "published": "", "identifier": "",
                       "licence": "", "abstract_source": "semanticscholar", "attribution": None, "excerpts": []}]}
    st = St()
    lr._render_sources(st, rep, lambda x: x)
    assert any("Semantic Scholar" in x for x in st.out)
    about = json.dumps(json.load(open("config/ui_help_pages.json"))["about"])
    assert "Semantic Scholar" in about


def test_arxiv_acknowledgement_on_about_once():
    about = json.dumps(json.load(open("config/ui_help_pages.json"))["about"])
    assert about.count("Thank you to arXiv for use of its open access interoperability.") == 1


def test_by_sa_notice_only_for_sa_sources():
    g = [{"n": 1, "title": "Wiki-like", "licence": "CC BY-SA 4.0"}, {"n": 2, "title": "Other", "licence": "cc by"},
         {"n": 3, "title": "Via attr", "licence": "", "attribution": {"licence": "cc by-sa"}}]
    t = ui_evidence.share_alike_notice(g)
    assert t.startswith("Parts of this answer adapt CC BY-SA material from: [1] Wiki-like; [3] Via attr;")
    assert t.endswith("remains under CC BY-SA 4.0.") and "Other" not in t
    assert ui_evidence.share_alike_notice([g[1]]) == ""
    rep = {"is_answer": True, "answer": "a", "question": "q", "groups": [dict(x, url="", published="", ns=[x["n"]], passages=[], excerpts=[]) for x in g]}
    st = St()
    lr.render_live_report(st, rep, public=True)
    assert any("adapt CC BY-SA" in x for x in st.out)


def test_arxiv_api_results_cached_with_ttl():
    from tests.test_item71_ddg import CFG
    n, now = [], [1000.0]
    tx = lambda url, params, headers, t: n.append(params) or SimpleNamespace(status_code=200, text="<feed/>")
    p = ArxivProvider(CFG["arxiv"], transport=tx, query_terms=CFG["query_terms"], wall_clock=lambda: now[0])
    p.call("agent frameworks", 5, 5.0); p.call("agent frameworks", 5, 5.0)
    assert len(n) == 1 and p.last_query.get("cache_hit")
    now[0] += ArxivProvider.CACHE_TTL_S + 1
    p.call("agent frameworks", 5, 5.0)
    assert len(n) == 2
    assert all(set(k) <= {"search_query", "start", "max_results", "sortBy", "sortOrder"} for k in n)


def test_generic_third_party_disclosure_shown_with_any_non_groq_checker(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("GROQ_API_KEY", "")
    text = disclosure.disclosure_text()
    assert "third-party AI and search services" in text
    assert any("third-party AI and search services" in t for _h, t in disclosure.disclosure_sections())


def test_europepmc_rate_gate_well_under_10_per_s():
    cfg = json.load(open("config/rate_limits.json"))
    e = cfg.get("portals", cfg)["europepmc"]
    assert e["rate_per_s"] <= 5.0 and e["burst"] <= 5 and "10 requests" in e["_comp_note"]
