"""Item #79: grounded /ask (retrieval, link following, citation check, no-sources path). Offline only."""
import time

import pytest
from fastapi.testclient import TestClient

from vera import ask_service, grounded_ask as ga, source_resolver as sr
from vera.ask_service import AskResult
from vera.pricing.config import PricingRecord

PRICING = PricingRecord(provider="p", model="m", input=0.0, output=0.0, source_url="u",
                        retrieved_at="2026-01-01", effective_from="2026-01-01")
CFG = {"enabled": True, "providers": ["arxiv"], "num_results": 3, "deadline_s": 5, "snippet_max_chars": 50,
       "no_sources_text": "NO SOURCES", "removed_citation_note": "NOTE",
       "resolve": {"enabled": True, "max_results_to_resolve": 2, "max_pages_per_result": 3, "max_depth": 2,
                   "max_ids_per_page": 2, "max_links_per_page": 3, "timeout_s": 1, "deadline_s": 5,
                   "title_match_min": 0.8, "scholarly_hosts": ["doi.org", "arxiv.org"]}}


def test_shipped_config_enables_grounding_with_bounded_resolution():
    g = ga.load_grounding()
    assert g and {"arxiv", "openalex"} <= set(g["providers"]) and g["resolve"]["max_depth"] >= 1
    assert all(isinstance(g["resolve"][k], (int, float)) for k in ("max_pages_per_result", "deadline_s", "timeout_s"))


def test_clean_text_strips_invisible_unicode_and_caps():
    assert ga.clean_text("ab​c‮ d  e", 5) == "abc d"


@pytest.mark.parametrize("answer,n,expected,removed", [
    ("A [1] and B [2].", 2, "A [1] and B [2].", 0),
    ("A [1, 7] B [9].", 2, "A [1] B .\n\nNOTE", 2),
    ("A [1-3].", 2, "A [1, 2].\n\nNOTE", 1),
])
def test_fix_citations(answer, n, expected, removed):
    assert ga.fix_citations(answer, n, "NOTE") == (expected, removed)


def test_retrieve_keeps_abstracts_resolves_links_and_lists_traced_originals():
    results = [{"url": "https://arxiv.org/abs/1", "title": "Direct", "snippet": "abstract text", "published": "2025-01-02T00"},
               {"url": "https://news.example/a", "title": "News", "snippet": ""},
               {"url": "https://news.example/b", "title": "News 2", "snippet": ""}]

    def resolver(r, rcfg, deadline_at):
        if r["url"].endswith("/a"):
            return [{"title": "Original", "url": "https://doi.org/10.1/x", "snippet": "orig abstract", "doi": "10.1/x",
                     "discovered_via": "news.example"}]
        return [{"title": "No abstract", "url": "https://doi.org/10.1/y", "snippet": "", "doi": "10.1/y",
                 "discovered_via": "news.example"}]

    sources, unused = ga.retrieve_all("q", CFG, search=lambda q: results, resolver=resolver)
    assert [s["title"] for s in sources] == ["Direct", "Original"] and [s["n"] for s in sources] == [1, 2]
    assert sources[0]["published"] == "2025-01-02" and sources[1]["discovered_via"] == "news.example"
    assert [u["title"] for u in unused] == ["No abstract"]


def test_retrieve_search_failure_returns_nothing():
    def boom(q):
        raise RuntimeError("down")
    assert ga.retrieve_all("q", CFG, search=boom) == ([], [])


def test_identifiers_meta_doi_first_then_arxiv_ssrn_and_links():
    html = ('<meta name="citation_doi" content="10.1145/123.456">'
            '<a href="https://arxiv.org/abs/2401.01234">x</a><a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=99">s</a>'
            '<a href="https://doi.org/10.5555/abc.">y</a>')
    assert sr.identifiers_in(html, 5) == ["10.1145/123.456", "10.48550/arxiv.2401.01234", "10.2139/ssrn.99", "10.5555/abc"]


def test_resolve_follows_one_more_hop_and_respects_page_limit():
    pages = {"https://news.example/a": '<a href="https://doi.org/landing">paper</a>',
             "https://doi.org/landing": '<meta name="citation_doi" content="10.9999/z">'}
    fetched = []

    def fetch(url, timeout_s):
        fetched.append(url)
        return pages.get(url), url, None if url in pages else "http 404"

    lookup = lambda doi, t: {"title": "Z", "url": f"https://doi.org/{doi}", "snippet": "abs", "doi": doi}
    out = sr.resolve({"url": "https://news.example/a", "title": "t"}, CFG["resolve"], time.monotonic() + 5,
                     fetch=fetch, lookup=lookup, by_title=lambda *a: None)
    assert [o["doi"] for o in out] == ["10.9999/z"] and out[0]["discovered_via"] == "news.example"
    assert fetched == ["https://news.example/a", "https://doi.org/landing"]


def test_resolve_falls_back_to_title_when_portal_blocks():
    out = sr.resolve({"url": "https://portal.example/p", "title": "A long exact paper title here"}, CFG["resolve"],
                     time.monotonic() + 5, fetch=lambda u, timeout_s: (None, None, "http 403"),
                     lookup=lambda d, t: None,
                     by_title=lambda title, t, m: {"title": title, "url": "https://doi.org/10.1/t", "snippet": "a", "doi": "10.1/t"})
    assert out and out[0]["doi"] == "10.1/t"


def test_resolve_stops_at_deadline():
    calls = []
    out = sr.resolve({"url": "https://news.example/a", "title": "t"}, CFG["resolve"], time.monotonic() - 1,
                     fetch=lambda u, timeout_s: calls.append(u) or ("", u, None), lookup=lambda d, t: None,
                     by_title=lambda *a: None)
    assert out == [] and calls == []


def test_lookup_title_rejects_weak_matches():
    class R:
        status_code = 200

        @staticmethod
        def json():
            return {"results": [{"title": "Something else entirely about cats", "doi": "https://doi.org/10.1/c",
                                 "abstract_inverted_index": {"x": [0]}}]}
    assert sr.lookup_title("AI coding assistants and developer productivity", 1, 0.8, http_get=lambda *a, **k: R) is None


def _l3_open(monkeypatch):
    """These tests exercise the retained state-A function; open the L3 seam in the test only (the shipped stub returns [])."""
    from vera import answer_contract
    monkeypatch.setattr(answer_contract, "examined_full_documents", lambda c: ["examined"])


def _no_router(monkeypatch):
    _l3_open(monkeypatch)
    from vera import scope_router
    monkeypatch.setattr(scope_router, "route", lambda *a, **k: scope_router.ScopeDecision(True, "layer1", "ok"))


def test_grounded_path_answers_from_sources_and_checks_citations(monkeypatch):
    _no_router(monkeypatch)
    seen = {}

    def fake_complete(messages, model, pricing, **kw):
        seen["system"] = messages[0]["content"]
        return AskResult("Claim [1]. Bogus [4].", 100, 0.0)

    monkeypatch.setattr(ask_service, "complete", fake_complete)
    src = [{"n": 1, "title": "T", "url": "https://arxiv.org/abs/1", "published": None, "snippet": "abstract"}]
    r = ask_service.answer_in_scope("Do AI assistants help?", "m", PRICING, grounding=(lambda: src, CFG))
    assert "[1] T" in seen["system"] and "Use ONLY the numbered sources" in seen["system"]
    assert r.answer == "Claim [1]. Bogus .\n\nNOTE" and r.sources == tuple(src)


def test_grounded_path_never_answers_from_memory(monkeypatch):
    _no_router(monkeypatch)
    monkeypatch.setattr(ask_service, "complete", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    r = ask_service.answer_in_scope("q", "m", PRICING, grounding=(lambda: [], CFG))
    assert r.answer == "NO SOURCES" and r.sources == ()


def _ask_with_kind(monkeypatch, kind):
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    import main
    from vera import public_mode as pm
    pm.reset_limiter(None)
    src = [{"n": 1, "title": "T", "url": "https://arxiv.org/abs/1", "published": "2025-01-01", "snippet": "secret abstract"}]
    monkeypatch.setattr(main, "GROUNDING", CFG)
    monkeypatch.setattr(main, "retrieve_all", lambda q, cfg: (src, [{"title": "U", "url": "https://doi.org/10.1/u",
                                                                      "published": None, "discovered_via": "x"}]))

    def fake_chain(q, entries, grounding):
        sources = grounding[0]()
        return AskResult("A [1]", 5, 0.0, sources=tuple(sources), result_kind=kind), "groq:m"

    monkeypatch.setattr(main, "answer_via_chain", fake_chain)
    monkeypatch.setattr(main, "store_from_env", lambda cfg: None)  # FIX-I6-1: never reach the production DB
    r = TestClient(main.app).post("/ask", json={"question": "q"}, headers={"X-API-Key": "k" * 20})
    assert r.status_code == 200, r.text
    return r


def test_ask_response_carries_sources_and_traced(monkeypatch):
    """Answer kind (L3 path preserved): public sources and traced sources are returned."""
    r = _ask_with_kind(monkeypatch, "answer")
    body = r.json()
    assert body["sources"][0]["title"] == "T" and "snippet" not in body["sources"][0]
    assert body["traced_sources"][0]["title"] == "U" and "secret abstract" not in r.text


def test_ask_response_non_answer_kind_has_no_sources_or_traced(monkeypatch):
    r = _ask_with_kind(monkeypatch, "reading_list")
    body = r.json()
    assert body["sources"] == [] and body["traced_sources"] == [] and body["claim_check"] is None
    assert "secret abstract" not in r.text
