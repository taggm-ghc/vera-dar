"""Item #71 sections 10-11: DOAJ provider, Google Scholar link-out, DuckDuckGo cache. Offline only; fixtures hand-made."""
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
import requests

import vera.search_and_fetch as sf
import vera.search_providers as sp
from tests.config_support import load_pair, real_run
from vera.eval_config import EvalConfigError, load_eval_config
from vera.search_providers import (DOAJProvider, DuckDuckGoLiteProvider, ProviderResponseError, ProviderUnavailable,
                                   build_chain)
from vera.search_query import build_query_terms, scholar_search_url

ROOT = Path(__file__).resolve().parent.parent
CFG = load_eval_config().search
DOAJ = CFG["doaj"]
DDG = CFG["duckduckgo_lite"]
QT = CFG["query_terms"]
QX = "How does GitHub Copilot affect bug rates in Python projects?"  # core + extra terms
Q = "What does published research show about AI coding assistants and developer productivity?"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real network transport called in a test")
    for name in ("arxiv_default_transport", "openalex_default_transport", "ddg_default_transport",
                 "doaj_default_transport"):
        monkeypatch.setattr(sp, name, boom)
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(sf, "_CHAIN_CACHE", {})


class R:
    def __init__(self, status=200, text="", headers=None):
        self.status_code, self.text, self.headers = status, text, headers or {}


def hit(i, year="2024", month="03", doi=True, fulltext=True, abstract="An abstract about assistants."):
    bib = {"title": f"Paper {i}", "year": year, "journal": {"title": f"Journal {i}", "language": ["EN"]},
           "identifier": [{"type": "eissn", "id": "1234-5678"}], "link": [{"type": "fulltext", "url": f"https://j.example/{i}"}]}
    if month:
        bib["month"] = month
    if abstract is not None:
        bib["abstract"] = abstract
    if doi:
        bib["identifier"].append({"type": "doi", "id": f"10.1000/x{i}"})
    if not fulltext:
        bib["link"] = []
    return {"id": f"d{i}", "bibjson": bib}


def body(*hits):
    return json.dumps({"total": len(hits), "results": list(hits)})


def doaj(responses, cfg=None, **kw):
    seq, calls = list(responses), []

    def tx(url, params, headers, timeout):
        calls.append({"url": url, "params": params, "headers": headers})
        return seq.pop(0)
    return DOAJProvider({**DOAJ, **(cfg or {})}, transport=tx, query_terms=QT, sleep=lambda s: None, **kw), calls


# ---------------------------------------------------------------------------------------------- DOAJ
def test_doaj_request_is_plain_terms_in_path_with_pagesize():
    p, calls = doaj([R(text=body(hit(1)))])
    p.call(Q, 7, 5.0)
    c = calls[0]
    assert c["url"].startswith("https://doaj.org/api/v4/search/articles/")
    terms = unquote(c["url"].rsplit("/", 1)[1])
    qt = build_query_terms(Q, QT)
    assert terms == " ".join(qt.core + qt.extra)
    assert '"' not in terms and " AND " not in terms and " " in terms and "%20" in c["url"]
    assert c["params"] == {"pageSize": 7} and "User-Agent" in c["headers"]


def test_doaj_parse_fields_year_filter_and_licence():
    p, _ = doaj([R(text=body(hit(1), hit(2, year="2021"), hit(3, year="2027"), hit(4, doi=False), hit(5, doi=False, fulltext=False),
                              hit(6, month=None, abstract="x" * 3000), hit(7, year="n/a")))])
    items = p.parse(p.call(Q, 10, 5.0), 10)
    assert [i["title"] for i in items] == ["Paper 1", "Paper 4", "Paper 6"]  # outside 2023-2026, no URL, no year: dropped
    a, b, c = items
    assert a["url"] == "https://doi.org/10.1000/x1" and a["doi"] == "10.1000/x1"
    assert a["published"] == "2024-03" and a["year"] == 2024 and a["publisher"] == "Journal 1"
    assert a["source_type"] == "journal-article" and a["language"] == ["EN"] and a["stored_text_kind"] == "abstract_metadata"
    assert a["licence"]["metadata"]["id"] == "CC0-1.0" and a["licence"]["content"]["id"] is None
    assert b["url"] == "https://j.example/4" and b["doi"] is None
    assert c["published"] == "2024" and len(c["snippet"]) <= DOAJ["snippet_max_chars"] + 3 and c["snippet"].endswith("...")
    assert p.cost_for(None) == (0.0, True, "keyless_doaj")


def test_doaj_zero_hits_retries_once_with_core_terms_and_records_it():
    slept = []
    p, calls = doaj([R(text=body()), R(text=body(hit(1)))], clock=lambda: 10.0)
    p._sleep = slept.append
    items = p.parse(p.call(QX, 5, 5.0), 5)
    assert len(items) == 1 and len(calls) == 2
    qt = build_query_terms(QX, QT)
    assert unquote(calls[1]["url"].rsplit("/", 1)[1]) == " ".join(qt.core)
    assert "retried once with core terms only" in p.last_query["note"]
    assert slept == [pytest.approx(DOAJ["min_interval_s"])]  # the retry keeps the gate spacing


def test_doaj_no_retry_when_hits_or_core_only_and_second_zero_is_empty():
    p, calls = doaj([R(text=body(hit(1)))])
    p.call(QX, 5, 5.0)
    assert len(calls) == 1
    p2, calls2 = doaj([R(text=body()), R(text=body())])
    assert p2.parse(p2.call(QX, 5, 5.0), 5) == [] and len(calls2) == 2  # at most one retry
    p3, calls3 = doaj([R(text=body())])
    p3.call(Q, 5, 5.0)  # core only: nothing to drop
    assert len(calls3) == 1


def test_doaj_bad_bodies_are_failures_not_empty():
    p, _ = doaj([])
    for text in ("not json", "[]", json.dumps({"error": "boom"})):
        with pytest.raises(ProviderResponseError):
            p.parse(R(text=text), 5)
    small, _ = doaj([], cfg={"max_response_bytes": 1024})
    with pytest.raises(ProviderResponseError, match="over 1024 bytes"):
        small.parse(R(text=" " * 2000), 5)


def test_doaj_daily_cap_counts_every_attempt_incl_retry_and_is_non_retryable():
    p, calls = doaj([R(text=body()), R(text=body()), R(text=body(hit(1)))], cfg={"daily_search_budget": 2},
                    today=lambda: "2026-10-04")
    p.call(QX, 5, 5.0)  # two attempts (retry) spend the cap
    with pytest.raises(ProviderUnavailable, match="daily search budget"):
        p.call(QX, 5, 5.0)
    assert len(calls) == 2
    p2, _ = doaj([R(text=body(hit(1)))], cfg={"daily_search_budget": 1}, today=lambda: "d1")
    p2.call(Q, 5, 5.0)
    with pytest.raises(sf.SearchError, match="unavailable"):
        search_q = sf.search_question
        search_q(Q, 5, providers=[p2], sleep=lambda s: None)
    day = ["d1"]
    p3, _ = doaj([R(text=body(hit(1))), R(text=body(hit(1)))], cfg={"daily_search_budget": 1}, today=lambda: day[0])
    p3.call(Q, 5, 5.0)
    day[0] = "d2"
    p3.call(Q, 5, 5.0)  # a new day resets the guard


def test_doaj_registered_in_chain_and_config():
    assert "doaj" in CFG["providers"]
    on = {**CFG, "doaj": {**DOAJ, "enabled": True}}  # shipped config has DOAJ switched off
    chain = build_chain(on, repo_root=ROOT)
    assert [p.name for p in chain] == ["arxiv", "openalex", "doaj", "duckduckgo_lite"]
    assert DOAJ["min_interval_s"] == 0.6 and DOAJ["daily_search_budget"] == 200 and (DOAJ["year_from"], DOAJ["year_to"]) == (2023, 2026)
    off = real_run()["search"]
    off["doaj"]["enabled"] = False
    assert not any(p.name == "doaj" for p in build_chain(off, repo_root=ROOT))


def test_doaj_goes_through_search_question_with_gate_and_meta():
    p, calls = doaj([R(text=body(hit(1), hit(2)))])
    res = sf.search_question(Q, 5, providers=[p], sleep=lambda s: None, min_candidates=1)
    assert [i["url"] for i in res] == ["https://doi.org/10.1000/x1", "https://doi.org/10.1000/x2"]
    assert "doaj" in res.meta["queries"]


@pytest.mark.parametrize("edit,msg", [
    (lambda r: r["search"]["doaj"].update(min_interval_s=0.4), "doaj.min_interval_s"),
    (lambda r: r["search"]["doaj"].update(daily_search_budget=0), "doaj.daily_search_budget"),
    (lambda r: r["search"]["doaj"].update(endpoint="http://doaj.org/x"), "doaj.endpoint"),
    (lambda r: r["search"]["doaj"].update(endpoint="https://example.com/x"), "host must be doaj.org"),
    (lambda r: r["search"]["doaj"].update(year_from=2027), "year_from"),
    (lambda r: r["search"]["doaj"].update(enabled="yes"), "doaj.enabled"),
    (lambda r: r["search"]["doaj"].pop("terms_source"), "doaj.terms_source"),
    (lambda r: r["search"].pop("doaj"), "search.doaj"),
    (lambda r: r["search"]["duckduckgo_lite"].update(cache_ttl_hours=0), "cache_ttl_hours"),
    (lambda r: r["search"]["duckduckgo_lite"].update(cache_dir="/etc/x"), "cache_dir"),
    (lambda r: r["search"]["duckduckgo_lite"].update(cache_dir="tmp/../x"), "cache_dir"),
])
def test_doaj_and_cache_config_validation(tmp_path, edit, msg):
    with pytest.raises(EvalConfigError, match=msg):
        load_pair(tmp_path, run_edit=edit)


# ---------------------------------------------------------------------------------------------- Scholar link-out
def test_scholar_url_is_a_pure_string_and_makes_no_network_call():
    url = scholar_search_url(Q, QT)  # the autouse fixture makes any requests call raise
    sp_ = urlsplit(url)
    assert (sp_.scheme, sp_.netloc, sp_.path) == ("https", "scholar.google.com", "/scholar")
    qt = build_query_terms(Q, QT)
    assert parse_qs(sp_.query) == {"q": [" ".join(qt.core + qt.extra)]}
    assert "AND" not in parse_qs(sp_.query)["q"][0]
    # no usable term: falls back to the raw question, never an empty query
    assert parse_qs(urlsplit(scholar_search_url("the of", QT)).query)["q"] == ["the of"]


def test_no_module_in_vera_references_scholar_host_except_the_url_function():
    hits = []
    for f in sorted((ROOT / "vera").rglob("*.py")):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if "scholar.google.com" in line:
                hits.append((f.relative_to(ROOT).as_posix(), n))
    # exactly one code reference (the f-less literal in scholar_search_url) plus its docstring mention
    assert {h[0] for h in hits} == {"vera/search_query.py"}, hits
    src = (ROOT / "vera" / "search_query.py").read_text(encoding="utf-8")
    code_lines = [ln for ln in src.splitlines() if "scholar.google.com" in ln and "return" in ln]
    assert len(code_lines) == 1
    # and no provider/transport is registered for it
    assert not any("scholar" in n for n in sp.PROVIDER_FACTORIES)


# ---------------------------------------------------------------------------------------------- DDG cache (R11-b/c)
def ddg_page(*pairs):
    from urllib.parse import quote
    rows = "".join(f"<tr><td><a href=\"//duckduckgo.com/l/?uddg={quote(u, safe='')}&amp;rut=a\" class='result-link'>{t}</a></td></tr>"
                   f"<tr><td class='result-snippet'>DDG SNIPPET {i}</td></tr>" for i, (u, t) in enumerate(pairs, 1))
    return f"<html><body><form><input name='q'></form><table>{rows}</table></body></html>"


PAGE = ddg_page(("https://www.reuters.com/a/", "Reuters headline"), ("https://papers.ssrn.com/x?id=1", "SSRN title"))


def ddg(responses, tmp_path, *, wall=None, cfg=None):
    seq, calls = list(responses), []

    def tx(url, data, headers, timeout):
        calls.append(data["q"])
        return seq.pop(0)
    p = DuckDuckGoLiteProvider({**DDG, **(cfg or {})}, transport=tx, query_terms=QT, sleep=lambda s: None,
                               today=lambda: "2026-10-04", cache_dir=tmp_path / "cache",
                               wall_clock=(lambda: wall[0]) if wall else None)
    return p, calls


def test_real_config_ddg_cap_and_cache_settings():
    assert DDG["daily_query_cap"] == 8 and DDG["cache_ttl_hours"] == 72
    assert DDG["cache_dir"] is None  # R-028 (2026-10-19): no query-derived cache file for public /ask (was tmp/search_cache/duckduckgo_lite)
    assert "tmp/search_cache/" in (ROOT / ".gitignore").read_text().splitlines()
    assert sp._ddg_cache_dir(DDG) is None


def test_cache_miss_writes_url_and_title_only_then_hit_makes_no_request_and_no_cap(tmp_path):
    p, calls = ddg([R(text=PAGE), R(text=PAGE)], tmp_path, cfg={"daily_query_cap": 2})
    live = p.parse(p.call(Q, 10, 5.0), 10)
    assert len(calls) == 2 and p._used == 2 and not p.last_call_cached
    files = sorted((tmp_path / "cache").glob("*.json"))
    assert len(files) == 2
    import hashlib
    assert {f.stem for f in files} == {hashlib.sha256(c.encode()).hexdigest() for c in calls}
    for f in files:
        doc = json.loads(f.read_text())
        assert all(set(i) == {"url", "title"} for i in doc["items"]) and "DDG SNIPPET" not in f.read_text()
    # second run: cap is spent (2/2) yet the call is served from cache
    assert p.gate_exempt(Q) is True
    cached = p.parse(p.call(Q, 10, 5.0), 10)
    assert len(calls) == 2 and p._used == 2 and p.last_call_cached and p.last_query["cache_hits"] == 2
    assert [i["url"] for i in cached] == [i["url"] for i in live]
    for i in live + cached:  # R11-c: only url + title + domain/meta fields; never a snippet
        assert i["snippet"] == "" and "DDG SNIPPET" not in json.dumps(i)
        assert i["licence"] is None and i["stored_text_kind"] == "link_metadata" and i["domain"]
    assert cached == live


def test_cached_calls_do_not_wait_on_or_stamp_the_gate(tmp_path):
    p, _ = ddg([R(text=PAGE), R(text=PAGE)], tmp_path)
    p.parse(p.call(Q, 10, 5.0), 10)
    now, slept = [100.0], []
    p.gate.stamp(100.0)  # a live call just ended: a live call would have to wait min_interval_s
    res = sf.search_question(Q, 10, providers=[p], sleep=slept.append, clock=lambda: now[0], min_candidates=1)
    assert len(res) == 2 and slept == [] and p.gate.last_end == 100.0


def test_expired_corrupt_and_malformed_entries_are_ignored(tmp_path):
    wall = [1000.0]
    p, calls = ddg([R(text=PAGE)] * 6, tmp_path, wall=wall)
    p.parse(p.call(Q, 10, 5.0), 10)
    assert len(calls) == 2
    wall[0] += 72 * 3600 - 1  # still fresh
    assert p.gate_exempt(Q)
    wall[0] += 2  # expired
    assert not p.gate_exempt(Q)
    p.parse(p.call(Q, 10, 5.0), 10)
    assert len(calls) == 4  # re-queried, entries rewritten
    for f in (tmp_path / "cache").glob("*.json"):
        f.write_text("{not json")
    assert not p.gate_exempt(Q)
    for f in (tmp_path / "cache").glob("*.json"):
        f.write_text(json.dumps({"stored_at": wall[0], "items": [{"url": "http://insecure.example/x", "title": "t"}]}))
    assert not p.gate_exempt(Q)  # unsafe URL in a cache entry: ignored
    for f in (tmp_path / "cache").glob("*.json"):
        f.write_text(json.dumps({"stored_at": wall[0], "items": [{"url": "https://nature.com/a"}]}))
    assert not p.gate_exempt(Q)  # missing title
    p.parse(p.call(Q, 10, 5.0), 10)
    assert len(calls) == 6


def test_challenge_is_never_cached(tmp_path):
    p, calls = ddg([R(status=202, text="")], tmp_path)
    with pytest.raises(ProviderResponseError):
        p.parse(p.call(Q, 10, 5.0), 10)
    assert not list((tmp_path / "cache").glob("*.json")) if (tmp_path / "cache").exists() else True


def test_no_cache_dir_means_no_cache_files(tmp_path):
    p = DuckDuckGoLiteProvider(DDG, transport=lambda *a: R(text=PAGE), query_terms=QT, sleep=lambda s: None)
    p.parse(p.call(Q, 10, 5.0), 10)
    assert p.cache_dir is None and not p.gate_exempt(Q)
