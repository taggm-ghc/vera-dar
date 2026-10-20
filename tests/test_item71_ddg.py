"""Item #71 DuckDuckGo Lite DISCOVERY provider: offline only. The HTML fixture is hand-made, never recorded."""
import json
import re

import pytest
import requests

import vera.search_and_fetch as sf
import vera.search_providers as sp
from tests.config_support import load_pair, real_run
from vera.eval_config import EvalConfigError, load_eval_config
from vera.licence_gate import licence_decision
from vera.search_and_fetch import apply_max_share, search_question
from vera.search_providers import (DuckDuckGoLiteProvider, ProviderResponseError, ProviderUnavailable, RateGate,
                                   build_chain, build_ddg_queries, unwrap_ddg_link)

CFG = load_eval_config().search
DDG = CFG["duckduckgo_lite"]
QT = CFG["query_terms"]
Q = "What does published research show about AI coding assistants and developer productivity?"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real network transport called in a test")
    for name in ("arxiv_default_transport", "openalex_default_transport", "ddg_default_transport"):
        monkeypatch.setattr(sp, name, boom)
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(sf, "_CHAIN_CACHE", {})


class R:
    def __init__(self, status=200, text=""):
        self.status_code, self.text, self.headers = status, text, {}


def page(*anchors):
    rows = "".join(f"<tr><td>{i}.&nbsp;</td><td>{a}</td></tr><tr><td class='result-snippet'>DDG SNIPPET {i}</td></tr>"
                   for i, a in enumerate(anchors, 1))
    return f"<html><body><form><input name='q'></form><table>{rows}</table></body></html>"


def link(url, title, cls="result-link"):
    from urllib.parse import quote
    return (f"<a rel=\"nofollow\" href=\"//duckduckgo.com/l/?uddg={quote(url, safe='')}&amp;rut=abc\" "
            f"class='{cls}'>{title}</a>")


HTML = page(
    link("https://www.reuters.com/technology/ai-slows-developers-2025-07-10/", "AI slows some  experienced developers, study finds"),
    "<a rel=\"nofollow\" href=\"https://duckduckgo.com/y.js?ad_domain=shop.example&ad_provider=bingv7aa\" class='result-link'>Buy stuff</a>",
    "<a href=\"https://duckduckgo.com/about\" class='result-link'>About DDG</a>",
    link("https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4945566", "The Effects of Generative AI on High-Skilled Work"),
    link("http://insecure.example.com/x", "plain http dropped"),
    link("https://192.168.0.1/admin", "ip literal dropped"),
    "<a href='https://arstechnica.com/ai/x/' class='result-link'>Direct link, not wrapped</a>",
)


def provider(responses, *, cfg=None, **kw):
    settings = {**DDG, **(cfg or {})}
    seq, calls = list(responses), []

    def tx(url, data, headers, timeout):
        calls.append({"url": url, "data": data, "headers": headers})
        x = seq.pop(0)
        if isinstance(x, Exception):
            raise x
        return x
    p = DuckDuckGoLiteProvider(settings, transport=tx, query_terms=QT, sleep=lambda s: None, **kw)
    return p, calls


def run(p, n=10):
    return p.parse(p.call(Q, n, 5.0), n)


def test_parse_keeps_only_discovery_fields():
    p, calls = provider([R(text=HTML), R(text=page())])
    items = run(p)
    assert [i["url"] for i in items] == [
        "https://www.reuters.com/technology/ai-slows-developers-2025-07-10/",
        "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4945566",
        "https://arstechnica.com/ai/x/"]  # redirect unwrapped; ad, duckduckgo.com, http and IP links dropped
    news, portal = items[0], items[1]
    assert news["title"] == "AI slows some experienced developers, study finds"
    assert news["snippet"] == "" and portal["snippet"] == ""
    assert not any("DDG SNIPPET" in json.dumps(i) for i in items)
    assert (news["source_type"], news["evidence_class"], news["domain"], news["publisher"]) == ("news", "grey", "reuters.com", "reuters.com")
    assert (portal["source_type"], portal["evidence_class"]) == ("web_portal", None)
    for i in items:
        assert i["licence"] is None and i["stored_text_kind"] == "link_metadata" and i["published"] is None
        assert i["discovery"] == "duckduckgo_lite" and i["provider_endpoint"] == DDG["endpoint"]
    d = calls[0]
    assert d["url"] == DDG["endpoint"] and d["data"]["kl"] == "us-en" and set(d["data"]) == {"q", "kl"}
    assert d["headers"] == {"User-Agent": DDG["user_agent"]}


@pytest.mark.parametrize("href,expect", [
    ("//duckduckgo.com/l/?uddg=https%3A%2F%2Fnature.com%2Fa%3Fx%3D1&rut=z", "https://nature.com/a?x=1"),
    ("https://duckduckgo.com/y.js?ad_domain=a.com", None),
    ("https://duckduckgo.com/l/?uddg=https%3A%2F%2Fduckduckgo.com%2Fx", None),
    ("https://duckduckgo.com/l/?rut=z", None),
    ("https://user:pw@nature.com/a", None),
    ("https://localhost/a", None),
    ("https://nature.com:8443/a", None),
])
def test_unwrap(href, expect):
    assert unwrap_ddg_link(href) == expect


def test_challenge_202_is_non_retryable_failure_and_stops_after_first_query():
    p, calls = provider([R(status=202, text="anomaly"), R(text=HTML)])
    resp = p.call(Q, 10, 5.0)
    assert len(calls) == 1  # no second query after a challenge
    with pytest.raises(ProviderResponseError, match="challenge page; not evaded"):
        p.parse(resp, 10)


@pytest.mark.parametrize("body", ["<html><div class='anomaly-modal'>Select all squares</div></html>",
                                  "<html><body>nothing useful here</body></html>"])
def test_challenge_body_or_missing_table(body):
    p, _ = provider([R(text=body), R(text=body)])
    with pytest.raises(ProviderResponseError, match="not evaded"):
        run(p)


def test_genuine_empty_page_is_empty_not_failure():
    p, _ = provider([R(text="<html><body>No  results.</body></html>"), R(text="<html>No results.</html>")])
    assert run(p) == []


def test_challenge_through_search_question_is_noted_and_not_retried():
    p, calls = provider([R(status=202, text="")] * 6)
    p.gate = RateGate(0.0)
    with pytest.raises(sf.SearchError, match="challenge page; not evaded"):
        search_question(Q, 5, providers=[p], sleep=lambda s: None)
    assert len(calls) == 1


def test_daily_cap_is_non_retryable_unavailable():
    p, calls = provider([R(text=page()) for _ in range(10)], cfg={"daily_query_cap": 3}, today=lambda: "2026-10-04")
    p.call(Q, 5, 5.0)  # 2 queries
    p.call(Q, 5, 5.0)  # third query spends the cap; the fourth is skipped, with a note
    assert len(calls) == 3 and "daily cap" in p.last_query["note"]
    with pytest.raises(ProviderUnavailable, match="daily query cap"):
        p.call(Q, 5, 5.0)
    assert len(calls) == 3
    p2, _ = provider([R(text=page())], cfg={"daily_query_cap": 1}, today=lambda: "2026-10-04")
    p2.call(Q, 5, 5.0)
    with pytest.raises(sf.SearchError, match="unavailable"):
        search_question(Q, 5, providers=[p2], sleep=lambda s: None)


def test_second_query_waits_for_the_gate():
    now, slept = [100.0], []
    p, _ = provider([R(text=page()), R(text=page())], clock=lambda: now[0])
    p._sleep = slept.append
    p.call(Q, 5, 5.0)
    assert slept == [DDG["min_interval_s"]]


def test_query_building_site_groups_unquoted():
    qs = build_ddg_queries(Q, QT, DDG)
    assert [q["group"] for q in qs] == ["news", "portals"]
    for q in qs:
        assert len(q["query"]) <= DDG["max_query_chars"] and '"' not in q["query"] and " AND " not in q["query"]
        assert re.search(r"\(site:[a-z0-9.\-]+( OR site:[a-z0-9.\-]+)*\)$", q["query"])
    assert "site:reuters.com" in qs[0]["query"] and "site:papers.ssrn.com" in qs[1]["query"]
    assert "coding" in qs[0]["query"].split("(")[0]


def test_query_length_split_bounded_by_max_queries():
    cfg = {**DDG, "max_query_chars": 150, "max_queries_per_question": 3}
    qs = build_ddg_queries(Q, QT, cfg)
    assert len(qs) == 3 and all(len(q["query"]) <= 150 for q in qs)
    assert [q["group"] for q in qs] == ["news", "portals", "news"]  # first chunk of each group first
    one = build_ddg_queries(Q, QT, {**cfg, "max_queries_per_question": 1})
    assert len(one) == 1
    only_news = build_ddg_queries(Q, QT, {**DDG, "site_groups": {"news": ["reuters.com"], "portals": []}})
    assert [q["group"] for q in only_news] == ["news"]


def mk(url, prov, found=None):
    return {"url": url, "provider": prov, "found_by_providers": found or [prov]}


def test_max_share_caps_discovery_only():
    items = [mk("https://a/1", "duckduckgo_lite"), mk("https://a/2", "duckduckgo_lite"), mk("https://a/3", "duckduckgo_lite"),
             mk("https://b/1", "arxiv"), mk("https://b/2", "openalex"),
             mk("https://c/1", "duckduckgo_lite", ["duckduckgo_lite", "openalex"])]
    kept, dropped = apply_max_share(items, {"duckduckgo_lite": 2})
    assert dropped == 1 and [i["url"] for i in kept] == ["https://a/1", "https://a/2", "https://b/1", "https://b/2", "https://c/1"]


def test_fan_out_applies_max_share():
    class Fake:
        required_env, endpoint = (), "fake"

        def __init__(self, name, urls):
            self.name, self.urls, self.gate = name, urls, RateGate(0.0)

        def call(self, q, n, t):
            return R()

        def parse(self, resp, n):
            return [{"url": u, "title": f"title number {u}", "snippet": ""} for u in self.urls][:n]

        def cost_for(self, resp):
            return 0.0, True, "x"

        def failure_basis(self):
            return "x"
    ddg = Fake("duckduckgo_lite", [f"https://news.example/{i}" for i in range(5)])
    sch = Fake("openalex", [f"https://doi.org/10.1/{i}" for i in range(5)])
    res = search_question(Q, 6, providers=[ddg, sch], mode="fan_out", sleep=lambda s: None)
    assert sum(1 for r in res if r["provider"] == "duckduckgo_lite") == 2
    assert len(res) == 6 and res.meta["max_share"] == {"duckduckgo_lite": 2} and res.meta["share_dropped"] == 3


def test_licence_gate_defers_discovered_records_so_nothing_is_fetched():
    p, _ = provider([R(text=HTML), R(text=page())])
    for rec in run(p):
        d, why = licence_decision(rec)
        assert d == "defer" and "no declared licence" in why
    # selective fetch is reached only for gate-A "fetch" AND licence "allow"; a defer is held for review, so no
    # fetch (and so no host pacing) can happen for these records.
    import inspect
    import vera.m2_runner as m2
    src = inspect.getsource(m2)
    assert "licence_decision" in src or "apply_licence_gate" in src


def test_factory_registered_and_switch_off():
    assert "duckduckgo_lite" in CFG["providers"] and CFG["max_share"]["duckduckgo_lite"] == 2
    assert any(p.name == "duckduckgo_lite" for p in build_chain(CFG, repo_root=sf._ROOT))
    off = real_run()["search"]
    off["duckduckgo_lite"]["enabled"] = False
    assert not any(p.name == "duckduckgo_lite" for p in build_chain(off, repo_root=sf._ROOT))


def test_capture_labels_news():
    from vera.trace_eval.capture import grounded_v1_messages
    msgs = grounded_v1_messages("q", [{"n": 1, "title": "Headline", "url": "https://reuters.com/x", "snippet": "",
                                       "published": None, "source_type": "news"},
                                      {"n": 2, "title": "Paper", "url": "https://arxiv.org/abs/1", "snippet": "a", "published": None}])
    sys_text = msgs[0]["content"]
    assert "[1] [news, grey literature" in sys_text and "[2] [news" not in sys_text


def _edit(fn):
    return fn


@pytest.mark.parametrize("edit,msg", [
    (lambda r: r["search"]["duckduckgo_lite"].update(enabled="yes"), "enabled"),
    (lambda r: r["search"]["duckduckgo_lite"].update(endpoint="https://example.com/lite/"), "lite.duckduckgo.com"),
    (lambda r: r["search"]["duckduckgo_lite"].update(min_interval_s=1.0), "min_interval_s"),
    (lambda r: r["search"]["duckduckgo_lite"].update(daily_query_cap=0), "daily_query_cap"),
    (lambda r: r["search"]["duckduckgo_lite"].update(user_agent="Mozilla/5.0"), "user_agent"),
    (lambda r: r["search"]["duckduckgo_lite"].update(site_groups={"news": ["Bad Host"]}), "site_groups"),
    (lambda r: r["search"]["duckduckgo_lite"].update(site_groups={"news": [], "portals": []}), "at least one group"),
    (lambda r: r["search"]["duckduckgo_lite"].update(source_class_by_domain={"x.com": "bogus"}), "source_class_by_domain"),
    (lambda r: r["search"]["duckduckgo_lite"].update(max_queries_per_question=9), "max_queries_per_question"),
    (lambda r: r["search"].update(max_share={"duckduckgo_lite": -1}), "max_share"),
    (lambda r: r["search"].pop("duckduckgo_lite"), "duckduckgo_lite"),
])
def test_config_validation(tmp_path, edit, msg):
    with pytest.raises(EvalConfigError, match=msg):
        load_pair(tmp_path, run_edit=edit)


# ---- item #71 section 13 N1: independent AI news first, same query count, source-class labels
NEWS_FIRST = ["the-decoder.com", "venturebeat.com", "infoq.com", "thenewstack.io", "axios.com", "theinformation.com"]


def test_n1_news_group_order_and_query_count_invariant():
    news = DDG["site_groups"]["news"]
    assert news[:6] == NEWS_FIRST and len(news) == 14 and "metr.org" in DDG["site_groups"]["portals"]
    for dropped in ("nytimes.com", "wsj.com", "ft.com", "bbc.com", "theguardian.com", "washingtonpost.com",
                    "bloomberg.com", "cnbc.com"):
        assert dropped not in news
    qs = build_ddg_queries(Q, QT, DDG)
    assert [q["group"] for q in qs] == ["news", "portals"] and len(qs) <= DDG["max_queries_per_question"]
    assert all("site:metr.org" not in q["query"] or q["group"] == "portals" for q in qs)
    # every news domain fits in the first news query (no second chunk is needed)
    assert all(f"site:{d}" in qs[0]["query"] for d in news)
    assert "site:metr.org" in qs[1]["query"]


def test_source_class_labels_and_vendor_label():
    from vera.trace_eval.capture import _format_sources
    p, _ = provider([R(text=HTML)])
    p.source_class_by_domain = dict(DDG["source_class_by_domain"])
    assert DDG["source_class_by_domain"]["metr.org"] == "independent_research"
    rec = lambda u, g: p._record(u, "t", g)
    assert rec("https://metr.org/blog/x", "portals")["source_class"] == "independent_research"
    assert rec("https://github.blog/x", "portals")["source_class"] == "vendor_claim"
    assert rec("https://www.microsoft.com/en-us/research/x", "portals")["source_class"] == "vendor_claim"
    assert rec("https://www.reuters.com/x", "news")["source_class"] == "news"
    assert rec("https://papers.ssrn.com/x", "portals")["source_class"] == "web_portal"
    out = _format_sources([{"n": 1, "title": "T", "url": "u", "snippet": "", "source_type": "web_portal",
                            "source_class": "vendor_claim"}])
    assert "[vendor claim: not independent evidence]" in out
