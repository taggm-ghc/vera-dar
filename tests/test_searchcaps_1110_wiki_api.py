"""SEARCHCAPS-1110 (3): Wikipedia article discovery never uses a DuckDuckGo site: query (OD-1110-SEARCHCAPS). Review F1
(2026-11-10): en.wikipedia.org/robots.txt disallows /w/, so the en.wikipedia.org/w/api.php route is DENIED and ships
with no endpoint: nothing is ever sent to en.wikipedia.org/w/. The api_search mechanics are tested with an injected
endpoint and fake transport only. Offline: fake transports only."""
import json

import pytest
import requests

from vera import api_permissions, rate_limit, search_and_fetch, selective_fetch, wiki_ref_portal as w

HOST = "en.wikipedia.org"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real network called in a test")
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(search_and_fetch, "cached_chain", boom)  # DuckDuckGo must never be reached
    rate_limit.set_registry_for_tests(rate_limit.Registry(rate_limit.load_config(env={}), sleep=lambda s: None))
    api_permissions.set_policy_for_tests(None)
    yield
    rate_limit.set_registry_for_tests(None)
    api_permissions.set_policy_for_tests(None)


class R:
    def __init__(self, status=200, body=b""):
        self.status_code, self.content, self.headers = status, body, {}


def _body(*titles):
    return json.dumps({"batchcomplete": True, "query": {"search": [{"ns": 0, "title": t} for t in titles]}}).encode()


def _shipped():
    c = w.load_config()
    assert c["enabled"] is False  # the shipped portal stays OFF; tests opt in on a copy
    return {**c, "enabled": True}


def _cfg():  # mechanics only: an injected endpoint on the article host (the shipped config has none)
    c = _shipped()
    return {**c, "search_api": {**c["search_api"], "endpoint": "https://en.wikipedia.org/w/api.php"}}


def test_shipped_search_api_config():
    a = w.load_config()["search_api"]
    assert "endpoint" not in a  # review F1: robots.txt Disallow: /w/
    p = a["params"]
    assert p["list"] == "search" and p["srprop"] == "" and p["srnamespace"] == "0" and p["maxlag"] == "5"
    ua = w.load_config()["user_agent"]
    assert "github.com/taggm-ghc/vera-dar" in ua and "@" not in ua  # contact URL, never an email


def test_api_search_titles_only_and_params():
    seen = {}

    def get(url, params, headers, timeout_s):
        seen.update(url=url, params=params, headers=headers)
        return R(body=_body("Photosynthesis", "Light-dependent reactions"))
    out = w.api_search("photosynthesis light", HOST, 5, _cfg(), get=get)
    assert out == [{"url": "https://en.wikipedia.org/wiki/Photosynthesis"},
                   {"url": "https://en.wikipedia.org/wiki/Light-dependent_reactions"}]
    assert seen["url"] == "https://en.wikipedia.org/w/api.php"
    assert seen["params"]["srsearch"] == "photosynthesis light" and seen["params"]["srlimit"] == "5"
    assert seen["headers"]["User-Agent"] == _cfg()["user_agent"]


@pytest.mark.parametrize("resp", [R(429), R(503), R(body=b'{"error": {"code": "maxlag"}}'), R(body=b"not json"),
                                  R(body=b"x" * 300000)])
def test_api_search_fails_closed(resp):
    try:
        out = w.api_search("photosynthesis", HOST, 5, _cfg(), get=lambda *a: resp)
    except w.PortalError:
        return
    assert out == []


def test_no_endpoint_or_foreign_host_means_no_search():
    c = _cfg()
    assert w.api_search("x y", HOST, 5, {**c, "search_api": None}, get=lambda *a: pytest.fail("sent")) == []
    bad = {**c, "search_api": {**c["search_api"], "endpoint": "https://evil.example/w/api.php"}}
    assert w.api_search("x y", HOST, 5, bad, get=lambda *a: pytest.fail("sent")) == []


def test_discover_uses_the_api_not_duckduckgo(monkeypatch):
    calls = []
    monkeypatch.setattr(w, "_api_get", lambda url, params, headers, t: (calls.append(url), R(body=_body("Zorblat")))[1])
    cands, _links, counts = w.discover("zorblat gait patterns", _cfg(), http_get=lambda *a, **k: R(404),
                                       url_check=lambda u: (True, "ok"), robots_ok=lambda u: False)
    assert calls == ["https://en.wikipedia.org/w/api.php"]
    assert counts["search"] == 1 and counts["articles_found"] == 1 and counts["robots_denied"] == 1


def test_permission_entry_is_denied_robots_first():
    url = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=x"
    d = api_permissions.decide(url, "metadata")
    assert not d.allowed and d.entry == "wikipedia_search_api" and d.why == "owner_deny"
    assert not api_permissions.decide(url, "fulltext").allowed
    assert not api_permissions.decide("https://en.wikipedia.org/w/index.php?title=X", "fulltext").allowed


def _record_pinned(monkeypatch, sent, body):
    def fake_pinned(url, *, headers=None, timeout=None, allow_redirects=False, stream=False, params=None,
                    api_purpose=None):
        sent.append(url)
        return R(body=body)
    monkeypatch.setattr(selective_fetch, "pinned_get", fake_pinned)


def test_nothing_is_sent_to_w_while_denied(monkeypatch):
    sent, pages = [], []
    _record_pinned(monkeypatch, sent, _body("Photosynthesis"))
    # shipped config (no endpoint): discovery sends nothing at all
    cands, _l, counts = w.discover("photosynthesis light", _shipped(), http_get=lambda u, **k: pages.append(u),
                                   url_check=lambda u: (True, "ok"), robots_ok=lambda u: True)
    assert cands == [] and counts["articles_found"] == 0 and sent == [] and pages == []
    # even with the endpoint injected, the denied permission entry stops the call before anything is sent
    with pytest.raises(api_permissions.ApiDenied):
        w.api_search("photosynthesis", HOST, 3, _cfg())
    assert sent == [] and rate_limit.get_registry().portal("wikipedia").n["requests"] == 0
    assert not any("/w/" in u for u in sent + pages)


def _permit_for_test(monkeypatch):
    pol = api_permissions.load_policy()
    e = dict(pol.entries["wikipedia_search_api"], owner_switch="permit", owner_decision="OD-1110-SEARCHCAPS")
    api_permissions.set_policy_for_tests(api_permissions.Policy(entries={**pol.entries, "wikipedia_search_api": e},
                                                                api_hosts=pol.api_hosts))


def test_gated_call_goes_through_the_wikipedia_portal_with_the_entry_ua(monkeypatch):
    _permit_for_test(monkeypatch)  # mechanics only: a test-local permit, never the shipped policy
    sent = []

    def fake_pinned(url, *, headers=None, timeout=None, allow_redirects=False, stream=False, params=None,
                    api_purpose=None):
        sent.append((url, dict(headers or {}), dict(params or {})))
        return R(body=_body("Photosynthesis"))
    monkeypatch.setattr(selective_fetch, "pinned_get", fake_pinned)
    out = w.api_search("photosynthesis", HOST, 3, _cfg())
    assert out == [{"url": "https://en.wikipedia.org/wiki/Photosynthesis"}]
    assert len(sent) == 1 and sent[0][0] == "https://en.wikipedia.org/w/api.php"
    assert sent[0][1]["User-Agent"].startswith("VERA-DAR-wiki-ref-portal/")
    assert rate_limit.get_registry().portal("wikipedia").n["requests"] == 1
