"""OPENALEX-FOLLOWUPS-1110 tests: F1 (per-item list-ingest codes for each OpenAlex refusal) and F3 (resp.request.url
scrubbed; a 404 is not a retry attempt). Offline only: fake transports and clocks; SYNTHETIC identifiers."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import requests

from vera import list_ingest, rate_limit
import vera.search_providers as sp
from vera.eval_config import load_eval_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_eval_config().search
KEY = "synthetic-test-key-000"
DAY = "2026-11-10"
WALL0 = 1_794_268_800.0


class T:
    def __init__(self):
        self.t = 1000.0

    def mono(self):
        return self.t

    def wall(self):
        return WALL0 + self.t


class Req:
    def __init__(self, url):
        self.url = url


class R:
    def __init__(self, status=200, doc=None, headers=None, url="https://api.openalex.org/works", request=None, history=()):
        self.status_code, self.headers, self.url = status, dict(headers or {}), url
        self.request, self.history = request, list(history)
        self._doc = {"results": []} if doc is None else doc
        self.text = json.dumps(self._doc)

    def json(self):
        return self._doc


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real network transport called in a test")
    monkeypatch.setattr(sp, "openalex_default_transport", boom)
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    yield


@pytest.fixture
def clk(monkeypatch):
    c = T()
    rate_limit.set_registry_for_tests(rate_limit.Registry(rate_limit.load_config(env={}), clock=c.mono, wall=c.wall,
                                                          sleep=lambda s: None, rand=lambda: 0.0))
    sp._QUOTA.reset()
    return c


def gate_on(monkeypatch):
    real = json.loads(rate_limit.CONFIG_PATH.read_text(encoding="utf-8"))["portals"]["openalex"]
    monkeypatch.setattr(sp, "_oa_raw_portal", lambda: {**real, "terms_gate": "on"})


def prov(clk, responder):
    sent = []

    def tx(url, params, headers, timeout):
        sent.append({"url": url, **params})
        out = responder(url, params)
        if isinstance(out, BaseException):
            raise out
        return out
    return sp.OpenAlexProvider(CFG["openalex"], transport=tx, today=lambda: DAY, clock=clk.mono, wall=clk.wall), sent


def no_inspire(monkeypatch):
    real = list_ingest._api_get

    def api_get(url, params, t):
        if "inspirehep" in url:
            raise OSError("inspire off in this test")
        return real(url, params, t)
    monkeypatch.setattr(list_ingest, "_api_get", api_get)


def route_via(monkeypatch, holder, inspire_answers=False):
    """list_ingest's gated GET (the R5 transport) goes to the current provider's fake transport. INSPIRE is off, or
    (inspire_answers=True, B7FIX-1112) answers with no hits, returned here and not counted in `sent`; under S6-1 an
    OpenAlex "not found" is definitive only when INSPIRE answered."""
    def api_get(url, params, t):
        if "inspirehep" in url:
            if inspire_answers:
                return R(200, {"hits": {"hits": []}})
            raise OSError("inspire off in this test")
        return holder[0]._transport(url, params, {}, t)
    monkeypatch.setattr(list_ingest, "_api_get", api_get)


def rc():
    return json.loads((ROOT / "config" / "list_ingest.json").read_text(encoding="utf-8"))["result_codes"]


EXPECT = {  # OpenAlexRefused code -> per-item code
    "terms_gate": "no-route-arxiv-lookup-paused",
    "terms_gate_config_invalid": "no-route-arxiv-lookup-config-invalid",
    "own_cap": "arxiv-lookup-own-cap",
    "seed_share": "arxiv-lookup-seed-share",
    "provider_quota": "arxiv-lookup-provider-quota",
    "cooldown": "arxiv-lookup-cooldown",
    "retry_limit": "arxiv-lookup-retry-limit",
}


def _ingest_refused(code):
    def boom(a):
        raise sp.OpenAlexRefused(code, "synthetic refusal")
    out = list_ingest.ingest(["arXiv:2601.00001"], resolvers=list_ingest.Resolvers(
        arxiv_original=boom, known=lambda *a: False), start_worker=False)
    return [x["code"] for x in out["results"]]


@pytest.mark.parametrize("code,want", sorted(EXPECT.items()))
def test_f1_each_refusal_has_its_own_item_code(code, want):
    assert _ingest_refused(code) == [want]


def test_f1_codes_distinct_and_none_is_generic_no_route():
    got = [_ingest_refused(c)[0] for c in EXPECT]
    assert len(set(got)) == len(EXPECT) and "no-route" not in got
    rcodes = rc()
    for v in EXPECT.values():  # every emitted code is declared in config result_codes, and nothing else collides
        assert v in rcodes.values()
    assert len(set(rcodes.values())) == len(rcodes)
    assert set(sp.OpenAlexRefused.__doc__.split("[oa-refusal")[0].replace("|", " ").split()) >= {"retry_limit"}


def test_f1_other_failures_and_other_kinds_get_no_openalex_item_code():
    """B7FIX-1112: after the list-ingest S6-1 merge a transient failure is the retryable lookup-unavailable (was the
    terminal no-route); the F1 intent is unchanged: neither case gets an OpenAlex per-item code."""
    def boom(a):
        raise OSError("transport down")
    out = list_ingest.ingest(["arXiv:2601.00001"], resolvers=list_ingest.Resolvers(
        arxiv_original=boom, known=lambda *a: False), start_worker=False)
    got = [x["code"] for x in out["results"]]
    assert got == ["lookup-unavailable"] and got[0] not in EXPECT.values()

    def pm(p):
        raise sp.OpenAlexRefused("own_cap", "x")  # a non-arXiv item never gets an arXiv-lookup code
    out = list_ingest.ingest(["PMID:12345678"], resolvers=list_ingest.Resolvers(pmid=pm, known=lambda *a: False),
                             start_worker=False)
    got = [x["code"] for x in out["results"]]
    assert got == ["lookup-unavailable"] and got[0] not in EXPECT.values()


def test_f1_real_refusals_end_to_end(clk, monkeypatch):
    """Real client refusals (not a hand-raised exception): retry_limit after a 500, cooldown after a 429."""
    gate_on(monkeypatch)
    p, sent = prov(clk, lambda u, prm: R(500))
    holder = [p]
    route_via(monkeypatch, holder, inspire_answers=True)
    monkeypatch.setattr(sp, "shared_openalex_provider", lambda: holder[0])
    first = list_ingest.ingest(["arXiv:2601.00001"], resolvers=list_ingest.Resolvers(known=lambda *a: False),
                               start_worker=False)
    second = list_ingest.ingest(["arXiv:2601.00001"], resolvers=list_ingest.Resolvers(known=lambda *a: False),
                                start_worker=False)
    # B7FIX-1112: a 5xx answer is transient under S6-1 (retryable lookup-unavailable), no longer a terminal not-found
    assert [x["code"] for x in first["results"]] == ["lookup-unavailable"]
    assert [x["code"] for x in second["results"]] == ["arxiv-lookup-retry-limit"] and len(sent) == 1
    p2, sent2 = prov(clk, lambda u, prm: R(429, headers={"Retry-After": "120"}))
    holder[0] = p2
    sp._QUOTA.reset()
    list_ingest.ingest(["arXiv:2601.00002"], resolvers=list_ingest.Resolvers(known=lambda *a: False), start_worker=False)
    third = list_ingest.ingest(["arXiv:2601.00003"], resolvers=list_ingest.Resolvers(known=lambda *a: False),
                               start_worker=False)
    assert [x["code"] for x in third["results"]] == ["arxiv-lookup-cooldown"] and len(sent2) == 1


# ----------------------------------------------------------------------------------------------------- F3
def test_f3_404_is_not_a_retry_attempt_but_500_is(clk, monkeypatch):
    gate_on(monkeypatch)
    p, sent = prov(clk, lambda u, prm: R(404, {}))
    for _ in range(3):  # list_ingest limit is 1 attempt; a 404 answer must not consume it
        r = sp.openalex_request("https://api.openalex.org/works/x", {"select": "doi"}, purpose="list_ingest",
                                timeout_s=1.0, item="arxiv:2601.00009", provider=p)
        assert r.status_code == 404
    assert len(sent) == 3
    p5, sent5 = prov(clk, lambda u, prm: R(500))
    sp.openalex_request("https://api.openalex.org/works/x", {"select": "doi"}, purpose="list_ingest", timeout_s=1.0,
                        item="arxiv:2601.00010", provider=p5)
    with pytest.raises(sp.OpenAlexRefused) as ei:
        sp.openalex_request("https://api.openalex.org/works/x", {"select": "doi"}, purpose="list_ingest",
                            timeout_s=1.0, item="arxiv:2601.00010", provider=p5)
    assert ei.value.code == "retry_limit" and len(sent5) == 1


def test_f3_404_then_list_ingest_reports_original_not_found_each_time(clk, monkeypatch):
    gate_on(monkeypatch)
    p, sent = prov(clk, lambda u, prm: R(404, {}))
    route_via(monkeypatch, [p], inspire_answers=True)
    monkeypatch.setattr(sp, "shared_openalex_provider", lambda: p)
    for _ in range(2):
        out = list_ingest.ingest(["arXiv:2601.00011"], resolvers=list_ingest.Resolvers(known=lambda *a: False),
                                 start_worker=False)
        assert [x["code"] for x in out["results"]] == ["no-route-arxiv-original-not-found"]
    assert len(sent) == 2
    # B7FIX-1112: with INSPIRE down the 404 is not definitive (S6-1): retryable each time, and still never retry-limit
    p2, sent2 = prov(clk, lambda u, prm: R(404, {}))
    route_via(monkeypatch, [p2])
    monkeypatch.setattr(sp, "shared_openalex_provider", lambda: p2)
    for _ in range(2):
        out = list_ingest.ingest(["arXiv:2601.00012"], resolvers=list_ingest.Resolvers(known=lambda *a: False),
                                 start_worker=False)
        assert [x["code"] for x in out["results"]] == ["lookup-unavailable"]
    assert len(sent2) == 2


@pytest.mark.parametrize("shape", ["request", "history", "both"])
def test_f3_request_url_is_scrubbed(clk, monkeypatch, caplog, shape):
    gate_on(monkeypatch)
    monkeypatch.setenv("OPENALEX_API_KEY", KEY)
    caplog.set_level(logging.DEBUG)

    def responder(u, prm):
        full = f"{u}?select=doi&api_key={prm['api_key']}"
        hop = R(302, {}, url=full, request=Req(full))
        return R(200, {"results": []}, url=full, request=Req(full) if shape != "history" else None,
                 history=[hop] if shape != "request" else [])
    p, sent = prov(clk, responder)
    resp = sp.openalex_request("https://api.openalex.org/works", {"select": "doi"}, purpose="search", timeout_s=1.0,
                               provider=p)
    assert sent[0]["api_key"] == KEY  # the key was really sent
    blob = " ".join([str(resp.url), str(getattr(resp.request, "url", "")),
                     *[str(h.url) + str(getattr(h.request, "url", "")) for h in resp.history]])
    assert KEY not in blob and KEY not in caplog.text
    if shape != "history":
        assert "api_key" not in str(resp.request.url) or KEY not in str(resp.request.url)
    # real requests objects: the PreparedRequest url is settable and scrubbed
    prep = requests.Request("GET", "https://api.openalex.org/works", params={"api_key": KEY}).prepare()
    rr = requests.Response()
    rr.status_code, rr.url, rr.request = 200, prep.url, prep
    sp._scrub_resp_url(rr, KEY)
    assert KEY not in rr.url and KEY not in rr.request.url


def test_f3_keyless_response_urls_untouched(clk, monkeypatch):
    gate_on(monkeypatch)
    u = "https://api.openalex.org/works?select=doi"
    p, _ = prov(clk, lambda url, prm: R(200, {"results": []}, url=u, request=Req(u)))
    resp = sp.openalex_request("https://api.openalex.org/works", {"select": "doi"}, purpose="search", timeout_s=1.0,
                               provider=p)
    assert resp.url == u and resp.request.url == u
