"""OPENALEX-FIX-1110 acceptance tests A1-A10 (plan work/OPENALEX-TERMS-1110/PLAN-fixes.md; decision records
CL-OPENALEX-METER, CL-OPENALEX-GATE, CL-LISTINGEST-PUBVER).

Offline only: fake transports and fake clocks; no socket, no DB, no model. Record shapes are the real ones: OpenAlex
`works` JSON with `locations[]` carrying doi.org `landing_page_url` and `version`, the real X-RateLimit-* header names,
Retry-After, and the real config files (config/rate_limits.json, config/list_ingest.json). SYNTHETIC identifiers."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest
import requests

from vera import (arxiv_portal, list_ingest, rate_limit, reference_portal as rp, request_origin, research_outcome,
                  source_resolver)
import vera.search_providers as sp
from vera.eval_config import load_eval_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_eval_config().search
KEY = "synthetic-test-key-000"
DAY = "2026-11-10"
WALL0 = 1_794_268_800.0  # 2026-11-10T00:00:00Z


class T:
    """One injected clock for the provider and the rate portal (no wall-clock waits)."""

    def __init__(self):
        self.t = 1000.0

    def mono(self):
        return self.t

    def wall(self):
        return WALL0 + self.t


class R:
    """A response in requests.Response shape (status_code, headers, text, json(), url)."""

    def __init__(self, status=200, doc=None, headers=None, url="https://api.openalex.org/works"):
        self.status_code, self.headers, self.url = status, dict(headers or {}), url
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
    reg = rate_limit.Registry(rate_limit.load_config(env={}), clock=c.mono, wall=c.wall, sleep=lambda s: None,
                              rand=lambda: 0.0)
    rate_limit.set_registry_for_tests(reg)
    sp._QUOTA.reset()
    return c


def gate(monkeypatch, value, *, missing=False):
    real = json.loads(rate_limit.CONFIG_PATH.read_text(encoding="utf-8"))["portals"]["openalex"]
    raw = {k: v for k, v in real.items() if k != "terms_gate"} if missing else {**real, "terms_gate": value}
    monkeypatch.setattr(sp, "_oa_raw_portal", lambda: raw)


def prov(clk, responder=None, sent=None):
    sent = [] if sent is None else sent

    def tx(url, params, headers, timeout):
        sent.append({"url": url, **params})
        out = responder(url, params) if responder else R()
        if isinstance(out, BaseException):
            raise out
        return out
    p = sp.OpenAlexProvider(CFG["openalex"], transport=tx, today=lambda: DAY, clock=clk.mono, wall=clk.wall)
    return p, sent


WORK = {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.48550/arxiv.2601.00001",
        "locations": [{"landing_page_url": "https://arxiv.org/abs/2601.00001", "version": "submittedVersion"},
                      {"landing_page_url": "https://doi.org/10.9999/synth.j.1", "version": "publishedVersion"}],
        "referenced_works": ["https://openalex.org/W20", "https://openalex.org/W21", "https://openalex.org/W22"]}


# ------------------------------------------------------------------------------------------------ route drivers
def r1(p):
    return p.call("quark gluon plasma viscosity", 5, 5.0)


def r2(p):
    tok = sp.OPENALEX_OA_FIELDS.set(True)
    try:
        return p.call("quark gluon plasma viscosity", 5, 5.0)
    finally:
        sp.OPENALEX_OA_FIELDS.reset(tok)


def r3(p, monkeypatch, s2=None):
    monkeypatch.setattr(sp, "shared_openalex_provider", lambda: p)
    s2_calls = []

    def http_get(url, params=None, headers=None, timeout=None):
        if "semanticscholar" in url:
            s2_calls.append(url)
            return s2 or R(404, {})
        return p._transport(url, params, headers, timeout)  # the OpenAlex leg, through the client's transport arg
    return source_resolver.lookup_doi("10.9999/synth.r3", 5.0, http_get=http_get), s2_calls


def r3_title(p, monkeypatch):
    monkeypatch.setattr(sp, "shared_openalex_provider", lambda: p)
    return source_resolver.lookup_title("Synthetic quark gluon plasma viscosity study", 5.0, 0.5,
                                        http_get=lambda u, params=None, headers=None, timeout=None:
                                        p._transport(u, params, headers, timeout))


def r4(p):
    return p.metadata_get({"filter": "doi:https://doi.org/10.48550/arxiv.2601.00001", "select": "id,doi"}, 5.0)


INSPIRE_EMPTY = R(200, {"hits": {"hits": []}})  # INSPIRE answered and knows no journal DOI (a definitive answer)


def r5(p, monkeypatch, inspire_answers=False):
    """inspire_answers=True (B7FIX-1112, A7 only): INSPIRE answers with no hits, outside `sent`, so the OpenAlex
    publishedVersion rule alone decides. Default: INSPIRE is down, and under S6-1 (list-ingest followups) "nothing
    found" is then not definitive (lookup-unavailable)."""
    monkeypatch.setattr(sp, "shared_openalex_provider", lambda: p)

    def api_get(url, params, t):
        if "inspirehep" in url:
            if inspire_answers:
                return INSPIRE_EMPTY
            raise OSError("inspire off in this test")
        return p._transport(url, params, {}, t)
    monkeypatch.setattr(list_ingest, "_api_get", api_get)
    return list_ingest._default_arxiv_original("2601.00001", list_ingest.load_config())


def r6(p, monkeypatch):
    from vera import search_and_fetch
    monkeypatch.setattr(search_and_fetch, "cached_chain", lambda *a, **k: ([p], 1, "m"))
    return rp._default_openalex_lookup(["W1"], 5, 3.0)


def no_inspire(monkeypatch):
    """list_ingest.ingest's INSPIRE leg is off in these tests (the OpenAlex leg is what is under test)."""
    real = list_ingest._api_get

    def api_get(url, params, t):
        if "inspirehep" in url:
            raise OSError("inspire off in this test")
        return real(url, params, t)
    monkeypatch.setattr(list_ingest, "_api_get", api_get)


def run(route, p, monkeypatch):
    """-> ("sent"|"refused", exception or None). Every route through its real entry point."""
    try:
        if route == "R1":
            r1(p)
        elif route == "R2":
            r2(p)
        elif route == "R3":
            r3(p, monkeypatch)
        elif route == "R4":
            r4(p)
        elif route == "R5":
            r5(p, monkeypatch)
        elif route == "R6":
            r6(p, monkeypatch)
    except Exception as e:  # noqa: BLE001
        return e
    return None


# ------------------------------------------------------------------------------------------------ A1
def test_a1_every_route_issues_exactly_one_request_through_openalex_request(clk, monkeypatch):
    gate(monkeypatch, "on")
    seen = []
    real = sp.openalex_request

    def spy(url, params, **kw):
        seen.append(kw["purpose"])
        return real(url, params, **kw)
    monkeypatch.setattr(sp, "openalex_request", spy)
    p, sent = prov(clk, lambda u, prm: R(200, {"results": [WORK], **WORK}))
    want = {"R1": "search", "R2": "acquisition", "R3": "resolve", "R4": "arxiv_lookup", "R5": "list_ingest",
            "R6": "refs"}
    for route, purpose in want.items():
        seen.clear()
        n0 = len(sent)
        run(route, p, monkeypatch)
        assert seen == [purpose], route
        assert len(sent) - n0 == 1, route


def test_a1_grep_no_openalex_request_outside_the_client():
    """Deterministic source check: the only network call for OpenAlex is OpenAlexProvider._send, and every OpenAlex
    URL constant or config URL is passed only to the client."""
    calls = []
    for f in sorted((ROOT / "vera").rglob("*.py")):
        src = f.read_text(encoding="utf-8")
        for m in re.finditer(r"openalex_default_transport\b(?!\s*\()", src):
            calls.append((f.name, "ref"))
        for m in re.finditer(r"openalex_default_transport\s*\(", src):
            calls.append((f.name, "call"))
        for m in re.finditer(r"([\w.]+)\(\s*(OPENALEX_WORK|OPENALEX_SEARCH|cfg\[\"openalex\"\]\[\"url\"\])", src):
            assert m.group(1) in ("_oa_get", "sp.openalex_request", "openalex_request"), (f.name, m.group(0))
        if f.name not in ("search_providers.py", "source_resolver.py", "api_permissions.py"):
            assert "api.openalex.org" not in src, f.name
    # definition + the single use in _send (the transport default), nowhere else
    assert sorted(calls) == [("search_providers.py", "call"), ("search_providers.py", "ref")], calls
    src = (ROOT / "vera" / "search_providers.py").read_text(encoding="utf-8")
    assert src.count("def openalex_default_transport(") == 1
    assert re.search(r"tx = transport or self\._transport or openalex_default_transport", src)


# ------------------------------------------------------------------------------------------------ A2
@pytest.mark.parametrize("status", [200, 404, 429, "exc"])
def test_a2_key_sent_on_r3_and_r5_and_never_leaked(clk, monkeypatch, caplog, status):
    gate(monkeypatch, "on")
    monkeypatch.setenv("OPENALEX_API_KEY", KEY)

    def responder(u, prm):
        if status == "exc":
            return requests.ConnectionError(f"boom for {u}?api_key={prm.get('api_key')}")
        return R(status, {"results": [], **WORK}, url=f"{u}?api_key={prm.get('api_key')}")
    caplog.set_level(logging.DEBUG)
    for route in ("R3", "R5"):
        sp._QUOTA.reset()  # a 429 on R3 would otherwise (correctly) cool R5 down: each route is checked on its own
        rate_limit.set_registry_for_tests(rate_limit.Registry(rate_limit.load_config(env={}), clock=clk.mono,
                                                              wall=clk.wall, sleep=lambda s: None, rand=lambda: 0.0))
        p, sent = prov(clk, responder)
        err = run(route, p, monkeypatch)
        assert sent and sent[-1]["api_key"] == KEY, route
        assert KEY not in str(err or ""), route
    assert KEY not in caplog.text
    sp._QUOTA.reset()
    rate_limit.set_registry_for_tests(rate_limit.Registry(rate_limit.load_config(env={}), clock=clk.mono,
                                                          wall=clk.wall, sleep=lambda s: None, rand=lambda: 0.0))
    p, sent = prov(clk, responder)
    if status != "exc":
        resp = sp.openalex_request("https://api.openalex.org/works", {"filter": "x"}, purpose="search", timeout_s=1.0,
                                   provider=p)
        assert KEY not in str(resp.url)
    else:
        with pytest.raises(requests.ConnectionError) as ei:
            sp.openalex_request("https://api.openalex.org/works", {"filter": "x"}, purpose="search", timeout_s=1.0,
                                provider=p)
        assert KEY not in str(ei.value)


def test_a2_keyless_sends_no_key(clk, monkeypatch):
    gate(monkeypatch, "on")
    for route in ("R3", "R5"):
        p, sent = prov(clk, lambda u, prm: R(200, WORK))
        run(route, p, monkeypatch)
        assert sent and "api_key" not in sent[-1]


# ------------------------------------------------------------------------------------------------ A3
@pytest.mark.parametrize("status,charged", [(200, 1), (204, 1), (400, 0), (404, 0), (429, 0), (500, 0), ("exc", 0)])
def test_a3_only_2xx_is_charged(clk, status, charged):
    p, sent = prov(clk, lambda u, prm: requests.Timeout("t") if status == "exc" else R(status))
    try:
        r1(p)
    except Exception:  # noqa: BLE001
        pass
    assert len(sent) == 1 and p._used == charged
    if not charged:
        assert any(k.startswith("oa_uncharged_") for k in p.oa_counts)


def test_a3_retry_charged_only_when_it_returns_2xx(clk):
    seq = iter([R(500), R(200)])
    p, sent = prov(clk, lambda u, prm: next(seq))
    r1(p)
    r1(p)
    assert len(sent) == 2 and p._used == 1


# ------------------------------------------------------------------------------------------------ A4
def hdr(limit, remaining, reset=None, **extra):
    h = {"X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": str(remaining)}
    if reset is not None:
        h["X-RateLimit-Reset"] = str(reset)
    return {**h, **extra}


@pytest.mark.parametrize("origin", [request_origin.SEED, request_origin.VISITOR])
def test_a4_header_seed_share(clk, origin):
    box = {"h": hdr(100, 51, 3600)}
    p, sent = prov(clk, lambda u, prm: R(200, headers=box["h"]))
    r1(p)  # learns Limit 100 / Remaining 51
    with request_origin.scope(origin):
        r1(p)  # 51 > 50: a seed is allowed
    box["h"] = hdr(100, 50, 3600)
    r1(p)  # learns Remaining 50
    n = len(sent)
    with request_origin.scope(origin):
        if origin == request_origin.SEED:
            with pytest.raises(sp.OpenAlexRefused) as ei:
                r1(p)
            assert ei.value.code == "seed_share" and sp._own_decision(ei.value) == "own_cap"
            assert len(sent) == n and not sp._QUOTA.blocked(clk.mono())
        else:
            r1(p)
            assert len(sent) == n + 1


@pytest.mark.parametrize("origin", [request_origin.SEED, request_origin.VISITOR])
def test_a4_remaining_zero_is_provider_quota_with_cooldown_until_reset(clk, origin):
    p, sent = prov(clk, lambda u, prm: R(200, headers=hdr(100, 0, 300)))
    r1(p)
    assert sp._QUOTA.blocked(clk.mono())
    for o in (origin, request_origin.VISITOR):
        with request_origin.scope(o), pytest.raises(sp.OpenAlexRefused) as ei:
            r1(p)
        assert ei.value.code == "provider_quota" and "own_cap" not in str(ei.value)
        assert "OpenAlex's daily allowance is used up until 2026-11-10T00:21:40Z" in str(ei.value)
        assert sp._own_decision(ei.value) is None and sp._is_rate_skip(ei.value)
    assert len(sent) == 1
    clk.t += 301
    p.s = p.s  # nothing else changes: after Reset the next request is sent and learns the new header
    r1(p)
    assert len(sent) == 2


def test_a4_fallback_cap_without_header_is_own_cap_and_no_cooldown(clk):
    p, sent = prov(clk)
    p._day, p._used = DAY, p._budget
    with pytest.raises(sp.OpenAlexRefused) as ei:
        r1(p)
    assert ei.value.code == "own_cap" and sp._own_decision(ei.value) == "own_cap"
    assert sent == [] and not sp._QUOTA.blocked(clk.mono())


@pytest.mark.parametrize("bad", [hdr("abc", 5, 10), hdr("", 5, 10), hdr(100, "abc", 10), hdr(100, 5, -5),
                                 hdr(100, 5, "soon"), {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "60"},
                                 {"X-RateLimit-Limit": "100"}])
def test_a4_unparseable_headers_ignored_fallback_count_applies(clk, bad):
    p, sent = prov(clk, lambda u, prm: R(200, headers=bad))
    r1(p)
    assert p._hdr is None and p._used == 1 and p.oa_counts.get("oa_header_ignored") == 1
    assert not sp._QUOTA.blocked(clk.mono())  # never a cooldown from a header we could not read
    p._used = p._budget
    with pytest.raises(sp.OpenAlexRefused) as ei:
        r1(p)
    assert ei.value.code == "own_cap"


def test_a4_header_from_yesterday_falls_back_to_the_count(clk):
    box = {"d": "2026-11-09"}
    p, sent = prov(clk, lambda u, prm: R(200, headers=hdr(100000, 99000, 86000)))
    p._today = lambda: box["d"]
    monkeypatch_key = pytest.MonkeyPatch()
    monkeypatch_key.setenv("OPENALEX_API_KEY", KEY)  # keyed: a fresh header would replace the count
    try:
        r1(p)  # header recorded on 2026-11-09
        assert p._hdr["day"] == "2026-11-09"
        box["d"] = DAY
        p._day, p._used = DAY, p._budget  # today's count is at the cap
        with request_origin.scope(request_origin.VISITOR), pytest.raises(sp.OpenAlexRefused) as ei:
            r1(p)
        assert ei.value.code == "own_cap"  # the stale header did not lift the cap
        box["d"] = "2026-11-09"  # same day as the header: it is fresh and does lift the count cap
        p._day, p._used = "2026-11-09", 10**6
        r1(p)
    finally:
        monkeypatch_key.undo()


def test_a4_keyed_fresh_header_replaces_the_count_keyless_keeps_it_as_floor(clk, monkeypatch):
    p, sent = prov(clk, lambda u, prm: R(200, headers=hdr(100000, 99000, 3600)))
    monkeypatch.setenv("OPENALEX_API_KEY", KEY)
    r1(p)
    p._used = p._budget  # count at the cap, but a fresh keyed header says plenty is left
    r1(p)
    assert len(sent) == 2
    monkeypatch.delenv("OPENALEX_API_KEY")
    with pytest.raises(sp.OpenAlexRefused) as ei:  # keyless: the count stays the floor
        r1(p)
    assert ei.value.code == "own_cap"


def test_a4_probe_not_vacuous_frozen_day_does_not_reset(clk):
    p, sent = prov(clk)
    r1(p)
    r1(p)
    assert p._day == DAY and p._used == 2  # same frozen day: no reset between calls
    clk.t += 86400  # monotonic time passes, the injected day does not
    r1(p)
    assert p._used == 3


def test_a4_credits_used_logged_per_origin_only(clk):
    p, sent = prov(clk, lambda u, prm: R(200, headers=hdr(100, 90, 3600, **{"X-RateLimit-Credits-Used": "1"})))
    r1(p)
    with request_origin.scope(request_origin.SEED):
        r1(p)
    assert p.oa_credits == {"visitor": 1.0, "seed": 1.0}


# ------------------------------------------------------------------------------------------------ A5
def test_a5_429_on_r3_blocks_r1_for_retry_after(clk, monkeypatch):
    gate(monkeypatch, "legacy")
    p, sent = prov(clk, lambda u, prm: R(429, headers={"Retry-After": "120"}))
    rec, s2 = r3(p, monkeypatch)
    assert rec is None and s2  # R3 fell through to Semantic Scholar
    n = len(sent)
    clk.t += 119
    with pytest.raises(sp.OpenAlexRefused) as ei:
        r1(p)
    assert ei.value.code == "cooldown" and len(sent) == n
    assert rate_limit.get_registry().portal("openalex").cooling_down()  # one cooldown, the rate portal's
    clk.t += 2
    p._transport = lambda u, prm, h, t: (sent.append(prm), R())[1]
    r1(p)
    assert len(sent) == n + 1


def test_a5_remaining_zero_blocks_every_route_for_reset(clk, monkeypatch):
    gate(monkeypatch, "on")
    p, sent = prov(clk, lambda u, prm: R(200, {"results": [WORK], **WORK}, headers=hdr(100, 0, 300)))
    r1(p)
    n = len(sent)
    for route in ("R1", "R2", "R3", "R4", "R5", "R6"):
        run(route, p, monkeypatch)
    assert len(sent) == n
    assert p.oa_counts.get("oa_provider_quota", 0) >= 5
    clk.t += 301
    run("R1", p, monkeypatch)
    assert len(sent) == n + 1


@pytest.mark.parametrize("why", ["own_cap", "seed_share", "terms_gate", "terms_gate_config_invalid", "retry_limit"])
def test_a5_own_refusals_block_nothing_and_a_later_visitor_call_is_sent(clk, monkeypatch, why):
    p, sent = prov(clk, lambda u, prm: R(500) if why == "retry_limit" else R())
    with request_origin.scope(request_origin.SEED):
        if why == "own_cap":
            p._day, p._used = DAY, p._budget
        elif why == "seed_share":
            p._day, p._used = DAY, request_origin.seed_cap(p._budget, 0.5)
        elif why == "terms_gate":
            gate(monkeypatch, "off")
        elif why == "terms_gate_config_invalid":
            gate(monkeypatch, "Legacy")
        elif why == "retry_limit":
            for _ in range(2):
                with pytest.raises(Exception):  # noqa: B017
                    r1(p)
                    raise RuntimeError("500 is returned, not raised")
        n = len(sent)
        with pytest.raises(sp.OpenAlexRefused) as ei:
            r1(p)
        assert ei.value.code == why and len(sent) == n
    assert not sp._QUOTA.blocked(clk.mono())
    assert not rate_limit.get_registry().portal("openalex").cooling_down()
    gate(monkeypatch, "legacy")
    p._day, p._used = DAY, 0
    p._attempts.clear()
    p._transport = lambda u, prm, h, t: (sent.append(prm), R())[1]
    r1(p)  # D1 regression: the visitor call goes out
    assert sent


# ------------------------------------------------------------------------------------------------ A5b
@pytest.mark.parametrize("why", ["budget", "seed_share", "cooldown"])
@pytest.mark.parametrize("origin", [request_origin.SEED, request_origin.VISITOR])
def test_a5b_r3_refusal_falls_through_and_lists_link_only(clk, monkeypatch, caplog, why, origin):
    gate(monkeypatch, "legacy")
    p, sent = prov(clk, lambda u, prm: R(200, {"title": "never", "results": []}))
    if why == "budget":
        p._day, p._used = DAY, p._budget
    elif why == "seed_share":
        if origin == request_origin.VISITOR:
            pytest.skip("the seed share never refuses a visitor (covered by A4)")
        p._day, p._used = DAY, request_origin.seed_cap(p._budget, 0.5)
    else:
        sp._QUOTA.block(clk.mono(), 600)
    s2 = R(200, {"title": "Synthetic S2 title", "publicationDate": "2024-01-01"})
    with request_origin.scope(origin):
        rec, s2_calls = r3(p, monkeypatch, s2=s2)
        assert sent == [] and s2_calls  # nothing sent to OpenAlex; Semantic Scholar asked instead
        assert rec["metadata_source"] == "semanticscholar" and rec["snippet"] == ""
        assert r3_title(p, monkeypatch) is None and sent == []
        from vera import grounded_ask
        caplog.set_level(logging.INFO)
        res = {"url": "https://example.org/page", "title": "Synthetic quark gluon plasma viscosity study"}
        found = grounded_ask._resolve_links(
            [res], {"deadline_s": 5, "max_results_to_resolve": 4},
            resolver=lambda r, c, d: source_resolver.resolve(r, {"timeout_s": 1, "max_pages_per_result": 0,
                                                                  "max_ids_per_page": 3, "max_depth": 1,
                                                                  "scholarly_hosts": [], "max_links_per_page": 3,
                                                                  "title_match_min": 0.5}, d))
    assert found == []  # no original traced: the result stays a link-only listing
    assert "failed" not in caplog.text.lower() and "openalex" not in caplog.text.lower()  # no error text
    assert sent == []


# ------------------------------------------------------------------------------------------------ A5c
@pytest.mark.parametrize("route,limit", [("R1", 2), ("R2", 2), ("R3", 1), ("R4", 1), ("R5", 1), ("R6", 1)])
@pytest.mark.parametrize("fail", ["500", "exc"])
def test_a5c_per_route_retry_limits(clk, monkeypatch, route, limit, fail):
    gate(monkeypatch, "on")
    p, sent = prov(clk, lambda u, prm: R(500) if fail == "500" else requests.ConnectionError("down"))
    p._day, p._used = DAY, 0  # plenty of budget left: the bound is the attempt counter, not the budget
    for _ in range(6):
        run(route, p, monkeypatch)
    assert len(sent) == limit and p._used == 0


def test_a5c_r1_chain_retries_stop_at_the_limit(clk, monkeypatch):
    from vera import search_and_fetch
    p, sent = prov(clk, lambda u, prm: R(500))
    with pytest.raises(Exception):  # noqa: B017
        search_and_fetch._attempts(p, "quark gluon plasma viscosity", 5, ledger=None, max_retries=3, backoff_s=0.0,
                                   deadline_at=clk.mono() + 60, timeout_s=1.0, sleep=lambda s: None, clock=clk.mono)
    assert len(sent) == 2  # the chain allows 3, the client stops at R1's 2


def test_a5c_a_2xx_clears_the_item_and_the_window_expires(clk):
    seq = iter([R(500), R(200), R(500), R(500), R(500)])
    p, sent = prov(clk, lambda u, prm: next(seq))
    for _ in range(4):
        try:
            r1(p)
        except sp.OpenAlexRefused:
            pass
    assert len(sent) == 4  # 500, 200 (clears), 500, 500; the 5th is refused
    with pytest.raises(sp.OpenAlexRefused) as ei:
        r1(p)
    assert ei.value.code == "retry_limit"
    clk.t += 901  # attempt_window_s
    r1(p)
    assert len(sent) == 5


# ------------------------------------------------------------------------------------------------ A6
EXPECT = {"legacy": {"R1", "R2", "R3"}, "off": set(), "on": {"R1", "R2", "R3", "R4", "R5", "R6"}}


@pytest.mark.parametrize("value", ["legacy", "off", "on"])
@pytest.mark.parametrize("route", ["R1", "R2", "R3", "R4", "R5", "R6"])
@pytest.mark.parametrize("origin", [request_origin.SEED, request_origin.VISITOR])
def test_a6_gate_value_by_route_by_origin(clk, monkeypatch, value, route, origin):
    gate(monkeypatch, value)
    p, sent = prov(clk, lambda u, prm: R(200, {"results": [WORK], **WORK}))
    with request_origin.scope(origin):
        err = run(route, p, monkeypatch)
    if route in EXPECT[value]:
        assert len(sent) == 1
    else:
        assert sent == []
        if route in ("R1", "R2", "R4", "R6"):
            assert sp.oa_refusal_code(err) == "terms_gate"
            assert sp.OA_TERMS_PENDING_TEXT in str(err) and sp.OA_CONFIG_INVALID_TEXT not in str(err)
            assert not re.search(r"budget|quota|spent|unavailable|limit", str(err).split("[oa-refusal")[0], re.I)


def test_a6_off_distinct_outcomes_per_route(clk, monkeypatch):
    gate(monkeypatch, "off")
    p, sent = prov(clk)
    rec, s2 = r3(p, monkeypatch)
    assert rec is None and s2 and sent == []  # R3: a miss (Semantic Scholar only)
    no_inspire(monkeypatch)
    out = list_ingest.ingest(["arXiv:2601.00001"], resolvers=list_ingest.Resolvers(known=lambda *a: False),
                             start_worker=False)
    assert [x["code"] for x in out["results"]] == ["no-route-arxiv-lookup-paused"]  # R5
    with pytest.raises(sp.OpenAlexRefused) as ei:
        r1(p)
    note = f"attempt 1: provider unavailable: {ei.value}"  # R1, as the search chain records it
    assert "not in use pending a terms review" in note and sp._own_decision(Exception(note)) == "terms_gate"
    assert research_outcome._cause_text("terms_gate") == sp.OA_TERMS_PENDING_TEXT


LOOKALIKES = ["Legacy", " legacy", "legacy ", "ON", "", None, 1, True, ["legacy"], "off ", "of"]


@pytest.mark.parametrize("bad", LOOKALIKES + ["<missing>"])
@pytest.mark.parametrize("origin", [request_origin.SEED, request_origin.VISITOR])
def test_a6_invalid_or_missing_gate_is_off_with_its_own_code(clk, monkeypatch, caplog, bad, origin):
    gate(monkeypatch, bad, missing=bad == "<missing>")
    caplog.set_level(logging.INFO)
    sp._OA_CFG_CACHE.pop("gate_logged", None)
    p, sent = prov(clk)
    for route in ("R1", "R2", "R3", "R4", "R6"):
        with request_origin.scope(origin):
            err = run(route, p, monkeypatch)
        if route != "R3":
            assert sp.oa_refusal_code(err) == "terms_gate_config_invalid", route
            assert sp.OA_CONFIG_INVALID_TEXT in str(err) and sp.OA_TERMS_PENDING_TEXT not in str(err)
            assert sp._own_decision(err) == "terms_gate_config_invalid"
    assert sent == []
    oa_logs = [r for r in caplog.records if r.name == "vera.openalex"]
    assert any(r.getMessage() == "R-OA-GATE-CONFIG-INVALID" and r.levelname == "ERROR" for r in oa_logs)
    assert all(r.getMessage() == "R-OA-GATE-CONFIG-INVALID" for r in oa_logs)  # rule id only, never the value
    no_inspire(monkeypatch)
    out = list_ingest.ingest(["arXiv:2601.00001"], resolvers=list_ingest.Resolvers(known=lambda *a: False),
                             start_worker=False)
    assert [x["code"] for x in out["results"]] == ["no-route-arxiv-lookup-config-invalid"]
    assert research_outcome._cause_text("terms_gate_config_invalid") == sp.OA_CONFIG_INVALID_TEXT


def test_a6_parity_shipped_gate_is_legacy_and_and_with_arxiv_switch(monkeypatch):
    raw = json.loads(rate_limit.CONFIG_PATH.read_text(encoding="utf-8"))["portals"]["openalex"]
    assert raw["terms_gate"] == "legacy" and sp.openalex_terms_gate() == "legacy"
    acq = json.loads((ROOT / "config" / "auto_acquire.json").read_text(encoding="utf-8"))
    assert acq["arxiv_portal"]["openalex"]["enabled"] is False  # T1 unchanged
    monkeypatch.setattr(rp, "load_config", lambda path=None: {"mode": "on", "adapters": {}, "caps": {}})
    monkeypatch.setattr(rp, "adapter_on", lambda c, n: False)
    from vera import arxiv_fulltext
    monkeypatch.setattr(arxiv_fulltext, "PORTAL_ONLY", True)
    on = json.loads(json.dumps(acq))
    on["arxiv_portal"]["enabled"] = True
    on["arxiv_portal"]["openalex"]["enabled"] = True
    assert arxiv_portal.switches(on)["oa_on"] is False  # legacy gate: the flag alone never opens R4
    gate(monkeypatch, "on")
    assert arxiv_portal.switches(on)["oa_on"] is True
    on["arxiv_portal"]["openalex"]["enabled"] = False
    assert arxiv_portal.switches(on)["oa_on"] is False  # the flag stays an AND with the gate


def test_a6_meter_config_invalid_is_off(clk, monkeypatch):
    real = json.loads(rate_limit.CONFIG_PATH.read_text(encoding="utf-8"))["portals"]["openalex"]
    for bad in ({**real, "meter": {**real["meter"], "charge_statuses": "all"}},
                {**real, "meter": {**real["meter"], "max_attempts": {"search": 2}}},
                {k: v for k, v in real.items() if k != "meter"}):
        monkeypatch.setattr(sp, "_oa_raw_portal", lambda b=bad: b)
        p, sent = prov(clk)
        with pytest.raises(sp.OpenAlexRefused) as ei:
            r1(p)
        assert ei.value.code == "terms_gate_config_invalid" and sent == []


# ------------------------------------------------------------------------------------------------ A7
def loc(doi, version, host="https://doi.org/"):
    return {"landing_page_url": f"{host}{doi}", "version": version}


@pytest.mark.parametrize("locations,want", [
    ([loc("10.1103/x", "acceptedVersion"), loc("10.1103/y", "publishedVersion")], "10.1103/y"),
    ([loc("10.1103/x", None)], None),
    ([{"landing_page_url": "https://doi.org/10.1103/x"}], None),
    ([loc("10.1101/2020.01.01.123456", "submittedVersion")], None),
    ([loc("10.1101/2020.01.01.123456", "publishedVersion")], None),
    ([loc("10.2139/ssrn.1", "publishedVersion")], None),
    ([loc("10.1103/x", "publishedVersion", host="https://repository.example.edu/handle/")], None),
    ([loc("10.1103/x", "publishedVersion", host="https://journals.aps.org/prl/abstract/")], None),
    ([loc("10.48550/arxiv.2601.00001", "publishedVersion")], None),
    ([loc("10.1103/x", "PublishedVersion")], None),
])
def test_a7_list_ingest_takes_only_published_version(clk, monkeypatch, locations, want):
    gate(monkeypatch, "on")
    p, sent = prov(clk, lambda u, prm: R(200, {"id": "https://openalex.org/W1", "locations": locations}))
    assert r5(p, monkeypatch, inspire_answers=True) == want
    assert len(sent) == 1 and sent[0]["select"] == "doi,locations"
    assert (arxiv_portal.published_version_dois(locations)[:1] or [None])[0] == want
    if want is None:  # B7FIX-1112: same input with INSPIRE down is not definitive (S6-1), never a silent None
        p2, sent2 = prov(clk, lambda u, prm: R(200, {"id": "https://openalex.org/W1", "locations": locations}))
        with pytest.raises(RuntimeError) as ei:
            r5(p2, monkeypatch)
        assert sp.oa_refusal_code(ei.value) is None and len(sent2) == 1


def _a7_ingest(p, monkeypatch, inspire_answers):
    monkeypatch.setattr(sp, "shared_openalex_provider", lambda: p)

    def api_get(url, params, t):
        if "inspirehep" in url:
            if inspire_answers:
                return INSPIRE_EMPTY
            raise OSError("inspire off")
        return p._transport(url, params, {}, t)
    monkeypatch.setattr(list_ingest, "_api_get", api_get)
    out = list_ingest.ingest(["arXiv:2601.00001"], resolvers=list_ingest.Resolvers(known=lambda *a: False),
                             start_worker=False)
    return [x["code"] for x in out["results"]]


def test_a7_list_ingest_no_original_code_kept(clk, monkeypatch):
    gate(monkeypatch, "on")
    p, sent = prov(clk, lambda u, prm: R(200, {"locations": [loc("10.1103/x", None)]}))
    assert _a7_ingest(p, monkeypatch, inspire_answers=True) == ["no-route-arxiv-original-not-found"]
    # B7FIX-1112: INSPIRE down + no published original is retryable (S6-1), not the terminal not-found code
    p2, sent2 = prov(clk, lambda u, prm: R(200, {"locations": [loc("10.1103/x", None)]}))
    assert _a7_ingest(p2, monkeypatch, inspire_answers=False) == ["lookup-unavailable"]
    assert len(sent) == 1 and len(sent2) == 1


def test_a7_result_codes_distinct():
    rc = json.loads((ROOT / "config" / "list_ingest.json").read_text(encoding="utf-8"))["result_codes"]
    codes = [rc["arxiv_lookup_gated"], rc["arxiv_lookup_config_invalid"], rc["arxiv_no_original"],
             rc["arxiv_lookup_limit"]]
    assert len(set(codes)) == 4 and rc["arxiv_lookup_gated"] == "no-route-arxiv-lookup-paused"


# ------------------------------------------------------------------------------------------------ A9
class _Fake:
    name = "openalex"

    def __init__(self, n_refs):
        self.n_refs, self.calls = n_refs, []

    def metadata_get(self, params, timeout_s):
        self.calls.append(params)
        if params["filter"].startswith("doi:"):
            return R(200, {"results": [WORK]})
        return R(200, {"results": [{"doi": f"https://doi.org/10.9999/synth.ref.{i}"} for i in range(self.n_refs)]})


@pytest.mark.parametrize("n_refs,max_refs,empty,calls", [(0, 3, 1, 2), (3, 3, 0, 2), (0, 0, 0, 1)])
def test_a9_oa_ref_empty_counter(monkeypatch, n_refs, max_refs, empty, calls, caplog):
    monkeypatch.setattr(sp._QUOTA, "blocked", lambda now: False)
    arxiv_portal.reset_for_tests()
    f = _Fake(n_refs)
    counts = {}
    caplog.set_level(logging.INFO)
    arxiv_portal.openalex_work(f"doi:https://doi.org/10.48550/arxiv.2601.0000{max_refs}", max_refs, 5.0, counts,
                               clock=lambda: 0.0, provider=f)
    assert counts.get("oa_ref_empty", 0) == empty and len(f.calls) == calls
    assert ("R-OA-REF-EMPTY" in caplog.text) is bool(empty)


# ------------------------------------------------------------------------------------------------ outcome texts
def test_outcome_sentences_distinct_and_statuses_mapped():
    from vera import portal_recorder, research_record
    texts = {c: research_outcome._cause_text(c) for c in ("terms_gate", "terms_gate_config_invalid", "provider_quota",
                                                           "retry_limit", "own_cap", "quota")}
    assert len(set(texts.values())) == len(texts)
    for c in ("terms_gate", "terms_gate_config_invalid", "provider_quota", "retry_limit"):
        assert c in research_outcome.PORTAL_STATUSES
        assert c in portal_recorder._STATUS and c in research_record._PORTAL_STATUS and c in research_record._PUBLIC_STATUS
    assert not re.search(r"budget|quota|spent", texts["terms_gate"], re.I)
