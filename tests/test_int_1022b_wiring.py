"""INT-1022-B: wiring of the RATE-1022 rate limiter and the WIKIREF-1022 reference portal into the call sites.
Offline: fake transports and fake portal searches only (no network, no model, no database)."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from tests import acq_support as A
from tests.test_yield_1021 import Q, _oa_cand, _R
from vera import acq_config, inline_acquire, rate_limit, search_and_fetch, wiki_ref_portal
from vera import auto_acquire as aa
from vera import search_providers as sp


def _wiki_fake(monkeypatch, cands, seen):
    def factory(cfg, *, http_get=None, url_check=None, clock=None):
        seen["http_get"] = http_get

        def search(question, n, timeout_s):
            seen["calls"] = seen.get("calls", 0) + 1
            seen["deadline"] = rate_limit.DEADLINE.get()
            return list(cands)
        return search
    monkeypatch.setattr(wiki_ref_portal, "make_portal_search", factory)


def _acq(cap, **kw):
    calls = {"base": 0, "ddg": 0, "epmc": 0}

    def b(q, n, t):
        calls["base"] += 1
        return [_oa_cand(1)]

    def d(q, n, t):
        calls["ddg"] += 1
        return []

    def tx(url, params, headers, timeout):
        calls["epmc"] += 1
        return _R({"resultList": {"result": [{"doi": "10.9999/yield.61", "pmcid": "PMC0000061", "isOpenAccess": "Y",
                                              "source": "MED", "title": "E"}]}})
    fn = sp.acquisition_search(b, A.cfg(max_search_calls_per_job=cap), transport=tx, terms_fn=lambda: (None, None),
                               ddg_search=d, **kw)
    return fn, calls


WIKI_CAND = {"url": "https://doi.org/10.9999/wiki.7", "doi": "10.9999/wiki.7", "title": "", "snippet": "",
             "provider": "wikipedia_portal", "acq_source": "wikipedia_portal", "portal": "wikipedia_refs",
             "ref_id_kind": "doi", "portal_rank": 1}


@pytest.mark.wiki_portal
def test_wiki_portal_runs_in_its_own_slot_and_merges_last(monkeypatch):
    seen = {}
    _wiki_fake(monkeypatch, [WIKI_CAND], seen)
    fn, calls = _acq(6)
    out = fn(Q, 10, 5)
    assert seen["calls"] == 1 and calls == {"base": 1, "ddg": 1, "epmc": 1}
    assert [c["doi"] for c in out][-1] == "10.9999/wiki.7"
    assert getattr(seen["http_get"], "rate_portal", None) == "wikipedia"  # default GET goes through the limiter
    assert sp.acq_source(WIKI_CAND) == "wikipedia_portal" and sp.acq_rank_tier(WIKI_CAND) == 1
    assert sp.acq_rank_tier({**WIKI_CAND, "ref_id_kind": "arxiv"}) == 2


@pytest.mark.wiki_portal
def test_wiki_portal_gets_no_slot_when_the_cap_is_spent(monkeypatch):
    seen = {}
    _wiki_fake(monkeypatch, [WIKI_CAND], seen)
    fn, calls = _acq(4)  # OpenAlex 2 + DDG 1 + Europe PMC 1: nothing left
    fn(Q, 10, 5)
    assert "calls" not in seen


@pytest.mark.wiki_portal
def test_injected_wiki_get_is_used_and_portal_runs_in_a_copied_context(monkeypatch):
    seen, fake_get = {}, object()
    _wiki_fake(monkeypatch, [], seen)
    fn, _ = _acq(6, wiki_get=fake_get)
    with rate_limit.deadline_scope(12345.0):
        fn(Q, 10, 5)
    assert seen["http_get"] is fake_get and seen["deadline"] == 12345.0  # C2: the pool thread sees the deadline


def test_offline_suite_keeps_the_portal_off_by_default(monkeypatch):
    seen = {}
    _wiki_fake(monkeypatch, [WIKI_CAND], seen)
    fn, _ = _acq(6)
    fn(Q, 10, 5)
    assert "calls" not in seen and wiki_ref_portal.load_config()["enabled"] is True  # shipped config stays on


def test_config_reason_code_for_a_rate_limit_skip():
    assert acq_config.reason(A.cfg(), "rate_limited") == "R-ACQ-RATELIMITED"


def test_guarded_get_maps_a_rate_limit_skip_to_its_own_code_not_unreachable():
    def get(*a, **k):
        raise rate_limit.RateLimited("europepmc", "cooldown", 3.0)
    deps = SimpleNamespace(http_get=get, url_check=lambda u: (True, ""))
    with pytest.raises(aa._Stop) as e:
        aa._guarded_get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?q=x", A.cfg(), deps, 1000, 5.0, True)
    assert e.value.code == "R-ACQ-RATELIMITED"


def test_transport_counts_the_request_in_the_portal(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _R({"results": []}))
    sp.openalex_default_transport("https://api.openalex.org/works", {}, {}, 5)
    assert rate_limit.get_registry().snapshot()["openalex"]["requests"] == 1


def test_rate_limit_skip_never_starts_the_openalex_fallback_cooldown():
    sp._QUOTA.reset()

    def b(q, n, t):
        raise rate_limit.RateLimited("openalex", "cooldown", 30.0)
    fn = sp.acquisition_search(b, A.cfg(max_search_calls_per_job=6), transport=lambda *a: _R({"resultList": {}}),
                               terms_fn=lambda: (None, None), ddg_search=lambda q, n, t: [])
    try:
        fn(Q, 10, 5)
    except Exception:  # noqa: BLE001  all sources empty/failed may re-raise; the gate state is what matters
        pass
    assert not sp._QUOTA.blocked(0.0) and not sp._is_quota_error(Exception("x")) and sp._is_rate_skip(
        rate_limit.RateLimited("openalex", "cooldown"))
    sp._QUOTA.reset()


def test_search_chain_skip_is_not_an_attempt():
    class P:
        name = "openalex"
        gate = SimpleNamespace(lock=threading.Lock(), wait_needed=lambda now: 0.0, stamp=lambda now: None)

        def failure_basis(self):
            return "keyless"

        def call(self, question, n, timeout_s):
            raise rate_limit.RateLimited("openalex", "cooldown", 9.0)
    with pytest.raises(search_and_fetch._ProviderFailed) as e:
        search_and_fetch._attempts_once(P(), Q, 5, ledger=None, max_retries=3, backoff_s=0.0, deadline_at=1e12,
                                        timeout_s=5.0, sleep=lambda s: None, clock=lambda: 0.0)
    assert len(e.value.notes) == 1 and "not an attempt" in e.value.notes[0]  # no retries were spent on a skip


def test_openai_clients_get_a_limited_transport_only_for_configured_enabled_portals():
    hc = rate_limit.openai_http_client("https://openrouter.ai/api/v1", 5.0)
    assert hc is not None and type(hc._transport).__name__ == "_LimitedTransport"
    hc.close()


def test_shipped_groq_portal_is_disabled_and_a_pure_pass_through():
    assert rate_limit.get_registry().portal("groq").c["enabled"] is False
    assert rate_limit.openai_http_client("https://api.groq.com/openai/v1", 5.0) is None  # SDK default client
    from vera.inference_chain import ChainEntry
    import os
    os.environ.setdefault("INT1022B_FAKE_KEY", "x")
    try:
        c = ChainEntry(provider="groq", model="m", pricing=None, base_url="https://api.groq.com/openai/v1",
                       api_key_env="INT1022B_FAKE_KEY").client()
        assert type(c._client._transport).__name__ != "_LimitedTransport"
    finally:
        os.environ.pop("INT1022B_FAKE_KEY", None)
    assert rate_limit.openai_http_client("https://unlisted.example.org/v1", 5.0) is None
    assert rate_limit.openai_http_client(None, 5.0) is None


def test_inline_run_binds_a_monotonic_rate_limit_deadline(monkeypatch):
    got = {}

    def body(*a, **k):
        got["d"] = rate_limit.DEADLINE.get()
        return "o"
    monkeypatch.setattr(inline_acquire, "_run_body", body)
    import time
    t = time.monotonic()
    assert inline_acquire._run("q", {"bounds": {"deadline_s": 45}}, {}, lambda: 0.0, None) == "o"
    assert got["d"] is not None and t < got["d"] <= time.monotonic() + 46
    assert rate_limit.DEADLINE.get() is None
