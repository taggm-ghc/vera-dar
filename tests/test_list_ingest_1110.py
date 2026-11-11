"""LIST-INGEST-1110: the search-free list-ingest route. Offline fakes only: no network, DB or env files."""
import copy
import json

import pytest
from fastapi.testclient import TestClient

import main
from tests import acq_support as A
from tests import fulltext_synth
from vera import auto_acquire, list_ingest, public_mode as pm, request_origin, search_providers

KEY = "k" * 20
IKEY = "i" * 24
H = {"X-API-Key": IKEY}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("VERA_API_KEY", KEY)
    monkeypatch.setenv("VERA_INGEST_API_KEY", IKEY)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.setenv("VERA_DB_URL_WO", "postgresql://wo.invalid/x")
    monkeypatch.setenv("VERA_DB_URL_RO", "postgresql://ro.invalid/x")
    monkeypatch.delenv("VERA_AUTOACQ_DISABLED", raising=False)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    pm.reset_limiter(pm.CapLimiter(pm.Caps(req_per_min_per_client=1000, req_per_day_per_client=1000,
                                           req_per_day_global=1000)))
    auto_acquire.reset_for_tests()
    list_ingest.reset_for_tests()
    yield
    pm.reset_limiter(None)
    auto_acquire.reset_for_tests()
    list_ingest.reset_for_tests()


def _no_worker(monkeypatch):
    real = auto_acquire.enqueue_candidates
    monkeypatch.setattr(auto_acquire, "enqueue_candidates", lambda *a, **k: real(*a, **{**k, "start_worker": False}))


def _forbid_search(monkeypatch):
    """Any search provider call fails the test."""
    def boom(*a, **k):
        raise AssertionError("a search provider was called")
    for name in ("acquisition_search", "europepmc_search", "openalex_search"):
        if hasattr(search_providers, name):
            monkeypatch.setattr(search_providers, name, boom)


class Rv(list_ingest.Resolvers):
    def __init__(self, known=(), arxiv=None, pmid=None):
        super().__init__(pmid=(lambda p: (pmid or {}).get(p)), arxiv_original=(lambda a: (arxiv or {}).get(a)),
                         known=lambda doi, url: doi in set(known))


def _ingest(items, monkeypatch, **kw):
    _no_worker(monkeypatch)
    _forbid_search(monkeypatch)
    return list_ingest.ingest(items, start_worker=False, **kw)


def codes(r):
    return [x["code"] for x in r["results"]]


def test_unauthenticated_is_401_and_wrong_key_too():
    c = TestClient(main.app, raise_server_exceptions=False)
    assert c.post("/ingest/list", json={"identifiers": ["10.9999/a.b"]}).status_code == 401
    assert c.post("/ingest/list", json={"identifiers": ["10.9999/a.b"]}, headers={"X-API-Key": "x" * 20}).status_code == 401
    assert c.get("/ingest/list/outcomes").status_code == 401


def test_dedicated_key_required_and_vera_api_key_refused(monkeypatch):
    c = TestClient(main.app, raise_server_exceptions=False)
    body = {"identifiers": ["10.9999/a.b"]}
    assert c.post("/ingest/list", json=body, headers={"X-API-Key": KEY}).status_code == 401  # the /ask key is not enough
    assert c.get("/ingest/list/outcomes", headers={"X-API-Key": KEY}).status_code == 401
    monkeypatch.delenv("VERA_INGEST_API_KEY")  # unset = disabled with a clear code
    r = c.post("/ingest/list", json=body, headers={"X-API-Key": KEY})
    assert r.status_code == 503 and "VERA_INGEST_API_KEY" in r.json()["detail"]
    monkeypatch.setenv("VERA_INGEST_API_KEY", KEY)  # equal to VERA_API_KEY = refused too
    assert c.post("/ingest/list", json=body, headers={"X-API-Key": KEY}).status_code == 503


def test_origin_cannot_be_sent_by_the_caller():
    c = TestClient(main.app, raise_server_exceptions=False)
    r = c.post("/ingest/list", json={"identifiers": ["10.9999/a.b"], "origin": "visitor"}, headers=H)
    assert r.status_code == 422


def test_too_many_items_is_413_and_nothing_queued(monkeypatch):
    n = list_ingest.load_config()["max_items_per_call"]
    c = TestClient(main.app, raise_server_exceptions=False)
    r = c.post("/ingest/list", json={"identifiers": [f"10.9999/x.{i}" for i in range(n + 1)]}, headers=H)
    assert r.status_code == 413
    assert auto_acquire._S.queue is None or auto_acquire._S.queue.empty()


def test_endpoint_queues_and_returns_codes(monkeypatch):
    _no_worker(monkeypatch)
    _forbid_search(monkeypatch)
    monkeypatch.setattr(list_ingest, "_default_known", lambda *a, **k: (lambda doi, url: doi == "10.9999/stored.1"))
    c = TestClient(main.app, raise_server_exceptions=False)
    r = c.post("/ingest/list", json={"identifiers": ["10.9999/new.1", "https://doi.org/10.9999/stored.1", "bogus"]},
               headers=H)
    assert r.status_code == 200
    assert [x["code"] for x in r.json()["results"]] == ["queued", "duplicate-already-stored", "invalid-identifier"]


def test_parse_identifier_kinds():
    p = list_ingest.parse_identifier
    assert p("10.1234/ABC.1") == ("doi", "10.1234/abc.1")
    assert p("https://doi.org/10.1234/ABC.1") == ("doi", "10.1234/abc.1")
    assert p("PMC1234567") == ("pmcid", "PMC1234567")
    assert p("pmid:123456") == ("pmid", "123456")
    assert p("arXiv:2301.12345v2") == ("arxiv", "2301.12345")
    assert p("https://arxiv.org/abs/2301.12345v1") == ("arxiv", "2301.12345")
    assert p("10.48550/arXiv.2301.12345") == ("arxiv", "2301.12345")
    assert p("https://example.org/page") == ("url", "https://example.org/page")
    assert p("http://example.org/page") is None and p("") is None and p("a b") is None


def test_arxiv_ids_go_to_the_journal_original_never_arxiv_text(monkeypatch):
    from vera import arxiv_fulltext
    monkeypatch.setattr(arxiv_fulltext, "evaluate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("arXiv text route")))
    rv = Rv(arxiv={"2301.12345": "10.9999/journal.77", "2302.00001": "10.48550/arxiv.2302.00001"})
    r = _ingest(["arXiv:2301.12345", "https://arxiv.org/abs/2302.00001", "2303.99999"], monkeypatch, resolvers=rv)
    assert codes(r) == ["queued", "no-route-arxiv-original-not-found", "no-route-arxiv-original-not-found"]
    queued = list(auto_acquire._S.queue.queue)
    assert len(queued) == 1
    c = queued[0].cands[0]
    assert c["doi"] == "10.9999/journal.77" and "arxiv" not in c and not c["doi"].startswith("10.48550")
    assert queued[0].origin == "seed" and queued[0].trigger == "list"


def test_pmid_and_url_routes(monkeypatch):
    rv = Rv(pmid={"111": {"pmcid": "PMC7654321", "doi": "10.9999/p.1"}, "222": None})
    r = _ingest(["pmid:111", "pmid:222", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC555555/",
                 "https://example.org/some-landing-page"], monkeypatch, resolvers=rv)
    assert codes(r) == ["queued", "no-route", "queued", "no-route"]


def test_duplicates_in_call_stored_and_queued_are_skipped(monkeypatch):
    rv = Rv(known={"10.9999/have.1"})
    r = _ingest(["10.9999/a.1", "doi:10.9999/A.1", "10.9999/have.1"], monkeypatch, resolvers=rv)
    assert codes(r) == ["queued", "duplicate-in-call", "duplicate-already-stored"]
    r2 = _ingest(["10.9999/a.1"], monkeypatch, resolvers=rv)  # a second call: already queued
    assert codes(r2) == ["duplicate-already-queued"]
    assert auto_acquire._S.queue.qsize() == 1


def test_item_caps_are_respected(monkeypatch):
    cfg = list_ingest.load_config()
    cfg["max_items_per_hour"] = 2
    r = _ingest([f"10.9999/c.{i}" for i in range(4)], monkeypatch, cfg=cfg, resolvers=Rv())
    assert codes(r) == ["queued", "queued", "rate-limited", "rate-limited"]


def test_shared_queue_caps_apply_and_seed_leaves_hourly_slots(monkeypatch):
    acq = A.cfg(jobs_per_hour=3, jobs_per_day=100)  # BG_MIN_FREE_HOURLY_JOBS=2: a seed job needs >= 2 free slots
    r = _ingest(["10.9999/q.1", "10.9999/q.2", "10.9999/q.3"], monkeypatch, acq=acq, resolvers=Rv())
    assert codes(r) == ["queued", "queued", "rate-limited"]


def test_disabled_config_and_kill_switch(monkeypatch):
    cfg = list_ingest.load_config()
    cfg["enabled"] = False
    assert codes(_ingest(["10.9999/d.1"], monkeypatch, cfg=cfg, resolvers=Rv())) == ["disabled"]
    monkeypatch.setenv("VERA_AUTOACQ_DISABLED", "true")
    assert codes(_ingest(["10.9999/d.2"], monkeypatch, resolvers=Rv())) == ["disabled"]


def test_job_runs_through_the_gates_without_search_and_stores(monkeypatch):
    doi = "10.9999/acq.epi4"
    web, _raw = A.epi_setup(doi)
    A.seed_access("www.ebi.ac.uk")
    store = A.Store()
    deps = A.deps([], web, store=store)
    deps.search = lambda *a, **k: (_ for _ in ()).throw(AssertionError("deps.search called"))
    r = _ingest([doi], monkeypatch, resolvers=Rv())
    assert codes(r) == ["queued"]
    res = auto_acquire.drain_for_tests(A.cfg(), deps, env=A.ENV_ON)
    assert len(res) == 1 and res[0]["stored"] == 1, res
    assert len(store.plans) == 1
    assert list_ingest.outcomes()[-1]["code"] == "stored"


def test_job_licence_reject_is_recorded_and_not_stored(monkeypatch):
    doi = "10.9999/acq.nd1"
    web, _raw = A.epi_setup(doi, rec_kw={"licence": "cc by-nd"})
    A.seed_access("www.ebi.ac.uk")
    store = A.Store()
    _ingest([doi], monkeypatch, resolvers=Rv())
    res = auto_acquire.drain_for_tests(A.cfg(), A.deps([], web, store=store), env=A.ENV_ON)
    assert res[0]["stored"] == 0 and not store.plans
    o = list_ingest.outcomes()[-1]
    assert o["state"] == "done" and o["code"] != "stored"


def test_visitor_jobs_run_before_list_jobs(monkeypatch):
    _ingest(["10.9999/lo.1"], monkeypatch, resolvers=Rv())
    q = auto_acquire._S.queue
    visitor = auto_acquire.Job(question="v", trigger="decline", created=0.0, origin="visitor")
    q.put_nowait(visitor)
    first = q.get_nowait()
    assert first.trigger == "list" and auto_acquire._defer_seed(q, first) is True  # seed job goes behind the visitor job


def test_public_ui_never_references_the_route():
    import pathlib
    root = pathlib.Path(main.__file__).parent
    for f in [root / "streamlit_app.py", *(root / "pages").glob("*.py"), root / "vera" / "public_mode.py"]:
        assert "/ingest/list" not in f.read_text(encoding="utf-8"), f


def test_visitor_job_still_queues_after_a_full_list_call(monkeypatch):
    acq = A.cfg(queue_max=4, jobs_per_hour=100, jobs_per_day=1000)
    r = _ingest([f"10.9999/f.{i}" for i in range(8)], monkeypatch, acq=acq, resolvers=Rv())
    assert codes(r).count("queued") == 2 and codes(r).count("queue-full") == 6  # queue_max 4 - reserve 2
    v = auto_acquire.enqueue_candidates("visitor question about outbreak", [A.cand("10.9999/v.1")], cfg=acq, start_worker=False)
    assert v["queued"] == [0]


def test_lookups_run_under_seed_origin_and_arxiv_lookups_are_capped(monkeypatch):
    seen = []

    def arx(a):
        seen.append(request_origin.current())
        return "10.9999/j." + a.replace(".", "")
    pm_seen = []

    def pmid(p):
        pm_seen.append(request_origin.current())
        return None
    cfg = list_ingest.load_config()
    cfg["max_arxiv_lookups_per_call"] = 2
    rv = list_ingest.Resolvers(arxiv_original=arx, pmid=pmid, known=lambda d, u: False)
    r = _ingest([f"arxiv:2301.0000{i}" for i in range(4)] + ["pmid:5"], monkeypatch, cfg=cfg, resolvers=rv)
    assert seen == ["seed", "seed"] and pm_seen == ["seed"]
    assert codes(r)[:4] == ["queued", "queued", "arxiv-lookup-limit", "arxiv-lookup-limit"]
    assert request_origin.current() == "visitor"


def test_caps_hold_under_concurrent_calls(monkeypatch):
    import threading
    import time as _t
    _no_worker(monkeypatch)
    cfg = list_ingest.load_config()
    cfg["max_items_per_hour"] = cfg["max_items_per_day"] = 10
    acq = A.cfg(queue_max=200, jobs_per_hour=1000, jobs_per_day=10000)
    real = auto_acquire.enqueue_candidates

    def slow(*a, **k):
        _t.sleep(0.01)
        return real(*a, **k)
    monkeypatch.setattr(auto_acquire, "enqueue_candidates", slow)
    res = []

    def call(n):
        res.append(list_ingest.ingest([f"10.9999/p{n}.{i}" for i in range(5)], cfg=cfg, acq=acq, resolvers=Rv(),
                                      start_worker=False))
    ts = [threading.Thread(target=call, args=(n,)) for n in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sum(codes(r).count("queued") for r in res) == 10


def test_topic_is_the_relevance_target_and_outcome_shows_decisive_code(monkeypatch):
    doi = "10.9999/acq.epi4"
    web, _raw = A.epi_setup(doi)
    A.seed_access("www.ebi.ac.uk")
    store = A.Store()
    _ingest([doi], monkeypatch, resolvers=Rv(), topic=A.Q_EPI)
    assert list_ingest.outcomes()[-1]["target"] == "topic"
    res = auto_acquire.drain_for_tests(A.cfg(), A.deps([], web, store=store), env=A.ENV_ON)
    assert res[0]["stored"] == 1 and list_ingest.outcomes()[-1]["code"] == "stored"
    # an off-topic topic makes the same document fail the content check
    web2, _ = A.epi_setup("10.9999/acq.epi5")
    store2 = A.Store()
    _ingest(["10.9999/acq.epi5"], monkeypatch, resolvers=Rv(), topic="medieval cathedral architecture and stained glass")
    res2 = auto_acquire.drain_for_tests(A.cfg(), A.deps([], web2, store=store2), env=A.ENV_ON)
    assert res2[0]["stored"] == 0 and not store2.plans
    o = list_ingest.outcomes()[-1]
    assert o["state"] == "done" and o["code"] == res2[0]["last_code"]


def test_bad_topic_is_refused_before_any_work(monkeypatch):
    r = _ingest(["10.9999/t.1"], monkeypatch, resolvers=Rv(), topic="x" * 400)
    assert codes(r) == ["invalid-topic"] and auto_acquire._S.queue is None


# PORTAL-RULE-1110 (Q4): a portal/excluded record is refused before it takes a queue slot; a URL host is never the reason.
def test_preprint_doi_is_refused_as_portal_and_takes_no_slot(monkeypatch):
    r = _ingest(["10.1101/2024.01.02.123456", "10.9999/journal.5"], monkeypatch, resolvers=Rv())
    assert codes(r) == ["refused-portal", "queued"]
    assert len(list(auto_acquire._S.queue.queue)) == 1
    assert len(list_ingest._ACCEPTED) == 1


def test_pmid_whose_record_is_a_preprint_is_refused_as_portal(monkeypatch):
    rv = Rv(pmid={"333": {"pmcid": "PMC1234567", "doi": "10.9999/p.2", "source": "PPR"},
                  "334": {"pmcid": "PMC1234568", "pub_types": ["Preprint"]}})
    r = _ingest(["pmid:333", "pmid:334"], monkeypatch, resolvers=rv)
    assert codes(r) == ["refused-portal", "refused-portal"]
    assert not list_ingest._ACCEPTED


def test_journal_doi_rejects_every_preprint_prefix_not_only_arxiv():
    from vera import arxiv_portal
    prefixes = arxiv_portal._preprint_prefixes()
    assert "10.1101/" in prefixes and "10.48550/" in prefixes
    for p in prefixes:
        assert list_ingest._journal_doi(p + "abc.123") is None
    assert list_ingest._journal_doi("https://doi.org/10.9999/Journal.77") == "10.9999/journal.77"


def test_arxiv_id_resolving_to_a_preprint_doi_reports_no_original(monkeypatch):
    rv = Rv(arxiv={"2301.12345": "10.1101/2024.01.02.123456"})
    r = _ingest(["arXiv:2301.12345"], monkeypatch, resolvers=rv)
    assert codes(r) == ["no-route-arxiv-original-not-found"]
    assert not list_ingest._ACCEPTED


def test_url_item_is_not_refused_for_its_portal_host(monkeypatch):
    r = _ingest(["https://pubmed.ncbi.nlm.nih.gov/12345/", "https://europepmc.org/article/MED/12345",
                 "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC555556/"], monkeypatch, resolvers=Rv())
    assert "refused-portal" not in codes(r)
    assert codes(r)[2] == "queued"


# ------------------------------------------------------------------------------------------------ F-LI1: one rate slot
def _openalex_transport(monkeypatch, rate_cfg=None):
    """DEFAULT lookup path: real api_permissions + rate_limit + pinned_get; only the network is faked (DNS + Session.get)."""
    import socket

    import requests

    from vera import rate_limit
    rc = rate_limit.load_config(rate_limit.CONFIG_PATH)
    if rate_cfg:
        rc["portals"]["openalex"].update(rate_cfg)
    rate_limit.set_registry_for_tests(rate_limit.Registry(rc))
    sent = []

    class Resp:
        status_code = 200
        headers = {}

        def json(self):
            # OPENALEX-FIX-1110 WP-B: real OpenAlex locations carry `version`; only publishedVersion counts now (M8)
            return {"locations": [{"landing_page_url": "https://doi.org/10.1234/journal.x", "version": "publishedVersion"}]}

        def close(self):
            pass

    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("151.101.3.42", 443))])
    monkeypatch.setattr(requests.Session, "get", lambda self, url, **kw: sent.append(url) or Resp())
    return sent


def test_openalex_lookup_takes_one_rate_slot_and_returns_doi(monkeypatch):
    from vera import rate_limit
    sent = _openalex_transport(monkeypatch)
    try:
        cfg = list_ingest.load_config()
        r = list_ingest._api_get(cfg["openalex"]["url"] + "https://doi.org/10.48550/arxiv.2101.00001",
                                 {"select": "doi,locations"}, 5.0)
        assert r.status_code == 200 and len(sent) == 1
        sent.clear()
        monkeypatch.setattr(list_ingest, "_api_get", lambda url, params, t, _o=list_ingest._api_get:
                            _o(url, params, t) if "openalex" in url else (_ for _ in ()).throw(OSError("skip inspire")))
        from vera import search_providers as sp
        with pytest.raises(sp.OpenAlexRefused) as ei:  # OPENALEX-FIX-1110: the shipped gate "legacy" refuses R5
            list_ingest._default_arxiv_original("2101.00001", cfg)
        assert ei.value.code == "terms_gate" and sent == []
        monkeypatch.setattr(sp, "openalex_terms_gate", lambda: "on")  # after the owner's terms ruling
        assert list_ingest._default_arxiv_original("2101.00001", cfg) == "10.1234/journal.x"
        assert len(sent) == 1
    finally:
        rate_limit.set_registry_for_tests(None)


def test_openalex_lookup_still_refused_when_slot_exhausted(monkeypatch):
    from vera import rate_limit
    sent = _openalex_transport(monkeypatch, {"rate_per_s": 0.0001, "burst": 1})
    try:
        cfg = list_ingest.load_config()
        url = cfg["openalex"]["url"] + "https://doi.org/10.48550/arxiv.2101.00001"
        list_ingest._api_get(url, {"select": "doi,locations"}, 5.0)  # takes the only token
        assert len(sent) == 1
        with pytest.raises(rate_limit.RateLimited):
            list_ingest._api_get(url, {"select": "doi,locations"}, 5.0)
        assert len(sent) == 1  # nothing sent when limited
    finally:
        rate_limit.set_registry_for_tests(None)


# LI-TRANSIENT-CODES-1110 (review S6-1): a transient failure is never reported as a final no-route / refused-portal.
def test_lookup_exception_is_lookup_unavailable_not_no_route(monkeypatch):
    def boom(x):
        raise OSError("timeout")
    rv = list_ingest.Resolvers(pmid=boom, arxiv_original=boom, known=lambda d, u: False)
    r = _ingest(["pmid:111", "arXiv:2301.12345"], monkeypatch, resolvers=rv)
    assert codes(r) == ["lookup-unavailable", "lookup-unavailable"]
    assert not list_ingest._ACCEPTED and (auto_acquire._S.queue is None or not list(auto_acquire._S.queue.queue))


def test_definitive_no_route_is_unchanged(monkeypatch):
    r = _ingest(["pmid:111"], monkeypatch, resolvers=Rv(pmid={}))
    assert codes(r) == ["no-route"]


def test_host_role_check_failure_is_check_unavailable_and_stores_nothing(monkeypatch):
    from vera import host_role

    def boom(rec):
        raise RuntimeError("config unreadable")
    monkeypatch.setattr(host_role, "role_for_record", boom)
    r = _ingest(["10.9999/journal.5"], monkeypatch, resolvers=Rv())
    assert codes(r) == ["check-unavailable"]
    assert not list_ingest._ACCEPTED and (auto_acquire._S.queue is None or not list(auto_acquire._S.queue.queue))


def _resp(sc, body=None):
    class R:
        status_code = sc

        def json(self):
            return body or {}
    return R()


def _lookup_with(monkeypatch, inspire, openalex):
    def fake(url, params, t):
        v = inspire if "inspire" in url else openalex
        if isinstance(v, Exception):
            raise v
        return v
    monkeypatch.setattr(list_ingest, "_api_get", fake)
    return list_ingest._default_arxiv_original("2101.00001", list_ingest.load_config())


def test_default_arxiv_lookup_transient_vs_definitive(monkeypatch):
    ok_empty = _resp(200, {"hits": {"hits": []}})
    oa_none = _resp(200, {"locations": []})
    assert _lookup_with(monkeypatch, ok_empty, _resp(404)) is None  # definitive: not found
    assert _lookup_with(monkeypatch, ok_empty, oa_none) is None
    for inspire, oa in ((ok_empty, _resp(429)), (ok_empty, _resp(503)), (ok_empty, OSError("t")),
                        (OSError("t"), oa_none), (_resp(500), oa_none), (OSError("t"), _resp(404))):
        with pytest.raises(Exception):
            _lookup_with(monkeypatch, inspire, oa)
    assert _lookup_with(monkeypatch, OSError("t"), _resp(200, {"locations": [
        {"landing_page_url": "https://doi.org/10.1234/j.x"}]})) == "10.1234/j.x"


# LI-TRANSIENT-CODES-1110 round 7 (S7-1): a permission denial is terminal (lookup-denied), not retryable.
def test_permission_denial_is_lookup_denied_not_unavailable(monkeypatch):
    from vera import api_permissions

    def denied(x):
        raise api_permissions.ApiDenied("R-API-DENIED", "inspire", "no entry")
    rv = list_ingest.Resolvers(pmid=denied, arxiv_original=denied, known=lambda d, u: False)
    r = _ingest(["pmid:111", "arXiv:2301.12345"], monkeypatch, resolvers=rv)
    assert codes(r) == ["lookup-denied", "lookup-denied"]
    assert not list_ingest._ACCEPTED


def test_http_401_403_from_lookup_services_is_denied(monkeypatch):
    ok_empty = _resp(200, {"hits": {"hits": []}})
    for inspire, oa in ((_resp(403), _resp(403)), (_resp(401), _resp(401)), (_resp(403), _resp(403))):
        with pytest.raises(list_ingest.LookupDenied):  # every lookup denied
            _lookup_with(monkeypatch, inspire, oa)
    rv = list_ingest.Resolvers(arxiv_original=lambda a: _lookup_with(monkeypatch, _resp(403), _resp(403)), known=lambda d, u: False)
    assert codes(_ingest(["arXiv:2301.12345"], monkeypatch, resolvers=rv)) == ["lookup-denied"]
    for inspire, oa in ((ok_empty, _resp(403)), (ok_empty, _resp(401))):  # only one denied: not terminal
        with pytest.raises(RuntimeError) as ei:
            _lookup_with(monkeypatch, inspire, oa)
        assert not isinstance(ei.value, list_ingest.LookupDenied)
    assert list_ingest._is_denied(OSError("t")) is False


def test_inspire_denial_falls_through_to_openalex_both_orders(monkeypatch):
    from vera import api_permissions
    found = _resp(200, {"locations": [{"landing_page_url": "https://doi.org/10.1234/j.x"}]})
    ins_found = _resp(200, {"hits": {"hits": [{"metadata": {"dois": [{"value": "10.5555/ins.1"}]}}]}})
    denied_exc = api_permissions.ApiDenied("R-API-DENIED", "x", "no entry")
    # INSPIRE denied (HTTP and exception form), OpenAlex answers: the item resolves
    assert _lookup_with(monkeypatch, _resp(403), found) == "10.1234/j.x"
    assert _lookup_with(monkeypatch, denied_exc, found) == "10.1234/j.x"
    # OpenAlex denied but INSPIRE answered first: resolved by INSPIRE
    assert _lookup_with(monkeypatch, ins_found, _resp(403)) == "10.5555/ins.1"
    # INSPIRE denied, OpenAlex has nothing / unavailable: not terminal-denied
    for oa in (_resp(200, {"locations": []}), _resp(404), _resp(503)):
        with pytest.raises(RuntimeError) as ei:
            _lookup_with(monkeypatch, _resp(403), oa)
        assert not isinstance(ei.value, list_ingest.LookupDenied)
    # both denied, in either exception/HTTP form: lookup-denied
    for i, o in ((denied_exc, denied_exc), (_resp(403), denied_exc), (denied_exc, _resp(401))):
        with pytest.raises(list_ingest.LookupDenied):
            _lookup_with(monkeypatch, i, o)
