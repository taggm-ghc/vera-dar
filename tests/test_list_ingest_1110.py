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
