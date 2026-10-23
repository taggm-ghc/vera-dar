"""INTEG-1023: end-to-end POST /ask over the integrated candidate (FRESH + amend + CONT + ROUTE-R3 + ARX-A2 route).

OFFLINE and in process (FastAPI TestClient via the ROUTE-R2 RouteWorld harness): only external I/O is stubbed (the
search provider chain, the Europe PMC transport, the full-text http_get) and the answer model / claim checker (the
_pipeline_support rig). Licence, injection, relevance, claim-check, caps and the DB/socket guards all run unchanged; the
DB/socket guard is fail-loud. The licence gate is the SHIPPED one (no test-only gate rule): arXiv HTML stays refused
(D-ARX-1 pending the owner). Background jobs are never started (tests/conftest.py autouse); the queue is inspected only.
"""
import pytest

from tests import arxiv_synth as X
from tests.test_acq_discovered import Router
from tests.test_route_ddg_first_r2 import (  # noqa: F401  (h, rig, clock, no_db_no_socket, world are fixtures)
    AX, DDG, EPMC, HOURS, NOW, OA, _cands, _cands_public, _grounded, _need_portals, _portals_of, _ro, canon, clock,
    ddg_row, epmc_row, h, no_db_no_socket, oa_rec, rig, world)
from vera import arxiv_fulltext, auto_acquire, inline_acquire


@pytest.fixture(autouse=True)
def _isolated_queue(monkeypatch):
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    monkeypatch.setattr(auto_acquire, "_worker_loop", lambda *a, **k: None)
    monkeypatch.setattr(arxiv_fulltext, "_sleep", lambda s: None)
    yield
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()


def _queued():
    q = auto_acquire._S.queue
    return [] if q is None else [c for j in list(q.queue) for c in (j.cands or ())]


def _by(cs, key, val):
    return [c for c in cs if str(c["ids"].get(key) or "").lower() == str(val).lower()]


# ============================================================================================ 1. the complete local path
def test_e2e_corpus_miss_ddg_first_epmc_jats_admitted_grounded_answer_with_citations(world):
    doi, pmc = "10.9999/integ.a1", "PMC9230001"
    world.doc(doi, pmc)
    world.p.beh[DDG] = [ddg_row(f"https://journals.example.org/doi/{doi}/full")]
    world.p.beh[OA] = [oa_rec(doi, "W92301", pmc)]
    world.p.beh[EPMC] = [epmc_row(doi, pmc)]
    world.store()  # corpus miss: nothing stored
    body = world.ask()
    assert "insufficient" in body["triggers"]
    # DDG is the first portal CALLED and the first portal RECORDED; records follow the actual call order
    assert world.p.log[0] == DDG, world.p.log
    ports = _need_portals(body)
    assert canon(ports[0]["portal"]) == "ddg" and ports[0]["status"] == "completed"
    recorded = [canon(p["portal"]) for p in ports if p["status"] not in ("not_attempted", "skipped")]
    assert recorded == [canon(n) for n in world.p.portal_calls]
    # eligible non-arXiv acquisition: Europe PMC JATS full text fetched, admitted, persisted
    assert any(f"/{pmc}/fullTextXML" in u for u in world.rig.web.urls), world.rig.web.urls
    assert not any("arxiv.org" in u for u in world.rig.web.urls)
    assert len(world.rig.store.plans) == 1
    ia = body["contract"]["inline_acquisition"]
    assert ia["status"] == "ok" and ia["admitted"] == 1 and ia["persisted"] is True
    # grounded answer with citations from the newly admitted evidence, in THIS request (accept_predicate)
    _grounded(body)
    ids = {a["document_id"] for a in body["admitted_evidence"]}
    assert ids and all(i.startswith("acq-") for i in ids) and {c["document_id"] for c in body["citations"]} == ids
    ro = _ro(body)
    assert ro["kind"] == "answered_with_new" and ro["complete"] is True
    c = _by(_cands(body), "doi", doi)
    assert len(c) == 1 and c[0]["terminal_reason"] == "admitted" and c[0]["ids"]["pmcid"] == pmc  # PMCIDPUBLIC-1023
    assert {"ddg", "openalex"} <= _portals_of(c[0])
    # FRESHBASIS amend: corpus_added_at = write time on the freshness clock; source_authored_at = pubYear, never 'recent'
    prov = world.rig.store.plans[0]["source"]["provenance"]
    assert prov["retrieved_at"] == NOW.isoformat(timespec="seconds")
    for s in body["sources"]:
        assert s["source_authored_precision"] == "year" and s["evidence_recency"] != "recent", s
        assert s["corpus_added_at"] != s["source_authored_at"]
    assert _queued() == []  # nothing unfinished: no background job


# ============================================================================================ 2. terminal reasons
def test_e2e_inaccessible_source_timeout_is_queued_background_and_search_incomplete(world):
    doi, pmc = "10.9999/integ.t1", "PMC9230101"
    world.doc(doi, pmc)
    world.web.over[f"/{pmc}/fullTextXML"] = TimeoutError("read timed out")
    world.p.beh[DDG] = [ddg_row(f"https://journals.example.org/doi/{doi}/full")]
    world.p.beh[OA] = [oa_rec(doi, "W92311", pmc)]
    world.store()
    body = world.ask()
    assert world.rig.store.plans == []
    c = _by(_cands(body), "doi", doi)
    assert len(c) == 1, _cands(body)
    assert c[0]["terminal_reason"] == "queued_background", c[0]  # transient timeout -> CONT-1023 best-effort queue
    ro = _ro(body)
    assert ro["complete"] is False and ro["kind"] != "no_valuable_source"
    if c[0]["terminal_reason"] == "queued_background":  # CONT-1023 best-effort queue (in memory only)
        assert c[0]["background"] in ("queued", "already_queued")
        assert [q.get("doi") for q in _queued()] == [doi]
        assert "best-effort" in ro["user_text"]


def test_e2e_licence_rejected_source_is_terminal_and_never_queued(world):
    doi, pmc = "10.9999/integ.l1", "PMC9230201"
    world.doc(doi, pmc)
    world.web.records[doi]["license"] = "cc by-nd"
    world.p.beh[OA] = [oa_rec(doi, "W92321", pmc)]
    world.p.beh[EPMC] = [epmc_row(doi, pmc)]
    world.store()
    body = world.ask()
    assert world.rig.store.plans == []
    c = _by(_cands(body), "doi", doi)
    assert len(c) == 1 and c[0]["terminal_reason"] == "licence_rejected", c
    assert c[0]["reason"].startswith(("R-LIC-", "R-ACQ-NOLIC", "R-ACQ-LICENCE")), c[0]
    assert c[0].get("background") != "queued" and _queued() == []


def _ask_with_discovery(world, raws, providers=None):
    """world.ask() with the /ask DISCOVERY search returning `raws` (the ARX-A2 hook is acq_discovered.discovered_candidates,
    i.e. records found by the discovery search; inline.use_discovered is on in the shipped config)."""
    from tests._gap_support import http_ask
    from vera import search_providers as sp
    rig = world.rig
    if providers is not None:
        rig.cfg["search"]["providers"] = list(providers)
    rig.setup(web=world.web, cands=[])

    def no_get(*a, **k):
        raise RuntimeError("offline: landing GET")
    real = sp.acquisition_search(auto_acquire.make_search(rig.cfg), rig.cfg, transport=world.p.transport,
                                 terms_fn=lambda: (None, None), url_check=lambda u: (False, "offline"), get=no_get)
    rig.deps.search = lambda question, n, t: (rig.search_calls.append(question), real(question, n, t))[1]
    resp, _s = http_ask(rig, world.mp, raws, offtopic=False)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_e2e_arxiv_html_candidate_refused_at_current_gate_with_precise_reason(world):
    world.web = Router(world.web.records, world.web.jats, X.routes(licence=X.CC_BY))
    world.store()
    body = _ask_with_discovery(world, [X.discovery_src()], providers=[DDG])
    assert world.p.log[0] == DDG, world.p.log
    urls = world.rig.web.urls
    assert X.ABS_URL in urls and X.HTML_URL in urls, urls  # fetched through the arXiv route (robots, abs, html)
    assert world.rig.store.plans == []  # never admitted under the shipped gate (D-ARX-1 pending)
    att = arxiv_fulltext.last_attempts(X.AID + X.VER)
    assert att and att[-1]["reason"] == "R-LIC-REJECT", att  # the gate's own text is kept in the attempts
    c = [c for c in _cands_public(body) if c["ids"].get("arxiv_id") == X.AID
         or str(c["ids"].get("doi") or "").lower() == X.DOI.lower() or c["ids"].get("url") in (X.ABS_URL, X.HTML_URL)]
    assert c, _cands_public(body)
    assert c[0]["reason"] == "R-ACQ-ARXIV_HTML_POLICY" and c[0]["terminal_reason"] == "licence_rejected", c[0]
    assert c[0].get("background") != "queued" and _queued() == []
    assert _ro(body)["kind"] != "answered_with_new"


def test_e2e_url_only_candidate_is_marked_followable_and_queued(world):
    url = "https://example.org/reports/outbreak-landing"
    world.p.beh[DDG] = [ddg_row(url, title="SYNTHETIC outbreak transmission incidence report")]
    world.store()
    body = world.ask(providers=[DDG])
    assert not any("example.org" in u for u in world.rig.web.urls)  # no request to the URL host
    c = [c for c in _cands_public(body) if c["ids"].get("url") == url]
    assert len(c) == 1, _cands_public(body)
    assert c[0]["terminal_reason"] == "queued_background" and c[0]["route"] == "follow_url", c[0]
    assert c[0]["reason"] == "R-ACQ-NODOI"
    assert [q.get("url") for q in _queued()] == [url]
    ro = _ro(body)
    assert ro["complete"] is False and "budget ran out" not in ro["user_text"]


# ============================================================================================ 3. freshness basis end to end
def test_e2e_freshness_trigger_uses_corpus_added_at_not_authoring_date(world):
    # added to the corpus 30 h ago, authored 1 h ago -> stale (trigger basis corpus_added_at)
    world.store(dict(doi="10.9999/integ.f1", published=HOURS(1), acquired=HOURS(30)))
    body = world.ask()
    assert body["triggers"] == ["stale"] and body["freshness"]["any_stale_or_unknown"] is True
    assert world.p.log and world.p.log[0] == DDG


def test_e2e_recently_added_old_publication_is_fresh_but_labelled_older(world):
    world.store(dict(doi="10.9999/integ.f2", published=NOW.replace(year=2019), acquired=HOURS(0.5)))
    body = world.ask()
    assert body["triggers"] == [] and body["freshness"]["any_stale_or_unknown"] is False
    assert world.p.log == [] and world.rig.store.plans == []
    _grounded(body)
    for s in body["sources"]:
        assert s["evidence_recency"] == "older" and s["corpus_added_at"] == HOURS(0.5).isoformat(timespec="seconds")
        assert s["source_authored_at"].startswith("2019-") and s["corpus_added_at"] != s["source_authored_at"]
