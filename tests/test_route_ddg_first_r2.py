"""ROUTE-R2 (P0): DDG-first multi-portal discovery, acceptance tests via POST /ask (TestClient, in process). OFFLINE.

Harness: the FRESH-C4 World/rig/http_ask pattern (tests/_pipeline_support, tests/_gap_support, tests/acq_support). The one
difference: the acquisition job's search is the REAL stack (search_providers.acquisition_search over auto_acquire.make_search
over search_and_fetch's chain), and ONLY the external I/O under it is stubbed: the provider chain (fake DuckDuckGo Lite /
OpenAlex / arXiv provider objects behind search_and_fetch.cached_chain + search_question) and the Europe PMC transport.
Every stub appends to ONE ordered call log (Portals.log), so the actual CALL ORDER is observable. No safety control is
disabled: licence, injection, relevance, claim-check and DB/socket guards all run; the DB/socket guards are fail-loud.

Contract under test (additive): body["research_outcome"] = {kind, user_text, complete, portals[], candidates[]};
portals = list in ACTUAL CALL ORDER of {portal, status in completed|timeout|quota|error|skipped|not_attempted, n_results};
candidates[] carry ids {doi, pmcid, openalex_id, arxiv_id, url} and a `portal` provenance field (str or list); body["triggers"].
On the base (9d0d7643) research_outcome does not exist, so tests that read it fail with "feature missing"; the
test_base_* test reads only the stubs' own call log, and so records the CURRENT call order.
"""
import json
import re
import socket
from datetime import datetime, timedelta, timezone

import pytest

from tests import accept_predicate as ap
from tests import acq_support as A
from tests import fulltext_synth
from tests import test_fulldoc_e2e as e2e
from tests._gap_support import http_ask
from tests._pipeline_support import h, rig  # noqa: F401  (fixtures)
from vera import auto_acquire, fulldoc_corpus, fulldoc_db
from vera import search_and_fetch
from vera import search_providers as sp

NOW = datetime(2026, 10, 23, 12, 0, 0, tzinfo=timezone.utc)
HOURS = lambda h_: NOW - timedelta(hours=h_)  # noqa: E731
NO_VALUABLE = "No sufficiently valuable source was found across the portals searched."
DDG, OA, AX, EPMC = sp.DDG_NAME, "openalex", "arxiv", "europepmc"
STATUSES = {"completed", "timeout", "quota", "error", "skipped", "not_attempted"}
CC0 = {"metadata": {"id": "CC0-1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/"}}


def canon(name):
    n = str(name or "").lower()
    if n in ("ddg", "duckduckgo", "duckduckgo_lite", "duckduckgo-lite"):
        return "ddg"
    if n in ("europepmc", "europe_pmc", "epmc", "europe pmc"):
        return "epmc"
    return n


# ----------------------------------------------------------------------------------------------------- guards / clock
@pytest.fixture
def no_db_no_socket(monkeypatch):
    hits = []

    def boom(kind):
        def _f(*a, **k):
            hits.append(kind)
            raise AssertionError(f"forbidden {kind} attempted")
        return _f
    import sqlalchemy
    monkeypatch.setattr(sqlalchemy, "create_engine", boom("sqlalchemy.create_engine"))
    for mod in ("psycopg2", "psycopg"):
        try:
            monkeypatch.setattr(__import__(mod), "connect", boom(f"{mod}.connect"))
        except ImportError:
            pass
    monkeypatch.setattr(socket.socket, "connect", boom("socket.connect"))
    monkeypatch.setattr(socket, "create_connection", boom("socket.create_connection"))
    return hits


@pytest.fixture
def clock(monkeypatch):
    """Single freshness clock hook; tolerant of the feature being absent (the stale tests then fail on the missing trigger)."""
    try:
        import vera.freshness as fr
    except ImportError:
        return None
    monkeypatch.setattr(fr, "utcnow", lambda: NOW, raising=False)
    return fr


# ----------------------------------------------------------------------------------------------------- portal stubs
class _Resp:
    def __init__(self, body, status=200):
        self.text, self.status_code, self.headers = (body if isinstance(body, str) else json.dumps(body)), status, {}


class _Provider:
    def __init__(self, name):
        self.name, self.last_query = name, None


class Portals:
    """Fake external portals + the ordered call log. Behaviour per portal: a list of results, or an Exception to raise.
    log entries: 'duckduckgo_lite' | 'openalex' | 'arxiv' | 'europepmc' (search) | 'europepmc_lookup' (PMC id lookup)."""

    def __init__(self):
        self.log = []
        self.beh = {DDG: [], OA: [], AX: [], EPMC: []}
        self.lookup = {}  # PMCID -> europepmc lite row (for DDG PMC links)
        self.chain_names = [DDG, AX, OA]  # arXiv listed before OpenAlex: "arXiv returns first"

    def chain(self):
        return [_Provider(n) for n in self.chain_names]

    def search_question(self, question, num_results=10, *, providers=None, **kw):
        out, errs = [], []
        provs = list(providers or ())
        for p in provs:
            self.log.append(p.name)
            b = self.beh.get(p.name, [])
            if isinstance(b, Exception):
                errs.append(b)
                continue
            out.extend(dict(r) for r in b)
        if errs and not out and len(errs) == len(provs):
            raise errs[0]
        return out

    def transport(self, url, params, headers, timeout):
        q = str((params or {}).get("query", ""))
        if "PMCID:" in q or "EXT_ID:" in q:
            self.log.append("europepmc_lookup")
            rows = [self.lookup[k] for k in self.lookup if k in q]
            return _Resp({"resultList": {"result": rows}})
        self.log.append(EPMC)
        b = self.beh[EPMC]
        if isinstance(b, Exception):
            raise b
        return _Resp({"resultList": {"result": list(b)}})

    @property
    def portal_calls(self):
        """portal names in first-call order, lookups excluded"""
        seen = []
        for n in self.log:
            if n != "europepmc_lookup" and n not in seen:
                seen.append(n)
        return seen


def oa_rec(doi, wid, pmcid=None, title="SYNTHETIC outbreak transmission incidence study"):
    return {"provider": "openalex", "doi": doi, "url": f"https://doi.org/{doi}", "title": title,
            "snippet": "transmission incidence outbreak pathogen", "openalex_id": f"https://openalex.org/{wid}", "licence": CC0,
            "stored_text_kind": "abstract_metadata",
            "oa": {"is_oa": True, "has_oa_location": True, "licence": "cc-by", "pmcid": pmcid}}


def ax_rec(n):
    aid = f"2401.{n:05d}"
    return {"provider": "arxiv", "url": f"https://arxiv.org/abs/{aid}", "doi": f"10.48550/arxiv.{aid}", "arxiv_id": aid,
            "title": f"SYNTHETIC outbreak transmission preprint {n}", "snippet": "transmission incidence outbreak pathogen",
            "licence": CC0, "stored_text_kind": "abstract_metadata"}


def epmc_row(doi, pmcid, oa="Y", title="SYNTHETIC outbreak transmission incidence study"):
    r = {"id": pmcid, "source": "MED", "pmcid": pmcid, "title": title, "pubYear": "2024", "isOpenAccess": oa, "inEPMC": "Y"}
    if doi:
        r["doi"] = doi
    return r


def ddg_row(url, title="DDG result", group="portals"):
    return {"url": url, "title": title, "discovery_group": group}


# ----------------------------------------------------------------------------------------------------- the world
class RouteWorld:
    def __init__(self, rig, mp, tmp_path):
        self.rig, self.mp, self.tmp = rig, mp, tmp_path
        rig.cfg["freshness_max_age_hours"] = 24
        if isinstance(rig.cfg.get("freshness"), dict):
            rig.cfg["freshness"]["max_age_hours"] = 24
        rig.cfg["scope_expansion"]["override_layer2_declines"] = True
        self.p = Portals()
        self.web = A.Web({}, {})
        self.n_docs = 0
        sp._QUOTA.reset()
        mp.setattr(search_and_fetch, "cached_chain", lambda settings=None: (self.p.chain(), 1, "fan_out"))
        mp.setattr(search_and_fetch, "search_question", self.p.search_question)

    def doc(self, doi, pmcid, oa="Y", topic="epi", key=None):
        """A document the (fake) Europe PMC can resolve and serve as JATS."""
        self.n_docs += 1
        k = key or doi or pmcid
        self.web.records[k] = A.epmc_record(doi or "", pmcid=pmcid, oa=oa)
        if not doi:
            self.web.records[k].pop("doi", None)
        self.web.jats[pmcid] = fulltext_synth.jats("acq_epi", topic, doi or f"10.9999/nodoi.{pmcid}", seed=100 + self.n_docs)
        return self

    def store(self, *specs):
        """Stored acquired chunks (C4 pattern); published/acquired times are controlled."""
        plans = []
        for i, s in enumerate(specs, start=1):
            web, _ = A.epi_setup(doi=s["doi"], pmcid=f"PMC90000{i}", topic=s.get("topic", "epi"), jats_kw={"seed": 200 + i})
            st = A.Store()
            auto_acquire.run_job(A.job(e2e.Q_EPI), A.cfg(), A.deps([A.cand(s["doi"])], web, store=st), clock=A.clock_at(), env=A.ENV_ON)
            assert st.plans, "fixture: stored document was not admitted"
            p = st.plans[0]
            prov = p["source"]["provenance"]
            prov["published"] = prov["item"]["published_year"] = s["published"].isoformat(timespec="seconds")
            acq = s["acquired"].isoformat(timespec="seconds")
            prov["item"]["fetched_at"] = prov["fetched_at"] = prov["retrieved_at"] = prov["acquisition"]["acquired_at"] = acq
            plans.append(p)
        src, spans = A.rows_from_plans(plans) if plans else ([], [])
        corpus = fulldoc_db.load_acquired_rows(src, spans, fulldoc_corpus.load_config(), A.cfg()) if plans else None
        self.mp.setattr(fulldoc_corpus, "get_acquired_corpus", lambda cfg=None, env=None, now=None: corpus)
        return self

    def ask(self, q=e2e.Q_EPI, providers=None, max_calls=None):
        rig = self.rig
        if providers is not None:
            rig.cfg["search"]["providers"] = list(providers)
        if max_calls is not None:
            rig.cfg["bounds"]["max_search_calls_per_job"] = max_calls
        rig.setup(web=self.web, cands=[])

        def no_get(*a, **k):
            raise RuntimeError("offline: landing GET")
        real = sp.acquisition_search(auto_acquire.make_search(rig.cfg), rig.cfg, transport=self.p.transport,
                                     terms_fn=lambda: (None, None), url_check=lambda u: (False, "offline"), get=no_get)

        def counted(question, n, t):
            rig.search_calls.append(question)
            return real(question, n, t)
        rig.deps.search = counted
        resp, _s = http_ask(rig, self.mp, [], q=q, offtopic=False)
        assert resp.status_code == 200, resp.text
        return resp.json()


@pytest.fixture
def world(rig, monkeypatch, tmp_path, clock, no_db_no_socket):  # noqa: F811
    monkeypatch.delenv("VERA_AUTOACQ_DISABLED", raising=False)
    w = RouteWorld(rig, monkeypatch, tmp_path)
    yield w
    assert no_db_no_socket == []


# ----------------------------------------------------------------------------------------------------- assertions
def _ro(body):
    assert "research_outcome" in body, f"feature missing: response has no research_outcome (keys: {sorted(body)})"
    return body["research_outcome"]


def _need_portals(body):
    ro = _ro(body)
    assert isinstance(ro.get("portals"), list) and ro["portals"], f"feature missing: research_outcome.portals empty/absent: {list(ro)}"
    for p in ro["portals"]:
        assert set(p) >= {"portal", "status", "n_results"} and p["status"] in STATUSES, p
    return ro["portals"]


def _status(body, name):
    hit = [p for p in _need_portals(body) if canon(p["portal"]) == canon(name)]
    assert hit, f"portal {name!r} not recorded in research_outcome.portals: {[p['portal'] for p in _need_portals(body)]}"
    return hit[0]


def _portals_of(c):
    v = c.get("portal", c.get("portals"))
    assert v, f"feature missing: candidate has no `portal` provenance field: {sorted(c)}"
    return {canon(x) for x in (v if isinstance(v, (list, tuple, set)) else [v])}


def _cands(body):
    cs = _ro(body).get("candidates") or []
    for c in cs:
        assert set(c.get("ids") or {}) >= {"doi", "pmcid", "openalex_id", "arxiv_id", "url"}, c.get("ids")
    return cs


def _cands_public(body):
    """ROUTE-R3: candidates with the public id set (pmcid may be present: PMCID-PUBLIC-1023)."""
    cs = _ro(body).get("candidates") or []
    for c in cs:
        assert set(c.get("ids") or {}) >= {"doi", "openalex_id", "arxiv_id", "url"}, c.get("ids")
    return cs


def _by_doi(cs, doi):
    return [c for c in cs if str(c["ids"].get("doi") or "").lower() == doi.lower()]


def _grounded(body):
    f = ap.grounded_failures(200, body)
    assert f == [], f


def _lower_doi_ids(cs):
    return [str(c["ids"].get("doi") or "").lower() for c in cs]


# ============================================================================================================ base: observed order
def test_base_observed_call_order_insufficient(world):
    """Records the CURRENT call order (stub log only; no research_outcome needed). Contract: DDG is the first portal called."""
    world.doc("10.9999/route.a1", "PMC9100001")
    world.p.beh[OA] = [oa_rec("10.9999/route.a1", "W91001", "PMC9100001")]
    world.p.beh[EPMC] = [epmc_row("10.9999/route.a1", "PMC9100001")]
    world.p.beh[DDG] = [ddg_row("https://journals.example.org/doi/10.9999/route.a1/full")]
    world.store()
    world.ask()
    assert world.p.log, "no portal was called at all"
    assert world.p.log[0] == DDG, f"wrong order: first portal call was {world.p.log[0]!r}; full order {world.p.log}"


# ============================================================================================================ 1
@pytest.mark.parametrize("trigger", ["insufficient", "stale"])
def test_1_ddg_attempted_first_for_both_triggers(world, trigger):
    world.doc("10.9999/route.b1", "PMC9100002")
    world.p.beh[OA] = [oa_rec("10.9999/route.b1", "W91002", "PMC9100002")]
    world.p.beh[EPMC] = [epmc_row("10.9999/route.b1", "PMC9100002")]
    world.p.beh[DDG] = [ddg_row("https://journals.example.org/doi/10.9999/route.b1/full")]
    if trigger == "stale":  # INTEG-1023: FRESHBASIS amend - trigger basis is corpus_added_at: added 30 h ago, published 1 h ago
        world.store(dict(doi="10.9999/route.old", published=HOURS(1), acquired=HOURS(30)))
    else:
        world.store()
    body = world.ask()
    assert "triggers" in body, "feature missing: no triggers field"
    assert trigger in body["triggers"], body["triggers"]
    assert world.p.log, f"no external search for trigger {trigger}"
    assert world.p.log[0] == DDG, f"wrong order: {world.p.log}"
    ports = _need_portals(body)
    assert canon(ports[0]["portal"]) == "ddg", [p["portal"] for p in ports]
    called = [canon(n) for n in world.p.portal_calls]
    recorded = [canon(p["portal"]) for p in ports if p["status"] not in ("not_attempted", "skipped")]
    assert recorded == called, f"portals not in actual call order: recorded {recorded} vs called {called}"


# ============================================================================================================ 2
@pytest.mark.parametrize("exc,status", [(RuntimeError("ddg backend failure"), "error"), (TimeoutError("ddg timed out"), "timeout")])
def test_2_ddg_failure_is_recorded_and_secondary_portals_still_run(world, exc, status):
    world.doc("10.9999/route.c1", "PMC9100003")
    world.p.beh[DDG] = exc
    world.p.beh[OA] = [oa_rec("10.9999/route.c1", "W91003", "PMC9100003")]
    world.p.beh[EPMC] = [epmc_row("10.9999/route.c1", "PMC9100003")]
    world.store()
    body = world.ask()
    assert world.p.log and world.p.log[0] == DDG, f"wrong order: {world.p.log}"
    assert OA in world.p.log and EPMC in world.p.log, f"DDG-first became DDG-only: calls {world.p.log}"
    assert _status(body, "ddg")["status"] == status
    assert _status(body, "openalex")["status"] == "completed" and _status(body, "europepmc")["status"] == "completed"
    assert _ro(body)["kind"] == "answered_with_new"
    _grounded(body)


# ============================================================================================================ 3
def test_3_ddg_hit_with_doi_is_enriched_by_scholarly_indexes_ids_preserved(world):
    doi = "10.9999/route.d1"
    world.doc(doi, "PMC9100004")
    world.p.beh[DDG] = [ddg_row(f"https://journals.example.org/doi/{doi}/full")]  # DDG carries only the DOI
    world.p.beh[OA] = [oa_rec(doi, "W91004", "PMC9100004", title="SYNTHETIC outbreak transmission incidence study (OpenAlex title)")]
    world.p.beh[EPMC] = [epmc_row(doi, "PMC9100004")]
    world.store()
    body = world.ask()
    hit = _by_doi(_cands(body), doi)
    assert len(hit) == 1, f"one paper from DDG+OpenAlex+Europe PMC must be one candidate: {_lower_doi_ids(_cands(body))}"
    ids = hit[0]["ids"]
    assert ids["doi"].lower() == doi and "W91004" in (ids["openalex_id"] or "") and ids["pmcid"] == "PMC9100004", ids
    assert {"ddg", "openalex"} <= _portals_of(hit[0]), _portals_of(hit[0])
    assert canon(_need_portals(body)[0]["portal"]) == "ddg"


def test_3b_ddg_pmc_link_is_mapped_to_doi_and_merged_with_openalex(world):
    doi = "10.9999/route.d2"
    world.doc(doi, "PMC9100005")
    world.p.lookup["PMC9100005"] = {"pmcid": "PMC9100005", "pmid": "91000005", "doi": doi}
    world.p.beh[DDG] = [ddg_row("https://pmc.ncbi.nlm.nih.gov/articles/PMC9100005/")]
    world.p.beh[OA] = [oa_rec(doi, "W91005", "PMC9100005")]
    world.store()
    body = world.ask()
    hit = _by_doi(_cands(body), doi)
    assert len(hit) == 1 and hit[0]["ids"]["pmcid"] == "PMC9100005" and "W91005" in (hit[0]["ids"]["openalex_id"] or ""), hit
    assert {"ddg", "openalex"} <= _portals_of(hit[0])


# ============================================================================================================ 4
def test_4_source_diversity_under_candidate_caps_arxiv_does_not_crowd_out_others(world):
    cap = int(world.rig.cfg["bounds"]["max_candidates_resolved"])
    world.p.beh[AX] = [ax_rec(i) for i in range(1, cap + 4)]  # arXiv answers first and has more than the cap
    others = [f"10.9999/route.e{i}" for i in range(1, 4)]
    for i, d in enumerate(others, 1):
        world.doc(d, f"PMC91001{i:02d}")
    world.p.beh[OA] = [oa_rec(d, f"W9101{i}", f"PMC91001{i:02d}") for i, d in enumerate(others, 1)]
    world.store()
    body = world.ask(providers=[OA, AX])
    from urllib.parse import unquote  # ROUTE-R3: compare normalised DOIs, not raw vs URL-encoded
    resolved = unquote(" ".join(world.rig.web.urls)).lower()  # Europe PMC lookups made by the unchanged resolver
    assert any(d.lower() in resolved for d in others), f"arXiv-only: the cap of {cap} was filled by arXiv candidates; resolved urls: {world.rig.web.urls}"
    # ROUTE-R3: the CAPPED acquisition candidates are the evaluated ones; the full listing also records those past the
    # cap as budget_exhausted (FRESH C3)
    # INTEG-1023: with CONT-1023 the past-cap candidates are queued_background (reason R-ACQ-DEFER_CAP), not evaluated here
    cs = [c for c in _cands_public(body) if c.get("terminal_reason") != "budget_exhausted" and c.get("reason") != "R-ACQ-DEFER_CAP"]
    assert 0 < len(cs) <= cap, [c.get("terminal_reason") for c in _cands_public(body)]
    non_ax = [c for c in cs if not str(c["ids"].get("doi") or "").lower().startswith("10.48550/arxiv")]
    assert non_ax, f"capped candidate list is all arXiv ({len(cs)} of {cap}) although eligible others exist"
    assert {d.lower() for d in others} & set(_lower_doi_ids(cs))


# ============================================================================================================ 5
def test_5_identifiers_survive_normalisation_and_dedup_two_provenances_kept(world):
    d1, d2 = "10.9999/ROUTE.F1", "10.9999/route.f2"
    world.doc(d1.lower(), "PMC9100201")
    world.doc(d2, "PMC9100202")
    world.p.beh[DDG] = [ddg_row(f"https://journals.example.org/doi/{d1}/full")]       # upper-case DOI from DDG
    world.p.beh[OA] = [oa_rec(d1.lower(), "W91201", "PMC9100201"), oa_rec(d2, "W91202")]  # lower-case DOI from OpenAlex
    world.p.beh[EPMC] = [epmc_row(d2, "PMC9100202")]                                   # d2 again, with its PMCID
    world.p.beh[AX] = [ax_rec(77)]
    world.store()
    body = world.ask(providers=[OA, AX])
    cs = _cands(body)
    dois = _lower_doi_ids(cs)
    assert dois.count(d1.lower()) == 1 and dois.count(d2) == 1, f"duplicate candidates for one paper: {dois}"
    c1, c2 = _by_doi(cs, d1)[0], _by_doi(cs, d2)[0]
    assert {"ddg", "openalex"} <= _portals_of(c1) and {"openalex", "epmc"} <= _portals_of(c2), (_portals_of(c1), _portals_of(c2))
    assert "W91201" in c1["ids"]["openalex_id"] and c1["ids"]["pmcid"] == "PMC9100201"
    assert "W91202" in c2["ids"]["openalex_id"] and c2["ids"]["pmcid"] == "PMC9100202"
    ax = [c for c in cs if c["ids"].get("arxiv_id")]
    assert ax and ax[0]["ids"]["arxiv_id"] == "2401.00077" and "arxiv.org" in str(ax[0]["ids"]["url"]), cs


# ============================================================================================================ 6
@pytest.mark.parametrize("route", ["doi", "pmcid_only"])
def test_6_non_arxiv_full_text_via_europe_pmc_jats_is_admitted_and_used(world, route):
    doi, pmc = "10.9999/route.g1", "PMC9100301"
    d = doi if route == "doi" else None
    world.doc(d, pmc)
    world.p.beh[EPMC] = [epmc_row(d, pmc)]  # only Europe PMC finds it: no arXiv, no OpenAlex, no DDG
    world.store()
    body = world.ask()
    assert any(f"/{pmc}/fullTextXML" in u for u in world.rig.web.urls), f"Europe PMC JATS never fetched: {world.rig.web.urls}"
    assert len(world.rig.store.plans) == 1
    _grounded(body)
    assert _ro(body)["kind"] == "answered_with_new"
    c = [c for c in _cands(body) if c["ids"]["pmcid"] == pmc]
    assert c and c[0]["terminal_reason"] == "admitted" and "epmc" in _portals_of(c[0])


# ============================================================================================================ 7
@pytest.mark.parametrize("variant", ["zero_results", "rejected_at_admission"])
def test_7_completed_searches_without_valuable_source(world, variant):
    # ROUTE-R3 (C3 canonical): no_valuable_source = the search completed and no candidate was admitted or blocked at
    # acquisition. The second fixture is fetched in full text and rejected at ADMISSION (off-topic: relevance).
    if variant == "rejected_at_admission":
        world.doc("10.9999/route.h1", "PMC9100401", topic="clim")
        world.p.beh[OA] = [oa_rec("10.9999/route.h1", "W91401", "PMC9100401")]
    world.store()
    body = world.ask()
    assert all(p["status"] == "completed" for p in _need_portals(body)), _need_portals(body)
    ro = _ro(body)
    if variant == "rejected_at_admission":
        reasons = [c.get("terminal_reason") for c in _cands_public(body)]
        assert reasons and all(r.endswith("_rejected") for r in reasons), reasons  # admission, not acquisition
    assert ro["kind"] == "no_valuable_source" and ro["complete"] is True, (ro["kind"], ro["complete"])
    assert ro["user_text"].startswith(NO_VALUABLE), ro["user_text"]  # FRESH may append a second sentence
    assert body["result_kind"] != "answer" and world.rig.h.answer_calls == [] and world.rig.store.plans == []


# ============================================================================================================ 8
INCOMPLETE = [("ddg", DDG, TimeoutError("timed out"), "timeout"), ("openalex", OA, RuntimeError("HTTP 429 rate limit exceeded"), "quota"),
              ("europepmc", EPMC, TimeoutError("timed out"), "timeout"), ("europepmc", EPMC, RuntimeError("HTTP 429 quota"), "quota")]


@pytest.mark.parametrize("who,key,exc,status", INCOMPLETE)
def test_8_incomplete_search_is_never_all_portals(world, who, key, exc, status):
    world.p.beh[key] = exc  # the others complete with zero results
    world.store()
    body = world.ask()
    assert _status(body, who)["status"] == status
    ro = _ro(body)
    assert ro["kind"] == "incomplete_search" and ro["complete"] is False, (ro["kind"], ro["complete"])
    assert not re.search(r"\ball\b[^.]*\bportals?\b", ro["user_text"], re.I), ro["user_text"]
    assert ro["user_text"] != NO_VALUABLE and body["result_kind"] != "answer"


# ============================================================================================================ distinctions
def test_status_not_attempted_is_distinct_from_completed_with_zero(world):
    # the per-job search-call cap leaves no slot for DDG / Europe PMC (OpenAlex AND+OR takes 2)
    world.store()
    body = world.ask(max_calls=2)
    st = {canon(p["portal"]): p for p in _need_portals(body)}
    assert st["ddg"]["status"] == "not_attempted" and st["epmc"]["status"] == "not_attempted", st
    assert st["openalex"]["status"] == "completed" and st["openalex"]["n_results"] == 0  # attempted, zero results
    assert world.p.log and set(world.p.log) == {OA}, world.p.log  # nothing marked not_attempted was called (AND-then-OR retry allowed)
    assert _ro(body)["kind"] == "incomplete_search" and _ro(body)["user_text"] != NO_VALUABLE


def test_status_error_quota_completed_zero_are_different_values(world):
    world.p.beh[DDG] = RuntimeError("ddg backend failure")           # error
    world.p.beh[OA] = RuntimeError("HTTP 429 rate limit exceeded")   # quota
    world.p.beh[EPMC] = []                                           # completed, zero results
    world.store()
    body = world.ask(providers=[OA])
    got = {canon(p["portal"]): p["status"] for p in _need_portals(body)}
    assert (got["ddg"], got["openalex"], got["epmc"]) == ("error", "quota", "completed"), got
    assert _status(body, "europepmc")["n_results"] == 0


# ============================================================================================================ 9
def test_9_same_request_grounded_answer_from_newly_admitted_evidence(world):
    doi = "10.9999/route.i1"
    world.doc(doi, "PMC9100501")
    world.p.beh[DDG] = [ddg_row(f"https://journals.example.org/doi/{doi}/full")]
    world.p.beh[OA] = [oa_rec(doi, "W91501", "PMC9100501")]
    world.store()
    body = world.ask()
    assert world.p.log[0] == DDG, f"wrong order: {world.p.log}"
    _grounded(body)
    ia = body["contract"]["inline_acquisition"]
    assert ia["status"] == "ok" and ia["admitted"] == 1 and ia["persisted"] is True
    assert len(world.rig.store.plans) == 1 and len(world.rig.h.answer_calls) == 1  # answered in THIS request
    ids = {a["document_id"] for a in body["admitted_evidence"]}
    assert ids and all(i.startswith("acq-") for i in ids) and {c["document_id"] for c in body["citations"]} == ids
    ro = _ro(body)
    assert ro["kind"] == "answered_with_new" and ro["complete"] is True
    assert canon(_need_portals(body)[0]["portal"]) == "ddg"
