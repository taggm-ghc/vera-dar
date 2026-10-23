"""FRESH-C4 (part of FRESH-1023): end-to-end POST /ask acceptance of the freshness / external-verification policy.

Owner cases A-J through main:app (TestClient, in process). OFFLINE: same stubs as tests/test_darb_http_e2e.py (scope router,
provider search seam under the REAL retrieve_all, Europe PMC / JATS web, answer LLM, claim checker, in-memory stores, DB /
socket guards). Stored chunks are the acquired corpus (fulldoc_db.load_acquired_rows) whose provenance carries the
publication and ingest times we control. Time: ONE clock hook `vera.freshness.utcnow` is monkeypatched to NOW; config key
`freshness_max_age_hours` = 24 is set on the rig's private config copy.

FRESH-AMEND-1023 (owner clarification 2026-10-23): the freshness TRIGGER = age of corpus_added_at (when the chunk was added to
the corpus: here provenance.retrieved_at written by the corpus writer, spec key `acquired`); the older/recent LABEL =
source_authored_at (provenance published, spec key `published`). A missing / invalid corpus_added_at triggers research and is
revised to the request's now() (vera.corpus_stamp).
Contract fields (additive): body["triggers"] (list of 'insufficient'|'stale'|'freshness_unknown'), body["freshness"]
({any_stale_or_unknown: bool, counts}), body["research_outcome"] ({kind, user_text, complete, portals[], candidates[]}).
Existing fields keep their behaviour; accept_predicate is used wherever a grounded answer is expected.
Mutation guard: disabling the insufficiency trigger fails C/D/I2/J; disabling the freshness trigger fails A-vs-B/E/H;
using the authoring date as the trigger fails B (added 30 h ago, published 1 h ago), E and F (published years ago, added
today); labelling recency by corpus_added_at fails F; dropping the backfill fails E2.
"""
import copy
import json
import re
import socket
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from tests import accept_predicate as ap
from tests import acq_support as A
from tests import test_fulldoc_e2e as e2e
from tests._gap_support import REAL_LOAD_CONFIG, http_ask
from tests._pipeline_support import h, rig  # noqa: F401  (fixtures)
from vera import acq_config, ask_service, auto_acquire, fulldoc_corpus, fulldoc_db

NOW = datetime(2026, 10, 23, 12, 0, 0, tzinfo=timezone.utc)
Q_CLIM = "How do temperature forcing and ocean circulation relate to warming and aerosol projection?"
OLD_DOI, NEW_DOI, NEW2_DOI, NEW3_DOI = "10.9999/old.epi1", "10.9999/acq.epi4", "10.9999/acq.epi5", "10.9999/acq.epi6"
HOURS = lambda h: NOW - timedelta(hours=h)  # noqa: E731
YEARS_AGO = datetime(2021, 6, 1, tzinfo=timezone.utc)
NEW_STAMPS = {"firstPublicationDate": (NOW - timedelta(hours=1)).date().isoformat(), "firstIndexDate": HOURS(1).isoformat(),
              "dateOfCreation": HOURS(1).isoformat(), "pubYear": str(NOW.year)}


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
    """Single clock hook. Tolerant of the feature being absent so each test fails on the missing BEHAVIOUR, not on setup;
    test_clock_hook_and_config_key_exist pins the hook itself."""
    try:
        import vera.freshness as fr
    except ImportError:
        return None
    monkeypatch.setattr(fr, "utcnow", lambda: NOW, raising=False)
    return fr


def test_clock_hook_and_config_key_exist(clock):
    assert clock is not None, "feature missing: module vera.freshness does not exist"
    assert callable(getattr(clock, "utcnow", None)), "feature missing: vera.freshness.utcnow"
    assert clock.utcnow() == NOW


# ----------------------------------------------------------------------------------------------------- the world
def _iso(t):
    return t.isoformat(timespec="seconds") if isinstance(t, datetime) else t


class World:
    """rig + stored acquired corpus + one HTTP /ask. `stored` specs: dict(doi, pmcid, topic, published, acquired)."""

    def __init__(self, rig, monkeypatch, tmp_path):
        self.rig, self.mp, self.tmp = rig, monkeypatch, tmp_path
        rig.cfg["freshness_max_age_hours"] = 24
        if isinstance(rig.cfg.get("freshness"), dict):
            rig.cfg["freshness"]["max_age_hours"] = 24
        rig.cfg["scope_expansion"]["override_layer2_declines"] = True
        self.plans, self.searches, self.answer_hook = [], [], None
        self.web = A.epi_setup(doi=NEW_DOI)[0]
        self.web.records[NEW_DOI].update(NEW_STAMPS)
        self.cands = [A.cand(NEW_DOI)]

    # -- stored chunks ------------------------------------------------------------------------------------------
    def store(self, *specs):
        plans = []
        for i, s in enumerate(specs, start=1):
            topic = s.get("topic", "epi")
            web, _ = A.epi_setup(doi=s["doi"], pmcid=s.get("pmcid", f"PMC90000{i}"), topic=topic,
                                 jats_kw={"seed": s.get("seed", 100 + i)})
            st = A.Store()
            q = e2e.Q_EPI if topic == "epi" else Q_CLIM
            auto_acquire.run_job(A.job(q), A.cfg(), A.deps([A.cand(s["doi"])], web, store=st), clock=A.clock_at(), env=A.ENV_ON)
            assert st.plans, f"fixture: stored document {s['doi']} was not admitted"
            p = st.plans[0]
            prov = p["source"]["provenance"]
            pub = s.get("published")
            # Publication time is item["published_year"] -> provenance["published"] (re-derived and verified by the loader, so both
            # are set consistently); ingest time is item["fetched_at"] / provenance["retrieved_at"] / acquisition.acquired_at.
            pv = _iso(pub)
            prov["published"] = prov["item"]["published_year"] = pv
            acq_t = _iso(s["acquired"])
            prov["item"]["fetched_at"] = prov["fetched_at"] = prov["retrieved_at"] = prov["acquisition"]["acquired_at"] = acq_t
            plans.append(p)
        self.plans = plans
        src, spans = A.rows_from_plans(plans) if plans else ([], [])
        corpus = fulldoc_db.load_acquired_rows(src, spans, fulldoc_corpus.load_config(), A.cfg()) if plans else None
        if plans:
            assert len(corpus.acquired) == len(plans), f"fixture: stored corpus unusable: {corpus.unusable}"
            for s, a in zip(specs, sorted(corpus.acquired, key=lambda a: [p["source"]["provenance"]["document_id"]
                                                                          for p in plans].index(a["document_id"]))):
                if "stamp" in s:  # a missing / invalid corpus_added_at in the served corpus (no per-chunk stamps)
                    corpus.stamps[a["document_id"]] = {"corpus_added_at": s["stamp"], "corpus_stamps": []}
        self.corpus = corpus
        self.mp.setattr(fulldoc_corpus, "get_acquired_corpus", lambda cfg=None, env=None, now=None: corpus)
        return self

    # -- the request ---------------------------------------------------------------------------------------------
    def ask(self, q=e2e.Q_EPI, cands=None, web=None):
        rig = self.rig
        rig.setup(web=web or self.web, cands=self.cands if cands is None else cands)
        if self.answer_hook:
            self.answer_hook()
        resp, searches = http_ask(rig, self.mp, [], q=q, offtopic=False)
        self.searches = searches
        assert resp.status_code == 200, resp.text
        return resp.json()

    # -- observations ----------------------------------------------------------------------------------------------
    @property
    def external(self):
        """Every sign of external search/acquisition in this request."""
        r = self.rig
        return {"discovery_searches": len(self.searches), "acq_searches": len(r.search_calls), "acquire_calls": len(r.outcomes),
                "http_urls": len(r.web.urls), "plans_written": len(r.store.plans)}

    def triggered(self):
        e = self.external
        return e["acquire_calls"] >= 1 and (e["discovery_searches"] + e["acq_searches"]) >= 1


@pytest.fixture
def world(rig, monkeypatch, tmp_path, clock, no_db_no_socket):  # noqa: F811
    monkeypatch.delenv("VERA_AUTOACQ_DISABLED", raising=False)
    w = World(rig, monkeypatch, tmp_path)
    yield w
    assert no_db_no_socket == []


def _need(body, key):
    assert key in body, f"feature missing: response has no {key!r} (keys: {sorted(body)})"
    return body[key]


def _doc_ids(body):
    return {c["document_id"] for c in body.get("citations") or []}


def _grounded(body):
    f = ap.grounded_failures(200, body)
    assert f == [], f


def _fresh_doc(doi, hours=2, **kw):
    return dict(doi=doi, published=HOURS(hours), acquired=HOURS(hours), **kw)


# ============================================================================================================ A
def test_A_sufficient_and_fresh_no_external_search(world):
    world.store(_fresh_doc(OLD_DOI, 2))
    body = world.ask()
    assert _need(body, "triggers") == []
    fr = _need(body, "freshness")
    assert fr["any_stale_or_unknown"] is False
    assert world.external == {"discovery_searches": 0, "acq_searches": 0, "acquire_calls": 0, "http_urls": 0, "plans_written": 0}
    _grounded(body)
    assert _doc_ids(body) == {a["document_id"] for a in body["admitted_evidence"]} and all(d.startswith("acq-") for d in _doc_ids(body))
    assert "discovered" not in ((body.get("contract") or {}).get("inline_acquisition") or {})
    ro = body.get("research_outcome")
    assert ro is None or ro.get("kind") != "answered_with_new"


# ============================================================================================================ B
def test_B_sufficient_but_stale_triggers_discovery_and_acquisition(world):
    # FRESH-AMEND-1023: ADDED to the corpus 30 h ago, published 1 h ago: authoring-time-as-freshness would call this fresh
    world.store(dict(doi=OLD_DOI, published=HOURS(1), acquired=HOURS(30)))
    body = world.ask()
    assert _need(body, "triggers") == ["stale"]
    assert _need(body, "freshness")["any_stale_or_unknown"] is True
    assert world.triggered(), world.external
    assert world.external["discovery_searches"] >= 1 and world.external["http_urls"] >= 1
    assert body["contract"]["inline_acquisition"]["status"] == "ok"


# ============================================================================================================ C
def test_C_insufficient_but_fresh_triggers(world):
    world.store(_fresh_doc("10.9999/old.clim1", 2, topic="clim"))  # fresh chunks, but none answers the epidemiology question
    body = world.ask()
    assert _need(body, "triggers") == ["insufficient"]
    assert _need(body, "freshness")["any_stale_or_unknown"] is False
    assert world.triggered(), world.external


# ============================================================================================================ D
def test_D_insufficient_and_stale_triggers_both_reasons(world):
    world.store(dict(doi="10.9999/old.clim1", topic="clim", published=HOURS(1), acquired=HOURS(72)))
    body = world.ask()
    # FRESH-1023 integration adaptation: the stale stored chunk (added 72 h ago) is about another topic (no candidate
    # passage for this question), so it is not "materially relevant" evidence and adds no freshness trigger;
    # insufficiency alone triggers (tightened back to == per FRESH-VERIFY-1023). Both-trigger behaviour on relevant
    # stale evidence is covered by tests/test_fresh_1023_integration.py.
    assert _need(body, "triggers") == ["insufficient"]
    assert world.triggered(), world.external


# ============================================================================================================ E
E_STAMPS = [("missing", None), ("year_only", "2026"), ("year_month", "2026-09"), ("date_only", "2026-10-23"),
            ("garbled", "31/31/2026 ??"), ("future", "2027-01-01T00:00:00+00:00")]


@pytest.mark.parametrize("kind,stamp", E_STAMPS)
def test_E_missing_or_untrustworthy_timestamp_triggers_verification(world, kind, stamp):
    # FRESH-AMEND-1023: the untrustworthy value is now corpus_added_at (the served corpus's write-time record); the
    # publication date is a valid recent one, so authoring time cannot explain the trigger.
    world.store(dict(doi=OLD_DOI, published=HOURS(2), acquired=HOURS(2), stamp=stamp))  # otherwise sufficient
    body = world.ask()
    assert _need(body, "triggers") == ["freshness_unknown"], kind  # not 'stale', not 'insufficient': the stamp is unknown
    fr = _need(body, "freshness")
    assert fr["any_stale_or_unknown"] is True
    assert world.triggered(), world.external


@pytest.mark.parametrize("kind,stamp", E_STAMPS)
def test_E2_invalid_corpus_stamp_is_revised_to_now_and_next_request_is_fresh(world, kind, stamp):
    """Owner refinement: missing / invalid / imprecise corpus_added_at -> trigger AND revised to the request's now()."""
    world.store(dict(doi=OLD_DOI, published=HOURS(2), acquired=HOURS(2), stamp=stamp))
    did = world.corpus.acquired[0]["document_id"]
    body = world.ask()
    assert _need(body, "triggers") == ["freshness_unknown"], kind
    bf = _need(body, "freshness").get("corpus_added_backfill")
    assert bf and bf["count"] >= 1 and bf["at"] == NOW.isoformat(), bf
    assert bf["persist"]["status"] == "not_needed" and bf["persist"]["not_persistable"] >= 1  # in-memory chunk: no DB row
    rec = world.corpus.stamps[did]["corpus_stamps"]
    assert rec and all(x["corpus_added_at"] == NOW.isoformat() for x in rec)  # the served corpus itself, in place
    body2 = world.ask()  # within 24 h: the repaired stamp is valid, so no freshness trigger for that chunk
    assert "freshness_unknown" not in _need(body2, "triggers") and "stale" not in body2["triggers"], body2["triggers"]
    assert not (_need(body2, "freshness").get("corpus_added_backfill"))


def test_E3_valid_corpus_stamp_is_never_overwritten(world):
    t = HOURS(30).isoformat(timespec="seconds")
    world.store(dict(doi=OLD_DOI, published=HOURS(2), acquired=HOURS(30)))
    did = world.corpus.acquired[0]["document_id"]
    body = world.ask()
    assert _need(body, "triggers") == ["stale"]
    assert not _need(body, "freshness").get("corpus_added_backfill")
    assert all(x["corpus_added_at"] == t for x in world.corpus.stamps[did]["corpus_stamps"])


# ============================================================================================================ F
def test_F_recently_ingested_old_publication_is_old(world):
    # FRESH-AMEND-1023: added to the corpus 30 min ago (no freshness trigger), published years ago -> labelled OLDER and
    # never called recent; both dates are exposed per source and are distinct.
    world.store(dict(doi=OLD_DOI, published=YEARS_AGO, acquired=HOURS(0.5)))  # acquired today, published years ago
    body = world.ask()
    assert _need(body, "triggers") == []
    assert _need(body, "freshness")["any_stale_or_unknown"] is False
    assert not world.triggered() and world.external["plans_written"] == 0, world.external
    _grounded(body)
    assert body["sources"]
    for s in body["sources"]:
        assert s["evidence_recency"] == "older", s
        assert s["source_authored_at"].startswith(YEARS_AGO.date().isoformat()) and s["source_authored_precision"] == "day"
        assert s["corpus_added_at"] == HOURS(0.5).isoformat(timespec="seconds")
        assert s["corpus_added_at"] != s["source_authored_at"]
    doc = _need(body, "freshness")["documents"][0]
    assert doc["freshness"] == "fresh" and doc["evidence_recency"] == "older"
    assert not re.search(r"\brecent\b", body["answer"], re.I) or "older" in body["answer"].lower()


# ============================================================================================================ G
def test_G_newly_acquired_licensed_chunks_admitted_and_cited_in_same_request(world):
    world.store()  # nothing stored: insufficient
    body = world.ask()
    assert "insufficient" in _need(body, "triggers")
    _grounded(body)
    ia = body["contract"]["inline_acquisition"]
    assert ia["status"] == "ok" and ia["admitted"] == 1 and ia["persisted"] is True
    assert len(world.rig.store.plans) == 1 and len(world.rig.h.answer_calls) == 1  # answered in THIS request
    new_ids = {a["document_id"] for a in body["admitted_evidence"]}
    assert len(new_ids) == 1 and all(i.startswith("acq-") for i in new_ids) and _doc_ids(body) == new_ids
    for s in body["sources"]:
        assert s["licence_decision"] == "allow" and s["source_type"] == "full_text_passage" and s["acquisition"] == "full_document_acquired"
    ro = _need(body, "research_outcome")
    assert ro["kind"] == "answered_with_new" and ro["complete"] is True and ro["user_text"]
    assert any(c["terminal_reason"] for c in ro["candidates"]) and ro["portals"]
    # FRESH-AMEND-1023: the corpus writer stamps corpus_added_at at write time on the (fake) freshness clock, and the
    # acquisition/added time is never used as the authoring date (Europe PMC pubYear stays a year-precision date).
    prov = world.rig.store.plans[0]["source"]["provenance"]
    assert prov["retrieved_at"] == NOW.isoformat(timespec="seconds")
    for s in body["sources"]:
        assert s["corpus_added_at"] == NOW.isoformat(timespec="seconds"), s
        assert s["source_authored_at"] == str(NOW.year) and s["source_authored_precision"] == "year", s
        assert s["evidence_recency"] == "unknown"  # a year-only date is never 'recent' (FRESH-VERIFY-1023 F1)


# ============================================================================================================ H
def _per_doc_answer(world):
    """Answer writer that cites one passage of EACH distinct document and says which is older / recent."""
    h = world.rig.h
    from vera import claim_check

    def complete(messages, model, pricing, **kw):
        h.answer_calls.append(messages)
        ps = json.loads(messages[-1]["content"])["passages"]
        by_url = {}
        for p in ps:
            by_url.setdefault(p["url"], p)
        parts = [f"Report {i}: transmission related to incidence in an outbreak [{p['n']}]." for i, p in enumerate(by_url.values(), 1)]
        return ask_service.AskResult(" ".join(parts), 10, 0.0)
    world.mp.setattr(ask_service, "complete", complete)


def test_H_new_evidence_contradicts_old_both_kept_and_marked(world):
    world.store(dict(doi=OLD_DOI, published=YEARS_AGO, acquired=HOURS(30), seed=3))  # sufficient but added 30 h ago
    world.answer_hook = lambda: _per_doc_answer(world)
    body = world.ask()
    assert _need(body, "triggers") == ["stale"]
    ids = _doc_ids(body)
    adm = {a["document_id"] for a in body["admitted_evidence"]}
    assert len(ids) == 2 and ids == adm, (ids, adm)  # the older and the newly acquired document are both cited
    stored_new = {p["source"]["provenance"].get("doi") or p["canonical_url"] for p in world.rig.store.plans}
    assert len(world.rig.store.plans) == 1 and NEW_DOI in json.dumps(sorted(stored_new))  # FRESH-1023 integration: a set is not JSON-serialisable (test bug)
    # the response marks which evidence is older and which is recent (per source), and represents the contradiction
    marks = {s["span_id"].split(":")[0]: json.dumps({k: v for k, v in s.items() if "fresh" in k.lower() or "recen" in k.lower()
                                                     or "older" in k.lower() or k == "age"}, sort_keys=True) for s in body["sources"]}
    assert len(set(marks.values())) == 2 and all(v != "{}" for v in marks.values()), f"feature missing: no older/recent marker per source: {marks}"
    ro = _need(body, "research_outcome")
    assert ro["kind"] == "answered_with_new"
    assert re.search(r"contradict|conflict|disagree|differ", json.dumps(ro) + body["answer"], re.I), "contradiction not represented"
    assert not (body.get("contract") or {}).get("fallback")


# ============================================================================================================ I
def test_I1_no_eligible_full_text_older_evidence_only_is_qualified(world):
    world.store(dict(doi=OLD_DOI, published=YEARS_AGO, acquired=HOURS(30)))
    world.web.records[NEW_DOI]["isOpenAccess"] = "N"  # candidate has no eligible (open licensed) full text
    body = world.ask()
    assert _need(body, "triggers") == ["stale"]
    ro = _need(body, "research_outcome")
    assert ro["kind"] == "qualified_older_only" and ro["user_text"] and ro["complete"] is True
    assert ro["candidates"] and all(c["terminal_reason"] for c in ro["candidates"])
    assert world.rig.store.plans == []  # nothing new admitted
    _grounded(body)  # grounded partial answer from the older stored chunks
    assert _doc_ids(body) and all(i.startswith("acq-") for i in _doc_ids(body))
    assert ro["user_text"] in body["answer"] or ro["user_text"] == body.get("research_notice")


def test_I2_no_eligible_full_text_and_nothing_stored_abstains(world):
    world.store()
    world.web.records[NEW_DOI]["isOpenAccess"] = "N"
    body = world.ask()
    assert "insufficient" in _need(body, "triggers")
    ro = _need(body, "research_outcome")
    assert ro["kind"] in ("abstain", "no_valuable_source") and ro["user_text"] and ro["complete"] is True
    assert body["result_kind"] != "answer" and not body.get("admitted_evidence") and not body.get("citations")
    assert world.rig.h.answer_calls == []  # no fabricated answer: the writer never ran
    assert ro["candidates"] and all(c["terminal_reason"] for c in ro["candidates"])


def test_I3_budget_exhausted_is_explicit_incomplete_search(world):
    import time

    class Slow(A.Web):
        def __call__(self, *a, **k):
            time.sleep(0.4)
            return super().__call__(*a, **k)
    world.store()
    slow = Slow(world.web.records, world.web.jats)
    world.rig.cfg["inline"]["deadline_s"] = 0.05
    body = world.ask(web=slow)
    ro = _need(body, "research_outcome")
    assert ro["kind"] == "incomplete_search" and ro["complete"] is False and ro["user_text"]
    assert body["result_kind"] != "answer" and world.rig.h.answer_calls == [] and world.rig.store.plans == []


# ============================================================================================================ J
def test_J_multiple_candidates_have_independent_terminal_outcomes(world):
    world.store()
    web = world.web
    web.records[NEW2_DOI] = A.epmc_record(NEW2_DOI, pmcid="PMC7654321", oa="N")  # not open access -> no eligible full text
    # NEW3_DOI is absent from Europe PMC -> unresolvable
    cands = [A.cand(NEW_DOI), A.cand(NEW2_DOI), A.cand(NEW3_DOI)]
    body = world.ask(cands=cands, web=web)
    ro = _need(body, "research_outcome")
    cs = ro["candidates"]
    assert len(cs) == 3 and len({c["candidate_id"] for c in cs}) == 3
    by = {d: next(c for c in cs if d in json.dumps(c["ids"])) for d in (NEW_DOI, NEW2_DOI, NEW3_DOI)}
    assert all(c["terminal_reason"] for c in cs)
    reasons = {d: c["terminal_reason"] for d, c in by.items()}
    # FRESH-1023 integration adaptation: NEW2 (not open access) and NEW3 (no Europe PMC record) both legitimately end
    # with no supported route; independence = one record per candidate with its own terminal reason, the admitted one
    # unaffected by the others' failures.
    assert reasons[NEW_DOI] == "admitted" and reasons[NEW2_DOI] != "admitted" and reasons[NEW3_DOI] != "admitted", reasons
    assert ro["kind"] == "answered_with_new"  # one candidate's failure did not stop the others
    _grounded(body)
    assert len(world.rig.store.plans) == 1 and NEW_DOI in json.dumps(world.rig.store.plans[0]["source"]["provenance"])
