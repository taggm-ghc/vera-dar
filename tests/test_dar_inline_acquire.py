"""DAR-INLINE-1021: inline open-access acquisition inside one /ask request (off-topic override path only).
Offline: classifier, search, web, store, answer model and claim checker are all stubbed with synthetic fixtures
(tests/acq_support.py, tests/test_fulldoc_e2e.Harness); no network, DB or LLM."""
import copy
import threading
import time

import pytest
from fastapi.testclient import TestClient

from tests import acq_support as A
from tests import test_fulldoc_e2e as e2e
from tests.test_acq_serve import BASE_NO_EPI, _decide, _serve, h  # noqa: F401  (h is a fixture)
from vera import acq_config, ask_service, auto_acquire, fulldoc_answer, inline_acquire, scope_router


@pytest.fixture(autouse=True)
def _legacy_decline_path(monkeypatch):
    """DDFIX-1023/SCOPE: an off_topic layer-2 verdict now takes the normal research path (tests/test_ddfix_scope_1023.py
    covers it). This module keeps covering the decline-path machinery, which other allowed layer-2 out-of-scope
    verdicts (e.g. category in_scope with in_scope=false) still reach, so the reroute is switched off here."""
    monkeypatch.setattr(ask_service, "_research_eligible", lambda *a, **k: False)

REAL = inline_acquire.acquire  # the shared fixture h stubs it to "off"; these tests restore the real one
OFF_Q = e2e.Q_EPI
GROUND = (None, {**e2e.GROUNDING, "no_sources_text": "NO"})
LIST_SRC = [{"n": 1, "title": "Paper One", "url": "https://arxiv.org/abs/2401.00001", "published": "2025-01-01",
             "snippet": "S", "identifier": "arXiv:2401.00001", "provider": "arxiv", "retrieved_at": "2026-10-01"}]


class Search:
    def __init__(self):
        self.calls = []

    def __call__(self, restricted=False):
        self.calls.append(restricted)
        return LIST_SRC


class Rig:
    """One synthetic inline scenario; `run()` asks the declined question through ask_service.answer_in_scope."""

    def __init__(self, h, mp):
        self.h, self.mp = h, mp
        self.cfg = A.cfg()
        self.cfg["scope_expansion"]["override_layer2_declines"] = True
        mp.setattr(acq_config, "load_config", lambda *a, **k: self.cfg)
        mp.setattr(inline_acquire, "acquire", self._spy)
        self.outcomes, self.search_calls, self.store, self.deps = [], [], A.Store(), None
        h.bundle(docs=BASE_NO_EPI)
        h.install()
        _serve(mp, None)  # nothing stored: the stored-corpus step cannot answer, the inline step runs
        mp.setattr(inline_acquire, "default_deps", lambda acq, env: self.deps)
        self.setup()

    def _spy(self, q, acq, **kw):
        o = REAL(q, acq, **kw)
        self.outcomes.append(o)
        return o

    def setup(self, topic="epi", web=None, store=None, lookup=None, cands=None, search=None):
        w = web or A.epi_setup(topic=topic)[0]
        self.store = store or self.store
        d = A.deps(cands if cands is not None else [A.cand("10.9999/acq.epi4")], w, lookup=lookup, store=self.store)
        inner = search or d.search

        def counted(q, n, t):
            self.search_calls.append(q)
            return inner(q, n, t)
        d.search = counted
        self.deps, self.web = d, w
        return self

    def run(self, q=OFF_Q, layer="layer2_llm", reason="llm_out:x", category="off_topic", accept=False):
        _decide(self.mp, accept, layer, reason, category)
        return ask_service.answer_in_scope(q, "m", e2e.PRICING, grounding=(Search(), GROUND[1]))


@pytest.fixture
def rig(h, monkeypatch):
    return Rig(h, monkeypatch)


# ------------------------------------------------------------------------------------------------ 1. answer in request
def test_inline_answer_in_the_same_request(rig):
    r = rig.run()
    assert r.result_kind == "answer" and r.accepted is False and r.acquire_trigger is None
    assert r.evidence is not None and r.evidence.admitted
    assert all(a["document_id"].startswith("acq-") for a in r.evidence.admitted)
    se = rig.cfg["scope_expansion"]
    assert r.answer.startswith(se["override_answer_prefix"] + " " + se["inline_answer_note"])
    ia = r.contract["inline_acquisition"]
    assert ia["status"] == "ok" and ia["admitted"] == 1 and ia["stored"] == 1 and ia["persisted"] is True
    assert set(ia["ms"]) >= {"total"} and r.contract["offtopic"] is True and r.contract["acquired_only"] is True
    assert len(rig.store.plans) == 1 and len(rig.search_calls) == 1
    assert len(rig.h.answer_calls) == 1 and len(rig.h.check_calls) == 1  # the unchanged writer and claim check ran


# ------------------------------------------------------------------------------------------------ 2. deadline
def test_deadline_expiry_gives_reading_list_with_deadline_notice(rig):
    class Slow(A.Web):
        def __call__(self, *a, **k):
            time.sleep(0.4)
            return super().__call__(*a, **k)
    base = A.epi_setup()[0]
    rig.setup(web=Slow(base.records, base.jats))
    rig.cfg["inline"]["deadline_s"] = 0.05
    r = rig.run()
    assert r.result_kind == "reading_list" and r.evidence is None and r.acquire_trigger is None
    assert rig.cfg["scope_expansion"]["inline_notice_deadline"] in r.answer
    assert rig.outcomes[0].status == "deadline" and rig.store.plans == []
    assert rig.h.answer_calls == []


def test_search_failure_gives_search_failed_notice(rig):
    def boom(q, n, t):
        raise RuntimeError("synthetic")
    rig.setup(search=boom)
    r = rig.run()
    assert r.result_kind == "reading_list" and r.acquire_trigger is None
    assert rig.cfg["scope_expansion"]["inline_notice_search_failed"] in r.answer
    assert rig.outcomes[0].status == "search_failed"


# ------------------------------------------------------------------------------------------------ 3. nothing admitted
def test_nothing_admitted_gives_none_notice_and_no_background_job(rig):
    rig.setup(topic="clim")  # a licensed document on an unrelated topic: the relevance gate rejects it
    r = rig.run()
    assert r.result_kind == "reading_list" and r.evidence is None and r.acquire_trigger is None
    assert rig.cfg["scope_expansion"]["inline_notice_none"] in r.answer
    o = rig.outcomes[0]
    assert o.status == "none" and o.corpus is None and rig.store.plans == []
    assert rig.h.answer_calls == []


# ------------------------------------------------------------------------------------------------ 4. persistence
class _Res:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class _Nested:
    def commit(self):
        pass

    def rollback(self):
        pass


class _Conn:
    def __init__(self, log):
        self.log, self.n = log, 0

    def execute(self, stmt, params=None):
        self.log.append(str(stmt))
        self.n += 1
        return _Res((100 + self.n,))

    def begin_nested(self):
        return _Nested()


class _Engine:
    def __init__(self, log):
        self.log = log

    def begin(self):
        conn = _Conn(self.log)

        class Ctx:
            def __enter__(s):
                return conn

            def __exit__(s, *a):
                return False
        return Ctx()


class _Look:
    def canonical_state(self, cu, sha):
        return (None, 0, False, False)


def test_persistence_is_insert_only_through_the_wo_store(rig):
    sql = []
    store = auto_acquire.WoStore(_Engine(sql), None, rig.cfg)
    store.lookup = _Look()
    rig.setup(store=store)
    r = rig.run()
    assert r.result_kind == "answer" and r.contract["inline_acquisition"]["stored"] == 1
    assert sql, "no SQL issued: the plan was not persisted through WoStore"
    for s in sql:
        head = s.strip().upper()
        assert head.startswith("INSERT INTO") or head.startswith("SELECT"), s
        for bad in ("UPDATE ", "DELETE ", "TRUNCATE", "ALTER ", "DROP ", "CREATE "):
            assert bad not in head, s
    assert sum(s.strip().upper().startswith("INSERT INTO") for s in sql) >= 5  # question, run, candidate, canonical, source, spans


def test_persist_failure_still_answers_with_unsaved_note(rig):
    class Broken(A.Store):
        def write(self, plan):
            raise RuntimeError("db down")
    rig.setup(store=Broken())
    r = rig.run()
    se = rig.cfg["scope_expansion"]
    assert r.result_kind == "answer" and r.evidence.admitted
    ia = r.contract["inline_acquisition"]
    assert ia["status"] == "ok" and ia["persisted"] is False and ia["stored"] == 0 and ia["admitted"] == 1
    assert se["inline_answer_note_unsaved"] in r.answer and se["inline_answer_note"] + " " not in r.answer.replace(
        se["inline_answer_note_unsaved"], "")
    assert len(rig.h.check_calls) == 1  # the claim check still ran


# ------------------------------------------------------------------------------------------------ 5. caps and kill switch
def test_hourly_job_cap_means_no_search(rig):
    rig.cfg["bounds"]["jobs_per_hour"] = 0
    r = rig.run()
    assert rig.search_calls == [] and rig.outcomes[0].status == "rate"
    # DAR-TRACE-1022: the background queue shares the cap, so no trigger and the capped notice (not "search started")
    assert r.result_kind == "reading_list" and r.acquire_trigger is None
    assert rig.cfg["scope_expansion"]["inline_notice_none"] not in r.answer
    assert rig.cfg["scope_expansion"]["inline_notice_capped"] in r.answer


def test_daily_document_cap_means_nothing_new_is_fetched_or_stored(rig):
    # R2 defect 1: the search still runs (stored documents may be reused), but nothing is resolved, fetched or stored
    rig.cfg["bounds"]["docs_per_day"] = 20  # DAR-TRACE-1022: shipped null (no cap); a configured cap still works
    rig.setup(lookup=A.Lookup(day=10 ** 6))
    r = rig.run()
    assert len(rig.search_calls) == 1 and rig.web.urls == [] and rig.store.plans == []
    assert rig.outcomes[0].status == "daily_cap"
    assert r.result_kind == "reading_list" and r.acquire_trigger is None  # DAR-TRACE-1022: doc-cap notice instead
    assert rig.cfg["scope_expansion"]["inline_notice_doc_cap"] in r.answer


def test_null_daily_document_cap_means_a_huge_db_count_does_not_stop_inline_acquisition(rig):
    assert rig.cfg["bounds"]["docs_per_day"] is None  # owner 2026-10-22: abolished
    rig.setup(lookup=A.Lookup(day=10 ** 6))
    r = rig.run()
    assert rig.outcomes[0].status == "ok" and r.result_kind == "answer"


def test_kill_switch_means_no_search_no_inline(rig, monkeypatch):
    monkeypatch.setenv("VERA_AUTOACQ_DISABLED", "1")
    r = rig.run()
    assert rig.search_calls == [] and rig.outcomes == []
    assert r.result_kind == "reading_list" and r.acquire_trigger is None
    assert inline_acquire.acquire(OFF_Q, rig.cfg, env={**A.ENV_ON, "VERA_AUTOACQ_DISABLED": "1"},
                                  deps_factory=lambda a, e: rig.deps).status == "off"
    assert rig.search_calls == []


def test_inline_flag_off_restores_background_only(rig):
    rig.cfg["inline"]["enabled"] = False
    r = rig.run()
    assert rig.search_calls == [] and r.result_kind == "reading_list" and r.acquire_trigger == "decline"


# ------------------------------------------------------------------------------------------------ 6. concurrency, dedupe
@pytest.fixture
def direct(monkeypatch):
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()
    monkeypatch.setattr(acq_config, "db_configured", lambda cfg, env: True)
    yield
    auto_acquire.reset_for_tests()
    inline_acquire.reset_for_tests()


def _blocking_deps(started, release, calls):
    web = A.epi_setup()[0]
    d = A.deps([A.cand("10.9999/acq.epi4")], web)

    def search(q, n, t):
        calls.append(q)
        started.set()
        assert release.wait(20)
        return [A.cand("10.9999/acq.epi4")]
    d.search = search
    return d


def test_max_concurrent_returns_busy_at_once_without_search(direct):
    cfg = A.cfg()
    cfg["inline"] = {**cfg["inline"], "max_concurrent": 1}
    started, release, calls = threading.Event(), threading.Event(), []
    d = _blocking_deps(started, release, calls)
    res = {}
    t = threading.Thread(target=lambda: res.setdefault("a", inline_acquire.acquire(
        "first question about outbreak transmission", cfg, env=A.ENV_ON, deps_factory=lambda a, e: d)))
    t.start()
    try:
        assert started.wait(10)
        t0 = time.monotonic()
        o = inline_acquire.acquire("a different question about outbreak incidence", cfg, env=A.ENV_ON,
                                   deps_factory=lambda a, e: d)
        assert o.status == "busy" and time.monotonic() - t0 < 2 and len(calls) == 1
    finally:
        release.set()
        t.join(30)
    assert res["a"].status in ("ok", "none")


def test_identical_in_flight_questions_search_once_and_share_the_outcome(direct):
    cfg = A.cfg()
    started, release, calls = threading.Event(), threading.Event(), []
    d = _blocking_deps(started, release, calls)
    res = {}

    def go(name):
        res[name] = inline_acquire.acquire(OFF_Q, cfg, env=A.ENV_ON, deps_factory=lambda a, e: d)
    t1 = threading.Thread(target=go, args=("a",))
    t1.start()
    assert started.wait(10)
    t2 = threading.Thread(target=go, args=("b",))
    t2.start()
    key = (auto_acquire._qkey(OFF_Q), None)
    end = time.monotonic() + 10
    while time.monotonic() < end:  # the second thread must be registered as a waiter before the first finishes
        with inline_acquire._I.lock:
            slot = inline_acquire._I.inflight.get(key)
            if slot and slot[2] == 1:
                break
        time.sleep(0.01)
    else:
        release.set()
        pytest.fail("second identical question never became a waiter")
    release.set()
    t1.join(30)
    t2.join(30)
    assert len(calls) == 1
    assert res["a"] is res["b"] and res["a"].status == "ok"
    assert len(d.store.plans) == 1


def test_finished_outcome_is_reused_within_ttl(direct):
    cfg = A.cfg()
    calls = []
    d = A.deps([A.cand("10.9999/acq.epi4")], A.epi_setup()[0])
    inner = d.search
    d.search = lambda q, n, t: calls.append(q) or inner(q, n, t)
    a = inline_acquire.acquire(OFF_Q, cfg, env=A.ENV_ON, deps_factory=lambda x, e: d)
    b = inline_acquire.acquire(OFF_Q, cfg, env=A.ENV_ON, deps_factory=lambda x, e: d)
    assert a.status == "ok" and b is a and len(calls) == 1


# ------------------------------------------------------------------------------------------------ 7/8. never on other paths
def test_in_scope_request_never_calls_inline(h, monkeypatch):
    calls = []
    monkeypatch.setattr(inline_acquire, "acquire", lambda *a, **k: calls.append(a) or REAL(*a, **k))
    # FRESH-1023: a stale or undated corpus hit now triggers research by design (owner P0); this guard keeps its
    # original meaning with the freshness step off. Fresh hits: tests/test_fresh_acceptance_c4.py::test_A_*.
    monkeypatch.setattr(ask_service, "_fresh_cfg", lambda acq: None)
    h.bundle()
    h.install()
    _decide(monkeypatch, True, "layer1", "ok")
    r = ask_service.answer_in_scope(e2e.Q_EPI, "m", e2e.PRICING, grounding=(Search(), GROUND[1]))
    assert r.result_kind == "answer" and calls == []


@pytest.mark.parametrize("layer,reason,cat", [
    ("layer1", "deny:medical", "off_topic"),
    ("layer1", "deny:injection", "off_topic"),
    ("layer1", "empty", "off_topic"),
    ("layer2_llm", "llm_out:x", "harmful"),
    ("layer2_llm", "llm_out:x", "advice"),
    ("layer2_llm", "llm_out:x", "programming_task"),
    ("layer2_llm", "llm_out:x", "rule_change"),
    ("layer2_llm", "llm_out:x", ""),
    ("layer2_llm", "classifier_malformed", "off_topic"),
    ("layer2_none", "unsure:decline", ""),
])
def test_other_declines_never_call_inline(rig, layer, reason, cat):
    r = rig.run(layer=layer, reason=reason, category=cat)
    assert rig.outcomes == [] and rig.search_calls == [] and r.acquire_trigger is None and r.evidence is None
    assert "inline_acquisition" not in (r.contract or {})


def test_override_off_never_calls_inline(rig):
    rig.cfg["scope_expansion"]["override_layer2_declines"] = False
    r = rig.run()
    assert rig.outcomes == [] and rig.search_calls == [] and r.result_kind == "reading_list"
    assert r.acquire_trigger == "decline"


# ------------------------------------------------------------------------------------------------ 9. claim check corpus
def test_answer_is_checked_against_the_ephemeral_corpus(rig, monkeypatch):
    seen = []
    real = fulldoc_answer.answer

    def spy(*a, **k):
        seen.append(k.get("reference_corpus"))
        return real(*a, **k)
    monkeypatch.setattr(fulldoc_answer, "answer", spy)
    r = rig.run()
    assert r.result_kind == "answer"
    corpus = rig.outcomes[0].corpus
    assert corpus is not None and len(seen) == 1 and seen[0] is corpus
    assert corpus.acquired and corpus.status == "ok"


# ------------------------------------------------------------------------------------------------ 10. background job
def test_background_job_still_stores_after_the_stage_refactor():
    auto_acquire.reset_for_tests()
    web, _ = A.epi_setup()
    store = A.Store()
    out = auto_acquire.run_job(A.job(e2e.Q_EPI), A.cfg(), A.deps([A.cand("10.9999/acq.epi4")], web, store=store),
                               clock=A.clock_at(), env=A.ENV_ON)
    assert out["stored"] == 1 and len(store.plans) == 1
    assert store.plans[0]["canonical_url"] and store.plans[0]["spans"]
    auto_acquire.reset_for_tests()


def test_background_job_shares_reserved_job_state_with_inline(direct):
    cfg = A.cfg()
    cfg["bounds"]["jobs_per_hour"] = 1
    d = A.deps([A.cand("10.9999/acq.epi4")], A.epi_setup()[0])
    a = inline_acquire.acquire(OFF_Q, cfg, env=A.ENV_ON, deps_factory=lambda x, e: d)
    b = inline_acquire.acquire("another question about outbreak incidence", cfg, env=A.ENV_ON,
                               deps_factory=lambda x, e: d)
    assert a.status == "ok" and b.status == "rate"  # the hourly job cap is shared


# ------------------------------------------------------------------------------------------------ 11. main.py refresh
def _main_ask(h, monkeypatch, rig_setup=None):
    main = h.install()
    refreshed, acquired = [], []
    monkeypatch.setattr(inline_acquire, "refresh_after_response", lambda: refreshed.append(1))
    monkeypatch.setattr(main, "_acquire", lambda q, t: acquired.append((q, t)))
    _decide(monkeypatch, False, "layer2_llm", "llm_out:x", "off_topic")
    r = TestClient(main.app).post("/ask", json={"question": OFF_Q}, headers={"X-API-Key": "k" * 20})
    assert r.status_code == 200, r.text
    return r.json(), refreshed, acquired


def test_main_schedules_background_refresh_when_documents_were_stored(rig, monkeypatch):
    body, refreshed, acquired = _main_ask(rig.h, monkeypatch)
    assert body["result_kind"] == "answer" and body["contract"]["inline_acquisition"]["stored"] == 1
    assert refreshed == [1] and acquired == []  # inline searched: no second background job


def test_main_does_not_refresh_when_nothing_was_stored(rig, monkeypatch):
    rig.setup(topic="clim")
    body, refreshed, acquired = _main_ask(rig.h, monkeypatch)
    assert body["result_kind"] == "reading_list" and refreshed == [] and acquired == []
