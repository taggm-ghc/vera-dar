"""DAR-INLINE-1021-R2: the run-2 regression (re-ask in a new process re-acquired lower-ranked documents instead of
answering from the already-stored ones) and tests for the VER L1-L6 fixes. Offline: synthetic fixtures only."""
import json

import pytest

from tests import acq_support as A
from tests.test_dar_inline_acquire import OFF_Q, direct, rig  # noqa: F401  (fixtures)
from tests.test_acq_serve import h  # noqa: F401  (fixture)
from vera import acq_config, ask_service, auto_acquire, fulldoc_corpus, inline_acquire
from vera.dar_corpus import doc_key_set

DOI = "10.9999/acq.epi4"


def _rows_from_plan(plan, source_id=11, run_id=7):
    """The DB rows the WoStore would have written for `plan` (what fulldoc_db.fetch_acquired_rows returns)."""
    src = {"run_id": run_id, "source_id": source_id, "canonical_url": plan["canonical_url"],
           "content": plan["source"]["content"], "content_hash": plan["source"]["content_hash"],
           "provenance": json.loads(json.dumps(plan["source"]["provenance"]))}
    spans = [{"source_id": source_id, "start_index": s["start_index"], "end_index": s["end_index"], "text": s["text"]}
             for s in plan["spans"]]
    return [src], spans


def _first_run_plan(rig):
    r = rig.run()
    assert r.result_kind == "answer" and len(rig.store.plans) == 1
    return rig.store.plans[0]


def _new_process(rig, plan, served_keys=()):
    """A NEW process: the served acquired corpus was not loaded when the stored attempt ran (None), but may finish
    loading (served_keys) before the inline step ranks; the document is in the DB (known) and its rows are readable."""
    inline_acquire.reset_for_tests()  # no outcome cache across processes
    calls = []
    lookup = A.Lookup(known={DOI})
    store = A.Store()
    rig.setup(store=store, lookup=lookup)
    rig.deps.corpus_keys = lambda: set(served_keys)

    def stored_rows(cus):
        calls.append(list(cus))
        s, sp = _rows_from_plan(plan)
        return [x for x in s if x["canonical_url"] in cus], sp
    rig.deps.stored_rows = stored_rows
    return store, calls


@pytest.mark.parametrize("ready_meanwhile", [False, True])
def test_reask_in_new_process_answers_from_stored_documents_without_reacquiring(rig, ready_meanwhile):
    plan = _first_run_plan(rig)
    keys = doc_key_set(DOI, f"https://doi.org/{DOI}", None) if ready_meanwhile else ()
    store, calls = _new_process(rig, plan, served_keys=keys)
    n_urls = len(rig.web.urls)
    r = rig.run()
    assert r.result_kind == "answer" and r.evidence.admitted, r.answer
    ia = r.contract["inline_acquisition"]
    assert ia["reused"] == 1 and ia["admitted"] == 0 and ia["stored"] == 0
    assert store.plans == [] and len(rig.web.urls) == n_urls  # no write, no resolve/fetch: nothing re-acquired
    assert calls and plan["canonical_url"] in calls[0]
    assert rig.cfg["scope_expansion"]["inline_answer_note_stored"] in r.answer


def test_documents_the_stored_attempt_tried_are_still_skipped(rig, monkeypatch):
    plan = _first_run_plan(rig)
    store, calls = _new_process(rig, plan)
    o = inline_acquire.acquire(OFF_Q, rig.cfg, env=A.ENV_ON, deps_factory=lambda a, e: rig.deps,
                               served=doc_key_set(DOI, f"https://doi.org/{DOI}", None))
    assert o.reused == 0 and o.status == "none" and store.plans == []


def test_reuse_lookup_failure_never_reacquires_a_known_document(rig):
    plan = _first_run_plan(rig)
    store, _ = _new_process(rig, plan)

    def boom(cus):
        raise RuntimeError("ro down")
    rig.deps.stored_rows = boom
    r = rig.run()
    assert store.plans == [] and r.result_kind == "reading_list"


def test_ephemeral_ids_never_collide_with_reused_row_ids(rig):
    plan = _first_run_plan(rig)
    s, sp = _rows_from_plan(plan, source_id=1, run_id=1)  # real ids equal to what synthetic ids used to be
    other = json.loads(json.dumps(plan))
    other["canonical_url"] = "https://doi.org/10.9999/other"
    c = inline_acquire.ephemeral_corpus([other], rig.cfg, auto_acquire._fd_cfg(),
                                        acq_config.ingest_cfg(__import__("vera.fulltext_loader").fulltext_loader.load_config(), rig.cfg),
                                        stored=(s, sp))
    assert c is not None and len(c.acquired) == 1  # same text under two URLs: the reader keeps exactly one


# ------------------------------------------------------------------------------------------------ VER L1-L6
def test_L1_cap_refusal_is_not_cached(direct):
    cfg = A.cfg()
    d = A.deps([A.cand(DOI)], A.epi_setup()[0])
    cfg["bounds"]["jobs_per_hour"] = 0
    assert inline_acquire.acquire(OFF_Q, cfg, env=A.ENV_ON, deps_factory=lambda a, e: d).status == "rate"
    cfg["bounds"]["jobs_per_hour"] = 100
    assert inline_acquire.acquire(OFF_Q, cfg, env=A.ENV_ON, deps_factory=lambda a, e: d).status == "ok"


def test_L2_deadline_scope_is_documented_honestly():
    note = acq_config.load_config()["inline"]["_calibration_note"]
    assert "search through persist" not in note and "persist" in note
    assert "bounds search through persist" not in inline_acquire.__doc__


def test_L3_admitted_but_unusable_corpus_is_unanswered_not_none(rig, monkeypatch):
    monkeypatch.setattr(inline_acquire, "ephemeral_corpus", lambda *a, **k: None)
    r = rig.run()
    assert rig.outcomes[0].status == "unanswered"
    assert rig.cfg["scope_expansion"]["inline_notice_unanswered"] in r.answer
    assert rig.cfg["scope_expansion"]["inline_notice_none"] not in r.answer


def test_L4_partial_persist_failure_is_not_reported_unsaved(direct):
    cfg = A.cfg()
    cfg["bounds"]["max_docs_per_job"] = 2
    w1, _ = A.epi_setup(doi="10.9999/acq.epi4", pmcid="PMC1234567")
    w2, _ = A.epi_setup(doi="10.9999/acq.epi5", pmcid="PMC7654321", topic="epi")
    web = A.Web({**w1.records, **w2.records}, {**w1.jats, **w2.jats})

    class Half(A.Store):
        def write(self, plan):
            if not self.plans:
                self.plans.append(None)
                raise RuntimeError("db hiccup")
            return super().write(plan)
    d = A.deps([A.cand("10.9999/acq.epi4"), A.cand("10.9999/acq.epi5")], web, store=Half())
    o = inline_acquire.acquire(OFF_Q, cfg, env=A.ENV_ON, deps_factory=lambda a, e: d)
    assert o.admitted == 2 and o.stored == 1 and o.persist_failed is False
    assert inline_acquire.answer_note(cfg, o) == cfg["scope_expansion"]["inline_answer_note"]


def test_L6_no_answer_path_means_no_inline_search(rig, monkeypatch):
    monkeypatch.setattr(fulldoc_corpus, "flag_enabled", lambda cfg: False)
    rig.run()
    assert rig.search_calls == [] and rig.outcomes == []


def test_L6_search_bound_zero_means_no_search(direct):
    cfg = A.cfg()
    cfg["bounds"]["max_search_calls_per_job"] = 0
    calls = []
    d = A.deps([A.cand(DOI)], A.epi_setup()[0])
    d.search = lambda q, n, t: calls.append(q) or []
    assert inline_acquire.acquire(OFF_Q, cfg, env=A.ENV_ON, deps_factory=lambda a, e: d).status == "off"
    assert calls == []


def test_L5_concurrent_jobs_cannot_overshoot_docs_per_day(direct):
    cfg = A.cfg()
    cfg["bounds"]["docs_per_day"] = 3
    cfg["bounds"]["max_docs_per_job"] = 2
    d = A.deps([], A.epi_setup()[0])
    a = auto_acquire.docs_budget(cfg, d, reserve=True)
    b = auto_acquire.docs_budget(cfg, d, reserve=True)  # a concurrent second job sees the first one's reservation
    assert (a, b) == (2, 1) and auto_acquire.docs_budget(cfg, d) == 0
    auto_acquire.settle_docs(a, 1)
    auto_acquire.settle_docs(b, 0)
    assert auto_acquire._S.docs_reserved == 0 and auto_acquire.docs_budget(cfg, d) == 2


def test_defect1_reuse_still_answers_when_docs_per_day_is_spent(rig):
    plan = _first_run_plan(rig)
    store, _ = _new_process(rig, plan)
    rig.deps.lookup.day = 10 ** 6  # docs_per_day spent
    n_urls = len(rig.web.urls)
    r = rig.run()
    assert r.result_kind == "answer" and r.contract["inline_acquisition"]["reused"] == 1
    assert store.plans == [] and len(rig.web.urls) == n_urls


# ------------------------------------------------------------------------------------------------ must-haves 1 and 2
def test_must1_in_scope_insufficient_question_is_answered_inline_with_in_scope_wording(rig):
    r = rig.run(accept=True, layer="layer2_llm", reason="llm_in:x", category="in_scope")
    assert r.result_kind == "answer" and r.evidence.admitted, r.answer
    se = rig.cfg["scope_expansion"]
    assert not r.answer.startswith(se["override_answer_prefix"]) and se["override_answer_prefix"] not in r.answer
    assert r.contract["inline_acquisition"]["stored"] == 1 and "offtopic" not in r.contract
    assert len(rig.h.check_calls) == 1  # the claim check ran


def test_must1_in_scope_flag_off_keeps_background_trigger_only(rig):
    rig.cfg["inline"]["in_scope"] = False
    r = rig.run(accept=True, layer="layer2_llm", reason="llm_in:x", category="in_scope")
    assert rig.search_calls == [] and r.result_kind != "answer"


def test_must1_layer1_dirty_in_scope_question_never_acquires(rig, monkeypatch):
    monkeypatch.setattr(ask_service, "_layer1_clean", lambda q: False)
    rig.run(accept=True, layer="layer2_llm", reason="llm_in:x", category="in_scope")
    assert rig.search_calls == [] and rig.outcomes == []


def test_must2_base_warmup_never_blocks_and_inline_still_answers(rig, monkeypatch):
    monkeypatch.setattr(ask_service, "_base_warming", lambda: True)

    def blocked(*a, **k):
        raise AssertionError("waited on the base corpus during warm-up")
    monkeypatch.setattr(fulldoc_corpus, "get_corpus", blocked)
    r = rig.run(accept=True, layer="layer2_llm", reason="llm_in:x", category="in_scope")
    assert r.result_kind == "answer" and r.contract["inline_acquisition"]["stored"] == 1
