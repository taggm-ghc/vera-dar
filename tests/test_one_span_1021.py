"""ONE-SPAN-1021 (owner approval 2026-10-21): DAR single-passage answer eligibility.

A fetched document with >= bounds.answer_min_admitted_spans (1) qualifying passages may answer THIS request through the
ephemeral corpus; it is persisted only with >= bounds.persist_min_admitted_spans (2). The relevance gate itself runs
unchanged (real admission over the synthetic document); the tests cap its count to model 0 / 1 / 2 qualifying passages.
Offline: classifier, search, web, store, answer model and claim checker are stubbed (tests/acq_support.py,
tests/test_fulldoc_e2e.Harness); no network, DB or LLM."""
import logging

import pytest

from tests import acq_support as A
from tests import test_fulldoc_e2e as e2e
from tests.test_acq_serve import h  # noqa: F401  (h is a fixture)
from tests.test_dar_inline_acquire import Rig
from vera import acq_config, auto_acquire, fulldoc_corpus, inline_acquire

REAL_GATE = auto_acquire.relevance_gate


def _cap(mp, n):
    """The real relevance gate runs (and must pass >= n); its count is capped at n qualifying passages."""
    mp.setattr(auto_acquire, "relevance_gate", lambda *a, **k: min(REAL_GATE(*a, **k), n))


@pytest.fixture
def rig(h, monkeypatch):
    return Rig(h, monkeypatch)


# ------------------------------------------------------------------------------------------------ config split
def test_config_splits_answer_and_persist_minimums():
    cfg = acq_config.load_config()
    b = cfg["bounds"]
    assert b["answer_min_admitted_spans"] == 1 and b["persist_min_admitted_spans"] == 2
    assert "min_admitted_spans" not in b
    assert acq_config.answer_min_spans(cfg) == 1 and acq_config.persist_min_spans(cfg) == 2


def test_code_floors_zero_never_answers_and_persist_never_below_two():
    assert acq_config.answer_min_spans({"bounds": {"answer_min_admitted_spans": 0, "persist_min_admitted_spans": 2}}) == 1
    assert acq_config.persist_min_spans({"bounds": {"persist_min_admitted_spans": 1}}) == 2
    assert acq_config.persist_min_spans({"bounds": {"min_admitted_spans": 3}}) == 3  # legacy key still read


# ------------------------------------------------------------------------------------------------ 1 passage
def test_one_passage_document_answers_in_the_same_request_but_is_not_persisted(rig, monkeypatch):
    _cap(monkeypatch, 1)
    r = rig.run()
    assert r.result_kind == "answer" and r.evidence is not None and r.evidence.admitted
    assert all(a["document_id"].startswith("acq-") for a in r.evidence.admitted)
    assert rig.store.plans == []  # never written to the DB
    ia = r.contract["inline_acquisition"]
    assert ia["n_answer_eligible"] == 1 and ia["n_persisted"] == 0 and ia["stored"] == 0
    assert ia["documents"] == [{"persisted": False, "reason": "below_persist_min"}]
    se = rig.cfg["scope_expansion"]
    assert se["inline_answer_note_answer_only"] in r.answer and se["inline_answer_note"] not in r.answer
    assert len(rig.h.answer_calls) == 1 and len(rig.h.check_calls) == 1  # unchanged writer and claim check ran


def test_one_passage_answer_schedules_no_refresh(rig, monkeypatch):
    from tests.test_dar_inline_acquire import _main_ask
    _cap(monkeypatch, 1)
    body, refreshed, acquired = _main_ask(rig.h, monkeypatch)
    assert body["result_kind"] == "answer" and body["contract"]["inline_acquisition"]["n_persisted"] == 0
    assert body["contract"]["inline_acquisition"]["n_answer_eligible"] == 1
    assert refreshed == [] and acquired == []


# ------------------------------------------------------------------------------------------------ 2 passages
def test_two_passage_document_answers_and_is_persisted(rig, monkeypatch):
    _cap(monkeypatch, 2)
    r = rig.run()
    assert r.result_kind == "answer" and len(rig.store.plans) == 1
    assert "admitted_spans" not in rig.store.plans[0]["source"]["provenance"]  # gate count is never stored
    ia = r.contract["inline_acquisition"]
    assert ia["n_answer_eligible"] == 1 and ia["n_persisted"] == 1 and ia["documents"] == [{"persisted": True}]
    assert rig.cfg["scope_expansion"]["inline_answer_note"] in r.answer


def test_mixed_one_and_two_passage_documents(rig, monkeypatch):
    seen = []

    def gate(q, plan, *a, **k):
        seen.append(plan["canonical_url"])
        return min(REAL_GATE(q, plan, *a, **k), 1 if "epi5" in plan["canonical_url"] else 2)
    monkeypatch.setattr(auto_acquire, "relevance_gate", gate)
    w1, raw1 = A.epi_setup(doi="10.9999/acq.epi4", pmcid="PMC1234567")
    w2, raw2 = A.epi_setup(doi="10.9999/acq.epi5", pmcid="PMC7654321", jats_kw={"seed": 3})
    web = A.Web({**w1.records, **w2.records}, {**w1.jats, **w2.jats})
    rig.setup(web=web, cands=[A.cand("10.9999/acq.epi4"), A.cand("10.9999/acq.epi5")])
    r = rig.run()
    ia = r.contract["inline_acquisition"]
    assert r.result_kind == "answer" and ia["n_answer_eligible"] == 2 and ia["n_persisted"] == 1
    assert sorted(d["persisted"] for d in ia["documents"]) == [False, True]
    assert len(rig.store.plans) == 1 and "epi4" in rig.store.plans[0]["canonical_url"]
    assert rig.cfg["scope_expansion"]["inline_answer_note_mixed"] in r.answer


# ------------------------------------------------------------------------------------------------ 0 passages
def test_zero_qualifying_passages_gives_reading_list(rig, monkeypatch):
    _cap(monkeypatch, 0)
    r = rig.run()
    assert r.result_kind == "reading_list" and r.evidence is None
    assert rig.cfg["scope_expansion"]["inline_notice_none"] in r.answer
    o = rig.outcomes[0]
    assert o.status == "none" and o.corpus is None and o.n_answer_eligible == 0 and rig.store.plans == []
    assert rig.h.answer_calls == []


# ------------------------------------------------------------------------------------------------ claim check, partial
def _marker():
    return fulldoc_corpus.load_config()["answer"]["insufficient_marker"]


def test_claim_check_still_removes_unsupported_sentences(rig, monkeypatch):
    _cap(monkeypatch, 1)
    bad = "Vaccination halved the serial interval in every hospital [1]."
    rig.h.answer_text = "Transmission was compared under the stated conditions [1]. " + bad
    rig.h.verdicts = {1: "unsupported"}
    r = rig.run()
    assert r.result_kind == "answer" and bad not in r.answer
    assert "Transmission was compared" in r.answer and rig.store.plans == []
    assert any(c.get("verdict") == "unsupported" for c in r.claims)


def test_all_sentences_unsupported_gives_no_answer(rig, monkeypatch):
    _cap(monkeypatch, 1)
    rig.h.verdicts = {0: "unsupported"}
    r = rig.run()
    assert r.result_kind == "reading_list" and rig.cfg["scope_expansion"]["inline_notice_unanswered"] in r.answer


def test_one_passage_answer_covering_part_of_the_question_is_marked_partial(rig, monkeypatch):
    _cap(monkeypatch, 1)
    rig.h.answer_text = ("Parts:\nP1: transmission and incidence\nP2: vaccination cost\nAnswers:\n"
                         "P1: Transmission was compared under the stated conditions [1].\nP2: " + _marker())
    r = rig.run()
    cc = r.claim_check
    assert r.result_kind == "answer" and cc["answer_completeness"] == "partial"
    assert cc["parts"]["answered"] == [1] and cc["parts"]["unanswered"] == [2]
    pp = fulldoc_corpus.load_config()["answer"]["per_part"]
    assert pp["partial_statement"] in r.answer  # qualified, never presented as complete; the gap is not filled


def test_unsupported_part_is_removed_and_answer_marked_partial(rig, monkeypatch):
    _cap(monkeypatch, 1)
    rig.h.answer_text = ("Parts:\nP1: transmission and incidence\nP2: vaccination cost\nAnswers:\n"
                         "P1: Transmission was compared under the stated conditions [1].\n"
                         "P2: Vaccination cost fell by half in every hospital [1].")
    rig.h.verdicts = {1: "unsupported"}
    r = rig.run()
    assert r.claim_check["answer_completeness"] == "partial" and "Vaccination cost fell" not in r.answer


def test_fully_covered_one_passage_answer_is_complete(rig, monkeypatch):
    _cap(monkeypatch, 1)
    r = rig.run()
    assert r.claim_check["answer_completeness"] == "complete"


# ------------------------------------------------------------------------------------------------ telemetry
def _lines(caplog, event):
    return [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith(f"autoacq {event} ")]


def test_inline_log_line_carries_both_counts_and_no_question_text(rig, monkeypatch, caplog):
    _cap(monkeypatch, 1)
    caplog.set_level(logging.INFO, logger="vera.autoacq")
    rig.run()
    lines = _lines(caplog, "inline")
    assert len(lines) == 1
    ln = lines[0]
    for k in ('"n_answer_eligible": 1', '"n_persisted": 0', '"n_below_persist_min": 1', '"n_ACQ-ANSWER-ONLY": 1'):
        assert k in ln, ln
    for word in ("transmission", "pathogen", "outbreak", "incidence"):
        assert word not in ln.lower()


def test_background_job_persists_only_at_two_or_more(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="vera.autoacq")
    for n, want in ((1, 0), (2, 1)):
        auto_acquire.reset_for_tests()
        _cap(monkeypatch, n)
        web, _ = A.epi_setup()
        store = A.Store()
        out = auto_acquire.run_job(A.job(e2e.Q_EPI), A.cfg(), A.deps([A.cand("10.9999/acq.epi4")], web, store=store),
                                   clock=A.clock_at(), env=A.ENV_ON)
        assert out["stored"] == want and len(store.plans) == want
        assert out["n_answer_eligible"] == 1 and out["n_persisted"] == want
        if n == 1:
            assert out["codes"] == {"R-ACQ-FEWSPANS": 1}
    lines = _lines(caplog, "job")
    assert len(lines) == 2 and '"n_answer_eligible": 1' in lines[0] and '"n_persisted": 0' in lines[0]
    assert '"n_persisted": 1' in lines[1]
    assert all(A.SENTINEL not in ln for ln in lines)
    auto_acquire.reset_for_tests()


def test_direct_acquire_outcome_counts(monkeypatch):
    monkeypatch.setattr(acq_config, "db_configured", lambda cfg, env: True)
    inline_acquire.reset_for_tests()
    auto_acquire.reset_for_tests()
    _cap(monkeypatch, 1)
    store = A.Store()
    d = A.deps([A.cand("10.9999/acq.epi4")], A.epi_setup()[0], store=store)
    o = inline_acquire.acquire(A.Q_EPI, A.cfg(), env=A.ENV_ON, deps_factory=lambda x, e: d)
    assert o.status == "ok" and o.corpus is not None and o.n_answer_eligible == 1 and o.n_persisted == 0
    assert store.plans == [] and o.codes.get("ACQ-ANSWER-ONLY") == 1
    inline_acquire.reset_for_tests()
    auto_acquire.reset_for_tests()


# ------------------------------------------------------------------------------------------------ note truthfulness
@pytest.mark.parametrize("docs,reused,stored,failed,key", [
    ([{"persisted": False, "reason": "below_persist_min"}], 0, 0, False, "inline_answer_note_answer_only"),
    ([{"persisted": False, "reason": "below_persist_min"}, {"persisted": False, "reason": "in_corpus"}], 0, 0, False,
     "inline_answer_note_mixed"),
    ([{"persisted": False, "reason": "below_persist_min"}, {"persisted": False, "reason": "write_failed"}], 0, 0, True,
     "inline_answer_note_mixed"),
    ([{"persisted": False, "reason": "below_persist_min"}], 1, 0, False, "inline_answer_note_mixed"),
    ([{"persisted": True}, {"persisted": False, "reason": "below_persist_min"}], 0, 1, False, "inline_answer_note_mixed"),
    ([{"persisted": False, "reason": "write_failed"}], 0, 0, True, "inline_answer_note_unsaved"),
    ([{"persisted": True}], 0, 1, False, "inline_answer_note"),
])
def test_answer_note_is_truthful_for_every_mix(docs, reused, stored, failed, key):
    cfg = acq_config.load_config()
    o = inline_acquire.Outcome("ok", admitted=len(docs), stored=stored, persist_failed=failed, reused=reused,
                               documents=docs)
    assert inline_acquire.answer_note(cfg, o) == cfg["scope_expansion"][key]
