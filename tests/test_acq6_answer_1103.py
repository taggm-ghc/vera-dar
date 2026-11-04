"""NONASK-1103 A6-4 (D3): honest split stored/fetched counts on the inline path; kill switch answer.inline_honest_counts.
Offline: everything below ask_service._inline is stubbed (as tests/test_inline_nonresponsive.py)."""
from types import SimpleNamespace

import pytest

from vera import ask_service, fulldoc_corpus, inline_acquire, ui_evidence

CFG = {"answer": {"inline_honest_counts": True,
                  "inline_honest_text": "VERA-DAR used {n_stored} stored and {n_fetched} fetched {n_word}; no answer."},
       "fallback_tokens": {"insufficient": "insufficient_full_document_evidence", "checker_unavailable": "checker_unavailable"}}


@pytest.fixture
def wired(monkeypatch):
    st = {"cfg": CFG, "fulldoc": []}
    dec = SimpleNamespace(accepted=True, layer="layer2_llm", reason="llm_in:x", tokens_used=3, cost_usd=0.0)
    monkeypatch.setattr(ask_service, "_layer1_clean", lambda q: True)
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda: st["cfg"])
    monkeypatch.setattr(inline_acquire, "notice", lambda acq, status: f"NOTICE:{status}")
    monkeypatch.setattr(inline_acquire, "acquire", lambda q, acq, served=None, **k: SimpleNamespace(
        corpus=object(), status="ok", stored=2, reused=1, admitted=5, public=lambda: {"status": "ok"}, queued=0))
    monkeypatch.setattr(inline_acquire, "lane_pass", lambda q, a: None)
    monkeypatch.setattr(inline_acquire, "background_note", lambda a, o: "")
    monkeypatch.setattr(inline_acquire, "answer_note", lambda a, o: "")
    monkeypatch.setattr(ask_service, "_acq_cfg_if_on", lambda: {})
    monkeypatch.setattr(fulldoc_corpus, "flag_enabled", lambda cfg: True)

    def run(fd):
        monkeypatch.setattr(ask_service, "_fulldoc", fd)
        return ask_service._inline("Q?", "m", SimpleNamespace(input=0, output=0), dec, None, None, {"inline": {}}, in_scope=True)
    st["run"] = run
    return st


def test_zero_passages_gives_honest_split_notice_no_writer(wired, monkeypatch):
    called = []
    monkeypatch.setattr("vera.fulldoc_answer.answer", lambda *a, **k: called.append(1))
    from vera import fulldoc_admission

    class Ev:
        status, admitted, examined_full_documents = "insufficient", (), ()
    monkeypatch.setattr(fulldoc_admission, "admit", lambda q, c, cfg: Ev())
    wired["cfg"] = {**CFG, "retrieval": {}, "admission": {}}
    out = wired["run"](ask_service.__dict__["_fulldoc"])
    assert out == "NOTICE:unanswered VERA-DAR used 3 stored and 3 fetched sources; no answer."
    assert not called and "tried" not in out.lower() and "memory" not in out.lower().replace("answered from memory", "")


def test_all_sentences_removed_gives_notice_with_counts_not_memory_answer(wired):
    out = wired["run"](lambda *a, **k: (None, "insufficient_full_document_evidence", 1, 0.0, "unsupported"))
    assert out == "NOTICE:unanswered VERA-DAR used 3 stored and 3 fetched sources; no answer." and "tried" not in out.lower()


def test_checker_unavailable_keeps_its_own_text(wired):
    out = wired["run"](lambda *a, **k: (None, "checker_unavailable", 1, 0.0, "x"))
    assert out == "NOTICE:unanswered"


def test_kill_switch_false_reproduces_pre_change(wired):
    wired["cfg"] = {**CFG, "answer": {**CFG["answer"], "inline_honest_counts": False}}
    out = wired["run"](lambda *a, **k: (None, "insufficient_full_document_evidence", 1, 0.0, "unsupported"))
    assert out == "NOTICE:unanswered"
    seen = {}
    wired["run"](lambda *a, **k: seen.update(c=ask_service._INLINE_COUNTS.get()))
    assert seen["c"] is None


def test_counts_context_does_not_leak(wired):
    wired["run"](lambda *a, **k: None)
    assert ask_service._INLINE_COUNTS.get() is None


def test_ui_counts_line_split():
    d = {"contract": {"inline_acquisition": {"counts": {"stored": 3, "fetched": 2}}}}
    assert ui_evidence.inline_counts_line(d) == "Sources used for this answer: 3 stored, 2 fetched for this request only."
    assert ui_evidence.inline_counts_line({}) is None


# ---- C6 (owner R-045): answer.always_attempt -- the writer runs with >= 1 admitted passage despite "insufficient" ----
def _ev6(n, status):
    from vera.fulldoc_evidence import EvidenceBundle
    docs = ({"document_id": "d0", "canonical_source_id": "c0", "title": "T0", "url": None, "admitted_spans": n},)
    adm = tuple({"span_id": f"s{i}", "document_id": "d0", "canonical_source_id": "c0", "title": "T0",
                 "text": "Synthetic passage.", "start": 0, "end": 18} for i in range(n))
    return EvidenceBundle(question="Q?", examined_full_documents=docs, admitted=adm, status=status)


@pytest.fixture
def fd6(monkeypatch):
    import copy
    from vera import fulldoc_admission, fulldoc_answer
    st = {"calls": [], "cfg": copy.deepcopy(fulldoc_corpus.load_config()), "ev": None, "result": None}
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda *a, **k: st["cfg"])
    monkeypatch.setattr(fulldoc_corpus, "flag_enabled", lambda *a, **k: True)
    monkeypatch.setattr(ask_service, "_pre_llm", lambda *a, **k: None)
    monkeypatch.setattr(fulldoc_admission, "admit", lambda *a, **k: st["ev"])

    def writer(question, ev, *a, **k):
        st["calls"].append(ev)
        return st["result"]
    monkeypatch.setattr(fulldoc_answer, "answer", writer)
    for name in ("answer_question", "complete"):  # never an answer from memory
        monkeypatch.setattr(ask_service, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("memory path")))

    def run():
        dec = SimpleNamespace(tokens_used=0, cost_usd=0.0, topic_key="")
        return ask_service._fulldoc("Q?", "m", None, dec, {}, None, acquired_only=True, ephemeral=object())
    st["run"] = run
    return st


def _answer6():
    from vera.fulldoc_answer import FullDocOutcome
    return FullDocOutcome(result=ask_service.AskResult(answer="Supported claim [1].", result_kind="answer", tokens_used=1, cost_usd=0.0), fallback=None)


def test_always_attempt_ships_true():
    assert fulldoc_corpus.load_config()["answer"]["always_attempt"] is True
    assert fulldoc_corpus.load_config()["answer"]["coverage_qualification_text"]


def test_insufficient_with_one_admitted_calls_writer_once_and_qualifies(fd6):
    fd6["ev"], fd6["result"] = _ev6(1, "insufficient"), _answer6()
    out = fd6["run"]()
    assert len(fd6["calls"]) == 1 and len(fd6["calls"][0].admitted) == 1 and fd6["calls"][0].grounded
    assert isinstance(out, ask_service.AskResult) and out.result_kind == "answer"
    shown = out.answer + " " + " ".join(n["text"] for n in out.notes)
    assert fd6["cfg"]["answer"]["coverage_qualification_text"] in shown


def test_admitted_status_gets_no_qualification(fd6):
    fd6["ev"], fd6["result"] = _ev6(1, "admitted"), _answer6()
    out = fd6["run"]()
    assert len(fd6["calls"]) == 1
    assert fd6["cfg"]["answer"]["coverage_qualification_text"] not in out.answer + str(out.notes)


def test_claim_check_removes_all_gives_notice_with_counts_no_memory(fd6):
    from vera.fulldoc_answer import FullDocOutcome
    fd6["ev"], fd6["result"] = _ev6(1, "insufficient"), FullDocOutcome(result=None, fallback="claim_check_insufficient")
    tok = ask_service._INLINE_COUNTS.set({"stored": 2, "fetched": 1})
    try:
        out = fd6["run"]()
    finally:
        ask_service._INLINE_COUNTS.reset(tok)
    assert len(fd6["calls"]) == 1
    assert isinstance(out, tuple) and out[1] == "claim_check_insufficient" and not isinstance(out, ask_service.AskResult)


def test_zero_admitted_never_calls_writer(fd6):
    fd6["ev"] = _ev6(0, "insufficient")
    out = fd6["run"]()
    assert fd6["calls"] == [] and not isinstance(out, ask_service.AskResult)


def test_kill_switch_false_is_previous_behaviour(fd6):
    fd6["cfg"]["answer"]["always_attempt"] = False
    fd6["ev"], fd6["result"] = _ev6(1, "insufficient"), _answer6()
    out = fd6["run"]()
    assert fd6["calls"] == [] and not isinstance(out, ask_service.AskResult)
