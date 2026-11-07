"""NOANSWER-RC-1107 (r2, owner 2026-11-07): non-answer text counts only documents with passages matching THIS question,
over the base and stored acquired corpora; never the size of a set searched regardless of subject (the pinned 9, the
acquired 30). Offline; no DB."""
import re
from types import SimpleNamespace

from vera import ask_service, fulldoc_corpus
from vera import ui_evidence as u

TOK = "insufficient_full_document_evidence"


def _rows(cands, pre="d"):
    return [{"document_id": f"{pre}{i}", "canonical_source_id": f"{pre}c{i}", "candidate_spans": c, "admitted_spans": 0}
            for i, c in enumerate(cands)]


def _ev(cands, pre="d", admitted=()):
    return SimpleNamespace(examined_full_documents=tuple(_rows(cands, pre)), admitted=tuple(admitted))


def _cfg():
    return fulldoc_corpus.load_config()["answer"]


def _data(cands, matched=None, adm=(), fallback=TOK, fetched=0):
    d = {"evidence_status": "insufficient", "examined_full_documents": _rows(cands), "corpus_source": {"kind": "db"},
         "admitted_evidence": [{"document_id": f"d{i}", "text": "t"} for i in adm], "contract": {"fallback": fallback}}
    if matched is not None:
        d["contract"]["search_counts"] = {"matched": matched}
    if fetched:
        d["corpus_source"] = {"kind": "bundle"}
        d["contract"]["inline_acquisition"] = {"documents": [{"reason": "below_persist_min"}] * fetched}
    return d


class _St:
    def __init__(self):
        self.info_calls, self.captions = [], []

    def info(self, t):
        self.info_calls.append(t)

    def caption(self, t):
        self.captions.append(t)

    def __getattr__(self, name):
        return lambda *a, **k: None


def test_flag_ships_true_with_notes_and_texts():
    a = _cfg()
    assert a["count_matching_documents"] is True
    assert "Observed spread" in a["_count_matching_documents_calibration_note"]
    assert "no stored full documents on this subject yet" in a["insufficient_no_match_text"]
    assert "{n_stored_phrase}" in a["insufficient_matched_text"] and "{n_stored_phrase}" in a["unsupported_matched_text"]
    for k in ("insufficient_no_match_text", "insufficient_matched_text", "insufficient_nocount_text",
              "unsupported_matched_text"):
        assert "searched" not in a[k] and "inspected" not in a[k] and "checked" not in a[k]


def test_matched_documents_counts_distinct_docs_with_candidates():
    assert u.matched_documents(_rows([0] * 9)) == 0
    assert u.matched_documents(_rows([0, 3, 0, 1, 0, 0, 0, 0, 0])) == 2
    assert u.matched_documents([{"document_id": "x"}]) is None and u.matched_documents(None) is None


def test_search_counts_combines_base_and_acquired_and_never_carries_set_sizes():
    assert u.search_counts(_rows([0] * 9), _rows([0, 2] + [0] * 28, "a")) == {"matched": 1}
    assert u.search_counts(_rows([1] * 9), None) == {"matched": 9}
    assert u.search_counts(_rows([0] * 9), [{"document_id": "x"}]) is None  # inexact -> no count
    assert u.search_counts([], None) is None and u.search_counts(None, None) is None


def test_headline_counts_only_matching_documents():
    assert u.fallback_text(TOK, n_inspected=9, n_matched=0) == _cfg()["insufficient_no_match_text"]
    two = u.fallback_text(TOK, n_inspected=9, n_matched=2)
    assert two.startswith("2 stored full documents had passages related to this question; none answered it")
    assert u.fallback_text(TOK, n_inspected=9, n_matched=1).startswith("1 stored full document had passages")
    assert u.fallback_text(TOK, n_inspected=9) == _cfg()["insufficient_nocount_text"]  # unknown: no count, no 9
    assert u.fallback_text(TOK, n_inspected=9, n_fetched=1, n_matched=0) == _cfg()["insufficient_nocount_text"]


def test_old_wording_only_with_flag_off():
    off = dict(_cfg(), count_matching_documents=False)
    assert "searched 9 full documents" in u.fallback_text(TOK, cfg=off, n_inspected=9, n_matched=0)
    assert "checked 30 full documents" in u.fallback_text("model_insufficient", cfg=off, n_inspected=30, n_selected=2)
    assert u.fallback_text("model_insufficient", n_inspected=30, n_selected=2).startswith(
        "Passages from 2 stored full documents looked related")


def test_api_headline_counts_base_and_acquired():
    used = {"ev_base": _ev([0] * 9), "ev_acq": _ev([0, 0, 2] + [0] * 27, "a")}
    assert ask_service._search_counts(used) == {"matched": 1}
    assert ask_service._insufficient(_cfg(), used["ev_base"], used).startswith("1 stored full document had passages")
    none = {"ev_base": _ev([0] * 9), "ev_acq": _ev([0] * 30, "a")}
    assert ask_service._insufficient(_cfg(), none["ev_base"], none) == _cfg()["insufficient_no_match_text"]
    bad = {"ev_base": _ev([0] * 9), "ev_acq": SimpleNamespace(examined_full_documents=({"document_id": "x"},))}
    assert ask_service._insufficient(_cfg(), bad["ev_base"], bad) == _cfg()["insufficient_nocount_text"]
    legacy = SimpleNamespace(examined_full_documents=tuple({"document_id": f"d{i}"} for i in range(9)), admitted=())
    assert ask_service._insufficient(_cfg(), legacy) == _cfg()["insufficient_nocount_text"]  # never silent fallback
    off = dict(_cfg(), count_matching_documents=False)
    assert "searched 9 full documents" in ask_service._insufficient(off, none["ev_base"], none)


def test_api_unsupported_counts_only_passage_documents():
    ev = _ev([3, 3] + [0] * 28, admitted=({"document_id": "d0"}, {"document_id": "d1"}, {"document_id": "d1"}))
    t = ask_service._unsupported(_cfg(), "unsupported_text", ev)
    assert t.startswith("Passages from 2 stored full documents looked related") and "30" not in t


def test_scope_line_never_states_set_size_or_inspected():
    assert u.evidence_scope_line(_data([0] * 9, matched=0)) == (
        "Evidence scope: 0 stored documents with passages matching the question; no passages selected as possible "
        "evidence.")
    live = u.evidence_scope_line(_data([3, 3] + [0] * 28, matched=2, adm=(0, 0, 0, 1, 1, 1), fallback="model_insufficient"))
    assert live == ("Evidence scope: 2 stored documents with passages matching the question; 6 passages from 2 "
                    "documents selected as possible evidence.")
    nocount = _data([0] * 9)
    nocount["examined_full_documents"] = [{"document_id": f"d{i}"} for i in range(9)]
    assert u.evidence_scope_line(nocount) == "Evidence scope: no passages selected as possible evidence."
    ans = u.evidence_scope_line(_data([3, 3] + [0] * 28, adm=(0, 1), fallback=None))
    assert "inspected" not in ans and "30" not in ans


def test_render_fallback_reason_uses_the_matched_count():
    """Kills M6 (n_matched=None in render_fallback_reason): the rendered headline carries the matched count."""
    st = _St()
    u.render_fallback_reason(st, _data([0] * 9, matched=3))
    assert len(st.info_calls) == 1 and st.info_calls[0].startswith("3 stored full documents had passages")
    st = _St()
    u.render_fallback_reason(st, _data([0] * 9, matched=0))
    assert st.info_calls == [_cfg()["insufficient_no_match_text"]]


def test_no_set_size_anywhere_in_a_zero_match_non_answer():
    """Owner 2026-11-07: a question with zero matching passages never shows 9 (base set) or 30 (acquired set)."""
    used = {"ev_base": _ev([0] * 9), "ev_acq": _ev([0] * 30, "a")}
    head = ask_service._insufficient(_cfg(), used["ev_base"], used)
    data = _data([0] * 9, matched=ask_service._search_counts(used)["matched"])
    st = _St()
    u.render_fallback_reason(st, data)
    for t in [head, *st.info_calls, *st.captions, u.evidence_scope_line(data)]:
        assert not re.search(r"\b(9|30)\b", t), t
        assert "inspected" not in t and not re.search(r"searched \d", t), t
