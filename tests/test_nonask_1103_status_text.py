"""NONASK-1103 WP-C: count plurals, truthful no-answer causes, acquisition statuses, E5 contract, contract merge.
Offline; no network or DB."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vera import acq_config, ask_service, fulldoc_corpus, inline_acquire, research_outcome, ui_evidence as ue
from vera.ask_service import AskResult

BANNED = re.compile(r"(?i)\b(gate|stage|model|provider|claim check|checker|writer|openai|groq|token)\b")
CODES = {"no_passages", "no_supported_answer", "verification_unavailable", "premise_unverified", "corpus_unusable",
         "reading_list"}


def _ev(n_docs=1, spans=0):
    adm = [{"document_id": f"d{i % max(n_docs, 1)}"} for i in range(spans)]
    return SimpleNamespace(examined_full_documents=[f"d{i}" for i in range(n_docs)], admitted=adm)


def test_plurals():
    t = fulldoc_corpus.load_config()["answer"]["insufficient_text"]
    assert "1 full document but" in ask_service._with_count(t, _ev(1))
    assert "9 full documents but" in ask_service._with_count(t, _ev(9))
    assert "(s)" not in ask_service._with_count(t, _ev(2))
    assert ue.fallback_text("insufficient_full_document_evidence", cfg=dict(fulldoc_corpus.load_config()["answer"], count_matching_documents=False), n_inspected=1).count(  # NOANSWER-RC-1107: legacy = flag off
        "1 full document ") == 1


def test_selected_text_never_says_found_no_passage():
    a = fulldoc_corpus.load_config()["answer"]
    for k in ("unsupported_text", "verification_unavailable_text"):
        assert "found no passage" not in a[k] and not BANNED.search(a[k]) and len(a[k]) <= 300
    txt = ask_service._with_count(a["unsupported_text"], _ev(5, spans=3), selected=True)
    assert "found no passage" not in txt
    assert "found no passage" in a["insufficient_text"]  # admitted 0 still says so
    assert "found no passage" not in ue.fallback_text("model_insufficient", n_inspected=2)


def test_with_cause_only_on_non_answers():
    ans = AskResult(answer="a", tokens_used=0, cost_usd=0.0, result_kind="answer")
    assert ask_service._with_cause(ans, "no_passages", "x") is ans
    r = AskResult(answer="n", tokens_used=0, cost_usd=0.0, result_kind="reading_list",
                  contract={"inline_acquisition": {"portals": [{"portal": "a", "status": "completed"},
                                                               {"portal": "b", "status": "failed"},
                                                               {"portal": "c", "status": "skipped"}]}})
    out = ask_service._with_cause(r, "no_supported_answer", "y" * 400)
    c = out.contract
    assert c["no_answer_cause"]["code"] in CODES and len(c["no_answer_cause"]["text"]) <= 300
    assert c["research"] == {"portals_attempted": 2, "portals_completed": 1}


def test_cause_text_banned_terms_and_codes():
    for code in ask_service._CAUSE_CODE.values():
        assert code in CODES
    a = fulldoc_corpus.load_config()["answer"]
    for k in ("insufficient_text", "unsupported_text", "verification_unavailable_text"):
        assert not BANNED.search(a[k])
    assert "no identifier VERA-DAR can fetch automatically yet" in research_outcome._CAUSE_TEXT["awaiting_route"]


def _o(stored=0, docs=(), admitted=0):
    return SimpleNamespace(stored=stored, documents=list(docs), admitted=admitted,
                           answer_only=sum(1 for d in docs if d.get("reason") == inline_acquire.BELOW_PERSIST_MIN))


def test_status_matrix():
    acq = json.loads(open(acq_config.CONFIG_PATH if hasattr(acq_config, "CONFIG_PATH") else "config/auto_acquire.json").read())
    f = inline_acquire._supply_status
    assert f(acq, {}, _o(docs=[{"persisted": False, "reason": inline_acquire.BELOW_PERSIST_MIN}], admitted=1)) == "relevance_rejected"
    assert f(acq, {"R-ACQ-IRRELEVANT": 2}, _o()) == "relevance_rejected"
    assert f(acq, {"R-ACQ-NODOI": 3, "R-ACQ-ARXIV_LICENCE_HOLD": 1}, _o()) == "no_route"
    assert f(acq, {}, _o()) == "resource_limited"
    assert f(acq, {"R-ACQ-NODOI": 1, "R-ACQ-IRRELEVANT": 1}, _o()) == "relevance_rejected"


def test_notice_lines():
    def d(s):
        return {"contract": {"inline_acquisition": {"acquisition": {"status": s, "stored": 0, "target": 6}}}}
    assert "enough relevant passages" in ue.acquisition_notice_line(d("relevance_rejected"))
    assert "fetch automatically" in ue.acquisition_notice_line(d("no_route"))
    assert "supply was limited" in ue.acquisition_notice_line(d("resource_limited"))
    assert "time limit" in ue.acquisition_notice_line(d("incomplete_deadline"))
    for s in ("relevance_rejected", "no_route"):
        assert not re.search(r"https?://|\.(org|com)\b", ue.acquisition_notice_line(d(s)))


def test_combined_answer_merges_contract(monkeypatch):
    from vera import fulldoc_answer
    ok = AskResult(answer="a [1]", tokens_used=1, cost_usd=0.0, result_kind="answer", contract={"trace": "T", "layers": "old"})
    monkeypatch.setattr("vera.fulldoc_admission.admit", lambda *a, **k: SimpleNamespace(admitted=[1]))
    monkeypatch.setattr(ask_service, "_merge_evidence", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(ask_service, "_base_warming", lambda: True)
    monkeypatch.setattr(fulldoc_corpus, "get_acquired_corpus", lambda cfg: None)
    monkeypatch.setattr(ask_service, "_reference_texts", lambda *a, **k: [])
    monkeypatch.setattr(ask_service, "_pre_llm", lambda **k: None)
    monkeypatch.setattr(fulldoc_answer, "answer", lambda *a, **k: SimpleNamespace(result=ok, fallback=None, tokens_used=1, cost_usd=0.0))
    monkeypatch.setattr(inline_acquire, "answer_note", lambda *a, **k: "")
    o = SimpleNamespace(corpus=object(), public=lambda: {"status": "ok"})
    fd = SimpleNamespace(evidence=None, tokens_used=0, cost_usd=0.0)
    res, _ = ask_service._combined_answer("q", "m", None, {}, None, {}, fd, o)
    assert res.contract["trace"] == "T" and res.contract["fallback"] is None
    assert res.contract["inline_acquisition"] == {"status": "ok"}


def test_fulldoc_answer_site_merges_contract():
    src = open(ask_service.__file__, encoding="utf-8").read()
    assert '**(out.result.contract or {}), "layers": contract.layers(), "fallback": None})' in src
