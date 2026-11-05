"""SCOPE-GATE-1104 S2 (plan Amendment 1, A1.3 S2-T1..T5; INTEGRATE-1104 steps 1-3). Offline stubs only.

Owner: "this should be a determination of whether the corpus can answer the prompt, or not. not whether a question is
in scope. even if it were in scope, the corpus may be insufficient". Every question that passes safety takes the normal
path; sufficiency is decided after retrieval; an insufficient corpus goes to acquisition (rounds until the target or
the request budget), and acquisition-off gives the normal reading list plus acquisition_off_note.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from vera import ask_service, fulldoc_corpus, inline_acquire
from vera import scope_router as sr
from vera.ask_service import AskResult

Q = "What is the evidence on sourdough fermentation and gut health?"
FT = fulldoc_corpus.load_config()["fallback_tokens"]
SAFE = sr.ScopeDecision(True, "layer2_llm", "llm_in:x", 5, 0.0, 1, sr.SAFE_CATEGORY)
LEGACY_OFF = sr.ScopeDecision(False, "layer2_llm", "llm_out:x", 5, 0.0, 1, "off_topic")
LEGACY_IN = sr.ScopeDecision(False, "layer2_llm", "llm_out:x", 5, 0.0, 1, "in_scope")


@pytest.fixture
def rig(monkeypatch):
    st = {"inline": [], "fulldoc": [], "rl": 0, "decision": SAFE, "acq": {"inline": {"in_scope": True, "deadline_s": 45}},
          "fd": (None, FT["insufficient"], 5, 0.0, "INSUFFICIENT"), "inline_ret": None, "outcomes": []}
    monkeypatch.setattr(sr, "route", lambda *a, **k: st["decision"])
    monkeypatch.setattr(ask_service, "_catstep", lambda d, *a, **k: d)
    monkeypatch.setattr(ask_service, "_acq_cfg_if_on", lambda: st["acq"])
    monkeypatch.setattr(ask_service, "_layer1_clean", lambda q: True)
    monkeypatch.setattr(ask_service, "_acq_on_insufficient", lambda a: False)
    monkeypatch.setattr(ask_service, "_fresh_cfg", lambda a: None)
    monkeypatch.setattr(ask_service, "_offtopic_reading_list",
                        lambda *a, **k: pytest.fail("restricted off-topic reading list used for a safe question"))

    def fulldoc(question, model, pricing, decision, chained, grounding, acquired_only=False, ephemeral=None, used=None,
                lanes=None):
        st["fulldoc"].append(decision)
        return st["fd"]
    monkeypatch.setattr(ask_service, "_fulldoc", fulldoc)

    def inline(question, model, pricing, decision, chained, grounding, acq, tried=None, in_scope=False, info=None):
        st["inline"].append({"acq": acq, "tried": set(tried or ()), "in_scope": in_scope})
        if info is not None and st["outcomes"]:
            out = st["outcomes"].pop(0)
            info["outcome"], info["tried_keys"] = out[0], out[1]
        return st["inline_ret"]
    monkeypatch.setattr(ask_service, "_inline", inline)

    def reading_list(question, model, pricing, decision, chained, grounding, contract, tokens=None, cost=None):
        st["rl"] += 1
        return AskResult(answer="RL", tokens_used=tokens or 0, cost_usd=cost or 0.0, result_kind="reading_list",
                         contract={"layers": ["L0"]})
    monkeypatch.setattr(ask_service, "_reading_list", reading_list)
    return st


def _ask(q=Q):
    return ask_service.answer_in_scope(q, "m", SimpleNamespace(input=0, output=0))


def _not_topic_declined(r):
    assert r.result_kind != "decline" and sr.decline_text() not in (r.answer or "")
    assert (r.contract or {}).get("offtopic") is not True
    assert sr.offtopic_notice() not in (r.answer or "")


@pytest.mark.parametrize("dec", [SAFE], ids=["safe_request"])
def test_s2_t1_insufficient_corpus_goes_to_acquisition(rig, dec):
    rig["decision"] = dec
    r = _ask()
    assert len(rig["fulldoc"]) == 1 and rig["fulldoc"][0].accepted is True  # corpus attempt as a safe question
    assert rig["fulldoc"][0].category == sr.SAFE_CATEGORY
    assert len(rig["inline"]) == 1 and rig["inline"][0]["in_scope"] is True
    _not_topic_declined(r)


@pytest.mark.parametrize("dec", [SAFE], ids=["safe_request"])
def test_s2_t2_acquisition_off_reading_list_plus_note(rig, dec):
    rig["decision"], rig["acq"] = dec, None
    r = _ask()
    assert r.result_kind == "reading_list" and rig["rl"] == 1 and not rig["inline"]
    assert r.answer.startswith("INSUFFICIENT") and r.answer.endswith(sr.acquisition_off_note())
    assert r.contract["inline_acquisition"] == {"status": "off"}
    _not_topic_declined(r)


def test_s2_t3_covered_question_answered_unchanged(rig):
    ans = AskResult(answer="A [1]", tokens_used=9, cost_usd=0.0, result_kind="answer", accepted=True)
    rig["fd"] = ans
    assert _ask() is ans and not rig["inline"]


@pytest.mark.parametrize("dec", [
    sr.ScopeDecision(False, "layer2_llm", "llm_out:x", 5, 0.0, 1, "harmful"),
    sr.ScopeDecision(False, "layer2_llm", "llm_out:x", 5, 0.0, 1, "advice"),
    sr.ScopeDecision(False, "layer2_llm", "llm_out:x", 5, 0.0, 1, "rule_change"),
    sr.ScopeDecision(False, "layer2_llm", "classifier_malformed", 5, 0.0, 2, ""),
    sr.ScopeDecision(False, "layer2_llm", "classifier_malformed:refusal_token", 5, 0.0, 1, "harmful"),
    sr.ScopeDecision(False, "layer2_llm", "classifier_malformed:category_conflict", 5, 0.0, 1, "advice"),
    sr.ScopeDecision(False, "layer2_llm", "llm_out:x", 5, 0.0, 1, ""),  # D2a: no category named
    sr.ScopeDecision(False, "layer2_llm", "llm_out:catstep", 5, 0.0, 2, ""),  # catstep found no refusal: still declined
    sr.ScopeDecision(False, "layer2_none", "none_mode_unsure", 0, 0.0, 0, ""),
    sr.ScopeDecision(False, "layer1", "deny:injection", 0, 0.0, 0, ""),
    LEGACY_OFF, LEGACY_IN,  # INTEGRATE-1104 round 2 safety fix (S2-T4 revised): a legacy label is never a pass
], ids=["harmful", "advice", "rule_change", "malformed", "refusal_token", "conflict", "no_category", "catstep_ok", "none",
        "injection", "legacy_off_topic", "legacy_in_scope"])
@pytest.mark.parametrize("acq_on", [True, False])
def test_s2_t5_refusal_malformed_none_keep_decline(rig, dec, acq_on):
    rig["decision"] = dec
    if not acq_on:
        rig["acq"] = None
    r = _ask()
    assert r.result_kind == "decline" and r.answer == sr.decline_text()
    assert not rig["fulldoc"] and not rig["inline"] and rig["rl"] == 0 and r.acquire_trigger is None


def test_catstep_ok_no_longer_relabels_off_topic(monkeypatch):
    from vera import catstep
    dec = sr.ScopeDecision(False, "layer2_llm", "llm_out:x", 5, 0.0, 1, "")
    monkeypatch.setattr(catstep, "applies", lambda d, q, cfg=None: True)
    monkeypatch.setattr(catstep, "categorise", lambda q, **k: SimpleNamespace(status="ok", tokens=1, cost=0.0, calls=1,
                                                                            key="k", refuse_category=""))
    out = ask_service._catstep(dec, Q, "m", None, None)
    assert out.accepted is False and out.category == "" and out.category not in ask_service._LEGACY_LABELS


@pytest.mark.parametrize("tok_key,retries", [("insufficient", True), ("corpus_unusable", True), ("nonresponsive", True),
                                             ("claim_check_disabled", False), ("checker_unavailable", False)])
def test_retry_tokens_keep_checker_exclusion(rig, tok_key, retries):
    rig["fd"] = (None, FT[tok_key], 5, 0.0, "T")
    _ask()
    assert len(rig["inline"]) == (1 if retries else 0)  # OD-CMF-1022g


def test_fulldoc_not_configured_still_acquires(rig):
    rig["fd"] = None
    r = _ask()
    assert len(rig["inline"]) == 1 and r.result_kind == "reading_list"
    assert r.contract["no_answer_cause"]["code"] == "reading_list"


def test_news_lane_w1_retry_on_other_token(rig, monkeypatch):
    from vera import news_intent, reporting_lane
    monkeypatch.setattr(reporting_lane, "lane_enabled", lambda *a, **k: True)
    monkeypatch.setattr(news_intent, "news_intent", lambda q: True)
    rig["fd"] = (None, "some_other_non_answer", 5, 0.0, "T")
    _ask("What made the news about the 2026 midterm elections?")
    assert len(rig["inline"]) == 1
    rig["inline"].clear()
    rig["fd"] = (None, FT["claim_check_disabled"], 5, 0.0, "T")
    _ask("What made the news about the 2026 midterm elections?")
    assert not rig["inline"]
    monkeypatch.setattr(news_intent, "news_intent", lambda q: False)
    rig["fd"] = (None, "some_other_non_answer", 5, 0.0, "T")
    _ask()
    assert not rig["inline"]


def _o(status, admitted, stop):
    return {"status": status, "admitted": admitted, "reused": 0, "acquisition": {"status": stop, "stored": admitted,
                                                                                 "target": 6}}


def test_rounds_continue_after_a_deadline_stop_until_no_progress(rig):
    rig["inline_ret"] = "NOTICE"
    rig["outcomes"] = [(_o("ok", 2, "incomplete_deadline"), {"a", "b"}),
                       (_o("ok", 1, "incomplete_deadline"), {"c"}),
                       (_o("ok", 0, "incomplete_deadline"), set())]
    r = _ask()
    assert len(rig["inline"]) == 3
    assert rig["inline"][1]["tried"] == {"a", "b"} and rig["inline"][2]["tried"] == {"a", "b", "c"}
    assert r.contract["inline_acquisition"]["rounds"] == 3


@pytest.mark.parametrize("stop", ["met", "resource_limited", ""])
def test_rounds_stop_on_non_deadline_outcome(rig, stop):
    rig["inline_ret"] = "NOTICE"
    rig["outcomes"] = [(_o("ok", 1, stop), {"a"}), (_o("ok", 1, "incomplete_deadline"), {"b"})]
    _ask()
    assert len(rig["inline"]) == 1


def test_rounds_stop_at_target(rig):
    rig["inline_ret"] = "NOTICE"
    rig["outcomes"] = [(_o("ok", 6, "incomplete_deadline"), {"a"}), (_o("ok", 1, "incomplete_deadline"), {"b"})]
    _ask()
    assert len(rig["inline"]) == 1


def test_answer_from_a_later_round_is_returned(rig, monkeypatch):
    ans = AskResult(answer="A [1]", tokens_used=3, cost_usd=0.0, result_kind="answer", accepted=True,
                    contract={"inline_acquisition": {"status": "ok"}})
    seq = ["NOTICE", ans]
    rig["outcomes"] = [(_o("ok", 1, "incomplete_deadline"), {"a"}), (_o("ok", 1, "met"), {"b"})]
    real = ask_service._inline

    def inline(*a, **k):
        real(*a, **k)
        return seq.pop(0)
    monkeypatch.setattr(ask_service, "_inline", inline)
    r = _ask()
    assert r.result_kind == "answer" and len(rig["inline"]) == 2


def test_log_basis_safe_and_none(caplog):
    with caplog.at_level(logging.INFO, logger="vera"):
        ask_service._log_basis(SAFE)
        ask_service._log_basis(sr.ScopeDecision(False, "layer2_llm", "llm_out:x", category="harmful"))
    msgs = [m.getMessage() for m in caplog.records if m.getMessage().startswith("scope basis=")]
    assert msgs == ["scope basis=safe layer=layer2_llm class=llm_in rid=-",
                    "scope basis=none layer=layer2_llm class=llm_out refusal=harmful rid=-"]


def test_round_spend_is_summed_with_the_classifier_counted_once(rig, monkeypatch):
    notes = [ask_service._Note("N1", 5 + 100, 0.0), ask_service._Note("N2", 5 + 50, 0.0)]  # each includes classifier 5
    rig["outcomes"] = [(_o("ok", 1, "incomplete_deadline"), {"a"}), (_o("ok", 1, "met"), {"b"})]
    real = ask_service._inline

    def inline(*a, **k):
        real(*a, **k)
        return notes.pop(0)
    monkeypatch.setattr(ask_service, "_inline", inline)
    r = _ask()
    assert len(rig["inline"]) == 2 and "N2" in r.answer
    # reading list stub: tokens = fd tokens (5); + round 1 (100) + round 2 (50), classifier (5) once
    assert r.tokens_used == 5 + 100 + 50
