"""MULTIQ-1108 (owner 2026-11-08: "Sometimes, people pose multi-question or multi-part question prompts. VD needs to be
able to handle both."). Offline: no DB, network or model. Detector cases are the plan's eval set (work/MULTIQ-1108/
plan.md section e); the Ask-path tests stub the scope route and the per-question answer path."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tests.test_ui_render_extract as base
from tests.test_fix1_offtopic import PRICING
from tests.test_selfq_relevance_1108 import VOCAB, _known
from vera import ask_service, multi_question as mq, page_text, scope_router as sr, spell_clarify as sc
from vera.ask_service import AskResult

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = json.loads((ROOT / "config" / "multi_question.json").read_text())
ON = dict(SHIPPED, mode="on")


# ---- detector (pure) ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q, n", [
    ("What is VERA-DAR? And how accurate are flu forecasts?", 2),
    ("Who founded OpenAI? When was Anthropic founded? Where is Google based?", 3),
    ("How do mRNA vaccines work, and what are their side effects?", 2),
    ("what causes inflation. how do central banks respond", 2),
    ("What is CRISPR?\nIs it used in agriculture?", 2),
    ("What is VERA-DAR? Who can use it?", 2),
    ("When did Einstein win his second Nobel Prize? And what is general relativity?", 2),
    ("Who founded OpenAI and when was Anthropic founded?", 2),
])
def test_multi_question_prompts_split(q, n):
    s = mq.detect(q, ON)
    assert s is not None and len(s.parts) == n and not s.over_cap
    assert mq._words(" ".join(p.text for p in s.parts)) == mq._words(q)  # the user's own words, nothing added or lost


@pytest.mark.parametrize("q", [
    "What is dengue? And why?",  # follow-on without a content word: merged, one question
    "Compare influenza vaccines and COVID-19 vaccines on cost and accuracy",
    "What is Vera dar magops?",
    "How do AWS, Google Cloud and Azure price GPU instances?",
    "What are the pros and cons of remote work?",
    "Is coffee safe and effective for weight loss?",
    "What is the difference between weather and climate?",
    "Which is better for heart health: walking, running or cycling?",
    "What did Dr. Smith say about U.S. vaccine policy?",
    "How, when and why did the Roman Empire fall?",
    "What is 3.5 percent of GDP? Please be brief.",
    "What is inflation, and what about deflation?",
])
def test_single_and_multi_part_questions_are_not_split(q):
    assert mq.detect(q, ON) is None


@pytest.mark.parametrize("q", [
    "What is the difference between what is legal and what is ethical?",
    "What happens when it rains and is cold?",
    "If it rains, is the match cancelled?",
    "For adults over sixty, what is the recommended dose?",
    "Tell me about vaccines, what are the risks?",  # one topic with a facet question: the topic is its context
    "I have a cough. Should I see a doctor?",  # context plus a question is one question
    'What does the phrase "Who am I? What is real?" mean in philosophy?',
    "What does “Why are we here? Who knows?” mean?",
])
def test_false_splits_stay_one_question(q):
    assert mq.detect(q, ON) is None


@pytest.mark.parametrize("q, n", [
    ("I have a cough. Should I see a doctor? What causes a cough?", 2),
    ("If it rains, is the match cancelled? Who referees the match?", 2),
    ('What does "Who am I? What is real?" mean in philosophy? Who coined the phrase?', 2),
    ("Who founded OpenAI and when was Anthropic founded?", 2),
    ("What is the difference between weather and climate? And when did Earth warm?", 2),
    ("For adults over sixty, what is the dose? Is it safe for children?", 2),
])
def test_true_multi_question_controls_still_split(q, n):
    s = mq.detect(q, ON)
    assert s is not None and len(s.parts) == n


def test_multi_part_shape_is_recorded_not_split():
    assert mq.shape("Compare influenza vaccines and COVID-19 vaccines on cost and accuracy", None) == "multi_part"
    assert mq.shape("What are the pros and cons of remote work?", None) == "single"


def test_cap_lists_the_rest_never_merges_or_drops():
    s = mq.detect("What is malaria? How is it spread? What are its symptoms? How is it treated?", ON)
    assert [p.text for p in s.parts] == ["What is malaria?", "How is it spread?", "What are its symptoms?"]
    assert s.over_cap == ("How is it treated?",) and s.n_questions == 4


def test_pronoun_part_carries_the_subject_for_answering_only():
    s = mq.detect("What is malaria? How is it spread? What are its symptoms?", ON)
    assert s.parts[1].carry == ("malaria",) and s.parts[2].carry == ("malaria",)
    assert s.parts[2].text == "What are its symptoms?" and s.parts[2].query == "What are its symptoms? (malaria)"
    s2 = mq.detect("How do mRNA vaccines work, and what are their side effects?", ON)
    assert set(s2.parts[1].carry) <= set(w.casefold() for w in "How do mRNA vaccines work".split())  # drift guard


def test_off_mode_and_failures_never_split():
    assert mq.detect("What is X? What is Y?", dict(ON, mode="off")) is None
    assert mq.mode_of({"mode": "bogus"}) == "off" and mq.mode_of(None) == "off"
    assert mq.detect(None, ON) is None


def test_config_is_grammar_only_and_ships_on():
    assert SHIPPED["mode"] == "on"  # MULTIQ-1109: owner "VD needs to be able to handle both"
    assert "_max_questions_calibration_note" in SHIPPED and "PROVISIONAL" in SHIPPED["_max_questions_calibration_note"]
    lists = {k for k, v in SHIPPED.items() if isinstance(v, list)}
    assert lists == {"interrogatives", "auxiliaries", "leading_conjunctions", "pronouns", "subordinators"}  # closed classes, no topics


def test_trace_allowlist_has_the_multiq_stage():
    cfg = json.loads((ROOT / "config" / "request_trace.json").read_text())["key_allowlist"]
    assert set(cfg["multiq"]["keys"]) == {"shape", "mode", "result_kind"}


# ---- Ask path ---------------------------------------------------------------------------------------------------------
def _answer(q, n=1):
    return AskResult(answer=f"Answer about {q.split('?')[0].split()[-1]} [1].", tokens_used=5, cost_usd=0.001,
                     accepted=True, result_kind="answer",
                     sources=({"n": 1, "title": f"Doc {n}", "url": f"https://example.org/{n}", "published": None},),
                     contract={"layers": ["L0"], "fallback": None})


def _noanswer(text="VERA-DAR could not find a passage that answers this question."):
    return AskResult(answer=text, tokens_used=2, cost_usd=0.0, accepted=True, result_kind="reading_list",
                     contract={"layers": ["L0"], "fallback": "insufficient"})


@pytest.fixture
def ask(monkeypatch):
    """answer_in_scope with the classifier and the per-question answer path stubbed. `parts` records the questions
    the single-question path received and the decision it ran under."""
    st = {"route": 0, "parts": [], "events": [], "mode": "on", "answers": {}}
    monkeypatch.setattr(mq, "load_config", lambda path=None: dict(SHIPPED, mode=st["mode"]))
    monkeypatch.setattr(ask_service, "_clarify_or_help", lambda q: None)
    monkeypatch.setattr(ask_service, "_help_answer", lambda q, covered_only, tokens=0, cost=0.0: None)
    monkeypatch.setattr(ask_service, "_catstep", lambda d, *a, **k: d)
    monkeypatch.setattr(ask_service, "_acq_cfg_if_on", lambda: None)
    monkeypatch.setattr(ask_service, "_fresh_cfg", lambda acq: None)
    monkeypatch.setattr(ask_service, "_rt_event", lambda *a, **k: st["events"].append((a, k)))

    def route(original, *a, **k):
        st["route"] += 1
        return st.get("decision") or sr.ScopeDecision(True, "layer2_llm", "ok", tokens_used=10, cost_usd=0.01)
    monkeypatch.setattr(sr, "route", route)

    def fulldoc(question, *a, **k):
        st["parts"].append((question, ask_service._MQ_PART.get()))
        r = st["answers"].get(question)
        return r if r is not None else _answer(question, len(st["parts"]))
    monkeypatch.setattr(ask_service, "_fulldoc", fulldoc)
    return st


def test_shadow_records_the_split_and_changes_nothing(ask):
    q = "What is CRISPR? Is it used in agriculture?"
    ask["mode"] = "off"
    off = ask_service.answer_in_scope(q, "m", PRICING)
    ask["mode"], ask["parts"], ask["events"] = "shadow", [], []
    sh = ask_service.answer_in_scope(q, "m", PRICING)
    assert sh == off and [p for p, _ in ask["parts"]] == [q]  # one question, the whole prompt, as before
    ev = [(a, k) for a, k in ask["events"] if a[0] == "multiq"]
    assert ev and ev[0][0][1] == "split" and ev[0][1]["counts"]["parts"] == 2
    assert ev[0][1]["detail"] == {"shape": "multi_question", "mode": "shadow"}
    assert "CRISPR" not in json.dumps([k for _, k in ev])  # counts and tokens only, never text


def test_on_routes_once_and_answers_each_question(ask):
    q = "What is CRISPR? Is it used in agriculture?"
    r = ask_service.answer_in_scope(q, "m", PRICING)
    assert ask["route"] == 1  # the whole prompt is classified once
    assert [p for p, _ in ask["parts"]] == ["What is CRISPR?", "Is it used in agriculture? (crispr)"]
    assert all(d is not None and d.tokens_used == 0 for _, d in ask["parts"])  # classifier cost counted once
    assert r.result_kind == "answer" and r.tokens_used == 10 + 5 + 5
    parts = r.contract["parts"]
    assert [p["question"] for p in parts] == ["What is CRISPR?", "Is it used in agriculture?"]  # user's own words
    assert all(p["result_kind"] == "answer" and p["sources"][0]["n"] == 1 for p in parts)
    assert r.sources == () and r.claims == () and r.evidence is None  # nothing written to memory from a combination
    assert "**1. What is CRISPR?**" in r.answer and "**2. Is it used in agriculture?**" in r.answer


def test_honest_partial_refusal(ask):
    q = "What is photosynthesis? Should I stop taking my insulin?"  # layer 1 declines the medical-advice question
    r = ask_service.answer_in_scope(q, "m", PRICING)
    p1, p2 = r.contract["parts"]
    assert p1["result_kind"] == "answer" and p2["result_kind"] == "decline"
    assert [p for p, _ in ask["parts"]] == ["What is photosynthesis?"]  # the declined question is never answered
    assert p2["answer"] == sr.decline_text() and r.answer.count(sr.decline_text()) == 1
    assert r.result_kind == "decline"  # fail closed for older clients: not every question was answered


def test_honest_no_answer_for_one_question(ask):
    ask["answers"]["Who founded the Moon Bank?"] = _noanswer()
    r = ask_service.answer_in_scope("What is CRISPR? Who founded the Moon Bank?", "m", PRICING)
    kinds = [p["result_kind"] for p in r.contract["parts"]]
    assert kinds == ["answer", "reading_list"] and r.result_kind == "reading_list"
    assert r.contract["parts"][1]["sources"] == [] and r.contract["fallback"] == "insufficient"


def test_whole_prompt_safety_decline_declines_everything(ask):
    ask["decision"] = sr.ScopeDecision(False, "layer1", "deny:harmful")
    r = ask_service.answer_in_scope("What is photosynthesis? How do I make a pipe bomb?", "m", PRICING)
    assert r.result_kind == "decline" and ask["parts"] == [] and "parts" not in (r.contract or {})


def test_over_cap_listed_once_not_answered(ask):
    r = ask_service.answer_in_scope("What is malaria? How is it spread? What are its symptoms? How is it treated?",
                                    "m", PRICING)
    assert len(ask["parts"]) == 3 and r.contract["over_cap"] == ["How is it treated?"]
    assert r.answer.count(SHIPPED["over_cap_text"]) == 1 and r.answer.count("How is it treated?") == 1


def test_no_sentence_twice_on_the_combined_answer(ask):
    same = _noanswer()
    ask["answers"].update({"Who founded the Moon Bank?": same, "When did the Moon Bank close?": same})
    r = ask_service.answer_in_scope("Who founded the Moon Bank? When did the Moon Bank close?", "m", PRICING)
    sents = [page_text._key(s) for s in page_text.sentences(r.answer)]
    assert len(sents) == len(set(sents))
    assert r.contract["parts"][1]["answer"] == SHIPPED["same_reason_text"].replace("{n}", "1")


def test_sense_rounds_of_several_questions_become_one(ask):
    def clar(q, word, opts):
        c = {"word": word, "text": f"'{word}' has more than one meaning. Did you mean:", "as_typed": q,
             "as_typed_label": "search as typed", "suggestions": [{"word": o, "question": f"{q} ({o})?"} for o in opts]}
        return AskResult(answer="x", tokens_used=0, cost_usd=0.0, result_kind="clarification",
                         contract={"clarification": c})
    ask["answers"]["What are leptons?"] = clar("What are leptons?", "leptons", ["particle physics", "GPU platform"])
    ask["answers"]["What is mercury?"] = clar("What is mercury?", "mercury", ["planet", "element", "god"])
    r = ask_service.answer_in_scope("What are leptons? What is mercury?", "m", PRICING)
    c = r.contract["clarification"]
    assert r.result_kind == "clarification" and c["final"] is True and c["questions"] == 2
    assert c["as_typed"] == "What are leptons? What is mercury?" and len(c["suggestions"]) == 3
    assert c["suggestions"][0]["question"] == "What are leptons? (particle physics)? What is mercury? (planet)?"
    assert c["suggestions"][2]["word"] == "GPU platform / god"


# ---- one combined typo round ----------------------------------------------------------------------------------------
@pytest.fixture
def blocked(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not be reached")
    monkeypatch.setattr(mq, "load_config", lambda path=None: dict(ON))
    monkeypatch.setattr(sr, "route", boom)
    for n in ("_fulldoc", "answer_question", "complete", "_reading_list", "_offtopic_reading_list", "_inline"):
        monkeypatch.setattr(ask_service, n, boom)
    monkeypatch.setattr(sc, "vocabulary", lambda *a, **k: VOCAB)
    monkeypatch.setattr(ask_service, "_corpus_best_relevance", lambda q: 0.0)
    monkeypatch.setattr(sc, "known_to_corpus", lambda w, vocab=None: _known(w))


def test_typos_in_two_questions_are_one_final_round(blocked):
    r = ask_service.answer_in_scope("What is vrea? How do brns heal?", "m", PRICING)
    c = r.contract["clarification"]
    assert r.result_kind == "clarification" and c["final"] is True and c["questions"] == 2
    assert "'vrea'" in c["text"] and "'brns'" in c["text"]
    assert any(s["question"] == "What is vera? How do burns heal?" for s in c["suggestions"])


def test_single_question_typo_round_is_unchanged(blocked):
    r = ask_service.answer_in_scope("what is vrea?", "m", PRICING)
    assert "final" not in r.contract["clarification"]


def test_final_round_resumes_with_spell_check_off():
    class UI:
        def __init__(self):
            self.payloads = []

        def info(self, *a, **k):
            pass

        def columns(self, n):
            return [self] * n

        def button(self, label, key=None, on_click=None, args=()):
            self.payloads.append(args[0])
    c = {"word": "vrea", "text": "?", "as_typed": "q", "suggestions": [{"word": "vera", "question": "q2"}]}
    for final, want in ((True, False), (False, None)):
        ui = UI()
        sc.render_clarification(ui, {"result_kind": "clarification", "contract": {"clarification": dict(c, final=final)}},
                                lambda p: None)
        assert ui.payloads[0].get("spell_check") is want and ui.payloads[-1]["spell_check"] is False


# ---- page ------------------------------------------------------------------------------------------------------------
def _combined():
    return AskResult(
        answer="combined", tokens_used=3, cost_usd=0.0, accepted=True, result_kind="reading_list",
        contract={"layers": ["L0", "L2"], "fallback": "insufficient", "over_cap": ["How is it treated?"],
                  "shared_notes": ["VERA-DAR looked only in its own documents."],
                  "multi_question": {"parts": 2, "over_cap": 1},
                  "parts": [{"index": 1, "question": "What is malaria?", "result_kind": "answer",
                             "answer": "Malaria is a disease spread by mosquitoes [1].", "fallback": None,
                             "sources": [{"n": 1, "title": "WHO malaria fact sheet", "url": "https://example.org/m",
                                          "published": None}], "candidates": []},
                            {"index": 2, "question": "Who founded the Moon Bank?", "result_kind": "reading_list",
                             "answer": "VERA-DAR could not find a passage that answers this question.",
                             "fallback": "insufficient", "sources": [], "candidates": []}]})


def test_page_renders_one_section_per_question(monkeypatch):
    from vera import progress
    monkeypatch.setattr(progress, "stream_enabled", lambda: False)  # legacy /ask call (the stream ends in render_result)
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    import main
    from vera import public_mode as pm
    pm.reset_limiter(None)
    monkeypatch.setattr(main, "answer_via_chain", lambda q, e, g: (_combined(), "groq:m"))
    monkeypatch.setattr(main, "store_from_env", lambda cfg: None)
    monkeypatch.setattr(main, "memory_from_env", lambda block: None)
    body = TestClient(main.app).post("/ask", json={"question": "What is malaria? Who founded the Moon Bank?"},
                                     headers={"X-API-Key": "k" * 20}).json()
    assert [p["index"] for p in body["contract"]["parts"]] == [1, 2]  # the API passes the parts through
    at = base._run(monkeypatch, body)
    md = [e.value for e in at.markdown]
    assert not at.exception, at.exception
    assert any("1. What is malaria?" in v for v in md) and any("2. Who founded the Moon Bank?" in v for v in md)
    assert any("Malaria is a disease" in e.value for e in at.success)
    assert any("could not find a passage" in e.value for e in at.info)
    assert any(b.label == "How is it treated?" for b in at.button)
    texts = [e.value for e in list(at.markdown) + list(at.info) + list(at.caption) + list(at.success)
             if not e.value.startswith("💡")]
    sents = [page_text._key(s) for v in texts for s in page_text.sentences(v) if len(s) > 25]
    assert len(sents) == len(set(sents))


# ---- MULTIQ-1109: quota / per-prompt ceiling reconciliation with SIX-1109 -----------------------------------------------
def test_share_replaces_the_per_prompt_ceiling_only_inside_a_part():
    acq = {"bounds": {"max_docs_per_prompt": 31, "max_docs_per_job": 2}}
    assert ask_service._mq_share_acq(acq) is acq and ask_service._mq_share_acq(None) is None
    tok = ask_service._MQ_DOCS.set(10)
    try:
        out = ask_service._mq_share_acq(acq)
    finally:
        ask_service._MQ_DOCS.reset(tok)
    assert out["bounds"]["max_docs_per_prompt"] == 10 and acq["bounds"]["max_docs_per_prompt"] == 31  # copy, not mutated
    tok = ask_service._MQ_DOCS.set(99)
    try:
        assert ask_service._mq_share_acq(acq)["bounds"]["max_docs_per_prompt"] == 31  # never raises the ceiling
    finally:
        ask_service._MQ_DOCS.reset(tok)


@pytest.mark.parametrize("q, n", [("What is CRISPR? Is it used in agriculture?", 2),
                                  ("Who founded OpenAI? When was Anthropic founded? Where is Google based?", 3)])
def test_each_question_gets_its_own_job_and_an_equal_share(ask, monkeypatch, q, n):
    seen = []
    monkeypatch.setattr(ask_service, "_acq_cfg_if_on", lambda: {"bounds": {"max_docs_per_prompt": 31}})
    budget = ask_service.RequestBudget(float("inf"))
    orig = ask["parts"]

    def fulldoc(question, *a, **k):
        seen.append((budget.inline_used, ask_service._MQ_DOCS.get()))
        budget.inline_used = True  # this question's research attempt ran
        orig.append((question, ask_service._MQ_PART.get()))
        return _answer(question, len(orig))
    monkeypatch.setattr(ask_service, "_fulldoc", fulldoc)
    r = ask_service.answer_in_scope(q, "m", PRICING, budget=budget)
    assert r.result_kind == "answer" and len(seen) == n
    assert all(used is False for used, _ in seen)  # the previous question's attempt never blocks the next one
    assert {sh for _, sh in seen} == {31 // n} and n * (31 // n) <= 31  # the prompt stays within the ceiling
