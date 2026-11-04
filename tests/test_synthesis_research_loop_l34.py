"""NONASK-1103 L3/L4 (flag-off new modules): invariants I1-I4 and F6 on mocks. Offline, no model, network or DB."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vera import research_loop as rl
from vera import search_query
from vera import synthesis as syn

ROOT = Path(__file__).resolve().parent.parent
ON = {"enabled": True}
SPANS = [{"n": 1, "snippet": "Span one says creatine improves sprint power."},
         {"n": 2, "snippet": "Span two says creatine raises muscle water."}]
VERIFIED = [{"text": "Creatine improves sprint power [1].", "cites": [1], "verdict": "supported"},
            {"text": "Creatine raises muscle water [2].", "cites": [2], "verdict": "supported"}]
REJECTED = [{"text": "Creatine causes kidney failure [2].", "cites": [2], "verdict": "unsupported"}]
BASELINE = "Creatine improves sprint power [1]. Creatine raises muscle water [2]."


class Counter:
    def __init__(self, ret=None, fn=None):
        self.calls, self.ret, self.fn = [], ret, fn

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return self.fn(*a, **k) if self.fn else self.ret


def _all_pass(claims, sources):
    return {c["id"]: {"verdict": "supported"} for c in claims}, "mock"


# ---------- config: both flags ship off ----------

def test_shipped_flags_are_off():
    assert json.loads((ROOT / "config" / "synthesis.json").read_text())["enabled"] is False
    assert json.loads((ROOT / "config" / "research_loop.json").read_text())["enabled"] is False
    assert syn.enabled() is False and rl.enabled() is False


def test_not_wired_into_answer_path():
    for f in ("vera/ask_service.py", "vera/fulldoc_answer.py", "streamlit_app.py"):
        src = (ROOT / f).read_text(encoding="utf-8")
        assert "vera.synthesis" not in src and "import synthesis" not in src and ", synthesis" not in src
        assert "research_loop" not in src


# ---------- I4: flag off = byte-identical, zero extra calls ----------

def test_synthesis_flag_off_identical_zero_calls():
    s, c = Counter("anything [1]."), Counter(fn=_all_pass)
    text, rec = syn.synthesise(BASELINE, VERIFIED, SPANS, rejected_claims=REJECTED, synthesizer=s, checker=c)
    assert text is BASELINE and text.encode() == BASELINE.encode()
    assert s.calls == [] and c.calls == [] and rec["outcome"] == "disabled"


def test_research_loop_flag_off_identical_zero_calls():
    held = [{"id": "a", "text": "x"}]
    search, assess = Counter([{"id": "b"}]), Counter({"verdict": "finalize"})
    out, rec = rl.run("Why is the sky blue?", {"verdict": rl.RESEARCH_FURTHER}, held, search=search, assess=assess)
    assert out is held and out == [{"id": "a", "text": "x"}]
    assert search.calls == [] and assess.calls == [] and rec["exit_reason"] == "disabled"


# ---------- L3: F6, I1, I2 ----------

def test_reintroduced_rejected_claim_is_blocked():
    draft = "Creatine improves sprint power [1]. creatine causes kidney failure [1]."
    c = Counter(fn=_all_pass)  # even a checker that would pass it never sees it
    text, rec = syn.synthesise(BASELINE, VERIFIED, SPANS, rejected_claims=REJECTED,
                               synthesizer=lambda v, s: draft, checker=c, cfg=ON)
    assert "kidney" not in text and rec["blocked_rejected"] == 1
    assert all("kidney" not in cl["text"] for a, _k in c.calls for cl in a[0])


def test_modified_proposition_is_rechecked_and_dropped_when_unsupported():
    draft = ("Creatine improves sprint power [1]. Creatine greatly improves sprint power [1]. "
             "Creatine raises muscle water [1].")  # reworded; same words but different span
    seen = []

    def checker(claims, sources):
        seen.append((claims, sources))
        return {c["id"]: {"verdict": "unsupported"} for c in claims}, "mock"

    text, rec = syn.synthesise(BASELINE, VERIFIED, SPANS, synthesizer=lambda v, s: draft, checker=checker, cfg=ON)
    assert text == "Creatine improves sprint power [1]."  # only the exact (text, span) verified claim keeps status
    assert rec["kept_exact"] == 1 and rec["rechecked"] == 2 and rec["dropped_unsupported"] == 2
    claims, sources = seen[0]
    assert [c["text"] for c in claims] == ["Creatine greatly improves sprint power [1].",
                                           "Creatine raises muscle water [1]."]
    # I2: the re-check sees admitted span text only, not the draft or the synthesiser's claims
    assert sources == [{"n": 1, "snippet": SPANS[0]["snippet"]}]
    assert all(draft not in json.dumps(s) for s in sources)


def test_modified_proposition_shown_only_after_recheck_pass():
    draft = "Creatine boosts short sprint power [1]. A new uncited sentence."
    text, rec = syn.synthesise(BASELINE, VERIFIED, SPANS, synthesizer=lambda v, s: draft, checker=_all_pass, cfg=ON)
    assert text == "Creatine boosts short sprint power [1]."
    assert rec["recheck_passed"] == 1 and rec["dropped_uncited"] == 1
    assert rec["sentences"][0]["source"] == "rechecked"


def test_non_admitted_span_citation_dropped():
    draft = "Creatine improves sprint power [3]. Creatine raises muscle water [2]."
    text, rec = syn.synthesise(BASELINE, VERIFIED, SPANS, synthesizer=lambda v, s: draft, checker=_all_pass, cfg=ON)
    assert text == "Creatine raises muscle water [2]." and rec["dropped_bad_span"] == 1


@pytest.mark.parametrize("synth,checker,outcome", [
    (None, _all_pass, "fail_closed_no_dependency"),
    (lambda v, s: "", _all_pass, "fail_closed_empty_draft"),
    (lambda v, s: 1 / 0, _all_pass, "fail_closed_synthesizer_error"),
    (lambda v, s: "New claim [1].", lambda c, s: (None, None), "fail_closed_checker_unavailable"),
    (lambda v, s: "New claim [1].", lambda c, s: ({0: {"verdict": "partial"}}, "m"), "fail_closed_nothing_shown"),
])
def test_fails_closed_to_baseline(synth, checker, outcome):
    text, rec = syn.synthesise(BASELINE, VERIFIED, SPANS, synthesizer=synth, checker=checker, cfg=ON)
    assert text is BASELINE and rec["outcome"] == outcome


def test_judge_checker_uses_existing_interface(monkeypatch):
    got = {}

    def fake_judge(claims, sources, cfg, call=None, question=None):
        got.update(claims=claims, sources=sources, cfg=cfg)
        return {c["id"]: {"verdict": "supported"} for c in claims}, "fake"

    monkeypatch.setattr(syn.claim_check, "judge", fake_judge)
    chk = syn.judge_checker({"checkers": []})
    assert chk([{"id": 0, "text": "t [1].", "cites": [1]}], SPANS[:1])[1] == "fake"
    assert got["cfg"] == {"checkers": []}


# ---------- L4: exits, terms only ----------

TERMS_CFG = dict(search_query._default_terms_cfg(), core_phrases=["muscle protein synthesis"])
Q = "Why does resistance training increase muscle protein synthesis in older adults?"
RF = {"verdict": rl.RESEARCH_FURTHER,
      "parts": [{"part": "1", "label": "resistance training older adults", "supported": False}]}


class Clock:
    def __init__(self, step=0.0):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


def _run(search, assess, gate=RF, deadline_s=45.0, clock=None):
    return rl.run(Q, gate, [{"id": "p0", "text": "held"}], search=search, assess=assess, cfg=ON,
                  terms_cfg=TERMS_CFG, deadline_s=deadline_s, clock=clock or Clock())


def test_exit_finalize_after_new_passage():
    out, rec = _run(Counter([{"id": "p1"}]), Counter({"verdict": "finalize"}))
    assert rec["exit_reason"] == "finalize" and [p["id"] for p in out] == ["p0", "p1"] and rec["rounds"] == 1


def test_exit_no_new_passage():
    out, rec = _run(Counter([{"id": "p0", "text": "held"}]), Counter(RF))
    assert rec["exit_reason"] == "no_new_passage" and len(out) == 1


def test_exit_repeated_query():
    n = iter(range(100))
    search = Counter(fn=lambda terms, rem: [{"id": "new%d" % next(n)}])
    out, rec = _run(search, Counter(RF))  # the need never changes, so the second round repeats the query key
    assert rec["exit_reason"] == "repeated_query" and len(search.calls) == 1


def test_exit_deadline_before_search():
    search = Counter([{"id": "p1"}])
    _out, rec = _run(search, Counter(RF), deadline_s=0.0)
    assert rec["exit_reason"] == "deadline" and search.calls == []


def test_exit_deadline_terminates_ever_progressing_loop():
    labels = iter("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi".split())

    def assess(held):
        return {"verdict": rl.RESEARCH_FURTHER, "parts": [{"label": next(labels) + " fibres", "supported": False}]}

    n = iter(range(1000))
    search = Counter(fn=lambda terms, rem: [{"id": "n%d" % next(n)}])
    _out, rec = _run(search, assess, deadline_s=45.0, clock=Clock(step=10.0))
    assert rec["exit_reason"] == "deadline" and 1 <= rec["rounds"] < 10
    assert all(0 < a[1] <= 45.0 for a, _k in search.calls)


def test_exit_not_triggered_and_terminal_verdicts():
    search = Counter([{"id": "p1"}])
    for v in ("finalize", "qualify", "abstain"):
        _o, rec = _run(search, Counter(RF), gate={"verdict": v})
        assert rec["exit_reason"] == "not_triggered"
    _o, rec = _run(search, Counter({"verdict": "qualify"}))
    assert rec["exit_reason"] == "qualify"


def test_exit_no_terms_never_falls_back_to_raw():
    search = Counter([{"id": "p1"}])
    out, rec = rl.run("Why is the?", {"verdict": rl.RESEARCH_FURTHER}, [], search=search, assess=Counter(RF),
                      cfg=ON, terms_cfg=TERMS_CFG, deadline_s=45.0, clock=Clock())
    assert rec["exit_reason"] == "no_terms" and search.calls == []


def test_exit_search_error():
    _o, rec = _run(Counter(fn=lambda t, r: 1 / 0), Counter(RF))
    assert rec["exit_reason"] == "search_error"


def test_no_raw_sentence_reaches_search():
    search = Counter([{"id": "p1"}])
    gate = {"verdict": rl.RESEARCH_FURTHER}  # no part labels: terms come from the question itself
    _o, rec = _run(search, Counter({"verdict": "finalize"}), gate=gate)
    assert len(search.calls) == 1
    (terms, _rem), _k = search.calls[0]
    assert isinstance(terms, tuple) and terms and all(isinstance(t, str) for t in terms)
    blob = " ".join(terms)
    assert Q not in blob and Q.lower() not in blob.lower() and "?" not in blob
    assert not {"why", "does", "in"} & {t.lower() for t in terms}
    assert "muscle protein synthesis" in terms
