"""NONASK-1103 E4/E4m/E6: the public evidence trace is built at both fulldoc constructors, survives ask_service's
contract merges to the /ask body and the /ask/stream final event, and renders inside the Report expander (redacted).
Offline: stub writer and checker, no network, no DB."""
import copy
import json
from types import SimpleNamespace

import pytest

import main
from tests.test_fulldoc_answer import GCFG, Stub, bundle, checker, span
from tests.test_stream_route_support import API, KEY, Q, client, lines, stream_env  # noqa: F401
from vera import ask_service, fulldoc_answer as fa, fulldoc_corpus
from vera.ask_service import AskResult
from vera.m7 import live_report

ORIG_ANSWER = fa.answer
TEXT = "Beta caching doubles throughput under heavy load [1]. Gamma results are short [2]."


def _setup(monkeypatch, per_part, verdicts, text=TEXT, ev=None):
    monkeypatch.setenv("FD_TEST_KEY", "x")
    cfg = copy.deepcopy(fulldoc_corpus.load_config())
    cfg["answer"]["per_part"]["enabled"] = per_part
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(fulldoc_corpus, "get_corpus", lambda c: object())
    monkeypatch.setattr(ask_service, "_base_warming", lambda: False)
    monkeypatch.setattr(ask_service, "_pre_llm", lambda **k: None)
    monkeypatch.setattr("vera.fulldoc_admission.admit", lambda *a, **k: ev or bundle())
    stub = Stub(text)

    def answer(question, ev_, model, pricing, **kw):
        kw.update(complete_fn=stub, check_call=checker(verdicts))
        return ORIG_ANSWER(question, ev_, model, pricing, **kw)

    monkeypatch.setattr(fa, "answer", answer)


def _fulldoc():
    dec = SimpleNamespace(tokens_used=0, cost_usd=0.0, accepted=True)
    return ask_service._fulldoc("q", "m", None, dec, {}, (None, GCFG))


@pytest.mark.parametrize("per_part", [True, False])
def test_trace_on_both_answer_constructors_and_ask_service_merge(monkeypatch, per_part):
    _setup(monkeypatch, per_part, ("supported", "supported"))
    res = _fulldoc()
    assert isinstance(res, AskResult) and res.result_kind == "answer"
    tr = res.contract["trace"]
    assert tr["view"] == "public" and [p["n"] for p in tr["passages"]] == [1, 2] and all(p["cited"] for p in tr["passages"])
    assert res.contract["layers"] is not None and res.contract["fallback"] is None  # merged, not replaced
    assert tr["adequacy"]["verdict"] == "finalize"
    dumped = json.dumps(tr)
    for bad in ("span_id", "document_id", "relevance", "s1", "d1", "x.org"):
        assert bad not in dumped


def _body_and_stream(monkeypatch, result):
    monkeypatch.setattr(main, "answer_via_chain", lambda *a, **k: (result, "p"))
    c = client()
    body = c.post("/ask", json={"question": Q}, headers=API)
    s = c.post("/ask/stream", json={"question": Q}, headers=API)
    return body, s


@pytest.mark.parametrize("per_part", [True, False])
def test_trace_reaches_ask_body_and_stream_final_event(monkeypatch, per_part):
    _setup(monkeypatch, per_part, ("supported", "supported"))
    res = _fulldoc()
    body, s = _body_and_stream(monkeypatch, res)
    assert body.status_code == 200, body.text
    t = body.json()["contract"]["trace"]
    assert t == res.contract["trace"] and t["view"] == "public"
    final = [e for e in lines(s) if e.get("type") == "result" or "contract" in json.dumps(e)][-1]
    assert json.dumps(final).count('"view": "public"') >= 1
    ev = json.dumps(final)
    assert '"passages"' in ev and "layers" in ev  # existing keys kept beside the trace


def test_trace_on_no_answer_outcome_and_contract(monkeypatch):
    _setup(monkeypatch, True, ("unsupported", "unsupported"))
    out = _fulldoc()
    assert isinstance(out, tuple) and out[1] == "claim_check_insufficient"
    tr = ask_service._FD_TRACE.get()
    assert tr and tr["view"] == "public" and all(not p["cited"] and p["reason"] for p in tr["passages"])
    assert ask_service._fd_trace_contract() == {"trace": tr, "trace_attempt": "stored"}  # SELFQ-RCA-1107 D2: origin tag
    # a later attempt starts clean: the stale trace never leaks into it
    monkeypatch.setattr(fulldoc_corpus, "get_corpus", lambda c: None)
    assert _fulldoc() is None and ask_service._FD_TRACE.get() is None


def test_no_answer_outcome_carries_trace_when_result_is_none(monkeypatch):
    _setup(monkeypatch, False, ("unsupported", "unsupported"))
    out = ORIG_ANSWER("q", bundle(), "m", None, chained={}, grounding_cfg=GCFG, cfg=fulldoc_corpus.load_config(),
                      complete_fn=Stub(TEXT), check_call=checker(("unsupported", "unsupported")))
    assert out.result is None and out.trace and out.trace["passages"]


def test_keep_partial_parts_defaults_false_and_a_switch(monkeypatch):
    cfg = fulldoc_corpus.load_config()
    assert cfg["answer"]["per_part"]["keep_partial_parts"] is False and cfg["adequacy"]["qualify_line"] is True
    two = "Parts:\nP1: first\nP2: second\nAnswers:\nP1: Beta caching doubles throughput [1].\nP2: Gamma results are short [2]."
    _setup(monkeypatch, True, ("supported", "partial"), text=two)
    off = _fulldoc()
    assert "Part 2" in off.answer and "do not support an answer to this part" in off.answer  # today's D8 behaviour
    cfg2 = copy.deepcopy(fulldoc_corpus.load_config())
    cfg2["answer"]["per_part"]["keep_partial_parts"] = True
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda *a, **k: cfg2)
    on = _fulldoc()
    assert "partly supported by the documents" in on.answer and on.claim_check["answer_completeness"] == "partial"
    assert on.answer.count("Part 2") == 1  # a part is never named twice
    assert on.contract["trace"]["adequacy"]["verdict"] == "qualify"


def test_pick_quote_snaps_to_sentence_inside_cap():
    t = "First sentence is here. Beta caching doubles throughput when warm, and it holds under load. Last one."
    q = fa.pick_quote(t, "Beta caching doubles throughput", 200)
    assert q == "Beta caching doubles throughput when warm, and it holds under load." and q in t
    q2 = fa.pick_quote(t, "Beta caching doubles throughput", 40)
    assert len(q2) <= 40 and q2 in t


# ---- E6 -----------------------------------------------------------------------------------------------------
class _St:
    def __init__(self):
        self.out, self.sidebar = [], SimpleNamespace(markdown=lambda *a, **k: self.out.append("SIDEBAR"))

    def __getattr__(self, name):
        if name == "expander":
            def exp(label, **k):
                self.out.append(("expander", label))
                return self
            return exp
        return lambda *a, **k: self.out.append((name, a[0] if a else None))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _trace():
    return {"version": 1, "view": "public", "parts": [{"label": "see host leak.example.net", "covered": False}],
            "passages": [{"n": 1, "cited": True, "supports": ["Beta doubles [1]."], "verdict": "supported",
                          "excerpt": "Beta caching doubles throughput.", "reason": None},
                         {"n": 2, "cited": False, "reason": "not used by the answer; no reason recorded"}],
            "adequacy": {"verdict": "qualify", "all_declared_parts_supported": False}}


def _body(trace):
    return {"answer": "A [1]", "result_kind": "answer", "contract": {"trace": trace} if trace else {}}


def test_build_live_report_copies_trace():
    assert live_report.build_live_report(_body(_trace()), "q", 1.0)["trace"] == _trace()
    assert live_report.build_live_report(_body(None), "q", 1.0)["trace"] is None
    assert live_report.build_live_report({"contract": {"trace": "junk"}}, "q", 1.0)["trace"] is None


def test_report_renders_trace_inside_verification_expander_and_redacts():
    rep = live_report.build_live_report(_body(_trace()), "q", 1.0)
    st = _St()
    live_report._render_verification(st, rep, lambda t: t.replace("leak.example.net", "[host removed]"), True)
    flat = json.dumps(st.out, default=str)
    assert st.out[0] == ("expander", "Research and verification")
    assert "How VERA-DAR chose the evidence" in flat and "Parts of your question" in flat
    assert "leak.example.net" not in flat, flat
    assert "host removed" in flat and "SIDEBAR" not in flat


def test_report_without_trace_or_other_content_has_no_expander():
    rep = live_report.build_live_report(_body(None), "q", 1.0)
    rep["claim_check"] = ""
    st = _St()
    live_report._render_verification(st, rep, lambda t: t, True)
    assert st.out == []


def test_report_public_never_renders_dev_trace():
    t = dict(_trace(), view="dev")
    rep = live_report.build_live_report(_body(t), "q", 1.0)
    st = _St()
    live_report._render_verification(st, rep, lambda x: x, True)
    assert st.out == []
