"""Observer inside pipeline_llm.call (S1). Fakes only: no DB, no network, no provider call."""
import pytest

from tests.m45_support import FakeLLM, corpus  # noqa: F401
import vera.m3.llm as m3_llm
import vera.pipeline_llm as pl
from vera.m4.evidence_relations import build_relations
from vera.m4.gate_c import evaluate_adequacy
from vera.m4.m4_runner import run_m4
from vera.m5.claim_verification import verify_claims
from vera.pipeline_llm import LLMError, LLMResult, call, observe_calls
from vera.run_budget import RunBudgetExceeded


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    def boom(system, user, **kw):
        raise AssertionError("complete_json (a real provider call) was reached from a test")
    monkeypatch.setattr(pl, "complete_json", boom)


def _res(**kw):
    return LLMResult(data=kw.get("data", {}), model="fake", tokens=kw.get("tokens", 10),
                     cost_usd=kw.get("cost", 0.001))


def test_no_observer_call_is_passthrough_and_default_not_invoked():
    seen = []
    out = call(lambda s, u: seen.append((s, u)) or _res(data={"k": 1}), "sys", "usr")
    assert out.data == {"k": 1} and seen == [("sys", "usr")]  # complete_json (patched to raise) untouched


def test_observer_sees_success_and_before_runs_first():
    order = []
    with observe_calls(lambda: order.append("before"), lambda r, e, t: order.append(("after", r.model, e, t >= 0))):
        out = call(lambda s, u: order.append("llm") or _res(), "s", "u")
    assert out.model == "fake" and order == ["before", "llm", ("after", "fake", None, True)]


def test_observer_scope_ends_with_block():
    hits = []
    with observe_calls(lambda: hits.append("b"), lambda r, e, t: hits.append("a")):
        pass
    call(lambda s, u: _res(), "s", "u")
    assert hits == []


def test_after_receives_exception_and_original_llmerror_reraised():
    got = []
    err = LLMError("provider down: detail")

    def bad(s, u):
        raise err
    with observe_calls(lambda: None, lambda r, e, t: got.append((r, e))):
        with pytest.raises(LLMError) as ei:
            call(bad, "s", "u")
    assert ei.value is err and got == [(None, err)]


def test_after_hook_bug_does_not_mask_original_error():
    def bad(s, u):
        raise LLMError("original")

    def broken_after(r, e, t):
        raise ValueError("observer bug")
    with observe_calls(lambda: None, broken_after):
        with pytest.raises(LLMError, match="original"):
            call(bad, "s", "u")


def test_before_raising_prevents_the_call():
    called = []

    def trip():
        raise RunBudgetExceeded("cap hit")
    with observe_calls(trip, lambda r, e, t: called.append("after")):
        with pytest.raises(RunBudgetExceeded):
            call(lambda s, u: called.append("llm") or _res(), "s", "u")
    assert called == []


def test_run_budget_exceeded_is_not_an_llm_error():
    assert not issubclass(RunBudgetExceeded, pl.LLMError)
    assert not issubclass(RunBudgetExceeded, m3_llm.LLMError)
    assert issubclass(RunBudgetExceeded, RuntimeError)


def _tripper():
    def trip():
        raise RunBudgetExceeded("cap hit")
    return observe_calls(trip, lambda r, e, t: None)


def test_trip_propagates_through_gate_c_missingness(corpus):  # noqa: F811
    llm = FakeLLM()
    with _tripper():
        with pytest.raises(RunBudgetExceeded):
            evaluate_adequacy(corpus, "q", llm=llm, use_llm_missingness=True)
    assert llm.calls == []


def test_trip_propagates_through_build_relations(corpus):  # noqa: F811
    llm = FakeLLM()
    with _tripper():
        with pytest.raises(RunBudgetExceeded):
            build_relations(corpus, llm=llm)
    assert llm.calls == []


def test_trip_propagates_through_verify_claims(corpus):  # noqa: F811
    resp = {"response_text": "x", "claims": [{"claim_text": "a claim", "evidence_span_ids": ["s1"]}]}
    calls = []
    with _tripper():
        with pytest.raises(RunBudgetExceeded):
            verify_claims(resp, corpus, llm=lambda s, u: calls.append(1) or _res())
    assert calls == []


def test_llm_none_under_observer_makes_no_gate_c_audit_call(monkeypatch, corpus):  # noqa: F811
    systems = []

    def counting(system, user, **kw):
        systems.append(system)
        return LLMResult(data={"relations": [], "gaps": []}, model="fake", tokens=5, cost_usd=0.001)
    monkeypatch.setattr(pl, "complete_json", counting)
    metered = []
    with observe_calls(lambda: None, lambda r, e, t: metered.append(r)):
        run_m4("t", "q", corpus, max_searches=0)
    assert len(systems) == 1 and "audit an evidence set" not in systems[0]  # relations only
    assert len(metered) == 1  # and it was metered
