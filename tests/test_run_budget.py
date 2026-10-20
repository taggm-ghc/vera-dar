"""RunBudget, ledger adapters and usage completeness (S1). Fakes only."""
import pytest

import vera.gate_a as gate_a
from vera.cost_ledger import CostLedger
from vera.m3.llm import CallLog, CallRecord
from vera.m6 import eval_rubric as R
from vera.m6.adapters import build_engineered
from vera.m6.judge import BudgetExceeded
from vera.pipeline_llm import LLMResult, call, observe_calls
from vera.run_budget import (CallObs, RunBudget, RunBudgetExceeded, attach_usage, engineered_usage,
                             obs_from_m2_ledger, obs_from_m3_log)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def mk(cap_usd=1.0, cap_s=600.0):
    c = Clock()
    b = RunBudget(cap_usd, cap_s, clock=c)
    b.start()
    return b, c


def ob(stage="m2", act="search", cost=0.01, known=True, ok=True, err="", tokens=5):
    return CallObs(stage, act, "m", tokens, cost, 0.1, ok, known, err)


def test_subclass_of_judge_budget_exceeded():
    assert issubclass(RunBudgetExceeded, BudgetExceeded)


def test_bad_caps_and_unstarted_clock_fail_verbosely():
    with pytest.raises(ValueError, match="positive caps"):
        RunBudget(0, 10)
    with pytest.raises(RuntimeError, match="start\\(\\) was never called"):
        RunBudget(1, 1).elapsed_s()


def test_cost_trip_message_is_verbose_and_sticky():
    b, _ = mk(cap_usd=0.05)
    b.record(ob("m3", "extract", cost=0.06))
    with pytest.raises(RunBudgetExceeded) as ei:
        b.check("m45", "relations")
    msg = str(ei.value)
    for needle in ("cost", "m45", "relations", "cap=$0.0500", "m3/extract", "No new call was started"):
        assert needle in msg, f"{needle!r} missing from: {msg}"
    assert b.tripped
    b.obs.clear()  # even if spend vanished, a tripped budget stays tripped
    with pytest.raises(RunBudgetExceeded):
        b.check("m5", "other")


def test_time_trip_with_fake_clock():
    b, c = mk(cap_s=60)
    b.check("m2", "search")
    c.t += 61
    with pytest.raises(RunBudgetExceeded, match="time"):
        b.check("m3", "x")
    assert b.remaining_s() == 0.0


def test_attach_live_counted_and_detach():
    b, _ = mk(cap_usd=0.5)
    b.attach_live("m2", lambda: 0.6)
    assert b.spent_usd() == pytest.approx(0.6)
    with pytest.raises(RunBudgetExceeded, match="live:m2"):
        b.check("m2", "fetch")
    b2, _ = mk()
    b2.attach_live("x", lambda: 0.2)
    with pytest.raises(ValueError):
        b2.attach_live("x", lambda: 0.1)
    b2.detach_live("x")
    assert b2.spent_usd() == 0
    with pytest.raises(KeyError, match="not attached"):
        b2.detach_live("x")


def test_remaining_usd():
    b, _ = mk(cap_usd=1.0)
    b.record(ob(cost=0.25))
    assert b.remaining_usd() == pytest.approx(0.75)


def test_observer_for_records_success_failure_and_unpriced():
    b, _ = mk()
    before, after = b.observer_for("m45", "m4m5")
    with observe_calls(before, after):
        call(lambda s, u: LLMResult({}, "gpt", 100, 0.002), "s", "u")
        call(lambda s, u: LLMResult({}, "gpt", 100, 0.0), "s", "u")  # tokens but cost 0
        with pytest.raises(RuntimeError):
            call(lambda s, u: (_ for _ in ()).throw(RuntimeError("boom")), "s", "u")
    ok, unpriced, failed = b.obs
    assert ok.cost_known and ok.cost_usd == 0.002 and ok.stage == "m45"
    assert not unpriced.cost_known and "no pricing" in unpriced.error
    assert not failed.ok and not failed.cost_known and "boom" in failed.error


def test_observer_budget_trip_is_not_swallowed_by_stage_code():
    b, _ = mk(cap_usd=0.01)
    b.record(ob(cost=0.02))
    before, after = b.observer_for("m45", "relations")
    with observe_calls(before, after):
        with pytest.raises(RunBudgetExceeded):
            call(lambda s, u: LLMResult({}), "s", "u")


def test_gate_a_swallows_inside_check_but_sticky_trip_still_stops_run():
    """Reviewer finding (gate_a.py _score_batch catches every Exception): a check placed INSIDE the
    llm_call turns into 'deferred candidates'. Prove it, and prove the sticky trip is the safety net."""
    b, _ = mk(cap_usd=0.01)
    b.record(ob(cost=0.02))

    def inside_check_llm_call(messages):
        b.check("m2", "llm_gate_a")  # WRONG place: raises RunBudgetExceeded inside gate_a's try/except
        raise AssertionError("unreachable")
    rows = gate_a.score_candidates("q", [{"url": "http://example.invalid/a", "title": "t", "snippet": "s"}],
                                   inside_check_llm_call)
    assert rows[0]["gate_a_decision"] == "defer"  # swallowed: looks like a normal deferred candidate
    assert b.tripped                              # but the trip was recorded...
    with pytest.raises(RunBudgetExceeded):        # ...and re-raises OUTSIDE gate_a, at the next boundary
        b.raise_if_tripped()


def test_check_outside_gate_a_stops_before_any_call():
    b, _ = mk(cap_usd=0.01)
    b.record(ob(cost=0.02))
    called = []
    with pytest.raises(RunBudgetExceeded):
        b.check("m2", "llm_gate_a")  # caller checks first, outside score_candidates
        gate_a.score_candidates("q", [], lambda m: called.append(1))
    assert called == []


# ---------------------------------------------------------------- ledger / log adapters
def test_obs_from_m2_ledger_rules():
    led = CostLedger()
    led.record(kind="search", provider="arxiv", cost_usd=0.008)
    led.record(kind="search", provider="arxiv", cost_usd=0.0, ok=False, detail="timeout")
    led.record(kind="llm_gate_a", provider="openai:x", cost_usd=0.001,
               units={"prompt_tokens": 10, "completion_tokens": 5})
    led.record(kind="llm_gate_a", provider="openai:x", cost_usd=0.0, ok=False, detail="boom")
    led.record(kind="llm_gate_a", provider="openai:x", cost_usd=0.0, cost_known=False)
    led.record(kind="fetch", provider="http", cost_usd=0.0)
    o = obs_from_m2_ledger(led)
    assert [x.cost_known for x in o] == [True, True, True, False, False, True]
    assert o[2].tokens == 15 and all(x.stage == "m2" for x in o)


def test_obs_from_m3_log_rules():
    log = CallLog()
    log.add(CallRecord("extract", "m", 10, 5, 0.001, 0.2))
    log.add(CallRecord("appraise", "m", 10, 5, 0.0, 0.2, error="no pricing record; cost unknown"))
    log.add(CallRecord("extract", "m", 0, 0, 0.0, 0.2, ok=False, error="RuntimeError('x')"))
    o = obs_from_m3_log(log)
    assert [x.cost_known for x in o] == [True, False, False]
    assert o[0].tokens == 15 and o[2].ok is False and o[0].stage == "m3"


# ---------------------------------------------------------------- completeness
def full_obs():
    return [ob("m2", "search", 0.01), ob("m2", "fetch", 0.0), ob("m3", "extract", 0.02), ob("m45", "m4m5", 0.03)]


def test_complete_usage_totals():
    u = engineered_usage(full_obs(), wall_clock_s=12.3456)
    assert u["complete"] is True and u["incomplete_reasons"] == []
    assert u["cost_usd"] == pytest.approx(0.06) and u["api_calls"] == 3  # fetch is not billable
    assert u["latency_s"] == 12.346 and u["by_stage"]["m3"]["calls"] == 1


def test_missing_stage_meter_is_incomplete_with_reason():
    u = engineered_usage([o for o in full_obs() if o.stage != "m3"], wall_clock_s=1)
    assert u["complete"] is False and any("stage m3 has no meter" in r for r in u["incomplete_reasons"])


def test_unknown_cost_is_incomplete_with_reason():
    obs = full_obs() + [ob("m45", "m4m5", 0.0, known=False, ok=False, err="RuntimeError('x')")]
    u = engineered_usage(obs, wall_clock_s=1)
    assert u["complete"] is False and any("m45/m4m5: cost unknown (RuntimeError" in r for r in u["incomplete_reasons"])


def test_empty_obs_and_bad_wall_clock():
    u = engineered_usage([], wall_clock_s=0)
    assert u["complete"] is False and len(u["incomplete_reasons"]) == 3
    with pytest.raises(ValueError, match="wall_clock_s"):
        engineered_usage(full_obs(), wall_clock_s=-1)
    with pytest.raises(ValueError, match="wall_clock_s"):
        engineered_usage(full_obs(), wall_clock_s=None)


# ---------------------------------------------------------------- the build_engineered hole
M5 = {"final_response": "r", "claims": [], "gate_c_trace": [], "gate_c_decision": "adequate"}
M3 = {"sources": []}
BUDGET = {"cost_usd": 1.0, "latency_s": 600.0}


def test_known_hole_build_engineered_drops_complete_and_scores_five():
    """Documents the measured defect: via build_engineered alone, incomplete usage scores 5."""
    eng = build_engineered(M5, M3, usage={"api_calls": 1, "tokens": 1, "cost_usd": 0.01, "latency_s": 1.0,
                                          "complete": False})
    if "complete" in eng["usage"]:
        pytest.skip("adapters.build_engineered now keeps usage.complete; the hole is closed upstream")
    assert R.score_cost_latency(eng, BUDGET)["score"] == 5


def test_attach_usage_keeps_incomplete_flag_so_it_scores_one_never_five():
    eng = build_engineered(M5, M3, usage={"cost_usd": 0.0, "latency_s": 0.0})
    usage = engineered_usage([ob("m2", "search", 0.01)], wall_clock_s=1.0)  # m3/m45 unmetered
    assert usage["complete"] is False
    attach_usage(eng, usage)
    assert eng["usage"]["complete"] is False
    out = R.score_cost_latency(eng, BUDGET)
    assert out["score"] == 1 and out["unverified"] is True


def test_attach_usage_complete_scores_from_numbers():
    eng = build_engineered(M5, M3)
    attach_usage(eng, engineered_usage(full_obs(), wall_clock_s=10))
    assert eng["usage"]["complete"] is True and R.score_cost_latency(eng, BUDGET)["score"] == 5


def test_attach_usage_refuses_usage_without_explicit_complete():
    for bad in ({"cost_usd": 0.01, "latency_s": 1}, {"cost_usd": 0.01, "latency_s": 1, "complete": 1}):
        with pytest.raises(ValueError, match="explicit boolean 'complete'"):
            attach_usage({}, bad)
    with pytest.raises(TypeError):
        attach_usage({}, None)
    with pytest.raises(ValueError, match="lacks"):
        attach_usage({}, {"complete": True, "cost_usd": None, "latency_s": 1})
