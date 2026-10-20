import copy

import pytest

from vera.m6.comparative_eval import apply_success_rule, evaluate_both
from vera.m6.m6_runner import run_m6
from vera.m7.fixture import (BASE_TAG, ENG_TAG, QUESTION, ScriptedJudge, build_corpus, build_engineered,
                             build_fixture_report, build_judge)

BASE = {"response_text": f"{BASE_TAG} base", "usage": {"api_calls": 1, "tokens": 1, "cost_usd": 0.001, "latency_s": 5}}


def _dim(**kw):
    d = {"relevance": {"coverage": 0.75, "score": 4}, "grounding": {"traceable_rate": 0.95, "fabricated": 0, "score": 5},
         "reasoning_integrity": {"score": 4, "addressed": 4}, "auditability": {"score": 4},
         "cost_latency": {"score": 3}}
    for k, v in kw.items():
        d[k] = {**d[k], **v}
    return d


BASE_D = _dim(relevance={"coverage": 0.30}, reasoning_integrity={"addressed": 1}, grounding={"traceable_rate": 0.1})


def test_all_required_pass():
    assert apply_success_rule(_dim(), BASE_D)["success"]


@pytest.mark.parametrize("override,failed", [
    (dict(relevance={"coverage": 0.55}), "relevance"),                       # below 60%
    (dict(relevance={"coverage": 0.40}), "relevance"),                       # <15pp over baseline 0.30? 10pp
    (dict(grounding={"traceable_rate": 0.89}), "grounding"),
    (dict(grounding={"fabricated": 1}), "grounding"),                        # hard fail
    (dict(reasoning_integrity={"score": 3}), "reasoning_integrity"),
    (dict(reasoning_integrity={"addressed": 2}), "reasoning_integrity"),     # only +1
    (dict(auditability={"score": 3}), "auditability"),
    (dict(cost_latency={"score": 2}), "cost_latency"),
])
def test_each_required_dimension_failing_fails_the_case(override, failed):
    r = apply_success_rule(_dim(**override), BASE_D)
    assert not r["success"] and failed in r["failed"]


def test_relevance_exactly_15pp_passes():
    base = _dim(relevance={"coverage": 0.45}, reasoning_integrity={"addressed": 1})
    assert apply_success_rule(_dim(relevance={"coverage": 0.60}), base)["checks"]["relevance"]["pass"]


def test_winning_four_of_five_does_not_rescue_a_failed_required_dimension():
    # Old rule ("wins >=3 of 5") would call this a success; the new rule must not.
    corpus = build_corpus()
    eng = build_engineered(corpus)
    eng["claims"][0]["citations"] = ["GHOST"]   # fabricated citation, everything else fine
    judge = build_judge()
    ev = evaluate_both(eng, BASE, corpus, QUESTION, judge=judge)
    assert ev["winner_count"]["engineered"] >= 3
    assert ev["outcome"] == "failed_case" and ev["overall_winner"] == "none"
    assert "grounding" in ev["failed_dimensions"]


def test_fixture_report_succeeds_and_carries_provenance():
    ev = build_fixture_report()["evaluation"]
    assert ev["outcome"] == "success" and ev["overall_winner"] == "engineered"
    assert set(ev["dimension_scores"]) == {"relevance", "grounding", "reasoning_integrity", "cost_latency", "auditability"}
    assert "Partial blinding" in ev["scoring_provenance"]["blinding"]
    assert ev["scoring_provenance"]["judge_family"] != "openai"
    assert len(ev["frozen_requirements_hash"]) == 16
    for d in ev["dimension_scores"].values():
        assert d["rationale"] and d["baseline_rationale"]


def test_over_budget_engineered_fails_case():
    corpus = build_corpus()
    eng = build_engineered(corpus)
    eng["usage"]["cost_usd"] = 3.0
    ev = evaluate_both(eng, BASE, corpus, QUESTION, judge=build_judge())
    assert ev["outcome"] == "failed_case" and "cost_latency" in ev["failed_dimensions"]


def test_reviewer_override_recorded_as_disagreement():
    corpus = build_corpus()
    ev = evaluate_both(build_engineered(corpus), BASE, corpus, QUESTION, judge=build_judge(),
                       reviewer_overrides={"relevance": {"engineered": 3, "reason": "req_rct only partly covered"}})
    d = ev["scoring_provenance"]["disagreements"]
    assert d and d[0]["judge"] == 5 and d[0]["final"] == 3 and ev["dimension_scores"]["relevance"]["engineered"] == 3


# ---- runner
def _run():
    corpus = build_corpus()
    return {"run_id": "r1", "question": QUESTION, "engineered": build_engineered(corpus), "corpus": corpus}


def _llm(q, m):
    return {"text": f"{BASE_TAG} answer", "prompt_tokens": 100, "completion_tokens": 100, "latency_s": 1.0}


def test_runner_refuses_without_budget_confirmation():
    with pytest.raises(RuntimeError, match="not confirmed by R1"):
        run_m6(_run(), budget_confirmed=False, judge=build_judge(), llm_call=_llm, store=False)


def test_runner_end_to_end_cost_summary(monkeypatch):
    rep = run_m6(_run(), budget_confirmed=True, judge=build_judge(), llm_call=_llm, store=False)
    cs = rep["cost_summary"]
    assert cs["api_calls"] == 42 + 3 + cs["breakdown"]["judge"]["api_calls"]
    assert cs["breakdown"]["judge"]["api_calls"] > 0
    assert cs["breakdown"]["baseline_3_runs"]["api_calls"] == 3
    assert cs["cost_usd"] > 0.31
    assert rep["evaluation"]["outcome"] in ("success", "failed_case")
    assert rep["engineered"]["audit"]["gate_b"] and rep["corpus"]["spans"]


def test_runner_stores_via_store_results(monkeypatch):
    saved = {}
    monkeypatch.setattr("vera.m6.m6_runner.store_results", lambda rid, b, r: saved.update(rid=rid, b=b, r=r))
    run_m6(_run(), budget_confirmed=True, judge=build_judge(), llm_call=_llm)
    assert saved["rid"] == "r1" and saved["b"]["response_text"] and saved["r"]["evaluation"]


def test_runner_judge_cap_fails_verbosely():
    from vera.m6.judge import BaseJudge, JudgeLimits

    class Tiny(BaseJudge):
        family, model = "stub", "stub"

        def _call(self, p):
            return '{"scores": {}}', 10, 10

        def _price(self, a, b):
            return 0.0
    with pytest.raises(RuntimeError, match="FAILED VERBOSELY.*cap reached"):
        run_m6(_run(), budget_confirmed=True, judge=Tiny(JudgeLimits(max_calls=1)), llm_call=_llm, store=False)


def test_runner_incomplete_usage_cannot_pass_cost():
    run = _run()
    run["engineered"]["usage"]["complete"] = False
    rep = run_m6(run, budget_confirmed=True, judge=build_judge(), llm_call=_llm, store=False)
    assert "cost_latency" in rep["evaluation"]["failed_dimensions"]
    assert rep["cost_summary"]["complete"] is False
    rep2 = run_m6(run, budget_confirmed=True, judge=build_judge(), llm_call=_llm, store=False,
                  extra_usage={"api_calls": 5, "tokens": 10, "cost_usd": 0.1, "latency_s": 30})
    assert "cost_latency" not in rep2["evaluation"]["failed_dimensions"]
