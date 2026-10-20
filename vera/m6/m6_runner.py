"""M6 orchestration: baseline -> five-dimension scoring -> success rule -> store.

Usage (live; spends money, writes prod Postgres):
    python -m vera.m6.m6_runner --run-id <id> --budget-confirmed

Reads  the M1-M5 tables via vera.m6.adapters.load_from_db (final answer, claims, spans,
       appraisals, sources, Gate C, relations) + config/vera_eval_freeze.json (frozen
       requirements and objections).
Writes vera_vjay.runs: baseline_response (TEXT), eval_metrics (JSONB), evaluated_at
(columns added by sql/m6_eval.sql; M1 owns the table itself).

Bounds: judge caps (calls/tokens/cost), 3 baseline calls, a wall-clock deadline.
Any limit hit => verbose failure (exception text is the report); nothing retried.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from dataclasses import asdict

from vera.cost_ledger import CostLedger
from vera.eval_config import EvalConfig, family_for, load_eval_config
from vera.m6.baseline_collection import collect_baseline
from vera.m6.comparative_eval import RUN_DEADLINE_S, evaluate_both
from vera.m6.judge import (BudgetExceeded, Judge, JudgeConfigError, JudgeLimits, build_default_judge,
                           preflight_judge)

# Budget and generator model come from config/vera_eval_run.json via vera.eval_config (one source);
# RUN_DEADLINE_S (M6's own wall clock) is defined in comparative_eval with its why-comment.


def _fail(msg: str):
    raise RuntimeError("M6 FAILED VERBOSELY: " + msg)


def load_run(run_id=None) -> dict:
    from vera.m6.adapters import load_from_db
    try:
        return load_from_db(run_id)
    except Exception as e:
        _fail(f"cannot assemble run from vera_vjay ({type(e).__name__}: {e}). Likely M1-M5 tables "
              "not applied or run not finalized (blocker owner: R7a for schema, R1 for go-ahead).")


def store_results(run_id, baseline: dict, report: dict) -> None:
    from sqlalchemy import text
    from vera.db import get_session
    with get_session() as s:
        n = s.execute(text("UPDATE vera_vjay.runs SET baseline_response = :b, "
                           "eval_metrics = CAST(:m AS jsonb), evaluated_at = now() WHERE run_id::text = :id"),
                      {"b": baseline["response_text"], "m": json.dumps(report, default=str),
                       "id": str(run_id)}).rowcount
        if n != 1:
            s.rollback()
            _fail(f"expected to update exactly 1 vera_vjay.runs row for run_id={run_id!r}, got {n}.")
        s.commit()


def build_report(run: dict, engineered: dict, baseline: dict, evaluation: dict,
                 judge_usage: dict, corpus: dict) -> dict:
    eu, bu = engineered.get("usage", {}), baseline["all_runs_usage"]
    total = {
        "api_calls": (eu.get("api_calls") or 0) + bu["api_calls"] + judge_usage["api_calls"],
        "cost_usd": round((eu.get("cost_usd") or 0) + bu["cost_usd"] + judge_usage["cost_usd"], 6),
        "breakdown": {"engineered_m1_m5": eu, "baseline_3_runs": bu, "judge": judge_usage},
        "complete": eu.get("complete") is not False,
        "note": "engineered usage is M1-M5 self-reported (complete=false means M2-M4 spend missing); judge is a different provider.",
    }
    return {"run_id": run.get("run_id"), "question": run["question"],
            "engineered": {k: engineered.get(k) for k in ("response_text", "claims", "usage", "audit")},
            "baseline": {"response_text": baseline["response_text"], "usage": baseline["usage"],
                         "model": baseline["model"], "method": baseline["method"]},
            "corpus": {"requirements": corpus["requirements"], "objections": corpus["objections"],
                       "spans": corpus["spans"], "sources": corpus["sources"]},
            "evaluation": evaluation, "cost_summary": total,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def run_m6(run: dict, *, budget_confirmed: bool, judge: Judge | None = None,
           llm_call=None, budget: dict | None = None, store: bool = True,
           reviewer_overrides: dict | None = None, model: str | None = None,
           extra_usage: dict | None = None, config: EvalConfig | None = None, preflight: bool = True,
           preflight_http_get=None, judge_selection: dict | None = None,
           ledger: CostLedger | None = None) -> dict:
    # A caller that already loaded and validated the config (run_mvp) passes it: it is reused for EVERYTHING
    # below, never re-read, so a file edited mid-run cannot change the scoring (review F8).
    cfg = config or load_eval_config()  # raises EvalConfigError listing every problem; nothing is spent
    if not budget_confirmed:
        _fail(f"cost/latency budget not confirmed by R1 (configured ${cfg.budget.cost_usd:.2f} / "
              f"{cfg.budget.latency_s:.0f} s in config/vera_eval_run.json). "
              "Pass --budget-confirmed once R1 has locked it. Blocker owner: R1.")
    model = model or cfg.generator_model
    family_for(model, cfg)  # fail closed BEFORE any spend: an unmapped override model cannot be fingerprinted
    t0 = time.monotonic()
    engineered, corpus = run["engineered"], run["corpus"]
    if extra_usage:
        from vera.m6.adapters import merge_usage
        merge_usage(engineered, extra_usage)
    ledger = ledger or CostLedger()
    if judge_selection:
        # Truthful state (review F10): the caller selected the judge via the live listing and a canary.
        lst = judge_selection.get("listing") or {}
        preflight_record = {"status": "performed_by_caller",
                            "reason": "the caller's judge selection ran the live listing and a canary before M6",
                            "listing_status": lst.get("status"), "listing_checked_at": lst.get("checked_at"),
                            "canary_spend": judge_selection.get("canary_spend"),
                            "chosen": [judge_selection.get("provider"), judge_selection.get("model")]}
    else:
        preflight_record = {"status": "skipped", "reason": "caller supplied the judge or preflight=False"}
    if judge is None:
        judge = build_default_judge(JudgeLimits(), ledger=ledger, config=cfg)  # raises if no usable judge
        if preflight:  # listing check, no tokens, BEFORE any baseline spend
            preflight_record = preflight_judge(judge, http_get=preflight_http_get)
    baseline = collect_baseline(run["question"], model, llm_call=llm_call)
    base_answer = {"response_text": baseline["response_text"], "usage": baseline["usage"]}
    try:
        evaluation = evaluate_both(engineered, base_answer, corpus, run["question"],
                                   judge=judge, budget=budget, config=cfg,
                                   reviewer_overrides=reviewer_overrides, judge_selection=judge_selection,
                                   generator_model=model, run_inputs=run.get("effective_inputs"))
    except BudgetExceeded as e:
        _fail(f"{e} | elapsed {time.monotonic()-t0:.0f}s. Baseline already collected "
              f"(3 runs, ${baseline['all_runs_usage']['cost_usd']}).")
    if time.monotonic() - t0 > RUN_DEADLINE_S:
        _fail(f"wall-clock deadline {RUN_DEADLINE_S}s exceeded after scoring; results not stored.")
    evaluation["scoring_provenance"]["preflight"] = preflight_record
    report = build_report(run, engineered, baseline, evaluation, judge.usage.as_dict(), corpus)
    report["baseline"]["response_text"] = baseline["response_text"]
    report["judge_cost_ledger"] = ledger.as_dicts()  # empty when a caller-supplied judge has no ledger
    if store:
        store_results(run.get("run_id"), baseline, report)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--budget-confirmed", action="store_true")
    ap.add_argument("--no-store", action="store_true")
    ap.add_argument("--extra-usage", help="JSON file: M2-M4 totals {api_calls,tokens,cost_usd,latency_s}")
    a = ap.parse_args(argv)
    try:
        run = load_run(a.run_id)
        extra = json.load(open(a.extra_usage)) if a.extra_usage else None
        cfg = load_eval_config()
        if not a.budget_confirmed:  # refuse before the listing/canary spend; run_m6 would refuse too
            _fail("cost/latency budget not confirmed by R1. Pass --budget-confirmed once R1 has locked it.")
        # Review F11: the same pool selection (live listing, family filter, canary, same-family refusal, no
        # env overrides) as the MVP path; the legacy env judge is not reachable from this CLI.
        from vera import judge_select as js
        ledger = CostLedger()
        sel = js.select_judge(os.environ, http_get=js.default_http_get, limits=JudgeLimits(), ledger=ledger,
                              judge_transport=js.default_judge_transport, config=cfg)
        rep = run_m6(run, budget_confirmed=a.budget_confirmed, store=not a.no_store, extra_usage=extra,
                     judge=sel.judge, judge_selection=sel.as_record(), config=cfg, ledger=ledger)
    except (RuntimeError, JudgeConfigError) as e:
        print(e, file=sys.stderr)
        return 2
    ev = rep["evaluation"]
    print(f"outcome={ev['outcome']} failed={ev['failed_dimensions']} "
          f"spend=${rep['cost_summary']['cost_usd']} calls={rep['cost_summary']['api_calls']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
