"""M6 comparative evaluation and the success rule.

SUCCESS RULE (replaces the flawed "wins >=3 of 5"; R1 + Opus review). The thresholds are NOT written here:
they are read from config/vera_eval_freeze.json (success_criteria) and config/vera_eval_run.json (the
fabrication rule) through vera.eval_config, and the human-readable "need" text is generated from the same
values. Shape of the rule, ALL required:
    Relevance           engineered coverage >= relevance_min AND >= relevance_delta above baseline
    Grounding           traceable rate >= grounding_min AND fabricated <= grounding_max_fabricated
    Reasoning Integrity engineered score >= reasoning_min AND >= reasoning_delta more objections addressed
    Auditability        engineered score >= audit_min
    Cost/Latency        engineered score >= cost_min (within budget)
  Any failure => outcome "failed_case" (a recorded result, not a bad demo).

Scoring provenance: LLM judge (different family) scores; an optional reviewer
(me/R1) may supply `reviewer_overrides` per dimension. Disagreements are listed
and the reviewer value is final. Partial blinding: engineered citations reveal
which answer is which, so the judge is not blind to origin. Disclosed in report.
Every report carries `run_fingerprint` + `effective_config` (vera.m6.fingerprint) beside the old
16-hex `frozen_requirements_hash`.
"""
from __future__ import annotations

from dataclasses import asdict

from vera.eval_config import EvalConfig, SuccessCriteria, load_eval_config
from vera.m6 import eval_rubric as R
from vera.m6 import fingerprint as FP
from vera.m6.judge import Judge

# Why: wall clock of M6 itself (not the engineered pipeline's latency budget; the values only happen to be
# equal). Recorded in the report; m6_runner enforces it. Defined here so the fingerprint layer never
# imports the runner.
RUN_DEADLINE_S = 600.0

BLINDING_NOTE = ("Partial blinding only: the engineered answer carries structured citations, so "
                 "the judge can tell which answer is which. Scores may favour the engineered answer.")
DIMENSIONS = ("relevance", "grounding", "reasoning_integrity", "cost_latency", "auditability")


def apply_success_rule(eng: dict, base: dict, success: SuccessCriteria | None = None) -> dict:
    sc = success or load_eval_config().success
    checks = {
        "relevance": {
            "pass": eng["relevance"]["coverage"] >= sc.relevance_min
            and eng["relevance"]["coverage"] - base["relevance"]["coverage"] >= sc.relevance_delta - 1e-9,
            "need": f"coverage >= {sc.relevance_min:.0%} and >= {sc.relevance_delta * 100:g} pp above baseline",
            "got": f"{eng['relevance']['coverage']:.0%} vs baseline {base['relevance']['coverage']:.0%}"},
        "grounding": {
            "pass": eng["grounding"]["traceable_rate"] >= sc.grounding_min
            and eng["grounding"]["fabricated"] <= sc.grounding_max_fabricated,
            "need": f">= {sc.grounding_min:.0%} traceable and at most {sc.grounding_max_fabricated} "
                    "fabricated citation(s) (hard fail)",
            "got": f"{eng['grounding']['traceable_rate']:.0%}, {eng['grounding']['fabricated']} fabricated"},
        "reasoning_integrity": {
            "pass": eng["reasoning_integrity"]["score"] >= sc.reasoning_min
            and eng["reasoning_integrity"]["addressed"] - base["reasoning_integrity"]["addressed"] >= sc.reasoning_delta,
            "need": f"score >= {sc.reasoning_min} and >= {sc.reasoning_delta} more objections addressed than baseline",
            "got": f"score {eng['reasoning_integrity']['score']}, addressed {eng['reasoning_integrity']['addressed']} "
                   f"vs {base['reasoning_integrity']['addressed']}"},
        "auditability": {"pass": eng["auditability"]["score"] >= sc.audit_min,
                         "need": f"score >= {sc.audit_min}", "got": f"score {eng['auditability']['score']}"},
        "cost_latency": {"pass": eng["cost_latency"]["score"] >= sc.cost_min,
                         "need": f"score >= {sc.cost_min} (within budget)",
                         "got": f"score {eng['cost_latency']['score']}"},
    }
    failed = [d for d, c in checks.items() if not c["pass"]]
    return {"checks": checks, "failed": failed, "success": not failed}


def _pricing_record(model: str) -> dict | None:
    """The exact generator pricing record in force (None, recorded as such, when there is none)."""
    from vera.pricing.config import latest_pricing_for, load_model_pricing
    rec = latest_pricing_for(model, load_model_pricing())
    return rec.model_dump(mode="json") if rec is not None else None


def _score_all(answer: dict, corpus: dict, judge: Judge, budget: dict, seed: int) -> dict:
    return {"relevance": R.score_relevance(answer, corpus, judge),
            "grounding": R.score_grounding(answer, corpus, judge),
            "reasoning_integrity": R.score_reasoning(answer, corpus, judge),
            "cost_latency": R.score_cost_latency(answer, budget),
            "auditability": R.score_auditability(answer, corpus, seed=seed)}


def _reconcile(scores: dict, overrides: dict | None) -> list[dict]:
    notes = []
    for dim, ov in (overrides or {}).items():
        for side in ("engineered", "baseline"):
            if side in ov and dim in scores[side] and ov[side] != scores[side][dim]["score"]:
                notes.append({"dimension": dim, "side": side, "judge": scores[side][dim]["score"],
                              "reviewer": ov[side], "final": ov[side],
                              "reason": ov.get("reason", "(no reason given)")})
                scores[side][dim]["score"] = ov[side]
                scores[side][dim]["reconciled"] = True
    return notes


def evaluate_both(engineered: dict, baseline: dict, evidence_corpus: dict, question: str,
                  *, judge: Judge, budget: dict | None = None, seed: int = 0,
                  reviewer_overrides: dict | None = None, config: EvalConfig | None = None,
                  judge_selection: dict | None = None, code_ver: dict | None = None,
                  generator_model: str | None = None, run_inputs: dict | None = None) -> dict:
    cfg = config or load_eval_config()
    configured = asdict(cfg.budget)
    # An empty/None budget means "use the configured one" (review F15), so it is labelled run_config.
    budget_source = "run_config" if (not budget or dict(budget) == configured) else "caller_override"
    budget = dict(budget) if budget else configured
    scores = {"engineered": _score_all(engineered, evidence_corpus, judge, budget, seed),
              "baseline": _score_all(baseline, evidence_corpus, judge, budget, seed)}
    disagreements = _reconcile(scores, reviewer_overrides)
    rule = apply_success_rule(scores["engineered"], scores["baseline"], cfg.success)
    per_dim = {}
    for d in DIMENSIONS:
        e, b = scores["engineered"][d]["score"], scores["baseline"][d]["score"]
        per_dim[d] = {"engineered": e, "baseline": b,
                      "winner": "engineered" if e > b else "baseline" if b > e else "tie",
                      "rationale": scores["engineered"][d]["rationale"],
                      "baseline_rationale": scores["baseline"][d]["rationale"]}
    model = generator_model or cfg.generator_model  # the baseline model actually used (review F2)
    eff = FP.effective_run_config(cfg, judge, budget=budget, budget_source=budget_source, seed=seed,
                                  pricing_record=_pricing_record(model),
                                  judge_selection=judge_selection, corpus=evidence_corpus,
                                  generator_model=model, scored_question=question,
                                  reviewer_overrides=reviewer_overrides, run_inputs=run_inputs)
    return {
        "question": question,
        "dimension_scores": per_dim,
        "detail": scores,
        "winner_count": {s: sum(v["winner"] == s for v in per_dim.values())
                         for s in ("engineered", "baseline", "tie")},
        "success_rule": rule,
        # outcome is decided by the rule, NOT by dimension wins
        "overall_winner": "engineered" if rule["success"] else "none",
        "outcome": "success" if rule["success"] else "failed_case",
        "failed_dimensions": rule["failed"],
        "budget": budget, "budget_source": budget_source,
        "frozen_requirements_hash": R.freeze_hash(evidence_corpus),  # old 16-hex hash, unchanged
        "run_fingerprint": FP.run_fingerprint(eff, code_ver=code_ver),
        "effective_config": eff,
        "scoring_provenance": {
            "judge_family": judge.family, "judge_model": judge.model,
            "judge_provider": getattr(judge, "provider", None),
            "judge_same_family_override": getattr(judge, "same_family_override", False),
            "judge_model_source": getattr(judge, "model_source", None),
            "judge_family_source": getattr(judge, "family_source", None),
            "judge_selection": judge_selection,
            "judge_measurement": {"usage": judge.usage.as_dict(),
                                  "calls": getattr(judge.usage, "log", [])},
            "reviewer_overrides_applied": bool(reviewer_overrides),
            "disagreements": disagreements, "blinding": BLINDING_NOTE},
    }
