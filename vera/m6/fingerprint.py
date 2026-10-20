"""Run fingerprint: a full sha256 over the canonical JSON of the whole effective run configuration.

This is the WIDER fingerprint (goal #2). It is a NEW field, `evaluation.run_fingerprint`, and does not alter
the old 16-hex `frozen_requirements_hash` (requirements + objections only), which is kept unchanged for old
reports and for M7. It covers both config files (with their own sha256), the effective budget, the generator
(model, family, baseline settings, the exact pricing record), the judge (identity, how model and family were
decided, call settings, prices, selection), the scoring module sources and the pipeline constants as of the M6
code. The code version (git commit, dirty flag) is recorded BESIDE the hash, not in it (plan D3).

Two hashes, two meanings: comparable = same config_sha256; reproducible = same config_sha256 plus the same
code_version.commit with dirty == false.

Known limit (plan 1b): M4/M5 run in an earlier process, so the pipeline values are those of the M6 code; run
M2-M6 from one commit. Canonicalisation is NOT RFC 8785 (accepted divergence): sorted keys, compact
separators, UTF-8, NaN/Infinity rejected.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path

from vera.eval_config import EvalConfig, family_for
from vera.m6.baseline_collection import MAX_OUTPUT_TOKENS, N_RUNS
from vera.m6.eval_rubric import AUDIT_K
from vera.m6.judge import JUDGE_MAX_TOKENS

# Why: names the serialisation so a verifier knows how to recompute the hash (stored beside it).
CANON = "python-json-sorted-compact-v1"
# Why: versions the layout of effective_config; a layout change is a visible schema bump, not a silent one.
SCHEMA = "vera-run-fingerprint-v2"  # v2 (review F1-F3): run_inputs section, used M4 values, narrower selection
_ROOT = Path(__file__).resolve().parent.parent.parent
# Why: the scoring logic and judge prompts live in these two modules; hashing their sources puts a prompt or
# cut-point edit inside the fingerprint (conservative: a pure refactor reads as "different", never as "same").
SCORING_MODULES = ("vera/m6/eval_rubric.py", "vera/m6/comparative_eval.py")
# Why: bound the best-effort git calls so a hung git can never stall an evaluation.
GIT_TIMEOUT_S = 5


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")  # NaN/Inf -> ValueError -> verbose failure


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def module_sha256(rel: str, root: Path = _ROOT) -> str:
    return sha256_hex((root / rel).read_bytes().replace(b"\r\n", b"\n"))


def pipeline_snapshot(corpus: dict | None = None, *, m4_used: dict | None = None) -> dict:
    """Named pipeline constants as of the M6 code (read, never edited, from the M2-M5 modules).

    m4_used = {"max_searches": int, "sub_questions": list} are the values the run ACTUALLY used for M4 (passed
    in by the runner; review F1). Without them the code defaults are recorded and labelled as such, never as
    "effective"."""
    if m4_used is not None and not {"max_searches", "sub_questions"} <= set(m4_used):
        raise ValueError("m4_used must carry both 'max_searches' and 'sub_questions' (fail verbosely, F1)")
    from vera import gate_a, pipeline_llm
    from vera.m3 import appraisal_rubric as ar
    from vera.m3.gate_b import DEFAULT_POLICY
    from vera.m4 import common, context_builder, gate_c, m4_runner
    from vera.m5 import claim_verification as cv
    return {
        "gate_a": {"model": gate_a.MODEL, "weights": dict(gate_a.WEIGHTS), "fetch_threshold": gate_a.FETCH_THRESHOLD,
                   "reject_threshold": gate_a.REJECT_THRESHOLD, "uncertainty_value": dict(gate_a.UNCERTAINTY_VALUE),
                   "batch_size": gate_a.BATCH_SIZE, "max_snippet_chars": gate_a.MAX_SNIPPET_CHARS},
        "gate_b_policy": asdict(DEFAULT_POLICY), "rubric_version": ar.RUBRIC_VERSION,
        "appraisal_weights": dict(ar.WEIGHTS), "evidence_window": list(ar.EVIDENCE_WINDOW),
        "out_of_window_indirectness_cap": ar.OUT_OF_WINDOW_INDIRECTNESS_CAP,
        "min_year": {"gate_c": gate_c.MIN_YEAR, "claim_verification": cv.MIN_YEAR},
        "gate_c": {"min_independent_sources": gate_c.MIN_INDEPENDENT_SOURCES,
                   "insufficient_below": gate_c.INSUFFICIENT_BELOW, "max_llm_gaps": gate_c.MAX_LLM_GAPS},
        "claim_verification": {"min_appraisal": cv.MIN_APPRAISAL, "min_overlap": cv.MIN_OVERLAP,
                               "max_claims": cv.MAX_CLAIMS, "hedges": list(cv.HEDGES)},
        "m4": {"max_units": common.MAX_UNITS, "unit_chars": context_builder.UNIT_CHARS,
               "max_searches": m4_used["max_searches"] if m4_used else m4_runner.DEFAULT_MAX_SEARCHES,
               "sub_questions": list(m4_used["sub_questions"]) if m4_used else common.sub_questions(corpus or {}),
               "values_source": "run (as used by M4)" if m4_used else "code defaults (runner did not supply)"},
        "llm": {"max_tokens": pipeline_llm.MAX_TOKENS, "max_json_retries": pipeline_llm.MAX_JSON_RETRIES,
                "fallback_model": pipeline_llm.FALLBACK_MODEL},
    }


def _judge_block(judge) -> dict:
    g = lambda name, default=None: getattr(judge, name, default)  # noqa: E731 - stubs lack most attributes
    limits = g("limits")
    return {"provider": g("provider"), "model": g("model"), "model_source": g("model_source"),
            "family": g("family"), "family_source": g("family_source"),
            "same_family_override": bool(g("same_family_override", False)),
            "call": {"temperature": 0, "max_tokens": JUDGE_MAX_TOKENS, "response_format": "json_object"},
            "limits": asdict(limits) if is_dataclass(limits) else None,
            "price_in": g("price_in"), "price_out": g("price_out"), "price_source": g("price_source")}


def selection_for_fingerprint(record: dict | None) -> dict | None:
    """Stable part of a judge-selection record: the chosen entry, the rule and where the pool came from (the
    configured pool spec is hashed in run_config.judge_settings). The live candidate list, listing timestamp,
    canary spend and refusals are volatile (any Groq catalogue change alters them) and stay in the report
    (scoring_provenance.judge_selection) BESIDE the hash, never in it (review F3)."""
    if not record:
        return None
    return {k: record.get(k) for k in ("provider", "model", "family", "family_source", "rule", "pool_source",
                                       "panel")}


def effective_run_config(cfg: EvalConfig, judge, *, budget: dict, budget_source: str, seed: int,
                         audit_k: int = AUDIT_K, pricing_record: dict | None = None,
                         judge_selection: dict | None = None, corpus: dict | None = None,
                         module_hashes: dict | None = None, generator_model: str | None = None,
                         scored_question: str | None = None, reviewer_overrides: dict | None = None,
                         run_inputs: dict | None = None) -> dict:
    """run_inputs (from the runner, review F1/F2) = {"m4": {"max_searches", "sub_questions"},
    "stage_limits": {"num_results", "max_fetch", "max_sources", "cap_usd", "cap_s"}}: the values the run
    really used. generator_model is the baseline model actually used (a model= override), scored_question the
    question actually scored, reviewer_overrides the overrides applied: all can change a result, so all are
    hashed."""
    model = generator_model or cfg.generator_model
    gen_family = family_for(model, cfg).family
    run_inputs = run_inputs or {}
    return {
        "schema": SCHEMA,
        "freeze": {"file_sha256": cfg.freeze_file_sha256, "requirements_hash16": cfg.requirements_hash16,
                   "question": cfg.question, "requirements": list(cfg.requirements),
                   "objections": list(cfg.objections), "success_criteria": asdict(cfg.success)},
        "run_config": {"file_sha256": cfg.run_file_sha256, "schema_version": cfg.schema_version,
                       "run_config_version": cfg.run_config_version,
                       "budget": {"effective": dict(budget), "source": budget_source,
                                  "configured": asdict(cfg.budget)},
                       "grounding_max_fabricated": cfg.success.grounding_max_fabricated,
                       "judge_settings": asdict(cfg.judge), "pipeline": dict(cfg.pipeline),
                       "search": json.loads(json.dumps(dict(cfg.search)))},
        "run_inputs": {"scored_question": scored_question if scored_question is not None else cfg.question,
                       "reviewer_overrides": reviewer_overrides or {},
                       "stage_limits": run_inputs.get("stage_limits"),
                       "search": run_inputs.get("search"),
                       "supplied_by_runner": bool(run_inputs)},
        "generator": {"model": model, "configured_model": cfg.generator_model, "family": gen_family,
                      "baseline": {"n_runs": N_RUNS, "max_output_tokens": MAX_OUTPUT_TOKENS,
                                   "selection": "median_by_output_length", "system_prompt": None},
                      "pricing_record": pricing_record},  # the exact record used, not the whole file
        "judge": dict(_judge_block(judge), selection=selection_for_fingerprint(judge_selection)),
        "scoring": {"seed": seed, "audit_k": audit_k,
                    "module_sha256": module_hashes or {m: module_sha256(m) for m in SCORING_MODULES}},
        "pipeline_as_of_m6_code": pipeline_snapshot(corpus, m4_used=run_inputs.get("m4")),
    }


def code_version(run=None) -> dict:
    """Best effort, recorded, never a silent default."""
    def _run(*args):
        return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True,
                              timeout=GIT_TIMEOUT_S, check=True).stdout.strip()
    try:
        commit = (run or _run)("rev-parse", "HEAD")
        dirty = bool((run or _run)("status", "--porcelain", "--", "vera", "config"))
    except Exception as e:  # noqa: BLE001
        return {"commit": None, "dirty": None, "source": f"unknown ({type(e).__name__})"}
    return {"commit": commit, "dirty": dirty, "source": "git"}


def run_fingerprint(effective: dict, *, code_ver: dict | None = None) -> dict:
    try:
        blob = canonical_json(effective)
    except (ValueError, TypeError) as e:
        raise ValueError(f"run fingerprint refused: effective config is not canonically serialisable "
                         f"({type(e).__name__}: {e}); NaN/Infinity and non-JSON values are rejected") from None
    return {"schema": SCHEMA, "canonicalization": CANON, "config_sha256": sha256_hex(blob),
            "components_sha256": {k: sha256_hex(canonical_json(v)) for k, v in effective.items()},
            "code_version": code_ver if code_ver is not None else code_version(),
            "limits": "pipeline values are as of the M6 code; run M2-M6 from one commit"}
