"""Slim live MVP runner (S4): one honest scored run, M2 -> M3 -> adapter -> M4/M5 -> M6.

    python -m vera.run_mvp --question-id N --stage {A,B,C,L1} --run-type {test,live} [--live ...]

DEFAULT IS DRY/OFFLINE: without --live this module runs the offline pre-flight only (no .env load, no
DB, no provider call) and prints what it would do. A live run needs the explicit --live flag AND
--budget-confirmed AND every pre-flight check to pass (checked again inside run_mvp before any
write or spend). Nothing here is executed live by the tests; every seam is injected through Deps.

Rules (plan vera-plan-slim-live-mvp.md sections 2, 4, 13, 15):
  * one RunBudget covers M2..M5 (cost and wall-clock); its checks run OUTSIDE Gate A (Gate A swallows
    exceptions) and outside any other code that catches broad exceptions;
  * after build_engineered the runner calls attach_usage (build_engineered drops usage.complete);
  * unknown cost anywhere means incomplete usage: stop BEFORE M6 with exit 4 (checked after M2 and after
    M3 when M6 is planned, so known-incomplete usage does not keep spending);
  * the judge is chosen once by pre-flight (select_judge) and fixed for the run; the selection is
    recorded; the judge transport is always passed explicitly;
  * every stage writes as the rw account; its identity is asserted before the first write;
  * failures are verbose: the report holds stage, cause and traceback (scrubbed), and a report-write
    failure is raised only AFTER the original error was logged.
Exit codes: 0 closed_ok, 1 preflight_failed, 2 stage failed, 3 budget stop, 4 usage incomplete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from vera.cost_ledger import CostLedger
from vera.gate_a import GateAConfig, score_candidates
from vera.eval_config import EvalConfigError, load_eval_config
from vera.judge_select import default_http_get, default_judge_transport, scrub
from vera.m3_to_m4 import adapt_m3_to_m4
from vera.m6.adapters import build_corpus, build_engineered
from vera.m6.eval_rubric import freeze_hash
from vera.m6.judge import JudgeLimits
from vera.m6.m6_runner import run_m6
from vera.m2_runner import SCHEMA
from vera.mvp_preflight import DB_URL_ENV_RW, EXPECTED_DB_ROLE, RUN_DIR_NAME, run_preflight
from vera.pipeline_llm import observe_calls
from vera.run_budget import (RunBudget, RunBudgetExceeded, attach_usage, engineered_usage,
                             obs_from_m2_ledger, obs_from_m3_log)
from vera.search_and_fetch import search_question
from vera.selective_fetch import fetch_candidate

logger = logging.getLogger("vera")

# --- exit codes (plan section 13, S4 hand-off) ---------------------------------------------------
EXIT_OK, EXIT_PREFLIGHT, EXIT_STAGE_FAILED, EXIT_BUDGET, EXIT_USAGE_INCOMPLETE = 0, 1, 2, 3, 4

# --- named constants (accepted divergence from the hardcoded-data standard; fold into
# config/vera_eval_run.json in goal #2 at the keys shown) ----------------------------------------
# Why: research_fn is not wired (U7f), so Gate C evaluates once and never asks for a re-search.
# key pipeline.max_searches
MVP_MAX_SEARCHES = 0
# Why: gitignored report location (runner plan Q10); defined once in mvp_preflight, which also checks it.
# key paths.run_dir
RUN_DIR = RUN_DIR_NAME
REGISTER_NAME = "register.jsonl"
# Why: stage plans from plan section 2; each stage runs at most once per invocation, no resume.
STAGE_PLANS = {"A": ("m2", "m3"), "B": ("m2", "m3", "m45"), "C": ("m2", "m3", "m45", "m6"),
               "L1": ("m2", "m3", "m45", "m6")}
# Why: A/B/C are throwaway test runs; only L1 is a live run, and only L1 stores the eval.
STAGE_RUN_TYPE = {"A": "test", "B": "test", "C": "test", "L1": "live"}
STORE_EVAL_STAGES = frozenset({"L1"})
# Why: states written to the report and register (plan section 2).
STATE_PREFLIGHT_FAILED, STATE_CLOSED_OK = "preflight_failed", "closed_ok"
STATE_BUDGET_STOP, STATE_USAGE_INCOMPLETE, STATE_FAILED = "budget_stop", "usage_incomplete", "failed"
# Why: bounds the traceback text stored in a report (scrubbed first), so a report stays small.
MAX_TRACEBACK_CHARS = 4000
# Why: UTC timestamp in the name of a run-less (pre-flight failure) report so reports never collide.
RUN_TS_FORMAT = "%Y%m%dT%H%M%SZ"
# Disclosed divergences, embedded in every report (plan section 5, last paragraph).
DIVERGENCES = (
    "rw account vera_claude_code_rw for every stage (U6); identity asserted before the first write",
    "in-process M6; runs.final_response is not written (U3, RD-3)",
    "max_searches=0: research_fn not wired, Gate C evaluates once (U7f)",
    "inert year, source-type and appraisal arms (U7h)",
    "judge chosen from the live Groq listing by name patterns now held in config/vera_eval_run.json "
    "(goal #2: single source; the patterns stay UNVERIFIED against the live list); Groq price ASSUMED unless "
    "overridden (U4)",
    "temporary duplicate of store_results, and two judge listing checks (judge_select.list_models for the pool, "
    "judge.preflight_judge for the legacy env judge) (U7c, CP-R2)",
    "pipeline constants (Gate A/B/C, MIN_YEAR, sub-questions, M3/M4/M5 thresholds) stay named code constants; "
    "they are fingerprinted as of the M6 code, not yet read from the run config (plan SHOULD items, post-MVP)",
    "activity labels m4m5/subq_tag instead of the cost-plan vocabulary (CP-R3)",
    "latency = wall-clock M2 start to M5 end (RD-4)",
    "no web research re-run for the plan (section 7)",
    "PF5 recognises a GPT generator by model-id prefix (startswith 'gpt'); a differently named OpenAI-family "
    "generator would fail PF5 (fail-closed, known limitation, accepted)",
    "search is arXiv (keyless, abstract pages only); its calls are recorded as KNOWN $0 with cost_basis "
    "no_published_price_keyless: no published price was seen, but no terms text states $0 either, so this is an "
    "assumption of the same kind as the old per-call constant, not a verified price",
    "scholarly-only discovery: grey literature (industry/lab reports) is under-represented; live search can "
    "leak post-cutoff pages even with date filters, so each run stores its candidate list and sha256 (node NS)",
    "optional seed-list fallback (agent-found candidate list, frozen by sha256) is used only when arXiv fails or "
    "returns fewer than min_candidates; when used it is added to the report; its discovery cost is unmetered "
    "by VERA-DAR and non-deterministic",
    "licence gate: no declared licence for the stored text is deferred (#63); arXiv abstract pages are governed "
    "by the CC0 metadata licence only; a Gate A fetch slot can be lost to a licence defer/reject",
)


class RunConfigError(ValueError):
    """Invalid stage / run-type combination or missing live confirmation."""


class StageError(RuntimeError):
    """A stage could not produce what the next stage needs (verbose; names the stage)."""


# ------------------------------------------------------------------------------------ real seams
def _sql_current_user(eng) -> str:
    from sqlalchemy import text
    with eng.connect() as c:
        return c.execute(text("SELECT current_user")).scalar()


def _sql_question_text(eng, question_id: int) -> str:
    from sqlalchemy import text
    with eng.connect() as c:
        return c.execute(text(f"SELECT research_question FROM {SCHEMA}.questions WHERE question_id = :q"),  # nosec B608 - SCHEMA constant (m2_runner); values bound
                         {"q": question_id}).scalar()


def _sql_create_run(eng, question_id: int, question: str) -> int:
    """U1: the runner creates the run row and writes runs.question (U3, partly)."""
    from sqlalchemy import text
    with eng.begin() as c:
        rid = c.execute(text(f"INSERT INTO {SCHEMA}.runs (question_id, question) VALUES (:q, :t) "  # nosec B608 - SCHEMA constant (m2_runner); values bound
                             "RETURNING run_id"), {"q": question_id, "t": question}).scalar_one()
    return int(rid)


def _sql_store_eval(eng, run_id: int, report: dict) -> None:
    """rw UPDATE of baseline_response / eval_metrics / evaluated_at; exactly one row or fail."""
    from sqlalchemy import text
    with eng.begin() as c:
        n = c.execute(text(f"UPDATE {SCHEMA}.runs SET baseline_response = :b, "  # nosec B608 - SCHEMA constant (m2_runner); values bound
                           "eval_metrics = CAST(:m AS jsonb), evaluated_at = now() WHERE run_id = :id"),
                      {"b": report["baseline"]["response_text"], "m": json.dumps(report, default=str),
                       "id": int(run_id)}).rowcount
    if n != 1:
        raise StageError(f"store_eval: expected to update exactly 1 {SCHEMA}.runs row for run_id={run_id}, got {n}")


def _git_info(repo_root: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=15,
                              check=True).stdout.strip()
    try:
        return {"head": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except Exception as e:  # noqa: BLE001  (report it; a missing git must not hide a run)
        return {"head": None, "dirty": None, "error": f"{type(e).__name__}"}


@dataclass
class Limits:
    judge: JudgeLimits = field(default_factory=JudgeLimits)
    budget_confirmed: bool = False
    # None = the stage module's own default (run_m2 num_results, GateAConfig.max_fetch, run_m3 max_sources).
    num_results: int | None = None
    max_fetch: int | None = None
    max_sources: int | None = None
    # None = the configured budget (config/vera_eval_run.json budget, via vera.eval_config): one source.
    cap_usd: float | None = None
    cap_s: float | None = None


@dataclass
class Deps:
    """Every seam. Tests pass fakes; Deps.real() builds the live ones (only main(--live) calls it)."""
    engine_factory: Callable[[str], object]
    current_user: Callable[[object], str]
    question_text: Callable[[object, int], str]
    create_run: Callable[[object, int, str], int]
    store_eval: Callable[[object, int, dict], None]
    m2_store_factory: Callable[[object], object]
    pipeline_store_factory: Callable[[object], object]
    run_m2: Callable
    search_fn: Callable
    score_fn: Callable
    fetch_fn: Callable
    m3_client_factory: Callable[[], object]
    run_m3: Callable
    run_m4_m5: Callable
    run_m6: Callable
    preflight: Callable
    baseline_llm_call: Callable | None
    judge_transport: Callable
    http_get: Callable
    env: Mapping[str, str]
    repo_root: Path
    run_dir: Path
    git_info: Callable[[], dict]
    now: Callable[[], datetime]
    clock: Callable[[], float] = time.monotonic
    ledger: CostLedger | None = None  # pre-flight (judge canary) ledger; fresh one when None

    @classmethod
    def real(cls, env: Mapping[str, str] | None = None) -> "Deps":
        from sqlalchemy.orm import Session
        from vera.db import get_role_engine
        from vera.m2_runner import PostgresStore, run_m2
        from vera.m3.llm import OpenAIJSONClient
        from vera.m3.m3_runner import run_m3
        from vera.m4_m5_runner import run_m4_m5
        from vera.pipeline_store import PipelineStore
        env = os.environ if env is None else env
        root = Path(__file__).resolve().parent.parent
        return cls(
            engine_factory=lambda role: get_role_engine(role, env=env),
            current_user=_sql_current_user, question_text=_sql_question_text,
            create_run=_sql_create_run, store_eval=_sql_store_eval,
            m2_store_factory=lambda eng: PostgresStore(Session(eng)),
            pipeline_store_factory=PipelineStore,
            run_m2=run_m2, search_fn=search_question, score_fn=score_candidates, fetch_fn=fetch_candidate,
            m3_client_factory=OpenAIJSONClient, run_m3=run_m3, run_m4_m5=run_m4_m5, run_m6=run_m6,
            preflight=run_preflight, baseline_llm_call=None,  # None = collect_baseline's real OpenAI call
            judge_transport=default_judge_transport, http_get=default_http_get, env=env, repo_root=root,
            run_dir=root / RUN_DIR, git_info=lambda: _git_info(root), now=lambda: datetime.now(timezone.utc))


# ------------------------------------------------------------------------------------ helpers
class MeteredM3Client:
    """Wraps an M3 LLM client: budget.check BEFORE each call (RunBudgetExceeded is not an LLMError, so
    M3's `except LLMError` cannot swallow it). Spend is read from the inner client's own CallLog."""

    def __init__(self, inner, budget: RunBudget):
        self._inner, self._budget = inner, budget

    @property
    def log(self):
        return self._inner.log

    def complete_json(self, system: str, user: str, purpose: str) -> dict:
        self._budget.check("m3", purpose)
        return self._inner.complete_json(system, user, purpose)


class RelationsCapture:
    """Wraps the injected pipeline store; remembers the graph passed to save_relations (R11a) and
    forwards every call unchanged, so what is stored is not altered."""

    def __init__(self, inner):
        self._inner, self.edges, self.seen = inner, [], False

    def save_relations(self, run_id, graph):
        self.edges, self.seen = list(graph.get("edges", [])), True
        return self._inner.save_relations(run_id, graph)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _validate(stage: str, run_type: str) -> None:
    if stage not in STAGE_PLANS:
        raise RunConfigError(f"unknown stage {stage!r}; expected one of {sorted(STAGE_PLANS)}")
    if STAGE_RUN_TYPE[stage] != run_type:
        raise RunConfigError(f"stage {stage} requires --run-type {STAGE_RUN_TYPE[stage]}, got {run_type!r}")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _gate_a_audit(m2) -> list[dict]:
    """Shape matches load_from_db's gate_a rows (source_id, score, decision, rationale, url)."""
    return [{"source_id": str(c["source_id"]), "score": c.get("gate_a_score"), "decision": c.get("gate_a_decision"),
             "rationale": c.get("gate_a_rationale"), "url": c.get("url")}
            for c in m2.admitted if c.get("source_id") is not None]


def _with_versions(m3: dict, m2) -> dict:
    """build_corpus reads sources[].version (defaults 1); M3 summaries do not carry it, M2 does."""
    ver = {str(c["source_id"]): c.get("source_version") for c in m2.admitted if c.get("source_version")}
    out = dict(m3)
    for bucket in ("sources", "excluded"):
        out[bucket] = [dict(s, version=ver[str(s["source_id"])]) if str(s["source_id"]) in ver else s
                       for s in m3.get(bucket, [])]
    return out


def _budget_record(budget: RunBudget) -> dict:
    return {"caps": {"cost_usd": budget.cap_usd, "latency_s": budget.cap_s},
            "spent_usd": round(budget.spent_usd(), 6), "elapsed_s": round(budget.elapsed_s(), 3),
            "tripped": budget.tripped}


@dataclass
class RunReport:
    exit_code: int
    state: str
    data: dict
    path: Path | None = None


def _unknown_cost_reasons(budget: RunBudget) -> list[str]:
    """Observations whose cost is unknown so far. Same wording as engineered_usage's reasons, but WITHOUT its
    'stage has no meter' reasons (later stages have not run yet, so a missing meter is expected here)."""
    return [f"{o.stage}/{o.activity}: cost unknown ({o.error[:80] or 'no detail'})"
            for o in budget.obs if not o.cost_known]


def _is_conn_error(exc: BaseException) -> bool:
    """C12: driver, SQLAlchemy and safety-layer errors may carry connection data in their text."""
    return any(c.__module__.split(".")[0] in ("sqlalchemy", "psycopg2", "psycopg", "safety") for c in type(exc).__mro__)


def _err_text(exc: BaseException) -> str:
    """C12: connection-class errors are shown by type name plus code only; others keep their text."""
    if _is_conn_error(exc):
        code = getattr(exc, "code", None)
        return type(exc).__name__ + (f"[{code}]" if isinstance(code, str) and code.replace("_", "").isalnum() else "")
    return f"{type(exc).__name__}: {exc}"


def _finish(deps: Deps, rec: dict, state: str, exit_code: int) -> RunReport:
    rec.update(state=state, exit_code=exit_code, finished_at=deps.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
    # Log the original outcome BEFORE writing, so a report-write failure can never replace it.
    logger.error("run %s finished: state=%s exit=%s cause=%s", rec.get("run_id"), state, exit_code,
                 scrub(str(rec.get("error", ""))[:300], deps.env))
    name = (f"{rec['run_id']}.json" if rec.get("run_id") is not None
            else f"preflight-{deps.now().strftime(RUN_TS_FORMAT)}.json")
    text = scrub(json.dumps(rec, indent=2, default=str, ensure_ascii=False), deps.env)
    deps.run_dir.mkdir(parents=True, exist_ok=True)
    path, tmp = deps.run_dir / name, deps.run_dir / (name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)  # atomic
    line = scrub(json.dumps({"run_id": rec.get("run_id"), "run_type": rec["run_type"], "stage_plan": rec["stage_plan"],
                             "state": state, "at": rec["finished_at"]}), deps.env)
    with open(deps.run_dir / REGISTER_NAME, "a") as f:
        f.write(line + "\n")
    return RunReport(exit_code, state, rec, path)


# ------------------------------------------------------------------------------------ stages
def search_snapshot(query: str, n: int, results) -> dict:
    """Node NS: the exact candidate list this run searched and its sha256 (repeatability record)."""
    from vera.m6.fingerprint import canonical_json, sha256_hex
    cands = [dict(c) for c in results]
    return {"query": query, "num_results": n, "count": len(cands), "candidates": cands,
            "candidate_list_sha256": sha256_hex(canonical_json(cands)),
            "route": dict(getattr(results, "meta", {}) or {})}


def _stage_m2(run_id, question, eng, budget: RunBudget, deps: Deps, limits: Limits, searches: list | None = None):
    ledger, captured = CostLedger(), {}
    searches = searches if searches is not None else []

    # The budget is checked here, in wrappers OUTSIDE Gate A's llm_call (Gate A swallows exceptions).
    def search_fn(q, n, *, ledger):
        budget.check("m2", "search")
        res = deps.search_fn(q, n, ledger=ledger)
        searches.append(search_snapshot(q, n, res))  # node NS: the candidate list and its sha256
        return res

    def score_fn(q, cands, *, cfg, ledger):
        budget.check("m2", "gate_a")
        out = deps.score_fn(q, cands, cfg=cfg, ledger=ledger)
        budget.check("m2", "gate_a_done")  # catches spend Gate A made (and any swallowed trip, sticky)
        return out

    def fetch_fn(url, *, ledger):
        budget.check("m2", "fetch")
        fr = deps.fetch_fn(url, ledger=ledger)
        if fr.ok:
            captured[url] = fr.content_text  # M3 input must not depend on a re-read of reused sources
        return fr

    kw = {} if limits.num_results is None else {"num_results": limits.num_results}
    if limits.max_fetch is not None:
        kw["cfg"] = GateAConfig(max_fetch=limits.max_fetch)
    budget.check("m2", "start")
    budget.attach_live("m2", ledger.total_usd)
    try:
        m2 = deps.run_m2(question, run_id, deps.m2_store_factory(eng), ledger=ledger, search_fn=search_fn,
                         score_fn=score_fn, fetch_fn=fetch_fn, **kw)
    finally:
        budget.detach_live("m2")
        for o in obs_from_m2_ledger(ledger):
            budget.record(o)
    budget.check("m2", "end")
    if not m2.admitted:
        raise StageError("M2 admitted 0 sources (nothing for M3): rejected="
                         f"{len(m2.rejected)} deferred={len(m2.deferred)} fetch_failed={len(m2.fetch_failed)}")
    return m2, captured


def _m3_sources(m2, captured: dict) -> list[dict]:
    out, seen = [], set()
    for c in m2.admitted:
        sid = c.get("source_id")
        if sid is None or sid in seen:
            continue
        content = captured.get(c["url"])
        if not content:
            raise StageError(f"no captured content for admitted source_id={sid} url={c['url']}: "
                             "M3 would read an empty source")
        seen.add(sid)
        out.append({"source_id": sid, "content": content, "title": c.get("title"), "url": c["url"],
                    "content_hash": c.get("content_hash"), "version": c.get("source_version")})
    return out


def _stage_m3(m2, captured, question, eng, budget: RunBudget, deps: Deps, limits: Limits) -> dict:
    inner = deps.m3_client_factory()
    client = MeteredM3Client(inner, budget)
    kw = {} if limits.max_sources is None else {"max_sources": limits.max_sources}
    sources = _m3_sources(m2, captured)
    budget.check("m3", "start")
    budget.attach_live("m3", lambda: inner.log.total_cost_usd)
    try:
        # M3's own cost cap is the remaining run budget, not a second literal.
        m3 = deps.run_m3(sources, question, llm=client, engine=eng,
                         max_cost_usd=budget.remaining_usd(), **kw)
    finally:
        budget.detach_live("m3")
        for o in obs_from_m3_log(inner.log):
            budget.record(o)
    budget.check("m3", "end")
    return m3


def _stage_m45(run_id, question, m3, freeze, eng, budget: RunBudget, deps: Deps):
    budget.check("m45", "start")
    with observe_calls(*budget.observer_for("m45", "subq_tag")):
        corpus = adapt_m3_to_m4(m3, freeze["requirements"])
    budget.check("m45", "tagged")
    store = RelationsCapture(deps.pipeline_store_factory(eng))
    with observe_calls(*budget.observer_for("m45", "m4m5")):
        m45 = deps.run_m4_m5(str(run_id), question, corpus, store=store, max_searches=MVP_MAX_SEARCHES)
    budget.check("m45", "end")
    return corpus, m45, store


# ------------------------------------------------------------------------------------ the run
def run_mvp(question_id: int, stage: str, run_type: str, *, deps: Deps, limits: Limits) -> RunReport:
    _validate(stage, run_type)
    plan = STAGE_PLANS[stage]
    if not limits.budget_confirmed:
        # Enforced here too (not only in main): a programmatic caller with default Limits() must not reach
        # the engine, create_run or any spend. Refused before the engine exists, so nothing is written
        # to the DB; only the local refusal report is.
        return _finish(deps, {"run_id": None, "run_type": run_type, "stage": stage, "stage_plan": list(plan),
                              "question_id": question_id, "divergences": list(DIVERGENCES), "stages": {},
                              "error": "refused: budget_confirmed is False; M2-M5 would spend before M6 "
                                       "could refuse. Set Limits(budget_confirmed=True) after approval."},
                       STATE_PREFLIGHT_FAILED, EXIT_PREFLIGHT)
    try:
        cfg = load_eval_config()  # one validated loader: every problem listed, before any spend or write
    except EvalConfigError as e:
        return _finish(deps, {"run_id": None, "run_type": run_type, "stage": stage, "stage_plan": list(plan),
                              "question_id": question_id, "divergences": list(DIVERGENCES), "stages": {},
                              "error": f"eval config invalid: {e}"}, STATE_PREFLIGHT_FAILED, EXIT_PREFLIGHT)
    rec: dict = {"run_id": None, "run_type": run_type, "stage": stage, "stage_plan": list(plan),
                 "question_id": question_id, "divergences": list(DIVERGENCES), "stages": {},
                 "parameters": {"max_searches": MVP_MAX_SEARCHES, "num_results": limits.num_results,
                                "max_fetch": limits.max_fetch, "max_sources": limits.max_sources,
                                "budget_cap_usd": limits.cap_usd or cfg.budget.cost_usd,
                                "budget_cap_s": limits.cap_s or cfg.budget.latency_s,
                                "judge_limits": vars(limits.judge) if hasattr(limits.judge, "__dict__") else None,
                                "db_url_env": DB_URL_ENV_RW}}
    # R11a order: the rw engine exists BEFORE PF8, which needs it. Connection only; no write yet.
    try:
        eng = deps.engine_factory("rw")
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"rw engine unavailable: {type(e).__name__}: {e}"
        return _finish(deps, rec, STATE_PREFLIGHT_FAILED, EXIT_PREFLIGHT)
    ledger = deps.ledger or CostLedger()
    try:
        # config=cfg: the pre-flight (and so the canary judge) uses the very config validated above, not a
        # second read of the files (review F8). A raise is reported, never allowed to skip the report (F7).
        pf = deps.preflight(question_id=question_id, env=deps.env, repo_root=deps.repo_root, limits=limits.judge,
                            ledger=ledger, judge_transport=deps.judge_transport, http_get=deps.http_get,
                            db_identity=lambda: deps.current_user(eng),
                            question_text_for=lambda q: deps.question_text(eng, q), config=cfg)
    except Exception as e:  # noqa: BLE001
        rec["error"] = scrub(f"pre-flight raised {type(e).__name__}: {e}", deps.env)[:600]
        rec["traceback"] = scrub(traceback.format_exc(), deps.env)[-MAX_TRACEBACK_CHARS:]
        return _finish(deps, rec, STATE_PREFLIGHT_FAILED, EXIT_PREFLIGHT)
    rec["preflight"] = [{"id": c.id, "ok": c.ok, "skipped": c.skipped, "detail": c.detail} for c in pf.checks]
    if not pf.ok or pf.selection is None:
        failed = ", ".join(c.id for c in pf.checks if not c.ok)
        rec["error"] = f"pre-flight failed: {failed}" if failed else "pre-flight passed but gave no judge selection"
        return _finish(deps, rec, STATE_PREFLIGHT_FAILED, EXIT_PREFLIGHT)
    rec["judge"] = pf.selection.as_record()
    role = deps.current_user(eng)  # belt and braces: refuse admin or anything but the rw account
    rec["account_role"] = role
    if role != EXPECTED_DB_ROLE:
        rec["error"] = f"db role mismatch: connected as {role!r}, expected {EXPECTED_DB_ROLE!r}; refusing to write"
        return _finish(deps, rec, STATE_PREFLIGHT_FAILED, EXIT_PREFLIGHT)

    freeze = dict(cfg.raw)
    question = freeze["question"]
    rec.update(question_sha256=_sha(question), freeze_hash=freeze_hash(freeze), git=deps.git_info(),
               config={"freeze_file_sha256": cfg.freeze_file_sha256, "run_file_sha256": cfg.run_file_sha256,
                       "run_config_version": cfg.run_config_version})
    try:
        run_id = deps.create_run(eng, question_id, question)  # int, RETURNING run_id
    except Exception as e:  # noqa: BLE001
        rec.update(error=f"create_run failed: {_err_text(e)}", failed_stage="create_run")
        return _finish(deps, rec, STATE_FAILED, EXIT_STAGE_FAILED)
    rec["run_id"] = run_id
    budget = RunBudget(limits.cap_usd or cfg.budget.cost_usd, limits.cap_s or cfg.budget.latency_s,
                       clock=deps.clock)
    budget.start()
    cur = {"stage": "m2"}
    searches: list = []
    rec["search"] = searches  # filled by _stage_m2 (also on a failed run: what was searched stays on record)

    def stop_if_cost_unknown(after: str):
        """Only when M6 is planned (it would refuse incomplete usage anyway, exit 4 guaranteed): stop NOW
        rather than after the remaining stages have spent. Returns an outcome tuple or None."""
        if "m6" not in plan:
            return None
        reasons = _unknown_cost_reasons(budget)
        if not reasons:
            return None
        rec.update(error=f"usage incomplete after {after}, stopped before later stages spend: "
                         + "; ".join(reasons[:10]),
                   failed_stage=after, budget=_budget_record(budget),
                   usage=engineered_usage(budget.obs, wall_clock_s=budget.elapsed_s()))  # partial; complete=False
        return STATE_USAGE_INCOMPLETE, EXIT_USAGE_INCOMPLETE

    def execute() -> tuple[str, int]:
        m2, captured = _stage_m2(run_id, question, eng, budget, deps, limits, searches)
        for sn in searches:
            r = sn["route"]
            if r.get("fallback_used"):
                rec["divergences"].append(f"seed-list FALLBACK used for search (sha256 {r.get('seed_sha256')}): "
                                          "agent-found candidates, unmetered discovery cost")
            if r.get("below_minimum"):
                rec["divergences"].append(f"search returned {sn['count']} candidate(s), below the minimum "
                                          f"{r.get('min_candidates')}, and no further route reached it")
        rec["stages"]["m2"] = {"status": "done", "admitted": len(m2.admitted), "deferred": len(m2.deferred),
                               "rejected": len(m2.rejected), "fetch_failed": len(m2.fetch_failed)}
        if (stopped := stop_if_cost_unknown("m2")):  # e.g. Gate A provider failure recorded cost_known=False
            return stopped
        cur["stage"] = "m3"
        m3 = _stage_m3(m2, captured, question, eng, budget, deps, limits)
        rec["stages"]["m3"] = {"status": "done", "counts": m3.get("counts"), "stopped_early": m3.get("stopped_early"),
                               "failures": len(m3.get("failures", []))}
        if (stopped := stop_if_cost_unknown("m3")):  # e.g. an M3 call with ok=False
            return stopped
        if "m45" not in plan:
            rec["budget"] = _budget_record(budget)
            return STATE_CLOSED_OK, EXIT_OK
        cur["stage"] = "m45"
        corpus, m45, store = _stage_m45(run_id, question, m3, freeze, eng, budget, deps)
        rec["stages"]["m45"] = {"status": "done", "tagging": corpus.get("tagging"),
                                "verification_status": m45.get("verification_status"),
                                "gate_c_trace": m45.get("gate_c_trace"), "relations_seen": store.seen}
        if not store.seen:
            rec["divergences"].append("relations not seen on save_relations: passed empty to build_engineered")
        usage = engineered_usage(budget.obs, wall_clock_s=budget.elapsed_s())
        rec.update(usage=usage, budget=_budget_record(budget))
        if "m6" not in plan:
            return STATE_CLOSED_OK, EXIT_OK
        if not usage["complete"]:  # RD-2/RD-5: unknown cost (e.g. from M4/M5), stop BEFORE any M6 spend
            rec["error"] = "engineered usage incomplete: " + "; ".join(usage["incomplete_reasons"][:10])
            return STATE_USAGE_INCOMPLETE, EXIT_USAGE_INCOMPLETE
        cur["stage"] = "m6"
        eng_ans = build_engineered(m45, m3, gate_a=_gate_a_audit(m2), relations={"edges": store.edges})
        attach_usage(eng_ans, usage)  # AFTER build: build_engineered drops usage.complete
        run = {"run_id": str(run_id), "question": question, "engineered": eng_ans,
               "corpus": build_corpus(_with_versions(m3, m2), freeze),
               # The values this run really used, for the run fingerprint (review F1/F2): M4 ran with
               # MVP_MAX_SEARCHES and the sub-questions derived from the requirements, and the stage limits
               # and effective caps can change the result.
               "effective_inputs": {
                   "search": [{"query": sn["query"], "candidate_list_sha256": sn["candidate_list_sha256"],
                               "route_used": sn["route"].get("route_used"),
                               "seed_sha256": sn["route"].get("seed_sha256"),
                               "below_minimum": sn["route"].get("below_minimum")} for sn in searches],
                   "m4": {"max_searches": MVP_MAX_SEARCHES, "sub_questions": corpus.get("sub_questions", [])},
                   "stage_limits": {"num_results": limits.num_results, "max_fetch": limits.max_fetch,
                                    "max_sources": limits.max_sources,
                                    "cap_usd": limits.cap_usd or cfg.budget.cost_usd,
                                    "cap_s": limits.cap_s or cfg.budget.latency_s}}}
        report = deps.run_m6(run, budget_confirmed=limits.budget_confirmed, judge=pf.selection.judge,
                             llm_call=deps.baseline_llm_call, store=False,  # never the admin get_session()
                             judge_selection=pf.selection.as_record(),
                             config=cfg)  # the config validated at the start; never re-read (review F8)
        ev = report.get("evaluation", {})
        rec["m6"] = {"outcome": ev.get("outcome"), "failed_dimensions": ev.get("failed_dimensions"),
                     "cost_summary": report.get("cost_summary")}
        if stage in STORE_EVAL_STAGES:
            deps.store_eval(eng, run_id, report)  # rw UPDATE, rowcount == 1
            rec["m6"]["stored"] = True
        return STATE_CLOSED_OK, EXIT_OK

    # _finish is called OUTSIDE the try: a report/register write failure must not be caught below and
    # rewrite an already-written closed_ok report as failed. It logs the outcome first, then raises.
    try:
        state, code = execute()
    except RunBudgetExceeded as e:
        rec.update(error=str(e), failed_stage=cur["stage"], budget=_budget_record(budget))
        state, code = STATE_BUDGET_STOP, EXIT_BUDGET
    except Exception as e:  # noqa: BLE001  (verbose failure: stage, cause, traceback kept)
        tb = scrub(traceback.format_exc(), deps.env)[-MAX_TRACEBACK_CHARS:]
        rec.update(error=_err_text(e), failed_stage=cur["stage"], traceback=(_err_text(e) if _is_conn_error(e) else tb),
                   budget=_budget_record(budget))
        state, code = STATE_FAILED, EXIT_STAGE_FAILED
    return _finish(deps, rec, state, code)


# ------------------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None, *, env: Mapping[str, str] | None = None,
         deps_factory: Callable[[], Deps] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vera.run_mvp", description=__doc__.split("\n")[0])
    ap.add_argument("--question-id", type=int, required=True)
    ap.add_argument("--stage", required=True, choices=sorted(STAGE_PLANS))
    ap.add_argument("--run-type", required=True, choices=("test", "live"))
    ap.add_argument("--live", action="store_true",
                    help="actually execute (spends money, writes the DB). Default: dry/offline pre-flight only")
    ap.add_argument("--budget-confirmed", action="store_true")
    ap.add_argument("--num-results", type=int)
    ap.add_argument("--max-fetch", type=int)
    ap.add_argument("--max-sources", type=int)
    a = ap.parse_args(argv)
    try:
        _validate(a.stage, a.run_type)
    except RunConfigError as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_PREFLIGHT
    env = os.environ if env is None else env
    if not a.live:
        # DRY: offline pre-flight only. No engine, no .env load, no provider call, no write.
        rep = run_preflight(question_id=a.question_id, env=env,
                            repo_root=Path(__file__).resolve().parent.parent, require_db=False,
                            canary=False, listing=False)
        print(rep.render())
        print(f"DRY RUN: stage {a.stage} ({'+'.join(STAGE_PLANS[a.stage])}) was NOT executed; nothing was "
              "spent or written. Re-run with --live --budget-confirmed after approval.")
        return EXIT_OK if rep.ok else EXIT_PREFLIGHT
    if not a.budget_confirmed:
        print("refused: --live needs --budget-confirmed (M6 refuses without it, after M2-M5 spend).",
              file=sys.stderr)
        return EXIT_PREFLIGHT
    deps = (deps_factory or (lambda: Deps.real(env)))()
    limits = Limits(budget_confirmed=True, num_results=a.num_results, max_fetch=a.max_fetch,
                    max_sources=a.max_sources)
    rep = run_mvp(a.question_id, a.stage, a.run_type, deps=deps, limits=limits)
    print(f"state={rep.state} exit={rep.exit_code} report={rep.path}")
    return rep.exit_code


if __name__ == "__main__":
    sys.exit(main())
