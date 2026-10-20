"""Zero-baseline-spend pre-flight for the slim live MVP (plan section 10.2, checks PF1-PF11).

Every check targets a predicted failure and runs BEFORE any baseline, M2 or M3 spend. The only spend
allowed here is the judge canary (a few hundred tokens, recorded, reset out of M6's judge budget).

Owns the two constants S4 imports (R11a): EXPECTED_DB_ROLE and DB_URL_ENV_RW. This module must NOT
import vera.db (no cycle: db.py imports DB_URL_ENV_RW from here) and never calls get_engine/get_session.

Offline mode (`--no-db` without `--listing`/`--canary`) loads no .env, reads only the process
environment, makes no provider or DB call, and states each skipped check.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from vera.cost_ledger import CostLedger
from vera.eval_config import EvalConfig, EvalConfigError, load_eval_config
from vera.judge_select import (JUDGE_POOL_ENV, Selection, _collect, _provider_facts,
                               default_http_get, default_judge_transport, parse_pool, scrub,
                               select_from)
from vera.gate_a import MODEL as GATE_A_MODEL
from vera.m6.baseline_collection import DEFAULT_MODEL as BASELINE_MODEL
from vera.m6.judge import JudgeConfigError, JudgeLimits, generation_family

# Why: U6 divergence: every MVP stage writes as the rw account; identity is asserted, not assumed.
# Goal #2 key: accounts.pipeline_write_role.
EXPECTED_DB_ROLE = "vera_claude_code_rw"
# Why: an env var NAME, not a value (the URL itself stays in .env / the process environment).
DB_URL_ENV_RW = "VERA_DB_URL_RW"
# The freeze pin (sha256 and the 16-hex requirements hash) is read from config/vera_eval_run.json
# (freeze_pin) by vera.eval_config; the interim hardcoded EXPECTED_FREEZE_HASH was retired in goal #2.
# Why: PF1 names. R1 approved only the GROQ and OPENAI keys plus the DB URL (2026-10-02) and directed
# that M2 be redone with a keyless search, so no search-provider key is required here. PF13 checks the
# search chain without any key (p3m3/vera-plan-m2-keyless-search.md).
REQUIRED_ENV_NAMES: tuple[str, ...] = ("OPENAI_API_KEY", "GROQ_API_KEY", DB_URL_ENV_RW)
# Why: gitignored report location (runner plan Q10); S4 owns RUN_DIR, PF11 only checks this name.
RUN_DIR_NAME = ".vera-runs"


@dataclass
class Check:
    id: str
    ok: bool
    detail: str
    fix: str = ""
    skipped: bool = False


@dataclass
class PreflightReport:
    ok: bool
    checks: list[Check] = field(default_factory=list)
    selection: Selection | None = None

    def render(self) -> str:
        lines = []
        for c in self.checks:
            tag = "SKIP" if c.skipped else ("PASS" if c.ok else "FAIL")
            lines.append(f"{c.id} {tag}: {c.detail}" + (f"  FIX: {c.fix}" if (not c.ok and c.fix) else ""))
        lines.append("PREFLIGHT " + ("OK" if self.ok else "FAILED (no run was created, nothing spent "
                                     "beyond the canary)"))
        return "\n".join(lines)


def _default_check_ignored(repo_root: Path, rel: str) -> bool:
    r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=repo_root, capture_output=True)
    return r.returncode == 0


def _default_pricing(model: str):
    from vera.pricing.config import latest_pricing_for, load_model_pricing
    return latest_pricing_for(model, load_model_pricing())


def _default_generator_model() -> str:
    from vera.pipeline_llm import default_model
    return default_model()


def run_preflight(*, question_id: int, env: Mapping[str, str], repo_root: Path,
                  limits: JudgeLimits | None = None, ledger: CostLedger | None = None,
                  judge_transport=None, http_get=None, db_identity: Callable[[], str] | None = None,
                  question_text_for: Callable[[int], str] | None = None,
                  check_ignored: Callable[[str], bool] | None = None, require_db: bool = True,
                  canary: bool = True, listing: bool = True, generator_model: str | None = None,
                  pricing_lookup: Callable | None = None, n_judges: int = 1,
                  baseline_model: str | None = None, gate_a_model: str | None = None,
                  config: EvalConfig | None = None) -> PreflightReport:
    """Run PF1-PF13 (PF13 = search chain; PF12 added in S5: baseline/Gate A model equals generator). Never raises for a failed check: every failure is a Check with its fix."""
    checks: list[Check] = []
    selection: Selection | None = None
    repo_root = Path(repo_root)

    def add(id_, ok, detail, fix="", skipped=False):
        checks.append(Check(id_, ok, scrub(detail, env), fix, skipped))

    # One loader for every config read in this check set (goal #2). A failure is reported verbosely by PF10
    # and makes PF2-PF4 fail ("config unavailable"), never silently fall back to code defaults.
    cfg_error = ""
    try:
        # `config`: the caller's already validated config is used as is (review F8: one load per run).
        eval_cfg = config or load_eval_config(repo_root / "config" / "vera_eval_freeze.json",
                                              repo_root / "config" / "vera_eval_run.json",
                                              model_selection_path=repo_root / "config" / "model-selection.json")
        providers = eval_cfg.judge.providers
    except EvalConfigError as e:
        eval_cfg, providers, cfg_error = None, (), str(e)

    # PF1 env names present (names only; values are never printed)
    missing = [n for n in REQUIRED_ENV_NAMES if not env.get(n)]
    add("PF1", not missing,
        "env names present: " + ", ".join(REQUIRED_ENV_NAMES) if not missing
        else "missing env name(s): " + ", ".join(missing),
        "export the missing name(s) in the process environment (values are never read from files here)")

    # PF2 VERA_JUDGE_POOL parses, providers in the pool
    pool_env = env.get(JUDGE_POOL_ENV)
    if eval_cfg is None:
        add("PF2", False, "run config unavailable, so the judge pool settings cannot be read (see PF10)",
            "fix PF10 first")
    elif pool_env:
        try:
            pool = parse_pool(pool_env, eval_cfg)
            add("PF2", True, f"{JUDGE_POOL_ENV} parses: {len(pool)} entr(y/ies), providers in {providers}")
        except JudgeConfigError as e:
            add("PF2", False, str(e), f"fix or unset {JUDGE_POOL_ENV}")
    else:
        add("PF2", True, f"{JUDGE_POOL_ENV} not set: default pool {providers}")

    # PF3 / PF4 / PF6 need the listing (and canary): skipped offline, stated
    cands = refused = lst = None
    if not listing:
        for id_ in ("PF3", "PF4"):
            add(id_, True, "SKIPPED: offline mode makes no provider call (use --listing)", skipped=True)
    else:
        try:
            key_env = _provider_facts(providers[0]).key_env if providers else None
            facts_err = ""
        except JudgeConfigError as e:  # unsupported provider: report it, never raise (review F7)
            key_env, facts_err = None, str(e)
        if eval_cfg is None:
            add("PF3", False, "NOT RUN: run config unavailable (see PF10)", "fix PF10 first")
            add("PF4", False, "NOT RUN: PF3 failed", "fix PF3 first")
        elif facts_err:
            add("PF3", False, facts_err, "use a supported provider in config/vera_eval_run.json judge.pool.providers")
            add("PF4", False, "NOT RUN: PF3 failed", "fix PF3 first")
        elif not env.get(key_env):
            add("PF3", False, f"{key_env} absent: listing not attempted", f"set {key_env}")
            add("PF4", False, "NOT RUN: PF3 failed", "fix PF3 first")
        else:
            try:
                cands, refused, lst = _collect(env, http_get=http_get,
                                               pool_env=pool_env if _pool_ok(pool_env, eval_cfg) else None,
                                               config=eval_cfg)
                add("PF3", True, f"groq listing ok: {len(lst['listed_models'])} model id(s)")
                if cands:
                    add("PF4", True, f"{len(cands)} candidate(s) after the family map; refused "
                        f"{len(refused)}: " + "; ".join(f"{r.model} ({r.reason[:60]})" for r in refused[:8]))
                else:
                    add("PF4", False,
                        "no non-OpenAI candidate with a known family; refused: "
                        + "; ".join(f"{r.model} ({r.reason[:60]})" for r in refused[:8]),
                        "review the UNVERIFIED family patterns against the live list, or set a pool")
            except JudgeConfigError as e:
                add("PF3", False, str(e), "check the key, connectivity and the pool setting")
                add("PF4", False, "NOT RUN: PF3 failed", "fix PF3 first")

    # PF5 generator is GPT and its mapped family equals the family the judge guard derives from the run config
    try:
        gen = generator_model or _default_generator_model()
        gen_family = generation_family(eval_cfg) if eval_cfg is not None else None
        mapped = eval_cfg.model_families.get(gen, {}).get("family") if eval_cfg is not None else None
        is_gpt = gen.lower().startswith("gpt")
        add("PF5", is_gpt and gen_family is not None and mapped == gen_family,
            f"generator model {gen!r} is {'GPT' if is_gpt else 'NOT GPT'}; mapped family {mapped!r}; "
            f"judge-guard generator family {gen_family!r} (from the run config)",
            "the generator is not the model the run config maps to the guarded family: review "
            "config/vera_eval_run.json (generator.model, model_families) before any run")
    except Exception as e:  # noqa: BLE001
        gen = None
        add("PF5", False, f"could not read the generator model: {e!r}", "check config/model-selection.json")

    # PF7 pricing record for the generator (silent $0 otherwise)
    try:
        rec = (pricing_lookup or _default_pricing)(gen) if gen else None
        add("PF7", rec is not None,
            f"pricing record present for generator {gen!r}" if rec is not None
            else f"no pricing record for generator {gen!r}: cost would silently be $0",
            "add a record to config/model-pricing.json")
    except Exception as e:  # noqa: BLE001
        add("PF7", False, f"pricing lookup failed: {e!r}", "check config/model-pricing.json")

    # PF10 freeze pin and run config validity. Read only; never edits either file.
    freeze = None
    freeze_path = repo_root / "config" / "vera_eval_freeze.json"
    if eval_cfg is not None:
        freeze = dict(eval_cfg.raw)
        add("PF10", True,
            f"freeze file sha256 {eval_cfg.freeze_file_sha256[:16]}... and requirements hash "
            f"{eval_cfg.requirements_hash16} match the pin in the run config (run_config_version "
            f"{eval_cfg.run_config_version}, run file sha256 {eval_cfg.run_file_sha256[:16]}...)")
    else:
        add("PF10", False, f"cannot load the eval config for {freeze_path}: {cfg_error[:600]}",
            "the immutable freeze file was edited or the run config is invalid: restore the freeze file; "
            "fix the run config as a new run_config_version")

    # PF11 report dir writable and git-ignored
    try:
        target = repo_root / RUN_DIR_NAME
        probe = target if target.exists() else repo_root
        writable = os.access(probe, os.W_OK)
        ignored = (check_ignored or (lambda rel: _default_check_ignored(repo_root, rel)))(f"{RUN_DIR_NAME}/")
        add("PF11", writable and ignored,
            f"{RUN_DIR_NAME}/ writable={writable} git-ignored={ignored}",
            f"add '{RUN_DIR_NAME}/' to .gitignore and make the directory writable")
    except Exception as e:  # noqa: BLE001
        add("PF11", False, f"git-ignore check failed: {e!r}", f"ensure git works and {RUN_DIR_NAME}/ is ignored")

    # PF8 / PF9 need the database
    if not require_db:
        for id_ in ("PF8", "PF9"):
            add(id_, True, "SKIPPED: --no-db (no database read)", skipped=True)
    else:
        if db_identity is None:
            add("PF8", False, "no db_identity callable provided", "pass the rw engine's current_user query")
        else:
            try:
                who = db_identity()
                add("PF8", who == EXPECTED_DB_ROLE,
                    f"current_user is {who!r}, expected {EXPECTED_DB_ROLE!r}",
                    f"connect via {DB_URL_ENV_RW} as {EXPECTED_DB_ROLE} (never the admin account)")
            except Exception as e:  # noqa: BLE001
                add("PF8", False, f"identity query failed: {e!r}", "check the rw connection")
        if question_text_for is None or freeze is None:
            add("PF9", False, "no question reader or no freeze file", "fix PF10 / pass the reader")
        else:
            try:
                got = question_text_for(question_id)
                a = hashlib.sha256((got or "").encode()).hexdigest()[:16]
                b = hashlib.sha256(freeze["question"].encode()).hexdigest()[:16]
                add("PF9", a == b, f"question {question_id}: sha256 {a} {'==' if a == b else '!='} frozen {b}",
                    "use the question_id whose text equals the frozen question")
            except Exception as e:  # noqa: BLE001
                add("PF9", False, f"question read failed: {e!r}", "check the question id and the connection")

    # PF12 baseline and Gate A models equal the generator model (S5 F7). The baseline is a fair
    # comparison only if it uses the same model as the pipeline; both constants are hardcoded in code.
    try:
        bm = baseline_model or BASELINE_MODEL
        gm = gate_a_model or GATE_A_MODEL
        same = bool(gen) and bm == gen and gm == gen
        add("PF12", same,
            f"baseline model {bm!r} and Gate A model {gm!r} "
            + (f"equal the generator model {gen!r}" if same else
               f"do not all equal the generator model {gen!r}: the baseline would compare different models"),
            "make baseline_collection.DEFAULT_MODEL and gate_a.MODEL equal the generator model")
    except Exception as e:  # noqa: BLE001
        add("PF12", False, f"could not compare baseline and generator models: {e!r}", "check the model constants")

    # PF6 canary: LAST, and only if every other check passed, so a failed check never spends tokens
    # (plan: all checks run before any spend). Check ids and report shape are unchanged.
    blockers = sorted((c.id for c in checks if not c.ok), key=lambda i: int(i[2:]))
    if not canary:
        add("PF6", True, "SKIPPED: canary not requested (it spends a few hundred tokens; use --canary)",
            skipped=True)
    elif cands is None:
        add("PF6", False, "NOT RUN: no candidate list (PF3/PF4 failed or listing skipped)", "fix PF3/PF4")
    elif not cands:
        add("PF6", False, "NOT RUN: no candidates", "fix PF4")
    elif blockers:
        add("PF6", True, f"SKIPPED: no canary spend because other check(s) failed: {', '.join(blockers)}",
            skipped=True)
    else:
        try:
            selection = select_from(cands, refused, lst, env, limits=limits, ledger=ledger,
                                    n=n_judges, judge_transport=judge_transport or default_judge_transport,
                                    config=eval_cfg)  # the validated config, not a fresh default load (F8)
            c = selection.candidate
            add("PF6", True, f"canary ok: {c.provider}:{c.model} family={c.family} rule={selection.rule}; "
                f"canary spend {selection.canary_spend}; failures before it: {len(selection.canary_failures)}")
        except JudgeConfigError as e:
            add("PF6", False, str(e), "see the failures listed; pick another model via the pool setting")

    # PF13 search chain: config parses, providers known, seed (if configured) present with a matching sha256.
    # Offline and KEY-FREE: no network call, no env name required by a keyless provider.
    try:
        if eval_cfg is None:
            add("PF13", False, "run config unavailable, so the search chain cannot be read (see PF10)", "fix PF10 first")
        else:
            from vera.search_providers import SearchConfigError, build_chain, file_sha256
            chain = build_chain(eval_cfg.search, repo_root=repo_root)
            need = sorted({e for p in chain for e in p.required_env if not env.get(e)})
            problems = [f"missing env {need}"] if need else []
            for p in chain:
                if p.name.startswith("seed:"):
                    if not p.path.is_file():
                        problems.append(f"seed file {p.path} not found")
                    elif file_sha256(p.path) != p.expected:
                        problems.append("seed file sha256 does not match the frozen value")
            add("PF13", not problems,
                ("search chain " + " -> ".join(p.name for p in chain) + f"; min_candidates "
                 f"{eval_cfg.search['min_candidates']}; no key required") if not problems else "; ".join(problems),
                "fix config search (providers, fallback_seed) or restore the seed file")
    except SearchConfigError as e:
        add("PF13", False, str(e), "fix config/vera_eval_run.json search as a new run_config_version")
    except Exception as e:  # noqa: BLE001
        add("PF13", False, f"search chain check failed: {e!r}", "check config/vera_eval_run.json search")

    checks.sort(key=lambda c: int(c.id[2:]))
    return PreflightReport(all(c.ok for c in checks), checks, selection)


def _pool_ok(pool_env, config=None) -> bool:
    try:
        return bool(pool_env) and bool(parse_pool(pool_env, config))
    except JudgeConfigError:
        return False  # PF2 reports it; the listing check proceeds without the broken pool


def main(argv: list[str] | None = None, *, env: Mapping[str, str] | None = None,
         repo_root: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vera.mvp_preflight")
    ap.add_argument("--question-id", type=int, required=True)
    ap.add_argument("--no-db", action="store_true", help="offline: no DB, no .env load, no provider call")
    ap.add_argument("--listing", action="store_true", help="with --no-db: also list Groq models ($0)")
    ap.add_argument("--canary", action="store_true", help="also run the judge canary (a little spend)")
    a = ap.parse_args(argv)
    env = os.environ if env is None else env
    root = repo_root or Path(__file__).resolve().parent.parent
    do_listing = (not a.no_db) or a.listing or a.canary
    db_identity = question_text_for = None
    if not a.no_db:
        try:
            from vera import m2_runner  # SCHEMA defined once there (as run_mvp does)
            from vera.db import get_role_engine  # S4 adds this; imported lazily, never at module import
            from sqlalchemy import text
            eng = get_role_engine("rw")

            def db_identity():
                with eng.connect() as c:
                    return c.execute(text("SELECT current_user")).scalar()

            def question_text_for(qid):
                with eng.connect() as c:
                    return c.execute(text(f"SELECT research_question FROM {m2_runner.SCHEMA}.questions "  # nosec B608 - m2_runner.SCHEMA constant; question id bound
                                          "WHERE question_id = :q"), {"q": qid}).scalar()
        except Exception as e:  # noqa: BLE001
            print(f"PF8/PF9 cannot run: rw engine unavailable ({scrub(repr(e), env)[:200]}). "
                  "Use --no-db or provide vera.db.get_role_engine.", file=sys.stderr)
    rep = run_preflight(question_id=a.question_id, env=env, repo_root=root, limits=JudgeLimits(),
                        ledger=CostLedger(), judge_transport=default_judge_transport if a.canary else None,
                        http_get=default_http_get if do_listing else None, db_identity=db_identity,
                        question_text_for=question_text_for, require_db=not a.no_db,
                        canary=a.canary, listing=do_listing)
    print(rep.render())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
