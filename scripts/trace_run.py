"""One entry point for TRACE eval automation: offline, dry-run, live, and run comparison.

    --offline   deterministic: fixture answers through the real capture, checks and aggregation. Zero provider calls,
                no env file, no network. Public JSON goes to the private run dir (never to eval_results/) unless --publish.
    --dry-run   prints the plan and limits; makes no call, loads no env, writes nothing.
    --live      OWNER-TRIGGERED. Sends questions through VERA-DAR's /ask (paid, rate-limited). Requires explicit
                --max-cases, --max-cost, --min-interval-s, --timeout-s, --total-timeout-s (concurrency is fixed at 1) and
                owner credentials from the environment at runtime: VERA_TRACE_EVAL_OWNER_CONFIRM (must equal the owner's
                confirmation phrase, see RUNBOOK) and VERA_API_KEY (loaded by the project env loader). There is no
                unauthenticated public trigger: nothing in the web app starts this script.
    --compare OLD NEW   regressions/improvements between two public aggregate files; "incomparable" unless the eval-set
                hash and scoring version match.

Post-deploy and scheduled runs are the same --live command with --trigger post-deploy|scheduled, started by an owner-run
cron/Render-cron/CI step (documented, not installed). Prints counts and paths only; never questions, answers, keys or hosts.
Exit codes: 0 ok, 2 refused (guard/arguments), 4 bad input, 5 run failed (failure record written, public file untouched)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vera.trace_eval import auto, capture, guidance  # noqa: E402

CONFIRM_ENV = "VERA_TRACE_EVAL_OWNER_CONFIRM"
CONFIRM_VALUE = "I-AUTHORIZE-LIVE-TRACE-EVAL"
LIVE_ENV_NAMES = (CONFIRM_ENV, "VERA_API_KEY", "VERA_API_BASE_URL (http target; optional)")
# Why: hard ceilings so a typo cannot turn a bounded owner run into an open-ended one.
CEIL = {"max_cases": 60, "max_cost": 5.0, "timeout_s": 120.0, "total_timeout_s": 3600.0}
QUESTIONS = {"frozen": "config/trace_questions_v1.json", "accept": "../work/dar-accept-1022/questions.json"}


def _parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--offline", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--live", action="store_true")
    g.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    ap.add_argument("--trigger", choices=auto.TRIGGERS, default="manual")
    ap.add_argument("--case-set", choices=["frozen", "accept"], default="frozen")
    ap.add_argument("--questions", help="question file (default per case set)")
    ap.add_argument("--target", choices=["http", "inprocess"], default="http")
    ap.add_argument("--max-cases", type=int)
    ap.add_argument("--max-cost", type=float)
    ap.add_argument("--min-interval-s", type=float, help="minimum seconds between /ask calls (rate)")
    ap.add_argument("--timeout-s", type=float, help="per-call timeout (http target)")
    ap.add_argument("--total-timeout-s", type=float, help="whole-run wall-clock limit")
    ap.add_argument("--private-dir", default=str(auto.PRIVATE_DIR), help="per-case results (gitignored)")
    ap.add_argument("--public-out", default=str(auto.PUBLIC_OUT), help="public aggregate file (--live; --offline with --publish)")
    ap.add_argument("--publish", action="store_true", help="offline only: also write the public file (default: private dir only)")
    ap.add_argument("--deployed-version", help="commit of the deployed build under test (default RENDER_GIT_COMMIT or unknown)")
    return ap.parse_args(argv)


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def _run_id() -> str:
    return auto.utc_now().strftime("%Y%m%dT%H%M%SZ")


def _limits(a) -> tuple[dict | None, str]:
    """Validated live limits, or (None, reason). Every limit must be given explicitly and be within its ceiling."""
    vals = {"max_cases": a.max_cases, "max_cost": a.max_cost, "min_interval_s": a.min_interval_s,
            "timeout_s": a.timeout_s, "total_timeout_s": a.total_timeout_s}
    missing = [k for k, v in vals.items() if v is None]
    if missing:
        return None, "live requires explicit " + ", ".join("--" + k.replace("_", "-") for k in missing)
    if vals["max_cases"] < 1 or vals["max_cost"] <= 0 or vals["min_interval_s"] < 0 or vals["timeout_s"] <= 0 or vals["total_timeout_s"] <= 0:
        return None, "limits must be positive (min-interval-s >= 0)"
    for k, c in CEIL.items():
        if vals[k] > c:
            return None, f"--{k.replace('_', '-')} exceeds the ceiling {c}"
    return vals, ""


def _load_live(a):
    cfg = guidance.load_config()
    return capture.load_case_set(_resolve(a.questions or QUESTIONS[a.case_set]), a.case_set, cfg)


def _report(res: dict) -> int:
    if res["status"] == "ok":
        print(f"OK run_dir={res['run_dir']} public={res.get('public') or '(private dir only)'} summary={json.dumps(res['summary'])}")
        return 0
    print(f"FAILED reason={res['reason']} failure_record={res.get('failure')}; public file untouched", file=sys.stderr)
    return 5


def main(argv=None) -> int:
    a = _parse(argv)
    if a.compare:
        try:
            old, new = (json.loads(_resolve(p).read_text(encoding="utf-8")) for p in a.compare)
        except (OSError, ValueError) as e:
            print(f"FAILED: {type(e).__name__}", file=sys.stderr)
            return 4
        out = auto.compare_runs(old, new)
        print(json.dumps(out, indent=1))
        return 0
    if a.offline:
        fx = json.loads(auto.OFFLINE_FIXTURE.read_text(encoding="utf-8"))
        doc = {**fx["questions"], "case_set": "offline-fixture", "checks_validated": True}
        res = auto.execute_run(doc=doc, source_sha=auto.canonical_sha(fx["questions"]), ask=auto.fixture_ask(fx), mode="offline",
                               trigger=a.trigger, private_dir=Path(a.private_dir), run_id=_run_id(), limit=a.max_cases,
                               max_cost=1.0, provider="offline-fixture", public_out=None,
                               publish_out=_resolve(a.public_out) if a.publish else None, deployed_version=a.deployed_version)
        return _report(res)
    lim, why = _limits(a)
    if a.dry_run:
        try:
            doc, sha = _load_live(a)
        except (ValueError, OSError, KeyError) as e:
            print(f"FAILED: {type(e).__name__}", file=sys.stderr)
            return 4
        n = min(len(doc["items"]), a.max_cases or len(doc["items"]))
        print(f"DRY-RUN plan: mode=live trigger={a.trigger} target={a.target} case_set={doc['case_set']} cases={n} "
              f"checks_validated={doc['checks_validated']} questions_sha256={sha[:12]} scoring={auto.SCORING_VERSION}")
        print(f"limits required by --live: {', '.join('--' + k.replace('_', '-') + ' (ceiling ' + str(c) + ')' for k, c in CEIL.items())}, "
              "--min-interval-s; concurrency fixed at 1")
        print("given limits: " + (json.dumps(lim) if lim else f"incomplete ({why})"))
        print(f"env names needed at runtime (values never read here): {', '.join(LIVE_ENV_NAMES)}")
        print(f"would write per-case results under {_resolve(a.private_dir)}/<run_id>/ and, only on full success, {_resolve(a.public_out)}")
        print("no call made, no env loaded, nothing written")
        return 0
    # --live: every guard runs before any env load, import of the API, or network use.
    if lim is None:
        print(f"REFUSED: {why}", file=sys.stderr)
        return 2
    if os.environ.get(CONFIRM_ENV) != CONFIRM_VALUE:
        print(f"REFUSED: {CONFIRM_ENV} is not set to the owner confirmation phrase (live runs are owner-triggered)", file=sys.stderr)
        return 2
    try:
        doc, sha = _load_live(a)
    except (ValueError, OSError, KeyError) as e:
        print(f"FAILED: {type(e).__name__}", file=sys.stderr)
        return 4
    from vera.env_files import load_vera_env
    load_vera_env()  # PLAN-ENV: credentials from the environment at runtime; nothing printed
    key = os.getenv("VERA_API_KEY", "")
    if not key:
        print("REFUSED: VERA_API_KEY is not set in the loaded environment", file=sys.stderr)
        return 2
    if a.target == "inprocess":
        raw_ask = capture.inprocess_ask(key)
    else:
        from vera.public_mode import upstream_base_url
        raw_ask = capture.http_ask(upstream_base_url(local_url_file=ROOT / ".vera-local-url"), key, timeout=int(lim["timeout_s"]))
    ask = auto.guarded_ask(raw_ask, min_interval_s=lim["min_interval_s"], total_timeout_s=lim["total_timeout_s"])
    res = auto.execute_run(doc=doc, source_sha=sha, ask=ask, mode="live", trigger=a.trigger, private_dir=Path(a.private_dir),
                           run_id=_run_id(), limit=lim["max_cases"], max_cost=lim["max_cost"], provider="vera-ask-route",
                           public_out=None, publish_out=_resolve(a.public_out), timed_out=lambda: ask.state["timed_out"],
                           deployed_version=a.deployed_version)
    return _report(res)


if __name__ == "__main__":
    sys.exit(main())
