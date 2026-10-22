"""Capture answer traces for the Trace Eval page and measure them with the deterministic checks A1-A9.

A live capture makes model calls (and, for ask_api, runs VERA-DAR's /ask route), so it needs owner authorization.
--dry-run and --measure-only make no model, network or DB call and load no env file.

    # plan only (no calls, no env, writes nothing):
    venv/bin/python scripts/trace_capture.py --variant ask_api --case-set frozen --dry-run \
        --out ../work/<task>/traces.jsonl --results-out ../work/<task>/results.json
    # live capture through the real /ask route (owner-authorized), then measure:
    venv/bin/python scripts/trace_capture.py --variant ask_api --target http --case-set frozen \
        --out ../work/<task>/traces.jsonl --results-out ../work/<task>/results.json
    # measure existing traces only (offline):
    venv/bin/python scripts/trace_capture.py --measure-only --out ../work/<task>/traces.jsonl \
        --results-out ../work/<task>/results.json

Prints only counts, cost and output paths; never answers, keys, hosts or env values."""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vera.trace_eval import capture, checks, guidance, measure  # noqa: E402

# Why: default question files per case set (relative to the checkout). The frozen set is withheld from the public repo
# (.gitignore: it carries the held-out split), so a clean checkout must be given --questions explicitly.
DEFAULT_QUESTIONS = {"frozen": "config/trace_questions_v1.json", "accept": "../work/dar-accept-1022/questions.json"}


def _parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", choices=list(capture.VARIANTS), default="ask_api")
    ap.add_argument("--target", choices=["http", "inprocess"], default="http",
                    help="ask_api only: POST to the running API (VERA_API_BASE_URL or the local default) or in-process")
    ap.add_argument("--case-set", choices=["frozen", "accept"], default="frozen")
    ap.add_argument("--questions", help="question file (default per case set)")
    ap.add_argument("--out", required=True, help="raw traces JSONL (full answers; keep untracked)")
    ap.add_argument("--results-out", help="results JSON for the page (measured after capture)")
    ap.add_argument("--split", choices=["dev", "heldout"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-cost", type=float, default=capture.DEFAULT_MAX_COST_USD)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="plan only: no env, no calls, no writes")
    ap.add_argument("--measure-only", action="store_true", help="measure --out into --results-out; no calls")
    return ap.parse_args(argv)


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (ROOT / q)


def _measure(out: Path, results_out: Path | None) -> int:
    if results_out is None:
        return 0
    res = measure.measure([out], results_out)
    print("\n".join(measure.aggregate_lines(res)))
    print(f"results={results_out}")
    return 0


def _load(a) -> tuple[dict, str]:
    cfg = guidance.load_config()
    return capture.load_case_set(_resolve(a.questions or DEFAULT_QUESTIONS[a.case_set]), a.case_set, cfg)


def _dry_run(a, out: Path, results_out: Path | None) -> int:
    fails = checks.self_test()
    if fails:
        print("DRY-RUN FAILED: checks self-test: " + "; ".join(fails), file=sys.stderr)
        return 1
    doc, sha = _load(a)
    items = [q for q in doc["items"] if not a.split or q.get("split") == a.split][: a.limit]
    cats = collections.Counter(q["category"] for q in items)
    print(f"DRY-RUN variant={a.variant} target={a.target if a.variant == 'ask_api' else 'n/a'} "
          f"case_set={doc['case_set']} checks_validated={doc['checks_validated']} questions={len(items)} "
          f"questions_sha256={sha[:12]}")
    print("categories: " + ", ".join(f"{k}={v}" for k, v in sorted(cats.items())))
    print(f"would write traces={out} (exists={out.exists()}) results={results_out}; no call made")
    return 0


def _live_callables(a):
    """Load env at runtime only and build the callables. Nothing loaded here is printed."""
    from vera.env_files import load_vera_env

    load_vera_env()  # PLAN-ENV
    if a.variant == "ask_api":
        import os

        key = os.getenv("VERA_API_KEY", "")
        if not key:
            raise SystemExit("FAILED: VERA_API_KEY is not set in the loaded environment")
        if a.target == "inprocess":
            return {"ask": capture.inprocess_ask(key)}, "", None
        from vera.public_mode import upstream_base_url

        return {"ask": capture.http_ask(upstream_base_url(local_url_file=ROOT / ".vera-local-url"), key)}, "", None
    from vera.pricing.config import latest_pricing_for, load_model_pricing, load_model_selection

    sel, prices = load_model_selection(), load_model_pricing()
    pricing = latest_pricing_for(sel.selected_model, prices) if sel else None
    if sel is None or pricing is None:
        raise SystemExit("FAILED: model selection or pricing record unavailable")
    return ({"chat": capture.default_chat(sel.selected_model), "search": capture.default_search},
            sel.selected_model, pricing)


def main(argv: list[str] | None = None) -> int:
    a = _parse(argv)
    out = _resolve(a.out)
    results_out = _resolve(a.results_out) if a.results_out else None
    if a.measure_only:
        if not out.is_file():
            print(f"FAILED: no traces at {out}", file=sys.stderr)
            return 4
        return _measure(out, results_out)
    try:
        if a.dry_run:
            return _dry_run(a, out, results_out)
        doc, sha = _load(a)
    except (ValueError, OSError, KeyError) as e:
        print(f"FAILED: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        return 4
    if out.exists() and not a.overwrite:
        print(f"REFUSED: {out} exists (use --overwrite)", file=sys.stderr)
        return 2
    fns, model, pricing = _live_callables(a)
    if a.overwrite and out.exists():
        out.write_text("", encoding="utf-8")  # truncate in place (no deletion)
    try:
        summary = capture.run_capture(doc, a.variant, out, pricing=pricing, model=model, max_cost_usd=a.max_cost,
                                      splits={a.split} if a.split else None, questions_sha256=sha, limit=a.limit,
                                      **fns)
    except Exception as e:  # noqa: BLE001  (auth/config errors abort the run; type and status only)
        print(f"ABORTED: {type(e).__name__}: {getattr(e, 'status', '')}", file=sys.stderr)
        return 5
    print(f"variant={a.variant} case_set={doc['case_set']} questions_sha256={sha[:12]} traces={out}")
    print(json.dumps(summary))
    rc = _measure(out, results_out)
    return rc if not summary["stopped"] else 6


if __name__ == "__main__":
    sys.exit(main())
