"""Capture raw answer traces for the item #70 trace evaluation (live calls; run by the orchestrator).

    venv/bin/python scripts/trace_capture.py --variant baseline --out tmp/trace_eval/baseline.jsonl

Prints only a summary (counts, cost); never answers or keys."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vera.env_files import load_vera_env  # noqa: E402

from vera.pricing.config import latest_pricing_for, load_model_pricing, load_model_selection  # noqa: E402
from vera.trace_eval import capture  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", choices=["baseline", "grounded_v1"], required=True)
    ap.add_argument("--questions", default="config/trace_questions_v1.json")
    ap.add_argument("--out", help="default tmp/trace_eval/<variant>.jsonl")
    ap.add_argument("--split", choices=["dev", "heldout"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-cost", type=float, default=capture.DEFAULT_MAX_COST_USD)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    out = ROOT / (a.out or f"tmp/trace_eval/{a.variant}.jsonl")
    if out.exists() and not a.overwrite:
        print(f"REFUSED: {out} exists (use --overwrite)", file=sys.stderr)
        return 2
    load_vera_env()  # PLAN-ENV
    sel, prices = load_model_selection(), load_model_pricing()
    pricing = latest_pricing_for(sel.selected_model, prices) if sel else None
    if sel is None or pricing is None:
        print("FAILED: model selection or pricing record unavailable", file=sys.stderr)
        return 3
    try:
        doc, sha = capture.load_questions(ROOT / a.questions)
    except (ValueError, OSError) as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 4
    if a.overwrite and out.exists():
        out.unlink()
    try:
        summary = capture.run_capture(
            doc, a.variant, out, chat=capture.default_chat(sel.selected_model), search=capture.default_search,
            pricing=pricing, model=sel.selected_model, max_cost_usd=a.max_cost,
            splits={a.split} if a.split else None, questions_sha256=sha, limit=a.limit)
    except Exception as e:  # noqa: BLE001  (auth errors abort the run, verbosely)
        print(f"ABORTED: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        return 5
    print(f"variant={a.variant} model={sel.selected_model} questions_sha256={sha[:12]} out={out}")
    print(summary)
    return 0 if not summary["stopped"] else 6


if __name__ == "__main__":
    sys.exit(main())
