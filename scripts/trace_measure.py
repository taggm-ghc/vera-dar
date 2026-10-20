"""Measure captured traces with the v1 deterministic checks (no network, no model calls).

    venv/bin/python scripts/trace_measure.py --traces tmp/trace_eval/baseline.jsonl tmp/trace_eval/grounded_v1.jsonl \
        --out eval_results/trace_eval_v1.json

Prints only aggregate numbers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vera.trace_eval import checks, measure  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traces", nargs="+")
    ap.add_argument("--out", default="eval_results/trace_eval_v1.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        fails = checks.self_test()
        w = [measure.wilson95(0, 20), measure.wilson95(10, 20), measure.wilson95(20, 20)]
        if fails:
            print("self-test FAILED: " + "; ".join(fails))
            return 1
        print(f"self-test ok ({checks.CHECKS_VERSION}); wilson 0/20, 10/20, 20/20 = {w}")
        return 0
    if not a.traces:
        ap.error("--traces is required unless --self-test")
    results = measure.measure(a.traces, a.out)
    print("\n".join(measure.aggregate_lines(results)))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
