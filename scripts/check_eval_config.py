"""Pre-demo / pre-run check of the eval config (goal #2, plan 4.4 mitigation 3). DB-free, no network.

    venv/bin/python scripts/check_eval_config.py

Loads both config files through the one validated loader (pin check included), then builds the M7 fixture
corpus, judge and report. Exit 0 = good; non-zero prints every problem. Reads no .env, opens no DB connection
and makes no provider call (the fixture judge is scripted)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vera.eval_config import EvalConfigError, load_eval_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    paths = [Path(a) for a in argv[:2]]  # optional: freeze path, run path (tests use temporary copies)
    try:
        cfg = load_eval_config(*paths)
    except EvalConfigError as e:
        print(f"EVAL CONFIG INVALID:\n{e}", file=sys.stderr)
        return 1
    print(f"eval config OK: run_config_version={cfg.run_config_version} "
          f"freeze_sha256={cfg.freeze_file_sha256[:16]}... run_sha256={cfg.run_file_sha256[:16]}... "
          f"requirements_hash16={cfg.requirements_hash16} budget=${cfg.budget.cost_usd}/{cfg.budget.latency_s}s "
          f"generator={cfg.generator_model}")
    if paths:
        return 0  # the M7 fixture reads the default (real) files; a custom pair is validated only
    try:
        from vera.m7.fixture import build_corpus, build_fixture_report, build_judge
        build_corpus(), build_judge()
        fp = build_fixture_report()["evaluation"]["run_fingerprint"]
    except Exception as e:  # noqa: BLE001 - verbose, non-zero
        print(f"M7 FIXTURE FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(f"M7 fixture OK: config_sha256={fp['config_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
