"""Standalone connectivity check — run locally before pushing.

    python scripts/test_db_connection.py

Prints only a success flag and the safety classification code (never host, port, user or dbname).
Reads EXTERNAL_DB_URL from R/config/env/.env-vera (or INTERNAL_DB_URL if RENDER=true), opens a
connection, runs `select 1`, and reports pass/fail. Touches no tables.

Ops class: ops-read. Installs connect_guard (app context) and refuses under a test runner.
"""
import os
import sys
from pathlib import Path

from vera.env_files import load_vera_env
from sqlalchemy import text

load_vera_env()  # PLAN-ENV

from vera.db import get_engine  # noqa: E402  (after load_vera_env, matches main.py's ordering)


def main() -> int:
    if os.environ.get("VERA_TEST_RUNNER") or os.environ.get("PYTEST_CURRENT_TEST"):
        print("refused: ops script, not for test runs", file=sys.stderr)
        return 3
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from safety import connect_guard
    connect_guard.install("production")  # classify-and-tag (app context); never disposable
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("select 1"))
            verdict = connect_guard.verdict_of(conn)
            safety_class = getattr(verdict, "cls", None) or "UNTAGGED"
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}", file=sys.stderr)  # C12: names only
        return 1
    print(f"OK: connected=true safety_class={safety_class}")   # M5: no host, port, user or dbname is printed
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
