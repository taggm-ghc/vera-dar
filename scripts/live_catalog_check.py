"""Operator script (K2.S-a, owner OD-S1): check that every (table, column) named in VERA's SQL exists in the live
catalog of schema vera_vjay. It replaces the former pytest test, because pytest must never connect to production.

Ops class: ops-read. Run as a script it installs connect_guard (app context) and refuses under a test runner.

NOT a test and never imported by tests. Agents never run it. It connects ONLY through VERA_DB_URL_RO, whose user
must be the read-only account `vera_eval_ro`, in a read-only session, after a privilege and read-only guard. It prints only counts and missing table.column names;
it never prints the host, user, URL, password or an exception message (class name only).

    python scripts/live_catalog_check.py --i-confirm-owner-run

Exit codes: 0 PASS, 1 FAIL, 2 missing confirmation flag (nothing was read), 3 guard failed, 4 cannot verify.

Result contract (owner-approved): the LAST output line is exactly one of
    RESULT: PASS            the checks completed and the live catalog satisfies the contract
    RESULT: FAIL            the checks completed and found a contract violation (missing table.column lines above it)
    RESULT: CANNOT_VERIFY   the checks could not be completed reliably (refused run, connection failure, missing
                            permission or credentials, failed guard, uninspectable schema, any unexpected error);
                            NEVER PASS in these cases
The first line is `live_catalog_check script_sha256=<hex of this file>`. No traceback and no exception message is ever
printed (class name only). Save the output with `> catalog-evidence.txt` and pass it to the release gate with
--catalog-evidence (the gate only reads that file).
"""
import hashlib
import os
import sys
from pathlib import Path
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
CONFIRM_FLAG = "--i-confirm-owner-run"
ROLE = "vera_eval_ro"
SCHEMA = "vera_vjay"
GUARD_QUERY = (
    "select current_user, r.rolsuper, r.rolcreaterole, r.rolcreatedb, "
    "has_schema_privilege(current_user, %s, 'CREATE'), "
    "current_setting('transaction_read_only'), "
    "coalesce((select bool_or(has_table_privilege(current_user, c.oid, 'INSERT,UPDATE,DELETE,TRUNCATE')) "
    "          from pg_class c join pg_namespace n on n.oid = c.relnamespace "
    "          where n.nspname = %s and c.relkind in ('r','p','v','m','f')), false) "
    "from pg_roles r where r.rolname = current_user")
CATALOG_QUERY = "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = %s"


def connect():
    """A read-only psycopg2 connection from VERA_DB_URL_RO (user must be the fixed role), or None if absent."""
    import psycopg2
    from urllib.parse import unquote
    raw = os.getenv("VERA_DB_URL_RO") or ""
    u = urlsplit(raw)
    if not (u.hostname and u.path.lstrip("/") and unquote(u.username or "") == ROLE and u.password):
        return None
    dsn = (f"postgresql://{quote(ROLE, safe='')}:{quote(unquote(u.password), safe='')}@{u.hostname}:{u.port or 5432}"
           f"/{u.path.lstrip('/')}")
    conn = psycopg2.connect(dsn, connect_timeout=10, sslmode="require",
                            options="-c default_transaction_read_only=on -c statement_timeout=20000")
    conn.set_session(readonly=True, autocommit=True)  # before any execute: no open transaction yet
    return conn


def privilege_ok(conn) -> bool:
    """Fixed role; not super/createrole/createdb; no CREATE on the schema; read-only session; no table write privilege."""
    with conn.cursor() as cur:
        cur.execute(GUARD_QUERY, (SCHEMA, SCHEMA))
        row = cur.fetchone()
    if row is None:
        return False
    user, sup, crole, cdb, create, ro, write = row
    return user == ROLE and ro == "on" and not (sup or crole or cdb or create or write)


def live_catalog(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute(CATALOG_QUERY, (SCHEMA,))
        cat: dict = {}
        for t, c in cur.fetchall():
            cat.setdefault(t, set()).add(c)
    return cat


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _run() -> tuple:
    """(exit code, result token). Prints only fixed messages, counts and repo-derived table.column names."""
    sys.path.insert(0, str(ROOT))
    from tests import sql_contract_support as sc
    try:
        conn = connect()
    except Exception as exc:  # noqa: BLE001  (class name only)
        print(f"cannot verify: connection failed ({type(exc).__name__})")
        return 4, "CANNOT_VERIFY"
    if conn is None:
        print("cannot verify: read-only credentials absent or not the read-only account")
        return 4, "CANNOT_VERIFY"
    try:
        if not privilege_ok(conn):
            print("ABORT: guard failed (not the read-only account, session not read-only, or elevated, CREATE or "
                  "table write privileges)")
            return 3, "CANNOT_VERIFY"
        cat = live_catalog(conn)
    except Exception as exc:  # noqa: BLE001
        print(f"cannot verify: query failed ({type(exc).__name__})")
        return 4, "CANNOT_VERIFY"
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
    if not cat:
        print("cannot verify: schema has no visible columns for this account")
        return 4, "CANNOT_VERIFY"
    found = sc.collect()
    problems = [f"{t}.{c}" for (t, c) in sorted(found) if t not in cat or c not in cat[t]]
    print(f"checked {len(found)} (table, column) pairs; {len(problems)} missing")
    for p in problems:
        print(f"missing: {p}")
    return (1, "FAIL") if problems else (0, "PASS")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    digest = script_sha256()
    print(f"live_catalog_check script_sha256={digest}")  # first line, before the confirm-flag check
    if CONFIRM_FLAG not in argv:
        print(f"refused: pass {CONFIRM_FLAG} (owner-run only); nothing was read")
        print("RESULT: CANNOT_VERIFY")
        return 2
    try:
        code, result = _run()
    except BaseException as exc:  # noqa: BLE001  (any unexpected error still ends with exactly one RESULT line)
        print(f"cannot verify: unexpected error ({type(exc).__name__})")
        code, result = 4, "CANNOT_VERIFY"
    print(f"RESULT: {result}")
    return code


def _ops_guard() -> None:
    """ENV-E (E7/C8): refuse under a test runner; classify-and-tag every connection as an APP context (never disposable)."""
    if os.environ.get("VERA_TEST_RUNNER") or os.environ.get("PYTEST_CURRENT_TEST"):
        digest = script_sha256()
        print(f"live_catalog_check script_sha256={digest}")
        print("refused: ops script, not for test runs; nothing was read")
        print("RESULT: CANNOT_VERIFY")
        raise SystemExit(2)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from safety import connect_guard
    connect_guard.install("production")


if __name__ == "__main__":
    _ops_guard()
    sys.exit(main())
