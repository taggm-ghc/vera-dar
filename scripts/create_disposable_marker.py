"""Owner tool: disposable-DB marker creation (PLAN-ENV v2.1 §2.4 V3). INERT IN THIS BUILD.

Binding owner decision 2026-10-19: disposable verification is disabled pending an independently reviewed isolation
design, so `main()` refuses unconditionally with DISPOSABLE-VERIFY-DISABLED-pending-isolation-review (exit 3). It opens
no connection, writes no marker file and creates no nonce. The helpers below are kept as tested, inert code for that
later design; tests drive them with fake cursors only.

V3 structural zero-row rule (no tunable threshold): creation is refused when any table in any production schema
(vera_vjay and every schema named in db/migrations) holds one or more rows, or when the database name is in PD or the
connected role is in PR.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from safety import endpoint_class as ec   # noqa: E402

BASE_SCHEMAS = ("vera_vjay",)
_IDENT_RE = re.compile(r"[a-z_][a-z0-9_]{0,62}")
_SCHEMA_DECL_RE = re.compile(r"(?i)\bCREATE\s+SCHEMA\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-z_][a-z0-9_]*)")
_SCHEMA_REF_RE = re.compile(r"(?i)\b(?:TABLE|INTO|ON|UPDATE|FROM|REFERENCES)\s+(?:IF\s+NOT\s+EXISTS\s+)?"
                            r"([a-z_][a-z0-9_]*)\.[a-z_][a-z0-9_]*")
REFUSAL_CODES = ("rows_present", "dbname", "role", "bad_identifier", "probe_error")


def production_schemas(migrations_dir: Path) -> set:
    found = set(BASE_SCHEMAS)
    for p in sorted(Path(migrations_dir).glob("*.sql")):
        text = p.read_text(encoding="utf-8", errors="replace")
        found |= {m.lower() for m in _SCHEMA_DECL_RE.findall(text)}
        found |= {m.lower() for m in _SCHEMA_REF_RE.findall(text) if m.lower() not in ("pg_catalog",
                                                                                         "information_schema")}
    return found


def structural_refusal(cur, schemas, sources) -> str | None:
    """None when creation may proceed; else a refusal code. Order: identity (PD, PR) first, then the zero-row rule."""
    try:
        cur.execute("SELECT current_database(), current_user")
        db, role = cur.fetchone()
        if sources.has_dbname(db):
            return "dbname"
        if sources.has_role(role):
            return "role"
        names = sorted(schemas)
        if not all(_IDENT_RE.fullmatch(s) for s in names):
            return "bad_identifier"
        cur.execute("SELECT table_schema, table_name FROM information_schema.tables "
                    "WHERE table_type = 'BASE TABLE' AND table_schema = ANY(%s)", (names,))
        tables = cur.fetchall()
        for schema, table in tables:
            if not (_IDENT_RE.fullmatch(schema) and _IDENT_RE.fullmatch(table)):
                return "bad_identifier"
            cur.execute(f'SELECT EXISTS (SELECT 1 FROM "{schema}"."{table}")')  # nosec B608 - schema/table fullmatch _IDENT_RE above, then quoted
            if cur.fetchone()[0]:
                return "rows_present"
        return None
    except Exception:
        return "probe_error"


def make_tag(nonce: bytes, dbname: str, oid: int) -> str:
    """HMAC-SHA256(N, 'vera-disposable-v1|' + dbname + '|' + oid). Identical to the inert verifier's tag."""
    return hmac.new(nonce, f"{ec.MARKER_TAG_PREFIX}{dbname}|{int(oid)}".encode(), hashlib.sha256).hexdigest()


def main(argv=None) -> int:
    print(f"create_disposable_marker refused: {ec.VERIFY_DISABLED}", file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())
