"""Runtime SQL guard (PLAN-ENV v2.1 §3 E8/C7, L3; guard file).

Every connection that is not DISPOSABLE_VERIFIED is PRODUCTION (C1), and DISPOSABLE_VERIFIED is unreachable in this
build (owner decision 2026-10-19), so the allowlist applies to every non-SQLite connection. This is defence in depth
on top of the connect guard (C5) and env stripping (C3); it is a best-effort lexical check, not a SQL parser.

Allowed first verbs: SELECT, INSERT, UPDATE (with a top-level, non-tautological WHERE), WITH (no DELETE, TRUNCATE,
DDL, or UPDATE without WHERE), BEGIN, COMMIT, ROLLBACK, SAVEPOINT, RELEASE, SET LOCAL, SHOW. Everything else is refused
with DestructiveRefused(code). The SQL text never appears in an exception.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

VERDICT_KEY = "vera_verdict"


def _sibling(name: str):
    try:
        return __import__(f"safety.{name}", fromlist=["x"])
    except ImportError:
        priv = f"_vera_safety_{name}"
        if priv in sys.modules:
            return sys.modules[priv]
        spec = importlib.util.spec_from_file_location(priv, Path(__file__).with_name(f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[priv] = mod
        spec.loader.exec_module(mod)
        return mod


_ec = _sibling("endpoint_class")


class DestructiveRefused(Exception):
    """Refused SQL. Carries a code only, never the statement."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code

    def __str__(self) -> str:
        return f"destructive SQL refused: {self.code}"

    def __repr__(self) -> str:
        return f"DestructiveRefused({self.code!r})"


_SHARED = sys.modules.setdefault("_vera_safety_shared_state", type(sys)("_vera_safety_shared_state"))
DestructiveRefused = _SHARED.__dict__.setdefault("DestructiveRefused", DestructiveRefused)


CODES = ("empty", "unterminated", "multi_statement", "verb_not_allowed", "update_without_where", "tautology",
         "forbidden_keyword", "denied_function", "session_set", "not_disposable")

_DOLLAR_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$")
_FORBIDDEN_RE = re.compile(r"\b(DELETE|TRUNCATE|DROP|ALTER|CREATE|GRANT|REVOKE|COPY|VACUUM|CALL|MERGE|REINDEX|"
                           r"EXECUTE|PREPARE|IMPORT)\b")
_DENIED_FN_RE = re.compile(r"\b(DBLINK_EXEC|DBLINK|PG_TERMINATE_BACKEND|PG_CANCEL_BACKEND|LO_UNLINK|LO_IMPORT|"
                           r"LO_EXPORT|PG_READ_FILE|PG_READ_BINARY_FILE|PG_WRITE_\w*)\s*\(")
_SET_CONFIG_RE = re.compile(r"\bSET_CONFIG\s*\(")
_TXN_VERBS = {"BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "SHOW"}
_EXEMPT_BEFORE_UPDATE = {"FOR", "KEY", "DO"}     # SELECT ... FOR [NO KEY] UPDATE; ON CONFLICT DO UPDATE
_TRUE_ONLY_RE = re.compile(r"\(*\s*(TRUE|NOT\s+FALSE|1|'T')\s*\)*")
_SELF_EQ_RE = re.compile(r"\(*\s*([\w.]+)\s*(=|>=|<=|IS\s+NOT\s+DISTINCT\s+FROM)\s*\1\s*\)*")
_CONST_CMP_RE = re.compile(r"\(*\s*\d+\s*(=|<>|!=|>|<|>=|<=)\s*\d+\s*\)*")
_OR_TRUE_RE = re.compile(r"\bOR\s+\(*\s*(TRUE|NOT\s+FALSE|([\w.]+)\s*=\s*\2\b|\d+\s*(=|<>|!=|>|<|>=|<=)\s*\d+)")


def _tok(kind: str, body: str) -> str:
    return f" __{kind}{hashlib.blake2s(body.encode('utf-8', 'surrogatepass'), digest_size=4).hexdigest()}__ "


def strip_sql(sql: str) -> str:
    """Remove comments; replace string, quoted-identifier and dollar-quoted literals with content-keyed tokens."""
    out: list = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c == "-" and sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j < 0 else j
            out.append(" ")
        elif c == "/" and sql.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if sql.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif sql.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            if depth:
                raise DestructiveRefused("unterminated")
            out.append(" ")
        elif c in "'\"":
            escapes = c == "'" and i > 0 and sql[i - 1] in "eE" and (i < 2 or not (sql[i - 2].isalnum() or sql[i - 2] == "_"))
            if escapes and out and out[-1] in "eE":
                out.pop()
            j, buf = i + 1, []
            while True:
                if j >= n:
                    raise DestructiveRefused("unterminated")
                if escapes and sql[j] == "\\":
                    buf.append(sql[j:j + 2])
                    j += 2
                    continue
                if sql[j] == c:
                    if j + 1 < n and sql[j + 1] == c:
                        buf.append(c)
                        j += 2
                        continue
                    break
                buf.append(sql[j])
                j += 1
            out.append(_tok("l" if c == "'" else "q", "".join(buf)))
            i = j + 1
        elif c == "$":
            m = _DOLLAR_RE.match(sql, i)
            if m and not (i > 0 and (sql[i - 1].isalnum() or sql[i - 1] == "_")):
                end = sql.find(m.group(0), m.end())
                if end < 0:
                    raise DestructiveRefused("unterminated")
                out.append(_tok("l", sql[m.end():end]))
                i = end + len(m.group(0))
            else:
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _matching_paren_body(s: str, open_idx: int) -> str:
    depth, j = 0, open_idx
    while j < len(s):
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                return s[open_idx + 1:j]
        j += 1
    return s[open_idx + 1:]


def _split_depth0(s: str, sep: str = ",") -> list:
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _check_updates(s: str) -> None:
    for m in re.finditer(r"\bUPDATE\b", s):
        before = s[:m.start()].split()
        if before and before[-1] in _EXEMPT_BEFORE_UPDATE:
            continue
        depth, j, seg = 0, m.end(), []
        while j < len(s):
            ch = s[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    break
            seg.append(ch if depth == 0 or ch in "()" else " ")
            j += 1
        flat = "".join(seg)
        wm = re.search(r"\bWHERE\b", flat)
        if wm is None:
            raise DestructiveRefused("update_without_where")
        # the WHERE clause text, with nested parentheses kept, up to RETURNING at depth 0
        start = m.end() + wm.end()
        clause = s[start:j]
        rm = re.search(r"\bRETURNING\b", clause)
        if rm:
            clause = clause[:rm.start()]
        c = " ".join(clause.split())
        if (not c or _TRUE_ONLY_RE.fullmatch(c) or _SELF_EQ_RE.fullmatch(c) or _CONST_CMP_RE.fullmatch(c)
                or _OR_TRUE_RE.search(c)):
            raise DestructiveRefused("tautology")


def _check_functions(s: str) -> None:
    if _DENIED_FN_RE.search(s):
        raise DestructiveRefused("denied_function")
    for m in _SET_CONFIG_RE.finditer(s):
        args = _split_depth0(_matching_paren_body(s, m.end() - 1))
        if len(args) != 3 or args[2].strip() != "TRUE":      # only transaction-local set_config(..., true)
            raise DestructiveRefused("denied_function")


def check_sql(statement) -> None:
    """Raise DestructiveRefused(code) unless the statement is on the allowlist."""
    if isinstance(statement, (bytes, bytearray)):
        statement = bytes(statement).decode("utf-8", "replace")
    if not isinstance(statement, str):
        raise DestructiveRefused("empty")
    s = strip_sql(statement).upper()
    parts = [p for p in s.split(";") if p.strip()]
    if not parts:
        raise DestructiveRefused("empty")
    if len(parts) > 1:
        raise DestructiveRefused("multi_statement")
    s = parts[0]
    words = s.replace("(", " ").split()
    verb = words[0]
    if verb == "SET":
        if len(words) < 2 or words[1] != "LOCAL":
            raise DestructiveRefused("session_set")
    elif verb not in {"SELECT", "INSERT", "UPDATE", "WITH"} | _TXN_VERBS:
        raise DestructiveRefused("verb_not_allowed")
    if _FORBIDDEN_RE.search(s):
        raise DestructiveRefused("forbidden_keyword")
    _check_functions(s)
    _check_updates(s)


def _is_sqlite(conn) -> bool:
    return getattr(getattr(conn, "dialect", None), "name", "") == "sqlite"


def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    if _is_sqlite(conn):
        return None
    try:
        verdict = conn.info.get(VERDICT_KEY)
    except Exception:
        verdict = None
    if verdict is not None and getattr(verdict, "cls", None) == _ec.DISPOSABLE_VERIFIED:   # unreachable
        return None
    check_sql(statement)
    return None


_STATE = _SHARED.__dict__.setdefault("sql_guard_state", {"listener": None})


def install_listener() -> None:
    from sqlalchemy import event
    from sqlalchemy.engine import Engine
    if _STATE["listener"] is None:
        _STATE["listener"] = before_cursor_execute
    if not event.contains(Engine, "before_cursor_execute", _STATE["listener"]):
        event.listen(Engine, "before_cursor_execute", _STATE["listener"])


def uninstall_listener() -> None:
    """TEST HELPER ONLY."""
    from sqlalchemy import event
    from sqlalchemy.engine import Engine
    if _STATE["listener"] is not None and event.contains(Engine, "before_cursor_execute", _STATE["listener"]):
        event.remove(Engine, "before_cursor_execute", _STATE["listener"])


def _as_text(query, raw_conn) -> str:
    if hasattr(query, "as_string"):
        return query.as_string(raw_conn)
    return query


class GuardedCursor:
    """Raw psycopg2 cursor proxy: every statement passes check_sql first."""

    def __init__(self, cur, raw_conn):
        object.__setattr__(self, "_cur", cur)
        object.__setattr__(self, "_raw_conn", raw_conn)

    def execute(self, query, vars=None):
        check_sql(_as_text(query, self._raw_conn))
        return self._cur.execute(query, vars)

    def executemany(self, query, vars_list):
        check_sql(_as_text(query, self._raw_conn))
        return self._cur.executemany(query, vars_list)

    def callproc(self, *a, **k):
        raise DestructiveRefused("verb_not_allowed")

    def copy_expert(self, *a, **k):
        raise DestructiveRefused("forbidden_keyword")

    copy_from = copy_to = copy_expert

    def __getattr__(self, name):
        return getattr(self._cur, name)

    def __setattr__(self, name, value):
        setattr(self._cur, name, value)

    def __iter__(self):
        return iter(self._cur)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._cur.__exit__(*exc)

    def __repr__(self) -> str:
        return "GuardedCursor(<guarded>)"


class GuardedConnection:
    """Raw psycopg2 connection proxy returned by the connect guard in app contexts."""

    def __init__(self, raw_conn, verdict):
        object.__setattr__(self, "_raw", raw_conn)
        object.__setattr__(self, "_vera_verdict", verdict)

    def cursor(self, *a, **k):
        return GuardedCursor(self._raw.cursor(*a, **k), self._raw)

    def __getattr__(self, name):
        return getattr(self._raw, name)

    def __setattr__(self, name, value):
        setattr(self._raw, name, value)

    def __enter__(self):
        self._raw.__enter__()
        return self

    def __exit__(self, *exc):
        return self._raw.__exit__(*exc)

    def __repr__(self) -> str:
        return "GuardedConnection(<guarded>)"


def wrap_connection(raw_conn, verdict) -> GuardedConnection:
    return GuardedConnection(raw_conn, verdict)
