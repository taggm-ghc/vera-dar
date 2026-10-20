"""Stdlib-only guards for db/apply_migrations.py (P1-MIG staged patch, NOT INTEGRATED).

Plan: p3m3/vera-gr/plan-migration-tooling-hardening-2026-10-17.md r2 (sections 2.1-2.5, V-1..V-9).
This module never imports a DB driver, sqlalchemy, dotenv or week-1v2 db.py, never resolves DNS and
never opens a socket. It never stores or prints a password or a host name.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit

# ---------------------------------------------------------------- exit codes
EXIT_OK, EXIT_DRIFT, EXIT_REFUSED, EXIT_IDENTITY, EXIT_CONNECT = 0, 1, 2, 3, 4


class Refusal(Exception):
    """Fail-closed refusal. Message must never contain a host or a credential."""

    def __init__(self, msg: str, code: int = EXIT_REFUSED):
        super().__init__(msg)
        self.code = code


# ---------------------------------------------------------------- 2.1 strict ids (N-1)
PREFIX_RE = re.compile(r"[0-9]{3}")                      # used with fullmatch only (V-5(1))
FILE_RE = re.compile(r"([0-9]{3})_vera_[a-z0-9_]+\.sql")  # fullmatch only
APPROVAL_RE = re.compile(r"OD-[A-Za-z0-9]+")             # fullmatch only (V-6)
LOCAL_DB_RE = re.compile(r"vera_local_[a-z0-9_]+")       # fullmatch only
# K18W INC-3 (WS-8a F-2): the bootstrap is recognised by exact pinned identity (name AND reviewed content hash), never
# by a name pattern. A changed 008 needs a reviewed edit of BOOTSTRAP_SHA256 here (the pin is part of the review).
BOOTSTRAP_NAME = "008_vera_schema_migrations.sql"
BOOTSTRAP_SHA256 = "06b795bc74942dba84b46ee520c34d1d84551bada938913f973ffc3528aa8aaa"
BOOTSTRAP_RE = re.compile(re.escape(BOOTSTRAP_NAME))   # kept for callers; matches the single pinned name only


def is_bootstrap(name: str, data: bytes) -> bool:
    """True only for the pinned bootstrap file: exact name and exact reviewed bytes."""
    return name == BOOTSTRAP_NAME and hmac.compare_digest(sha256_hex(data), BOOTSTRAP_SHA256)


def validate_prefixes(args: list[str]) -> list[str]:
    """Every id is exactly three ASCII digits; no duplicates; strictly ascending (2.1)."""
    if not args:
        raise Refusal("no migration id given")
    for n, a in enumerate(args, 1):
        if not isinstance(a, str) or PREFIX_RE.fullmatch(a) is None:
            # K18W (QA F16): the argument is never echoed (a pasted password must not reach logs); position only.
            raise Refusal(f"invalid migration id (id argument {n}): must be exactly three ASCII digits (no paths, globs or suffixes)")
    if len(set(args)) != len(args):
        raise Refusal("duplicate migration id")
    if args != sorted(args):
        raise Refusal("migration ids must be strictly ascending")
    return args


def resolve(migrations_dir: Path, prefix: str) -> Path:
    """Top-level, regular, non-symlink file; exact name pattern; exactly one match. No glob (2.1)."""
    PREFIX_RE.fullmatch(prefix) or _raise(f"invalid migration id {prefix!r}")
    base = migrations_dir.resolve()
    hits = []
    with os.scandir(base) as it:
        for e in it:
            m = FILE_RE.fullmatch(e.name)
            if m is None or m.group(1) != prefix:
                continue
            if e.name.endswith("_rollback.sql"):
                raise Refusal(f"{e.name}: rollback-named file at top level is never applied (V-5(2))")
            if e.is_symlink() or not e.is_file(follow_symlinks=False):
                raise Refusal(f"{e.name}: not a regular file (symlinks refused)")
            hits.append(Path(e.path))
    if not hits:
        raise Refusal(f"migration {prefix} not found at top level of migrations/ (pending/ and rollback/ are never reachable)")
    if len(hits) > 1:
        raise Refusal(f"migration {prefix} is ambiguous ({len(hits)} files)")
    f = hits[0]
    if f.resolve().parent != base:
        raise Refusal(f"{f.name}: resolves outside migrations/")
    return f


def _raise(msg: str):
    raise Refusal(msg)


# ---------------------------------------------------------------- lint (V-5(2)-(4), V-9(b))
# Header rule, refined from V-5(2): a case-sensitive whole word ROLLBACK or DESTRUCTIVE, not preceded by a
# letter or '-'. The plan's bare-substring rule would refuse 001u ("NON-DESTRUCTIVE", line 1) and 006
# ("Rollback (destructive ...)", line 5) - see design.md PD-1.
HEADER_REFUSE_RE = re.compile(r"(?<![A-Za-z-])(ROLLBACK|DESTRUCTIVE)(?![A-Za-z])")
NEVER_VIA_RE = re.compile(r"never\s+(run\s+)?via\s+(db/)?apply_migrations\.py", re.I)
# K18W INC-4 (WS-8d N-3): the joined header is also matched with backticks/quotes removed, and the path may be split at
# any '/' (e.g. "via db/" + newline + "apply_migrations.py") or carry a leading path ("./db/", "vera/db/").
NEVER_VIA_JOINED_RE = re.compile(r"never\s+(run\s+)?via\s+(?:[\w.-]+\s*/\s*)*apply_migrations\.py", re.I)
HEADER_QUOTES_RE = re.compile(r"[`'\"\u2018\u2019\u201c\u201d]")
# r2 (review R-5): the same phrase with ALL whitespace removed, so a path split inside a word ("apply_" / "migrations.py")
NEVER_VIA_NOWS_RE = re.compile(r"never(run)?via(?:[\w.-]+/)*apply_migrations\.py", re.I)
FRESH_RE = re.compile(r"FRESH INSTALL ONLY")
# K18W INC-3 (WS-8a F-3): header lines are matched with their comment markers removed AND joined, so a warning that
# wraps across lines ("Never run via\n-- db/apply_migrations.py") is still found.
HEADER_MARK_RE = re.compile(r"^\s*(--+|/\*+|\*+/?)?\s?")
TOP_TXN_RE = re.compile(r"(BEGIN|START\s+TRANSACTION|COMMIT|ROLLBACK|END|ABORT|SAVEPOINT|RELEASE|PREPARE\s+TRANSACTION)\b", re.I)
# K18W (QA F12): statement-level session changes would leak into later files or defeat the tool's own SET LOCAL
# timeouts and identity check. Matched at statement start only (UPDATE ... SET col = ... is not affected).
TOP_SESSION_RE = re.compile(
    r"(SET|RESET)\s+(SESSION\s+|LOCAL\s+)?(ROLE|SESSION\s+AUTHORIZATION|AUTHORIZATION|statement_timeout|lock_timeout|"
    r"search_path|default_transaction_read_only|transaction_read_only|idle_in_transaction_session_timeout|ALL)\b", re.I)
# K18W (QA F1): Postgres ends a -- comment at \r as well as \n; a bare CR (and NUL / bidi controls) can hide statements
# from this lint and from a human reviewer, so any of them is a hard refusal.
HIDDEN_CHARS_RE = re.compile(r"\r(?!\n)|\x00|[\u202a-\u202e\u2066-\u2069\u200b-\u200f\ufeff]")
# K18W INC-3 (WS-8b write opt-in): a migration body must never change the transaction or session read/write mode,
# the tool's own per-file opt-in is the only one. Matched on comment-stripped text that keeps literals and dollar
# bodies, so set_config('default_transaction_read_only', ...) and EXECUTE '...' forms are caught too.
WRITE_MODE_RE = [
    (re.compile(r"\bSET\s+SESSION\s+CHARACTERISTICS\b", re.I), "SET SESSION CHARACTERISTICS"),
    (re.compile(r"\b(default_)?transaction_read_only\b", re.I), "transaction_read_only / default_transaction_read_only"),
    (re.compile(r"\bREAD\s+WRITE\b", re.I), "READ WRITE transaction mode"),
    # K18W INC-4 (WS-8d N-1): role- or database-level RESET ALL silently removes a persistent
    # default_transaction_read_only=on (DB-design control C1a). Covers ALTER ROLE|USER|DATABASE ... [IN DATABASE ...]
    # RESET ALL, any case/whitespace/comment spacing, and a double-quoted name that contains ';'.
    (re.compile(r'\bALTER\s+(ROLE|USER|DATABASE)\b(?:"[^"]*(?:""[^"]*)*"(?!")|[^;"])*?\bRESET\s+ALL\b', re.I),
     "role/database-level RESET ALL (removes default_transaction_read_only)"),
    # Explicit form of the named role/database-level SET/RESET (also caught by the token rule above; labelled here).
    (re.compile(r'\bALTER\s+(ROLE|USER|DATABASE)\b(?:"[^"]*(?:""[^"]*)*"(?!")|[^;"])*?\b(SET|RESET)\s+"?'
                r'(default_)?transaction_read_only\b', re.I),
     "role/database-level SET/RESET of default_transaction_read_only"),
    # Direct catalog edits of per-role/per-database settings bypass ALTER ROLE/DATABASE entirely.
    (re.compile(r"\bpg_db_role_setting\b", re.I), "pg_db_role_setting catalog reference"),
]
# r2 (review R-1): RESET ALL anywhere (any statement, literal or identifier) is refused, matched on comment-stripped text
# from both the INC-3 stripper and a nesting-aware, quote-aware one, so a nested /* /* */ ; */ comment or a quoted
# name that desynchronises strip_comments() cannot hide it. Fail-closed: a literal 'RESET ALL' is refused too (X-7).
RESET_ALL_ANY_RE = re.compile(r"\bRESET\s+ALL\b", re.I)
# r2 (review R-3): renaming or creating a role/database moves or drops per-role/per-database settings (advisory).
ROLE_DDL_ADVISORY = [
    re.compile(r'\bALTER\s+(ROLE|USER|GROUP|DATABASE)\b(?:"[^"]*(?:""[^"]*)*"(?!")|[^;"])*?\bRENAME\b', re.I),
    re.compile(r"\bCREATE\s+(ROLE|USER|GROUP|DATABASE)\b", re.I),
]
# K18W INC-3 (WS-8a F-6/V11): small additions to the advisory lint. Hard: statements that escape the database
# boundary or change server/session state from inside dynamic SQL.
EXTRA_HARD = [
    (re.compile(r"\bCOPY\b[^;]*\b(FROM|TO)\s+PROGRAM\b", re.I), "COPY ... PROGRAM"),
    (re.compile(r"\bALTER\s+SYSTEM\b", re.I), "ALTER SYSTEM"),
    (re.compile(r"\bset_config\s*\(", re.I), "set_config(...) call"),
    # statement start inside dynamic SQL or a PL/pgSQL body (after a quote, ';', or BEGIN/THEN/ELSE/LOOP/EXECUTE);
    # top-level statements are covered by TOP_SESSION_RE, and `UPDATE t SET role = ...` is not matched.
    (re.compile(r"(?:['$;]|\b(?:BEGIN|THEN|ELSE|LOOP|EXECUTE)\b)\s*['\"]?\s*(SET|RESET)\s+(SESSION\s+|LOCAL\s+)?"
                r"(ROLE|SESSION\s+AUTHORIZATION)\b", re.I),
     "role/authorization change inside dynamic SQL or a PL/pgSQL body"),
]
HARD_DESTRUCTIVE = [
    re.compile(p, re.I) for p in (
        r"\bDROP\s+(TABLE|SCHEMA|DATABASE|ROLE|USER|OWNED|VIEW|MATERIALIZED\s+VIEW|SEQUENCE|TYPE|EXTENSION)\b",
        r"\bTRUNCATE\s+(?!ON\b)(TABLE\s+)?[A-Za-z_\"]", r"\bDELETE\s+FROM\b", r"\bALTER\s+TABLE\b[^;]*\bDROP\s+(COLUMN\b|(?!CONSTRAINT|DEFAULT|NOT\b|IDENTITY|EXPRESSION)[A-Za-z_\"])",
        r"\bREASSIGN\s+OWNED\b",
        # K18W INC-3 (V11): further object classes and data-removing forms
        r"\bDROP\s+(DOMAIN|FOREIGN\s+TABLE|PROCEDURE|AGGREGATE|SERVER|PUBLICATION|SUBSCRIPTION|COLLATION|ROUTINE)\b",
        r"\bMERGE\b[^;]*\bTHEN\s+DELETE\b", r"\bALTER\s+(TABLE|VIEW|SEQUENCE|SCHEMA)\b[^;]*\bRENAME\b",
    )
]
ADVISORY_DESTRUCTIVE = [
    re.compile(p, re.I) for p in (
        r"\bDROP\s+(TRIGGER|FUNCTION|INDEX|POLICY|RULE)\b", r"\bALTER\s+TABLE\b[^;]*\bDROP\s+(CONSTRAINT|DEFAULT|NOT\s+NULL)\b",
        r"\bREVOKE\b",
    )
]


def strip_sql(text: str) -> str:
    """Remove -- and /* */ comments, '...' and "..." literals and $tag$...$tag$ bodies (replaced by a space)."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("--", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
        elif c in "'\"":
            j = i + 1
            while j < n:
                if text[j] == c:
                    if j + 1 < n and text[j + 1] == c:
                        j += 2
                        continue
                    break
                j += 1
            i = j + 1
            out.append(" '' " if c == "'" else ' "q" ')
        elif c == "$" and not (i > 0 and (text[i - 1].isalnum() or text[i - 1] in "_$")):
            # K18W (QA F2): '$' inside an identifier (t$a$) is not a dollar-quote opener in Postgres.
            m = re.match(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$", text[i:])
            if m:
                tag = m.group(0)
                j = text.find(tag, i + len(tag))
                i = n if j < 0 else j + len(tag)
                out.append(" $body$ ")
            else:
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def strip_comments_nested(text: str) -> str:
    """r2: remove -- and NESTED /* */ comments (Postgres semantics) outside '...' literals and "..." identifiers;
    keeps literal, identifier and dollar-body contents. Used only for the RESET ALL refusal."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c in "'\"":
            j = i + 1
            while j < n:
                if text[j] == c:
                    if j + 1 < n and text[j + 1] == c:
                        j += 2
                        continue
                    break
                j += 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("--", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            i = j
            out.append(" ")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def strip_comments(text: str) -> str:
    """Remove -- and /* */ comments only (outside '...' literals); keeps literals and dollar bodies."""
    out, i, n, q = [], 0, len(text), None
    while i < n:
        c = text[i]
        if q:
            out.append(c)
            if c == q:
                q = None
            i += 1
        elif c == "'":
            q = c
            out.append(c)
            i += 1
        elif text.startswith("--", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _depth0(s: str) -> str:
    """Blank everything inside (...) (literals are already removed by strip_sql); unbalanced ')' is ignored."""
    out, d = [], 0
    for c in s:
        if c == "(":
            d += 1
            out.append(" ")
        elif c == ")":
            d = max(d - 1, 0)
            out.append(" ")
        else:
            out.append(c if d == 0 else " ")
    return "".join(out)


@dataclass
class Lint:
    hard: list[str] = field(default_factory=list)          # refusals in every mode
    destructive: list[str] = field(default_factory=list)   # need --allow-destructive <same ID>
    advisory: list[str] = field(default_factory=list)


def lint_sql(name: str, text: str) -> Lint:
    lint = Lint()
    if HIDDEN_CHARS_RE.search(text):
        lint.hard.append("bare CR, NUL or invisible/bidi control character (can hide statements)")
    head = text.splitlines()[:10]
    for ln in head:
        if HEADER_REFUSE_RE.search(ln):
            lint.hard.append("header marks file ROLLBACK/DESTRUCTIVE")
            break
    joined = " ".join(HEADER_MARK_RE.sub("", ln, count=1) for ln in head)   # INC-3 (F-3): multi-line header phrases
    unquoted = HEADER_QUOTES_RE.sub("", joined)                              # INC-4 (N-3): `db/apply_migrations.py`
    if any(NEVER_VIA_RE.search(ln) for ln in head) or NEVER_VIA_RE.search(joined) or \
            NEVER_VIA_JOINED_RE.search(unquoted) or NEVER_VIA_NOWS_RE.search("".join(unquoted.split())):
        lint.hard.append("header says never via apply_migrations.py")
    if any(FRESH_RE.search(ln) for ln in head) or FRESH_RE.search(" ".join(joined.split())):
        lint.hard.append("header says FRESH INSTALL ONLY")
    if "vera_local_meta" in text:
        lint.hard.append("mentions vera_local_meta (V-9(b))")
    body = strip_sql(text)
    if re.search(r"\bDROP\s+SCHEMA\b", body, re.I):
        lint.hard.append("DROP SCHEMA")
    elif re.search(r"\bDROP\s+SCHEMA\b", strip_comments(text), re.I):
        lint.hard.append("DROP SCHEMA (inside a DO body or literal)")   # r3 ruling: hard refusal
    if re.search(r"\bCONCURRENTLY\b", body, re.I):
        lint.hard.append("CONCURRENTLY (cannot run in a transaction)")
    for stmt in body.split(";"):
        s = stmt.strip()
        if s and TOP_TXN_RE.match(s):
            lint.hard.append(f"top-level transaction control: {s.split()[0].upper()}")
        elif s and TOP_SESSION_RE.match(s):
            lint.hard.append(f"session-level setting changed: {' '.join(s.split()[:2]).upper()}")
        if s and re.match(r"UPDATE\b", s, re.I) and not re.search(r"\bWHERE\b", s, re.I):
            lint.destructive.append("UPDATE WITHOUT WHERE")   # INC-3 (V11): bulk update of every row
        elif s and re.match(r"(UPDATE|WITH)\b", s, re.I):
            # INC-4 (WS-8d N-4): WITH ... UPDATE, or a WHERE that sits only inside a subquery. Parenthesised text is
            # blanked so only a top-level UPDATE and a top-level WHERE after it count. Heuristic, advisory.
            # r2 (review R-4): the main verb is the first top-level SELECT/INSERT/UPDATE/DELETE/MERGE/VALUES/TABLE, so
            # WITH ... SELECT ... FOR UPDATE and WITH ... INSERT ... ON CONFLICT DO UPDATE are not flagged.
            d0 = _depth0(s)
            m = re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|MERGE|VALUES|TABLE)\b", d0, re.I)
            if m and m.group(1).upper() == "UPDATE" and not re.search(r"\bWHERE\b", d0[m.end():], re.I):
                lint.destructive.append("UPDATE WITHOUT WHERE")
    # r2: destructive classes are matched on comment-stripped text that KEEPS literals and dollar bodies, so dynamic
    # SQL such as DO $$ BEGIN EXECUTE 'DROP TABLE x'; END $$ is still flagged (r1 stripped bodies first).
    nocomment = strip_comments(text)
    for rx, label in WRITE_MODE_RE:   # INC-3 (WS-8b): never in a migration body, in any position
        if rx.search(nocomment):
            lint.hard.append(f"changes transaction read/write mode: {label}")
    if RESET_ALL_ANY_RE.search(nocomment) or RESET_ALL_ANY_RE.search(strip_comments_nested(text)):   # INC-4 r2
        lint.hard.append("RESET ALL (anywhere; can remove a role/database-level default_transaction_read_only)")
    for rx, label in EXTRA_HARD:      # INC-3 (V11)
        if rx.search(nocomment):
            lint.hard.append(label)
    for rx in HARD_DESTRUCTIVE:
        for m in rx.finditer(nocomment):
            lint.destructive.append(" ".join(m.group(0).split()[:3]).upper())
    for rx in ADVISORY_DESTRUCTIVE:
        for m in rx.finditer(nocomment):
            lint.advisory.append(" ".join(m.group(0).split()[:3]).upper())
    for rx in ROLE_DDL_ADVISORY:      # INC-4 r2 (review R-3)
        for m in rx.finditer(nocomment):
            lint.advisory.append("ROLE/DATABASE " + ("RENAME" if m.group(0).upper().startswith("ALTER") else "CREATE")
                                 + " (per-role/per-database settings move or are absent)")
    if name.endswith("_rollback.sql"):
        lint.hard.append("rollback-named file")
    return lint


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- 2.4 / V-3 target classification, no connect
URL_KEYS = ("EXTERNAL_DB_URL", "INTERNAL_DB_URL", "DATABASE_URL", "VERA_DB_URL_RO", "VERA_DB_URL_RW", "VERA_DB_URL_WO")
PARAM_KEYS = ("DB_HOST", "DB_HOST_EXTERNAL_SUFFIX", "DB_PORT", "DB_NAME", "DB_ADMIN_ROLE", "LOCAL_PG_HOST",
              "LOCAL_PG_PORT", "LOCAL_PG_DB", "LOCAL_PG_ROLE", "LOCAL_PG_RO_ROLE")
SECRET_KEYS = ("DB_ADMIN_PASSWORD", "VERA_MIGRATE_PRODUCTION")   # presence only, value never kept
LOOPBACK = {"127.0.0.1", "::1", "localhost"}   # r3 (V QA-1): "" removed; socket dirs must be a single absolute path
HOST_RE = re.compile(r"[A-Za-z0-9._:/-]+")       # fullmatch: no ',' (libpq multi-host), no whitespace, not empty


@dataclass(frozen=True)
class Endpoint:
    host: str          # kept in memory for fingerprinting/connect only; NEVER printed
    port: int
    dbname: str
    role: str = ""

    def fingerprint(self) -> str:
        return sha256_hex(f"{self.host.lower()}|{self.port}|{self.dbname}".encode())[:16]

    def is_loopback(self) -> bool:
        return HOST_RE.fullmatch(self.host or "") is not None and (
            self.host in LOOPBACK or (self.host.startswith("/") and ".." not in self.host))


def endpoint(host: str, port: str, dbname: str, role: str = "") -> Endpoint:
    """Strict constructor (r3): refuses empty or multi-host values and non-numeric or multi-port values."""
    if HOST_RE.fullmatch(host or "") is None:
        raise Refusal("host value is empty, multi-host or malformed")
    if re.fullmatch(r"[0-9]{1,5}", str(port or "")) is None:
        raise Refusal("port value is empty, multi-port or malformed")
    if re.fullmatch(r"[A-Za-z0-9_]+", dbname or "") is None:
        raise Refusal("database name is empty or malformed")
    return Endpoint(host, int(port), dbname, role)


def parse_env_file(path: Path) -> tuple[dict[str, str], set[str]]:
    """Dedicated parser (V-3): returns (non-secret allowed keys, names of secret keys present).
    Never calls load_dotenv, never writes os.environ, never keeps a password (URL passwords are dropped)."""
    keep: dict[str, str] = {}
    present: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return keep, present
    for lineno, raw in enumerate(lines, 1):
        s = raw.strip()
        commented = s.startswith("#")
        if commented:   # K18W (QA F4): a commented-out secret is still a secret on disk; detection only, never kept
            s = s.lstrip("#").strip()
        if not s or "=" not in s:
            continue
        k, _, v = EXPORT_RE.sub("", s, count=1).partition("=")   # K18W: `export<TAB>KEY=` too
        k = k.strip()
        if commented:
            if k in SECRET_KEYS or key_is_secret(k) or value_has_password(v):
                present.add(f"{report_key(k, lineno)}:commented")
            continue
        if k in SECRET_KEYS or key_is_secret(k):
            present.add(report_key(k, lineno))
            continue
        if value_has_password(v):              # r4 (SEC S-2, V R2e): every value, whatever the key name
            present.add(f"{report_key(k, lineno)}:value-password")
            continue
        if k in URL_KEYS or k in PARAM_KEYS:
            v = v.strip().strip("'\"")
            if k in URL_KEYS:
                v = _url_triplet(k, v, path.name)
            keep[k] = v
    return keep, present


SECRET_SUFFIXES = ("_PASSWORD", "_PASS", "_PWD", "_PASSWD")
NOT_SECRET_NAMES = frozenset({"PWD", "OLDPWD"})
# K18W (QA F4): any '_'-separated token that is a password word, or a name containing PASSWORD/PASSWD anywhere
# (ADMINPASSWORD, DB_PASSWORD_ADMIN, PGPASSWORD). BYPASS / PASSPORT / COMPASS stay innocent.
SECRET_TOKENS = frozenset({"PASS", "PWD", "PW", "PASSWD", "PASSWORD", "DBPASS", "PGPASS", "PASSPHRASE", "SSLPASSWORD"})
EXPORT_RE = re.compile(r"^export\s+")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def report_key(k: str, lineno: int) -> str:
    """r5 (W-1): print a key only if it is a plain identifier; otherwise only its line number."""
    return k if IDENT_RE.fullmatch(k) else f"line {lineno}"
PW_VALUE_RE = re.compile(r"://[^/@\s]*:[^@\s]*@|(?:^|[\s?&;'\"])(?:ssl)?password\s*=", re.I)   # K18W: sslpassword


def key_is_secret(k: str) -> bool:
    """r4: case-insensitive; PASSWORD/PASS/PWD/PASSWD names and suffixes. K18W: token rule (SECRET_TOKENS)."""
    u = k.upper()
    if u in NOT_SECRET_NAMES:   # r5 (W-2): the shell's working-directory variables are not passwords
        return False
    if "PASSWORD" in u or "PASSWD" in u:
        return True
    return u.endswith(SECRET_SUFFIXES) or any(t in SECRET_TOKENS for t in re.split(r"[^A-Z0-9]+", u))


def decoded_forms(v: str, rounds: int = 3) -> list[str]:
    """K18W INC-3 (WS-8a F-4): the value plus its percent-decoded forms (libpq decodes URI userinfo and query
    keywords/values once; up to `rounds` passes are checked, which is stricter). Values are never kept or printed."""
    out = [v or ""]
    for _ in range(rounds):
        nxt = unquote(out[-1])
        if nxt == out[-1]:
            break
        out.append(nxt)
    return out


def value_has_password(v: str) -> bool:
    """r4: URL userinfo password, key/value DSN `password=` or URI query `?password=` (presence only).
    INC-3 (F-4): also checked after percent-decoding (`?%70assword=`, `%3A` in userinfo)."""
    return any(PW_VALUE_RE.search(f) for f in decoded_forms(v))


def _url_triplet(k: str, v: str, where: str) -> str:
    """Parse a DB URL into host|port|db; any malformed or multi-host value is a Refusal that names only the key."""
    try:
        u = urlsplit(v)
        if "," in (u.netloc or ""):
            raise ValueError("multi-host")
        port = u.port or 5432
        host = u.hostname or ""
    except ValueError:
        raise Refusal(f"{where}: {k} is malformed or multi-host (value suppressed)") from None
    return f"{host}|{port}|{(u.path or '/').lstrip('/')}"


def env_secret_hits(environ) -> list[str]:
    """r4: names of process-env keys whose name or value carries password material (values never kept)."""
    return sorted(report_key(k, 0) if IDENT_RE.fullmatch(k) else "non-identifier-name"
                  for k, v in environ.items()
                  if k != "VERA_MIGRATE_PRODUCTION" and (key_is_secret(k) or value_has_password(str(v))))


def env_url_passwords(environ) -> list[str]:
    """Names of process-env URL keys that carry a password (presence only)."""
    out = []
    for k in URL_KEYS:
        v = environ.get(k)
        try:
            has_pw = bool(v) and any(urlsplit(x.strip().strip("'\"")).password is not None for x in decoded_forms(v))
        except ValueError:   # K18W (QA F18): a malformed URL is a named refusal, never an internal error
            raise Refusal(f"process env {k} is malformed (value suppressed)") from None
        if has_pw:
            out.append(f"{k}:url-password")
    return out


def production_fingerprints(env_files: list[Path]) -> tuple[set[str], set[str]]:
    """Fingerprints (hash of host|port|db) of every production endpoint named in the given files."""
    fps: set[str] = set()
    secrets_seen: set[str] = set()
    for p in env_files:
        kv, present = parse_env_file(p)
        secrets_seen |= present
        for k in URL_KEYS:
            if k in kv:
                h, port, db = kv[k].split("|", 2)
                fps.add(Endpoint(h, int(port), db).fingerprint())
        if "DB_HOST" in kv and "DB_NAME" in kv:
            if re.fullmatch(r"[0-9]{1,5}", kv.get("DB_PORT", "5432") or "5432") is None:
                raise Refusal(f"{p.name}: DB_PORT is malformed or multi-port (value suppressed)")
            port = int(kv.get("DB_PORT", "5432") or 5432)
            for h in {kv["DB_HOST"], kv["DB_HOST"] + kv.get("DB_HOST_EXTERNAL_SUFFIX", "")}:
                fps.add(Endpoint(h, port, kv["DB_NAME"]).fingerprint())
    return fps, secrets_seen


def strip_inherited(environ, patterns) -> list[str]:
    """PLAN-ENV C3: delete inherited names matching fnmatch `patterns` (case-insensitive) from a mutable mapping.
    Returns the removed NAMES (sorted); values are never read into the result."""
    import fnmatch  # noqa: PLC0415
    hit = sorted(k for k in list(environ) if any(fnmatch.fnmatchcase(str(k).upper(), p.upper()) for p in patterns))
    for k in hit:
        del environ[k]
    return hit


def classify(ep: Endpoint | None, prod_fps: set[str]) -> str:
    """'production' | 'local' | 'unknown'. Fails closed: non-loopback counts as production (2.4).
    PLAN-ENV C2 (owner decision 2026-10-19): a 'local' result is a ROUTING HINT ONLY. It never authorises a connection;
    any connecting local op additionally needs DISPOSABLE_VERIFIED, which is unreachable (VERIFY_DISABLED)."""
    if ep is None:
        return "unknown"
    if ep.fingerprint() in prod_fps or not ep.is_loopback():
        return "production"
    if LOCAL_DB_RE.fullmatch(ep.dbname):
        return "local"
    return "unknown"


def marker_matches(expected: str, got: str | None) -> bool:
    return bool(got) and hmac.compare_digest(expected.encode(), got.encode())
