"""Pragmatic SQL column extractor + read-only catalog loader for tests/test_sql_contract.py.

EXTRACTION (regex over SQL string literals found via `ast`; no real SQL parser):
  * Every string constant (incl. docstrings and the literal parts of f-strings) in vera/**/*.py that
    looks like a statement against the VERA-DAR schema (contains `vera_vjay.` or `{S}.` plus SELECT/INSERT/
    UPDATE/ALTER) is treated as one statement. In sql/*.sql only ALTER TABLE ... ADD COLUMN statements are checked. Docstring prose is cut to
    keyword..';'/blank-line regions.
  * Tables: names after FROM / JOIN / INTO / UPDATE / ALTER TABLE, with optional alias.
  * Columns: alias.col refs; INSERT column lists; UPDATE ... SET targets (incl. ON CONFLICT DO UPDATE);
    ALTER TABLE ... ADD COLUMN names; and, ONLY for single-table statements, unqualified identifiers
    left after stripping keywords, params (:x), casts, function names, `AS` aliases and string literals.
  * sa.Table(...) declarations (SQLAlchemy Core in vera/m3/store.py) via ast: every Column name.
LIMITS: multi-table statements only get alias-qualified checks (unqualified columns there are NOT
checked); CTEs/subqueries with their own aliases, dynamically built SQL, f-string interpolated
identifiers and SQL outside string literals are not seen; `{S}` is assumed to be the vera_vjay schema.
Only schema vera_vjay is ever queried (never 'internship').

CONTRACT-FIX-1105 (release gate a2 at 9a8ea14):
  * Function calls are NOT tables: `FROM/JOIN schema.fn(` is ignored (scripts/live_catalog_check.py, pinned, reads only
    information_schema.columns, so it cannot check a function; a function contract would always FAIL there).
  * Docstrings (module/class/function) are not parsed as SQL; only other string constants are.
  * Pending-migration exemption, deny-by-default: a (table, column) pair is left out of the REQUIRED contract only if a
    draft named in PENDING_DRAFT_ALLOWLIST exists as a regular file db/migrations/pending/<name>, its text has
    `CREATE TABLE|VIEW vera_vjay.<table>` (outside comments, strings and $$ bodies) whose parsed column list contains
    <column>, and no applied migration (db/migrations/*.sql) creates that table. Anything else stays required. Delete the
    draft or its allowlist entry (do so when the migration is applied) and the pair is required again.
  * collect() == collect_required() (extracted minus exempt; the pinned live script calls collect()); collect_all() is
    everything extracted; pending_exemptions() / exemption_report() name every exempt pair. The pinned live script prints
    only its own lines, so the exemption is reported by tests/test_sql_contract_pending_1105.py (a warning on every suite
    run) and by exemption_report(), not in the catalog evidence file.
  * Functions are checked OFFLINE instead (unknown_function_calls() must be empty): every called vera_vjay function must be
    created by an applied db/migrations/*.sql or by a draft in PENDING_FUNCTION_DRAFT_ALLOWLIST.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "vera_vjay"
RO_ACCOUNT = "vera_eval_ro"

KEYWORDS = set("""select from where and or not null is in as on join left right inner outer full cross insert into values
update set delete returning order by group having limit offset desc asc distinct case when then else end exists
between like ilike true false conflict do nothing excluded with union all using cast jsonb text int integer bigint
numeric boolean timestamptz timestamp date interval over partition filter nulls first last any some only if
alter table add column default now coalesce max min count sum lower upper""".split())
_TBL = re.compile(r"\b(?:from|join|into|update|table)\s+(?:if\s+exists\s+)?(?:(?:\{S\}|vera_vjay)\.)(\w+)"
                  r"(?:\s+(?:as\s+)?(?!(?:join|where|set|on|values|left|right|inner|order|group|limit|add|"
                  r"select|using|returning|cross|full)\b)(\w+))?", re.I)


# CONTRACT-FIX-1105: reviewed allowlist of pending DRAFT FILE NAMES (not object names). Each entry lets the objects that
# draft verifiably creates leave the required contract until it is applied. Remove the entry when the draft is applied.
PENDING_DRAFT_ALLOWLIST = (
    "016-DRAFT-lists-to-db.sql.draft",   # lstreg_active_lists_v, lstreg_active_items_v (vera/list_tables.py, topic_tagging.py)
    "017-DRAFT-topic-tags.sql.draft",    # toptag_document_topics (vera/corpus_description.py)
)
PENDING_DIR = "db/migrations/pending"
_DRAFT_NAME = re.compile(r"^\d{3}-DRAFT-[A-Za-z0-9-]+\.sql\.draft$")
_CREATE = re.compile(r"\bcreate\s+(?:or\s+replace\s+)?(?:(?:temp|temporary|unlogged|materialized)\s+)?"
                     r"(table|view|function)\s+(?:if\s+not\s+exists\s+)?vera_vjay\.(\w+)", re.I)
_LITERAL = re.compile(r"'(?:[^']|'')*'")
_DOLLAR = re.compile(r"\$(?:[A-Za-z_]\w*)?\$")
_NOT_COLUMN = {"constraint", "primary", "unique", "check", "foreign", "exclude", "like"}


def _docstring_ids(tree) -> set[int]:
    ids = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def _py_strings(path: Path):
    tree = ast.parse(path.read_text())
    docs = _docstring_ids(tree)
    for n in ast.walk(tree):
        if id(n) in docs:
            continue
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            yield n.value, n.lineno
        elif isinstance(n, ast.JoinedStr):
            yield "".join(v.value for v in n.values if isinstance(v, ast.Constant)), n.lineno


def _looks_sql(s: str) -> bool:
    return bool(re.search(r"(vera_vjay|\{S\})\.\w+", s)) and bool(
        re.search(r"\b(select\b.*\bfrom|insert\s+into|update\s+\S+\s+set|alter\s+table)\b", s, re.I | re.S))


def extract_columns(sql: str) -> set[tuple[str, str]]:
    """Return {(table, column)} referenced by one statement text (see module docstring for limits)."""
    sql = sql.replace("{S}", SCHEMA)
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)           # string literals
    tables: dict[str, str] = {}                          # alias/name -> table
    fn_sources = 0
    for m in _TBL.finditer(sql):
        if re.match(r"\s*\(", sql[m.end(1):]) and not re.match(r"(?:into|table)\b", m.group(0), re.I):
            fn_sources += 1                              # FROM/JOIN schema.fn(...) is a function call, not a table
            continue
        t, alias = m.group(1).lower(), (m.group(2) or "").lower()
        tables[t] = t
        if alias and alias not in KEYWORDS:
            tables[alias] = t
    real = set(tables.values())
    out: set[tuple[str, str]] = set()
    for a, c in re.findall(r"\b(\w+)\.(\w+)\b", sql):
        if a.lower() in tables and a.lower() != SCHEMA:
            out.add((tables[a.lower()], c.lower()))
    for t, cols in re.findall(r"insert\s+into\s+vera_vjay\.(\w+)\s*\(([^)]*)\)", sql, re.I):
        out |= {(t.lower(), c.strip().lower()) for c in cols.split(",") if c.strip()}
    for t, col in re.findall(r"alter\s+table\s+vera_vjay\.(\w+)\s+add\s+column\s+(?:if\s+not\s+exists\s+)?(\w+)", sql, re.I):
        out.add((t.lower(), col.lower()))
    if len(real) == 1 and not fn_sources:               # a function source makes it multi-source: no unqualified check
        (t,) = real
        for seg in re.findall(r"\bset\b(.*?)(?=\bwhere\b|\breturning\b|;|$)", sql, re.I | re.S):
            out |= {(t, c.lower()) for c in re.findall(r"(?:^|,)\s*(\w+)\s*=", seg)}
        body = re.sub(r"\bcast\s*\(.*?\bas\s+\w+\s*\)", " ", sql, flags=re.I | re.S)
        body = re.sub(r"\balter\s+table\b.*?\badd\s+column\b[^;]*;?", " ", body, flags=re.I | re.S)
        body = re.sub(r"\b\w+\.\w+\b", " ", body)
        body = re.sub(r"::\w+|:\w+|\bas\s+\w+", " ", body, flags=re.I)
        body = re.sub(r"\b\w+(?=\s*\()", " ", body)      # function names / table-name before col list
        for tok in re.findall(r"\b[a-z_]\w*\b", body, re.I):
            tl = tok.lower()
            if tl not in KEYWORDS and tl not in tables and tl != SCHEMA:
                out.add((t, tl))
    out = {(t, c) for t, c in out if c not in KEYWORDS and c != "*"}
    return out


def extract_statements(text: str) -> set[tuple[str, str]]:
    """Cut prose out of docstrings: a statement runs from its keyword to ';' or a blank line."""
    out: set[tuple[str, str]] = set()
    for m in re.finditer(r"\b(select|insert|update|alter)\b.*?(?:;|\n\s*\n|$)", text, re.I | re.S):
        if _looks_sql(m.group(0)):
            out |= extract_columns(m.group(0))
    return out


def _sa_tables(path: Path):
    """Yield (table, column, lineno) from sa.Table('name', metadata, sa.Column('c'..), ..., schema=SCHEMA)."""
    src = path.read_text()
    if "Table(" not in src:
        return
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) == "Table" and n.args:
            if isinstance(n.args[0], ast.Constant):
                schema_ok = any(k.arg == "schema" for k in n.keywords)
                if not schema_ok:
                    continue
                for a in n.args[2:]:
                    if isinstance(a, ast.Call) and getattr(a.func, "attr", "") == "Column" and a.args \
                            and isinstance(a.args[0], ast.Constant):
                        yield n.args[0].value, a.args[0].value, a.lineno


def collect_all() -> dict[tuple[str, str], list[str]]:
    """{(table, column): [where-found, ...]} across vera/**/*.py, sql/*.sql: EVERYTHING extracted, no exemption."""
    found: dict[tuple[str, str], list[str]] = {}
    for py in sorted((ROOT / "vera").rglob("*.py")):
        rel = py.relative_to(ROOT)
        for s, line in _py_strings(py):
            if _looks_sql(s):
                for tc in extract_statements(s):
                    found.setdefault(tc, []).append(f"{rel}:{line}")
        for t, c, line in _sa_tables(py):
            found.setdefault((t.lower(), c.lower()), []).append(f"{rel}:{line} (sa.Table)")
    for f in sorted((ROOT / "sql").glob("*.sql")):
        # only ALTER ... ADD COLUMN statements are checked in .sql files (DO-block PL/pgSQL is not SQL-parsed)
        for stmt in re.findall(r"alter\s+table\s+[^;]*?add\s+column[^;]*;", f.read_text(), re.I | re.S):
            for tc in extract_columns(stmt):
                    found.setdefault(tc, []).append(str(f.relative_to(ROOT)))
    return found


def _strip_sql(text: str) -> str:
    """Drop -- comments, nested /* */ comments, '...' literals and $tag$ bodies in ONE left-to-right scan, so none of them
    can pair up wrongly or hide DDL. Anything unterminated is dropped to the end (fail-safe: it then proves nothing)."""
    out, i, n = [], 0, len(text)
    while i < n:
        two = text[i:i + 2]
        if two == "--":
            j = text.find("\n", i)
            i = n if j < 0 else j
            out.append(" ")
        elif two == "/*":
            depth, i = 1, i + 2
            while i < n and depth:
                if text.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            out.append(" ")
        elif text[i] == "'":
            m = _LITERAL.match(text, i)
            i = m.end() if m else n
            out.append("''")
        elif text[i] == "$" and not (i and (text[i - 1].isalnum() or text[i - 1] == "_")) and _DOLLAR.match(text, i):
            tag = _DOLLAR.match(text, i).group(0)
            j = text.find(tag, i + len(tag))
            i = n if j < 0 else j + len(tag)
            out.append(" ")
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _split_top(s: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _paren_body(s: str, start: int) -> str | None:
    """Text inside the parenthesis opening at s[start] (which must be '('), or None if unbalanced."""
    depth = 0
    for i in range(start, len(s)):
        depth += {"(": 1, ")": -1}.get(s[i], 0)
        if depth == 0:
            return s[start + 1:i]
    return None


def _view_columns(rest: str) -> frozenset:
    m = re.match(r"\s*\(([^()]*)\)", rest)                 # explicit column list
    if m:
        return frozenset(c.strip().lower() for c in m.group(1).split(",") if c.strip())
    m = re.search(r"\bas\s+select\b", rest, re.I)
    if not m:
        return frozenset()
    body, depth, end = rest[m.end():], 0, None
    for fm in re.finditer(r"[()]|\bfrom\b|;", body, re.I):
        tok = fm.group(0)
        depth += {"(": 1, ")": -1}.get(tok, 0)
        if depth == 0 and tok.lower() in ("from", ";"):
            end = fm.start(); break
    cols = set()
    for item in _split_top(body[:end] if end is not None else ""):
        a = re.search(r"\bas\s+(\w+)\s*$", item, re.I) or re.search(r"(?:^|\.)(\w+)\s*$", item)
        if not a:
            return frozenset()                           # `*` or an expression without a name: prove nothing
        cols.add(a.group(1).lower())
    return frozenset(cols)


def draft_objects(text: str) -> dict[str, tuple[str, frozenset]]:
    """{name: (kind, columns)} for every CREATE TABLE|VIEW|FUNCTION vera_vjay.name in one migration text."""
    text = _strip_sql(text)
    out: dict[str, tuple[str, frozenset]] = {}
    for m in _CREATE.finditer(text):
        kind, name, rest = m.group(1).lower(), m.group(2).lower(), text[m.end():]
        cols: frozenset = frozenset()
        if kind == "table":
            p = re.match(r"\s*\(", rest)
            body = _paren_body(rest, p.end() - 1) if p else None
            if body is not None:
                cols = frozenset(e.split()[0].lower() for e in _split_top(body)
                                 if e.split()[0].lower() not in _NOT_COLUMN and re.match(r"\w+$", e.split()[0]))
        elif kind == "view":
            cols = _view_columns(rest)
        out[name] = (kind, cols)
    return out


def _draft_file(root: Path, name: str) -> Path | None:
    """db/migrations/pending/<name> if the name has the draft shape, no component from root down to the file is a
    symlink, and it is a regular file; else None."""
    if not _DRAFT_NAME.match(name):
        return None
    cur = root
    for part in (*Path(PENDING_DIR).parts, name):
        cur = cur / part
        if cur.is_symlink():
            return None
    return cur if cur.is_file() else None


def pending_objects(root: Path | None = None) -> dict[str, tuple[str, frozenset]]:
    """{table_or_view: (draft filename, columns)} proven by ALLOWLISTED drafts present as regular files; tables or views
    an applied migration (db/migrations/*.sql) also creates are never included. Functions are never included."""
    root = ROOT if root is None else root
    applied: set[str] = set()
    for f in sorted((root / "db" / "migrations").glob("*.sql")):
        applied |= set(draft_objects(f.read_text()))
    out: dict[str, tuple[str, frozenset]] = {}
    for name in PENDING_DRAFT_ALLOWLIST:
        f = _draft_file(root, name)
        if f is None:
            continue
        for obj, (kind, cols) in draft_objects(f.read_text()).items():
            if kind in ("table", "view") and cols and obj not in applied and obj not in out:
                out[obj] = (name, cols)
    return out


def pending_exempt(table: str, root: Path | None = None) -> bool:
    """True only if `table` is a table/view created by an allowlisted pending draft (see pending_objects). Pair-level
    exemption additionally needs the column to be in that draft's column list."""
    return table.lower() in pending_objects(root)


def pending_exemptions(found: dict | None = None, root: Path | None = None) -> dict[tuple[str, str], str]:
    """{(table, column): draft filename} for every extracted pair the drafts prove; deny-by-default for the rest."""
    found = collect_all() if found is None else found
    objs = pending_objects(root)
    return {(t, c): objs[t][0] for (t, c) in sorted(found) if t in objs and c in objs[t][1]}


def collect_required() -> dict[tuple[str, str], list[str]]:
    """The effective contract that must exist live now: collect_all() minus pending_exemptions(). Same structure."""
    found = collect_all()
    exempt = pending_exemptions(found)
    return {k: v for k, v in found.items() if k not in exempt}


def collect() -> dict[tuple[str, str], list[str]]:
    """What scripts/live_catalog_check.py (pinned) checks: the required contract (see collect_required)."""
    return collect_required()


def exemption_report(found: dict | None = None) -> str:
    ex = pending_exemptions(found)
    drafts = sorted(set(ex.values()))
    lines = [f"SQL contract pending-migration EXEMPTION: {len(ex)} (table, column) pairs exempt via {len(drafts)} "
             f"allowlisted draft(s): {', '.join(drafts) or 'none'}"]
    lines += [f"exempt: {t}.{c} ({d})" for (t, c), d in ex.items()]
    return "\n".join(lines)


# CONTRACT-FIX-1105 F2: function calls are not in the column contract, so this OFFLINE check covers them. Reviewed allowlist
# of pending DRAFT FILE NAMES whose CREATE FUNCTION may satisfy a call from vera/ (separate from the table/view list).
PENDING_FUNCTION_DRAFT_ALLOWLIST = (
    "018-DRAFT-trace-purge-erasure.sql.draft",   # purge_expired_trace (vera/retention_purge.py, flag off)
)
# word right before `schema.name(` that makes it a table followed by a column list, not a call
_NOT_CALL_PREV = {"into", "table", "references", "on", "update", "exists", "view", "index", "only", "trigger"}


def collect_function_calls(root: Path | None = None) -> dict[str, list[str]]:
    """{function: [where-found]} for every `vera_vjay.name(` / `{S}.name(` in a non-docstring string in vera/**/*.py,
    except a table followed by a column list (INSERT INTO t (..), REFERENCES t (..), ON t (..), ...)."""
    root = ROOT if root is None else root
    found: dict[str, list[str]] = {}
    for py in sorted((root / "vera").rglob("*.py")):
        for s, line in _py_strings(py):
            for m in re.finditer(r"(?<![\w.])(?:\{S\}|vera_vjay)\.(\w+)\s*\(", s):
                prev = re.search(r"(\w+)\s*$", s[:m.start()])
                if prev and prev.group(1).lower() in _NOT_CALL_PREV:
                    continue
                found.setdefault(m.group(1).lower(), []).append(f"{py.relative_to(root)}:{line}")
    return found


def known_functions(root: Path | None = None) -> dict[str, str]:
    """{function: origin} created by an applied db/migrations/*.sql (not pending/) or an allowlisted pending function draft."""
    root = ROOT if root is None else root
    out: dict[str, str] = {}
    for f in sorted((root / "db" / "migrations").glob("*.sql")):
        for obj, (kind, _) in draft_objects(f.read_text()).items():
            if kind == "function":
                out.setdefault(obj, f.name)
    for name in PENDING_FUNCTION_DRAFT_ALLOWLIST:
        f = _draft_file(root, name)
        if f is None:
            continue
        for obj, (kind, _) in draft_objects(f.read_text()).items():
            if kind == "function":
                out.setdefault(obj, f"pending/{name}")
    return out


def unknown_function_calls(root: Path | None = None) -> dict[str, list[str]]:
    """Called functions no applied migration or allowlisted function draft creates (must be empty)."""
    known = known_functions(root)
    return {fn: w for fn, w in collect_function_calls(root).items() if fn not in known}
