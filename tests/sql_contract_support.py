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


def _py_strings(path: Path):
    tree = ast.parse(path.read_text())
    for n in ast.walk(tree):
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
    for m in _TBL.finditer(sql):
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
    if len(real) == 1:
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


def collect() -> dict[tuple[str, str], list[str]]:
    """{(table, column): [where-found, ...]} across vera/**/*.py, sql/*.sql."""
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
