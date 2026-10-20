"""Destructive-operation guard (PLAN-ENV v2.1 §3 E5; guard file).

- `@require_disposable` and `destructive_guard()` are for explicit test fixtures that would run destructive SQL. They
  refuse unless the connection's Verdict is DISPOSABLE_VERIFIED, which is unreachable in this build (owner decision
  2026-10-19), so today they always refuse with DestructiveRefused('not_disposable').
- The static scan finds destructive SQL text in string literals (docstrings included) under vera/, scripts/, db/,
  main.py and streamlit_app.py. Hits are allowed only in the pinned ALLOWLIST, each with its reason.
"""
from __future__ import annotations

import ast
import functools
import importlib.util
import re
import sys
from contextlib import contextmanager
from pathlib import Path


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
_sg = _sibling("sql_guard")
_cg = _sibling("connect_guard")
DestructiveRefused = _sg.DestructiveRefused


def _is_disposable(conn) -> bool:
    v = _cg.verdict_of(conn)
    return v is not None and getattr(v, "cls", None) == _ec.DISPOSABLE_VERIFIED


def require_disposable(fn):
    """Decorator: the first positional argument is the connection to be used destructively."""

    @functools.wraps(fn)
    def wrapper(conn, *a, **kw):
        if not _is_disposable(conn):
            raise DestructiveRefused("not_disposable")
        return fn(conn, *a, **kw)

    return wrapper


@contextmanager
def destructive_guard(conn):
    if not _is_disposable(conn):
        raise DestructiveRefused("not_disposable")
    yield conn


ALLOWLIST = {
    "db/migration_guard.py": "lint patterns for migration files (the lint itself)",
    "db/apply_migrations.py": "migration runner behind its own gate (D-ENV-5, ships disabled)",
    "scripts/live_catalog_check.py": "a privilege name inside a read-only SELECT",
    "vera/dar_store.py": "a docstring describing what the store never does",
    "vera/m6/test_m6_schema_sync.py": "keyword strings matched against migration text (never executed as SQL)",
}
SCAN_DIRS = ("vera", "scripts", "db")
SCAN_FILES = ("main.py", "streamlit_app.py")
DESTRUCTIVE_RE = re.compile(
    r"(?i)\b(DELETE\s+FROM|TRUNCATE\s+(TABLE\s+)?[\w\"]|DROP\s+(TABLE|SCHEMA|DATABASE|ROLE|INDEX|VIEW|FUNCTION|OWNED|"
    r"OWNER|EXTENSION|TRIGGER)\b|ALTER\s+(TABLE|ROLE|SCHEMA|DATABASE|DEFAULT\s+PRIVILEGES)\b|"
    r"GRANT\s+[\w, ]+\s+ON\b|REVOKE\s+[\w, ]+\s+(ON|FROM)\b|VACUUM\s+FULL)")


def scan_source(text: str) -> list:
    """Line numbers of string literals (docstrings included) that contain destructive SQL text."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [0]
    hits = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and DESTRUCTIVE_RE.search(node.value):
            hits.add(node.lineno)
    return sorted(hits)


def _files(repo_root: Path):
    root = Path(repo_root)
    for d in SCAN_DIRS:
        base = root / d
        if base.is_dir():
            for p in sorted(base.rglob("*.py")):
                if "__pycache__" not in p.parts:
                    yield p
    for f in SCAN_FILES:
        if (root / f).is_file():
            yield root / f


def scan_tree(repo_root: Path) -> dict:
    out = {}
    for p in _files(repo_root):
        hits = scan_source(p.read_text(encoding="utf-8", errors="replace"))
        if hits:
            out[p.relative_to(repo_root).as_posix()] = hits
    return out


def unallowed_findings(repo_root: Path) -> dict:
    return {k: v for k, v in scan_tree(repo_root).items() if k not in ALLOWLIST}
