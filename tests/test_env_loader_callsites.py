"""DARINT-ENVFIX: every .env read goes through vera.env_precedence.load_env (PRB-DARINT-002).
Static AST scan of vera/ and scripts/, plus load_env behaviour on a synthetic repo root (never the real .env)."""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from vera import env_precedence as ep

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {ROOT / "vera" / "env_precedence.py"}
BANNED = {"load_dotenv", "dotenv_values"}


def _py_files():
    for d in ("vera", "scripts"):
        for p in sorted((ROOT / d).rglob("*.py")):
            if p not in ALLOWED:
                yield p


def _bare_calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name in BANNED:
                hits.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return hits


def test_no_bare_dotenv_calls_outside_env_precedence():
    hits = [h for p in _py_files() for h in _bare_calls(p)]
    assert not hits, "bare load_dotenv()/dotenv_values() calls (use vera.env_precedence.load_env): " + ", ".join(hits)


# --- PLAN-ENV s1.4 static bans (vera/, scripts/, db/, main.py, streamlit_app.py; comments and docstrings exempt) ----
SCAN_DIRS = ("vera", "scripts", "db")
SCAN_FILES = ("main.py", "streamlit_app.py")
ENV_FILES_PY = ROOT / "vera" / "env_files.py"
# Guard files whose literals ARE the denylist; they name the forbidden strings on purpose.
LITERAL_GUARD_FILES = {ROOT / "vera" / "env_precedence.py", ROOT / "scripts" / "vera_test_runner.py",
                       ROOT / "scripts" / "check_config.py"}
# Pre-existing repo-root anchors inside vera/ subpackages (R/vera, never R/config); not env loading.
PARENT_CLIMB_PINNED = {ROOT / "vera" / "pricing" / "config.py", ROOT / "vera" / "trace_eval" / "guidance.py",
                       ROOT / "vera" / "m6" / "test_m6_schema_sync.py", ROOT / "vera" / "m6" / "fingerprint.py"}
FORBIDDEN_LITERALS = ("AI-Internship", "week-1v2", ".env.db-accounts")


def _scan_files():
    for d in SCAN_DIRS:
        yield from sorted((ROOT / d).rglob("*.py"))
    for f in SCAN_FILES:
        yield ROOT / f


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _docstring_nodes(tree) -> set:
    ids = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def _rel(p, node):
    return f"{p.relative_to(ROOT)}:{node.lineno}"


def test_no_env_precedence_load_env_outside_env_files():
    hits = []
    for p in _scan_files():
        if p in (ENV_FILES_PY, ROOT / "vera" / "env_precedence.py"):
            continue
        for n in ast.walk(_tree(p)):
            if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[-1] == "env_precedence" \
                    and any(a.name == "load_env" for a in n.names):
                hits.append(_rel(p, n))
            elif isinstance(n, ast.Attribute) and n.attr == "load_env":
                hits.append(_rel(p, n))
    assert not hits, "use vera.env_files.load_vera_env, not env_precedence.load_env: " + ", ".join(hits)


def test_no_legacy_path_literals():
    hits = []
    for p in _scan_files():
        if p in LITERAL_GUARD_FILES:
            continue
        tree = _tree(p)
        doc = _docstring_nodes(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc \
                    and any(f in n.value for f in FORBIDDEN_LITERALS):
                hits.append(_rel(p, n))
    assert not hits, "legacy checkout/credential-file literals: " + ", ".join(hits)


def test_no_path_join_to_dot_env():
    hits = []
    for p in _scan_files():
        for n in ast.walk(_tree(p)):
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) and isinstance(n.right, ast.Constant) \
                    and n.right.value == ".env":
                hits.append(_rel(p, n))
    assert not hits, 'Path(...) / ".env" joins: ' + ", ".join(hits)


def _is_triple_parent(n) -> bool:
    depth, cur = 0, n
    while isinstance(cur, ast.Attribute) and cur.attr == "parent":
        depth, cur = depth + 1, cur.value
    return depth >= 3


def test_parent_climbs_only_in_env_files():
    hits = []
    for p in _scan_files():
        if p == ENV_FILES_PY or p in PARENT_CLIMB_PINNED:
            continue
        for n in ast.walk(_tree(p)):
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == "parents" \
                    and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, int) and n.slice.value >= 2:
                hits.append(_rel(p, n))
            elif _is_triple_parent(n):
                hits.append(_rel(p, n))
    assert not hits, "parents[2] / .parent.parent.parent outside vera/env_files.py: " + ", ".join(sorted(set(hits)))


def test_live_catalog_check_not_exempt_and_reads_no_env_file():
    assert ROOT / "scripts" / "live_catalog_check.py" not in ALLOWED
    assert "_read_env_file" not in (ROOT / "scripts" / "live_catalog_check.py").read_text(encoding="utf-8")


# --- behaviour on a synthetic root ---------------------------------------------------------------------------
@pytest.fixture
def repo(tmp_path):
    old = os.umask(0o077)
    root = tmp_path / "parent" / "repo"
    root.mkdir(parents=True)
    (root / ".env-vera").write_text("VERA_OWNER_KEY=x\nOTHER_PLAIN=y\n", encoding="utf-8")
    yield root
    os.umask(old)


def _load(environ, root):
    return ep.load_env(environ, cfg=ep.load_config(), root=root, modules={}, log=False)


def test_marker_preserving_environ_does_not_yield_owner_key(repo):
    env = {"CLAUDECODE": "1", "PATH": "/usr/bin:/bin"}
    audit = _load(env, repo)
    assert audit.context == "agent"
    assert "VERA_OWNER_KEY" not in env


def test_marker_less_default_context_shows_residual(repo):
    """Gap A residual (documented, not fixed in code): env -i drops the agent marker, so the default context
    local_dev loads .env-vera, including VERA_OWNER_KEY."""
    env = {"PATH": "/usr/bin:/bin"}
    audit = _load(env, repo)
    assert audit.context == "local_dev"
    assert env.get("VERA_OWNER_KEY") == "x"
