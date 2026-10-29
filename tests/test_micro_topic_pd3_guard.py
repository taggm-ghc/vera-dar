"""PD-3 replacement guards (TOPICREG-FIX-1028), alongside tests/test_micro_topic.py (which is unchanged). The read-time key stays
read-only even when the DB-backed registry is enabled. Offline: fake engine only."""
import ast
import re
from pathlib import Path

import pytest

from vera import micro_topic as mt, topic_registry as tr

VERA = Path(__file__).resolve().parent.parent / "vera"
ON = {"registry_table": {"read_from_db": True}}
ROW = {"topic_key": "a-one", "version": 1, "label": "Label a-one", "field": "f1", "anchor_text": "alpha beta gamma delta epsilon zeta",
       "parent_openalex_topic": "none", "parent_topic_key": None, "origin": "seed", "state": "active"}


class _Res:
    def __init__(self, scalar=None, rows=None):
        self._s, self._r = scalar, rows or []

    def scalar(self):
        return self._s

    def mappings(self):
        return self

    def all(self):
        return self._r

    def first(self):
        return self._r[0] if self._r else None


class _Engine:
    def __init__(self):
        self.log = []

    def connect(self):
        eng = self

        class C:
            def __enter__(s):
                return s

            def __exit__(s, *a):
                return False

            def execute(s, stmt, params=None):
                q = str(stmt)
                eng.log.append(q)
                if "pg_get_userbyid" in q:
                    return _Res(rows=[{"owner": "adm", "app_memberships": 0, "current_reaches_owner": False, "session_reaches_owner": False}])
                if "to_regclass" in q and q.startswith("SELECT to_regclass"):
                    return _Res(scalar="vera_vjay.topreg_topics")
                return _Res(rows=[dict(ROW)])
        return C()


def test_a_enabled_db_loader_issues_only_read_only_selects(monkeypatch):
    mt.reset_for_tests()
    eng = _Engine()
    import vera.db as vdb
    monkeypatch.setattr(vdb, "get_role_engine", lambda role, **kw: eng)
    reg = mt.load_registry(ON)
    assert [e["id"] for e in reg.entries] == ["a-one"] and eng.log
    for stmt in eng.log:
        assert re.match(r"^\s*(SET TRANSACTION READ ONLY|SELECT)\b", stmt), stmt
        assert not re.search(r"\b(INSERT|UPDATE|DELETE|MERGE|COPY|TRUNCATE)\b", stmt), stmt
    assert eng.log[0] == "SET TRANSACTION READ ONLY"
    mt.reset_for_tests()


def _tree(name):
    return ast.parse((VERA / name).read_text(encoding="utf-8"))


def test_b_micro_topic_only_uses_load_active_registry_data_from_topic_registry():
    tree = _tree("micro_topic.py")
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for bad in ("register_proposed_topic", "TOPIC_SQL_INSERT", "TOPIC_SQL_ACTIVATE", "TOPIC_SQL_SUSPEND", "TOPIC_SQL_PARENT_ACTIVE"):
        assert bad not in names
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "topic_registry"}
    assert used == {"load_active_registry_data"}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module and "topic_registry" in n.module]
    src = (VERA / "micro_topic.py").read_text(encoding="utf-8")
    assert not re.search(r"TOPIC_SQL_|register_proposed", src)


def test_c_read_path_functions_reference_only_the_three_read_statements():
    tree = _tree("topic_registry.py")
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    allowed = {"TOPIC_SQL_EXISTS", "TOPIC_SQL_OWNER", "TOPIC_SQL_ROWS"}
    for name in ("load_active_registry_data", "read_topic_rows", "registry_data_from_rows"):
        refs = {n.id for n in ast.walk(fns[name]) if isinstance(n, ast.Name) and n.id.startswith("TOPIC_SQL_")}
        assert refs <= allowed, (name, refs)
    used = set()
    for name in ("load_active_registry_data", "read_topic_rows", "registry_data_from_rows"):
        used |= {n.id for n in ast.walk(fns[name]) if isinstance(n, ast.Name) and n.id.startswith("TOPIC_SQL_")}
    assert used == allowed
    for name in ("load_active_registry_data", "read_topic_rows", "registry_data_from_rows"):
        assert not [n for n in ast.walk(fns[name]) if isinstance(n, ast.Name) and n.id == "register_proposed_topic"]


def test_d_no_vera_code_writes_acquisition_micro_topic_outside_acquisition_time():
    # No module in vera/ writes a 'micro_topic' key into provenance today. The allowlist is the acquisition-time writer(s), if
    # one is added later it must be named here deliberately (and reviewed), not inferred.
    ALLOWED_WRITERS: set = set()
    hits = []
    for p in VERA.rglob("*.py"):
        if p.name in ALLOWED_WRITERS:
            continue
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Dict) and any(isinstance(k, ast.Constant) and k.value == "micro_topic" for k in n.keys):
                hits.append((p.name, n.lineno, "dict literal"))
            if isinstance(n, ast.Subscript) and isinstance(n.ctx, (ast.Store, ast.Del)) and isinstance(n.slice, ast.Constant) \
                    and n.slice.value == "micro_topic":
                hits.append((p.name, n.lineno, "subscript store"))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("setdefault", "update", "pop") \
                    and any(isinstance(a, ast.Constant) and a.value == "micro_topic" for a in n.args):
                hits.append((p.name, n.lineno, n.func.attr))
            if isinstance(n, ast.keyword) and n.arg == "micro_topic":
                hits.append((p.name, getattr(n, "lineno", 0), "keyword"))
    assert hits == [], hits
