"""Inert marker/HMAC code (PLAN-ENV v2.1 §2.4, V3; owner decision 2026-10-19).

verify_disposable refuses unconditionally; the HMAC helpers are tested DIRECTLY with fakes and are unreachable from
classification. The owner creation tool is tested with a fake cursor only; its main() refuses."""
import ast
import importlib.util
import secrets
from pathlib import Path

import pytest

from safety import endpoint_class as ec

ROOT = Path(__file__).resolve().parents[1]


def _load_marker_tool():
    spec = importlib.util.spec_from_file_location("_t_create_disposable_marker",
                                                  ROOT / "scripts" / "create_disposable_marker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeCursor:
    def __init__(self, ro="on", ident=("vera_test_x", "tester", 41001, 5433), tag=None, fail_on=None, log=None):
        self.ro, self.ident, self.tag, self.fail_on = ro, ident, tag, fail_on
        self.log = [] if log is None else log
        self._next = None

    def execute(self, q, params=None):
        self.log.append(q)
        if self.fail_on and self.fail_on in q:
            raise TimeoutError("canceling statement due to statement timeout")
        if q.startswith("SHOW"):
            self._next = (self.ro,)
        elif "current_database(), current_user, " in q:
            self._next = self.ident
        elif "disposable_marker" in q:
            if self.tag == "MISSING_TABLE":
                raise RuntimeError('relation "vera_local_meta.disposable_marker" does not exist')
            self._next = None if self.tag is None else (self.tag,)

    def fetchone(self):
        return self._next


class FakeConn:
    def __init__(self, cur):
        self.cur = cur

    def cursor(self):
        return self.cur


@pytest.fixture
def marker():
    nonce = secrets.token_bytes(32)
    return {"nonce": nonce.hex(), "dbname": "vera_test_x", "oid": 41001, "port": 5433}, nonce


def _check(cur, m, sources=None, events=None):
    return ec._inert_marker_check(FakeConn(cur), m, sources=sources or ec.Sources.empty(), events=events)


def test_correct_tag_passes_inert_helper(marker):
    m, nonce = marker
    tag = ec._inert_marker_tag(nonce, "vera_test_x", 41001)
    events = []
    assert _check(FakeCursor(tag=tag), m, events=events) == "marker_ok"
    assert events == ["read_only", "identity", "marker"]      # negative checks before the marker read


@pytest.mark.parametrize("kw,code", [
    (dict(tag="00" * 32), "tag_mismatch"),
    (dict(tag="MISSING_TABLE"), "probe_error"),
    (dict(tag=None), "no_marker"),
    (dict(fail_on="SHOW"), "probe_error"),
    (dict(fail_on="disposable_marker"), "probe_error"),
    (dict(ro="off"), "not_read_only"),
])
def test_failures(marker, kw, code):
    m, nonce = marker
    kw.setdefault("tag", ec._inert_marker_tag(nonce, "vera_test_x", 41001))
    assert _check(FakeCursor(**kw), m) == code


def test_oid_mismatch_from_restored_dump(marker):
    m, nonce = marker
    tag = ec._inert_marker_tag(nonce, "vera_test_x", 41001)
    assert _check(FakeCursor(ident=("vera_test_x", "tester", 99999, 5433), tag=tag), m) == "tag_mismatch"
    assert _check(FakeCursor(ident=("vera_test_x", "tester", 41001, 6543), tag=tag), m) == "tag_mismatch"


def test_negative_checks_pd_pr_first(marker, tmp_path):
    m, nonce = marker
    f = tmp_path / ".env-vera"
    f.write_text("VERA_DB_URL_RO=postgresql://prod_role:x@canary-prod.invalid/vera_test_x\n")
    src = ec.Sources.from_files(f, None, {})
    events = []
    tag = ec._inert_marker_tag(nonce, "vera_test_x", 41001)
    assert _check(FakeCursor(tag=tag), m, sources=src, events=events) == "negative_check"
    assert "marker" not in events
    assert _check(FakeCursor(ident=("other_db", "prod_role", 41001, 5433), tag=tag), m, sources=src) == \
        "negative_check"


def test_inert_helpers_unreachable_from_classification():
    for rel in ("safety/connect_guard.py", "safety/sql_guard.py", "safety/destructive.py", "scripts/check_config.py"):
        text = (ROOT / rel).read_text()
        assert "_inert_marker" not in text and "marker_file" not in text, rel
    tree = ast.parse((ROOT / "safety" / "endpoint_class.py").read_text())
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and not n.name.startswith("_inert")):
        names = {getattr(x, "id", None) for x in ast.walk(fn)}
        assert not names & {"_inert_marker_check", "_inert_marker_tag"}, fn.name


def test_verify_disposable_ignores_a_perfect_marker(marker):
    m, nonce = marker
    cur = FakeCursor(tag=ec._inert_marker_tag(nonce, "vera_test_x", 41001))
    v = ec.verify_disposable(FakeConn(cur), "postgresql://t:x@localhost:5433/vera_test_x", marker_file=m, environ={})
    assert v.cls == ec.PRODUCTION and ec.VERIFY_DISABLED in v.reasons
    assert cur.log == []


# ---- owner creation tool (V3 structural zero-row rule), fake cursor only -------------------------------------
class CreationCursor:
    def __init__(self, db="vera_test_x", role="tester", tables=(("vera_vjay", "runs"),), rows=None):
        self.db, self.role, self.tables, self.rows = db, role, list(tables), rows or {}
        self.log, self._one, self._all = [], None, None

    def execute(self, q, params=None):
        self.log.append(q)
        if q.startswith("SELECT current_database()"):
            self._one = (self.db, self.role)
        elif "information_schema.tables" in q:
            self._all = [t for t in self.tables if t[0] in params[0]]
        elif q.startswith("SELECT EXISTS"):
            key = q.split("FROM ")[1].rstrip(")").replace('"', "")
            self._one = (self.rows.get(key, 0) >= 1,)

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._all


def test_creation_tool_structural_rule(tmp_path):
    tool = _load_marker_tool()
    schemas = tool.production_schemas(ROOT / "db" / "migrations")
    assert "vera_vjay" in schemas
    assert tool.structural_refusal(CreationCursor(), schemas, ec.Sources.empty()) is None     # empty schemas pass
    one_row = CreationCursor(tables=[("vera_vjay", "runs"), ("vera_vjay", "claims")], rows={"vera_vjay.claims": 1})
    assert tool.structural_refusal(one_row, schemas, ec.Sources.empty()) == "rows_present"
    f = tmp_path / ".env-vera"
    f.write_text("VERA_DB_URL_RO=postgresql://prod_role:x@canary-prod.invalid/canaryprod_db\n")
    src = ec.Sources.from_files(f, None, {})
    assert tool.structural_refusal(CreationCursor(db="canaryprod_db"), schemas, src) == "dbname"
    assert tool.structural_refusal(CreationCursor(role="prod_role"), schemas, src) == "role"
    assert tool.structural_refusal(CreationCursor(tables=[("vera_vjay", 'x"; drop')]), schemas, src) == \
        "bad_identifier"


def test_creation_tool_tag_matches_inert_verifier():
    tool = _load_marker_tool()
    n = secrets.token_bytes(32)
    assert tool.make_tag(n, "vera_test_x", 7) == ec._inert_marker_tag(n, "vera_test_x", 7)


def test_creation_tool_main_refuses(capsys):
    tool = _load_marker_tool()
    assert tool.main([]) == 3
    assert ec.VERIFY_DISABLED in capsys.readouterr().err
