"""ENV-E (E7/C8, C12): static/fake tests. Ops-script headers and guard install, no connection-error text leaks,
hide_parameters at the engine sites. No connection, no env file, no network."""
import ast
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
OPS = {"memory_recall_drill.py": "ops-write", "ingest_fulltext.py": "ops-write",
       "test_db_connection.py": "ops-read", "live_catalog_check.py": "ops-read"}


@pytest.mark.parametrize("name,cls", sorted(OPS.items()))
def test_ops_header_install_and_test_refusal(name, cls):
    src = (REPO / "scripts" / name).read_text(encoding="utf-8")
    assert f"Ops class: {cls}" in src.split('"""', 2)[1]
    assert 'connect_guard.install("production")' in src
    assert "VERA_TEST_RUNNER" in src and "PYTEST_CURRENT_TEST" in src


def _load(name):
    spec = importlib.util.spec_from_file_location("enve_probe_" + name[:-3], REPO / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    return spec, mod


@pytest.mark.parametrize("var", ["VERA_TEST_RUNNER", "PYTEST_CURRENT_TEST"])
def test_ops_guard_refuses_under_test_before_install(monkeypatch, var):
    monkeypatch.setenv(var, "1")
    from safety import connect_guard
    before = connect_guard.installed_context()
    for name in ("memory_recall_drill.py", "ingest_fulltext.py", "live_catalog_check.py"):
        spec, mod = _load(name)
        spec.loader.exec_module(mod)
        with pytest.raises(SystemExit) as ei:
            mod._ops_guard()
        assert ei.value.code in (2, 3)
    assert connect_guard.installed_context() == before


def test_test_db_connection_refuses_under_test_without_connecting(monkeypatch, capsys):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "x")
    spec, mod = _load("test_db_connection.py")
    spec.loader.exec_module(mod)
    assert mod.main() == 3
    assert "refused" in capsys.readouterr().err


def test_no_str_of_connection_errors_in_ops_paths():
    src = (REPO / "scripts" / "test_db_connection.py").read_text(encoding="utf-8")
    assert 'f"FAILED: {exc}"' not in src
    for rel in ("scripts/grounded_trace_run.py", "vera/run_mvp.py"):
        s = (REPO / rel).read_text(encoding="utf-8")
        assert "_err_text(" in s and "_is_conn_error" in s
        assert 'f"create_run failed: {type(e).__name__}: {e}"' not in s


def _fake_errors():
    from sqlalchemy.exc import OperationalError
    from safety.connect_guard import ConnectRefused
    return [OperationalError("select 1", {}, Exception("host db-sentinel.invalid user u pw hunter2")),
            ConnectRefused("pg_env")]


@pytest.mark.parametrize("rel", ["vera/run_mvp.py", "scripts/grounded_trace_run.py"])
def test_err_text_names_only_for_connection_class(rel):
    spec = importlib.util.spec_from_file_location("enve_err_probe", REPO / rel)
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    ns = {}
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_is_conn_error", "_err_text")]
    exec(compile(ast.Module(body=fns, type_ignores=[]), rel, "exec"), ns)
    for e in _fake_errors():
        out = ns["_err_text"](e)
        assert "sentinel" not in out and "hunter2" not in out and "select" not in out.lower(), out
        assert out.startswith(type(e).__name__)
    assert ns["_err_text"](ValueError("plain")).endswith("plain")


def test_engine_sites_pass_hide_parameters():
    for rel in ("vera/db.py", "vera/dar_health.py", "vera/dar_runtime.py", "vera/memory_store.py",
                "vera/corpus_admission.py", "vera/corpus_description.py"):
        s = (REPO / rel).read_text(encoding="utf-8")
        assert "hide_parameters=True" in s or '"hide_parameters": True' in s, rel
    assert (REPO / "vera/db.py").read_text(encoding="utf-8").count("hide_parameters=True") == 2


def test_engine_factories_receive_hide_parameters(monkeypatch):
    from vera import db
    seen = []
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_database_url", lambda: "postgresql://u:p@h.invalid/d")
    monkeypatch.setattr(db, "create_engine", lambda url, **kw: seen.append(kw) or object())
    monkeypatch.setattr(db, "sessionmaker", lambda **kw: None)
    db.get_engine()
    db.get_role_engine("rw", env={db.DB_URL_ENV_RW: "postgresql://u:p@h.invalid/d"},
                       engine_factory=lambda url, **kw: seen.append(kw))
    assert len(seen) == 2 and all(k.get("hide_parameters") is True for k in seen)


# ---- FIX-IMPL: M2 (connect() reaches the driver under the production guard), M5 (no endpoint print), M1 (launchers)
def test_live_catalog_connect_reaches_driver_under_production_guard(monkeypatch):
    """Fake driver only: connect() builds a DSN from the parsed URL; the guard (production context) lets it through
    and the kwargs carry the read-only options. .invalid host, fake password, no network."""
    import types
    import psycopg2
    from safety import connect_guard as cg
    seen = []

    class FakeConn:
        def set_session(self, **kw):
            seen.append(("set_session", kw))

    fake_driver = types.SimpleNamespace(connect=lambda *a, **k: (seen.append(("connect", a, k)), FakeConn())[1])
    g = cg.Guard()
    g.install("production", sqlalchemy=False, drivers=[(fake_driver, "connect")])
    try:
        monkeypatch.setattr(psycopg2, "connect", fake_driver.connect)
        monkeypatch.setenv("VERA_DB_URL_RO", "postgresql://vera_eval_ro:p%40ss%2Fw@db-fixture.invalid:5433/fixturedb")
        spec, mod = _load("live_catalog_check.py")
        spec.loader.exec_module(mod)
        conn = mod.connect()
    finally:
        g.uninstall()
    assert conn is not None, "the guard refused or the driver was not reached"   # guard returns its tagged wrapper
    calls = [s for s in seen if s[0] == "connect"]
    assert len(calls) == 1
    # the guard rebuilt the kwargs from the parse; the code-supplied args survived the allowlist
    kw = calls[0][2]
    assert kw.get("options") == "-c default_transaction_read_only=on -c statement_timeout=20000"
    assert kw.get("sslmode") == "require" and kw.get("connect_timeout") == 10
    assert kw.get("host") == "db-fixture.invalid" and kw.get("user") == "vera_eval_ro"
    assert ("set_session", {"readonly": True, "autocommit": True}) in seen


def test_live_catalog_connect_passes_one_dsn_string_not_split_kwargs():
    src = (REPO / "scripts" / "live_catalog_check.py").read_text(encoding="utf-8")
    assert "psycopg2.connect(dsn," in src and "psycopg2.connect(host=" not in src


def test_test_db_connection_prints_only_flag_and_class(monkeypatch, capsys):
    import types
    for n in ("VERA_TEST_RUNNER", "PYTEST_CURRENT_TEST"):
        monkeypatch.delenv(n, raising=False)
    spec, mod = _load("test_db_connection.py")
    spec.loader.exec_module(mod)
    from safety import connect_guard

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *_):
            return None

    engine = types.SimpleNamespace(connect=lambda: FakeConn(),
                                   url="postgresql://sentinel_user:pw@sentinel-host.invalid:5999/sentinel_db")
    monkeypatch.setattr(mod, "get_engine", lambda: engine)
    monkeypatch.setattr(connect_guard, "install", lambda *_a, **_k: None)
    assert mod.main() == 0
    out = capsys.readouterr()
    text = out.out + out.err
    assert "connected=true" in text and "safety_class=" in text
    for s in ("sentinel", "5999", "pw@", "invalid"):
        assert s not in text


@pytest.mark.parametrize("name", ["run.sh", "run_demo.sh"])
def test_launchers_default_to_explicit_local_dev_context(name):
    """R-018 item 6 parity: a default launcher start exports an explicit local_dev context (caller value wins),
    before the app is exec'd."""
    src = (REPO / name).read_text(encoding="utf-8")
    line = 'export VERA_ENV_CONTEXT="${VERA_ENV_CONTEXT:-local_dev}"'
    assert line in src
    assert src.index(line) < src.rindex("\nexec ")
    assert "production" not in line
