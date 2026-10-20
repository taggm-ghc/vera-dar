"""safety/connect_guard.py (PLAN-ENV v2.1 §3 E0/C5, V1, V2, L1, §5). Driver stubs only; no DB, no network.

The test boundary installs the process-global guard in the "test" context (E2). These tests therefore exercise the
global guard read-only, and drive app-mode behaviour through PRIVATE Guard instances installed on fake driver modules.
"""
import os
import types

import pytest

from safety import connect_guard as cg
from safety import endpoint_class as ec
from safety import sql_guard as sg

URL = "postgresql://app_ro:s3cret@db-app.test:5432/appdb"


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *a, **kw):
        self.calls.append((a, kw))
        return types.SimpleNamespace(cursor=lambda *x, **y: None, close=lambda: None)


@pytest.fixture
def clean_env(monkeypatch):
    for n in list(os.environ):
        if n.upper().startswith(ec.AMBIENT_PREFIXES):
            monkeypatch.delenv(n)
    return monkeypatch


def _guard(ctx):
    rec = Recorder()
    mod = types.SimpleNamespace(connect=rec)
    g = cg.Guard()
    g.install(ctx, sqlalchemy=False, drivers=[(mod, "connect")])
    return g, mod, rec


def test_global_guard_is_shared_and_guarded_in_tests():
    assert cg.installed_context() == "test"
    with pytest.raises(cg.ConnectRefused) as ei:
        cg.guard_url("postgresql://t:x@localhost:5433/vera_test_x")
    assert ei.value.code in (ec.VERIFY_DISABLED, "pg_env")
    with pytest.raises(cg.ConnectRefused) as ei:
        cg.install("production")                  # guarded -> app never loosens
    assert ei.value.code == "context_conflict"


def test_rebuild_ignores_raw_url_extras_and_merges_allowlist():
    p = ec.parse("postgresql://u:p%40ss@h.test:6000/db?sslmode=require&application_name=a")
    kw = cg.rebuild_kwargs(p, {"connect_timeout": 5, "options": "-c statement_timeout=30000"})
    assert kw == {"host": "h.test", "port": 6000, "dbname": "db", "user": "u", "password": "p@ss",
                  "sslmode": "require", "application_name": "a", "connect_timeout": 5,
                  "options": "-c statement_timeout=30000", "passfile": "/dev/null", "sslcertmode": "disable",
                  "gssencmode": "disable"}


@pytest.mark.parametrize("args", [
    {"host": "x"}, {"hostaddr": "10.0.0.1"}, {"port": 1}, {"dbname": "x"}, {"user": "x"}, {"passfile": "/x"},
    {"service": "x"}, {"password": "x"}, {"sslrootcert": "/x"}, {"keepalives": 1},
    {"options": "-c search_path=evil"}, {"options": "-c statement_timeout=0; DROP"}, {"connect_timeout": "x"},
    {"connect_timeout": True}, {"sslmode": "disable"}, {"application_name": "a b"},
])
def test_v1_refuses_endpoint_and_unknown_keys(args):
    p = ec.parse("postgresql://u:p@h.test/db?sslmode=require")
    with pytest.raises(cg.ConnectRefused) as ei:
        cg.rebuild_kwargs(p, args)
    assert ei.value.code == "bad_param"


def test_v1_real_engine_configs_keep_timeouts(clean_env):
    """The three app engines (dar_health, memory_store, dar_runtime) keep connect_timeout and statement_timeout."""
    from vera import dar_runtime
    captured = {}

    def factory(url, **kw):
        captured.update(kw)
        return object()

    dar_runtime._engine_for({"db_url_env": "X_URL", "connect_timeout_s": 7, "statement_timeout_ms": 1500},
                            {"X_URL": URL}, factory)
    kwargs, verdict = cg.Guard().guard_url(URL, captured["connect_args"], context="production")
    assert kwargs["connect_timeout"] == 7 and kwargs["options"] == "-c statement_timeout=1500"
    assert verdict.cls == ec.PRODUCTION
    import inspect
    from vera import dar_health, memory_store
    for mod in (dar_health, memory_store):
        src = inspect.getsource(mod)
        assert '"options": f"-c statement_timeout={int(' in src and '"connect_timeout": int(' in src


@pytest.mark.parametrize("ctx", cg.GUARDED_CONTEXTS)
def test_guarded_contexts_always_refuse_before_driver(clean_env, ctx):
    g, mod, rec = _guard(ctx)
    with pytest.raises(cg.ConnectRefused) as ei:
        mod.connect("postgresql://t:x@localhost:5433/vera_test_x")
    assert ei.value.code == ec.VERIFY_DISABLED
    assert rec.calls == []


def test_refused_url_never_reaches_driver(clean_env):
    g, mod, rec = _guard("production")
    for url, code in (("postgresql://u:p@h.test/db?hostaddr=1.2.3.4", "bad_param"), ("nope", "bad_scheme"),
                      ("postgresql://u:p@/db", "empty_host"), (None, "no_url")):
        with pytest.raises(cg.ConnectRefused) as ei:
            mod.connect(url)
        assert ei.value.code == code
    for kw in ({"host": "other.test"}, {"connection_factory": object}, {"hostaddr": "10.0.0.9"}):
        with pytest.raises(cg.ConnectRefused):
            mod.connect(URL, **kw)
    with pytest.raises(cg.ConnectRefused):
        mod.connect(URL, "extra-positional")
    assert rec.calls == []


@pytest.mark.parametrize("name", ["PGHOST", "PGPASSFILE", "PGSERVICE", "KRB5CCNAME", "GSSAPI_X"])
@pytest.mark.parametrize("ctx", ["production", "local_dev", "agent", "test"])
def test_ambient_pg_names_refuse_in_every_context(clean_env, name, ctx):
    g, mod, rec = _guard(ctx)
    clean_env.setenv(name, "x")
    with pytest.raises(cg.ConnectRefused) as ei:
        mod.connect(URL)
    assert ei.value.code == "pg_env" and rec.calls == []


def test_v2e_percent_password_connects_in_app_mode(clean_env, capsys, caplog):
    g, mod, rec = _guard("local_dev")
    pw_dec = "pa:ss@w/rd%x"
    pw_enc = "pa%3Ass%40w%2Frd%25x"
    conn = mod.connect(f"postgresql://u:{pw_enc}@h.test/db", connect_timeout=3)
    assert len(rec.calls) == 1 and rec.calls[0][0] == ()
    kw = rec.calls[0][1]
    assert kw["password"] == pw_dec and kw["host"] == "h.test" and kw["connect_timeout"] == 3
    assert kw["passfile"] == "/dev/null" and kw["gssencmode"] == "disable" and kw["sslcertmode"] == "disable"
    assert isinstance(conn, sg.GuardedConnection)
    v = cg.verdict_of(conn)
    assert v.cls == ec.PRODUCTION
    out = capsys.readouterr()
    for chan in (out.out, out.err, caplog.text, repr(conn), repr(v), str(v.as_dict()), repr(g)):
        assert pw_dec not in chan and pw_enc not in chan


def test_v2b_unparsable_in_app_mode_fails_closed(clean_env):
    g, mod, rec = _guard("agent")
    with pytest.raises(cg.ConnectRefused) as ei:
        mod.connect("postgresql://u:p@a.test,b.test/db")
    assert ei.value.code == "multihost" and rec.calls == []


def test_install_contexts():
    g = cg.Guard()
    with pytest.raises(cg.ConnectRefused) as ei:
        g.install("nonsense", sqlalchemy=False, drivers=[])
    assert ei.value.code == "bad_context"
    g.install("production", sqlalchemy=False, drivers=[])
    g.install("production", sqlalchemy=False, drivers=[])          # idempotent
    with pytest.raises(cg.ConnectRefused) as ei:
        g.install("local_dev", sqlalchemy=False, drivers=[])
    assert ei.value.code == "context_conflict"
    g.install("test", sqlalchemy=False, drivers=[])                 # app -> guarded tightens
    assert g.context == "test"
    with pytest.raises(cg.ConnectRefused):
        g.install("production", sqlalchemy=False, drivers=[])


def test_idempotent_wrap_and_uninstall_restores():
    rec = Recorder()
    mod = types.SimpleNamespace(connect=rec)
    g = cg.Guard()
    g.install("production", sqlalchemy=False, drivers=[(mod, "connect")])
    g.install("production", sqlalchemy=False, drivers=[(mod, "connect")])
    assert mod.connect._vera_original is rec
    g.uninstall()
    assert mod.connect is rec


@pytest.mark.boundary_probe
def test_sqlalchemy_real_path_refuses_before_driver(monkeypatch):
    """create_engine(url).connect() in the test process: the boundary (outermost) and the global guard both refuse;
    whichever fires first, psycopg2's C entry point is never reached. (The raw-driver path is covered with fake driver
    modules above; tests/test_env_guard.py bans literal production connector calls in tests.)"""
    import psycopg2
    raw = Recorder()
    monkeypatch.setattr(psycopg2, "_connect", raw)
    from sqlalchemy import create_engine
    for ca in ({}, {"hostaddr": "10.0.0.1"}):
        eng = create_engine("postgresql+psycopg2://u:p@h.test/db", connect_args=ca)
        with pytest.raises(Exception) as ei:
            eng.connect()
        assert type(ei.value).__name__ in ("ConnectRefused", "DBBoundaryError")
    assert raw.calls == []


class _Rec:
    def __init__(self):
        self.info = {}


def test_sqlalchemy_hook_refuses_via_dialect_dispatch(clean_env):
    """Drive a private guard's listener exactly as SQLAlchemy's pool does, with the engine its create_engine saw."""
    import sqlalchemy
    g = cg.Guard()
    g.install("production", sqlalchemy=True, drivers=[])
    try:
        eng = sqlalchemy.create_engine("postgresql+psycopg2://u:p@h.test/db", connect_args={"hostaddr": "10.0.0.1"})
        assert g.dialect_info[eng.dialect][1] == {"hostaddr": "10.0.0.1"}
        assert any(f == g.listener for f in eng.dialect.dispatch.do_connect)
        with pytest.raises(cg.ConnectRefused) as ei:
            g.on_do_connect(eng.dialect, _Rec(), (), {"host": "h.test", "dbname": "db", "user": "u"})
        assert ei.value.code == "bad_param"
        eng2 = sqlalchemy.create_engine("postgresql+psycopg2://u:p@h.test/db", connect_args={"host": "evil.test"})
        with pytest.raises(cg.ConnectRefused):
            g.on_do_connect(eng2.dialect, _Rec(), (), {})
    finally:
        g.uninstall()
    assert not any(f == g.listener for f in sqlalchemy.create_engine("sqlite://").dialect.dispatch.do_connect)


class _Dialect:
    name = "postgresql"

    def __init__(self):
        self.calls = []

    def connect(self, *a, **kw):
        self.calls.append(kw)
        return object()


def test_do_connect_hook_app_mode_tags_and_rebuilds(clean_env):
    g = cg.Guard()
    g.install("production", sqlalchemy=False, drivers=[])
    d = _Dialect()
    g.dialect_info[d] = (URL + "?sslmode=require", {"connect_timeout": 4})
    rec = _Rec()
    g.on_do_connect(d, rec, (), {"host": "ignored.test", "dbname": "ignored"})
    assert d.calls[0]["host"] == "db-app.test" and d.calls[0]["dbname"] == "appdb"
    assert d.calls[0]["connect_timeout"] == 4 and d.calls[0]["sslmode"] == "require"
    assert rec.info[sg.VERDICT_KEY].cls == ec.PRODUCTION
    assert g.on_do_connect(types.SimpleNamespace(name="sqlite"), rec, (), {}) is None


def test_do_connect_hook_fallback_for_untagged_engine(clean_env):
    g = cg.Guard()
    g.install("production", sqlalchemy=False, drivers=[])
    d = _Dialect()
    g.on_do_connect(d, _Rec(), (), {"host": "h.test", "port": 5432, "dbname": "db", "user": "u",
                                    "password": "p:w", "connect_timeout": 3})
    assert d.calls[0]["password"] == "p:w" and d.calls[0]["connect_timeout"] == 3
    for cp in ({"host": "h.test", "dbname": "db", "user": "u", "hostaddr": "10.0.0.1"},
               {"host": "h.test", "dbname": "db", "user": "u", "service": "x"}):
        with pytest.raises(cg.ConnectRefused):
            g.on_do_connect(_Dialect(), _Rec(), (), cp)
    with pytest.raises(cg.ConnectRefused):
        g.on_do_connect(_Dialect(), _Rec(), ("dsn",), {})


def test_pool_reconnect_reverifies_no_cache(clean_env):
    """Every new DBAPI connection re-runs the decision; a fake DNS change shows up on the next connect."""
    answers = iter([["198.51.100.1"], ["192.0.2.66"]])
    src = ec.Sources.empty()
    src._n_sources = 1
    src.add_resolved_ips(["192.0.2.66"])
    prev = ec.get_default_sources()
    ec.set_default_sources(src)
    try:
        g, mod, rec = _guard("production")
        g.resolver = lambda host: next(answers)
        c1 = mod.connect(URL)
        c2 = mod.connect(URL)
        assert "ip" not in cg.verdict_of(c1).reasons and "ip" in cg.verdict_of(c2).reasons
        assert len(rec.calls) == 2
    finally:
        ec.set_default_sources(prev)
    g2, mod2, rec2 = _guard("test")
    for _ in range(2):
        with pytest.raises(cg.ConnectRefused):
            mod2.connect(URL)
    assert rec2.calls == []


class _DialectVia:
    name = "postgresql"

    def __init__(self, mod):
        self.mod = mod

    def connect(self, *a, **kw):
        return self.mod.connect(*a, **kw)


def test_l1_reentry_driver_called_once(clean_env):
    """A SQLAlchemy connect passes through the do_connect hook and the driver wrapper; the guard acts once."""
    g, mod, rec = _guard("production")
    dd = _DialectVia(mod)
    g.dialect_info[dd] = (URL, {})
    conn = g.on_do_connect(dd, _Rec(), (), {})
    assert len(rec.calls) == 1 and not isinstance(conn, sg.GuardedConnection)
    assert rec.calls[0][1]["passfile"] == "/dev/null"


def test_codes_only_messages():
    e = cg.ConnectRefused("bad_param")
    assert str(e) == "connect refused: bad_param" and e.args == ("bad_param",)
