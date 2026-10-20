"""K18W WS-2 (RB-4 test isolation): negative and positive tests for the in-tree boundary changes.

Runs only under the VERA-DAR test boundary (tests/conftest.py, isolated netns). Every target below is synthetic: hosts
end in `.invalid`, credentials are sentinel strings, sockets are bound in tmp_path or on loopback inside the netns.
Nothing here connects to a database, a network service or a host socket.

Covered:
  M-2  a constant "reset" or raw write of the deny switch cannot loosen check(); persisting writes are reported as
       tamper and the sealed decision is restored (owner: "a test resetting the deny switch").
  OA-2 local sockets (loopback, AF_UNIX, abstract) are refused unless the test requests them (@pytest.mark.vera_sockets).
  L-4/credentials  the netns re-exec env and the release-gate child envs carry no DB URL, password or provider key.
  O9   tests/__init__.py carries no policy; operator scripts can import tests.sql_contract_support outside the boundary.
  M3-G the root conftest enforces the netns gate itself (dropping tests/conftest.py is not enough to bypass it).
"""
import ast
import importlib.util
import os
import socket
import sys
from pathlib import Path

import pytest

bt = sys.modules["tests.vera_test_boundary"]  # the single instance loaded by the conftests
REPO = Path(__file__).resolve().parent.parent
SENT = "postgresql://sentinel_user:sentinel_pw@db-sentinel.invalid:5432/sentinel_db"
SENT_TEST = "postgresql://sentinel_tester:sentinel_pw@db-sentinel-test.invalid:5432/sentinel_test_db"
CP_TEST = {"host": "db-sentinel-test.invalid", "port": 5432, "dbname": "sentinel_test_db", "user": "sentinel_tester"}


def _allowed_path_setup(monkeypatch):
    """The ONE configuration under which check() allows a target: live item, live session, disposable test DB."""
    monkeypatch.setattr(bt, "_orig_getaddrinfo", lambda host, port=None: [(2, 1, 6, "", ("127.0.0.1", 0))])
    monkeypatch.setenv("VERA_TEST_DB_URL", SENT_TEST)
    monkeypatch.setenv("VERA_TEST_DB_DISPOSABLE", "1")
    monkeypatch.setitem(bt.STATE, "live_enabled", True)
    bt.begin_item(live_item=True)


# ============================================================================== M-2: deny switch cannot be reset
def test_session_deny_decision_is_sealed():
    assert "deny_all" in bt._SEALED and "deny_policy" in bt._SEALED
    assert bt._SEALED["deny_all"] == (bt._SEALED["deny_policy"] != "ok")      # fail closed unless the policy is ok
    assert bt.tamper_keys() == []                                             # this session is untampered so far


@pytest.mark.boundary_probe
def test_constant_reset_of_deny_switch_cannot_loosen_check(monkeypatch):
    """The X-era `_reset_policy_state()` wrote deny_policy="ok", deny_all=False. With a fail-closed sealed session that
    must NOT allow anything."""
    _allowed_path_setup(monkeypatch)
    monkeypatch.setitem(bt._SEALED, "deny_all", True)
    monkeypatch.setitem(bt._SEALED, "deny_policy", "missing")
    monkeypatch.setitem(bt.STATE, "deny_all", False)          # the reset (restored by monkeypatch)
    monkeypatch.setitem(bt.STATE, "deny_policy", "ok")
    try:
        assert bt.effective_deny_all() is True
        with pytest.raises(bt.DBBoundaryError):
            bt.check((), dict(CP_TEST))
    finally:
        bt.begin_item(live_item=False)


def test_positive_control_explicit_loosening_of_both_values_is_still_refused_owner_decision(monkeypatch):
    """Not vacuous: with the sealed value (explicitly) loosened too, the deny switch is no longer what stops the target
    (precondition code moves from "deny_all" to VERIFY_DISABLED). Owner decision 2026-10-19
    (DISPOSABLE-VERIFY-DISABLED-pending-isolation-review): check() still refuses, records "db" and pins nothing."""
    _allowed_path_setup(monkeypatch)
    monkeypatch.setitem(bt._SEALED, "deny_all", False)
    monkeypatch.setitem(bt.STATE, "deny_all", False)
    try:
        target = ("db-sentinel-test.invalid", 5432, "sentinel_test_db", "sentinel_tester")
        kw = dict(live_item=True, live_enabled=True, env=os.environ, deny_fp=bt.STATE["deny_fp"],
                  deny_user_fp=bt.STATE["deny_user_fp"], deny_db_fp=bt.STATE["deny_db_fp"],
                  deny_suffix=bt.STATE["deny_suffix"], deny_ip_fp=bt.STATE["deny_ip_fp"])
        assert bt.live_preconditions(target, deny_all=True, **kw) == "deny_all"
        assert bt.live_preconditions(target, deny_all=False, **kw) == bt.VERIFY_DISABLED
        assert bt.decide(target, deny_all=False, **kw) is False
        cp = dict(CP_TEST)
        with pytest.raises(bt.DBBoundaryError):
            bt.check((), cp)
        assert "hostaddr" not in cp and bt.STATE["violations"][-1]["kind"] == "db"
    finally:
        bt.begin_item(live_item=False)


def test_raw_persisting_write_is_reported_as_tamper_by_name_only(monkeypatch):
    saved = (bt.STATE["deny_all"], bt.STATE["deny_policy"])
    monkeypatch.setitem(bt.STATE, "tamper_count", bt.STATE["tamper_count"])   # undone at teardown
    try:
        bt.STATE["deny_all"] = not bt._SEALED["deny_all"]       # raw write, NOT monkeypatch: would persist
        bt.STATE["deny_policy"] = "sentinel-policy"
        assert set(bt.tamper_keys()) >= {"deny_all", "deny_policy"}
        msg = bt.tamper_message()
        assert msg and "deny_all" in msg and "sentinel-policy" not in msg     # names, never values
        # fail closed: tamper_message() put the sealed decision back
        assert bt.STATE["deny_all"] == bt._SEALED["deny_all"] and bt.STATE["deny_policy"] == bt._SEALED["deny_policy"]
        assert bt.tamper_keys() == []
    finally:
        bt.STATE["deny_all"], bt.STATE["deny_policy"] = saved


def test_replaced_socket_wrapper_is_reported_as_tamper(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda self, address: None)
    assert "wrapper:connect" in bt.tamper_keys()
    monkeypatch.undo()
    assert "wrapper:connect" not in bt.tamper_keys()


@pytest.mark.parametrize("key,value", [("netns", "live-session"), ("seccomp", "none"), ("installed", False),
                                       ("live_enabled", True)])
def test_session_isolation_facts_are_sealed(monkeypatch, key, value):
    if bt.STATE[key] == value:
        pytest.skip("value already equal in this session")
    monkeypatch.setitem(bt.STATE, key, value)
    assert key in bt.tamper_keys()


def test_tamper_hook_is_wired_after_teardown_and_withholds_evidence():
    src = (REPO / "tests" / "conftest.py").read_text()
    tree = ast.parse(src)
    fns = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert 'call.when == "teardown"' in fns["pytest_runtest_makereport"] and "tamper_message()" in fns["pytest_runtest_makereport"]
    assert "seal_session()" in fns["pytest_sessionstart"]
    term = fns["pytest_terminal_summary"]
    assert term.index('STATE["tamper_count"]') < term.index("VERA-ISOLATION-EVIDENCE")
    assert 'STATE["tamper_count"]' in fns["pytest_sessionfinish"]


# ============================================================================== hidden DB connections
@pytest.mark.boundary_probe
def test_hidden_db_connections_are_refused_before_any_socket():
    # direct psycopg2.connect calls are exercised only in tests/test_env_guard.py (its connector lint allows no other file)
    from sqlalchemy import create_engine
    for fn in (lambda: create_engine(SENT).connect(),
               lambda: create_engine(SENT, pool_pre_ping=True).connect(),
               lambda: create_engine("postgresql+psycopg2://sentinel_user@db-sentinel.invalid/sentinel_db").connect()):
        before = sum(1 for v in bt.STATE["violations"] if v["kind"] == "db")
        with pytest.raises(Exception):
            fn()
        assert sum(1 for v in bt.STATE["violations"] if v["kind"] == "db") == before + 1


@pytest.mark.boundary_probe
@pytest.mark.vera_sockets
def test_socket_request_does_not_open_the_db_boundary():
    from sqlalchemy import create_engine
    with pytest.raises(bt.DBBoundaryError):
        create_engine("postgresql://sentinel_user:sentinel_pw@127.0.0.1:5432/sentinel_db").connect()


def test_sqlite_still_works():
    from sqlalchemy import create_engine, text
    with create_engine("sqlite://").connect() as c:
        assert c.execute(text("select 1")).scalar() == 1


# ============================================================================== OA-2: sockets only on request
def _unix_server(name):
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(name)
    srv.listen(1)
    return srv


@pytest.mark.boundary_probe
def test_local_sockets_refused_without_request(tmp_path, monkeypatch):
    assert bt.STATE["sockets_requested"] is False
    monkeypatch.chdir(tmp_path)                       # under basetemp: allowed by the roots rule, but not requested
    usrv = _unix_server("u.sock")
    tsrv = socket.create_server(("127.0.0.1", 0))
    host, port = tsrv.getsockname()
    abstract = b"\0vera-k18w-%d" % os.getpid()
    asrv = _unix_server(abstract)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        attempts = {
            "unix_connect": lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).connect("u.sock"),
            "unix_connect_ex": lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).connect_ex("u.sock"),
            "abstract": lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).connect(abstract),
            "tcp_create_connection": lambda: socket.create_connection((host, port), timeout=2),
            "tcp_connect": lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port)),
            "udp_sendto": lambda: udp.sendto(b"x", ("127.0.0.1", 9)),
        }
        for name, fn in attempts.items():
            before = len(bt.STATE["violations"])
            with pytest.raises(bt.NetworkBoundaryError) as ei:
                fn()
            assert str(ei.value) == bt.LOCAL_SOCKET_REFUSAL_TEXT, name       # diagnosable: names the marker
            assert len(bt.STATE["violations"]) == before + 1, name
    finally:
        usrv.close(); tsrv.close(); asrv.close(); udp.close()


@pytest.mark.vera_sockets
def test_requested_local_sockets_work(tmp_path, monkeypatch):
    assert bt.STATE["sockets_requested"] is True
    monkeypatch.chdir(tmp_path)
    usrv = _unix_server("u.sock")
    tsrv = socket.create_server(("127.0.0.1", 0))
    try:
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.connect("u.sock")
        c.close()
        socket.create_connection(tsrv.getsockname(), timeout=2).close()
    finally:
        usrv.close(); tsrv.close()
    assert bt.STATE["violations"] == []


@pytest.mark.boundary_probe
@pytest.mark.vera_sockets
def test_socket_request_never_opens_non_loopback(monkeypatch):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(bt.NetworkBoundaryError) as ei:
            s.connect(("192.0.2.1", 9))
        assert str(ei.value) == bt.NET_REFUSAL_TEXT
    finally:
        s.close()
    with pytest.raises(bt.NetworkBoundaryError):
        socket.create_connection(("192.0.2.1", 9), timeout=1)


def test_unaddressed_socketpair_and_loopback_resolution_need_no_request():
    a, b = socket.socketpair()
    a.close(); b.close()
    assert socket.getaddrinfo("localhost", 80)
    assert bt.STATE["violations"] == []


def test_socket_request_is_per_item_and_reset():
    bt.begin_item(live_item=False, sockets=True)
    assert bt.STATE["sockets_requested"] is True
    bt.end_item()
    assert bt.STATE["sockets_requested"] is False
    bt.begin_item(live_item=False)
    assert bt.STATE["sockets_requested"] is False


# ============================================================================== credential inheritance
CRED_ENV = {"VERA_DB_URL_RO": SENT, "VERA_DB_URL_WO": SENT, "VERA_DB_URL_RW": SENT, "DATABASE_URL": SENT,
            "EXTERNAL_DB_URL": SENT, "PGPASSWORD": "sentinel_pw", "DB_PASSWORD": "sentinel_pw",
            "OPENAI_API_KEY": "GUARD-SENTINEL", "HTTPS_PROXY": "http://proxy.invalid:3128",
            "WSL_INTEROP": "/run/WSL/1_interop"}


def test_reexec_env_carries_no_credentials():
    env = bt.reexec_env({"PATH": "/usr/bin:/bin", "HOME": "/home/sentinel", "LANG": "C.UTF-8",
                         "VERA_TEST_DB_URL": SENT_TEST, **CRED_ENV})
    assert not set(CRED_ENV) & set(env)
    assert env["PATH"] == "/usr/bin:/bin" and env["HOME"] == "/home/sentinel"
    assert env["VERA_TEST_DB_URL"] == SENT_TEST             # PASSTHROUGH: the explicit disposable-DB designation
    assert "GUARD-SENTINEL" not in repr(env) and "sentinel_pw@db-sentinel.invalid" not in repr(env)


def test_reexec_uses_reexec_env():
    src = (REPO / "tests" / "vera_test_boundary.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "maybe_reexec")
    seg = ast.get_source_segment(src, fn)
    # INT3 MED-A: maybe_reexec no longer execs at all (a non-isolated session is refused), so it can pass no env
    # anywhere; reexec_env stays the only env builder for the (now library-only) chain.
    assert "execve" not in seg and "reexec_argv" not in seg and "_ORIG_ENV.items()" not in seg
    assert "ISOLATION_REQUIRED_MESSAGE" in seg
    assert bt.reexec_env({"PGPASSWORD": "x", "WSL_INTEROP": "y", "PATH": "/usr/bin"}) == {"PATH": "/usr/bin"}


def test_no_guarded_name_visible_to_test_code():
    assert [k for k in os.environ if bt.guarded(k)] == []


def _gate():
    spec = importlib.util.spec_from_file_location("k18w_release_gate_under_test", REPO / "scripts" / "release_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                     # module level: imports, constants, defs; __main__ block inert
    return mod


def test_release_gate_child_envs_carry_no_credentials(tmp_path):
    g = _gate()
    outer = {"PATH": "/opt/evil/bin:/usr/bin", "HOME": "/home/sentinel", "LANG": "C.UTF-8", **CRED_ENV}
    suite = g.child_env(g._child_env(outer), "0" * 32, str(tmp_path))
    child_check = g._child_env(outer)
    for env in (suite, child_check):
        leaked = [k for k in env if "DB_URL" in k or k in ("DATABASE_URL", "PGPASSWORD")]
        assert leaked == [], leaked
    assert suite["PATH"] == "/usr/bin:/bin"
    assert not set(CRED_ENV) & set(suite)


# ============================================================================== O9: tests/__init__.py carries no policy
def test_tests_package_init_has_no_side_effects():
    init = REPO / "tests" / "__init__.py"
    tree = ast.parse(init.read_text())
    body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert body == [], "tests/__init__.py must not carry policy (owner O9: wrong abstraction layer)"


ALLOWED_SUPPORT_IMPORTS = {"__future__", "ast", "re", "pathlib"}


def test_sql_contract_support_is_side_effect_free_for_operator_scripts():
    """scripts/live_catalog_check.py (owner-run evidence, outside the test boundary) does `from tests import
    sql_contract_support`. That import must not install the boundary, exit, or pull application code (F-RB4B-1)."""
    tree = ast.parse((REPO / "tests" / "sql_contract_support.py").read_text())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    assert mods <= ALLOWED_SUPPORT_IMPORTS, mods
    top = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef,
                                                       ast.Assign, ast.AnnAssign))
           and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert top == [], [type(n).__name__ for n in top]


def test_live_catalog_check_import_contract_unchanged():
    src = (REPO / "scripts" / "live_catalog_check.py").read_text()
    assert "from tests import sql_contract_support" in src


# ============================================================================== M3-G: root conftest gate
def _root_conftest(request):
    want = str(REPO / "conftest.py")
    for plugin in request.config.pluginmanager.get_plugins():
        if getattr(plugin, "__file__", None) and os.path.realpath(plugin.__file__) == want:
            return plugin
    pytest.fail("root conftest not registered")


def test_root_conftest_enforces_netns_gate(request, monkeypatch):
    rc = _root_conftest(request)
    calls = []
    monkeypatch.setattr(bt, "_isolation_fail", lambda cfg, text: calls.append(text))
    monkeypatch.setitem(bt.STATE, "netns", "undecided")
    rc.pytest_sessionstart(request.session)
    assert calls == [bt.NETNS_MESSAGE]
    calls.clear()
    monkeypatch.setitem(bt.STATE, "netns", "isolated")
    rc.pytest_sessionstart(request.session)
    assert calls == []
    # tests/conftest.py never ran (dropped): the root conftest makes the netns decision itself, never as a live session
    decided = []
    monkeypatch.setitem(bt.STATE, "netns_decided", False)
    monkeypatch.setitem(bt.STATE, "netns", "undecided")
    monkeypatch.setattr(bt, "maybe_reexec", lambda cfg, live: decided.append(live))
    rc.pytest_sessionstart(request.session)
    assert decided == [False] and calls == [bt.NETNS_MESSAGE]     # still undecided after the (faked) decision: refused
