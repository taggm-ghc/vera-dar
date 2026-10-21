"""K18W INT2: regression tests for the WS-8c verifier findings (verification-report.md f5d40bdfaee9...).

Each test fails on the INT2 base (120ab53 + integrated-K18W.diff 690ad47c...) and passes after the INT2 fix, unless
marked "control" (legitimate behaviour that must keep working). Everything is synthetic: DB targets end in .invalid,
credentials are sentinel strings, the only sockets are this test's own listeners on loopback inside the network
namespace or in tmp_path. Nothing here connects to a database, a host socket or the network.

  F-1  raw psycopg2 entry points (psycopg2._connect, _psycopg._connect, SQLAlchemy creator=) refused; connector lint
       flags them; an unsanctioned launch and a visible Postgres socket fail closed; the runner pins argv/env/masks.
  F-2  persisting raw unwrap / importlib.reload(psycopg2) / listener removal / spawn unwrap reported and re-wrapped.
  F-3  pure-Python _socket.socket.connect / sendto and ctypes symbol lookups refused (audit hook).
  F-4  scripts/vera_test_runner.py refuses --noconftest, a planted -c ini, tests/pytest.ini, paths outside tests/.
  F-5  multiprocessing stdlib children under -I get the boundary explicitly (see also
       test_env_guard.py::test_multiprocessing_spawn_child_has_boundary, the end-to-end proof).
  F-6  the boundary snapshot (_ORIG_ENV) and stash keep no credential values.
  LOW  importlib.reload(boundary) keeps the seal and is itself reported (LAST in this file: on the base it resets
       the boundary module for the rest of the session).
"""
import ast
import importlib
import importlib.util
import os
import socket
import sys
import types
from pathlib import Path

import pytest

bt = sys.modules["tests.vera_test_boundary"]
REPO = Path(__file__).resolve().parent.parent
DSN = "host=db-sentinel.invalid port=5432 dbname=sentinel_db user=sentinel_user connect_timeout=1"
DSN_TEST = "host=db-sentinel-test.invalid port=5432 dbname=sentinel_test_db user=sentinel_tester"


class _FakeCfg:
    class pluginmanager:
        @staticmethod
        def getplugin(name):
            return None


def _raise(text):
    raise SystemExit(text)


def _fresh_decision(monkeypatch, status, env):
    monkeypatch.setitem(bt.STATE, "netns_decided", False)
    monkeypatch.setitem(bt.STATE, "netns", bt.STATE["netns"])
    monkeypatch.setitem(bt.STATE, "unix_check", bt.STATE["unix_check"])
    monkeypatch.setitem(bt.STATE, "config", bt.STATE["config"])
    monkeypatch.setitem(bt.STATE, "bootstrap_dir", None)          # never remove the real session's bootstrap
    monkeypatch.setattr(bt, "netns_status", lambda *a, **k: status)
    monkeypatch.setattr(bt, "_ORIG_ENV", dict(env))
    monkeypatch.setattr(bt, "_isolation_fail", lambda cfg, text: _raise(text))
    monkeypatch.setattr(bt, "_install_filter_or_fail", lambda cfg: None)


def _keep_tamper(monkeypatch):
    monkeypatch.setitem(bt.STATE, "tamper_count", bt.STATE["tamper_count"])


def _runner():
    spec = importlib.util.spec_from_file_location("vera_test_runner_under_test", REPO / "scripts" / "vera_test_runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================== F-1 raw DB driver entry points
@pytest.mark.boundary_probe
@pytest.mark.parametrize("site", [("psycopg2", "_connect"), ("psycopg2._psycopg", "_connect")])
def test_raw_psycopg2_entry_point_is_refused(site):
    fn = getattr(importlib.import_module(site[0]), site[1])        # table-driven: the lint below flags literal calls
    before = len(bt.STATE["violations"])
    with pytest.raises(bt.DBBoundaryError):
        fn(DSN)
    assert [v["kind"] for v in bt.STATE["violations"][before:]] == ["db"]


@pytest.mark.boundary_probe
def test_sqlalchemy_creator_with_raw_entry_point_is_refused():
    import psycopg2
    from sqlalchemy import create_engine
    raw = getattr(psycopg2, "_connect")
    eng = create_engine("postgresql+psycopg2://", creator=lambda: raw(DSN))
    with pytest.raises(bt.DBBoundaryError):
        eng.connect()


def test_raw_entry_point_runs_only_inside_a_checked_connect(monkeypatch):
    """control + F-1: the depth rule, on a fake driver (no libpq, no socket)."""
    fake = types.ModuleType("fake_pg_int2")
    calls = []
    fake.raw = lambda dsn, **k: calls.append(dsn) or "conn"          # stands in for psycopg2._connect
    fake.connect = lambda dsn, **k: fake.raw(dsn)                     # check() adds hostaddr/passfile kwargs
    bt._wrap_raw_driver(fake, "raw")
    bt._wrap_driver(fake, "connect")
    monkeypatch.setattr(bt, "_orig_getaddrinfo", lambda host, port=None: [(2, 1, 6, "", ("127.0.0.1", 0))])
    monkeypatch.setenv("VERA_TEST_DB_URL", "postgresql://sentinel_tester:sentinel_pw@db-sentinel-test.invalid:5432/sentinel_test_db")
    monkeypatch.setenv("VERA_TEST_DB_DISPOSABLE", "1")
    monkeypatch.setitem(bt.STATE, "live_enabled", True)
    monkeypatch.setitem(bt.STATE, "deny_all", False)
    monkeypatch.setitem(bt._SEALED, "deny_all", False)
    bt.begin_item(live_item=True)
    try:
        # Owner decision 2026-10-19 (DISPOSABLE-VERIFY-DISABLED-pending-isolation-review): decide() is always False, so
        # the formerly allowed disposable target is refused by check() before the raw entry point is reached.
        with pytest.raises(bt.DBBoundaryError):
            fake.connect(DSN_TEST)
        assert calls == [] and bt.STATE["violations"][-1]["kind"] == "db"
        with pytest.raises(bt.DBBoundaryError):
            fake.raw(DSN_TEST)                                       # direct raw call: refused as well
        assert calls == []
        # The depth rule itself (raw runs only from inside a checked connect) on a stubbed check() that admits the
        # call; the real check() above refuses. Fake driver only: no libpq, no socket.
        monkeypatch.setattr(bt, "check", lambda cargs=(), cparams=None: None)
        assert fake.connect(DSN_TEST) == "conn" and calls == [DSN_TEST]
        with pytest.raises(bt.DBBoundaryError):
            fake.raw(DSN_TEST)                                       # outside a checked call: still refused
    finally:
        bt.begin_item(live_item=False)
    assert calls == [DSN_TEST]


def test_session_wraps_the_raw_entry_points():
    import psycopg2
    import psycopg2._psycopg as _pg
    assert getattr(getattr(psycopg2, "_connect"), "_vera_boundary", False)
    assert getattr(getattr(_pg, "_connect"), "_vera_boundary", False)


def test_connector_lint_flags_raw_entry_points(tmp_path):
    from tests.test_env_guard import _code_token_hits
    p = tmp_path / "t.py"
    p.write_text("import psycopg2\nfrom psycopg2 import extensions as _ext\npsycopg2._connect('x')\n"
                 "psycopg2.extensions.connection('x')\n_ext.connection('x')\n")
    assert {h[1] for h in _code_token_hits(p)} >= {"psycopg2._connect", "psycopg2.extensions.connection",
                                                   "_ext.connection"}


# ============================================================================== F-1 launch + kernel facts
def test_unsanctioned_launch_is_refused(monkeypatch):
    _fresh_decision(monkeypatch, "isolated", {"PATH": "/usr/bin:/bin"})
    with pytest.raises(SystemExit) as ei:
        bt.maybe_reexec(_FakeCfg, False)
    assert ei.value.code == bt.RUNNER_MESSAGE and "scripts/vera_test_runner.py" in bt.RUNNER_MESSAGE


def test_unsanctioned_launch_is_refused_before_any_reexec(monkeypatch):
    _fresh_decision(monkeypatch, "not_isolated", {"PATH": "/usr/bin:/bin"})
    monkeypatch.setattr(os, "execve", lambda *a: pytest.fail("must not re-exec an unsanctioned launch"))
    with pytest.raises(SystemExit) as ei:
        bt.maybe_reexec(_FakeCfg, False)
    assert ei.value.code == bt.RUNNER_MESSAGE


@pytest.mark.parametrize("env", [{"VERA_TEST_RUNNER": "1"}, {"VERA_TEST_NETNS_REEXEC": "1"},
                                 {"VERA_TEST_OFFLINE_ESCAPE": "1"}, {"VERA_GATE_NONCE": "0123456789abcdef" * 2}])
def test_sanctioned_launches(env):
    assert bt.launch_sanctioned(env)


@pytest.mark.parametrize("env", [{}, {"VERA_TEST_RUNNER": "yes"}, {"VERA_GATE_NONCE": "x" * 32},
                                 {"VERA_GATE_NONCE": "0123456789ABCDEF" * 2}, {"VERA_TEST_OFFLINE_ESCAPE": "0"}])
def test_unsanctioned_launch_signals(env):
    assert not bt.launch_sanctioned(env)


def test_child_of_a_sanctioned_session_counts_only_as_a_real_child(monkeypatch):
    monkeypatch.setitem(bt.STATE, "is_child", False)
    monkeypatch.setitem(bt.STATE, "bootstrap_reused", False)
    assert not bt.launch_sanctioned({bt.MARKER: "1"})          # the marker alone (outside a child) is not enough
    monkeypatch.setitem(bt.STATE, "is_child", True)
    assert not bt.launch_sanctioned({bt.MARKER: "1"})          # nor a marker without the parent's bootstrap dir
    monkeypatch.setitem(bt.STATE, "bootstrap_reused", True)
    assert bt.launch_sanctioned({bt.MARKER: "1"})


def test_live_session_is_not_runner_gated(monkeypatch):
    """control: --run-live-stages (owner opt-in) keeps its own path."""
    _fresh_decision(monkeypatch, "not_isolated", {"PATH": "/usr/bin:/bin"})
    monkeypatch.setattr(os, "execve", lambda *a: pytest.fail("a live session must not re-exec"))
    bt.maybe_reexec(_FakeCfg, True)
    assert bt.STATE["netns"] == "live-session"


def _pg_socket(monkeypatch, d):
    d.mkdir()
    monkeypatch.chdir(d)                                   # relative bind: no sun_path length limit
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(".s.PGSQL.5432")                                # bound, never listened on or connected to
    return s


def test_visible_postgres_socket_fails_closed(monkeypatch, tmp_path):
    d = tmp_path / "pg"
    s = _pg_socket(monkeypatch, d)
    try:
        (d / ".s.PGSQL.5432.lock").write_text("")
        assert bt.db_socket_files((str(d),)) == [str(d / ".s.PGSQL.5432")]
        monkeypatch.setattr(bt, "DB_SOCKET_SCAN_DIRS", (str(d),))
        _fresh_decision(monkeypatch, "isolated", {"VERA_TEST_RUNNER": "1"})
        with pytest.raises(SystemExit) as ei:
            bt.maybe_reexec(_FakeCfg, False)
        assert ei.value.code == bt.DB_SOCKET_MESSAGE
        files, dirs = bt.escape_lists(1001, "someone")      # the re-exec chain binds /dev/null over a loose socket
        assert str(d / ".s.PGSQL.5432") in files
    finally:
        s.close()


def test_reexec_chain_masks_postgres_socket_dirs():
    files, dirs = bt.escape_lists(1001, "someone")
    assert "/run/postgresql" in dirs and "/var/run/postgresql" in dirs
    argv = bt.reexec_argv(["python"], 1001, 1002, "someone")
    assert "/var/run/postgresql" in argv[argv.index("--dirs"):argv.index("--cmd")]


def test_no_postgres_socket_visible_in_this_session():
    assert bt.db_socket_files() == []


@pytest.mark.skipif(os.environ.get("VERA_TEST_RUNNER") != "1", reason="only under scripts/vera_test_runner.py")
def test_runner_session_attestation():
    """Under the runner: the launch environment (/proc/self/environ) holds only the allowlist, -I and the pycache
    prefix are active, no Postgres socket and no credential file of the host is visible."""
    raw = Path("/proc/self/environ").read_bytes().split(b"\0")
    names = {e.split(b"=", 1)[0].decode() for e in raw if e}
    extra = set()  # DEV-A: allowed only when the runner bound the fixed read-only bundle (VERA_TEST_RO_BUNDLE=1 at launch)
    if "VERA_REAL_DAR9_BUNDLE" in names:
        assert os.environ["VERA_REAL_DAR9_BUNDLE"] == str(REPO.parent / "work" / "fd-1020" / "bundle")
        extra = {"VERA_REAL_DAR9_BUNDLE"}
    assert names <= {"PATH", "HOME", "LANG", "TMPDIR", "TEMP", "TMP", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTHONDONTWRITEBYTECODE",
                     "PYTHONPYCACHEPREFIX", "VERA_TEST_RUNNER", "VERA_GATE_NONCE", "USER"} | extra, sorted(names)
    assert sys.flags.isolated and sys.pycache_prefix == "/tmp/vera-pycache"
    assert not os.path.exists("/var/run/postgresql/.s.PGSQL.5432") and not os.path.exists("/run/postgresql")
    assert not any(n.startswith(".s.PGSQL.") for n in os.listdir("/tmp"))
    real_home = os.path.expanduser("~" + os.environ.get("USER", "nobody"))
    for f in (".pgpass", ".bashrc", ".ssh", ".codex", ".gitconfig"):
        assert not os.path.exists(os.path.join(real_home, f)), f
    assert all(Path(p).stat().st_size == 0 for p in (REPO / ".env",) if p.exists())

    # orchestrator WSL item: no Windows drive or other host-shared mount, no interop route (visibility checks only;
    # no Windows binary is executed). The WSLInterop binfmt handler is kernel-global (residual): what is checked is
    # that nothing it could launch is reachable and that its interop channel is gone.
    assert not os.path.exists("/mnt/c") and os.listdir("/mnt") == []
    assert bt.foreign_mounts(visible_only=True) == [], bt.foreign_mounts(visible_only=True)
    assert not [n for n in names if n.upper().startswith("WSL")] and not os.path.exists("/run/WSL")
    binfmt = Path("/proc/sys/fs/binfmt_misc/WSLInterop")
    assert not binfmt.exists() or binfmt.read_text().strip() == ""
    assert bt.wsl_interop_open() is False


# ============================================================================== F-2 tamper detection
@pytest.mark.parametrize("how", ["raw_unwrap", "reload_psycopg2", "listener_removed", "raw_entry_unwrap",
                                 "spawn_unwrap", "dotenv_restored"])
def test_persisting_unwrap_is_reported_and_rewrapped(monkeypatch, how):
    import dotenv
    import psycopg2
    from sqlalchemy import event
    from sqlalchemy.engine import Engine
    _keep_tamper(monkeypatch)
    assert bt.tamper_keys() == []
    if how == "raw_unwrap":
        psycopg2.connect, key = psycopg2.connect._vera_original, "wrapper:psycopg2.connect"
    elif how == "reload_psycopg2":
        importlib.reload(psycopg2)
        key = "wrapper:psycopg2.connect"
    elif how == "listener_removed":
        event.remove(Engine, "do_connect", bt.STATE["db_listener"] or bt._on_do_connect)
        key = "wrapper:sqlalchemy.do_connect_listener"
    elif how == "raw_entry_unwrap":
        setattr(psycopg2, "_connect", getattr(psycopg2, "_connect")._vera_original)
        key = "wrapper:psycopg2._connect"
    elif how == "spawn_unwrap":
        os.system, key = os.system._vera_original, "wrapper:os.system"
    else:
        dotenv.load_dotenv, key = bt._ORIG["load"], "wrapper:dotenv.load_dotenv"
    assert key in bt.tamper_keys()
    msg = bt.tamper_message()
    assert msg and key.split(":", 1)[1] in msg
    assert bt.tamper_keys() == []                               # fail closed: re-wrapped and re-sealed
    assert getattr(psycopg2.connect, "_vera_boundary", False) and getattr(os.system, "_vera_boundary", False)
    assert event.contains(Engine, "do_connect", bt.STATE["db_listener"])
    assert dotenv.load_dotenv is bt._noop_load_dotenv


@pytest.mark.boundary_probe
def test_db_still_refused_after_rewrap():
    from sqlalchemy import create_engine
    with pytest.raises(bt.DBBoundaryError):
        create_engine("postgresql+psycopg2://u:sentinel_pw@db-sentinel.invalid:5432/x").connect()


# ============================================================================== F-3 audit hook
@pytest.mark.boundary_probe
def test_raw_base_class_socket_calls_are_refused():
    import _socket
    srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]
    socks = []
    try:
        c = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        socks.append(c)
        with pytest.raises(bt.NetworkBoundaryError):
            c.connect(("127.0.0.1", port))                       # raw base class (verifier A7)
        c2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socks.append(c2)
        with pytest.raises(bt.NetworkBoundaryError):
            _socket.socket.connect(c2, ("127.0.0.1", port))      # pure-Python call of the C method (verifier A7)
        with pytest.raises(bt.NetworkBoundaryError):
            _socket.socket.connect_ex(c2, ("127.0.0.1", port))
        u = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        socks.append(u)
        with pytest.raises(bt.NetworkBoundaryError):
            u.sendto(b"x", ("127.0.0.1", port))
        with pytest.raises(socket.gaierror):
            _socket.getaddrinfo("example.org", 443)              # raw resolver, non-loopback name
    finally:
        for s in socks + [srv]:
            s.close()


@pytest.mark.vera_sockets
def test_raw_socket_to_own_listener_allowed_on_request():
    """control: the hook applies the same decision as the wrappers (OA-2 opt-in still works)."""
    import _socket
    srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    c = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    try:
        c.connect(srv.getsockname())
    finally:
        c.close()
        srv.close()


@pytest.mark.boundary_probe
@pytest.mark.parametrize("name", ["connect", "socket", "system", "execve", "syscall", "PQconnectdb"])
def test_ctypes_lookup_of_network_and_exec_symbols_is_refused(name):
    import ctypes
    libc = ctypes.CDLL(None)                                     # fresh handle: no cached function object
    with pytest.raises(bt.NetworkBoundaryError):
        getattr(libc, name)


def test_ctypes_harmless_symbol_still_resolves():
    """control"""
    import ctypes
    assert ctypes.CDLL(None).getpid() == os.getpid()


def test_audit_hook_installed():
    assert bt.STATE["audit_hook"] is True


def test_kernel_layer_probe_call_sites_are_pinned():
    pinned = {("test_env_guard.py", f) for f in ("test_raw_socket_libpq_and_ctypes_cannot_reach_the_network",
                                                 "test_native_windows_exec_blocked", "test_git_fsmonitor_config_is_overridden",
                                                 "_shell_status", "test_io_uring_setup_refused")}
    found = set()
    for p in sorted((REPO / "tests").glob("*.py")):
        if p.name == "vera_test_boundary.py":
            continue
        for fn in ast.walk(ast.parse(p.read_text())):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for n in ast.walk(fn):
                    if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("kernel_layer_probe"):
                        found.add((p.name, fn.name))
    assert found == pinned


# ============================================================================== F-4 sanctioned runner
def test_runner_refuses_bypass_arguments():
    R = _runner()
    root = str(REPO)
    for args in (["--noconftest", "tests"], ["-c", "tests/evil.ini", "tests"], ["-c=tests/evil.ini", "tests"],
                 ["--rootdir", "/tmp", "tests"], ["--confcutdir", "tests", "tests"], ["-p", "no:conftest", "tests"],
                 ["-p", "tests.evil", "tests"], ["-o", "addopts=--noconftest", "tests"], ["--override-ini=x=y", "tests"],
                 ["--run-live-stages", "tests"], ["--pyargs", "tests"], ["db/test_x.py"], ["scripts"],
                 ["tests/../db/test_x.py"], ["vera"], [], ["--basetemp", str(REPO), "tests"],
                 ["--ignore=/etc", "tests"]):
        _, bad = R.validate_args(args, root)
        assert bad, args


def test_runner_accepts_the_gate_argv_and_common_options():
    """control"""
    R = _runner()
    root = str(REPO)
    gate = ["-q", "tests", "--ignore=tests/test_m2_integration.py"]       # INT3 MED-C: the gate-wiring proposal argv
    out, bad = R.validate_args(gate, root)
    assert bad == [] and "tests" in out and "-c" not in out and "--rootdir" not in out   # added by the runner only
    pinned = ["-q", "-c", f"{root}/pytest.ini", "--rootdir", root, "-p", "no:cacheprovider", "tests"]
    assert R.validate_args(pinned, root)[1]                                # INT3: now refused, even with pinned values
    out, bad = R.validate_args(["-q", "-rfE", "-k", "a and not b", "--basetemp", "/tmp/bt", "--tb=short",
                                "tests/test_env_guard.py::test_x"], root)
    assert bad == [] and out[:2] == ["-q", "-rfE"]


def test_runner_detects_planted_configuration(tmp_path):
    R = _runner()
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    for d in ("tests/sub", "vera"):
        (tmp_path / d).mkdir(parents=True)
    for f in ("conftest.py", "tests/conftest.py"):
        (tmp_path / f).write_text("")
    assert R.planted_configs(str(tmp_path)) == []
    planted = {"tests/pytest.ini": "[pytest]\naddopts = --noconftest\n", "tests/sub/tox.ini": "[pytest]\n",
               "sitecustomize.py": "", "vera/x.pth": "", "tests/sub/conftest.py": "", "tests/setup.cfg": "[tool:pytest]\n"}
    for f, text in planted.items():
        (tmp_path / f).write_text(text)
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts = '--noconftest'\n")
    found = R.planted_configs(str(tmp_path))
    assert set(planted) <= set(found) and "pyproject.toml [tool.pytest*]" in found
    (tmp_path / "pytest.ini").unlink()
    assert any("missing" in f for f in R.planted_configs(str(tmp_path)))


def test_runner_environment_is_allowlisted():
    R = _runner()
    env = R.sandbox_env({"PATH": "/evil", "VERA_DB_URL_RO": "postgresql://u:sentinel_pw@db-sentinel.invalid/x",
                         "PGPASSWORD": "sentinel_pw", "OPENAI_API_KEY": "GUARD-SENTINEL", "LD_PRELOAD": "/x.so",
                         "VERA_GATE_NONCE": "0123456789abcdef" * 2, "USER": "someone"})
    assert set(env) == {"PATH", "HOME", "LANG", "TMPDIR", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTHONDONTWRITEBYTECODE",
                        "PYTHONPYCACHEPREFIX", "VERA_TEST_RUNNER", "VERA_GATE_NONCE", "USER"}
    assert env["PATH"] == "/usr/bin:/bin" and env["VERA_TEST_RUNNER"] == "1"
    assert "VERA_GATE_NONCE" not in R.sandbox_env({"VERA_GATE_NONCE": "../../x"})
    assert R.launch_refusals({"LD_PRELOAD": "/x.so"}, True) and R.launch_refusals({}, False)
    assert R.launch_refusals({}, True) == []


def test_runner_sandbox_script_masks_sockets_home_and_credentials():
    R = _runner()
    script = R.sandbox_script("/home/u/repo", "/home/u/repo/.venv", [("/home/u/main/.git/worktrees/w", True)],
                              ["/home/u/repo/.env"], 1000, 1000, "/home/u/repo/.venv/bin/python", ["tests"],
                              R.sandbox_env({}), path_exists=lambda p: True, realpath=lambda p: p)
    for needle in ("tmpfs /home", "tmpfs /tmp", "tmpfs /run", "tmpfs /dev/shm", "tmpfs /var/tmp",
                   "tmpfs /var/run/postgresql", "--bind /dev/null /dev/vsock", "--bind /dev/null /home/u/repo/.env",
                   "remount,bind,ro /home/u/repo/.venv", "/usr/bin/env -i", "-c /home/u/repo/pytest.ini",
                   "--rootdir /home/u/repo", "pycache_prefix=/tmp/vera-pycache", "-p no:cacheprovider",
                   "--map-user=1000", "VERA_TEST_RUNNER=1"):
        assert needle in script, needle
    assert script.index("tmpfs /home") > script.index("--bind /home/u/repo ")          # staged before /home is hidden
    assert script.index("--move") > script.index("tmpfs /home")
    sym = R.sandbox_script("/home/u/repo", "/home/u/v", [], [], 1000, 1000, "/home/u/v/bin/python", ["tests"],
                           R.sandbox_env({}), path_exists=lambda p: True,
                           realpath=lambda p: "/run/postgresql" if p == "/var/run/postgresql" else p)
    assert "tmpfs /var/run/postgresql" not in sym and "tmpfs /run " in sym              # via /var/run -> /run
    argv = R.unshare_argv(script)
    assert argv[0] == "/usr/bin/unshare" and {"--net", "--mount", "--user", "--pid", "--ipc", "--kill-child"} <= set(argv)
    with pytest.raises(ValueError):
        R.sandbox_script("/tmp/repo", "/home/u/v", [], [], 1000, 1000, "/x", ["tests"], {})


def test_runner_plan_refuses_and_runs_without_executing():
    R = _runner()
    kind, why = R.plan(["--", "--noconftest", "tests"], {}, root=str(REPO), isolated=True)
    assert kind == "refuse" and any("--noconftest" in w for w in why)
    kind, why = R.plan(["--", "tests"], {}, root=str(REPO), isolated=False)
    assert kind == "refuse" and any("python -I" in w for w in why)


def test_runner_does_not_live_in_a_collected_or_imported_package():
    """control (owner invariant, operational half): the runner is a script; tests/__init__.py stays empty."""
    assert (REPO / "tests" / "__init__.py").read_text() == ""
    src = (REPO / "scripts" / "vera_test_runner.py").read_text()
    tree = ast.parse(src)
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert imported <= {"__future__", "fnmatch", "os", "re", "secrets", "shlex", "shutil", "stat", "subprocess", "sys"}


MOUNTINFO = (
    "22 1 8:1 / / rw - ext4 /dev/sdc rw\n"
    "40 22 0:50 / /mnt/c rw,noatime - 9p drvfs rw,aname=drvfs;path=C:\\\n"
    "41 22 0:51 / /usr/lib/wsl/drivers ro - 9p drivers ro\n"
    "42 22 0:52 / /home/u/work/web rw - 9p web rw\n"
    "43 22 0:53 / /srv/win\\040share rw - drvfs x rw\n"
    "44 22 0:54 / /run/user/1000 rw - tmpfs tmpfs rw\n")


def test_runner_hides_windows_drives_and_host_shared_mounts():
    R = _runner()
    found = R.foreign_mounts(MOUNTINFO)
    assert found == ["/home/u/work/web", "/mnt/c", "/srv/win share", "/usr/lib/wsl/drivers"]
    masks, bad = R.foreign_masks(found, ["/home/u/repo", "/home/u/repo/.venv"])
    assert masks == ["/srv/win share", "/usr/lib/wsl/drivers"] and bad == []     # /mnt/c, /home/...: under tmpfs
    _, bad = R.foreign_masks(found, ["/home/u/work/web/repo"])
    assert bad and "9p" in bad[0]                                                  # a checkout on a Windows share
    script = R.sandbox_script("/home/u/repo", "/home/u/v", [], [], 1000, 1000, "/home/u/v/bin/python", ["tests"],
                              R.sandbox_env({"WSL_INTEROP": "/run/WSL/1_interop", "WSL_DISTRO_NAME": "x"}),
                              path_exists=lambda p: True, realpath=lambda p: p, foreign=masks)
    for needle in ("tmpfs /usr/lib/wsl/drivers", "tmpfs '/srv/win share'", "-t tmpfs -o mode=0755 tmpfs /mnt",
                   "--bind /dev/null /proc/sys/fs/binfmt_misc/WSLInterop", "unset WSL_INTEROP WSL_DISTRO_NAME WSLENV"):
        assert needle in script, needle
    assert "WSL_INTEROP=" not in script and "WSL_DISTRO_NAME=" not in script


def test_reexec_chain_hides_mnt_and_host_shared_mounts(monkeypatch):
    parse = bt.foreign_mounts                                   # the real parser, before it is faked below
    monkeypatch.setattr(bt, "foreign_mounts", lambda *a, **k: ["/mnt/c", "/usr/lib/wsl/drivers", "/x/web",
                                                              str(bt.REPO.parent)])
    files, dirs = bt.escape_lists(1001, "someone")
    assert "/mnt" in dirs and "/usr/lib/wsl/drivers" in dirs and "/x/web" in dirs and "/mnt/c" not in dirs
    assert str(bt.REPO.parent) not in dirs                        # never hide the checkout itself
    assert parse(lambda: MOUNTINFO) == ["/home/u/work/web", "/mnt/c", "/srv/win share", "/usr/lib/wsl/drivers"]
    covered = MOUNTINFO + "45 22 0:60 / /mnt rw - tmpfs tmpfs rw\n46 22 0:61 / /usr/lib/wsl/drivers rw - tmpfs tmpfs rw\n"
    assert parse(lambda: covered, visible_only=True) == ["/home/u/work/web", "/srv/win share"]


# ============================================================================== F-5 multiprocessing under -I
def test_mp_child_argv_rewrites_only_the_stdlib_child_shapes():
    exe = sys.executable
    rt = [exe, "-B", "-I", "-c", "from multiprocessing.resource_tracker import main;main(7)"]
    new = bt.mp_child_argv(rt)
    assert new[:4] == rt[:4] and new[4].endswith(rt[4]) and ".install()" in new[4] and "_exit(98)" in new[4]
    sp = [exe, "-B", "-I", "-X", "utf8", "-c",
          "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5, pipe_handle=9)", "--multiprocessing-fork"]
    assert bt.mp_child_argv(sp)[-1] == "--multiprocessing-fork"
    for bad in ([exe, "-I", "-c", "import os; os.system('true')"],
                [exe, "-I", "-c", sp[6] + "; import os"],
                [exe, "-I", "-c", rt[4], "extra"],
                ["/bin/sh", "-I", "-c", rt[4]],
                [exe, "-B", "-c", rt[4]],                                # not isolated: sitecustomize does the job
                [exe, "-I", "--evil", "-c", rt[4]]):
        assert bt.mp_child_argv(bad) is None, bad


@pytest.mark.boundary_probe
def test_isolated_generic_python_child_still_refused():
    import multiprocessing.util as mpu
    with pytest.raises(bt.ChildProcessBoundaryError):
        mpu.spawnv_passfds(sys.executable, [sys.executable, "-I", "-c", "pass"], [])


# ============================================================================== F-6 credentials in the snapshot
def test_snapshot_and_stash_drop_credentials(monkeypatch):
    monkeypatch.setattr(bt, "_ORIG_ENV", {"PATH": "/x", "PGPASSWORD": "sentinel_pw", "USER": "u",
                                          "VERA_DB_URL_RO": "postgresql://u:sentinel_pw@db-sentinel.invalid/x"})
    monkeypatch.setattr(bt, "_STASH", {"VERA_DB_URL_RO": "postgresql://u:sentinel_pw@db-sentinel.invalid/x",
                                       "OPENAI_API_KEY": "GUARD-SENTINEL"})
    gone = bt.drop_credentials_from_snapshot()
    assert gone == ["PGPASSWORD", "VERA_DB_URL_RO"]
    assert bt._ORIG_ENV == {"PATH": "/x", "USER": "u"} and bt._STASH == {"OPENAI_API_KEY": "GUARD-SENTINEL"}


def test_session_snapshot_holds_no_guarded_names():
    assert [k for k in bt._ORIG_ENV if bt.guarded(k)] == []
    assert [k for k in bt._STASH if not bt.PROVIDER_KEY.search(k)] == []
    src = (REPO / "tests" / "vera_test_boundary.py").read_text()
    body = src[src.index("def install()"):src.index("def drop_credentials_from_snapshot")]
    assert "drop_credentials_from_snapshot()" in body


# ============================================================================== LOW: boundary reload (keep LAST)
def test_boundary_reload_keeps_the_seal_and_is_reported(monkeypatch):
    _keep_tamper(monkeypatch)
    monkeypatch.setitem(bt.STATE, "reloads", bt.STATE.get("reloads", 0))
    monkeypatch.setitem(bt._SEALED, "reloads", bt._SEALED.get("reloads", 0))
    state, sealed, before = bt.STATE, bt._SEALED, dict(bt._SEALED)
    importlib.reload(bt)
    assert bt.STATE is state and bt._SEALED is sealed and bt.STATE["installed"] is True
    assert bt._SEALED["deny_all"] == before["deny_all"] and bt._SEALED["session"] == before["session"]
    assert bt.STATE["tamper_count"] >= 1                         # the session fails closed (restored by monkeypatch)
    assert "boundary_reloaded" in bt.tamper_keys()
    assert bt.tamper_message() and bt.tamper_keys() == []
