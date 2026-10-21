"""K18W INT3 regression tests for the WS-8g findings MED-A, MED-B and MED-C (each FAILS on INT2, PASSES on INT3).

MED-A: OFFLINE_ESCAPE, the runner marker, a gate nonce, the old re-exec marker and the child marker are honoured only
       when the session is ALREADY isolated (network, sockets, credential files); otherwise the session is refused
       and nothing is re-exec'd. The refusal message no longer advertises the escape.
MED-B: Postgres sockets outside the fixed directories (the checkout, /var/lib/postgresql) are detected in-boundary,
       and the runner sweeps the whole visible tree for socket files of ANY name (and unlistable directories).
MED-C: the runner refuses '@' argument files, path arguments through a symlink, -c/--rootdir/-p/--noconftest/
       --confcutdir outright, and planted symlinks in the checkout.
Nothing here opens a DB connection; the only sockets are bound locally in tmp_path and never listened on.
"""
from __future__ import annotations

import importlib.util
import os
import socket
import stat
import sys
from pathlib import Path

import pytest

bt = sys.modules["tests.vera_test_boundary"]
REPO = Path(__file__).resolve().parent.parent
FAKE_PGPASS = "localhost:5432:*:*:int3-sentinel-not-a-secret\n"


class _FakeCfg:
    class pluginmanager:
        @staticmethod
        def getplugin(name):
            return None


class _Exec(Exception):
    pass


def _raise(text):
    raise SystemExit(text)


def _runner():
    spec = importlib.util.spec_from_file_location("vera_test_runner_int3", REPO / "scripts" / "vera_test_runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _decide(monkeypatch, status, env, execs):
    monkeypatch.setitem(bt.STATE, "netns_decided", False)
    monkeypatch.setitem(bt.STATE, "netns", bt.STATE["netns"])
    monkeypatch.setitem(bt.STATE, "unix_check", bt.STATE["unix_check"])
    monkeypatch.setitem(bt.STATE, "config", bt.STATE["config"])
    monkeypatch.setitem(bt.STATE, "bootstrap_dir", None)          # never remove the real session's bootstrap
    monkeypatch.setattr(bt, "netns_status", lambda *a, **k: status)
    monkeypatch.setattr(bt, "_ORIG_ENV", dict(env))
    monkeypatch.setattr(bt, "_isolation_fail", lambda cfg, text: _raise(text))
    monkeypatch.setattr(bt, "_install_filter_or_fail", lambda cfg: None)

    def spy(*a):
        execs.append(a[0] if a else None)
        raise _Exec

    monkeypatch.setattr(os, "execve", spy)


SIGNALS = {"runner": {"VERA_TEST_RUNNER": "1"}, "escape": {"VERA_TEST_OFFLINE_ESCAPE": "1"},
           "nonce": {"VERA_GATE_NONCE": "0123456789abcdef" * 2}, "reexec-marker": {"VERA_TEST_NETNS_REEXEC": "1"},
           "child": {"VERA_TEST_BOUNDARY_CHILD": "1"}}


# ============================================================================== MED-A
@pytest.mark.parametrize("signal", sorted(SIGNALS))
def test_int3_signal_on_a_non_isolated_session_is_refused_not_reexecd(monkeypatch, signal):
    env = {"PATH": "/usr/bin:/bin", **SIGNALS[signal]}
    if signal == "child":
        env = {"PATH": "/usr/bin:/bin", bt.MARKER: "1"}
        monkeypatch.setitem(bt.STATE, "is_child", True)
        monkeypatch.setitem(bt.STATE, "bootstrap_reused", True)
    assert bt.launch_sanctioned(env)                                   # precondition: the signal is accepted as such
    execs = []
    _decide(monkeypatch, "not_isolated", env, execs)
    try:
        bt.maybe_reexec(_FakeCfg, False)
        outcome = "ran"
    except _Exec:
        outcome = "re-exec"
    except SystemExit as e:
        outcome = e.code
    assert execs == [], f"VULNERABLE: {signal} on a non-isolated session fell back to the re-exec chain"
    assert outcome == getattr(bt, "ISOLATION_REQUIRED_MESSAGE", "<missing>"), outcome


def test_int3_refusal_messages_do_not_advertise_the_escape():
    for name in ("RUNNER_MESSAGE", "ISOLATION_REQUIRED_MESSAGE", "CREDENTIAL_MESSAGE", "DB_SOCKET_MESSAGE"):
        text = getattr(bt, name, "")
        assert text, name
        assert bt.OFFLINE_ESCAPE not in text and "OFFLINE" not in text.upper(), name


def _fake_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".pgpass").write_text(FAKE_PGPASS)
    (home / ".pgpass").chmod(0o600)
    return home


def test_int3_readable_credential_file_refuses_an_isolated_session(monkeypatch, tmp_path):
    home = _fake_home(tmp_path)
    execs = []
    _decide(monkeypatch, "isolated", {"PATH": "/usr/bin:/bin", "HOME": str(home), "VERA_TEST_RUNNER": "1"}, execs)
    monkeypatch.setattr(bt, "DB_SOCKET_SCAN_DIRS", ())
    monkeypatch.setattr(bt, "escape_lists", lambda uid, user: ([], []))
    monkeypatch.setattr(bt, "escape_capable", lambda *a, **k: False)
    try:
        bt.maybe_reexec(_FakeCfg, False)
        outcome = "ran"
    except SystemExit as e:
        outcome = e.code
    assert outcome == getattr(bt, "CREDENTIAL_MESSAGE", "<missing>"), \
        "VULNERABLE: a readable ~/.pgpass did not stop the escape/runner signal"
    assert str(home) not in str(outcome)                               # the message names no path


def test_int3_credential_visibility_pure_cases(tmp_path):
    home = _fake_home(tmp_path)
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / ".env.example").write_text("X=1\n")                         # examples are never credentials
    (repo / ".venv").mkdir()
    (repo / ".venv" / "a.pem").write_text("x")                          # the pinned venv: skipped (INT4)
    pv = str(repo / ".venv")
    vis = bt.credential_files_visible(homes=[str(home)], repo=str(repo), venv=pv)
    assert vis == [str(home / ".pgpass")]
    (repo / "tests" / "k.pem").write_text("fake")
    (repo / ".env").symlink_to("/dev/null")                            # the runner's mask shape: not visible
    assert bt.credential_files_visible(homes=[str(home)], repo=str(repo), venv=pv) == sorted([str(home / ".pgpass"),
                                                                                   str(repo / "tests" / "k.pem")])
    (home / ".pgpass").write_text("")                                    # empty: not a credential
    (repo / "tests" / "k.pem").unlink()
    assert bt.credential_files_visible(homes=[str(home)], repo=str(repo), venv=pv) == []


def test_int3_isolation_problems_name_the_failed_check(monkeypatch, tmp_path):
    monkeypatch.setattr(bt, "netns_status", lambda *a, **k: "not_isolated")
    assert bt.session_isolation_problems(1001, "someone") == ["network"]
    monkeypatch.setattr(bt, "netns_status", lambda *a, **k: "isolated")
    monkeypatch.setattr(bt, "db_socket_visible", lambda *a, **k: True)
    monkeypatch.setattr(bt, "escape_capable", lambda *a, **k: True)
    monkeypatch.setattr(bt, "credential_files_visible", lambda *a, **k: ["x"])
    assert bt.session_isolation_problems(1001, "someone") == ["db-socket", "escape-socket", "credential-file"]
    assert bt.session_isolation_problems(1001, "someone", live=True) == ["db-socket", "escape-socket"]


# ============================================================================== MED-B
def _bound_socket(monkeypatch, d, name=".s.PGSQL.5432"):
    d.mkdir(parents=True)
    monkeypatch.chdir(d)                                   # relative bind: no sun_path length limit
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(name)                                           # bound, never listened on or connected to
    return s


def test_int3_postgres_socket_in_the_checkout_fails_closed(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    s = _bound_socket(monkeypatch, repo / "deep" / "dir")
    try:
        execs = []
        _decide(monkeypatch, "isolated", {"PATH": "/usr/bin:/bin", "VERA_TEST_OFFLINE_ESCAPE": "1"}, execs)
        monkeypatch.setattr(bt, "REPO", repo)
        monkeypatch.setattr(bt, "DB_SOCKET_SCAN_DIRS", ())
        monkeypatch.setattr(bt, "escape_capable", lambda *a, **k: False)
        try:
            bt.maybe_reexec(_FakeCfg, False)
            outcome = "ran"
        except SystemExit as e:
            outcome = e.code
        assert outcome == bt.DB_SOCKET_MESSAGE, "VULNERABLE: a .s.PGSQL socket in the checkout was not detected"
    finally:
        s.close()


def test_int3_db_scan_covers_var_lib_postgresql():
    assert "/var/lib/postgresql" in bt.DB_SOCKET_SCAN_DIRS


def test_int3_runner_script_sweeps_every_socket_and_unlistable_dir():
    R = _runner()
    script = R.sandbox_script("/home/u/repo", "/home/u/repo/.venv", [], [], 1000, 1000, "/home/u/repo/.venv/bin/python",
                              ["tests"], R.sandbox_env({}), path_exists=lambda p: True, realpath=lambda p: p)
    assert getattr(R, "FIND", None) in R.LAUNCH_BINARIES
    assert "-type s -exec /bin/sh -c '/usr/bin/mount -n --bind /dev/null \"$1\"" in script   # any name, anywhere
    assert "-type s -exec /bin/sh -c '/usr/bin/mount -n -t tmpfs -o mode=0755 tmpfs \"${1%/*}\"'" in script  # pass 2
    assert "! -readable -o ! -executable" in script and "tmpfs {} \\;" in script         # unlistable dirs hidden
    assert 'fail "socket-sweep' in script
    assert script.index('fail "socket-sweep') < script.index("exec /usr/bin/unshare --user --map-user")


@pytest.mark.skipif(os.environ.get("VERA_TEST_RUNNER") != "1", reason="only under scripts/vera_test_runner.py")
def test_int3_runner_session_sees_no_socket_file():
    """Under the runner no PRE-EXISTING Unix socket file of any name is visible outside /proc and /sys (WS-8g GC2
    reached one in the checkout). The runner's fresh per-session tmpfs mounts (/tmp, /run, /dev/shm, /var/tmp,
    /var/crash) are skipped: sockets there were created by this session itself (tmp_path binds, multiprocessing).
    lstat only; nothing is opened. Reports counts per top-level directory, never a full path."""
    fresh = ("/tmp", "/run", "/dev/shm", "/var/tmp", "/var/crash")
    fresh += (os.path.realpath(os.environ["TMPDIR"]),)   # the runner's own per-run temp dir (R/work/test-tmp/r<hex>), created by this session
    found = {}
    for d, dirs, files in os.walk("/"):
        if d == "/":
            dirs[:] = [x for x in dirs if x not in ("proc", "sys")]
        dirs[:] = [x for x in dirs if os.path.join(d, x) not in fresh]
        for f in files + dirs:
            try:
                if stat.S_ISSOCK(os.lstat(os.path.join(d, f)).st_mode):
                    top = "/" + os.path.join(d, f).lstrip("/").split("/", 1)[0]
                    found[top] = found.get(top, 0) + 1
            except OSError:
                continue
    assert found == {}, f"VULNERABLE: socket file(s) visible under the runner, per top dir: {found}"


# ============================================================================== MED-C
def _fake_root(tmp_path):
    root = tmp_path / "root"
    (root / "tests" / "real").mkdir(parents=True)
    (root / "pytest.ini").write_text("[pytest]\n")
    (root / "tests" / "real" / "test_a.py").write_text("def test_a():\n    pass\n")
    (root / "tests" / "args.txt").write_text("--noconftest\n")
    return root


def test_int3_runner_refuses_argument_files(tmp_path):
    R = _runner()
    root = _fake_root(tmp_path)
    (root / "@tests").symlink_to("tests")                              # the WS-8g GE3 shape
    for args in (["@tests/args.txt"], ["tests", "@tests/args.txt"], ["-k", "@tests/args.txt", "tests"],
                 ["--deselect=@tests/args.txt", "tests"], ["--deselect", "@tests/args.txt", "tests"]):
        _, bad = R.validate_args(args, str(root))
        assert bad, f"VULNERABLE: argument file accepted: {args}"


def test_int3_runner_refuses_paths_through_a_symlink(tmp_path):
    R = _runner()
    root = _fake_root(tmp_path)
    (root / "tests" / "link").symlink_to("real")
    for args in (["tests/link/test_a.py"], ["tests/link"], ["tests", "--ignore=tests/link"]):
        _, bad = R.validate_args(args, str(root))
        assert bad, f"VULNERABLE: symlinked path accepted: {args}"
    out, bad = R.validate_args(["tests/real/test_a.py"], str(root))     # control
    assert bad == [] and out == ["tests/real/test_a.py"]


def test_int3_runner_refuses_override_options_even_with_pinned_values():
    R = _runner()
    root = str(REPO)
    for args in (["-p", "no:cacheprovider", "tests"], ["-pno:cacheprovider", "tests"],
                 ["-c", f"{root}/pytest.ini", "tests"], ["-c=pytest.ini", "tests"], ["--rootdir", root, "tests"],
                 ["--rootdir=" + root, "tests"], ["--noconftest", "tests"], ["--confcutdir=tests", "tests"]):
        _, bad = R.validate_args(args, root)
        assert bad, f"VULNERABLE: override accepted: {args}"
    out, bad = R.validate_args(["-q", "tests", "--ignore=tests/test_m2_integration.py"], root)   # control: gate argv
    assert bad == []


def test_int3_planted_configs_flags_symlinks(tmp_path):
    R = _runner()
    root = _fake_root(tmp_path)
    assert R.planted_configs(str(root)) == []                           # control
    (root / "@tests").symlink_to("tests")
    (root / "tests" / "lnk.py").symlink_to(tmp_path / "elsewhere.py")
    found = R.planted_configs(str(root))
    assert any("@tests" in f for f in found) and any("lnk.py" in f for f in found), found
