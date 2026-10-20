"""K18W WS-2: version-agnostic BEHAVIOURAL regressions (no new API names), so the same file shows the vulnerable
behaviour on the original candidate 120ab53 and the blocked behaviour on the integrated tree.

Runs only under the VERA-DAR test boundary in an isolated netns. Synthetic targets only (`.invalid`, sentinel credentials,
loopback/AF_UNIX inside the netns and tmp_path). Every raw write to boundary state is restored in `finally`.
"""
import os
import socket
import sys

import pytest

bt = sys.modules["tests.vera_test_boundary"]
SENT_TEST = "postgresql://sentinel_tester:sentinel_pw@db-sentinel-test.invalid:5432/sentinel_test_db"
CP_TEST = {"host": "db-sentinel-test.invalid", "port": 5432, "dbname": "sentinel_test_db", "user": "sentinel_tester"}


class _FakeCfg:
    class pluginmanager:
        @staticmethod
        def getplugin(name):
            return None


class _Stop(Exception):
    pass


@pytest.mark.boundary_probe
def test_constant_reset_of_deny_switch_does_not_allow_db(monkeypatch):
    """X-era fail-open (PF-18 M-2): a session that started fail-closed is 'reset' to deny_policy=ok/deny_all=False by
    a raw write. Secure behaviour: the DB target is still refused."""
    if bt.STATE["deny_policy"] == "ok":
        pytest.skip("session deny policy is ok here; the fail-open precondition (fail-closed start) is absent")
    monkeypatch.setattr(bt, "_orig_getaddrinfo", lambda host, port=None: [(2, 1, 6, "", ("127.0.0.1", 0))])
    monkeypatch.setenv("VERA_TEST_DB_URL", SENT_TEST)
    monkeypatch.setenv("VERA_TEST_DB_DISPOSABLE", "1")
    monkeypatch.setitem(bt.STATE, "live_enabled", True)
    saved = (bt.STATE["deny_policy"], bt.STATE["deny_all"])
    bt.begin_item(live_item=True)
    try:
        bt.STATE["deny_policy"], bt.STATE["deny_all"] = "ok", False      # the old _reset_policy_state()
        allowed = True
        try:
            bt.check((), dict(CP_TEST))
        except bt.DBBoundaryError:
            allowed = False
        assert not allowed, "VULNERABLE: a reset deny switch allowed a DB connection (fail-open)"
    finally:
        bt.STATE["deny_policy"], bt.STATE["deny_all"] = saved
        bt.begin_item(live_item=False)


@pytest.mark.boundary_probe
def test_loopback_connect_without_request_is_refused():
    srv = socket.create_server(("127.0.0.1", 0))
    try:
        connected = True
        try:
            socket.create_connection(srv.getsockname(), timeout=2).close()
        except OSError:
            connected = False
        assert not connected, "VULNERABLE: an unmarked test reached a loopback service (OA-2)"
    finally:
        srv.close()


@pytest.mark.boundary_probe
def test_unix_connect_without_request_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind("s.sock")
    srv.listen(1)
    try:
        cli = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connected = True
        try:
            cli.connect("s.sock")
        except OSError:
            connected = False
        cli.close()
        assert not connected, "VULNERABLE: an unmarked test reached a local AF_UNIX socket (OA-2)"
    finally:
        srv.close()


def test_netns_reexec_does_not_carry_db_credentials(monkeypatch):
    """Credential inheritance: the re-exec into the namespace must not carry DB URLs or passwords."""
    monkeypatch.setitem(bt.STATE, "netns_decided", False)
    monkeypatch.setitem(bt.STATE, "netns", "undecided")
    monkeypatch.setitem(bt.STATE, "config", bt.STATE["config"])
    monkeypatch.setitem(bt.STATE, "bootstrap_dir", None)          # never remove the real session's bootstrap
    monkeypatch.setattr(bt, "netns_status", lambda *a, **k: "not_isolated")
    monkeypatch.setattr(bt, "_ORIG_ENV", {"PATH": os.environ.get("PATH", ""), bt.RUNNER_MARKER: "1",   # INT2 precondition
                                          "VERA_DB_URL_RO": "postgresql://u:sentinel_pw@db-sentinel.invalid/x",
                                          "PGPASSWORD": "sentinel_pw", "OPENAI_API_KEY": "GUARD-SENTINEL"})
    seen = {}

    def spy(exe, argv, env):
        seen["env"] = env
        raise _Stop

    monkeypatch.setattr(os, "execve", spy)
    monkeypatch.setattr(bt, "_isolation_fail", lambda cfg, text: (_ for _ in ()).throw(SystemExit(text)))
    with pytest.raises(SystemExit):                                # INT3 MED-A: refused, never re-exec'd
        bt.maybe_reexec(_FakeCfg, False)
    assert "env" not in seen, "VULNERABLE: a not-isolated session was re-exec'd"
    env = bt.reexec_env(bt._ORIG_ENV)                              # the (now unused) chain helper still scrubs
    leaked = sorted(k for k in ("VERA_DB_URL_RO", "PGPASSWORD", "OPENAI_API_KEY") if k in env)
    assert leaked == [], f"VULNERABLE: re-exec env carries {leaked}"
