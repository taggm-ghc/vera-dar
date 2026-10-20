"""UI-1 api_kickstart tests. MOCK-ONLY: spawn, Popen and health are fakes; no API, socket, DB or real child
process is ever started. A real launch smoke is UNVERIFIED."""
import fcntl
import os
import stat
import threading

import pytest

import vera.api_kickstart as k

URL = "http://127.0.0.1:8001"
ENV = {"VERA_UI_AUTOSTART_API": "1", "PATH": "/x", "VERA_PUBLIC_MODE": "dev", "CANARY_SECRET": "s"}


@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    monkeypatch.setattr(k, "REPO", tmp_path)
    monkeypatch.setitem(k._LATCH, "spawned", False)
    monkeypatch.setitem(k._LATCH, "failed", False)


@pytest.fixture
def cfg():
    c = k.load_config()
    c.update(max_polls=3, backoff_initial_s=0.0)
    return c


def run(cfg, health, spawn, url=URL, env=ENV, public=False, **kw):
    kw.setdefault("context_fn", lambda: ("local_dev", "explicit"))  # never reads real .env files
    return k.ensure_api(url, env=env, public_mode=public, cfg=cfg, health=health, spawn=spawn,
                        sleep=lambda s: None, **kw)


def test_healthy_no_spawn(cfg):
    calls = []
    r = run(cfg, lambda u: k.OK, lambda *a: calls.append(a))
    assert r.status == k.HEALTHY and not calls


def test_unreachable_spawn_then_healthy(cfg):
    up, msgs, calls = {"v": False}, [], []
    r = run(cfg, lambda u: k.OK if up["v"] else k.DOWN,
            lambda c, a, e: (calls.append(a), up.update(v=True)), notify=msgs.append)
    assert r.status == k.STARTED and r.ok and len(calls) == 1 and "starting" in msgs[0]


def test_timeout_one_spawn_then_latched_in_process_and_file(cfg):
    calls = []
    r = run(cfg, lambda u: k.DOWN, lambda c, a, e: calls.append(a))
    assert r.status == k.FAILED and "run.sh" in r.message and "127.0.0.1" not in r.message
    assert run(cfg, lambda u: k.DOWN, lambda c, a, e: calls.append(a)).status == k.FAILED
    k._LATCH.update(spawned=False, failed=False)  # simulate a new process: file latch still blocks (D1)
    assert run(cfg, lambda u: k.DOWN, lambda c, a, e: calls.append(a)).status == k.FAILED
    assert len(calls) == 1


def test_health_retries_bounded(cfg):
    n = {"c": 0}
    run(cfg, lambda u: (n.update(c=n["c"] + 1), k.DOWN)[1], lambda *a: None)
    assert n["c"] <= cfg["max_polls"] + 4


def test_wait_deadline_from_config(cfg):
    cfg.update(max_polls=100, backoff_initial_s=5.0, backoff_factor=1.0, max_wait_s=20.0)
    t = {"now": 0.0}
    n = {"c": 0}
    k.ensure_api(URL, env=ENV, public_mode=False, cfg=cfg, spawn=lambda *a: None,
                 health=lambda u: (n.update(c=n["c"] + 1), k.DOWN)[1],
                 sleep=lambda s: t.update(now=t["now"] + s), clock=lambda: t["now"])
    assert t["now"] <= 20.0 and n["c"] < 12


@pytest.mark.parametrize("url", [
    "https://127.0.0.1:8001", "http://localhost:8001", "http://[::1]:8001", "http://127.0.0.1.:8001",
    "http://user@127.0.0.1:8001", "http://HTTP://127.0.0.1:8001", "http://127.0.0.1.evil.com:8001",
    "http://127.0.0.1:8002", "http://0.0.0.0:8001", "https://x.onrender.com", "http://127.0.0.1:8001/x",
    "http://127.0.0.1:80", "http://[127.0.0.1]:8001", "HTTP://127.0.0.1:8001", ""])
def test_non_default_or_malformed_url_never_spawns(cfg, url):
    calls = []
    r = run(cfg, lambda u: k.DOWN, lambda *a: calls.append(a), url=url)
    assert r.status == k.NOT_LOCAL and not calls


def test_default_url_parser_strict():
    assert k.local_target_port(URL) == 8001
    assert k.local_target_port("http://127.0.0.1:80", "http://127.0.0.1:80") is None  # port < 1024
    assert k.local_target_port("http://[::1]:8001", "http://[::1]:8001") is None  # bind is 127.0.0.1 only


@pytest.mark.parametrize("env,public", [
    ({**ENV, "VERA_UI_AUTOSTART_API": "true"}, False), ({**ENV, "VERA_UI_AUTOSTART_API": ""}, False),
    ({k_: v for k_, v in ENV.items() if k_ != "VERA_UI_AUTOSTART_API"}, False),
    (ENV, True), ({**ENV, "RENDER": "true"}, False)])
def test_gates_each_missing_no_spawn(cfg, env, public):
    calls = []
    r = run(cfg, lambda u: k.DOWN, lambda *a: calls.append(a), env=env, public=public)
    assert not calls and not r.ok


def test_foreign_listener_no_spawn(cfg):
    calls = []
    r = run(cfg, lambda u: k.FOREIGN_PORT, lambda *a: calls.append(a))
    assert r.status == k.FOREIGN and not calls


def test_default_health_shapes(cfg, monkeypatch):
    class R:
        def __init__(s, code, body):
            s.status_code, s._b = code, body

        def json(s):
            if isinstance(s._b, Exception):
                raise s._b
            return s._b

    for resp, want in [(R(200, {"status": "ok"}), k.OK), (R(200, {"x": 1}), k.FOREIGN_PORT),
                       (R(200, ValueError()), k.FOREIGN_PORT), (R(500, {"status": "ok"}), k.FOREIGN_PORT)]:
        monkeypatch.setattr(k.requests, "get", lambda *a, _r=resp, **kw: _r)
        assert k.default_health(URL, cfg) == want

    def boom(*a, **kw):
        raise k.requests.ConnectionError()
    monkeypatch.setattr(k.requests, "get", boom)
    assert k.default_health(URL, cfg) == k.DOWN


@pytest.mark.parametrize("tail", [
    "python -m uvicorn main:app", ["-m", "uvicorn", "main:app", "--host", "0.0.0.0"],
    ["-m", "uvicorn", "main:app", "--reload"], ["-m", "uvicorn", "main:app", "--workers", "4"],
    ["-m", "uvicorn", "main:app;rm"], ["-m", "uvicorn", "evil:app"], ["-c", "x"], [1, 2, 3]])
def test_argv_validation_rejects(cfg, tail):
    with pytest.raises(ValueError):
        k.build_argv({**cfg, "argv_tail": tail}, 8001)


def test_argv_shape(cfg):
    a = k.build_argv(cfg, 8001, python="/v/python")
    assert a == ["/v/python", "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8001", "--workers", "1"]


def test_spawn_popen_args_env_and_private_files(cfg, monkeypatch):
    seen = {}

    class P:
        pid = 4242
    monkeypatch.setattr(k.subprocess, "Popen", lambda argv, **kw: seen.update(argv=argv, **kw) or P())
    full = {n: "v" for n in cfg["env_allowlist"]}
    full.update(CANARY_SECRET="s", RENDER="x")
    k._spawn(cfg, k.build_argv(cfg, 8001), full)
    assert seen["shell"] is False and isinstance(seen["argv"], list) and "--reload" not in seen["argv"]
    assert set(seen["env"]) == set(cfg["env_allowlist"])
    assert not any(n.startswith("VERA_DB_" + "URL") for n in cfg["env_allowlist"])
    for name in (cfg["log_file"], cfg["pid_file"]):
        assert stat.S_IMODE(os.stat(k.REPO / "tmp" / name).st_mode) == 0o600
    assert (k.REPO / "tmp" / cfg["pid_file"]).read_text() == "4242"
    assert not (k.REPO / ".vera-local-url").exists()


def test_concurrent_calls_single_spawn(cfg):
    state, calls, results = {"up": False}, [], []
    inside, release = threading.Event(), threading.Event()

    def slow(c, a, e):
        calls.append(a)
        inside.set()
        release.wait(5)
        state["up"] = True

    def h(u):
        return k.OK if state["up"] else k.DOWN
    t = threading.Thread(target=lambda: results.append(run(cfg, h, slow)))
    t.start()
    assert inside.wait(5)

    def sl(s):  # waiter's first sleep deterministically lets the spawner finish
        release.set()
        t.join(5)

    r2 = k.ensure_api(URL, env=ENV, public_mode=False, cfg=cfg, health=h, context_fn=lambda: "local_dev",
                      spawn=lambda *a: calls.append("SECOND"), sleep=sl)
    t.join(5)
    assert len(calls) == 1 and r2.status == k.WAITED


def test_held_lock_never_spawns(cfg):
    fd = k._open_private(k._run_dir(cfg) / cfg["lock_file"], os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    calls = []
    try:
        r = run(cfg, lambda u: k.DOWN, lambda *a: calls.append(a))
    finally:
        os.close(fd)
    assert not calls and r.status == k.FAILED


def test_rerun_after_spawn_does_not_respawn(cfg):
    calls = []
    run(cfg, lambda u: k.DOWN, lambda c, a, e: calls.append(a))
    k._LATCH["failed"] = False
    (k._run_dir(cfg) / cfg["failed_file"]).unlink()
    run(cfg, lambda u: k.DOWN, lambda c, a, e: calls.append(a))
    assert len(calls) == 1  # spawned latch alone blocks (D1: 3 calls, 3 spawns before)


def test_stale_pidfile_foreign_cmdline_not_trusted(cfg, tmp_path):
    proc = tmp_path / "proc"
    (proc / "77").mkdir(parents=True)
    (proc / "77" / "cmdline").write_bytes(b"/usr/bin/vim\0file\0")
    pf = k._run_dir(cfg) / cfg["pid_file"]
    pf.write_text("77")
    argv = k.build_argv(cfg, 8001)
    assert not k.pid_is_ours(pf, argv, proc)
    (proc / "77" / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    assert k.pid_is_ours(pf, argv, proc)
    pf.write_text("99999999")
    assert not k.pid_is_ours(pf, argv, proc)
    pf.write_text("junk")
    assert not k.pid_is_ours(pf, argv, proc)


def test_live_pid_ours_prevents_spawn(cfg, tmp_path):
    proc = tmp_path / "proc"
    (proc / "55").mkdir(parents=True)
    argv = k.build_argv(cfg, 8001)
    (proc / "55" / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    (k._run_dir(cfg) / cfg["pid_file"]).write_text("55")
    calls = []
    r = run(cfg, lambda u: k.DOWN, lambda *a: calls.append(a), proc_root=proc)
    assert not calls and r.status == k.FAILED


def test_context_propagated_to_child_env(cfg):
    seen = {}
    run(cfg, lambda u: k.DOWN, lambda c, a, e: seen.update(e), context_fn=lambda: ("local_dev", "explicit"))
    assert seen["VERA_ENV_CONTEXT"] == "local_dev" and "VERA_ENV_CONTEXT" in cfg["env_allowlist"]
    assert k._child_env(cfg, seen)["VERA_ENV_CONTEXT"] == "local_dev"


def test_parent_env_context_calls_load_env(monkeypatch):
    import types
    # ENV-A2: parent_env_context now goes through env_files.load_vera_env (PLAN-ENV v2.1 1.4), so that is the seam.
    fake = types.SimpleNamespace(load_env=lambda: types.SimpleNamespace(context="local_dev", selected_by="default"))
    monkeypatch.setattr("vera.env_files.load_vera_env", fake.load_env)
    assert k.parent_env_context() == ("local_dev", "default")


def test_default_selected_parent_passes_no_context_to_child(cfg):
    """D5: a default-selected parent must not make the child 'explicit' (which would load restricted DB URLs)."""
    seen = {}
    env = {**ENV, "VERA_ENV_CONTEXT": "stale"}
    run(cfg, lambda u: k.DOWN, lambda c, a, e: seen.update(e), env=env, context_fn=lambda: ("local_dev", "default"))
    assert seen and "VERA_ENV_CONTEXT" not in seen and "VERA_ENV_CONTEXT" not in k._child_env(cfg, seen)


@pytest.mark.parametrize("bad", ["local_dev", ("local_dev",), ["local_dev", "explicit"], None])
def test_non_pair_context_fn_result_fails_closed_to_default(cfg, bad):
    """Env verifier gap: a bare string (or any non-pair) must not be treated as explicit."""
    seen = {}
    run(cfg, lambda u: k.DOWN, lambda c, a, e: seen.update(e), context_fn=lambda: bad)
    assert "VERA_ENV_CONTEXT" not in seen


def test_agent_explicit_parent_passes_context_to_child(cfg):
    """G2 (AGENTDEV verifier): an agent_explicit parent's child must resolve agent_explicit, not agent_marker."""
    seen = {}
    run(cfg, lambda u: k.DOWN, lambda c, a, e: seen.update(e), context_fn=lambda: ("agent", "agent_explicit"))
    assert seen["VERA_ENV_CONTEXT"] == "agent" and k._child_env(cfg, seen)["VERA_ENV_CONTEXT"] == "agent"


def test_explicit_selected_parent_passes_context_to_child(cfg):
    seen = {}
    run(cfg, lambda u: k.DOWN, lambda c, a, e: seen.update(e), context_fn=lambda: ("local_dev", "explicit"))
    assert seen["VERA_ENV_CONTEXT"] == "local_dev" and k._child_env(cfg, seen)["VERA_ENV_CONTEXT"] == "local_dev"


@pytest.mark.parametrize("ctx", ["agent", "render", "test", "", "LOCAL_DEV"])
def test_context_outside_configured_allowlist_refused_no_spawn(cfg, ctx):
    calls = []
    cfg = {**cfg, "autostart_allowed_contexts": ["local_dev"]}
    r = run(cfg, lambda u: k.DOWN, lambda *a: calls.append(a), context_fn=lambda: ctx)
    assert not calls and not r.ok and r.status == k.DISABLED


def test_context_gate_on_hold_in_shipped_config():
    """OD-DARINT-AGENTDEV (2026-10-18): the hold is lifted; local_dev and agent may auto-start (agent children
    load .env under agent_explicit, which never yields VERA_OWNER_KEY or DB URLs)."""
    assert k.load_config()["autostart_allowed_contexts"] == ["local_dev", "agent"]


@pytest.mark.parametrize("ctx", ["agent", "local_dev"])
def test_hold_allows_any_resolved_context_and_propagates_it(cfg, ctx):
    seen = {}
    cfg = {**cfg, "autostart_allowed_contexts": None}
    run(cfg, lambda u: k.DOWN, lambda c, a, e: seen.update(e), context_fn=lambda: (ctx, "explicit"))
    assert seen["VERA_ENV_CONTEXT"] == ctx


def test_context_resolution_error_refused_no_spawn(cfg):
    calls = []

    def boom():
        raise RuntimeError("bad config")
    r = run(cfg, lambda u: k.DOWN, lambda *a: calls.append(a), context_fn=boom)
    assert not calls and not r.ok
