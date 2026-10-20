"""UI-1: when the UI finds the local API unreachable, start it ONCE as a local child process.

Gates (all must hold, else no spawn): dev mode (not public), env flag VERA_UI_AUTOSTART_API == "1",
RENDER != "true", the URL in use equals the DEFAULT local URL (public_mode.DEFAULT_LOCAL_URL) and parses
strictly as http://127.0.0.1:<1024-65535> (a typed sidebar URL, a tampered .vera-local-url, localhost,
userinfo, uppercase, brackets, trailing dot, lookalikes are all refused).
Command: <venv python> -m uvicorn main:app from config, plus code-appended --host 127.0.0.1 --port N
--workers 1; config may not supply --host/--port/--workers/--reload. shell=False, never run.sh/start.sh.
One spawn per process (module latch survives Streamlit reruns) and per host user (flock + 0600 pidfile
under <repo>/tmp, gitignored; a pid is trusted only if /proc/<pid>/cmdline matches our argv; we never kill).
Bounded wait (config, ~20s) then latch "failed" in-process and in a 0600 file honoured for max_wait_s.
A port that answers but is not VERA-DAR-shaped is never spawned over and the caller must not send the key.
Child output goes to a 0600 log; no .vera-local-url write.
The parent env context is passed to the child as VERA_ENV_CONTEXT only when the parent selected it explicitly
(the child inherits the selection mode, D5). The context gate is configurable (config autostart_allowed_contexts: a
list, or null = not enforced); shipped [local_dev, agent] per OD-DARINT-AGENTDEV. An unresolvable context refuses.

KNOWN LIMITS (accepted divergences, documented):
- main.py loads .env files through vera.env_precedence.load_env (PRB-DARINT-002 wiring), not a bare
  load_dotenv(): no parent-directory walk; restricted names (DB URLs, VERA_OWNER_KEY) load only as the
  resolved context allows (agent contexts never get VERA_OWNER_KEY). The env allowlist alone still does not
  control what the child's load_env reads from the repo's own .env files.
- With the flag on, a dev /ask on the auto-started API writes whatever the child's env/.env allows
  (production rows if a write URL is present). Owner-intentional; stated here next to the flag.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "config" / "api_kickstart.json"
PROC_ROOT = Path("/proc")
_URL_RE = re.compile(r"\Ahttp://127\.0\.0\.1:([0-9]{4,5})\Z")
_FORBIDDEN = ("--host", "--port", "--workers", "--reload")
_SAFE_ARG = re.compile(r"\A[A-Za-z0-9_.:\-]+\Z")

HEALTHY, STARTED, WAITED, DISABLED, NOT_LOCAL, FOREIGN, FAILED = (
    "healthy", "started", "waited_for_other_start", "disabled", "not_local", "foreign_listener", "failed")
OK, DOWN, FOREIGN_PORT = "ok", "down", "foreign"
LOCAL_CONTEXT = "local_dev"

_LATCH = {"spawned": False, "failed": False}  # process-level: survives Streamlit reruns


@dataclass(frozen=True)
class KickResult:
    status: str
    message: str  # user-facing; no host, no env values
    ok: bool


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text())


def default_health(base_url: str, cfg: dict) -> str:
    """OK = VERA-DAR-shaped /health; DOWN = nothing listening; FOREIGN_PORT = something else answers."""
    try:
        r = requests.get(base_url.rstrip("/") + cfg["health_path"], timeout=cfg["health_timeout_s"])  # nosec B113 - timeout set from config/api_kickstart.json (non-literal)
    except requests.ConnectionError:
        return DOWN
    except requests.RequestException:
        return FOREIGN_PORT
    try:
        body = r.json()
    except ValueError:
        return FOREIGN_PORT
    return OK if r.status_code == 200 and isinstance(body, dict) and body.get("status") == "ok" else FOREIGN_PORT


def local_target_port(base_url, default_url=None):
    """Port if base_url is exactly the default local URL AND strictly http://127.0.0.1:<port>, else None."""
    if default_url is None:
        from vera.public_mode import DEFAULT_LOCAL_URL as default_url
    if not isinstance(base_url, str) or base_url != default_url:
        return None
    m = _URL_RE.match(base_url)
    port = int(m.group(1)) if m else 0
    return port if 1024 <= port <= 65535 else None


def build_argv(cfg: dict, port: int, python: str | None = None) -> list[str]:
    tail = cfg.get("argv_tail")
    if not isinstance(tail, list) or tail != ["-m", "uvicorn", "main:app"]:
        raise ValueError("argv_tail must be exactly ['-m','uvicorn','main:app']")
    for a in tail:
        if not _SAFE_ARG.match(a) or any(a.startswith(f) for f in _FORBIDDEN):
            raise ValueError("unsafe argv element")
    if cfg.get("bind_host") != "127.0.0.1":
        raise ValueError("bind_host must be 127.0.0.1")
    return [python or sys.executable, *tail, "--host", "127.0.0.1", "--port", str(port), "--workers", "1"]


def _child_env(cfg: dict, env) -> dict:
    return {k: env[k] for k in cfg["env_allowlist"] if k in env}


def _manual_help(port) -> str:
    return (f"Start it yourself: in the VERA-DAR repo run './run.sh {port}', then reload this page. "
            "Child output, if any, is in tmp/api_kickstart.log.")


def _run_dir(cfg: dict) -> Path:
    d = REPO / cfg["run_dir"]
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    return d


def _open_private(path: Path, flags: int):
    return os.open(path, flags | os.O_CREAT | os.O_NOFOLLOW, 0o600)


def pid_is_ours(pid_path: Path, argv: list[str], proc_root: Path | None = None) -> bool:
    """True only if the pidfile names a live process whose cmdline equals our argv. Never kills."""
    proc_root = proc_root or PROC_ROOT
    try:
        pid = int(pid_path.read_text().strip())
        cmd = (proc_root / str(pid) / "cmdline").read_bytes().split(b"\0")
    except (OSError, ValueError):
        return False
    return [c.decode("utf-8", "replace") for c in cmd if c] == argv


def _spawn(cfg: dict, argv: list[str], env) -> subprocess.Popen:
    d = _run_dir(cfg)
    fd = _open_private(d / cfg["log_file"], os.O_WRONLY | os.O_APPEND)
    try:
        proc = subprocess.Popen(argv, cwd=str(REPO), env=_child_env(cfg, env), shell=False,
                                stdin=subprocess.DEVNULL, stdout=fd, stderr=subprocess.STDOUT,
                                start_new_session=True)
    finally:
        os.close(fd)
    pfd = _open_private(d / cfg["pid_file"], os.O_WRONLY | os.O_TRUNC)
    with os.fdopen(pfd, "w") as f:
        f.write(str(proc.pid))
    return proc


def _poll(base_url, cfg, health, sleep, clock) -> bool:
    start, delay = clock(), cfg["backoff_initial_s"]
    for _ in range(cfg["max_polls"]):
        if health(base_url) == OK:
            return True
        if clock() - start + delay > cfg["max_wait_s"]:
            break
        sleep(delay)
        delay = min(delay * cfg["backoff_factor"], cfg["backoff_max_s"])
    return health(base_url) == OK


def _failed_recently(cfg, now) -> bool:
    try:
        return now - (_run_dir(cfg) / cfg["failed_file"]).stat().st_mtime < cfg["max_wait_s"]
    except OSError:
        return False


def _fail(cfg, port, text) -> KickResult:
    _LATCH["failed"] = True
    try:
        os.close(_open_private(_run_dir(cfg) / cfg["failed_file"], os.O_WRONLY | os.O_TRUNC))
        os.utime(_run_dir(cfg) / cfg["failed_file"])
    except OSError:
        pass
    return KickResult(FAILED, f"{text} {_manual_help(port)}", False)


def parent_env_context() -> tuple[str, str]:
    """(context, selected_by) of the parent's resolved env (PRB-DARINT-002 s9); the child inherits both, never a
    literal. selected_by is env_precedence's mode ("explicit", "default", ...)."""
    from vera import env_files
    res = env_files.load_vera_env()
    return res.context, res.selected_by


def ensure_api(base_url, *, env=None, public_mode=None, cfg=None, health=None, spawn=None,
               sleep=time.sleep, clock=time.monotonic, notify=None, proc_root=None,
               wall=time.time, context_fn=None) -> KickResult:
    """Make sure the local API answers. Bounded, one spawn per process, never kills anything."""
    env = os.environ if env is None else env
    cfg = cfg or load_config()
    health = health or (lambda u: default_health(u, cfg))
    spawn = spawn or _spawn
    notify = notify or (lambda _m: None)
    state = health(base_url)
    if state == OK:
        return KickResult(HEALTHY, "API is running.", True)
    if state == FOREIGN_PORT:
        return KickResult(FOREIGN, "Something is listening at the API address but it is not VERA-DAR; "
                          "not starting a second API and not sending your key.", False)
    if public_mode is None:
        from vera.public_mode import is_public_mode
        public_mode = is_public_mode(env)
    port = local_target_port(base_url)
    if public_mode or str(env.get("RENDER", "")).strip().lower() == "true" or port is None:
        return KickResult(NOT_LOCAL, "The API is not responding and will not be auto-started "
                          "(only a local dev UI using the default local address may do that).", False)
    if str(env.get(cfg["enable_env"], "")) != "1":
        return KickResult(DISABLED, f"The API is not responding; auto-start is off "
                          f"(set {cfg['enable_env']}=1 to enable). {_manual_help(port)}", False)
    if _LATCH["failed"] or _failed_recently(cfg, wall()):
        return KickResult(FAILED, f"An earlier auto-start attempt failed. {_manual_help(port)}", False)
    try:
        resolved = (context_fn or parent_env_context)()
        # fail closed: anything but a (context, selected_by) pair is treated as default-selected (no context passed)
        context, selected_by = resolved if (isinstance(resolved, tuple) and len(resolved) == 2) else (resolved, "default")
    except Exception as exc:  # EnvPrecedenceError or import problems: fail closed, no spawn
        return KickResult(DISABLED, f"Auto-start refused: env context could not be resolved "
                          f"({type(exc).__name__}). {_manual_help(port)}", False)
    allowed = cfg.get("autostart_allowed_contexts")  # None = not enforced; shipped list per OD-DARINT-AGENTDEV
    if allowed is not None and context not in allowed:
        return KickResult(DISABLED, f"Auto-start refused: env context {context!r} is not in the configured "
                          f"autostart_allowed_contexts. {_manual_help(port)}", False)
    # D5 (env verifier): the child inherits the parent's SELECTION MODE. Only an explicitly selected parent context is
    # passed as VERA_ENV_CONTEXT (which env_precedence treats as explicit); a default-selected parent passes none, so
    # the child also resolves by default and restricted names (e.g. DB URLs) stay skipped as they are for the parent.
    env = {k: v for k, v in env.items() if k != "VERA_ENV_CONTEXT"}
    if selected_by in ("explicit", "agent_explicit"):  # G2: agent_explicit parents pass their context too
        env["VERA_ENV_CONTEXT"] = context
    try:
        argv = build_argv(cfg, port)
    except ValueError as exc:
        return _fail(cfg, port, f"Auto-start config rejected ({exc}).")
    lock_fd = _open_private(_run_dir(cfg) / cfg["lock_file"], os.O_RDWR)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:  # another start is in progress: wait, never spawn
            if _poll(base_url, cfg, health, sleep, clock):
                return KickResult(WAITED, "API was being started by another session and is now running.", True)
            return _fail(cfg, port, "Another session started the API but it never became healthy.")
        try:
            if health(base_url) == OK:
                return KickResult(HEALTHY, "API is running.", True)
            if _LATCH["spawned"] or pid_is_ours(_run_dir(cfg) / cfg["pid_file"], argv, proc_root):
                # already started (this process or an earlier one); do not respawn, just wait
                if _poll(base_url, cfg, health, sleep, clock):
                    return KickResult(STARTED, "The API is running (started earlier).", True)
                return _fail(cfg, port, "The API was started earlier but is not healthy.")
            _LATCH["spawned"] = True
            notify(f"The API was not responding; starting it locally on port {port}...")
            try:
                spawn(cfg, argv, env)
            except OSError as exc:
                return _fail(cfg, port, f"Could not start the API ({type(exc).__name__}).")
            if _poll(base_url, cfg, health, sleep, clock):
                return KickResult(STARTED, f"The API was not running, so VERA-DAR started it locally on port {port}.", True)
            return _fail(cfg, port, f"Started the API process but /health did not report ok within "
                         f"{cfg['max_wait_s']:.0f}s.")
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)
