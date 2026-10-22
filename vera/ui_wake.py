"""UIWAKE-1022: the Streamlit UI wakes a sleeping or slow API before it sends /ask.

- `ensure_awake` probes GET <health_path> with a short timeout; on failure it shows a "Starting..." status and
  retries on the configured backoff schedule within `max_total_s` (config/ui_wake.json).
- The question is queued in st.session_state (`queue_question`) and sent by the caller only once the API is
  healthy (`take_ready_question`), so a rerun during the wait does not lose it.
- Optional restart kick (outward action, owner-enabled): if the wake window ends unhealthy, public mode only, and the
  env var named by `restart_hook_env` is set, POST to it at most once per cooldown per process. Only a failed health
  check reaches this path. The hook URL is read from the server env at call time; it is never returned, stored in
  session state, rendered or logged. Unset = no code path runs.
- No host, URL or exception text is ever rendered; messages come from config.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "ui_wake.json"
COOLDOWN_FLOOR_S = 900.0  # owner requirement: a restart kick at most once per 15 min per process
PENDING_KEY = "vera_wake_pending"
LAST_OK_KEY = "vera_wake_last_ok"

# Test seams (module level so AppTest runs, which share this process, can replace them).
http_get = requests.get
http_post = requests.post
clock = time.monotonic
sleep = time.sleep

_hook_lock = threading.Lock()
_hook_last_at: float | None = None  # process-level cooldown state


@dataclass(frozen=True)
class WakeConfig:
    health_path: str
    health_timeout_s: float
    page_load_recheck_s: float
    backoff_s: tuple
    max_total_s: float
    starting_message: str
    ready_message: str
    failed_message: str
    restart_requested_message: str
    restart_hook_env: str
    restart_hook_cooldown_s: float
    restart_hook_timeout_s: float


def load_config(path: Path = CONFIG_PATH) -> WakeConfig:
    raw = json.loads(Path(path).read_text())
    fields = {k: raw[k] for k in WakeConfig.__dataclass_fields__}
    fields["backoff_s"] = tuple(float(x) for x in fields["backoff_s"])
    fields["restart_hook_cooldown_s"] = max(COOLDOWN_FLOOR_S, float(fields["restart_hook_cooldown_s"]))
    return WakeConfig(**fields)


def is_healthy(base_url: str, cfg: WakeConfig) -> bool:
    base = (base_url or "").strip().rstrip("/")
    if not base.lower().startswith(("http://", "https://")):
        return False
    try:
        return http_get(f"{base}{cfg.health_path}", timeout=cfg.health_timeout_s).status_code == 200
    except Exception:  # noqa: BLE001  (any failure = not healthy; the error text may carry the host)
        return False


def _hook_url(cfg: WakeConfig, env=None) -> str:
    env = os.environ if env is None else env
    url = str(env.get(cfg.restart_hook_env, "") or "").strip()
    return url if url.lower().startswith("https://") else ""


def kick_restart(cfg: WakeConfig, env=None) -> bool:
    """POST the owner's restart hook once per cooldown. Returns True only if a request was sent now.
    Called only by ensure_awake after a failed wake window; never with user input."""
    global _hook_last_at
    url = _hook_url(cfg, env)
    if not url:
        return False
    with _hook_lock:
        now = clock()
        if _hook_last_at is not None and now - _hook_last_at < cfg.restart_hook_cooldown_s:
            return False
        _hook_last_at = now  # set before the call: a failed or slow call still counts against the cooldown
    try:
        http_post(url, timeout=cfg.restart_hook_timeout_s)
    except Exception:  # noqa: BLE001  (never surface or log the URL or the error text)
        pass
    return True


def reset_hook_cooldown() -> None:
    """Tests only."""
    global _hook_last_at
    with _hook_lock:
        _hook_last_at = None


def _in_script_run() -> bool:
    """False when the page module is merely imported or executed bare (no Streamlit script run, e.g. a
    multiprocessing spawn child re-running __main__): then nothing probes, waits or calls the hook."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:  # noqa: BLE001
        return False


def ensure_awake(st, base_url: str, *, allow_restart_hook: bool, cfg: WakeConfig | None = None) -> bool:
    """True once GET /health answers 200. Shows a status while waking; on failure a fixed friendly error."""
    if not _in_script_run():
        return True
    cfg = cfg or load_config()
    if is_healthy(base_url, cfg):
        st.session_state[LAST_OK_KEY] = clock()
        return True
    deadline = clock() + cfg.max_total_s
    ok = False
    with st.status(cfg.starting_message.format(max_total_s=cfg.max_total_s), expanded=False) as box:
        for step in cfg.backoff_s:
            wait = min(step, deadline - clock())
            if wait <= 0:
                break
            sleep(wait)
            if is_healthy(base_url, cfg):
                ok = True
                break
        box.update(label=cfg.ready_message if ok else cfg.failed_message, state="complete" if ok else "error")
    if ok:
        st.session_state[LAST_OK_KEY] = clock()
        return True
    st.session_state.pop(LAST_OK_KEY, None)
    st.error(cfg.failed_message)
    if allow_restart_hook and kick_restart(cfg):
        st.info(cfg.restart_requested_message)
    return False


def page_load_check(st, base_url: str, *, allow_restart_hook: bool, cfg: WakeConfig | None = None) -> None:
    """On page load: at most one health check per `page_load_recheck_s` per session; wakes the API if asleep."""
    cfg = cfg or load_config()
    last = st.session_state.get(LAST_OK_KEY)
    if last is not None and clock() - last < cfg.page_load_recheck_s:
        return
    ensure_awake(st, base_url, allow_restart_hook=allow_restart_hook, cfg=cfg)


def queue_question(state, payload: dict) -> None:
    state[PENDING_KEY] = dict(payload)


def take_ready_question(st, base_url: str, *, allow_restart_hook: bool, cfg: WakeConfig | None = None) -> dict | None:
    """The queued question once the API is healthy (and removes it from the queue), else None. A queued question
    survives a rerun during the wait; after a failed wake window it is dropped (the text input keeps it)."""
    pending = st.session_state.get(PENDING_KEY)
    if not pending:
        return None
    ok = ensure_awake(st, base_url, allow_restart_hook=allow_restart_hook, cfg=cfg)
    st.session_state.pop(PENDING_KEY, None)
    return pending if ok else None
