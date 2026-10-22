"""WED-P161 (P1.6.1): warm the DB-backed full-document corpus and its evidence index at startup, off the request path.

ONE daemon thread, started at most once per process, only when fulldoc.enabled is true-like, the DB run-id env var is
set, the bundle env var is NOT set, and fulldoc.warm_at_startup is not false-like (absent => on). It calls the same
get_corpus() a request would (fulldoc_db.cached_load is single-flight, so a request arriving meanwhile waits on the
same load instead of starting a second one) and then builds the evidence index under the corpus' own lock.
One attempt, no retries; the DB I/O is bounded by the existing connect / statement timeouts and the failure cache.
Logs carry a reason code and at most an exception class name: never a host, key, URL or run id. Never raises.
While it runs, /health/dar reports fulldoc status "loading" without touching the corpus (vera/dar_health.py)."""
from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

IDLE, WARMING, DONE, FAILED = "idle", "warming", "done", "failed"
R_WARM_UNUSABLE = "warm_corpus_unusable"
R_WARM_ERROR = "warm_error"

_LOCK = threading.Lock()
_STATE: dict = {"state": IDLE, "thread": None}

_FALSE = ("false", "0", "no", "off")
_TRUE = ("true", "1", "yes", "on")


def reset_for_tests() -> None:
    with _LOCK:
        _STATE.update(state=IDLE, thread=None)


def state() -> str:
    with _LOCK:
        return _STATE["state"]


def warming() -> bool:
    return state() == WARMING


def wanted(cfg: dict, env) -> bool:
    """True only when a DB-backed corpus would be loaded by get_corpus and the warm flag is not off. Never raises."""
    try:
        fd = cfg.get("fulldoc")
        if not isinstance(fd, dict) or "enabled" not in fd:
            return False
        en = fd["enabled"]
        if en is not True and str(en).strip().lower() not in _TRUE:
            return False
        w = fd.get("warm_at_startup", True)
        if w is False or str(w).strip().lower() in _FALSE:
            return False
        from vera import fulldoc_corpus
        if (env.get(cfg["bundle_dir_env"]) or "").strip():
            return False  # bundle only, or both sources (R-SOURCE): nothing to warm
        return bool((env.get(fulldoc_corpus._db_run_env(cfg)) or "").strip())
    except Exception:  # noqa: BLE001
        return False


def loading_status() -> dict:
    """The /health/dar fulldoc block while warming: same keys as readiness() for the DB source, codes and counts only."""
    return {"enabled": True, "bundle_configured": False, "status": "loading", "n_acquired": 0, "n_unusable": 0,
            "pins_ok": None, "source": "db", "db_configured": True, "codes": []}


def _run(cfg: dict, env) -> None:
    t0 = time.monotonic()
    final, reason = FAILED, R_WARM_ERROR
    try:
        from vera import fulldoc_corpus
        c = fulldoc_corpus.get_corpus(cfg, env)
        if c is None or c.status != fulldoc_corpus.STATUS_OK or not c.acquired:
            reason = R_WARM_UNUSABLE
            logger.warning("fulldoc warm-up: reason=%s elapsed_s=%.1f", reason, time.monotonic() - t0)
            return
        c._evidence_dar()
        final = DONE
        logger.info("fulldoc warm-up: done n_acquired=%d elapsed_s=%.1f", len(c.acquired), time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001  non-fatal: the request path loads lazily as before
        logger.warning("fulldoc warm-up: reason=%s (%s) elapsed_s=%.1f", R_WARM_ERROR, type(exc).__name__,
                       time.monotonic() - t0)
    finally:
        with _LOCK:
            _STATE["state"] = final


def start(cfg: dict | None = None, env=None) -> threading.Thread | None:
    """Start the warm-up thread when wanted; returns it (or None). Idempotent per process. Never raises."""
    try:
        env = os.environ if env is None else env
        if cfg is None:
            from vera import fulldoc_corpus
            cfg = fulldoc_corpus.load_config()
        if not wanted(cfg, env):
            return None
        with _LOCK:
            if _STATE["thread"] is not None:
                return None
            t = threading.Thread(target=_run, args=(cfg, env), name="fulldoc-warm", daemon=True)
            _STATE.update(state=WARMING, thread=t)
        t.start()
        return t
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            if _STATE["state"] == WARMING:
                _STATE["state"] = FAILED
        logger.warning("fulldoc warm-up: reason=%s (%s)", R_WARM_ERROR, type(exc).__name__)
        return None
