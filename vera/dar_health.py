"""DAR readiness for startup logging and /health (P0-E). Read-only, bounded, cached, never raises.

Output carries CODES and COUNTS only: no hosts, URLs, env values or individual source titles. Uses the read-only DB URL
(env var NAME taken from the DAR runtime config) via an injectable engine_factory. Performs no writes and no DDL.
Overall API health is separate: callers must not make "status ok" depend on this."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "dar_health.json"
_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")
_LOCK = threading.Lock()
_CACHE: dict = {}  # {"value": dict, "at": float, "ttl": float}
_REFRESHING = threading.Event()


def reset_for_tests() -> None:
    with _LOCK:
        _CACHE.clear()
    _REFRESHING.clear()


def _cfg() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def _result(dar, routing, mig, stored, corpus, fallback, reason) -> dict:
    return {"dar": dar, "migrations": mig, "stored_corpus": stored, "routing": routing,
            "corpus": corpus, "fallback": fallback, "reason_code": reason}


def _unknown(reason: str) -> dict:
    """Readiness not known (not yet probed, or the health check itself failed): dar and routing are None, never a
    claimed outage or "routing off" (D-DARINT-050 point 2: API availability is distinct from DAR readiness)."""
    return _result(None, None, {"006": None, "007": None}, "unknown", None, None, reason)


def _probe_db(url: str, cfg: dict, engine_factory):
    """(migrations dict, counts dict|None). Raises on any DB error. SELECT only; engine always disposed."""
    from sqlalchemy import text  # lazy
    if engine_factory is None:
        from sqlalchemy import create_engine as engine_factory  # lazy
    schema = cfg["schema"]
    groups = cfg["migration_tables"]
    counts_cfg = cfg["count_tables"]
    idents = [schema] + [t for ts in groups.values() for t in ts] + list(counts_cfg.values())
    if not all(_IDENT.match(i) for i in idents):
        raise ValueError("bad identifier in config")
    engine = engine_factory(url, connect_args={"connect_timeout": int(cfg["connect_timeout_s"]),
                                               "options": f"-c statement_timeout={int(cfg['statement_timeout_ms'])}"},
                            pool_pre_ping=True, hide_parameters=True)
    try:
        with engine.connect() as c:
            present = {}
            for mig, tables in groups.items():
                ok = True
                for t in tables:
                    row = c.execute(text("SELECT to_regclass(:n) IS NOT NULL"), {"n": f"{schema}.{t}"}).fetchall()
                    ok = ok and bool(row and row[0][0])
                present[mig] = ok
            counts = None
            if present.get("007"):
                counts = {}
                for key, t in counts_cfg.items():  # identifiers validated above; COUNT only, no row/text loads
                    row = c.execute(text(f"SELECT count(*) FROM {schema}.{t}")).fetchall()  # nosec B608 - schema/table names regex-checked by _IDENT above; COUNT only
                    counts[key] = int(row[0][0]) if row else 0
            return present, counts
    finally:
        try:
            engine.dispose()
        except Exception:  # noqa: BLE001
            pass


class _Captured(Exception):
    """Raised by the capturing engine factory to hand back the URL the DAR runtime resolved (never logged or stored)."""

    def __init__(self, url):
        super().__init__("captured")
        self.url = url


def _runtime_url(rt: dict) -> str:
    """The RO URL exactly as dar_runtime itself resolves it (its own env handling, incl. any injected env), via its
    _engine_for (or earlier _fetch_version) path with a factory that captures the URL before any engine is built. "" when absent/unsupported."""
    from vera import dar_runtime

    def _capture(url, **kw):
        raise _Captured(url)
    try:
        if hasattr(dar_runtime, "_engine_for"):
            dar_runtime._engine_for(rt, os.environ, _capture)
        else:  # earlier runtime layout
            dar_runtime._fetch_version(rt, os.environ, _capture)
    except _Captured as c:
        return c.url
    except Exception:  # noqa: BLE001  (db url missing, or no such helper)
        pass
    return ""


def _compute(env, engine_factory) -> tuple[dict, float]:
    cfg = _cfg()
    from vera import dar_runtime, domain_registry
    ttl = float(cfg["cache_ttl_s"])
    try:
        rstatus = dar_runtime.status()
    except Exception:  # noqa: BLE001
        rstatus = "unavailable:config_error"
    routing = "off" if (rstatus == "disabled" or rstatus == "unavailable:config_error") else "on"
    rt = {}
    try:
        rt = (domain_registry.load_config().get("runtime") or {})
    except Exception:  # noqa: BLE001
        pass
    if env is None:  # production path: the runtime's own URL resolution
        url = _runtime_url(rt).strip()
    else:
        url = (env.get(str(rt.get("db_url_env") or cfg["default_db_url_env"])) or "").strip()
    unknown = {"006": None, "007": None}
    if not url:
        reason = "no_db_url"
        if routing == "off":
            return _result("disabled", "off", unknown, "unknown", None, False, "routing_off"), ttl
        return _result("unavailable", routing, unknown, "unknown", None, True, reason), float(cfg["failure_cache_ttl_s"])
    try:
        mig, counts = _probe_db(url, cfg, engine_factory)
    except Exception as exc:  # noqa: BLE001  class name only, never the message (may embed the URL)
        logger.warning("DAR health probe failed (%s)", type(exc).__name__)
        fb = routing == "on"
        dar = "unavailable" if fb else "disabled"
        return _result(dar, routing, unknown, "unknown", None, fb, "db_unreachable"), float(cfg["failure_cache_ttl_s"])
    if counts is None:
        stored = "missing_tables"
    elif counts.get("documents", 0) > 0 and counts.get("chunks", 0) > 0:
        stored = "ready"
    else:
        stored = "empty"
    if routing == "off":
        return _result("disabled", "off", mig, stored, counts, False, "routing_off"), ttl
    if not mig.get("007"):
        return _result("degraded", routing, mig, stored, None, True, "migration_007_missing"), ttl
    if stored != "ready":
        return _result("degraded", routing, mig, stored, counts, True, "corpus_empty"), ttl
    if rstatus not in ("ok", "unavailable:not_built"):  # not_built = lazy first build, not a failure
        reason = re.sub(r"[^a-z0-9_]", "_", rstatus.split(":", 1)[-1].lower())[:40] or "runtime_not_ok"
        return _result("degraded", routing, mig, stored, counts, True, "runtime_" + reason), ttl
    return _result("ready", routing, mig, stored, counts, False, "ok"), ttl


def _fulldoc_status() -> dict:
    """C-3/S8: codes and counts only (fulldoc_corpus.readiness never raises). Evaluated per call, not cached here.
    Attached only by the authenticated /health/dar handler (never by public /health: amplification)."""
    try:
        from vera import fulldoc_corpus  # lazy
        return fulldoc_corpus.readiness()
    except Exception:  # noqa: BLE001
        return {"enabled": False, "bundle_configured": False, "status": "error", "n_acquired": 0, "n_unusable": 0,
                "pins_ok": None, "source": "bundle"}


def _with_fd(value: dict, include: bool) -> dict:
    return {**value, "fulldoc": _fulldoc_status()} if include else dict(value)


def dar_readiness(env=None, engine_factory=None, now=None, force: bool = False, include_fulldoc: bool = True) -> dict:
    """Readiness dict (codes and counts only). Cached for the configured TTL; never raises."""
    clock = now or time.monotonic
    try:
        with _LOCK:
            hit = _CACHE.get("value")
            if hit is not None and not force and clock() - _CACHE["at"] < _CACHE["ttl"]:
                return _with_fd(hit, include_fulldoc)
        value, ttl = _compute(env, engine_factory)
        with _LOCK:
            _CACHE.update(value=value, at=clock(), ttl=ttl)
        return _with_fd(value, include_fulldoc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DAR health error (%s)", type(exc).__name__)
        return _with_fd(_unknown("health_error"), include_fulldoc)


def cached_readiness(env=None, engine_factory=None) -> dict:
    """Never touches the DB on the caller's thread: returns the last value (or a not_probed placeholder) and, when stale,
    refreshes in a daemon thread. Intended for /health."""
    try:
        with _LOCK:
            value, at, ttl = _CACHE.get("value"), _CACHE.get("at", 0.0), _CACHE.get("ttl", 0.0)
        if value is None or time.monotonic() - at >= ttl:
            if not _REFRESHING.is_set():
                _REFRESHING.set()

                def _run():
                    try:
                        dar_readiness(env, engine_factory, force=True, include_fulldoc=False)
                    finally:
                        _REFRESHING.clear()
                threading.Thread(target=_run, daemon=True).start()
        if value is None:
            return _unknown("not_probed")
        return dict(value)
    except Exception:  # noqa: BLE001
        return _unknown("health_error")


def startup_line(env=None, engine_factory=None) -> str:
    """Log one codes-only line at startup and return it. Never raises."""
    try:
        r = dar_readiness(env, engine_factory, force=True, include_fulldoc=False)
        c = r["corpus"] or {}
        m = r["migrations"]
        missing = ",".join(k for k in ("007", "006") if m.get(k) is False) or "none"
        line = ("DAR readiness: dar=%s routing=%s m006=%s m007=%s stored_corpus=%s documents=%s chunks=%s domains=%s "
                "fallback=%s missing_migrations=%s reason=%s"
                % (r["dar"], r["routing"], m.get("006"), m.get("007"), r["stored_corpus"], c.get("documents"),
                   c.get("chunks"), c.get("domains"), r["fallback"], missing, r["reason_code"]))
        logger.info(line)
        return line
    except Exception as exc:  # noqa: BLE001
        line = "DAR readiness: dar=unavailable reason=health_error (%s)" % type(exc).__name__
        logger.warning(line)
        return line


def startup_log_async(env=None, engine_factory=None) -> None:
    """Fire-and-forget startup_line in a daemon thread so import/startup is never delayed by the bounded DB probe."""
    try:
        threading.Thread(target=startup_line, args=(env, engine_factory), daemon=True).start()
    except Exception:  # noqa: BLE001
        pass
