"""CHUNK-TAXO-JOB-1109: read-only display status for the API pages (WP6 imports status()).

Reads the latest finished tax_job_runs row written by the chunk-taxonomy job (one SELECT, through the rw engine the API already
has), caches it with a TTL, never raises, and is safe when migration 024 is absent. No chunk text is read or returned.
Returns {state, version, metrics, display_ok, reasons, last_run}; state is "ok" (a finished run exists), "none" (024 present, no
finished run) or "absent" (024 missing or unreachable). display_ok is False unless a finished run says True.
"""
from __future__ import annotations

import logging
import os
import threading
import time

from vera import chunk_taxo_store as store

log = logging.getLogger("vera.chunk_taxo")
TTL_S = 300.0
_LOCK = threading.Lock()
_CACHE: dict = {"at": None, "val": None}


def _empty(state: str) -> dict:
    return {"state": state, "version": None, "metrics": {}, "display_ok": False, "reasons": [], "last_run": None}


def reset_for_tests() -> None:
    with _LOCK:
        _CACHE.update(at=None, val=None)


def _read(engine_factory, env) -> dict:
    from vera.db import get_role_engine  # noqa: PLC0415
    factory = engine_factory
    if factory is None:
        from sqlalchemy import create_engine  # noqa: PLC0415
        factory = create_engine

    def make(url, **kw):
        return factory(url, pool_size=1, max_overflow=0, **kw,
                       connect_args={"connect_timeout": 3, "options": "-c statement_timeout=3000"})
    eng = get_role_engine("rw", env=env, engine_factory=make)
    try:
        with eng.connect() as conn:
            if not store.probe_schema(conn)[2]:
                return _empty("absent")
            row = store.run_sql(conn, "job_run_latest").first()
    finally:
        try:
            eng.dispose()
        except Exception:  # noqa: BLE001
            pass
    if row is None:
        return _empty("none")
    from vera.chunk_taxo_job import _numeric_only  # noqa: PLC0415
    fin = row[1]
    return {"state": "ok", "version": str(row[0])[:128], "metrics": _numeric_only(row[2]) or {},
            "display_ok": row[3] is True, "reasons": [str(r)[:64] for r in (row[4] or [])][:16],
            "last_run": fin.isoformat() if hasattr(fin, "isoformat") else None}


def status(engine_factory=None, env=None, clock=time.monotonic, ttl: float = TTL_S) -> dict:
    """Cached latest-run status. Never raises."""
    try:
        with _LOCK:
            at = _CACHE["at"]
            if at is not None and clock() - at < ttl and _CACHE["val"] is not None:
                return dict(_CACHE["val"])
        try:
            val = _read(engine_factory, env if env is not None else os.environ)
        except Exception as exc:  # noqa: BLE001
            log.warning("chunk_taxo status read failed: CT_STATUS_ERROR (%s)", type(exc).__name__)
            val = _empty("absent")
        with _LOCK:
            _CACHE.update(at=clock(), val=val)
        return dict(val)
    except Exception:  # noqa: BLE001
        return _empty("absent")
