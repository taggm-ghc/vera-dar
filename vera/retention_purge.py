"""PURGE-WIRING-1105: daily in-app retention purge of the request trace (DESIGN PRIV-D2-KIT-1105 section 3).

Called from the request_trace writer thread (tick()). Contract:
  * flag off (retention_purge.enabled is not True): tick() returns at once; no engine is created, no connection opened;
  * at most one run per interval_s (default 86400); first run initial_delay_s after the writer starts;
  * never raises, never blocks a request (it runs on the writer thread, never on a request thread);
  * any error: log the exception class only, back off (initial_delay_s doubling, capped at interval_s);
  * the ONLY SQL is `SELECT events_deleted, requests_deleted FROM vera_vjay.purge_expired_trace()` (migration 018,
    not applied until the owner approves it), on a dedicated engine from the wo URL with its own statement_timeout
    (the flush engine's 2 s is too short); the engine is disposed after each run;
  * success: log line `vera_trace purge events=N requests=M` (counts only) and a trace-log event with stage
    retention_purge (no request id; the events table keys on request_id, so no DB row is written for it).
"""
from __future__ import annotations

import json
import logging
import os
import time

log = logging.getLogger("vera.trace")

PURGE_SQL = "SELECT events_deleted, requests_deleted FROM vera_vjay.purge_expired_trace()"
DEFAULTS = {"interval_s": 86400, "initial_delay_s": 300, "statement_timeout_ms": 60000}


def _num(cfg: dict, key: str) -> float:
    v = cfg.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0 < v <= 10**7):
        return float(DEFAULTS[key])
    return float(v)


class Purger:
    def __init__(self, pcfg, dcfg, factory=None, env=None, clock=time.monotonic):
        self.pcfg = pcfg if isinstance(pcfg, dict) else {}
        self.dcfg = dcfg if isinstance(dcfg, dict) else {}
        self._factory = factory
        self._env = env if env is not None else os.environ
        self._clock = clock
        self.on = self.pcfg.get("enabled") is True
        self.interval = _num(self.pcfg, "interval_s")
        self.delay = _num(self.pcfg, "initial_delay_s")
        self.fails = 0
        self.runs = 0
        self.next_due = clock() + self.delay if self.on else None

    def tick(self, emit=None) -> bool:
        """Run the purge if due. Returns True only when a purge completed. Never raises."""
        if not self.on:
            return False
        try:
            if self._clock() < self.next_due:
                return False
            return self._run(emit)
        except Exception as exc:  # noqa: BLE001
            self._backoff(exc)
            return False

    def _backoff(self, exc) -> None:
        self.fails += 1
        wait = min(self.interval, self.delay * (2 ** min(self.fails - 1, 20)))
        self.next_due = self._clock() + wait
        log.warning("vera_trace purge failed: RT_PURGE_ERROR (%s), retry in %ds", type(exc).__name__, int(wait))

    def _run(self, emit) -> bool:
        url = (self._env.get(str(self.dcfg.get("db_url_env_wo") or "VERA_DB_URL_WO")) or "").strip()
        if not url:
            raise RuntimeError("no_wo_url")
        factory = self._factory
        if factory is None:
            from sqlalchemy import create_engine as factory
        ms = int(_num(self.pcfg, "statement_timeout_ms"))
        eng = factory(url, hide_parameters=True, pool_size=1, max_overflow=0,
                      connect_args={"connect_timeout": int(self.dcfg.get("connect_timeout_s") or 2),
                                    "options": f"-c statement_timeout={ms}"})
        try:
            from sqlalchemy import text
            with eng.begin() as conn:
                row = conn.execute(text(PURGE_SQL)).first()
            ev, rq = (int(row[0]), int(row[1])) if row is not None else (0, 0)
        finally:
            try:
                eng.dispose()
            except Exception:  # noqa: BLE001
                pass
        self.fails = 0
        self.runs += 1
        self.next_due = self._clock() + self.interval
        log.info("vera_trace purge events=%d requests=%d", ev, rq)
        if emit is not None:
            emit({"stage": "retention_purge", "code": "purged",
                  "counts": {"events_deleted": ev, "requests_deleted": rq}})
        return True
