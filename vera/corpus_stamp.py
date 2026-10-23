"""corpus_added_at backfill (FRESH-AMEND-1023, owner refinement 2026-10-23, verbatim): corpus_added_at "is a per-chunk stamp
set when the chunk is written to the corpus. It drives the 24-hour freshness trigger. If it's missing or invalid, that counts
as unknown and triggers research AND triggers setting that value to the then now()"; and for an imprecise (partial) value
"its value in this case should be revised to the then now()".

backfill(stamps, now, stamper) for the stamp dicts vera.freshness listed as missing/invalid on chunks USED in this request:
  1. in memory: each dict's corpus_added_at is set to `now` (the request's injectable clock) in place. The dicts are the
     served corpus's own chunk stamps (shared by reference), so the next request sees the repaired value at once.
     Re-checked first: a stamp that became valid meanwhile (another request) is never overwritten.
  2. persisted through the existing app write path (the vera_pipeline_wo account, env VERA_DB_URL_WO; acquisition's
     db_url_env_wo): ONE conditional single-row UPDATE per stored chunk (evidence_spans.span_id > 0), touching only a
     NULL / future created_at (idempotent; a valid value is never overwritten; never bulk). Without the WO URL nothing is
     written and the record says persist "skipped" with the reason. In-memory-only chunks (no stored span id) are
     recorded as not persistable.
Valid stamps are immutable: no path changes a valid corpus_added_at (in memory: re-checked before writing; in the DB: the
UPDATE's WHERE only matches NULL / future created_at). Kill switch (review D1): the in-memory stamp always applies; the
DB persist step is skipped (status "skipped", reason "kill switch") while VERA_AUTOACQ_DISABLED (acq_config.killed) is on,
and when the WO URL is absent. Privileges (review D2): persistence runs ONLY with the existing app-path permissions; no
grant is requested. The WO account vera_vjay_wo has INSERT and SELECT(span_id) only on evidence_spans
(db/accounts/create_vera_accounts.sql:39-58); this conditional UPDATE would need UPDATE(created_at) AND SELECT(created_at)
there, so today the persist step is recorded as "failed" (reason "no privilege") harmlessly and the in-memory stamp
still applies. In production evidence_spans.created_at is NOT NULL DEFAULT now() (db/migrations/003), so persisted
backfills should be rare.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta

from vera import freshness as fz

SCHEMA = "vera_vjay"
SQL_STAMP = (f"UPDATE {SCHEMA}.evidence_spans SET created_at = CAST(:now AS timestamptz) "
             f"WHERE span_id = :span_id AND (created_at IS NULL OR created_at > CAST(:limit AS timestamptz))")


class SpanStamper:
    """Conditional single-row UPDATE through an injected SQLAlchemy engine (the WO app account)."""

    def __init__(self, engine):
        self.engine = engine

    def stamp(self, span_id: int, now: datetime, skew_s: float) -> int:
        from sqlalchemy import text
        with self.engine.begin() as c:
            r = c.execute(text(SQL_STAMP), {"now": now.isoformat(), "span_id": int(span_id),
                                            "limit": (now + timedelta(seconds=skew_s)).isoformat()})
            return int(getattr(r, "rowcount", 0) or 0)


_ENGINES: dict = {}  # review D4: one engine per WO URL, reused (no per-call engine)
_ENG_LOCK = threading.Lock()


def default_stamper(acq: dict | None, env=os.environ):
    """(stamper | None, skip reason). Uses only the existing env var NAMES from config/auto_acquire.json."""
    from vera import acq_config
    if acq_config.killed(acq, env):
        return None, "kill switch"
    name = str((acq or {}).get("db_url_env_wo") or "VERA_DB_URL_WO")
    url = (env.get(name) or "").strip()
    if not url:
        return None, f"{name} not set"
    try:
        with _ENG_LOCK:
            eng = _ENGINES.get(url)
            if eng is None:
                from sqlalchemy import create_engine
                ms = int((acq or {}).get("db_statement_timeout_ms") or 15000)
                eng = create_engine(url, hide_parameters=True, pool_pre_ping=True, pool_size=1, max_overflow=0,
                                    connect_args={"connect_timeout": 5, "options": f"-c statement_timeout={ms}"})
                _ENGINES[url] = eng
    except Exception as exc:  # noqa: BLE001
        return None, f"write engine unavailable ({type(exc).__name__})"
    return SpanStamper(eng), ""


def reset_for_tests() -> None:
    with _ENG_LOCK:
        for e in _ENGINES.values():
            try:
                e.dispose()
            except Exception:  # noqa: BLE001
                pass
        _ENGINES.clear()


def _no_privilege(exc) -> bool:
    for e in (exc, getattr(exc, "orig", None)):
        if getattr(e, "pgcode", None) == "42501" or getattr(e, "sqlstate", None) == "42501":
            return True
    return False


def persistable(s) -> bool:
    """A stored chunk (positive evidence_spans.span_id) can be persisted; in-memory chunks cannot."""
    sid = s.get("span_id") if isinstance(s, dict) else None
    return isinstance(sid, int) and not isinstance(sid, bool) and sid > 0


def backfill(stamps, now: datetime, cfg: dict | None, stamper=None, skip_reason: str = "") -> dict:
    """Never raises. Returns the research-record entry {count, chunk_ids, at, persist: {...}}."""
    skew_s = float((cfg or {}).get("freshness_future_skew_seconds", fz.DEFAULT_FUTURE_SKEW_S))
    skew = timedelta(seconds=skew_s)
    now_iso = fz._utc(now).isoformat()
    seen, done, ids = set(), [], []
    for s in stamps or ():
        if not isinstance(s, dict) or id(s) in seen:
            continue
        seen.add(id(s))
        if fz.stamp_valid(s.get("corpus_added_at"), now, skew):
            continue  # became valid meanwhile: never overwrite a valid value
        s["corpus_added_at"] = now_iso
        s["corpus_added_backfilled"] = True
        done.append(s)
        ids.append(s.get("span_id") if s.get("span_id") is not None else s.get("chunk_id") or s.get("document_id"))
    persist = {"status": "not_needed", "updated": 0, "not_persistable": 0, "reason": ""}
    stored = [s for s in done if persistable(s)]
    persist["not_persistable"] = len(done) - len(stored)
    if stored:
        if stamper is None:
            persist.update(status="skipped", reason=skip_reason or "no write path")
        else:
            try:
                persist["updated"] = sum(stamper.stamp(s["span_id"], fz._utc(now), skew_s) for s in stored)
                persist["status"] = "done"
            except Exception as exc:  # noqa: BLE001
                logging.getLogger("vera").warning("corpus_added_at backfill not persisted (%s)", type(exc).__name__)
                persist.update(status="failed", reason="no privilege" if _no_privilege(exc) else type(exc).__name__)
    return {"count": len(done), "chunk_ids": ids, "at": now_iso, "persist": persist}
