"""CHUNK-USAGE-1110 (WP11, work/CHUNK-VALUE-1110/plan.md section 6b): per-chunk match-to-use counters. Best-effort telemetry.

What is counted (per stored retrieval chunk = (evidence_spans partition span_id, chunk_start, chunk_end), document offsets):
  matched   the chunk cleared the question's relevance rules in fulldoc_admission.admit (scored candidate);
  selected  it was admitted as evidence for the writer;
  used      it was cited in a delivered answer (vera.fulldoc_answer, after the claim check).
Only counts and UTC dates reach the database (vera_vjay.chunk_use, migration 025, owner-run). No question text, run id, session
or time of day is ever held here or stored. Chunks without a positive integer partition id (bundle corpora, documents fetched
inline before storage) have no key and are skipped.

Safety contract (never slows or fails an ask):
  * note_* only append to a bounded in-memory dict under a lock taken with a short timeout; any error or a full queue drops the
    increment. No database call, no network and no thread start on the request path except one lazy daemon-thread start.
  * a background daemon thread flushes batches (executemany of one fixed upsert; a failed batch is dropped, never retried) and
    probes `to_regclass('vera_vjay.chunk_use')` first: until the table exists the queue is discarded, so shipping this code before
    migration 025 is inert (flag default on, effective only if the table exists).
  * demotion (config demote.weight, 0 = OFF): a rank-only multiplier on the admission sort key for chunks with
    matched_count >= k_min and used_count = 0, from an in-memory snapshot refreshed by the background thread; it never removes a
    candidate. weight 0 means no snapshot is read at all.
Tests inject fakes with set_*_for_tests; no test needs a database.
"""
from __future__ import annotations

import contextvars
import json
import logging
import math
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("vera.chunk_usage")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "chunk_usage.json"
LOCK_TIMEOUT_S = 0.05

SQL_PROBE = "SELECT pg_catalog.to_regclass('vera_vjay.chunk_use') IS NOT NULL"
SQL_UPSERT = (
    "INSERT INTO vera_vjay.chunk_use (span_id, chunk_start, chunk_end, matched_count, selected_count, used_count) "
    "VALUES (:span_id, :chunk_start, :chunk_end, :m, :s, :u) "
    "ON CONFLICT (span_id, chunk_start, chunk_end) DO UPDATE SET "
    "matched_count = vera_vjay.chunk_use.matched_count + EXCLUDED.matched_count, "
    "selected_count = vera_vjay.chunk_use.selected_count + EXCLUDED.selected_count, "
    "used_count = vera_vjay.chunk_use.used_count + EXCLUDED.used_count, "
    "last_seen_day = (now() AT TIME ZONE 'utc')::date")
SQL_SNAPSHOT = (
    "SELECT span_id, chunk_start, chunk_end, matched_count FROM vera_vjay.chunk_use "
    "WHERE used_count = 0 AND matched_count >= :k_min ORDER BY matched_count DESC, span_id, chunk_start LIMIT :max_rows")

DEFAULTS = {"enabled": False, "queue_max": 2000, "batch_max": 200, "flush_s": 5.0, "probe_s": 600.0,
            "connect_timeout_s": 3, "statement_timeout_ms": 3000}
DEMOTE_DEFAULTS = {"weight": 0.0, "k_min": 50, "threshold": 0.0, "refresh_s": 900.0, "max_rows": 50000}
_BOUNDS = {"queue_max": (10, 100000), "batch_max": (1, 5000), "flush_s": (0.5, 3600), "probe_s": (5, 86400),
           "connect_timeout_s": (1, 30), "statement_timeout_ms": (100, 60000)}
_DBOUNDS = {"weight": (0.0, 100.0), "k_min": (1, 10 ** 9), "threshold": (0.0, 100.0), "refresh_s": (5, 86400),
            "max_rows": (1, 1000000)}

_LOCK = threading.Lock()
_STATE: dict = {"cfg": None, "pending": {}, "thread": None, "table_ok": None, "probed_at": 0.0, "engine": None,
                "engine_factory": None, "snap": {}, "snap_at": None, "snap_busy": False, "clock": time.monotonic,
                "dropped": 0, "flushed": 0, "failed_batches": 0}
_SUPPRESS: contextvars.ContextVar = contextvars.ContextVar("chunk_usage_suppress", default=False)


# ------------------------------------------------------------------------------------------------ config
def _num(v, lo, hi):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and lo <= v <= hi


def _clean(raw) -> dict:
    c = dict(DEFAULTS)
    d = dict(DEMOTE_DEFAULTS)
    if isinstance(raw, dict):
        if raw.get("enabled") is True:
            c["enabled"] = True
        for k, (lo, hi) in _BOUNDS.items():
            if _num(raw.get(k), lo, hi):
                c[k] = raw[k]
        dr = raw.get("demote")
        if isinstance(dr, dict):
            for k, (lo, hi) in _DBOUNDS.items():
                if _num(dr.get(k), lo, hi):
                    d[k] = dr[k]
    c["demote"] = d
    return c


def config() -> dict:
    """Validated config, cached. A missing or bad file means OFF (enabled false, demotion weight 0). Never raises."""
    cfg = _STATE["cfg"]
    if cfg is None:
        try:
            cfg = _clean(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            cfg = _clean({})
        _STATE["cfg"] = cfg
    return cfg


def enabled() -> bool:
    try:
        return config().get("enabled") is True
    except Exception:  # noqa: BLE001
        return False


def demotion_weight() -> float:
    try:
        return float(config()["demote"]["weight"]) if enabled() else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


# ------------------------------------------------------------------------------------------------ keys
def chunk_key(stamps, start, end):
    """(partition span_id, start, end) for the retrieval chunk [start, end): the stored partition stamp that holds the chunk's
    midpoint. None when there is no positive integer partition id (document-level stamps, bundle or inline documents)."""
    if not isinstance(start, int) or not isinstance(end, int) or isinstance(start, bool) or isinstance(end, bool) or end <= start:
        return None
    mid = (start + end) // 2
    for x in stamps or ():
        if not isinstance(x, dict):
            continue
        sid, s, e = x.get("span_id"), x.get("start"), x.get("end")
        if isinstance(sid, int) and not isinstance(sid, bool) and sid > 0 and isinstance(s, int) and isinstance(e, int) \
                and not isinstance(s, bool) and not isinstance(e, bool) and s <= mid < e:
            return (sid, start, end)
    return None


def _range_of(span_id):
    """(start, end) of a corpus span id 'doc:start-end', else None."""
    try:
        _, _, rng = str(span_id).rpartition(":")
        a, _, b = rng.partition("-")
        return int(a), int(b)
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------ recording (request path)
@contextmanager
def suppressed():
    """Count nothing inside (non-answer probes such as the relevance probe of ask_service)."""
    tok = _SUPPRESS.set(True)
    try:
        yield
    finally:
        _SUPPRESS.reset(tok)


def _note(keys, slot: int) -> None:
    """Add 1 to counter `slot` (0 matched, 1 selected, 2 used) of each distinct key. Drops silently on any problem."""
    try:
        if not keys or _SUPPRESS.get() or not enabled():
            return
        cfg = config()
        if not _LOCK.acquire(timeout=LOCK_TIMEOUT_S):
            _STATE["dropped"] += len(keys)
            return
        try:
            pend = _STATE["pending"]
            for k in set(keys):
                row = pend.get(k)
                if row is None:
                    if len(pend) >= int(cfg["queue_max"]):
                        _STATE["dropped"] += 1
                        continue
                    row = pend[k] = [0, 0, 0]
                row[slot] += 1
        finally:
            _LOCK.release()
        _ensure_thread()
    except Exception:  # noqa: BLE001  never raises into /ask
        pass


def note_admission(corpus, matched, admitted, stamp_holder) -> None:
    """Hook in fulldoc_admission.admit: `matched` = candidate dicts that cleared the relevance rules, `admitted` = the final
    admitted entries; `stamp_holder(corpus, document_id_str)` is that module's stored-chunk-stamp accessor."""
    try:
        if _SUPPRESS.get() or not enabled():
            return
        mk, sk = [], []
        for c in matched or ():
            k = chunk_key(stamp_holder(corpus, str(c["document_id"]))["corpus_stamps"], c.get("start"), c.get("end"))
            if k is not None:
                mk.append(k)
        for a in admitted or ():
            sk.extend(_entry_keys(a, corpus, stamp_holder))
        _note(mk, 0)
        _note(sk, 1)
    except Exception:  # noqa: BLE001
        pass


def _entry_keys(a, corpus=None, stamp_holder=None) -> list:
    """Keys of the retrieval chunks behind one admitted entry (a merged entry expands to its member chunks)."""
    stamps = a.get("corpus_chunks")
    if stamp_holder is not None and corpus is not None and not stamps:
        stamps = stamp_holder(corpus, str(a["document_id"]))["corpus_stamps"]
    out = []
    ids = a.get("merged_span_ids") or (a.get("span_id"),)
    for sid in ids:
        rng = _range_of(sid)
        k = chunk_key(stamps, rng[0], rng[1]) if rng else None
        if k is not None:
            out.append(k)
    if not out:
        k = chunk_key(stamps, a.get("start"), a.get("end"))
        if k is not None:
            out.append(k)
    return out


def note_used(ev, citations) -> None:
    """Hook in fulldoc_answer after the claim check: the admitted entries whose span_id a delivered citation names."""
    try:
        if _SUPPRESS.get() or not enabled():
            return
        cited = {str(c.get("span_id")) for c in citations or () if isinstance(c, dict) and c.get("span_id")}
        if not cited:
            return
        keys = []
        for a in getattr(ev, "admitted", ()) or ():
            if isinstance(a, dict) and str(a.get("span_id")) in cited:
                keys.extend(_entry_keys(a))
        _note(keys, 2)
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------------------------------------ demotion (request path: dict lookup only)
def demoter(stamp_holder, corpus):
    """None when demotion is off (weight 0, flag off, or no snapshot yet): admit then changes nothing. Otherwise f(c) -> a factor
    in (0, 1] for a candidate dict; it multiplies the (negative) relevance sort key."""
    try:
        w = demotion_weight()
        if w <= 0:
            return None
        _refresh_snapshot_async()
        snap = _STATE["snap"]
        if not snap:
            return None
        d = config()["demote"]
        thr = float(d["threshold"])

        def f(c) -> float:
            try:
                k = chunk_key(stamp_holder(corpus, str(c["document_id"]))["corpus_stamps"], c.get("start"), c.get("end"))
                m = snap.get(k) if k is not None else None
                if m is None:
                    return 1.0
                return 1.0 / (1.0 + w * max(0.0, math.log(m + 1) - thr))  # used_count = 0 in the snapshot, so ln((m+1)/1)
            except Exception:  # noqa: BLE001
                return 1.0
        return f
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------ background thread
def _ensure_thread() -> None:
    t = _STATE["thread"]
    if t is not None and t.is_alive():
        return
    with _LOCK:
        t = _STATE["thread"]
        if t is not None and t.is_alive():
            return
        t = threading.Thread(target=_loop, name="chunk-usage-flush", daemon=True)
        _STATE["thread"] = t
        t.start()


def _loop() -> None:
    while True:
        try:
            flush_once()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(float(config()["flush_s"]))


def _engine():
    if _STATE["engine"] is None:
        factory = _STATE["engine_factory"]
        cfg = config()
        if factory is None:
            from vera.db import get_role_engine  # noqa: PLC0415

            def make(url, **kw):
                from sqlalchemy import create_engine  # noqa: PLC0415
                return create_engine(url, pool_size=1, max_overflow=0, hide_parameters=True, **{k: v for k, v in kw.items() if k != "hide_parameters"},
                                     connect_args={"connect_timeout": int(cfg["connect_timeout_s"]),
                                                   "options": f"-c statement_timeout={int(cfg['statement_timeout_ms'])}"})
            _STATE["engine"] = get_role_engine("rw", env=os.environ, engine_factory=make)
        else:
            _STATE["engine"] = factory()
    return _STATE["engine"]


def _drop_engine() -> None:
    eng, _STATE["engine"] = _STATE["engine"], None
    try:
        if eng is not None:
            eng.dispose()
    except Exception:  # noqa: BLE001
        pass


def _table_exists() -> bool:
    """to_regclass probe, cached probe_s (re-probed so a table created later switches the writer on without a restart)."""
    now = _STATE["clock"]()
    if _STATE["table_ok"] is not None and now - _STATE["probed_at"] < float(config()["probe_s"]):
        return bool(_STATE["table_ok"])
    ok = False
    try:
        from sqlalchemy import text  # noqa: PLC0415
        with _engine().connect() as conn:
            ok = bool(conn.execute(text(SQL_PROBE)).scalar())
    except Exception as exc:  # noqa: BLE001
        log.info("chunk_usage probe failed (%s)", type(exc).__name__)
        _drop_engine()
    _STATE["table_ok"], _STATE["probed_at"] = ok, now
    return ok


def flush_once() -> int:
    """One flush pass (the background thread calls it; tests call it directly). Returns rows written. Never raises."""
    try:
        if not enabled():
            return 0
        cfg = config()
        if not _table_exists():
            with _LOCK:
                _STATE["pending"].clear()  # no table: nothing is kept (inert until migration 025 is applied)
            return 0
        written = 0
        while True:
            with _LOCK:
                pend = _STATE["pending"]
                keys = sorted(pend)[: int(cfg["batch_max"])]  # sorted: stable lock order across instances
                batch = [(k, pend.pop(k)) for k in keys]
            if not batch:
                break
            params = [{"span_id": k[0], "chunk_start": k[1], "chunk_end": k[2], "m": v[0], "s": v[1], "u": v[2]} for k, v in batch]
            try:
                from sqlalchemy import text  # noqa: PLC0415
                with _engine().begin() as conn:
                    conn.execute(text(SQL_UPSERT), params)
                written += len(params)
                _STATE["flushed"] += len(params)
            except Exception as exc:  # noqa: BLE001  a failed batch is dropped, never retried
                _STATE["failed_batches"] += 1
                _STATE["dropped"] += len(params)
                log.info("chunk_usage batch dropped (%s)", type(exc).__name__)
                _drop_engine()
                break
        return written
    except Exception:  # noqa: BLE001
        return 0


def _refresh_snapshot_async() -> None:
    now = _STATE["clock"]()
    d = config()["demote"]
    if _STATE["snap_busy"] or (_STATE["snap_at"] is not None and now - _STATE["snap_at"] < float(d["refresh_s"])):
        return
    _STATE["snap_busy"] = True
    threading.Thread(target=refresh_snapshot, name="chunk-usage-snap", daemon=True).start()


def refresh_snapshot() -> int:
    """Load {key: matched_count} for chunks with used_count = 0 and matched_count >= k_min. On any error keep the old snapshot."""
    n = 0
    try:
        d = config()["demote"]
        if _table_exists():
            from sqlalchemy import text  # noqa: PLC0415
            with _engine().connect() as conn:
                rows = conn.execute(text(SQL_SNAPSHOT), {"k_min": int(d["k_min"]), "max_rows": int(d["max_rows"])}).fetchall()
            _STATE["snap"] = {(int(r[0]), int(r[1]), int(r[2])): int(r[3]) for r in rows}
            n = len(rows)
    except Exception as exc:  # noqa: BLE001
        log.info("chunk_usage snapshot failed (%s)", type(exc).__name__)
        _drop_engine()
    finally:
        _STATE["snap_at"] = _STATE["clock"]()
        _STATE["snap_busy"] = False
    return n


# ------------------------------------------------------------------------------------------------ tests
def reset_for_tests() -> None:
    _STATE.update({"cfg": None, "pending": {}, "thread": None, "table_ok": None, "probed_at": 0.0, "engine": None,
                   "engine_factory": None, "snap": {}, "snap_at": None, "snap_busy": False, "clock": time.monotonic,
                   "dropped": 0, "flushed": 0, "failed_batches": 0})


def set_config_for_tests(raw) -> None:
    _STATE["cfg"] = _clean(raw)


def set_engine_factory_for_tests(factory) -> None:
    _STATE["engine_factory"] = factory
    _STATE["engine"] = None


def pending_for_tests() -> dict:
    return {k: tuple(v) for k, v in _STATE["pending"].items()}
