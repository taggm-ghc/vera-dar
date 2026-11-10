"""CHUNK-USAGE-1110 (WP11, work/CHUNK-VALUE-1110/plan.md section 6b): per-chunk, per-association match-to-use counters. Best-effort.

Rows are keyed (partition span_id, chunk_start, chunk_end, node_id) in vera_vjay.chunk_use (migration 025, owner-run). node_id 0 is the
per-chunk TOTAL row (every event, with or without routing); node_id > 0 is one routing association (the node, taken from a candidate's
"domain:<id>" branch or its "nodes" list, through which the chunk matched). What is counted: matched = cleared the question's relevance
rules in fulldoc_admission.admit; selected = admitted as evidence; used = cited in a delivered answer. Counts and UTC dates only; no
question text, run id, session or time of day. Chunks without a positive integer partition id (bundle corpora, inline documents) are skipped.

Association weight (owner 2026-11-10: lower the weight, then soft-remove the association, never purge the chunk):
  * initial_weight = the baseline similarity of the association; weight starts equal to it and never exceeds it. The taxonomy job stores
    its chunk-link score as score_kind 'cosine' = round(clamp(1 - cosine distance, -1, 1), 6) (vera/chunk_taxo_job.py _similarity;
    NULL when unknown; it is computed per stored partition against the assigned node, not per 400-character chunk). chunk_usage does
    not read the tax tables: the baseline comes from set_similarity_source(fn) (the router supplies fn(span_id, node_id) -> float|None);
    None or out of range => 1.0 (>= 0 clamps to MIN_INITIAL). refresh_initial_weights() re-bases after a re-link and keeps the decay ratio.
  * after assoc.k matches through a node with no use (matched_since_use >= k) each further match multiplies weight by assoc.decay, not
    below assoc.floor * initial_weight; a use moves weight back toward initial_weight by the fraction assoc.restore per use, never above it.
  * when weight < assoc.remove_below (relative to initial_weight by default; assoc.relative false = absolute) removed_at is set (SOFT);
    a use that lifts weight back to or above the threshold clears it; restore_association() clears it by hand. Nothing here deletes a
    row or a chunk: the app roles hold no DELETE and the chunk stays globally searchable.
  * step() is the Python statement of the SQL upsert's arithmetic; the tests pin it and the rehearsal checks the SQL against it.
Read API for the router: association(), weight_ratio(), removed_associations() (an in-memory snapshot refreshed off the request path).

Safety contract (never slows or fails an ask): note_* only append to a bounded in-memory dict under a short-timeout lock; any error or a
full queue drops the increment; a background daemon thread probes `to_regclass('vera_vjay.chunk_use')`, then flushes bounded batches
(a failed batch is dropped, never retried); until the table exists the queue is discarded, so shipping this code before migration 025 is
inert. Demotion (config demote.scale, 0 = OFF) is rank-only: a chunk whose every matching association has a lowered weight sorts later.
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
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("vera.chunk_usage")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "chunk_usage.json"
LOCK_TIMEOUT_S = 0.05

SQL_PROBE = "SELECT pg_catalog.to_regclass('vera_vjay.chunk_use') IS NOT NULL"
_T = "vera_vjay.chunk_use"
_MSU = f"(CASE WHEN EXCLUDED.used_count > 0 THEN 0 ELSE {_T}.matched_since_use + EXCLUDED.matched_count END)"
_NEWW = (f"(CASE WHEN EXCLUDED.node_id = 0 THEN 1.0 "
         f"WHEN EXCLUDED.used_count > 0 THEN LEAST({_T}.initial_weight, {_T}.weight + ({_T}.initial_weight - {_T}.weight) * "
         f"(1.0 - power(1.0 - CAST(:restore AS double precision), EXCLUDED.used_count))) "
         f"WHEN {_MSU} >= CAST(:k AS bigint) THEN GREATEST(CAST(:floor AS double precision) * {_T}.initial_weight, "
         f"{_T}.weight * power(CAST(:decay AS double precision), LEAST(EXCLUDED.matched_count, {_MSU} - CAST(:k AS bigint) + 1))) "
         f"ELSE {_T}.weight END)")
_THR = (f"(CASE WHEN CAST(:rel AS boolean) THEN CAST(:thr AS double precision) * {_T}.initial_weight "
        f"ELSE CAST(:thr AS double precision) END)")
SQL_UPSERT = (
    f"INSERT INTO {_T} (span_id, chunk_start, chunk_end, node_id, matched_count, selected_count, used_count, matched_since_use, "
    "initial_weight, weight) VALUES (:span_id, :chunk_start, :chunk_end, :node_id, :m, :s, :u, :msu0, :init, :init) "
    "ON CONFLICT (span_id, chunk_start, chunk_end, node_id) DO UPDATE SET "
    f"matched_count = {_T}.matched_count + EXCLUDED.matched_count, selected_count = {_T}.selected_count + EXCLUDED.selected_count, "
    f"used_count = {_T}.used_count + EXCLUDED.used_count, matched_since_use = {_MSU}, weight = {_NEWW}, "
    f"removed_at = (CASE WHEN EXCLUDED.node_id = 0 THEN NULL WHEN {_NEWW} < {_THR} THEN COALESCE({_T}.removed_at, now()) "
    f"WHEN EXCLUDED.used_count > 0 THEN NULL ELSE {_T}.removed_at END), "
    "last_seen_day = (now() AT TIME ZONE 'utc')::date")
SQL_SNAPSHOT = (
    f"SELECT span_id, chunk_start, chunk_end, node_id, initial_weight, weight, (removed_at IS NOT NULL) FROM {_T} "
    "WHERE node_id > 0 AND (weight < initial_weight OR removed_at IS NOT NULL) ORDER BY node_id, span_id, chunk_start LIMIT :max_rows")
SQL_REFRESH_INIT = (
    f"UPDATE {_T} SET weight = LEAST(CAST(:new AS double precision), CASE WHEN initial_weight > 0 THEN weight / initial_weight * CAST(:new AS double precision) "
    "ELSE CAST(:new AS double precision) END), initial_weight = CAST(:new AS double precision) "
    "WHERE span_id = :span_id AND node_id = :node_id AND node_id > 0")
SQL_RESTORE = (f"UPDATE {_T} SET weight = initial_weight, removed_at = NULL, matched_since_use = 0 "
               "WHERE span_id = :span_id AND chunk_start = :chunk_start AND chunk_end = :chunk_end AND node_id = :node_id AND node_id > 0")

MIN_INITIAL = 0.01
DEFAULTS = {"enabled": False, "queue_max": 2000, "batch_max": 200, "flush_s": 5.0, "probe_s": 600.0,
            "connect_timeout_s": 3, "statement_timeout_ms": 3000}
ASSOC_DEFAULTS = {"k": 50, "decay": 0.9, "floor": 0.05, "restore": 0.5, "remove_below": 0.2, "relative": True}
DEMOTE_DEFAULTS = {"scale": 0.0}
SNAP_DEFAULTS = {"refresh_s": 900.0, "max_rows": 50000}
_BOUNDS = {"queue_max": (10, 100000), "batch_max": (1, 5000), "flush_s": (0.5, 3600), "probe_s": (5, 86400),
           "connect_timeout_s": (1, 30), "statement_timeout_ms": (100, 60000)}
_ABOUNDS = {"k": (1, 10 ** 9), "decay": (0.01, 0.999), "floor": (0.0, 0.9), "restore": (0.0, 1.0), "remove_below": (0.0, 1.0)}
_SBOUNDS = {"refresh_s": (5, 86400), "max_rows": (1, 1000000)}

_LOCK = threading.Lock()
_NODES: "OrderedDict[tuple, frozenset]" = OrderedDict()  # chunk key -> nodes of its latest admission (for 'used' attribution)
_NODES_MAX = 512
_STATE: dict = {"sim_fn": None, "cfg": None, "pending": {}, "thread": None, "table_ok": None, "probed_at": 0.0, "engine": None,
                "engine_factory": None, "snap": {}, "snap_at": None, "snap_busy": False, "clock": time.monotonic,
                "dropped": 0, "flushed": 0, "failed_batches": 0}
_SUPPRESS: contextvars.ContextVar = contextvars.ContextVar("chunk_usage_suppress", default=False)


# ------------------------------------------------------------------------------------------------ config
def _num(v, lo, hi):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and lo <= v <= hi


def _clean(raw) -> dict:
    c = dict(DEFAULTS)
    d, a, sn = dict(DEMOTE_DEFAULTS), dict(ASSOC_DEFAULTS), dict(SNAP_DEFAULTS)
    if isinstance(raw, dict):
        if raw.get("enabled") is True:
            c["enabled"] = True
        for k, (lo, hi) in _BOUNDS.items():
            if _num(raw.get(k), lo, hi):
                c[k] = raw[k]
        ar = raw.get("assoc")
        if isinstance(ar, dict):
            for k, (lo, hi) in _ABOUNDS.items():
                if _num(ar.get(k), lo, hi):
                    a[k] = ar[k]
            if ar.get("relative") is False:
                a["relative"] = False
        dr = raw.get("demote")
        if isinstance(dr, dict) and _num(dr.get("scale"), 0.0, 100.0):
            d["scale"] = float(dr["scale"])
        sr = raw.get("snapshot")
        if isinstance(sr, dict):
            for k, (lo, hi) in _SBOUNDS.items():
                if _num(sr.get(k), lo, hi):
                    sn[k] = sr[k]
    a["floor"] = min(a["floor"], a["remove_below"] * 0.5) if a["relative"] else a["floor"]  # removal must stay reachable
    c.update(assoc=a, demote=d, snapshot=sn)
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


def demotion_scale() -> float:
    try:
        return float(config()["demote"]["scale"]) if enabled() else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def step(state: dict, m: int, u: int, cfg: dict | None = None) -> dict:
    """The upsert's arithmetic for one node row. state = {initial, weight, msu, removed}; m matches and u uses in the batch.
    Returns the new state. Pure; the SQL (SQL_UPSERT) is checked against it in the rehearsal."""
    a = (cfg or config())["assoc"]
    init, w, msu, rem = float(state["initial"]), float(state["weight"]), int(state["msu"]), bool(state["removed"])
    if u > 0:
        msu = 0
        w = min(init, w + (init - w) * (1.0 - (1.0 - a["restore"]) ** u))
    else:
        msu += m
        if msu >= a["k"]:
            w = max(a["floor"] * init, w * a["decay"] ** min(m, msu - a["k"] + 1))
    thr = a["remove_below"] * init if a["relative"] else a["remove_below"]
    if w < thr:
        rem = True
    elif u > 0:
        rem = False
    return {"initial": init, "weight": w, "msu": msu, "removed": rem}


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
    """Add 1 to counter `slot` (0 matched, 1 selected, 2 used) of each distinct 4-key. Drops silently on any problem."""
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


def _nodes_of(c) -> tuple:
    """Association nodes of a retrieval candidate: ints in c['nodes'] and 'domain:<id>' branches (ids > 0)."""
    out = set()
    try:
        for n in c.get("nodes") or ():
            if isinstance(n, int) and not isinstance(n, bool) and n > 0:
                out.add(n)
        for b in c.get("branches") or ():
            if isinstance(b, str) and b.startswith("domain:") and b[7:].isdigit() and int(b[7:]) > 0:
                out.add(int(b[7:]))
    except Exception:  # noqa: BLE001
        pass
    return tuple(sorted(out))


def _expand(k3, nodes) -> list:
    return [k3 + (0,)] + [k3 + (n,) for n in nodes]


def _remember_nodes(k3, nodes) -> None:
    try:
        with _LOCK:
            _NODES[k3] = frozenset(nodes)
            _NODES.move_to_end(k3)
            while len(_NODES) > _NODES_MAX:
                _NODES.popitem(last=False)
    except Exception:  # noqa: BLE001
        pass


def note_admission(corpus, matched, admitted, stamp_holder) -> None:
    """Hook in fulldoc_admission.admit: `matched` = candidate dicts that cleared the relevance rules, `admitted` = the final
    admitted entries; `stamp_holder(corpus, document_id_str)` is that module's stored-chunk-stamp accessor."""
    try:
        if _SUPPRESS.get() or not enabled():
            return
        mk, sk, by_range = [], [], {}
        for c in matched or ():
            k3 = chunk_key(stamp_holder(corpus, str(c["document_id"]))["corpus_stamps"], c.get("start"), c.get("end"))
            if k3 is None:
                continue
            nodes = _nodes_of(c)
            by_range[(str(c["document_id"]), c.get("start"), c.get("end"))] = nodes
            mk.extend(_expand(k3, nodes))
            _remember_nodes(k3, nodes)
        for a in admitted or ():
            for k3, rng in _entry_keys(a, corpus, stamp_holder, with_ranges=True):
                sk.extend(_expand(k3, by_range.get((str(a.get("document_id")),) + rng, ())))
        _note(mk, 0)
        _note(sk, 1)
    except Exception:  # noqa: BLE001
        pass


def _entry_keys(a, corpus=None, stamp_holder=None, with_ranges=False) -> list:
    """3-keys of the retrieval chunks behind one admitted entry (a merged entry expands to its member chunks)."""
    stamps = a.get("corpus_chunks")
    if stamp_holder is not None and corpus is not None and not stamps:
        stamps = stamp_holder(corpus, str(a["document_id"]))["corpus_stamps"]
    out = []
    ids = a.get("merged_span_ids") or (a.get("span_id"),)
    for sid in ids:
        rng = _range_of(sid)
        k = chunk_key(stamps, rng[0], rng[1]) if rng else None
        if k is not None:
            out.append((k, rng))
    if not out:
        k = chunk_key(stamps, a.get("start"), a.get("end"))
        if k is not None:
            out.append((k, (a.get("start"), a.get("end"))))
    return out if with_ranges else [k for k, _ in out]


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
                for k3 in _entry_keys(a):
                    with _LOCK:
                        nodes = tuple(sorted(_NODES.get(k3, ())))
                    keys.extend(_expand(k3, nodes))
        _note(keys, 2)
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------------------------------------ read API (router) and demotion
def set_similarity_source(fn) -> None:
    """fn(span_id, node_id) -> float | None: the association's baseline similarity (the taxonomy job's cosine link score). The writer
    calls it on the flush thread only. None clears it (baseline 1.0)."""
    _STATE["sim_fn"] = fn


def _baseline(span_id, node_id) -> float:
    if node_id <= 0:
        return 1.0
    try:
        fn = _STATE["sim_fn"]
        v = fn(span_id, node_id) if fn is not None else None
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            return min(1.0, max(MIN_INITIAL, float(v)))
    except Exception:  # noqa: BLE001
        pass
    return 1.0


def _snapshot() -> dict:
    """{(span_id, start, end, node_id): (initial_weight, weight, removed)} of decayed or removed associations. Never blocks."""
    if enabled():
        _refresh_snapshot_async()
    return _STATE["snap"]


def association(span_id, chunk_start, chunk_end, node_id):
    """None when the association has no lowered weight (treat as full weight, not removed); else a dict with initial_weight,
    weight, ratio (weight / initial_weight) and removed. Cheap, in-memory, safe on the request path."""
    try:
        v = _snapshot().get((int(span_id), int(chunk_start), int(chunk_end), int(node_id)))
        if v is None:
            return None
        init, w, rem = v
        return {"initial_weight": init, "weight": w, "ratio": (w / init if init > 0 else 0.0), "removed": bool(rem)}
    except Exception:  # noqa: BLE001
        return None


def weight_ratio(span_id, chunk_start, chunk_end, node_id) -> float:
    a = association(span_id, chunk_start, chunk_end, node_id)
    return 1.0 if a is None else float(a["ratio"])


def removed_associations() -> frozenset:
    """Keys (span_id, chunk_start, chunk_end, node_id) of soft-removed associations. The chunk itself is never removed."""
    try:
        return frozenset(k for k, v in _snapshot().items() if v[2])
    except Exception:  # noqa: BLE001
        return frozenset()


def demoter(stamp_holder, corpus):
    """None when demotion is off (scale 0, flag off, or no lowered association known): admit then changes nothing. Otherwise
    f(candidate) -> a factor in (0, 1] multiplying the (negative) relevance sort key. A candidate with no routing node, or with
    any node of full weight, is not demoted."""
    try:
        sc = demotion_scale()
        if sc <= 0:
            return None
        snap = _snapshot()
        if not snap:
            return None

        def f(c) -> float:
            try:
                nodes = _nodes_of(c)
                k3 = chunk_key(stamp_holder(corpus, str(c["document_id"]))["corpus_stamps"], c.get("start"), c.get("end"))
                if k3 is None or not nodes:
                    return 1.0
                ratio = max(((snap[k3 + (n,)][1] / snap[k3 + (n,)][0]) if k3 + (n,) in snap else 1.0) for n in nodes)
                return 1.0 / (1.0 + sc * (1.0 - min(1.0, max(0.0, ratio))))
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
            a = cfg["assoc"]
            g = {"k": int(a["k"]), "decay": float(a["decay"]), "floor": float(a["floor"]), "restore": float(a["restore"]),
                 "thr": float(a["remove_below"]), "rel": bool(a["relative"])}
            params = [dict(g, span_id=k[0], chunk_start=k[1], chunk_end=k[2], node_id=k[3], m=v[0], s=v[1], u=v[2],
                           msu0=(0 if v[2] > 0 else v[0]), init=_baseline(k[0], k[3])) for k, v in batch]
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
    d = config()["snapshot"]
    if _STATE["snap_busy"] or (_STATE["snap_at"] is not None and now - _STATE["snap_at"] < float(d["refresh_s"])):
        return
    _STATE["snap_busy"] = True
    threading.Thread(target=refresh_snapshot, name="chunk-usage-snap", daemon=True).start()


def refresh_snapshot() -> int:
    """Load decayed or removed associations into the in-memory snapshot. On any error keep the old snapshot."""
    n = 0
    try:
        d = config()["snapshot"]
        if _table_exists():
            from sqlalchemy import text  # noqa: PLC0415
            with _engine().connect() as conn:
                rows = conn.execute(text(SQL_SNAPSHOT), {"max_rows": int(d["max_rows"])}).fetchall()
            _STATE["snap"] = {(int(r[0]), int(r[1]), int(r[2]), int(r[3])): (float(r[4]), float(r[5]), bool(r[6])) for r in rows}
            n = len(rows)
    except Exception as exc:  # noqa: BLE001
        log.info("chunk_usage snapshot failed (%s)", type(exc).__name__)
        _drop_engine()
    finally:
        _STATE["snap_at"] = _STATE["clock"]()
        _STATE["snap_busy"] = False
    return n


def _exec_updates(sql: str, params: list) -> int:
    """Run one fixed UPDATE per param dict in one transaction (job or owner side, not the request path). Never raises."""
    try:
        if not enabled() or not params or not _table_exists():
            return 0
        from sqlalchemy import text  # noqa: PLC0415
        n = 0
        with _engine().begin() as conn:
            for p in params:
                r = conn.execute(text(sql), p)
                n += int(getattr(r, "rowcount", 0) or 0)
        return n
    except Exception as exc:  # noqa: BLE001
        log.info("chunk_usage update failed (%s)", type(exc).__name__)
        _drop_engine()
        return 0


def refresh_initial_weights(items) -> int:
    """After the taxonomy job re-links: items = (span_id, node_id, similarity). Re-bases initial_weight on the new similarity
    (clamped to [MIN_INITIAL, 1]) and rescales weight to keep its decay ratio. removed_at is left as it is. Returns rows updated."""
    ps = []
    for sid, node, sim in items or ():
        try:
            if isinstance(sim, (int, float)) and not isinstance(sim, bool) and math.isfinite(sim) and int(node) > 0:
                ps.append({"span_id": int(sid), "node_id": int(node), "new": min(1.0, max(MIN_INITIAL, float(sim)))})
        except Exception:  # noqa: BLE001
            continue
    return _exec_updates(SQL_REFRESH_INIT, ps)


def restore_association(span_id, chunk_start, chunk_end, node_id) -> bool:
    """Reverse a decay/soft removal by hand: weight back to initial_weight, removed_at cleared. True if a row was changed."""
    try:
        p = {"span_id": int(span_id), "chunk_start": int(chunk_start), "chunk_end": int(chunk_end), "node_id": int(node_id)}
    except Exception:  # noqa: BLE001
        return False
    return _exec_updates(SQL_RESTORE, [p]) > 0


# ------------------------------------------------------------------------------------------------ tests
def reset_for_tests() -> None:
    _NODES.clear()
    _STATE.update({"sim_fn": None, "cfg": None, "pending": {}, "thread": None, "table_ok": None, "probed_at": 0.0, "engine": None,
                   "engine_factory": None, "snap": {}, "snap_at": None, "snap_busy": False, "clock": time.monotonic,
                   "dropped": 0, "flushed": 0, "failed_batches": 0})


def set_config_for_tests(raw) -> None:
    _STATE["cfg"] = _clean(raw)


def set_engine_factory_for_tests(factory) -> None:
    _STATE["engine_factory"] = factory
    _STATE["engine"] = None


def pending_for_tests() -> dict:
    return {k: tuple(v) for k, v in _STATE["pending"].items()}
