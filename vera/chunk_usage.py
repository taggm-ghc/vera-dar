"""CHUNK-USAGE-1110 (WP11, work/CHUNK-VALUE-1110/plan.md section 6b): per-chunk and per-association match-to-use telemetry. Best-effort.

Two tables (migration 025, owner-run):
  vera_vjay.chunk_use        one row per stored retrieval chunk (partition span_id, chunk_start, chunk_end): matched / selected / used counts.
  vera_vjay.chunk_use_assoc  one row per ASSOCIATION at PARTITION level (span_id, node_kind, node_id): the unit of the taxonomy job's chunk
                             link and of its stored cosine. node_kind codes (NODE_KINDS): 1 dar_domain (research_domains.domain_id),
                             2 tax_domain, 3 tax_topic, 4 tax_subtopic. The kinds have separate id sequences, so the kind is part of the key.
Counts, weights and UTC dates only; no question text, run id, session or time of day. Chunks without a positive integer partition id
(bundle corpora, documents fetched inline before storage) have no key and are skipped.

What is counted, and when (F3): fulldoc_admission.admit only STAGES a pass (which chunks matched, which were admitted, and through which nodes);
nothing is counted. commit_answer(ev, citations) runs in fulldoc_answer for the bundle the writer actually answered from, and counts, once per
stage: matched (relevance survivors of that pass), selected (admitted entries that are in the delivered bundle) and used (cited in the delivered
answer). Acquisition, inline, relevance-probe, discarded-base and no-answer passes are never committed. A merged bundle commits the union of its
entries' stages. Per-request node attribution rides with the staged entries (no global state shared between requests).

Association weight (owner 2026-11-10: lower the weight, then soft-remove the association, never purge the chunk):
  * initial_weight = the baseline similarity of the association. The taxonomy job stores a chunk link's score as score_kind 'cosine'
    (round(clamp(1 - cosine distance, -1, 1), 6), vera/chunk_taxo_job.py _similarity; per stored partition against the assigned node); the
    multilink branch also writes 'cosine_r<rank>' and 'cosine_soft'. link_similarity() accepts those kinds only. A DAR/tax DOMAIN has no
    stored chunk-to-domain cosine (domain links are derived through topics), so its baseline is domain_similarity(): the MAX over the chunk's
    topic links inside that domain. Both are pure helpers for the code that wires the source.
  * F5: the baseline comes from set_similarity_source(fn(span_id, node_kind, node_id) -> float | None). Until a production source is wired
    (none is, in this batch) NO association row is ever written: a row without a known baseline is dropped, never defaulted to 1.0.
  * selected_since_use counts selections without a citation. After k of them each further selection multiplies weight by decay, not below
    floor * initial_weight; a use restores toward initial_weight by the fraction restore per use, never above it.
  * removed_at is set (soft) when weight < remove_below (times initial_weight when assoc.relative, the default); a use that lifts weight back
    to or above the threshold clears it; restore_association() clears it by hand. Nothing deletes a row or a chunk: the app roles hold no DELETE
    and the chunk stays globally searchable.
  * The SQL lives in vera/chunk_usage_sql/chunk_usage_upsert_*.sql and is the exact text the migration rehearsal executes; step() is its Python statement.
Read API for the router: association(), weight_ratio(), removed_associations() (an in-memory snapshot refreshed off the request path).

Safety contract (never slows or fails an ask): every request-path function takes its lock with a short timeout, appends to a bounded
in-memory structure, and drops on any error; a background daemon thread probes `to_regclass('vera_vjay.chunk_use')` and
`('vera_vjay.chunk_use_assoc')`, then flushes bounded batches (a failed batch is dropped, never retried); until both tables exist the queue is
discarded, so shipping this code before migration 025 is inert. Demotion (config demote.scale, 0 = OFF) is rank-only.
Tests inject fakes with set_*_for_tests; no test needs a database.
"""
from __future__ import annotations

import contextvars
import json
import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("vera.chunk_usage")

_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = _ROOT / "config" / "chunk_usage.json"
LOCK_TIMEOUT_S = 0.05
NODE_KINDS = {"dar_domain": 1, "tax_domain": 2, "tax_topic": 3, "tax_subtopic": 4}


def _sql(name: str) -> str:
    """The app SQL file without comment lines and the trailing semicolon (SQLAlchemy text form)."""
    try:
        lines = (Path(__file__).resolve().parent / "chunk_usage_sql" / name).read_text(encoding="utf-8").splitlines()
    except OSError:  # missing file => "" => the writer stays inert (flush_once skips an empty statement); import never fails
        return ""
    return "\n".join(x for x in lines if not x.lstrip().startswith("--")).strip().rstrip(";")


SQL_PROBE = ("SELECT pg_catalog.to_regclass('vera_vjay.chunk_use') IS NOT NULL "
             "AND pg_catalog.to_regclass('vera_vjay.chunk_use_assoc') IS NOT NULL")
SQL_UPSERT_TOTAL = _sql("chunk_usage_upsert_total.sql")
SQL_UPSERT_ASSOC = _sql("chunk_usage_upsert_assoc.sql")
SQL_SNAPSHOT = ("SELECT span_id, node_kind, node_id, initial_weight, weight, (removed_at IS NOT NULL) FROM vera_vjay.chunk_use_assoc "
                "WHERE weight < initial_weight OR removed_at IS NOT NULL ORDER BY node_kind, node_id, span_id LIMIT :max_rows")
SQL_REFRESH_INIT = (
    "UPDATE vera_vjay.chunk_use_assoc SET weight = LEAST(CAST(:new AS double precision), CASE WHEN initial_weight > 0 "
    "THEN weight / initial_weight * CAST(:new AS double precision) ELSE CAST(:new AS double precision) END), "
    "initial_weight = CAST(:new AS double precision) WHERE span_id = :span_id AND node_kind = :node_kind AND node_id = :node_id")
SQL_RESTORE = ("UPDATE vera_vjay.chunk_use_assoc SET weight = initial_weight, removed_at = NULL, selected_since_use = 0 "
               "WHERE span_id = :span_id AND node_kind = :node_kind AND node_id = :node_id")

MIN_INITIAL = 0.01
DEFAULTS = {"enabled": False, "queue_max": 2000, "batch_max": 200, "flush_s": 5.0, "probe_s": 600.0,
            "connect_timeout_s": 3, "statement_timeout_ms": 3000, "stage_max": 256}
ASSOC_DEFAULTS = {"k": 50, "decay": 0.9, "floor": 0.05, "restore": 0.5, "remove_below": 0.2, "relative": True}
DEMOTE_DEFAULTS = {"scale": 0.0}
SNAP_DEFAULTS = {"refresh_s": 900.0, "max_rows": 10000}
_BOUNDS = {"queue_max": (10, 100000), "batch_max": (1, 5000), "flush_s": (0.5, 3600), "probe_s": (5, 86400),
           "connect_timeout_s": (1, 30), "statement_timeout_ms": (100, 60000), "stage_max": (8, 5000)}
_ABOUNDS = {"k": (1, 10 ** 9), "decay": (0.01, 0.999), "floor": (0.0, 0.9), "restore": (0.0, 1.0), "remove_below": (0.0, 1.0)}
_SBOUNDS = {"refresh_s": (5, 86400), "max_rows": (1, 25000)}  # 25k entries is about 5-8 MB of Python objects

_LOCK = threading.Lock()
_STAGES: "OrderedDict[int, tuple]" = OrderedDict()  # id(admitted entry) -> (entry, _Stage, member chunk keys); entries pinned so ids stay unique
_STATE: dict = {"sim_fn": None, "cfg": None, "tot": {}, "assoc": {}, "thread": None, "table_ok": None, "probed_at": 0.0,
                "engine": None, "engine_factory": None, "snap": {}, "snap_at": None, "snap_busy": False, "clock": time.monotonic,
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
    if a["relative"]:
        a["floor"] = min(a["floor"], a["remove_below"] * 0.5)  # removal must stay reachable
    c.update(assoc=a, demote=d, snapshot=sn)
    return c


def config() -> dict:
    """Validated config, cached. A missing or bad file means OFF. Never raises."""
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


def step(state: dict, s: int, u: int, cfg: dict | None = None) -> dict:
    """The association upsert's arithmetic for one commit: s selections and u uses. state = {initial, weight, ssu, removed}.
    Pure; vera/chunk_usage_sql/chunk_usage_upsert_assoc.sql is the same rule in SQL (the rehearsal runs the SQL itself)."""
    a = (cfg or config())["assoc"]
    init, w, ssu, rem = float(state["initial"]), float(state["weight"]), int(state["ssu"]), bool(state["removed"])
    if u > 0:
        ssu = 0
        w = min(init, w + (init - w) * (1.0 - (1.0 - a["restore"]) ** u))
    else:
        ssu += s
        if ssu >= a["k"]:
            w = max(a["floor"] * init, w * a["decay"] ** min(s, ssu - a["k"] + 1))
    thr = a["remove_below"] * init if a["relative"] else a["remove_below"]
    if w < thr:
        rem = True
    elif u > 0:
        rem = False
    return {"initial": init, "weight": w, "ssu": ssu, "removed": rem}


# ------------------------------------------------------------------------------------------------ similarity helpers
_KIND_RE = re.compile(r"^(cosine|cosine_r[0-9]{1,3}|cosine_soft)$")


def link_similarity(score, score_kind):
    """Baseline from one tax link row (score, score_kind): kinds 'cosine', 'cosine_r<rank>' and 'cosine_soft' only; else None.
    Clamped to [MIN_INITIAL, 1]."""
    if not isinstance(score_kind, str) or not _KIND_RE.match(score_kind):
        return None
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
        return None
    return min(1.0, max(MIN_INITIAL, float(score)))


def domain_similarity(links):
    """Baseline of a domain association: the MAX link_similarity over the chunk's topic links inside that domain (links =
    iterable of (score, score_kind)); None when no usable link exists. Domain links are derived through topics, so no chunk-to-domain
    cosine is stored anywhere."""
    best = None
    for sc, kind in links or ():
        v = link_similarity(sc, kind)
        if v is not None and (best is None or v > best):
            best = v
    return best


# ------------------------------------------------------------------------------------------------ keys
def chunk_key(stamps, start, end):
    """(partition span_id, start, end) for the retrieval chunk [start, end): the stored partition stamp that holds the chunk's midpoint.
    None when there is no positive integer partition id (document-level stamps, bundle or inline documents)."""
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


def _nodes_of(c) -> tuple:
    """Association nodes (kind code, id) of a retrieval candidate: 'domain:<id>' branches are DAR domains (kind 1); c['nodes'] items are
    (kind name or code, id) pairs or {'kind':..,'id':..} dicts (the router's taxonomy nodes)."""
    out = set()
    try:
        for b in c.get("branches") or ():
            if isinstance(b, str) and b.startswith("domain:") and b[7:].isdigit() and int(b[7:]) > 0:
                out.add((NODE_KINDS["dar_domain"], int(b[7:])))
        for n in c.get("nodes") or ():
            kind, nid = (n.get("kind"), n.get("id")) if isinstance(n, dict) else (n[0], n[1])
            kind = NODE_KINDS.get(kind, kind)
            if kind in NODE_KINDS.values() and isinstance(nid, int) and not isinstance(nid, bool) and nid > 0:
                out.add((kind, nid))
    except Exception:  # noqa: BLE001
        pass
    return tuple(sorted(out))


@contextmanager
def suppressed():
    """Stage nothing inside (non-answer probes such as the relevance probe of ask_service)."""
    tok = _SUPPRESS.set(True)
    try:
        yield
    finally:
        _SUPPRESS.reset(tok)


class _Stage:
    """One admission pass: the chunks that matched (with their nodes) and which of them were admitted. Committed at most once."""
    __slots__ = ("matched", "committed")

    def __init__(self, matched):
        self.matched = matched  # {(span_id, start, end): nodes}
        self.committed = False


def _entry_keys(a, stamps) -> list:
    """3-keys of the retrieval chunks behind one admitted entry (a merged entry expands to its member chunks)."""
    out = []
    for sid in a.get("merged_span_ids") or (a.get("span_id"),):
        rng = _range_of(sid)
        k = chunk_key(stamps, rng[0], rng[1]) if rng else None
        if k is not None:
            out.append(k)
    if not out:
        k = chunk_key(stamps, a.get("start"), a.get("end"))
        if k is not None:
            out.append(k)
    return out


def stage_admission(corpus, matched, admitted, stamp_holder) -> None:
    """Hook in fulldoc_admission.admit. `matched` = candidate dicts that cleared the relevance rules, `admitted` = the final admitted
    entries; `stamp_holder(corpus, document_id_str)` is that module's stored-chunk-stamp accessor. Counts NOTHING: it only records the pass
    so commit_answer can count it if, and only if, this pass's evidence reaches the delivered answer."""
    try:
        if _SUPPRESS.get() or not enabled() or not admitted:
            return
        m = {}
        for c in matched or ():
            k3 = chunk_key(stamp_holder(corpus, str(c["document_id"]))["corpus_stamps"], c.get("start"), c.get("end"))
            if k3 is not None:
                m[k3] = _nodes_of(c)
        if not m:
            return
        st = _Stage(m)
        rows = []
        for a in admitted:
            keys = _entry_keys(a, a.get("corpus_chunks") or stamp_holder(corpus, str(a["document_id"]))["corpus_stamps"])
            rows.append((a, st, keys))
        if not _LOCK.acquire(timeout=LOCK_TIMEOUT_S):
            return
        try:
            cap = int(config()["stage_max"])
            for a, st_, keys in rows:
                _STAGES[id(a)] = (a, st_, keys)
                _STAGES.move_to_end(id(a))
            while len(_STAGES) > cap:
                _STAGES.popitem(last=False)
        finally:
            _LOCK.release()
    except Exception:  # noqa: BLE001  never raises into /ask
        pass


def commit_answer(ev, citations) -> None:
    """Hook in fulldoc_answer after the claim check: count the stages behind the bundle the writer answered from. matched = the stage's
    relevance survivors; selected = its admitted entries that are in `ev`; used = those cited in `citations`. Each stage commits once."""
    try:
        if _SUPPRESS.get() or not enabled():
            return
        entries = [a for a in (getattr(ev, "admitted", ()) or ()) if isinstance(a, dict)]
        if not entries:
            return
        cited = {str(c.get("span_id")) for c in citations or () if isinstance(c, dict) and c.get("span_id")}
        if not _LOCK.acquire(timeout=LOCK_TIMEOUT_S):
            return
        try:
            found = []
            for a in entries:
                hit = _STAGES.get(id(a))
                if hit is not None and hit[0] is a and not hit[1].committed:
                    found.append(hit)
            stages = {id(h[1]): h[1] for h in found}
            for st in stages.values():
                st.committed = True
        finally:
            _LOCK.release()
        if not found:
            return
        mt, sel, use = [], [], []
        for st in stages.values():
            mt.extend(st.matched)
        for a, st, keys in found:
            sel.extend(keys)
            if str(a.get("span_id")) in cited:
                use.extend(keys)
        for slot, ks in ((0, mt), (1, sel), (2, use)):
            _add(ks, slot, {k: n for st in stages.values() for k, n in st.matched.items()})
    except Exception:  # noqa: BLE001
        pass


def _add(keys, slot: int, nodes_by_key) -> None:
    """Add 1 to counter `slot` (0 matched, 1 selected, 2 used) of each distinct chunk key, and of each distinct (partition, node)."""
    if not keys:
        return
    cfg = config()
    tot, assoc = set(keys), set()
    for k in tot:
        for kind, nid in nodes_by_key.get(k, ()):
            assoc.add((k[0], kind, nid))  # partition level: one count per partition per commit
    if not _LOCK.acquire(timeout=LOCK_TIMEOUT_S):
        _STATE["dropped"] += len(tot)
        return
    try:
        for table, ks in (("tot", tot), ("assoc", assoc)):
            pend = _STATE[table]
            for k in ks:
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


# ------------------------------------------------------------------------------------------------ read API (router) and demotion
def set_similarity_source(fn) -> None:
    """fn(span_id, node_kind_code, node_id) -> float | None: the association's baseline similarity (see link_similarity and
    domain_similarity). Called on the flush thread only. None clears it; with no source NO association row is written (F5)."""
    _STATE["sim_fn"] = fn


def _baseline(span_id, node_kind, node_id):
    """Baseline in [MIN_INITIAL, 1] or None (unknown => the row is not written)."""
    try:
        fn = _STATE["sim_fn"]
        v = fn(span_id, node_kind, node_id) if fn is not None else None
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0:
            return min(1.0, max(MIN_INITIAL, float(v)))
    except Exception:  # noqa: BLE001
        pass
    return None


def _snapshot() -> dict:
    """{(span_id, node_kind, node_id): (initial_weight, weight, removed)} of decayed or removed associations. Never blocks."""
    if enabled():
        _refresh_snapshot_async()
    return _STATE["snap"]


def association(span_id, node_kind, node_id):
    """None when the association has no lowered weight (treat as full weight, not removed); else a dict with initial_weight, weight,
    ratio (weight / initial_weight) and removed. node_kind is a code or a name of NODE_KINDS. In-memory, safe on the request path."""
    try:
        kind = NODE_KINDS.get(node_kind, node_kind)
        v = _snapshot().get((int(span_id), int(kind), int(node_id)))
        if v is None:
            return None
        init, w, rem = v
        return {"initial_weight": init, "weight": w, "ratio": (w / init if init > 0 else 0.0), "removed": bool(rem)}
    except Exception:  # noqa: BLE001
        return None


def weight_ratio(span_id, node_kind, node_id) -> float:
    a = association(span_id, node_kind, node_id)
    return 1.0 if a is None else float(a["ratio"])


def removed_associations() -> frozenset:
    """Keys (span_id, node_kind, node_id) of soft-removed associations. The chunk itself is never removed."""
    try:
        return frozenset(k for k, v in _snapshot().items() if v[2])
    except Exception:  # noqa: BLE001
        return frozenset()


def demoter(stamp_holder, corpus):
    """None when demotion is off (scale 0, flag off, or no lowered association known): admit then changes nothing. Otherwise
    f(candidate) -> a factor in (0, 1] multiplying the (negative) relevance sort key. A candidate with no routing node, or with any
    node of full weight, is not demoted."""
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
                ratio = max(((snap[(k3[0],) + n][1] / snap[(k3[0],) + n][0]) if (k3[0],) + n in snap else 1.0) for n in nodes)
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
    if not _LOCK.acquire(timeout=LOCK_TIMEOUT_S):
        return
    try:
        t = _STATE["thread"]
        if t is not None and t.is_alive():
            return
        t = threading.Thread(target=_loop, name="chunk-usage-flush", daemon=True)
        _STATE["thread"] = t
        t.start()
    finally:
        _LOCK.release()


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
                return create_engine(url, pool_size=1, max_overflow=0, hide_parameters=True,
                                     **{k: v for k, v in kw.items() if k != "hide_parameters"},
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
    """to_regclass probe of both tables, cached probe_s (re-probed so a table created later switches the writer on without a restart)."""
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


def _take(table: str, n: int) -> list:
    with _LOCK:
        pend = _STATE[table]
        keys = sorted(pend)[:n]  # sorted: stable lock order across instances
        return [(k, pend.pop(k)) for k in keys]


def flush_once() -> int:
    """One flush pass (the background thread calls it; tests call it directly). Returns rows written. Never raises."""
    try:
        if not enabled():
            return 0
        cfg = config()
        if not _table_exists():
            with _LOCK:
                _STATE["tot"].clear()
                _STATE["assoc"].clear()  # no tables: nothing is kept (inert until migration 025 is applied)
            return 0
        a = cfg["assoc"]
        g = {"k": int(a["k"]), "decay": float(a["decay"]), "floor": float(a["floor"]), "restore": float(a["restore"]),
             "thr": float(a["remove_below"]), "rel": bool(a["relative"])}
        written = 0
        for table, sql in (("tot", SQL_UPSERT_TOTAL), ("assoc", SQL_UPSERT_ASSOC)):
            while True:
                batch = _take(table, int(cfg["batch_max"]))
                if not batch:
                    break
                if table == "tot":
                    params = [{"span_id": k[0], "chunk_start": k[1], "chunk_end": k[2], "m": v[0], "s": v[1], "u": v[2]} for k, v in batch]
                else:
                    params = []
                    for k, v in batch:
                        init = _baseline(k[0], k[1], k[2])
                        if init is None:  # F5: no known baseline similarity => no association row
                            _STATE["dropped"] += 1
                            continue
                        params.append(dict(g, span_id=k[0], node_kind=k[1], node_id=k[2], m=v[0], s=v[1], u=v[2],
                                           ssu0=(0 if v[2] > 0 else v[1]), init=init))
                if not params or not sql:
                    continue
                try:
                    from sqlalchemy import text  # noqa: PLC0415
                    with _engine().begin() as conn:
                        conn.execute(text(sql), params)
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
    """Load decayed or removed associations into the in-memory snapshot (capped by snapshot.max_rows). On any error keep the old one."""
    n = 0
    try:
        d = config()["snapshot"]
        if _table_exists():
            from sqlalchemy import text  # noqa: PLC0415
            with _engine().connect() as conn:
                rows = conn.execute(text(SQL_SNAPSHOT), {"max_rows": int(d["max_rows"])}).fetchall()
            _STATE["snap"] = {(int(r[0]), int(r[1]), int(r[2])): (float(r[3]), float(r[4]), bool(r[5])) for r in rows}
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
    """After the taxonomy job re-links: items = (span_id, node_kind, node_id, similarity). Re-bases initial_weight on the new similarity
    (clamped to [MIN_INITIAL, 1]) and rescales weight to keep its decay ratio. removed_at is left as it is. Returns rows updated."""
    ps = []
    for sid, kind, node, sim in items or ():
        try:
            kind = NODE_KINDS.get(kind, kind)
            if isinstance(sim, (int, float)) and not isinstance(sim, bool) and math.isfinite(sim) and sim > 0 and kind in NODE_KINDS.values():
                ps.append({"span_id": int(sid), "node_kind": int(kind), "node_id": int(node), "new": min(1.0, max(MIN_INITIAL, float(sim)))})
        except Exception:  # noqa: BLE001
            continue
    return _exec_updates(SQL_REFRESH_INIT, ps)


def restore_association(span_id, node_kind, node_id) -> bool:
    """Reverse a decay/soft removal by hand: weight back to initial_weight, removed_at cleared. True if a row was changed."""
    try:
        p = {"span_id": int(span_id), "node_kind": int(NODE_KINDS.get(node_kind, node_kind)), "node_id": int(node_id)}
    except Exception:  # noqa: BLE001
        return False
    return _exec_updates(SQL_RESTORE, [p]) > 0


# ------------------------------------------------------------------------------------------------ tests
def reset_for_tests() -> None:
    _STAGES.clear()
    _STATE.update({"sim_fn": None, "cfg": None, "tot": {}, "assoc": {}, "thread": None, "table_ok": None, "probed_at": 0.0,
                   "engine": None, "engine_factory": None, "snap": {}, "snap_at": None, "snap_busy": False,
                   "clock": time.monotonic, "dropped": 0, "flushed": 0, "failed_batches": 0})


def set_config_for_tests(raw) -> None:
    _STATE["cfg"] = _clean(raw)


def set_engine_factory_for_tests(factory) -> None:
    _STATE["engine_factory"] = factory
    _STATE["engine"] = None


def pending_for_tests() -> dict:
    return {"tot": {k: tuple(v) for k, v in _STATE["tot"].items()}, "assoc": {k: tuple(v) for k, v in _STATE["assoc"].items()}}
