"""PORT-P2: per-/ask discovery recorder. Maps one request's research record (vera.research_record.summarize output)
plus the answer's numbered sources and claim verdicts onto the discovery store: one discovery_event per portal
attempt, one attribution per (portal, candidate document) - every portal that found a document is credited - and
one outcome row per document under the versioned metric definitions in config/portal_registry.json.

Never blocks or changes an answer: record_ask() swallows every error, is a no-op when the store is disabled, and
writes on one background worker through a bounded queue (config store.max_pending; overflow dropped) with a short
SQLite busy timeout (store.busy_timeout_s).
Stores a hash of the question only (no question text, no caller identity).
"""
from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import uuid
from typing import Mapping

from vera import portal_store as ds

logger = logging.getLogger("vera.discovery")

# research_record._PORTAL_STATUS plus the raw search_providers statuses -> discovery_event.status
_STATUS = {"completed": "completed", "timeout": "timeout", "quota": "quota", "skipped_quota": "quota",
           "failed": "error", "error": "error", "skipped_budget": "skipped", "skipped": "skipped"}
_RELEVANT_OUTCOMES = ("admitted", "reused", "not_used")  # not_used = admitted but over the document budget
_ATTEMPTED_OUTCOMES = ("admitted", "rejected", "not_used")
_ADMISSION_REJECTIONS = ("licence_rejected", "provenance_rejected", "injection_rejected", "relevance_rejected",
                         "credibility_rejected", "freshness_rejected")
_STORE = None
_STORE_CFG: str | None = None


def query_hash(question: str) -> str:
    return hashlib.sha256(" ".join(str(question or "").casefold().split()).encode("utf-8")).hexdigest()


_CFG_CACHE: dict = {}


def _registry_cfg() -> dict:
    """config/portal_registry.json, re-read only when its mtime changes (not on every request)."""
    from vera import portal_registry
    m = portal_registry.CONFIG_PATH.stat().st_mtime_ns
    if _CFG_CACHE.get("mtime") != m:
        cfg = json.loads(portal_registry.CONFIG_PATH.read_text(encoding="utf-8"))
        _CFG_CACHE.update(mtime=m, cfg=cfg, reg=None)
    return _CFG_CACHE["cfg"]


def _registry(cfg: Mapping):
    if _CFG_CACHE.get("cfg") is cfg and _CFG_CACHE.get("reg") is not None:
        return _CFG_CACHE["reg"]
    from vera import portal_registry
    reg = portal_registry.load_registry(cfg)
    if _CFG_CACHE.get("cfg") is cfg:
        _CFG_CACHE["reg"] = reg
    return reg


def _store_for(cfg: Mapping):
    global _STORE, _STORE_CFG
    sig = json.dumps(cfg.get("store") or {}, sort_keys=True)
    if _STORE_CFG != sig:
        _STORE, _STORE_CFG = ds.open_store(cfg), sig
    return _STORE


def _match_sources(c: Mapping, sources) -> list:
    """Numbers of the answer sources that are this candidate's document (canonical URL, URL or identity key)."""
    ref = c.get("canonical_ref")
    want = {str(v) for v in (c.get("canonical_url"), c.get("url")) if v}
    key = ds.document_key(c)
    out = []
    for s in sources or ():
        if not isinstance(s, Mapping):
            continue
        cand = {str(v) for v in (s.get("canonical_source_id"), s.get("url")) if v}
        if (ref and ref == ds.canonical_ref(s.get("canonical_source_id"))) or (want & cand) \
                or (key is not None and ds.document_key({"url": s.get("url")}) == key):
            out.append(s.get("n"))
    return out


def stage_outcome(c: Mapping, sources, claims, fresh_docs: Mapping, accepted: bool) -> dict:
    """Stage flags for one candidate under metric definitions v1 (config metrics.definitions)."""
    oc = c.get("outcome")
    tr = c.get("terminal_reason")
    relevant = oc in _RELEVANT_OUTCOMES
    attempted = oc in _ATTEMPTED_OUTCOMES
    succeeded = oc in ("admitted", "not_used") or (oc == "rejected" and tr in _ADMISSION_REJECTIONS)
    ns = set(_match_sources(c, sources))
    chunks = c.get("n_chunks") if oc == "admitted" else None  # reused: the stored passage count is not on the record
    mine = [r for r in claims or () if isinstance(r, Mapping) and ns & set(r.get("citations") or ())]
    verdicts = [r.get("verdict") for r in mine]
    shown = [r.get("verdict") for r in mine if not r.get("partial_removed")]  # FA8-3: a removed partial is not shown
    checked = bool(claims)
    cited = (any(v in ("supported", "partial") for v in shown)) if checked else None
    contradictory = (any(v == "unsupported" for v in verdicts)) if checked else None
    docs = [fresh_docs.get(str(s.get("document_id"))) for s in sources or ()
            if isinstance(s, Mapping) and s.get("n") in ns]
    fresh = any(d and d.get("freshness") == "fresh" for d in docs) if docs else None
    has_chunks = bool(ns) and (chunks is None or chunks >= 1)
    valuable = bool(relevant and accepted and has_chunks and checked and "supported" in verdicts
                    and not contradictory)
    return {"relevance_qualified": relevant, "acq_attempted": attempted, "acq_succeeded": succeeded,
            "admitted_chunks": chunks, "cited": cited, "contradictory": contradictory, "freshness_qualified": fresh,
            "valuable": valuable, "terminal_reason": tr, "source_date": next(
                (d.get("source_date") for d in docs if d and d.get("source_date")), None),
            "acquired_at": next((d.get("acquired_at") for d in docs if d and d.get("acquired_at")), None)}


def record(store, rec: Mapping, *, question: str = "", sources=(), claims=(), accepted: bool = False,
           domain: str | None = None, registry=None, definitions_version: int = 1,
           request_id: str | None = None) -> str | None:
    """Write one request. Returns the request id, or None when the record has no research attempt."""
    if store is None or not isinstance(rec, Mapping) or not rec.get("triggered"):
        return None
    rid = request_id or uuid.uuid4().hex
    qh = query_hash(question)
    key = registry.resolve_key if registry is not None else (lambda n: str(n))
    cost_basis = (lambda k: getattr(registry.portals.get(k), "cost_basis", None)) if registry is not None \
        else (lambda k: None)
    events: dict = {}
    for p in rec.get("portals") or ():
        if not isinstance(p, Mapping):
            continue
        k = key(p.get("portal"))
        events[k] = store.add_event(request_id=rid, portal_key=k, status=_STATUS.get(p.get("status"), "error"),
                                    n_results=int(p.get("results") or 0), query_hash=qh, domain=domain,
                                    latency_ms=p.get("ms"), cost_usd=p.get("cost_usd"), cost_basis=cost_basis(k))
    after = ((rec.get("freshness") or {}).get("after") or (rec.get("freshness") or {}).get("before") or {})
    fresh_docs = {str(d.get("document_id")): d for d in after.get("documents") or () if isinstance(d, Mapping)}
    seen: set = set()
    for i, c in enumerate(rec.get("candidates") or ()):
        if not isinstance(c, Mapping):
            continue
        portals = list(dict.fromkeys(key(x) for x in (c.get("found_by") or [c.get("portal")]) if x))
        doc = None
        for k in portals:
            if k not in events:  # a candidate proves its portal returned results even without an attempt record
                events[k] = store.add_event(request_id=rid, portal_key=k, status="completed", query_hash=qh,
                                            domain=domain, cost_basis=cost_basis(k))
            doc = store.add_attribution(event_id=events[k], ids=c, rank=i + 1) or doc
        if doc is None or doc in seen:
            continue  # unique publications: one outcome row per document per request
        seen.add(doc)
        store.add_outcome(request_id=rid, doc_key=doc, definitions_version=definitions_version,
                          **stage_outcome(c, sources, claims, fresh_docs, accepted))
    return rid


_QUEUE: queue.Queue | None = None
_WORKER: threading.Thread | None = None
_QLOCK = threading.Lock()


def _worker() -> None:
    while True:
        job = _QUEUE.get()
        try:
            job()
        except Exception as e:  # noqa: BLE001  recording never affects anything else
            logger.warning("discovery record skipped (%s)", type(e).__name__)
        finally:
            _QUEUE.task_done()


def _submit(job, max_pending: int) -> bool:
    """Hand a write to ONE daemon worker through a bounded queue; when full the record is dropped (never awaited)."""
    global _QUEUE, _WORKER
    with _QLOCK:
        if _QUEUE is None:
            _QUEUE = queue.Queue(maxsize=max(1, int(max_pending)))
            _WORKER = threading.Thread(target=_worker, name="vera-portal-recorder", daemon=True)
            _WORKER.start()
    try:
        _QUEUE.put_nowait(job)
        return True
    except queue.Full:
        logger.warning("discovery record dropped (queue full)")
        return False


def drain(timeout_s: float = 2.0) -> bool:
    """Tests/dev: wait (bounded) until queued records are written. True when the queue emptied in time."""
    import time
    end = time.monotonic() + timeout_s
    while _QUEUE is not None and _QUEUE.unfinished_tasks and time.monotonic() < end:
        time.sleep(0.01)
    return _QUEUE is None or not _QUEUE.unfinished_tasks


def record_ask(rec: Mapping, *, question: str = "", result=None, domain: str | None = None) -> None:
    """The /ask hook (vera.ask_service). No-op unless config store.enabled; never raises and never waits on the store:
    the request thread only snapshots the inputs and enqueues (bounded); the write runs on a background worker with
    the store's short SQLite busy timeout."""
    try:
        cfg = _registry_cfg()
        store = _store_for(cfg)
        if store is None or not isinstance(rec, Mapping) or not rec.get("triggered"):
            return
        reg = _registry(cfg)
        ver = int(((cfg.get("metrics") or {}).get("definitions") or {}).get("version") or 1)
        snap = json.loads(json.dumps(rec, default=str))
        sources = [dict(x) for x in (getattr(result, "sources", ()) or ()) if isinstance(x, Mapping)]
        claims = [dict(x) for x in (getattr(result, "claims", ()) or ()) if isinstance(x, Mapping)]
        accepted = bool(getattr(result, "accepted", False)) and getattr(result, "result_kind", None) == "answer"
        _submit(lambda: record(store, snap, question=question, sources=sources, claims=claims, accepted=accepted,
                               domain=domain, registry=reg, definitions_version=ver),
                int((cfg.get("store") or {}).get("max_pending", 8)))
    except Exception as e:  # noqa: BLE001  recording never affects the answer
        logger.warning("discovery record skipped (%s)", type(e).__name__)
