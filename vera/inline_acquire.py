"""DAR-INLINE-1021: inline open-access full-text acquisition inside one /ask request (off-topic override path only).

Called by vera.ask_service on the layer-2 off_topic override path, after the stored acquired corpus could not answer:

  search (the job's own search + AND/PMC profile) -> rank (shared stage) -> resolve OA + fetch JATS + every existing
  gate, in parallel (small pool, per-document timeout, early stop at the document budget) -> persist through the
  EXISTING INSERT-only WoStore (vera_pipeline_wo) -> an EPHEMERAL in-memory corpus of exactly the just-admitted
  documents, built by the acquired-path reader (vera.fulldoc_db.load_acquired_rows: same chunking, spans, hashes,
  licence re-decision and injection re-scan) -> the caller answers through the unchanged admission, answer writer and
  claim check with reference_corpus = that corpus. No global rebuild here; main.py refreshes the stored acquired
  corpus in the background after the response.

Bounds: the kill switch and flag (acq_config.enabled) re-checked per call; jobs_per_hour / jobs_per_day and
docs_per_day shared with the background queue (auto_acquire._S); at most inline.max_concurrent acquisitions
server-wide (above it: "busy" at once); identical in-flight questions wait (bounded, capped waiters) on the first
one's outcome; outcomes are reused for result_ttl_s so a provider-chain retry never acquires twice. One deadline
(inline.deadline_s) bounds search through the parallel fetch and gates; persist and the ephemeral corpus build run after it
(VER L2).

ONE-SPAN-1021 (owner approval): a document answers this request with >= bounds.answer_min_admitted_spans (1) qualifying
passages but is persisted only with >= bounds.persist_min_admitted_spans (2); below that it is in the ephemeral corpus
only (Outcome.documents: persisted false, reason below_persist_min; counts n_answer_eligible / n_persisted). It does
not count against docs_per_day: only STORED documents do (used = out.stored; the cap is null = off since 2026-10-22).

Persist failure: the answer still goes out (every licence, content, injection and relevance gate already passed, and
the claim check still runs); the note then says the documents could NOT be added, and the failure is logged (class
name only). Privacy (AC-8) as auto_acquire: the question lives only in memory (salted HMAC keys), logs carry codes,
counts and durations only.
"""
from __future__ import annotations

import contextvars
import hashlib
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from vera import acq_config, acq_rounds, ddg_meter, ddg_reduce, rate_limit
from vera import auto_acquire as aa
from vera import request_trace as rt
from vera import research_outcome, search_providers, source_type, stream_ctx

logger = logging.getLogger("vera.autoacq")

# statuses: a search was made -> "ok" (corpus set), "none", "deadline", "search_failed"
#           no search was made -> "off", "busy", "rate", "daily_cap", "error"
SEARCHED = frozenset({"ok", "none", "deadline", "search_failed", "unanswered"})
CAPPED = frozenset({"rate", "daily_cap"})  # DAR-TRACE-1022: the shared hourly/daily caps refused the acquisition
# DAR-TRACE-1022 (review S1): provider/time failures are cached only failure_ttl_s; deterministic none/unanswered keep
# result_ttl_s (re-running them early would only burn the shared jobs_per_hour cap).
TRANSIENT = frozenset({"search_failed", "deadline"})
BELOW_PERSIST_MIN = "below_persist_min"  # ONE-SPAN-1021 per-document telemetry reason
INF = float("inf")
# STREAM-1031: licence-hold reason keys (a candidate that is open but held back for licence reasons), counts only.
HELD_KEYS = ("arxiv_licence_hold", "preprint_licence_unverified", "preprint_not_permitted", "licence_absent", "no_licence")
# STREAM-1031 closed "why" set for acquire/limited (INTERFACES.md): 1 no more open sources, 2 sources busy, 3 search
# limit reached, 4 stalled, 5 answering service at its limit.
# MIDTERMS-WP8-1106: 6 = the provisional no-progress heuristic stopped the rounds (disclosed, owner D2).
WHY_NO_MORE, WHY_BUSY, WHY_CAPPED, WHY_STALLED, WHY_EXHAUSTED, WHY_HEURISTIC = 1, 2, 3, 4, 5, 6


def _emit(stage: str, state: str, **counts) -> None:
    """STREAM-1031: counts-only progress event, only on the streamed path (legacy /ask emits nothing from here).
    Never raises; never carries a provider, host, URL, title, DOI or question text."""
    if not stream_ctx.active():
        return
    try:
        from vera import progress
        progress.emit(stage, state, **{k: int(v) for k, v in counts.items()})
    except Exception:  # noqa: BLE001  telemetry never breaks the run
        pass


@dataclass
class Outcome:
    status: str
    corpus: object = None
    guard: bool = False  # BATCH3-1109: "off" because the shared intent guard refused this question (not a config switch)
    stored: int = 0
    admitted: int = 0
    persist_failed: bool = False
    reused: int = 0  # R2: already-stored acquired documents answered from (no new acquisition)
    # ONE-SPAN-1021: per newly fetched answer-eligible document, in rank order: {"persisted": bool[, "reason": code]}
    # (reason "below_persist_min" = fewer than persist_min_admitted_spans qualifying passages: answers, never stored)
    documents: list = field(default_factory=list)
    codes: dict = field(default_factory=dict)
    ms: dict = field(default_factory=dict)
    # P0-5 (inline.use_discovered only; None = flag off, and public() is then unchanged): one entry per discovered
    # record: {service, record_id, url, [epistemic], gate_a {decision, factors, uncertainty}, outcome, [reason],
    # [gate_b {decision, basis}]}
    discovered: list | None = None
    # FRESH-1023 C3 (additive; None = not searched): portal attempts [{portal,status,results}] and one independent
    # record per candidate [{portal,doi,pmcid,openalex_id,arxiv_id,url,published,outcome,reason[,persisted,persist_reason]}]
    portals: list | None = None
    candidates: list | None = None
    # CONT-1023 (additive; None = nothing was eligible): the background hand-off of this request's unfinished
    # candidates {"status": code, "queued": n, "already_queued": n, "refused": n}
    background: dict | None = None
    # T1 (IMPL-1028-T1O1; None = router off): {"route", "basis": [rule ids], "scores": {"s","g"}}; per request, never
    # stored with identity, never the question text
    source_route: dict | None = None
    # OD-1031-P1 item 4: {"status": met|resource_limited|incomplete_deadline, "stored": n, "target": n}; counts only
    # (no titles, hosts, URLs or provider names). None = no search ran. Observational: changes no cap or gate.
    acquisition: dict | None = None
    # RETRIEVAL-1109 (additive; None = not run): THIS run's rate-limiter accounting {"waited_s", "permits": {portal: n},
    # "skips": {portal: {reason: n}}} (rate_limit.request_scope); portal names and counts only. The limiter's
    # politeness state is process-wide; this record is per run, so it never carries another request's skips.
    limiter: dict | None = None
    # RETRIEVAL-1109 A3: reporting-lane results of variant rounds (news questions), a list of lane_evidence.LaneResult.
    # Request-local like the round-0 lane pass: never public, cached (no_cache is set when it holds a page), stored,
    # queued or logged; with_web merges it into the request's lane result.
    lanes_extra: list | None = None
    # NONASK-1103 A6 D1 (S5): True when answer-only documents (variant-round or lane pages) are in the corpus; such an
    # outcome is never put in the result TTL cache (request-local). Not public.
    no_cache: bool = False
    # A6-W (inline.web_read_enabled): request-local web route result (vera.web_route.WebResult). NEVER public, cached,
    # stored, queued or logged: page text lives only for this request; ask_service merges its pages as lane "web".
    web: object = None

    @property
    def n_answer_eligible(self) -> int:
        return self.admitted

    @property
    def n_persisted(self) -> int:
        return self.stored

    @property
    def answer_only(self) -> int:
        return sum(1 for d in self.documents if d.get("reason") == BELOW_PERSIST_MIN)

    def public(self) -> dict:
        return {"status": self.status, "admitted": self.admitted, "stored": self.stored, "reused": self.reused,
                "persisted": not self.persist_failed, "n_answer_eligible": self.n_answer_eligible,
                "n_persisted": self.n_persisted, "documents": [dict(d) for d in self.documents], "ms": dict(self.ms),
                **({"discovered": [dict(d) for d in self.discovered]} if self.discovered is not None else {}),
                **({"portals": [dict(d) for d in self.portals]} if self.portals is not None else {}),
                **({"candidates": [dict(d) for d in self.candidates]} if self.candidates is not None else {}),  # PMCID-PUBLIC-1023
                **({"diag_status_counts": research_outcome.diag_counts(self.candidates)}
                   if self.candidates is not None else {}),  # P0DTA-DIAG: aggregate per status
                **({"background": dict(self.background)} if self.background is not None else {}),
                **({"source_route": dict(self.source_route)} if self.source_route is not None else {}),
                **({"acquisition": dict(self.acquisition)} if self.acquisition is not None else {})}

    @property
    def queued(self) -> int:
        return int((self.background or {}).get("queued") or 0) + int((self.background or {}).get("already_queued") or 0)


class _Inline:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.inflight: dict = {}   # qkey -> [threading.Event, Outcome | None, waiters]
        self.results: OrderedDict = OrderedDict()  # qkey -> (monotonic time, Outcome)


_I = _Inline()


def reset_for_tests() -> None:
    global _I
    _I = _Inline()


_REASON_TOKEN = re.compile(r"R-ACQ-[A-Z0-9_]{1,40}(-[A-Z0-9_]{1,40})?\Z")


def _trace_reason(code):
    """WP7: the trace layer redacts any string that is not lower-case token-shaped, so the upper-case reason codes
    (R-ACQ-DEFER_CAP) all showed as "<str>". A reason code of the fixed R-ACQ-* shape is traced lower-cased
    (r-acq-defer_cap); anything else is None (never free text, Deny-All)."""
    return code.lower() if isinstance(code, str) and _REASON_TOKEN.match(code) else None


def _trace_done(o, st) -> None:
    """Trace one inline acquisition attempt: counts, status, reason-code counts, phase ms and up to 20 candidates as
    provider/outcome/reason-code/status only (never doi, url, title or free text). Never raises."""
    try:
        rt.event("acquire", "done",
                 counts={"admitted": int(o.admitted or 0), "stored": int(o.stored or 0), "reused": int(o.reused or 0),
                         "candidates": len(o.candidates or [])},
                 detail={"status": str(o.status), "codes": dict(o.codes or {}), "ms": dict(o.ms or {})})
        for c in (o.candidates or [])[:20]:
            rt.event("acquire", "candidate",
                     detail={"provider": c.get("portal"), "outcome": c.get("outcome"), "reason": _trace_reason(c.get("reason")),
                             "diag": c.get("status")})
    except Exception:  # noqa: BLE001
        pass


def icfg_of(acq: dict | None) -> dict:
    v = (acq or {}).get("inline") if isinstance(acq, dict) else None
    return v if isinstance(v, dict) else {}


def inline_enabled(acq: dict | None, env=os.environ) -> bool:
    return acq_config.enabled(acq, env) and icfg_of(acq).get("enabled") is True


def _b(acq, key, default):
    try:
        return float(icfg_of(acq).get(key, default))
    except (TypeError, ValueError):
        return float(default)


def default_deps(acq: dict, env):
    """The background job's live wiring, minus its post-commit refresh (main.py refreshes after the response) and with
    a non-blocking served-key set (the stored acquired corpus only; never waits on the base-corpus load)."""
    from vera import fulldoc_corpus
    d = aa.default_deps(acq, env)

    def keys():
        c = fulldoc_corpus.get_acquired_corpus()
        dar = getattr(c, "_dar", None)
        return set(dar.doc_keys()) if dar is not None and hasattr(dar, "doc_keys") else set()
    def stored_rows(cus):  # R2: read-only (ro account), the served acquired-row filter limited to these URLs
        from vera import fulldoc_db
        setup = fulldoc_corpus._acq_setup(fulldoc_corpus.load_config(), env)
        if setup is None:
            return [], []
        acfg, connect, rid = setup
        return fulldoc_db.fetch_acquired_rows(connect, rid, acq_config.marker(acfg), len(cus), canonical_urls=cus)
    d.corpus_keys, d.refresh, d.stored_rows = keys, None, stored_rows
    return d


def _cache_key(question: str, served, disc=None) -> tuple:
    """Defect 3: the outcome cache/in-flight key is the question PLUS a digest of the document set the stored-corpus
    attempt tried, so a cached "none" is not reused once new documents became available. served None = unknown.
    P0-5: with discovered candidates (flag on) a digest of their identities is appended; disc None = unchanged key."""
    if served is None:
        k = (aa._qkey(question), None)
    else:
        k = (aa._qkey(question), hashlib.sha256("\n".join(sorted(map(str, served))).encode()).hexdigest()[:16])
    if disc is None:
        return k
    from vera import acq_discovered
    return k + ("d:" + acq_discovered.digest(disc[0]),)


def _subquery_parts(question: str, acq: dict) -> list | None:
    """NONASK-1103 S3: [(query text, entities)] for a question that decomposes into 2 or more sub-queries (vera.subquery,
    deterministic, no model call or network), else None. Off with search.subquery.enabled == false (kill switch).
    Never raises: a failure means the unchanged single search."""
    try:
        if ((acq.get("search") or {}).get("subquery") or {}).get("enabled") is not True:
            return None
        from vera import search_query, subquery
        plan = subquery.decompose(question)
        if len(plan.parts) < 2:
            return None
        out = []
        for p in plan.parts:  # a part with no registry entity is still centred on its list item (its label)
            ents = tuple(p.entities) or ((p.label,) if plan.rule == "entity_list" and p.label else ())
            out.append((search_query.part_query(p.text, ents), ents))
        return out
    except Exception:  # noqa: BLE001
        return None


def news_question(question: str) -> bool:
    """PERF-MISS-1105-FIX2 F2/F5: True when the question has news / current-events intent (vera.news_intent: a news
    cue, or an event noun with a recent year). A bare current-state cue (vera.temporality, e.g. "status quo") is not
    news intent. Never raises (False on error)."""
    try:
        from vera import news_intent
        return bool(news_intent.news_intent(str(question or "")))
    except Exception:  # noqa: BLE001
        return False


def lane_pass(question: str, acq: dict | None = None):
    """NONASK-1103 S4/S7 (R-037, R-042): the Official and Current-reporting lanes, per sub-query and, for reporting,
    per temporality (vera.lane_evidence.collect). Request-local: the result is NEVER put in the inline result cache,
    the background queue (enqueue_candidates/follow_url never see lane items), a store or a log. None when both lane
    flags are off (the shipped default: no work at all) or the question is empty; never raises."""
    try:
        from vera import lane_evidence
        if not str(question or "").strip() or not lane_evidence.any_enabled():
            return None
        from vera import scope_router
        if not scope_router.intent_clean(str(question)):
            return None
        return lane_evidence.collect(str(question))
    except Exception as e:  # noqa: BLE001
        logger.warning("lane pass failed (%s)", type(e).__name__)
        return None


def _trace_lane(question: str, lanes) -> None:
    """PERF-MISS-1105-FIX2 F2: one request-trace event for the lane pass: portal called or skipped, the reason
    (circuit, spacing, no_items, throttled, failed, off ...), and item, page and link counts. Closed tokens and counts
    only (never a URL, title, outlet or question text). Never raises."""
    try:
        news = news_question(question)
        if lanes is None:
            rt.event("lane", "skipped", counts={"items": 0, "pages": 0, "links": 0},
                     detail={"status": "skipped", "reason": "off", "news": news})
            return
        diag = [d for d in (getattr(lanes, "diag", None) or []) if isinstance(d, dict)]
        called = [d for d in diag if d.get("status") == "called"]
        first = (called or diag or [{}])[0]
        counts = {"items": sum(int(d.get("items", 0) or 0) for d in diag), "pages": len(lanes.pages),
                  "links": len(lanes.links), "calls": len(called), "skips": len(diag) - len(called)}
        # WP7 (d): extracted-term and raw (pre outlet policy) item counts, when the lane's diag entries carry them.
        # Counts keys are identifier tokens (request_trace.clean_counts), so no detail-key allowlist entry is needed;
        # a diag entry without the key adds nothing (never a guessed zero).
        for src, dst in (("terms", "terms"), ("raw_items", "raw_items")):
            vals = [d.get(src) for d in diag if isinstance(d.get(src), int) and not isinstance(d.get(src), bool)]
            if vals:
                counts[dst] = max(vals) if dst == "terms" else sum(vals)
        chk = getattr(lanes, "checks", None) or {}  # H19-FIX-1109 T6: page-check counters (counts + one token)
        for k in ("nd_read", "nd_link_only", "page_link_only", "acc_exc", "a11_drop"):
            if isinstance(chk.get(k), int) and not isinstance(chk.get(k), bool):
                counts[k] = chk[k]
        det = {"portal": str(first.get("portal") or "-"), "status": "called" if called else "skipped",
               "reason": str(first.get("reason") or ("no_reporting_part" if not diag else "-")), "news": news}
        if isinstance(chk.get("acc_exc_first"), str):
            det["acc_exc_first"] = chk["acc_exc_first"]
        rt.event("lane", "done" if called else "skipped", counts=counts, detail=det)
    except Exception:  # noqa: BLE001
        pass


class LaneTask:
    """PERF-MISS-1105-FIX2 F2 (owner rule feedback_nonblocking_spawned_steps): the lane pass started CONCURRENTLY with
    inline acquisition (it needs only the question), in a daemon thread with a copy of the request context (trace,
    progress sink, stream ctx). result() joins it; while the caller waits, the stream shows the news-lane activity
    (progress.ticking). Request-local, like lane_pass; never raises."""

    def __init__(self, question: str, acq: dict | None):
        self.question = question
        self._done = threading.Event()
        self._value = None
        ctx = contextvars.copy_context()
        self._t = threading.Thread(target=ctx.run, args=(self._run, question, acq), name="vera-lane", daemon=True)
        self._t.start()

    def _run(self, question, acq):
        try:
            self._value = lane_pass(question, acq)  # its activity is shown only while the caller waits (result())
        except Exception:  # noqa: BLE001
            self._value = None
        finally:
            self._done.set()

    def result(self):
        if not self._done.is_set():
            from vera import progress
            with progress.ticking(progress.ACT_NEWS):
                while not self._done.wait(1.0):  # bounded by the lane's own connect/read timeouts and read cap
                    c = stream_ctx.current()
                    if c is not None:
                        c.touch()
        _trace_lane(self.question, self._value)
        return self._value


def start_lane_pass(question: str, acq: dict | None = None) -> LaneTask:
    """Start the lane pass now (concurrently with acquire()); call .result() where the lane result is needed."""
    return LaneTask(question, acq)


def _with_parts(plan, fn, *a):
    """Run the search with the request's sub-queries visible to acquisition_search, always reset afterwards."""
    tok = search_providers.SUBQUERIES.set(plan)
    try:
        return fn(*a)
    finally:
        search_providers.SUBQUERIES.reset(tok)


def acquire(question: str, acq: dict, *, env=os.environ, clock=time.monotonic, deps_factory=None,
            served=None, discovered=None, prefer_recent: bool = False, topic_key: str = "",
            persist: bool = False) -> Outcome:
    """Never raises. See the module docstring for statuses. served (R2): doc keys the caller's stored-corpus attempt
    already tried (skipped here); None = the currently served acquired corpus (deps.corpus_keys). discovered (P0-5):
    the request's discovery results (list of source dicts), used only when inline.use_discovered is true."""
    try:
        if not inline_enabled(acq, env) or not str(question or "").strip():
            return Outcome("off")
        from vera import scope_router  # SCOPE-1022: the shared intent rule also guards inline acquisition
        if not scope_router.intent_clean(str(question)):
            return Outcome("off", guard=True)
        disc = None
        if discovered is not None:
            from vera import acq_discovered
            if acq_discovered.use_discovered(acq):
                disc = acq_discovered.discovered_candidates(discovered, acq, question)
        key = _cache_key(question, served, disc) if disc is not None else _cache_key(question, served)
        if persist:  # MIDTERMS-WP8-1106: a keep-searching request never replays, nor shares a slot with, a plain one
            key = key + ("persist",)
        now = clock()
        busy = False  # STREAM-1031: a busy refusal is also a stream event (emitted after the lock is released)
        with _I.lock:
            ttl = _b(acq, "result_ttl_s", 600)
            fttl = min(ttl, _b(acq, "failure_ttl_s", 60))  # DAR-TRACE-1022: a TRANSIENT failure is replayed <= 60 s
            for k in [k for k, (t, o) in _I.results.items() if now - t >= (fttl if o.status in TRANSIENT else ttl)]:
                del _I.results[k]
            if key in _I.results:
                return _I.results[key][1]
            slot = _I.inflight.get(key)
            owner = slot is None
            if not owner:
                if slot[2] >= int(_b(acq, "max_waiters", 4)):
                    busy = True
                else:
                    slot[2] += 1
            elif _I.active >= int(_b(acq, "max_concurrent", 2)):
                busy = True
            else:
                _I.active += 1
                slot = _I.inflight[key] = [threading.Event(), None, 0, stream_ctx.current()]
    except Exception as e:  # noqa: BLE001
        logger.warning("autoacq inline failed (%s)", type(e).__name__)
        return Outcome("error")
    if busy:
        _emit("acquire", "busy")
        return Outcome("busy")
    if not owner:  # an identical question is in flight: wait (bounded) for its outcome
        _wait_for_holder(slot, acq)
        with _I.lock:
            slot[2] -= 1
        if slot[1] is None:
            _emit("acquire", "busy")
            return Outcome("busy")
        return slot[1]
    out = Outcome("error")  # an unexpected failure is never reported as "searched" (no search notice is shown)
    try:
        out = _run(question, acq, env, clock, deps_factory, served,
                   **({"disc": disc} if disc is not None else {}),  # P0-5: flag off = the unchanged call
                   **({"prefer_recent": True} if prefer_recent else {}),
                   **({"topic_key": topic_key} if topic_key else {}),
                   **({"persist": True} if persist else {}))
    except Exception as e:  # noqa: BLE001
        logger.warning("autoacq inline failed (%s)", type(e).__name__)
    finally:
        with _I.lock:
            _I.active -= 1
            _I.inflight.pop(key, None)
            if out.status in SEARCHED and not out.no_cache:  # VER L1: never cache a cap/busy refusal; A6 S5: nor answer-only text
                _I.results[key] = (clock(), out)
                while len(_I.results) > int(_b(acq, "result_cache_max", 8)):
                    _I.results.popitem(last=False)
        slot[1] = out
        slot[0].set()
    return out


def _check_stop(ctx, st: dict) -> bool:
    """STREAM-1031: True when the request must start nothing new (disconnect, stall, providers exhausted). The first
    reason seen is recorded in st so the final status does not depend on a later clock reading."""
    r = ctx.stop_reason()
    if r is not None and st.get("stop") is None:
        st["stop"] = r
        logger.info("autoacq inline stop reason=%s since_progress_s=%d", r, int(ctx.since_progress()))  # code + count only
    return r is not None


def _supply_status(acq: dict, codes: dict, o) -> str:
    """NONASK-1103 WP-C: why a search that worked still stored fewer than the target (counts only, no titles/hosts).
    relevance_rejected: documents were fetched but none had enough relevant passages to keep; no_route: every
    candidate lacked a route VERA-DAR can fetch automatically; else resource_limited (true under-supply)."""
    def n(*keys):
        return sum(int(codes.get(acq_config.reason(acq, k), 0) or 0) for k in keys)
    rel = n("irrelevant", "few_spans") + int(o.answer_only)
    if int(o.stored) == 0 and rel > 0:
        return "relevance_rejected"
    noroute = n("no_doi", "not_oa", "arxiv_licence_hold") + int(codes.get("R-ACQ-PID-NOROUTE", 0) or 0)
    if noroute > 0 and rel == 0 and not o.documents and int(o.admitted) == 0:
        return "no_route"
    return "resource_limited"  # budget >= target, search worked, routes ran out: supply


def _left_reason(acq: dict, ctx, st: dict, late: bool) -> str:
    """WP7 (b): why a candidate with no recorded outcome was left unevaluated, by the real stop: the deadline, a
    stream stop (disconnect, stall, providers exhausted), the no-progress round stop, or the document budget. Only a
    candidate left with none of these is the plain candidate cut (R-ACQ-DEFER_CAP)."""
    if late:
        return acq_config.reason(acq, "deadline")
    if ctx is not None and _stop_reason(ctx, st) is not None:
        return acq_config.reason(acq, "stopped")
    stop = (st.get("acq6") or {}).get("stop")
    if stop == "no_progress":
        return acq_config.reason(acq, "no_progress")
    if stop == "ceiling31":
        return acq_config.reason(acq, "doc_budget")
    if stop == "disconnect" or (isinstance(stop, str) and stop in (stream_ctx.STALLED, stream_ctx.PROVIDERS_EXHAUSTED)):
        return acq_config.reason(acq, "stopped")
    return acq_config.reason(acq, "defer_cap")


def _stop_reason(ctx, st: dict):
    return st.get("stop") or getattr(ctx, "_stop", None)


def _provider_busy(acq: dict, codes: dict) -> bool:
    """PERF-MISS-1105-FIX2 F1: "sources are busy" only for a real provider cooldown, open breaker or daily cap
    (R-ACQ-RATELIMITED-<reason> with a rate_limit.PROVIDER_BUSY_REASONS reason). This process's own concurrency or
    pacing limiter, a caller deadline, or the plain legacy code never make the request "busy"."""
    for c, n in (codes or {}).items():
        if n and acq_config.limiter_why(acq, c) in rate_limit.PROVIDER_BUSY_REASONS:
            return True
    return False


def _emit_final(o: Outcome, acq: dict, codes: dict, stop) -> None:
    """The Stage A Outcome.acquisition surfaced as the final acquire event (counts only)."""
    a = o.acquisition or {}
    if a.get("status") == "met":
        _emit("acquire", "done", stored=a["stored"], target=a["target"])
    elif a.get("status") in ("resource_limited", "relevance_rejected", "no_route"):
        why = (WHY_STALLED if stop == stream_ctx.STALLED else WHY_EXHAUSTED if stop == stream_ctx.PROVIDERS_EXHAUSTED
               else WHY_HEURISTIC if a.get("stop") == "no_progress"
               else WHY_BUSY if _provider_busy(acq, codes) else WHY_NO_MORE)
        _emit("acquire", "limited", stored=a["stored"], target=a["target"], why=why)


def _wait_for_holder(slot, acq) -> None:
    """Wait for the in-flight identical question's outcome. Legacy: at most waiter_wait_s (45). Streamed path
    (STREAM-1031, OD-1031-NODEADLINE): 1 s ticks on the holder's Event, bounded by the holder's lifetime (the Event is
    always set when it ends), its stall rule, and this waiter's own disconnect; against a legacy holder (no stream ctx)
    the legacy bound applies. Each tick counts as progress for this waiter."""
    own, holder = stream_ctx.current(), slot[3] if len(slot) > 3 else None
    if own is None or holder is None:
        slot[0].wait(timeout=_b(acq, "waiter_wait_s", 45))
        return
    while not slot[0].wait(timeout=1.0):
        if own.cancelled or holder.should_stop():
            return
        own.touch()


def _run(question, acq, env, clock, deps_factory, served=None, disc=None, prefer_recent=False,
         topic_key="", persist=False) -> Outcome:
    """RATE-1022 C1: every rate-limit acquire() in this inline run sees its deadline. rate_limit uses time.monotonic,
    so the scope is now + deadline_s on that clock (the injected `clock` may be a test clock)."""
    # STREAM-1031 / OD-1031-NODEADLINE: on the streamed path there is no inline deadline (per-wait max_wait_s still applies)
    # RETRIEVAL-1109 A2 (owner H-T): on the streamed path a limiter wait up to rate_limits.json max_wait_stream_s is
    # waited out instead of skipped (stream_wait_scope); rates, Crawl-delay slots and cooldowns are unchanged.
    # request_scope: this run's own limiter accounting (waits, skips), reported on this Outcome only.
    streamed = stream_ctx.active()
    scope = None if streamed else time.monotonic() + float(_b(acq, "deadline_s", 45))
    with rate_limit.deadline_scope(scope), rate_limit.stream_wait_scope(streamed), rate_limit.request_scope() as led:
        o = _run_body(question, acq, env, clock, deps_factory, served,
                      **({"disc": disc} if disc is not None else {}),
                      **({"prefer_recent": True} if prefer_recent else {}),
                      **({"topic_key": topic_key} if topic_key else {}),
                      **({"persist": True} if persist else {}))
    try:
        o.limiter = led.summary()
        logger.info("autoacq ratelimit_request waited_ms=%d skips=%s", int(o.limiter["waited_s"] * 1000),
                    ",".join(f"{p}:{r}={n}" for p, rs in o.limiter["skips"].items() for r, n in rs.items()) or "-")
    except Exception:  # noqa: BLE001  telemetry never breaks the run
        pass
    return o


QUEUED = "queued_background"  # CONT-1023: candidate outcome = handed to the background acquisition queue


def _transient(acq, code) -> bool:
    """CONT-1023: a per-candidate failure that may succeed later (timeout, 5xx/non-200, quota, connection error)."""
    c = str(code or "")
    if c in {acq_config.reason(acq, k) for k in ("unreachable", "deadline", "crawl_delay")}:
        return True
    if acq_config.is_rate_limited(acq, c):  # F1: every per-reason limiter skip (nothing was sent)
        return True
    if c == acq_config.reason(acq, "fetch_timeout"):  # P0DTA-DIAG: the per-fetch wall clock (was R-ACQ-DEADLINE)
        return True
    return c.startswith("R-ACQ-ERR-") and any(w in c for w in ("Timeout", "Connection"))


def continue_eligible(acq, rec: dict) -> bool:
    """CONT-1023: True when the request ended this candidate's acquisition for a reason that can change (the inline
    deadline, the per-request candidate cut or document budget, a transient failure). Terminal reasons (licence, ND,
    injection, abstract-only, irrelevant, no route, already held) are never eligible."""
    o, why = rec.get("outcome"), rec.get("reason")
    if o in ("not_evaluated", "not_used"):
        return True
    if o == "dropped":  # the cut; or (owner) a URL-only candidate: its URL may lead to the source (follow_url)
        return why == acq_config.reason(acq, "defer_cap") or (why == acq_config.reason(acq, "no_doi")
                                                               and bool(rec.get("url")))
    return o == "rejected" and _transient(acq, why)


def _continue(question, acq, env, clock, recs: list, cands: list, deps_factory=None) -> dict | None:
    """CONT-1023: enqueue the eligible unfinished candidates as BEST-EFFORT background jobs (in-memory queue: lost on
    restart or spin-down, never retried; vera.auto_acquire.enqueue_candidates);
    mark each record queued_background (keeping why it was unfinished) or with the refusal. None = none eligible."""
    from vera import research_outcome as ro
    pick = [i for i, r in enumerate(recs) if continue_eligible(acq, r)]
    for i, r in enumerate(recs):
        if i not in pick and r.get("outcome") in ("dropped", "rejected"):
            r["background"] = "no_route" if ro.reason_from_code(r.get("reason")) == "no_route" else "terminal"
    # P0DTA-PRIO D4: within the hourly reserve (auto_acquire.BG_MIN_FREE_HOURLY_JOBS) routable full text is queued
    # first, then the rest (URL-only follow_url unchanged); only those the reserve blocks are skipped for quota
    pick = sorted(pick, key=lambda i: not aa.routable_fulltext(cands[i], acq))
    if not pick:
        return None
    res = aa.enqueue_candidates(question, [cands[i] for i in pick], cfg=acq, env=env, clock=clock,
                                deps_factory=deps_factory)  # None (production): the worker's own live wiring
    for j, i in enumerate(pick):
        r = recs[i]
        if j in res.get("queued", ()) or j in res.get("already", ()):
            r["unfinished"], r["outcome"] = r.get("outcome"), QUEUED
            r["background"] = "queued" if j in res.get("queued", ()) else "already_queued"
            if r.get("reason") == acq_config.reason(acq, "no_doi"):
                r["route"] = "follow_url"  # no identifier yet: the job follows the URL (auto_acquire.follow_url)
        elif res.get("status") == acq_config.reason(acq, "rate"):
            r["background"] = "skipped_quota"  # P0DTA-PRIO D4: blocked by the hourly/daily job cap or its reserve
        else:
            r["background"] = res.get("status")
    return {"status": res.get("status"), "queued": len(res.get("queued", ())),
            "already_queued": len(res.get("already", ())), "refused": len(res.get("refused", ()))}


def _cand_year(c) -> int | None:
    for k in ("year", "published", "publication_date"):
        m = re.match(r"\s*(\d{4})", str(c.get(k) or ""))
        if m:
            return int(m.group(1))
    return None


def _canonical_ref(url) -> str | None:
    from vera.portal_store import canonical_ref  # PORT-P2: additive match key
    return canonical_ref(url)


def _cand_record(c: dict) -> dict:
    """FRESH-1023 C3: one candidate's public record from what the candidate dict actually carries (None if absent)."""
    from vera import search_providers
    oa = c.get("oa") if isinstance(c.get("oa"), dict) else {}
    try:
        portal = search_providers.acq_portals(c)  # ROUTE-R3: every finding portal
        portal = portal[0] if len(portal) == 1 else portal
    except Exception:  # noqa: BLE001
        portal = c.get("provider")
    return {"portal": portal or c.get("provider"), "doi": aa._doi(c),
            "pmcid": aa._pmcid(c) or None, "openalex_id": c.get("openalex_id") or None,
            "arxiv_id": _arxiv_id_str(c.get("arxiv_id") or c.get("arxiv")), "url": c.get("url") or None,
            **({"discovered_url": c["discovered_url"]} if isinstance(c.get("discovered_url"), str) else {}),  # DF-6
            "published": c.get("published") or c.get("publication_date") or c.get("year") or None,
            "outcome": None, "reason": None, "found_by": list(c.get("found_by") or [portal or c.get("provider")])}


def _arxiv_id_str(v):
    """P0DTA-ARXFIX: arxiv_id is a string (id + version), never the route's {"id","version"} dict."""
    if isinstance(v, dict):
        return f"{v.get('id') or ''}{v.get('version') or ''}" or None
    return v or None


def _run_body(question, acq, env, clock, deps_factory, served=None, disc=None, prefer_recent=False,
              topic_key="", persist=False) -> Outcome:
    from vera import fulltext_loader as fl
    t0 = clock()
    ctx = stream_ctx.current()  # STREAM-1031: set only by POST /ask/stream; None = the legacy path, unchanged below
    deadline = INF if ctx is not None else t0 + _b(acq, "deadline_s", 45)  # OD-1031-NODEADLINE
    live = acq_config.live(acq)
    rounds_on = icfg_of(acq).get("acq6_rounds_enabled") is True  # A6 D1 kill switch (false = the single search)
    ms: dict = {}
    codes: dict = {}

    def count(code):
        codes[code] = codes.get(code, 0) + 1

    entries: dict = {}  # P0-5: id(discovered candidate) -> its public entry (disc only)
    st: dict = {"portals": None, "recs": [], "cands": []}  # FRESH-1023 C3: portal attempts + per-candidate records
    st["entries"] = entries  # WORDING-LABELS-1110 (b): _settle also updates the reading-list (discovered) entry

    def done(o: Outcome) -> Outcome:
        if st.get("lanes_extra"):  # RETRIEVAL-1109 A3: variant-round lane results, request-local (never cached)
            o.lanes_extra = list(st["lanes_extra"])
            o.no_cache = True  # A8 minor 1: never cached, links-only results included (matches the field comment)
        if st.get("web") is not None:  # A6-W: request-local; an outcome holding page text is never cached (R-037)
            o.web = st["web"]
            if o.web.pages:
                o.no_cache = True
        if st["portals"] is not None:
            o.portals = list(st["portals"])
            late = clock() >= deadline
            for r in st["recs"]:
                if r["outcome"] is None:
                    r["outcome"] = "not_evaluated"
                    r["reason"] = r.pop("skip", None) or _left_reason(acq, ctx, st, late)  # P0DTA-DIAG: recorded at the skip; else the real stop
            wid = st.get("web_ids") or ()
            for c, r in zip(st["cands"], st["recs"]):  # A6-W N4: web-route candidates stay terminal (never queued)
                if id(c) in wid:
                    r.update(outcome="web_answer_only", reason="web_route", background="terminal")
            # P0DTA-DIAG: one terminal status per candidate (status only)
            for r in st["recs"]:
                research_outcome.annotate_diag(r)
            o.background = _continue(question, acq, env, clock, st["recs"], st["cands"], deps_factory)  # CONT-1023
            o.candidates = [dict(r) for r in st["recs"]]
        if disc is not None and disc[1]:  # P0DTA-DIAG: held discovered records (preprint, no identifier) are candidates too
            held = [research_outcome.annotate_diag({
                "portal": h.get("service"), "doi": None, "pmcid": None, "openalex_id": None, "arxiv_id": None,
                "url": h.get("url") or None, "record_id": h.get("record_id"), "published": None,
                "outcome": h.get("outcome"), "reason": h.get("reason"), "found_by": [h.get("service")], "held": True})
                for h in disc[1]]
            o.candidates = (o.candidates or []) + held
        if disc is not None:
            o.discovered = [entries.get(id(c)) or {**c["discovery"], "outcome": "not_reached"} for c in disc[0]] + \
                [dict(h) for h in disc[1]]
            if disc[1]:
                for h in disc[1]:
                    count(h["reason"])
            # P0DTA-DIAG: same status vocabulary on the discovered entries
            for d in o.discovered:
                research_outcome.annotate_diag(d)
        if st["portals"] is not None:  # a search ran: report the outcome against the round target (counts only)
            tgt = acq_rounds.target_of(acq)
            stop = _stop_reason(ctx, st) if ctx is not None else None
            fb = st.get("fresh_budget")  # the pass's fresh-document budget (None if never reserved)
            if (o.admitted + (o.reused if count_reused(acq) else 0) + _n_ao(st) >= tgt) if rounds_on else (o.stored >= tgt):  # A6 D1: answer-eligible incl. answer-only
                aq_status = "met"
            elif ctx is not None and stop == stream_ctx.DISCONNECTED:  # STREAM-1031
                aq_status = "incomplete_disconnected"
            elif st.get("search_failed"):  # every provider failed: the search, not supply, is the cause
                aq_status = "search_failed"
            elif isinstance(fb, int) and fb < tgt:  # budget below the target: kill switch (2) or spent daily budget (0)
                aq_status = "capped"
            elif ctx is None and (clock() >= deadline or acq_config.reason(acq, "deadline") in codes):
                aq_status = "incomplete_deadline"  # legacy path only: the stream path has no deadline status
            else:
                aq_status = _supply_status(acq, codes, o)
            o.acquisition = {"status": aq_status, "stored": int(o.stored), "target": int(tgt),
                             **({"stop": stop} if ctx is not None and aq_status == "resource_limited"
                                and stop in (stream_ctx.STALLED, stream_ctx.PROVIDERS_EXHAUSTED) else {})}
            r6 = st.get("acq6")
            if r6 and (r6["rounds"] > 1 or r6.get("persist") or r6.get("stop") == "no_progress"):  # A6 D1 telemetry; WP8: always under persist
                o.acquisition.update({"rounds": int(r6["rounds"]), "stop": r6["stop"]})
            if r6 and r6.get("persist"):
                o.acquisition["persist"] = True
            if r6 and r6.get("stop") == "no_progress":  # WP8 (owner D2): the heuristic caused this stop; disclosed
                o.acquisition["heuristic"] = True
            if ctx is not None:
                _emit_final(o, acq, codes, stop)
        elif ctx is not None and o.status in CAPPED:  # the shared job caps refused: nothing searched
            _emit("acquire", "limited", stored=0, target=int(acq_rounds.target_of(acq)), why=WHY_CAPPED)
        ms["total"] = int((clock() - t0) * 1000)
        o.source_route = st.get("route")
        o.codes, o.ms = dict(sorted(codes.items())), ms
        aa._log("inline", status=o.status, admitted=o.admitted, stored=o.stored, persist_failed=o.persist_failed,
                n_answer_eligible=o.n_answer_eligible, n_persisted=o.n_persisted,
                n_below_persist_min=o.answer_only, reused=o.reused,
                **{f"n_{k}": v for k, v in o.codes.items()}, **{f"ms_{k}": v for k, v in ms.items()})
        _trace_done(o, st)
        try:  # RATE-1022 E2: portal names and counters only
            logger.info("autoacq %s", rate_limit.get_registry().log_line())
        except Exception:  # noqa: BLE001  telemetry never breaks the run
            pass
        return o

    if int(acq_config.bound(acq, "max_search_calls_per_job")) <= 0:  # VER L6: the job's search bound applies too
        return done(Outcome("off"))
    why = aa.reserve_job(acq, question, clock=clock)  # the SAME hourly/daily job caps as the background queue
    if why:
        count(why)
        return done(Outcome("rate"))
    deps = (deps_factory or default_deps)(acq, env)
    reserved, used = 0, 0
    try:
        budget = reserved = aa.docs_budget(acq, deps, clock=clock, reserve=True,
                                           per_job=acq_config.inline_fresh_docs(acq))  # OD-1031-CEIL31 per-prompt limit; VER L5: reserved under the lock
        st["fresh_budget"] = max(0, min(budget, acq_config.inline_fresh_docs(acq)))  # SA-FIX: read by done()

        def track(n_stored):  # ROUNDS-1028 (observational): the job's stored count and route exhaustion -> topic round
            if budget <= 0:
                return  # a spent daily budget is a cap effect, not a result of the topic's routes
            acq_rounds.observe(question, acq, n_stored, acq_rounds.routes_exhausted(
                len(ranked), len(plans), fresh_budget, acq_config.reason(acq, "deadline") in codes), "inline",
                hint=topic_key)
        # R2 defect 1: a spent docs_per_day no longer stops before the search: reusing already-stored documents
        # acquires nothing, so it is not capped. With budget 0 nothing new is resolved, fetched or stored.
        s = acq.get("search") or {}
        t = clock()
        search_failed = False
        # NONASK-1103 S3: per-sub-query search. ONE reservation (above) and ONE DDG call serve the whole question; only
        # the OpenAlex/Europe PMC/arXiv calls fan out, inside the scaled budget (search_providers.subquery_budget).
        sq_plan = _subquery_parts(question, acq)
        st["n_parts"] = len(sq_plan) if sq_plan else 1
        with search_providers.portal_scope() as portals_l:
            st["portals"] = portals_l
            _emit("search", "start", step=1)
            try:
                # P0DTA-PRIO D3: the search phase ends a fetch reserve (per_doc_timeout_s) before the inline deadline
                # (STREAM-1031: no inline deadline, so the search keeps only its own timeout_s)
                # DDG-REDUCE-1110 WP0: counted as "inline" (a caller's own label, e.g. freshness, is kept)
                with ddg_meter.site("inline" if ddg_meter.current() == "base" else ddg_meter.current()):
                    cands = list(_with_parts(sq_plan, deps.search, question, int(s.get("results_per_provider", 10)),
                                             float(s.get("timeout_s", 20)) if ctx is not None else
                                             max(1.0, min(float(s.get("timeout_s", 20)),
                                                          deadline - clock() - _b(acq, "per_doc_timeout_s", 25)))) or [])
                if not portals_l:  # FRESH-1023: a search callable that records no portals counts as ONE portal
                    search_providers.record_portal("acquisition_search", "completed", len(cands))
            except Exception:  # noqa: BLE001
                if not portals_l:
                    search_providers.record_portal("acquisition_search", "failed", 0)
                count(acq_config.reason(acq, "search_failed"))
                rt.event("acquire", "search_failed")
                st["search_failed"] = True  # SA-FIX: read by done() (acquisition status)
                if not (disc is not None and disc[0]):
                    return done(Outcome("search_failed"))
                cands, search_failed = [], True  # P0-5: the discovered candidates still go through acquisition
        ms["search"] = int((clock() - t) * 1000)
        _web_read(question, acq, cands, st, ctx, count, clock=clock, deadline=deadline,  # A6-W: round 0 only (N3)
                  max_fetches=max(0, min(budget, acq_config.inline_fresh_docs(acq))))
        if ctx is not None:
            ctx.touch()  # the search rung finished: progress
            _check_stop(ctx, st)  # T10d: a disconnect/stop during the search starts no fetch below
        fdcfg = aa._fd_cfg()
        icfg = acq_config.ingest_cfg(fl.load_config(), acq)
        # R2: skip only what the stored-corpus attempt actually tried. The served acquired corpus may have finished
        # loading in the background since (a new process), so its keys are NOT "already tried".
        skip = set(served) if served is not None else set(deps.corpus_keys() or ())
        if disc is not None:  # P0-5: discovered first (merge keeps the first occurrence; DOI/PMCID de-duplication)
            from vera import acq_discovered
            from vera.search_providers import merge_sources
            cands = merge_sources(list(disc[0]), cands)
        if prefer_recent:  # FRESH-1023 C3: stable, newest first, unknown year last (ties then break toward recent)
            cands = sorted(cands, key=lambda c: (1, 0) if not isinstance(c, dict) or _cand_year(c) is None
                           else (0, -_cand_year(c)))
        rec_by: dict = {}
        for c in cands:
            if isinstance(c, dict):
                rec_by[id(c)] = _cand_record(c)
                if id(c) in st.get("web_ids", ()):  # A6-W N4: read on the answer-only web route; never handed to the queue
                    rec_by[id(c)].update(outcome="web_answer_only", reason="web_route", background="terminal")
                st["recs"].append(rec_by[id(c)])
                st["cands"].append(c)
        drops: dict = {}
        route_rec = None
        if source_type.enabled(acq):  # T1: per-request, question-derived, off by default
            route_rec = source_type.classify(question, acq)
            st["route"] = route_rec
        route = route_rec["route"] if route_rec else None
        protected = _protected_keys(cands, acq, deps) if aa._topic_fit_active(acq, route) else set()
        ranked = aa.rank_candidates(question, [c for c in cands if id(c) not in st.get("web_ids", ())] if st.get("web_ids")
                                    else cands, acq, skip, fdcfg, count, drops=drops, route=route,
                                    protected=protected, parts=[(t, e) for t, e in (sq_plan or ())])
        for cid, why in drops.items():
            if cid in rec_by:
                rec_by[cid]["outcome"], rec_by[cid]["reason"] = "dropped", why
        if disc is not None:
            entries.update(acq_discovered.gate_a(disc[0], cands, ranked, skip, acq, drops))
        # DAR-INLINE-1021-R2: ranked candidates ALREADY stored as acquired documents are answered from their stored rows
        # (read-only) instead of being skipped as R-ACQ-KNOWN while lower-ranked documents are acquired in their place
        # (the run-2 failure: a new process, the served acquired corpus still loading in the background).
        t = clock()
        reuse_info: dict = {}
        reused, ranked_left = _stored_for(ranked, acq, deps, count, question=question, fdcfg=fdcfg, icfg=icfg,
                                          clock=clock, deadline=deadline, info=reuse_info,
                                          **({"ctx": ctx, "max_checks": int(live["max_reuse_checks_per_prompt"])}
                                             if ctx is not None else {}))
        for cid, (oc, why) in reuse_info.items():
            if cid in rec_by:
                rec_by[cid]["outcome"], rec_by[cid]["reason"] = oc, why
        if disc is not None:
            left = {id(r[2]) for r in ranked_left}
            for r in ranked:
                e = entries.get(id(r[2]))
                if e is not None and id(r[2]) not in left:
                    e["outcome"] = "already_stored"
        web_n = int(st.get("n_web_fetched", 0) or 0)  # WP6-P2b S2: web fetches count against max_docs_per_prompt
        ranked = aa.reserve_reads(ranked_left, max(0, min(budget, acq_config.inline_fresh_docs(acq)) - web_n), acq, route)
        ms["reuse"] = int((clock() - t) * 1000)
        _emit("search", "done", found=len(cands), shortlisted=len(ranked))
        # ACQ S4-0 L4b-0 (B3): the fresh budget is separate from reuse; documents answered from stored rows no longer
        # reduce it. The per-prompt value comes from acq_config.inline_fresh_docs (max_docs_per_prompt 31 with the flag on, else
        # max_docs_per_job 2), still bounded by the remaining daily budget.
        fresh_budget = max(0, min(budget, acq_config.inline_fresh_docs(acq)) - web_n)
        t = clock()
        per: dict = {}  # FRESH-1023 C3: always collected (per-candidate records); entries stay disc-only
        sk: dict = {}  # P0DTA-DIAG: rank index -> why it was skipped, recorded where it happened
        if ctx is not None and _check_stop(ctx, st):  # STREAM-1031: stopped before any fetch: nothing starts
            plans = []
            sk.update({i: acq_config.reason(acq, "stopped") for i in range(len(ranked))})  # WP7: the stop, not the cut
        elif fresh_budget > 0 and ranked and deadline - clock() < _b(acq, "per_doc_timeout_s", 25):
            # P0DTA-PRIO D3: one full fetch no longer fits: nothing starts; each candidate is a deadline skip
            # (P0DTA-INTEG: recorded through the DIAG skip map, so done() and annotate_diag give skipped/deadline)
            count(acq_config.reason(acq, "deadline"))
            sk.update({i: acq_config.reason(acq, "deadline") for i in range(len(ranked))})
            plans = []
        else:
            plans = _evaluate_parallel(question, ranked, acq, deps, icfg, fdcfg, clock, deadline, fresh_budget, count,
                                       ms, per=per, skips=sk, **({"ctx": ctx, "st": st}
                                                                if ctx is not None else {})) if fresh_budget > 0 else []
        if fresh_budget <= 0:  # nothing evaluated: the daily document quota (budget 0) or the per-job document limit
            sk.update({i: acq_config.reason(acq, "daily_cap" if budget <= 0 else "doc_budget") for i in range(len(ranked))})
        by_plan: dict = {}
        kept_ids = {id(p) for p in plans}
        rec_plan: dict = {}
        for i, r in enumerate(ranked):
            rc = rec_by.get(id(r[2]))
            got = per.get(i)
            if rc is None or got is None:
                if rc is not None and i in sk:
                    rc["skip"] = sk[i]
                continue  # unfinished: done() marks it not_evaluated
            if isinstance(got, str):
                rc["outcome"], rc["reason"] = "rejected", got
            elif id(got) in kept_ids:
                rc["outcome"] = "admitted"
                if not rc.get("pmcid") and got.get("resolved_pmcid"):  # DDGFIX-1023 DF-7: the PMCID the route used
                    rc["pmcid"] = got["resolved_pmcid"]
                rc["n_chunks"] = len(got.get("spans") or ())  # PORT-P2: additive (url and ids unchanged)
                rc["canonical_url"], rc["canonical_ref"] = got.get("canonical_url"), _canonical_ref(got.get("canonical_url"))
                rec_plan[id(got)] = rc
            else:
                rc["outcome"], rc["reason"] = "not_used", None
        if disc is not None:
            kept = {id(p) for p in plans}
            for i, r in enumerate(ranked):
                e = entries.get(id(r[2]))
                if e is None:
                    continue
                got = per.get(i)
                if got is None:
                    e["outcome"] = "not_evaluated"  # deadline or the document budget was reached first
                    if i in sk:  # DDFIX-1023/REL: the recorded skip reason, not the generic outcome text
                        e["reason"] = sk[i]
                elif isinstance(got, str):
                    e["outcome"], e["reason"] = "rejected", got
                    if got in (acq_config.reason(acq, "irrelevant"), acq_config.reason(acq, "few_spans")):
                        e["gate_b"] = {"decision": "reject", "basis": acq_discovered.GATE_B_REJECT_BASIS}
                elif id(got) in kept:
                    e["outcome"], e["gate_b"] = "admitted", {"decision": "admit", "basis": acq_discovered.GATE_B_BASIS}
                    by_plan[id(got)] = e
                else:
                    e["outcome"] = "not_used"  # admitted but over the document budget
        if search_failed and not plans and not reused[0]:
            return done(Outcome("search_failed"))
        ms["evaluate"] = int((clock() - t) * 1000)
        ao_ids: set = set()  # A6 S5: ids of plans found ONLY by variant queries (answer-only, never stored)
        if rounds_on and fresh_budget > 0 and not search_failed and len(per) >= len(ranked):
            t = clock()
            extra, reused = _extra_rounds(question, acq, deps, icfg, fdcfg, clock, deadline, ctx, st, count, ms, skip,
                                          plans, reused, fresh_budget, route, **({"persist": True} if persist else {}))
            ao_ids = {id(p) for p in extra}
            plans = plans + extra
            ms["rounds"] = int((clock() - t) * 1000)
        if not plans and not reused[0]:
            if budget <= 0:
                count(acq_config.reason(acq, "daily_cap"))
                return done(Outcome("daily_cap"))
            late = clock() >= deadline or acq_config.reason(acq, "deadline") in codes
            track(0)
            return done(Outcome("deadline" if late else "none"))
        out = Outcome("ok", admitted=len(plans))
        out.reused = len(reused[0])
        failed = 0
        t = clock()
        for plan in plans:  # one INSERT-only transaction per document (WoStore); DOI dedupe = canonical_state
            e = by_plan.get(id(plan))
            rc = rec_plan.get(id(plan))
            if id(plan) in ao_ids or plan.get("answer_only") or plan.get("lane") or plan.get("source_lane"):
                # A6 S5: variant-only documents answer only (no write); like below_persist_min plans they may sit in the
                # result cache (they passed the licence gate). Lane/web pages (R-037) are never cached either.
                if plan.get("answer_only") or plan.get("lane") or plan.get("source_lane"):
                    out.no_cache = True
                count(acq_config.reason(acq, BELOW_PERSIST_MIN))
                out.documents.append({"persisted": False, "reason": BELOW_PERSIST_MIN})
                continue
            if not aa.persist_eligible(plan, acq):  # ONE-SPAN-1021: answers this request only; never written
                count(acq_config.reason(acq, BELOW_PERSIST_MIN))
                out.documents.append({"persisted": False, "reason": BELOW_PERSIST_MIN})
                if rc is not None:
                    rc["persisted"], rc["persist_reason"] = False, BELOW_PERSIST_MIN
                if e is not None:
                    e["persisted"] = False
                continue
            try:
                w = deps.store.write(plan)
                if e is not None:
                    e["persisted"] = w.get("status") == "stored"
                if rc is not None:
                    rc["persisted"] = w.get("status") == "stored"
                    rc["persist_reason"] = None if rc["persisted"] else "in_corpus"
                if w.get("status") == "stored":
                    out.stored += 1
                    if ctx is not None:
                        ctx.touch()  # a document was stored: progress
                    count(acq_config.reason(acq, "stored"))
                    out.documents.append({"persisted": True})
                else:
                    count(acq_config.reason(acq, "in_corpus"))
                    out.documents.append({"persisted": False, "reason": "in_corpus"})
            except Exception as e:  # noqa: BLE001  answer anyway, marked unsaved (see module docstring)
                failed += 1
                if rc is not None:
                    rc["persisted"], rc["persist_reason"] = False, "write_failed"
                count(aa.error_code(acq, e))
                out.documents.append({"persisted": False, "reason": "write_failed"})
                logger.warning("autoacq inline persist failed (%s)", type(e).__name__)
        out.persist_failed = failed > 0 and out.stored == 0  # VER L4: the note says unsaved only if none was saved
        used = out.stored  # DAR-TRACE-1022: only STORED documents count against docs_per_day (owner wording "stored documents/day"; the cap itself is null = off since 2026-10-22)
        ms["write"] = int((clock() - t) * 1000)
        t = clock()
        out.corpus = ephemeral_corpus(plans, acq, fdcfg, icfg, stored=reused)
        track(out.stored)
        ms["corpus"] = int((clock() - t) * 1000)
        if out.corpus is None:
            out.status = "unanswered"  # VER L3: documents passed the checks but no usable corpus; never "none"
        return done(out)
    finally:
        aa.settle_docs(reserved, used, clock=clock)
        try:
            if deps.close is not None:
                deps.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("autoacq close failed (%s)", type(e).__name__)


def _ckeys(c: dict) -> set:
    """A6 D1: the canonical keys of one candidate (doc_key_set plus its DOI/PMCID identity): the `tried` set."""
    from vera.dar_corpus import doc_key_set
    doi = aa._doi(c)
    isbn = aa.candidate_isbn(c)
    ks = set(doc_key_set(doi, c.get("url"), (isbn or {}).get("isbn")))
    ks.add("doi:" + doi if doi else "pmc:" + str(aa._pmcid(c) or id(c)))
    return ks


def web_read_enabled(acq: dict | None) -> bool:
    """A6-W kill switch inline.web_read_enabled (ships true; false = no web route at all, pre-A6-W behaviour)."""
    return icfg_of(acq).get("web_read_enabled") is True


def _web_read(question, acq, cands, st, ctx, count, clock=time.monotonic, deadline=None, max_fetches=None,
              again: bool = False) -> None:
    """A6-W (R-045/R-047/R-048/R-050, review N3/N4): read the round-0 general-web results (no DOI/PMCID/arXiv id)
    through vera.web_route (read gate before any fetch, vera.public_fetch, use_for_answer gate; always answer-only).
    Their candidate records are marked terminal so CONT-1023 never hands them to enqueue_candidates/run_job (N4).
    Never raises; counts only reach the log.
    WP6-P2b S2: before EACH fetch the route checks the stop/disconnect signal (streamed path) or the non-stream
    `deadline` less the existing fetch reserve (per_doc_timeout_s); fetches started are bounded by `max_fetches` (the
    pass's max_docs_per_prompt budget) and recorded in st["n_web_fetched"], which the caller subtracts from the fresh
    budget. One call per question (a second call is a no-op), one fetch per host (redirects included).
    again=True (PERF-MISS-1105-FIX2 F5, news-intent variant rounds only): a further pass over that round's new web
    results, merged into st["web"]; the same gates, and hosts already read in this request are not read again."""
    if not web_read_enabled(acq) or (ctx is not None and ctx.stop_reason() is not None) or ("web" in st and not again):
        return
    reserve = _b(acq, "per_doc_timeout_s", 25)

    def should_stop():
        if ctx is not None:
            return "stopped" if ctx.stop_reason() is not None else None
        if deadline is not None and deadline - clock() < reserve:
            return "deadline"
        return None
    try:
        from vera import web_route
        web = [c for c in cands if isinstance(c, dict) and web_route.is_web_candidate(c)]
        prev = st.get("web") if again else None
        if again:  # F5: one fetch per host per request, across passes (every host offered before, read or not)
            seen_hosts = set(st.get("web_hosts") or ())
            web = [c for c in web if web_route._host(str(c.get("url") or "")) not in seen_hosts]
        if not web:
            return
        if max_fetches is not None and max_fetches <= 0:
            return
        st["web_hosts"] = set(st.get("web_hosts") or ()) | {web_route._host(str(c.get("url") or "")) for c in web}
        res = web_route.read_web(question, web, should_stop=should_stop, max_fetches=max_fetches)
        if prev is not None:
            res = web_route.WebResult(pages=list(prev.pages) + list(res.pages), listed=list(prev.listed) + list(res.listed),
                                      dev=list(prev.dev) + list(res.dev), fetched=int(prev.fetched) + int(res.fetched))
        st["web"], st["n_web"], st["n_web_fetched"] = res, len(res.pages), int(res.fetched)
        if any((d.get("gate") or {}).get("read") == "read_deadline" for d in res.dev):
            count(acq_config.reason(acq, "deadline"))
        st["web_ids"] = set(st.get("web_ids") or ()) | {id(c) for c in web}  # marked terminal where built (N4)
        count("R-ACQ-WEB-READ") if res.pages else None
        if ctx is not None:
            ctx.touch()
    except Exception as e:  # noqa: BLE001
        logger.warning("autoacq web route failed (%s)", type(e).__name__)


def count_reused(acq) -> bool:
    """OD-1109-SIX (owner 2026-11-09, verbatim: "ingest at least 6 new documents found from external search results,
    including from following links from portals until arriving at original hosts of sources"): already-stored documents
    re-found by a search are NOT new, so they do not count toward the round target unless
    bounds.acq_round_count_reused is exactly true (the pre-SIX behaviour)."""
    return ((acq or {}).get("bounds") or {}).get("acq_round_count_reused") is True


def rounds_share_job(acq) -> bool:
    """OD-1109-QUOTA: one question's search rounds count as ONE job against jobs_per_hour (bounds.acq_rounds_one_job)."""
    return ((acq or {}).get("bounds") or {}).get("acq_rounds_one_job") is True


def _rn(acq, reused) -> int:
    """Reused stored documents counted toward the round target (0 under OD-1109-SIX; see count_reused)."""
    return len(reused[0]) if count_reused(acq) else 0


def new_documents(o) -> int:
    """New documents this acquisition obtained (fresh admitted full texts + request-local web pages read), for the
    page statement. Never raises."""
    try:
        web = getattr(o, "web", None)
        return int(getattr(o, "admitted", 0) or 0) + len(getattr(web, "pages", None) or ())
    except Exception:  # noqa: BLE001
        return 0


def _n_ao(st: dict) -> int:
    """Answer-only request-local pages counted toward the round target: web route pages (A6-W) and, RETRIEVAL-1109 A3,
    variant-round reporting-lane pages."""
    return int(st.get("n_web", 0) or 0) + int(st.get("n_lane", 0) or 0)


def _merge_lanes(lanes, extra):
    """RETRIEVAL-1109 A3: the round-0 lane result plus the variant rounds' lane results (links deduplicated by URL,
    pages by final URL, attributions once). `lanes` unchanged when there is nothing extra."""
    extra = [x for x in (extra or ()) if x is not None and (getattr(x, "links", None) or getattr(x, "pages", None))]
    if not extra:
        return lanes
    from vera import lane_evidence
    base = lanes if lanes is not None else lane_evidence.LaneResult()
    links, pages = list(base.links), list(base.pages)
    attrs, diag = list(base.attributions), list(getattr(base, "diag", None) or [])
    seen_l = {str(lk.get("url")) for lk in links if isinstance(lk, dict)}
    seen_p = {str(getattr(pg, "url", "")) for pg in pages}
    for x in extra:
        for lk in x.links:
            if isinstance(lk, dict) and str(lk.get("url")) not in seen_l:
                seen_l.add(str(lk.get("url")))
                links.append(lk)
        for pg in x.pages:
            if str(getattr(pg, "url", "")) not in seen_p:
                seen_p.add(str(getattr(pg, "url", "")))
                pages.append(pg)
        attrs += [a for a in x.attributions if a not in attrs]
        diag += list(getattr(x, "diag", None) or [])
    return lane_evidence.LaneResult(links=links, pages=pages, attributions=attrs, diag=diag)


def variant_lane_on(acq: dict) -> bool:
    """RETRIEVAL-1109 A3 switch: config/auto_acquire.json inline.variant_lane_enabled, exactly true."""
    return icfg_of(acq).get("variant_lane_enabled") is True


def _variant_lane(question: str, q: str, acq: dict, st: dict, collect=None) -> int:
    """RETRIEVAL-1109 A3 (NOANSWER-ROOT-1109 plan A3): one reporting-lane pass for a news question's variant round,
    with the VARIANT text as the lane query (the original question still decides news intent). Every lane rule is
    unchanged: GDELT spacing/backoff/circuit/daily cap (vera.reporting_lane), outlet verdicts, ND link-only unless the
    nc_basis + nd_news gates allow answer-only, robots and page signals. Pages are answer-only and request-local.
    Returns the number of pages read; never raises."""
    try:
        from vera import lane_evidence, reporting_lane, subquery
        if not variant_lane_on(acq) or not reporting_lane.lane_enabled():
            return 0
        # A8 minor 2: lane_evidence.collect counts reads per call, so the variant rounds together stop once they have
        # read the lane's per-request read cap (round 0 keeps its own cap): up to 3 x cap - 1 lane pages per request (round 0 may read up to cap; each of the two variant rounds starts only while fewer than cap have been read, so the total is bounded by 3 x cap - 1).
        if int(st.get("n_lane", 0) or 0) >= int(lane_evidence._read_cap()):
            return 0
        plan = subquery.SubQueryPlan(original=str(question), parts=(subquery.SubPart("V", str(q)),), rule="single")
        res = (collect or lane_evidence.collect)(str(question), plan=plan)
        if res is None:
            return 0
        st.setdefault("lanes_extra", []).append(res)
        return len(res.pages)
    except Exception as e:  # noqa: BLE001
        logger.warning("variant lane failed (%s)", type(e).__name__)
        return 0


def with_web(o, lanes, question=None):
    """A6-W: the request's lane result with the web route's pages and reading-list links appended (a new LaneResult;
    None when neither exists). Request-local only, like lane_pass."""
    web = getattr(o, "web", None) if o is not None else None
    from vera import lane_evidence
    from vera import lane_router
    lanes = _merge_lanes(lanes, getattr(o, "lanes_extra", None) if o is not None else None)  # RETRIEVAL-1109 A3
    if question and lane_router.reporting_cue(str(question), str(question)):  # NT-3, LINKS-RC-1107: news-intent only
        lanes = lane_evidence.merge_wiki_links(lanes, question)
    if web is None or not (web.pages or web.listed):
        return lanes
    keep = ("lane", "label", "title", "url", "published", "read", "reference")
    links = [{k: d.get(k) for k in keep if d.get(k) is not None} for d in web.listed]
    links += [{"lane": "web", "label": getattr(p, "label", "web source"), "title": p.title, "url": p.url, "read": True}
              for p in web.pages]
    base = lanes if lanes is not None else lane_evidence.LaneResult()
    return lane_evidence.LaneResult(links=list(base.links) + links, pages=list(base.pages) + list(web.pages),
                                    attributions=list(base.attributions))


# Provisional stopping heuristic (owner D2, OD-1106-MIDTERMS-D1D4, 2026-11-05): not a hard limit; a keep-searching
# request (persist, vera.request_intent) overrides it; a stop it causes is disclosed (acquisition.heuristic, why 6).
# Origin (2026-11-04): the round loop ends after NO_PROGRESS_ROUNDS consecutive variant rounds that
# yield ZERO new usable documents (no new answer-eligible document: fresh admitted or reused stored), so an empty-result question does not spend the hourly jobs down to the background floor.
# Exit reason "no_progress" (Outcome.acquisition["stop"], telemetry only). CALIBRATION NOTE: 2 is a structural choice,
# not a tuned value: one empty round can be a provider blip, two in a row on different variants indicate the variant
# list is not reaching new sources; no live distribution of per-round yields exists yet. Re-check against owner-run
# logs of rounds/stop per question (how often a third empty round would have found something) and change only then.
NO_PROGRESS_ROUNDS = 2


# MULTIQ-1109 F3: set by ask_service._answer_parts around ONE question of a multi-question prompt. ROUND_CAP = the most
# searches (round 0 + variant rounds) that question may run; ROUND_CUT collects True when the cap ended its search early.
# None (the default, and every single-question request) = unchanged behaviour.
ROUND_CAP: contextvars.ContextVar = contextvars.ContextVar("multiq_round_cap", default=None)
ROUND_CUT: contextvars.ContextVar = contextvars.ContextVar("multiq_round_cut", default=None)


def _extra_rounds(question, acq, deps, icfg, fdcfg, clock, deadline, ctx, st, count, ms, skip, plans, reused,
                  fresh_budget, route, persist=False):
    """NONASK-1103 A6 D1 (A6-2, A6-5): rounds 2 and later. While fewer than the target (acq_rounds, 6) answer-eligible
    documents (fresh + reused + answer-only/lane pages) exist and the previous ranked list is exhausted, take the next
    query variant (vera.query_variants6: finite list V, never the raw sentence) and search again. Every round reserves
    one job (aa.reserve_job, which also refuses below aa.BG_MIN_FREE_HOURLY_JOBS free hourly jobs); a refusal stops the
    loop. `tried` canonical keys are never re-evaluated. Documents found only here are ANSWER-ONLY (S5): returned for the
    caller to answer from, never stored or cached.
    Bounds: STREAMED path (N1, OD-1031-NODEADLINE unchanged): no wall-clock bound; at most |V| rounds, each
    <= search timeout_s + one evaluation pass (per-fetch per_doc_timeout_s), a stop/disconnect check before every round
    and fetch start, the job floor and stall_s. NON-STREAM path: `deadline` (45 s from start) is absolute across all
    rounds: a round starts only if its search can end per_doc_timeout_s before it, and no fetch starts if one no longer
    fits (N5), so the worst case is deadline_s + one straggler per_doc_timeout_s.
    MIDTERMS-WP8-1106 (owner D1/D2): persist (a keep-searching request) lifts ONLY the no-progress heuristic; every
    other bound above (met, ceiling31, the job floor, stream stops, the deadline, search_failed, the finite variant
    list) still applies. No new limit.
    Returns (extra answer-only plans, reused). st["acq6"] = {"rounds": searches incl. round 0, "stop": reason}."""
    from vera import query_variants6
    tgt = acq_rounds.target_of(acq)
    per_doc = _b(acq, "per_doc_timeout_s", 25)
    s = acq.get("search") or {}
    tried: set = set()
    # WP7 (a): round-0 candidates cut by the shortlist (R-ACQ-DEFER_CAP) were never evaluated, so they are NOT tried
    # yet; they are evaluated, best first, before any new variant search, and join `tried` only once evaluated.
    pending = _cut_deferred(acq, st)
    pend_ids = {id(c) for c in pending}
    for c in st["cands"]:
        if id(c) not in pend_ids:
            tried |= _ckeys(c)
    extra: list = []
    rounds, stop = 1, "variants_exhausted"
    r6 = st["acq6"] = {"rounds": 1, "stop": stop}
    if persist:
        r6["persist"] = True
    idle, last = 0, None  # consecutive variant rounds that added zero new usable documents (NO_PROGRESS_ROUNDS)
    news = news_question(question)  # F5
    try:
        variants = query_variants6.variants(question)
    except Exception:  # noqa: BLE001
        variants = []
    stop_pre = None
    passes = 0
    while pending and passes <= len(pend_ids):  # each pass takes at least one candidate off `pending`; the cap is a guard
        passes += 1
        eligible = len(plans) + len(extra) + _rn(acq, reused) + _n_ao(st)
        remaining = fresh_budget - len(plans) - len(extra)
        if eligible >= tgt:
            stop_pre = "met"
        elif remaining <= 0:
            stop_pre = "ceiling31"
        elif ctx is not None and _check_stop(ctx, st):
            stop_pre = "disconnect" if st.get("stop") == stream_ctx.DISCONNECTED else str(st.get("stop"))
        elif ctx is None and deadline - clock() < per_doc:  # N5: one full fetch no longer fits: nothing starts
            count(acq_config.reason(acq, "deadline"))
            stop_pre = "deadline"
        if stop_pre:
            break
        drops_p: dict = {}
        ranked = aa.rank_candidates(question, pending, acq, set(skip) | tried, fdcfg, count, drops=drops_p, route=route,
                                    protected=None)
        in_ranked = {id(r[2]) for r in ranked}
        cut_again = acq_config.reason(acq, "defer_cap")
        nxt = []
        for c in pending:
            why = drops_p.get(id(c))
            if id(c) in in_ranked:
                continue
            if why == cut_again:
                nxt.append(c)  # beyond this pass's cut: stays pending, not tried
            elif why:
                _settle(st, id(c), "dropped", why)  # duplicate / known / licence: its real reason
        pending = nxt
        reuse_info = {}
        got_reused, left = _stored_for(ranked, acq, deps, count, question=question, fdcfg=fdcfg, icfg=icfg, clock=clock,
                                       deadline=deadline, info=reuse_info,
                                       **({"ctx": ctx, "max_checks": max(0, int(acq_config.live(acq)["max_reuse_checks_per_prompt"])
                                                                          - sum(1 for r in st["recs"] if r.get("outcome") in ("reused", "reuse_rejected")))}
                                          if ctx is not None else {}))  # the per-prompt check bound is shared with round 0
        if got_reused[0]:
            reused = (list(reused[0]) + list(got_reused[0]), list(reused[1]) + list(got_reused[1]))
        for cid, (oc, why) in reuse_info.items():
            _settle(st, cid, oc, why)
        need = max(1, min(remaining, tgt - (len(plans) + len(extra) + _rn(acq, reused))))
        taken = aa.reserve_reads(left, need, acq, route)
        taken_ids = {id(r[2]) for r in taken}
        reuse_ids = {cid for cid in reuse_info}
        for r in ranked:
            if id(r[2]) in taken_ids or id(r[2]) in reuse_ids:
                tried |= _ckeys(r[2])  # evaluated (or settled by reuse) now: only then is it tried
            else:
                pending.append(r[2])  # trimmed by the read reserve: stays pending for a later pass
        if not taken:
            continue
        per_p: dict = {}
        sk_p: dict = {}
        got = _evaluate_parallel(question, taken, acq, deps, icfg, fdcfg, clock, deadline, need, count, ms,
                                 per=per_p, skips=sk_p, **({"ctx": ctx, "st": st} if ctx is not None else {}))
        extra += got
        kept = {id(p) for p in got}
        for i, r in enumerate(taken):
            res = per_p.get(i)
            if isinstance(res, str):
                _settle(st, id(r[2]), "rejected", res)
            elif res is not None and id(res) in kept:
                _settle(st, id(r[2]), "admitted", None)
            elif res is not None:
                _settle(st, id(r[2]), "not_used", None)
            else:  # never finished (stop, budget): not tried, so a later pass or the relabel below sees it
                pending.append(r[2])
                tried.difference_update(_ckeys(r[2]))
                if i in sk_p:
                    _settle(st, id(r[2]), "not_evaluated", sk_p[i])
    r6["deferred_evaluated"] = len(pend_ids) - len(pending)
    if stop_pre:
        stop = stop_pre
    for q in ([] if stop_pre else variants):
        cur = len(extra) + _rn(acq, reused) + int(st.get("n_lane", 0) or 0)  # usable = answer-eligible documents gained
        # (fresh or reused; RETRIEVAL-1109 A3: variant-round reporting-lane pages too)
        if last is not None:
            idle = idle + 1 if cur == last else 0
            if idle >= NO_PROGRESS_ROUNDS:  # checked BEFORE the next reservation: no job is spent on a third round
                if not persist:
                    stop = "no_progress"
                    break
                r6["heuristic_overridden"] = 1  # WP8: keep-searching overrides the heuristic (telemetry only)
        last = cur
        eligible = len(plans) + len(extra) + _rn(acq, reused) + _n_ao(st)
        remaining = fresh_budget - len(plans) - len(extra)
        if eligible >= tgt:
            stop = "met"
            break
        if remaining <= 0:
            stop = "ceiling31"
            break
        if ctx is not None and _check_stop(ctx, st):
            stop = "disconnect" if st.get("stop") == stream_ctx.DISCONNECTED else str(st.get("stop"))
            break
        if ctx is None and deadline - clock() - per_doc < 1.0:  # N5: the search must end a fetch reserve before it
            count(acq_config.reason(acq, "deadline"))
            stop = "deadline"
            break
        _cap = ROUND_CAP.get()
        if isinstance(_cap, int) and not isinstance(_cap, bool) and rounds >= _cap:  # MULTIQ-1109 F3: fair share of searches
            stop = "round_cap"
            _cut = ROUND_CUT.get()
            if _cut is not None:
                _cut.append(True)
            rt.event("acquire", "round_cap", counts={"rounds": rounds})
            break
        # OD-1109-QUOTA: with bounds.acq_rounds_one_job exactly true, a question's variant rounds share the ONE job its
        # round 0 reserved (normal use must not exhaust jobs_per_hour); per-source rate limits, robots and Crawl-delay
        # still bound every fetch. Otherwise each round reserves one job (the pre-QUOTA rule).
        why = None if rounds_share_job(acq) else aa.reserve_job(acq, question, clock=clock,
                                                                floor=aa.BG_MIN_FREE_HOURLY_JOBS)
        if why:
            count(why)
            stop = "rate"
            break
        rounds += 1
        r6["rounds"] = rounds
        _emit("search", "start", step=rounds)
        with search_providers.portal_scope() as pl:
            try:
                # N3: DDG once per question; no DDG/Wikipedia here. PERF-MISS-1105-FIX2 F5: a news-intent question is
                # not forced to scholarly-only portals (a scholarly-only round cannot find current reporting); the
                # web results it finds go through the same A6-W web route gates as round 0.
                so_tok = search_providers.SCHOLARLY_ONLY.set(not news)
                site_tok = ddg_meter.SITE.set("variant")  # DDG-REDUCE-1110 WP0: counted as a variant round
                nd_tok = ddg_reduce.NO_DDG.set(ddg_reduce.on("variant_rounds_no_ddg"))  # C2: news rounds keep GDELT, not DDG
                try:
                    cands = list(_with_parts(None, deps.search, q, int(s.get("results_per_provider", 10)),
                                             float(s.get("timeout_s", 20)) if ctx is not None else
                                             max(1.0, min(float(s.get("timeout_s", 20)), deadline - clock() - per_doc))) or [])
                finally:
                    ddg_reduce.NO_DDG.reset(nd_tok)
                    ddg_meter.SITE.reset(site_tok)
                    search_providers.SCHOLARLY_ONLY.reset(so_tok)
                if not pl:
                    search_providers.record_portal("acquisition_search", "completed", len(cands))
            except Exception:  # noqa: BLE001
                if not pl:
                    search_providers.record_portal("acquisition_search", "failed", 0)
                count(acq_config.reason(acq, "search_failed"))
                rt.event("acquire", "search_failed")
                stop = "search_failed"
                st["portals"].extend(pl)
                break
        st["portals"].extend(pl)
        if ctx is not None:
            ctx.touch()
        # RETRIEVAL-1109 A3: news rounds also query the reporting lane (variant text). Streamed path only: the lane's
        # own connect/read timeouts and GDELT spacing are not bounded by the non-streamed 45 s deadline.
        if news and ctx is not None and variant_lane_on(acq):
            n_lane = _variant_lane(question, q, acq, st)
            st["n_lane"] = int(st.get("n_lane", 0) or 0) + n_lane  # answer-only pages count toward the round target
            r6["lane_pages"] = int(r6.get("lane_pages", 0)) + n_lane
            r6["lane_calls"] = int(r6.get("lane_calls", 0)) + 1
        fresh = [c for c in cands if isinstance(c, dict) and not (_ckeys(c) & tried)]
        for c in cands:
            if isinstance(c, dict):
                tried |= _ckeys(c)
        if news and fresh:  # F5: this round's general-web results through the gated web route (answer-only)
            n_before = int(st.get("n_web", 0) or 0)
            _web_read(question, acq, fresh, st, ctx, count, clock=clock, deadline=deadline, again=True,
                      max_fetches=max(0, min(remaining, tgt - eligible)))
            from vera import web_route
            # every general-web result stays on the gated web route (read, link or nothing), never the paper pipeline
            fresh = [c for c in fresh if not web_route.is_web_candidate(c)]
            r6["web_pages"] = int(r6.get("web_pages", 0)) + int(st.get("n_web", 0) or 0) - n_before
        if not fresh:
            continue
        ranked = aa.rank_candidates(question, fresh, acq, set(skip) | tried, fdcfg, count, drops={}, route=route,
                                    protected=None)
        reuse_info: dict = {}
        got_reused, left = _stored_for(ranked, acq, deps, count, question=question, fdcfg=fdcfg, icfg=icfg, clock=clock,
                                       deadline=deadline, info=reuse_info,
                                       **({"ctx": ctx, "max_checks": int(acq_config.live(acq)["max_reuse_checks_per_prompt"])}
                                          if ctx is not None else {}))
        if got_reused[0]:
            reused = (list(reused[0]) + list(got_reused[0]), list(reused[1]) + list(got_reused[1]))
        need = max(1, min(remaining, tgt - (len(plans) + len(extra) + _rn(acq, reused))))
        left = aa.reserve_reads(left, need, acq, route)
        if not left:
            continue
        if ctx is not None and _check_stop(ctx, st):
            stop = "disconnect" if st.get("stop") == stream_ctx.DISCONNECTED else str(st.get("stop"))
            break
        if ctx is None and deadline - clock() < per_doc:  # N5: one full fetch no longer fits: nothing starts
            count(acq_config.reason(acq, "deadline"))
            stop = "deadline"
            break
        extra += _evaluate_parallel(question, left, acq, deps, icfg, fdcfg, clock, deadline, need, count, ms,
                                    per={}, skips={}, **({"ctx": ctx, "st": st} if ctx is not None else {}))
    else:
        eligible = len(plans) + len(extra) + _rn(acq, reused) + _n_ao(st)
        stop = stop_pre or ("met" if eligible >= tgt else "variants_exhausted")
    r6["stop"] = stop
    for c in pending:  # still unevaluated: label by the real stop (WP7 b), not by the cut
        _relabel_left(acq, ctx, st, c, stop)
    return extra, reused


def _cut_deferred(acq: dict, st: dict) -> list:
    """WP7 (a): the round-0 candidates dropped only by the shortlist cut (outcome dropped, R-ACQ-DEFER_CAP), in the
    order the ranking stage recorded them (best first). Web-route candidates are terminal and never included."""
    why = acq_config.reason(acq, "defer_cap")
    wid = st.get("web_ids") or ()
    return [c for c, r in zip(st["cands"], st["recs"])
            if isinstance(c, dict) and id(c) not in wid and r.get("outcome") == "dropped" and r.get("reason") == why]


def _settle(st: dict, cid: int, outcome: str, reason) -> None:
    """WP7 (a): record the real outcome of a formerly cut candidate on its public record."""
    for c, r in zip(st["cands"], st["recs"]):
        if id(c) == cid:
            r["outcome"], r["reason"] = outcome, reason
            e = (st.get("entries") or {}).get(cid)
            if isinstance(e, dict):  # WORDING-LABELS-1110 (b): the reading list shows this entry; keep it in step
                e["outcome"], e["reason"] = outcome, reason
            return


def _relabel_left(acq: dict, ctx, st: dict, cand, stop) -> None:
    """WP7 (b): a cut candidate the extra rounds never reached gets the real stop as its reason (outcome
    not_evaluated, still continue-eligible), unless the loop simply met its target or ran out of variants."""
    why = {"disconnect": "stopped", stream_ctx.STALLED: "stopped", stream_ctx.PROVIDERS_EXHAUSTED: "stopped",
           stream_ctx.DISCONNECTED: "stopped", "no_progress": "no_progress", "ceiling31": "doc_budget",
           "deadline": "deadline"}.get(stop)
    if why:
        _settle(st, id(cand), "not_evaluated", acq_config.reason(acq, why))


# UNCALIBRATED planner prior: max candidates in the one protect-lookup batch (the discovered/search pool is bounded
# by discovered_max 6 + provider result caps). Calibrate from the live candidate-pool sizes in dev-mode candidates[];
# over the cap the lookup is refused (fail safe), never truncated. NOTE: deps.stored_rows is the existing reader and
# returns full source rows and spans (no identifier-only variant exists); a narrower SELECT would need a new reader.
PROTECT_LOOKUP_MAX = 60


def _protected_keys(cands, acq, deps):
    """O1 (IMPL-1028-T1O1): doc_key_set keys of the candidates ALREADY stored as acquired documents (reuse candidates),
    from ONE read-only batch through the same store reader _stored_for uses (deps.stored_rows, SELECT only). They are
    never evicted by the reserved-slot swap. Returns None when the lookup is unavailable or fails: the caller then
    skips the swap (fail safe)."""
    fn = getattr(deps, "stored_rows", None)
    if fn is None:
        return None
    try:
        from vera.corpus_admission import canonical_url
        from vera.dar_corpus import doc_key_set
        from vera.url_key import legacy_keys
        items = []
        for c in list(cands or ())[:PROTECT_LOOKUP_MAX]:
            if not isinstance(c, dict):
                continue
            s = {"doi": aa._doi(c), "url": c.get("url") or ""}
            isbn = aa.candidate_isbn(c)
            items.append((c, [canonical_url(s), *legacy_keys(s)],
                          doc_key_set(s["doi"], c.get("url"), (isbn or {}).get("isbn"))))
        if len(list(cands or ())) > PROTECT_LOOKUP_MAX:
            return None  # over the batch cap: protection cannot be established -> no O1 reordering (fail safe)
        srcs, _spans = fn(sorted({k for _c, ks, _d in items for k in ks}))
    except Exception as e:  # noqa: BLE001
        logger.warning("autoacq inline protect lookup failed (%s)", type(e).__name__)
        return None
    have = {r.get("canonical_url") for r in srcs or ()}
    out: set = set()
    for _c, ks, dk in items:
        if have & set(ks):
            out |= dk
    return out


def _stored_for(ranked, acq, deps, count, question=None, fdcfg=None, icfg=None, clock=None, deadline=None,
                info: dict | None = None, ctx=None, max_checks: int | None = None):
    """((sources, spans) of the ranked candidates already stored as acquired documents, at most max_docs_per_job, in
    rank order), and the ranked candidates left to evaluate. Read-only; a lookup failure reuses nothing (the
    evaluate stage's known check still prevents duplicates).

    DAR-TRACE-1022: a stored document is reused only when it passes THIS question's relevance gate (the same admission
    over a corpus of that document alone, >= answer_min_admitted_spans). Before, any ranked stored document was taken
    unchecked and filled max_docs_per_job, so fresh_budget fell to 0, no fresh candidate was evaluated, and the
    ephemeral corpus admitted nothing (status "unanswered", no answer-writer call). A failing stored document is
    counted R-ACQ-REUSE-IRRELEVANT (or R-ACQ-REUSE-UNUSABLE when the reader refused its rows) and never re-acquired
    (it is known). The checks stop at `deadline` (review S5): a row not checked in time is not reused.
    STREAM-1031 (ctx given; the deadline is infinite): at most max_checks relevance checks per prompt
    (live.max_reuse_checks_per_prompt) and none once ctx.should_stop(); an unchecked row is not reused and, being
    known, not re-acquired."""
    from vera.corpus_admission import canonical_url
    fn = getattr(deps, "stored_rows", None)
    if fn is None or not ranked:
        return ([], []), ranked
    from vera.url_key import legacy_keys
    csrcs = [{"doi": doi, "url": cand.get("url") or ""} for _sc, doi, cand, _i in ranked]
    cus = [canonical_url(s) for s in csrcs]
    # R3NORM-1028 read-time compatibility: also ask for rows stored under the legacy spelling of each key
    legacy = [dict.fromkeys(k for k in legacy_keys(s) if k not in cus) for s in csrcs]
    try:
        srcs, spans = fn(cus + [k for ks in legacy for k in ks])
    except Exception as e:  # noqa: BLE001
        logger.warning("autoacq inline reuse failed (%s)", type(e).__name__)
        return ([], []), ranked
    newest: dict = {}
    for r in sorted(srcs or (), key=lambda r: -int(r.get("source_id") or 0)):
        newest.setdefault(r.get("canonical_url"), r)  # one (the newest) stored row per canonical URL
    by_src: dict = {}
    for sp in spans or ():
        by_src.setdefault(sp.get("source_id"), []).append(sp)
    take, rest = [], []
    n_checks = 0
    for cu0, lks, r in zip(cus, legacy, ranked):
        cu = next((k for k in [cu0, *lks] if k in newest), cu0)  # current key first, then a legacy-keyed row
        if cu in newest and len(take) < int(acq_config.bound(acq, "max_docs_per_job")):
            row = newest.pop(cu)
            if question is not None:
                if ctx is not None and ((max_checks is not None and n_checks >= max_checks) or ctx.should_stop()):
                    why0 = acq_config.reason(acq, "doc_budget")  # STREAM-1031: bounded by count/stop, never a deadline
                    count(why0)
                    if info is not None:
                        info[id(r[2])] = ("reuse_rejected", why0)
                    continue
                if clock is not None and deadline is not None and clock() >= deadline:
                    count(acq_config.reason(acq, "deadline"))
                    if info is not None:
                        info[id(r[2])] = ("reuse_rejected", acq_config.reason(acq, "deadline"))
                    continue
                n_checks += 1
                why = _reuse_relevant(question, row, by_src.get(row.get("source_id"), []), acq, fdcfg, icfg)
                if ctx is not None:
                    ctx.touch()
                if why:
                    count(why)
                    if info is not None:
                        info[id(r[2])] = ("reuse_rejected", why)
                    continue
            take.append(row)
            if info is not None:
                info[id(r[2])] = ("reused", None)
            count("ACQ-REUSED")
        elif cu in newest:
            newest.pop(cu)  # stored but over the per-job cap: never re-acquired (evaluate would stop it as known)
            count(acq_config.reason(acq, "in_corpus"))
            if info is not None:
                info[id(r[2])] = ("reuse_rejected", acq_config.reason(acq, "in_corpus"))
        else:
            rest.append(r)
    ids = {r["source_id"] for r in take}
    return (take, [sp for sp in spans or () if sp.get("source_id") in ids]), rest


REUSE_IRRELEVANT = "R-ACQ-REUSE-IRRELEVANT"
REUSE_UNUSABLE = "R-ACQ-REUSE-UNUSABLE"  # review S4: the reader refused the stored rows (a safety refusal, not relevance)


def _reuse_relevant(question, row, spans, acq, fdcfg, icfg):
    """DAR-TRACE-1022: None when the stored row passes this question's relevance gate (auto_acquire.relevance_gate
    semantics: the answer path's admission over a corpus of this document alone), else the refusal code. Fails closed
    (REUSE_UNUSABLE) on any error."""
    import json

    from vera import fulldoc_admission, fulldoc_db
    try:
        src = json.loads(json.dumps(row, default=str))
        c = fulldoc_db.load_acquired_rows([src], [dict(sp) for sp in spans], fdcfg, acq, ingest_cfg=icfg)
        if c.status != "ok" or not c.acquired:
            return REUSE_UNUSABLE
        ev = fulldoc_admission.admit(question, c, fdcfg)
        return None if len(ev.admitted or ()) >= acq_config.answer_min_spans(acq) else REUSE_IRRELEVANT
    except Exception as e:  # noqa: BLE001
        logger.warning("autoacq inline reuse check failed (%s)", type(e).__name__)
        return REUSE_UNUSABLE


def _runner(question, acq, deps, icfg, fdcfg, clock, deadline, durs, per_doc, fl, queue_slots: bool = False):
    """The per-candidate evaluation closure shared by the legacy and the streamed pool. queue_slots (streamed path,
    PERF-MISS-1105-FIX2 F1): the candidate's limited fetches queue for a free per-portal slot within its own
    per-document budget (rate_limit.queue_scope) instead of skipping after max_wait_s, so parallel workers never
    self-skip on a single-connection portal."""
    def one(idx, score, doi, cand, isbn):
        if not queue_slots:
            return _one(idx, score, doi, cand, isbn)
        pd = per_doc
        if isinstance(cand.get("arxiv"), dict):
            from vera import arxiv_fulltext
            pd = max(per_doc, arxiv_fulltext.doc_budget(acq, clock))
        with rate_limit.queue_scope(time.monotonic() + float(pd)):
            return _one(idx, score, doi, cand, isbn)

    def _one(idx, score, doi, cand, isbn):
        d: dict = {}
        durs.append(d)
        job = aa.Job(question=question, trigger="inline", created=clock())
        try:
            pd = per_doc
            if isinstance(cand.get("arxiv"), dict):  # P0DTA-ARXFIX C: 2 x crawl-delay + fetch timeout, not the flat bound
                from vera import arxiv_fulltext
                pd = max(per_doc, arxiv_fulltext.doc_budget(acq, clock))
            dl = deadline
            if isinstance(cand.get("arxiv"), dict):  # CONTROL-GAPS-1107 r2: + one Crawl-delay for a cold tdmrep.json
                from vera import arxiv_fulltext
                dl = deadline + arxiv_fulltext.deadline_extension(acq)
            plan, _isbn = aa.evaluate_candidate(job, score, doi, cand, isbn, acq, deps, icfg, fdcfg, clock,
                                                min(dl, clock() + pd), d,
                                                min_spans=acq_config.answer_min_spans(acq))  # ONE-SPAN-1021
            return idx, plan, None
        except aa._Stop as e:
            aa.record_licence_hold(deps.store, e)  # LICENCE-ALLOWLIST-1104: unknown licence -> observations; never raises
            return idx, None, e.code
        except fl.Refused as e:
            aa.record_licence_hold(deps.store, e)  # LICENCE-ALLOWLIST-1104: unknown licence -> observations; never raises
            return idx, None, str(e.code)
        except Exception as e:  # noqa: BLE001
            return idx, None, aa.error_code(acq, e)
        finally:
            job.question = ""
    return one


def _slow_read(pending, started, n, slow, every: float = 10.0, now=time.monotonic) -> None:
    """NOISY-UI: at 10 s and every 10 s after, the oldest still-running source read is reported (k of n, seconds).
    Counts only. Cleared when nothing is pending."""
    from vera import progress
    if not pending:
        slow["mark"] = 0
        progress.clear_activity()
        return
    idx, t0 = min((started[f] for f in pending if f in started), key=lambda x: x[1], default=(None, None))
    k = len(pending)  # PROGRESS-COUNT-1106: reads run in parallel; the rank index of the oldest is not a position
    if idx is None:
        return
    age = now() - t0
    mark = int(age // every)
    if mark >= 1 and mark != slow["mark"]:
        slow["mark"] = mark
        progress.set_activity(progress.ACT_SLOW_READ, k=k, n=n, elapsed=int(age))
    elif mark < 1:
        slow["mark"] = 0


def _evaluate_stream(question, ranked, acq, deps, icfg, fdcfg, clock, budget, count, ms, per, skips, ctx, st, fl) -> list:
    """STREAM-1031 / OD-1031-NODEADLINE: the streamed-path evaluation. No inline deadline: waits are <= 1 s ticks (never
    wait(timeout=inf)); ctx.should_stop() is checked between candidates: once set, NO new candidate starts, in-flight
    ones finish (bounded by their own per-document timeouts) and their admitted documents are kept for the persist step.
    One pass over the shortlist: at most len(ranked) evaluations. Each finished candidate is a progress event
    (ctx.touch) and an acquire/update (counts only)."""
    if not ranked:
        return []
    per_doc = _b(acq, "per_doc_timeout_s", 25)
    workers = max(1, int(_b(acq, "fetch_workers", 4)))
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vera-inline")
    durs: list = []
    one = _runner(question, acq, deps, icfg, fdcfg, clock, INF, durs, per_doc, fl,  # each fetch keeps its own timeout
                  queue_slots=True)  # F1: queue for a portal slot, never a self-inflicted skip
    tgt = acq_rounds.target_of(acq)
    held_codes = {acq_config.reason(acq, k) for k in HELD_KEYS}
    got: dict = {}
    seen = {"held": 0}
    pending: set = set()
    started: dict = {}  # NOISY-UI: future -> (index, monotonic start)
    slow = {"mark": 0}
    nxt = 0
    try:
        while True:
            stopped = _check_stop(ctx, st)
            # start candidates only while running, below the document budget (admitted + in flight) and with a free worker
            while not stopped and nxt < len(ranked) and len(got) + len(pending) < budget and len(pending) < workers:
                fut = pool.submit(contextvars.copy_context().run, one, nxt, *ranked[nxt])
                started[fut] = (nxt, time.monotonic())
                pending.add(fut)
                nxt += 1
            if not pending:
                break
            fin, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            _slow_read(pending, started, len(ranked), slow)
            for f in fin:
                idx, plan, code = f.result()
                if per is not None:
                    per[idx] = plan if plan is not None else code
                if plan is not None:
                    got[idx] = plan
                else:
                    count(code)
                    if code in held_codes:
                        seen["held"] += 1
                ctx.touch()
                # PROGRESS-COUNT-1106: eligible plans may exceed the target (budget > target); the shown count is the
                # number toward the target, so it is clamped (the surplus is still persisted by the persist step)
                _elig = sum(1 for p in got.values() if aa.persist_eligible(p, acq))
                _emit("acquire", "update", checked=len(per) if per is not None else len(got) + 0, usable=len(got),
                      held=seen["held"],
                      stored=min(_elig, int(tgt)), target=int(tgt),
                      shortlisted=len(ranked))
            if len(got) >= budget:
                break  # the document budget is reached: stragglers are not waited for (legacy behaviour)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if skips is not None:
        why = acq_config.reason(acq, "doc_budget" if len(got) >= budget else "stopped")  # F4b: never reached, not cut by a cap
        skips.update({i: why for i in range(len(ranked)) if i not in (per or {})})
    for d in durs:
        for k in ("fetch", "verify"):
            ms[k] = max(ms.get(k, 0), int(d.get(k, 0)))
    return [got[i] for i in sorted(got)][:budget]


def _evaluate_parallel(question, ranked, acq, deps, icfg, fdcfg, clock, deadline, budget, count, ms,
                       per: dict | None = None, skips: dict | None = None, ctx=None, st: dict | None = None) -> list:
    """Every ranked candidate through the job's evaluate stage in a small pool; per-document timeout capped by the
    inline deadline; stops at `budget` admitted documents. Returns the admitted row plans (rank order). per (P0-5,
    flag on only): filled with rank index -> plan or reason code for each finished evaluation."""
    from vera import fulltext_loader as fl
    if ctx is not None:  # STREAM-1031: no inline deadline; own loop
        return _evaluate_stream(question, ranked, acq, deps, icfg, fdcfg, clock, budget, count, ms, per, skips, ctx,
                                st if st is not None else {}, fl)
    if not ranked or clock() >= deadline:
        if ranked:
            count(acq_config.reason(acq, "deadline"))
            if skips is not None:
                skips.update({i: acq_config.reason(acq, "deadline") for i in range(len(ranked))})
        return []
    per_doc = _b(acq, "per_doc_timeout_s", 25)
    pool = ThreadPoolExecutor(max_workers=max(1, int(_b(acq, "fetch_workers", 4))), thread_name_prefix="vera-inline")
    durs: list = []
    one = _runner(question, acq, deps, icfg, fdcfg, clock, deadline, durs, per_doc, fl)

    collect_until = deadline
    if any(isinstance(r[2], dict) and isinstance(r[2].get("arxiv"), dict) for r in ranked):
        # CONTROL-GAPS-1107 r2 SAFETY BOUND: + one Crawl-delay (<= 15 s) while tdmrep.json is cold, decided BEFORE
        # any fetch starts (a worker caching tdmrep.json must not shrink the window it needs)
        from vera import arxiv_fulltext
        collect_until = deadline + arxiv_fulltext.deadline_extension(acq)
    futs = {pool.submit(contextvars.copy_context().run, one, i, *r) for i, r in enumerate(ranked)}  # RATE-1022 C2
    got: dict = {}
    try:
        pending = futs
        while pending and len(got) < budget:
            left = collect_until - clock()
            if left <= 0:
                break
            fin, pending = wait(pending, timeout=left, return_when=FIRST_COMPLETED)
            for f in fin:
                idx, plan, code = f.result()
                if per is not None:
                    per[idx] = plan if plan is not None else code
                if plan is not None:
                    got[idx] = plan
                else:
                    count(code)
        if pending and skips is not None:  # P0DTA-DIAG: per-candidate skip reason at the moment of stopping
            why = acq_config.reason(acq, "doc_budget" if len(got) >= budget else "deadline")
            skips.update({i: why for i in range(len(ranked)) if i not in (per or {})})
        if pending and len(got) < budget:
            count(acq_config.reason(acq, "deadline"))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # stragglers stop at their own (deadline-capped) timeouts
    for d in durs:
        for k in ("fetch", "verify"):
            ms[k] = max(ms.get(k, 0), int(d.get(k, 0)))  # slowest document (parallel), not the sum
    return [got[i] for i in sorted(got)][:budget]


def ephemeral_corpus(plans: list, acq: dict, fdcfg: dict, icfg: dict, stored=None):
    """The in-memory corpus of exactly these admitted documents (plus `stored` = (sources, spans): already-stored
    acquired rows reused for this question), built by the acquired-path reader from the same row plans the writer
    stored (one synthetic run per document, so each verifies alone). None when unusable. R2: synthetic run AND source
    ids are negative, so they never collide with real (positive) ids of reused rows; the reader dedupes by canonical
    key and text hash."""
    import json

    from vera import acquired_pins, fulldoc_db
    srcs, spans = [], []
    # H1B-PROMOTE-1107: reused STORED rows must never carry the in-memory tag (stripped here; the manifest decides)
    st_src, st_spans = stored if stored else ([], [])
    srcs.extend({k: v for k, v in json.loads(json.dumps(r, default=str)).items() if k != acquired_pins.EPHEMERAL_KEY}
                for r in st_src)
    spans.extend(dict(sp) for sp in st_spans)
    for i, plan in enumerate(plans, 1):
        srcs.append({"run_id": -i, "source_id": -i, "canonical_url": plan["canonical_url"],
                     "content": plan["source"]["content"], "content_hash": plan["source"]["content_hash"],
                     "provenance": json.loads(json.dumps(plan["source"]["provenance"])),
                     acquired_pins.EPHEMERAL_KEY: True})  # H1B: in-memory fetch, never read back from the DB
        spans.extend({"source_id": -i, "start_index": s["start_index"], "end_index": s["end_index"], "text": s["text"]}
                     for s in plan["spans"])
    if not srcs:
        return None
    c = fulldoc_db.load_acquired_rows(srcs, spans, fdcfg, acq, ingest_cfg=icfg)
    if c.status == "ok" and c.acquired:
        c.seed_per_doc = True  # NONASK-1103 A6-3: admit() seeds each document's own candidates (admission.seed_per_doc_spans)
        return c
    return None


def notice(acq: dict | None, status: str) -> str | None:
    """Honest reading-list notice after an inline search ("none" / "deadline" / "search_failed" / "unanswered");
    None when no search was made (the caller keeps today's behaviour)."""
    se = (acq or {}).get("scope_expansion") or {}
    if status in CAPPED:  # DAR-TRACE-1022: the background queue shares these caps, so "a search has started" is false
        t = se.get("inline_notice_capped" if status == "rate" else "inline_notice_doc_cap")  # review S2
        return t if isinstance(t, str) and t else None
    if status not in SEARCHED and status != "unanswered":
        return None
    t = se.get({"none": "inline_notice_none", "deadline": "inline_notice_deadline",
                "search_failed": "inline_notice_search_failed", "ok": "inline_notice_unanswered",
                "unanswered": "inline_notice_unanswered"}[status])
    return t if isinstance(t, str) else ""


BACKGROUND_NOTE = research_outcome.BACKGROUND_SENTENCE  # NOANSWER-PAGE-1108 D6: one wording (said once on the page)


BACKGROUND_NOTE_PERSIST = (
    "VERA-DAR has tried every search available for this question. {n} source(s) it found but could not finish now may "
    "be checked later, but this is not guaranteed, and this answer will not update.")
# H5-CRAWL-1107 r3 (need-to-know): plain wording, no back-end detail; shown only when sources were really queued.
# Review F1: used when the keep-searching rounds stopped for any reason other than exhausting the variants.
BACKGROUND_NOTE_PERSIST_STOPPED = (
    "VERA-DAR stopped searching before every search option was tried ({reason}). {n} source(s) it found but could "
    "not finish now may be checked later, but this is not guaranteed, and this answer will not update.")
_STOP_REASON_TEXT = {"deadline": "the time limit was reached", "ceiling31": "the per-question reading limit was reached",
                     "rate": "the current search limit was reached", "disconnect": "the connection ended",
                     "search_failed": "a search could not be completed", stream_ctx.STALLED: "no further progress was made",
                     stream_ctx.PROVIDERS_EXHAUSTED: "the answering service reached its limit"}
_NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def stop_clause(aq: dict | None) -> str:
    """MIDTERMS-WP8-1106 (owner D1/D2): one need-to-know sentence on why the search rounds stopped (no portal,
    provider, host or internal cause), with a leading space; "" for no stop or any other stop. Shared by the
    shortfall line (vera.ui_evidence) and the non-answer notice (vera.ask_service._inline_notice)."""
    aq = aq if isinstance(aq, dict) else {}
    stop, persist = aq.get("stop"), aq.get("persist") is True
    if stop == "no_progress" and not persist:
        n = _NUM_WORDS.get(NO_PROGRESS_ROUNDS, str(NO_PROGRESS_ROUNDS))
        return (f" VERA-DAR stopped after {n} further searches found nothing new; add \"keep searching\" to your "
                "question to try every remaining search.")
    if persist and stop == "variants_exhausted":
        return " Every search option available for this question was tried."
    if stop == "ceiling31":
        return " The per-question reading limit was reached."
    if stop == "rate":
        return " The current search limit was reached; try again later."
    return ""


def background_note(acq: dict | None, o) -> str:
    """CONT-1023: one honest sentence when this request queued candidates for background acquisition (best-effort: no
    claim they will run or succeed); "" otherwise. Override: scope_expansion.inline_notice_background.
    MIDTERMS-WP8-1106 (owner D1): under persist the hand-off count itself is stated (Outcome.queued, the real
    best-effort hand-off; nothing is claimed as running: the jobs run no search and their status is not observable).
    Override: scope_expansion.inline_notice_background_persist ({n} = the count)."""
    n = int(getattr(o, "queued", 0) or 0)
    if not n:
        return ""
    if ((getattr(o, "acquisition", None) or {}).get("persist")
            and str((getattr(o, "background", None) or {}).get("status") or "") == "ACQ-QUEUED"):
        stop = (getattr(o, "acquisition", None) or {}).get("stop")
        se = (acq or {}).get("scope_expansion") or {}
        if stop == "variants_exhausted":  # review F1: the exhaustion claim only when every variant was searched
            key, dflt = "inline_notice_background_persist", BACKGROUND_NOTE_PERSIST
        else:
            key, dflt = "inline_notice_background_persist_stopped", BACKGROUND_NOTE_PERSIST_STOPPED
        tp = se.get(key)
        tp = tp if isinstance(tp, str) and tp else dflt
        reason = _STOP_REASON_TEXT.get(stop, "it reached a stopping point")
        try:
            return tp.format(n=n, reason=reason)
        except (KeyError, IndexError, ValueError):
            return dflt.format(n=n, reason=reason)
    t = ((acq or {}).get("scope_expansion") or {}).get("inline_notice_background")
    return t if isinstance(t, str) and t else BACKGROUND_NOTE


def answer_note(acq: dict | None, o: Outcome) -> str:
    se = (acq or {}).get("scope_expansion") or {}
    # ONE-SPAN-1021: never say "added" for a document used for this answer only. answer_only: EVERY answering document
    # is a new below_persist_min one; mixed (neutral wording): answer-only documents beside any other kind (stored,
    # already held, unsaved after a write failure, or reused).
    only = o.answer_only
    if only:
        key = ("inline_answer_note_answer_only" if only == len(o.documents) and not o.reused
               else "inline_answer_note_mixed")
    else:
        key = ("inline_answer_note_unsaved" if o.persist_failed else
               "inline_answer_note_stored" if o.reused and not o.admitted else "inline_answer_note")  # R2: reuse only
    t = se.get(key)
    return t if isinstance(t, str) else ""


def refresh_after_response() -> None:
    """Background (after the response): pick up the stored documents in the served acquired corpus. Never raises."""
    try:
        from vera import fulldoc_corpus
        fulldoc_corpus.invalidate_acquired()
        fulldoc_corpus.refresh_acquired(sync=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("autoacq inline refresh failed (%s)", type(e).__name__)
