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

from vera import acq_config, rate_limit
from vera import auto_acquire as aa
from vera import search_providers

logger = logging.getLogger("vera.autoacq")

# statuses: a search was made -> "ok" (corpus set), "none", "deadline", "search_failed"
#           no search was made -> "off", "busy", "rate", "daily_cap", "error"
SEARCHED = frozenset({"ok", "none", "deadline", "search_failed", "unanswered"})
CAPPED = frozenset({"rate", "daily_cap"})  # DAR-TRACE-1022: the shared hourly/daily caps refused the acquisition
# DAR-TRACE-1022 (review S1): provider/time failures are cached only failure_ttl_s; deterministic none/unanswered keep
# result_ttl_s (re-running them early would only burn the shared jobs_per_hour cap).
TRANSIENT = frozenset({"search_failed", "deadline"})
BELOW_PERSIST_MIN = "below_persist_min"  # ONE-SPAN-1021 per-document telemetry reason


@dataclass
class Outcome:
    status: str
    corpus: object = None
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
                **({"background": dict(self.background)} if self.background is not None else {})}

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


def acquire(question: str, acq: dict, *, env=os.environ, clock=time.monotonic, deps_factory=None,
            served=None, discovered=None, prefer_recent: bool = False) -> Outcome:
    """Never raises. See the module docstring for statuses. served (R2): doc keys the caller's stored-corpus attempt
    already tried (skipped here); None = the currently served acquired corpus (deps.corpus_keys). discovered (P0-5):
    the request's discovery results (list of source dicts), used only when inline.use_discovered is true."""
    try:
        if not inline_enabled(acq, env) or not str(question or "").strip():
            return Outcome("off")
        from vera import scope_router  # SCOPE-1022: the shared intent rule also guards inline acquisition
        if not scope_router.intent_clean(str(question)):
            return Outcome("off")
        disc = None
        if discovered is not None:
            from vera import acq_discovered
            if acq_discovered.use_discovered(acq):
                disc = acq_discovered.discovered_candidates(discovered, acq)
        key = _cache_key(question, served, disc) if disc is not None else _cache_key(question, served)
        now = clock()
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
                    return Outcome("busy")
                slot[2] += 1
            elif _I.active >= int(_b(acq, "max_concurrent", 2)):
                return Outcome("busy")
            else:
                _I.active += 1
                slot = _I.inflight[key] = [threading.Event(), None, 0]
    except Exception as e:  # noqa: BLE001
        logger.warning("autoacq inline failed (%s)", type(e).__name__)
        return Outcome("error")
    if not owner:  # an identical question is in flight: wait (bounded) for its outcome
        slot[0].wait(timeout=_b(acq, "waiter_wait_s", 45))
        with _I.lock:
            slot[2] -= 1
        return slot[1] if slot[1] is not None else Outcome("busy")
    out = Outcome("error")  # an unexpected failure is never reported as "searched" (no search notice is shown)
    try:
        out = _run(question, acq, env, clock, deps_factory, served,
                   **({"disc": disc} if disc is not None else {}),  # P0-5: flag off = the unchanged call
                   **({"prefer_recent": True} if prefer_recent else {}))
    except Exception as e:  # noqa: BLE001
        logger.warning("autoacq inline failed (%s)", type(e).__name__)
    finally:
        with _I.lock:
            _I.active -= 1
            _I.inflight.pop(key, None)
            if out.status in SEARCHED:  # VER L1: never cache a cap/busy refusal
                _I.results[key] = (clock(), out)
                while len(_I.results) > int(_b(acq, "result_cache_max", 8)):
                    _I.results.popitem(last=False)
        slot[1] = out
        slot[0].set()
    return out


def _run(question, acq, env, clock, deps_factory, served=None, disc=None, prefer_recent=False) -> Outcome:
    """RATE-1022 C1: every rate-limit acquire() in this inline run sees its deadline. rate_limit uses time.monotonic,
    so the scope is now + deadline_s on that clock (the injected `clock` may be a test clock)."""
    with rate_limit.deadline_scope(time.monotonic() + float(_b(acq, "deadline_s", 45))):
        return _run_body(question, acq, env, clock, deps_factory, served,
                         **({"disc": disc} if disc is not None else {}),
                         **({"prefer_recent": True} if prefer_recent else {}))


QUEUED = "queued_background"  # CONT-1023: candidate outcome = handed to the background acquisition queue


def _transient(acq, code) -> bool:
    """CONT-1023: a per-candidate failure that may succeed later (timeout, 5xx/non-200, quota, connection error)."""
    c = str(code or "")
    if c in {acq_config.reason(acq, k) for k in ("unreachable", "rate_limited", "deadline")}:
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
            "arxiv_id": c.get("arxiv_id") or c.get("arxiv") or None, "url": c.get("url") or None,
            "published": c.get("published") or c.get("publication_date") or c.get("year") or None,
            "outcome": None, "reason": None, "found_by": list(c.get("found_by") or [portal or c.get("provider")])}


def _run_body(question, acq, env, clock, deps_factory, served=None, disc=None, prefer_recent=False) -> Outcome:
    from vera import fulltext_loader as fl
    t0 = clock()
    deadline = t0 + _b(acq, "deadline_s", 45)
    ms: dict = {}
    codes: dict = {}

    def count(code):
        codes[code] = codes.get(code, 0) + 1

    entries: dict = {}  # P0-5: id(discovered candidate) -> its public entry (disc only)
    st: dict = {"portals": None, "recs": [], "cands": []}  # FRESH-1023 C3: portal attempts + per-candidate records

    def done(o: Outcome) -> Outcome:
        if st["portals"] is not None:
            o.portals = list(st["portals"])
            late = clock() >= deadline
            for r in st["recs"]:
                if r["outcome"] is None:
                    r["outcome"] = "not_evaluated"
                    r["reason"] = acq_config.reason(acq, "deadline") if late else acq_config.reason(acq, "defer_cap")
            o.background = _continue(question, acq, env, clock, st["recs"], st["cands"], deps_factory)  # CONT-1023
            o.candidates = [dict(r) for r in st["recs"]]
        if disc is not None:
            o.discovered = [entries.get(id(c)) or {**c["discovery"], "outcome": "not_reached"} for c in disc[0]] + \
                [dict(h) for h in disc[1]]
            if disc[1]:
                for h in disc[1]:
                    count(h["reason"])
        ms["total"] = int((clock() - t0) * 1000)
        o.codes, o.ms = dict(sorted(codes.items())), ms
        aa._log("inline", status=o.status, admitted=o.admitted, stored=o.stored, persist_failed=o.persist_failed,
                n_answer_eligible=o.n_answer_eligible, n_persisted=o.n_persisted,
                n_below_persist_min=o.answer_only, reused=o.reused,
                **{f"n_{k}": v for k, v in o.codes.items()}, **{f"ms_{k}": v for k, v in ms.items()})
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
        budget = reserved = aa.docs_budget(acq, deps, clock=clock, reserve=True)  # VER L5: reserved under the lock
        # R2 defect 1: a spent docs_per_day no longer stops before the search: reusing already-stored documents
        # acquires nothing, so it is not capped. With budget 0 nothing new is resolved, fetched or stored.
        s = acq.get("search") or {}
        t = clock()
        search_failed = False
        with search_providers.portal_scope() as portals_l:
            st["portals"] = portals_l
            try:
                cands = list(deps.search(question, int(s.get("results_per_provider", 10)),
                                         max(1.0, min(float(s.get("timeout_s", 20)), deadline - clock()))) or [])
                if not portals_l:  # FRESH-1023: a search callable that records no portals counts as ONE portal
                    search_providers.record_portal("acquisition_search", "completed", len(cands))
            except Exception:  # noqa: BLE001
                if not portals_l:
                    search_providers.record_portal("acquisition_search", "failed", 0)
                count(acq_config.reason(acq, "search_failed"))
                if not (disc is not None and disc[0]):
                    return done(Outcome("search_failed"))
                cands, search_failed = [], True  # P0-5: the discovered candidates still go through acquisition
        ms["search"] = int((clock() - t) * 1000)
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
                st["recs"].append(rec_by[id(c)])
                st["cands"].append(c)
        drops: dict = {}
        ranked = aa.rank_candidates(question, cands, acq, skip, fdcfg, count, drops=drops)
        for cid, why in drops.items():
            if cid in rec_by:
                rec_by[cid]["outcome"], rec_by[cid]["reason"] = "dropped", why
        if disc is not None:
            entries.update(acq_discovered.gate_a(disc[0], cands, ranked, skip, acq))
        # DAR-INLINE-1021-R2: ranked candidates ALREADY stored as acquired documents are answered from their stored rows
        # (read-only) instead of being skipped as R-ACQ-KNOWN while lower-ranked documents are acquired in their place
        # (the run-2 failure: a new process, the served acquired corpus still loading in the background).
        t = clock()
        reuse_info: dict = {}
        reused, ranked_left = _stored_for(ranked, acq, deps, count, question=question, fdcfg=fdcfg, icfg=icfg,
                                          clock=clock, deadline=deadline, info=reuse_info)
        for cid, (oc, why) in reuse_info.items():
            if cid in rec_by:
                rec_by[cid]["outcome"], rec_by[cid]["reason"] = oc, why
        if disc is not None:
            left = {id(r[2]) for r in ranked_left}
            for r in ranked:
                e = entries.get(id(r[2]))
                if e is not None and id(r[2]) not in left:
                    e["outcome"] = "already_stored"
        ranked = ranked_left
        ms["reuse"] = int((clock() - t) * 1000)
        job_max = int(acq_config.bound(acq, "max_docs_per_job"))
        fresh_budget = max(0, min(budget, job_max - len(reused[0])))
        t = clock()
        per: dict = {}  # FRESH-1023 C3: always collected (per-candidate records); entries stay disc-only
        plans = _evaluate_parallel(question, ranked, acq, deps, icfg, fdcfg, clock, deadline, fresh_budget, count,
                                   ms, per=per) if fresh_budget > 0 else []
        by_plan: dict = {}
        kept_ids = {id(p) for p in plans}
        rec_plan: dict = {}
        for i, r in enumerate(ranked):
            rc = rec_by.get(id(r[2]))
            got = per.get(i)
            if rc is None or got is None:
                continue  # unfinished: done() marks it not_evaluated
            if isinstance(got, str):
                rc["outcome"], rc["reason"] = "rejected", got
            elif id(got) in kept_ids:
                rc["outcome"] = "admitted"
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
        if not plans and not reused[0]:
            if budget <= 0:
                count(acq_config.reason(acq, "daily_cap"))
                return done(Outcome("daily_cap"))
            late = clock() >= deadline or acq_config.reason(acq, "deadline") in codes
            return done(Outcome("deadline" if late else "none"))
        out = Outcome("ok", admitted=len(plans))
        out.reused = len(reused[0])
        failed = 0
        t = clock()
        for plan in plans:  # one INSERT-only transaction per document (WoStore); DOI dedupe = canonical_state
            e = by_plan.get(id(plan))
            rc = rec_plan.get(id(plan))
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


def _stored_for(ranked, acq, deps, count, question=None, fdcfg=None, icfg=None, clock=None, deadline=None,
                info: dict | None = None):
    """((sources, spans) of the ranked candidates already stored as acquired documents, at most max_docs_per_job, in
    rank order), and the ranked candidates left to evaluate. Read-only; a lookup failure reuses nothing (the
    evaluate stage's known check still prevents duplicates).

    DAR-TRACE-1022: a stored document is reused only when it passes THIS question's relevance gate (the same admission
    over a corpus of that document alone, >= answer_min_admitted_spans). Before, any ranked stored document was taken
    unchecked and filled max_docs_per_job, so fresh_budget fell to 0, no fresh candidate was evaluated, and the
    ephemeral corpus admitted nothing (status "unanswered", no answer-writer call). A failing stored document is
    counted R-ACQ-REUSE-IRRELEVANT (or R-ACQ-REUSE-UNUSABLE when the reader refused its rows) and never re-acquired
    (it is known). The checks stop at `deadline` (review S5): a row not checked in time is not reused."""
    from vera.corpus_admission import canonical_url
    fn = getattr(deps, "stored_rows", None)
    if fn is None or not ranked:
        return ([], []), ranked
    cus = [canonical_url({"doi": doi, "url": cand.get("url") or ""}) for _sc, doi, cand, _i in ranked]
    try:
        srcs, spans = fn(cus)
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
    for cu, r in zip(cus, ranked):
        if cu in newest and len(take) < int(acq_config.bound(acq, "max_docs_per_job")):
            row = newest.pop(cu)
            if question is not None:
                if clock is not None and deadline is not None and clock() >= deadline:
                    count(acq_config.reason(acq, "deadline"))
                    if info is not None:
                        info[id(r[2])] = ("reuse_rejected", acq_config.reason(acq, "deadline"))
                    continue
                why = _reuse_relevant(question, row, by_src.get(row.get("source_id"), []), acq, fdcfg, icfg)
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
        c = fulldoc_db.load_acquired_rows([src], [dict(sp) for sp in spans], fdcfg, acq, ingest_cfg=icfg, max_docs=1)
        if c.status != "ok" or not c.acquired:
            return REUSE_UNUSABLE
        ev = fulldoc_admission.admit(question, c, fdcfg)
        return None if len(ev.admitted or ()) >= acq_config.answer_min_spans(acq) else REUSE_IRRELEVANT
    except Exception as e:  # noqa: BLE001
        logger.warning("autoacq inline reuse check failed (%s)", type(e).__name__)
        return REUSE_UNUSABLE


def _evaluate_parallel(question, ranked, acq, deps, icfg, fdcfg, clock, deadline, budget, count, ms,
                       per: dict | None = None) -> list:
    """Every ranked candidate through the job's evaluate stage in a small pool; per-document timeout capped by the
    inline deadline; stops at `budget` admitted documents. Returns the admitted row plans (rank order). per (P0-5,
    flag on only): filled with rank index -> plan or reason code for each finished evaluation."""
    from vera import fulltext_loader as fl
    if not ranked or clock() >= deadline:
        if ranked:
            count(acq_config.reason(acq, "deadline"))
        return []
    per_doc = _b(acq, "per_doc_timeout_s", 25)
    pool = ThreadPoolExecutor(max_workers=max(1, int(_b(acq, "fetch_workers", 4))), thread_name_prefix="vera-inline")
    durs: list = []

    def one(idx, score, doi, cand, isbn):
        d: dict = {}
        durs.append(d)
        job = aa.Job(question=question, trigger="inline", created=clock())
        try:
            plan, _isbn = aa.evaluate_candidate(job, score, doi, cand, isbn, acq, deps, icfg, fdcfg, clock,
                                                min(deadline, clock() + per_doc), d,
                                                min_spans=acq_config.answer_min_spans(acq))  # ONE-SPAN-1021
            return idx, plan, None
        except aa._Stop as e:
            return idx, None, e.code
        except fl.Refused as e:
            return idx, None, str(e.code)
        except Exception as e:  # noqa: BLE001
            return idx, None, aa.error_code(acq, e)
        finally:
            job.question = ""

    futs = {pool.submit(contextvars.copy_context().run, one, i, *r) for i, r in enumerate(ranked)}  # RATE-1022 C2
    got: dict = {}
    try:
        pending = futs
        while pending and len(got) < budget:
            left = deadline - clock()
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

    from vera import fulldoc_db
    srcs, spans = [], []
    st_src, st_spans = stored if stored else ([], [])
    srcs.extend(json.loads(json.dumps(r, default=str)) for r in st_src)
    spans.extend(dict(sp) for sp in st_spans)
    for i, plan in enumerate(plans, 1):
        srcs.append({"run_id": -i, "source_id": -i, "canonical_url": plan["canonical_url"],
                     "content": plan["source"]["content"], "content_hash": plan["source"]["content_hash"],
                     "provenance": json.loads(json.dumps(plan["source"]["provenance"]))})
        spans.extend({"source_id": -i, "start_index": s["start_index"], "end_index": s["end_index"], "text": s["text"]}
                     for s in plan["spans"])
    if not srcs:
        return None
    c = fulldoc_db.load_acquired_rows(srcs, spans, fdcfg, acq, ingest_cfg=icfg, max_docs=len(srcs))
    return c if c.status == "ok" and c.acquired else None


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


BACKGROUND_NOTE = ("Further sources VERA-DAR found for this question may be acquired in the background on a best-effort "
                   "basis; any that are added must first pass its licence, content and relevance checks.")


def background_note(acq: dict | None, o) -> str:
    """CONT-1023: one honest sentence when this request queued candidates for background acquisition (best-effort: no
    claim they will run or succeed); "" otherwise. Override: scope_expansion.inline_notice_background."""
    if not getattr(o, "queued", 0):
        return ""
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
