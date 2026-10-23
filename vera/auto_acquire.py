"""WED-ACQ2 / P1.7: bounded background acquisition of open-access FULL TEXT from /ask demand.

Plan: records/wed-acq2-auto-corpus-expansion-plan.md (s4, s11). One daemon worker thread, a bounded in-memory queue.

  enqueue(question, trigger)            O(1), never blocks /ask, never raises
  worker -> run_job(job, cfg, deps):    search (existing providers, shared /ask instances) -> evaluate (DOI, not in the
      collection by DOI/URL/ISBN, deterministic relevance pre-rank, no model call) -> resolve OA via Europe PMC (OA flag,
      declared licence, refused publication types) -> fetch JATS (allowlisted hosts, https, SSRF guard on every hop, XML
      only, byte and time caps) -> convert (vera.jats_text) -> EVERY vera.fulltext_loader filter + licence rule (label
      overlay only) -> injection scan -> relevance gate (the answer path's own admission over the single document, which
      also proves the reader accepts the exact rows) -> write with the write-only account (one transaction per
      document) -> refresh the acquired corpus.

Privacy (AC-8): the visitor's question lives only in the in-memory Job and is dropped when the job ends or expires. It
is never written to the DB (the questions/runs rows carry the fixed config text), never logged (log lines carry reason
codes, counts and durations only), never put in a thread name or an exception message, and query terms are not stored.
Every exception inside the worker is caught and logged by class name only, so threading.excepthook never sees one.

Network and DB are injected (Deps) so the offline tests never open a socket or a connection.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlsplit

from vera import acq_config, rate_limit

logger = logging.getLogger("vera.autoacq")
CODE_VERSION = "wed-acq2-p17-1"
THREAD_NAME = "vera-autoacq"


# ------------------------------------------------------------------------------------------------ jobs and state
@dataclass
class Job:
    question: str
    trigger: str
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created: float = 0.0


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.queue: queue.Queue | None = None
        self.worker: threading.Thread | None = None
        self.jobs_hour: deque = deque()
        self.jobs_day: deque = deque()
        self.docs_day: deque = deque()
        self.docs_reserved = 0  # DAR-INLINE-1021-R2 (VER L5): documents reserved by in-progress inline jobs
        self.recent: dict = {}  # salted HMAC of the normalised question -> monotonic time (memory only)
        self.salt = os.urandom(32)


_S = _State()


def reset_for_tests() -> None:
    global _S
    _S = _State()


def _qkey(question: str) -> str:
    norm = re.sub(r"\s+", " ", str(question or "").strip().lower())
    return hmac.new(_S.salt, norm.encode("utf-8"), hashlib.sha256).hexdigest()


def _prune(dq: deque, now: float, window: float) -> None:
    while dq and now - dq[0] >= window:
        dq.popleft()


def _log(event: str, **counts) -> None:
    """Codes, counts and durations only. Never the question, query terms, titles, URLs or identifiers."""
    safe = {k: v for k, v in counts.items() if isinstance(v, (int, float, bool)) or
            (isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_.:\-]{0,64}", v))}
    logger.info("autoacq %s %s", event, json.dumps(safe, sort_keys=True))


def enqueue(question: str, trigger: str, *, cfg: dict | None = None, env=os.environ, clock=time.monotonic,
            start_worker: bool = True, deps_factory=None) -> str:
    """Returns a reason code (ACQ-QUEUED or an R-ACQ-* refusal). Never raises, never blocks."""
    try:
        cfg = cfg if cfg is not None else acq_config.load_config()
        if not acq_config.enabled(cfg, env):
            return acq_config.reason(cfg or {}, "disabled")
        if trigger not in ("decline", "insufficient") or not str(question or "").strip():
            return acq_config.reason(cfg, "disabled")
        from vera import scope_router  # SCOPE-1022: the shared intent rule also guards the queue (belt and braces)
        if not scope_router.intent_clean(str(question)):
            return acq_config.reason(cfg, "intent_denied")
        now = clock()
        with _S.lock:
            win = float(acq_config.bound(cfg, "dedupe_window_s"))
            for k in [k for k, t in _S.recent.items() if now - t >= win]:
                del _S.recent[k]
            key = _qkey(question)
            if key in _S.recent:
                return acq_config.reason(cfg, "duplicate_question")
            _prune(_S.jobs_hour, now, 3600.0)
            _prune(_S.jobs_day, now, 86400.0)
            if len(_S.jobs_hour) >= int(acq_config.bound(cfg, "jobs_per_hour")) or \
                    len(_S.jobs_day) >= int(acq_config.bound(cfg, "jobs_per_day")):
                return acq_config.reason(cfg, "rate")
            if _S.queue is None:
                _S.queue = queue.Queue(maxsize=int(acq_config.bound(cfg, "queue_max")))
            try:
                _S.queue.put_nowait(Job(question=str(question), trigger=trigger, created=now))
            except queue.Full:
                return acq_config.reason(cfg, "queue_full")
            _S.recent[key] = now
            _S.jobs_hour.append(now)
            _S.jobs_day.append(now)
            if start_worker and (_S.worker is None or not _S.worker.is_alive()):
                _S.worker = threading.Thread(target=_worker_loop, args=(cfg, env, clock, deps_factory),
                                             name=THREAD_NAME, daemon=True)
                _S.worker.start()
        return "ACQ-QUEUED"
    except Exception as e:  # noqa: BLE001
        logger.warning("autoacq enqueue failed (%s)", type(e).__name__)
        return "R-ACQ-ERROR"


def _worker_loop(cfg, env, clock, deps_factory) -> None:
    q = _S.queue
    while True:
        try:
            job = q.get(timeout=300)
        except queue.Empty:
            with _S.lock:  # idle: end, unless a job arrived meanwhile (enqueue checks is_alive under the same lock)
                if q.empty():
                    _S.worker = None
                    return
            continue
        deps = None
        try:
            deps = (deps_factory or default_deps)(cfg, env)
            run_job(job, cfg, deps, clock=clock, env=env)
        except Exception as e:  # noqa: BLE001  (class only: never the message, which may quote a URL)
            logger.warning("autoacq job failed (%s)", type(e).__name__)
        finally:
            job.question = ""  # drop the visitor's text
            try:
                if deps is not None and deps.close is not None:
                    deps.close()
            except Exception as e:  # noqa: BLE001
                logger.warning("autoacq close failed (%s)", type(e).__name__)
            q.task_done()


def drain_for_tests(cfg, deps, clock=time.monotonic, env=None) -> list:
    """Run every queued job synchronously in this thread (tests only)."""
    out = []
    q = _S.queue
    while q is not None and not q.empty():
        job = q.get_nowait()
        try:
            out.append(run_job(job, cfg, deps, clock=clock, env=env))
        finally:
            job.question = ""
            q.task_done()
    return out


# ------------------------------------------------------------------------------------------------ dependencies
@dataclass
class Deps:
    search: object            # (question, n, timeout_s) -> list[candidate dict]
    http_get: object          # requests.get-compatible (url, headers=, timeout=, allow_redirects=, stream=)
    lookup: object            # RO: known(doi, url, isbn) -> bool ; docs_last_24h() -> int
    store: object             # WO: write(plan) -> dict ids
    corpus_keys: object       # () -> set of doc_key_set keys of the documents currently served
    url_check: object = None  # SSRF guard (selective_fetch.is_public_url by default)
    refresh: object = None    # () -> None, after a commit
    close: object = None      # () -> None, after the job (disposes the engines)
    stored_rows: object = None  # DAR-INLINE-1021-R2 (inline only): (canonical_urls) -> (sources, spans) already stored


def make_search(cfg: dict, chain_fn=None, search_fn=None):
    """The job's search: the REAL vera.search_and_fetch.search_question over the shared /ask provider instances named in
    config (shared rate gates and budgets), one attempt per provider, then this job's route record is removed.
    chain_fn / search_fn are injectable so a test drives the real search through a fake provider transport."""
    names = set((cfg.get("search") or {}).get("providers") or ())

    def search(question, n, timeout_s):
        from vera import search_and_fetch
        chain = (chain_fn or (lambda: search_and_fetch.cached_chain()[0]))()
        provs = [p for p in chain if getattr(p, "name", None) in names]
        if not provs:
            return []
        # ACQ-SEARCH-1021: acquisition-only OpenAlex profile (config search.openalex_profile); /ask never sets it.
        from vera import search_providers
        profile = (cfg.get("search") or {}).get("openalex_profile") or None
        fn = search_fn or search_and_fetch.search_question
        tok = search_providers.OPENALEX_PROFILE.set(profile)
        try:
            out = search_and_clear(fn, provs, question, n, timeout_s)
            if out or profile is None:
                return out
            # DAR-INLINE-1021-R2: the AND profile found nothing -> retry ONCE with the plain OR query (profile None).
            search_providers.OPENALEX_PROFILE.reset(tok)
            tok = search_providers.OPENALEX_PROFILE.set(None)
            return search_and_clear(fn, provs, question, n, timeout_s)
        finally:
            search_providers.OPENALEX_PROFILE.reset(tok)
    return search


def search_and_clear(search_fn, provs, question, n, timeout_s) -> list:
    """Run the shared /ask provider instances (shared rate gates and budgets), then remove ONLY this job's route record
    (`last_query`) so no question-derived terms outlive the job (AC-8). The record each provider held BEFORE the search
    is captured; afterwards a provider whose record changed is restored to it only when the new record is this job's
    own (every query word comes from this job's question). A concurrent /ask's record is never touched."""
    before = {id(p): getattr(p, "last_query", None) for p in provs}
    qwords = set(re.findall(r"[a-z0-9]+", str(question or "").lower()))
    try:
        return list(search_fn(question, n, providers=provs, min_candidates=1, max_retries=1,  # ONE attempt (0 = none)
                              deadline_s=float(timeout_s), timeout_s=float(timeout_s), mode="fan_out"))
    finally:
        for p in provs:
            cur = getattr(p, "last_query", None)
            if cur is None or cur is before[id(p)] or not isinstance(cur, dict):
                continue
            words = set(re.findall(r"[a-z0-9]+", " ".join(str(v) for k, v in cur.items()
                                                           if k in ("query", "retry_query")).lower())) - {"or", "and"}
            if words and words <= qwords:
                try:
                    p.last_query = before[id(p)]
                except Exception:  # noqa: BLE001
                    pass


def served_doc_keys(fulldoc_corpus, warm_state=None) -> set:
    """Document keys already served (base + acquired), WITHOUT waiting on the base corpus load. LIVE-DIAG-1021: while
    the startup warm-up (or any first load) is in flight, get_corpus() makes this job a single-flight follower that
    waits up to db_bounds.single_flight_wait_s (60 s) inside the job's 120 s deadline. The base keys are read only
    once the warm-up is DONE (a cache hit); otherwise they are skipped, which is safe because RoLookup.known() checks
    every stored full-document row (base and acquired) before any fetch or write."""
    if warm_state is None:
        from vera import fulldoc_warm
        warm_state = fulldoc_warm.state
    out = set()
    try:
        base_ready = warm_state() == "done"
    except Exception:  # noqa: BLE001
        base_ready = False
    for get in ((fulldoc_corpus.get_corpus,) if base_ready else ()) + (fulldoc_corpus.get_acquired_corpus,):
        dar = getattr(get(), "_dar", None)
        if dar is not None and hasattr(dar, "doc_keys"):
            out |= set(dar.doc_keys())
    return out


def default_deps(cfg: dict, env) -> Deps:  # live wiring; exercised offline by tests/test_acq_wiring.py
    import requests

    from sqlalchemy import create_engine

    from vera import fulldoc_corpus, search_and_fetch, selective_fetch
    ms = int(cfg.get("db_statement_timeout_ms") or 15000)
    kw = {"connect_args": {"connect_timeout": 5, "options": f"-c statement_timeout={ms}", "keepalives": 1, "keepalives_idle": 10, "keepalives_interval": 5, "keepalives_count": 3},
          "hide_parameters": True,
          "pool_pre_ping": True}
    wo = create_engine(env[cfg["db_url_env_wo"]], **kw)
    ro = create_engine(env[cfg["db_url_env_ro"]], **kw)
    from vera import search_providers
    search = search_providers.acquisition_search(make_search(cfg), cfg)  # YIELD-1021: + DDG first, Europe PMC

    def keys():
        return served_doc_keys(fulldoc_corpus)

    def refresh():
        fulldoc_corpus.invalidate_acquired()
        fulldoc_corpus.refresh_acquired(sync=True)

    def close():
        wo.dispose()
        ro.dispose()

    # RATE-1022 A6: www.ebi.ac.uk is the europepmc portal (resolve_oa + JATS fetch); other hosts pass through.
    return Deps(search=search, http_get=rate_limit.limited_by_url(requests.get), lookup=RoLookup(ro, cfg),
                store=WoStore(wo, ro, cfg),
                corpus_keys=keys, url_check=selective_fetch.is_public_url, refresh=refresh, close=close)


# ------------------------------------------------------------------------------------------------ evaluation
def _fd_cfg() -> dict:
    from vera import fulldoc_corpus
    return fulldoc_corpus.load_config()


def rank_question(question: str, cfg: dict | None) -> str:
    """DAR-TRACE-1022: the question with the acquisition profile's generic terms (search.openalex_profile.generic_terms:
    "different", "types", "key", ...) removed, for the pre-fetch ORDER only. Unchanged when nothing distinctive would
    remain. The relevance gate itself still scores the full question."""
    generic = {str(g).lower() for g in (((cfg or {}).get("search") or {}).get("openalex_profile") or {}).get(
        "generic_terms") or ()}
    if not generic:
        return question
    words = re.findall(r"\S+", str(question or ""))
    kept = [w for w in words if re.sub(r"[^\w-]", "", w).lower() not in generic]
    return " ".join(kept) if len(kept) < len(words) and kept else question


def prerank_score(question: str, cand: dict, fdcfg: dict) -> float:
    """Deterministic relevance of the candidate's title + abstract to the question (the answer path's stem overlap),
    used only to ORDER candidates; the decision gate is the admission over the full text (relevance_gate)."""
    from vera import fulldoc_admission as fa
    r = fa._routing(fdcfg)
    icfg, n = r.get("identification") or {}, int((r.get("corpus") or {}).get("relevance_stem_len") or 5)
    if not icfg:
        return 0.0
    return round(fa._relevance(question, f"{cand.get('title') or ''} {cand.get('snippet') or ''}", icfg, n), 6)


def candidate_isbn(cand: dict) -> dict | None:
    """ISBN post-script (owner 2026-10-21): only for books / book chapters, only from provider metadata, validated."""
    kind = str(cand.get("source_type") or cand.get("type") or "").strip().lower().replace("_", "-")
    if kind not in ("book", "book-chapter", "monograph", "edited-book", "reference-book", "book-part"):
        return None
    vals = []
    for k in ("isbn", "isbns", "isbn13", "isbn10"):
        v = cand.get(k)
        if isinstance(v, (list, tuple)):
            vals.extend(v)
        elif v:
            vals.append(v)
    return acq_config.first_isbn(vals)


def _doi(c: dict) -> str | None:
    d = str(c.get("doi") or "").strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d if re.fullmatch(r"10\.\d{4,9}/\S+", d) else None


def _pmcid(c: dict) -> str | None:
    """PLAN-NONDOI-1022: the candidate's PMCID (e.g. a Europe PMC open-access hit), or None. Used only when the
    candidate has no DOI."""
    from vera.search_providers import _pmcid_of
    oa = c.get("oa") if isinstance(c.get("oa"), dict) else {}
    return _pmcid_of(c.get("pmcid") or oa.get("pmcid"))


# ------------------------------------------------------------------------------------------------ resolve + fetch
class _Stop(Exception):
    def __init__(self, code: str, n_adm: int = 0):
        super().__init__(code)
        self.code = code
        self.n_adm = n_adm  # ONE-SPAN-1021: qualifying passages counted by the relevance gate (0 before it ran)


def _guarded_get(url: str, cfg: dict, deps: Deps, max_bytes: int, timeout_s: float, want_json: bool,
                 clock=time.monotonic, deadline: float | None = None, chain: list | None = None):
    """GET with the host allowlist checked on every hop, https only, the SSRF guard, manual redirects, content type, size
    cap, and a wall-clock limit over the WHOLE fetch (requests' timeout is per read; plan V-D7). Every streamed
    response is closed, including on an early stop.

    P0-5 (discovered candidates only, i.e. `chain` is a list): every requested hop is appended to `chain`; a revisited
    hop stops with reason `cycle` (before: only the hop cap, as unreachable); userinfo or a port other than 443 stops
    with `host`; Accept-Encoding: identity is sent and any other Content-Encoding is refused. chain=None (every
    existing caller) is byte-identical to before."""
    from urllib.parse import urljoin

    from vera import selective_fetch as sf
    allowed = {h.lower() for h in ((cfg.get("fetch") or {}).get("allowed_hosts") or ())}
    ctypes = {c.lower() for c in ((cfg.get("fetch") or {}).get("content_types") or ())}
    check = deps.url_check or sf.is_public_url
    end = clock() + float(timeout_s)
    if deadline is not None:
        end = min(end, deadline)

    def late():
        if clock() > end:
            raise _Stop(acq_config.reason(cfg, "deadline"))

    current, hops, resp = url, 0, None
    visited = set()
    try:
        while True:
            late()
            sp = urlsplit(current)
            if sp.scheme != "https" or (sp.hostname or "").lower() not in allowed:
                raise _Stop(acq_config.reason(cfg, "host"))
            if chain is not None:  # P0-5 (new path only)
                from vera import acq_discovered
                if not acq_discovered._host_ok(current, allowed):
                    raise _Stop(acq_config.reason(cfg, "host"))
                if acq_discovered._norm(current) in visited:
                    raise _Stop(acq_config.reason(cfg, "cycle"))
                visited.add(acq_discovered._norm(current))
            ok, _why = check(current)
            if not ok:
                raise _Stop(acq_config.reason(cfg, "host"))
            if chain is not None:
                chain.append(acq_discovered._clean(current, 2000))
            hdrs = {"User-Agent": sf.USER_AGENT} if chain is None else {"User-Agent": sf.USER_AGENT,
                                                                        "Accept-Encoding": "identity"}
            try:
                resp = deps.http_get(current, headers=hdrs, timeout=timeout_s,
                                     allow_redirects=False, stream=True)
            except rate_limit.RateLimited:  # RATE-1022 B2: skipped, nothing sent; never recorded as unreachable
                raise _Stop(acq_config.reason(cfg, "rate_limited")) from None
            except Exception:  # noqa: BLE001  unreachable = unusable (never substituted by an abstract)
                raise _Stop(acq_config.reason(cfg, "unreachable")) from None
            if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
                hops += 1
                _close(resp)
                if hops > sf.MAX_REDIRECTS:
                    raise _Stop(acq_config.reason(cfg, "unreachable"))
                current = urljoin(current, resp.headers["Location"])
                resp = None
                continue
            break
        if resp.status_code != 200:
            raise _Stop(acq_config.reason(cfg, "unreachable"))
        if chain is not None and str(resp.headers.get("Content-Encoding") or "identity").strip().lower() != "identity":
            raise _Stop(acq_config.reason(cfg, "encoding"))
        ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if want_json:
            if ctype not in ("application/json", "text/json"):
                raise _Stop(acq_config.reason(cfg, "unreachable"))
        elif ctype not in ctypes:
            raise _Stop(acq_config.reason(cfg, "convert"))
        chunks, size = [], 0
        try:
            for chunk in resp.iter_content(65536):
                size += len(chunk)
                if size > max_bytes:
                    raise _Stop(acq_config.reason(cfg, "too_large"))
                late()
                chunks.append(chunk)
        except _Stop:
            raise
        except Exception:  # noqa: BLE001
            raise _Stop(acq_config.reason(cfg, "unreachable")) from None
        return b"".join(chunks), current
    finally:
        _close(resp)


def _close(resp) -> None:
    try:
        if resp is not None and hasattr(resp, "close"):
            resp.close()
    except Exception:  # noqa: BLE001
        pass


def resolve_oa(doi: str | None, cfg: dict, deps: Deps, clock=time.monotonic, deadline: float | None = None,
               pmcid: str | None = None, chain: list | None = None) -> dict:
    """Europe PMC REST search by DOI (core result), or by PMCID when the candidate has no DOI (PLAN-NONDOI-1022).
    Raises _Stop with a reason code when not usable. The result's "doi" is the core record's well-formed DOI or None."""
    rcfg = cfg.get("resolve") or {}
    # INT-1022-B (orchestrator disposition b): refuse BEFORE any network call when there is neither a DOI nor a
    # well-formed PMCID (never a query on an empty or malformed identifier).
    pmcid = str(pmcid or "").strip().upper() or None
    if not doi and not (pmcid and re.fullmatch(r"PMC\d{1,10}", pmcid)):
        raise _Stop(acq_config.reason(cfg, "not_oa"))
    from urllib.parse import urlencode
    url = rcfg["search_url"] + "?" + urlencode({"query": f'DOI:"{doi}"' if doi else f'PMCID:"{pmcid}"', "resultType": "core", "format": "json",
                                                "pageSize": 1})
    raw, _final = _guarded_get(url, cfg, deps, 2_000_000, float(acq_config.bound(cfg, "fetch_timeout_s")), True,
                               clock=clock, deadline=deadline, **({"chain": chain} if chain is not None else {}))
    try:
        doc = json.loads(raw.decode("utf-8"))
        res = ((doc.get("resultList") or {}).get("result") or [])
        r = res[0] if res and isinstance(res[0], dict) else None
    except Exception:  # noqa: BLE001
        r = None
    rdoi = str((r or {}).get("doi") or "").strip().lower()
    if r is None or (doi and rdoi != doi) or (not doi and (not pmcid or
                                                           str(r.get("pmcid") or "").strip().upper() != pmcid)):
        raise _Stop(acq_config.reason(cfg, "not_oa"))
    pmcid = str(r.get("pmcid") or "").strip().upper()
    if rcfg.get("require_open_access", True) and (str(r.get("isOpenAccess") or "").upper() != "Y"
                                                   or not re.fullmatch(r"PMC\d{1,10}", pmcid)):
        raise _Stop(acq_config.reason(cfg, "not_oa"))
    if str(r.get("source") or "").upper() in {s.upper() for s in rcfg.get("refuse_sources") or ()}:
        raise _Stop(acq_config.reason(cfg, "pub_type"))
    ptypes = (r.get("pubTypeList") or {}).get("pubType") or []
    ptypes = [ptypes] if isinstance(ptypes, str) else ptypes
    refuse = {p.lower() for p in rcfg.get("refuse_pub_types") or ()}
    if any(str(p).strip().lower() in refuse for p in ptypes):
        raise _Stop(acq_config.reason(cfg, "pub_type"))
    lic = str(r.get("license") or "").strip()
    lic_url = acq_config.licence_url_for(lic, cfg)
    if not lic or not lic_url:
        raise _Stop(acq_config.reason(cfg, "no_licence"))  # unclear licence = hold (never stored)
    book = r.get("bookOrReportDetails") if isinstance(r.get("bookOrReportDetails"), dict) else {}
    return {"pmcid": pmcid, "licence": lic, "licence_url": lic_url, "title": " ".join(str(r.get("title") or "").split()),
            "authors": " ".join(str(r.get("authorString") or "").split()),
            "year": str(r.get("pubYear") or "").strip(), "pub_types": [str(p) for p in ptypes],
            "isbn_values": [book.get(k) for k in ("isbn13", "isbn10", "isbn") if book.get(k)], "is_book": bool(book),
            "doi": _doi({"doi": rdoi})}


def _family(licence_or_url: str) -> str | None:
    s = str(licence_or_url or "").strip().lower()
    m = re.search(r"licenses/([a-z\-]+)/", s)
    if m:
        return m.group(1)
    if "publicdomain/zero" in s or re.fullmatch(r"cc0(\s*1\.0)?", s):
        return "zero"
    m = re.fullmatch(r"cc[\s\-]+([a-z][a-z\-]*)(?:[\s\-]+\d(?:\.\d)?)?", re.sub(r"\s+", " ", s))
    return m.group(1).replace(" ", "-") if m else None


# ------------------------------------------------------------------------------------------------ the job
def build_document(doi: str | None, cand: dict, res: dict, raw: bytes, final_url: str, cfg: dict, icfg: dict, job: Job,
                   score: float, now_iso: str, landing_url: str | None = None) -> tuple[dict, dict, dict, str]:
    """(rec, item, meta, text) after conversion and EVERY fulltext_loader filter; raises _Stop / Refused."""
    from vera import fulltext_loader as fl
    from vera import jats_text
    try:
        conv = jats_text.convert(raw, (cfg.get("fetch") or {}).get("fulltext_format", "jats-xml"), icfg["converter"])
    except Exception:  # noqa: BLE001  (ConversionError or a hostile file)
        raise _Stop(acq_config.reason(cfg, "convert")) from None
    text = conv["text"]
    meta = {k: v for k, v in conv.items() if k != "text"}
    meta.update({"document_id": None, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                 "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "raw_file": None})
    jm = meta.get("meta") or {}
    if not doi:  # PLAN-NONDOI-1022 rule 6: a DOI stated in the JATS article-id gives the DOI identity (one paper, one URL)
        doi = _doi({"doi": jm.get("doi")})
    did = "acq-" + hashlib.sha256((doi if doi else "pmcid:" + res["pmcid"]).encode("utf-8")).hexdigest()[:16]
    meta["document_id"] = did
    authors = res.get("authors") or "; ".join(str(a) for a in (jm.get("authors") or []) if a)
    year = res.get("year") or str(jm.get("year") or "")
    item = {"document_id": did, "title": res.get("title") or cand.get("title") or "", "doi": doi,
            "landing_url": landing_url or cfg["resolve"]["landing_url_template"].format(pmcid=res["pmcid"]),
            "fulltext_url": final_url, "fulltext_format": (cfg.get("fetch") or {}).get("fulltext_format", "jats-xml"),
            "text_file": f"acquired/{did}.xml", "licence": res["licence"], "licence_url": res["licence_url"],
            "authors": authors, "published_year": year, "fetched_at": now_iso,
            "attribution": None, "verifier_status": icfg["verified_status"], "read_status": "VERBATIM"}
    if not doi:  # PLAN-NONDOI-1022: PMCID identity (doi stays None; the loader waives the DOI only for this kind)
        item["identifier"] = {"kind": "pmcid", "id": res["pmcid"], "url": item["landing_url"]}
    # WED-P17-VER v2: EVERY <license> element of the JATS (href, license-type, ali:license_ref, license-p text) is
    # classified BEFORE the loader runs; the loader itself reads only the first element (base path unchanged). The
    # summary is kept in the conversion record so the acquired-path reader re-applies the same decision.
    meta["jats_licences"] = jats_licences(raw)
    licence_elements_decision(meta["jats_licences"], res["licence"], icfg)
    meta["jats_licences"] = [{**e, "text": (e.get("text") or "")[:1000] or None} for e in meta["jats_licences"]]  # stored
    rec = fl.prepare_from_parts(item, text, meta, icfg, now=now_iso)  # every filter + the licence rule (reject first)
    return rec, item, meta, text


_TEXT_FAMILY = re.compile(r"\b(?:cc|creative\s+commons)[\s\-]+(by(?:[\s\-]+(?:nc|sa|nd))*)\b|\b(cc0)\b|"
                          r"creative\s+commons\s+attribution((?:[\s\-]+(?:non\s*commercial|share\s*alike|"
                          r"no\s*derivatives?))*)")


def _text_family(s: str) -> str | None:
    t = re.sub(r"\s+", " ", str(s or "").lower())
    m = _TEXT_FAMILY.search(t)
    if not m:
        return None
    if m.group(2):
        return "zero"
    if m.group(1):
        return re.sub(r"[\s\-]+", "-", m.group(1))
    parts = ["by"] + [{"n": "nc", "s": "sa"}.get(x.strip()[0], "nd") for x in re.findall(
        r"non\s*commercial|share\s*alike|no\s*derivatives?", m.group(3) or "")]
    return "-".join(parts)


def jats_licences(raw: bytes) -> list:
    """[{href, type, ref, text}] for every <license> element (namespace-agnostic). Parsed with defusedxml, like the
    converter (the same bytes already passed its guards). Statement text is truncated; it is a licence notice."""
    import defusedxml.ElementTree as DET
    out = []
    try:
        root = DET.fromstring(raw)
    except Exception:  # noqa: BLE001  (the converter already accepted these bytes; fail closed on a surprise)
        return [{"href": None, "type": None, "ref": None, "text": "unparseable licence block"}]

    def ln(tag):
        return str(tag).rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""
    for el in root.iter():
        if ln(el.tag) != "license":
            continue
        href = next((v for k, v in el.attrib.items() if ln(k) == "href"), None)
        refs = [" ".join("".join(x.itertext()).split()) for x in el.iter() if ln(x.tag) == "license_ref"]
        text = " ".join(" ".join("".join(x.itertext()).split()) for x in el.iter() if ln(x.tag) == "license-p")
        out.append({"href": (href or "").strip() or None, "type": (el.get("license-type") or "").strip() or None,
                    "ref": (" ".join(refs)).strip() or None, "text": text or None})  # SEC L1: full text decided
    return out


def licence_elements_decision(elements: list, declared: str, icfg: dict) -> None:
    """Raises Refused: any reject marker (ND, no educational use, fee only) in any element's href, type, ref or text ->
    reject; an element with no recognisable family, more than one family, or a family that differs from the declared
    licence -> hold (unclear = hold). No licence element at all: the declared licence governs (returns)."""
    from vera import fulltext_loader as fl
    from vera.licence_gate import reject_reason
    codes = icfg["reason_codes"]
    why = reject_reason(declared, None)  # a rejecting declared licence is a reject, never only a hold
    if why:
        raise fl.Refused(codes["licence_reject"], f"declared licence is {why}")
    fams = set()
    for e in elements or ():
        why = reject_reason(" ".join(x for x in (e.get("type"), e.get("text")) if x),
                            " ".join(x for x in (e.get("href"), e.get("ref")) if x))
        if why:
            raise fl.Refused(codes["licence_reject"], f"a licence element of the document is {why}")
        ef = {f for f in (_family(e.get("href") or ""), _family(e.get("ref") or ""), _text_family(e.get("text") or ""))
              if f}
        if not ef:
            raise fl.Refused(codes["licence_hold"], "a licence element of the document is not a recognised licence: held")
        fams |= ef
    if not fams:
        return
    fd = _family(declared)
    if len(fams) > 1 or fd is None or fams != {fd}:
        raise fl.Refused(codes["licence_mismatch"], "licence elements disagree with each other or the declared licence: held")


def build_plan(rec: dict, item: dict, meta: dict, cfg: dict, job: Job, score: float, value: dict, isbn: dict | None,
               now_iso: str) -> dict:
    """The G4 row plan (vera.fulldoc_db.build_rows, live chunking_params: CAL parity) plus the acquisition record."""
    from vera import fulldoc_db
    item_sha = hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    pins = {"manifest_sha256": item_sha, "sums_sha256": rec["content_hash"]}
    plan = fulldoc_db.build_rows(rec, item, meta, 0, pins, fulldoc_db.chunking_params(), n_documents=1)
    prov = plan["source"]["provenance"]
    # WED-P17-VER item 3: no trigger kind and no question-derived score are stored (the disclosure names source,
    # licence and time added; job_id is random and carries no visitor information).
    prov["acquisition"] = {"marker": acq_config.marker(cfg), "job_id": job.job_id,
                           "providers": list((cfg.get("search") or {}).get("providers") or ()),
                           "resolver": "europepmc", "acquired_at": now_iso, "value": value,
                           "code_version": CODE_VERSION}
    if isbn:  # post-script, after every existing field (owner 2026-10-21)
        prov["isbn"] = isbn["isbn"]
        if isbn.get("isbn10"):
            prov["isbn10"] = isbn["isbn10"]
    plan["candidate"] = {"source_url": item["landing_url"], "title": item["title"],
                         "rationale": rec["rationale"]}
    plan["canonical_url"] = rec["canonical_url"]
    return plan


def relevance_gate(question: str, plan: dict, cfg: dict, icfg: dict, fdcfg: dict) -> int:
    """Admitted span count when the answer path's own admission runs over a corpus that holds ONLY this document,
    loaded from the exact rows by the acquired-path reader (so a pass also proves the reader accepts them)."""
    from vera import fulldoc_admission, fulldoc_db
    src = {"run_id": -1, "source_id": 1, "canonical_url": plan["canonical_url"], "content": plan["source"]["content"],
           "content_hash": plan["source"]["content_hash"], "provenance": json.loads(json.dumps(plan["source"]["provenance"]))}
    spans = [{"source_id": 1, "start_index": s["start_index"], "end_index": s["end_index"], "text": s["text"]}
             for s in plan["spans"]]
    corpus = fulldoc_db.load_acquired_rows([src], spans, fdcfg, cfg, ingest_cfg=icfg)
    if corpus.status != "ok" or not corpus.acquired:
        codes = [u.get("code") for u in corpus.unusable]
        raise _Stop(str(codes[0]) if codes and codes[0] else acq_config.reason(cfg, "convert"))
    ev = fulldoc_admission.admit(question, corpus, fdcfg)
    return len(ev.admitted or ())


def rank_candidates(question: str, cands, cfg: dict, served: set, fdcfg: dict, count) -> list:
    """Stage (shared by the background job and DAR-INLINE-1021): DOI or PMCID required (PLAN-NONDOI-1022), not already
    served (DOI/URL/ISBN), deduplicated by identity, ordered by the deterministic pre-rank, capped at max_candidates_resolved."""
    from vera import search_providers
    from vera.dar_corpus import doc_key_set
    ranked, seen = [], set()
    rq = rank_question(question, cfg)
    for pos, c in enumerate(cands or ()):
        if not isinstance(c, dict):
            continue
        doi = _doi(c)
        pmcid = None if doi else _pmcid(c)
        if not doi and not pmcid:
            count(acq_config.reason(cfg, "no_doi"))
            continue
        ident = ("doi:" + doi) if doi else ("pmc:" + pmcid)
        why = search_providers.acq_prefilter(c)  # YIELD-1021: licence/OA pre-filter BEFORE the cut (metadata only)
        if why:
            count(acq_config.reason(cfg, "not_oa" if why == "not_oa" else "no_licence"))
            continue
        isbn = candidate_isbn(c)
        keys = doc_key_set(doi, c.get("url"), (isbn or {}).get("isbn"))
        if ident in seen or keys & served:
            count(acq_config.reason(cfg, "in_corpus"))
            continue
        seen.add(ident)
        ranked.append((prerank_score(rq, c, fdcfg), doi, c, isbn, pos))
    # YIELD-1021: DDG-mapped first. DAR-TRACE-1022: ties keep the providers' own relevance order (merge position), not
    # the DOI string (Europe PMC lite carries no abstract, so most title scores tie and the DOI sort discarded its order).
    ranked.sort(key=lambda x: (search_providers.acq_rank_tier(x[2]), -x[0], x[4]))
    ranked = [r[:4] for r in ranked]
    cut = ranked[: int(acq_config.bound(cfg, "max_candidates_resolved"))]
    by: dict = {}
    for _s, _d, c, _i in cut:
        by[search_providers.acq_source(c)] = by.get(search_providers.acq_source(c), 0) + 1
    _log("ranked", **{f"src_{k}": v for k, v in sorted(by.items())})
    return cut


def evaluate_candidate(job: Job, score: float, doi: str, cand: dict, isbn, cfg: dict, deps: Deps, icfg: dict,
                       fdcfg: dict, clock=time.monotonic, deadline: float | None = None, dur: dict | None = None,
                       min_spans: int | None = None):
    """Stage: known check -> resolve OA -> fetch JATS -> convert + every loader filter + licence rule -> injection scan
    -> row plan -> relevance gate (min_spans; default acq_config.persist_min_spans, the persistent-corpus value; the
    inline answer path passes acq_config.answer_min_spans). Returns (plan, isbn) with plan["admitted_spans"] = the
    gate's count; raises _Stop (n_adm set by the gate) / fl.Refused. Writes nothing."""
    dur = {} if dur is None else dur
    if deps.lookup.known(doi, cand.get("url"), (isbn or {}).get("isbn")):
        raise _Stop(acq_config.reason(cfg, "in_corpus"))
    t = clock()
    # P0-5: a discovered candidate (only present with inline.use_discovered) records its URL chain; Step B
    # (inline.landing_route) tries ONE DOI -> landing -> JATS location when Europe PMC has no open-access record.
    disc = cand.get("discovery") if isinstance(cand.get("discovery"), dict) else None
    chain = [] if disc is not None else None
    route, landing = "europepmc", None
    if disc is None:
        res = resolve_oa(doi, cfg, deps, clock=clock, deadline=deadline, pmcid=None if doi else _pmcid(cand))
    else:
        from vera import acq_discovered
        try:
            res = resolve_oa(doi, cfg, deps, clock=clock, deadline=deadline, pmcid=None if doi else _pmcid(cand),
                             chain=chain)
        except _Stop as e:
            if not (doi and e.code == acq_config.reason(cfg, "not_oa") and acq_discovered.landing_on(cfg)):
                raise
            route = "landing"
            raw, final, landing, res = acq_discovered.landing_fetch(doi, cand, cfg, deps, icfg, clock=clock,
                                                                    deadline=deadline, chain=chain)
    if not doi and res.get("doi"):  # PLAN-NONDOI-1022 rule 6: promote to the DOI identity, re-check known
        doi = res["doi"]
        if deps.lookup.known(doi, None, (isbn or {}).get("isbn")):
            raise _Stop(acq_config.reason(cfg, "in_corpus"))
    if isbn is None and res.get("is_book"):
        isbn = acq_config.first_isbn(res.get("isbn_values"))
        if isbn and deps.lookup.known(doi, None, isbn["isbn"]):
            raise _Stop(acq_config.reason(cfg, "in_corpus"))
    if route == "europepmc":
        url = cfg["resolve"]["fulltext_url_template"].format(pmcid=res["pmcid"])
        raw, final = _guarded_get(url, cfg, deps, int(acq_config.bound(cfg, "max_fetch_bytes")),
                                  float(acq_config.bound(cfg, "fetch_timeout_s")), False, clock=clock,
                                  deadline=deadline, **({"chain": chain} if chain is not None else {}))
    dur["fetch"] = dur.get("fetch", 0) + int((clock() - t) * 1000)
    t = clock()
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rec, item, meta, text = build_document(doi, cand, res, raw, final, cfg, icfg, job, score, now_iso,
                                           **({"landing_url": landing} if landing else {}))
    if not doi and item.get("doi"):  # PLAN-NONDOI-1022: promoted from the JATS article-id; re-check known
        if deps.lookup.known(item["doi"], None, (isbn or {}).get("isbn")):
            raise _Stop(acq_config.reason(cfg, "in_corpus"))
    if acq_config.injection_refused(f"{item.get('title') or ''}\n{text}", cfg):  # SEC L2: title too
        raise _Stop(acq_config.reason(cfg, "inject"))
    value = {"open_access": True, "licence_family": _family(res["licence"]) or "",
             "pub_types_n": len(res.get("pub_types") or [])}
    plan = build_plan(rec, item, meta, cfg, job, score, value, isbn, now_iso)
    if disc is not None:
        acq_discovered.annotate(plan, cand, res, route, chain, cfg)
    n_adm = relevance_gate(job.question, plan, cfg, icfg, fdcfg)
    dur["verify"] = dur.get("verify", 0) + int((clock() - t) * 1000)
    need = acq_config.persist_min_spans(cfg) if min_spans is None else max(1, int(min_spans))
    if n_adm < need:
        raise _Stop(acq_config.reason(cfg, "few_spans") if n_adm else acq_config.reason(cfg, "irrelevant"), n_adm)
    plan["admitted_spans"] = n_adm  # ONE-SPAN-1021: read by the caller's persist decision; never written to the DB
    if disc is not None:
        acq_discovered.gate_b_admit(plan)  # P0-5 D3: admit + basis, no score or count
    return plan, isbn


def persist_eligible(plan: dict, cfg: dict) -> bool:
    """ONE-SPAN-1021: True when the plan's gate count meets the persistent-corpus minimum (>= 2)."""
    return int(plan.get("admitted_spans") or 0) >= acq_config.persist_min_spans(cfg)


def error_code(cfg: dict, e: Exception) -> str:
    return (acq_config.reason(cfg, "write_failed") if type(e).__name__ in ("WriteFailed", "IntegrityError",
                                                                          "OperationalError", "ProgrammingError")
            else f"R-ACQ-ERR-{type(e).__name__}"[:40])


def reserve_job(cfg: dict, question: str, clock=time.monotonic) -> str | None:
    """DAR-INLINE-1021: count one inline acquisition against the SAME jobs_per_hour / jobs_per_day caps as enqueue.
    None = reserved; else the refusal reason code. Also marks the question as recently acquired (no duplicate
    background job for it)."""
    now = clock()
    with _S.lock:
        _prune(_S.jobs_hour, now, 3600.0)
        _prune(_S.jobs_day, now, 86400.0)
        if len(_S.jobs_hour) >= int(acq_config.bound(cfg, "jobs_per_hour")) or \
                len(_S.jobs_day) >= int(acq_config.bound(cfg, "jobs_per_day")):
            return acq_config.reason(cfg, "rate")
        _S.jobs_hour.append(now)
        _S.jobs_day.append(now)
        _S.recent[_qkey(question)] = now
    return None


def docs_budget(cfg: dict, deps: Deps, clock=time.monotonic, reserve: bool = False) -> int:
    """Documents this job may still store: min(docs_per_day remaining (local and DB count, minus documents other
    in-progress inline jobs reserved), max_docs_per_job). Fails closed (0) when the DB count is unavailable.
    VER L5: reserve=True reserves the returned number under the same lock (release it with settle_docs), so two
    concurrent inline jobs can no longer both spend the same remaining daily budget.
    DAR-TRACE-1022 (owner 2026-10-22): docs_per_day null = no daily cap: max_docs_per_job only, and the RO daily count
    is not read at all (a failed count can no longer block acquisition)."""
    cap = acq_config.docs_per_day_cap(cfg)
    if cap is None:
        n = max(0, int(acq_config.bound(cfg, "max_docs_per_job")))
        if reserve:
            with _S.lock:
                _S.docs_reserved += n
        return n
    try:
        db_day = int(deps.lookup.docs_last_24h())
    except Exception:  # noqa: BLE001
        return 0
    now = clock()
    with _S.lock:
        _prune(_S.docs_day, now, 86400.0)
        local_day = len(_S.docs_day)
        n = max(0, min(cap - max(local_day, db_day) - _S.docs_reserved, int(acq_config.bound(cfg, "max_docs_per_job"))))
        if reserve:
            _S.docs_reserved += n
    return n


def settle_docs(reserved: int, used: int, clock=time.monotonic) -> None:
    """Release a docs_budget(reserve=True) reservation and count the `used` documents against docs_per_day."""
    with _S.lock:
        _S.docs_reserved = max(0, _S.docs_reserved - int(reserved))
        for _ in range(int(used)):
            _S.docs_day.append(clock())


def note_docs(n: int, clock=time.monotonic) -> None:
    with _S.lock:
        for _ in range(int(n)):
            _S.docs_day.append(clock())


def run_job(job: Job, cfg: dict, deps: Deps, clock=time.monotonic, env=None) -> dict:
    """Never raises. Returns {"codes": {code: n}, "stored": n, "durations_ms": {...}} (also logged, counts only)."""
    from vera import fulltext_loader as fl
    t0 = clock()
    codes: dict = {}
    dur: dict = {}
    stored = 0
    tel = {"answer_eligible": 0}  # ONE-SPAN-1021: documents meeting answer_min_admitted_spans (none answer here)

    def count(code):
        codes[code] = codes.get(code, 0) + 1

    def done():
        return _done(job, codes, stored, dur, t0, clock, tel)

    try:
        if not acq_config.enabled(cfg, os.environ if env is None else env):  # kill switch re-checked per job
            count(acq_config.reason(cfg, "disabled"))
            return done()
        if clock() - job.created > float(acq_config.bound(cfg, "question_max_age_s")):
            count(acq_config.reason(cfg, "expired"))
            return done()
        deadline = t0 + float(acq_config.bound(cfg, "job_deadline_s"))
        now = clock()
        with _S.lock:
            _prune(_S.docs_day, now, 86400.0)
            local_day = len(_S.docs_day) + _S.docs_reserved  # VER L5: inline reservations count too
        cap = acq_config.docs_per_day_cap(cfg)  # DAR-TRACE-1022 (owner 2026-10-22): None = no daily document cap
        if cap is None:
            budget = int(acq_config.bound(cfg, "max_docs_per_job"))
        else:
            try:
                db_day = int(deps.lookup.docs_last_24h())
            except Exception:  # noqa: BLE001  fail closed: no count, no acquisition
                count(acq_config.reason(cfg, "daily_cap"))
                return done()
            if max(local_day, db_day) >= cap:
                count(acq_config.reason(cfg, "daily_cap"))
                return done()
            budget = min(cap - max(local_day, db_day), int(acq_config.bound(cfg, "max_docs_per_job")))
        # max_groq_tokens_per_job: this job makes NO model call (deterministic evaluation; lowest-cost default). The
        # bound is kept in config so a future model-assisted evaluation must stay within it (0 = none allowed).
        s = cfg.get("search") or {}
        t = clock()
        try:
            n_calls = int(acq_config.bound(cfg, "max_search_calls_per_job"))
            cands = list(deps.search(job.question, int(s.get("results_per_provider", 10)),
                                     float(s.get("timeout_s", 20))) or []) if n_calls > 0 else []
        except Exception:  # noqa: BLE001
            count(acq_config.reason(cfg, "search_failed"))
            return done()
        dur["search"] = int((clock() - t) * 1000)
        fdcfg = _fd_cfg()
        icfg = acq_config.ingest_cfg(fl.load_config(), cfg)
        served = set(deps.corpus_keys() or ())
        from vera.dar_corpus import doc_key_set
        ranked = rank_candidates(job.question, cands, cfg, served, fdcfg, count)
        for score, doi, cand, isbn in ranked:
            if stored >= budget:
                break
            if clock() >= deadline:
                count(acq_config.reason(cfg, "deadline"))
                break
            try:
                plan, isbn = evaluate_candidate(job, score, doi, cand, isbn, cfg, deps, icfg, fdcfg, clock, deadline, dur)
                tel["answer_eligible"] += 1
                t = clock()
                out = deps.store.write(plan)
                dur["write"] = dur.get("write", 0) + int((clock() - t) * 1000)
                if out.get("status") == "known":
                    raise _Stop(acq_config.reason(cfg, "in_corpus"))
                stored += 1
                with _S.lock:
                    _S.docs_day.append(clock())
                served |= doc_key_set(doi or plan["source"]["provenance"].get("doi"),
                                      cand.get("url"), (isbn or {}).get("isbn"))  # PLAN-NONDOI-1022: promoted DOI
                count(acq_config.reason(cfg, "stored"))
            except _Stop as e:
                count(e.code)
                if e.n_adm >= acq_config.answer_min_spans(cfg):  # would answer inline; below persist min: not stored
                    tel["answer_eligible"] += 1
            except fl.Refused as e:
                count(str(e.code))
            except Exception as e:  # noqa: BLE001
                count(error_code(cfg, e))
        if stored and deps.refresh is not None:
            try:
                deps.refresh()
            except Exception as e:  # noqa: BLE001
                logger.warning("autoacq refresh failed (%s)", type(e).__name__)
    except Exception as e:  # noqa: BLE001
        count(f"R-ACQ-ERR-{type(e).__name__}"[:40])
    return done()


def _done(job, codes, stored, dur, t0, clock, tel=None) -> dict:
    dur["total"] = int((clock() - t0) * 1000)
    n_elig = int((tel or {}).get("answer_eligible", 0))
    out = {"codes": dict(sorted(codes.items())), "stored": stored, "durations_ms": dur, "trigger": job.trigger,
           "n_answer_eligible": n_elig, "n_persisted": stored}
    _log("job", stored=stored, trigger=job.trigger, n_answer_eligible=n_elig, n_persisted=stored,
         **{f"n_{k}": v for k, v in out["codes"].items()}, **{f"ms_{k}": v for k, v in dur.items()})
    return out


# ------------------------------------------------------------------------------------------------ DB (injected engines)
_SCHEMA = "vera_vjay"


def _sql(s: str) -> str:
    return s.replace("{S}", _SCHEMA)


class RoLookup:
    """Read-only account: membership by DOI canonical, URL, ISBN (acquired rows) and the 24 h acquired count."""

    def __init__(self, engine_ro, cfg: dict):
        self.engine, self.cfg = engine_ro, cfg

    def _rows(self, sql, params):
        from sqlalchemy import text
        with self.engine.connect() as c:
            return list(c.execute(text(_sql(sql)), params).fetchall())

    def known(self, doi, url, isbn) -> bool:
        from vera.corpus_admission import canonical_url
        cu = canonical_url({"doi": doi, "url": url or ""})
        r = self._rows("SELECT s.source_id FROM {S}.sources s JOIN {S}.canonical_sources cs "
                       "ON cs.canonical_source_id = s.canonical_source_id WHERE cs.canonical_url = :u "
                       "AND s.provenance->>'stored_text_kind' = 'fulldoc_chunks' LIMIT 1", {"u": cu})
        if r:
            return True
        if isbn:
            r = self._rows("SELECT s.source_id FROM {S}.sources s WHERE s.provenance->>'isbn' = :i "
                           "AND s.provenance->>'stored_text_kind' = 'fulldoc_chunks' LIMIT 1", {"i": str(isbn)})
            if r:
                return True
        return False

    def docs_last_24h(self) -> int:
        r = self._rows("SELECT count(*) FROM {S}.sources s WHERE s.provenance->'acquisition'->>'marker' = :m "
                       "AND s.created_at > now() - interval '24 hours'", {"m": acq_config.marker(self.cfg)})
        return int(r[0][0]) if r else 0

    def canonical_state(self, cu: str, text_sha: str):
        """(canonical_source_id | None, max version, fulldoc already stored?, same text anywhere?)."""
        r = self._rows("SELECT canonical_source_id FROM {S}.canonical_sources WHERE canonical_url = :u", {"u": cu})
        csid = r[0][0] if r else None
        ver, has_fd = 0, False
        if csid is not None:
            v = self._rows("SELECT coalesce(max(version), 0), bool_or(provenance->>'stored_text_kind' = "
                           "'fulldoc_chunks') FROM {S}.sources WHERE canonical_source_id = :c", {"c": csid})
            ver, has_fd = int(v[0][0] or 0), bool(v[0][1])
        h = self._rows("SELECT 1 FROM {S}.sources WHERE content_hash = :h OR provenance->>'text_sha256' = :h LIMIT 1",
                       {"h": text_sha})
        return csid, ver, has_fd, bool(h)


class WoStore:
    """Write-only account (vera_pipeline_wo): plain INSERT ... RETURNING in savepoints, one transaction per document
    (question, run, candidate, canonical [or a new version under an existing canonical], source, spans), the WoCorpusStore
    / FulltextStore pattern. The question row holds the fixed config text, never the visitor's."""

    def __init__(self, engine_wo, engine_ro, cfg: dict):
        self.engine, self.lookup, self.cfg = engine_wo, RoLookup(engine_ro, cfg), cfg

    @staticmethod
    def _one(conn, sql, params):
        from sqlalchemy import text
        row = conn.execute(text(_sql(sql)), params).first()
        return row[0] if row else None

    def _insert_new(self, conn, sql, params):
        from sqlalchemy.exc import IntegrityError
        sp = conn.begin_nested()
        try:
            v = self._one(conn, sql, params)
            sp.commit()
            return v
        except IntegrityError as exc:
            sp.rollback()
            if getattr(getattr(exc, "orig", None), "pgcode", None) not in (None, "23505"):
                raise
            return None

    def write(self, plan: dict) -> dict:
        prov = plan["source"]["provenance"]
        csid, ver, has_fd, same = self.lookup.canonical_state(plan["canonical_url"], prov["text_sha256"])
        if has_fd or same:
            return {"status": "known"}
        qtext = str(self.cfg["question_text"])
        with self.engine.begin() as c:
            self._one(c, "SELECT pg_advisory_xact_lock(hashtext(:h))", {"h": prov["text_sha256"]})
            qid = self._one(c, "INSERT INTO {S}.questions (research_question) VALUES (:q) RETURNING question_id",
                            {"q": qtext})
            rid = self._one(c, "INSERT INTO {S}.runs (question_id, question) VALUES (:q, :t) RETURNING run_id",
                            {"q": qid, "t": qtext})
            cc = plan["candidate"]
            ov = (self.cfg.get("ingest_overlay") or {}).get("candidate") or {}
            cid = self._insert_new(c, "INSERT INTO {S}.candidates (run_id, source_url, title, snippet, gate_a_decision, "
                                      "gate_a_rationale, fetch_status) VALUES (:r,:u,:t,NULL,:d,:ra,:fs) "
                                      "RETURNING candidate_id",
                                   {"r": rid, "u": cc["source_url"], "t": cc["title"], "d": ov.get("gate_a_decision"),
                                    "ra": cc["rationale"], "fs": ov.get("fetch_status", "fetched")})
            if cid is None:
                raise RuntimeError("candidate already present in a new run")
            if csid is None:
                csid = self._insert_new(c, "INSERT INTO {S}.canonical_sources (canonical_url) VALUES (:u) "
                                           "RETURNING canonical_source_id", {"u": plan["canonical_url"]})
                if csid is None:
                    raise RuntimeError("canonical source appeared concurrently")
            src = plan["source"]
            sid = self._one(c, "INSERT INTO {S}.sources (canonical_source_id, candidate_id, version, content, "
                               "content_hash, provenance) VALUES (:cs,:cid,:v,:t,:h,CAST(:p AS jsonb)) RETURNING source_id",
                            {"cs": csid, "cid": cid, "v": ver + 1, "t": src["content"], "h": src["content_hash"],
                             "p": json.dumps(prov)})
            from sqlalchemy import text
            for sp in plan["spans"]:
                c.execute(text(_sql("INSERT INTO {S}.evidence_spans (source_id, text, start_index, end_index, "
                                    "relevance_score, evidence_type) VALUES (:s,:t,:a,:b,NULL,:e)")),
                          {"s": sid, "t": sp["text"], "a": sp["start_index"], "b": sp["end_index"],
                           "e": sp["evidence_type"]})
        return {"status": "stored", "run_id": rid, "source_id": sid, "canonical_source_id": csid}
