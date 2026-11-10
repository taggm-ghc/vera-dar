"""LIST-INGEST-1110: the search-free list-ingest route (SEED-SCI-1110 S10, CHUNK-VALUE-1110 H.8), behind POST /ingest/list.

A bounded list of known identifiers (DOI, PMCID, PMID, arXiv id, or an original URL) is queued, one job per item, on the
EXISTING background acquisition queue (vera.auto_acquire), which runs the unchanged stages: Europe PMC OA resolve,
robots/TDM and rate gates per hop, licence gate and ND reject, injection scan, content/relevance check, and the
chunk-value filter where the base has one. Differences from a question-driven job, and nothing else:
  * NO search of any kind: not DuckDuckGo, not OpenAlex/Europe PMC search, not Wikipedia. The only lookups are by
    identifier (Europe PMC id lookup for a PMID; INSPIRE or OpenAlex by arXiv id for the journal original).
  * origin is always "seed" (set here, never read from the caller): seed share of the queue caps, low priority behind
    visitor jobs. The route is not part of the public UI.
  * arXiv is a portal only (OD-1110-ARXIV-PORTAL): an arXiv id or arxiv.org URL is resolved to its journal original
    (a non-10.48550 DOI); no original = "no-route-arxiv-original-not-found". arXiv text is never fetched or stored.
  * idempotent: an identifier already stored (read-only lookup) or already queued is skipped.
Per-item result codes are returned at once; each queued item's final job code is recorded in memory (bounded) and read
back with outcomes(). Logs carry counts only, never identifiers.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Callable, Mapping

logger = logging.getLogger("vera.listingest")
CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "list_ingest.json"
QUESTION = "list ingest of known identifiers"  # inert: the list job ignores it (rank question "", title-based content check)

_DOI = re.compile(r"^(?:doi:|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/\S+)$", re.I)
_PMCID = re.compile(r"^(?:pmcid:)?(PMC\d{4,9})$", re.I)
_PMID = re.compile(r"^(?:pmid:)(\d{1,9})$", re.I)
_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.", re.I)

_LOCK = threading.Lock()
_ACCEPTED: deque = deque()  # monotonic times of items accepted by this route (hour/day caps)
_OUTCOMES: "OrderedDict[str, dict]" = OrderedDict()


class ListIngestConfigError(ValueError):
    pass


def load_config(path: Path | None = None) -> dict:
    try:
        c = json.loads((path or CFG_PATH).read_text(encoding="utf-8"))
        for k in ("max_items_per_call", "max_items_per_hour", "max_items_per_day", "lookup_timeout_s", "outcomes_kept",
                  "max_arxiv_lookups_per_call", "topic_max_len"):
            if not isinstance(c.get(k), (int, float)) or isinstance(c.get(k), bool) or c[k] <= 0:
                raise ValueError(k)
        if not isinstance(c.get("queue_reserve_for_visitors"), int) or c["queue_reserve_for_visitors"] < 0 or \
                not isinstance(c.get("ingest_key_env"), str):
            raise ValueError("reserve")
        if not isinstance(c.get("result_codes"), dict) or not isinstance(c.get("enabled"), bool):
            raise ValueError("shape")
        return c
    except Exception as e:  # noqa: BLE001
        raise ListIngestConfigError("config/list_ingest.json unreadable or invalid") from e


def reset_for_tests() -> None:
    with _LOCK:
        _ACCEPTED.clear()
        _OUTCOMES.clear()


def parse_identifier(raw) -> tuple[str, str] | None:
    """(kind, canonical) with kind in doi | pmcid | pmid | arxiv | url, else None. Metadata only: no request."""
    s = str(raw or "").strip()
    if not s or len(s) > 500 or re.search(r"[\s\x00-\x1f]", s):
        return None
    from vera import arxiv_fulltext as axf
    m = _DOI.match(s)
    if m:
        doi = m.group(1).lower().rstrip(".,;")
        a = _ARXIV_DOI.match(doi)
        return ("arxiv", doi[a.end():]) if a else ("doi", doi)
    m = _PMCID.match(s)
    if m:
        return "pmcid", m.group(1).upper()
    m = _PMID.match(s)
    if m:
        return "pmid", m.group(1)
    m = axf._ID_TEXT.match(s)
    if m:
        return "arxiv", m.group(1).lower()
    if re.match(r"^https://", s, re.I):
        parsed = axf.parse_arxiv_url(s)
        if parsed:
            return "arxiv", parsed[0]
        return "url", s
    return None


# ------------------------------------------------------------------------------------------------ resolution
class Resolvers:
    """Identifier-only lookups; each is injectable (tests pass fakes; defaults use the gated API calls). Each returns
    None when nothing is found and raises on transport errors (the item then gets no-route)."""

    def __init__(self, pmid: Callable | None = None, arxiv_original: Callable | None = None,
                 known: Callable | None = None):
        self.pmid, self.arxiv_original, self.known = pmid, arxiv_original, known


def _default_pmid(pmid: str, cfg: Mapping, acq: Mapping) -> dict | None:
    from vera import search_providers
    got = search_providers.europepmc_ids_lookup([("pmid", pmid)], float(cfg["lookup_timeout_s"]),
                                                (acq.get("search") or {}).get("europepmc") or {})
    return got.get("pmid:" + pmid)


def _journal_doi(v) -> str | None:
    m = _DOI.match(str(v or "").strip())
    d = m.group(1).lower() if m else None
    if not d or _ARXIV_DOI.match(d):
        return None
    try:  # PORTAL-RULE-1110 Q4: every preprint-server DOI prefix (config arxiv_portal.openalex.preprint_doi_prefixes), not only arXiv's
        from vera import arxiv_portal
        if d.startswith(arxiv_portal._preprint_prefixes()):
            return None
    except Exception:  # noqa: BLE001  config unreadable: the host_role refusal in _run still applies
        pass
    return d


def _api_get(url: str, params: dict, timeout_s: float):
    """The gated identifier lookup: vera.api_permissions.call (entry, conditions, rate portal, UA) over the pinned
    public-only GET, purpose "metadata" (same pattern as vera.portal_inspire). Raises ApiDenied / RateLimited with
    nothing sent when the entry is missing, denied or limited."""
    from vera import api_permissions
    from vera import selective_fetch as sf
    with api_permissions.purpose_scope("metadata", url):
        return api_permissions.call(sf.pinned_get, url, params=params, timeout=timeout_s, purpose="metadata",
                                    allow_redirects=False)


def _default_arxiv_original(arxiv_id: str, cfg: Mapping) -> str | None:
    """The journal DOI of an arXiv id through INSPIRE, then OpenAlex (metadata-only identifier lookups through the
    existing api_permissions entries; no search, no body text)."""
    t = float(cfg["lookup_timeout_s"])
    try:
        r = _api_get(cfg["inspire"]["url"], {"q": f"arxiv:{arxiv_id}", "fields": "dois", "size": 1}, t)
        if getattr(r, "status_code", 0) == 200:
            for h in (r.json().get("hits") or {}).get("hits") or ():
                for d in (h.get("metadata") or {}).get("dois") or ():
                    j = _journal_doi(d.get("value") if isinstance(d, dict) else d)
                    if j:
                        return j
    except Exception as e:  # noqa: BLE001
        logger.info("list-ingest inspire lookup failed (%s)", type(e).__name__)
    r = _api_get(cfg["openalex"]["url"] + "https://doi.org/10.48550/arxiv." + arxiv_id,
                 {"select": "doi,locations"}, t)
    if getattr(r, "status_code", 0) != 200:
        return None
    for loc in r.json().get("locations") or ():
        j = _journal_doi((loc or {}).get("landing_page_url"))
        if j:
            return j
    return None


def _default_known(deps_factory, cfg, env) -> Callable:
    def known(doi, url):
        from vera import auto_acquire
        d = (deps_factory or auto_acquire.default_deps)(cfg, env)
        try:
            return bool(d.lookup.known(doi, url, None))
        finally:
            if d.close is not None:
                d.close()
    return known


# ------------------------------------------------------------------------------------------------ the route
def ingest(items, *, topic: str | None = None, cfg: dict | None = None, acq: dict | None = None, env=os.environ,
           clock=time.monotonic, resolvers: Resolvers | None = None, deps_factory=None, start_worker: bool = True) -> dict:
    """Queue each identifier; returns {"results": [{"input", "kind", "code"}...], "counts": {code: n}}. Never raises
    for a per-item problem. Raises ValueError("too_many") when len(items) > max_items_per_call (nothing is queued).
    The whole call (identifier lookups included) runs under origin=seed. `topic` (optional) is the relevance target for
    the content check; absent = each document's own title."""
    from vera import acq_config, request_origin
    cfg = cfg if cfg is not None else load_config()
    acq = acq if acq is not None else acq_config.load_config()
    rc = cfg["result_codes"]
    items = list(items or ())
    if len(items) > int(cfg["max_items_per_call"]):
        raise ValueError("too_many")
    topic = (str(topic).strip() if topic is not None else "") or None
    if topic is not None:  # same intent rule as a question; never stored or logged
        from vera import scope_router
        if len(topic) > int(cfg["topic_max_len"]) or not scope_router.intent_clean(topic):
            c = rc.get("invalid_topic", "invalid-topic")
            return {"results": [{"input": str(r)[:200], "kind": None, "code": c} for r in items],
                    "counts": {c: len(items)}}
    with request_origin.scope(request_origin.SEED):  # F2: forced; nothing from the caller selects the origin
        results = _run(items, topic, cfg, acq, rc, env, clock, resolvers or Resolvers(), deps_factory, start_worker)
    counts: dict = {}
    for x in results:
        counts[x["code"]] = counts.get(x["code"], 0) + 1
    logger.info("listingest %s", json.dumps({"n": len(results), **{k.replace("-", "_"): v for k, v in counts.items()}},
                                             sort_keys=True))
    return {"results": results, "counts": counts}


def _run(items, topic, cfg, acq, rc, env, clock, rv, deps_factory, start_worker) -> list:
    from vera import acq_config, auto_acquire
    known = rv.known or _default_known(deps_factory, acq, env)
    results, seen = [], set()
    n_arxiv = 0

    def out(raw, kind, code):
        results.append({"input": str(raw)[:200], "kind": kind, "code": rc.get(code, code)})

    for raw in items:
        if not cfg["enabled"]:
            out(raw, None, "disabled")
            continue
        p = parse_identifier(raw)
        if p is None:
            out(raw, None, "invalid")
            continue
        kind, val = p
        cand: dict = {}
        rec = None
        try:
            if kind == "arxiv":  # portal only: resolve to the journal original; never arXiv text
                if n_arxiv >= int(cfg["max_arxiv_lookups_per_call"]):
                    out(raw, kind, "arxiv_lookup_limit")
                    continue
                n_arxiv += 1
                doi = (rv.arxiv_original or (lambda a: _default_arxiv_original(a, cfg)))(val)
                doi = _journal_doi(doi)
                if not doi:
                    out(raw, kind, "arxiv_no_original")
                    continue
                cand = {"doi": doi, "followed_from": "arxiv-portal"}
            elif kind == "pmid":
                rec = (rv.pmid or (lambda x: _default_pmid(x, cfg, acq)))(val)
                if not rec or not (rec.get("pmcid") or _journal_doi(rec.get("doi"))):
                    out(raw, kind, "no_route")
                    continue
                cand = ({"pmcid": rec["pmcid"]} if rec.get("pmcid") else {}) | (
                    {"doi": _journal_doi(rec.get("doi"))} if _journal_doi(rec.get("doi")) else {})
            elif kind == "doi":
                cand = {"doi": val}
            elif kind == "pmcid":
                cand = {"pmcid": val}
            else:  # url: only an identifier the URL itself carries (DOI, PMC article) is followed; no landing fetch
                c2 = auto_acquire.follow_url({"url": val})
                if c2.get("arxiv_id") or not (c2.get("doi") or c2.get("pmcid")) or _ARXIV_DOI.match(str(c2.get("doi") or "")):
                    out(raw, kind, "no_route")
                    continue
                cand = {k: c2[k] for k in ("doi", "pmcid") if c2.get(k)} | {"url": val}
        except Exception as e:  # noqa: BLE001  transport/permission/rate failure on an identifier lookup
            logger.info("list-ingest lookup failed (%s)", type(e).__name__)
            out(raw, kind, "no_route")
            continue
        if cand.get("doi") and _ARXIV_DOI.match(cand["doi"]):  # belt and braces: never route an arXiv DOI to the pipeline
            out(raw, kind, "arxiv_no_original")
            continue
        try:  # PORTAL-RULE-1110 Q4: a portal/excluded record (DOI prefix, Europe PMC PPR source, preprint type) is never queued.
            # Only the record's own DOI, source and publication types decide; a URL item is not refused for its host.
            from vera import host_role
            role = host_role.role_for_record({
                "doi": cand.get("doi") or "", "source": (rec or {}).get("source") or "",
                "pub_types": (rec or {}).get("pub_types") or (rec or {}).get("pubTypeList") or []}).role
        except Exception:  # noqa: BLE001  fail closed
            role = "PORTAL"
        if role in ("PORTAL", "EXCLUDED"):
            out(raw, kind, "portal")
            continue
        ref = ("doi:" + cand["doi"]) if cand.get("doi") else ("pmc:" + cand["pmcid"])
        if ref in seen:
            out(raw, kind, "duplicate_in_call")
            continue
        seen.add(ref)
        try:
            if known(cand.get("doi"), cand.get("url")):
                out(raw, kind, "duplicate_stored")
                continue
        except Exception:  # noqa: BLE001  lookup unavailable: the job re-checks (known stage) before any fetch
            pass
        now = clock()
        with _LOCK:  # F3: check and reserve in one critical section; released below if nothing was queued
            while _ACCEPTED and now - _ACCEPTED[0] >= 86400.0:
                _ACCEPTED.popleft()
            hour = sum(1 for t in _ACCEPTED if now - t < 3600.0)
            if hour >= int(cfg["max_items_per_hour"]) or len(_ACCEPTED) >= int(cfg["max_items_per_day"]):
                out(raw, kind, "rate_limited")
                continue
            _ACCEPTED.append(now)
        cand["li_ref"] = ref
        if topic:
            cand["li_topic"] = True
        r = auto_acquire.enqueue_candidates(topic or QUESTION, [cand], cfg=acq, env=env, clock=clock,
                                            start_worker=start_worker, deps_factory=deps_factory,
                                            trigger=auto_acquire.LIST_TRIGGER,
                                            reserve_slots=int(cfg["queue_reserve_for_visitors"]))
        if r["queued"]:
            with _LOCK:
                _OUTCOMES[ref] = {"state": "queued", "code": None, "target": "topic" if topic else "title"}
                _OUTCOMES.move_to_end(ref)
                while len(_OUTCOMES) > int(cfg["outcomes_kept"]):
                    _OUTCOMES.popitem(last=False)
            out(raw, kind, "queued")
            continue
        with _LOCK:  # not queued: give the reserved slot back
            try:
                _ACCEPTED.remove(now)
            except ValueError:
                pass
        if r["already"] or r["status"] == "already_queued":
            out(raw, kind, "duplicate_queued")
        elif r["status"] == acq_config.reason(acq, "queue_full"):
            out(raw, kind, "queue_full")
        elif r["status"] == acq_config.reason(acq, "rate"):
            out(raw, kind, "rate_limited")
        else:
            out(raw, kind, "disabled")
    return results


def record_outcome(cands, job_result: Mapping) -> None:
    """Called by auto_acquire after a list job: the item's final state, from the job's reason codes (memory only)."""
    ref = ((cands or [{}])[0] or {}).get("li_ref")
    if not ref:
        return
    codes = dict((job_result or {}).get("codes") or {})
    stored = int((job_result or {}).get("stored") or 0)
    final = "stored" if stored else (str((job_result or {}).get("last_code") or "") or (sorted(codes)[0] if codes else "no-result"))
    with _LOCK:
        _OUTCOMES[ref] = {**_OUTCOMES.get(ref, {}), "state": "done", "code": final, "stored": stored}
        _OUTCOMES.move_to_end(ref)


def outcomes(limit: int = 200) -> list:
    with _LOCK:
        return [{"ref": k, **v} for k, v in list(_OUTCOMES.items())[-max(1, int(limit)):]]
