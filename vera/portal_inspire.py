"""ROUTES-1108 INSPIRE batch: INSPIRE-HEP as a REFERENCE PORTAL adapter of vera/reference_portal.py.

Owner decision OD-1108-SRC (2026-11-08) item 2: the INSPIRE-HEP API is CLEARED for metadata and references only, at
15 requests / 5 s, CC0 metadata, with email fields dropped. Terms (GENSCI-SOURCES-1108 I1-I3, summarizer reads):
abstracts reusable only when their source is arXiv or CERN (none are read here), "reconstructing the fulltexts of
articles from snippets is not allowed" (no snippet is read here), "the download of email addresses is not permitted"
(no email field is requested or read).

One literature search per question (config adapters.inspire_refs): GET <endpoint>?q=<question content terms>&size=..
&sort=..&fields=<identifier fields only>. Every request goes through vera.api_permissions.call (entry
inspire_literature: deny by default, owner permit OD-1108-SRC, rate portal inspire, purpose metadata). If the API
allowlist module is not present the adapter sends NOTHING (fail closed). The parser reads only DOI and arXiv
identifier values of each hit and of each hit's references; every other field is ignored. Candidates are ordinary
acquisition candidates (vera.reference_portal), full text only through a permitted route (arXiv CC, SCOAP3, ...).

UNVERIFIED until one response is recorded through the permitted route: the field paths dois[].value,
arxiv_eprints[].value, references[].reference.dois[] and references[].reference.arxiv_eprint (INSPIRE REST docs, not
read raw). A response of any other shape yields no candidates (fail closed), never an error.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Callable, Mapping

ADAPTER = "inspire_refs"
_ENTRY_HOST = "inspirehep.net"
_COOL = {"until": 0.0}
_CACHE: dict = {}
_LOCK = threading.Lock()


def _query(question: str, acfg: Mapping) -> str:
    from vera import reference_portal as rp, wiki_ref_portal as w
    q = w.build_query(question, {"query": rp._wiki_query(), "relevance": {"cue_terms": []}})
    words = q.split()[: int(acfg["max_query_terms"])]
    return " ".join(words) if words and words != str(question or "").split() else ""


def _default_get(url: str, params: dict, timeout_s: float):
    """The gated call: vera.api_permissions.call (entry, conditions, rate portal, UA) over the pinned public-only GET.
    Raises ImportError when the API allowlist is not shipped (nothing is sent)."""
    from vera import api_permissions  # noqa: F401  (ImportError -> fail closed)
    from vera import selective_fetch as sf
    # pinned_get re-gates every hop under the explicit purpose (1108b: purpose_scope binds it to this url). call() takes the
    # entry's rate portal; holding() tells pinned_get it is already held, so the portal is not taken twice (concurrency skip).
    with api_permissions.purpose_scope("metadata", url), api_permissions.holding("inspire"):
        return api_permissions.call(sf.pinned_get, url, params=params, timeout=timeout_s, purpose="metadata",
                                    allow_redirects=False)


def _ids_of(values, kind: str) -> list:
    from vera import wiki_ref_portal as w
    norm = w.norm_doi if kind == "doi" else w.norm_arxiv
    out = []
    for v in values if isinstance(values, list) else ([values] if isinstance(values, str) else []):
        if isinstance(v, Mapping):
            v = v.get("value")
        n = norm(v) if isinstance(v, str) else None
        if n:
            out.append({kind: n})
    return out


def parse(body: bytes, acfg: Mapping) -> list:
    """INSPIRE JSON -> [{"doi": ..} | {"arxiv": ..}] in hit order: each hit's own identifiers, then its references
    (capped per hit). Only identifier values are read. Any shape error -> [] (fail closed)."""
    try:
        doc = json.loads(body.decode("utf-8"))
        hits = doc["hits"]["hits"]
        if not isinstance(hits, list):
            return []
    except Exception:  # noqa: BLE001
        return []
    out = []
    per = int(acfg["max_refs_per_seed"])
    for h in hits[: int(acfg["size"])]:
        md = h.get("metadata") if isinstance(h, Mapping) else None
        if not isinstance(md, Mapping):
            continue
        # ARXIV-PORTAL-1110: a hit's journal DOI when it has one, else its arXiv id (then an arXiv portal seed, followed
        # by vera/arxiv_portal.py); the same DOI-first rule as for references
        out += _ids_of(md.get("dois"), "doi")[:1] or _ids_of(md.get("arxiv_eprints"), "arxiv")[:1]
        n = 0
        for r in md.get("references") or () if isinstance(md.get("references"), list) else ():
            ref = r.get("reference") if isinstance(r, Mapping) else None
            if not isinstance(ref, Mapping) or n >= per:
                continue
            got = _ids_of(ref.get("dois"), "doi")[:1] or _ids_of(ref.get("arxiv_eprint"), "arxiv")[:1]
            if got:
                out += got
                n += 1
    return out


def adapter(question: str, base, cfg: Mapping, deps, counts: dict, clock: Callable = time.monotonic,
            deadline: float | None = None, get: Callable | None = None) -> list:
    """-> [candidates]. ONE API call per question at most; cached; a 429 starts a cooldown (Retry-After, at least
    cooldown_min_s) and is never retried in the question. Never raises."""
    from vera import wiki_ref_portal as w
    acfg = cfg["adapters"][ADAPTER]
    q = _query(question, acfg)
    if not q:
        return []
    now = clock()
    with _LOCK:
        hit = _CACHE.get(q)
    if hit is not None and now - hit[0] < float(acfg["cache_ttl_s"]):
        counts["inspire_cache_hit"] = 1
        ids = hit[1]
    else:
        if now < _COOL["until"]:
            counts["inspire_cooldown"] = 1
            return []
        left = float(cfg["caps"]["timeout_s"]) if deadline is None else min(float(cfg["caps"]["timeout_s"]),
                                                                              deadline - clock())
        if left <= 0.5:
            counts["inspire_deadline"] = 1
            return []
        params = {"q": q, "size": int(acfg["size"]), "sort": str(acfg["sort"]), "fields": ",".join(acfg["fields"])}
        counts["inspire_calls"] = 1
        resp = None
        try:
            resp = (get or _default_get)(str(acfg["endpoint"]), params, left)
            st = int(getattr(resp, "status_code", 0) or 0)
            if st == 429:
                ra = (getattr(resp, "headers", None) or {}).get("Retry-After")
                try:
                    wait = max(float(acfg["cooldown_min_s"]), float(ra))
                except (TypeError, ValueError):
                    wait = float(acfg["cooldown_min_s"])
                _COOL["until"] = clock() + wait
                counts["inspire_429"] = 1
                return []
            if st != 200:
                counts["inspire_failed"] = 1
                return []
            body = b""
            for chunk in resp.iter_content(65536):
                body += chunk
                if len(body) > int(acfg["max_response_bytes"]):
                    counts["inspire_too_large"] = 1
                    return []
            ids = parse(body, acfg)
        except ImportError:
            counts["inspire_api_gate_missing"] = 1  # API allowlist not shipped: nothing was sent
            return []
        except Exception as e:  # noqa: BLE001  ApiDenied, RateLimited, transport errors: no retry
            counts["inspire_" + (getattr(e, "code", None) or "failed").lower().replace("-", "_")] = 1
            return []
        finally:
            try:
                if resp is not None and hasattr(resp, "close"):
                    resp.close()
            except Exception:  # noqa: BLE001
                pass
        with _LOCK:
            _CACHE[q] = (now, ids)
            while len(_CACHE) > int(acfg["cache_max"]):
                _CACHE.pop(next(iter(_CACHE)))
    out = []
    for i in ids:
        c = w.to_candidate(i, len(out))
        if c is None:
            continue
        c.update({"provider": f"portal:{ADAPTER}", "acq_source": "reference_portal", "portal": ADAPTER,
                  "title": "", "snippet": ""})
        out.append(c)
    counts["inspire_ids"] = len(out)
    return out


ARXIV_FIELDS = ("dois.value", "references.reference.dois")
_ARX_CACHE: dict = {}


def parse_arxiv_record(body: bytes, max_refs: int) -> tuple[list, list]:
    """INSPIRE JSON for ONE arXiv e-print -> ([journal DOIs], [reference DOIs]) (DOIs only; arXiv-only references are
    not followed: they would be further portals). Any shape error -> ([], []) (fail closed)."""
    try:
        hits = json.loads(body.decode("utf-8"))["hits"]["hits"]
        md = hits[0]["metadata"] if isinstance(hits, list) and hits and isinstance(hits[0], Mapping) else None
        if not isinstance(md, Mapping):
            return [], []
    except Exception:  # noqa: BLE001
        return [], []
    own = [i["doi"] for i in _ids_of(md.get("dois"), "doi")[:1]]
    refs: list = []
    for r in md.get("references") if isinstance(md.get("references"), list) else ():
        ref = r.get("reference") if isinstance(r, Mapping) else None
        if len(refs) >= max_refs or not isinstance(ref, Mapping):
            continue
        refs += [i["doi"] for i in _ids_of(ref.get("dois"), "doi")[:1] if i["doi"] not in refs and i["doi"] not in own]
    return own, refs[:max_refs]


def lookup_arxiv(aid: str, endpoint: str, max_refs: int, timeout_s: float, counts: dict, clock: Callable = time.monotonic,
                 get: Callable | None = None, cache_ttl_s: float = 86400.0, max_bytes: int = 4000000):
    """ARXIV-PORTAL-1110: ONE INSPIRE metadata call for an arXiv e-print (q=arxiv:<id>, identifier fields only) ->
    ([journal DOIs], [reference DOIs]) or None when nothing was obtained. Through the same gate as the portal adapter
    (_default_get: api_permissions inspire_literature, purpose metadata, rate portal inspire); cached; a 429 starts the
    shared cooldown (at least 5 s) and is never retried. Never raises."""
    now = clock()
    with _LOCK:
        hit = _ARX_CACHE.get(aid)
    if hit is not None and now - hit[0] < cache_ttl_s:
        counts["arxivportal_cache_hit"] = counts.get("arxivportal_cache_hit", 0) + 1
        return hit[1]
    if now < _COOL["until"]:
        counts["arxivportal_cooldown"] = counts.get("arxivportal_cooldown", 0) + 1
        return None
    if timeout_s <= 0.5:
        counts["arxivportal_deadline"] = counts.get("arxivportal_deadline", 0) + 1
        return None
    params = {"q": f"arxiv:{aid}", "size": 1, "fields": ",".join(ARXIV_FIELDS)}
    counts["arxivportal_calls"] = counts.get("arxivportal_calls", 0) + 1
    resp = None
    try:
        resp = (get or _default_get)(endpoint, params, timeout_s)
        st = int(getattr(resp, "status_code", 0) or 0)
        if st == 429:
            ra = (getattr(resp, "headers", None) or {}).get("Retry-After")
            try:
                wait = max(5.0, float(ra))
            except (TypeError, ValueError):
                wait = 5.0
            _COOL["until"] = clock() + wait
            counts["arxivportal_429"] = counts.get("arxivportal_429", 0) + 1
            return None
        if st != 200:
            counts["arxivportal_failed"] = counts.get("arxivportal_failed", 0) + 1
            return None
        body = b""
        for chunk in resp.iter_content(65536):
            body += chunk
            if len(body) > max_bytes:
                counts["arxivportal_too_large"] = counts.get("arxivportal_too_large", 0) + 1
                return None
        got = parse_arxiv_record(body, max_refs)
    except Exception as e:  # noqa: BLE001  ApiDenied, RateLimited, ImportError, transport errors: no retry, nothing more
        k = "arxivportal_" + (getattr(e, "code", None) or type(e).__name__).lower().replace("-", "_")
        counts[k] = counts.get(k, 0) + 1
        return None
    finally:
        try:
            if resp is not None and hasattr(resp, "close"):
                resp.close()
        except Exception:  # noqa: BLE001
            pass
    with _LOCK:
        _ARX_CACHE[aid] = (now, got)
        while len(_ARX_CACHE) > 512:
            _ARX_CACHE.pop(next(iter(_ARX_CACHE)))
    return got


def reset_for_tests() -> None:
    with _LOCK:
        _CACHE.clear()
        _ARX_CACHE.clear()
    _COOL["until"] = 0.0
