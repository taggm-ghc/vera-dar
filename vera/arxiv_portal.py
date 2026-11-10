"""ARXIV-PORTAL-1110: arXiv as a PORTAL to original sources, never a source.

Owner, 2026-11-10, verbatim: "arxiv is a portal to sources, not a source, itself" (and earlier, owner memory
project_reference_portals.md: prefer routes that fetch nothing from the portal itself).

An arXiv-only candidate (an arXiv id and no journal DOI) is a SEED. It is never read (vera/arxiv_fulltext.PORTAL_ONLY
closes that route in code) and ranking drops it with R-ACQ-ARXIV_PORTAL. `extend` follows each seed through metadata only:

- (a) its published version: the journal DOI INSPIRE-HEP records for the e-print;
- (b) its references: the DOIs INSPIRE records in its reference list, at most arxiv_portal.max_refs_per_record.

Each becomes an ordinary DOI candidate labelled portal_via "via arXiv <id>". It goes through the unchanged acquisition
gates on its original host (Europe PMC route, robots/TDM, rate limit, licence gate). Nothing is requested from arxiv.org.
INSPIRE calls go through vera.portal_inspire.lookup_arxiv (API allowlist entry inspire_literature, rate portal inspire),
at most arxiv_portal.max_records_per_call per call, and only when the INSPIRE portal adapter is on and permitted. A record INSPIRE does not know
(most non-physics arXiv papers) yields nothing more: the seed stays a reading-list portal entry.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Callable

logger = logging.getLogger("vera.auto_acquire")
# The arXiv family: arxiv.org and every subdomain (export., ar5iv.labs., info., national mirrors such as de./in.), plus
# known full-text mirrors and derivatives of arXiv e-prints. Never read as a source, on any route or redirect hop
# (vera/web_route.make_hop_gate, is_web_candidate). Add a host here when another mirror is found.
ARXIV_HOSTS = ("arxiv.org", "ar5iv.org", "arxiv-vanity.com", "alphaxiv.org", "xxx.lanl.gov")


def is_arxiv_host(host) -> bool:
    from vera import host_role  # S1: one normalisation (Unicode dots, fullwidth, case, trailing dot, IDNA), not a copy
    h, _path, _problem = host_role._split("//" + str(host or "").strip())
    return bool(h) and any(h == d or h.endswith("." + d) for d in ARXIV_HOSTS)
_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.I)


def is_arxiv_only_work(r) -> bool:
    """OD-1110-ARXIV-ABSTRACT (owner 2026-11-10, "No, portal only"): True for a search record whose only version is the
    arXiv e-print: an arXiv DataCite DOI (10.48550/arXiv.*), or no DOI and an arXiv-family URL or provider. A record
    carrying a journal DOI is the published work (not arXiv-only). Metadata only."""
    if not isinstance(r, dict):
        return False
    from urllib.parse import urlsplit
    from vera.search_providers import normalise_doi
    doi = normalise_doi(r.get("doi"))
    if doi:
        return bool(_ARXIV_DOI.match(doi))
    try:
        host = urlsplit(str(r.get("url") or "")).hostname or ""
    except ValueError:
        host = ""
    return is_arxiv_host(host) or str(r.get("provider") or "").lower() == "arxiv"


def cfg_of(acq: dict | None = None) -> dict:
    from vera import acq_config
    a = acq if isinstance(acq, dict) else acq_config.load_config()
    v = a.get("arxiv_portal")
    return v if isinstance(v, dict) else {}


def seed_id(c) -> str | None:
    """The bare arXiv id of an arXiv-ONLY candidate (no journal DOI, no PMCID), else None. Metadata only."""
    if not isinstance(c, dict):
        return None
    from vera import auto_acquire as aa
    from vera import work_key
    doi = aa._doi(c)
    if (doi and not _ARXIV_DOI.match(doi)) or aa._pmcid(c):
        return None
    ids = work_key.identifiers(c)
    if "doi" in ids or "pmcid" in ids:
        return None
    a = c.get("arxiv_id") or ids.get("arxiv")
    return work_key._arxiv_bare(a.get("id") if isinstance(a, dict) else a)


def is_seed(c, acq) -> bool:
    """True when ranking must drop c as an arXiv portal entry (never read). Only while the evidence route is closed."""
    from vera import arxiv_fulltext
    return arxiv_fulltext.PORTAL_ONLY and seed_id(c) is not None


_OA_CACHE: dict = {}


def _doi_of(v) -> str | None:
    from vera.search_providers import normalise_doi
    d = normalise_doi(v)
    return d if d and not _ARXIV_DOI.match(d) else None


def _preprint_prefixes() -> tuple:
    """DOI prefixes of preprint servers (config arxiv_portal.openalex.preprint_doi_prefixes), lowercased, "10.x/" form."""
    try:
        v = (cfg_of().get("openalex") or {}).get("preprint_doi_prefixes")
    except Exception:  # noqa: BLE001
        v = None
    return tuple(p.strip().lower().rstrip("/") + "/" for p in (v or ()) if isinstance(p, str) and p.strip())


_W_ID = re.compile(r"^(?:https?://openalex\.org/)?(W\d{1,12})$", re.I)
OA_REFS_BATCH_MAX = 50  # OpenAlex OR-filter values per request (structural API bound, not a decision threshold)


def _oa_get(p, params: dict, timeout_s: float) -> list:
    """One OpenAlex metadata request through the provider (budget split, allowlist, rate portal, key). -> results list.
    Raises sp.ProviderResponseError on a non-200 (429 as a quota error)."""
    from vera import search_providers as sp
    resp = p.metadata_get(params, float(timeout_s))
    st = int(getattr(resp, "status_code", 200) or 200)
    if st == 429:
        raise sp.ProviderResponseError("openalex 429 quota")
    if st != 200:
        raise sp.ProviderResponseError(f"openalex status {st}")
    doc = resp.json() if hasattr(resp, "json") else {}
    res = doc.get("results") if isinstance(doc, dict) else None
    return [r for r in res if isinstance(r, dict)] if isinstance(res, list) else []


def openalex_work(flt: str, max_refs: int, timeout_s: float, counts: dict, clock: Callable = time.monotonic,
                  provider=None, cache_ttl_s: float = 86400.0, deadline: float | None = None):
    """PORTAL-WIRE-B-1110 (owner 2026-11-10: follow references; OpenAlex treated as a PORTAL, so metadata and links
    only): at most TWO OpenAlex metadata calls for one record. Call 1: the work (filter `flt`, select
    id,doi,locations,referenced_works: identifiers and links, never an abstract). Call 2, only when max_refs > 0 and the
    work lists referenced works: ONE batched lookup of the first max_refs referenced W ids (filter openalex:W1|W2|...,
    select doi only) -> their DOIs. Through the shared /ask OpenAlex provider (OpenAlexProvider.metadata_get): each call
    is spent against its daily budget with the visitor/seed split (SEARCHCAPS-1110), the API allowlist entry openalex,
    the openalex rate portal and the optional key. The process-wide quota cooldown is honoured and started only by a
    PROVIDER quota error, never by VERA-DAR's own refusal (review D1). Nothing is retried; a failed call 2 keeps call 1's
    result with no references. -> {"doi", "locations": [url], "refs": [doi]} or None. Never raises."""
    from vera import search_providers as sp
    now = clock()
    key = (flt, int(max_refs))
    hit = _OA_CACHE.get(key)
    if hit is not None and now - hit[0] < cache_ttl_s:
        counts["oa_cache_hit"] = counts.get("oa_cache_hit", 0) + 1
        return hit[1]
    if sp._QUOTA.blocked(now):
        counts["oa_quota_cooldown"] = counts.get("oa_quota_cooldown", 0) + 1
        return None
    if timeout_s <= 0.5:
        counts["oa_deadline"] = counts.get("oa_deadline", 0) + 1
        return None
    p = provider
    if p is None:
        from vera import search_and_fetch
        p = next((x for x in search_and_fetch.cached_chain()[0] if getattr(x, "name", None) == "openalex"), None)
    if p is None or not hasattr(p, "metadata_get"):
        counts["oa_no_provider"] = counts.get("oa_no_provider", 0) + 1
        return None

    def _fail(e) -> None:
        own_why = sp._own_decision(e)
        if own_why is None and sp._is_quota_error(e) and not sp._is_rate_skip(e):  # D1: a provider quota only
            from vera import reference_portal
            sp._QUOTA.block(clock(), float(reference_portal._openalex_cooldown()))
            counts["oa_429"] = counts.get("oa_429", 0) + 1
        elif own_why is not None:
            counts["oa_own_" + own_why] = counts.get("oa_own_" + own_why, 0) + 1
        else:
            counts["oa_failed"] = counts.get("oa_failed", 0) + 1

    try:
        counts["oa_calls"] = counts.get("oa_calls", 0) + 1
        works = _oa_get(p, {"filter": flt, "per_page": 1, "select": "id,doi,locations,referenced_works"}, timeout_s)
    except Exception as e:  # noqa: BLE001  never retried
        _fail(e)
        return None
    if not works:
        got = {"doi": None, "locations": [], "published": [], "refs": []}
    else:
        w = works[0]
        locs = []
        for x in w.get("locations") or ():
            if isinstance(x, dict):
                for k in ("landing_page_url", "pdf_url"):
                    u = str(x.get(k) or "").strip()
                    if u.startswith(("https://", "http://")) and u not in locs:
                        locs.append(u)
        # S2 (review round 3): the published version is only a location OpenAlex marks version == "publishedVersion";
        # the work DOI itself is the arXiv DOI (or a preprint's) and is never taken as published. Preprint-server DOI
        # prefixes (config) are excluded on top.
        pre = _preprint_prefixes()
        pub = [d for d in (_doi_of(x.get("landing_page_url")) for x in (w.get("locations") or ())
                           if isinstance(x, dict) and x.get("version") == "publishedVersion")
               if d and not d.lower().startswith(pre)]
        got = {"doi": _any_doi(w.get("doi")), "locations": locs, "published": list(dict.fromkeys(pub))[:1], "refs": []}
        wids = []
        for r in w.get("referenced_works") or ():
            m = _W_ID.match(str(r or "").strip())
            if m and m.group(1).upper() not in wids:
                wids.append(m.group(1).upper())
        wids = wids[: max(0, min(int(max_refs), OA_REFS_BATCH_MAX))]
        left = timeout_s if deadline is None else min(timeout_s, deadline - clock())
        if wids and left > 0.5 and not sp._QUOTA.blocked(clock()):
            try:
                counts["oa_calls"] = counts.get("oa_calls", 0) + 1
                counts["oa_ref_calls"] = counts.get("oa_ref_calls", 0) + 1
                refs = _oa_get(p, {"filter": "openalex:" + "|".join(wids), "per_page": len(wids), "select": "doi"}, left)
                got["refs"] = list(dict.fromkeys(d for d in (_any_doi(r.get("doi")) for r in refs) if d))[: len(wids)]
            except Exception as e:  # noqa: BLE001  never retried; call 1's result stands
                _fail(e)
    _OA_CACHE[key] = (now, got)
    while len(_OA_CACHE) > 512:
        _OA_CACHE.pop(next(iter(_OA_CACHE)))
    return got


def _any_doi(v) -> str | None:
    from vera.search_providers import normalise_doi
    return normalise_doi(v) or None


def openalex_lookup(aid: str, max_refs: int, timeout_s: float, counts: dict, clock: Callable = time.monotonic,
                    provider=None, cache_ttl_s: float = 86400.0):
    """OD-1110-ARXIV-OPENALEX, widened by PORTAL-WIRE-B-1110 (owner wants references followed: two calls per record):
    for an arXiv record INSPIRE does not know, openalex_work on the work 10.48550/arXiv.<id>. Its published version is
    a doi.org landing page of a location OpenAlex marks publishedVersion (S2; never the work DOI); its references are the DOIs of its referenced works
    (call 2, at most max_refs). -> ([published-version DOI], [reference DOIs]) or None. Never raises."""
    got = openalex_work(f"doi:https://doi.org/10.48550/arxiv.{aid}", max_refs, timeout_s, counts, clock=clock,
                        provider=provider, cache_ttl_s=cache_ttl_s)
    if got is None:
        return None
    own = [d for d in (_doi_of(x) for x in got.get("published") or ()) if d][:1]  # S2: publishedVersion locations only
    refs = [d for d in (_doi_of(x) for x in got.get("refs") or ()) if d and d not in own][: max(0, int(max_refs))]
    return own, refs


def reset_for_tests() -> None:
    _OA_CACHE.clear()


def _cand(doi: str, aid: str, kind: str, rank: int) -> dict:
    return {"doi": doi, "url": f"https://doi.org/{doi}", "title": "", "snippet": "", "provider": "portal:arxiv",
            "acq_source": "arxiv_portal", "portal": "arxiv", "portal_via": f"via arXiv {aid}", "portal_kind": kind,
            "portal_rank": rank}


def switches(acq: dict | None = None) -> dict | None:
    """The lookup switches shared by extend and vera/portal_follow: None when the arXiv portal is off, else
    {pc, rcfg, insp, oa_on}. INSPIRE needs its reference-portal adapter on and permitted; OpenAlex needs
    arxiv_portal.openalex.enabled (ships off pending the terms review). Both need reference_portals mode "on"."""
    pc = cfg_of(acq)
    from vera import arxiv_fulltext, reference_portal
    if pc.get("enabled") is not True or not arxiv_fulltext.PORTAL_ONLY:
        return None
    rcfg = reference_portal.load_config()
    insp = (rcfg.get("mode") == "on" and reference_portal.adapter_on(rcfg, "inspire_refs")
            and reference_portal.inspire_permitted())
    # the reference-portal subsystem's master mode governs both lookups (off = no portal request at all)
    oa_on = rcfg.get("mode") == "on" and (pc.get("openalex") or {}).get("enabled") is True
    return {"pc": pc, "rcfg": rcfg, "insp": bool(insp), "oa_on": bool(oa_on)}


def lookup_id(aid: str, sw: dict, counts: dict, *, clock: Callable = time.monotonic, deadline: float | None = None,
              get: Callable | None = None, openalex_provider=None, max_refs: int | None = None):
    """One arXiv id -> ((published-version DOIs, reference DOIs), via) or (None, via): INSPIRE first, then OpenAlex
    when INSPIRE does not know it (or is off). Metadata only; nothing is requested from arxiv.org."""
    pc, rcfg = sw["pc"], sw["rcfg"]
    mr = int(pc.get("max_refs_per_record", 10)) if max_refs is None else int(max_refs)
    left = float(rcfg["caps"]["timeout_s"]) if deadline is None else min(float(rcfg["caps"]["timeout_s"]),
                                                                          deadline - clock())
    got, via = None, "inspire"
    if sw["insp"]:
        from vera import portal_inspire
        icfg = rcfg["adapters"]["inspire_refs"]
        got = portal_inspire.lookup_arxiv(aid, str(icfg["endpoint"]), mr, left, counts, clock=clock, get=get,
                                          cache_ttl_s=float(icfg["cache_ttl_s"]),
                                          max_bytes=int(icfg["max_response_bytes"]))
    if not (got and (got[0] or got[1])) and sw["oa_on"]:  # INSPIRE does not know it (or is off): OpenAlex
        left = left if deadline is None else min(left, deadline - clock())
        got = openalex_lookup(aid, mr, left, counts, clock=clock, provider=openalex_provider,
                              cache_ttl_s=float((pc.get("openalex") or {}).get("cache_ttl_s", 86400)))
        via = "openalex"
    return got, via


def extend(question: str, cands: list, acq: dict | None = None, *, clock: Callable = time.monotonic,
           deadline: float | None = None, get: Callable | None = None, openalex_provider=None) -> list:
    """cands + the published-version and reference candidates of every arXiv seed (seeds stay in the list; ranking
    drops them as portal entries), then (PORTAL-WIRE-B-1110) the original-host hand-offs vera/portal_follow finds for
    the question's most relevant other portal records. INSPIRE first (when its portal adapter is on and permitted); a
    seed INSPIRE does not know gets the OpenAlex lookup (at most two calls) when arxiv_portal.openalex is on
    (OD-1110-ARXIV-OPENALEX). Both need reference_portals mode "on". Never raises; any failure returns cands unchanged."""
    out = _extend_seeds(question, cands, acq, clock=clock, deadline=deadline, get=get,
                        openalex_provider=openalex_provider)
    try:
        from vera import portal_follow
        return portal_follow.extend(question, out, acq, clock=clock, deadline=deadline, get=get,
                                    openalex_provider=openalex_provider)
    except Exception:  # noqa: BLE001  the portal never breaks the search
        logger.warning("autoacq portalfollow skipped")
        return out


def _extend_seeds(question: str, cands: list, acq: dict | None = None, *, clock: Callable = time.monotonic,
                  deadline: float | None = None, get: Callable | None = None, openalex_provider=None) -> list:
    try:
        sw = switches(acq)
        if sw is None or (not sw["insp"] and not sw["oa_on"]):
            return list(cands or ())
        pc = sw["pc"]
        seeds: list = []
        for c in cands or ():
            aid = seed_id(c)
            if aid and aid not in seeds:
                seeds.append(aid)
        counts: dict = {"seeds": len(seeds)}
        out = list(cands or ())
        have = {str(c.get("doi") or "").lower() for c in out if isinstance(c, dict)}
        for aid in seeds[: int(pc.get("max_records_per_call", 3))]:
            got, via = lookup_id(aid, sw, counts, clock=clock, deadline=deadline, get=get,
                                 openalex_provider=openalex_provider)
            if not got:
                continue
            own, refs = got
            for kind, dois in (("published_version", own), ("reference", refs)):
                for i, d in enumerate(dois):
                    if d.lower() in have:
                        continue
                    have.add(d.lower())
                    c = _cand(d, aid, kind, i)
                    c["portal_lookup"] = via
                    out.append(c)
                    counts[f"{kind}_{via}"] = counts.get(f"{kind}_{via}", 0) + 1
        logger.info("autoacq arxivportal %s", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        return out
    except Exception:  # noqa: BLE001  the portal never breaks the search
        logger.warning("autoacq arxivportal skipped")
        return list(cands or ())
