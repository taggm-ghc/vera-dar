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
    h = str(host or "").strip().lower().rstrip(".")
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


def _cand(doi: str, aid: str, kind: str, rank: int) -> dict:
    return {"doi": doi, "url": f"https://doi.org/{doi}", "title": "", "snippet": "", "provider": "portal:arxiv",
            "acq_source": "arxiv_portal", "portal": "arxiv", "portal_via": f"via arXiv {aid}", "portal_kind": kind,
            "portal_rank": rank}


def extend(question: str, cands: list, acq: dict | None = None, *, clock: Callable = time.monotonic,
           deadline: float | None = None, get: Callable | None = None) -> list:
    """cands + the published-version and reference candidates of every arXiv seed (seeds stay in the list; ranking
    drops them as portal entries). Never raises; any failure returns cands unchanged."""
    try:
        pc = cfg_of(acq)
        from vera import arxiv_fulltext, reference_portal
        if pc.get("enabled") is not True or not arxiv_fulltext.PORTAL_ONLY or not reference_portal.inspire_permitted():
            return list(cands or ())
        from vera import portal_inspire
        rcfg = reference_portal.load_config()
        if rcfg.get("mode") != "on" or not reference_portal.adapter_on(rcfg, "inspire_refs"):
            return list(cands or ())  # the same INSPIRE switches as the portal adapter (OD-1108-DISPOSITIONS-2 H9 (b))
        icfg = rcfg["adapters"]["inspire_refs"]
        seeds: list = []
        for c in cands or ():
            aid = seed_id(c)
            if aid and aid not in seeds:
                seeds.append(aid)
        counts: dict = {"seeds": len(seeds)}
        out = list(cands or ())
        have = {str(c.get("doi") or "").lower() for c in out if isinstance(c, dict)}
        for aid in seeds[: int(pc.get("max_records_per_call", 3))]:
            left = float(rcfg["caps"]["timeout_s"]) if deadline is None else min(float(rcfg["caps"]["timeout_s"]),
                                                                                  deadline - clock())
            got = portal_inspire.lookup_arxiv(aid, str(icfg["endpoint"]), int(pc.get("max_refs_per_record", 10)), left,
                                              counts, clock=clock, get=get, cache_ttl_s=float(icfg["cache_ttl_s"]),
                                              max_bytes=int(icfg["max_response_bytes"]))
            if not got:
                continue
            own, refs = got
            for kind, dois in (("published_version", own), ("reference", refs)):
                for i, d in enumerate(dois):
                    if d.lower() in have:
                        continue
                    have.add(d.lower())
                    out.append(_cand(d, aid, kind, i))
                    counts[kind] = counts.get(kind, 0) + 1
        logger.info("autoacq arxivportal %s", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        return out
    except Exception:  # noqa: BLE001  the portal never breaks the search
        logger.warning("autoacq arxivportal skipped")
        return list(cands or ())
