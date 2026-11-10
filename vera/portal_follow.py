"""PORTAL-WIRE-B-1110: follow promising portal records to the ORIGINAL host of the original source document.

Owner, 2026-11-10, verbatim: "That sample illustrates a result from a portal through which we can links to original
sources. For promising sources, we should follow links recursively until we get to the original host of the original
source doc". OD-1110-PORTALS-ALL: portals (OpenAlex, INSPIRE, Wikipedia, search engines, preprint servers) are never
evidence; their abstracts and snippets are used here only to judge which records are promising.

For the question's top-k portal records (relevance-ranked; config/portal_follow.json) a bounded walk runs over links
and identifiers:

  record -> DOI / URL / arXiv id / OpenAlex W id
  URL    -> host_role.is_hop_allowed(url) first. ORIGINAL or REPOSITORY_OF_RECORD: hand off (stop).
            RESOLVER (doi.org, handle): its identifier. PORTAL: never fetched; only an identifier in the link is
            followed (portal -> portal chain through metadata). EXCLUDED, unknown host, malformed: dead end.
  DOI    -> a portal DOI (host_roles doi_prefix_roles: arXiv, bioRxiv, SSRN, ...): arXiv DOIs become arXiv ids, other
            preprint DOIs are dead ends (no route yet: plan D-b). Any other DOI: with an OpenAlex lookup available,
            its locations (landing/pdf URLs, each classified as above) and, for a top-k record, its references;
            without one, the DOI itself is handed off as an identifier (the acquisition routes resolve it with their
            own per-hop host checks).
  arXiv  -> vera/arxiv_portal.lookup_id (INSPIRE, then OpenAlex): published-version DOI and reference DOIs.
  W id   -> OpenAlex work (two calls at most: the work, one batched reference lookup).

Bounds (config): records per question, hops per record (default 4), hand-offs per record, references per record,
metadata lookups per question, and a time budget. A visited set per question detects cycles. Hand-offs are ordinary
DOI/URL candidates (labelled portal_via) that then pass the unchanged acquisition gates: licence, relevance,
robots/TDM, rate limits and the route allowlists. Nothing here reads page text or touches the database.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import Callable

logger = logging.getLogger("vera.auto_acquire")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "portal_follow.json"
DEFAULTS = {"enabled": False, "max_records_per_question": 3, "max_hops_per_record": 4, "max_handoffs_per_record": 11,
            "max_refs_per_record": 10, "max_lookups_per_question": 6, "deadline_s": 6.0, "min_relevance_terms": 1,
            "portal_providers": ["openalex", "duckduckgo_lite", "duckduckgo", "arxiv", "inspire", "wikipedia",
                                 "semantic_scholar", "crossref", "doaj"]}
_INT_KEYS = ("max_records_per_question", "max_hops_per_record", "max_handoffs_per_record", "max_refs_per_record",
             "max_lookups_per_question", "min_relevance_terms")
_EVIDENCE = ("ORIGINAL", "REPOSITORY_OF_RECORD")
_ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf|html)/([a-z\-]+(?:\.[a-z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.I)
_DOI_IN = re.compile(r"(?:doi\.org/|doi:\s*)(10\.\d{4,9}/[^\s\"'<>&#?]+)", re.I)
_W_IN = re.compile(r"openalex\.org/(?:works/)?(W\d{1,12})\b", re.I)
_WORD = re.compile(r"[a-z][a-z0-9]{3,}")
_STOP = frozenset("what which when where does have that this with from about into their there these those were been "
                  "being they them than then also only over such more most some very will would could should".split())


def load_config(path=None) -> dict:
    """Defaults merged with the file; an unreadable or invalid file turns the follow-up OFF (fail closed)."""
    try:
        raw = json.loads(Path(path or CONFIG_PATH).read_text())
        if not isinstance(raw, dict):
            raise ValueError("not an object")
        cfg = {**DEFAULTS, **{k: v for k, v in raw.items() if not k.startswith("_")}}
        for k in _INT_KEYS:
            if isinstance(cfg[k], bool) or int(cfg[k]) < 0:
                raise ValueError(k)
            cfg[k] = int(cfg[k])
        cfg["deadline_s"] = float(cfg["deadline_s"])
        cfg["enabled"] = cfg.get("enabled") is True
        cfg["portal_providers"] = [str(x).lower() for x in cfg.get("portal_providers") or ()]
        return cfg
    except Exception:  # noqa: BLE001
        return {**DEFAULTS, "enabled": False}


# ----------------------------------------------------------------------------------------------------- selection
def _terms(text) -> set:
    return {w for w in _WORD.findall(str(text or "").lower()) if w not in _STOP}


def relevance(question: str, rec: dict) -> int:
    """Distinct content words of the question in the record's title or snippet (lexical; the snippet is used for this
    judgement only, never as evidence)."""
    return len(_terms(question) & _terms(f"{rec.get('title') or ''} {rec.get('snippet') or ''}"))


def is_portal_record(rec, cfg: dict) -> bool:
    """A record found through a portal: its provider is a portal service, or its own role (DOI prefix, source flag,
    URL host; vera/host_role.py) is PORTAL. Stored chunks and portal_follow / arxiv_portal hand-offs are not."""
    if not isinstance(rec, dict) or isinstance(rec.get("dar_span"), dict) or rec.get("portal_via"):
        return False
    prov = str(rec.get("provider") or "").lower()
    if prov in cfg["portal_providers"] or prov.startswith("portal:"):
        return True
    from vera import host_role
    return host_role.role_for_record({k: rec.get(k) for k in ("doi", "url", "source", "type", "source_type")
                                      if rec.get(k)}).role == "PORTAL"


def select(question: str, records, cfg: dict) -> list:
    """The promising portal records: relevance-ranked (stable: search order breaks ties), at least
    min_relevance_terms shared content words when the question has any, at most max_records_per_question."""
    has_terms = bool(_terms(question))
    scored = []
    for i, r in enumerate(records or ()):
        if not is_portal_record(r, cfg):
            continue
        rel = relevance(question, r)
        if has_terms and rel < cfg["min_relevance_terms"]:
            continue
        scored.append((-rel, i, r))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in scored[: cfg["max_records_per_question"]]]


# ----------------------------------------------------------------------------------------------------- the walk
def _norm_doi(v) -> str | None:
    from vera.search_providers import normalise_doi
    return normalise_doi(v) or None


def ids_in_link(url: str) -> list:
    """Identifiers carried by a link itself (no fetch): [("arxiv", id) | ("doi", doi) | ("openalex", W id)]."""
    out, u = [], str(url or "")
    m = _ARXIV_ID.search(u)
    if m:
        out.append(("arxiv", m.group(1).lower()))
    m = _DOI_IN.search(u)
    if m and _norm_doi(m.group(1)):
        out.append(("doi", _norm_doi(m.group(1))))
    m = _W_IN.search(u)
    if m:
        out.append(("openalex", m.group(1).upper()))
    return out


def _roots(rec: dict) -> list:
    """Start nodes of one record: its DOI, arXiv id, OpenAlex W id and URL (identifiers first)."""
    out = []
    d = _norm_doi(rec.get("doi"))
    if d:
        out.append(("doi", d))
    a = rec.get("arxiv_id")
    a = a.get("id") if isinstance(a, dict) else a
    if a:
        out.append(("arxiv", str(a).lower()))
    w = _W_IN.search(str(rec.get("openalex_id") or ""))
    if w:
        out.append(("openalex", w.group(1).upper()))
    if rec.get("url"):
        out.append(("url", str(rec["url"]).strip()))
    return out


class Lookups:
    """The metadata expanders (injectable for tests). arxiv(aid, counts, deadline, max_refs) -> ((own, refs), via) |
    (None, via); openalex(flt, counts, deadline, max_refs) -> {"doi", "locations", "refs"} | None, or None when no
    OpenAlex lookup is allowed."""

    def __init__(self, arxiv: Callable | None, openalex: Callable | None, cost: Callable | None = None):
        self.arxiv, self.openalex = arxiv, openalex
        # cost(kind, refs_on) -> worst-case HTTP calls of one lookup (kind "arxiv" | "openalex"); default: OpenAlex 1 (+1 for
        # the references batch), arXiv 1
        self.cost = cost or (lambda kind, refs_on: (1 + int(refs_on)) if kind == "openalex" else 1)


def http_calls(counts: dict) -> int:
    """Actual metadata HTTP requests counted so far: OpenAlex (oa_calls, both calls of a work) + the INSPIRE arXiv lookup
    (arxivportal_calls). Cache hits and cooldown skips are not requests and are not counted."""
    return int(counts.get("oa_calls", 0)) + int(counts.get("arxivportal_calls", 0))


def default_lookups(acq=None, *, clock: Callable = time.monotonic, get=None, openalex_provider=None) -> Lookups:
    """INSPIRE / OpenAlex through vera/arxiv_portal's switches (reference_portals mode, the INSPIRE adapter, and
    arxiv_portal.openalex.enabled, which ships off pending the OpenAlex terms review)."""
    from vera import arxiv_portal
    sw = arxiv_portal.switches(acq)
    if sw is None:
        return Lookups(None, None)
    ax = None
    if sw["insp"] or sw["oa_on"]:
        def ax(aid, counts, deadline, max_refs):
            return arxiv_portal.lookup_id(aid, sw, counts, clock=clock, deadline=deadline, get=get,
                                          openalex_provider=openalex_provider, max_refs=max_refs)
    oa = None
    if sw["oa_on"]:
        ttl = float((sw["pc"].get("openalex") or {}).get("cache_ttl_s", 86400))

        def oa(flt, counts, deadline, max_refs):
            left = float(sw["rcfg"]["caps"]["timeout_s"])
            if deadline is not None:
                left = min(left, deadline - clock())
            return arxiv_portal.openalex_work(flt, max_refs, left, counts, clock=clock, provider=openalex_provider,
                                              cache_ttl_s=ttl, deadline=deadline)
    def cost(kind, refs_on):
        oa_cost = 1 + int(refs_on)
        if kind == "openalex":
            return oa_cost
        return int(bool(sw["insp"])) + (oa_cost if sw["oa_on"] else 0)  # INSPIRE, then OpenAlex when it does not know it
    return Lookups(ax, oa, cost)


def _bump(counts: dict, k: str, n: int = 1) -> None:
    counts[k] = counts.get(k, 0) + n


def follow(question: str, records, cfg: dict | None = None, *, lookups: Lookups | None = None,
           clock: Callable = time.monotonic, deadline: float | None = None, have=None, counts: dict | None = None):
    """-> (hand-off candidates, counts). records: the question's search/portal records; have: DOIs/URLs (lower-case)
    already candidates (never handed off twice). Never raises; a failure returns what was found so far."""
    cfg = cfg or load_config()
    counts = {} if counts is None else counts
    out: list = []
    if not cfg["enabled"]:
        return out, counts
    from vera import host_role
    lookups = lookups or Lookups(None, None)
    end = clock() + cfg["deadline_s"]
    end = end if deadline is None else min(end, deadline)
    have = set(have or ())
    seen: set = set()  # per question: cycle detection and de-duplication across records
    base_calls = http_calls(counts)  # the cap counts actual HTTP requests (INSPIRE + OpenAlex), not expander calls

    def budget_ok(kind: str, refs_on: bool) -> bool:
        if clock() >= end:
            _bump(counts, "deadline")
            return False
        if http_calls(counts) - base_calls + lookups.cost(kind, refs_on) > cfg["max_lookups_per_question"]:
            _bump(counts, "lookup_cap")
            return False
        return True

    try:
        picked = select(question, records, cfg)
        counts["records"] = len(picked)
        for rank, rec in enumerate(picked):
            if clock() >= end:
                _bump(counts, "deadline")
                break
            title = " ".join(str(rec.get("title") or "").split())[:300]
            portal = str(rec.get("provider") or "portal").lower()
            label = f"{portal} record"
            n_hand = 0
            q = deque((kind, val, 0, (label,), "original", True) for kind, val in _roots(rec))
            while q:
                kind, val, hops, via, nkind, refs_ok = q.popleft()
                if hops > cfg["max_hops_per_record"]:
                    _bump(counts, "hop_cap")
                    continue
                key = (kind, str(val).lower().rstrip("/"))
                if key in seen:
                    _bump(counts, "cycle")
                    continue
                seen.add(key)
                if n_hand >= cfg["max_handoffs_per_record"]:
                    _bump(counts, "handoff_cap")
                    break
                if clock() >= end:
                    _bump(counts, "deadline")
                    break

                def push(k, v, label_, nk=nkind, rok=refs_ok):
                    q.append((k, v, hops + 1, via + (label_,), nk, rok))

                def hand(url, doi, role, rule):
                    nonlocal n_hand
                    keys = {str(url or "").lower().rstrip("/"), str(doi or "").lower()} - {""}
                    if keys & have:
                        _bump(counts, "already_candidate")
                        return
                    have.update(keys)
                    n_hand += 1
                    out.append({"doi": doi or "", "url": url, "title": title if nkind == "original" else "",
                                "snippet": "", "provider": "portal:follow", "acq_source": "portal_follow",
                                "portal": portal, "portal_via": "via " + " > ".join(via), "portal_kind": nkind,
                                "portal_rank": rank, "portal_lookup": "follow", "portal_hops": hops,
                                "host_role": role, "host_role_rule": rule})
                    _bump(counts, f"handoff_{role.lower()}")

                if kind == "url":
                    hr = host_role.role_for_url(val)
                    if not host_role.is_hop_allowed(val, "answer"):
                        if hr.role == "PORTAL":  # never fetched: only an identifier in the link itself is followed
                            ids = ids_in_link(val)
                            for k, v in ids:
                                push(k, v, f"link {k}")
                            _bump(counts, "portal_hop" if ids else "dead_portal")
                        else:
                            _bump(counts, f"dead_{hr.role.lower()}")
                        continue
                    if hr.role in _EVIDENCE:
                        d = next((v for k, v in ids_in_link(val) if k == "doi"), None)
                        hand(val, d, hr.role, hr.rule_id)
                    elif hr.role == "RESOLVER":
                        ids = ids_in_link(val)
                        for k, v in ids:
                            push(k, v, "resolver")
                        if not ids:
                            _bump(counts, "dead_resolver")
                    else:
                        _bump(counts, "dead_unknown")  # owner: only ORIGINAL or REPOSITORY_OF_RECORD is an end
                elif kind == "doi":
                    rr = host_role.role_for_record({"doi": val})
                    if rr.role in ("PORTAL", "EXCLUDED"):
                        m = _ARXIV_DOI.match(val)
                        if m:
                            push("arxiv", m.group(1).lower(), "arXiv DOI")
                        else:
                            _bump(counts, "dead_preprint_doi")  # no route for non-arXiv preprints yet (plan D-b)
                        continue
                    got = None
                    if lookups.openalex is not None and budget_ok("openalex", refs_ok):
                        got = lookups.openalex(f"doi:https://doi.org/{val}", counts, end,
                                               cfg["max_refs_per_record"] if refs_ok else 0)
                    if got and got.get("locations"):
                        for u in got["locations"]:
                            push("url", u, "OpenAlex location")
                    else:
                        hand(f"https://doi.org/{val}", val, "RESOLVER", "doi_identifier")
                    if got and refs_ok:
                        for i, d in enumerate(got.get("refs") or ()):
                            if i >= cfg["max_refs_per_record"]:
                                break
                            push("doi", d, "reference", "reference", False)
                elif kind == "arxiv":
                    if lookups.arxiv is None or not budget_ok("arxiv", refs_ok):
                        _bump(counts, "dead_arxiv_no_lookup")
                        continue
                    got, lk = lookups.arxiv(val, counts, end, cfg["max_refs_per_record"] if refs_ok else 0)
                    if not got:
                        _bump(counts, "dead_arxiv_unknown")
                        continue
                    own, refs = got
                    for d in own:
                        push("doi", d, f"arXiv {val} ({lk})", nkind)
                    if refs_ok:
                        for d in list(refs)[: cfg["max_refs_per_record"]]:
                            push("doi", d, f"arXiv {val} reference ({lk})", "reference", False)
                elif kind == "openalex":
                    if lookups.openalex is None or not budget_ok("openalex", refs_ok):
                        _bump(counts, "dead_openalex_no_lookup")
                        continue
                    got = lookups.openalex(f"openalex:{val}", counts, end,
                                           cfg["max_refs_per_record"] if refs_ok else 0)
                    if not got:
                        _bump(counts, "dead_openalex_unknown")
                        continue
                    for d in got.get("published") or ():  # S2: publishedVersion locations only, never the work DOI
                        push("doi", d, f"OpenAlex {val}")
                    for u in got.get("locations") or ():
                        push("url", u, f"OpenAlex {val} location")
                    if refs_ok:
                        for d in list(got.get("refs") or ())[: cfg["max_refs_per_record"]]:
                            push("doi", d, f"OpenAlex {val} reference", "reference", False)
    except Exception as e:  # noqa: BLE001  the follow-up never breaks the search
        _bump(counts, "error_" + type(e).__name__)
    return out, counts


def extend(question: str, cands: list, acq: dict | None = None, *, clock: Callable = time.monotonic,
           deadline: float | None = None, get=None, openalex_provider=None, cfg: dict | None = None,
           lookups: Lookups | None = None) -> list:
    """cands + the original-host hand-offs of the question's promising portal records (arXiv seeds are left to
    vera/arxiv_portal's own loop). Logs counts only. Never raises."""
    try:
        cfg = cfg or load_config()
        cands = list(cands or ())
        if not cfg["enabled"]:
            return cands
        from vera import arxiv_portal
        recs = [c for c in cands if isinstance(c, dict) and arxiv_portal.seed_id(c) is None]
        have = set()
        for c in cands:
            if isinstance(c, dict):
                have |= {str(c.get("doi") or "").lower(), str(c.get("url") or "").lower().rstrip("/")} - {""}
        lk = lookups or default_lookups(acq, clock=clock, get=get, openalex_provider=openalex_provider)
        found, counts = follow(question, recs, cfg, lookups=lk, clock=clock, deadline=deadline, have=have)
        logger.info("autoacq portalfollow %s", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        return cands + found
    except Exception:  # noqa: BLE001
        logger.warning("autoacq portalfollow skipped")
        return list(cands or ())
