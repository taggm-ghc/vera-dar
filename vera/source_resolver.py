"""Item #79 phase B: follow link-only search results to their original scholarly sources.

A result with no abstract (DuckDuckGo, news, portals, OpenAlex records without an abstract) is followed:
its page is fetched under selective_fetch's guards, scholarly identifiers are read from it (the page's own
citation DOI first, then arXiv links, then DOIs it links to), and each identifier is resolved through
OpenAlex to a title and date (metadata only). A page with no identifier is followed one more hop through its
links to known scholarly hosts. If that finds nothing (for example a portal that refuses the fetch), the result's
title is looked up in OpenAlex and accepted only on a close title match. Every count, hop and time limit comes from the `resolve` settings in
config/ask-provider-chain.json; the work stops at the deadline and reports what it skipped.

PORTAL-WIRE-B-1110 (owner 2026-11-10, OD-1110-PORTALS-ALL): OpenAlex and Semantic Scholar are PORTALS (indexes). Their
abstracts are never evidence, stored or answer-only, so this module neither requests nor returns one: the OpenAlex
select omits abstract_inverted_index, Semantic Scholar is asked for title and date only, and every record has an empty
snippet. A record is a discovery entry (title, link, identifier) that grounded_ask lists and hands to acquisition; the
original's own text is obtained, if at all, by the acquisition routes on the original host.
"""
import logging
import re
import time
from urllib.parse import urljoin, urlsplit

import requests
from vera import strict_json

from vera.selective_fetch import fetch_html
from vera.search_providers import USER_AGENT
from vera import api_permissions
from vera.api_permissions import ApiDenied


def _api_get(url: str, **kw):
    """API-ALLOW-1107: every request in this module is an API call (OpenAlex, Semantic Scholar): the permission gate
    (config/api_permissions.json; a denial sends nothing and is a miss), then the entry's rate-limit portal (RATE-1022
    A7: api.openalex.org -> openalex, api.semanticscholar.org -> semantic_scholar)."""
    return api_permissions.call(requests.get, url, **kw)


_LIMITED_GET = _api_get
_MISS = (requests.RequestException, ValueError, ApiDenied)  # API-ALLOW-1107: a denied API is a miss, never an error

logger = logging.getLogger("vera")

OPENALEX_WORK = "https://api.openalex.org/works/doi:{doi}"
OPENALEX_FIELDS = "id,doi,title,publication_date,type"  # PORTAL-WIRE-B-1110: no abstract_inverted_index (portal text)
_META_DOI = re.compile(r"<meta[^>]+name=[\"'](?:citation_doi|dc\.identifier|prism\.doi|DC\.Identifier)[\"'][^>]*"
                       r"content=[\"']\s*(?:doi:|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[^\"'\s<>]+)", re.I)
_META_DOI_REV = re.compile(r"<meta[^>]+content=[\"']\s*(?:doi:|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[^\"'\s<>]+)"
                           r"[\"'][^>]*name=[\"'](?:citation_doi|dc\.identifier|prism\.doi)[\"']", re.I)
_ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.I)
_DOI = re.compile(r"(?:doi\.org/|doi:\s*)(10\.\d{4,9}/[^\s\"'<>&#?]+)", re.I)
_HREF = re.compile(r"href=[\"']([^\"'#]+)[\"']", re.I)
_SSRN = re.compile(r"ssrn\.com/(?:sol3/papers\.cfm\?abstract_id=|abstract=)(\d+)", re.I)
OPENALEX_SEARCH = "https://api.openalex.org/works"


def _clean_doi(doi: str) -> str:
    return doi.rstrip(".,;)]}").lower()


def identifiers_in(html: str, max_ids: int) -> list[str]:
    """DOIs in priority order: the page's own citation DOI, arXiv links (as arXiv DOIs), then linked DOIs."""
    found: list[str] = []
    for d in _META_DOI.findall(html) + _META_DOI_REV.findall(html):
        found.append(_clean_doi(d))
    for a in _ARXIV.findall(html):
        found.append(f"10.48550/arxiv.{a.lower()}")
    for sid in _SSRN.findall(html):
        found.append(f"10.2139/ssrn.{sid}")
    for d in _DOI.findall(html):
        found.append(_clean_doi(d))
    return list(dict.fromkeys(found))[:max_ids]


def scholarly_links(html: str, base_url: str, hosts: list[str], max_links: int) -> list[str]:
    out = []
    for href in _HREF.findall(html):
        url = urljoin(base_url, href)
        host = (urlsplit(url).hostname or "").lower()
        if url.startswith(("https://", "http://")) and any(host == h or host.endswith("." + h) for h in hosts):
            out.append(url)
    return list(dict.fromkeys(out))[:max_links]


S2_PAPER = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"


def _s2(doi: str, timeout_s: float, http_get=_LIMITED_GET) -> dict:
    """Semantic Scholar record, title and date only (keyless; may rate-limit, which counts as a miss)."""
    try:
        r = http_get(S2_PAPER.format(doi=doi), params={"fields": "title,publicationDate"},
                     headers={"User-Agent": USER_AGENT}, timeout=timeout_s)
        return strict_json.response_json(r) if r.status_code == 200 else {}
    except _MISS:
        return {}


def _record(doi: str, title, published) -> dict:
    """A discovery record: metadata only, never text. PORTAL-WIRE-B-1110: the snippet is always empty (an index's
    abstract is never evidence) and evidence_eligible is False."""
    return {"title": title or doi, "url": f"https://doi.org/{doi}", "published": published, "snippet": "",
            "source_type": "scholarly", "doi": doi, "stored_text_kind": "metadata_only", "evidence_eligible": False}


def _oa_get(url: str, params: dict, timeout_s: float, item: str, http_get):
    """OPENALEX-FIX-1110 (R3): every OpenAlex request here goes through the metered client (terms gate scope "search",
    key when configured, 2xx-only charging, budget and seed share, one cooldown, no retry). A refusal by the client
    raises with nothing sent and is a miss (fail-closed item 5: lookup_doi then asks Semantic Scholar)."""
    from vera import search_providers as sp
    tx = None if http_get is _LIMITED_GET else (lambda u, p, h, t: http_get(u, params=p, headers=h, timeout=t))
    return sp.openalex_request(url, params, purpose="resolve", timeout_s=timeout_s, item=item, transport=tx)


def _oa_miss() -> tuple:
    from vera import rate_limit, search_providers as sp
    return (*_MISS, sp.ProviderUnavailable, rate_limit.RateLimited)


def lookup_doi(doi: str, timeout_s: float, http_get=_LIMITED_GET) -> dict | None:
    """Original source for a DOI: OpenAlex metadata (title, date), else Semantic Scholar's. None when neither index
    knows the DOI. Never an abstract (PORTAL-WIRE-B-1110)."""
    try:
        r = _oa_get(OPENALEX_WORK.format(doi=doi), {"select": OPENALEX_FIELDS}, timeout_s, "doi:" + doi, http_get)
        w = strict_json.response_json(r, keys="exact") if r.status_code == 200 else None
    except _oa_miss():
        w = None
    if w:
        return _record(doi, w.get("title"), w.get("publication_date"))
    s2 = _s2(doi, timeout_s, http_get)
    if not s2:
        return None
    return dict(_record(doi, s2.get("title"), s2.get("publicationDate")), metadata_source="semanticscholar")


def _words(t: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(t or "").lower()))


def lookup_title(title: str, timeout_s: float, min_match: float, http_get=_LIMITED_GET) -> dict | None:
    """OpenAlex title search; accept only a record whose title word overlap (Jaccard) is at least min_match."""
    want = _words(title)
    if len(want) < 4:
        return None
    try:
        r = _oa_get(OPENALEX_SEARCH, {"search": title, "per-page": 3, "select": OPENALEX_FIELDS}, timeout_s,
                    "title:" + " ".join(sorted(want)), http_get)
        if r.status_code != 200:
            return None
        items = strict_json.response_json(r, keys="exact").get("results") or []
    except _oa_miss():
        return None
    for w in items:
        got = _words(w.get("title"))
        if got and len(want & got) / len(want | got) >= min_match:
            doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", str(w.get("doi") or ""), flags=re.I).lower() or None
            if not doi:
                continue
            return _record(doi, w.get("title"), w.get("publication_date"))
    return None


def resolve(result: dict, cfg: dict, deadline_at: float, *, fetch=fetch_html, lookup=lookup_doi,
            by_title=lookup_title, clock=time.monotonic) -> list[dict]:
    """Original sources behind one link-only result (possibly none). Bounded by cfg and deadline_at."""
    url = str(result.get("url") or "")
    via = urlsplit(url).hostname
    if result.get("doi"):
        rec = lookup(_clean_doi(result["doi"]), cfg["timeout_s"])
        return [dict(rec, discovered_via=via)] if rec else []
    direct = identifiers_in(url, 1)  # the link itself may be an arXiv, SSRN or DOI link
    if direct:
        rec = lookup(direct[0], cfg["timeout_s"])
        if rec:
            return [dict(rec, discovered_via=via)]
    pages, frontier, seen, out = 0, [(url, 0)], set(), []
    while frontier and pages < cfg["max_pages_per_result"] and clock() < deadline_at:
        page, depth = frontier.pop(0)
        if page in seen:
            continue
        seen.add(page)
        pages += 1
        html, final, err = fetch(page, timeout_s=cfg["timeout_s"])
        if err or not html:
            continue
        ids = identifiers_in(html, cfg["max_ids_per_page"])
        for doi in ids:
            if clock() >= deadline_at or len(out) >= cfg["max_ids_per_page"]:
                break
            rec = lookup(doi, cfg["timeout_s"])
            if rec:
                out.append(dict(rec, discovered_via=urlsplit(url).hostname))
        if out:
            break
        if not ids and depth + 1 < cfg["max_depth"]:
            frontier += [(u, depth + 1) for u in scholarly_links(html, final or page, cfg["scholarly_hosts"],
                                                                 cfg["max_links_per_page"])]
    if not out and result.get("title") and clock() < deadline_at:  # blocked portal or no identifier on the page
        rec = by_title(str(result["title"]), cfg["timeout_s"], float(cfg["title_match_min"]))
        if rec:
            out.append(dict(rec, discovered_via=via))
    return out
