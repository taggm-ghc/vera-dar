"""Item #79 phase B: follow link-only search results to their original scholarly sources.

A result with no abstract (DuckDuckGo, news, portals, OpenAlex records without an abstract) is followed:
its page is fetched under selective_fetch's guards, scholarly identifiers are read from it (the page's own
citation DOI first, then arXiv links, then DOIs it links to), and each identifier is resolved through
OpenAlex to a title, date and abstract. A page with no identifier is followed one more hop through its
links to known scholarly hosts. If that finds nothing (for example a portal that refuses the fetch), the result's
title is looked up in OpenAlex and accepted only on a close title match. Every count, hop and time limit comes from the `resolve` settings in
config/ask-provider-chain.json; the work stops at the deadline and reports what it skipped.
"""
import logging
import re
import time
from urllib.parse import urljoin, urlsplit

import requests

from vera.selective_fetch import fetch_html
from vera.search_providers import OPENALEX_METADATA_LICENCE, USER_AGENT, licence_record, reconstruct_abstract

logger = logging.getLogger("vera")

OPENALEX_WORK = "https://api.openalex.org/works/doi:{doi}"
OPENALEX_FIELDS = "id,doi,title,publication_date,abstract_inverted_index,type"
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


def _s2(doi: str, timeout_s: float, http_get=requests.get) -> dict:
    """Semantic Scholar record (keyless; may rate-limit, which counts as a miss)."""
    try:
        r = http_get(S2_PAPER.format(doi=doi), params={"fields": "title,abstract,publicationDate"},
                     headers={"User-Agent": USER_AGENT}, timeout=timeout_s)
        return r.json() if r.status_code == 200 else {}
    except (requests.RequestException, ValueError):
        return {}


def _record(doi: str, title, published, abstract: str) -> dict:
    """An OpenAlex abstract carries OpenAlex's CC0 metadata licence; any other origin is set by the caller."""
    rec = {"title": title or doi, "url": f"https://doi.org/{doi}", "published": published,
           "snippet": abstract or "", "source_type": "scholarly", "doi": doi, "stored_text_kind": "abstract_metadata"}
    if abstract:
        rec["licence"] = licence_record(metadata=OPENALEX_METADATA_LICENCE)
    return rec


def with_abstract(rec: dict, timeout_s: float, http_get=requests.get) -> dict:
    """Fill a missing abstract from Semantic Scholar; the record keeps an empty snippet if none is found."""
    if rec.get("snippet") or not rec.get("doi"):
        return rec
    s2 = _s2(rec["doi"], timeout_s, http_get)
    # Semantic Scholar's abstract licence is not declared per record: licence None, so the licence gate holds it
    return dict(rec, snippet=s2["abstract"], licence=None, abstract_source="semanticscholar") if s2.get("abstract") else rec


def lookup_doi(doi: str, timeout_s: float, http_get=requests.get) -> dict | None:
    """Original source for a DOI: OpenAlex metadata, abstract from OpenAlex or Semantic Scholar.
    None when neither index knows the DOI; an empty snippet means the original has no free abstract."""
    try:
        r = http_get(OPENALEX_WORK.format(doi=doi), params={"select": OPENALEX_FIELDS},
                     headers={"User-Agent": USER_AGENT}, timeout=timeout_s)
        w = r.json() if r.status_code == 200 else None
    except (requests.RequestException, ValueError):
        w = None
    if w:
        rec = _record(doi, w.get("title"), w.get("publication_date"),
                      reconstruct_abstract(w.get("abstract_inverted_index"), 4000))
        return with_abstract(rec, timeout_s, http_get)
    s2 = _s2(doi, timeout_s, http_get)
    if not s2:
        return None
    rec = _record(doi, s2.get("title"), s2.get("publicationDate"), "")
    return dict(rec, snippet=s2.get("abstract") or "", licence=None, abstract_source="semanticscholar")


def _words(t: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(t or "").lower()))


def lookup_title(title: str, timeout_s: float, min_match: float, http_get=requests.get) -> dict | None:
    """OpenAlex title search; accept only a record whose title word overlap (Jaccard) is at least min_match."""
    want = _words(title)
    if len(want) < 4:
        return None
    try:
        r = http_get(OPENALEX_SEARCH, params={"search": title, "per-page": 3, "select": OPENALEX_FIELDS},
                     headers={"User-Agent": USER_AGENT}, timeout=timeout_s)
        if r.status_code != 200:
            return None
        items = r.json().get("results") or []
    except (requests.RequestException, ValueError):
        return None
    for w in items:
        got = _words(w.get("title"))
        if got and len(want & got) / len(want | got) >= min_match:
            abstract = reconstruct_abstract(w.get("abstract_inverted_index"), 4000)
            doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", str(w.get("doi") or ""), flags=re.I).lower() or None
            if not doi:
                continue
            return with_abstract(_record(doi, w.get("title"), w.get("publication_date"), abstract), timeout_s, http_get)
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
