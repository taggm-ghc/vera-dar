"""WIKIREF-1022: English Wikipedia as a DISCOVERY PORTAL only.

Owner decision 2026-10-22: "ignore wikipedia for information, it is usable only as a portal to search for urls to
sources". Wikipedia text is NEVER evidence: it is never stored, chunked, quoted, logged, or sent to the answer writer,
and it never appears in an emitted candidate. This module only:

  1. searches English Wikipedia (MediaWiki Action API, generator=search, at most max_articles articles) and receives
     the articles' wikitext in the same request (prop=revisions), continuation within the request budget;
  2. extracts, transiently and in memory, the scholarly identifiers the articles CITE (DOI, PMCID, PMID, arXiv) with
     strict patterns, grouped per citation (a {{cite journal}} with doi + pmc + pmid is ONE source);
  3. optionally orders them by the stem overlap between the question and the sentence that carries the <ref>
     (computed transiently, then discarded);
  4. emits candidate dicts in the format vera.search_providers' DDG mapping produces (_ddg_cand), so they flow
     through the UNCHANGED acquisition path: rank_candidates -> resolve (Europe PMC by DOI) -> JATS fetch -> licence,
     injection, relevance and admission gates.

Emitted candidates carry identifiers and fixed labels only: title and snippet are empty, no Wikipedia URL, title,
sentence or citation text. Non-scholarly citation URLs (news, blogs) are never candidates; at most max_links of them
are returned separately as reading-list links (held, like the DDG rule "unclear-licence pages held as links only").

Network is injected (http_get, url_check, clock) so the offline tests never open a socket. No model call, no DB, no
env read. The question is used only to build the search query and the transient ranking.
"""
from __future__ import annotations

import json
import logging
import re
import time
from vera import strict_json
from pathlib import Path
from typing import Callable, Mapping
from urllib.parse import urlencode, urlsplit

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "wiki_ref_portal.json"
PORTAL = "wikipedia_refs"
SOURCE = "wikipedia_portal"
REQUIRED_KEYS = ("enabled", "api_url", "allowed_hosts", "user_agent", "max_articles", "max_refs", "max_links",
                 "timeout_s", "budget", "query", "rank", "skip_disambiguation")
logger = logging.getLogger("vera.autoacq")

# Identifier syntax (format definitions, not tunables). DOI: the Crossref-recommended pattern; PMCID/PMID: NCBI forms;
# arXiv: new-style YYMM.NNNN(N) and old-style archive(.SC)/YYMMNNN, an optional version is dropped.
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
PMCID_RE = re.compile(r"(?:PMC)?(\d{1,10})", re.I)
PMID_RE = re.compile(r"\d{1,9}")
ARXIV_RE = re.compile(r"(?:arxiv:)?(\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", re.I)
_DOI_TRAIL = re.compile(r"(?:[.,;:]|/full|/abstract|/epdf|/pdf|\.pdf)+$", re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF = re.compile(r"<ref(?:\s[^>/]*)?>(.*?)</ref\s*>", re.S | re.I)
_TEMPLATE = re.compile(r"\{\{\s*(?:cite|citation|vcite)[^{}]*\}\}", re.I | re.S)
_PARAM = re.compile(r"\|\s*([a-z0-9\-]+)\s*=\s*([^|}]*)", re.I)
_DOI_TPL = re.compile(r"\{\{\s*doi\s*\|\s*([^|}]*)\}\}", re.I)
_URL = re.compile(r"https?://[^\s|\]\[}{<>\"']+", re.I)
_SENT_END = re.compile(r"[.!?]\s|\n")
_WORD = re.compile(r"[a-z0-9]+")


class PortalError(Exception):
    """A portal-level failure; the message is a fixed code, never question or article text."""


# ------------------------------------------------------------------------------------------------ config
def load_config(path: Path = CONFIG_PATH) -> dict:
    """The config, or {"enabled": False} when missing, unreadable or incomplete (fail closed: the portal stays off)."""
    try:
        c = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"enabled": False}
    return validate(c)


def validate(c) -> dict:
    """c when it carries every required key, else {"enabled": False}."""
    if not isinstance(c, dict) or any(k not in c for k in REQUIRED_KEYS):
        return {"enabled": False}
    return c


def enabled(cfg: Mapping | None) -> bool:
    return isinstance(cfg, Mapping) and cfg.get("enabled") is True


def is_candidate(c) -> bool:
    return isinstance(c, Mapping) and c.get("acq_source") == SOURCE


# ------------------------------------------------------------------------------------------------ identifiers
def norm_doi(v) -> str | None:
    s = _COMMENT.sub("", str(v or "")).strip()
    s = re.sub(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)", "", s, flags=re.I).strip()
    s = _DOI_TRAIL.sub("", s)
    return s.lower() if DOI_RE.fullmatch(s) else None


def norm_pmcid(v) -> str | None:
    m = PMCID_RE.fullmatch(_COMMENT.sub("", str(v or "")).strip())
    return f"PMC{int(m.group(1))}" if m and int(m.group(1)) > 0 else None


def norm_pmid(v) -> str | None:
    s = _COMMENT.sub("", str(v or "")).strip()
    return str(int(s)) if PMID_RE.fullmatch(s) and int(s) > 0 else None


def norm_arxiv(v) -> str | None:
    m = ARXIV_RE.fullmatch(_COMMENT.sub("", str(v or "")).strip())
    return m.group(1).lower() if m else None


def _ids_from_url(url: str) -> dict:
    from vera.search_providers import ddg_identifier
    ident = ddg_identifier(url) or {}
    k, v = ident.get("kind"), ident.get("value")
    out = {}
    if k == "doi" and norm_doi(v):
        out["doi"] = norm_doi(v)
    elif k == "pmcid" and norm_pmcid(v):
        out["pmcid"] = norm_pmcid(v)
    elif k == "pmid" and norm_pmid(v):
        out["pmid"] = norm_pmid(v)
    elif k == "arxiv" and norm_arxiv(v):
        out["arxiv"] = norm_arxiv(v)
    return out


def citation_ids(chunk: str) -> tuple[dict, list]:
    """One citation's text -> ({doi?, pmcid?, pmid?, arxiv?}, [non-scholarly https URLs]). Strict: a value that does
    not FULLY match its pattern is dropped (no prefix salvage), so markup or prose in a value never passes."""
    ids, urls = {}, []
    body = _COMMENT.sub("", chunk or "")
    for m in _PARAM.finditer(body):
        k, v = m.group(1).lower(), m.group(2)
        if k == "doi" and norm_doi(v):
            ids.setdefault("doi", norm_doi(v))
        elif k == "pmc" and norm_pmcid(v):
            ids.setdefault("pmcid", norm_pmcid(v))
        elif k == "pmid" and norm_pmid(v):
            ids.setdefault("pmid", norm_pmid(v))
        elif k in ("arxiv", "eprint") and norm_arxiv(v):
            ids.setdefault("arxiv", norm_arxiv(v))
    for m in _DOI_TPL.finditer(body):
        if norm_doi(m.group(1)):
            ids.setdefault("doi", norm_doi(m.group(1)))
    for m in _URL.finditer(body):
        u = m.group(0)
        got = _ids_from_url(u)
        if got:
            for k, v in got.items():
                ids.setdefault(k, v)
        else:
            urls.append(u)
    return ids, urls


def _stems(text: str, n: int, strip: set) -> set:
    return {w[:n] for w in _WORD.findall(str(text or "").lower()) if w not in strip and len(w) > 1}


def _context_before(text: str, pos: int, max_chars: int) -> str:
    """The sentence ending at pos (transient; for ranking only)."""
    lo = max(0, pos - max_chars)
    seg = _REF.sub(" ", text[lo:pos])
    ends = list(_SENT_END.finditer(seg))
    return seg[ends[-1].end():] if ends else seg


def extract_citations(wikitext: str, question: str, cfg: Mapping) -> list:
    """Wikitext -> [{"ids": {...}, "urls": [...], "score": float}] in article order. Inline <ref> citations first
    (scored against their citing sentence when rank.use_citing_sentence), then cite templates outside <ref> (score 0).
    The wikitext and the sentences are local variables only: nothing derived from them except identifiers, scores and
    URLs leaves this function."""
    text = str(wikitext or "")
    rk = cfg["rank"]
    strip = {str(t).lower() for t in (cfg["query"].get("strip_terms") or ())}
    n = int(rk["stem_len"])
    q = _stems(question, n, strip)
    out = []
    for m in _REF.finditer(text):
        ids, urls = citation_ids(m.group(1))
        if not ids and not urls:
            continue
        score = 0.0
        if rk.get("use_citing_sentence") is True and q:
            score = len(q & _stems(_context_before(text, m.start(), int(rk["max_context_chars"])), n, strip)) / len(q)
        out.append({"ids": ids, "urls": urls, "score": round(score, 6)})
    rest = _REF.sub(" ", text)
    for m in _TEMPLATE.finditer(rest):
        ids, urls = citation_ids(m.group(0))
        if ids or urls:
            out.append({"ids": ids, "urls": urls, "score": 0.0})
    for m in _DOI_TPL.finditer(rest):
        if norm_doi(m.group(1)):
            out.append({"ids": {"doi": norm_doi(m.group(1))}, "urls": [], "score": 0.0})
    return out


def _keys(ids: Mapping) -> set:
    return {f"{k}:{v}" for k, v in ids.items()}


def dedupe(cites: list) -> list:
    """Merge citations that share any identifier (union of ids, max score, first position kept)."""
    out, index = [], {}
    for c in cites:
        if not c["ids"]:
            continue
        hit = next((index[k] for k in _keys(c["ids"]) if k in index), None)
        if hit is None:
            hit = {"ids": dict(c["ids"]), "score": c["score"], "order": len(out)}
            out.append(hit)
        else:
            for k, v in c["ids"].items():
                hit["ids"].setdefault(k, v)
            hit["score"] = max(hit["score"], c["score"])
        for k in _keys(hit["ids"]):
            index[k] = hit
    return out


# ------------------------------------------------------------------------------------------------ candidates
def to_candidate(ids: Mapping, rank: int) -> dict | None:
    """Identifiers -> a candidate in the DDG-mapped format (search_providers._ddg_cand), or None (PMID-only: needs the
    Europe PMC id lookup). Only identifiers and fixed labels: no Wikipedia text."""
    pmcid = ids.get("pmcid")
    base = {"title": "", "snippet": "", "provider": SOURCE, "acq_source": SOURCE, "portal": PORTAL,
            "portal_rank": rank, **({"pmcid": pmcid, "oa": {"pmcid": pmcid}} if pmcid else {}),
            **({"pmid": ids["pmid"]} if ids.get("pmid") else {})}
    if ids.get("doi"):
        return {"url": f"https://doi.org/{ids['doi']}", "doi": ids["doi"], "ref_id_kind": "doi", **base}
    if ids.get("arxiv"):
        doi = f"10.48550/arxiv.{ids['arxiv']}"
        return {"url": f"https://doi.org/{doi}", "doi": doi, "ref_id_kind": "arxiv", "arxiv": ids["arxiv"], **base}
    if pmcid:
        return {"url": f"https://europepmc.org/article/PMC/{pmcid}", "doi": None, "ref_id_kind": "pmcid", **base}
    return None


def build_query(question: str, cfg: Mapping) -> str:
    q = cfg["query"]
    strip = {str(t).lower() for t in q.get("strip_terms") or ()}
    from vera.search_query import drop_generic  # DDFIX-1023/REL: idioms such as "de facto", "status quo"
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", drop_generic(str(question or "")))
    keep = [w for w in words if w.lower() not in strip][: int(q["max_terms"])]
    return (" ".join(keep) if keep else " ".join(str(question or "").split()))[: int(q["max_query_chars"])]


# ------------------------------------------------------------------------------------------------ guarded GET
def get_json(params: dict, cfg: Mapping, *, http_get: Callable, url_check: Callable | None = None,
             clock: Callable = time.monotonic, deadline: float | None = None) -> dict:
    """One GET of the API: https, allowlisted host, public-address check, no redirect, JSON only, byte cap, wall-clock
    limit timeout_s over the whole streamed body (capped by deadline). Raises PortalError(code). Never retried."""
    url = str(cfg["api_url"]) + "?" + urlencode(params)
    sp = urlsplit(url)
    if sp.scheme != "https" or (sp.hostname or "").lower() not in {h.lower() for h in cfg["allowed_hosts"]}:
        raise PortalError("host")
    if url_check is None:
        from vera.selective_fetch import is_public_url as url_check
    ok = url_check(url)
    if not (ok[0] if isinstance(ok, tuple) else ok):
        raise PortalError("host")
    timeout_s = float(cfg["timeout_s"])
    end = clock() + timeout_s if deadline is None else min(clock() + timeout_s, deadline)
    if clock() >= end:
        raise PortalError("deadline")
    max_bytes = int(cfg["budget"]["max_response_bytes"])
    resp = None
    try:
        try:
            resp = http_get(url, headers={"User-Agent": str(cfg["user_agent"]), "Accept": "application/json"},
                            timeout=max(0.1, min(timeout_s, end - clock())), allow_redirects=False, stream=True)
        except Exception as e:  # noqa: BLE001
            if "rate limit" in str(e).lower():
                raise PortalError("rate_limited") from None
            raise PortalError("unreachable") from None
        status = int(getattr(resp, "status_code", 0) or 0)
        if status == 429:
            raise PortalError("rate_limited")
        if status != 200:
            raise PortalError("unreachable")
        ctype = ((getattr(resp, "headers", None) or {}).get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise PortalError("unreachable")
        chunks, size = [], 0
        try:
            for chunk in resp.iter_content(65536):
                size += len(chunk)
                if size > max_bytes:
                    raise PortalError("too_large")
                if clock() > end:
                    raise PortalError("deadline")
                chunks.append(chunk)
        except PortalError:
            raise
        except Exception:  # noqa: BLE001
            raise PortalError("unreachable") from None
        try:
            doc = strict_json.loads(b"".join(chunks).decode("utf-8"))
        except Exception:  # noqa: BLE001
            raise PortalError("unreachable") from None
        if not isinstance(doc, dict) or "error" in doc:
            raise PortalError("unreachable")
        return doc
    finally:
        try:
            if resp is not None and hasattr(resp, "close"):
                resp.close()
        except Exception:  # noqa: BLE001
            pass


def search_params(question: str, cfg: Mapping) -> dict:
    return {"action": "query", "format": "json", "formatversion": "2", "generator": "search",
            "gsrsearch": build_query(question, cfg), "gsrnamespace": "0", "gsrlimit": str(int(cfg["max_articles"])),
            "gsrprop": "", "prop": "revisions|pageprops", "rvprop": "content", "rvslots": "main",
            "ppprop": "disambiguation", "redirects": "1"}


# ------------------------------------------------------------------------------------------------ portal
def discover(question: str, cfg: Mapping | None = None, *, http_get: Callable | None = None,
             url_check: Callable | None = None, clock: Callable = time.monotonic, deadline: float | None = None,
             max_refs: int | None = None) -> tuple[list, list, dict]:
    """-> (candidates, reading-list links, counts). Never raises: a failure returns what was gathered so far plus a
    counts["error"] code. Counts carry numbers and fixed codes only (no titles, no question)."""
    cfg = load_config() if cfg is None else cfg
    counts = {"requests": 0, "articles": 0, "citations": 0, "ids": 0, "candidates": 0, "held_pmid_only": 0,
              "links": 0}
    if not enabled(cfg):
        counts["error"] = "disabled"
        return [], [], counts
    if http_get is None:
        import requests
        http_get = requests.get
    budget = int(cfg["budget"]["max_requests_per_question"])
    max_articles = int(cfg["max_articles"])
    params = search_params(question, cfg)
    pages: dict = {}
    while True:
        if counts["requests"] >= budget:
            counts["error"] = "budget"
            break
        counts["requests"] += 1
        try:
            doc = get_json(params, cfg, http_get=http_get, url_check=url_check, clock=clock, deadline=deadline)
        except PortalError as e:
            counts["error"] = str(e)
            break
        for p in ((doc.get("query") or {}).get("pages") or ()) if isinstance(doc.get("query"), dict) else ():
            if not isinstance(p, dict) or not isinstance(p.get("pageid"), int):
                continue
            slot = pages.setdefault(p["pageid"], {"index": p.get("index"), "content": None, "disambig": False})
            if isinstance(p.get("index"), int):
                slot["index"] = p["index"]
            if isinstance(p.get("pageprops"), dict) and "disambiguation" in p["pageprops"]:
                slot["disambig"] = True
            for r in p.get("revisions") or ():
                c = (((r or {}).get("slots") or {}).get("main") or {}).get("content") if isinstance(r, dict) else None
                if isinstance(c, str):
                    slot["content"] = c
        # Follow "continue" only while the batch is incomplete (prop=revisions split the content). After
        # batchcomplete, "continue" carries gsroffset = the NEXT search page: never followed (max_articles holds).
        cont = doc.get("continue")
        if (doc.get("batchcomplete") or not isinstance(cont, dict)
                or all(v["content"] is not None for v in pages.values())):
            break
        params = {**search_params(question, cfg), **{k: str(v) for k, v in cont.items()}}
    ordered = sorted(pages.values(), key=lambda v: (v["index"] if isinstance(v["index"], int) else 1 << 30))
    cites = []
    for i, art in enumerate(ordered[:max_articles]):
        if art["content"] is None or (art["disambig"] and cfg["skip_disambiguation"] is True):
            continue
        counts["articles"] += 1
        for c in extract_citations(art["content"], question, cfg):
            c["article"] = i
            cites.append(c)
        art["content"] = None  # discard the article text as soon as its identifiers are out
    counts["citations"] = len(cites)
    links, seen_links = [], set()
    from vera.search_providers import safe_target_url
    for c in cites:
        for u in c["urls"]:
            s = safe_target_url(u)
            if s and s not in seen_links and len(links) < int(cfg["max_links"]):
                seen_links.add(s)
                links.append(s)
    groups = dedupe([c for c in cites if c["ids"]])
    counts["ids"] = len(groups)
    groups.sort(key=lambda g: (-g["score"], g["order"]))
    cap = int(cfg["max_refs"]) if max_refs is None else min(int(cfg["max_refs"]), int(max_refs))
    out = []
    for g in groups:
        if len(out) >= cap:
            break
        cand = to_candidate(g["ids"], len(out))
        if cand is None:
            counts["held_pmid_only"] += 1
            continue
        out.append(cand)
    counts["candidates"], counts["links"] = len(out), len(links)
    logger.info("autoacq wikiref %s", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return out, links, counts


def make_portal_search(cfg: Mapping | None = None, *, http_get: Callable | None = None,
                       url_check: Callable | None = None, clock: Callable = time.monotonic) -> Callable:
    """A search(question, n, timeout_s) -> [candidates] callable for search_providers.acquisition_search's pool. The
    deadline is now + timeout_s; n caps the emitted candidates together with max_refs."""
    def search(question, n, timeout_s):
        cands, _links, _counts = discover(question, cfg, http_get=http_get, url_check=url_check, clock=clock,
                                          deadline=clock() + float(timeout_s), max_refs=n or None)
        return cands
    return search
