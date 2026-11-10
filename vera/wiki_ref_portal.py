"""WIKI-REDESIGN-1107: English Wikipedia as a REFERENCE-LIST portal only (supersedes the WIKIREF-1022 API design).

Owner 2026-11-07: "for Wikipedia, we shouldn't use its page contents. instead, only search for reference links to follow
those for original sources". Owner 2026-10-22: Wikipedia is never evidence, only a portal. Standing rules: robots.txt
first and default-deny; never violate robots, terms or a deny-list.

Flow (no page under /w/, /api/ or Special: is ever fetched; en.wikipedia.org/robots.txt disallows them for `*`):

  1. DISCOVER articles with ONE call to Wikipedia's own search API (SEARCHCAPS-1110, owner OD-1110-SEARCHCAPS
     "Keep low, use it better": Wikipedia lookups go through Wikipedia's API instead of a DuckDuckGo `site:` query,
     which keeps DuckDuckGo's small daily cap for visitor questions). The call is the MediaWiki Action API
     list=search (titles only: srprop is empty, so no snippet or prose is returned), gated by config/api_permissions.json
     entry wikipedia_search_api (owner_switch permit, OD-1110-SEARCHCAPS; the API is judged on its documented API
     terms, OD-1107-HITL), sent serially through the "wikipedia" rate portal with the project's descriptive
     User-Agent (a project URL as the contact, no email; Wikimedia API etiquette) and maxlag. Only main-namespace
     article URLs https://en.wikipedia.org/wiki/<Title> are kept.
  2. ROBOTS: each article URL goes through vera.selective_fetch.robots_allowed (RFC 9309 matcher, AI-agent groups and
     statements, fail closed) BEFORE any request to it; a deny or an unreadable robots.txt means no fetch. The
     host's Crawl-delay is honoured (crawl_reserve: never sent early). Public-address check, no redirect, byte cap,
     deadline.
  3. EXTRACT only the reference metadata: the COinS spans (<span class="Z3988" title="...">, OpenURL ContextObject
     key/value pairs that MediaWiki's citation templates emit per reference). The parser reads tag attributes only
     and has no text handler, so article prose is never read, stored, scored, logged or returned. References with no
     COinS (bare links, explanatory notes) are skipped.
  4. RELEVANCE from the reference's OWN metadata (title and venue) against the question; news/time cue words and
     years are not counted as question stems (LINKS-VERIFY-1107 finding 3).
  5. HAND-OFF: each kept reference becomes an ordinary acquisition candidate: a scholarly identifier (DOI, PMCID,
     arXiv) in the search_providers candidate format, or the ORIGINAL https URL as a URL-only candidate (a
     web.archive.org snapshot is unwrapped to the page it archives; other archive mirrors are dropped). Everything
     after that is the unchanged pipeline: resolve, robots, licence, TDM opt-out, H19 ND/news links-only, full-text
     only, injection, relevance and admission gates. No Wikipedia URL, title or text is ever in a candidate, and the
     portal no longer offers a "via Wikipedia citations" reading list (record_links is not called).

Network is injected (site_search, http_get, url_check, robots_ok, clock) so the offline tests never open a socket. No
model call, no DB, no env read.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Mapping
from urllib.parse import parse_qsl, quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "wiki_ref_portal.json"
PORTAL = "wikipedia_refs"
SOURCE = "wikipedia_portal"
REQUIRED_KEYS = ("enabled", "article_host", "allowed_hosts", "user_agent", "max_articles", "max_refs", "timeout_s",
                 "max_response_bytes", "query", "relevance")
logger = logging.getLogger("vera.autoacq")

# Identifier syntax (format definitions, not tunables).
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
PMCID_RE = re.compile(r"(?:PMC)?(\d{1,10})", re.I)
PMID_RE = re.compile(r"\d{1,9}")
ARXIV_RE = re.compile(r"(?:arxiv:)?(\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", re.I)
_DOI_TRAIL = re.compile(r"(?:[.,;:]|/full|/abstract|/epdf|/pdf|\.pdf)+$", re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_WORD = re.compile(r"[a-z0-9]+")
_YEAR = re.compile(r"^(?:1[5-9]|20)\d\d$")
# Main-namespace article path only: no query, no namespace prefix (":"), no sub-page, nothing under /w/ or /api/.
_ARTICLE_PATH = re.compile(r"^/wiki/[^:/?#\s]{1,250}$")
# Wayback snapshot: /web/<14-digit-or-shorter timestamp>[flag_]/<original absolute URL>
_WAYBACK = re.compile(r"^https?://(?:www\.)?web\.archive\.org/web/\d{1,14}(?:[a-z]{2}_)?/(https?://.+)$", re.I)
_MIRRORS = re.compile(r"(?:^|\.)(?:archive\.(?:ph|today|is|li|vn|fo|md)|archive\.org|webcitation\.org|"
                      r"ghostarchive\.org|wikiwix\.com)$", re.I)
_WIKI_FAMILY = re.compile(r"(?:^|\.)(?:wikipedia|wikimedia|wikidata|wiktionary|wikibooks|wikiquote|wikisource|"
                          r"wikiversity|wikivoyage|wikinews|mediawiki)\.org$", re.I)
_CTRL = re.compile(r"[\x00-\x1f\x7f]")
_MAX_TITLE_CHARS = 300  # safety bound on an editor-supplied title (display/relevance only), not a tunable


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
    s = re.sub(r"^(?:doi:\s*|info:doi/|https?://(?:dx\.)?doi\.org/)", "", s, flags=re.I).strip()
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


# ------------------------------------------------------------------------------------------------ URLs
def _host(url: str) -> str:
    try:
        return (urlsplit(str(url or "")).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def article_url(url: str, cfg: Mapping) -> str | None:
    """The canonical https://<article_host>/wiki/<Title> for a search result, or None (any other host, namespace,
    query, sub-page or path, including everything under /w/ and /api/). The fragment is dropped."""
    try:
        sp = urlsplit(str(url or "").strip())
        port = sp.port
    except ValueError:
        return None
    if sp.scheme != "https" or (sp.hostname or "").lower() != str(cfg["article_host"]).lower() \
            or sp.username or sp.password or port not in (None, 443) or sp.query:
        return None
    path = sp.path or ""
    if not _ARTICLE_PATH.fullmatch(path) or ":" in unquote(path):
        return None
    return f"https://{str(cfg['article_host']).lower()}{path}"


_MAX_UNWRAP = 3  # nested Wayback snapshots unwrapped at most this deep (a format bound, not a tunable)


def unwrap_archive(url: str) -> str | None:
    """A web.archive.org snapshot -> the original URL it archives, unwrapping nested snapshots up to _MAX_UNWRAP deep
    (r2: the inner URL is re-checked every time); a URL that is, or still points into, any archive mirror after that
    -> None (dropped, fail closed); a URL with whitespace or control characters -> None; every other URL unchanged.
    Gates and fetches then see the ORIGINAL page, never an archive."""
    u = str(url or "").strip()
    for _ in range(_MAX_UNWRAP + 1):
        if not u or re.search(r"[\s\x00-\x1f\x7f]", u):
            return None
        m = _WAYBACK.match(u)
        if not m:
            break
        u = m.group(1).strip()
    else:
        return None  # nested deeper than the bound
    return None if (_WAYBACK.match(u) or _MIRRORS.search(_host(u))) else u


# ------------------------------------------------------------------------------------------------ COinS
class _CoinsParser(HTMLParser):
    """Collects the title attribute of every <span class="... Z3988 ..."> (COinS). Attributes only: there is no
    handle_data, so no text node of the page is ever read. Bounded by max_refs_scanned."""

    def __init__(self, limit: int):
        super().__init__(convert_charrefs=True)
        self.coins: list[str] = []
        self.limit = limit

    def handle_starttag(self, tag, attrs):
        if tag != "span" or len(self.coins) >= self.limit:
            return
        a = dict(attrs)
        if "Z3988" in str(a.get("class") or "").split() and a.get("title"):
            self.coins.append(str(a["title"]))


def _clean(s, n: int = _MAX_TITLE_CHARS) -> str:
    s = " ".join(_CTRL.sub(" ", str(s or "")).split())
    return s[:n]


def parse_coins(title_attr: str) -> dict | None:
    """One COinS ContextObject -> {"title", "venue", "genre", "ids": {doi?, pmcid?, pmid?, arxiv?}, "url"?} or None.
    Only OpenURL keys are read; values are clipped and stripped of control characters."""
    try:
        pairs = parse_qsl(str(title_attr or ""), keep_blank_values=False, max_num_fields=200)
    except ValueError:
        return None
    kv: dict = {}
    rft_ids: list = []
    for k, v in pairs:
        if k == "rft_id":
            rft_ids.append(v.strip())
        elif k.startswith("rft.") and k not in kv:
            kv[k] = v
    title = _clean(kv.get("rft.atitle") or kv.get("rft.btitle") or kv.get("rft.title") or "")
    venue = _clean(kv.get("rft.jtitle") or (kv.get("rft.btitle") if kv.get("rft.atitle") else "") or "")
    ids, url = {}, None
    for r in rft_ids:
        low = r.lower()
        if low.startswith("info:doi/") and norm_doi(r):
            ids.setdefault("doi", norm_doi(r))
        elif low.startswith("info:pmid/") and norm_pmid(r[10:]):
            ids.setdefault("pmid", norm_pmid(r[10:]))
        elif low.startswith("info:arxiv/") and norm_arxiv(r[11:]):
            ids.setdefault("arxiv", norm_arxiv(r[11:]))
        elif low.startswith(("http://", "https://")):
            if re.search(r"[\s\x00-\x1f\x7f]", r):
                continue  # r2: a URL with whitespace or control characters is malformed: ignored
            got = _ids_from_url(r)
            for k, v in got.items():
                ids.setdefault(k, v)
            if not got and url is None:
                url = r
    if not ids and not url:
        return None
    rec = {"title": title, "venue": venue, "genre": _clean(kv.get("rft.genre") or "", 40), "ids": ids}
    if url:
        rec["url"] = url
    return rec


def extract_references(html: str, limit: int) -> list:
    """Article HTML -> [reference records] in page order, from COinS only. Nothing else of the page leaves."""
    p = _CoinsParser(limit)
    try:
        p.feed(str(html or ""))
        p.close()
    except Exception:  # noqa: BLE001  malformed markup: what was parsed so far stands
        pass
    out = []
    for t in p.coins:
        rec = parse_coins(t)
        if rec is not None:
            out.append(rec)
    return out


# ------------------------------------------------------------------------------------------------ relevance
def ref_match(question: str, rec: Mapping, cfg: Mapping) -> tuple[int, int]:
    """(matched, total) question content stems found in the reference's OWN title and venue. Content stems: words of
    3+ characters, minus strip_terms, the news/time cue terms and years (LINKS-VERIFY-1107 finding 3), cut to stem_len.
    A stem matches a title/venue word with the same prefix, or a word that contains (or is contained in) a question
    word when both have at least min_contain_len characters (nucleosynthesis / synthesis). Wikipedia prose is never
    an input."""
    r = cfg["relevance"]
    n, mc = int(r["stem_len"]), int(r.get("min_contain_len", 6))
    drop = {str(t).lower() for t in (cfg["query"].get("strip_terms") or ())} | \
        {str(t).lower() for t in (r.get("cue_terms") or ())}
    qwords = {w for w in _WORD.findall(str(question or "").lower())
              if len(w) > 2 and w not in drop and not _YEAR.match(w)}
    rwords = {w for w in _WORD.findall((str(rec.get("title") or "") + " " + str(rec.get("venue") or "")).lower())
              if len(w) > 2}
    rstems = {w[:n] for w in rwords}
    qstems: dict = {}
    for w in qwords:
        qstems.setdefault(w[:n], set()).add(w)
    hit = sum(1 for stem, ws in qstems.items()
              if stem in rstems or any(len(a) >= mc and len(b) >= mc and (a in b or b in a) for a in ws for b in rwords))
    return hit, len(qstems)


def relevant(match: tuple[int, int], cfg: Mapping) -> bool:
    """Kept when at least min_hits question stems match (all of them when the question has fewer)."""
    hit, total = match
    return total > 0 and hit >= min(total, max(1, int(cfg["relevance"]["min_hits"])))


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


def to_url_candidate(url: str, title: str = "", rank: int = 0) -> dict | None:
    """The ORIGINAL source URL (archive unwrapped) as a URL-only candidate, the shape of
    search_providers._ddg_url_cand: never fetched here; the web route and follow_url take it through their gates.
    https only, no Wikimedia-family host, no archive mirror. r2: the title is the URL's HOST, never the editor-written
    COinS title (no editor text reaches a prompt or the UI through the candidate); `title` is accepted and ignored."""
    from vera.search_providers import safe_target_url
    orig = unwrap_archive(url)
    if not orig or _WIKI_FAMILY.search(_host(orig)):
        return None
    safe = safe_target_url(orig)
    if not safe:
        return None
    return {"url": safe, "title": _host(safe), "snippet": "", "doi": None, "provider": SOURCE,
            "acq_source": SOURCE, "portal": PORTAL, "portal_rank": rank, "ref_id_kind": "url",
            "discovered_url": safe}


def build_query(question: str, cfg: Mapping) -> str:
    q = cfg["query"]
    strip = {str(t).lower() for t in q.get("strip_terms") or ()}
    strip |= {str(t).lower() for t in ((cfg.get("relevance") or {}).get("cue_terms") or ())}
    from vera.search_query import drop_generic  # DDFIX-1023/REL: idioms such as "de facto", "status quo"
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", drop_generic(str(question or "")))
    keep = [w for w in words if w.lower() not in strip][: int(q["max_terms"])]
    return (" ".join(keep) if keep else " ".join(str(question or "").split()))[: int(q["max_query_chars"])]


# ------------------------------------------------------------------------------------------------ guarded GET
def _guard_path(url: str) -> None:
    """Defence in depth: the portal never requests /w/ (action API, index.php), /api/ (REST) or Special: pages."""
    path = unquote(urlsplit(url).path or "")
    if path.startswith(("/w/", "/api/")) or "/w/" in path or not path.startswith("/wiki/") or ":" in path:
        raise PortalError("path")


def get_article(url: str, cfg: Mapping, *, http_get: Callable, url_check: Callable, clock: Callable,
                deadline: float | None) -> str:
    """One GET of an allowed article page: https, the article host only, public address, no redirect, text/html only,
    byte cap, wall-clock limit over the streamed body. Raises PortalError(code). Never retried."""
    _guard_path(url)
    if _host(url) != str(cfg["article_host"]).lower() or _host(url) not in {h.lower() for h in cfg["allowed_hosts"]}:
        raise PortalError("host")
    ok = url_check(url)
    if not (ok[0] if isinstance(ok, tuple) else ok):
        raise PortalError("host")
    timeout_s = float(cfg["timeout_s"])
    end = clock() + timeout_s if deadline is None else min(clock() + timeout_s, deadline)
    if clock() >= end:
        raise PortalError("deadline")
    max_bytes = int(cfg["max_response_bytes"])
    resp = None
    try:
        try:
            resp = http_get(url, headers={"User-Agent": str(cfg["user_agent"]), "Accept": "text/html"},
                            timeout=max(0.1, min(timeout_s, end - clock())), allow_redirects=False, stream=True)
        except Exception as e:  # noqa: BLE001
            raise PortalError("rate_limited" if "rate limit" in str(e).lower() else "unreachable") from None
        status = int(getattr(resp, "status_code", 0) or 0)
        if status == 429:
            raise PortalError("rate_limited")
        if status != 200:
            raise PortalError("unreachable")  # redirects included: never followed
        ctype = ((getattr(resp, "headers", None) or {}).get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "text/html":
            raise PortalError("unreachable")
        return _read_body(resp, max_bytes, end, clock).decode("utf-8", "replace")
    finally:
        try:
            if resp is not None and hasattr(resp, "close"):
                resp.close()
        except Exception:  # noqa: BLE001
            pass


def _read_body(resp, max_bytes: int, end: float, clock: Callable) -> bytes:
    """r2: the whole body must arrive before `end` (one overall deadline, not a per-read one). The reads run in a
    daemon worker; the caller waits at most until `end`, then closes the response (which unblocks a slow-drip read)
    and raises PortalError("deadline"). Byte cap enforced as the chunks arrive."""
    box: dict = {"chunks": [], "err": None, "done": False}
    stop = threading.Event()

    def pump():
        size = 0
        try:
            for chunk in resp.iter_content(16384):
                if stop.is_set():
                    return
                size += len(chunk)
                if size > max_bytes:
                    box["err"] = "too_large"
                    return
                box["chunks"].append(chunk)
        except Exception:  # noqa: BLE001
            box["err"] = "unreachable"
        finally:
            box["done"] = True

    t = threading.Thread(target=pump, name="wikiref-read", daemon=True)
    t.start()
    t.join(max(0.0, end - clock()))
    if t.is_alive() or not box["done"]:
        stop.set()
        try:
            resp.close()
        except Exception:  # noqa: BLE001
            pass
        raise PortalError("deadline")
    if box["err"]:
        raise PortalError(box["err"])
    if clock() > end:
        raise PortalError("deadline")
    return b"".join(box["chunks"])


def _api_get(url: str, params: dict, headers: dict, timeout_s: float):
    """The gated API call (as vera.portal_inspire): config/api_permissions.json entry, conditions, rate portal and
    User-Agent through vera.api_permissions.call, over the pinned public-only GET; never a redirect."""
    from vera import api_permissions
    from vera import selective_fetch as sf
    with api_permissions.purpose_scope("metadata", url):
        return api_permissions.call(sf.pinned_get, url, params=params, headers=headers, timeout=timeout_s,
                                    purpose="metadata", allow_redirects=False)


def parse_api_search(body: bytes, host: str, n: int) -> list:
    """MediaWiki list=search JSON -> [{"url": https://<host>/wiki/<Title>}] (titles only). Any shape error or an API
    error object (maxlag included) -> [] (fail closed)."""
    try:
        doc = json.loads(body.decode("utf-8"))
        hits = doc["query"]["search"]
    except Exception:  # noqa: BLE001
        return []
    out = []
    for h in hits if isinstance(hits, list) else []:
        t = h.get("title") if isinstance(h, Mapping) else None
        if isinstance(t, str) and t.strip() and h.get("ns", 0) == 0:
            out.append({"url": f"https://{host}/wiki/{quote(t.strip().replace(' ', '_'), safe='_()-,.')}"})
        if len(out) >= n:
            break
    return out


def api_search(query: str, host: str, n: int, cfg: Mapping, *, get: Callable | None = None) -> list:
    """SEARCHCAPS-1110: ONE Wikipedia search API call (cfg["search_api"]) -> article URLs. [] when the API is not
    configured or the endpoint is not on the article host (fail closed: no DuckDuckGo fallback). Raises on transport
    errors (discover counts them as a failed search)."""
    a = cfg.get("search_api") if isinstance(cfg, Mapping) else None
    if not isinstance(a, Mapping) or not a.get("endpoint"):
        return []
    ep = str(a["endpoint"])
    if urlsplit(ep).scheme != "https" or _host(ep) != str(host).lower():
        return []
    params = {**{str(k): str(v) for k, v in (a.get("params") or {}).items()},
              "srsearch": str(query), "srlimit": str(max(1, min(int(n), int(a.get("max_results", 10)))))}
    resp = (get or _api_get)(ep, params, {"User-Agent": str(cfg["user_agent"]), "Accept": "application/json"},
                             float(a.get("timeout_s", cfg["timeout_s"])))
    try:
        if int(getattr(resp, "status_code", 0) or 0) != 200:
            raise PortalError("rate_limited" if int(getattr(resp, "status_code", 0) or 0) == 429 else "unreachable")
        body = getattr(resp, "content", b"") or b""
        if len(body) > int(a.get("max_response_bytes", 200000)):
            raise PortalError("too_large")
        return parse_api_search(body, str(host).lower(), int(n))
    finally:
        try:
            if hasattr(resp, "close"):
                resp.close()
        except Exception:  # noqa: BLE001
            pass


def _default_site_search(query: str, host: str, n: int, cfg: Mapping | None = None) -> list:
    """SEARCHCAPS-1110: the Wikipedia search API (was a DuckDuckGo Lite site: query until OD-1110-SEARCHCAPS)."""
    return api_search(query, host, n, cfg if cfg is not None else load_config())


def _default_robots(http_get: Callable, url_check: Callable, ua: str) -> Callable:
    def ok(url: str) -> bool:
        from vera import selective_fetch as sf
        return sf.robots_allowed(url, http_get=http_get, url_check=url_check, user_agent=ua)
    return ok


def _crawl_ok(url: str, ua: str) -> bool:
    """robots Crawl-delay of the host (cached by robots_allowed): the request is never sent early."""
    from vera import selective_fetch as sf
    return sf.crawl_reserve(_host(url), sf.robots_crawl_delay(url, ua))


# ------------------------------------------------------------------------------------------------ portal
def discover(question: str, cfg: Mapping | None = None, *, site_search: Callable | None = None,
             http_get: Callable | None = None, url_check: Callable | None = None, robots_ok: Callable | None = None,
             clock: Callable = time.monotonic, deadline: float | None = None,
             max_refs: int | None = None) -> tuple[list, list, dict]:
    """-> (candidates, [], counts). Never raises: a failure returns what was gathered so far plus counts["error"].
    The second element (reading-list links) is always empty since WIKI-REDESIGN-1107 (no news dump). Counts carry
    numbers and fixed codes only (no titles, no question, no URL)."""
    cfg = load_config() if cfg is None else cfg
    counts = {"search": 0, "articles_found": 0, "robots_denied": 0, "articles": 0, "refs": 0, "relevant": 0,
              "candidates": 0, "candidates_url": 0, "held_pmid_only": 0, "held_url": 0, "duplicates": 0}
    if not enabled(cfg):
        counts["error"] = "disabled"
        return [], [], counts
    ua = str(cfg["user_agent"])
    if http_get is None or url_check is None:
        from vera import selective_fetch as sf
        http_get = http_get or sf.pinned_get
        url_check = url_check or sf.is_public_url
    robots_ok = robots_ok or _default_robots(http_get, url_check, ua)
    max_articles = int(cfg["max_articles"])
    try:
        counts["search"] = 1
        if site_search is not None:
            results = list(site_search(build_query(question, cfg), str(cfg["article_host"]),
                                       max(max_articles * 3, 5)) or [])
        else:
            results = _default_site_search(build_query(question, cfg), str(cfg["article_host"]),
                                           max(max_articles * 3, 5), cfg)
    except Exception:  # noqa: BLE001  a failed search ends the portal for this question (the other providers stand)
        counts["error"] = "search"
        results = []
    arts = []
    for r in results:
        u = article_url((r or {}).get("url") if isinstance(r, Mapping) else None, cfg)
        if u and u not in arts:
            arts.append(u)
    arts = arts[:max_articles]
    counts["articles_found"] = len(arts)
    refs = []
    scan_limit = int(cfg["relevance"].get("max_refs_scanned", 500))
    for i, url in enumerate(arts):
        if deadline is not None and clock() >= deadline:
            counts["error"] = "deadline"
            break
        try:
            allowed = bool(robots_ok(url))
        except Exception:  # noqa: BLE001  fail closed
            allowed = False
        if not allowed:
            counts["robots_denied"] += 1
            continue
        try:
            if not _crawl_ok(url, ua):
                counts["error"] = "crawl_delay"
                continue
        except Exception:  # noqa: BLE001
            counts["error"] = "crawl_delay"
            continue
        try:
            html = get_article(url, cfg, http_get=http_get, url_check=url_check, clock=clock, deadline=deadline)
        except PortalError as e:
            counts["error"] = str(e)
            continue
        counts["articles"] += 1
        for rec in extract_references(html, scan_limit):
            rec["article"] = i
            refs.append(rec)
        html = None  # the page is discarded as soon as its reference metadata is out
    counts["refs"] = len(refs)
    scored = []
    for k, rec in enumerate(refs):
        m = ref_match(question, rec, cfg)
        if relevant(m, cfg):
            scored.append((round(m[0] / m[1], 6), k, rec))
    counts["relevant"] = len(scored)
    scored.sort(key=lambda t: (-t[0], t[1]))
    cap = int(cfg["max_refs"]) if max_refs is None else min(int(cfg["max_refs"]), int(max_refs))
    out, seen = [], set()
    for s, _k, rec in scored:
        if len(out) >= cap:
            break
        ids = rec["ids"]
        keys = {f"{a}:{b}" for a, b in ids.items()}
        if ids:
            if keys & seen:
                counts["duplicates"] += 1
                continue
            cand = to_candidate(ids, len(out))
            if cand is None:
                counts["held_pmid_only"] += 1
                continue
        else:
            cand = to_url_candidate(rec.get("url") or "", rank=len(out))
            if cand is None:
                counts["held_url"] += 1
                continue
            keys = {"url:" + cand["url"].lower().rstrip("/")}
            if keys & seen:
                counts["duplicates"] += 1
                continue
            counts["candidates_url"] += 1
        seen |= keys
        cand["ref_score"] = s
        out.append(cand)
    counts["candidates"] = len(out)
    logger.info("autoacq wikiref %s", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return out, [], counts


_LINK_SINK: dict = {}  # NT-3 API kept for lane_evidence.merge_wiki_links; the portal no longer records links
_LINK_SINK_MAX = 32
_LINK_LOCK = threading.Lock()


def record_links(question: str, links: list) -> None:
    """NT-3 API (kept for callers and tests). WIKI-REDESIGN-1107: discover never calls it, so no Wikipedia-citation
    reading list reaches the news lane; followed sources appear in the normal lanes under their own type."""
    try:
        key = " ".join(str(question or "").split()).lower()
        if not key or not links:
            return
        with _LINK_LOCK:
            while len(_LINK_SINK) >= _LINK_SINK_MAX:
                _LINK_SINK.pop(next(iter(_LINK_SINK)), None)
            _LINK_SINK[key] = list(links)
    except Exception:  # noqa: BLE001
        logger.debug("autoacq wikiref record_links failed", exc_info=True)


def take_links(question: str) -> list:
    """Pop (once) the links recorded for this question; [] when none."""
    with _LINK_LOCK:
        return _LINK_SINK.pop(" ".join(str(question or "").split()).lower(), None) or []


def make_portal_search(cfg: Mapping | None = None, *, http_get: Callable | None = None,
                       url_check: Callable | None = None, clock: Callable = time.monotonic,
                       site_search: Callable | None = None, robots_ok: Callable | None = None) -> Callable:
    """A search(question, n, timeout_s) -> [candidates] callable for search_providers.acquisition_search's pool. The
    deadline is now + timeout_s; n caps the emitted candidates together with max_refs."""
    def search(question, n, timeout_s):
        cands, _links, _counts = discover(question, cfg, site_search=site_search, http_get=http_get,
                                          url_check=url_check, robots_ok=robots_ok, clock=clock,
                                          deadline=clock() + float(timeout_s), max_refs=n or None)
        return cands
    return search
