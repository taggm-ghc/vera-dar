"""ROUTES-1108 slice 1: generic REFERENCE PORTAL (plan work/ROUTES-PLAN-1108/plan.md section 3).

Owner 2026-11-08: "most sites should be treated as portals if through those sites we find links to orginal sources".
Owner 2026-10-22 / 2026-11-07: a portal is never evidence; only its links to original sources are followed.

A portal page yields ONLY identifiers and approved outbound links. Its prose is never read: the page parser
(_LinkParser) implements handle_starttag only (attributes of <a>, <link>, <meta> and COinS <span>), so no text node
reaches any variable. Nothing of a portal page is stored, scored on text, logged or returned except the identifiers.

Eligibility of a page (all must hold; default deny), config/reference_portals.json + config/portal_hosts.json:
  1. host not in exclusions.hosts (exclusion wins over any registry row);
  2. registry row for the exact host with status "active" and terms "permit"; https, port 443, no userinfo, no query,
     path full-matching paths_allow; rate_portal naming a config/rate_limits.json portal that lists the host;
     a host with no usable row is NOT fetched and is counted host_unchecked (the terms-reading queue);
  3. robots.txt allows the URL for our user agent, * and every AI-agent group, and states no AI block
     (vera.selective_fetch.robots_allowed: fail closed);
  4. no Content-Signal ai-input=no (an ai-train=no only host stays usable: D-P1, provisional);
  5. no TDM reservation: /.well-known/tdmrep.json rule (unreadable = reserved), TDM-Reservation header, or the
     tdm-reservation meta tag of the page itself (then the page's links are discarded).
Requests: public-address check, no redirect followed, text/html only, byte cap, one overall deadline, robots
Crawl-delay (crawl_reserve) and the host's rate portal (vera.rate_limit).

Adapters (special cases of the same flow): openalex_rw (OpenAlex referenced works of the seeds the acquisition search
already returned; ONE keyless API call at most), wikipedia_title (/wiki/<Title> built from the question; no on-wiki
search), generic_page (result pages already returned by DuckDuckGo and OpenAlex landing URLs).

Every candidate is an ordinary acquisition candidate (acq_source "reference_portal") that passes the unchanged
acquisition gates; it is acquired only as full text through a permitted route.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Mapping
from urllib.parse import quote, unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "reference_portals.json"
SOURCE = "reference_portal"
MODES = ("off", "shadow", "on")
_HANDLE_OK = re.compile(r"[0-9][0-9.]{0,20}/[A-Za-z0-9._\-]{1,64}")
_LANDING_PATH = re.compile(r"/[A-Za-z0-9._~/\-]{0,200}")
REQUIRED = ("mode", "user_agent", "host_registry", "caps", "exclusion_signals", "exclusions", "classifier",
            "relevance", "adapters")
REQUIRED_CAPS = ("max_pages_per_question", "max_pages_per_host", "max_links_per_page", "max_candidates_per_question",
                 "max_hrefs_scanned", "max_page_bytes", "timeout_s", "shadow_deadline_s", "tdmrep_max_bytes")
MUST_EXCLUDE = ("ai_input_no", "tdm_reservation", "tdmrep_unreadable_is_reserved", "robots_block", "terms_forbid")
logger = logging.getLogger("vera.autoacq")


class PortalConfigError(ValueError):
    pass


class Skip(Exception):
    """A page is not used; .code is a fixed reason code (never a URL or page text)."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


# ------------------------------------------------------------------------------------------------ config
def validate(c) -> dict:
    if not isinstance(c, dict) or any(k not in c for k in REQUIRED):
        raise PortalConfigError("missing keys")
    if c["mode"] not in MODES:
        raise PortalConfigError("mode")
    caps = c["caps"]
    if not isinstance(caps, dict) or any(k not in caps for k in REQUIRED_CAPS):
        raise PortalConfigError("caps")
    for k in REQUIRED_CAPS:
        v = caps[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            raise PortalConfigError(f"caps.{k}")
    sig = c["exclusion_signals"]
    if not isinstance(sig, dict) or any(sig.get(k) is not True for k in MUST_EXCLUDE):
        raise PortalConfigError("exclusion_signals")  # a loosened exclusion signal fails closed
    ex = c["exclusions"]
    if not isinstance(ex, dict) or not isinstance(ex.get("hosts"), list) or not ex["hosts"]:
        raise PortalConfigError("exclusions")
    ua = str(c["user_agent"])
    if not ua or "@" in ua:
        raise PortalConfigError("user_agent")  # no email anywhere in a User-Agent
    if not isinstance(c["adapters"], dict) or not isinstance(c["classifier"], dict):
        raise PortalConfigError("adapters")
    for p in c["classifier"].get("drop_host_patterns") or ():
        re.compile(p)
    return c


def load_config(path: Path | None = None) -> dict:
    """Raises PortalConfigError on a missing, unreadable or invalid file (callers treat that as mode off)."""
    try:
        return validate(json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8")))
    except PortalConfigError:
        raise
    except Exception as e:  # noqa: BLE001
        raise PortalConfigError(type(e).__name__) from None


def mode(cfg: Mapping | None = None) -> str:
    try:
        return str((cfg if cfg is not None else load_config())["mode"])
    except Exception:  # noqa: BLE001  unreadable config = off (no request)
        return "off"


def adapter_on(cfg: Mapping, name: str) -> bool:
    a = (cfg.get("adapters") or {}).get(name)
    return isinstance(a, dict) and a.get("enabled") is True


def inspire_permitted() -> bool:
    """ROUTES-1108-OFF guard: INSPIRE is called only when the portal mode is "on" (checked by the caller), the adapter
    is enabled (adapter_on) AND the api_permissions entry's owner_switch is "permit". Any error = False (no call)."""
    try:
        from vera import api_permissions as ap
        pol = ap.get_policy()
        return pol.error is None and pol.entries["inspire_literature"].get("owner_switch") == "permit"
    except Exception:  # noqa: BLE001
        return False


def _rate_hosts() -> dict:
    """portal name -> set of hosts from config/rate_limits.json (empty on any error: every row then invalid)."""
    try:
        doc = json.loads((ROOT / "config" / "rate_limits.json").read_text(encoding="utf-8"))
        return {k: {str(h).lower() for h in (v.get("hosts") or ())} for k, v in (doc.get("portals") or {}).items()
                if isinstance(v, dict)}
    except Exception:  # noqa: BLE001
        return {}


def load_hosts(cfg: Mapping, path: Path | None = None, rate_hosts: Mapping | None = None) -> dict:
    """host -> usable row (status active, terms permit, valid paths_allow and rate portal). Invalid rows are dropped
    (deny). Any error reading the file -> {} (everything denied)."""
    try:
        doc = json.loads(Path(path or (ROOT / str(cfg["host_registry"]))).read_text(encoding="utf-8"))
        rows = doc["entries"]
    except Exception:  # noqa: BLE001
        return {}
    rh = _rate_hosts() if rate_hosts is None else rate_hosts
    out = {}
    for r in rows if isinstance(rows, list) else ():
        try:
            host = str(r["host"]).lower()
            if r.get("status") != "active" or r.get("terms") != "permit":
                continue
            rx = re.compile(str(r["paths_allow"]))
            portal = r.get("rate_portal")
            if not portal or host not in rh.get(str(portal), set()):
                continue  # no rate portal covering the host: never requested
            out[host] = {"host": host, "paths": rx, "rate_portal": str(portal)}
        except Exception:  # noqa: BLE001  a malformed row is ignored (deny)
            continue
    return out


def _host(url: str) -> str:
    try:
        return (urlsplit(str(url or "")).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def excluded(host: str, cfg: Mapping) -> bool:
    h = str(host or "").lower().rstrip(".")
    for e in (cfg.get("exclusions") or {}).get("hosts") or ():
        e = str(e).lower().strip(".")
        if e and (h == e or h.endswith("." + e)):
            return True
    return False


def page_row(url: str, cfg: Mapping, hosts: Mapping) -> dict:
    """The registry row allowing this exact URL as a portal page. Raises Skip(code)."""
    try:
        sp = urlsplit(str(url or "").strip())
        port = sp.port
    except ValueError:
        raise Skip("bad_url") from None
    host = (sp.hostname or "").lower().rstrip(".")
    if not host:
        raise Skip("bad_url")
    if excluded(host, cfg):
        raise Skip("excluded_host")
    row = hosts.get(host)
    if row is None:
        raise Skip("host_unchecked")
    if sp.scheme != "https" or sp.username or sp.password or port not in (None, 443) or sp.query:
        raise Skip("bad_url")
    raw = sp.path or ""
    if not row["paths"].fullmatch(raw) or not row["paths"].fullmatch(unquote(raw)):
        raise Skip("path_not_allowed")  # SECREV M1: /wiki/User%3AX decodes to a namespace page: both forms must match
    return row


# ------------------------------------------------------------------------------------------------ parsing
class _LinkParser(HTMLParser):
    """Attributes only: href of <a>/<link>, citation_* identifier metas, the tdm-reservation meta and COinS span titles.
    There is deliberately NO handle_data / handle_comment / handle_entityref: no text node of the page is ever read."""

    _META_IDS = {"citation_doi": "doi", "dc.identifier": "doi", "citation_pmid": "pmid", "citation_arxiv_id": "arxiv"}

    def __init__(self, max_hrefs: int):
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []
        self.coins: list[str] = []
        self.meta_ids: list[tuple[str, str]] = []
        self.tdm_reserved = False
        self.max = int(max_hrefs)

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs if isinstance(k, str)}
        if tag in ("a", "link") and a.get("href") and len(self.hrefs) < self.max:
            self.hrefs.append(a["href"].strip())
        elif tag == "meta":
            name = a.get("name", "").strip().lower()
            if name == "tdm-reservation" and a.get("content", "").strip() == "1":
                self.tdm_reserved = True
            elif name in self._META_IDS and a.get("content") and len(self.meta_ids) < self.max:
                self.meta_ids.append((self._META_IDS[name], a["content"].strip()))
        elif tag == "span" and "Z3988" in a.get("class", "").split() and a.get("title") and len(self.coins) < self.max:
            self.coins.append(a["title"])

    handle_startendtag = handle_starttag


def _landing_hosts() -> set:
    try:
        doc = json.loads((ROOT / "config" / "auto_acquire.json").read_text(encoding="utf-8"))
        return {str(h).lower() for h in ((doc.get("landing_route") or {}).get("publisher_hosts") or ())}
    except Exception:  # noqa: BLE001
        return set()


def _doi_excluded(doi: str, cfg: Mapping) -> bool:
    """SECREV L5: DOIs under a prefix of an excluded publisher (bioRxiv/medRxiv 10.1101, OD-7) never become candidates."""
    d = str(doi or "").lower()
    return any(d.startswith(str(p).lower().rstrip("/") + "/") for p in (cfg["exclusions"].get("doi_prefixes") or ()))


def classify(href: str, page_url: str, cfg: Mapping, landing_hosts: set) -> dict | None:
    """One href -> {"ids": {...}} or {"url": ..., "kind": "handle"|"landing"} or None (dropped: deny-all)."""
    from vera import search_providers as sp
    from vera import wiki_ref_portal as w
    try:
        u = urljoin(page_url, str(href or "").strip())
    except ValueError:
        return None
    if not u.lower().startswith(("http://", "https://")) or re.search(r"[\s\x00-\x1f\x7f]", u):
        return None
    if "web.archive.org" in u.lower():
        u = w.unwrap_archive(u) or ""
        if not u:
            return None
    host = _host(u)
    if not host or host == _host(page_url) and not sp.ddg_identifier(u):
        return None  # same-site navigation is never a reference (unless it literally carries an identifier)
    if excluded(host, cfg) or any(re.search(p, host) for p in (cfg["classifier"].get("drop_host_patterns") or ())):
        return None
    kinds = set(cfg["classifier"].get("kinds") or ())
    ident = sp.ddg_identifier(u)
    if ident:
        k, v = ident["kind"], ident["value"]
        if k not in kinds:
            return None
        if k == "handle":
            if not _HANDLE_OK.fullmatch(str(v)):
                return None  # SECREV L3: a handle suffix is an identifier, never free text
            safe = sp.safe_target_url(f"https://hdl.handle.net/{v}")
            return {"url": safe, "kind": "handle"} if safe else None
        norm = {"doi": w.norm_doi, "pmcid": w.norm_pmcid, "pmid": w.norm_pmid, "arxiv": w.norm_arxiv}[k](v)
        if k == "doi" and norm and _doi_excluded(norm, cfg):
            return None
        return {"ids": {k: norm}} if norm else None
    if "landing" in kinds and host in landing_hosts:
        try:
            lp = urlsplit(u)
        except ValueError:
            return None
        if lp.query or lp.fragment or not _LANDING_PATH.fullmatch(lp.path or "/"):
            return None  # SECREV L3: landing URLs carry a plain path only (no free text in a query)
        safe = sp.safe_target_url(u)
        return {"url": safe, "kind": "landing"} if safe else None
    return None


def extract(html: str, page_url: str, question: str, cfg: Mapping, landing_hosts: set | None = None) -> list:
    """Page HTML -> ordered reference records [{"ids"|"url", "kind", "score"}]. Raises Skip("tdm_reservation") when
    the page's own meta reserves TDM. COinS records are ordered by their own title/venue match to the question (never
    page prose); plain links keep page order after them. Capped at caps.max_links_per_page."""
    from vera import wiki_ref_portal as w
    caps = cfg["caps"]
    p = _LinkParser(int(caps["max_hrefs_scanned"]))
    try:
        p.feed(str(html or ""))
        p.close()
    except Exception:  # noqa: BLE001  malformed markup: what was parsed so far stands
        pass
    if p.tdm_reserved:
        raise Skip("tdm_reservation")
    lh = _landing_hosts() if landing_hosts is None else landing_hosts
    rcfg = {"relevance": cfg["relevance"], "query": _wiki_query()}
    recs: list = []
    for t in p.coins:
        rec = w.parse_coins(t)
        if rec is None:
            continue
        hit, total = w.ref_match(question, rec, rcfg)
        score = hit / total if total else 0.0
        ids = {k: v for k, v in (rec.get("ids") or {}).items() if v}
        if ids:
            recs.append({"ids": ids, "kind": "coins", "score": score})
        elif rec.get("url"):
            c = classify(rec["url"], page_url, cfg, lh)
            if c:
                recs.append({**c, "score": score})
    recs.sort(key=lambda r: -r["score"])  # stable: page order within equal scores
    for kind, value in p.meta_ids:
        norm = {"doi": w.norm_doi, "pmid": w.norm_pmid, "arxiv": w.norm_arxiv}[kind](value)
        if norm:
            recs.append({"ids": {kind: norm}, "kind": "meta", "score": 0.0})
    for h in p.hrefs:
        c = classify(h, page_url, cfg, lh)
        if c:
            recs.append({**c, "score": 0.0})
    return recs[: int(caps["max_links_per_page"])]


_WQ: dict = {}


def _wiki_query() -> dict:
    """The strip-term list shared with the Wikipedia portal config (question words, never page text)."""
    if not _WQ:
        try:
            from vera import wiki_ref_portal as w
            _WQ.update(w.load_config()["query"])
        except Exception:  # noqa: BLE001
            _WQ.update({"strip_terms": [], "max_terms": 8, "max_query_chars": 300})
    return dict(_WQ)


# ------------------------------------------------------------------------------------------------ candidates
def to_candidate(rec: Mapping, rank: int, adapter: str) -> dict | None:
    """A reference record -> an ordinary acquisition candidate. Only identifiers, the target URL and fixed labels:
    never any portal page text (title is "" or the target host)."""
    from vera import wiki_ref_portal as w
    if rec.get("ids"):
        c = w.to_candidate(rec["ids"], rank)
    elif rec.get("url"):
        c = w.to_url_candidate(rec["url"], rank=rank)
        if c is not None:
            c["ref_id_kind"] = rec.get("kind") or "url"
    else:
        c = None
    if c is None:
        return None
    c.update({"provider": f"portal:{adapter}", "acq_source": SOURCE, "portal": adapter, "title": c.get("title") or "",
              "snippet": ""})
    return c


def _keys(c: Mapping) -> set:
    from vera import work_key
    try:
        return set(work_key.keys(dict(c)))
    except Exception:  # noqa: BLE001
        return {"url:" + str(c.get("url") or "").lower()}


# ------------------------------------------------------------------------------------------------ fetching
@dataclass
class Deps:
    http_get: Callable | None = None
    url_check: Callable | None = None
    robots_ok: Callable | None = None       # (url, ua) -> bool
    no_ai_input: Callable | None = None     # (url, ua) -> bool   (Content-Signal ai-input=no of the host)
    ai_train_no: Callable | None = None     # (url, ua) -> bool   (Content-Signal ai-train=no; used only if configured)
    tdmrep_ok: Callable | None = None       # (url, rate_portal) -> bool (False = reserved or unreadable)
    crawl_ok: Callable | None = None        # (url, ua) -> bool
    limited: Callable | None = None         # (rate_portal, fn) -> fn
    openalex_lookup: Callable | None = None  # (seed ids, per_page, timeout_s) -> [OpenAlex-parsed records]


def _default_deps(cfg: Mapping, d: Deps | None) -> Deps:
    from vera import rate_limit, selective_fetch as sf
    d = d or Deps()
    http_get = d.http_get or sf.pinned_get
    url_check = d.url_check or sf.is_public_url
    lim = d.limited or (lambda portal, fn: rate_limit.limited(portal, fn))
    return Deps(
        http_get=http_get, url_check=url_check,
        robots_ok=d.robots_ok or (lambda u, ua: sf.robots_allowed(u, http_get=http_get, url_check=url_check,
                                                                  user_agent=ua)),
        no_ai_input=d.no_ai_input or (lambda u, ua: not _robots_cached(u, ua) or sf.robots_no_ai(u, ua)),
        ai_train_no=d.ai_train_no or (lambda u, ua: not _robots_cached(u, ua) or _signal_says(sf.robots_content_signal(u, ua),
                                                                                         "ai-train", "no")),
        tdmrep_ok=d.tdmrep_ok or (lambda u, portal: _tdmrep_ok(u, cfg, http_get, url_check, limited=lim,
                                                                 rate_portal=portal)),
        crawl_ok=d.crawl_ok or (lambda u, ua: _robots_cached(u, ua)
                                and sf.crawl_reserve(_host(u), sf.robots_crawl_delay(u, ua))),
        limited=lim,
        openalex_lookup=d.openalex_lookup or _default_openalex_lookup)


def _robots_cached(url: str, ua: str) -> bool:
    """SECREV L4: the parsed robots.txt robots_allowed just cached for this host is still present. A miss (evicted by
    another thread) means the Content-Signal and Crawl-delay cannot be read: callers fail closed."""
    from vera import selective_fetch as sf
    sp_ = urlsplit(url)
    with sf._ROBOTS_LOCK:
        return (sp_.scheme, (sp_.hostname or "").lower(), ua) in sf._ROBOTS


def _signal_says(signal: str, key: str, value: str) -> bool:
    for part in re.split(r"[,\s]+", str(signal or "").lower()):
        k, _, v = part.partition("=")
        if k.strip() == key and v.strip() == value:
            return True
    return False


_TDM: dict = {}
_TDM_LOCK = threading.Lock()
_TDM_TTL_S = 3600.0  # same order as the robots cache TTLs (a format/cache bound, not a decision threshold)


def _tdmrep_ok(url: str, cfg: Mapping, http_get: Callable, url_check: Callable, clock: Callable = time.monotonic,
               limited: Callable | None = None, rate_portal: str | None = None) -> bool:
    """False when /.well-known/tdmrep.json reserves the URL's path, or cannot be read (fail closed). 404/410 = no
    reservation. Cached per host. One GET through the host's rate portal (SECREV L2), no redirect, byte cap and one
    overall deadline on the body (SECREV L1), response always closed."""
    from vera import public_fetch, wiki_ref_portal as w
    host = _host(url)
    now = clock()
    with _TDM_LOCK:
        hit = _TDM.get(host)
    if hit is None or now - hit[0] > _TDM_TTL_S:
        rules = "unreadable"
        try:
            t = f"https://{host}/.well-known/tdmrep.json"
            ok = url_check(t)
            if ok[0] if isinstance(ok, tuple) else ok:
                get = limited(rate_portal, http_get) if (limited and rate_portal) else http_get
                end = clock() + float(cfg["caps"]["timeout_s"])
                resp = get(t, headers={"User-Agent": str(cfg["user_agent"]), "Accept": "application/json"},
                           timeout=float(cfg["caps"]["timeout_s"]), allow_redirects=False, stream=True)
                try:
                    st = int(getattr(resp, "status_code", 0) or 0)
                    if st in (404, 410):
                        rules = []
                    elif st == 200:
                        body = w._read_body(resp, int(cfg["caps"]["tdmrep_max_bytes"]), end, clock)
                        rules = json.loads(body.decode("utf-8"))
                        public_fetch.tdmrep_reserved(rules, "/")  # shape check; ValueError -> unreadable
                finally:
                    try:
                        resp.close()
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            rules = "unreadable"
        with _TDM_LOCK:
            _TDM[host] = (now, rules)
            while len(_TDM) > 256:
                _TDM.pop(next(iter(_TDM)))
        hit = (now, rules)
    rules = hit[1]
    if rules == "unreadable":
        return not cfg["exclusion_signals"]["tdmrep_unreadable_is_reserved"]
    try:
        return not public_fetch.tdmrep_reserved(rules, urlsplit(url).path or "/")
    except Exception:  # noqa: BLE001
        return False


def fetch_page(url: str, row: Mapping, cfg: Mapping, deps: Deps, clock: Callable, deadline: float | None) -> str:
    """Every gate in order, then ONE GET. Raises Skip(code). Never retried."""
    from vera import acq_discovered, wiki_ref_portal as w
    ua = str(cfg["user_agent"])
    ok = deps.url_check(url)
    if not (ok[0] if isinstance(ok, tuple) else ok):
        raise Skip("not_public")
    try:
        allowed = bool(deps.robots_ok(url, ua))
    except Exception:  # noqa: BLE001
        allowed = False
    if not allowed:
        raise Skip("robots_block")
    try:
        no_ai = bool(deps.no_ai_input(url, ua))
    except Exception:  # noqa: BLE001
        no_ai = True
    if no_ai:
        raise Skip("ai_input_no")
    if cfg["exclusion_signals"].get("ai_train_no_only") is True:  # D-P1 (provisional): false = ai-train=no allowed
        try:
            train_no = bool(deps.ai_train_no(url, ua))
        except Exception:  # noqa: BLE001
            train_no = True
        if train_no:
            raise Skip("ai_train_no")
    try:
        tdm_ok = bool(deps.tdmrep_ok(url, row["rate_portal"]))
    except Exception:  # noqa: BLE001
        tdm_ok = False
    if not tdm_ok:
        raise Skip("tdm_reservation")
    try:
        if not deps.crawl_ok(url, ua):
            raise Skip("crawl_delay")
    except Skip:
        raise
    except Exception:  # noqa: BLE001
        raise Skip("crawl_delay") from None
    timeout_s = float(cfg["caps"]["timeout_s"])
    end = clock() + timeout_s if deadline is None else min(clock() + timeout_s, deadline)
    if clock() >= end:
        raise Skip("deadline")
    get = deps.limited(row["rate_portal"], deps.http_get)
    resp = None
    try:
        try:
            resp = get(url, headers={"User-Agent": ua, "Accept": "text/html"},
                       timeout=max(0.1, min(timeout_s, end - clock())), allow_redirects=False, stream=True)
        except Exception as e:  # noqa: BLE001
            raise Skip("rate_limited" if "rate limit" in str(e).lower() else "unreachable") from None
        status = int(getattr(resp, "status_code", 0) or 0)
        if status == 429:
            raise Skip("rate_limited")
        if status != 200:
            raise Skip("unreachable")  # redirects included: never followed
        headers = getattr(resp, "headers", None) or {}
        if acq_discovered.tdm_header_reserved(headers):
            raise Skip("tdm_reservation")
        ctype = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype not in ("text/html", "application/xhtml+xml"):
            raise Skip("not_html")
        try:
            return w._read_body(resp, int(cfg["caps"]["max_page_bytes"]), end, clock).decode("utf-8", "replace")
        except w.PortalError as e:
            raise Skip(str(e)) from None
    finally:
        try:
            if resp is not None and hasattr(resp, "close"):
                resp.close()
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------------------------------------ adapters
def wikipedia_title_urls(question: str, cfg: Mapping) -> list:
    """[alternative URLs] for ONE page: /wiki/<Title> from the question's content terms, as-is then naive singular.
    Question words only; no search request."""
    from vera import wiki_ref_portal as w
    a = cfg["adapters"]["wikipedia_title"]
    q = _wiki_query()
    terms = w.build_query(question, {"query": q, "relevance": {"cue_terms": []}}).split()
    if not terms or terms == str(question or "").split():
        return []  # nothing but stop words: no title
    title = "_".join(terms)
    title = title[:1].upper() + title[1:]
    alts = [title]
    last = terms[-1]
    if len(last) > 3 and last.lower().endswith("s") and not last.lower().endswith("ss"):
        alts.append(title[:-1])
    host = str(a["host"]).lower()
    return [f"https://{host}/wiki/{quote(t, safe='_()-,.')}" for t in alts[: int(a["max_title_tries"])]]


def generic_page_urls(ddg_raw, base, cfg: Mapping) -> list:
    src = set((cfg["adapters"]["generic_page"].get("sources") or ()))
    out = []
    if "ddg" in src:
        out += [str(r.get("url") or "") for r in (ddg_raw or ()) if isinstance(r, Mapping)]
    if "openalex_landing" in src:
        for c in base or ():
            if not isinstance(c, Mapping):
                continue
            oa = c.get("oa") if isinstance(c.get("oa"), Mapping) else {}
            for u in (c.get("url"), oa.get("landing_page_url")):
                if isinstance(u, str) and not re.match(r"^https?://(dx\.)?doi\.org/", u, re.I):
                    out.append(u)
    return [u for u in out if u]


_OA_CACHE: dict = {}
_OA_LOCK = threading.Lock()


def openalex_rw(base, cfg: Mapping, deps: Deps, counts: dict, clock: Callable, deadline: float | None) -> list:
    """Referenced works of the OpenAlex seeds the search already returned: ONE keyless call at most, cached."""
    from vera import search_providers as sp
    a = cfg["adapters"]["openalex_rw"]
    seeds = []
    for c in base or ():
        wid = str((c or {}).get("openalex_id") or "").rsplit("/", 1)[-1] if isinstance(c, Mapping) else ""
        if re.fullmatch(r"W\d{1,12}", wid) and wid not in seeds:
            seeds.append(wid)
    seeds = seeds[: int(a["max_seeds"])] if a.get("or_seeds") is True else seeds[:1]
    if not seeds or int(a["max_calls_per_question"]) < 1:
        return []
    key = "|".join(sorted(seeds))
    now = clock()
    with _OA_LOCK:
        hit = _OA_CACHE.get(key)
    if hit is not None and now - hit[0] < float(a["cache_ttl_s"]):
        counts["openalex_cache_hit"] = 1
        recs = hit[1]
    else:
        if sp._QUOTA.blocked(now):
            counts["openalex_skipped_quota"] = 1
            return []
        left = float(cfg["caps"]["timeout_s"]) if deadline is None else min(float(cfg["caps"]["timeout_s"]),
                                                                              deadline - clock())
        if left <= 0.5:
            counts["openalex_deadline"] = 1
            return []
        counts["openalex_calls"] = 1
        try:
            recs = list(deps.openalex_lookup(seeds, int(a["per_page"]), left) or [])
        except Exception as e:  # noqa: BLE001  never retried in this question
            if sp._is_quota_error(e) and not sp._is_rate_skip(e):
                sp._QUOTA.block(clock(), float(_openalex_cooldown()))
                counts["openalex_429"] = 1
            else:
                counts["openalex_failed"] = 1
            return []
        with _OA_LOCK:
            _OA_CACHE[key] = (now, recs)
            while len(_OA_CACHE) > int(a["cache_max"]):
                _OA_CACHE.pop(next(iter(_OA_CACHE)))
    out = []
    for r in recs:
        if not isinstance(r, Mapping):
            continue
        c = dict(r)
        c.update({"provider": "portal:openalex_rw", "acq_source": SOURCE, "portal": "openalex_rw",
                  "portal_rank": len(out)})
        out.append(c)
    counts["openalex_refs"] = len(out)
    return out


def _openalex_cooldown() -> float:
    try:
        doc = json.loads((ROOT / "config" / "auto_acquire.json").read_text(encoding="utf-8"))
        return float((doc.get("search") or {}).get("openalex_quota_cooldown_s", 900))
    except Exception:  # noqa: BLE001
        return 900.0


def _default_openalex_lookup(seeds: list, per_page: int, timeout_s: float) -> list:
    """The shared /ask OpenAlex provider instance (its daily budget), keyless, through the openalex rate portal:
    filter cited_by:<seeds> with the OA fields; parsed by OpenAlexProvider.parse. No key is ever sent."""
    from vera import search_and_fetch, search_providers as sp
    chain = search_and_fetch.cached_chain()[0]
    p = next((x for x in chain if getattr(x, "name", None) == "openalex"), None)
    if p is None:
        return []
    p._spend()
    params = {"filter": "cited_by:" + "|".join(seeds), "per_page": int(per_page),
              "select": sp.OPENALEX_SELECT + "," + sp.OPENALEX_OA_SELECT_EXTRA}
    resp = sp.openalex_default_transport(p.endpoint, params, {"User-Agent": sp.USER_AGENT}, float(timeout_s))
    st = int(getattr(resp, "status_code", 200) or 200)
    if st == 429:
        raise sp.ProviderResponseError("openalex 429 quota")
    if st != 200:
        raise sp.ProviderResponseError(f"openalex status {st}")
    return p.parse(resp, int(per_page))


# ------------------------------------------------------------------------------------------------ run
def collect(question: str, *, base=None, ddg_raw=None, cfg: Mapping | None = None, deps: Deps | None = None,
            hosts: Mapping | None = None, clock: Callable = time.monotonic, deadline: float | None = None) -> tuple:
    """-> (candidates, counts). Never raises. counts carry numbers and fixed codes only."""
    counts: dict = {"pages_considered": 0, "pages_read": 0, "links": 0, "candidates": 0, "duplicates": 0}
    try:
        cfg = load_config() if cfg is None else validate(dict(cfg))
    except PortalConfigError:
        counts["error"] = "config"
        return [], counts
    if cfg["mode"] == "off":
        counts["error"] = "off"
        return [], counts
    deps = _default_deps(cfg, deps)
    hosts = load_hosts(cfg) if hosts is None else hosts
    caps = cfg["caps"]
    cap_q = int(caps["max_candidates_per_question"])
    out, seen = [], set()

    def add(c) -> None:
        if c is None or len(out) >= cap_q:
            return
        if _doi_excluded(str(c.get("doi") or ""), cfg):
            counts["excluded_doi"] = counts.get("excluded_doi", 0) + 1
            return
        ks = _keys(c)
        if ks & seen:
            counts["duplicates"] += 1
            return
        seen.update(ks)
        out.append(c)

    try:
        if adapter_on(cfg, "openalex_rw") and cfg["mode"] == "shadow" and \
                cfg["adapters"]["openalex_rw"].get("run_in_shadow") is not True:
            counts["openalex_shadow_skipped"] = 1  # SECREV M2: shadow never spends the shared keyless quota/slot
        elif adapter_on(cfg, "openalex_rw"):
            for c in openalex_rw(base, cfg, deps, counts, clock, deadline):
                add(c)
        if adapter_on(cfg, "inspire_refs") and cfg["mode"] == "on" and inspire_permitted():  # INSPIRE batch (OD-1108-SRC); a static import, never a dynamic one
            try:  # an adapter failure (module not shipped yet, ...) never drops the other adapters (SECREV re-check)
                from vera import portal_inspire
                for c in portal_inspire.adapter(question, base, cfg, deps, counts, clock, deadline) or ():
                    add(c)
            except Exception as e:  # noqa: BLE001
                counts["inspire_error"] = type(e).__name__
        groups: list = []  # [(adapter, [alternative urls])]
        if adapter_on(cfg, "wikipedia_title"):
            alts = wikipedia_title_urls(question, cfg)
            if alts:
                groups.append(("wikipedia_title", alts))
        if adapter_on(cfg, "generic_page"):
            groups += [("generic_page", [u]) for u in generic_page_urls(ddg_raw, base, cfg)]
        per_host: dict = {}
        landing = _landing_hosts()
        done_urls = set()
        for adapter, alts in groups:
            if counts["pages_read"] >= int(caps["max_pages_per_question"]) or len(out) >= cap_q:
                break
            if deadline is not None and clock() >= deadline:
                counts["error"] = "deadline"
                break
            for url in alts:
                if url in done_urls:
                    break
                done_urls.add(url)
                counts["pages_considered"] += 1
                try:
                    row = page_row(url, cfg, hosts)
                    h = row["host"]
                    if per_host.get(h, 0) >= int(caps["max_pages_per_host"]):
                        raise Skip("host_cap")
                    html = fetch_page(url, row, cfg, deps, clock, deadline)
                    per_host[h] = per_host.get(h, 0) + 1  # pages READ per host (alternatives of one page share it)
                    recs = extract(html, url, question, cfg, landing)
                    html = None  # the page is discarded as soon as its identifiers are out
                except Skip as s:
                    counts[s.code] = counts.get(s.code, 0) + 1
                    if s.code == "unreachable":
                        continue  # try the next alternative (e.g. the singular title)
                    break
                counts["pages_read"] += 1
                counts["links"] += len(recs)
                for rec in recs:
                    add(to_candidate(rec, len(out), adapter))
                break
    except Exception as e:  # noqa: BLE001  the portal never breaks the search
        counts["error"] = type(e).__name__
    counts["candidates"] = len(out)
    logger.info("autoacq refportal mode=%s %s", cfg["mode"], " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return out, counts


_SHADOW_BUSY = threading.Lock()


def extend(question: str, merged: list, *, base=None, ddg_raw=None, clock: Callable = time.monotonic,
           deadline: float | None = None, cfg: Mapping | None = None, deps: Deps | None = None,
           background: bool = True) -> list:
    """The acquisition_search hook. off: merged unchanged, no work. shadow: a background run (one at a time, its own
    deadline) that only logs counts; merged returned unchanged at once. on: synchronous within `deadline`; portal
    candidates merged after every other provider (merge_sources keeps first occurrences). Never raises."""
    try:
        c = load_config() if cfg is None else cfg
        m = mode(c)
    except Exception:  # noqa: BLE001
        return merged
    if m == "off":
        return merged
    if m == "shadow":
        if not _SHADOW_BUSY.acquire(blocking=False):
            logger.info("autoacq refportal mode=shadow skipped=busy")
            return merged
        b, d = list(base or ()), list(ddg_raw or ())

        def run():
            try:
                collect(question, base=b, ddg_raw=d, cfg=c, deps=deps, clock=clock,
                        deadline=clock() + float(c["caps"]["shadow_deadline_s"]))
            finally:
                _SHADOW_BUSY.release()
        if background:
            threading.Thread(target=run, name="refportal-shadow", daemon=True).start()
        else:
            run()
        return merged
    try:
        cands, _counts = collect(question, base=base, ddg_raw=ddg_raw, cfg=c, deps=deps, clock=clock,
                                 deadline=deadline)
        if not cands:
            return merged
        from vera import search_providers as sp
        return sp.merge_sources(merged, cands)
    except Exception:  # noqa: BLE001
        return merged


def reset_for_tests() -> None:
    with _OA_LOCK:
        _OA_CACHE.clear()
    with _TDM_LOCK:
        _TDM.clear()
    _WQ.clear()
