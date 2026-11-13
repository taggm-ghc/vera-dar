"""ALLDOMAINS-WPG-IMPL-1111 (plan work/ALLDOMAINS-1111 WP-G, absorbing work/USFED-ROUTE-1111): the generic
ORIGINAL_HTML full-text route. USFED_HTML renamed, driven by config/permitted_originals.json rows. Nothing here opens a
socket: every request goes through auto_acquire._guarded_get (SSRF guard and pinned transport, robots and TDM on every hop,
host_role "store", rate portal, Crawl-delay, content type, byte cap, wall clock), called with the row's own allowlist.

Inherited USFED controls, by step (each has a parity test in tests/test_original_html_wpg_1111.py):
  2   route-own allowlist: exact host + path prefix, https, port 443, no userinfo/query/fragment (normalise_url, match_row);
      the Europe PMC fetch.allowed_hosts is NOT widened (the route passes a derived cfg to _guarded_get)
  3   _guarded_get on every hop with route_hop=False (the agency page is the source itself, 3b-i), path_check on EVERY hop
      (3c: a same-host redirect outside the prefix is refused), chain=[] (userinfo/port/cycle/Content-Encoding identity)
  3a  robots per OD-1111-ROBOTS-4XX (404/410 only = no restriction) is the shared WP-L gate (acq_discovered), not a copy
  3b  401/403 on a document -> R-ACQ-ACCESS_DENIED in the shared WP-L _DENIED memory (acq_discovered.access_denied_stop);
      3b-ii the denial is checked before the first request and on every hop; 3b-iii per-host breaker + the snapshot
      `denied` list; 3b-iv one canonical URL form. No bypass: no retry, no variant URL, no other lane.
  3d  untrusted URLs (catalogue, index, sitemap) are normalised then re-checked (normalise_url)
  5   page licence checks and gov_public_domain.decide on the FULL visible text, before any pruning
  6   generic main-content extraction (vera.html_main), provenance captured before pruning; 6c boilerplate from the snapshot
  7   whole-or-nothing: OVERSIZE (decoded bytes), NO_BODY, THIN; nothing partial is ever stored
  9   storage only when mode == "on" AND the row has store: true; shadow/off store nothing (counts only)
Modes: off (ships) -> shadow -> on. Never raises to a caller outside auto_acquire._Stop / fulltext_loader.Refused.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = _ROOT / "config" / "permitted_originals.json"
MODES = ("off", "shadow", "on")
SUPPORTED_BASES = ("us_federal_pd",)  # wave 1; "page_declared" needs its own plan, licence review and C0 sample
ROUTE = "original_html"
LICENCE_URL = "https://www.copyright.gov/title17/92chap1.html"  # 17 U.S.C. 105 (the allowlist entry's evidence_url)
_UNRESERVED = re.compile(r"%(2[dDeE]|3[0-9]|4[1-9a-fA-F]|5[0-9aA]|5[fF]|6[1-9a-fA-F]|7[0-9aAeE])")
_HOST_RE = re.compile(r"[a-z0-9]([a-z0-9.-]{0,251}[a-z0-9])?")
_HEX = re.compile(r"[0-9a-f]{64}")
_TITLE_RE = re.compile(r"[^\x00-\x1f\x7f]{1,200}")
_TERM_RE = re.compile(r"[a-z0-9]{1,40}")
_BAD_PATH = re.compile(r"[?#;\\]|\.\.|//|%")
_SLUG = re.compile(r"[a-z0-9][a-z0-9-]*")  # lower-case slug, no leading hyphen, no '.', '~' or '_' (G-C condition)
_STOP = frozenset({"a", "an", "the", "of", "is", "are", "what", "whats", "and", "in", "to"})
logger = logging.getLogger("vera.original_html")

_cache: dict = {}
_LOCK = threading.Lock()
_HOST_DENIED: dict = {}  # host -> monotonic time of the first R-ACQ-ACCESS_DENIED on it (3b-iii (1): process lifetime)


def reset_for_tests() -> None:
    with _LOCK:
        _cache.clear()
        _HOST_DENIED.clear()


# ------------------------------------------------------------------------------------------------ config
def load_config(path: Path | None = None) -> dict:
    p = Path(path or CONFIG_PATH)
    key = ("cfg", str(p))
    with _LOCK:
        if key not in _cache:
            try:
                c = json.loads(p.read_text(encoding="utf-8"))
                _cache[key] = c if isinstance(c, dict) else {}
            except (OSError, ValueError):
                _cache[key] = {}
        return _cache[key]


def mode(cfg: dict | None = None) -> str:
    m = (cfg if cfg is not None else load_config()).get("mode")
    return m if m in MODES else "off"


def _row_valid(r) -> bool:
    """A malformed or unsupported row is ignored (fail closed): the route then simply does not claim its URLs."""
    try:
        if not isinstance(r, dict) or r.get("enabled") is not True:
            return False
        h = r.get("host")
        if not isinstance(h, str) or h != h.lower() or not _HOST_RE.fullmatch(h) or "." not in h:
            return False
        try:  # D5: a DNS name only, never an IP literal (and no all-numeric last label)
            ipaddress.ip_address(h)
            return False
        except ValueError:
            pass
        if not re.fullmatch(r"[a-z][a-z0-9-]*", h.rsplit(".", 1)[1]):
            return False
        pp, st = r.get("path_prefixes", []), r.get("path_stems", [])
        if not isinstance(pp, list) or not isinstance(st, list) or not (pp or st):
            return False
        if not all(isinstance(x, str) and x.startswith("/") and not _BAD_PATH.search(x) for x in pp + st):
            return False
        if r.get("licence_basis") not in SUPPORTED_BASES:
            return False
        if not isinstance(r.get("id"), str) or not r["id"] or not isinstance(r.get("review"), dict) \
                or not r["review"].get("task") or not r["review"].get("date"):
            return False
        return r.get("text_only") is True and isinstance(r.get("store"), bool)
    except Exception:  # noqa: BLE001
        return False


def rows(cfg: dict | None = None) -> list:
    c = cfg if cfg is not None else load_config()
    return [r for r in c.get("rows") or () if _row_valid(r)]


# ------------------------------------------------------------------------------------------------ URL form (3d, 3b-iv)
def normalise_url(u) -> str | None:
    """The one canonical form of a candidate URL, or None when it must be refused: https, port 443 or none, no userinfo,
    no query, no fragment, lower-case host, percent-decoding of unreserved characters ONLY, dot-segments resolved (a path
    that climbs above the root is refused), no encoded separators, no empty segments, no trailing slash (except root).
    Pure; never raises."""
    try:
        s = str(u or "")
        if not s or len(s) > 2000 or any(ord(ch) < 33 or ord(ch) > 126 for ch in s) or "\\" in s or ";" in s:
            return None  # D5: no path parameters (`..;/` tricks)
        sp = urlsplit(s)
        if sp.scheme != "https" or sp.query or sp.fragment or sp.username is not None or sp.password is not None:
            return None
        if "?" in s or "#" in s or "@" in sp.netloc:
            return None
        if sp.port not in (None, 443):
            return None
        host = (sp.hostname or "").lower()
        if not host or host.endswith(".") or not _HOST_RE.fullmatch(host):
            return None
        path = _UNRESERVED.sub(lambda m: chr(int(m.group(1), 16)), sp.path or "/")
        if re.search(r"%(2[fF]|5[cC]|00)", path) or "//" in path:
            return None
        out: list = []
        for seg in path.split("/")[1:]:
            if seg == ".":
                continue
            if seg == "..":
                if not out:
                    return None
                out.pop()
                continue
            out.append(seg)
        p = "/" + "/".join(x for x in out if x)
        if re.search(r"%(?![0-9A-Fa-f]{2})", p):
            return None
        return f"https://{host}{p}"
    except Exception:  # noqa: BLE001
        return None


def match_row(url, cfg: dict | None = None):
    """(row, canonical_url) when the NORMALISED url has a row's exact host and a path starting with one of its prefixes,
    else None. The check runs on the normalised form (3d)."""
    canon = normalise_url(url)
    if canon is None:
        return None
    sp = urlsplit(canon)
    for r in rows(cfg):
        if sp.hostname == r["host"] and path_ok(r, sp.path):
            return r, canon
    return None


def path_ok(row: dict, path: str) -> bool:
    """D1: whole path segments. A `path_prefixes` entry P matches `P` itself and anything under `P/` (so `/dark-matter`
    does not match `/dark-matterX` or `/dark-matter-evil/x`). A `path_stems` entry S (for a host whose pages are named by
    appending a slug to a stem in ONE segment, e.g. DOE `/science/doe-explains<slug>`) matches S followed by a non-empty
    single-segment slug; the bare stem (an index page, links only) and any deeper path never match."""
    for p in row.get("path_prefixes") or ():
        b = p.rstrip("/")
        if path == b or path.startswith(b + "/"):
            return True
    for s in row.get("path_stems") or ():
        if path.startswith(s) and _SLUG.fullmatch(path[len(s):]):
            return True
    return False


def path_checker(row: dict, index=None):
    """The per-hop check handed to auto_acquire._guarded_get: every hop (redirect targets included) must normalise and
    stay on the row's exact host AND inside a row prefix or stem, by whole segments (3c). PREON-ENFORCE-1112: when
    `index` (a frozenset of canonical URLs, mode "on" only) is given, every hop must also be a member of it."""
    def check(url: str) -> bool:
        c = normalise_url(url)
        if c is None:
            return False
        if index is not None and c not in index:
            return False
        sp = urlsplit(c)
        return sp.hostname == row["host"] and path_ok(row, sp.path)
    return check


# ------------------------------------------------------------------------------------------------ snapshot (read only)
def snapshot_digest(body: dict) -> str:
    """sha256 of the canonical JSON of the snapshot's data blocks (entries, denied, boilerplate)."""
    d = {k: body.get(k) for k in ("entries", "denied", "boilerplate")}
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _snapshot_valid(s, cfg: dict) -> bool:
    """Structure, sha256 and every entry re-checked (https, a row's host and prefix, one canonical form, no query).
    Off-pattern values are refused without being echoed."""
    try:
        if not isinstance(s, dict) or s.get("version") != 1 or not isinstance(s.get("harvested_at"), str):
            return False
        if not isinstance(s.get("robots_status"), dict) or not isinstance(s.get("script_version"), str):
            return False
        if s.get("sha256") != snapshot_digest(s):
            return False
        for e in s["entries"]:
            m = match_row(e.get("url"), cfg)
            if (m is None or m[1] != e["url"] or e.get("host") != urlsplit(e["url"]).hostname
                    or not _TITLE_RE.fullmatch(str(e.get("title") or "")) or str(e["title"]) != str(e["title"]).strip()
                    or not isinstance(e.get("slug_terms"), list)
                    or not all(isinstance(t, str) and _TERM_RE.fullmatch(t) for t in e["slug_terms"])):
                return False
        for d in s["denied"]:
            if normalise_url(d.get("url")) != d.get("url") or not isinstance(d.get("code"), str) \
                    or not isinstance(d.get("seen_at"), str):
                return False
        for h, lst in s["boilerplate"].items():
            if not _HOST_RE.fullmatch(str(h)) or not isinstance(lst, list) or not all(
                    isinstance(x, str) and _HEX.fullmatch(x) for x in lst):
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def load_snapshot(cfg: dict | None = None) -> dict | None:
    """The validated snapshot, or None (missing or invalid: the route claims nothing, fail closed). READ ONLY: this module
    has no write path to it (a test scans the source); only scripts/refresh_* writes it."""
    c = cfg if cfg is not None else load_config()
    rel = str((c.get("snapshot") or {}).get("path") or "")
    if not rel:
        return None
    p = (_ROOT / rel) if not Path(rel).is_absolute() else Path(rel)
    key = ("snap", str(p), json.dumps(c.get("rows"), sort_keys=True, default=str))
    with _LOCK:
        if key not in _cache:
            try:
                s = json.loads(p.read_text(encoding="utf-8"))
                _cache[key] = s if _snapshot_valid(s, c) else None
            except (OSError, ValueError):
                _cache[key] = None
        return _cache[key]


def boilerplate_for(host: str, cfg: dict | None = None) -> set:
    s = load_snapshot(cfg)
    return set(((s or {}).get("boilerplate") or {}).get(host) or ())


def snapshot_denied(canon: str, cfg: dict | None = None) -> bool:
    s = load_snapshot(cfg)
    return any(d.get("url") == canon for d in (s or {}).get("denied") or ())


# ------------------------------------------------------------------------------------------------ published index (mode "on")
# PREON-ENFORCE-1112: the `_pre_on_condition` of config/permitted_originals.json, enforced. In mode "on" the route may claim,
# fetch and store only a URL that is a member of the harvested snapshot's entries. A missing, invalid, mislocated, stale,
# future-dated or empty index admits nothing (fail closed). Shadow and off never call this code (behaviour unchanged).
# UNCALIBRATED PRIOR: the 30-day default age is a conservative first value, no harvest has run yet; calibrate from the
# first real harvest cadence (limits.index_max_age_days overrides it, int 1..365, anything else refuses).
INDEX_MAX_AGE_DAYS = 30


def index_urls(cfg: dict | None = None):
    """frozenset of canonical entry URLs of the validated, fresh snapshot, or None (refuse). Never raises."""
    try:
        c = cfg if cfg is not None else load_config()
        rel = str((c.get("snapshot") or {}).get("path") or "")
        if not rel:
            return None
        sp = Path(rel)
        p = sp if sp.is_absolute() else _ROOT / rel
        root = (_ROOT / "config").resolve()
        if root != p.resolve() and root not in p.resolve().parents:
            return None  # Q24: the snapshot must sit inside <root>/config
        age = (c.get("limits") or {}).get("index_max_age_days", INDEX_MAX_AGE_DAYS)
        if isinstance(age, bool) or not isinstance(age, int) or not 1 <= age <= 365:
            return None
        snap = load_snapshot(c)
        if snap is None:
            return None
        from datetime import datetime, timedelta

        from vera import freshness as _fz
        at = datetime.fromisoformat(str(snap.get("harvested_at") or ""))
        if at.tzinfo is None:
            return None
        now = _fz.utcnow()
        if at > now or now - at > timedelta(days=age):
            return None
        out = set()
        for e in snap.get("entries") or ():
            u = e.get("url") if isinstance(e, dict) else None
            if isinstance(u, str) and normalise_url(u) == u:
                out.add(u)
        return frozenset(out)
    except Exception:  # noqa: BLE001
        return None


def in_index(url, cfg: dict | None = None, index=None) -> bool:
    """True when the normalised url is an exact member of the index (computed from cfg unless `index` is given)."""
    canon = normalise_url(url)
    idx = index if index is not None else index_urls(cfg)
    return canon is not None and idx is not None and canon in idx


# ------------------------------------------------------------------------------------------------ claim and match
def claims_url(url, cfg: dict | None = None) -> bool:
    """True when the route would take this URL now: mode shadow or on, the validated snapshot present, and a row matches.
    In mode "on" the URL must also be in the published index. Metadata only; makes no request."""
    c = cfg if cfg is not None else load_config()
    if mode(c) == "off" or load_snapshot(c) is None:
        return False
    if mode(c) == "on" and not in_index(url, c):
        return False
    return match_row(url, c) is not None


def claims_cand(cand, cfg: dict | None = None) -> bool:
    return isinstance(cand, dict) and any(claims_url(cand.get(k), cfg) for k in ("url", "discovered_url"))


def cand_url(cand, cfg: dict | None = None) -> str | None:
    c = cfg if cfg is not None else load_config()
    idx = index_urls(c) if mode(c) == "on" else None
    for k in ("url", "discovered_url"):
        m = match_row(cand.get(k), c) if isinstance(cand, dict) else None
        if m and (mode(c) != "on" or (idx is not None and m[1] in idx)):
            return m[1]
    return None


def _terms(s: str) -> list:
    out = []
    for t in re.split(r"[^a-z0-9]+", str(s or "").lower()):
        if not t or t in _STOP:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]  # quarks -> quark; plural only, no stemming beyond that
        out.append(t)
    return out


def match_subject(subject: str, cfg: dict | None = None) -> list:
    """Catalogue candidates for a definitional subject: every subject token must occur (exact token match after
    case/plural/hyphen normalisation) among the entry's title and slug tokens; extra entry tokens are allowed. Ranked by
    fewest extra tokens; at most limits.max_candidates_per_host per host and limits.max_candidates_per_question in all.
    Local data only. No match, an empty subject, or mode off -> []."""
    c = cfg if cfg is not None else load_config()
    s = load_snapshot(c)
    want = set(_terms(subject))
    if mode(c) == "off" or s is None or not want:
        return []
    lim = c.get("limits") or {}
    per_host, total = int(lim.get("max_candidates_per_host", 1)), int(lim.get("max_candidates_per_question", 3))
    hits = []
    for e in s["entries"]:
        have = set(_terms(e["title"])) | {t for x in e["slug_terms"] for t in _terms(x)}
        if want <= have and claims_url(e["url"], c) and not snapshot_denied(e["url"], c):
            hits.append((len(have - want), e["url"], e))
    hits.sort(key=lambda x: (x[0], x[1]))
    out, n = [], {}
    for _x, u, e in hits:
        if n.get(e["host"], 0) >= per_host or len(out) >= total:
            continue
        n[e["host"]] = n.get(e["host"], 0) + 1
        out.append({"url": u, "title": e["title"], "source": "permitted_catalogue", "original_html": True})
    return out


# ------------------------------------------------------------------------------------------------ untrusted index / sitemap input
class SitemapRefused(ValueError):
    """The sitemap is not parsed. Carries a short code, never the document."""


def parse_sitemap(raw: bytes, row: dict, cfg: dict | None = None) -> dict:
    """Parse one sitemap or sitemap index for the catalogue refresh (WP-C). Refuses: over `limits.max_sitemap_bytes`
    (decoded bytes), gzip content (magic bytes), any DOCTYPE or entity declaration, a malformed or non-sitemap root.
    Parsed with defusedxml (DTD, entities and external references forbidden). Returns {"urls": [...], "children": [...]}:
    every <loc> is untrusted, so it is normalised and re-checked against the row (exact host and prefix) for pages, and
    for the row's exact host (https, no query) for child sitemaps; others are dropped, never fetched."""
    c = cfg if cfg is not None else load_config()
    cap = int((c.get("limits") or {}).get("max_sitemap_bytes", 5000000))
    if not isinstance(raw, (bytes, bytearray)) or len(raw) > cap:
        raise SitemapRefused("oversize")
    if bytes(raw[:2]) == b"\x1f\x8b":
        raise SitemapRefused("gzip")
    low = bytes(raw).lower()
    if b"<!doctype" in low or b"<!entity" in low:
        raise SitemapRefused("doctype")
    try:
        from defusedxml import ElementTree as DET
        root = DET.fromstring(bytes(raw), forbid_dtd=True, forbid_entities=True, forbid_external=True)
    except Exception:  # noqa: BLE001
        raise SitemapRefused("malformed") from None
    tag = root.tag.rsplit("}", 1)[-1]
    if tag not in ("urlset", "sitemapindex"):
        raise SitemapRefused("root")
    locs = [(e.text or "").strip() for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "loc"]
    urls, children = [], []
    for loc in locs:
        n = normalise_url(loc)
        if n is None or urlsplit(n).hostname != row["host"]:
            continue
        if tag == "sitemapindex":
            if n not in children and (n.endswith(".xml") or "sitemap" in n):
                children.append(n)
        elif match_row(n, c) is not None and match_row(n, c)[0]["id"] == row["id"] and n not in urls:
            urls.append(n)
    return {"urls": urls, "children": children}


def parse_index_links(html, base: str, row: dict, cfg: dict | None = None) -> list:
    """Page links on an index page (links only; the index text is never evidence): each href is joined to `base`,
    normalised and re-checked against the row; anything else is dropped."""
    from urllib.parse import urljoin

    from vera import html_main
    doc = html_main.parse(html)
    out: list = []
    for n in html_main.walk(doc):
        if n.tag == "a" and n.attrs.get("href"):
            try:
                u = urljoin(base, n.attrs["href"].strip())
            except ValueError:
                continue
            m = match_row(u, cfg)
            if m is not None and m[0]["id"] == row["id"] and m[1] not in out:
                out.append(m[1])
    return out


# ------------------------------------------------------------------------------------------------ denial breaker (3b-iii)
def host_tripped(host: str) -> bool:
    with _LOCK:
        return host in _HOST_DENIED


def trip_host(host: str) -> None:
    with _LOCK:
        _HOST_DENIED.setdefault(host, time.monotonic())


# ------------------------------------------------------------------------------------------------ loader limits (step 7)
def relaxed_ingest_cfg(icfg: dict, oc: dict | None = None) -> dict:
    """The fulltext_ingest config with this route's single length floor (plan step 7; §5.6 SF-A, USFED-SHADOWFIX-1112).
    The floor `limits.min_text_chars` is measured on the converted text that is stored (len(text)), the same measure the
    route checks first under R-ACQ-ORIGINAL_THIN, so `min_fulltext_chars` IS that floor. The article abstract-only tests
    do not apply to an HTML page (it has no abstract): `min_body_chars` is 1, because the converter's `body_chars` omits
    the lead text between the page title and the first <h2> (where an explainer's definition sits), and
    `min_body_to_abstract_ratio` is 0, because an HTML heading named "Summary" is not an article abstract; `has_body`
    is still required (the route raises NO_BODY first). Every other filter is unchanged. A copy; the input is never
    mutated. Used only when the caller passes route="original_html" (the route itself, or the acquired-path reader from
    the row's acquisition resolver), never from an item's own claim."""
    lim = ((oc if oc is not None else load_config()).get("limits") or {})
    out = dict(icfg)
    out["min_fulltext_chars"] = int(lim.get("min_text_chars", 1500))
    out["min_body_chars"] = 1
    out["min_body_to_abstract_ratio"] = 0
    return out


# ------------------------------------------------------------------------------------------------ the route body
def _fetch_cfg(cfg: dict, row: dict, limits: dict) -> dict:
    """A derived acquisition cfg for _guarded_get: ONLY this row's host, text/html only. The shared cfg is not mutated and
    the Europe PMC allowlist is not widened."""
    out = dict(cfg)
    out["fetch"] = {"allowed_hosts": [row["host"]], "content_types": ["text/html"], "fulltext_format": "html"}
    return out


def evaluate(job, score, doi, cand, isbn, cfg, deps, icfg, fdcfg, clock=time.monotonic, deadline=None, dur=None,
             min_spans=None):
    """auto_acquire.evaluate_candidate for an ORIGINAL_HTML-routed candidate. Same contract as the other routes: returns
    (plan, isbn); raises auto_acquire._Stop / fulltext_loader.Refused. Writes nothing."""
    from vera import acq_config, acq_discovered, html_main, jats_text, page_licence, public_fetch
    from vera import auto_acquire as aa
    from vera import fulltext_loader as fl
    from vera import gov_public_domain
    dur = {} if dur is None else dur
    oc = load_config()
    md = mode(oc)
    m = match_row(cand_url(cand), oc)
    if md == "off" or m is None or load_snapshot(oc) is None:
        raise aa._Stop(acq_config.reason(cfg, "host"))  # defensive: claims() already said no
    row, canon = m
    host = row["host"]
    idx = index_urls(oc) if md == "on" else None
    if md == "on" and (idx is None or canon not in idx):
        raise aa._Stop(acq_config.reason(cfg, "host"))  # PREON-ENFORCE-1112: not in the published index, no request
    lim = oc.get("limits") or {}
    denied_code = acq_discovered.refusal_code(cfg, "access_denied")
    # 3b-ii / 3b-iii: no request for a denied document, a denied host (breaker) or a URL on the snapshot's denied list
    if host_tripped(host) or snapshot_denied(canon, oc) or acq_discovered.is_denied(urls=(canon,)):
        raise aa._Stop(denied_code)
    t = clock()
    try:
        raw, final = aa._guarded_get(canon, _fetch_cfg(cfg, row, lim), deps, int(lim.get("max_html_bytes", 2000000)),
                                     float(lim.get("fetch_timeout_s", 20)), False, clock=clock, deadline=deadline,
                                     chain=[], route_hop=False, path_check=path_checker(row, idx))
    except aa._Stop as e:
        if str(e.code).startswith(denied_code):
            trip_host(host)  # the first denial on a host stops every request to it for this process (fail closed wider)
        if e.code == acq_config.reason(cfg, "too_large"):
            raise aa._Stop(acq_config.reason(cfg, "original_oversize")) from None  # step 7: decoded-byte cap
        raise
    dur["fetch"] = dur.get("fetch", 0) + int((clock() - t) * 1000)
    t = clock()
    fm = match_row(final, oc)
    if fm is None or fm[0]["id"] != row["id"] or (idx is not None and fm[1] not in idx):
        raise aa._Stop(acq_config.reason(cfg, "host"))
    if len(raw) > int(lim.get("max_html_bytes", 2000000)):
        raise aa._Stop(acq_config.reason(cfg, "original_oversize"))
    try:
        doc = html_main.parse(raw, (oc.get("extract") or {}).get("max_depth", 200),
                              (oc.get("extract") or {}).get("max_nodes", 400000))
    except html_main.HtmlRefused:
        raise aa._Stop(acq_config.reason(cfg, "convert")) from None
    # 5 + 6e: licence and marker checks on the FULL visible text, provenance captured, all BEFORE any pruning
    full = html_main.visible_text(doc)
    prov = html_main.capture_provenance(doc, oc.get("provenance"))
    html_s = raw.decode("utf-8", "replace")
    sigs = public_fetch.canonical_signals(html_s, final)
    pl = page_licence.check(final, {"text": full, "licence_href": prov["licence_href"], "url": final,
                                    "canonical_url": sigs[0] if sigs else "", "canonical_signals": sigs})
    pd = gov_public_domain.decide(final, full, prov["licence_href"])
    if (row["licence_basis"] != "us_federal_pd" or gov_public_domain.mode() == "off" or not pd.get("applies")
            or not pl.get("ok")):
        raise aa._Stop(acq_config.reason(cfg, "original_not_pd"))
    # 6a-6d: generic main-content step, then the converter; a page with no headed sections is NO_BODY
    ex = html_main.extract_main(raw, oc.get("extract") or {}, host, boilerplate_for(host, oc), doc=doc)
    try:
        conv = jats_text.convert(ex["html"], "html", icfg["converter"])
    except Exception:  # noqa: BLE001  (ConversionError: no headed sections / hostile file)
        raise aa._Stop(acq_config.reason(cfg, "original_no_body")) from None
    if not conv.get("has_body"):  # only the title heading: converts without error, but there is no body section
        raise aa._Stop(acq_config.reason(cfg, "original_no_body"))
    text = conv["text"]
    if len(text) < int(lim.get("min_text_chars", 1500)):  # step 7: the one floor, on the stored text (SF-A)
        raise aa._Stop(acq_config.reason(cfg, "original_thin"))
    now_iso = _now_iso()
    meta = {k: v for k, v in conv.items() if k != "text"}
    did = "acq-" + hashlib.sha256(("url:" + canon).encode("utf-8")).hexdigest()[:16]
    meta.update({"document_id": did, "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_file": None,
                 "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "jats_licences": []})
    lic = pd["licence"]
    title = ((meta.get("meta") or {}).get("title") or cand.get("title") or "").strip()
    agency = str(row.get("agency") or host)
    item = {"document_id": did, "title": title, "doi": None,
            "identifier": {"kind": "original_url", "id": canon, "url": canon},
            "landing_url": canon, "fulltext_url": canon, "fulltext_format": "html", "text_file": f"acquired/{did}.html",
            "licence": lic["id"], "licence_url": LICENCE_URL, "licence_source": "us-federal-public-domain rule",
            "discovered_via": ROUTE, "authors": agency,
            "published_year": (prov.get("published") or "")[:4] or "n.d.",  # the page's own date, else "no date"
            "fetched_at": now_iso,
            "verifier_status": icfg["verified_status"], "read_status": "VERBATIM"}
    item["attribution"] = aa.tasl_attribution(title, agency, canon, lic["id"], LICENCE_URL)
    # no JATS licence element exists on an HTML page: the licence is the PD rule's (decided above on the full text); the
    # acquired-path reader checks this row's licence record by its own original_html branch (fulldoc_db)
    rec = fl.prepare_from_parts(item, text, meta, icfg, now=now_iso, route=ROUTE)  # every filter; route-own floors
    if acq_config.injection_refused(f"{title}\n{text}", cfg):
        raise aa._Stop(acq_config.reason(cfg, "inject"))
    plan = aa.build_plan(rec, item, meta, cfg, job, score, {"open_access": True, "licence_family": "us-federal-pd",
                                                          "pub_types_n": 0}, isbn, now_iso)
    acq_discovered.annotate(plan, cand, {"pub_types": []}, "landing", [canon, final], cfg)
    a = plan["source"]["provenance"]["acquisition"]
    a["resolver"], a["repository"] = ROUTE, host
    a["content_version"] = {"pub_types": [], "source": f"{agency} web page (original host)"}
    a["licence_evidence"] = {"governing": "us-federal-public-domain rule (17 U.S.C. 105) decided on the full visible "
                                          "page text; no index or catalogue licence used", "index_metadata_licence": "not used"}
    a["inspection"]["converter"] = "html (generic main-content extraction)"
    a["original_html"] = {"row": row["id"], "mode": md, "licence_basis": row["licence_basis"],
                          "gov_pd": {k: pd.get(k) for k in ("rule", "applies", "reason_code", "basis", "organization",
                                                            "image_credit", "dataset_retrieved")},
                          "page": prov, "extract": {"root": ex["root"], "pruned": ex["stats"]},
                          "retrieved_at": now_iso}
    n_adm = aa.relevance_gate(job.question, plan, cfg, icfg, fdcfg)
    dur["verify"] = dur.get("verify", 0) + int((clock() - t) * 1000)
    need = acq_config.persist_min_spans(cfg) if min_spans is None else max(1, int(min_spans))
    if n_adm < need:
        raise aa._Stop(acq_config.reason(cfg, "few_spans") if n_adm else acq_config.reason(cfg, "irrelevant"), n_adm)
    plan["admitted_spans"] = n_adm
    acq_discovered.gate_b_admit(plan)
    if md != "on" or row.get("store") is not True:
        # shadow (or a row that may not store yet): the steps above ran; nothing is stored; counts only (no URL, title)
        aa._log("original_html_shadow", mode=md, store=bool(row.get("store")), admitted=n_adm, chars=len(text),
                pruned=sum(ex["stats"].values()))
        raise aa._Stop(acq_config.reason(cfg, "original_shadow"), n_adm)
    return plan, isbn


def _now_iso() -> str:
    from datetime import timezone

    from vera import freshness as _fz
    return _fz.utcnow().astimezone(timezone.utc).isoformat(timespec="seconds")
