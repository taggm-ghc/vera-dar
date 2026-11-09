"""NONASK-1103 S4 (plan 6.3, 6.7, 6.8a; owner decisions R-037 and 2026-11-03 "honour site terms").

Official lane: for each registry entity named in a sub-query, the registry's own documentation entry URLs. No search
call is made and no host is ever guessed (an entity without an active registry row gets nothing).

Access gate (Deny All, owner 2026-11-04; OD-OL-1 verified the 10 active rows): only a row whose
source_permissions.access_verdict is explicitly "allow" is ever fetched; unknown, missing or forbid yields links only. A page is
fetched only when robots.txt permits it (vera.public_fetch checks robots on the same pinned path and fails closed) and
its normalised path is on the row's `allowed_paths` for that host (Deny All, owner 2026-11-04). The access verdicts are operational policy
decisions, not definitive legal determinations of permitted AI use (owner qualification, OD-OL-1).

Hop pin (OFFICIAL-LANE-S4A, gate G12): every hop (start URL, each redirect target and its robots.txt request) must be on
one of the row's exact hosts and on its allowed_paths (or be that host's /robots.txt); web_route.gated_fetch checks this BEFORE the hop is sent, so
a redirect to an off-registry host is refused, never requested (reason read_lane_hop_refused).

Page text is held in memory for this answer only (R-037): LanePage keeps it out of repr(), so a log line, a trace or a
traceback that renders the object never carries it. This module imports no store, cache or queue writer.

The lane flag ships OFF (config/subquery.json official_lane.enabled). Enable only via docs/lanes-enable-checklist.md.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Callable
from urllib.parse import quote, unquote, urlsplit

_ROOT = Path(__file__).resolve().parent.parent
_CFG_PATH = _ROOT / "config" / "subquery.json"
VERDICTS = ("allow", "links_only", "unknown")
LANE = "official"


@dataclass(repr=False)
class LanePage:
    """One page read for this answer only. `text` is internal (prompt and claim check); never shown or stored."""
    lane: str
    entity: str
    title: str
    url: str
    outlet: str = ""
    published: str | None = None
    attribution: dict | None = None
    text: str = field(default="", repr=False)

    def __repr__(self) -> str:  # never the page text
        return f"LanePage(lane={self.lane!r}, url={self.url!r}, chars={len(self.text)})"

    __str__ = __repr__


def load_config(path: Path | None = None) -> dict:
    try:
        cfg = json.loads((path or _CFG_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"enabled": False}
    ol = cfg.get("official_lane") if isinstance(cfg, dict) else None
    return ol if isinstance(ol, dict) else {"enabled": False}


def lane_enabled(cfg: dict | None = None) -> bool:
    """Exactly true only (fail closed on any odd value)."""
    try:
        return (cfg if cfg is not None else load_config()).get("enabled") is True
    except Exception:  # noqa: BLE001
        return False


def fetch_allowed(ent: str, cfg: dict | None) -> bool:
    """RETRIEVAL-1109 H-2: optional `fetch_entities` (list of registry entity names) narrows READING to those rows; every
    other row gives links only. Key absent = no narrowing (previous behaviour). A present but odd value reads nothing
    (fail closed)."""
    if not isinstance(cfg, dict) or "fetch_entities" not in cfg:
        return True
    fe = cfg.get("fetch_entities")
    return isinstance(fe, list) and ent in fe


def verdict(row: dict) -> str:
    """allow | links_only. Deny All (owner standard 2026-11-04, audit D3; replaces the R-050 "unknown means read" rule
    for this lane): "allow" ONLY on an explicit, verified access verdict (source_permissions.access_verdict == "allow",
    which a terms_verdict links_only always overrides to forbid). unknown, missing, forbid or any odd value means
    links_only. Checked by run() before any fetch or site: query."""
    try:
        from vera import source_permissions
        v = source_permissions.access_verdict(row if isinstance(row, dict) else {})
    except Exception:  # noqa: BLE001
        return "links_only"
    return "allow" if v == "allow" else "links_only"


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


# Cloud metadata endpoint names (checklist A3, SEC M5): never an official host, whatever a registry row says. The
# addresses themselves are blocked by public_fetch (_EXTRA_BLOCKED, link-local); this is the name-level backstop.
METADATA_HOSTS = frozenset({"metadata.google.internal", "metadata", "metadata.goog", "instance-data",
                            "instance-data.ec2.internal", "metadata.azure.com"})
_HOST_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)+$")


def valid_host(host) -> bool:
    """Checklist B1 (SEC M6): a registry host must be an ASCII, lower-case, dotted DNS name (A-labels only: an IDN must
    be written as xn--...), IDNA round-trip stable, with no wildcard, port, path, userinfo or IP literal, and not a
    metadata name. Anything else is rejected (fail closed)."""
    if not isinstance(host, str) or not host.isascii() or host != host.lower() or not _HOST_RE.match(host):
        return False
    if host in METADATA_HOSTS or host.endswith(".internal") or host.rsplit(".", 1)[-1].isdigit():
        return False
    try:
        return host.encode("ascii").decode("idna").encode("idna").decode("ascii") == host  # SEC2 N1: real A-labels
    except UnicodeError:
        return False


def row_hosts(row: dict) -> set:
    """The row's valid exact hosts (invalid entries are ignored, never repaired)."""
    return {d for d in ((row or {}).get("domains") or []) if valid_host(d)}


def host_ok(url: str, row: dict) -> bool:
    """Exact host match against the row's valid domains (no wildcard, no suffix match, no metadata name)."""
    h = _host(url)
    return bool(h) and h not in METADATA_HOSTS and h in row_hosts(row)


# Deny All; approve specified exceptions (owner standard 2026-11-04). A path is read only when, after normalisation,
# it equals an `exact` entry or starts with a `prefix` entry (each ending in "/") listed for that host in the row's
# `allowed_paths`; everything else is link-only. `avoid_paths` is documentation only (never consulted by code).
_PCHAR_SAFE = "/!$&'()*+,=:@-._~"
_NEVER_PAGES = ("/robots.txt", "/llms.txt", "/llms-full.txt")  # SEC3 S3-3: control files are never lane pages
_PATH_CHARS = re.compile(r"^/[A-Za-z0-9._~!$&'()*+,=:@/-]*$")  # RFC 3986 pchar minus ';' and '%' (decoded once)


def normalise_path(url: str, case_insensitive: bool = False):
    """The normalised path, or None (deny). Rules: percent-decode once; deny anything still encoded, non-ASCII, a query,
    ';', a backslash, any character outside RFC 3986 pchar, an empty, '.' or '..' segment ('//' included). A fragment
    is never sent and is ignored. Lower-cased only when the row declares a case-insensitive host."""
    try:
        sp = urlsplit(url)
    except ValueError:
        return None
    raw = sp.path or "/"
    if sp.query or not raw.isascii():
        return None
    path = unquote(raw, errors="strict")
    if "%" in path or not path.isascii() or not _PATH_CHARS.match(path):
        return None
    if quote(path, safe=_PCHAR_SAFE) != raw:  # SEC3 S3-2: the checked path must be byte-identical to the one sent
        return None
    segs = path.split("/")[1:]
    if any(sg in (".", "..") for sg in segs) or any(sg == "" for sg in segs[:-1]):
        return None
    return path.lower() if case_insensitive else path


def path_allowed(url: str, row: dict) -> bool:
    """True only when the normalised path matches one of the row's allowed exact paths or prefixes for this exact host.
    A missing, malformed or empty allowlist allows nothing (link only)."""
    try:
        h = _host(url)
        spec = ((row or {}).get("allowed_paths") or {}).get(h)
        if not isinstance(spec, dict):
            return False
        path = normalise_path(url, (row or {}).get("path_case_insensitive") is True)
        if path is None:
            return False
        if set(spec) - {"exact", "prefix"}:  # SEC3 S3-1: unknown keys mean a malformed spec: nothing allowed
            return False
        ex, px = spec.get("exact", []), spec.get("prefix", [])
        if not isinstance(ex, list) or not isinstance(px, list):  # a string is never iterated as characters
            return False
        exact = [e for e in ex if isinstance(e, str) and e.startswith("/")]
        prefix = [x for x in px if isinstance(x, str) and x.startswith("/") and x.endswith("/")]
        if path in _NEVER_PAGES or path.startswith("/.well-known/"):
            return False
        return path in exact or any(path.startswith(x) for x in prefix)
    except Exception:  # noqa: BLE001
        return False


def page_ok(url: str, row: dict) -> bool:
    """A page may be read: exact registry host AND an allowed path."""
    try:
        return host_ok(url, row) and path_allowed(url, row)
    except Exception:  # noqa: BLE001
        return False


def hop_ok(url: str, row: dict) -> bool:
    """G12 hop predicate (checked before each hop is sent): page_ok, or exactly /robots.txt on a registry host (robots
    must be read to honour it; its body is never a lane page). Never raises (False on error)."""
    try:
        if host_ok(url, row) and normalise_path(url) == "/robots.txt":
            return True
        return page_ok(url, row)
    except Exception:  # noqa: BLE001
        return False


def _rows(registry: list | None) -> dict:
    if registry is None:
        from vera import subquery
        registry = subquery.load_entity_registry()
    out = {}
    for r in registry or ():
        if not (isinstance(r, dict) and r.get("state") == "active" and r.get("entity")):
            continue
        doms = r.get("domains") or []
        if not doms or not all(valid_host(d) for d in doms):  # B1: one bad host drops the whole row (fail closed)
            continue
        out[r["entity"]] = r
    return out


_LAST_FETCH: dict = {}  # host -> monotonic time of the last request sent (per process); never holds page text
_LAST_LOCK = threading.Lock()
_clock = time.monotonic


def _delay_ok(url: str, row: dict) -> bool:
    """Row `crawl_delay_s` (config; the host's robots.txt Crawl-delay): True, and the slot is taken, when the last request
    to this host was at least that long ago; False = too early, so the request is NOT sent (link only)."""
    try:
        delay = float(row.get("crawl_delay_s") or 0)
    except (TypeError, ValueError):
        return False
    if delay <= 0:
        return True
    h, now = _host(url), _clock()
    with _LAST_LOCK:
        last = _LAST_FETCH.get(h)
        if last is not None and now - last < delay:
            return False
        _LAST_FETCH[h] = now
    return True


class OfficialLane:
    """Inject `fetch` (vera.public_fetch.fetch signature: url -> {"text", "url", "content_type"}) in tests."""

    def __init__(self, cfg: dict | None = None, *, registry: list | None = None, fetch: Callable | None = None,
                 site_search: Callable | None = None):
        self.cfg = cfg if cfg is not None else load_config()
        self._rows = _rows(registry)
        self._fetch = fetch
        self._site_search = site_search  # (host, terms) -> [{"url", "title"}]; None = the shared DDG provider
        self._robots_cache: dict = {}  # robots.txt rules per host for this lane object (one request)

    def _do_fetch(self, url: str, row: dict) -> dict:
        """WP6-P2b D3: the per-hop READ gate (registry, deny list, login path) inside public_fetch on every hop.
        G12: hop_allowed pins every hop to the row's exact hosts and allowed_paths, checked before the hop is sent."""
        from vera import web_route
        return web_route.gated_fetch(url, fetch=self._fetch, registry=list(self._rows.values()),
                                     robots_cache=self._robots_cache, hop_allowed=lambda u: hop_ok(u, row))

    def _site_urls(self, row: dict, terms: list) -> list:
        """Plan 6.3 step 2: ONE "site:<host> <terms>" DDG query per allow-verdict entity, through the shared DDG
        provider (its cache, daily cap and rate gate). Links-only rows never reach here. Results off the row's exact
        hosts are dropped. A spent cap, a challenge or any error gives nothing (never raises)."""
        hosts = [str(d).lower() for d in (row.get("domains") or [])]
        if not hosts or not terms or (self._site_search is None and self.cfg.get("site_search") is not True):
            return []
        try:
            got = (self._site_search or _ddg_site_search)(hosts[0], list(terms))
        except Exception:  # noqa: BLE001
            return []
        return [(str(r.get("url")), str(r.get("title") or "")) for r in got or [] if isinstance(r, dict)
                and host_ok(str(r.get("url") or ""), row) and str(r.get("url") or "").startswith("https://")]

    def run(self, entities, terms=(), should_stop=None) -> dict:
        """{'links': [public-safe dicts], 'pages': [LanePage]} for the entities of one sub-query. Flag off: nothing.
        Never raises; a refused or failed fetch leaves the link only (the reason is a short code, never page text).
        `should_stop()` (optional) returns a short code or None and is checked before each fetch (checklist D3)."""
        out = {"links": [], "pages": []}
        if not lane_enabled(self.cfg):
            return out
        seen: set = set()
        for ent in entities or ():
            row = self._rows.get(ent)
            if row is None:  # no registry hit: the entity is named, no host is assigned (no guessing)
                continue
            urls = [(u, f"{ent} documentation") for u in row.get("docs_entry_urls") or []]
            # R-050: site: only for readable rows; an access-restricted row (verdict links_only) gets links only below
            readable = verdict(row) == "allow" and fetch_allowed(ent, self.cfg)  # H-2: only the approved entities
            if readable and _gate((row.get("docs_entry_urls") or [""])[0], row).read:
                urls += self._site_urls(row, terms)
            for url, title in urls:
                if url in seen or not host_ok(url, row):
                    continue
                seen.add(url)
                link = {"lane": LANE, "label": "Official", "entity": ent, "title": title or f"{ent} documentation",
                        "url": url, "read": False}
                perm = _gate(url, row)  # vera.source_permissions (I-A6, R-049/R-050): access + use_for_answer
                stop = _stop_code(should_stop)
                if not readable:
                    link["reason"] = perm.read_reason if not perm.read else "read_registry_not_allowed"  # D3: link only
                elif not perm.read:
                    link["reason"] = perm.read_reason  # never fetched: link only
                elif not perm.use_for_answer:
                    link["reason"] = perm.use_reason  # barred from answer use: reading-list reference only
                elif not page_ok(url, row):
                    link["reason"] = "path_not_allowed"  # deny all: not on the row's allowed_paths, never requested
                elif stop:
                    link["reason"] = stop  # deadline or stop: listed without a fetch
                elif not _delay_ok(url, row):
                    link["reason"] = "crawl_delay"  # inside the host's Crawl-delay: not sent early
                else:
                    try:
                        got = self._do_fetch(url, row)
                        final = str(got.get("url") or url)
                        from vera import web_route
                        post = web_route.evaluate_fetched(final, got, registry=list(self._rows.values()),
                                                          registry_row=row)  # noai, Content-Signal, derivative, login
                        if not page_ok(final, row):
                            link["reason"] = "off_registry"
                        elif not post.read:
                            link["reason"] = post.read_reason
                        elif not post.use_for_answer:
                            link["reason"] = post.use_reason
                        elif not (got.get("text") or "").strip():
                            link["reason"] = "empty"
                        else:
                            out["pages"].append(LanePage(LANE, ent, link["title"], final, outlet=_host(final),
                                                         text=str(got["text"])))
                            link["read"] = True
                    except Exception as exc:  # noqa: BLE001  FetchRefused or transport error: link only
                        link["reason"] = str(getattr(exc, "reason", "") or "fetch_refused")[:40]
                out["links"].append(link)
        return out


def _stop_code(should_stop) -> str:
    if should_stop is None:
        return ""
    try:
        return str(should_stop() or "")[:40]
    except Exception:  # noqa: BLE001  a failing stop check stops (fail closed)
        return "stopped"


def _ddg_site_search(host: str, terms: list) -> list:
    """The shared /ask DuckDuckGo Lite instance (vera.search_and_fetch.cached_chain): one explicit site: query."""
    from vera import search_and_fetch, search_providers as sp
    chain, _m, _mode = search_and_fetch.cached_chain()
    ddg = next((p for p in chain if getattr(p, "name", None) == sp.DDG_NAME), None)
    if ddg is None:
        return []
    q = {"group": "official", "query": f"{' '.join(terms)} (site:{host})"[:int(ddg.s.get("max_query_chars", 500))]}
    n = int(ddg.s.get("results_per_query", 10) or 10)
    return ddg.parse(ddg.call(q["query"], n, float(ddg.s.get("timeout_s", 10) or 10), queries=[q]), n)


def _gate(url: str, row: dict):
    """Pre-fetch permission record from the shared vera.source_permissions module (registry access_verdict and
    derivative_verdict, owner deny-list). robots.txt is checked by public_fetch on the pinned path. Lane pages are
    never ingested, whatever the ingest gate says (R-037)."""
    from vera import source_permissions
    return source_permissions.read_gate(url, registry=[row])
