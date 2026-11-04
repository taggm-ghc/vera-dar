"""NONASK-1103 S4 (plan 6.3, 6.7, 6.8a; owner decisions R-037 and 2026-11-03 "honour site terms").

Official lane: for each registry entity named in a sub-query, the registry's own documentation entry URLs. No search
call is made and no host is ever guessed (an entity without an active registry row gets nothing).

Terms gate (owner, 2026-11-03): every registry row carries `terms_verdict` (allow | links_only | unknown) and a
`terms_evidence` pointer. A page is fetched ONLY when the verdict is "allow" AND robots.txt permits the fetch
(vera.public_fetch checks robots on the same pinned path and fails closed). Any other verdict, including a missing or
unknown one, yields a link only. The fetched URL (after redirects) must stay on one of the row's exact hosts.

Page text is held in memory for this answer only (R-037): LanePage keeps it out of repr(), so a log line, a trace or a
traceback that renders the object never carries it. This module imports no store, cache or queue writer.

The lane flag ships OFF (config/subquery.json official_lane.enabled). Enable only via docs/lanes-enable-checklist.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

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


def verdict(row: dict) -> str:
    """allow | links_only; unknown, missing or malformed verdicts mean links_only (owner 2026-11-03)."""
    v = row.get("terms_verdict") if isinstance(row, dict) else None
    return "allow" if v == "allow" and row.get("terms_evidence") else "links_only"


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def host_ok(url: str, row: dict) -> bool:
    """Exact host match against the row's domains (no wildcard, no suffix match)."""
    h = _host(url)
    return bool(h) and h in {str(d).lower() for d in (row.get("domains") or [])}


def _rows(registry: list | None) -> dict:
    if registry is None:
        from vera import subquery
        registry = subquery.load_entity_registry()
    return {r["entity"]: r for r in registry if isinstance(r, dict) and r.get("state") == "active" and r.get("entity")}


class OfficialLane:
    """Inject `fetch` (vera.public_fetch.fetch signature: url -> {"text", "url", "content_type"}) in tests."""

    def __init__(self, cfg: dict | None = None, *, registry: list | None = None, fetch: Callable | None = None,
                 site_search: Callable | None = None):
        self.cfg = cfg if cfg is not None else load_config()
        self._rows = _rows(registry)
        self._fetch = fetch
        self._site_search = site_search  # (host, terms) -> [{"url", "title"}]; None = the shared DDG provider

    def _do_fetch(self, url: str) -> dict:
        if self._fetch is not None:
            return self._fetch(url)
        from vera import public_fetch
        return public_fetch.fetch(url)

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

    def run(self, entities, terms=()) -> dict:
        """{'links': [public-safe dicts], 'pages': [LanePage]} for the entities of one sub-query. Flag off: nothing.
        Never raises; a refused or failed fetch leaves the link only (the reason is a short code, never page text)."""
        out = {"links": [], "pages": []}
        if not lane_enabled(self.cfg):
            return out
        seen: set = set()
        for ent in entities or ():
            row = self._rows.get(ent)
            if row is None:  # no registry hit: the entity is named, no host is assigned (no guessing)
                continue
            urls = [(u, f"{ent} documentation") for u in row.get("docs_entry_urls") or []]
            if _gate((row.get("docs_entry_urls") or [""])[0], row).read:  # site: only for readable hosts
                urls += self._site_urls(row, terms)
            for url, title in urls:
                if url in seen or not host_ok(url, row):
                    continue
                seen.add(url)
                link = {"lane": LANE, "label": "Official", "entity": ent, "title": title or f"{ent} documentation",
                        "url": url, "read": False}
                perm = _gate(url, row)  # vera.source_permissions (I-A6, R-049/R-050): access + use_for_answer
                if not perm.read:
                    link["reason"] = perm.read_reason  # never fetched: link only
                elif not perm.use_for_answer:
                    link["reason"] = perm.use_reason  # barred from answer use: reading-list reference only
                else:
                    try:
                        got = self._do_fetch(url)
                        final = str(got.get("url") or url)
                        if not host_ok(final, row):
                            link["reason"] = "off_registry"
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
