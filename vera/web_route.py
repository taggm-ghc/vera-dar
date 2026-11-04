"""NONASK-1103 I-A6 D5 (A6-W): request-local, answer-only web route. Pure: no store, cache, queue or logger ever sees
page text (R-037/R-045/R-047/R-048/R-050). Wiring into the inline loop is done by the caller.

Flow per call: select http(s) candidates with no DOI/PMCID/arXiv id -> READ gate (source_permissions.read_gate) BEFORE any
fetch -> at most one fetch per host -> vera.public_fetch.fetch (robots fail-closed) -> source_permissions.evaluate on the
response headers and meta-robots -> pages (READ and USE_FOR_ANSWER) or listed (reading list only, no page text).
Ingest is always False here (answer-only); the dev record says why. Robots Crawl-delay: no hook exists (see config).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from vera import source_permissions as sp
from vera.official_lane import LanePage

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "web_route.json"
LANE = "web"
_NON_WEB_HOSTS = ("doi.org", "dx.doi.org", "arxiv.org", "ncbi.nlm.nih.gov", "europepmc.org")
_ID_KEYS = ("doi", "DOI", "pmcid", "PMCID", "arxiv_id", "arxiv", "pmid", "PMID")


def load_config(path: Path = CONFIG_PATH) -> dict:
    try:
        c = json.loads(Path(path).read_text(encoding="utf-8"))
        return c if isinstance(c, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


@dataclass(repr=False)
class WebPage(LanePage):
    """LanePage-compatible; lane 'web'. `text` is internal and never in repr."""
    answer_only: bool = True
    stored: bool = False
    epistemic: str = "web"
    label: str = "web source"

    def __repr__(self) -> str:
        return f"WebPage(url={self.url!r}, chars={len(self.text)})"

    __str__ = __repr__


@dataclass
class WebResult:
    pages: list = field(default_factory=list)
    listed: list = field(default_factory=list)   # reading-list entries, no page text
    dev: list = field(default_factory=list)      # per-candidate reason codes, no text, no host

    def __repr__(self) -> str:
        return f"WebResult(pages={len(self.pages)}, listed={len(self.listed)}, dev={len(self.dev)})"


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def is_web_candidate(c) -> bool:
    """http(s) URL, no DOI/PMCID/arXiv id (DDG/general web results only)."""
    if not isinstance(c, dict):
        return False
    url = str(c.get("url") or c.get("discovered_url") or "")
    if urlsplit(url).scheme not in ("http", "https") or not _host(url):
        return False
    if any(c.get(k) for k in _ID_KEYS) or any((c.get("ids") or {}).get(k) for k in ("doi", "pmcid", "arxiv", "pmid")):
        return False
    h = _host(url)
    return not any(h == d or h.endswith("." + d) for d in _NON_WEB_HOSTS)


def _clip(s, n: int) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _reference(c: dict, url: str, title: str) -> str:
    try:
        from vera import apa7
        rec = {"type": "webpage", "title": title or url, "URL": url}
        if c.get("issued"):
            rec["issued"] = c["issued"]
        if c.get("authors") or c.get("author"):
            rec["author"] = c.get("author") or c.get("authors")
        return apa7.reference(rec, links=True)
    except Exception:  # noqa: BLE001  reference is cosmetic; never block the route
        return f"{title or url}. {url}"


def _listed(c, url, title, cfg, phrase, qmax):
    return {"lane": LANE, "label": cfg.get("label", "web source"), "epistemic": "web", "title": title, "url": url,
            "reference": _reference(c, url, title),
            "characterisation": _clip(cfg.get("characterisation", {}).get("listed", "web source"), qmax) +
            (": " + _clip(c.get("snippet"), qmax) if c.get("snippet") else ""),
            "phrase": phrase, "read": False}


def _login_wall(text: str, cfg: dict) -> bool:
    head = (text or "")[: int(cfg.get("marker_scan_chars", 600))].lower()
    return any(m in head for m in cfg.get("login_markers", []))


def read_web(question, candidates, *, fetch=None, registry=None, deny_list=None, cfg=None, clock=time.monotonic,
             rate=None) -> WebResult:
    """Never raises for a single bad candidate. `fetch` is vera.public_fetch.fetch-compatible (url -> dict).
    `clock` and `rate` are accepted for the caller's rate gate; Crawl-delay has no hook (documented gap)."""
    cfg = cfg if cfg is not None else load_config()
    qmax = int(cfg.get("quote_max_chars", 200))
    ph = cfg.get("phrases", {})
    if fetch is None:
        from vera import public_fetch
        fetch = public_fetch.fetch
    from vera import public_fetch as _pf
    out = WebResult()
    hosts: set = set()
    for c in candidates or ():
        if not is_web_candidate(c):
            continue
        url = str(c.get("url") or c.get("discovered_url"))
        host = _host(url)
        title = _clip(c.get("title"), qmax) or url
        dev = {"gate": {}, "fetched": False, "ingest": False}
        out.dev.append(dev)
        try:
            pre = sp.read_gate(url, registry=registry, deny_list=deny_list)
            dev["gate"]["read"] = pre.read_reason
            if not pre.read:
                dev["gate"]["use"], dev["gate"]["ingest"] = pre.use_reason, "ingest_not_readable"
                out.listed.append(_listed(c, url, title, cfg, pre.read_phrase, qmax))
                continue
            if host in hosts:
                dev["gate"]["read"] = "read_host_round_cap"
                out.listed.append(_listed(c, url, title, cfg, ph.get("host_round_cap", ""), qmax))
                continue
            hosts.add(host)
            try:
                got = fetch(url)
                dev["fetched"] = True
            except _pf.FetchRefused as exc:
                reason = str(getattr(exc, "reason", "") or "refused")[:40]
                if reason == "robots":
                    dev["gate"]["read"] = "read_robots_disallow"
                    phrase = sp._phrase(sp.load_config(), "read", "read_robots_disallow")
                elif reason == "crawl_delay":
                    dev["gate"]["read"] = "read_crawl_delay"  # listed now; readable in a later request, never early
                    phrase = ph.get("crawl_delay", ph.get("fetch_failed", ""))
                else:
                    dev["gate"]["read"] = "fetch_" + reason
                    phrase = ph.get("fetch_failed", "")
                out.listed.append(_listed(c, url, title, cfg, phrase, qmax))
                continue
            final = str(got.get("url") or url)
            text = str(got.get("text") or "")
            mr = str(got.get("meta_robots") or "")  # WP6-P2: normalised signal strings only (no raw headers)
            head = f'<meta name="robots" content="{mr}">' if mr and '"' not in mr else ""
            sig = {k: str(got[f]) for k, f in (("X-Robots-Tag", "x_robots_tag"), ("Content-Signal", "content_signal"))
                   if got.get(f)}
            post = sp.evaluate(final, headers=sig, html_head=head, registry=registry,
                               robots_allowed=True, deny_list=deny_list,
                               page_signals={"login_required": True} if _login_wall(text, cfg) else None)
            dev["gate"].update(read=post.read_reason, use=post.use_reason, ingest="ingest_answer_only_wp",
                               ingest_evaluated=post.ingest_reason)
            if not post.read:
                out.listed.append(_listed(c, url, title, cfg, post.read_phrase, qmax))
            elif not post.use_for_answer:
                out.listed.append(_listed(c, url, title, cfg, post.use_phrase, qmax))
            elif not text.strip():
                dev["gate"]["read"] = "read_empty"
                out.listed.append(_listed(c, url, title, cfg, ph.get("empty", ""), qmax))
            else:
                out.pages.append(WebPage(LANE, "", title, final, outlet=_host(final), text=text))
        except Exception as exc:  # noqa: BLE001  one candidate never sinks the route; class name only, no text
            dev["gate"]["error"] = type(exc).__name__
            out.listed.append(_listed(c, url, title, cfg, ph.get("fetch_failed", ""), qmax))
    return out
