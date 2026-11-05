"""NONASK-1103 I-A6 D5 (A6-W): request-local, answer-only web route. Pure: no store, cache, queue or logger ever sees
page text (R-037/R-045/R-047/R-048/R-050). Wiring into the inline loop is done by the caller.

Flow per call: select http(s) candidates with no DOI/PMCID/arXiv id -> READ gate (source_permissions.read_gate) BEFORE any
fetch -> at most one fetch per host -> vera.public_fetch.fetch (robots fail-closed) -> source_permissions.evaluate on the
response headers and meta-robots -> pages (READ and USE_FOR_ANSWER) or listed (reading list only, no page text).
Ingest is always False here (answer-only); the dev record says why. Robots Crawl-delay is enforced in public_fetch.
WP6-P2b (security-push2 S1/S2/M2): the READ gate (deny list, registry access_verdict forbid / links_only, login-path
heuristic) and the per-host round cap also run on EVERY redirect hop inside public_fetch, before anything is sent to
that hop (hop_gate); before each fetch the caller's stop/deadline signal and fetch budget (the existing
max_docs_per_prompt bound) are checked; robots.txt rules are cached per host for the whole call.
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
    fetched: int = 0                             # fetches started (counted against the caller's per-prompt bound)

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


_LOGIN_PATH_MARKERS = ("/login", "/signin", "/sign-in", "/sign_in", "/logon", "/auth/", "/account/login", "/subscribe",
                       "/paywall", "/register")  # fallback only; config/web_route.json login_path_markers governs


def _login_path(url: str, cfg: dict) -> bool:
    """Pre-request login/paywall heuristic for a hop (discard-only, uncalibrated, see _calibration_note): the URL path
    looks like a sign-in, registration, subscription or paywall page."""
    try:
        path = (urlsplit(url).path or "/").lower()
    except ValueError:
        return True
    marks = cfg.get("login_path_markers", _LOGIN_PATH_MARKERS)
    return any(path == m.rstrip("/") or path.startswith(m.rstrip("/") + "/") or path.startswith(m) for m in marks)


def _read_phrase(code: str, ph: dict) -> str:
    if code == "read_host_round_cap":
        return ph.get("host_round_cap", "")
    return sp._phrase(sp.load_config(), "read", code) or ph.get("fetch_failed", "")


def _login_wall(text: str, cfg: dict) -> bool:
    head = (text or "")[: int(cfg.get("marker_scan_chars", 600))].lower()
    return any(m in head for m in cfg.get("login_markers", []))


# WP6-P2b (verification-push2 D3): shared by the web route and both lanes (official_lane, lane_evidence) so every
# public-page path applies the same per-hop READ gate and the same post-fetch evaluate().
def make_hop_gate(start: str, *, registry=None, deny_list=None, cfg=None, hosts=None):
    """hop_gate for vera.public_fetch.fetch: the READ gate (deny list, registry access_verdict forbid / terms
    links_only), the pre-request login/paywall path heuristic and, when `hosts` is given, the per-round host cap,
    on every hop BEFORE robots.txt or the request is sent. Returns a refusal code or None."""
    cfg = cfg if cfg is not None else load_config()

    def gate(hop: str):
        g = sp.read_gate(hop, registry=registry, deny_list=deny_list)
        if not g.read:
            return g.read_reason
        if _login_path(hop, cfg):
            return "read_login_paywall"
        h = _host(hop)
        if hosts is not None and h != _host(start):
            if h in hosts:
                return "read_host_round_cap"  # S2 pacing: one fetch per host per round, redirects included
            hosts.add(h)
        return None
    return gate


def evaluate_fetched(final: str, got: dict, *, registry=None, registry_row=None, deny_list=None, cfg=None):
    """source_permissions.evaluate on a public_fetch result: X-Robots-Tag, Content-Signal (header and robots.txt),
    meta robots, the login/paywall text heuristic and the registry derivative rules. Signal strings only."""
    cfg = cfg if cfg is not None else load_config()
    text = str(got.get("text") or "")
    mr = str(got.get("meta_robots") or "")  # WP6-P2: normalised signal strings only (no raw headers)
    head = f'<meta name="robots" content="{mr}">' if mr and '"' not in mr else ""
    sig = {k: str(got[f]) for k, f in (("X-Robots-Tag", "x_robots_tag"), ("Content-Signal", "content_signal"))
           if got.get(f)}
    return sp.evaluate(final, headers=sig, html_head=head, registry=registry, registry_row=registry_row,
                       robots_allowed=True, deny_list=deny_list,
                       page_signals={"login_required": True} if _login_wall(text, cfg) else None)


def gated_fetch(url: str, *, fetch=None, registry=None, deny_list=None, cfg=None, robots_cache=None,
                hop_allowed=None):
    """One public page for a lane: the pre-request login path check, then fetch with the per-hop gate. Returns the
    fetch dict; raises public_fetch.FetchRefused (reason = a short code) when a hop is refused.
    NEWS-LANE-1104 W6 (SEC S1): `hop_allowed(hop_url) -> bool`, when given, is checked first on every hop (the start
    URL, each redirect target and its robots.txt request) BEFORE anything is sent; False refuses the hop."""
    from vera import public_fetch
    cfg = cfg if cfg is not None else load_config()
    f = fetch or public_fetch.fetch
    if _login_path(url, cfg):
        raise public_fetch.FetchRefused("read_login_paywall")
    if hop_allowed is not None and not hop_allowed(url):
        raise public_fetch.FetchRefused("read_lane_hop_refused")
    base = make_hop_gate(url, registry=registry, deny_list=deny_list, cfg=cfg)

    def _gate(hop: str):
        try:
            if hop_allowed is not None and not hop_allowed(hop):
                return "read_lane_hop_refused"
        except Exception:  # noqa: BLE001  a failing predicate refuses (fail closed)
            return "read_lane_hop_refused"
        return base(hop)
    kw = {"hop_gate": _gate}
    if robots_cache is not None:
        kw["robots_cache"] = robots_cache
    return f(url, **kw) if _takes_kwargs(f) else f(url)


def _takes_kwargs(f) -> bool:
    """True when an injected fetch accepts hop_gate (public_fetch.fetch, or a stub with **kwargs / hop_gate)."""
    import inspect
    try:
        ps = inspect.signature(f).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(p.kind is p.VAR_KEYWORD or p.name == "hop_gate" for p in ps)


def read_web(question, candidates, *, fetch=None, registry=None, deny_list=None, cfg=None, clock=time.monotonic,
             rate=None, should_stop=None, max_fetches=None) -> WebResult:
    """Never raises for a single bad candidate. `fetch` is vera.public_fetch.fetch-compatible
    (url, hop_gate=..., robots_cache=...) -> dict. `should_stop()` returns a short code ("deadline", "stopped") or None
    and is checked before EACH fetch; `max_fetches` bounds fetches started (None = no bound here; the inline caller
    passes its remaining per-prompt document budget). Candidates not fetched for either reason are listed only.
    `clock` and `rate` are accepted for the caller's rate gate."""
    cfg = cfg if cfg is not None else load_config()
    qmax = int(cfg.get("quote_max_chars", 200))
    ph = cfg.get("phrases", {})
    if fetch is None:
        from vera import public_fetch
        fetch = public_fetch.fetch
    from vera import public_fetch as _pf
    out = WebResult()
    hosts: set = set()
    robots_cache: dict = {}  # M2: parsed robots.txt rules per host for this call (rules only)
    halted = None  # first stop/deadline/budget code; every later candidate is listed without a fetch

    def hop_gate(hop: str, start: str):
        return make_hop_gate(start, registry=registry, deny_list=deny_list, cfg=cfg, hosts=hosts)(hop)
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
            if halted is None and max_fetches is not None and out.fetched >= max(0, int(max_fetches)):
                halted = "fetch_budget"
            if halted is None and should_stop is not None:
                halted = should_stop() or None
            if halted is not None:  # S2: nothing more is fetched in this call
                dev["gate"]["read"] = "read_" + str(halted)[:30]
                out.listed.append(_listed(c, url, title, cfg, ph.get(str(halted), ph.get("not_read", "")), qmax))
                continue
            hosts.add(host)
            out.fetched += 1
            try:
                got = fetch(url, hop_gate=lambda hop, _s=url: hop_gate(hop, _s), robots_cache=robots_cache)
                dev["fetched"] = True
            except _pf.FetchRefused as exc:
                reason = str(getattr(exc, "reason", "") or "refused")[:40]
                if reason.startswith("read_"):  # S1: a hop refused by the READ gate before any request to it
                    dev["gate"]["read"] = reason
                    phrase = _read_phrase(reason, ph)
                elif reason == "robots":
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
            post = evaluate_fetched(final, got, registry=registry, deny_list=deny_list, cfg=cfg)
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
