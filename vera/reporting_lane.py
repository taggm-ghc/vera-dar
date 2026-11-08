"""NONASK-1103 S7: current-reporting lane portal clients (plan 6.4, 6.10): GDELT DOC 2.0 and HN Algolia.

Links-only: each item is title, url, published date, outlet domain; the portal label is kept internal (need-to-know).
One call per sub-query and portal, extracted terms only (vera.query_terms), never the user's sentence.
429/5xx/timeout: backoff 60/120/240 s with +-20% jitter, max 3 retries, then a 15-minute circuit-open (in process).
Fallback order from config (shipped: GDELT only). The lane flag ships ON since NEWS-LANE-1104 (config/reporting_lane.json).

GDELT JSON artlist field names (url, url_mobile, title, seendate, socialimage, domain, language, sourcecountry) were
confirmed by a live 200 on 2026-11-04 (work/NEWS-LANE-1104/compliance.md); parsing stays defensive. HTTP is injected
(`http_get`); the default uses urllib and is never exercised over the network by tests.

NEWS-LANE-1104 (owner 2026-11-04 "enable"): HN is out of fallback_order (no first-party terms; work/NEWS-LANE-1104/
compliance.md). Outlet terms policy (outlet_verdict): each item's host is matched (exact or subdomain) against the
`outlets` list; "exclude" items are dropped; "links_only" and unlisted items (unlisted_verdict) carry read_ok=False and
are emitted only when emit_link_only is true (set once vera.lane_evidence honours read_ok, wiring W2); "allow" items
carry read_ok=True. A per-host crawl_delay_s (from robots.txt) limits how often items of that host are emitted for
reading. With outlet_query "allowlist" the GDELT query is restricted to the allow hosts (domain: operators).
GDELT calls keep min_interval_s apart in process (first-party 429 text: "one every 5 seconds").
The default HTTP client follows no redirects and only calls the two fixed API hosts.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode, urlsplit

from vera.query_terms import build_query, extract_terms

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "reporting_lane.json"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
HN_URL = "https://hn.algolia.com/api/v1/search"
HN_ITEM = "https://news.ycombinator.com/item?id="
GDELT_ATTRIBUTION = {"text": "News index: GDELT Project", "url": "https://www.gdeltproject.org/"}

_DEFAULTS = {"enabled": False, "fallback_order": ["gdelt", "hn"], "user_agent": "VERA-DAR/1.0",
             "backoff_s": [60, 120, 240], "jitter_frac": 0.2, "max_retries": 3, "circuit_open_s": 900,
             "outlets": [], "unlisted_verdict": "links_only", "emit_link_only": False, "outlet_query": "all",
             "min_interval_s": 6,
             "gdelt": {"maxrecords": 50, "timespan": "3months", "connect_timeout_s": 5, "read_timeout_s": 20},
             "hn": {"hitsPerPage": 20, "window_days": 90, "connect_timeout_s": 5, "read_timeout_s": 10}}


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    cfg = {**_DEFAULTS, **{k: v for k, v in raw.items() if not k.startswith("_")}}
    cfg["enabled"] = cfg.get("enabled") is True  # anything but literal true is OFF
    if Path(path) == CONFIG_PATH:  # LISTS-TO-DB-1104: news.outlets + news.fallback_order from the DB when enabled
        try:
            from vera import list_tables
            cfg = list_tables.overlay("reporting_lane.json", cfg)
        except Exception:  # noqa: BLE001  loader missing or faulty: keep the file value
            pass
    return cfg


def lane_enabled(cfg: dict | None = None) -> bool:
    return (cfg or load_config())["enabled"] is True


def gdelt_attribution() -> dict:
    """Credit line data for the UI (GDELT terms: cite the project and link it)."""
    return dict(GDELT_ATTRIBUTION)


def _domain(url: str) -> str:
    try:
        h = (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""
    return h[4:] if h.startswith("www.") else h


def _http_url(u: Any) -> str:
    u = str(u or "").strip()
    return u if urlsplit(u).scheme in ("http", "https") and _domain(u) else ""


def _parse_date(v: Any) -> str | None:
    """Return ISO date (YYYY-MM-DD) or None. Accepts GDELT 20261103T120000Z, ISO strings, epoch seconds."""
    if v in (None, ""):
        return None
    try:
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(float(v), tz=timezone.utc).date().isoformat()
        s = str(v).strip()
        for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt).date().isoformat()
            except ValueError:
                pass
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date().isoformat()
    except (ValueError, OverflowError, OSError):
        return None


def _item(title: Any, url: str, date: str | None, outlet: str, portal: str) -> dict:
    return {"title": " ".join(str(title or "").split())[:300] or outlet, "url": url, "published": date,
            "outlet": outlet, "lane": "reporting", "_portal": portal}  # _portal internal, never shown


def parse_gdelt(body: Any) -> list[dict]:
    """Defensive artlist parse; tolerates missing fields and wrong shapes (field names UNVERIFIED)."""
    if isinstance(body, (bytes, str)):
        try:
            body = json.loads(body)
        except ValueError:
            return []
    arts = body.get("articles") if isinstance(body, dict) else None
    if not isinstance(arts, list):
        return []
    out = []
    for a in arts:
        if not isinstance(a, dict):
            continue
        url = _http_url(a.get("url") or a.get("url_mobile"))
        if not url:
            continue
        outlet = _domain(url)  # SEC minor: the label comes from the URL host, never GDELT's own domain field
        out.append(_item(a.get("title"), url, _parse_date(a.get("seendate") or a.get("date")), outlet, "gdelt"))
    return out


def parse_hn(body: Any) -> list[dict]:
    if isinstance(body, (bytes, str)):
        try:
            body = json.loads(body)
        except ValueError:
            return []
    hits = body.get("hits") if isinstance(body, dict) else None
    if not isinstance(hits, list):
        return []
    out = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        url = _http_url(h.get("url"))
        if not url and h.get("objectID"):  # Ask/Show posts: the HN item page itself
            url = HN_ITEM + str(h["objectID"])
        if not url:
            continue
        date = _parse_date(h.get("created_at_i")) or _parse_date(h.get("created_at"))
        out.append(_item(h.get("title") or h.get("story_title"), url, date, _domain(url), "hn"))
    return out


def sort_items(items: list[dict]) -> list[dict]:
    """Newest first; undated last."""
    return sorted(items, key=lambda i: (i.get("published") is None, _neg(i.get("published"))))


def _neg(d: str | None) -> str:
    return "".join(chr(0x10FFFF - ord(c)) for c in d) if d else ""


_WORD = re.compile(r"[A-Za-z0-9]+")
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def event_core(text: str, cfg: dict | None = None) -> dict | None:
    """WP4 (MIDTERMS-WP4-1106): the event core of a news question, or None when it names no event noun.

    Only config event nouns (news_intent.event_nouns) and the latest named year are used, so filler, request or meta
    words can never reach GDELT. Returns {"phrase": "midterm elections", "words": [...], "year": "2026"|None,
    "synonyms": [...]} (synonyms come from the optional config news_intent.event_synonyms {noun: [..]}).
    GDELT DOC 2.0 query syntax used: a double-quoted exact phrase, and OR inside parentheses (the same form this
    module already uses for domain: operators); terms separated by a space are ANDed.
    """
    ni = (cfg if cfg is not None else load_config()).get("news_intent")
    if not isinstance(ni, dict):
        return None
    nouns = [str(n).lower().split() for n in ni.get("event_nouns") or [] if str(n).strip()]
    if not nouns:
        return None
    toks = [t.lower() for t in _WORD.findall(str(text or ""))]
    run: list[str] = []
    i = 0
    while i < len(toks):
        hit = next((n for n in sorted(nouns, key=len, reverse=True) if toks[i:i + len(n)] == n), None)
        if hit:
            run.extend(hit)
            i += len(hit)
        elif run:
            break
        else:
            i += 1
    if not run:
        return None
    years = _YEAR.findall(str(text or ""))
    syn = ni.get("event_synonyms") if isinstance(ni.get("event_synonyms"), dict) else {}
    syns: list[str] = []
    for w in run:
        for x in syn.get(w) or []:
            x = " ".join(_WORD.findall(str(x))).lower()
            if x and x not in syns and x != " ".join(run):
                syns.append(x)
    return {"phrase": " ".join(run), "words": list(dict.fromkeys(run)), "year": max(years) if years else None,
            "synonyms": syns}


def core_queries(core: dict) -> list[str]:
    """[precise, broader]: the quoted phrase (with any synonyms OR-ed) plus the year; then the event words OR-ed plus
    the year. The broader query is dropped when it equals the precise one."""
    y = " " + core["year"] if core.get("year") else ""
    alts = [f'"{core["phrase"]}"'] + [f'"{x}"' for x in core.get("synonyms") or []]
    q1 = (alts[0] if len(alts) == 1 else "(" + " OR ".join(alts) + ")") + y
    words = list(dict.fromkeys(list(core["words"]) + [w for x in core.get("synonyms") or [] for w in x.split()]))
    q2 = (words[0] if len(words) == 1 else "(" + " OR ".join(words) + ")") + y
    return [q1] if q2 == q1 or q2 == core["phrase"] + y else [q1, q2]


VERDICTS = ("allow", "links_only", "exclude")
API_HOSTS = frozenset({"api.gdeltproject.org", "hn.algolia.com"})
MAX_BODY = 2_000_000


def _new_state() -> dict:
    return {"open": {}, "last": {}, "emit": {}, "fails": {}, "next": {}, "lock": threading.Lock()}


_SHARED = _new_state()  # per-process lane state (circuit, backoff, spacing, crawl delay); never holds page text


def _host_match(host: str, entry: Any) -> bool:
    e = str(entry or "").lower().strip().rstrip(".")
    e = e[4:] if e.startswith("www.") else e
    return bool(e) and (host == e or host.endswith("." + e))


def outlet_row(url: str, cfg: dict | None = None) -> dict | None:
    """The outlets row for the URL host (exact host or a parent domain; www. ignored), else None."""
    h = _domain(url)
    if not h:
        return None
    for o in (cfg or load_config()).get("outlets") or []:
        if isinstance(o, dict) and _host_match(h, o.get("host")):
            return o
    return None


def _raw_outlet_verdict(url: str, cfg: dict | None = None) -> str:
    """allow | links_only | exclude from the row alone (unlisted host gets unlisted_verdict; an "allow" row without
    terms_evidence or any unknown verdict counts as links_only). PRIVATE: only vera.news_access (design A1) may call it;
    every lane must use outlet_verdict(), which adds the ND access checks (H19 review F6)."""
    cfg = cfg or load_config()
    if not _domain(url):
        return "exclude"
    row = outlet_row(url, cfg)
    if row is None:
        v = cfg.get("unlisted_verdict", "links_only")
        return v if v in ("links_only", "exclude") else "links_only"
    v = row.get("terms_verdict")
    if v == "allow" and not row.get("terms_evidence"):
        return "links_only"
    return v if v in VERDICTS else "links_only"


def outlet_verdict(url: str, cfg: dict | None = None) -> str:
    """allow | links_only | exclude. An unlisted host gets unlisted_verdict (default links_only). An "allow" row
    without terms_evidence, or any unknown verdict, counts as links_only (fail closed).
    H19: an "allow" row that carries an ND `access` block is "allow" only when licence_rules.json nd_news.enabled is
    true (ships off) AND vera.news_access.full_text_ok(url) holds (NC-BASIS-1 SATISFIED and every record-level A-check);
    otherwise, and on any error, links_only (review F1)."""
    cfg = cfg or load_config()
    v = _raw_outlet_verdict(url, cfg)
    if v == "allow":
        row = outlet_row(url, cfg)
        if isinstance(row, dict) and isinstance(row.get("access"), dict):
            try:
                from vera import licence_gate, news_access
                if not (licence_gate.nd_news_enabled() and news_access.full_text_ok(url, cfg)):
                    return "links_only"
            except Exception:  # noqa: BLE001  fail closed
                return "links_only"
    return v


def allow_hosts(cfg: dict | None = None) -> list[str]:
    cfg = cfg or load_config()
    return [str(o["host"]).lower() for o in cfg.get("outlets") or [] if isinstance(o, dict) and o.get("host")
            and outlet_verdict("https://" + str(o["host"]) + "/", cfg) == "allow"]


def _no_redirect_handler():
    """urllib redirect handler that refuses every redirect (a 3xx is returned as an HTTPError, never followed)."""
    import urllib.request

    class _H(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    return _H()


def _default_http_get(url: str, headers: dict, timeout: tuple) -> tuple[int, dict, bytes]:
    sp = urlsplit(url)
    try:
        port = sp.port
    except ValueError:
        return 0, {}, b""
    if (sp.scheme != "https" or (sp.hostname or "").lower() not in API_HOSTS or port not in (None, 443)
            or sp.username is not None or sp.password is not None):
        return 0, {}, b""  # only the two fixed API hosts, https, default port, no userinfo
    # API-ALLOW-1107: permission gate (config/api_permissions.json) BEFORE anything is sent; ApiDenied propagates to
    # ReportingLane._call, which skips the portal. A permitted call goes through the entry's rate-limit portal.
    from vera import api_permissions, rate_limit
    d = api_permissions.require(url, "metadata")  # API-URL-GATE-1108b (6): explicit purpose (content_scope checked)
    headers = api_permissions.headers_for(d, headers)
    permit = rate_limit.get_registry().portal(str(d.conditions["rate_portal"])).acquire()
    if not permit.ok:
        permit.raise_()  # RateLimited: nothing sent
    try:
        status, rhdrs, body = _open(url, headers, timeout)
    except BaseException as e:
        permit.record(error=e)
        raise
    permit.record(status=status or None, headers=rhdrs)
    return status, rhdrs, body


def _open(url: str, headers: dict, timeout: tuple) -> tuple[int, dict, bytes]:
    """The urllib request itself; only ever called by _default_http_get after the permission gate."""
    import urllib.error
    import urllib.request
    opener = urllib.request.build_opener(_no_redirect_handler())
    req = urllib.request.Request(url, headers=headers)
    deadline = time.monotonic() + float(timeout[0]) + float(timeout[1])  # total wall-clock bound for the body read
    try:
        with opener.open(req, timeout=timeout[1]) as r:  # nosec - fixed https API hosts only, no redirects
            ctype = str(r.headers.get("Content-Type", "")).lower()
            parts, n = [], 0
            while n <= MAX_BODY:
                if time.monotonic() > deadline:
                    return 0, {}, b""
                chunk = r.read(min(65536, MAX_BODY + 1 - n))
                if not chunk:
                    break
                parts.append(chunk)
                n += len(chunk)
            body = b"".join(parts)
            if len(body) > MAX_BODY or not any(t in ctype for t in ("json", "javascript", "text/plain")):
                return 0, {}, b""  # oversize or not JSON: not used, not retried
            return r.status, dict(r.headers), body
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), b""


class ReportingLane:
    """Holds the per-portal circuit state in process. Inject http_get, sleep, clock, rand for tests."""

    def __init__(self, cfg: dict | None = None, *, http_get: Callable | None = None, sleep: Callable = time.sleep,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 rand: Callable[[], float] = random.random):
        self.cfg = cfg or load_config()
        self._get = http_get or _default_http_get
        self._sleep, self._clock, self._wall, self._rand = sleep, clock, wall, rand
        # NEWS-LANE-1104 SEC B1: the production lane (default HTTP client, default clock) shares one per-process
        # state, so the circuit, backoff and min interval hold across requests (lane_evidence builds a lane per
        # request). An injected http_get or clock (tests) gets its own state.
        self.last_why: str | None = None  # PERF-MISS-1105-FIX2 F2: closed token for the last _call (trace only)
        st = _SHARED if (http_get is None and clock is time.monotonic) else _new_state()
        self._open_until, self._last_call, self._last_emit, self._fails = st["open"], st["last"], st["emit"], st["fails"]
        self._next = st["next"]  # WP4: broader core query deferred to the next call (key: the precise query)
        self._lock = st["lock"]

    def available(self, portal: str) -> bool:
        with self._lock:
            return self._clock() >= self._open_until.get(portal, 0.0)

    def _headers(self) -> dict:
        return {"User-Agent": self.cfg["user_agent"], "Accept": "application/json"}

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        steps = self.cfg["backoff_s"]
        base = steps[min(attempt, len(steps) - 1)]
        if retry_after is not None and retry_after > base:
            base = retry_after
        j = float(self.cfg["jitter_frac"])
        return base * (1.0 + j * (2.0 * self._rand() - 1.0))

    def _space(self, portal: str) -> bool:
        """Keep calls to one portal at least min_interval_s apart (per process). Never sleeps in the request path:
        a call inside the interval is skipped (False = portal unavailable for this call, the caller falls through)."""
        gap = float(self.cfg.get("min_interval_s") or 0)
        with self._lock:
            now = self._clock()
            last = self._last_call.get(portal)
            if last is not None and now - last < gap:
                return False
            self._last_call[portal] = now
        return True

    def _call(self, portal: str, url: str, timeout: tuple) -> Any | None:
        """One attempt per logical call; never sleeps for backoff inside the request (SEC B1). On 429/5xx/timeout
        the portal is marked unavailable for the next backoff step (60/120/240 s +-jitter, or Retry-After when
        longer), and after max_retries consecutive failures for circuit_open_s; every wait is clamped to
        circuit_open_s. None = unavailable (the caller falls through)."""
        if not self._space(portal):
            self.last_why = "spacing"  # PERF-MISS-1105-FIX2 F2: why the call returned None (trace only)
            return None
        status, headers, body = None, {}, b""
        from vera import api_permissions, rate_limit
        try:
            status, headers, body = self._get(url, self._headers(), timeout)
        except api_permissions.ApiDenied:  # API-ALLOW-1107: not permitted, nothing sent; not a failure or a backoff
            self.last_why = "api_denied"
            return None
        except rate_limit.RateLimited:  # the portal's limiter skipped the call (nothing sent): like spacing
            self.last_why = "spacing"
            return None
        except Exception:  # timeout / network error counts as a transient failure
            status = None
        if status == 200:
            with self._lock:
                self._fails[portal] = 0
            self.last_why = "ok"
            return body
        if status is not None and status not in (0, 429) and not 500 <= status < 600:
            self.last_why = "http_4xx"
            return None  # other 4xx: not retryable, not a circuit event
        self.last_why = "throttled" if status == 429 else "failed"
        # 0 (client refused: oversize, non-JSON, read deadline) counts as a transient failure (SEC-RR R3)
        ra = {str(k).lower(): v for k, v in (headers or {}).items()}.get("retry-after")
        try:
            retry_after = float(ra) if ra is not None else None
        except ValueError:
            retry_after = None
        cap = float(self.cfg["circuit_open_s"])
        with self._lock:
            n = self._fails.get(portal, 0)
            self._fails[portal] = n + 1
            wait = cap if n >= int(self.cfg["max_retries"]) else self._backoff(n, retry_after)
            self._open_until[portal] = self._clock() + max(0.0, min(wait, cap))
        return None

    def gdelt(self, terms: list[str], query: str | None = None) -> list[dict] | None:
        c = self.cfg["gdelt"]
        q = query or build_query(terms)
        if not q:
            return None
        if self.cfg.get("outlet_query") == "allowlist":
            hosts = allow_hosts(self.cfg)
            if not hosts:
                return []
            dq = " OR ".join("domain:" + h for h in hosts)
            q = f"{q} ({dq})" if len(hosts) > 1 else f"{q} {dq}"
        url = GDELT_URL + "?" + urlencode({"query": q, "mode": "artlist", "format": "json", "sort": "datedesc",
                                           "maxrecords": c["maxrecords"], "timespan": c["timespan"]})
        body = self._call("gdelt", url, (c["connect_timeout_s"], c["read_timeout_s"]))
        return None if body is None else parse_gdelt(body)

    def hn(self, terms: list[str]) -> list[dict] | None:
        c = self.cfg["hn"]
        q = build_query(terms)
        if not q:
            return None
        since = int(self._wall() - timedelta(days=int(c["window_days"])).total_seconds())
        url = HN_URL + "?" + urlencode({"query": q, "tags": "story", "hitsPerPage": c["hitsPerPage"],
                                        "numericFilters": f"created_at_i>{since}"})
        body = self._call("hn", url, (c["connect_timeout_s"], c["read_timeout_s"]))
        return None if body is None else parse_hn(body)

    def search(self, sub_query: str, known_entities=()) -> dict:
        """One call per sub-query. Returns {'items': [...], 'attribution': {...}|None, 'terms': [...]}.

        The first available portal in fallback order that answers (even with zero hits) is used; an unavailable
        or failed portal falls through to the next. Lane flag OFF -> no call at all.
        """
        # PERF-MISS-1105-FIX2 F2: res["diag"] = one {portal, status: called|skipped, reason, items} per portal tried
        # (closed tokens and counts only; for the request trace, never shown to a user).
        res = {"items": [], "attribution": None, "terms": [], "diag": []}
        if not self.cfg["enabled"]:
            res["diag"].append({"portal": "-", "status": "skipped", "reason": "off", "items": 0})
            return res
        terms = extract_terms(sub_query, known_entities)
        res["terms"] = terms
        if not terms:
            res["diag"].append({"portal": "-", "status": "skipped", "reason": "no_terms", "items": 0})
            return res
        for portal in self.cfg["fallback_order"]:
            fn = {"gdelt": self.gdelt, "hn": self.hn}.get(portal)
            if fn is None:
                continue
            if not self.available(portal):
                res["diag"].append({"portal": portal, "status": "skipped", "reason": "circuit", "items": 0})
                continue
            self.last_why = None
            core = event_core(sub_query, self.cfg) if portal == "gdelt" else None
            qs = core_queries(core) if core else []
            attempts, deferred = 1, False
            first = qs[0] if qs else None
            was_deferred = bool(qs and self._next.get(qs[0]))
            if was_deferred:  # a broader query was deferred earlier: use it now (cleared once a call is made)
                first = qs[1]
            items = self.gdelt(terms, first) if qs else fn(terms)
            if was_deferred and items is not None:
                self._next.pop(qs[0], None)
            # an empty precise result earns ONE broader core query. The lane never sleeps in the request path
            # (SEC B1) and never shortens min_interval_s: if the spacing has not elapsed the broader query is
            # deferred to the next call (retry_deferred). A GDELT 429 opens the backoff window, so available()
            # is False and nothing is retried.
            if qs and first == qs[0] and items is not None and not items and len(qs) > 1 and self.available(portal):
                self.last_why = None
                items2 = self.gdelt(terms, qs[1])
                if items2 is not None:
                    items, attempts = items2, 2
                elif self.last_why == "spacing":
                    self._next[qs[0]] = qs[1]
                    deferred = True
            if items is None:
                why = self.last_why or "no_query"
                src = {"spacing": "local_spacing", "throttled": "gdelt_response"}.get(why)
                res["diag"].append({"portal": portal, "status": "skipped" if why in ("spacing", "no_query", "api_denied") else "called",
                                    "reason": why, "items": 0, "attempts": attempts,
                                    **({"throttle_source": src} if src else {})})
                continue
            raw_n = len(items)
            res["items"] = self._apply_outlet_policy(sort_items(items))
            res["diag"].append({"portal": portal, "status": "called",
                                "reason": "ok" if res["items"] else ("no_items" if raw_n == 0 else "outlet_policy"),
                                "items": len(res["items"]), "raw_items": raw_n, "attempts": attempts,
                                **({"retry_deferred": True} if deferred else {})})
            if portal == "gdelt":
                res["attribution"] = gdelt_attribution()
            return res
        return res

    def _apply_outlet_policy(self, items: list[dict]) -> list[dict]:
        """Drop exclude; tag read_ok; drop read_ok=False items unless emit_link_only; per-host crawl delay for reads."""
        out, now = [], self._clock()
        emit_link = self.cfg.get("emit_link_only") is True
        for it in items:
            v = outlet_verdict(it.get("url", ""), self.cfg)
            if v == "exclude":
                continue
            if v == "allow":
                row = outlet_row(it["url"], self.cfg) or {}
                delay = float(row.get("crawl_delay_s") or 0)
                key = str(row.get("host") or "").lower()
                with self._lock:
                    last = self._last_emit.get(key)
                    ok = delay <= 0 or last is None or now - last >= delay
                    if ok and delay > 0:
                        self._last_emit[key] = now
                if ok:
                    out.append(dict(it, read_ok=True))
                    continue
            if emit_link:  # links_only, unlisted, or an allow host inside its crawl delay: link only
                out.append(dict(it, read_ok=False))
        return out
