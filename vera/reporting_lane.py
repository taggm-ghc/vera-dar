"""NONASK-1103 S7: current-reporting lane portal clients (plan 6.4, 6.10): GDELT DOC 2.0 and HN Algolia.

Links-only: each item is title, url, published date, outlet domain; the portal label is kept internal (need-to-know).
One call per sub-query and portal, extracted terms only (vera.query_terms), never the user's sentence.
429/5xx/timeout: backoff 60/120/240 s with +-20% jitter, max 3 retries, then a 15-minute circuit-open (in process).
Fallback order GDELT then HN. The lane flag ships OFF (config/reporting_lane.json).

TODO gate (confirm format before enable): the GDELT JSON artlist field names (url, title, seendate, domain, ...) are
UNVERIFIED (both 2026-11-03 probes returned HTTP 429). Parsing is defensive; confirm against a live 200 before enable.
Enabling also needs SD.1 (disclosure) done and SEC passed. HTTP is injected (`http_get`); the default uses urllib and
is never exercised by tests.
"""
from __future__ import annotations

import json
import random
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
             "gdelt": {"maxrecords": 50, "timespan": "3months", "connect_timeout_s": 5, "read_timeout_s": 20},
             "hn": {"hitsPerPage": 20, "window_days": 90, "connect_timeout_s": 5, "read_timeout_s": 10}}


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    cfg = {**_DEFAULTS, **{k: v for k, v in raw.items() if not k.startswith("_")}}
    cfg["enabled"] = cfg.get("enabled") is True  # anything but literal true is OFF
    return cfg


def lane_enabled(cfg: dict | None = None) -> bool:
    return (cfg or load_config())["enabled"] is True


def gdelt_attribution() -> dict:
    """Credit line data for the UI (GDELT terms: cite the project and link it)."""
    return dict(GDELT_ATTRIBUTION)


def _domain(url: str) -> str:
    try:
        h = (urlsplit(url).hostname or "").lower()
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
        outlet = str(a.get("domain") or "").strip().lower() or _domain(url)
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


def _default_http_get(url: str, headers: dict, timeout: tuple) -> tuple[int, dict, bytes]:
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout[1]) as r:  # nosec - fixed https hosts only
            return r.status, dict(r.headers), r.read(2_000_000)
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
        self._open_until: dict[str, float] = {}
        self._lock = threading.Lock()

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

    def _call(self, portal: str, url: str, timeout: tuple) -> Any | None:
        """One logical call with 429/5xx/timeout backoff. None = unavailable (circuit opened)."""
        retries = int(self.cfg["max_retries"])
        for attempt in range(retries + 1):
            status, headers, body, retry_after = None, {}, b"", None
            try:
                status, headers, body = self._get(url, self._headers(), timeout)
            except Exception:  # timeout / network error counts as a transient failure
                status = None
            if status == 200:
                return body
            if status is not None and status not in (429,) and not 500 <= status < 600:
                return None  # other 4xx: not retryable, not a circuit event
            ra = {str(k).lower(): v for k, v in (headers or {}).items()}.get("retry-after")
            try:
                retry_after = float(ra) if ra is not None else None
            except ValueError:
                retry_after = None
            if attempt == retries:
                break
            self._sleep(self._backoff(attempt, retry_after))
        with self._lock:
            self._open_until[portal] = self._clock() + float(self.cfg["circuit_open_s"])
        return None

    def gdelt(self, terms: list[str]) -> list[dict] | None:
        c = self.cfg["gdelt"]
        q = build_query(terms)
        if not q:
            return None
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
        res = {"items": [], "attribution": None, "terms": []}
        if not self.cfg["enabled"]:
            return res
        terms = extract_terms(sub_query, known_entities)
        res["terms"] = terms
        if not terms:
            return res
        for portal in self.cfg["fallback_order"]:
            fn = {"gdelt": self.gdelt, "hn": self.hn}.get(portal)
            if fn is None or not self.available(portal):
                continue
            items = fn(terms)
            if items is None:
                continue
            res["items"] = sort_items(items)
            if portal == "gdelt":
                res["attribution"] = gdelt_attribution()
            return res
        return res
