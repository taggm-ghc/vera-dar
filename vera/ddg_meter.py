"""DDG-REDUCE-1110 WP0: a per-call-site DuckDuckGo meter. COUNTS ONLY (never a query, URL, title or question text).

A call site labels the work it starts with `site(label)` (a context variable, copied into the provider worker threads);
the DuckDuckGo provider reports every call it makes through `record(...)`: how many live POSTs it sent and how many
site-group queries the local cache served. The numbers go to
  * the request trace, as one event per call: stage "search", code "ddg_<site>", counts {posts, cache_hits};
  * a process-wide aggregate counter (`snapshot()`), per site and in total.
Sites: base (the grounding / reading-list search, the default), domain (domain-routing branches), inline (acquisition
round 0), variant (acquisition variant rounds), official (the official lane's site: query), freshness (the OD-1109-SIX
search on an answered question).
"""
from __future__ import annotations

import contextlib
import contextvars
import re
import threading

SITES = ("base", "domain", "inline", "variant", "official", "freshness")
SITE: contextvars.ContextVar[str] = contextvars.ContextVar("vera_ddg_site", default="base")
_LOCK = threading.Lock()
_AGG: dict[str, dict[str, int]] = {}


def current() -> str:
    v = SITE.get()
    return v if isinstance(v, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,23}", v) else "base"


@contextlib.contextmanager
def site(label: str):
    """Label the enclosed DuckDuckGo calls (and contexts copied from it) with the call site `label`."""
    tok = SITE.set(label)
    try:
        yield
    finally:
        SITE.reset(tok)


def record(posts: int, cache_hits: int = 0, *, calls: int = 1) -> None:
    """One provider call: `posts` live POSTs, `cache_hits` site-group queries served locally. Never raises."""
    try:
        s = current()
        with _LOCK:
            for key in (s, "total"):
                a = _AGG.setdefault(key, {"calls": 0, "posts": 0, "cache_hits": 0})
                a["calls"] += int(calls)
                a["posts"] += int(posts)
                a["cache_hits"] += int(cache_hits)
        from vera import request_trace as rt
        rt.event("search", f"ddg_{s}", counts={"posts": int(posts), "cache_hits": int(cache_hits)})
    except Exception:  # noqa: BLE001  a meter never fails a request
        pass


def snapshot() -> dict[str, dict[str, int]]:
    with _LOCK:
        return {k: dict(v) for k, v in _AGG.items()}


def reset() -> None:
    with _LOCK:
        _AGG.clear()
