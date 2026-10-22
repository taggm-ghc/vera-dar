"""RATE-1022: dynamic, per-portal rate-limit adapters for every external service VERA calls.

One `PortalAdapter` per portal (config/rate_limits.json, `portals`): a token bucket (rate_per_s, burst), a concurrency
cap, a daily request cap, and a circuit breaker with half-open probing. It adapts from responses:
  * Retry-After (seconds or HTTP-date) sets a cooldown, clamped by retry_after_max_s;
  * X-RateLimit-* / x-ratelimit-remaining-* headers (per-portal header specs) pace the remaining quota over its reset
    window, and a remaining count at or below `reserve` sets a cooldown until the reset;
  * AIMD: a throttle (429 / 503 / anti-bot 202, per-portal `throttle_statuses`, or an explicit throttled=True) halves
    the rate (decrease_factor) and sets a jittered, doubling cooldown; each success adds increase_frac x base rate,
    never above the documented base rate or a header-declared ceiling;
  * the breaker opens after `breaker_failures` consecutive throttles/failures, then admits `half_open_max` probe(s).

Deadline-aware and non-blocking by design: acquire() waits at most min(max_wait_s, deadline - now - min_call_s). A
portal that is cooling down, open, over its daily cap, or that would need a longer wait returns a Skip at once, so the
caller moves on to other providers. Nothing here sleeps for long inside a request.

Accounting (AGENTS.md 7a; owner rule feedback_rate_limit_not_an_attempt): every request SENT is counted in
`requests` (telemetry, portal accounting), throttled ones included. Outcome.counts_as_attempt is False for
RATE_LIMITED and for skips, so callers exclude them from eligible-attempt allowances. Spend and retry ceilings are
the callers' and are unchanged; the daily_cap here counts every request sent.

Process-level shared state: get_registry() returns one Registry per process (thread-safe; asyncio callers use
acquire_async). Telemetry (snapshot()) carries portal names, counters and state only: never hosts, URLs or keys.
No numbers live in this file: every limit is read from config/rate_limits.json (validated by load_config).
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import math
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "rate_limits.json"
CONTACT_EMAIL_ENV = "VERA_CONTACT_EMAIL"  # optional, unset by default; never logged
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,190}\.[A-Za-z]{2,24}")
_RATE_EPS = 1e-6           # numerical floor for a rate (avoids division by zero); not a tuning value
_DAY_S = 86400.0           # seconds per UTC day (daily_cap roll-over)
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")

# Caller's absolute monotonic deadline for the current request (inline budget). Set with deadline_scope(); threads
# started by a ThreadPoolExecutor need contextvars.copy_context().run to inherit it (see INTEGRATION.md).
DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar("vera_rate_deadline", default=None)


class RateLimitConfigError(ValueError):
    """config/rate_limits.json is missing, malformed or out of range."""


class RateLimited(RuntimeError):
    """Raised by the wrappers when a portal is skipped. The text says 'rate limit' so existing quota-error checks
    (search_providers._is_quota_error) classify it as RATE_LIMITED, not as a counted failure."""

    def __init__(self, portal: str, reason: str, retry_in_s: float = 0.0):
        self.portal, self.reason, self.retry_in_s = portal, reason, max(0.0, float(retry_in_s))
        super().__init__(f"{portal} rate limit: skipped ({reason}); retry in {self.retry_in_s:.1f}s")


class Outcome:
    OK = "ok"
    RATE_LIMITED = "rate_limited"
    FAILED = "failed"
    SKIPPED = "skipped"

    @staticmethod
    def counts_as_attempt(outcome: str) -> bool:
        """Only OK and FAILED use up attempt allowances; RATE_LIMITED and SKIPPED never do (owner rule)."""
        return outcome in (Outcome.OK, Outcome.FAILED)


# ------------------------------------------------------------------------------------------------ config
_DEFAULT_KEYS = {  # key: (type, lo, hi). Every value comes from the JSON file; these are only validation ranges.
    "increase_frac": (float, 0.0, 1.0), "decrease_factor": (float, 0.01, 0.99), "min_rate_factor": (float, 0.001, 1.0),
    "breaker_failures": (int, 1, 100), "breaker_open_s": (float, 0.1, 86400.0), "breaker_open_max_s": (float, 0.1, 86400.0),
    "half_open_max": (int, 1, 16), "jitter_frac": (float, 0.0, 1.0), "max_wait_s": (float, 0.0, 60.0),
    "min_call_s": (float, 0.0, 60.0), "retry_after_max_s": (float, 1.0, 86400.0), "base_cooldown_s": (float, 0.0, 86400.0),
    "max_cooldown_s": (float, 0.0, 86400.0), "poll_s": (float, 0.001, 5.0),
}
_PORTAL_KEYS = {"rate_per_s": (float, 1e-6, 1000.0), "burst": (float, 1.0, 1000.0), "max_concurrency": (int, 1, 64)}


def _num(where: str, v, typ, lo, hi):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (typ is int and not isinstance(v, int)):
        raise RateLimitConfigError(f"{where}: need a {typ.__name__}")
    if not (lo <= v <= hi) or (isinstance(v, float) and not math.isfinite(v)):
        raise RateLimitConfigError(f"{where}: {v} outside [{lo}, {hi}]")
    return typ(v)


def _ints(where: str, v) -> tuple:
    if not isinstance(v, list) or not all(isinstance(x, int) and not isinstance(x, bool) and 100 <= x <= 599 for x in v):
        raise RateLimitConfigError(f"{where}: need a list of HTTP status codes")
    return tuple(v)


def _header_specs(where: str, v) -> tuple:
    out = []
    for i, h in enumerate(v or ()):
        if not isinstance(h, dict) or not isinstance(h.get("remaining"), str):
            raise RateLimitConfigError(f"{where}[{i}]: need {{remaining, reset, reset_format, reserve}}")
        fmt = h.get("reset_format", "seconds")
        if fmt not in ("seconds", "epoch", "duration", "auto"):
            raise RateLimitConfigError(f"{where}[{i}].reset_format: seconds | epoch | duration | auto")
        out.append({"remaining": h["remaining"].lower(), "reset": str(h.get("reset") or "").lower() or None,
                    "limit": str(h.get("limit") or "").lower() or None, "reset_format": fmt,
                    "reserve": float(_num(f"{where}[{i}].reserve", h.get("reserve", 0), float if isinstance(h.get("reserve", 0), float) else int, 0, 10**9)),
                    # pace=false: the header counts budget units (credits, tokens), not requests: only the reserve applies
                    "pace": h.get("pace", True) is True})
    return tuple(out)


def _interval_spec(where: str, v):
    if v is None:
        return None
    if not isinstance(v, dict) or not isinstance(v.get("limit"), str) or not isinstance(v.get("interval"), str):
        raise RateLimitConfigError(f"{where}: need {{limit, interval}} header names")
    return {"limit": v["limit"].lower(), "interval": v["interval"].lower()}


def load_config(path: Path | str = CONFIG_PATH, env: Mapping | None = None) -> dict:
    """Read and validate config/rate_limits.json -> {"defaults": {...}, "portals": {name: {...}}}. A portal's
    `with_key` overrides apply only when the env var named by `key_env` is set (presence only; the value is never
    read into state or logged). Raises RateLimitConfigError."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RateLimitConfigError(f"rate limit config unreadable: {type(e).__name__}") from None
    return parse_config(raw, env)


def parse_config(raw, env: Mapping | None = None) -> dict:
    """Validate an already-parsed config object (see load_config)."""
    env = os.environ if env is None else env
    if not isinstance(raw, dict) or not isinstance(raw.get("defaults"), dict) or not isinstance(raw.get("portals"), dict):
        raise RateLimitConfigError("rate limit config: need objects 'defaults' and 'portals'")
    d = {k: _num(f"defaults.{k}", raw["defaults"].get(k), *spec) for k, spec in _DEFAULT_KEYS.items()}
    portals = {}
    for name, p in raw["portals"].items():
        if name.startswith("_"):
            continue
        if not isinstance(p, dict) or not re.fullmatch(r"[a-z][a-z0-9_]{1,40}", name):
            raise RateLimitConfigError(f"portals.{name}: need an object with a lowercase name")
        if p.get("key_env") and str(env.get(str(p["key_env"])) or "").strip() and isinstance(p.get("with_key"), dict):
            p = {**p, **p["with_key"]}
        if not isinstance(p.get("sources"), list) or not p["sources"]:
            raise RateLimitConfigError(f"portals.{name}.sources: every default needs its source (url, date)")
        c = {k: _num(f"portals.{name}.{k}", p.get(k), *spec) for k, spec in _PORTAL_KEYS.items()}
        for k, spec in _DEFAULT_KEYS.items():  # per-portal override of any default
            c[k] = _num(f"portals.{name}.{k}", p[k], *spec) if k in p else d[k]
        cap = p.get("daily_cap")
        c["daily_cap"] = None if cap is None else _num(f"portals.{name}.daily_cap", cap, int, 1, 10**7)
        c["enabled"] = p.get("enabled", True)
        if not isinstance(c["enabled"], bool):
            raise RateLimitConfigError(f"portals.{name}.enabled: true or false")
        hosts = p.get("hosts") or []
        if not isinstance(hosts, list) or not all(isinstance(h, str) and h == h.lower() for h in hosts):
            raise RateLimitConfigError(f"portals.{name}.hosts: need a list of lowercase host names")
        c["hosts"] = tuple(hosts)
        c["throttle_statuses"] = _ints(f"portals.{name}.throttle_statuses", p.get("throttle_statuses", [429, 503]))
        c["failure_statuses"] = _ints(f"portals.{name}.failure_statuses", p.get("failure_statuses", [500, 502, 504]))
        c["remaining_headers"] = _header_specs(f"portals.{name}.remaining_headers", p.get("remaining_headers"))
        c["interval_header"] = _interval_spec(f"portals.{name}.interval_header", p.get("interval_header"))
        cp = p.get("contact_param")
        if cp is not None and not (isinstance(cp, str) and re.fullmatch(r"[a-z_]{1,20}", cp)):
            raise RateLimitConfigError(f"portals.{name}.contact_param: a query parameter name or null")
        c["contact_param"] = cp
        if c["breaker_open_max_s"] < c["breaker_open_s"] or c["max_cooldown_s"] < c["base_cooldown_s"]:
            raise RateLimitConfigError(f"portals.{name}: a max below its base")
        portals[name] = c
    return {"defaults": d, "portals": portals}


# ------------------------------------------------------------------------------------------------ header parsing
def _hget(headers, name: str):
    if not headers or not name:
        return None
    try:
        v = headers.get(name)
        if v is None:
            for k in headers:
                if str(k).lower() == name:
                    return headers[k]
        return v
    except Exception:  # noqa: BLE001
        return None


def parse_duration(v) -> float | None:
    """'7.66s', '2m59.56s', '1h2m', '250ms', '12' (seconds) -> seconds."""
    s = str(v or "").strip().lower()
    if not s:
        return None
    try:
        return max(0.0, float(s))
    except ValueError:
        pass
    parts = _DURATION_RE.findall(s)
    if not parts or "".join(a + b for a, b in parts) != s:
        return None
    mult = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
    return sum(float(a) * mult[b] for a, b in parts)


def parse_retry_after(v, wall_now: float) -> float | None:
    """Retry-After: delta-seconds or an HTTP-date -> seconds from now (>= 0), or None if absent/garbled."""
    s = str(v or "").strip()
    if not s:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    try:
        dt = parsedate_to_datetime(s)
    except (TypeError, ValueError, IndexError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, dt.timestamp() - wall_now)


def _reset_seconds(v, fmt: str, wall_now: float) -> float | None:
    if v is None:
        return None
    if fmt == "duration":
        return parse_duration(v)
    try:
        x = float(str(v).strip())
    except ValueError:
        return parse_duration(v) if fmt == "auto" else None
    if fmt == "auto" and x > 10**12:  # epoch milliseconds
        x = x / 1000.0
    if fmt == "epoch" or (fmt == "auto" and x > 10**9):
        return max(0.0, x - wall_now)
    return max(0.0, x)


def _int_header(v) -> int | None:
    x = _float_header(v)
    return None if x is None else int(x)


def _float_header(v) -> float | None:
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


# ------------------------------------------------------------------------------------------------ adapter
@dataclass
class Skip:
    portal: str
    reason: str  # cooldown | breaker_open | half_open_busy | daily_cap | deadline | pacing | concurrency
    retry_in_s: float = 0.0
    ok: bool = False
    outcome: str = Outcome.SKIPPED

    def raise_(self):
        raise RateLimited(self.portal, self.reason, self.retry_in_s)


@dataclass
class Permit:
    adapter: "PortalAdapter"
    probe: bool = False
    waited_s: float = 0.0
    ok: bool = True
    outcome: str | None = None
    _done: bool = field(default=False, repr=False)

    def record(self, status: int | None = None, headers: Mapping | None = None, error: BaseException | None = None,
               throttled: bool | None = None) -> str:
        """Record the result of the request this permit admitted; returns the Outcome. Idempotent; also releases."""
        if self._done:
            return self.outcome or Outcome.OK
        self.outcome = self.adapter._record(self, status, headers, error, throttled)
        self.release()
        return self.outcome

    def release(self) -> None:
        if not self._done:
            self._done = True
            self.adapter._release(self)

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if not self._done:
            if ev is not None:
                self.record(error=ev)
            else:
                self.release()
        return False


class PortalAdapter:
    def __init__(self, name: str, c: Mapping, *, clock: Callable[[], float], wall: Callable[[], float],
                 sleep: Callable[[float], None], rand: Callable[[], float]):
        self.name, self.c = name, dict(c)
        self._clock, self._wall, self._sleep, self._rand = clock, wall, sleep, rand
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self.base = float(c["rate_per_s"])
        self.rate = self.base
        self.ceiling = self.base           # lowered by an interval header, never raised above base
        self.min_rate = self.base * float(c["min_rate_factor"])
        self.tokens = float(c["burst"])
        self._last_refill = clock()
        self.cooldown_until = 0.0
        self.breaker, self.open_until, self.open_s = "closed", 0.0, float(c["breaker_open_s"])
        self.fail_streak = self.throttle_streak = 0
        self.inflight = self.probes = 0
        self.day, self.day_used = None, 0
        self.n = {"requests": 0, "ok": 0, "rate_limited": 0, "failed": 0, "header_updates": 0}
        self.skips: dict[str, int] = {}

    # ---- internals (lock held)
    def _refill(self, now: float) -> None:
        dt = max(0.0, now - self._last_refill)
        self.tokens = min(float(self.c["burst"]), self.tokens + dt * self.rate)
        self._last_refill = now

    def _roll_day(self) -> None:
        day = datetime.fromtimestamp(self._wall(), timezone.utc).date().isoformat()
        if day != self.day:
            self.day, self.day_used = day, 0

    def _skip(self, reason: str, retry_in: float = 0.0) -> Skip:
        self.skips[reason] = self.skips.get(reason, 0) + 1
        return Skip(self.name, reason, retry_in)

    def _budget(self, now: float, deadline: float | None) -> float:
        b = float(self.c["max_wait_s"])
        if deadline is not None:
            b = min(b, deadline - now - float(self.c["min_call_s"]))
        return b

    def _admit_locked(self, now: float, budget: float):
        """-> Permit (with the token wait it must sleep), Skip, or None (concurrency full: caller may wait)."""
        if not self.c["enabled"]:
            self.inflight += 1
            self.n["requests"] += 1
            return Permit(self)
        if budget < 0:
            return self._skip("deadline")
        self._roll_day()
        cap = self.c["daily_cap"]
        if cap is not None and self.day_used >= cap:
            nxt = (math.floor(self._wall() / _DAY_S) + 1) * _DAY_S - self._wall()
            return self._skip("daily_cap", nxt)
        if now < self.cooldown_until:
            return self._skip("cooldown", self.cooldown_until - now)
        probe = False
        if self.breaker == "open":
            if now < self.open_until:
                return self._skip("breaker_open", self.open_until - now)
            self.breaker = "half_open"
        if self.breaker == "half_open":
            if self.probes >= int(self.c["half_open_max"]):
                return self._skip("half_open_busy", float(self.c["poll_s"]))
            probe = True
        if self.inflight >= int(self.c["max_concurrency"]):
            return None
        self._refill(now)
        wait = 0.0 if self.tokens >= 1.0 else (1.0 - self.tokens) / max(self.rate, _RATE_EPS)
        if wait > budget:
            return self._skip("deadline" if budget < float(self.c["max_wait_s"]) else "pacing", wait)
        self.tokens -= 1.0  # reserved now (may go negative): later callers queue behind this one
        self.inflight += 1
        self.probes += 1 if probe else 0
        self.day_used += 1
        self.n["requests"] += 1
        return Permit(self, probe=probe, waited_s=wait)

    def _concurrency_wait_steps(self, budget: float) -> int:
        return int(max(0.0, budget) / float(self.c["poll_s"])) + 1

    # ---- public
    def acquire(self, deadline: float | None = None):
        """Permit or Skip. Blocks at most min(max_wait_s, deadline - now - min_call_s); never longer."""
        deadline = DEADLINE.get() if deadline is None else deadline
        with self._cond:
            now = self._clock()
            budget = self._budget(now, deadline)
            res = self._admit_locked(now, budget)
            steps = self._concurrency_wait_steps(budget)
            while res is None and steps > 0:  # bounded: at most budget/poll_s slices, also with a frozen fake clock
                steps -= 1
                left = self._budget(self._clock(), deadline)
                if left <= 0:
                    break
                self._cond.wait(min(float(self.c["poll_s"]), left))
                now = self._clock()
                res = self._admit_locked(now, self._budget(now, deadline))
            if res is None:
                res = self._skip("concurrency", float(self.c["poll_s"]))
        if isinstance(res, Permit) and res.waited_s > 0:
            self._sleep(res.waited_s)
        return res

    async def acquire_async(self, deadline: float | None = None):
        """asyncio twin of acquire(): never blocks the event loop."""
        deadline = DEADLINE.get() if deadline is None else deadline
        now = self._clock()
        budget = self._budget(now, deadline)
        steps = self._concurrency_wait_steps(budget)
        while True:
            with self._lock:
                now = self._clock()
                res = self._admit_locked(now, self._budget(now, deadline))
            if res is not None or steps <= 0:
                break
            steps -= 1
            await asyncio.sleep(float(self.c["poll_s"]))
        if res is None:
            with self._lock:
                res = self._skip("concurrency", float(self.c["poll_s"]))
        if isinstance(res, Permit) and res.waited_s > 0:
            await asyncio.sleep(res.waited_s)
        return res

    def _release(self, permit: Permit) -> None:
        with self._cond:
            self.inflight = max(0, self.inflight - 1)
            if permit.probe:
                self.probes = max(0, self.probes - 1)
            self._cond.notify()

    def _classify(self, status, error, throttled) -> str:
        if throttled is True or (status is not None and int(status) in self.c["throttle_statuses"]):
            return Outcome.RATE_LIMITED
        if isinstance(error, RateLimited):
            return Outcome.RATE_LIMITED
        if error is not None or (status is not None and (int(status) in self.c["failure_statuses"])):
            return Outcome.FAILED
        return Outcome.OK

    def _apply_headers(self, headers, now: float, wall_now: float) -> None:
        if not headers:
            return
        iv = self.c["interval_header"]
        if iv:
            lim, per = _int_header(_hget(headers, iv["limit"])), parse_duration(_hget(headers, iv["interval"]))
            if lim and per:
                self.ceiling = max(self.min_rate, min(self.base, lim / per))
                self.rate = min(self.rate, self.ceiling)
                self.n["header_updates"] += 1
        for h in self.c["remaining_headers"]:
            rem = _float_header(_hget(headers, h["remaining"]))
            if rem is None:
                continue
            reset = _reset_seconds(_hget(headers, h["reset"]), h["reset_format"], wall_now) if h["reset"] else None
            self.n["header_updates"] += 1
            if rem <= h["reserve"]:
                hold = reset if reset is not None else float(self.c["base_cooldown_s"])
                self.cooldown_until = max(self.cooldown_until, now + min(hold, float(self.c["retry_after_max_s"])))
                self.tokens = min(self.tokens, 0.0)
            elif h["pace"] and reset and reset > 0:
                paced = (rem - h["reserve"]) / reset  # spread what is left over the window
                if paced < self.rate:  # may go below min_rate: the server's quota is the binding constraint
                    self.rate = max(_RATE_EPS, paced)

    def _breaker_fail(self, now: float, permit: Permit) -> None:
        self.fail_streak += 1
        if self.breaker == "half_open" and permit.probe:
            self.open_s = min(float(self.c["breaker_open_max_s"]), self.open_s * 2.0)
            self.breaker, self.open_until = "open", now + self._jitter(self.open_s)
        elif self.breaker == "closed" and self.fail_streak >= int(self.c["breaker_failures"]):
            self.breaker, self.open_until = "open", now + self._jitter(self.open_s)

    def _jitter(self, s: float) -> float:
        """Upward-only jitter: never shorter than the server asked or the backoff computed."""
        return s * (1.0 + float(self.c["jitter_frac"]) * max(0.0, min(1.0, self._rand())))

    def _record(self, permit: Permit, status, headers, error, throttled) -> str:
        with self._lock:
            if not self.c["enabled"]:
                return self._classify(status, error, throttled)
            now, wall_now = self._clock(), self._wall()
            out = self._classify(status, error, throttled)
            if out == Outcome.RATE_LIMITED:
                self.n["rate_limited"] += 1
                self.throttle_streak += 1
                self.rate = max(min(self.rate, self.min_rate), self.rate * float(self.c["decrease_factor"]))
                self.tokens = min(self.tokens, 0.0)
                ra = parse_retry_after(_hget(headers, "retry-after"), wall_now)
                if ra is not None:
                    cool = self._jitter(min(ra, float(self.c["retry_after_max_s"])))
                else:
                    back = float(self.c["base_cooldown_s"]) * (2.0 ** (self.throttle_streak - 1))
                    cool = self._jitter(min(back, float(self.c["max_cooldown_s"])))
                self.cooldown_until = max(self.cooldown_until, now + cool)
                self._breaker_fail(now, permit)
            elif out == Outcome.FAILED:
                self.n["failed"] += 1
                self._breaker_fail(now, permit)
            else:
                self.n["ok"] += 1
                self.fail_streak = self.throttle_streak = 0
                self.rate = min(self.ceiling, self.rate + float(self.c["increase_frac"]) * self.base)
                if self.breaker != "closed":
                    self.breaker, self.open_s = "closed", float(self.c["breaker_open_s"])
            self._apply_headers(headers, now, wall_now)  # last: the server's own numbers override the AIMD step
            return out

    def snapshot(self) -> dict:
        """Counters and state only: no hosts, URLs, keys or contact values."""
        with self._lock:
            now = self._clock()
            return {"requests": self.n["requests"], "ok": self.n["ok"], "rate_limited": self.n["rate_limited"],
                    "failed": self.n["failed"], "header_updates": self.n["header_updates"],
                    "skips": dict(sorted(self.skips.items())), "skipped": sum(self.skips.values()),
                    "rate_per_s": round(self.rate, 6), "base_rate_per_s": self.base, "ceiling_per_s": round(self.ceiling, 6),
                    "breaker": ("half_open" if self.breaker == "open" and now >= self.open_until else self.breaker),
                    "cooldown_s": round(max(0.0, self.cooldown_until - now), 3), "inflight": self.inflight,
                    "day_used": self.day_used, "daily_cap": self.c["daily_cap"], "enabled": self.c["enabled"]}


# ------------------------------------------------------------------------------------------------ registry
class Registry:
    def __init__(self, cfg: Mapping, *, clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, rand: Callable[[], float] = random.random):
        self.adapters = {n: PortalAdapter(n, c, clock=clock, wall=wall, sleep=sleep, rand=rand)
                         for n, c in cfg["portals"].items()}
        self._by_host = {h: n for n, c in cfg["portals"].items() for h in c["hosts"]}

    def portal(self, name: str) -> PortalAdapter:
        try:
            return self.adapters[name]
        except KeyError:
            raise RateLimitConfigError(f"no rate-limit portal named {name!r} in config/rate_limits.json") from None

    def portal_for_url(self, url: str) -> PortalAdapter | None:
        host = (urlsplit(str(url)).hostname or "").lower()
        n = self._by_host.get(host)
        return self.adapters.get(n) if n else None

    def acquire(self, portal: str, deadline: float | None = None):
        return self.portal(portal).acquire(deadline)

    def snapshot(self) -> dict:
        return {n: a.snapshot() for n, a in sorted(self.adapters.items())}

    def log_line(self) -> str:
        """One aggregate line for logs: 'ratelimit openalex=req:3,429:1,skip:2,rate:0.5,breaker:closed ...'."""
        parts = []
        for n, s in self.snapshot().items():
            parts.append(f"{n}=req:{s['requests']},429:{s['rate_limited']},fail:{s['failed']},skip:{s['skipped']},"
                         f"rate:{s['rate_per_s']:g},breaker:{s['breaker']}")
        return "ratelimit " + " ".join(parts)


_REG: Registry | None = None
_REG_LOCK = threading.Lock()


def get_registry(path: Path | str | None = None) -> Registry:
    """The process-wide registry (created on first use from config/rate_limits.json)."""
    global _REG
    with _REG_LOCK:
        if _REG is None:
            _REG = Registry(load_config(path or CONFIG_PATH))
        return _REG


def set_registry_for_tests(reg: Registry | None) -> None:
    global _REG
    with _REG_LOCK:
        _REG = reg


def health_aggregates() -> dict:
    """For /health/dar: per-portal counters and state (no hosts/keys). Empty dict if config fails to load."""
    try:
        return get_registry().snapshot()
    except RateLimitConfigError:
        return {}


@contextlib.contextmanager
def deadline_scope(deadline: float | None):
    """Bind the caller's absolute monotonic deadline for every acquire() in this context."""
    tok = DEADLINE.set(deadline)
    try:
        yield
    finally:
        DEADLINE.reset(tok)


def contact_params(portal: str, env: Mapping | None = None, registry: Registry | None = None) -> dict:
    """{contact_param: email} only when the portal declares one AND VERA_CONTACT_EMAIL is set and well-formed.
    Unset by default: no email is ever added otherwise."""
    env = os.environ if env is None else env
    p = (registry or get_registry()).portal(portal).c.get("contact_param")
    v = str(env.get(CONTACT_EMAIL_ENV) or "").strip()
    return {p: v} if p and _EMAIL_RE.fullmatch(v) else {}


# ------------------------------------------------------------------------------------------------ wrappers
def _status_headers(resp):
    return getattr(resp, "status_code", None), getattr(resp, "headers", None)


def call_with_limit(adapter: PortalAdapter, fn: Callable, *a, classify: Callable[[Any], bool] | None = None,
                    deadline: float | None = None, **kw):
    """Acquire, call fn, record. Raises RateLimited on a skip (fn is NOT called). `classify(resp) -> True` marks a
    response throttled (e.g. a DuckDuckGo anti-bot page served with 200)."""
    got = adapter.acquire(deadline)
    if not got.ok:
        got.raise_()
    try:
        resp = fn(*a, **kw)
    except BaseException as e:
        got.record(error=e)
        raise
    status, headers = _status_headers(resp)
    thr = None
    if classify is not None:
        try:
            thr = bool(classify(resp)) or None
        except Exception:  # noqa: BLE001  a classifier bug never breaks the call
            thr = None
    got.record(status=status, headers=headers, throttled=thr)
    return resp


def limited(portal: str, fn: Callable | None = None, *, classify: Callable[[Any], bool] | None = None,
            registry: Registry | None = None):
    """Wrap a transport / requests.get-compatible callable for ONE portal. Usable as limited("openalex", fn) or as a
    decorator @limited("openalex"). The registry is resolved at call time so tests can swap it."""
    def deco(f):
        def wrapped(*a, **kw):
            return call_with_limit((registry or get_registry()).portal(portal), f, *a, classify=classify, **kw)
        wrapped.__wrapped__ = f
        wrapped.rate_portal = portal
        return wrapped
    return deco(fn) if fn is not None else deco


def limited_by_url(fn: Callable, *, registry: Registry | None = None, classify: Callable[[Any], bool] | None = None):
    """Wrap a requests.get-compatible callable used for several hosts: the portal is chosen from the URL (first
    positional arg or url=). Hosts not in any portal pass through unlimited (and uncounted)."""
    def wrapped(*a, **kw):
        url = a[0] if a else kw.get("url", "")
        ad = (registry or get_registry()).portal_for_url(url)
        if ad is None:
            return fn(*a, **kw)
        return call_with_limit(ad, fn, *a, classify=classify, **kw)
    wrapped.__wrapped__ = fn
    return wrapped


def httpx_transport(portal: str, inner=None, *, registry: Registry | None = None):
    """An httpx transport that rate-limits every request of an OpenAI-SDK client (Groq, OpenRouter):
        OpenAI(..., http_client=httpx.Client(transport=rate_limit.httpx_transport("groq")))
    A skip raises RateLimited inside the SDK, which surfaces as an OpenAIError (APIConnectionError), so the existing
    provider chain moves to the next entry. Response headers (x-ratelimit-*, retry-after) feed the adapter."""
    import httpx

    class _LimitedTransport(httpx.BaseTransport):
        def __init__(self, inner_t):
            self.inner = inner_t or httpx.HTTPTransport()

        def handle_request(self, request):
            return call_with_limit((registry or get_registry()).portal(portal), self.inner.handle_request, request)

        def close(self):
            self.inner.close()

    return _LimitedTransport(inner)


def openai_http_client(base_url: str | None, timeout_s: float):
    """INT-1022-B (RATE-1022 A9/A10): an httpx.Client for an OpenAI-SDK client whose base URL maps to a configured
    portal (groq, openrouter), else None (the SDK's own client; never an error at request time). Config errors fail
    open to None here: a broken rate-limit file must not take the model chain down (the release gate validates it)."""
    if not base_url:
        return None
    try:
        ad = get_registry().portal_for_url(base_url)
    except RateLimitConfigError:
        return None
    if ad is None or not ad.c.get("enabled", True):  # a disabled portal is a pure pass-through: the SDK's own client
        return None
    import httpx
    return httpx.Client(transport=httpx_transport(ad.name), timeout=timeout_s)
