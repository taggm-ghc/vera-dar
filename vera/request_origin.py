"""SEARCHCAPS-1110 (owner OD-1110-SEARCHCAPS): who a request is for, and the per-service visitor reserve.

Every request is a VISITOR request unless an authenticated API caller marks it origin="seed" (POST /ask only; see
main.AskRequest). The mark lives in a context variable for the life of that request and of the background jobs it
queues (vera.auto_acquire.Job.origin); it is never stored, logged with the question, or associated with a caller.

Policy per search service comes from config/rate_limits.json (caps as data), portal keys:
  * origin.seed_allowed (default true): false = seed requests never use the service (DuckDuckGo: owner "seeding never
    uses DuckDuckGo");
  * origin.visitor_reserve_frac (default 0.0): the share of the service's daily cap kept for visitor requests. A seed
    request is refused once the day's spend reaches floor(cap * (1 - frac)); a visitor request may use the whole cap.
  * search_daily_cap (optional; with_key may override it): the provider's daily search cap; absent = the provider's own
    settings value (config/vera_eval_run.json).
When the policy cannot be read, visitors are unaffected and seed requests are refused (fail closed for seeds only).
"""
from __future__ import annotations

import contextlib
import contextvars
import math
from typing import Mapping

VISITOR = "visitor"
SEED = "seed"
ORIGINS = (VISITOR, SEED)

ORIGIN: contextvars.ContextVar[str] = contextvars.ContextVar("vera_request_origin", default=VISITOR)


def current() -> str:
    v = ORIGIN.get()
    return v if v in ORIGINS else SEED  # an unknown value is treated as the restricted origin


def is_seed() -> bool:
    return current() != VISITOR


@contextlib.contextmanager
def scope(origin: str):
    """Mark the enclosed work (and contexts copied from it) as `origin`. Raises ValueError for an unknown origin."""
    if origin not in ORIGINS:
        raise ValueError("unknown request origin")
    tok = ORIGIN.set(origin)
    try:
        yield
    finally:
        ORIGIN.reset(tok)


def seed_cap(cap: int, reserve_frac: float) -> int:
    """The part of a daily cap a seed request may use: floor(cap * (1 - reserve_frac)), never below 0."""
    frac = min(1.0, max(0.0, float(reserve_frac)))
    return max(0, int(math.floor(int(cap) * (1.0 - frac) + 1e-9)))


def _portal_cfg(portal: str) -> Mapping | None:
    try:
        from vera import rate_limit
        return rate_limit.get_registry().portal(portal).c
    except Exception:  # noqa: BLE001  unknown portal or unreadable config
        return None


def policy(portal: str) -> dict:
    """{"seed_allowed", "visitor_reserve_frac", "search_daily_cap", "ok"} for a rate-limit portal. ok=False when the
    config could not be read: then seed_allowed is False (fail closed for seeds) and nothing else changes."""
    c = _portal_cfg(portal)
    if not isinstance(c, Mapping):
        return {"seed_allowed": False, "visitor_reserve_frac": 1.0, "search_daily_cap": None, "ok": False}
    return {"seed_allowed": bool(c.get("seed_allowed", True)),
            "visitor_reserve_frac": float(c.get("visitor_reserve_frac", 0.0)),
            "search_daily_cap": c.get("search_daily_cap"), "ok": True}


def refusal(portal: str, used: int, cap: int) -> str | None:
    """Why the CURRENT request may not spend one more call of `portal` given `used` of `cap` today, else None.
    Visitors: only the full cap. Seeds: excluded services, then the seed share."""
    if used >= cap:
        return "cap"
    if not is_seed():
        return None
    p = policy(portal)
    if not p["seed_allowed"]:
        return "seed_excluded"
    if used >= seed_cap(cap, p["visitor_reserve_frac"]):
        return "seed_share"
    return None
