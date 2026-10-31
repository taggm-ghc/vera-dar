"""STREAM-1031 / OD-1031-TOKENS: provider-exhaustion stop. No app-side token caps; the only stop is "every provider
in the chain is rate-limited/out of quota right now".

State is process-local (accepted divergence, plan s3.7 item 5): per chain entry an "exhausted until" time on an
injectable monotonic clock. Only HTTP 429 / quota errors mark an entry; other errors never do. A success clears it.
Retry-After (numeric seconds, from exc.response.headers) sets the time, else exhaustion.recheck_s (PROVISIONAL prior,
config/ask-provider-chain.json). At recovery at most one probe per entry goes out (try_acquire). Logs carry counts,
status codes and Retry-After values only, never message text, keys or question text.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from contextvars import ContextVar
from pathlib import Path

logger = logging.getLogger("vera")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "ask-provider-chain.json"
DEFAULT_RECHECK_S = 60.0
QUOTA_CODES = frozenset({"insufficient_quota", "rate_limit_exceeded", "quota_exceeded", "rate_limited",
                         "resource_exhausted", "tokens_exceeded"})


class ProvidersExhausted(RuntimeError):
    """Every configured provider is exhausted now; no LLM call is made. Carries no request data."""


_lock = threading.Lock()
_clock = time.monotonic
_keys: list[str] = []
_until: dict[str, float] = {}      # key -> monotonic time the exhaustion ends
_status: dict[str, int] = {}       # key -> last status code that marked it
_retry: dict[str, float] = {}      # key -> Retry-After / recheck seconds used
_probing: set[str] = set()


def recheck_s(path: Path | None = None) -> float:
    try:
        v = float(json.loads(Path(path or CONFIG_PATH).read_text())["exhaustion"]["recheck_s"])
        return v if v > 0 else DEFAULT_RECHECK_S
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_RECHECK_S


def set_clock(clock) -> None:
    global _clock
    _clock = clock


def reset_for_tests() -> None:
    global _clock
    with _lock:
        _clock = time.monotonic
        _keys.clear()
        _until.clear()
        _status.clear()
        _retry.clear()
        _probing.clear()


def register_chain(keys) -> None:
    """The chain's entry keys, in order. A changed set drops state of keys no longer present."""
    ks = list(keys)
    with _lock:
        if ks != _keys:
            _keys[:] = ks
            for d in (_until, _status, _retry):
                for k in [k for k in d if k not in ks]:
                    d.pop(k, None)
            _probing.intersection_update(ks)


def _status_of(exc) -> int | None:
    s = getattr(exc, "status_code", None)
    if s is None:
        s = getattr(getattr(exc, "response", None), "status_code", None)
    return s if isinstance(s, int) else None


def is_exhaustion(exc) -> bool:
    if _status_of(exc) == 429:
        return True
    if type(exc).__name__ == "RateLimitError":
        return True
    code = getattr(exc, "code", None)
    return isinstance(code, str) and code.lower() in QUOTA_CODES


def _retry_after(exc) -> float | None:
    try:
        v = float(exc.response.headers.get("retry-after"))
        return v if v >= 0 else None
    except (AttributeError, TypeError, ValueError):
        return None


def note_failure(key: str, exc) -> bool:
    """Release any probe slot; mark `key` exhausted when exc is a 429/quota error. Returns True when marked."""
    with _lock:
        _probing.discard(key)
    if not is_exhaustion(exc):
        return False
    ra = _retry_after(exc)
    secs = ra if ra is not None else recheck_s()
    with _lock:
        _until[key] = _clock() + secs
        _status[key] = _status_of(exc) or 429
        _retry[key] = secs
    return True


def note_success(key: str) -> None:
    with _lock:
        _probing.discard(key)
        _until.pop(key, None)
        _status.pop(key, None)
        _retry.pop(key, None)


def try_acquire(key: str) -> bool:
    """May a call go to `key` now? False while exhausted; at recovery only one caller gets the probe slot."""
    with _lock:
        u = _until.get(key)
        if u is None:
            return True
        if _clock() < u or key in _probing:
            return False
        _probing.add(key)
        return True


def exhausted() -> bool:
    """True only when every registered entry is exhausted right now (one healthy entry -> False)."""
    with _lock:
        if not _keys:
            return False
        now = _clock()
        return all(k in _until and now < _until[k] for k in _keys)


def stop_and_raise() -> None:
    """Log one verbose line, stop the active stream ctx (if any) and raise ProvidersExhausted."""
    from vera import stream_ctx
    with _lock:
        n = len(_keys)
        codes = [_status.get(k) for k in _keys if k in _status]
        ras = [round(_retry[k], 1) for k in _keys if k in _retry]
    logger.warning("/ask providers exhausted: providers=%d statuses=%s retry_after_s=%s; no further LLM calls",
                   n, codes, ras)
    c = stream_ctx.current()
    if c is not None:
        c.stop(stream_ctx.PROVIDERS_EXHAUSTED)
    raise ProvidersExhausted("all providers exhausted")


# OD-1031-EXHAUST-GRACE. WORK: per-request state dict set by inference_chain.answer_via_chain ("work" = True once the
# scope decision exists, i.e. work has started). GRACE: set only while a mid-response completion runs; it lets the
# guard pass and routes the OpenAI client/checker to the scoped completion (never set for a fresh prompt).
WORK: ContextVar = ContextVar("vera_exhaust_work", default=None)
GRACE: ContextVar = ContextVar("vera_exhaust_grace", default=None)
_grace_openai: tuple | None = None  # (model, pricing) set once by main at startup


def set_grace_openai(model, pricing) -> None:
    global _grace_openai
    _grace_openai = (model, pricing) if (model and pricing is not None) else None


def grace_openai():
    return _grace_openai


def note_work(**kw) -> None:
    """Record that work has started (and what exists) for a possible mid-response completion. No-op off the chain."""
    st = WORK.get()
    if st is not None:
        st["work"] = True
        st.update(kw)


def grace_cfg(path: Path | None = None) -> dict:
    try:
        g = json.loads(Path(path or CONFIG_PATH).read_text())["exhaustion"]["grace"]
        return g if isinstance(g, dict) else {}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def grace_openai_enabled(path: Path | None = None) -> bool:
    return grace_cfg(path).get("openai_completion") is True


def guard() -> None:
    """Call before a token-using call. Raises ProvidersExhausted when every provider is exhausted."""
    if GRACE.get() is not None:
        return
    if exhausted():
        stop_and_raise()
