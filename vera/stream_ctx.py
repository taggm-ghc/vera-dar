"""STREAM-1031: per-request context of one streamed prompt (POST /ask/stream). Request-scoped, in-memory only; nothing
here survives the request. The legacy /ask never sets it, so `active()` is False there and every bounded behaviour
(45 s inline cut, the client wait constant, 125 s retry guard) stays exactly as before (OD-1031-NODEADLINE: the cut is removed on the
streaming path only).

Stop reasons (closed set; the first writer wins, except that "disconnected" overrides an earlier reason): "disconnected" (client gone; stop at the current candidate, keep the
admitted documents), "stalled" (no progress event for stall_s), "providers_exhausted" (OD-1031-TOKENS).
"""
from __future__ import annotations

import threading
import time
from contextvars import ContextVar

DISCONNECTED = "disconnected"
STALLED = "stalled"
PROVIDERS_EXHAUSTED = "providers_exhausted"
STOP_REASONS = frozenset({DISCONNECTED, STALLED, PROVIDERS_EXHAUSTED})


class StreamCancelled(Exception):
    """Raised at a safe point (before a token-using step) when the client is gone; never carries request data."""


class StreamCtx:
    def __init__(self, stall_s: float, clock=time.monotonic):
        self.stall_s = float(stall_s)
        self.clock = clock
        self.started = clock()
        self._last = self.started
        self._lock = threading.Lock()
        self._stop: str | None = None

    def touch(self) -> None:
        """A progress event happened (candidate evaluated, document stored, rung advanced)."""
        with self._lock:
            self._last = self.clock()

    def since_progress(self) -> float:
        with self._lock:
            return self.clock() - self._last

    def stop(self, reason: str) -> None:
        if reason in STOP_REASONS:
            with self._lock:
                if self._stop is None or (reason == DISCONNECTED and self._stop != DISCONNECTED):
                    self._stop = reason  # EXG-FIX-1031 D3: first reason wins, except a client disconnect always cancels

    def cancel(self) -> None:  # client disconnect
        self.stop(DISCONNECTED)

    @property
    def cancelled(self) -> bool:
        return self._stop == DISCONNECTED

    def stalled(self) -> bool:
        return self.since_progress() > self.stall_s

    def stop_reason(self) -> str | None:
        """The recorded stop reason, or "stalled" when the no-progress window has elapsed (recorded on first sight)."""
        if self._stop is None and self.stalled():
            self.stop(STALLED)
        return self._stop

    def should_stop(self) -> bool:
        return self.stop_reason() is not None


CTX: ContextVar[StreamCtx | None] = ContextVar("vera_stream_ctx", default=None)


def current() -> StreamCtx | None:
    return CTX.get()


def active() -> bool:
    return CTX.get() is not None


def check_cancelled() -> None:
    """Call before a token-using step: raises StreamCancelled when the client is gone. No-op off the stream path."""
    c = CTX.get()
    if c is not None and c.cancelled:
        raise StreamCancelled()
