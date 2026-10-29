"""PROG-1027: closed stage-event emitter and in-memory progress store (contract version "1").

Single definition of INTERFACE-CONTRACT v1.x s2. Events are exactly
{"v": "1", "seq": int, "ts": str, "stage": Stage, "state": State, "counts": {key: int}}; no free text, titles, URLs,
queries, reasons, categories or basis values are ever carried. `emit` drops anything outside the contract and logs a
drop counter only. Every public call is a no-op without a sink, never raises and never blocks.

Access control (plan s3b): a record is readable only with the per-browser-session token that created it. The store keeps
only sha256(token). Lookups run one constant-time compare path (dummy hash for unknown/expired/malformed ids).
Limits live in config/progress.json (PROVISIONAL, plan s3c); an invalid config disables progress, never /ask.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path

log = logging.getLogger("vera.progress")

CONTRACT_VERSION = "1"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "progress.json"
KILL_SWITCH_ENV = "VERA_PROGRESS_DISABLED"
_KILL_VALUES = ("1", "true", "yes", "on")  # case-insensitive
RID_RE = re.compile(r"^[0-9a-f]{32}\Z")
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43}\Z")


class Stage(str, Enum):
    RECEIVED = "received"
    SCOPE = "scope"
    CORPUS = "corpus"
    CORPUS_VERDICT = "corpus_verdict"
    SEARCH = "search"
    ACQUIRE = "acquire"
    WRITE = "write"
    CHECK = "check"
    RETRY = "retry"
    DONE = "done"


class State(str, Enum):
    START = "start"
    UPDATE = "update"
    DONE = "done"
    SKIPPED = "skipped"
    SUFFICIENT = "sufficient"  # CORPUS_VERDICT only
    INSUFFICIENT = "insufficient"
    STALE = "stale"
    ANSWERED = "answered"  # DONE only
    PARTIAL = "partial"
    NO_ANSWER = "no_answer"
    DECLINED = "declined"
    ERROR = "error"


COUNT_KEYS = frozenset({"inspected", "admitted", "found", "checked", "usable", "budget", "stored", "target", "queued", "attempt"})
_LIFECYCLE = frozenset({State.START, State.UPDATE, State.DONE, State.SKIPPED})
VALID: dict[Stage, frozenset] = {
    Stage.CORPUS_VERDICT: frozenset({State.SUFFICIENT, State.INSUFFICIENT, State.STALE}),
    Stage.DONE: frozenset({State.ANSWERED, State.PARTIAL, State.NO_ANSWER, State.DECLINED, State.ERROR}),
    Stage.RETRY: frozenset({State.START}),
    Stage.RECEIVED: frozenset({State.DONE}),
    **{s: _LIFECYCLE for s in (Stage.SCOPE, Stage.CORPUS, Stage.SEARCH, Stage.ACQUIRE, Stage.WRITE, Stage.CHECK)},
}
_QUEUED_TERMINALS = frozenset({State.ANSWERED, State.PARTIAL, State.NO_ANSWER})


# ------------------------------------------------------------------------------------------------ config
@dataclass(frozen=True)
class Config:
    enabled: bool
    max_records: int
    max_events_per_record: int
    max_events_per_poll: int
    ttl_s: float
    poll_interval_s: float
    poll_margin_s: float
    first_event_grace_s: float
    get_timeout_s: float
    max_count: int


_INT_KEYS = {"max_records": (1, 100000), "max_events_per_record": (2, 10000), "max_events_per_poll": (1, 10000),
             "max_count": (1, 10**9)}
_NUM_KEYS = {"ttl_s": (1, 86400), "poll_interval_s": (0.1, 60), "poll_margin_s": (0, 600), "first_event_grace_s": (0, 600),
             "get_timeout_s": (0.1, 60)}


def validate_config(raw) -> Config:
    """Raise ValueError naming every bad key (verbose failure). Keys starting with "_" are comments."""
    problems: list[str] = []
    if not isinstance(raw, dict):
        raise ValueError("progress config must be a JSON object")
    known = {"enabled", *_INT_KEYS, *_NUM_KEYS}
    for k in raw:
        if k not in known and not str(k).startswith("_"):
            problems.append(f"unknown key {k!r}")
    if not isinstance(raw.get("enabled"), bool):
        problems.append("enabled must be a boolean")
    vals: dict = {"enabled": raw.get("enabled") is True}
    for k, (lo, hi) in _INT_KEYS.items():
        v = raw.get(k)
        if type(v) is not int or not lo <= v <= hi:
            problems.append(f"{k} must be an int in [{lo}, {hi}]")
        vals[k] = v
    for k, (lo, hi) in _NUM_KEYS.items():
        v = raw.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            problems.append(f"{k} must be a number in [{lo}, {hi}]")
        vals[k] = v
    if not problems and raw["max_events_per_poll"] < raw["max_events_per_record"]:
        problems.append("max_events_per_poll must be >= max_events_per_record")
    if not problems and raw["get_timeout_s"] > 2 * raw["poll_interval_s"]:
        problems.append("get_timeout_s must be <= 2 x poll_interval_s")
    if problems:
        raise ValueError("; ".join(problems))
    return Config(**vals)


def load_config(path: Path = CONFIG_PATH) -> Config | None:
    """None (progress disabled) when the file is missing or invalid; logs the reason as keys/types only, never raises."""
    try:
        return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001
        log.error("progress config unusable, progress disabled (%s): %s", type(exc).__name__,
                  str(exc)[:300] if isinstance(exc, ValueError) else "")
        return None


CFG: Config | None = load_config()


def configure(cfg: Config | None) -> None:
    """Replace the active config and reset the store (tests; never called by app code)."""
    global CFG, STORE
    CFG = cfg
    STORE = Store()


def enabled() -> bool:
    """Single flag source: config `enabled` (ships false); env VERA_PROGRESS_DISABLED=1/true/yes/on forces off."""
    return CFG is not None and CFG.enabled and os.environ.get(KILL_SWITCH_ENV, "").strip().lower() not in _KILL_VALUES


def valid_rid(rid) -> bool:
    return isinstance(rid, str) and RID_RE.match(rid) is not None


def valid_token(tok) -> bool:
    return isinstance(tok, str) and TOKEN_RE.match(tok) is not None


# ------------------------------------------------------------------------------------------------ drop counters
_drop_lock = threading.Lock()
DROPS: dict[str, int] = {}


def _drop(kind: str) -> None:
    """Counter only: never a value, never a key name from the caller."""
    with _drop_lock:
        DROPS[kind] = DROPS.get(kind, 0) + 1


# ------------------------------------------------------------------------------------------------ record and store
clock = time.monotonic  # injectable in tests


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Record:
    """Append-only event list for one request. seq (from 1) and ts are assigned here under the lock."""

    def __init__(self, rid: str, token_hash: bytes):
        self.rid = rid
        self.token_hash = token_hash
        self.created = clock()
        self.closed_at: float | None = None
        self.queued = 0
        self._events: list[dict] = []
        self._lock = threading.Lock()
        self._done_seen = False

    @property
    def closed(self) -> bool:
        return self.closed_at is not None

    def append(self, stage: str, state: str, counts: dict) -> bool:
        with self._lock:
            if self.closed_at is not None:
                _drop("closed")
                return False
            terminal = stage == Stage.DONE.value
            if terminal and self._done_seen:
                _drop("second_done")
                return False
            if not terminal and len(self._events) >= (CFG.max_events_per_record if CFG else 0):
                _drop("full")  # the terminal is always kept
                return False
            self._events.append({"v": CONTRACT_VERSION, "seq": len(self._events) + 1, "ts": _utc_iso(),
                                 "stage": stage, "state": state, "counts": dict(counts)})
            if terminal:
                self._done_seen = True
            return True

    def close(self) -> None:
        with self._lock:
            if self.closed_at is None:
                self.closed_at = clock()

    def events_after(self, after: int, limit: int) -> list[dict]:
        with self._lock:
            return [dict(e, counts=dict(e["counts"])) for e in self._events if e["seq"] > after][:limit]


# One fixed per-process hash compared when no record exists, so known and unknown ids take the same path.
DUMMY_HASH = sha256(secrets.token_bytes(32)).digest()


class Store:
    def __init__(self):
        self._lock = threading.Lock()
        self._records: dict[str, Record] = {}  # insertion order = creation order

    def _expired(self, rec: Record, now: float) -> bool:
        base = rec.closed_at if rec.closed_at is not None else rec.created
        return now - base > CFG.ttl_s

    def _purge(self, now: float) -> None:
        for rid in [r for r, rec in self._records.items() if self._expired(rec, now)]:
            del self._records[rid]

    def create(self, rid: str, token_hash: bytes) -> Record | None:
        """None when the rid exists (first writer wins) or the store is full of open records."""
        with self._lock:
            now = clock()
            self._purge(now)
            if rid in self._records:
                _drop("duplicate_rid")
                return None
            if len(self._records) >= CFG.max_records:
                victim = next((r for r, rec in self._records.items() if rec.closed), None)
                if victim is None:
                    _drop("store_full")
                    return None
                del self._records[victim]
                _drop("evicted_closed")
            rec = Record(rid, token_hash)
            self._records[rid] = rec
            return rec

    def get_authorized(self, rid, header) -> Record | None:
        """Constant-time path (plan s3b): always hash the header and compare against the stored hash or DUMMY_HASH."""
        try:
            h = sha256((header if isinstance(header, str) else "").encode("utf-8", "replace")).digest()
            rec = None
            if valid_rid(rid) and CFG is not None:
                with self._lock:
                    cand = self._records.get(rid)
                    if cand is not None and not self._expired(cand, clock()):
                        rec = cand
            ok = hmac.compare_digest(h, rec.token_hash if rec is not None else DUMMY_HASH)
            return rec if (rec is not None and ok) else None
        except Exception as exc:  # noqa: BLE001
            log.warning("progress lookup failed (%s)", type(exc).__name__)
            return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


STORE = Store()
SINK: ContextVar[Record | None] = ContextVar("vera_progress_sink", default=None)


# ------------------------------------------------------------------------------------------------ producer API
def emit(stage, state, **counts) -> None:
    """Append one contract event to this context's record. No-op without a sink; never raises; never blocks."""
    rec = SINK.get()
    if rec is None:
        return
    try:
        st, sa = Stage(stage), State(state)
        if sa not in VALID[st]:
            _drop("state")
            return
        cap = CFG.max_count if CFG is not None else 0
        clean = {k: v for k, v in counts.items() if k in COUNT_KEYS and type(v) is int and 0 <= v <= cap}
        if len(clean) != len(counts):
            _drop("counts")  # a counter only, never the rejected values
        rec.append(st.value, sa.value, clean)
    except Exception as exc:  # noqa: BLE001
        _drop("exception")
        log.warning("progress emit failed (%s)", type(exc).__name__)


def set_queued(n) -> None:
    """INTERFACE-CONTRACT s2: the only transport of the background count (PROG-L's call site). No-op without a sink."""
    rec = SINK.get()
    if rec is None:
        return
    if type(n) is int and CFG is not None and 0 <= n <= CFG.max_count:
        rec.queued = n
    else:
        _drop("queued")


def open_record(request) -> Record | None:
    """Create the record for this request, or None. Never raises. None without both headers, with the flag off, or
    when `request` has no `headers` (in-process callers pass a SimpleNamespace)."""
    try:
        if not enabled():
            return None
        hdrs = getattr(request, "headers", None)
        if hdrs is None:
            return None
        rid, tok = hdrs.get("x-vera-request-id"), hdrs.get("x-vera-session")
        if not (valid_rid(rid) and valid_token(tok)):
            return None
        return STORE.create(rid, sha256(tok.encode("ascii")).digest())
    except Exception as exc:  # noqa: BLE001
        log.warning("progress open failed (%s)", type(exc).__name__)
        return None


# ------------------------------------------------------------------------------------------------ terminal T-map
def outcome_of(result_kind, completeness) -> State:
    """Closed T-map (plan s3a): answer+partial -> PARTIAL, answer -> ANSWERED, decline -> DECLINED, else NO_ANSWER
    (fail closed, mirrors answer_contract.outcome_for)."""
    if result_kind == "answer":
        return State.PARTIAL if completeness == "partial" else State.ANSWERED
    if result_kind == "decline":
        return State.DECLINED
    return State.NO_ANSWER


def outcome(resp) -> State:
    """Derive the terminal from an AskResponse (or its dict). Reads only result_kind and claim_check.answer_completeness."""
    try:
        if isinstance(resp, dict):
            rk, cc = resp.get("result_kind"), resp.get("claim_check")
        else:
            rk, cc = getattr(resp, "result_kind", None), getattr(resp, "claim_check", None)
        ac = cc.get("answer_completeness") if isinstance(cc, dict) else getattr(cc, "answer_completeness", None)
        return outcome_of(rk, ac)
    except Exception:  # noqa: BLE001
        return State.NO_ANSWER


def terminal_counts(state: State, rec: Record) -> dict:
    """`queued` rides only on answered/partial/no_answer (contract s2)."""
    return {"queued": rec.queued} if state in _QUEUED_TERMINALS else {}
