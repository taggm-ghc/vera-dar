"""FLOOR6-1028 (INTERIM, in-process only): per-micro-topic acquisition rounds and their outcome labels.

Owner 2026-10-28, verbatim: "Really, it should be: as each round of topic addition (initial or subsequent) should
target at least 6 new docs. If there simply are not 6 more new docs to be found, that's a resource limitation, not a
process failure."

A round belongs to one micro-topic key and targets `bounds.acq_round_target_new_docs` (6) NEW stored documents. The
job caps stand (6 jobs/h, 30/day, 2 docs/job, owner 2026-10-22), so one round spans several jobs. Each job of the round
reports how many documents it stored and whether its search routes were exhausted (no candidate left unevaluated for
lack of time or budget). Outcomes:
  - "open": fewer than the target so far and the routes are not exhausted (the next job of the topic continues it);
  - "met": >= target new documents stored in the round (the round closes; the next job opens a new round);
  - "resource_limited": a job of the round exhausted its routes with the round below target. A resource limitation,
    NOT an error or failure; the round closes and a later round may find more.
INTERIM: state lives in this process only and is lost on restart or spin-down (recorded divergence, plan s24); the
durable home is the acq_jobs table of the 010 migration draft (plan s23, work/dbstate-1028), which needs a round field.
WIRED (ROUNDS-1028, observational only): inline_acquire and auto_acquire call observe() at the end of each job that
ran its search. The topic key is the read-time micro_topic anchor key (the same key A4 serving and catstep's anchor
step use: micro_topic registry, best BM25 anchor at or above the scholarly threshold), a caller-supplied catstep key
when present, else "unassigned". observe() changes no cap, admission, licence or serving behaviour and never raises;
it logs one class-token line (key, counts, outcome; never question text). Integers and keys only; no question text
is held. The open-round table is bounded (bounds.acq_round_max_open, oldest least-recently-updated evicted).
"""
from __future__ import annotations

import itertools
import logging
import re
import threading
from dataclasses import dataclass, field

MET = "met"
RESOURCE_LIMITED = "resource_limited"
OPEN = "open"
DEFAULT_TARGET = 6
DEFAULT_MAX_OPEN = 200
UNASSIGNED = "unassigned"
logger = logging.getLogger("vera.autoacq")
_TOKEN = re.compile(r"[A-Za-z0-9_.:\-]{1,64}")


def target_of(acq: dict | None) -> int:
    v = ((acq or {}).get("bounds") or {}).get("acq_round_target_new_docs")
    return int(v) if isinstance(v, int) and not isinstance(v, bool) and v > 0 else DEFAULT_TARGET


def max_open_of(acq: dict | None) -> int:
    v = ((acq or {}).get("bounds") or {}).get("acq_round_max_open")
    return int(v) if isinstance(v, int) and not isinstance(v, bool) and v > 0 else DEFAULT_MAX_OPEN


@dataclass
class Round:
    round_id: int
    topic_key: str
    target: int
    new_docs: int = 0
    jobs: int = 0
    outcome: str = OPEN
    paths: dict = field(default_factory=dict)  # "inline" / "background" -> jobs

    def as_dict(self) -> dict:
        return {"round_id": self.round_id, "topic_key": self.topic_key, "target": self.target,
                "new_docs": self.new_docs, "jobs": self.jobs, "outcome": self.outcome, "paths": dict(self.paths)}


class RoundTracker:
    """Thread-safe. One open round per topic key; closed rounds are kept (bounded) for reporting."""

    def __init__(self, keep_closed: int = 200, max_open: int = DEFAULT_MAX_OPEN):
        self._max_open = max(1, int(max_open))
        self.evicted = 0
        self._lock = threading.Lock()
        self._open: dict = {}
        self._closed: list = []
        self._ids = itertools.count(1)
        self._keep = int(keep_closed)

    def record_job(self, topic_key: str, stored: int, exhausted: bool, acq: dict | None = None,
                   path: str = "background") -> dict:
        """Add one finished job's stored count to the topic's open round (opening one if none). Returns the round."""
        if not topic_key:
            raise ValueError("topic_key required")
        stored = max(0, int(stored))
        with self._lock:
            r = self._open.pop(topic_key, None)  # re-inserted below: dict order = least recently updated first
            if r is None:
                while len(self._open) >= self._max_open:  # bound: evict the oldest open round (state only, no cap effect)
                    del self._open[next(iter(self._open))]
                    self.evicted += 1
                r = Round(next(self._ids), topic_key, target_of(acq))
            self._open[topic_key] = r
            r.jobs += 1
            r.new_docs += stored
            r.paths[path] = r.paths.get(path, 0) + 1
            if r.new_docs >= r.target:
                r.outcome = MET
            elif exhausted:
                r.outcome = RESOURCE_LIMITED
            if r.outcome != OPEN:
                del self._open[topic_key]
                self._closed.append(r)
                del self._closed[: max(0, len(self._closed) - self._keep)]
            return r.as_dict()

    def open_round(self, topic_key: str) -> dict | None:
        with self._lock:
            r = self._open.get(topic_key)
            return r.as_dict() if r else None

    def open_count(self) -> int:
        with self._lock:
            return len(self._open)

    def closed(self) -> list:
        with self._lock:
            return [r.as_dict() for r in self._closed]


def job_exhausted(entries) -> bool:
    """Inline/background job: routes exhausted = no discovered candidate was left 'not_evaluated' (deadline or budget
    first) or 'not_used' (admitted but over the job's document budget). Empty discovery counts as exhausted."""
    return not any((e or {}).get("outcome") in ("not_evaluated", "not_used") for e in (entries or ()))


_TRACKER: RoundTracker | None = None
_TLOCK = threading.Lock()


def tracker(acq: dict | None = None) -> RoundTracker:
    """The process-wide tracker (in-process only: DV-ACQ-FLOOR6-1)."""
    global _TRACKER
    with _TLOCK:
        if _TRACKER is None:
            _TRACKER = RoundTracker(max_open=max_open_of(acq))
        return _TRACKER


def reset_for_tests() -> None:
    global _TRACKER
    with _TLOCK:
        _TRACKER = None


def topic_key(question: str, acq: dict | None, hint: str = "") -> str:
    """Micro-topic key of a question: a caller-supplied catstep key (token-shaped) when present, else the read-time
    micro_topic anchor key (best BM25 registry anchor at or above the scholarly threshold, as A4 and catstep's anchor
    step), else "unassigned". Pure and read-only; never raises."""
    try:
        if isinstance(hint, str) and hint and hint != UNASSIGNED and _TOKEN.fullmatch(hint):
            return hint
        from vera import micro_topic
        mcfg = (acq or {}).get("micro_topic") or {}
        reg = micro_topic.load_registry(mcfg)
        b = reg.best(question)
        if b is not None and b[0] >= micro_topic._threshold(mcfg):
            k = micro_topic.key_id(micro_topic._entry_key("registry", b[1]["id"], round(b[0], 4), reg,
                                                          b[1].get("field")))
            if k:
                return k
    except Exception:  # noqa: BLE001
        pass
    return UNASSIGNED


def observe(question: str, acq: dict | None, stored: int, exhausted: bool, path: str, hint: str = "") -> dict | None:
    """Wiring entry point: record one finished job against its topic's round and log one class-token line (topic key
    and counts only, never question text). Observational: never raises, changes nothing else."""
    try:
        key = topic_key(question, acq, hint)
        r = tracker(acq).record_job(key, stored, bool(exhausted), acq, path)
        logger.info("autoacq round topic=%s round=%d target=%d new_docs=%d jobs=%d stored=%d exhausted=%s outcome=%s path=%s",
                    key if _TOKEN.fullmatch(key) else "other", r["round_id"], r["target"], r["new_docs"], r["jobs"],
                    max(0, int(stored)), str(bool(exhausted)).lower(), r["outcome"], path)
        return r
    except Exception:  # noqa: BLE001
        return None


def routes_exhausted(n_ranked: int, n_admitted: int, budget: int, deadline_hit: bool) -> bool:
    """Inline job: no ranked candidate was left unevaluated. The evaluation stage runs until the job's document budget
    is filled or every candidate has finished, so: nothing ranked, or (budget not filled and no deadline cut it short).
    A filled budget keeps the round open (the routes may hold more)."""
    if n_ranked <= 0:
        return True
    return n_admitted < max(0, budget) and not deadline_hit
