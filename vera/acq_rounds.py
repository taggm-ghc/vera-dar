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
NOT WIRED: no question-to-topic key exists in the acquisition paths yet (plan s4.2, leaf L5); callers pass a key once
it does. Integers and keys only; no question text is held.
"""
from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field

MET = "met"
RESOURCE_LIMITED = "resource_limited"
OPEN = "open"
DEFAULT_TARGET = 6


def target_of(acq: dict | None) -> int:
    v = ((acq or {}).get("bounds") or {}).get("acq_round_target_new_docs")
    return int(v) if isinstance(v, int) and not isinstance(v, bool) and v > 0 else DEFAULT_TARGET


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

    def __init__(self, keep_closed: int = 200):
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
            r = self._open.get(topic_key)
            if r is None:
                r = self._open[topic_key] = Round(next(self._ids), topic_key, target_of(acq))
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

    def closed(self) -> list:
        with self._lock:
            return [r.as_dict() for r in self._closed]


def job_exhausted(entries) -> bool:
    """Inline/background job: routes exhausted = no discovered candidate was left 'not_evaluated' (deadline or budget
    first) or 'not_used' (admitted but over the job's document budget). Empty discovery counts as exhausted."""
    return not any((e or {}).get("outcome") in ("not_evaluated", "not_used") for e in (entries or ()))
