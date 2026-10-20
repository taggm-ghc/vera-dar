"""Run-level budget and usage metering for the slim live MVP (S1).

Three jobs:
  1. RunBudget: one $/seconds cap for a run (M2-M5), sticky once tripped, checked BEFORE each
     call so no new call starts after a trip. It raises RunBudgetExceeded, which is a
     judge.BudgetExceeded (RuntimeError) and deliberately NOT an LLMError.
  2. obs_from_m2_ledger / obs_from_m3_log: turn the existing M2 CostLedger and M3 CallLog into CallObs.
  3. engineered_usage / attach_usage: make engineered usage COMPLETE or explicitly INCOMPLETE and
     keep that flag all the way to M6's Cost/Latency scoring.

Known swallowing site (reviewer finding, verified): vera/gate_a.py `_score_batch` catches every
Exception around `llm_call`, so a budget check placed INSIDE a Gate A llm_call would quietly become
"deferred candidates". Callers must run `budget.check(...)` OUTSIDE the wrapped score_fn / llm_call
(for example before calling Gate A and between stages). As a safety net the trip is sticky: even if
an exception is swallowed, `budget.tripped` stays set and the next `check` / `raise_if_tripped`
re-raises, so the swallowed trip still stops the run at the next boundary.

Named constants (accepted divergence from the hardcoded-data standard; fold into
config/vera_eval_run.json in goal #2): see UNKNOWN_COST_NOTE and REQUIRED_STAGES below.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from vera.m6.judge import BudgetExceeded

# Stages that must each have produced at least one meter for usage to be called complete.
# Why: a stage with zero observations means its spend was not captured (an unmetered seam), and
# "complete" must not be claimed from silence. Goal #2 key: pipeline.usage.required_stages.
REQUIRED_STAGES = ("m2", "m3", "m45")

# Marker M3's client writes when a model has no pricing record (vera/m3/llm.py). Why: that record has
# cost 0.0 but the cost is unknown, not free (RD-2 strict policy).
M3_UNPRICED_MARKER = "no pricing record"

# Why: error text is stored in reports; bound it so a provider response cannot bloat or leak.
MAX_ERROR_CHARS = 120

# Activity of an M2 ledger entry that is plain HTTP (not a billable API call).
M2_NON_BILLABLE_KIND = "fetch"


class RunBudgetExceeded(BudgetExceeded):
    """The run-level cost or time cap tripped. Not an LLMError: stage code must not swallow it."""


@dataclass
class CallObs:
    stage: str
    activity: str
    model: str
    tokens: int
    cost_usd: float
    latency_s: float
    ok: bool
    cost_known: bool
    error: str = ""


class RunBudget:
    def __init__(self, cap_usd: float, cap_s: float, *, clock: Callable[[], float] = time.monotonic):
        if not cap_usd or cap_usd <= 0 or not cap_s or cap_s <= 0:
            raise ValueError(f"RunBudget needs positive caps, got cap_usd={cap_usd!r}, cap_s={cap_s!r}")
        self.cap_usd, self.cap_s, self._clock = float(cap_usd), float(cap_s), clock
        self._t0: float | None = None
        self.obs: list[CallObs] = []
        self._live: dict[str, Callable[[], float]] = {}
        self.tripped: str = ""
        self._last: CallObs | None = None

    # -- clock ---------------------------------------------------------------------------------
    def start(self) -> None:
        self._t0 = self._clock()

    def elapsed_s(self) -> float:
        if self._t0 is None:
            raise RuntimeError("RunBudget.start() was never called; elapsed time is undefined.")
        return self._clock() - self._t0

    def remaining_s(self) -> float:
        return max(0.0, self.cap_s - self.elapsed_s())

    # -- spend ---------------------------------------------------------------------------------
    def record(self, obs: CallObs) -> None:
        if not isinstance(obs, CallObs):
            raise TypeError(f"RunBudget.record expects CallObs, got {type(obs).__name__}")
        self.obs.append(obs)
        self._last = obs

    def attach_live(self, name: str, spent_fn: Callable[[], float]) -> None:
        """Count a running total (e.g. a CostLedger.total_usd) while a stage is still running.
        Detach it before record()-ing the same spend as CallObs, or it is counted twice."""
        if name in self._live:
            raise ValueError(f"live meter {name!r} is already attached")
        self._live[name] = spent_fn

    def detach_live(self, name: str) -> None:
        if name not in self._live:
            raise KeyError(f"live meter {name!r} is not attached (attached: {sorted(self._live)})")
        del self._live[name]

    def spent_usd(self) -> float:
        return sum(o.cost_usd or 0.0 for o in self.obs) + sum(f() or 0.0 for f in self._live.values())

    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd())

    # -- checks --------------------------------------------------------------------------------
    def _by_stage_activity(self) -> dict:
        out: dict[str, float] = {}
        for o in self.obs:
            k = f"{o.stage}/{o.activity}"
            out[k] = round(out.get(k, 0.0) + (o.cost_usd or 0.0), 6)
        for n, f in self._live.items():
            out[f"live:{n}"] = round(f() or 0.0, 6)
        return out

    def _trip(self, kind: str, stage: str, activity: str) -> None:
        last = f"{self._last.stage}/{self._last.activity} ({self._last.model})" if self._last else "none"
        self.tripped = (
            f"run budget exceeded ({kind}) at stage={stage} activity={activity}: "
            f"cap=${self.cap_usd:.4f}/{self.cap_s:.0f}s, spent=${self.spent_usd():.6f}, "
            f"elapsed={self.elapsed_s():.1f}s, spent by stage/activity={self._by_stage_activity()}, "
            f"last call={last}. No new call was started.")
        raise RunBudgetExceeded(self.tripped)

    def raise_if_tripped(self) -> None:
        if self.tripped:
            raise RunBudgetExceeded(self.tripped)

    def check(self, stage: str, activity: str) -> None:
        """Raise RunBudgetExceeded if over a cap. Sticky. Call it OUTSIDE any code that swallows
        exceptions (see module docstring: gate_a)."""
        self.raise_if_tripped()
        if self.spent_usd() >= self.cap_usd:
            self._trip("cost", stage, activity)
        if self.elapsed_s() >= self.cap_s:
            self._trip("time", stage, activity)

    def observer_for(self, stage: str, activity: str):
        """(before, after) for pipeline_llm.observe_calls: before() checks the budget, after() records."""
        def before() -> None:
            self.check(stage, activity)

        def after(result, exc, latency_s: float) -> None:
            if exc is not None:
                # A failed attempt may still have spent tokens we cannot see: cost unknown (RD-2).
                self.record(CallObs(stage, activity, "unknown", 0, 0.0, latency_s, False, False,
                                    f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]))
                return
            tokens, cost = int(result.tokens or 0), float(result.cost_usd or 0.0)
            # tokens with zero cost means no pricing record: unknown, not free.
            known = not (tokens > 0 and cost == 0.0)
            self.record(CallObs(stage, activity, result.model, tokens, cost, latency_s, True, known,
                                "" if known else "tokens used but cost 0.0 (no pricing record?)"))
        return before, after


# ---------------------------------------------------------------------------- adapters from ledgers
def obs_from_m2_ledger(ledger) -> list[CallObs]:
    """M2 CostLedger -> CallObs. A failed LLM (Gate A) entry has cost 0 but unknown real cost, so it is
    cost_known=False. Failed search/fetch attempts keep the ledger's own cost_known (RD-2)."""
    out = []
    for e in ledger.entries:
        units = e.units or {}
        tokens = int(units.get("prompt_tokens", 0) or 0) + int(units.get("completion_tokens", 0) or 0)
        known = bool(e.cost_known) and not (e.kind.startswith("llm") and not e.ok)
        out.append(CallObs("m2", e.kind, e.provider, tokens, float(e.cost_usd or 0.0), 0.0,
                           bool(e.ok), known, (e.detail or "")[:MAX_ERROR_CHARS]))
    return out


def obs_from_m3_log(log) -> list[CallObs]:
    """M3 CallLog -> CallObs. ok=False or an unpriced-model marker means cost unknown (RD-2)."""
    out = []
    for r in log.records:
        known = bool(r.ok) and M3_UNPRICED_MARKER not in (r.error or "")
        out.append(CallObs("m3", r.purpose, r.model, int(r.prompt_tokens + r.completion_tokens),
                           float(r.cost_usd or 0.0), float(r.latency_s or 0.0), bool(r.ok), known,
                           (r.error or "")[:MAX_ERROR_CHARS]))
    return out


# ---------------------------------------------------------------------------- usage assembly
def engineered_usage(obs: list[CallObs], *, required: tuple[str, ...] = REQUIRED_STAGES,
                     wall_clock_s: float) -> dict:
    """Totals plus a computed `complete` flag. `complete` is False (never defaulted True) when a
    required stage has no meter or any call has unknown cost; reasons are listed."""
    if wall_clock_s is None or wall_clock_s < 0:
        raise ValueError(f"wall_clock_s must be a non-negative number, got {wall_clock_s!r}")
    seen = {o.stage for o in obs}
    reasons = [f"stage {s} has no meter" for s in required if s not in seen]
    reasons += [f"{o.stage}/{o.activity}: cost unknown ({o.error[:80] or 'no detail'})"
                for o in obs if not o.cost_known]
    by_stage: dict[str, dict] = {}
    for o in obs:
        d = by_stage.setdefault(o.stage, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
        d["calls"] += 1
        d["tokens"] += o.tokens
        d["cost_usd"] = round(d["cost_usd"] + (o.cost_usd or 0.0), 6)
    return {"api_calls": sum(1 for o in obs if not (o.stage == "m2" and o.activity == M2_NON_BILLABLE_KIND)),
            "tokens": sum(o.tokens for o in obs),
            "cost_usd": round(sum(o.cost_usd or 0.0 for o in obs), 6),
            "latency_s": round(wall_clock_s, 3),
            "latency_basis": "wall_clock_m2_start_to_m5_end",
            "complete": not reasons, "incomplete_reasons": reasons, "by_stage": by_stage}


def attach_usage(engineered: dict, usage: dict) -> dict:
    """Set engineered["usage"] AFTER build_engineered (which drops usage.complete, measured hole) and
    refuse any usage without an explicit boolean `complete`, so incomplete usage can never be scored
    as complete by omission (score_cost_latency scores 1 only when complete is exactly False)."""
    if not isinstance(usage, dict):
        raise TypeError(f"usage must be a dict from engineered_usage(), got {type(usage).__name__}")
    if not isinstance(usage.get("complete"), bool):
        raise ValueError("usage has no explicit boolean 'complete' key (would be scored as complete "
                         f"by omission); keys present: {sorted(usage)}")
    missing = [k for k in ("cost_usd", "latency_s") if usage.get(k) is None]
    if missing:
        raise ValueError(f"usage lacks {missing}; Cost/Latency cannot be computed from it")
    engineered["usage"] = dict(usage)
    return engineered
