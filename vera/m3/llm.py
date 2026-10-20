"""Thin, cost-logging JSON-mode LLM wrapper shared by M3 stages.

Every call is appended to a CallLog (purpose, model, tokens, cost) so M6 can
report cost per stage. Default model is the repo's selected model
(config/model-selection.json; gpt-4.1-nano = lowest cost). No capability gap
requiring a pricier model has been demonstrated for span extraction or
rubric-guided appraisal, so none is used.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol

from vera.pricing.config import latest_pricing_for, load_model_pricing, load_model_selection

logger = logging.getLogger("vera.m3")

DEFAULT_MODEL = "gpt-4.1-nano"


class LLMError(RuntimeError):
    """Raised (verbosely) when an LLM call fails or returns unusable output."""


@dataclass
class CallRecord:
    purpose: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    ok: bool = True
    error: str = ""


@dataclass
class CallLog:
    records: list[CallRecord] = field(default_factory=list)

    def add(self, rec: CallRecord) -> None:
        self.records.append(rec)
        logger.info("m3 llm call %s", asdict(rec))

    @property
    def total_cost_usd(self) -> float:
        return round(sum(r.cost_usd for r in self.records), 6)

    @property
    def total_tokens(self) -> int:
        return sum(r.prompt_tokens + r.completion_tokens for r in self.records)

    def summary(self) -> dict:
        by_purpose: dict[str, dict] = {}
        for r in self.records:
            d = by_purpose.setdefault(r.purpose, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
            d["calls"] += 1
            d["tokens"] += r.prompt_tokens + r.completion_tokens
            d["cost_usd"] = round(d["cost_usd"] + r.cost_usd, 6)
        return {
            "calls": len(self.records),
            "failed_calls": sum(1 for r in self.records if not r.ok),
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "by_purpose": by_purpose,
        }


class LLMClient(Protocol):
    log: CallLog

    def complete_json(self, system: str, user: str, purpose: str) -> dict: ...


class OpenAIJSONClient:
    """Real client. OpenAI is imported lazily so tests never need it."""

    def __init__(self, model: str | None = None, log: CallLog | None = None, timeout_s: float = 60.0):
        sel = load_model_selection()
        self.model = model or (sel.selected_model if sel else DEFAULT_MODEL)
        self.log = log if log is not None else CallLog()
        self.timeout_s = timeout_s
        self._client = None
        self._pricing = latest_pricing_for(self.model, load_model_pricing())

    def _get(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(timeout=self.timeout_s, max_retries=1)
        return self._client

    def _cost(self, pt: int, ct: int) -> float:
        if self._pricing is None:
            return 0.0  # unknown price is logged as 0 but flagged in the record error
        return round(pt / 1e6 * self._pricing.input + ct / 1e6 * self._pricing.output, 6)

    def complete_json(self, system: str, user: str, purpose: str) -> dict:
        t0 = time.monotonic()
        try:
            resp = self._get().chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            u = resp.usage
            raw = resp.choices[0].message.content or ""
            self.log.add(CallRecord(purpose, self.model, u.prompt_tokens, u.completion_tokens,
                                    self._cost(u.prompt_tokens, u.completion_tokens),
                                    round(time.monotonic() - t0, 3),
                                    error="" if self._pricing else "no pricing record; cost unknown"))
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"{purpose}: model returned non-JSON output: {exc}") from exc
        except Exception as exc:  # provider failure: log and re-raise verbosely
            self.log.add(CallRecord(purpose, self.model, 0, 0, 0.0, round(time.monotonic() - t0, 3),
                                    ok=False, error=repr(exc)))
            raise LLMError(f"{purpose}: LLM call failed: {exc!r}") from exc
