"""LLM judge for M6 scoring, with hard resource caps.

Rules enforced here (R1 requirements, not preferences):
  * The judge MUST be a different model family from the model that produced
    the answers (engineered + baseline are OpenAI gpt-4.1-nano). A same-family
    judge is refused, not silently accepted.
  * Every judge has a call / token / cost cap. Hitting a cap raises
    BudgetExceeded with a verbose report (bounded-loops rule); nothing is
    retried past the cap.
  * Judge output must be JSON. Unparseable output is retried at most
    `max_parse_retries` times, then fails verbosely.

Selection (goal #2, R1 decision D2): the judge is chosen at run time from a pool of live candidates
(vera.judge_select), never from a pinned default. The model-to-family map and the pool settings live in
config/vera_eval_run.json (vera/eval_config.py). The legacy env selector below stays for A/B work:
  VERA_JUDGE_MODEL = provider selector (groq | openai), default groq
  groq       GROQ_API_KEY     model VERA_JUDGE_GROQ_MODEL     (REQUIRED: no default; mixtral-8x7b-32768 was
                              shut down by Groq on 2025-03-20)
  openai     OPENAI_API_KEY   model VERA_JUDGE_OPENAI_MODEL   (REQUIRED). Same family as the generator, so
             REFUSED unless VERA_JUDGE_ALLOW_SAME_FAMILY=1 (then disclosed). Off by default.
The family of a model comes from the run config's model_families map; an unmapped model is refused (fail
closed) unless VERA_JUDGE_FAMILY_OVERRIDE=<family> is set, which is disclosed (family_source="override").
The override is refused outright for the openai provider, and for any model whose id contains a refuse_openai
pattern (gpt / openai): an override can never contradict the model id.
Prices per 1M tokens: VERA_JUDGE_PRICE_IN / VERA_JUDGE_PRICE_OUT override the provider defaults, which are
ASSUMED list prices; verify against the provider price page.
Mistral is not a supported provider (R1 2026-10-02: it never worked; its key is not held).
Only GROQ_API_KEY and OPENAI_API_KEY are ever used.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from datetime import datetime, timezone

from vera.cost_ledger import CostLedger
from vera.eval_config import EvalConfig, family_for, load_eval_config

# Why: the judge's max_tokens is a fixed call setting (not a threshold); named once, imported by
# vera.judge_select and recorded in the run fingerprint.
JUDGE_MAX_TOKENS = 4000


class BudgetExceeded(RuntimeError):
    """A loop hit its time/token/cost/step cap. Message is the verbose report."""


class JudgeConfigError(RuntimeError):
    pass


def generation_family(config: EvalConfig | None = None) -> str:
    """Family of the generator model, derived from the run config (not a constant): the judge must differ."""
    cfg = config or load_eval_config()
    return family_for(cfg.generator_model, cfg).family


@dataclass
class JudgeLimits:
    max_calls: int = 60
    max_tokens: int = 400_000
    max_cost_usd: float = 0.25
    max_parse_retries: int = 1


@dataclass
class JudgeUsage:
    calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    log: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"api_calls": self.calls, "tokens": self.tokens,
                "cost_usd": round(self.cost_usd, 6), "latency_s": round(self.latency_s, 3)}


class Judge(Protocol):
    family: str
    model: str
    usage: JudgeUsage

    def judge_json(self, purpose: str, prompt: str) -> dict: ...


def parse_json_loose(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ``` fences / prose."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


class BaseJudge:
    """Shared cap + retry logic. Subclasses implement `_call`."""

    family = "unknown"
    model = "unknown"

    provider = "unknown"  # selector name (groq/openai), recorded in reports
    same_family_override = False
    model_source = "class"  # how the model was chosen: explicit / env / class
    family_source = "class"  # how the family was decided: map / override / pattern / class
    price_source = "assumed_default"

    def __init__(self, limits: JudgeLimits | None = None, ledger: CostLedger | None = None, *,
                 generator_family: str | None = None, config: EvalConfig | None = None):
        self.limits = limits or JudgeLimits()
        self.usage = JudgeUsage()
        self.ledger = ledger  # optional; every call is recorded for A/B measurement
        gen = generator_family or generation_family(config)
        if self.family == gen and os.getenv("VERA_JUDGE_ALLOW_SAME_FAMILY") == "1":
            self.same_family_override = True
        elif self.family == gen:
            raise JudgeConfigError(
                f"Judge family {self.family!r} equals the generation family {gen!r}; "
                "R1 requires a different model family for the judge.")

    def _call(self, prompt: str) -> tuple[str, int, int]:  # text, in_tok, out_tok
        raise NotImplementedError

    def _price(self, in_tok: int, out_tok: int) -> float:
        raise NotImplementedError

    def _check_caps(self, purpose: str) -> None:
        u, l = self.usage, self.limits
        if u.calls >= l.max_calls or u.tokens >= l.max_tokens or u.cost_usd >= l.max_cost_usd:
            raise BudgetExceeded(
                f"Judge cap reached before call for {purpose!r}: calls {u.calls}/{l.max_calls}, "
                f"tokens {u.tokens}/{l.max_tokens}, cost ${u.cost_usd:.4f}/${l.max_cost_usd:.2f}. "
                "Goal (scoring all dimensions) is incomplete. Needed to continue: raise the cap "
                "(R1 decision) or narrow the rubric inputs. Calls so far: "
                + ", ".join(e["purpose"] for e in u.log[-10:]))

    def judge_json(self, purpose: str, prompt: str) -> dict:
        last_err = None
        for _attempt in range(self.limits.max_parse_retries + 1):
            self._check_caps(purpose)
            t0 = time.monotonic()
            text, tin, tout = self._call(prompt)
            dt = time.monotonic() - t0
            cost = self._price(tin, tout)
            self.usage.calls += 1
            self.usage.tokens += tin + tout
            self.usage.cost_usd += cost
            self.usage.latency_s += dt
            self.usage.log.append({"purpose": purpose, "model": self.model, "tokens_in": tin,
                                   "tokens_out": tout, "tokens": tin + tout, "cost_usd": cost,
                                   "latency_s": round(dt, 3)})
            if self.ledger is not None:
                self.ledger.record(
                    kind="llm_judge", provider=f"{self.provider}:{self.model}", cost_usd=cost,
                    units={"purpose": purpose, "tokens_in": tin, "tokens_out": tout,
                           "latency_s": round(dt, 3)})
            try:
                return parse_json_loose(text)
            except json.JSONDecodeError as e:
                last_err = e
        raise RuntimeError(
            f"Judge returned unparseable JSON for {purpose!r} after "
            f"{self.limits.max_parse_retries + 1} attempts: {last_err}")


# Provider facts (invariants, not eval parameters): key env var NAME, base URL, ASSUMED list $/1M in/out.
# Prices are unverified; override with VERA_JUDGE_PRICE_IN/OUT (recorded as price_source="env").
# Chat URL = base + "/chat/completions"; listing URL = base + "/models". No Mistral entry (R1 2026-10-02).
@dataclass(frozen=True)
class ProviderFacts:
    key_env: str
    base_url: str
    price_in: float
    price_out: float


_PROVIDERS = {
    "groq": ProviderFacts("GROQ_API_KEY", "https://api.groq.com/openai/v1", 0.24, 0.24),
    "openai": ProviderFacts("OPENAI_API_KEY", "https://api.openai.com/v1", 0.1, 0.4),
}


class OpenAICompatJudge(BaseJudge):
    """Chat-completions judge for Groq / OpenAI (same wire format, plain HTTPS).

    No code default model (D2): the model is explicit (the pool selection passes it) or comes from
    VERA_JUDGE_<PROVIDER>_MODEL; otherwise construction fails verbosely. The family comes from the run
    config's model map (fail closed) unless the caller passes a family (pool selection, source "pattern")."""

    def __init__(self, provider: str, limits: JudgeLimits | None = None,
                 api_key: str | None = None, ledger: CostLedger | None = None, *,
                 model: str | None = None, family: str | None = None, family_source: str | None = None,
                 config: EvalConfig | None = None):
        if provider not in _PROVIDERS:
            raise JudgeConfigError(f"Unknown judge provider {provider!r} (known: {sorted(_PROVIDERS)}).")
        cfg = config or load_eval_config()
        facts = _PROVIDERS[provider]
        self.provider = provider
        env_model = os.getenv(f"VERA_JUDGE_{provider.upper()}_MODEL")
        if model:
            self.model, self.model_source = model, "explicit"
        elif env_model:
            self.model, self.model_source = env_model, "env"
        else:
            raise JudgeConfigError(
                f"No judge model configured for provider {provider!r}: there is no code or config default "
                "(Groq shut down mixtral-8x7b-32768 on 2025-03-20; R1 decision D2 replaced defaults with live "
                f"pool selection, see vera.judge_select). Set VERA_JUDGE_{provider.upper()}_MODEL to a model the "
                "provider lists and that model_families maps, or use the pool selection.")
        override = os.getenv("VERA_JUDGE_FAMILY_OVERRIDE")
        if provider == "openai" and override and override.strip():
            raise JudgeConfigError(
                "VERA_JUDGE_FAMILY_OVERRIDE is refused for the openai provider: that provider always means "
                f"the openai family (the generator's family); got override {override.strip()!r}. Unset it.")
        if family:
            self.family, self.family_source = family, family_source or "explicit"
        else:
            try:
                res = family_for(self.model, cfg, override=override)
            except Exception as e:  # EvalConfigError -> JudgeConfigError, same verbose text
                raise JudgeConfigError(str(e)) from None
            self.family, self.family_source = res.family, res.source
        mid = self.model.lower()
        hit = next((p for p in cfg.judge.refuse_openai_patterns if p in mid), None)
        if self.family != "openai" and hit:
            raise JudgeConfigError(
                f"Family {self.family!r} for {self.model!r} contradicts the id, which contains {hit!r} "
                "(OpenAI family); fix the map, the override or the pool pattern. Refused.")
        if provider == "openai" and self.family != "openai":
            raise JudgeConfigError(
                f"Provider 'openai' with model {self.model!r} resolved to family {self.family!r}; the openai "
                "provider always means the openai family. Refused.")
        super().__init__(limits, ledger, generator_family=generation_family(cfg))
        self.base_url = facts.base_url
        self.url = facts.base_url + "/chat/completions"
        self.api_key = api_key or os.getenv(facts.key_env)
        if not self.api_key:
            raise JudgeConfigError(
                f"{facts.key_env} is not set, so judge provider {provider!r} is unavailable. "
                "Blocker owner: R4b (credential custody). Set the key or choose another "
                "VERA_JUDGE_MODEL; this run will not fall back to a same-family judge.")
        env_prices = os.getenv("VERA_JUDGE_PRICE_IN") is not None or os.getenv("VERA_JUDGE_PRICE_OUT") is not None
        self.price_in = float(os.getenv("VERA_JUDGE_PRICE_IN", facts.price_in))
        self.price_out = float(os.getenv("VERA_JUDGE_PRICE_OUT", facts.price_out))
        self.price_source = "env" if env_prices else "assumed_default"

    def _call(self, prompt: str):
        import requests
        r = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"},
            json={"model": self.model, "max_tokens": JUDGE_MAX_TOKENS, "temperature": 0,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=120)
        if not r.ok:  # verbose, no retry (bounded-loops rule)
            raise RuntimeError(f"Judge call to {self.provider} model {self.model!r} failed: "
                               f"HTTP {r.status_code}: {r.text[:300]}")
        d = r.json()
        u = d.get("usage", {})
        return (d["choices"][0]["message"]["content"],
                u.get("prompt_tokens", 0), u.get("completion_tokens", 0))

    def _price(self, tin, tout):
        return tin / 1e6 * self.price_in + tout / 1e6 * self.price_out


# Why: one listing call needs a bound; same value the run config uses for pool selection.
PREFLIGHT_TIMEOUT_S = 15


def preflight_judge(judge, *, http_get=None, timeout_s: float = PREFLIGHT_TIMEOUT_S) -> dict:
    """One listing call, no tokens, no retry. Raises JudgeConfigError BEFORE any baseline spend.
    The API key is sent as a header and never appears in an error message."""
    if http_get is None:
        import requests
        http_get = requests.get
    url = judge.base_url + "/models"
    try:
        r = http_get(url, headers={"Authorization": f"Bearer {judge.api_key}"}, timeout=timeout_s)
    except Exception as e:  # noqa: BLE001
        raise JudgeConfigError(f"Preflight: cannot list {judge.provider} models ({type(e).__name__}); "
                               "nothing spent; run stopped.") from None
    if not r.ok:
        raise JudgeConfigError(f"Preflight: {judge.provider} /models HTTP {r.status_code}: {str(r.text)[:300]}")
    try:
        rows = r.json().get("data", [])
    except Exception as e:  # noqa: BLE001
        raise JudgeConfigError(f"Preflight: {judge.provider} /models returned unreadable JSON "
                               f"({type(e).__name__}); nothing spent.") from None
    ids = sorted(m["id"] for m in rows if isinstance(m, dict) and isinstance(m.get("id"), str))
    if judge.model not in ids:
        raise JudgeConfigError(
            f"Preflight: {judge.provider} does not list model {judge.model!r} ({len(ids)} models listed). "
            "Nothing was spent. Choose a listed model; see console.groq.com/docs/deprecations.")
    return {"status": "ok", "provider": judge.provider, "model": judge.model, "listed_models": len(ids),
            "checked_at": datetime.now(timezone.utc).isoformat()}


def build_default_judge(limits: JudgeLimits | None = None, ledger: CostLedger | None = None, *,
                        config: EvalConfig | None = None) -> Judge:
    """Legacy env-selected judge (VERA_JUDGE_MODEL = groq | openai; default groq). The model
    must be supplied through VERA_JUDGE_<PROVIDER>_MODEL (no default). The normal MVP path is the live
    pool selection in vera.judge_select."""
    choice = os.getenv("VERA_JUDGE_MODEL", "groq").strip().lower() or "groq"
    if choice in _PROVIDERS:
        return OpenAICompatJudge(choice, limits, ledger=ledger, config=config)
    raise JudgeConfigError(
        f"VERA_JUDGE_MODEL={choice!r} is not a known judge provider "
        f"({', '.join(_PROVIDERS)}).")
