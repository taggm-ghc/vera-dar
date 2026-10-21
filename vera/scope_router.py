"""Scope router for VERA-DAR's answer path (p3m3 item #72 W1, revision R72-a).

Layer 1 is a deterministic, zero-cost pre-check driven by config/scope_router.json: injection markers and deny
categories DECLINE, an AI term plus a software-development term (or a named coding tool) ACCEPTs, anything else is
UNSURE. Layer 2 (config `layer2_mode`) resolves UNSURE: "llm" asks the selected model for a JSON verdict
{in_scope, reason}; "none" applies `none_mode_unsure_action`. "embedding" is noted in config but not implemented.

Declines always return the fixed `decline_template`; no model-generated text is ever shown for a decline. The
classifier prompt holds no secrets, keys, hosts or internal names (OWASP LLM07; enforced by a test). A malformed
classifier reply fails closed (decline). OpenAI SDK errors propagate to the routing layer like the answer call's.
The default OpenAI client (default_chat with client=None) raises OpenAIFallbackDisabled in the API process unless
`openai_fallback.enabled` is true (ask_service latch); chain entries supply their own client.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "scope_router.json"
LAYER2_MODES = ("llm", "none", "embedding")

ACCEPT, DECLINE, UNSURE = "accept", "decline", "unsure"


class ScopeConfigError(ValueError):
    """config/scope_router.json is missing, malformed or selects an unimplemented mode."""


@dataclass(frozen=True)
class ScopeDecision:
    accepted: bool
    layer: str  # "layer1" | "layer2_llm" | "layer2_none"
    reason: str
    tokens_used: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0


@dataclass(frozen=True)
class RouterConfig:
    raw: dict
    layer2_mode: str
    none_mode_unsure_action: str
    decline_template: str
    injection: tuple[re.Pattern, ...]
    deny: dict[str, tuple[re.Pattern, ...]]
    ai_terms: tuple[re.Pattern, ...]
    dev_terms: tuple[re.Pattern, ...]
    tool_names: tuple[re.Pattern, ...]
    layer2: dict


def _compile(patterns: list[str], where: str) -> tuple[re.Pattern, ...]:
    try:
        return tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    except re.error as exc:
        raise ScopeConfigError(f"scope_router.json: bad regex in {where}: {exc}") from exc


def parse_config(raw: dict, layer2_mode: str | None = None) -> RouterConfig:
    try:
        mode = layer2_mode or raw["layer2_mode"]
        if mode not in LAYER2_MODES:
            raise ScopeConfigError(f"scope_router.json: layer2_mode {mode!r} not in {LAYER2_MODES}")
        if mode == "embedding":
            raise ScopeConfigError("scope_router.json: layer2_mode 'embedding' is noted but not implemented")
        unsure = raw["none_mode_unsure_action"]
        if unsure not in (ACCEPT, DECLINE):
            raise ScopeConfigError(f"scope_router.json: none_mode_unsure_action {unsure!r} must be accept|decline")
        l1 = raw["layer1"]
        return RouterConfig(
            raw=raw,
            layer2_mode=mode,
            none_mode_unsure_action=unsure,
            decline_template=raw["decline_template"],
            injection=_compile(l1["injection_patterns"], "injection_patterns"),
            deny={k: _compile(v, f"deny_categories.{k}") for k, v in l1["deny_categories"].items()},
            ai_terms=_compile(l1["ai_terms"], "ai_terms"),
            dev_terms=_compile(l1["dev_terms"], "dev_terms"),
            tool_names=_compile(l1["tool_names"], "tool_names"),
            layer2=raw["layer2"],
        )
    except KeyError as exc:
        raise ScopeConfigError(f"scope_router.json: missing key {exc}") from exc


@lru_cache(maxsize=1)
def load_config(path: str = str(CONFIG_PATH)) -> RouterConfig:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScopeConfigError(f"cannot load scope router config {Path(path).name}: {exc}") from exc
    return parse_config(raw)


def normalise(question: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", question or "")).strip().lower()


def _any(patterns: tuple[re.Pattern, ...], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def layer1(question: str, cfg: RouterConfig) -> tuple[str, str]:
    """Return (ACCEPT|DECLINE|UNSURE, reason). Zero cost, deterministic."""
    q = normalise(question)
    if not q:
        return DECLINE, "empty"
    if _any(cfg.injection, q):
        return DECLINE, "deny:injection"
    for name, pats in cfg.deny.items():
        if _any(pats, q):
            return DECLINE, f"deny:{name}"
    if _any(cfg.tool_names, q):
        return ACCEPT, "tool_name"
    if _any(cfg.ai_terms, q) and _any(cfg.dev_terms, q):
        return ACCEPT, "ai_and_dev_terms"
    return UNSURE, "no_rule"


# chat(messages, layer2_cfg) -> (content, prompt_tokens, completion_tokens)
ChatFn = Callable[[list[dict], str, dict], tuple[str, int, int]]


def classifier_messages(question: str, cfg: RouterConfig) -> list[dict]:
    l2 = cfg.layer2
    q = (question or "")[: int(l2["max_question_chars"])]
    return [{"role": "system", "content": l2["system_prompt"]},
            {"role": "user", "content": l2["user_template"].format(question=q)}]


def default_chat(messages: list[dict], model: str, l2: dict, client=None,
                 extra_body: dict | None = None) -> tuple[str, int, int]:
    from vera.ask_service import _get_client

    c = (client or _get_client()).chat.completions.create(
        model=model, messages=messages, temperature=l2["temperature"], max_tokens=l2["max_tokens"],
        response_format={"type": "json_object"}, timeout=l2["timeout_s"],
        **({"extra_body": extra_body} if extra_body else {}),
    )
    return c.choices[0].message.content or "", c.usage.prompt_tokens, c.usage.completion_tokens


def parse_verdict(content: str) -> tuple[bool | None, str]:
    """Return (in_scope, reason); in_scope None when the reply is malformed (caller fails closed)."""
    try:
        obj = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None, "classifier_malformed"
    if not isinstance(obj, dict) or not isinstance(obj.get("in_scope"), bool):
        return None, "classifier_malformed"
    return obj["in_scope"], str(obj.get("reason", ""))[:200]


def route(question: str, model: str, pricing=None, cfg: RouterConfig | None = None,
          chat: ChatFn | None = None) -> ScopeDecision:
    cfg = cfg or load_config()
    verdict, reason = layer1(question, cfg)
    if verdict != UNSURE:
        return ScopeDecision(verdict == ACCEPT, "layer1", reason)
    if cfg.layer2_mode == "none":
        return ScopeDecision(cfg.none_mode_unsure_action == ACCEPT, "layer2_none", "unsure:" + cfg.none_mode_unsure_action)
    content, pt, ct = (chat or default_chat)(classifier_messages(question, cfg), model, cfg.layer2)
    cost = 0.0
    if pricing is not None:
        cost = round(pt / 1_000_000 * pricing.input + ct / 1_000_000 * pricing.output, 8)
    in_scope, why = parse_verdict(content)
    if in_scope is None:
        return ScopeDecision(False, "layer2_llm", why, pt + ct, cost, 1)
    return ScopeDecision(in_scope, "layer2_llm", ("llm_in:" if in_scope else "llm_out:") + why, pt + ct, cost, 1)


def decline_text(cfg: RouterConfig | None = None) -> str:
    return (cfg or load_config()).decline_template
