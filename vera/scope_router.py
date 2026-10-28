"""Scope router for VERA-DAR's answer path (p3m3 item #72 W1, revision R72-a).

Layer 1 is a deterministic, zero-cost pre-check driven by config/scope_router.json: injection markers and deny
categories DECLINE, an AI term plus a software-development term (or a named coding tool) ACCEPTs, anything else is
UNSURE. Layer 2 (config `layer2_mode`) resolves UNSURE: "llm" asks the selected model for a JSON verdict
{in_scope, reason}; "none" applies `none_mode_unsure_action`. "embedding" is noted in config but not implemented.

Declines always return the fixed `decline_template`; no model-generated text is ever shown for a decline. Exception (FIX1-1021, config
`offtopic_reading_list`, default off in code): a well-formed layer-2 out-of-scope verdict may instead yield a fixed notice plus a
scholarly-only reading list (see ask_service.answer_in_scope); the notice is fixed text, never model-generated. The
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

from vera.strict_json import base_norm as _base, loads_or_none, norm_key as _norm

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
    category: str = ""  # DAR-1021: layer-2 verdict category ("" when absent or not in LAYER2_CATEGORIES)


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
        mr = raw["layer2"].get("malformed_retry", 1)
        if isinstance(mr, bool) or mr not in (0, 1):  # PATHA-1028 Q2: a bound, clamped to 0/1; fail verbose
            raise ScopeConfigError(f"scope_router.json: layer2.malformed_retry {mr!r} must be 0 or 1")
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


_LETTER_RUN_RE = re.compile(r"(?<![^\W\d_])[^\W\d_](?: +[^\W\d_])+(?![^\W\d_])")


def _join_letter_run(m: re.Match) -> str:
    run = m.group(0)
    letters = run.replace(" ", "")
    if len(letters) < 4 or not re.search(r"(?<! ) (?! )", run):  # need 4+ letters and a single-space separator
        return run
    # single-spaced letters join into a word; gaps of 2+ spaces are word breaks
    return " ".join(g.replace(" ", "") for g in re.split(r" {2,}", run))


def collapse_letter_spaced(question: str) -> str:
    """SCOPE2-1022: collapse letter-spaced text (4+ single letters separated by single spaces; "H O W  M A N Y" ->
    "HOW MANY"). Word breaks are gaps of 2+ spaces. For layer-1 matching only; callers never pass the result on."""
    return _LETTER_RUN_RE.sub(_join_letter_run, unicodedata.normalize("NFKC", question or ""))


def _any(patterns: tuple[re.Pattern, ...], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def layer1(question: str, cfg: RouterConfig) -> tuple[str, str]:
    """Return (ACCEPT|DECLINE|UNSURE, reason). Zero cost, deterministic."""
    q = normalise(question)
    if not q:
        return DECLINE, "empty"
    # SCOPE2-1022: also match a letter-spaced-collapsed view (matching only; the question itself is never altered)
    views = [q]
    qc = normalise(collapse_letter_spaced(question))
    if qc != q:
        views.append(qc)
    for v in views:
        if _any(cfg.injection, v):
            return DECLINE, "deny:injection"
        for name, pats in cfg.deny.items():
            if _any(pats, v):
                return DECLINE, f"deny:{name}"
    if _any(cfg.tool_names, q):
        return ACCEPT, "tool_name"
    if _any(cfg.ai_terms, q) and _any(cfg.dev_terms, q):
        return ACCEPT, "ai_and_dev_terms"
    return UNSURE, "no_rule"


_MARKER_RE = re.compile(r"<<+\s*question|question\s*>>+", re.IGNORECASE)  # SEC C1: 2+ angle brackets
_VERDICT_TOKENS_RE = re.compile(r"\b(categor(y|ies)|in[\s_-]*scope|off[\s_-]*topic)\b", re.IGNORECASE)  # SEC C1


def has_marker(question: str) -> bool:
    """DAR-1021 SEC H1: the question contains the classifier's own data markers (an attempt to close the data block)."""
    return bool(_MARKER_RE.search(unicodedata.normalize("NFKC", str(question or ""))))


def _neutralise_for_classifier(q: str) -> str:
    """DAR-1021 SEC H1: the classifier sees no braces or verdict-field words from the question, so the question cannot
    forge a JSON verdict or name its own category. Only the classifier input changes; the question is not altered."""
    q = unicodedata.normalize("NFKC", q).replace("{", "(").replace("}", ")")
    return _VERDICT_TOKENS_RE.sub("[x]", q)


# chat(messages, layer2_cfg) -> (content, prompt_tokens, completion_tokens)
ChatFn = Callable[[list[dict], str, dict], tuple[str, int, int]]


def classifier_messages(question: str, cfg: RouterConfig) -> list[dict]:
    l2 = cfg.layer2
    q = _neutralise_for_classifier((question or "")[: int(l2["max_question_chars"])])
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


LAYER2_CATEGORIES = ("in_scope", "off_topic", "advice", "harmful", "programming_task", "rule_change")
REFUSE_CATEGORIES = frozenset({"advice", "harmful", "programming_task", "rule_change"})  # SCOPE-1022: never answered, listed or acquired

# HOTFIX-DUPKEY-1027: strict classifier-reply parsing. Code constants (a security floor), not config.
# STRICTJSON-1027 C5: the normaliser and strict loader live in vera/strict_json.py (moved verbatim); behaviour unchanged.
MAX_REPLY_CHARS = 16384  # SEC-H-6: layer-2 max_tokens is 512 (well under 4 KB of JSON); re-calibrate if it exceeds ~2048


def _norm_ws(s: str) -> str:
    """Text scans: whitespace runs kept as one space, so merged words such as "categorical advice" do not match."""
    return re.sub(r"\s+", " ", _base(s)).casefold().strip()


def _strict_loads(content):
    """The decoded reply, or None when it is malformed (non-str, oversize, bad JSON, too deep, duplicate or non-ASCII
    key); the caller fails closed. A plain json.loads keeps the LAST of duplicate keys (R2-SCOPE-S-2). NaN and 1e999
    stay accepted here (allow_constants=True), exactly as in the hotfix."""
    return loads_or_none(content, allow_constants=True, max_chars=MAX_REPLY_CHARS)


_REFUSE_ORDER = ("harmful", "rule_change", "advice", "programming_task")  # SCOPE v2 precedence
_CATS = r"(harmful|rule[ _]?change|advice|programming[ _]?task)"
_RAW_QUOTED = re.compile(r'\\*"' + _CATS + r'\\*"')  # SEC-H-2/H-4: any run of backslashes before each quote
_CAT_IN_TEXT = re.compile(r"categor(?:y|ies)\W{0,8}" + _CATS)  # SEC-H-2/H-6: bounded, no \w* backtracking
_CAT_SPELLINGS = {"harmful": "harmful", "advice": "advice", "rule_change": "rule_change", "rulechange": "rule_change",
                  "programming_task": "programming_task", "programmingtask": "programming_task"}


def _canon(m: re.Match) -> str:
    return _CAT_SPELLINGS[m.group(1).replace(" ", "_")]


_ASSIGN = re.compile(r"categor(?:y|ies)(?![\s(;,.]*" + _CATS.replace("(", "(?:", 1) + r")[\W_]{1,8}" + _CATS)  # "category <sep> X" anywhere; not ";" "," "." or space/paren


def refusal_token(content, categories=REFUSE_CATEGORIES) -> str:
    """HOTFIX-DUPKEY-1027 v2 (A1-OVERREFUSE-1028): the first refusal category (by _REFUSE_ORDER, restricted to
    `categories`) the reply names, else "". Runs on the strictly decoded reply (duplicate / non-ASCII keys at any depth,
    bad JSON, depth and size already make it malformed). Every key and value EXCEPT the top-level free-text "reason" is
    scanned for a category (equality, "category: X" phrasing, quoted labels, embedded JSON). The reason declines only if
    it (a) equals a category, (b) contains an assignment "category: X"/"categories=X" anywhere (any 1-8 non-word separator characters except a bare space, paren, ; , or .), or (c) contains "{" or "["
    (decoded and walked, quoted labels scanned). Prose such as 'not "advice"' does not decline. Total: never raises;
    a non-str, oversize, malformed or non-object reply, or any internal fault, gives the fail-closed value."""
    fc = next((c for c in _REFUSE_ORDER if c in categories), "harmful")
    try:
        obj = _strict_loads(content)
        if not isinstance(obj, dict):
            return fc
        hits = set()
        stack = []

        def scan_str(s, prose):
            n = _norm(s)
            if prose:  # rule (a): quotes, punctuation, whitespace and case are all ignored
                n = re.sub(r"[\W_]+", "", n)
            if n in _CAT_SPELLINGS:
                hits.add(_CAT_SPELLINGS[n])
            t = _norm_ws(s)
            if prose:
                hits.update(_canon(m) for m in _ASSIGN.finditer(t))  # rule (b): assignment form anywhere in the reason
                if "{" not in s and "[" not in s:
                    return
            else:
                hits.update(_canon(m) for m in _CAT_IN_TEXT.finditer(t))
            hits.update(_canon(m) for m in _RAW_QUOTED.finditer(t))
            if "{" in s or "[" in s:
                try:  # lenient decode: duplicate keys are all kept as (key, value) pairs
                    stack.append(json.loads(s, object_pairs_hook=lambda pairs: pairs))
                except (ValueError, RecursionError):  # not JSON: scan it as text (single quotes, bare keys, ...)
                    hits.update(_canon(m) for m in _CAT_IN_TEXT.finditer(t))

        top_reason = obj.get("reason")
        if isinstance(top_reason, str):
            scan_str(top_reason, prose=True)
        stack.append([(k, v) for k, v in obj.items() if k != "reason" or not isinstance(v, str)])
        while stack:  # iterative walk: no recursion limit on deep replies
            node = stack.pop()
            if isinstance(node, dict):
                stack.extend(node.items())
            elif isinstance(node, (list, tuple)):
                stack.extend(node)
            elif isinstance(node, str):
                scan_str(node, prose=False)
        return next((c for c in _REFUSE_ORDER if c in categories and c in hits), "")
    except Exception:  # noqa: BLE001  SEC-H-7: fail closed
        return fc


def parse_category(content: str) -> str:
    """DAR-1021: the classifier's optional `category` (a closed enum). Anything else, including a missing field, a
    non-string or an unknown value, is "" so every gate keyed on it fails closed."""
    obj = _strict_loads(content)
    c = obj.get("category") if isinstance(obj, dict) else None
    return c if isinstance(c, str) and c in LAYER2_CATEGORIES else ""


def parse_verdict(content: str) -> tuple[bool | None, str]:
    """Return (in_scope, reason); in_scope None when the reply is malformed (caller fails closed)."""
    obj = _strict_loads(content)
    if not isinstance(obj, dict) or not isinstance(obj.get("in_scope"), bool):
        return None, "classifier_malformed"
    c = obj.get("category")
    if isinstance(c, str) and not unicodedata.normalize("NFKC", c).isascii():  # lookalike value, e.g. Cyrillic а
        return None, "classifier_malformed"
    return obj["in_scope"], str(obj.get("reason", ""))[:200]


def route(question: str, model: str, pricing=None, cfg: RouterConfig | None = None,
          chat: ChatFn | None = None) -> ScopeDecision:
    cfg = cfg or load_config()
    verdict, reason = layer1(question, cfg)
    if verdict != UNSURE:
        return ScopeDecision(verdict == ACCEPT, "layer1", reason)
    if has_marker(question):  # DAR-1021 SEC H1: fixed refusal, no classifier call, never a reading list or acquisition
        return ScopeDecision(False, "layer1", "deny:injection_marker")
    if cfg.layer2_mode == "none":
        return ScopeDecision(cfg.none_mode_unsure_action == ACCEPT, "layer2_none", "unsure:" + cfg.none_mode_unsure_action)
    messages = classifier_messages(question, cfg)
    retries = int(cfg.layer2.get("malformed_retry", 1))
    pt = ct = calls = 0
    while True:  # PATHA-1028 Q2: at most 1 + malformed_retry (<= 2) classifier calls
        content, p_, c_ = (chat or default_chat)(messages, model, cfg.layer2)
        pt, ct, calls = pt + p_, ct + c_, calls + 1
        cost = 0.0
        if pricing is not None:
            cost = round(pt / 1_000_000 * pricing.input + ct / 1_000_000 * pricing.output, 8)
        d = _layer2_decision(content, pt, ct, cost, calls)
        if calls > retries or not _nullified(d) or _raw_names_refusal(content):
            if calls > 1:
                import logging
                logging.getLogger("vera").info("scope classifier retry: calls=%d nullified=%s", calls, _nullified(d))
            return d


def _nullified(d: ScopeDecision) -> bool:
    """PATHA-1028 Q2: a verdict with no usable classification (plan s26.1 P5 and P10b): a malformed, oversize, bad-JSON,
    duplicate-key or non-ASCII-category reply, or a well-formed in_scope=false reply whose category is missing or not
    in the closed enum. category_conflict and refusal_token verdicts are safety refusals, not nullified."""
    return d.reason == "classifier_malformed" or (d.reason.startswith("llm_out:") and d.category == "")


_RAW_REFUSAL = re.compile(r"(?<![a-z])(harmful|rule[ _-]?change|advice|programming[ _-]?task)(?![a-z])")


def _raw_names_refusal(content) -> bool:
    """PATHA-1028 Q2 / s4.1 D1: True when the raw reply text names a refusal category anywhere (lenient, no decode), so
    a nullified reply that mentions e.g. "harmful" is declined at once and never retried. Non-str or any fault: True."""
    try:
        return not isinstance(content, str) or bool(_RAW_REFUSAL.search(_norm_ws(content)))
    except Exception:  # noqa: BLE001
        return True


def _layer2_decision(content, pt: int, ct: int, cost: float, calls: int) -> ScopeDecision:
    """One classifier reply -> ScopeDecision (unchanged A1 rules; moved out of route for the Q2 retry)."""
    in_scope, why = parse_verdict(content)
    if in_scope is None:
        return ScopeDecision(False, "layer2_llm", why, pt + ct, cost, calls)
    category = parse_category(content)
    if in_scope and category in REFUSE_CATEGORIES:  # SCOPE-1022: in_scope=true with an advice/harm/... category is a
        # contradictory verdict; fail closed with the fixed refusal (never answered, listed or acquired)
        return ScopeDecision(False, "layer2_llm", "classifier_malformed:category_conflict", pt + ct, cost, calls, category)
    tok = refusal_token(content)
    if tok and (in_scope or category not in REFUSE_CATEGORIES):  # HOTFIX-DUPKEY-1027: a refusal category named anywhere
        # else in the reply (nested, a key, a reason) is never answered, listed or acquired; fail closed, keep the category
        return ScopeDecision(False, "layer2_llm", "classifier_malformed:refusal_token", pt + ct, cost, calls, tok)
    return ScopeDecision(in_scope, "layer2_llm", ("llm_in:" if in_scope else "llm_out:") + why, pt + ct, cost, calls,
                         category)


def intent_clean(question: str, cfg: RouterConfig | None = None) -> bool:
    """SCOPE-1022: the single layer-1 intent rule shared by scope routing, discovery and acquisition. False when layer 1
    hard-blocks the question (injection, explicit personal advice, harmful operational request, coding task), when it
    holds the classifier's data markers, or on any error (fail closed). Topic words alone never make it False."""
    try:
        cfg = cfg or load_config()
        verdict, _why = layer1(question, cfg)
        return verdict != DECLINE and not has_marker(question)
    except Exception:  # noqa: BLE001
        return False


def decline_text(cfg: RouterConfig | None = None) -> str:
    return (cfg or load_config()).decline_template


def offtopic_enabled(cfg: RouterConfig | None = None) -> bool:
    """FIX1-1021: True only for an explicit boolean true in `offtopic_reading_list` (missing/other -> off, the refusal)."""
    return (cfg or load_config()).raw.get("offtopic_reading_list") is True


def offtopic_applies(decision: ScopeDecision, cfg: RouterConfig | None = None) -> bool:
    """True only for a declined question whose decline came from a WELL-FORMED layer-2 LLM out-of-scope verdict (so
    layer 1 matched no deny/injection rule). Layer-1 declines, malformed replies and layer2_mode none never qualify."""
    cfg = cfg or load_config()
    return (offtopic_enabled(cfg) and not decision.accepted and decision.layer == "layer2_llm"
            and decision.reason.startswith("llm_out:"))


def offtopic_providers(cfg: RouterConfig | None = None) -> list[str]:
    provs = (cfg or load_config()).raw.get("offtopic_providers")
    if not isinstance(provs, list) or not provs or not all(isinstance(p, str) for p in provs):
        raise ScopeConfigError("scope_router.json: offtopic_providers must be a non-empty list of names")
    return list(provs)


def offtopic_openalex_profile(cfg: RouterConfig | None = None) -> dict | None:
    """RL-REL-1021: the OpenAlex term-AND profile for the off-topic reading list, or None (previous OR query) when the
    key is missing or malformed."""
    p = (cfg or load_config()).raw.get("offtopic_openalex_profile")
    if not isinstance(p, dict) or not isinstance(p.get("and_extra_max"), int) or p["and_extra_max"] < 1:
        return None
    return {"and_extra_max": p["and_extra_max"], "generic_terms": list(p.get("generic_terms") or ())}


def acquisition_off_note(cfg: RouterConfig | None = None) -> str:
    """PATHA-1028 CT-2: the fixed sentence appended to the reading list while automatic search is off ("" when the key
    is missing or not a string)."""
    t = (cfg or load_config()).raw.get("acquisition_off_note")
    return t.strip() if isinstance(t, str) else ""


def offtopic_notice(cfg: RouterConfig | None = None) -> str:
    cfg = cfg or load_config()
    t = cfg.raw.get("offtopic_notice")
    if not isinstance(t, str) or not t.strip():
        raise ScopeConfigError("scope_router.json: offtopic_notice missing")
    scope = str(cfg.raw.get("scope", "")).rstrip(".")
    scope = scope[:1].lower() + scope[1:]  # FIX1 verify D3: mid-sentence insertion, lower-case first letter
    return t.replace("{scope}", scope)
