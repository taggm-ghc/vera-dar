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

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from pathlib import Path
from typing import Callable

from vera.strict_json import base_norm as _base, loads_or_none, norm_key as _norm

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "scope_router.json"
LAYER2_MODES = ("llm", "none", "embedding")

ACCEPT, DECLINE, UNSURE = "accept", "decline", "unsure"


from vera.category_table import ScopeConfigError  # noqa: E402,F401  (defined there so the table parser has no cycle)


@dataclass(frozen=True)
class ScopeDecision:
    accepted: bool
    layer: str  # "layer1" | "layer2_llm" | "layer2_none"
    reason: str
    tokens_used: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    category: str = ""  # DAR-1021: layer-2 verdict category ("" when absent or not in LAYER2_CATEGORIES)
    raw_refusal_named: bool = False  # CATSTEP-1028 (plan s27.1 C1): the final raw reply names a refusal word (lenient)
    topic_key: str = ""  # CATSTEP-1028 (plan s27.3): topic key derived by vera/catstep.py ("" = step not run)
    ooe_label: bool = False  # FINAL-REVIEW-1028 D4: final reply was invalid ONLY for an out-of-enum label (A' path)


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
    mode_claims: tuple[re.Pattern, ...] = ()  # SEC-V3-1028 / owner 2026-10-28: nullified, not refused


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
            layer2={**raw["layer2"], "system_prompt": render_category_prompt(raw["layer2"]["system_prompt"])},
            mode_claims=_compile(l1.get("mode_claim_patterns", []), "mode_claim_patterns"),
        )
    except KeyError as exc:
        raise ScopeConfigError(f"scope_router.json: missing key {exc}") from exc


def _read_owner_json(path, what: str):
    """Read an owner-written config file (read-only; the one json.loads of config in this module)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScopeConfigError(f"cannot load {what} {Path(path).name}: {exc}") from exc
    return raw


@lru_cache(maxsize=1)
def load_config(path: str = str(CONFIG_PATH)) -> RouterConfig:
    return parse_config(_read_owner_json(path, "scope router config"))


def strip_format_chars(text: str) -> str:
    """SEC-V3-1028: drop Unicode format characters (category Cf: soft hyphen U+00AD, zero-width space U+200B, ...),
    which would otherwise split a word so no layer-1 pattern sees it."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def normalise(question: str) -> str:
    return re.sub(r"\s+", " ", strip_format_chars(unicodedata.normalize("NFKC", question or ""))).strip().lower()


MODE_CLAIM_TOKEN = "injection:mode_claim_nullified"  # the only thing logged about a nullified mode claim


def _strip_mode_claims(text: str, cfg: RouterConfig) -> tuple[str, int]:
    n = 0
    for pat in cfg.mode_claims:
        text, k = pat.subn(" ", text)
        n += k
    return text, n


def _meaningful(text: str) -> bool:
    return bool(re.search(r"[^\W_]", text or ""))


def nullify_mode_claims(question: str, cfg: RouterConfig | None = None) -> tuple[str, int]:
    """SEC-V3-1028 + owner directive 2026-10-28 ("* [mode|etc.] enabled|activated|toggled|etc." pattern should be
    nullified): a mode-switch claim grants nothing and is never obeyed. Returns (question with every mode-claim clause
    removed, number removed). With no claim the question is returned unchanged; otherwise the text is NFKC-normalised,
    format characters (Cf) are dropped and whitespace is collapsed. The caller routes the rest through every normal
    check; an empty rest is declined by layer1 (deny:mode_claim). Never raises (on error: the question unchanged)."""
    try:
        cfg = cfg or load_config()
        q = question or ""
        t = strip_format_chars(unicodedata.normalize("NFKC", q))
        out, n = _strip_mode_claims(t, cfg)
        tc = collapse_letter_spaced(t)  # REL-FIX3-1028 D4: the same views as layer1 (a letter-spaced claim too)
        if tc != t:
            out_c, n_c = _strip_mode_claims(tc, cfg)
            if n_c > n:
                out, n = out_c, n_c
        if not n:
            return q, 0
        return re.sub(r"^[\s.!?;:,]+", "", re.sub(r"\s+", " ", out)).strip(), n
    except Exception:  # noqa: BLE001
        return question, 0


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


def _views(question: str) -> list[str]:
    """The layer-1 matching views: the normalised question and (SCOPE2-1022) its letter-spaced-collapsed form when it
    differs (matching only; the question itself is never altered)."""
    q = normalise(question)
    views = [q]
    qc = normalise(collapse_letter_spaced(strip_format_chars(question or "")))
    if qc != q:
        views.append(qc)
    return views


def injection_unstripped(question: str, cfg: RouterConfig | None = None) -> bool:
    """REL-FIX3-1028 D1: an injection pattern or a classifier data marker on the UN-STRIPPED text (every view), checked
    before any mode claim is nullified, so stripping a claim can never delete the last word of an injection
    ("ignore all previous instructions mode on"). A hit rejects the whole input. Fails closed (True) on error."""
    try:
        cfg = cfg or load_config()
        return has_marker(question) or any(_any(cfg.injection, v) for v in _views(question))
    except Exception:  # noqa: BLE001
        return True


# INJ-SEP-1028 (owner decision 2026-10-28 on OD-1028(21) clarification 3): reject the unauthorized instruction, not
# automatically the research question. Conservative: only WHOLE sentences that match an injection pattern are removed
# (like the mode-claim stripper); the rest is kept only when it is a well-formed standalone question, contains no
# injection on any view and no data marker, and it then goes through every normal gate (layer 1 on the rest AND the
# original deny categories, layer 2, catstep, evidence). A quoted span is masked (untrusted data, never obeyed) only
# inside a third-person question about it. Anything else: the whole input is declined as before (deny:injection).
INSTRUCTION_REMOVED_TOKEN = "injection:instruction_removed"  # the only thing logged about a removed instruction
_QUOTE_RE = re.compile("\"[^\"\\n]{1,300}\"|\u201c[^\u201d\\n]{1,300}\u201d|\u2018[^\u2019\\n]{1,300}\u2019"
                       "|(?<![^\\W_])'[^'\\n]{1,300}'(?![^\\W_])")
_PH_RE = re.compile("\ue000(\\d+)\ue001")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")
_INTERROG_RE = re.compile(r"^(what|why|how|which|who|whom|whose|when|where|is|are|was|were|does|do|did|has|have|had"
                          r"|can|could|should|would|will|may|might)\b")
_RESEARCH_VERB_RE = re.compile(r"^(explain|describe|summari[sz]e|compare|define|outline|discuss|review)\b")
# INJSEP-SECREVIEW-1028 D1: a wh-word followed by an auxiliary ("what does", "why do", "how is"); never "why not",
# "how about" or an auxiliary-first form ("do ... now?"), which can direct the quoted text at VERA
_META_START_RE = re.compile(r"^(what|why|how|which|who|where|when)\s+(is|are|was|were|does|do|did|has|have|had|can"
                            r"|could|would|should|might|may|will)\b")
_CONNECTIVE_RE = re.compile(r"^(then|and|also|now|instead|so|but|next|finally|afterwards|after that|otherwise|plus|or"
                            r"|additionally|furthermore|moreover|meanwhile|first|second|lastly)\b")
_SECOND_PERSON_RE = re.compile(r"\b(you|your|yours|yourself|yourselves|u|ur|thee|thou|thy|thine|ye|vera(-dar)?)\b")


def separation_enabled(cfg: RouterConfig) -> bool:
    """INJ-SEP-1028: config `injection_separation`; only an explicit boolean true enables it (INJSEP-SECREVIEW-1028
    D4: fail closed like offtopic_enabled; the shipped config sets true). Off restores the whole-input decline."""
    return cfg.raw.get("injection_separation") is True


def _hits_injection(text: str, cfg: RouterConfig) -> bool:
    return any(_any(cfg.injection, v) for v in _views(text))


def _meta_quote_question(outer: str, cfg: RouterConfig) -> bool:
    """A third-person interrogative ABOUT the quoted text (what/why/how/does ... ?), not directed at VERA."""
    o = normalise(outer)
    return (bool(_META_START_RE.match(o)) and o.endswith("?") and not _SECOND_PERSON_RE.search(o)
            and not _hits_injection(outer, cfg))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def _well_formed(sentences: list[str], quoted: bool = False) -> bool:
    """The remainder stands alone (INJSEP-SECREVIEW-1028 D1/D2): 3+ words, and EVERY sentence is an interrogative
    ending in '?' or opens with a research verb (explain, describe, ...); no sentence opens with a connective that leans
    on removed text, and no sentence holds a second-person token (you, your, VERA, ...: a residual instruction signal,
    even after a research verb). With a masked quote every sentence must be a third-person wh-question about it.
    Declaratives and imperatives are not kept: the input is declined whole instead (conservative)."""
    norm = [normalise(s) for s in sentences]
    if not norm or len(re.findall(r"[^\W_]+", " ".join(norm))) < 3:
        return False
    for s in norm:
        if _CONNECTIVE_RE.match(s) or _SECOND_PERSON_RE.search(s):
            return False
        if quoted:
            ok = bool(_META_START_RE.match(s)) and s.endswith("?")
        else:
            ok = bool((_INTERROG_RE.match(s) and s.endswith("?")) or _RESEARCH_VERB_RE.match(s))
        if not ok:
            return False
    return True


def separate_instructions(question: str, cfg: RouterConfig) -> tuple[str, str, int] | None:
    """INJ-SEP-1028: (rest, rest_for_matching, sentences_removed) or None when separation is unreliable (the caller
    declines the whole input). rest keeps the user's own words (quoted data intact); rest_for_matching has eligible
    quoted spans masked and is what layer 1 matches. Never raises (None on error)."""
    try:
        if not separation_enabled(cfg) or has_marker(question):
            return None
        t = strip_format_chars(unicodedata.normalize("NFKC", question or ""))
        spans: list[str] = []

        def _ph(m: re.Match) -> str:
            spans.append(m.group(0))
            return f"\ue000{len(spans) - 1}\ue001"

        masked = _QUOTE_RE.sub(_ph, t.replace("\ue000", "").replace("\ue001", ""))
        kept, kept_m, removed, quoted = [], [], 0, 0
        for s in _sentences(masked):
            out_s = _PH_RE.sub(lambda m: spans[int(m.group(1))], s)
            match_s = out_s
            if _PH_RE.search(s) and _meta_quote_question(_PH_RE.sub(" ", s), cfg):
                match_s, quoted = _PH_RE.sub(' "x" ', s), quoted + 1  # untrusted data: masked, never matched as orders
            if _hits_injection(match_s, cfg):
                removed += 1
                continue
            kept.append(out_s)
            kept_m.append(match_s)
        if not kept or not (removed or quoted):
            return None
        rest, rest_m = " ".join(kept), " ".join(kept_m)
        if _hits_injection(rest_m, cfg) or has_marker(rest):  # re-check the joined rest on every view
            return None
        if not _well_formed(_sentences(nullify_mode_claims(rest_m, cfg)[0]), quoted=bool(quoted)):
            return None
        return rest, rest_m, removed
    except Exception:  # noqa: BLE001
        return None


def strip_instructions(question: str, cfg: RouterConfig | None = None) -> tuple[str, int]:
    """INJ-SEP-1028: (question with removed injection sentences, number removed). Unchanged (question, 0) when nothing
    was removed or separation is unreliable (layer 1 then declines the whole input). Never raises."""
    try:
        cfg = cfg or load_config()
        if not _hits_injection(question, cfg):
            return question, 0
        sep = separate_instructions(question, cfg)
        if sep is None or not sep[2]:
            return question, 0
        return sep[0], sep[2]
    except Exception:  # noqa: BLE001
        return question, 0


def layer1(question: str, cfg: RouterConfig) -> tuple[str, str]:
    """Return (ACCEPT|DECLINE|UNSURE, reason). Zero cost, deterministic."""
    q = normalise(question)
    if not q:
        return DECLINE, "empty"
    views = orig_views = _views(question)
    separated = False
    # REL-FIX3-1028 D1: injection is checked on the UN-STRIPPED views first. INJ-SEP-1028: a hit rejects the whole
    # input unless the instruction separates reliably; the rest is then matched instead (original deny checked below)
    if any(_any(cfg.injection, v) for v in views):
        sep = separate_instructions(question, cfg)
        if sep is None:
            return DECLINE, "deny:injection"
        views, separated = _views(sep[1]), True
        q = views[0]
    # SEC-V3-1028 / owner 2026-10-28: mode-switch claims are nullified (stripped) on every view, never obeyed; the
    # rest is checked below as usual (injection again included). Nothing meaningful left -> the neutral decline.
    raw_views = views
    stripped = [_strip_mode_claims(v, cfg) for v in views]
    if any(n for _v, n in stripped):
        views = [re.sub(r"\s+", " ", v).strip() for v, _n in stripped]
        if not all(_meaningful(v) for v in views):  # REL-FIX3-1028 D4: any view that is only a claim
            return DECLINE, "deny:mode_claim"
        q = views[0]
    # REFUSE-FIX-1028 V1: injection is checked on every view before any deny category, and each category on every
    # view before the next one, so a softer reason (coding_task, which now gets a reading list) can never mask an
    # injection or harmful hit that only the letter-spaced view shows. Every outcome here is still DECLINE.
    if any(_any(cfg.injection, v) for v in views):
        return DECLINE, "deny:injection"
    # INJ-SEP-1028 (+ INJSEP-SECREVIEW-1028 D5: runs BEFORE the deny loop): removing an instruction never clears a
    # deny rule the ORIGINAL matches; any such hit keeps the plain injection decline, never a softer path such as
    # the coding reading list (_coding_hint), whether the hit is in the removed sentence or the kept rest
    if separated and any(_any(pats, v) for pats in cfg.deny.values() for v in orig_views):
        return DECLINE, "deny:injection"
    deny_views = views if views is raw_views else views + raw_views  # D1: the un-stripped text is checked too
    for name, pats in cfg.deny.items():
        if any(_any(pats, v) for v in deny_views):
            return DECLINE, f"deny:{name}" if not separated else "deny:injection"
    if _any(cfg.tool_names, q):
        return ACCEPT, "tool_name"
    if _any(cfg.ai_terms, q) and _any(cfg.dev_terms, q):
        return ACCEPT, "ai_and_dev_terms"
    return UNSURE, "no_rule"


_MARKER_RE = re.compile(r"<<+\s*question|question\s*>>+", re.IGNORECASE)  # SEC C1: 2+ angle brackets
_VERDICT_TOKENS_RE = re.compile(r"\b(categor(y|ies)|in[\s_-]*scope|off[\s_-]*topic)\b", re.IGNORECASE)  # SEC C1


def has_marker(question: str) -> bool:
    """DAR-1021 SEC H1: the question contains the classifier's own data markers (an attempt to close the data block)."""
    return bool(_MARKER_RE.search(strip_format_chars(unicodedata.normalize("NFKC", str(question or "")))))


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


# OOE-APRIME-C-1028 (OD-1028(38), owner "a' + c"): the layer-2 categories come from ONE owner-curated, versioned table,
# config/classifier_categories.json. Read-only at runtime: loaded once at import into immutable structures (frozen rows,
# tuples, frozensets, a read-only alias map); no code path writes or extends it (the model never creates a research
# route). An invalid table raises ScopeConfigError at import; main.py imports this module at process start
# (OOE-APRIME-C-1028 D2), so a bad table stops the service from starting, and any later lazy import fails too.
CATEGORIES_PATH = Path(__file__).resolve().parent.parent / "config" / "classifier_categories.json"
from vera.category_table import (ATTESTATION_SCHEMA, Category, DESCRIPTION_SIMILARITY_MAX, DISPOSITIONS,  # noqa: E402,F401
                                REFUSING_DISPOSITIONS, description_similarity, descriptions_sha256, parse_categories,
                                verify_attestation)
ATTESTATION_PATH = Path(__file__).resolve().parent.parent / "config" / "category_attestation.json"
SEMSIM_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "semantic_similarity.json"


def load_categories(path: str | Path = CATEGORIES_PATH) -> tuple[Category, ...]:
    raw = _read_owner_json(path, "classifier category table")
    att = sem = None
    if isinstance(raw, dict) and raw.get("attestation_required") is True:
        att = _read_owner_json(ATTESTATION_PATH, "category attestation")
        sem = _read_owner_json(SEMSIM_CONFIG_PATH, "semantic similarity config")
    return parse_categories(raw, att, sem)


CATEGORY_TABLE: tuple[Category, ...] = load_categories()
LAYER2_CATEGORIES = tuple(c.code for c in CATEGORY_TABLE)
CATEGORY_ALIASES = MappingProxyType({a: c.code for c in CATEGORY_TABLE for a in c.aliases})  # output label -> code
REFUSE_CATEGORIES = frozenset(c.code for c in CATEGORY_TABLE if c.disposition in REFUSING_DISPOSITIONS)  # SCOPE-1022: never answered, listed or acquired


def resolve_label(label) -> str:
    """A classifier output label -> its code ("" when not a code or alias). Aliases never apply to user text."""
    if not isinstance(label, str):
        return ""
    return label if label in LAYER2_CATEGORIES else CATEGORY_ALIASES.get(label, "")


def render_category_prompt(template: str) -> str:
    """The classifier prompt's category enum and meanings, rendered from the table into the {{category_enum}} and
    {{category_meanings}} slots of layer2.system_prompt."""
    if not isinstance(template, str) or "{{category_enum}}" not in template or "{{category_meanings}}" not in template:
        raise ScopeConfigError("scope_router.json: layer2.system_prompt must hold {{category_enum}} and {{category_meanings}}")
    enum = " or ".join(f'"{c.code}"' for c in CATEGORY_TABLE)
    meanings = "; ".join(f'"{c.code}": {c.description}' for c in CATEGORY_TABLE)
    return template.replace("{{category_enum}}", enum).replace("{{category_meanings}}", meanings)

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


_REFUSE_ORDER = tuple(c.code for c in sorted((c for c in CATEGORY_TABLE if c.code in REFUSE_CATEGORIES),
                                               key=lambda c: c.refusal_rank))  # SCOPE v2 precedence (table refusal_rank)
_CATS = "(" + "|".join(c.replace("_", "[ _]?") for c in _REFUSE_ORDER) + ")"
_RAW_QUOTED = re.compile(r'\\*"' + _CATS + r'\\*"')  # SEC-H-2/H-4: any run of backslashes before each quote
_CAT_IN_TEXT = re.compile(r"categor(?:y|ies)\W{0,8}" + _CATS)  # SEC-H-2/H-6: bounded, no \w* backtracking
_CAT_SPELLINGS = {**{c: c for c in _REFUSE_ORDER}, **{c.replace("_", ""): c for c in _REFUSE_ORDER}}


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
    return resolve_label(c)  # OOE-APRIME-C-1028: a code, or an owner-curated alias of one, from the table


def _label_only_invalid(content) -> bool:
    """OOE-APRIME-C-1028 (A'): True when the reply is well-formed (strict decode, boolean in_scope, ASCII category) and
    invalid ONLY because its non-empty category label is neither a table code nor an alias. Never raises."""
    try:
        obj = _strict_loads(content)
        if not isinstance(obj, dict) or not isinstance(obj.get("in_scope"), bool):
            return False
        c = obj.get("category")
        return (isinstance(c, str) and unicodedata.normalize("NFKC", c).isascii() and bool(c.strip())
                and not resolve_label(c))
    except Exception:  # noqa: BLE001
        return False


def parse_verdict(content: str, unknown_label_ok: bool = False) -> tuple[bool | None, str]:
    """Return (in_scope, reason); in_scope None when the reply is malformed (caller fails closed). unknown_label_ok
    (OOE-APRIME-C-1028, used only after the retry) reads an out-of-enum label as no recognised category."""
    obj = _strict_loads(content)
    if not isinstance(obj, dict) or not isinstance(obj.get("in_scope"), bool):
        return None, "classifier_malformed"
    c = obj.get("category")
    if isinstance(c, str) and not unicodedata.normalize("NFKC", c).isascii():  # lookalike value, e.g. Cyrillic а
        return None, "classifier_malformed"
    if isinstance(c, str) and c.strip() and not resolve_label(c) and not unknown_label_ok:  # OOE-A-PRESTAGE-1028:
        # a non-empty label outside the closed enum is invalid classifier output (OD-1028(21) c4), not user misuse;
        # nullified and retried once (OOE-APRIME-C-1028: still out-of-enum after the retry -> no recognised category)
        return None, "classifier_malformed"
    return obj["in_scope"], str(obj.get("reason", ""))[:200]


def _masked_quote(question: str, cfg: RouterConfig) -> bool:
    """INJSEP-SECREVIEW-2: True when separation masked a quoted span (rest_for_matching differs from rest). Never raises."""
    try:
        sep = separate_instructions(question, cfg)
        return sep is not None and sep[0] != sep[1]
    except Exception:  # noqa: BLE001
        return False


def route(question: str, model: str, pricing=None, cfg: RouterConfig | None = None,
          chat: ChatFn | None = None) -> ScopeDecision:
    cfg = cfg or load_config()
    original = question
    question = strip_instructions(question, cfg)[0]  # INJ-SEP-1028: nor a removed instruction sentence
    question = nullify_mode_claims(question, cfg)[0]  # SEC-V3-1028: the classifier never sees a mode claim either
    # REL-FIX3-1028 D1: layer 1 runs on the ORIGINAL text (it checks injection before stripping mode claims itself),
    # and the marker check covers the original as well as the stripped text
    verdict, reason = layer1(original, cfg)
    marker = has_marker(original) or has_marker(question)
    if verdict == ACCEPT and _masked_quote(original, cfg):
        verdict, reason = UNSURE, "masked_quote"  # INJSEP-SECREVIEW-2: a masked quoted span always gets layer 2
    if verdict == DECLINE and reason in (UNSUPPORTED_L1_REASON, "deny:mode_claim") and marker:
        # REFUSE-FIX-1028 V1: the marker check runs before the unsupported-use path (fixed refusal, never a list)
        return ScopeDecision(False, "layer1", "deny:injection_marker")
    if verdict == DECLINE and reason == UNSUPPORTED_L1_REASON:
        return _coding_hint(question, model, pricing, cfg, chat)
    if verdict != UNSURE:
        return ScopeDecision(verdict == ACCEPT, "layer1", reason)
    if marker:  # DAR-1021 SEC H1: fixed refusal, no classifier call, never a reading list or acquisition
        return ScopeDecision(False, "layer1", "deny:injection_marker")
    if cfg.layer2_mode == "none":
        return ScopeDecision(cfg.none_mode_unsure_action == ACCEPT, "layer2_none", "unsure:" + cfg.none_mode_unsure_action)
    return _classify(question, model, pricing, cfg, chat)


def _classify(question: str, model: str, pricing, cfg: RouterConfig, chat: ChatFn | None) -> ScopeDecision:
    """The layer-2 classifier call with the PATHA-1028 Q2 retry (unchanged; moved out of route for REFUSE-FIX2-1028)."""
    messages = classifier_messages(question, cfg)
    retries = int(cfg.layer2.get("malformed_retry", 1))
    pt = ct = calls = 0
    ooe_labels: list[str] = []  # OOE-APRIME-C-1028: sanitised labels of label-only invalid replies, for the label log
    while True:  # PATHA-1028 Q2: at most 1 + malformed_retry (<= 2) classifier calls
        content, p_, c_ = (chat or default_chat)(messages, model, cfg.layer2)
        pt, ct, calls = pt + p_, ct + c_, calls + 1
        cost = 0.0
        if pricing is not None:
            cost = round(pt / 1_000_000 * pricing.input + ct / 1_000_000 * pricing.output, 8)
        d = _layer2_decision(content, pt, ct, cost, calls)
        label_only = d.reason == "classifier_malformed" and _label_only_invalid(content)
        if label_only:
            lab = _ooe_label(content)
            if lab and lab not in ooe_labels:
                ooe_labels.append(lab)
        if calls > retries or not _nullified(d) or _raw_names_refusal(content):
            if label_only and not _raw_names_refusal(content):  # OOE-APRIME-C-1028 (A'): still out-of-enum after the
                # retry -> no recognised category: the absent-category path (catstep gates: layer 1, intent_clean, ...)
                d = _layer2_decision(content, pt, ct, cost, calls, unknown_label_ok=True)
                if d.reason.startswith(("llm_out:", "llm_in:")) and d.category == "":
                    from dataclasses import replace as _replace_l
                    d = _replace_l(d, ooe_label=True)
            if calls > 1:
                import logging
                logging.getLogger("vera").info("scope classifier retry: calls=%d nullified=%s", calls, _nullified(d))
            if d.reason.startswith("llm_out:") and d.category == "" and _raw_names_refusal(content):
                from dataclasses import replace as _replace
                d = _replace(d, raw_refusal_named=True)  # CATSTEP-1028 C1: never categorised (keeps the decline)
            if ooe_labels:  # OOE-APRIME-C-1028 D3: the outcome that actually happens (logged after it is final)
                if d.reason.startswith("llm_out:") and d.category == "" and not d.raw_refusal_named:
                    outcome = "catstep_after_retry"  # out-of-enum or absent category: the catstep path (its gates)
                elif d.reason.startswith("classifier_malformed") or d.raw_refusal_named:
                    outcome = "declined"
                elif d.accepted and d.category == "":
                    outcome = "accepted_after_retry"  # FINAL-REVIEW-1028 D1: in_scope=true, label still out-of-enum
                else:
                    outcome = "retry_resolved"
                import logging
                logging.getLogger("vera").info("scope classifier out-of-enum category: label=%s outcome=%s",
                                               ",".join(ooe_labels), outcome)
            return d


# REFUSE-FIX2-1028 (OD-1028(26): layer 2 is the primary intent classifier): closed set of layer-2 categories that
# confirm a layer-1 coding hint as unsupported use; and the closed fallback tokens logged when layer 2 cannot confirm.
CODING_HINT_CONFIRM = frozenset(c.code for c in CATEGORY_TABLE  # OOE-APRIME-C-1028: from the table
                                if c.disposition in ("unsupported_use", "off_topic", "in_scope"))
CODING_HINT_FALLBACKS = ("l2_unavailable", "l2_mode_none", "l2_malformed")


def _coding_fallback(token: str, d: ScopeDecision | None = None) -> ScopeDecision:
    import logging
    logging.getLogger("vera").info("scope coding hint: layer2=%s", token)
    return ScopeDecision(False, "layer1", f"{UNSUPPORTED_L1_REASON}:{token}", *(
        (d.tokens_used, d.cost_usd, d.llm_calls) if d is not None else ()))


def _coding_hint(question: str, model: str, pricing, cfg: RouterConfig, chat: ChatFn | None) -> ScopeDecision:
    """REFUSE-FIX2-1028: a layer-1 deny:coding_task hit is a HINT; layer 2 also runs (OD-1028(26)).
    - a CLEAN layer-2 verdict (reason llm_out:/llm_in:, no refusal token or conflict) whose category is
      programming_task, off_topic or in_scope -> ScopeDecision(False, "layer1", "deny:coding_task") carrying the
      classifier spend: the unsupported-use sentence plus the reading list (unsupported_use); never an answer, code or
      acquisition (layer 1 stays DECLINE, so intent_clean is False);
    - any other refusal category (advice, harmful, rule_change, also via refusal_token or category_conflict) -> that
      layer-2 decision unchanged, refused by the existing paths;
    - layer2_mode none, a classifier error, or a malformed / nullified / unexpected verdict -> the fixed decline
      (reason deny:coding_task:<token>, token from CODING_HINT_FALLBACKS, also logged). Never raises."""
    try:
        if cfg.layer2_mode == "none":
            return _coding_fallback("l2_mode_none")
        try:
            d = _classify(question, model, pricing, cfg, chat)
        except Exception as exc:  # noqa: BLE001  unavailable classifier: the decline, as before V1
            import logging
            logging.getLogger("vera").warning("scope coding hint classifier unavailable (%s)", type(exc).__name__)
            return _coding_fallback("l2_unavailable")
        if d.category in REFUSE_CATEGORIES - {UNSUPPORTED_CATEGORY} and not d.accepted:
            return d  # advice / harmful / rule_change (clean, refusal_token or category_conflict): existing refusal
        if d.reason.startswith(("llm_out:", "llm_in:")) and d.category in CODING_HINT_CONFIRM:
            return ScopeDecision(False, "layer1", UNSUPPORTED_L1_REASON, d.tokens_used, d.cost_usd, d.llm_calls)
        if d.ooe_label and d.category == "" and not d.raw_refusal_named:  # FINAL-REVIEW-1028 D4 (OD-1028(21), (38)):
            # a label-only-invalid reply twice is never a decline solely for the label: unsupported use + reading list
            return ScopeDecision(False, "layer1", UNSUPPORTED_L1_REASON, d.tokens_used, d.cost_usd, d.llm_calls)
        return _coding_fallback("l2_malformed", d)
    except Exception:  # noqa: BLE001  fail closed
        return ScopeDecision(False, "layer1", f"{UNSUPPORTED_L1_REASON}:l2_unavailable")


def _nullified(d: ScopeDecision) -> bool:
    """PATHA-1028 Q2: a verdict with no usable classification (plan s26.1 P5 and P10b): a malformed, oversize, bad-JSON,
    duplicate-key or non-ASCII-category reply, or a well-formed in_scope=false reply whose category is missing or not
    in the closed enum. category_conflict and refusal_token verdicts are safety refusals, not nullified."""
    return d.reason == "classifier_malformed" or (d.reason.startswith("llm_out:") and d.category == "")


_RAW_REFUSAL = re.compile(r"(?<![a-z])(" + "|".join(c.replace("_", "[ _-]?") for c in _REFUSE_ORDER) + r")(?![a-z])")


def _raw_names_refusal(content) -> bool:
    """PATHA-1028 Q2 / s4.1 D1: True when the raw reply text names a refusal category anywhere (lenient, no decode), so
    a nullified reply that mentions e.g. "harmful" is declined at once and never retried. Non-str or any fault: True."""
    try:
        return not isinstance(content, str) or bool(_RAW_REFUSAL.search(_norm_ws(content)))
    except Exception:  # noqa: BLE001
        return True


def _ooe_label(content) -> str:
    """OOE-A-PRESTAGE-1028: the out-of-enum category label only, sanitised (<= 32 chars; never question text); ""
    when there is none."""
    try:
        obj = _strict_loads(content)
        c = obj.get("category") if isinstance(obj, dict) else None
        if isinstance(c, str) and c.strip() and not resolve_label(c) and c.isascii():
            lab = c.strip().lower()  # FINAL-REVIEW-1028 D2: only a label-shaped token is ever logged
            return lab if re.fullmatch(r"[a-z][a-z0-9_]{0,31}", lab) else "nonlabel"
    except Exception:  # noqa: BLE001
        pass
    return ""


def _layer2_decision(content, pt: int, ct: int, cost: float, calls: int, unknown_label_ok: bool = False) -> ScopeDecision:
    """One classifier reply -> ScopeDecision (unchanged A1 rules; moved out of route for the Q2 retry)."""
    in_scope, why = parse_verdict(content, unknown_label_ok)
    if in_scope is None:
        return ScopeDecision(False, "layer2_llm", why, pt + ct, cost, calls)
    category = parse_category(content)
    if in_scope and category in REFUSE_CATEGORIES:  # SCOPE-1022: in_scope=true with an advice/harm/... category is a
        # contradictory verdict; fail closed with the fixed refusal (never answered, listed or acquired)
        return ScopeDecision(False, "layer2_llm", "classifier_malformed:category_conflict", pt + ct, cost, calls, category)
    tok = refusal_token(content)
    if category == UNSUPPORTED_CATEGORY and tok and tok != category:  # REFUSE-FIX-1028 V1: a programming_task verdict
        # that also names a stronger refusal category (harmful, rule_change, advice) keeps the fixed refusal, so the
        # unsupported-use reading list never masks it
        return ScopeDecision(False, "layer2_llm", "classifier_malformed:refusal_token", pt + ct, cost, calls, tok)
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


UNSUPPORTED_L1_REASON = "deny:coding_task"  # REFUSE-FIX-1028 V1 (OD-1028(21)): unsupported use, not a forbidden topic
UNSUPPORTED_CATEGORY = next(c.code for c in CATEGORY_TABLE if c.disposition == "unsupported_use")  # OOE-APRIME-C-1028


def unsupported_use_text(cfg: RouterConfig | None = None) -> str:
    """REFUSE-FIX-1028 V1: the fixed neutral unsupported-use sentence ("" when missing or not a string)."""
    t = (cfg or load_config()).raw.get("unsupported_use_text")
    return t.strip() if isinstance(t, str) else ""


def unsupported_use(decision: ScopeDecision, cfg: RouterConfig | None = None) -> bool:
    """REFUSE-FIX-1028 V1 (OD-1028(21) and clarification): True for a request to write or fix code: a layer-1
    deny:coding_task hit CONFIRMED by a clean layer-2 verdict (REFUSE-FIX2-1028, _coding_hint; layer 1 checks injection
    and every other deny category first; route checks the data markers first) or a CLEAN layer-2 verdict (reason llm_out:) whose category is programming_task. Such a request gets the
    fixed unsupported-use sentence plus the scholarly reading list: no code, no model answer, no acquisition.
    refusal_token, category_conflict and malformed verdicts never qualify. False when the sentence is not configured
    or on any error (fail closed: decline_template)."""
    try:
        cfg = cfg or load_config()
        if decision.accepted or not unsupported_use_text(cfg):
            return False
        if decision.layer == "layer1":  # REFUSE-FIX2-1028: only after layer 2 confirmed the hint (route's
            # _coding_hint is the sole producer of this exact reason, always with >= 1 classifier call)
            return decision.reason == UNSUPPORTED_L1_REASON and decision.llm_calls >= 1
        return (decision.layer == "layer2_llm" and decision.reason.startswith("llm_out:")
                and decision.category == UNSUPPORTED_CATEGORY)
    except Exception:  # noqa: BLE001
        return False


def offtopic_applies(decision: ScopeDecision, cfg: RouterConfig | None = None) -> bool:
    """True only for a declined question whose decline came from a WELL-FORMED layer-2 LLM out-of-scope verdict (so
    layer 1 matched no deny/injection rule), or for an unsupported-use request (REFUSE-FIX-1028 V1, unsupported_use).
    Other layer-1 declines, malformed replies and layer2_mode none never qualify."""
    cfg = cfg or load_config()
    if not offtopic_enabled(cfg) or decision.accepted:
        return False
    return (decision.layer == "layer2_llm" and decision.reason.startswith("llm_out:")) or unsupported_use(decision, cfg)


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


# ---- SELFREF-1028 (owner directive 2026-10-28): self-reflective questions answered from fixed public text -------------
_SELF_TRAIL_RE = re.compile(r"[\s?.!]+$")
_SELF_LEAD_RE = re.compile(r"^(hi|hello|hey|ok|okay|so|and)[ ,]+|^please[ ,]+")


def _self_cfg(cfg: RouterConfig | None) -> dict:
    sr = (cfg or load_config()).raw.get("self_reflective")
    return sr if isinstance(sr, dict) else {}


def self_reflective(question: str, cfg: RouterConfig | None = None) -> bool:
    """True only for a benign second-person question about VERA-DAR's identity, capabilities or how it works that
    FULLY matches one pattern of the closed list `self_reflective.patterns`. Injection, deny and marker checks run
    first (layer 1 verdict DECLINE or a classifier marker -> False), so an instruction/config/prompt request never
    qualifies. Fails closed (False) on a missing key, a bad pattern or any error."""
    try:
        cfg = cfg or load_config()
        pats = _self_cfg(cfg).get("patterns")
        if not isinstance(pats, list) or not pats or has_marker(question):
            return False
        if layer1(question, cfg)[0] == DECLINE:
            return False
        q = normalise(question)
        if len(q) > 120:
            return False
        q = _SELF_TRAIL_RE.sub("", q)
        q = _SELF_LEAD_RE.sub("", q)
        return any(re.fullmatch(p, q, re.IGNORECASE) for p in pats if isinstance(p, str))
    except Exception:  # noqa: BLE001
        return False


def self_reflective_text(cfg: RouterConfig | None = None) -> str:
    """The fixed reply: the configured identity line plus the public disclosure text. "" when not configured."""
    try:
        ident = _self_cfg(cfg).get("identity_text")
        if not isinstance(ident, str) or not ident.strip():
            return ""
        from vera.disclosure import disclosure_text
        return f"{ident.strip()} {disclosure_text()}"
    except Exception:  # noqa: BLE001
        return ""
