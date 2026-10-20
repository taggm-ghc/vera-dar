"""Memory write gate (item #84): a pure, deterministic structural filter.

No DB, no model call, no network, no I/O. It decides whether one candidate
finding may be offered for storage. It is a heuristic filter that catches crude
and known patterns only; a pass is not proof of safety, and it is not a
security guarantee (the claim checker's verdict is one input, not the gate).

Injection/invisible-character logic is adapted from AI-Internship
week-1v2 intake/detectors.py (phrase, delimiter, role-line lists) and
intake/extract.py (is_invisible); the lists are copied minimally here.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Sequence

VERDICT_SUPPORTED = "supported"

# Refusal reasons (stable identifiers).
NOT_ALLOWED_TYPE = "fact_type_not_allowed"
NOT_SUPPORTED = "not_supported"
NO_CITATION = "no_citation"
PROVIDER_NOT_ALLOWED = "provider_not_allowed"
LICENCE = "licence"
NO_IDENTIFIER = "no_identifier"
INVISIBLE_UNICODE = "invisible_unicode"
INJECTION_PATTERN = "injection_pattern"
TOOL_DUMP = "tool_dump"
PERSONAL = "personal"
ECHOES_QUESTION = "echoes_question"
VOLATILE = "volatile"
LENGTH = "length"
MARKUP = "markup"
FRAGMENT = "fragment"
ALL_REASONS = (NOT_ALLOWED_TYPE, NOT_SUPPORTED, NO_CITATION, PROVIDER_NOT_ALLOWED,
               LICENCE, NO_IDENTIFIER, INVISIBLE_UNICODE, INJECTION_PATTERN,
               TOOL_DUMP, PERSONAL, ECHOES_QUESTION, VOLATILE, LENGTH, MARKUP, FRAGMENT)


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "ask-provider-chain.json"
CONFIG_KEY = "memory"


@dataclass(frozen=True)
class GateConfig:
    """All thresholds and lists come from the `memory` block of config/ask-provider-chain.json."""
    allowed_fact_types: tuple
    allowed_providers: tuple
    allowed_licence: tuple
    min_chars: int
    min_words: int
    terminal_punctuation: str
    ellipsis_patterns: tuple
    markup_patterns: tuple
    fragment_start_words: tuple
    fragment_patterns: tuple
    max_chars: int
    max_question_overlap: float
    min_overlap_tokens: int
    long_digit_run: int
    invisible_categories: tuple
    invisible_ranges: tuple
    injection_phrases: tuple
    delimiter_patterns: tuple
    role_line_patterns: tuple
    tool_dump_patterns: tuple
    personal_words: tuple
    personal_patterns: tuple
    volatile_terms: tuple
    leading_marker_pattern: str
    inline_strip_patterns: tuple
    citation_marker_pattern: str
    provider_separator: str
    # Words that, alone inside parentheses, mark a remnant of a stripped citation such as "(as in)".
    dangling_parenthetical_words: tuple = ()


class MemoryGateConfigError(ValueError):
    """Raised verbosely when the `memory` config block is missing or incomplete."""


def gate_config_from_dict(block: dict) -> GateConfig:
    out = {}
    for f in fields(GateConfig):
        if f.name not in block and f.default is not MISSING:
            continue
        if f.name not in block:
            raise MemoryGateConfigError(f"config `{CONFIG_KEY}` block is missing key {f.name!r}")
        v = block[f.name]
        out[f.name] = (tuple(tuple(x) if isinstance(x, list) else x for x in v)
                       if isinstance(v, list) else v)
    return GateConfig(**out)


def load_gate_config(path: Path = CONFIG_PATH) -> GateConfig:
    try:
        block = json.loads(Path(path).read_text())[CONFIG_KEY]
    except (OSError, ValueError, KeyError) as exc:
        raise MemoryGateConfigError(f"cannot load `{CONFIG_KEY}` block from {path}: {type(exc).__name__}") from exc
    return gate_config_from_dict(block)


@dataclass(frozen=True)
class SourceRef:
    provider: str
    licence_decision: str
    identifier: str          # DOI / arXiv id / URL of the cited source
    url: str = ""


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    reasons: tuple = field(default_factory=tuple)


_TOKEN = re.compile(r"[a-z0-9]+")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).casefold()


def _has_invisible(text: str, cfg: GateConfig) -> bool:
    for ch in text:
        cp = ord(ch)
        if unicodedata.category(ch) in cfg.invisible_categories or any(
                lo <= cp <= hi for lo, hi in cfg.invisible_ranges):
            return True
    return False


def _strip_invisible(text: str, cfg: GateConfig) -> str:
    return "".join(ch for ch in text if not _has_invisible(ch, cfg))


def _injection_hit(text: str, cfg: GateConfig) -> bool:
    norm = re.sub(r"[ \t\r\f\v]+", " ", unicodedata.normalize("NFKC", _strip_invisible(text, cfg)))
    sq = _squash(norm)
    if any(_squash(p) in sq for p in cfg.injection_phrases):
        return True
    if any(re.search(p, sq, re.I) for p in cfg.delimiter_patterns):
        return True
    return any(re.search(p, norm, re.I | re.M) for p in cfg.role_line_patterns)


def _tool_dump(text: str, cfg: GateConfig) -> bool:
    return (any(re.search(p, text, re.I) for p in cfg.tool_dump_patterns)
            or re.search(r"\d{%d,}" % cfg.long_digit_run, text) is not None)


def _personal(text: str, cfg: GateConfig) -> bool:
    words = "|".join(re.escape(w) for w in cfg.personal_words)
    return (bool(words) and re.search(r"\b(%s)\b" % words, text, re.I) is not None) or any(
        re.search(p, text) for p in cfg.personal_patterns)


def _drop_dangling_parentheticals(text: str, cfg: GateConfig) -> str:
    """Remove parentheses left empty or holding only connector words (e.g. "(as in)") once markers are gone.
    A parenthetical with any other word, number or symbol is real content and is kept."""
    words = {w.casefold() for w in cfg.dangling_parenthetical_words}
    if not words:
        return text

    def _repl(m):
        parts = [p.replace(".", "").casefold() for p in re.split(r"[\s,;:]+", m.group(1)) if p.strip(".")]
        return "" if all(p in words for p in parts) else m.group(0)

    t = re.sub(r"\s*\(([^()]*)\)", _repl, text)
    return re.sub(r"\s+([,.;:!?])", r"\1", t)


def strip_citation_markers(text: str, cfg: GateConfig) -> str:
    """Remove `[n]` / `[n, m]` citation markers and the dangling parentheticals they leave behind
    (e.g. "(as in [1])" -> nothing); use before checking and before storing."""
    t = re.sub(cfg.citation_marker_pattern, "", text or "")
    return _drop_dangling_parentheticals(t, cfg).strip()


def normalise_claim(text: str, cfg: GateConfig) -> str:
    """Strip citation markers, one leading list marker, markdown emphasis/backticks and stray <br>; collapse
    whitespace. Table rows (pipe-delimited cells) are left untouched so the markup rule still refuses them."""
    t = strip_citation_markers(text, cfg)
    if any(re.search(p, t) for p in cfg.markup_patterns[:1]):
        return t
    t = re.sub(cfg.leading_marker_pattern, "", t)
    for p in cfg.inline_strip_patterns:
        t = re.sub(p, " " if p.startswith("<") else "", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def _provider_allowed(provider: str, cfg: GateConfig) -> bool:
    """Composite providers such as `openalex+semanticscholar` need every part allow-listed."""
    parts = [p.strip() for p in (provider or "").split(cfg.provider_separator)]
    return bool(parts) and all(p and p in cfg.allowed_providers for p in parts)


def _overlap(text: str, question: str, cfg: GateConfig) -> float:
    q = set(_TOKEN.findall(_squash(question)))
    if len(q) < cfg.min_overlap_tokens:
        return 0.0
    t = set(_TOKEN.findall(_squash(text)))
    return len(q & t) / len(q)


def _markup(text: str, cfg: GateConfig) -> bool:
    return any(re.search(p, text, re.M) for p in cfg.markup_patterns)


def _fragment(text: str, cfg: GateConfig) -> bool:
    """Not a complete standalone sentence: no terminal punctuation, trailing ellipsis, lowercase or connector
    opening, or a known dangling-subject opening ("In the X shows that")."""
    s = text.strip()
    if not s or s[-1] not in cfg.terminal_punctuation:
        return True
    if any(re.search(p, s) for p in cfg.ellipsis_patterns):
        return True
    if not s[0].isupper():
        return True
    first = (re.findall(r"[A-Za-z]+", s) or [""])[0].lower()
    if first in cfg.fragment_start_words:
        return True
    return any(re.search(p, s, re.I) for p in cfg.fragment_patterns)


def check_memory_write(fact_type: str, text: str, verdict: str, citations: Sequence[SourceRef],
                       question: str, cfg: GateConfig) -> GateResult:
    """Return GateResult(allowed, sorted unique reasons). Structural filter only.

    `[n]` markers are stripped from `text` before checks; callers store strip_citation_markers(text, cfg)."""
    reasons: set[str] = set()
    raw = strip_citation_markers(text, cfg)
    text = normalise_claim(text, cfg)
    if fact_type not in cfg.allowed_fact_types:
        reasons.add(NOT_ALLOWED_TYPE)
    if verdict != VERDICT_SUPPORTED:
        reasons.add(NOT_SUPPORTED)
    if not citations:
        reasons.add(NO_CITATION)
    for s in citations:
        if not _provider_allowed(s.provider, cfg):
            reasons.add(PROVIDER_NOT_ALLOWED)
        if s.licence_decision not in cfg.allowed_licence:
            reasons.add(LICENCE)
        if not (s.identifier or "").strip():
            reasons.add(NO_IDENTIFIER)
    if _has_invisible(text, cfg):
        reasons.add(INVISIBLE_UNICODE)
    if _injection_hit(raw, cfg) or _injection_hit(text, cfg):
        reasons.add(INJECTION_PATTERN)
    if _tool_dump(text, cfg):
        reasons.add(TOOL_DUMP)
    if _personal(text, cfg):
        reasons.add(PERSONAL)
    if _overlap(text, question or "", cfg) > cfg.max_question_overlap:
        reasons.add(ECHOES_QUESTION)
    low = _squash(text)
    if any(re.search(r"\b%s\b" % re.escape(t), low) for t in cfg.volatile_terms):
        reasons.add(VOLATILE)
    if not cfg.min_chars <= len(text.strip()) <= cfg.max_chars or len(text.split()) < cfg.min_words:
        reasons.add(LENGTH)
    if _markup(text, cfg):
        reasons.add(MARKUP)
    if _fragment(text, cfg):
        reasons.add(FRAGMENT)
    return GateResult(not reasons, tuple(sorted(reasons)))
