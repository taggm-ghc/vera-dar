"""FDA policy helpers (PLAN-FDA POL, D7). Pure functions, stdlib only, no I/O.

require_supported_claim(records) -> bool
    D7: an answer is published only if at least one claim record is `supported`; partial-only answers fall back.
pointer_only(span_text, cfg) -> bool
    True when the span's predicate content is only a cross-reference ("see Box/Table/Fig. N", "as described in ...")
    or a reference list. Thresholds come from cfg (a `pointer_filter` sub-dict, or the dict itself) with safe defaults.
    UNCALIBRATED: the defaults below are priors; a later CAL WP calibrates them on calset-clean.
"""
from __future__ import annotations

import re

DEFAULTS = {
    "min_content_tokens": 6,      # UNCALIBRATED prior: content words left after removing pointers => not pointer-only
    "min_token_len": 3,
    "ref_list_min_entries": 3,    # UNCALIBRATED prior: citation-like lines needed to call a span a reference list
    "ref_list_min_fraction": 0.6,  # UNCALIBRATED prior: share of non-empty lines that look like references
}

_TARGET = (r"(?:box(?:es)?|tables?|tab\.|fig(?:ure)?s?\.?|section|sec\.|appendix|appendices|supplement\w*|"
           r"supporting information|s\d+\s+(?:text|table|fig\w*)|ref(?:erence)?s?\.?|\[\d+(?:\s*[,–-]\s*\d+)*\])")
_POINTER = re.compile(
    r"\b(?:(?:see|cf\.?|compare|refer\s+to|consult)(?:\s+also)?|(?:as\s+)?(?:described|shown|reported|detailed|presented|"
    r"discussed|illustrated|listed|summari[sz]ed|outlined|given|provided|explained)\s+(?:in|by)|in)\s+"
    + _TARGET + r"(?:\s*[\w.\-]*\d[\w.\-]*)?(?:\s*(?:,|and|&|or)\s*" + _TARGET + r"(?:\s*[\w.\-]*\d[\w.\-]*)?)*",
    re.IGNORECASE)
_BARE_REF = re.compile(r"\[\d+(?:\s*[,–-]\s*\d+)*\]|\((?:fig(?:ure)?s?\.?|tables?|box)\s*[\w.\-]+\)", re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_]+")
_STOP = frozenset("the and for are was were with that this these those from have has had not but can will would could should "
                  "may which their there than then into also such been being its our your they them you who whom what when "
                  "where how why all any each more most other some only over under between about after before both during "
                  "through use used using see".split())
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_REF_LINE = re.compile(r"(?:\bet al\b|\bdoi\b|10\.\d{4,}/|\b\d+\s*[:(]\s*\d+|\bpp?\.\s*\d+|\b\d+\s*\(\d+\)\s*[:,]?\s*\d*)",
                       re.IGNORECASE)


def _num(cfg, key):
    sub = cfg.get("pointer_filter") if isinstance(cfg, dict) and isinstance(cfg.get("pointer_filter"), dict) else cfg
    v = (sub or {}).get(key) if isinstance(sub, dict) else None
    try:
        return type(DEFAULTS[key])(v) if v is not None else DEFAULTS[key]
    except (TypeError, ValueError):
        return DEFAULTS[key]


def require_supported_claim(records) -> bool:
    """D7: True only if at least one record has verdict == 'supported' (partial / unsupported never suffice)."""
    return any(isinstance(r, dict) and r.get("verdict") == "supported" for r in (records or ()))


def _is_reference_list(text: str, cfg) -> bool:
    lines = [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()]
    if len(lines) < int(_num(cfg, "ref_list_min_entries")):
        return False
    refs = sum(1 for ln in lines if _YEAR.search(ln) and (_REF_LINE.search(ln) or re.match(r"^\[?\d+[\].)]", ln)))
    return refs >= int(_num(cfg, "ref_list_min_entries")) and refs / len(lines) >= float(_num(cfg, "ref_list_min_fraction"))


def pointer_only(span_text: str, cfg: dict) -> bool:
    """True when the span carries no predicate content beyond cross-references / a reference list."""
    text = span_text if isinstance(span_text, str) else ""
    cfg = cfg if isinstance(cfg, dict) else {}
    if not text.strip():
        return False
    if _is_reference_list(text, cfg):
        return True
    stripped, n = _POINTER.subn(" ", text)
    if n == 0 and not _BARE_REF.search(text):
        return False
    stripped = _BARE_REF.sub(" ", stripped)
    min_len = int(_num(cfg, "min_token_len"))
    content = [w for w in _WORD.findall(stripped.lower()) if len(w) >= min_len and w not in _STOP]
    return len(content) < int(_num(cfg, "min_content_tokens"))
