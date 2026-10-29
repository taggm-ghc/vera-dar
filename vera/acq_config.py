"""WED-ACQ2 / P1.7: shared configuration and pure helpers for automatic corpus expansion.

Interface module shared by the acquisition job (vera/auto_acquire.py) and the reader/serving side (vera/fulldoc_db.py,
vera/fulldoc_corpus.py, vera/ask_service.py, vera/disclosure.py). Pure: no network, no DB, no model calls. Env var
NAMES come from config/auto_acquire.json; values are only tested for presence here and never logged or returned.

Kill switch (plan s4.5, AC-12): the feature is ON only when `enabled` is exactly true in the config, the kill-switch
env var is not true-like, and both DB env vars are set. Absent or unreadable config = OFF (code default OFF).
"""
from __future__ import annotations

import copy
import json
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "auto_acquire.json"
_TRUE = ("true", "1", "yes", "on")


class ConfigError(RuntimeError):
    """The tracked acquisition config holds a key-like value (a secret must live in the environment). The message names only dotted
    config PATHS, never values."""


def load_config(path: Path = CONFIG_PATH) -> dict | None:
    """The parsed config, or None when absent/unreadable/not an object (=> feature OFF). Raises ConfigError (after logging the
    offending dotted paths, never values, at ERROR) when the config holds a key-like value (ACQ S4-0 L1/L4c): startup fails
    loud instead of silently switching the feature off. Request-path callers catch ConfigError and treat it as OFF."""
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    bad = refuse_key_like_values(cfg)
    if bad:
        msg = "acquisition config refused: key-like value at path(s) " + ", ".join(bad)
        logging.getLogger("vera").error(msg)
        raise ConfigError(msg)
    return cfg


def refuse_key_like_values(cfg, _depth: int = 0) -> list:
    """Paths (never values) of string values in `cfg` that look like a secret (vera.acq_urls.key_like_value). Keys
    belong in the process environment; the tracked config holds env var NAMES only. Empty list = clean."""
    from vera import acq_urls
    out: list = []

    def walk(node, path):
        if len(path) > 12:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + [str(k)])
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, path + [str(i)])
        elif isinstance(node, str) and acq_urls.key_like_value(node, ".".join(path)):
            out.append(".".join(path))
    walk(cfg, [])
    return out


def _true_like(v) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() in _TRUE)


def flag_on(cfg: dict | None) -> bool:
    """The config flag alone (used by the disclosure, which must describe what the release config allows)."""
    return isinstance(cfg, dict) and cfg.get("enabled") is True


def killed(cfg: dict | None, env) -> bool:
    name = str((cfg or {}).get("kill_switch_env") or "VERA_AUTOACQ_DISABLED")
    return _true_like(env.get(name))


def db_configured(cfg: dict | None, env) -> bool:
    c = cfg or {}
    return all((env.get(str(c.get(k) or "")) or "").strip() for k in ("db_url_env_wo", "db_url_env_ro"))


def enabled(cfg: dict | None, env) -> bool:
    """Effective switch: flag true AND not killed AND both DB env vars set."""
    return flag_on(cfg) and not killed(cfg, env) and db_configured(cfg, env)


def serving_enabled(cfg: dict | None, env) -> bool:
    """Reader side: acquired documents are served when the flag is on and the switch is not killed (RO env is checked
    by the reader's own connection factory)."""
    return flag_on(cfg) and not killed(cfg, env)


def bound(cfg: dict, key: str, default=None):
    v = ((cfg or {}).get("bounds") or {}).get(key, default)
    if v is None:
        raise KeyError(f"auto_acquire bound {key!r} missing")
    return v


def inline_fresh_docs(cfg: dict) -> int:
    """ACQ S4-0 L4b-0 (B3): fresh-document budget of one inline pass, separate from the reuse budget. Key
    bounds.inline_fresh_docs; absent = max_docs_per_job (the current cap). Anything not a non-negative int fails
    closed to 0 (a malformed cap acquires nothing)."""
    b = (cfg or {}).get("bounds") or {}
    v = b.get("inline_fresh_docs", b.get("max_docs_per_job"))
    if v is None:
        raise KeyError("auto_acquire bound 'max_docs_per_job' missing")
    try:
        return max(0, int(v)) if not isinstance(v, bool) else 0
    except (TypeError, ValueError):
        return 0


def docs_per_day_cap(cfg: dict):
    """OWNER DECISION 2026-10-22 ("abolish the 20-a-day limit", relayed by the orchestrator, DAR-TRACE-1022):
    bounds.docs_per_day null or absent = NO daily document cap (the job caps jobs_per_hour / jobs_per_day and
    max_docs_per_job still bound acquisition). A number (int >= 0) restores the cap; anything else is treated as 0
    (fail closed: a malformed cap acquires nothing)."""
    v = ((cfg or {}).get("bounds") or {}).get("docs_per_day")
    if v is None:
        return None
    try:
        return max(0, int(v)) if not isinstance(v, bool) else 0
    except (TypeError, ValueError):
        return 0


def answer_min_spans(cfg: dict) -> int:
    """ONE-SPAN-1021 (owner approval): qualifying, provenance-verified passages a fetched document needs to supply
    evidence to THIS request's ephemeral answer corpus. Floor 1: a document with no qualifying passage never answers."""
    b = (cfg or {}).get("bounds") or {}
    return max(1, int(b.get("answer_min_admitted_spans", persist_min_spans(cfg))))


def persist_min_spans(cfg: dict) -> int:
    """ONE-SPAN-1021: qualifying passages a document needs to be written to the persistent corpus. Floor 2 (owner:
    persistent ingestion keeps a minimum of TWO qualifying passages per document); legacy key min_admitted_spans read
    when the new key is absent."""
    b = (cfg or {}).get("bounds") or {}
    v = b.get("persist_min_admitted_spans", b.get("min_admitted_spans"))
    if v is None:
        raise KeyError("auto_acquire bound 'persist_min_admitted_spans' missing")
    return max(2, int(v))


def reason(cfg: dict, key: str) -> str:
    return str(((cfg or {}).get("reason_codes") or {}).get(key) or f"R-ACQ-{key.upper()}")


def marker(cfg: dict) -> str:
    return str(cfg["marker"])


# ------------------------------------------------------------------------------------------------ triggers
def trigger_for_decline(cfg: dict | None, layer: str, why: str) -> bool:
    """True only for an off-domain decline. deny:* (incl. deny:injection), empty and classifier_malformed NEVER trigger
    (AC-1, FM-7, FM-12). A trigger means acquisition only; answering a declined question is the separate
    override_layer2_declines switch (ships off)."""
    if not isinstance(cfg, dict):
        return False
    t = cfg.get("triggers") or {}
    r = str(why or "")
    if any(r.startswith(p) for p in (t.get("never_on_reasons_prefix") or ["deny:", "empty"])):
        return False
    if r.startswith("deny") or r == "empty" or r.startswith("classifier_malformed"):  # belt and braces: config
        return False  # cannot re-enable these (plan s11 V1: a malformed classifier reply is typical of injection)
    return any(r.startswith(p) for p in (t.get("on_decline_reasons_prefix") or ()))


# ------------------------------------------------------------------------------------------------ ingest overlay
def ingest_cfg(base: dict, cfg: dict) -> dict:
    """fulltext_ingest config with the acquisition label overlay (plan AC-5): labels plus the non-DOI identity
    allowlist (PLAN-NONDOI-1022); every filter key and limit stays the base value. Deep copy; the base dict is never
    mutated."""
    out = copy.deepcopy(base)
    ov = (cfg or {}).get("ingest_overlay") or {}
    for k in ("verified_status", "admitted_by", "question_text", "licence_observations_enabled",
              "embedded_licence_unknown_holds", "non_doi_identifier_kinds"):
        if k in ov:
            out[k] = ov[k]
    if isinstance(ov.get("candidate"), dict):
        out["candidate"] = {**(out.get("candidate") or {}), **{k: v for k, v in ov["candidate"].items()}}
    return out


# ------------------------------------------------------------------------------------------------ injection scan
@lru_cache(maxsize=8)
def _compiled(patterns: tuple) -> tuple:
    return tuple(re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns)


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")  # DAR-1021 SEC L2: zero-width/format chars
    return re.sub(r"\s+", " ", t.lower())


def injection_hits(text: str, cfg: dict) -> int:
    """Number of configured instruction-like patterns found in the text (counts only; matches are never returned)."""
    sc = (cfg or {}).get("injection_scan") or {}
    pats = _compiled(tuple(str(p) for p in (sc.get("patterns") or ())))
    t = _norm(text)
    return sum(1 for p in pats if p.search(t))


def injection_refused(text: str, cfg: dict) -> bool:
    return injection_hits(text, cfg) > int(((cfg or {}).get("injection_scan") or {}).get("max_hits", 0))


# ------------------------------------------------------------------------------------------------ ISBN (owner 2026-10-21)
def _digits(raw) -> str:
    return re.sub(r"[\s\-]", "", str(raw or "")).upper()


def _isbn13_ok(s: str) -> bool:
    if not re.fullmatch(r"97[89]\d{10}", s):
        return False
    total = sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(s[:12]))
    return (10 - total % 10) % 10 == int(s[12])


def _isbn10_ok(s: str) -> bool:
    if not re.fullmatch(r"\d{9}[\dX]", s):
        return False
    total = sum((10 - i) * (10 if c == "X" else int(c)) for i, c in enumerate(s))
    return total % 11 == 0


def isbn10_to_13(s: str) -> str:
    core = "978" + s[:9]
    total = sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(core))
    return core + str((10 - total % 10) % 10)


def normalise_isbn(raw) -> dict | None:
    """{"isbn": <ISBN-13, no hyphens>} or {"isbn": ..., "isbn10": <source ISBN-10>}; None when absent or invalid
    (check digit verified; never guessed)."""
    s = _digits(raw)
    if _isbn13_ok(s):
        return {"isbn": s}
    if _isbn10_ok(s):
        return {"isbn": isbn10_to_13(s), "isbn10": s}
    return None


def first_isbn(values) -> dict | None:
    """The first valid ISBN among provider values (str or list), in order; None when none is valid."""
    if values is None:
        return None
    if isinstance(values, (str, int)):
        values = [values]
    for v in values:
        n = normalise_isbn(v)
        if n:
            return n
    return None


def override_layer2_declines(cfg: dict | None) -> bool:
    """plan s11 V1: answering a declined question from acquired documents; exactly true only (ships false)."""
    se = (cfg or {}).get("scope_expansion") or {}
    return isinstance(cfg, dict) and se.get("enabled") is True and se.get("override_layer2_declines") is True


def _refuse_categories() -> frozenset:
    from vera.scope_router import REFUSE_CATEGORIES
    return REFUSE_CATEGORIES


def layer2_category_ok(cfg: dict | None, decision) -> bool:
    """DAR-1021: a decline may acquire (and, with the override, answer) only for a well-formed layer-2 LLM verdict whose
    closed-enum category is in scope_expansion.layer2_categories_allowed. Missing/invalid config or category = False."""
    se = (cfg or {}).get("scope_expansion") or {} if isinstance(cfg, dict) else {}
    allowed = se.get("layer2_categories_allowed")
    if not isinstance(allowed, list) or not all(isinstance(a, str) for a in allowed):
        return False
    cat = getattr(decision, "category", "")
    return getattr(decision, "layer", "") == "layer2_llm" and isinstance(cat, str) and bool(cat) and cat in allowed \
        and cat not in _refuse_categories()  # CMF-1022 R2: refusal categories never pass, even if listed in config


def override_answer_prefix(cfg: dict | None) -> str:
    """DAR-1021 SEC M1: fixed text placed before an answer to a declined question (from acquired documents only)."""
    se = (cfg or {}).get("scope_expansion") or {} if isinstance(cfg, dict) else {}
    t = se.get("override_answer_prefix")
    return t if isinstance(t, str) else ""


def reading_list_notice(cfg: dict | None) -> str:
    """DAR-1021: the sentence appended to the off-topic reading list when acquisition was started for the question."""
    se = (cfg or {}).get("scope_expansion") or {} if isinstance(cfg, dict) else {}
    t = se.get("reading_list_notice_answering" if override_layer2_declines(cfg) else "reading_list_notice_listing")
    return t if isinstance(t, str) else ""


def licence_url_for(declared, cfg: dict) -> str | None:
    """Version-less CC family URL for a declared Europe PMC licence string; None (= hold) when unmapped."""
    m = ((cfg or {}).get("resolve") or {}).get("licence_url_map") or {}
    key = re.sub(r"\s+", " ", str(declared or "").strip().lower())
    v = m.get(key) if key and not key.startswith("_") else None
    return v if isinstance(v, str) and v.startswith("https://") else None
