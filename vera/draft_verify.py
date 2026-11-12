"""DRAFTVERIFY-IMPL-1111 (plan work/DRAFT-VERIFY-1111): config, request state, leak guards and the /ask hooks for the
draft-first stages. Every stage ships OFF (config/draft_verify.json); with all stages off `any_on()` is False and no
other function here is reached, so /ask is byte-identical.

Hard rules kept here (plan 2.1 to 2.6):
  * the draft is the model's prior, never evidence: it lives only in the DRAFT ContextVar, is never passed to the writer,
    and `leaks()` is the zero-leak check (5-gram, draft-only numbers, draft-only named entities);
  * new stages are optional and Groq-only: `skip_reason()` returns a code (exhausted, grace, budget_short) and the caller
    continues on today's pipeline; nothing here raises into /ask (`prepare` and `after_answer` swallow everything);
  * traces carry codes and counts only, never draft or claim text.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from vera import request_trace as rt

log = logging.getLogger("vera")
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "draft_verify.json"
MODES = ("off", "shadow", "on")
STAGES = ("draft_query", "stance", "gap")
DEFAULTS = {"stages": {s: "off" for s in STAGES}, "k_chunks": 3, "min_contra_docs": 1, "num_tolerance": 0.10,
            "max_gap_rounds": 2, "max_triggered_claims": 4, "max_queries_per_claim": 2, "max_gap_queries": 16,
            "max_restance": 2, "max_claims": 8, "draft_model": "openai/gpt-oss-20b", "drafter_family": "gpt-oss",
            "draft_timeout_s": 20, "stance_timeout_s": 20, "draft_max_chars": 1200, "reserve_s": 60,
            "disconfirm_terms": ("criticism", "limitations")}
# (lo, hi) per numeric key; a value outside, a bool or a non-number REFUSES the whole file (not clamped, plan Q11)
_BOUNDS = {"k_chunks": (1, 50), "min_contra_docs": (1, 50), "num_tolerance": (0.0, 10.0), "max_gap_rounds": (0, 5),
           "max_triggered_claims": (0, 20), "max_queries_per_claim": (0, 5), "max_gap_queries": (0, 64),
           "max_restance": (0, 5), "max_claims": (1, 30), "draft_timeout_s": (1, 120), "stance_timeout_s": (1, 120),
           "draft_max_chars": (100, 10000), "reserve_s": (0, 170)}

_STATE: dict = {"cfg": None}

DRAFT: ContextVar = ContextVar("dv_draft", default=None)          # Draft (text + claims); never leaves this request
QUERY: ContextVar = ContextVar("dv_second_query", default=None)   # terms-only second retrieval query (mode on)
GAP_QUERIES: ContextVar = ContextVar("dv_gap_queries", default=None)  # ordered gap queries for _extra_rounds (mode on)
STANCE: ContextVar = ContextVar("dv_stance", default=None)        # last aggregate result (codes/counts, for tests)


def _clean(raw) -> dict:
    """Validated config. Any invalid value refuses the whole file: all stages off (never clamped)."""
    c = json.loads(json.dumps(DEFAULTS))
    c["disconfirm_terms"] = tuple(DEFAULTS["disconfirm_terms"])
    if not isinstance(raw, dict):
        return c
    st = raw.get("stages", {})
    if not isinstance(st, dict) or any(k not in STAGES or v not in MODES for k, v in st.items()):
        return _refused(c)
    for k, (lo, hi) in _BOUNDS.items():
        if k in raw:
            v = raw[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
                return _refused(c)
            c[k] = v
    for k in ("draft_model", "drafter_family"):
        if k in raw:
            if not isinstance(raw[k], str) or not raw[k].strip():
                return _refused(c)
            c[k] = raw[k].strip()
    if "disconfirm_terms" in raw:
        d = raw["disconfirm_terms"]
        if not isinstance(d, list) or not all(isinstance(x, str) and x.strip() for x in d) or len(d) > 4:
            return _refused(c)
        c["disconfirm_terms"] = tuple(x.strip() for x in d)
    c["stages"] = {**{s: "off" for s in STAGES}, **st}
    return c


def _refused(c: dict) -> dict:
    c = dict(c)
    c["stages"] = {s: "off" for s in STAGES}
    c["refused"] = True
    return c


def config() -> dict:
    """Cached validated config. Missing or bad file => every stage off. Never raises."""
    c = _STATE["cfg"]
    if c is None:
        try:
            c = _clean(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            c = _refused(_clean({}))
        _STATE["cfg"] = c
    return c


def set_config_for_tests(raw: dict | None) -> None:
    _STATE["cfg"] = None if raw is None else _clean(raw)


def stage_mode(stage: str) -> str:
    try:
        return config()["stages"].get(stage, "off")
    except Exception:  # noqa: BLE001
        return "off"


def any_on() -> bool:
    return any(stage_mode(s) != "off" for s in STAGES)


def reset() -> None:
    for v in (DRAFT, QUERY, GAP_QUERIES, STANCE):
        v.set(None)


def second_query() -> str | None:
    """The terms-only second retrieval query, only while draft_query is exactly on. None otherwise."""
    return QUERY.get() if stage_mode("draft_query") == "on" else None


def gap_queries() -> list:
    """Ordered gap queries for the acquisition rounds, only while gap is exactly on. [] otherwise."""
    g = GAP_QUERIES.get() if stage_mode("gap") == "on" else None
    return list(g) if g else []


@dataclass(frozen=True)
class Draft:
    text: str
    claims: tuple  # atomic claim strings (internal only)


# ------------------------------------------------------------------------------------------------ plan 2.6 fallback
def skip_reason(remaining_s: float | None, need_s: float, cfg: dict | None = None) -> str | None:
    """Why an optional stage must be skipped (the ask then continues on today's pipeline), else None:
    'grace' (an exhaustion grace completion is running: new stages never run there), 'exhausted' (every provider is in
    its stop window), 'budget_short' (remaining request time minus the writer/checker reserve cannot cover the call).
    remaining_s None or infinite (streamed path: no deadline) never skips for time."""
    cfg = cfg or config()
    try:
        from vera import provider_exhaustion as px
        if px.GRACE.get() is not None:
            return "grace"
        if px.exhausted():
            return "exhausted"
    except Exception:  # noqa: BLE001  cannot tell: do not run an optional stage
        return "exhausted"
    if remaining_s is not None and math.isfinite(remaining_s) and remaining_s - float(cfg["reserve_s"]) < float(need_s):
        return "budget_short"
    return None


def remaining_s(budget) -> float | None:
    try:
        return float(budget.deadline) - time.monotonic()
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------ leak guards
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _words(t: str) -> list:
    return [w.lower() for w in _WORD.findall(t or "")]


def leaks(outputs, draft_texts, allowed_texts, ngram: int = 5) -> dict:
    """Zero-leak check (plan 4): what in `outputs` (published text, panel, trace, grace payload) occurs ONLY in the draft.
    Returns {'ngram': [...], 'numbers': [...], 'entities': [...]}; all empty = no leak. A token or n-gram counts as
    draft-only when it occurs in a draft text and in no `allowed` text (question and admitted chunks)."""
    out_t, dr_t, al_t = (" \n ".join(str(x) for x in v) for v in (outputs, draft_texts, allowed_texts))
    ow, dw, aw = _words(out_t), _words(dr_t), _words(al_t)
    grams = lambda ws: {" ".join(ws[i:i + ngram]) for i in range(len(ws) - ngram + 1)}  # noqa: E731
    g_out = grams(ow)
    res = {"ngram": sorted((g_out & grams(dw)) - grams(aw)), "numbers": [], "entities": []}
    a_nums = {n.replace(",", "") for n in _NUM.findall(al_t)}
    d_nums = {n.replace(",", "") for n in _NUM.findall(dr_t)}
    res["numbers"] = sorted({n.replace(",", "") for n in _NUM.findall(out_t)} & (d_nums - a_nums))
    a_set = set(aw)
    d_ents = {m.lower() for m in re.findall(r"(?<=[a-z,;] )[A-Z][a-zA-Z]{2,}", dr_t)} - a_set
    res["entities"] = sorted(d_ents & set(ow))
    return res


def leak_free(outputs, draft_texts, allowed_texts) -> bool:
    return not any(leaks(outputs, draft_texts, allowed_texts).values())


def aspect(claim_text: str, question: str, chunk_texts) -> str | None:
    """Neutral silence aspect (plan 2.3): built ONLY from claim terms that also occur in the question or an admitted
    chunk; a term that occurs only in the draft is never used. None when no such term exists (silence is then counted
    in the trace and not shown)."""
    allowed = set(_words(question)) | {w for t in chunk_texts for w in _words(t)}
    seen, terms = set(), []
    stop = _stop()
    for w in _words(claim_text):
        if w in allowed and w not in stop and w not in seen and len(w) > 2 and not w.isdigit():
            seen.add(w)
            terms.append(w)
    return " ".join(terms[:4]) or None


def silence_text(asp: str) -> str:
    return f"the sources found do not address {asp}"


def _stop() -> frozenset:
    try:
        from vera import search_query
        cfg = search_query._default_terms_cfg()
        return frozenset(cfg["stopwords"]) | frozenset(cfg["meta_words"]) if cfg else frozenset()
    except Exception:  # noqa: BLE001
        return frozenset()


def terms_query(text: str, max_terms: int = 8) -> str:
    """Terms-only query from free text (PII filtered, stopwords dropped; never the raw sentence). '' if no terms."""
    try:
        from vera import query_variants_acq as qv
        from vera import search_query
        cfg = search_query._default_terms_cfg()
        if not cfg:
            return ""
        qt = search_query.build_query_terms(qv.scrub(text), cfg)
        if qt.empty or qt.mode == "fallback_raw":
            return ""
        terms = list(dict.fromkeys([*qt.core, *search_query.and_terms(qt, max_terms), *qt.extra]))[:max_terms]
        return " ".join(terms)
    except Exception:  # noqa: BLE001
        return ""


# ------------------------------------------------------------------------------------------------ /ask hooks
def prepare(question: str, budget=None, call=None) -> None:
    """Before _fulldoc (after an accepted decision): stage draft_query. Never raises; on any skip the ask is unchanged."""
    try:
        reset()
        mode = stage_mode("draft_query")
        if mode == "off" and stage_mode("stance") == "off" and stage_mode("gap") == "off":
            return
        cfg = config()
        why = skip_reason(remaining_s(budget), cfg["draft_timeout_s"], cfg)
        if why:
            rt.event("draft_verify", "skip_" + why)
            return
        from vera import draft_hypothesis
        d = draft_hypothesis.make(question, cfg, call=call)
        if d is None:
            rt.event("draft_verify", "draft_unavailable")
            return
        DRAFT.set(d)
        q = terms_query(d.text) if mode != "off" else ""
        rt.event("draft_verify", "draft_made", counts={"claims": len(d.claims), "query_terms": len(q.split())})
        if mode == "on" and q:
            QUERY.set(q)
    except Exception as exc:  # noqa: BLE001  optional stage: never fails /ask
        log.info("draft_verify prepare skipped (%s)", type(exc).__name__)
        reset()


def after_answer(question: str, ev, budget=None, call=None) -> None:
    """After an answered _fulldoc: stances (shadow or on) and the gap-trigger plan. Records codes and counts only; the
    answer is never changed in this wave. Never raises."""
    try:
        d = DRAFT.get()
        if d is None or (stage_mode("stance") == "off" and stage_mode("gap") == "off") or ev is None:
            return
        cfg = config()
        chunks = [a for a in (getattr(ev, "admitted", ()) or ()) if isinstance(a, dict) and a.get("text")]
        if not chunks or not d.claims:
            rt.event("draft_verify", "stance_no_input")
            return
        why = skip_reason(remaining_s(budget), cfg["stance_timeout_s"], cfg)
        if why:
            rt.event("draft_verify", "stance_skip_" + why)
            return
        from vera import claim_stance, gap_queries as gq
        res = claim_stance.run(d.claims, chunks, cfg, call=call)
        if res is None:
            rt.event("draft_verify", "stance_unavailable")
            return
        STANCE.set(res)
        counts = {k: sum(1 for a in res["aggregates"].values() if a["status"] == k) for k in claim_stance.STATUSES}
        rt.event("draft_verify", "stance", counts={**counts, "same_family": int(bool(res["same_family"]))})
        trig = claim_stance.triggers(res["aggregates"], d.claims, cfg, chunks)
        if stage_mode("gap") != "off":
            qs = gq.build(trig, d.claims, chunks, question, cfg)
            rt.event("draft_verify", "gap_plan", counts={"triggered": len(trig), "queries": len(qs)})
            if stage_mode("gap") == "on" and not res["same_family"]:
                GAP_QUERIES.set(qs)
    except Exception as exc:  # noqa: BLE001
        log.info("draft_verify stance skipped (%s)", type(exc).__name__)
