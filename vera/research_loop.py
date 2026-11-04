"""NONASK-1103 L4: bounded supplemental research on Gate C "research further". FLAG-OFF (config/research_loop.json).

NOT wired into ask_service, fulldoc_answer or streamlit_app; wiring follows calibration (evidence-policy plan L4).
Owner-flip-packet option (a): there is NO separate round count. The loop ends on
  deadline        the existing inline deadline (config/auto_acquire.json inline.deadline_s, 45 s);
  no_new_passage  a round adds no admitted passage not already held;
  repeated_query  a round would repeat a query key (auto_acquire._qkey over the extracted terms).
It also ends on the gate's own dispositions (finalize, qualify, abstain), when no search terms can be extracted
(no_terms; the raw question is never a fallback), and on a search error (search_error). Every exit records
`exit_reason`. With the flag off nothing is called and the passages come back unchanged (I4).

Only extracted terms (vera.search_query.build_query_terms core + extra) reach the injected search callable, never
the user's sentence. The search callable must return ADMITTED passages only (every existing licence, injection,
relevance and per-prompt document bound applies inside it); this module adds no admission rule. Logs codes and counts
only, never the question, terms or passage text.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Callable

from vera import auto_acquire as aa
from vera import search_query

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parent.parent / "config"
CONFIG_PATH = _ROOT / "research_loop.json"
ACQ_PATH = _ROOT / "auto_acquire.json"
RESEARCH_FURTHER = "research_further"
TERMINAL = ("finalize", "qualify", "abstain")


def load_cfg(path: Path | None = None) -> dict:
    try:
        cfg = json.loads((path or CONFIG_PATH).read_text(encoding="utf-8"))
        return cfg if isinstance(cfg, dict) else {"enabled": False}
    except Exception:  # noqa: BLE001  unreadable config = off
        return {"enabled": False}


def enabled(cfg: dict | None = None) -> bool:
    return (cfg if cfg is not None else load_cfg()).get("enabled") is True


def inline_deadline_s(path: Path | None = None) -> float:
    """The existing inline.deadline_s (no new number)."""
    return float(json.loads((path or ACQ_PATH).read_text(encoding="utf-8"))["inline"]["deadline_s"])


def _verdict(rec) -> str:
    return str((rec or {}).get("verdict", "")) if isinstance(rec, dict) else str(rec or "")


def _need_text(question: str, gate_rec) -> str:
    """The missing evidence need: labels of unsupported declared parts, else the question (terms are extracted from it)."""
    parts = (gate_rec or {}).get("parts") if isinstance(gate_rec, dict) else None
    labels = [str(p.get("label") or "") for p in parts or () if isinstance(p, dict) and not p.get("supported")]
    labels = [x for x in labels if x.strip()]
    return " ; ".join(labels) if labels else str(question or "")


def extract_terms(text: str, terms_cfg) -> tuple:
    """Core phrases plus extra terms; () when nothing but the raw text would remain (mode fallback_raw)."""
    if not terms_cfg:
        return ()
    qt = search_query.build_query_terms(str(text or ""), terms_cfg)
    if qt.empty or qt.mode == "fallback_raw":
        return ()
    return tuple(dict.fromkeys(qt.core + qt.extra))


def _pid(p) -> str:
    if isinstance(p, dict) and p.get("id") is not None:
        return "id:" + str(p["id"])
    body = p.get("text", "") if isinstance(p, dict) else p
    return "h:" + hashlib.sha256(str(body).encode("utf-8")).hexdigest()


def run(question: str, gate_rec, passages: list, *, search: Callable | None = None, assess: Callable | None = None,
        cfg: dict | None = None, terms_cfg=None, deadline_s: float | None = None, clock=time.monotonic) -> tuple:
    """(passages, record).

    gate_rec  the Gate C record (or verdict string); the loop runs only on verdict "research_further".
    passages  the admitted passages held so far; new ones are appended, never replaced.
    search    callable(terms: tuple[str, ...], remaining_s: float) -> list of admitted passages.
    assess    callable(passages) -> the next Gate C record (or verdict string).
    """
    cfg = cfg if cfg is not None else load_cfg()
    if not enabled(cfg):
        return passages, {"enabled": False, "exit_reason": "disabled", "rounds": 0}
    rec = {"enabled": True, "exit_reason": None, "rounds": 0, "searches": 0, "new_passages": 0}

    def done(reason, held):
        rec["exit_reason"] = reason
        logger.info("research_loop exit=%s rounds=%d searches=%d new=%d", reason, rec["rounds"], rec["searches"],
                    rec["new_passages"])
        return held, rec

    held = list(passages or ())
    if _verdict(gate_rec) != RESEARCH_FURTHER:
        return done("not_triggered", held)
    if search is None or assess is None:
        return done("no_dependency", held)
    if terms_cfg is None:
        terms_cfg = search_query._default_terms_cfg()
    limit = float(deadline_s if deadline_s is not None else inline_deadline_s())
    end = clock() + limit
    seen_ids = {_pid(p) for p in held}
    seen_q: set = set()
    current = gate_rec
    while True:
        v = _verdict(current)
        if v in TERMINAL:
            return done(v, held)
        if v != RESEARCH_FURTHER:
            return done("unknown_verdict", held)
        remaining = end - clock()
        if remaining <= 0:
            return done("deadline", held)
        terms = extract_terms(_need_text(question, current), terms_cfg)
        if not terms:
            return done("no_terms", held)
        key = aa._qkey(" ".join(sorted(terms)))
        if key in seen_q:
            return done("repeated_query", held)
        seen_q.add(key)
        rec["rounds"] += 1
        rec["searches"] += 1
        try:
            found = search(terms, remaining) or []
        except Exception:  # noqa: BLE001
            return done("search_error", held)
        new = []
        for p in found:
            k = _pid(p)
            if k not in seen_ids:
                seen_ids.add(k)
                new.append(p)
        if not new:
            return done("no_new_passage", held)
        held.extend(new)
        rec["new_passages"] += len(new)
        if end - clock() <= 0:
            return done("deadline", held)
        current = assess(held)
