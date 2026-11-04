"""NONASK-1103 S7: deterministic current-state (temporality) detector.

Pure Python: no DB, network or model. The lexicon, negations and patterns come from config/temporality.json
(curated lists, not tuned numbers; see its _calibration_note). A model fallback (S7b) is NOT implemented here.

detect(text, volatile_entity=False) -> TemporalResult(current, cues). An explicit year counts only together with a
lexicon cue. A volatile registry entity (passed by the caller) also triggers detection.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "temporality.json"
_cache: dict | None = None


@dataclass(frozen=True)
class TemporalResult:
    current: bool
    cues: tuple = field(default_factory=tuple)


def load_config(path: Path | None = None) -> dict:
    global _cache
    if path is None and _cache is not None:
        return _cache
    p = path or _CFG_PATH
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {"lexicon": [], "negations": [], "named_recent_events": [], "volatile_entity_triggers": True,
               "year_requires_lexicon_cue": True, "year_pattern": r"\b(?:19|20)\d{2}\b"}
    if path is None:
        _cache = cfg
    return cfg


def _phrase_re(phrase: str) -> re.Pattern:
    words = [re.escape(w) for w in re.split(r"[\s]+", phrase.strip()) if w]
    return re.compile(r"(?<![\w-])" + r"\s+".join(words) + r"(?![\w-])", re.I)


def detect(text: str, volatile_entity: bool = False, cfg: dict | None = None) -> TemporalResult:
    cfg = cfg if cfg is not None else load_config()
    t = text or ""
    masked = t
    for neg in cfg.get("negations", []):
        masked = _phrase_re(neg).sub(" ", masked)
    cues: list[str] = []
    for term in cfg.get("lexicon", []):
        if _phrase_re(term).search(masked):
            cues.append(term)
    for ev in cfg.get("named_recent_events", []):
        if ev and _phrase_re(ev).search(t):
            cues.append("event:" + ev)
    year = cfg.get("year_pattern")
    if year and re.search(year, t) and (cues or not cfg.get("year_requires_lexicon_cue", True)):
        cues.append("year")
    if volatile_entity and cfg.get("volatile_entity_triggers", True):
        cues.append("volatile_entity")
    return TemporalResult(current=bool(cues), cues=tuple(cues))
