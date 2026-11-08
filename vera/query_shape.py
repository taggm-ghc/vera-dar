"""QUERY-SHAPE-1108: question-shape decisions for the acquisition discovery query (config/query_shape.json).

shape(question) -> {"recency": bool, "review": bool}. A definitional question ("what are quarks?", "explain X") is
detected from grammar frames in config (no keyword lists in code) and gets no OpenAlex date filter and one
review-filtered call. A question with a current-state cue (vera.temporality) or news intent (vera.news_intent) always
keeps the recency filter. Pure; never raises (the default shape keeps the existing behaviour)."""
from __future__ import annotations

import json
import re
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "query_shape.json"
DEFAULT = {"recency": True, "review": False}


def load_config(path: Path = CONFIG_PATH) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def is_definitional(question: str, cfg: dict | None = None) -> bool:
    cfg = load_config() if cfg is None else cfg
    q = " ".join(str(question or "").split())
    return any(re.search(p, q, re.I) for p in cfg.get("definitional_patterns") or ())


def shape(question: str, cfg: dict | None = None) -> dict:
    try:
        cfg = load_config() if cfg is None else cfg
        if not is_definitional(question, cfg):
            return dict(DEFAULT)
        from vera import news_intent, temporality
        if temporality.detect(question).current or news_intent.news_intent(question):
            return dict(DEFAULT)
        return {"recency": False, "review": bool(cfg.get("review_filter"))}
    except Exception:  # noqa: BLE001
        return dict(DEFAULT)
