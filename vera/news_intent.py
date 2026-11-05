"""NEWS-LANE-1104: deterministic news / current-events intent detector for the current-reporting lane.

Pure Python: no DB, network or model. The cue lists come from config/reporting_lane.json `news_intent` (curated word
lists, see its _calibration_note). A question has news intent when it carries a news cue ("news", "newsworthy",
"headlines", ...), or names a recent year (this year or the previous `recent_years_back` years, or later) together with
an event noun ("election", "midterm", "summit", ...). A year alone, or an event noun alone, is not enough.

It complements vera.temporality (which looks for "latest", "current", ...); temporality is unchanged.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Iterable

_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def _cfg(cfg: dict | None) -> dict:
    if cfg is None:
        from vera import reporting_lane
        cfg = reporting_lane.load_config()
    ni = cfg.get("news_intent") if isinstance(cfg, dict) else None
    return ni if isinstance(ni, dict) else {}


def _has(text: str, phrases: Iterable[str]) -> list[str]:
    hits = []
    for p in phrases or []:
        words = [re.escape(w) for w in str(p).split() if w]
        if words and re.search(r"(?<![\w-])" + r"\s+".join(words) + r"(?![\w-])", text, re.I):
            hits.append(str(p))
    return hits


def cues(text: str, *, today: date | None = None, cfg: dict | None = None) -> list[str]:
    """The cues found (empty list = no news intent). Never raises."""
    try:
        ni = _cfg(cfg)
        t = str(text or "")
        found = ["cue:" + c for c in _has(t, ni.get("cues", []))]
        nouns = _has(t, ni.get("event_nouns", []))
        if nouns:
            y0 = (today or date.today()).year - int(ni.get("recent_years_back", 1))
            years = [int(y) for y in _YEAR.findall(t) if int(y) >= y0]
            if years:
                found.append(f"event:{nouns[0]}+{years[0]}")
        return found
    except Exception:  # noqa: BLE001  a detector never breaks the answer path
        return []


def news_intent(text: str, *, today: date | None = None, cfg: dict | None = None) -> bool:
    return bool(cues(text, today=today, cfg=cfg))
