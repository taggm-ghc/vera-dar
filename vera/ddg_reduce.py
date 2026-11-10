"""DDG-REDUCE-1110 (owner OD-1110-SEARCHCAPS "keep low, use it better"): switches that lower the DuckDuckGo POSTs a
visitor question spends. Config config/ddg_reduce.json; every switch ships true (the new behaviour) and false restores the
previous behaviour for that one path. A missing file, an unreadable file or a missing key is false (previous behaviour).
Counts of what each path spends: vera.ddg_meter."""
from __future__ import annotations

import contextlib
import contextvars
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "ddg_reduce.json"

# Set by a call site that must not spend DuckDuckGo (variant rounds, answered non-news questions): the acquisition
# search budget drops its DuckDuckGo slot. Distinct from search_providers.SCHOLARLY_ONLY, which also drops Wikipedia
# and the id lookup and is kept for variant rounds of non-news questions.
NO_DDG: contextvars.ContextVar[bool] = contextvars.ContextVar("vera_ddg_none", default=False)


def load(path: Path | None = None) -> dict:
    try:
        doc = json.loads((path or CONFIG_PATH).read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def on(key: str) -> bool:
    return load().get(key) is True


def fallback_only() -> bool:
    """C6: DuckDuckGo runs only when the other routes return too little (config ddg_fallback_only)."""
    return on("ddg_fallback_only")


def fallback_min() -> int:
    """C6: the fewest usable candidates from the other routes that make DuckDuckGo unnecessary (UNCALIBRATED prior)."""
    v = load().get("ddg_fallback_min_results")
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 1 else 3


def fallback_min_remaining_s() -> float:
    """C6: the deadline left (seconds) below which the fallback is skipped rather than started (config
    ddg_fallback_min_remaining_s, default 1.0: a DuckDuckGo call needs about a second to be worth starting)."""
    v = load().get("ddg_fallback_min_remaining_s")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 else 1.0


def usable_count(question: str, cands, extra_generic=()) -> int:
    """C6 (F2): how many of `cands` (dicts with title / snippet, url) count toward ddg_fallback_min_results. With
    ddg_fallback_relevant_only on, a candidate the EXISTING pre-filters would drop (search_query.relevance_tiers tier 2,
    the DDFIX-1023 off-topic test; search_providers.acq_prefilter nd / not_oa) does not count, so loosely matched hits do
    not suppress the fallback. No new heuristic. Never raises: on any error every candidate counts."""
    items = [c for c in (cands or ()) if isinstance(c, dict)]
    if not on("ddg_fallback_relevant_only"):
        return len(list(cands or ()))
    try:
        from vera import search_providers, search_query
        kc = search_query.key_concepts(question, extra_generic=tuple(extra_generic or ()))
        tiers = search_query.relevance_tiers(kc, [f"{c.get('title') or ''} {c.get('snippet') or ''}" for c in items])
        return sum(1 for c, t in zip(items, tiers) if t != 2 and not search_providers.acq_prefilter(c))
    except Exception:  # noqa: BLE001
        return len(items)


@contextlib.contextmanager
def no_ddg(active: bool = True):
    tok = NO_DDG.set(bool(active) or NO_DDG.get())
    try:
        yield
    finally:
        NO_DDG.reset(tok)
