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


@contextlib.contextmanager
def no_ddg(active: bool = True):
    tok = NO_DDG.set(bool(active) or NO_DDG.get())
    try:
        yield
    finally:
        NO_DDG.reset(tok)
