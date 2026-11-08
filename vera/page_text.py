"""NOANSWER-PAGE-1108: each sentence once per rendered page, and no heading over an empty section.

`Seen` records the sentences already shown on the page; `new(text)` returns only the sentences of `text` not shown yet
(in order, whitespace and case folded for the comparison only). `Recording` wraps a Streamlit-like target and records
the text of every element it writes, so a later section can drop what an earlier one already said. `Probe` is a sink
that writes nothing and only records whether anything would have been rendered (an empty section gets no heading).
Pure: no I/O, no Streamlit import.
"""
from __future__ import annotations

import re

# A sentence ends at . ! or ? followed by whitespace and a capital letter, digit, quote or bracket.
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(\[*])")


def sentences(text) -> list[str]:
    return [s.strip() for s in _SPLIT.split(text if isinstance(text, str) else "") if s.strip()]


def _key(s: str) -> str:
    return " ".join(s.replace("*", "").replace("\\", "").split()).casefold()  # markdown escapes (plain()) ignored


class Seen:
    def __init__(self):
        self._keys: set = set()

    def add(self, text) -> None:
        for s in sentences(text):
            self._keys.add(_key(s))

    def new(self, text) -> str:
        """The sentences of text not yet shown (nor repeated within text), joined by single spaces; "" when none."""
        out, keys = [], set(self._keys)
        for s in sentences(text):
            k = _key(s)
            if k not in keys:
                keys.add(k)
                out.append(s)
        return " ".join(out)

    def take(self, text) -> str:
        """new(text), then recorded as shown."""
        out = self.new(text)
        self.add(out)
        return out


class Recording:
    """Pass-through wrapper for the text elements the wrapped renderers use (render_fallback_reason, render_discovery):
    every text written through it is recorded in `seen`. Explicit methods only (no dynamic attribute lookup)."""

    def __init__(self, target, seen: Seen):
        self._t, self._seen = target, seen

    def _rec(self, body):
        if isinstance(body, str):
            self._seen.add(body)
        return body

    def info(self, body, *a, **k):
        return self._t.info(self._rec(body), *a, **k)

    def caption(self, body, *a, **k):
        return self._t.caption(self._rec(body), *a, **k)

    def markdown(self, body, *a, **k):
        return self._t.markdown(self._rec(body), *a, **k)

    def warning(self, body, *a, **k):
        return self._t.warning(self._rec(body), *a, **k)

    def success(self, body, *a, **k):
        return self._t.success(self._rec(body), *a, **k)

    def write(self, body, *a, **k):
        return self._t.write(self._rec(body), *a, **k)

    def json(self, body, *a, **k):
        return self._t.json(body, *a, **k)


class Probe:
    """Writes nothing; `shown` turns True on the first element a renderer would have written (render_trace,
    the research captions: markdown and caption only)."""

    def __init__(self):
        self.shown = False

    def _hit(self, *a, **k):
        self.shown = True
        return self

    markdown = caption = info = warning = success = write = table = json = _hit
