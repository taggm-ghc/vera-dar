"""Deterministic question-part detection (facets and concepts); no model call."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

COMPARATIVE = r"\b(?:more|less|fewer|greater|higher|lower)\b[^.]{0,40}?\bthan\b"

DEFAULT_TABLE: dict[str, dict[str, list[str]]] = {
    # REPLAY-1028 (coordinator 2026-10-28): span cues are PART-SPECIFIC only. Weak cues ("than", "but", "while",
    # "also", "however", "same", comparatives) satisfied the contrast part on off-topic passages and were removed.
    "contrast": {
        "question": ["difference", "differences", "differ", "differs", "different",
                     "vs", "versus", "compared", "compare", "comparison",
                     "contrast", "distinguish"],
        "span": ["unlike", "whereas", "differ*", "in contrast"],
    },
    "common": {
        "question": ["commonality", "commonalities", "similarity", "similarities",
                     "similar", "in common", "shared", "share", "both", "alike"],
        "span": ["both", "similar*", "share*", "common*"],
    },
}


@dataclass(frozen=True)
class Part:
    """One part of a question: a facet or a concept."""

    kind: str
    label: str
    cues: tuple[str, ...]


def table(cfg: Any = None) -> dict:
    """Return cfg['facets'] when a valid non-empty dict, else DEFAULT_TABLE."""
    if isinstance(cfg, dict):
        f = cfg.get("facets")
        if isinstance(f, dict) and f:
            return f
    return DEFAULT_TABLE


def _cue_re(cue: str) -> re.Pattern:
    if cue == "<comparative>":
        return re.compile(COMPARATIVE, re.I)
    prefix = cue.endswith("*")
    words = cue.rstrip("*").split()
    body = r"\s+".join(re.escape(w) for w in words)
    return re.compile(r"\b" + body + ("" if prefix else r"\b"), re.I)


def facets(question: str, cfg: Any = None) -> list[Part]:
    """Facets whose question cues occur, in order of first occurrence; [] if < 2."""
    found: list[tuple[int, str, Part]] = []
    for name, spec in table(cfg).items():
        pos = None
        for cue in spec.get("question", []):
            m = _cue_re(cue).search(question or "")
            if m and (pos is None or m.start() < pos):
                pos = m.start()
        if pos is not None:
            found.append((pos, name, Part("facet", name, tuple(spec.get("span", [])))))
    found.sort(key=lambda t: t[0])
    parts = [p for _, _, p in found]
    return parts if len(parts) >= 2 else []


def has_facet_cue(question: str, cfg: Any = None) -> bool:
    """True when any facet's question cue occurs (one facet is enough here)."""
    return any(_cue_re(c).search(question or "") for spec in table(cfg).values() for c in spec.get("question", []))


_SIDES = (re.compile(r"\bbetween\s+(.+?)\s+and\s+(.+?)\s*(?:[?.!;,]|$)", re.I),
          re.compile(r"\bcompar\w*\s+(.+?)\s+(?:and|with|to|versus|vs\.?)\s+(.+?)\s*(?:[?.!;,]|$)", re.I),
          re.compile(r"(?:^|[:;,]\s*|\b(?:is|are|does|do)\s+)([^?.!;,:]+?)\s+(?:vs\.?|versus)\s+(.+?)\s*(?:[?.!;,]|$)", re.I))


def comparison_sides(question: str, cfg: Any = None) -> list:
    """REVIEW-1028-E1FA7 F3: the two compared sides of an EXPLICIT comparison ("between X and Y", "compare X and/with
    Y", "X vs Y"), only when the question also has a facet cue (so "the relationship between diet and cancer" is not
    a comparison). [] otherwise. Sides are raw text; the caller maps them to key concepts."""
    q = question or ""
    if not has_facet_cue(q, cfg):
        return []
    for rx in _SIDES:
        m = rx.search(q)
        if m and m.group(1).strip() and m.group(2).strip():
            return [m.group(1).strip(), m.group(2).strip()]
    return []


def _member_re(member: str) -> re.Pattern | None:
    words = [w for w in re.split(r"[\s\-]+", member.strip()) if w]
    if not words:
        return None
    last = words[-1]
    if len(last) > 5:
        last_re = re.escape(last[:max(5, len(last) - 3)]) + r"\w*"
    else:
        last_re = re.escape(last)
    body = r"[\s\-]+".join([re.escape(w) for w in words[:-1]] + [last_re])
    return re.compile(r"\b" + body + r"\b", re.I)


def part_fit(span_text: str, part: Part) -> int:
    """Facet: distinct span cues found. Concept: 1 if any member matches, else 0.

    Concept inflection: a last word longer than 5 letters matches on its first
    max(5, len-3) letters as a prefix ("boxing" -> "boxin*", so "boxing" but not
    "boxers").
    """
    text = span_text or ""
    if part.kind == "facet":
        return sum(1 for c in set(part.cues) if _cue_re(c).search(text))
    for m in part.cues:
        rx = _member_re(m)
        if rx and rx.search(text):
            return 1
    return 0


def concept_parts(concepts: Iterable[str]) -> list[Part]:
    """Concept Parts from strings with '|' alternatives; [] when fewer than 2."""
    out: list[Part] = []
    seen: set[tuple[str, ...]] = set()
    for c in concepts or []:
        members = tuple(m.strip() for m in str(c).split("|") if m.strip())
        key = tuple(m.lower() for m in members)
        if not members or key in seen:
            continue
        seen.add(key)
        out.append(Part("concept", members[0], members))
    return out if len(out) >= 2 else []
