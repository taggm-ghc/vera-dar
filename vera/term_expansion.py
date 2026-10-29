"""PLAN-1028-FIRSTASK WP-FA1 (E1): term-concept groups (acronyms and spelled-out forms). Pure; no network, DB or env.

A group is one question CONCEPT matched by any of its phrases ("mixed martial arts" | "mma"). Admission rewrites
the text before tokenizing: every phrase of a group the QUESTION triggered becomes the letters-only token
"xg<id>" (vera.fulldoc_admission keeps that token whole, never cut to stem_len). SECREV C1: the token carries a
WEIGHT = the number of content stems of the group's spelled-out (first non-ambiguous) phrase, so a spelled-out
question keeps today's arithmetic exactly and an alias-only span gains the same weight.

Ambiguity guard (SECREV C1): an `ambiguous` phrase ("mma": methylmalonic acid, methyl methacrylate) alone never
triggers its group; it triggers only when the question also has another phrase of the group or one of the group's
`cues`. A group that the question did not trigger never rewrites anything (span text included).

The groups are a SEED (config/term_expansions.json); the target is a DB table (owner rule: tables are DB tables).
Switch: config/fulldoc_answer.json admission.term_groups.enabled (exactly true; default false).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = _ROOT / "config" / "term_expansions.json"
_FD_CFG = _ROOT / "config" / "fulldoc_answer.json"
_ID = re.compile(r"[a-z]{3,40}")
TOKEN_PREFIX = "xg"
MAX_VARIANTS_PER_GROUP = 2  # discovery bound (plan WP-FA1 query_variants); a resource bound, not a decision threshold


@dataclass(frozen=True)
class Group:
    id: str
    phrases: tuple  # ((text, ambiguous), ...) in config order; the first non-ambiguous phrase is the spelled-out form
    cues: tuple = ()

    @property
    def token(self) -> str:
        return TOKEN_PREFIX + self.id

    @property
    def spelled(self) -> str:
        return next((t for t, amb in self.phrases if not amb), self.phrases[0][0])


def _phrase_rx(phrase: str) -> re.Pattern:
    words = [re.escape(w) for w in phrase.casefold().split()]
    return re.compile(r"(?<![^\W_])" + r"[\s\-]+".join(words) + r"s?(?![^\W_])", re.IGNORECASE)


def parse(data) -> tuple:
    """Validated groups, or () (fail closed) on any schema fault: ids letters-only and unique, every phrase >= 2
    characters, no phrase in two groups, at least two phrases per group."""
    try:
        out, seen_ids, seen_ph = [], set(), set()
        for g in data["groups"]:
            gid = str(g["id"])
            if not _ID.fullmatch(gid) or gid in seen_ids:
                return ()
            seen_ids.add(gid)
            ph = []
            for p in g["phrases"]:
                t = " ".join(str(p["text"]).casefold().split())
                if len(t) < 2 or t in seen_ph:
                    return ()
                seen_ph.add(t)
                ph.append((t, p.get("ambiguous") is True))
            if len(ph) < 2:
                return ()
            cues = tuple(" ".join(str(c).casefold().split()) for c in (g.get("cues") or ()) if str(c).strip())
            out.append(Group(gid, tuple(ph), cues))
        return tuple(out)
    except (KeyError, TypeError, ValueError, AttributeError):
        return ()


@lru_cache(maxsize=4)
def load(path: str | None = None) -> tuple:
    try:
        return parse(json.loads(Path(path or DEFAULT_PATH).read_text()))
    except (OSError, ValueError):
        return ()


def enabled(cfg) -> bool:
    tg = ((cfg or {}).get("admission") or {}).get("term_groups") if isinstance(cfg, dict) else None
    return isinstance(tg, dict) and tg.get("enabled") is True


def groups_for(cfg) -> tuple:
    """The configured groups when admission.term_groups.enabled is exactly true, else ()."""
    if not enabled(cfg):
        return ()
    p = cfg["admission"]["term_groups"].get("path")
    return load(str(_ROOT / p) if isinstance(p, str) and p else None)


@lru_cache(maxsize=1)
def default_groups() -> tuple:
    """groups_for(the shipped config/fulldoc_answer.json); () when the file is missing or the switch is off."""
    try:
        return groups_for(json.loads(_FD_CFG.read_text()))
    except (OSError, ValueError):
        return ()


def _has(text: str, phrase: str) -> bool:
    return bool(_phrase_rx(phrase).search(text))


def triggered(question: str, groups) -> tuple:
    """Groups the question names: a non-ambiguous phrase, or an ambiguous one plus a second group phrase or a cue."""
    q = str(question or "")
    out = []
    for g in groups or ():
        hits = [(t, amb) for t, amb in g.phrases if _has(q, t)]
        if any(not amb for _t, amb in hits):
            out.append(g)
        elif hits and (len(hits) >= 2 or any(_has(q, c) for c in g.cues)):
            out.append(g)
    return tuple(out)


def canonicalize(text: str, groups) -> str:
    """Every phrase of `groups` (pass only TRIGGERED groups) replaced by the group token, longest phrase first,
    whole-word, case-insensitive. Pure text rewrite; tokenizer settings are unchanged."""
    # a literal group token already in the text (document or question) is defused first, so text cannot inject a
    # group match by spelling "xg<id>" (REVIEW-1028-E1FA7 probe "inject")
    s = re.sub(r"(?<![^\W_])" + TOKEN_PREFIX + r"(?=[^\W\d_])", TOKEN_PREFIX + "_", str(text or ""), flags=re.IGNORECASE)
    pairs = sorted(((t, g.token) for g in groups or () for t, _a in g.phrases), key=lambda p: (-len(p[0]), p[0]))
    for t, tok in pairs:
        s = _phrase_rx(t).sub(f" {tok} ", s)
    return s


def collision_counts(texts, groups) -> dict:
    """Plan s.6 calibration helper (REVIEW-1028-E1FA7 F1): per ambiguous phrase, how many texts contain it
    {"alias": n, "alias_triggered": n, "alias_untriggered": n}. An untriggered hit is a potential collision (another
    sense, e.g. methylmalonic acid) that the guard keeps out; a triggered hit in an off-domain text is a guard failure
    to inspect. Counts only; run it over the held corpus (in work/, never saving text) and the scope_set questions."""
    out = {}
    for g in groups or ():
        for t, amb in g.phrases:
            if not amb:
                continue
            c = {"alias": 0, "alias_triggered": 0, "alias_untriggered": 0}
            for x in texts or ():
                if _has(str(x), t):
                    c["alias"] += 1
                    c["alias_triggered" if g in triggered(str(x), (g,)) else "alias_untriggered"] += 1
            out[f"{g.id}:{t}"] = c
    return out


def query_variants(question: str, groups, max_per_group: int = MAX_VARIANTS_PER_GROUP) -> list:
    """Discovery variants: for each triggered group, its phrases the question does not already contain (spelled-out
    form first), at most `max_per_group` each. An untriggered ambiguous alias adds nothing."""
    q = str(question or "")
    out = []
    for g in triggered(q, groups):
        alts = sorted((t for t, _a in g.phrases if not _has(q, t)), key=lambda t: t != g.spelled)
        out.extend(alts[:max(int(max_per_group), 0)])
    return out
