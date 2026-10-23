"""Provider-neutral query translation (p3m3 item #71, gap G1).

`build_query_terms(question, cfg)` turns a reader-facing question into search terms: stopwords and meta-words
are dropped, domain phrases from config are kept whole as the CORE concept, and the remaining content words
are the EXTRA terms. Every word list comes from config `search.query_terms` (validated by
vera.eval_config); nothing here encodes a topic.

Fallback is never silent and never empty: when no core phrase and no extra term survives, `mode` is
"fallback_raw" and the caller uses its old behaviour (recorded in the provider's `last_query`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping
from urllib.parse import urlencode

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")
_TOKEN_OR_BREAK = re.compile(r"[A-Za-z][A-Za-z0-9\-]*|[|?!.,;:()]")

# DDFIX-1023/REL: discourse IDIOMS that carry no topic ("de facto status quo best practices" made the acquisition query
# `de AND facto AND status AND quo`). Removed only as whole idioms, never as bare words ("HIV status", "best
# treatments" keep theirs). Shared by query building, the Wikipedia portal query and admission question stems. Config
# search.query_terms.generic_phrases overrides it (config/ was protected for this fix; recorded divergence).
GENERIC_PHRASES = ("de facto", "status quo", "best practices", "best practice", "state of the art", "state-of-the-art",
                   "keep track", "on track", "track record")
# Verbs of the question's framing ("how does X affect Y"): never an AND term of the acquisition query and never a key
# concept for pre-fetch relevance (they stay in the OR queries). Config search.query_terms.filler_words overrides.
FILLER_WORDS = frozenset({"affect", "affects", "affected", "affecting", "drive", "drives", "driven", "driving",
                          "improve", "improves", "improved", "improving", "effective", "interact", "interacts",
                          "influence", "influences", "controlling", "relate", "relates", "related"})


@dataclass(frozen=True)
class QueryTerms:
    core: tuple[str, ...]    # whole domain phrases found in the question (AND-ed)
    extra: tuple[str, ...]   # other content words (OR-ed)
    mode: str                # "core_and_extra" | "core_only" | "extra_only" | "fallback_raw"
    # DDFIX-1023/REL: runs of adjacent non-filler extra terms in question order (a stopword, filler word, idiom or
    # punctuation ends a run), e.g. (("agent", "archetypes"), ("multi-agent", "orchestration")).
    runs: tuple = ()

    @property
    def anchor(self) -> tuple:
        """The longest run (the earliest on a tie); () when no run has two terms."""
        best = max(self.runs, key=len, default=())
        return tuple(best) if len(best) > 1 else ()

    @property
    def empty(self) -> bool:
        return not self.core and not self.extra


def _phrase_regex(phrase: str) -> re.Pattern:
    words = [re.escape(w) for w in phrase.lower().split()]
    return re.compile(r"\b" + r"\s+".join(words) + r"s?\b")


def build_query_terms(question: str, cfg: Mapping) -> QueryTerms:
    text = " ".join(question.lower().split())
    drop = set(cfg["stopwords"]) | set(cfg["meta_words"])
    core: list[str] = []
    # longest phrases first so "ai coding assistant" wins over "coding assistant"
    for phrase in sorted(cfg["core_phrases"], key=lambda p: (-len(p.split()), p)):
        m = _phrase_regex(phrase).search(text)
        if m:
            core.append(m.group(0))
            text = text[:m.start()] + " | " + text[m.end():]  # consumed (and a run break): not repeated among the extra terms
    # keep a core phrase once even if a shorter phrase matched inside another (dedupe by containment)
    core = [c for i, c in enumerate(core) if not any(c != d and c in d for d in core)]
    (extra, runs), _seen = _extra_terms(text, cfg, drop, generic=True)
    if not core and not extra:  # only idioms: keep them rather than fall back to the raw question
        (extra, runs), _seen = _extra_terms(text, cfg, drop, generic=False)
    if core and extra:
        mode = "core_and_extra"
    elif core:
        mode = "core_only"
    elif extra:
        mode = "extra_only"
    else:
        mode = "fallback_raw"
    return QueryTerms(tuple(core), tuple(extra), mode, tuple(tuple(r) for r in runs))


def _generic_cfg(cfg: Mapping) -> tuple:
    return tuple(cfg.get("generic_phrases") or GENERIC_PHRASES)


def filler_words(cfg: Mapping | None = None) -> frozenset:
    return frozenset((cfg or {}).get("filler_words") or FILLER_WORDS)


def and_terms(qt: QueryTerms, k: int, exclude=()) -> list:
    """DDFIX-1023/REL D1: up to `k` distinctive terms to AND: one term from each run in turn (the anchor run first, then
    the others in question order; within a run, question order), so every concept of the question is represented
    before any concept gets a second term; filler words and `exclude` (e.g. the profile's generic_terms) never.
    Returned in question order. [] when no run term is left (the caller keeps its previous rule)."""
    ex = {str(x).lower() for x in exclude or ()}
    runs = [[t for t in r if t not in ex] for r in qt.runs]
    runs = [r for r in runs if r]
    if not runs:
        return []
    a = qt.anchor
    order = sorted(range(len(runs)), key=lambda i: 0 if a and tuple(qt.runs[i]) == a else 1)
    picked: list = []
    depth = 0
    while len(picked) < k and any(depth < len(runs[i]) for i in order):
        for i in order:
            if depth < len(runs[i]) and len(picked) < k:
                picked.append(runs[i][depth])
        depth += 1
    pos = {t: n for n, t in enumerate(qt.extra)}
    return sorted(picked, key=lambda t: pos.get(t, 0))


def drop_generic(text: str, cfg: Mapping | None = None) -> str:
    """The text with the generic IDIOMS removed (case-insensitive, whole phrases; bare words are never removed);
    unchanged when nothing else would remain. Shared by portal queries and admission question stems."""
    low = " ".join(str(text or "").split())
    out = low
    for ph in sorted(_generic_cfg(cfg or {}), key=lambda p: -len(p)):
        out = re.compile(_phrase_regex(ph).pattern, re.I).sub(" ", out)
    out = " ".join(out.split())
    return out if _TOKEN.search(out) else low


def _extra_terms(text: str, cfg: Mapping, drop: set, *, generic: bool):
    """-> ((extra terms in question order, runs of adjacent kept non-filler terms), seen). A dropped word, a filler
    word (kept as an extra term), punctuation or a removed idiom ends a run."""
    if generic:
        for ph in sorted(_generic_cfg(cfg), key=lambda p: -len(p)):
            text = _phrase_regex(ph).sub(" | ", text)
    fill = filler_words(cfg)
    extra, seen, runs, cur = [], set(), [], []
    for tok in _TOKEN_OR_BREAK.findall(text):
        if len(tok) < cfg["min_term_chars"] or tok in drop or not tok[0].isalpha():
            if cur:
                runs.append(cur)
            cur = []
            continue
        if tok in seen or len(extra) >= cfg["max_extra_terms"]:
            continue
        seen.add(tok)
        extra.append(tok)
        if tok in fill:
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        runs.append(cur)
    return (extra, runs), seen


def scholar_search_url(question: str, terms_cfg: Mapping) -> str:
    """Human link-out to Google Scholar (item #71 R10-b). A PURE STRING BUILDER: VERA-DAR never requests, fetches or
    parses this URL (Scholar's robots.txt disallows /scholar and Google's terms forbid automated access against
    robots.txt). A person clicks it in their own browser. This is the only place in vera/ that names the host."""
    qt = build_query_terms(question, terms_cfg)
    words = list(qt.core) + list(qt.extra)
    q = " ".join(words) if words else " ".join(question.split())
    return "https://scholar.google.com/scholar?" + urlencode({"q": q})


# ---- DDFIX-1023/REL: deterministic pre-fetch concept relevance (title + abstract), used to ORDER candidates before
# the max_candidates_resolved cut. It never admits anything: the full-text relevance gate still decides.
_RUN_CFG = Path(__file__).resolve().parent.parent / "config" / "vera_eval_run.json"


@lru_cache(maxsize=1)
def _default_terms_cfg() -> Mapping | None:
    try:
        return json.loads(_RUN_CFG.read_text())["search"]["query_terms"]
    except Exception:  # noqa: BLE001  ranking then keeps the existing order (no concepts)
        return None


def key_concepts(question: str, cfg: Mapping | None = None, extra_generic=()) -> tuple[tuple, dict]:
    """(key concepts, weights). Concepts: core phrases + distinctive extra terms (stop, meta, filler words and
    `extra_generic`, e.g. the acquisition profile's generic_terms, removed; idioms never reach them). Weight of a
    concept: 1 + the number of other concepts that contain it as a hyphen component ("agent" in "multi-agent": 2).
    ((), {}) without a terms config."""
    cfg = cfg if cfg is not None else _default_terms_cfg()
    if not cfg:
        return (), {}
    qt = build_query_terms(question, cfg)
    gen = {str(g).lower() for g in extra_generic or ()} | filler_words(cfg)
    out = tuple(t for t in (*qt.core, *qt.extra) if t not in gen)
    w = {c: 1 + sum(1 for d in out if d != c and c in re.split(r"[\s\-]+", d)) for c in out}
    return out, w


def _concept_regex(term: str) -> re.Pattern:
    """Whole-word match; a hyphenated compound also matches "multi agent" / "multiagent"; the last word may inflect
    (a word over five letters is matched on all but its last three letters, min five: orchestrat-ion/-ing)."""
    words = term.lower().replace("-", " ").split()
    *head, last = words
    stem = last if len(last) <= 5 else last[:max(5, len(last) - 3)]
    sep = r"[\s\-]?" if "-" in term else r"\s+"
    return re.compile(r"\b" + "".join(re.escape(w) + sep for w in head) + re.escape(stem), re.I)


def matched_concepts(kc: tuple, text: str) -> set:
    """Concepts found in `text`; a matched compound also counts its component concepts ("multiagent" -> agent)."""
    concepts = kc[0]
    t = str(text or "")
    got = {c for c in concepts if _concept_regex(c).search(t)}
    for c in list(got):
        got |= {d for d in concepts if d != c and d in re.split(r"[\s\-]+", c)}
    return got


# A title + abstract with fewer words is not judged (neutral: never dropped, never demoted). CALIBRATION NOTE: planner
# prior, UNCALIBRATED against live data. Seen only on the owner's five live false positives of 2026-10-23 (titles of
# 7-17 words, no abstract) and offline fixtures. Change it when live R-ACQ-IRRELEVANT pre-fetch drops are reviewed
# against their titles (a relevant short-titled record dropped -> raise; off-topic short titles taking slots -> lower).
MIN_JUDGE_WORDS = 4


def relevance_tiers(kc: tuple, texts) -> list:
    """DDFIX-1023/REL D4, per text: 0 = relevant, 1 = weak, 2 = off topic (title + abstract match NONE of the question's
    key concepts). Weak = only lower-weight concepts matched while the question has a central concept (weight > 1),
    e.g. "Pedestrian Archetypes" for an agent-orchestration question; with no central concept every match is 0.
    Texts under MIN_JUDGE_WORDS words are 0 (not judged). Relative to the pool: when no text in it matches any
    concept, every text is 0 (the previous behaviour; the full-text relevance gate still decides). Tiers ORDER
    candidates before the caps (0, then 1); only tier 2 is dropped (R-ACQ-IRRELEVANT) by the callers."""
    concepts, w = kc
    texts = [str(t or "") for t in texts]
    if not concepts:
        return [0] * len(texts)
    top = max(w.values())
    out, any_hit = [], False
    for t in texts:
        if len(_TOKEN.findall(t)) < MIN_JUDGE_WORDS:
            out.append(0)
            continue
        m = matched_concepts(kc, t)
        any_hit = any_hit or bool(m)
        out.append(2 if not m else 1 if top > 1 and max(w[c] for c in m) < top else 0)
    return out if any_hit else [0] * len(texts)
