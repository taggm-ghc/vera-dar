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

# DDFIX-1023/REL: discourse idioms and words that carry no topic on their own ("de facto status quo best practices"
# made the acquisition query `de AND facto AND status AND quo` and returned status-quo-bias papers). They are never
# discovery terms unless nothing else is left. Config search.query_terms.generic_words / generic_phrases override
# these defaults (config/ was protected for this fix, so the defaults live here; recorded divergence).
GENERIC_PHRASES = ("de facto", "status quo", "best practices", "best practice", "state of the art", "state-of-the-art",
                   "keep track", "on track", "track record")
GENERIC_WORDS = frozenset({"de", "facto", "status", "quo", "best", "practice", "practices", "track", "tracks"})


@dataclass(frozen=True)
class QueryTerms:
    core: tuple[str, ...]    # whole domain phrases found in the question (AND-ed)
    extra: tuple[str, ...]   # other content words (OR-ed)
    mode: str                # "core_and_extra" | "core_only" | "extra_only" | "fallback_raw"
    # DDFIX-1023/REL: the longest run of adjacent extra terms in the question (a hyphenated compound breaks ties), e.g.
    # ("multi-agent", "orchestration"); () when no run has two terms. Used to AND the acquisition query.
    anchor: tuple[str, ...] = ()

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
    extra, seen = _extra_terms(text, cfg, drop, generic=True)
    if not core and not extra[0]:  # only generic words: keep them rather than fall back to the raw question
        extra, seen = _extra_terms(text, cfg, drop, generic=False)
    extra, runs = extra
    if core and extra:
        mode = "core_and_extra"
    elif core:
        mode = "core_only"
    elif extra:
        mode = "extra_only"
    else:
        mode = "fallback_raw"
    anchor = max(runs, key=lambda r: (len(r), any("-" in t for t in r)), default=[]) if runs else []
    return QueryTerms(tuple(core), tuple(extra), mode, tuple(anchor) if len(anchor) > 1 else ())


def _generic_cfg(cfg: Mapping) -> tuple[tuple, frozenset]:
    phrases = tuple(cfg.get("generic_phrases") or GENERIC_PHRASES)
    return phrases, frozenset(cfg.get("generic_words") or GENERIC_WORDS)


def drop_generic(text: str, cfg: Mapping | None = None) -> str:
    """The text with the generic phrases and words removed (case-insensitive, whole words); unchanged when nothing
    else would remain. For portals that build their own query from the question (DDFIX-1023/REL)."""
    phrases, gw = _generic_cfg(cfg or {})
    low = " ".join(str(text or "").split())
    out = low
    for ph in sorted(phrases, key=lambda p: -len(p)):
        out = re.compile(_phrase_regex(ph).pattern, re.I).sub(" ", out)
    kept = [w for w in out.split() if re.sub(r"[^\w-]", "", w).lower() not in gw]
    return " ".join(kept) if any(_TOKEN.search(w) for w in kept) else low


def _extra_terms(text: str, cfg: Mapping, drop: set, *, generic: bool):
    """-> ((extra terms in question order, runs of adjacent kept terms), seen). A dropped word, punctuation or a
    removed generic phrase ends a run."""
    gw: frozenset = frozenset()
    if generic:
        phrases, gw = _generic_cfg(cfg)
        for ph in sorted(phrases, key=lambda p: -len(p)):
            text = _phrase_regex(ph).sub(" | ", text)
    extra, seen, runs, cur = [], set(), [], []
    for tok in _TOKEN_OR_BREAK.findall(text):
        if (len(tok) < cfg["min_term_chars"] or tok in drop or tok in gw or not tok[0].isalpha()):
            if cur:
                runs.append(cur)
            cur = []
            continue
        if tok in seen or len(extra) >= cfg["max_extra_terms"]:
            continue
        seen.add(tok)
        extra.append(tok)
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


def key_concepts(question: str, cfg: Mapping | None = None, extra_generic=()) -> tuple[tuple, tuple]:
    """(key concepts, anchor) of the question. Concepts: core phrases + distinctive extra terms (stop, meta and generic
    words removed, plus `extra_generic`, e.g. the acquisition profile's generic_terms). Anchor: the run of adjacent
    concepts holding a hyphenated compound ("multi-agent orchestration"), else (). ((), ()) without a terms config."""
    cfg = cfg if cfg is not None else _default_terms_cfg()
    if not cfg:
        return (), ()
    qt = build_query_terms(question, cfg)
    gen = {str(g).lower() for g in extra_generic or ()}
    _phrases, gw = _generic_cfg(cfg)
    out = tuple(t for t in (*qt.core, *qt.extra) if t not in gen and t not in gw)
    anchor = tuple(t for t in qt.anchor if t in out)
    return out, (anchor if len(anchor) > 1 and any("-" in t for t in anchor) else ())


def _concept_regex(term: str) -> re.Pattern:
    """Whole-word match; a hyphenated compound also matches "multi agent" / "multiagent"; the last word may inflect
    (a word over five letters is matched on all but its last three letters, min five: orchestrat-ion/-ing)."""
    words = term.lower().replace("-", " ").split()
    *head, last = words
    stem = last if len(last) <= 5 else last[:max(5, len(last) - 3)]
    sep = r"[\s\-]?" if "-" in term else r"\s+"
    return re.compile(r"\b" + "".join(re.escape(w) + sep for w in head) + re.escape(stem), re.I)


def concept_hits(concepts, text: str) -> int:
    """Number of distinct key concepts found in `text`."""
    t = str(text or "")
    return sum(1 for c in concepts if _concept_regex(c).search(t))


# A title + abstract with fewer words is not judged (neutral, never dropped): too little text for a negative judgement
# (e.g. a three-word title with no abstract). CALIBRATION NOTE: planner prior, UNCALIBRATED against live data. Seen only
# on the owner's five live false positives of 2026-10-23 (titles of 7-17 words, no abstract; all judged and dropped)
# and the offline fixtures. Change it when live R-ACQ-IRRELEVANT pre-fetch drops are reviewed against their titles
# (a relevant short-titled record dropped -> raise; off-topic short titles taking slots -> lower).
MIN_JUDGE_WORDS = 4


def off_topic(kc: tuple, text: str) -> bool:
    """True when the candidate's title + abstract (at least MIN_JUDGE_WORDS words) matches none of the question's key
    concepts, or misses every term of its compound anchor: a lexical false positive of the search ("status quo bias",
    "Pedestrian Archetypes"). Ranking only: such candidates go behind every other candidate before the cut; nothing is
    dropped and the full-text relevance gate still decides admission."""
    concepts, anchor = kc
    if not concepts or len(_TOKEN.findall(str(text or ""))) < MIN_JUDGE_WORDS:
        return False
    if not concept_hits(concepts, text):
        return True
    return bool(anchor) and not concept_hits(anchor, text)


def off_topic_flags(kc: tuple, texts) -> list:
    """off_topic() per text, RELATIVE to the pool: when no text in the pool is on topic (judged and matching), nothing is
    flagged, so a pool without any relevant candidate keeps the previous behaviour (the full-text relevance gate still
    decides). Only when a relevant candidate exists do the lexical false positives give way to it."""
    texts = [str(t or "") for t in texts]
    flags = [off_topic(kc, t) for t in texts]
    on = any(not f and len(_TOKEN.findall(t)) >= MIN_JUDGE_WORDS and concept_hits(kc[0], t)
             for f, t in zip(flags, texts))
    return flags if on else [False] * len(texts)
