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
    # NONASK-1103 S2: the named entity's own terms in a sub-query; the run holding one is the anchor (taken first)
    central: tuple = ()

    @property
    def anchor(self) -> tuple:
        """The longest run (the earliest on a tie); () when no run has two terms. NONASK-1103 S2: the run holding a
        central (named-entity) term, when there is one."""
        cen = [r for r in self.runs if any(t in self.central for t in r)]
        best = max(cen, key=len, default=()) if cen else max(self.runs, key=len, default=())
        return tuple(best) if (len(best) > 1 or cen) else ()

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


def _groups(groups):
    """E1 term groups: the given ones, else the shipped default (() unless admission.term_groups.enabled is true)."""
    if groups is not None:
        return tuple(groups)
    from vera import term_expansion
    return term_expansion.default_groups()


def key_concepts(question: str, cfg: Mapping | None = None, extra_generic=(), groups=None,
                 central=()) -> tuple[tuple, dict]:
    """(key concepts, weights). Concepts: core phrases + distinctive extra terms (stop, meta, filler words and
    `extra_generic`, e.g. the acquisition profile's generic_terms, removed; idioms never reach them). Weight of a
    concept: 1 + the number of other concepts that contain it as a hyphen component ("agent" in "multi-agent": 2).
    ((), {}) without a terms config.
    NONASK-1103 S2: `central` (the named entities of a sub-query) makes each matching concept CENTRAL: its weight is
    (top weight + 1), so a record that lacks the entity is weak (tier 1) however many generic concepts it matches.
    No new number: the weight is derived from the others."""
    cfg = cfg if cfg is not None else _default_terms_cfg()
    if not cfg:
        return (), {}
    qt = build_query_terms(question, cfg)
    gen = {str(g).lower() for g in extra_generic or ()} | filler_words(cfg)
    out = tuple(t for t in (*qt.core, *qt.extra) if t not in gen)
    gs = _groups(groups)
    if gs:
        # IMPL-1028-E1FA7 (E1): a term group the question triggers is ONE concept "phrase1|phrase2|..." matched by any
        # member (_concept_regex); the question words of its phrases are folded into it. Ambiguous aliases follow
        # vera.term_expansion.triggered (never alone without a second group phrase or a cue).
        from vera import term_expansion
        for g in term_expansion.triggered(question, gs):
            words = {w for t, _a in g.phrases if term_expansion._has(question, t) for w in t.split()}
            first = next((i for i, c in enumerate(out) if c in words), len(out))
            rest = [c for c in out if c not in words]
            out = tuple(rest[:first] + ["|".join(t for t, _a in g.phrases)] + rest[first:])
    w = {c: 1 + sum(1 for d in out if d != c and "|" not in d and c in re.split(r"[\s\-]+", d)) for c in out}
    cen = entity_concepts(out, central)
    if cen:
        top = max(w.values())
        w = {c: (top + 1 if c in cen else v) for c, v in w.items()}
    return out, w


def entity_concepts(concepts, entities) -> set:
    """NONASK-1103 S2: the concepts that name one of `entities` (case-insensitive; a concept matches when it equals the
    entity or when one is a whole-word part of the other: "aws" for "AWS", "azure" for "Microsoft Azure")."""
    ents = [" ".join(re.findall(r"[a-z0-9+#.\-]+", str(e).lower())) for e in entities or ()]
    ents = [e for e in ents if e]
    out = set()
    for c in concepts:
        if "|" in c:
            continue
        for e in ents:
            if c == e or re.search(r"(?<![a-z0-9])" + re.escape(c) + r"(?![a-z0-9])", e) \
                    or re.search(r"(?<![a-z0-9])" + re.escape(e) + r"(?![a-z0-9])", c):
                out.add(c)
    return out


def subquery_terms(text: str, entities, cfg: Mapping) -> QueryTerms:
    """NONASK-1103 S2: the query terms of ONE sub-query, with the named entity central: every run is rebuilt so the
    entity's own term leads its run (and_terms then takes it first). Built only from the part's own words (drift
    guard S4: the terms are a subset of the sub-query's terms)."""
    qt = build_query_terms(text, cfg)
    cen = entity_concepts(qt.extra, entities)
    if not cen or not qt.runs:
        return qt
    runs = tuple(tuple(sorted(r, key=lambda t: 0 if t in cen else 1)) for r in qt.runs)
    # the run holding the entity is the anchor (the longest), so the entity is taken first by and_terms
    runs = tuple(sorted(runs, key=lambda r: 0 if any(t in cen for t in r) else 1))
    return QueryTerms(qt.core, qt.extra, qt.mode, runs, tuple(sorted(cen)))


def query_variants(question: str, groups=None) -> list:
    """E1 discovery variants (bounded): the other phrases of each term group the question triggers, e.g. "mixed martial
    arts" for "boxing vs MMA"; [] when term groups are off. Not yet sent by vera.search_providers (IMPL-1028-E1FA7
    note: the acquisition-query wiring is a separate leaf)."""
    from vera import term_expansion
    return term_expansion.query_variants(question, _groups(groups))


def _concept_regex(term: str) -> re.Pattern:
    """Whole-word match; a hyphenated compound also matches "multi agent" / "multiagent"; the last word may inflect
    (a word over five letters is matched on all but its last three letters, min five: orchestrat-ion/-ing)."""
    if "|" in term:  # E1 group concept: any member
        return re.compile("|".join(_concept_regex(m).pattern for m in term.split("|") if m.strip()), re.I)
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
        if "|" not in c:
            got |= {d for d in concepts if d != c and d in re.split(r"[\s\-]+", c)}
    return got


# A title + abstract with fewer words is not judged off topic (never dropped; tier 1 unless it matches a central concept). CALIBRATION NOTE: planner
# prior, UNCALIBRATED against live data. Seen only on the owner's five live false positives of 2026-10-23 (titles of
# 7-17 words, no abstract) and offline fixtures. Change it when live R-ACQ-IRRELEVANT pre-fetch drops are reviewed
# against their titles (a relevant short-titled record dropped -> raise; off-topic short titles taking slots -> lower).
MIN_JUDGE_WORDS = 4


def _has_phrase(text: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase.lower()).replace(r"\ ", r"\s+") + r"(?![a-z0-9])", text) is not None


def entity_hit(text: str, ctx: Mapping) -> bool:
    """NONASK-1103 S2: does `text` really name the entity described by registry context `ctx` (keys: names, full_aliases,
    parents, context_terms, generic_phrases)? A full alias counts alone. A plain name or parent-company hit counts only
    after the entity's generic phrases ("azure sky", "google it", "snowflake schema") are removed AND only with a
    context term. No numeric threshold; the lists are curated registry data. With no context terms the hit is not
    narrowed (the unchanged behaviour for an entity the registry does not describe)."""
    low = " ".join(str(text or "").lower().split())
    if any(_has_phrase(low, a) for a in ctx.get("full_aliases") or ()):
        return True
    for g in sorted(ctx.get("generic_phrases") or (), key=len, reverse=True):
        low = re.sub(r"(?<![a-z0-9])" + re.escape(g.lower()) + r"(?![a-z0-9])", " ", low)
    named = any(_has_phrase(low, n) for n in (*(ctx.get("names") or ()), *(ctx.get("parents") or ())))
    if not named:
        return False
    cterms = ctx.get("context_terms") or ()
    return True if not cterms else any(_has_phrase(low, c) for c in cterms)


def part_tiers(question: str, texts, part_of, parts, *, cfg=None, extra_generic=(), ctx_for=None) -> list:
    """NONASK-1103 S2/S3: the pre-fetch relevance tier of each text. `part_of[i]` = index of the sub-query that found
    text i (None = question level). With fewer than 2 `parts` every text is judged against the question's concepts as
    one pool (today). Otherwise a text found for part k is judged against THAT part's concepts with the named entity
    central AND really present (entity_hit via `ctx_for(entity) -> ctx or None`); tiers stay relative to each group."""
    texts = list(texts)
    kc_q = key_concepts(question, cfg, extra_generic=extra_generic)
    if len(parts or ()) < 2:
        return relevance_tiers(kc_q, texts)
    groups: dict = {}
    for i, k in enumerate(part_of):
        groups.setdefault(k if isinstance(k, int) and 0 <= k < len(parts) else None, []).append(i)
    out = [0] * len(texts)
    for k, idx in groups.items():
        if k is None:
            kc, ents = kc_q, None
        else:
            kc = key_concepts(parts[k][0], cfg, extra_generic=extra_generic, central=parts[k][1])
            ents = [c for c in (ctx_for(e) for e in parts[k][1]) if c] if ctx_for else None
        for i, t in zip(idx, relevance_tiers(kc, [texts[i] for i in idx], entities=ents or None)):
            out[i] = t
    return out


def relevance_tiers(kc: tuple, texts, entities=None) -> list:
    """DDFIX-1023/REL D4, per text: 0 = relevant, 1 = weak, 2 = off topic (title + abstract match NONE of the question's
    key concepts). Weak = only lower-weight concepts matched while the question has a central concept (weight > 1),
    e.g. "Pedestrian Archetypes" for an agent-orchestration question; with no central concept every match is 0.
    Texts under MIN_JUDGE_WORDS words are never dropped: 0 if they match a central concept, else 1 (D6). Relative to the pool: when no text in it matches any
    concept, every text is 0 (the previous behaviour; the full-text relevance gate still decides). Tiers ORDER
    candidates before the caps (0, then 1); only tier 2 is dropped (R-ACQ-IRRELEVANT) by the callers."""
    concepts, w = kc
    texts = [str(t or "") for t in texts]
    if not concepts:
        return [0] * len(texts)
    ent_c = [(entity_concepts(concepts, c.get("names") or ()), c) for c in entities or ()]

    def _mc(t):  # matched concepts; an entity concept counts only on a real entity hit (disambiguation)
        m = matched_concepts(kc, t)
        for cs, c in ent_c:
            if cs:
                m = (m | cs) if entity_hit(t, c) else (m - cs)
        return m
    top = max(w.values())
    out, any_hit = [], False
    for t in texts:
        if len(_TOKEN.findall(t)) < MIN_JUDGE_WORDS:
            # DDFIX-1023/INTEG D6: an unjudged short text is never dropped, but it is tier 0 only when it matches a
            # central concept as a long text would; otherwise tier 1, so a 3-word weak or off-topic title ("Status Quo
            # Bias") can no longer sort with the relevant records ahead of them (both cuts use these tiers).
            m = _mc(t)
            out.append(0 if m and (top <= 1 or max(w[c] for c in m) >= top) else 1)
            continue
        m = _mc(t)
        any_hit = any_hit or bool(m)
        out.append(2 if not m else 1 if top > 1 and max(w[c] for c in m) < top else 0)
    return out if any_hit else [0] * len(texts)


def part_query(text: str, entities, cfg: Mapping | None = None) -> str:
    """NONASK-1103 S2: the provider query text of ONE sub-query: the named entity's own term first, then the part's other
    terms, in the part's order. Built only from the part's own words (drift guard S4); the part's text when it has no
    terms config or no terms."""
    cfg = cfg if cfg is not None else _default_terms_cfg()
    if not cfg:
        return text
    qt = subquery_terms(text, entities, cfg)
    if qt.empty:
        return text
    words = [*qt.core, *(t for run in qt.runs for t in run)]
    words += [t for t in qt.extra if t not in words]
    return " ".join(dict.fromkeys(words)) or text
