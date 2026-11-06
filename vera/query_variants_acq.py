"""NONASK-1103 A6-1: deterministic term-variant generator for the inline acquisition loop (acq6 plan s.4 step 2).

THIN WRAPPER (orchestrator note): the term-group stage is search_query.query_variants itself (it returns [] when the
flag-off lexicon yields nothing, so nothing here depends on it); this module adds only drop-terms, entity-central terms,
the config seed list, OR, de-duplication against `tried` and raw-sentence-free extraction. vera.subquery is not in
f54e195, so entity-central terms use search_query.key_concepts/anchor; swap to vera.subquery after I-S23 merges.
Pure: no LLM, no network, no clock, no randomness. Given the question it returns the finite, ordered list of search
queries to try AFTER the first search, never the raw sentence. Order (each stage de-duplicated against `tried` and
against earlier variants):
  (a) drop terms: the base AND terms minus one term per step (last term first);
  (b) the existing term-group lexicon (vera.term_expansion via vera.search_query.query_variants);
  (c) entity-central terms: vera.subquery parts/entities (S23), then the anchor run / key concepts (vera.search_query);
  (d) route-aware seed list (config/acq_variants.json, SEED/FALLBACK, owner-editable): anchor terms + one seed phrase.
      WP6 (MIDTERMS-WP6-1106): news intent (vera.news_intent, the reporting-lane detector) uses `news_terms`; every other
      question is the technical route and uses `practitioner_terms`;
  (e) the OR form of the base terms.
Terms come from vera.search_query (stopwords, meta words and filler words removed); emails, URLs and long digit strings
are removed from the question BEFORE extraction (PII filter). Not wired into the loop yet (follows I-S23).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from vera import search_query

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "acq_variants.json"
FALLBACK_SEEDS = ("best practices", "patterns", "architecture", "guide", "reference", "documentation",
                  "design patterns", "framework")
FALLBACK_NEWS_SEEDS = ("results", "analysis", "aftermath", "outcome", "reaction")
FALLBACK_MAX_TERMS = 4
_WH = frozenset({"where", "when", "which", "while", "what", "whose", "whom", "how", "why"})  # not in the shipped stopwords
_PII = re.compile(r"https?://\S+|www\.\S+|\S+@\S+|\+?\d[\d\-\s().]{6,}\d|\b\d{5,}\b", re.I)


def load_seeds(path: Path = CONFIG_PATH, *, news: bool = False) -> tuple[tuple, int]:
    """(seed phrases, max terms per query) from config for the route: `news_terms` when news=True, else
    `practitioner_terms` (the technical route). The built-in fallback when the file is absent or invalid."""
    key, fallback = ("news_terms", FALLBACK_NEWS_SEEDS) if news else ("practitioner_terms", FALLBACK_SEEDS)
    try:
        d = json.loads(Path(path).read_text())
        seeds = tuple(s.strip() for s in d[key] if isinstance(s, str) and s.strip())
        return (seeds or fallback), max(int(d.get("max_terms_per_query", FALLBACK_MAX_TERMS)), 1)
    except Exception:  # noqa: BLE001  seed/fallback: never fails the loop
        return fallback, FALLBACK_MAX_TERMS


def is_news_route(question: str) -> bool:
    """True when the question has news intent: the same detector as the reporting lane (vera.news_intent). Never
    raises (False on error, so the technical route applies)."""
    try:
        from vera import news_intent
        return bool(news_intent.news_intent(str(question or "")))
    except Exception:  # noqa: BLE001
        return False


def scrub(question: str) -> str:
    """The question with emails, URLs, phone-like and long digit strings removed (PII filter)."""
    return " ".join(_PII.sub(" ", str(question or "")).split())


def norm(query: str) -> str:
    """Canonical dedupe form: lower-case, whitespace-collapsed, OR operators order-insensitive."""
    q = " ".join(str(query or "").lower().split())
    if " or " in q:
        return " or ".join(sorted(p.strip() for p in q.split(" or ")))
    return q


def _q(terms) -> str:
    return " ".join(t for t in terms if t)


def _subquery_central(clean: str, cfg, mt: int) -> list[str]:
    """Entity-central terms from vera.subquery (S23): each part's query with its named entity first
    (search_query.part_query), then each named entity on its own. Pure, no model call; [] on any failure
    (the key_concepts anchor run below still applies)."""
    try:
        from vera import subquery
        plan = subquery.decompose(clean)
        out: list[str] = []
        for p in plan.parts:
            ents = tuple(p.entities)
            if ents or len(plan.parts) > 1:
                q = search_query.part_query(p.text, ents, cfg)
                if q and q.strip() and q.strip().lower() != clean.lower():
                    out.append(" ".join(q.split()[:3 * mt + 4]))
        out.extend(e for e in plan.entity_hits() if isinstance(e, str))
        return out
    except Exception:  # noqa: BLE001
        return []


def _drop_bare_flag() -> bool:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("drop_bare_one_word_variants") is True
    except Exception:  # noqa: BLE001
        return False


def variants(question: str, tried=(), *, terms_cfg=None, groups=None, seeds=None, max_terms=None) -> list[str]:
    """Ordered, de-duplicated query variants; [] when the question has no content terms. `tried` = queries already run
    (strings; compared in `norm` form)."""
    cfg = terms_cfg if terms_cfg is not None else search_query._default_terms_cfg()
    if not cfg:
        return []
    seed_list, mt = load_seeds(news=is_news_route(question))
    seed_list = tuple(seeds) if seeds is not None else seed_list
    mt = int(max_terms) if max_terms else mt
    clean = scrub(question)
    qt = search_query.build_query_terms(clean, cfg)
    qt = search_query.QueryTerms(qt.core, tuple(t for t in qt.extra if t not in _WH), qt.mode, tuple(tuple(t for t in r if t not in _WH) for r in qt.runs if any(t not in _WH for t in r)))
    if qt.empty or qt.mode == "fallback_raw":
        return []
    gen = search_query.filler_words(cfg) | _WH
    base = [t for t in (*qt.core, *search_query.and_terms(qt, mt)) if t not in gen]
    if not base:
        base = [t for t in qt.extra if t not in gen][:mt]
    base = list(dict.fromkeys(base))
    if not base:
        return []
    raw = norm(clean)
    seen = {norm(t) for t in tried}
    out: list[str] = []

    drop_bare = len(base) > 1 and _drop_bare_flag()

    def add(query: str, *, entity: bool = False) -> None:
        n = norm(query)
        if drop_bare and not entity and len(query.split()) == 1 and "-" not in query:
            return  # R3: a bare one-word variant ("agent", "color") matches the wrong field; entity-central queries exempt
        if n and n != raw and n not in seen and len(query.split()) <= 3 * mt + 4 \
                and not search_query.is_degenerate_query(query):  # F1-QUERY-1106: never a bare year / "us"
            seen.add(n)
            out.append(query.strip())

    # (a) drop terms (the first query, all base terms, is the loop's own first search; it is in `tried` or skipped)
    for i in range(len(base) - 1, -1, -1):
        if len(base) > 1:
            add(_q(base[:i] + base[i + 1:]))
    # (b) the term-group lexicon
    for v in search_query.query_variants(clean, groups):
        words = set(v.lower().split())
        add(_q([v] + [t for t in base if t not in words][:max(mt - 1, 1)]))
        add(v)
    # (c) entity-central terms
    for sq in _subquery_central(clean, cfg, mt):  # A6 D1: vera.subquery entity-central queries come first
        add(sq, entity=True)
    kc, w = search_query.key_concepts(clean, cfg, groups=groups)
    kc = [c for c in kc if "|" not in c and c not in gen]
    anchor = list(qt.anchor)
    if anchor:
        add(_q(anchor))
    for c in sorted(kc, key=lambda c: (-w.get(c, 1), kc.index(c)))[:mt]:
        add(c)
    # (d) practitioner seed list: the entity-central terms (anchor run, else the first two base terms) + one seed
    head = anchor[:2] or base[:2]
    if is_news_route(question):  # F1-QUERY-1106: the news seed head is the event phrase ("midterm elections")
        try:
            from vera import reporting_lane
            core = reporting_lane.event_core(clean)
            if core and core.get("phrase"):
                head = [core["phrase"]] + ([core["year"]] if core.get("year") else [])
        except Exception:  # noqa: BLE001
            pass
    for s in seed_list:
        add(_q(head + [s]))
    # (e) the OR form
    if len(base) > 1:
        add(" OR ".join(base))
    return out
