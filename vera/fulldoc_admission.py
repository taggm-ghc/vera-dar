"""D-055 Option C: evidence evaluation and admission.

admit(question, corpus, cfg) -> vera.fulldoc_evidence.EvidenceBundle. Deterministic; no LLM, network, DB or env reads.
Every retrieved candidate span is EVALUATED and either admitted or recorded in `rejected` with a reason code.
Examined (every acquired, verified document whose chunks were scored) is kept distinct from admitted (spans that
enter reasoning context).

Relevance: the same definition as DarCorpus.relevance (share of the question's content stems found in the span, stems
= first `stem_len` chars of tokens, tokenized as vera.domain_identify.tokenize does), computed without a DarCorpus
from config/fulldoc_answer.json admission.relevance (cfg["routing"] may override, for tests).
If that config is missing the relevance is 0.0 for everything (fail closed => insufficient).
"""
from __future__ import annotations

import hashlib
import inspect
import math
import re
from vera.fulldoc_evidence import EvidenceBundle

REASONS = ("not_acquired", "licence", "attribution_incomplete", "span_mismatch", "not_body_section", "excluded_section",
           "question_coverage", "salient_stem", "low_relevance", "pointer_only", "selector_mismatch", "overlap", "cap_doc",
           "cap_total")

# CAL-impl (PLAN-FDA Option A). Signal "idf_weighted" replaces the raw stem-coverage rule; the default "legacy" keeps the
# shipped behaviour until INT applies the config delta. FIXED in weighted mode, not configurable away: >= 2 matched
# non-selector stems, body-only ranges, C1 exclusions, licence/attribution gates, span re-verification.
MIN_MATCHED_STEMS_FLOOR = 2
_ET_AL = re.compile(r"\bet\.?\s+al\b\.?", re.IGNORECASE)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize(text: str, icfg: dict) -> list[str]:
    """Same rule as vera.domain_identify.tokenize (casefold, token_pattern, min_token_len, stopwords), kept local so
    this gate module imports nothing from domain routing/registry (tests/test_dar_routing.py isolation rule)."""
    pat, stop, min_len = re.compile(icfg["token_pattern"]), set(icfg.get("stopwords") or []), int(icfg["min_token_len"])
    return [t for t in pat.findall(str(text or "").casefold()) if len(t) >= min_len and t not in stop]


def _routing(cfg: dict) -> dict:
    """Scoring settings: cfg["admission"]["relevance"] (shipped), or a test override cfg["routing"]; {} = fail closed."""
    r = cfg.get("routing") if isinstance(cfg, dict) else None
    if isinstance(r, dict):
        return r
    rel = ((cfg.get("admission") or {}).get("relevance") if isinstance(cfg, dict) else None) or {}
    if not rel:
        return {}
    return {"identification": rel, "corpus": {"relevance_stem_len": rel.get("stem_len")}}


class _TermCtx:
    """IMPL-1028-E1FA7 (E1, vera.term_expansion): the term groups ONE question triggered and each group token's weight
    (the number of content stems of the group's spelled-out phrase; SECREV C1). None everywhere = feature off."""
    __slots__ = ("groups", "w", "ph")

    def __init__(self, groups, w, ph=None):
        self.groups, self.w = groups, w
        self.ph = ph or {}  # group token -> the content stems of its spelled-out phrase (matched-stem floor, F2)


def _term_ctx(question: str, cfg: dict, icfg: dict, n: int):
    """admission.term_groups.enabled exactly true AND the question triggers >= 1 group -> _TermCtx, else None."""
    if not icfg or n <= 0:
        return None
    try:
        from vera import term_expansion as te
        gs = te.triggered(question, te.groups_for(cfg))
        if not gs:
            return None
        gen = {str(g).casefold() for g in (icfg.get("question_generic_terms") or []) if isinstance(g, str)}
        ph = {g.token: {t[:n] for t in tokenize(g.spelled, icfg) if t not in gen} for g in gs}
        return _TermCtx(gs, {k: max(1, len(v)) for k, v in ph.items()}, ph)
    except Exception:  # odd seed/config: the feature stays off (today's behaviour)
        return None


def _canon(text: str, tg) -> str:
    if tg is None:
        return text
    from vera.term_expansion import canonicalize
    return canonicalize(text, tg.groups)


def _cut(t: str, n: int, tg) -> str:
    """A token's stem: its first n chars, except an E1 group token, which is kept whole (never cut to stem_len)."""
    return t if tg is not None and t in tg.w else t[:n]


def _wsum(stems, tg) -> int:
    """Weighted size of a stem set: 1 per stem, a group token its weight (tg None: plain len)."""
    return len(stems) if tg is None else sum(tg.w.get(x, 1) for x in stems)


def _floor_n(canon: set, raw: set, tg) -> int:
    """REVIEW-1028-E1FA7 F2 (SECREV C1: the weight is for relevance and coverage only): the count used by the
    matched-stem FLOOR. A plain stem counts 1; a group token counts the stems of its spelled-out phrase actually
    present in the raw (un-rewritten) text, at least 1. So the spelled-out form keeps today's count exactly and an
    alias alone ("UFC", "MMA") is ONE hit, never clearing a floor the base would not."""
    if tg is None:
        return len(canon)
    return sum(max(1, len(tg.ph.get(x, set()) & raw)) if x in tg.w else 1 for x in canon)


def _stems(text: str, icfg: dict, n: int, tg=None) -> set:
    try:
        if tg is None:
            return {t[:n] for t in tokenize(text, icfg)}
        return {_cut(t, n, tg) for t in tokenize(_canon(text, tg), icfg)}
    except Exception:  # missing/odd config: fail closed
        return set()


def coverage_ok(covered: set, q_stems: set, min_cov: float, tg=None) -> bool:
    """DDFIX-1023/REL: THE per-document question-coverage rule of the legacy signal (admission.min_question_coverage):
    the document's admitted passages cover at least `min_cov` of the question's content stems. The one place a
    later, approved threshold or rule change touches (the idf_weighted signal keeps its own rule below).
    E1 (tg): numerator and denominator are weighted (a group token counts as the stems it replaced)."""
    if tg is not None:
        return bool(q_stems) and _wsum(set(covered) & set(q_stems), tg) / _wsum(q_stems, tg) >= min_cov
    return bool(q_stems) and len(set(covered) & set(q_stems)) / len(q_stems) >= min_cov


def _q_stems(question: str, icfg: dict, n: int, tg=None) -> set:
    """DAR-TRACE-1022: the question's CONTENT stems: its stems minus the stems of relevance.question_generic_terms
    (question words and generic wording such as "did", "key", "types", "different"). Absent key = all stems (the
    previous behaviour). A question left with no content stem has none (it is never grounded)."""
    # DDFIX-1023/REL: the SAME generic idioms and words the discovery queries drop (vera.search_query.drop_generic:
    # "de facto", "status quo", "best practices", ...) are removed from the question text first, so a relevant paper is
    # not judged on whether it also says "status quo". Unchanged when nothing else would remain.
    from vera.search_query import drop_generic
    question = drop_generic(question)
    if tg is not None:  # E1: group phrases -> group token, after the idioms and before tokenizing
        question = _canon(question, tg)
    gen = icfg.get("question_generic_terms") if isinstance(icfg, dict) else None
    if not isinstance(gen, list) or not gen:
        if tg is None:
            return _stems(question, icfg, n)
        try:  # already rewritten above: tokenize directly (a second rewrite would defuse the group token)
            return {_cut(t, n, tg) for t in tokenize(question, icfg)}
        except Exception:
            return set()
    try:
        # INT-1022-B (fix2 verifier must-fix 1): generic terms are dropped as WHOLE TOKENS before stemming, never by
        # stem prefix (that dropped commodity->common, imports->important, majority->major, description->describe).
        drop = {str(g).casefold() for g in gen if isinstance(g, str)}
        return {_cut(t, n, tg) for t in tokenize(question, icfg) if t not in drop}
    except Exception:  # odd config: fail closed (no content stems)
        return set()


def _relevance(question: str, text: str, icfg: dict, n: int, tg=None) -> float:
    q = _q_stems(question, icfg, n, tg)
    if tg is not None:
        return _wsum(q & _stems(text, icfg, n, tg), tg) / _wsum(q, tg) if q else 0.0
    return len(q & _stems(text, icfg, n)) / len(q) if q else 0.0


def _matched(question: str, text: str, icfg: dict, n: int, tg=None) -> tuple[int, int]:
    """(question content stems found in the span, question content stems): the matched-stem FLOOR counts. E1 (tg):
    _floor_n (a group counts the spelled-out stems really present, at least 1; never its weight)."""
    q = _q_stems(question, icfg, n, tg)
    if tg is not None:
        return (_floor_n(q & _stems(text, icfg, n, tg), _stems(text, icfg, n), tg),
                _floor_n(q, _q_stems(question, icfg, n), tg))
    return len(q & _stems(text, icfg, n)), len(q)


def idf_weights(stems, df: dict, n_chunks: int) -> dict:
    """Smoothed IDF idf = ln((N+1)/(df+1)) + 1 over body evidence chunks. An out-of-vocabulary stem (df = 0) gets the
    MAXIMUM smoothed IDF, ln(N+1)+1, and so keeps its full weight in the denominator (never a default of 0)."""
    n = max(int(n_chunks), 0)
    return {st: math.log((n + 1) / (int(df.get(st, 0)) + 1)) + 1.0 for st in stems}


def _surname_stems(authors, icfg: dict, n: int) -> set:
    """Stems of author surnames from a manifest author field (a string such as "Gostic et al." or a list). The field is
    split on ; & and and commas; the LAST content token of each piece is its surname (a "Given Surname" piece, a bare
    surname, or one piece of a comma list). "et al." is ignored. Accepted looseness: "Surname, Given" also yields the
    given name, which only matters if the question itself contains that word."""
    if isinstance(authors, (list, tuple)):
        entries = [str(a) for a in authors]
    else:
        entries = re.split(r"\s*(?:;|&|\band\b|,)\s*", _ET_AL.sub(" ", str(authors or "")))
    out: set = set()
    for e in entries:
        toks = tokenize(_ET_AL.sub(" ", e), icfg)
        if toks:
            out.add(toks[-1][:n])
    return out


def _evidence_texts(corpus, acquired) -> list:
    """Body evidence chunk texts (the IDF universe). The real corpus provides them; otherwise evidence-range lines."""
    f = getattr(corpus, "evidence_chunk_texts", None)
    if callable(f):
        try:
            return [t for t in f() if isinstance(t, str)]
        except Exception:
            return []
    out = []
    for d in acquired:
        try:
            full = corpus.text(d["document_id"])
            for r in d.get("evidence_ranges") or ():
                out.extend(ln for ln in full[r[0]:r[1]].split("\n") if ln.strip())
        except Exception:
            continue
    return out


def _df_table(corpus, acquired, icfg: dict, n: int, tg=None) -> tuple[dict, int]:
    key = ("_fda_df", n, str(icfg.get("token_pattern")), icfg.get("min_token_len"), tuple(icfg.get("stopwords") or ()))
    if tg is not None:  # E1: df on canonicalised chunks, so a group token is in vocabulary (SECREV C1)
        key = key + (tuple(sorted(tg.w)),)
    cache = getattr(corpus, "__dict__", None)
    if cache is not None and key in cache.get("_fda_cache", {}):
        return cache["_fda_cache"][key]
    texts = _evidence_texts(corpus, acquired)
    df: dict = {}
    for t in texts:
        for st in _stems(t, icfg, n, tg):
            df[st] = df.get(st, 0) + 1
    res = (df, len(texts))
    if cache is not None:
        cache.setdefault("_fda_cache", {})[key] = res
    return res


def _call_retrieve(corpus, question: str, k: int, evidence_only: bool):
    if evidence_only:
        try:
            ok = "evidence_only" in inspect.signature(corpus.retrieve).parameters
        except (TypeError, ValueError):
            ok = False
        if ok:
            return corpus.retrieve(question, k, evidence_only=True)
    return corpus.retrieve(question, k)


def _pointer(span_text: str, pcfg: dict) -> bool:
    """POL hook: vera.fulldoc_policy.pointer_only(span_text, cfg). Fails CLOSED (True = reject) if the hook errors."""
    try:
        from vera import fulldoc_policy
        return bool(fulldoc_policy.pointer_only(span_text, pcfg))
    except Exception:
        return True


def attribution(d: dict, cfg: dict) -> dict | None:
    """CC BY attribution for a document, from its verified acquisition record only (never the model): creator, title,
    source (DOI URL, else landing URL), licence id + URL and an indication of changes. None if any part is missing:
    such a document is not citable (fail closed)."""
    doi = str(d.get("doi") or "").strip()
    source = f"https://doi.org/{doi}" if doi else str(d.get("url") or "").strip()
    authors = d.get("authors")
    authors = ", ".join(str(a) for a in authors) if isinstance(authors, (list, tuple)) else str(authors or "").strip()
    out = {"creator": authors, "title": str(d.get("title") or "").strip(), "source": source,
           "licence": str(d.get("licence") or "").strip(), "licence_url": str(d.get("licence_url") or "").strip()}
    if not all(out.values()):
        return None
    tmpl = ((cfg.get("answer") or {}).get("attribution_changes_text") if isinstance(cfg, dict) else None) or ""
    if not tmpl:
        return None
    out["changes"] = tmpl.format(format=str(d.get("fulltext_format") or "the published format"))
    return out


def _unusable(question, corpus):
    out = []
    for u in (getattr(corpus, "unusable", None) or []):
        if isinstance(u, dict):
            out.append({"document_id": u.get("document_id"), "code": u.get("code"), "reason": u.get("reason")})
    return tuple(out)


def _in_one_range(d: dict, start: int, end: int) -> bool:
    ranges = d.get("evidence_ranges")
    return isinstance(ranges, (list, tuple)) and any(
        isinstance(r, (list, tuple)) and len(r) == 2 and r[0] <= start and end <= r[1] for r in ranges)


def _stamp_holder(corpus, did_s) -> dict:
    """FRESH-AMEND-1023: the corpus's write-time record of one document (vera.fulldoc_db: corpus.stamps), shared by
    reference so a corpus_added_at backfill applies in place to the served corpus. A document without per-chunk stamps
    gets ONE document-level stamp, created lazily in the corpus's own record."""
    st_by = getattr(corpus, "stamps", None)
    if not isinstance(st_by, dict):
        st_by = {}
    h = st_by.get(did_s)
    if not isinstance(h, dict):
        h = {"corpus_added_at": None, "corpus_stamps": []}
        try:
            st_by[did_s] = h
        except Exception:  # noqa: BLE001
            pass
    if not isinstance(h.get("corpus_stamps"), list) or not h["corpus_stamps"]:
        h["corpus_stamps"] = [{"span_id": None, "document_id": did_s, "start": None, "end": None,
                               "corpus_added_at": h.get("corpus_added_at"), "basis": "document"}]
    return h


def _corpus_chunks(h, start, end) -> list:
    """The stamps of the stored chunks overlapping [start, end) (all of them when no range is given)."""
    st = h["corpus_stamps"]
    if start is None or end is None:
        return list(st)
    hit = [x for x in st if isinstance(x, dict) and (x.get("start") is None or x.get("end") is None
                                                     or (x["start"] < end and start < x["end"]))]
    return hit or list(st)


def _try_merge(c: dict, admitted: list, corpus, d: dict, mcfg: dict, n_chars: int, max_chars):
    """WED-P16-RET read-time merge: a candidate that overlaps or abuts (gap <= max_gap_chars) admitted passages of the
    SAME document becomes one contiguous passage = the exact document slice [min start, max end). Allowed only when the
    result is <= max_passage_chars, lies inside ONE body evidence range and keeps the admitted-character budget. The
    merged passage keeps the position (rank) of its best-ranked member and the highest member relevance.
    Returns (members, start, end, text) or None (no merge: the caller applies the ordinary overlap/cap rules)."""
    gap = max(int(mcfg.get("max_gap_chars", 0)), 0)
    cap = int(mcfg.get("max_passage_chars", 0))
    members = [a for a in admitted if a["document_id"] == c["document_id"]
               and c["start"] <= a["end"] + gap and a["start"] <= c["end"] + gap]
    if not members or cap <= 0:
        return None
    start = min([c["start"]] + [a["start"] for a in members])
    end = max([c["end"]] + [a["end"] for a in members])
    if end - start > cap or not _in_one_range(d, start, end):
        return None
    if any(a["document_id"] == c["document_id"] and a not in members and start < a["end"] + gap and a["start"] < end + gap
           for a in admitted):
        return None
    try:
        full = corpus.text(c["document_id"])
    except Exception:
        return None
    if not isinstance(full, str) or not 0 <= start < end <= len(full):
        return None
    extra = (end - start) - sum(a["end"] - a["start"] for a in members)
    if max_chars is not None and n_chars + extra > max_chars:
        return None
    return members, start, end, full[start:end]


def _body_text(corpus, d: dict) -> str:
    """The document's body evidence text (its evidence ranges joined), '' on any fault."""
    try:
        full = corpus.text(d["document_id"])
        return "\n".join(full[r[0]:r[1]] for r in (d.get("evidence_ranges") or ())
                         if isinstance(r, (list, tuple)) and len(r) == 2)
    except Exception:
        return ""


def topical_fit(q_stems: set, body: str, df: dict, n_chunks: int, icfg: dict, n: int, top_n: int, tg=None) -> float:
    """WP-FA7-d "aboutness" of one document: (term frequency of the question's content stems in the body) / (term
    frequency of the body's top_n stems by tf * idf over the request's evidence chunks). 0.0 for an empty body.
    Lexical only (no embedding model at answer time). Title bonus from the plan is NOT implemented (order mode only)."""
    try:
        toks = [_cut(t, n, tg) for t in tokenize(_canon(body, tg), icfg)]
    except Exception:
        return 0.0
    if not toks:
        return 0.0
    tf: dict = {}
    for t in toks:
        tf[t] = tf.get(t, 0) + 1
    idf = idf_weights(tf, df, n_chunks)
    top = sorted(tf, key=lambda t: (-tf[t] * idf[t], t))[:max(int(top_n), 1)]
    den = sum(tf[t] for t in top)
    return sum(tf.get(x, 0) for x in q_stems) / den if den else 0.0


def _alloc_parts(question: str, pa: dict, icfg: dict, tg) -> list:
    """WP-FA7 parts (REVIEW-1028-E1FA7 F3). Allocation runs ONLY for a genuine multi-part question: >= 2 facets
    (vera.question_parts: contrast and commonality), or an explicit comparison ("between X and Y", "X vs Y", "compare
    X with Y") with a facet cue. Concept parts are the compared sides mapped to key concepts (an E1 group the side
    names, else the side's key-concept words, ANY of which matches); never raw multi-word runs. Concepts first."""
    from vera import question_parts as qp
    facets = qp.facets(question, pa)
    sides = qp.comparison_sides(question, pa)
    if not facets and not sides:
        return []
    concepts: list = []
    if sides and pa.get("concepts", True) is not False:
        try:
            from vera import search_query as sq
            kc = [c for c in sq.key_concepts(question, groups=tg.groups if tg is not None else ())[0]]
        except Exception:
            kc = []
        drop = {str(g).casefold() for g in (icfg.get("question_generic_terms") or []) if isinstance(g, str)}
        for spec in qp.table(pa).values():
            drop |= {str(c).casefold() for c in spec.get("question", ()) if " " not in str(c)}
        for side in sides:
            grp = None
            if tg is not None:
                from vera.term_expansion import _has
                grp = next((g for g in tg.groups if any(_has(side, t) for t, _a in g.phrases)), None)
            if grp is not None:
                concepts.append("|".join(t for t, _a in grp.phrases))
                continue
            words = {w.casefold() for w in re.findall(r"[^\W\d_][\w\-]*", side)}
            members = [c for c in kc if "|" not in c and c.casefold() in words and c.casefold() not in drop]
            if members:
                concepts.append("|".join(members))
    cparts = qp.concept_parts(concepts) if len(concepts) >= 2 else []
    return cparts + facets if (cparts or facets) else []


def _fits(text: str, parts: list) -> list:
    """Per part: does the passage serve it? A facet needs a part-specific cue AND >= 1 concept match (REPLAY-1028:
    a weak cue on an off-topic passage must not satisfy a facet); with no concept parts the passage's own relevance
    pass (>= 2 matched question stems) is that concept match."""
    from vera import question_parts as qp
    conc = [p for p in parts if p.kind == "concept"]
    has_concept = (not conc) or any(qp.part_fit(text, p) >= 1 for p in conc)
    return [qp.part_fit(text, p) >= 1 and (p.kind == "concept" or has_concept) for p in parts]


def admit(question: str, corpus, cfg: dict) -> EvidenceBundle:
    question = str(question or "")
    cfg = cfg if isinstance(cfg, dict) else {}
    acquired = list(getattr(corpus, "acquired", None) or []) if corpus is not None else []
    acquired = [d for d in acquired if isinstance(d, dict) and d.get("document_id") is not None]
    unusable = _unusable(question, corpus) if corpus is not None else ()
    src = getattr(corpus, "source_kind", None)
    src = src if src in ("db", "bundle") else None
    if corpus is None or getattr(corpus, "status", None) != "ok" or not acquired:
        return EvidenceBundle(question=question, unusable=unusable, status="corpus_unusable",
                              trace={"candidates": 0, "admitted": 0, "rejected_by_reason": {}, "examined": 0,
                                     "unusable": len(unusable)})
    docs = {str(d["document_id"]): d for d in acquired}
    adm = cfg.get("admission") or {}
    min_rel = float(adm.get("min_relevance", 0.4))
    min_hits = int(adm.get("min_matched_stems", 2))  # absolute floor: one shared word never grounds an answer
    adaptive = adm.get("adaptive_min_matched_stems") is True  # DAR-TRACE-1022 (exactly true; absent = off)
    max_doc = int(adm.get("max_spans_per_doc", 2))
    max_total = int(adm.get("max_spans_total", 6))
    max_chars = adm.get("max_admitted_chars")  # WED-P16-RET: total admitted characters (absent = no character budget)
    max_chars = None if max_chars is None else int(max_chars)
    k = int((cfg.get("retrieval") or {}).get("candidate_k", 12))
    weighted = str(adm.get("signal", "legacy")) == "idf_weighted"
    wcfg = adm.get("idf_signal") or {}
    pcfg = adm.get("pointer_filter") or {}
    use_pointer = bool(pcfg.get("enabled", False))
    scfg = adm.get("metadata_selectors") or {}
    routing = _routing(cfg)
    icfg = routing.get("identification") or {}
    try:
        stem_n = int((routing.get("corpus") or {})["relevance_stem_len"])
    except (KeyError, TypeError, ValueError):
        stem_n = 0
        icfg = {}  # no usable relevance config: every relevance is 0.0
    gv2 = adm.get("generic_terms_v2")  # IMPL-1028-E1FA7 / DIAG-ANS3 fix 6: extra generic question words (default off)
    if isinstance(gv2, dict) and gv2.get("enabled") is True and icfg:
        icfg = dict(icfg, question_generic_terms=list(icfg.get("question_generic_terms") or [])
                    + [str(x) for x in (gv2.get("add") or []) if isinstance(x, str)])
    tg = _term_ctx(question, cfg, icfg, stem_n)  # E1 term groups: None = off or no group in the question

    q_text = _ET_AL.sub(" ", question) if weighted and scfg.get("enabled", True) else question
    et_al = bool(_ET_AL.search(question)) if weighted and scfg.get("enabled", True) else False
    selector_docs: set = set()
    q_ns: set = set()  # non-selector question stems (weighted mode)
    idf: dict = {}
    if weighted:
        q_all = _stems(q_text, icfg, stem_n, tg)
        sel: set = set()
        if scfg.get("enabled", True):
            for d in acquired:
                hit = _surname_stems(d.get("authors"), icfg, stem_n) & q_all
                if hit:
                    sel |= hit
                    selector_docs.add(str(d["document_id"]))
        q_ns = q_all - sel
        q_raw = _stems(q_text, icfg, stem_n) if tg is not None else q_ns  # F2: floor counts on the raw question
        df, n_chunks = _df_table(corpus, acquired, icfg, stem_n, tg)
        idf = idf_weights(q_ns, df, n_chunks)
        if tg is not None:  # E1: a group token weighs as the stems it replaced
            idf = {x: v * tg.w.get(x, 1) for x, v in idf.items()}
        min_rel = float(wcfg.get("min_relevance", 0.30))
        min_hits = max(MIN_MATCHED_STEMS_FLOOR, min_hits)  # fixed floor, never configurable away
    restrict = bool(selector_docs) and bool(scfg.get("restrict_candidates", True))

    try:
        cands = list(_call_retrieve(corpus, question, k, bool((cfg.get("retrieval") or {}).get("evidence_only"))) or [])
    except Exception:
        cands = []
    try:
        inspected = corpus.inspection() or {}
    except Exception:
        inspected = {}

    rejected, scored, n_cand = [], [], {}
    relaxed: dict = {}  # id(candidate) -> document id, for spans that passed only under the 1-stem relaxed floor
    for c in cands:
        if not isinstance(c, dict):
            rejected.append({"span_id": None, "document_id": None, "reason": "span_mismatch"})
            continue
        sid, did = c.get("span_id"), c.get("document_id")
        did_s = str(did) if did is not None else None

        def rej(reason, sid=sid, did=did):
            rejected.append({"span_id": sid, "document_id": did, "reason": reason})
        d = docs.get(did_s) if did_s is not None else None
        if d is None:
            rej("not_acquired")
            continue
        n_cand[did_s] = n_cand.get(did_s, 0) + 1
        if d.get("licence_decision") != "allow":
            rej("licence")
            continue
        attr = attribution(d, cfg)
        if attr is None:  # not citable without complete attribution data
            rej("attribution_incomplete")
            continue
        text, start, end = c.get("text"), c.get("start"), c.get("end")
        ok = isinstance(text, str) and bool(text) and isinstance(start, int) and isinstance(end, int) \
            and not isinstance(start, bool) and not isinstance(end, bool)
        if ok:
            try:
                full = corpus.text(did)
                ok = isinstance(full, str) and 0 <= start < end <= len(full) and full[start:end] == text
            except Exception:
                ok = False
        if not ok:
            rej("span_mismatch")
            continue
        ranges = d.get("evidence_ranges")  # body sections only: never title/abstract/back matter (review B1)
        if not isinstance(ranges, (list, tuple)) or not any(
                isinstance(r, (list, tuple)) and len(r) == 2 and r[0] <= start and end <= r[1] for r in ranges):
            exr = d.get("excluded_ranges")  # C1: inside a recorded excluded range => the more specific sub-reason
            if isinstance(exr, (list, tuple)) and any(
                    isinstance(x, dict) and x.get("start", 0) < end and start < x.get("end", 0) for x in exr):
                rej("excluded_section")
            else:
                rej("not_body_section")
            continue
        if restrict and did_s not in selector_docs:  # the question names an acquired document's author
            rej("selector_mismatch")
            continue
        if weighted:
            sp = _stems(text, icfg, stem_n, tg) & q_ns
            tot = sum(idf.values())
            rel = sum(idf[x] for x in sp) / tot if tot else 0.0
            hits, q_n = _floor_n(sp, _stems(text, icfg, stem_n), tg), _floor_n(q_ns, q_raw, tg)
        else:
            rel = _relevance(question, text, icfg, stem_n, tg)
            hits, q_n = _matched(question, text, icfg, stem_n, tg)
        # review B2: no relaxation for one-word questions, UNLESS admission.adaptive_min_matched_stems is true
        # (DAR-TRACE-1022, orchestrator decision; calibration note work/dar-trace-1022/CALIBRATION.md): then the floor
        # is min(min_matched_stems, the question's content stems), never below 1, and only in the legacy signal (the
        # weighted signal keeps its fixed MIN_MATCHED_STEMS_FLOOR).
        need = max(1, min(min_hits, q_n)) if (adaptive and not weighted) else min_hits
        if rel < min_rel or hits < need or q_n == 0:
            rej("low_relevance")
            continue
        if use_pointer and _pointer(text, pcfg):  # POL hook: a bare cross-reference / reference list is not evidence
            rej("pointer_only")
            continue
        try:
            score = float(c.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        scored.append((-rel, -score, str(sid), c, dict(d, _attribution=attr), rel))
        if adaptive and not weighted and q_n == 1 and need < min_hits:
            relaxed[id(c)] = did_s
    if relaxed:
        # INT-1022-B (fix2 verifier must-fix 2): a ONE-content-stem question admitted under the relaxed floor needs
        # >= 2 passing spans in the same document, or that stem in the document's title; a single passing mention
        # (e.g. "horse" in an office-worker diet study) is not enough.
        q1 = _q_stems(question, icfg, stem_n, tg)
        per = {}
        for v in relaxed.values():
            per[v] = per.get(v, 0) + 1
        kept = []
        for t in scored:
            did_r = relaxed.get(id(t[3]))
            if did_r is not None and per[did_r] < 2 and not (q1 & _stems(str(t[4].get("title") or ""), icfg, stem_n, tg)):
                rejected.append({"span_id": t[3].get("span_id"), "document_id": t[3].get("document_id"),
                                 "reason": "low_relevance"})
            else:
                kept.append(t)
        scored = kept
    scored.sort(key=lambda t: t[:3])
    extra_trace: dict = {}
    tfc = adm.get("topical_fit") if isinstance(adm.get("topical_fit"), dict) else {}
    if tfc.get("order") is True and scored:
        # WP-FA7-d ORDER mode only (never a gate): documents in a lower fit band fill only leftover slots.
        lo, hi = tfc.get("fit_lo"), tfc.get("fit_hi")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and not isinstance(lo, bool) \
                and not isinstance(hi, bool):
            dft, nct = _df_table(corpus, acquired, icfg, stem_n, tg)
            qs = q_ns if weighted else _q_stems(question, icfg, stem_n, tg)
            fit = {k2: topical_fit(qs, _body_text(corpus, docs[k2]), dft, nct, icfg, stem_n,
                                   int(tfc.get("top_n") or 20), tg)
                   for k2 in {str(t[3]["document_id"]) for t in scored}}
            band = {k2: 0 if v >= hi else 1 if v >= lo else 2 for k2, v in fit.items()}
            scored.sort(key=lambda t: (band[str(t[3]["document_id"])],) + t[:3])
            extra_trace["topical_fit"] = {"bands": [sum(1 for v in band.values() if v == b) for b in (0, 1, 2)]}
        else:
            extra_trace["topical_fit"] = "unset"  # order mode needs calibrated fit_lo and fit_hi (plan s.6)
    pa = adm.get("part_allocation") if isinstance(adm.get("part_allocation"), dict) else {}
    parts = _alloc_parts(question, pa, icfg, tg) if pa.get("enabled") is True else []

    mcfg = adm.get("merge_adjacent") or {}
    merge_on = bool(mcfg.get("enabled", False))  # absent/false = no merging (pre-P16R behaviour)
    admitted, per_doc, n_chars, n_merged = [], {}, 0, 0
    rk: dict = {}  # id(admitted entry) -> rank position (allocation only: admitted stays in rank order)

    def place(pos, t, record: bool) -> bool:
        """The existing fill rules for one scored span. record=False (allocation pass 1): nothing is rejected and a
        span that would merge is left to pass 2, so pass 1 can only ever ADD spans that pass every normal rule."""
        nonlocal n_chars, n_merged
        _, _, _, c, d, rel = t
        did_s = str(c["document_id"])
        sid = c.get("span_id")
        m = _try_merge(c, admitted, corpus, d, mcfg, n_chars, max_chars) if merge_on else None
        if m is not None and not record:
            return False
        if m is not None:
            members, start, end, text = m
            keep = members[0]  # best-ranked member: admitted is in rank order
            n_chars += len(text) - sum(len(a["text"]) for a in members)
            ids = [x for a in members for x in a.get("merged_span_ids", (a["span_id"],))] + [sid]
            keep.update({"span_id": f"{c['document_id']}:{start}-{end}", "start": start, "end": end, "text": text,
                         "corpus_chunks": _corpus_chunks(_stamp_holder(corpus, did_s), start, end),
                         "text_sha256": _sha(text), "relevance": max([rel] + [a["relevance"] for a in members]),
                         "merged_span_ids": tuple(ids)})
            admitted[:] = [a for a in admitted if a is keep or all(a is not x for x in members)]
            per_doc[did_s] = sum(1 for a in admitted if str(a["document_id"]) == did_s)
            n_merged += 1
            return True
        reason = None
        if any(a["document_id"] == c["document_id"] and c["start"] < a["end"] and a["start"] < c["end"]
               for a in admitted):
            reason = "overlap"
        elif per_doc.get(did_s, 0) >= max_doc:
            reason = "cap_doc"
        elif len(admitted) >= max_total or (max_chars is not None and n_chars + len(c["text"]) > max_chars):
            reason = "cap_total"
        if reason is not None:
            if record:
                rejected.append({"span_id": sid, "document_id": c["document_id"], "reason": reason})
            return False
        per_doc[did_s] = per_doc.get(did_s, 0) + 1
        n_chars += len(c["text"])
        entry = {
            "span_id": sid, "document_id": c["document_id"], "canonical_source_id": d.get("canonical_source_id"),
            "title": d.get("title"), "url": d.get("url"), "published": d.get("published"),
            "authors": d.get("authors"), "licence": d.get("licence"), "start": c["start"], "end": c["end"],
            "text": c["text"], "text_sha256": _sha(c["text"]), "relevance": rel,
            "licence_url": d.get("licence_url"), "attribution": d["_attribution"],
            "corpus_added_at": _stamp_holder(corpus, did_s).get("corpus_added_at"),
            "corpus_chunks": _corpus_chunks(_stamp_holder(corpus, did_s), c["start"], c["end"])}
        admitted.append(entry)
        if parts:
            rk[id(entry)] = pos
            admitted.sort(key=lambda a: rk.get(id(a), 0))
        return True

    done: set = set()
    quota = 0
    if parts:
        # WP-FA7 pass 1: each part's quota (max_spans_total / parts, at least 1) from its fitting spans in rank
        # order, under the unchanged overlap, per-document and character rules; spans already admitted that serve
        # a part count toward its quota. Only spans that passed every relevance rule are candidates (never a floor
        # bypass).
        quota = max(1, max_total // len(parts))
        fit_of = {id(t[3]): _fits(t[3]["text"], parts) for t in scored}
        for pi, p in enumerate(parts):
            have = sum(1 for a in admitted if _fits(a["text"], parts)[pi])
            for pos, t in enumerate(scored):
                if have >= quota:
                    break
                if id(t[3]) in done or not fit_of[id(t[3])][pi]:
                    continue
                if place(pos, t, False):
                    done.add(id(t[3]))
                    have += 1
    for pos, t in enumerate(scored):  # pass 2 (the only pass when allocation is off): global rank
        if id(t[3]) not in done:
            place(pos, t, True)

    if weighted:
        thr = float(wcfg.get("min_weighted_coverage", 0.60))
        sal_k = int(wcfg.get("salient_k", 2))
        tot = sum(idf.values())
        salient = sorted(q_ns, key=lambda x: (-idf[x], x))[:max(sal_k, 0)]
        cover = {}
        for a in admitted:
            cover.setdefault(str(a["document_id"]), set()).update(_stems(a["text"], icfg, stem_n, tg) & q_ns)
        weak, why = set(), {}
        for did_s, v in cover.items():
            if not tot or sum(idf[x] for x in v) / tot < thr:
                weak.add(did_s)
                why[did_s] = "question_coverage"
            elif any(x not in v for x in salient):
                weak.add(did_s)
                why[did_s] = "salient_stem"
        if weak:
            rejected.extend({"span_id": a["span_id"], "document_id": a["document_id"],
                             "reason": why[str(a["document_id"])]} for a in admitted if str(a["document_id"]) in weak)
            admitted = [a for a in admitted if str(a["document_id"]) not in weak]
            per_doc = {k: v for k, v in per_doc.items() if k not in weak}
    else:
        q_stems = _q_stems(question, icfg, stem_n, tg)
        min_cov = float(adm.get("min_question_coverage", 1.0))
        # review B2: each cited document's admitted passages must together cover the question's content terms
        cover = {}
        for a in admitted:
            cover.setdefault(str(a["document_id"]), set()).update(_stems(a["text"], icfg, stem_n, tg) & q_stems)
        weak = {k for k, v in cover.items() if not coverage_ok(v, q_stems, min_cov, tg)}
        if weak:
            rejected.extend({"span_id": a["span_id"], "document_id": a["document_id"], "reason": "question_coverage"}
                            for a in admitted if str(a["document_id"]) in weak)
            admitted = [a for a in admitted if str(a["document_id"]) not in weak]
            per_doc = {k: v for k, v in per_doc.items() if k not in weak}

    unsupported: tuple = ()
    if parts:
        served = [sum(1 for a in admitted if _fits(a["text"], parts)[i]) for i in range(len(parts))]
        unsupported = tuple(p.label for p, k2 in zip(parts, served) if k2 == 0)
        extra_trace["parts"] = {"n": len(parts), "quota": quota, "admitted_by_part": served,
                                "unsupported": len(unsupported)}
    if tg is not None:
        extra_trace["term_groups"] = len(tg.groups)
    examined = []
    for d in acquired:
        did_s = str(d["document_id"])
        ins = inspected.get(d["document_id"], inspected.get(did_s)) if isinstance(inspected, dict) else None
        n_chunks = ins.get("chunks_inspected") if isinstance(ins, dict) else None
        examined.append({
            "document_id": d["document_id"], "canonical_source_id": d.get("canonical_source_id"),
            "title": d.get("title"), "url": d.get("url"), "doi": d.get("doi"), "published": d.get("published"),
            "licence": d.get("licence"), "licence_decision": d.get("licence_decision"),
            "text_sha256": d.get("text_sha256"), "chars": d.get("chars"),
            "chunks_inspected": n_chunks if isinstance(n_chunks, int) else 0,
            "candidate_spans": n_cand.get(did_s, 0), "admitted_spans": per_doc.get(did_s, 0),
            "corpus_added_at": _stamp_holder(corpus, did_s).get("corpus_added_at"),
            "corpus_chunks": _corpus_chunks(_stamp_holder(corpus, did_s), None, None)})
    by_reason: dict = {}
    for r in rejected:
        by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
    return EvidenceBundle(
        question=question, examined_full_documents=tuple(examined), admitted=tuple(admitted), unusable=unusable,
        rejected=tuple(rejected), status="admitted" if admitted else "insufficient",
        corpus_source=src, parts_unsupported=unsupported,
        trace={"candidates": len(cands), "admitted": len(admitted), "rejected_by_reason": by_reason,
               "examined": len(examined), "unusable": len(unusable),
               **({"merged": n_merged} if n_merged else {}),
               **({"signal": "idf_weighted", "selector_documents": sorted(selector_docs), "et_al": et_al,
                   "non_selector_stems": len(q_ns)} if weighted else {}),
               **extra_trace})
