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


def _stems(text: str, icfg: dict, n: int) -> set:
    try:
        return {t[:n] for t in tokenize(text, icfg)}
    except Exception:  # missing/odd config: fail closed
        return set()


def _q_stems(question: str, icfg: dict, n: int) -> set:
    """DAR-TRACE-1022: the question's CONTENT stems: its stems minus the stems of relevance.question_generic_terms
    (question words and generic wording such as "did", "key", "types", "different"). Absent key = all stems (the
    previous behaviour). A question left with no content stem has none (it is never grounded)."""
    gen = icfg.get("question_generic_terms") if isinstance(icfg, dict) else None
    if not isinstance(gen, list) or not gen:
        return _stems(question, icfg, n)
    try:
        # INT-1022-B (fix2 verifier must-fix 1): generic terms are dropped as WHOLE TOKENS before stemming, never by
        # stem prefix (that dropped commodity->common, imports->important, majority->major, description->describe).
        drop = {str(g).casefold() for g in gen if isinstance(g, str)}
        return {t[:n] for t in tokenize(question, icfg) if t not in drop}
    except Exception:  # odd config: fail closed (no content stems)
        return set()


def _relevance(question: str, text: str, icfg: dict, n: int) -> float:
    q = _q_stems(question, icfg, n)
    return len(q & _stems(text, icfg, n)) / len(q) if q else 0.0


def _matched(question: str, text: str, icfg: dict, n: int) -> tuple[int, int]:
    """(question content stems found in the span, question content stems)."""
    q = _q_stems(question, icfg, n)
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


def _df_table(corpus, acquired, icfg: dict, n: int) -> tuple[dict, int]:
    key = ("_fda_df", n, str(icfg.get("token_pattern")), icfg.get("min_token_len"), tuple(icfg.get("stopwords") or ()))
    cache = getattr(corpus, "__dict__", None)
    if cache is not None and key in cache.get("_fda_cache", {}):
        return cache["_fda_cache"][key]
    texts = _evidence_texts(corpus, acquired)
    df: dict = {}
    for t in texts:
        for st in _stems(t, icfg, n):
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

    q_text = _ET_AL.sub(" ", question) if weighted and scfg.get("enabled", True) else question
    et_al = bool(_ET_AL.search(question)) if weighted and scfg.get("enabled", True) else False
    selector_docs: set = set()
    q_ns: set = set()  # non-selector question stems (weighted mode)
    idf: dict = {}
    if weighted:
        q_all = _stems(q_text, icfg, stem_n)
        sel: set = set()
        if scfg.get("enabled", True):
            for d in acquired:
                hit = _surname_stems(d.get("authors"), icfg, stem_n) & q_all
                if hit:
                    sel |= hit
                    selector_docs.add(str(d["document_id"]))
        q_ns = q_all - sel
        df, n_chunks = _df_table(corpus, acquired, icfg, stem_n)
        idf = idf_weights(q_ns, df, n_chunks)
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
            sp = _stems(text, icfg, stem_n) & q_ns
            tot = sum(idf.values())
            rel = sum(idf[x] for x in sp) / tot if tot else 0.0
            hits, q_n = len(sp), len(q_ns)
        else:
            rel = _relevance(question, text, icfg, stem_n)
            hits, q_n = _matched(question, text, icfg, stem_n)
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
        q1 = _q_stems(question, icfg, stem_n)
        per = {}
        for v in relaxed.values():
            per[v] = per.get(v, 0) + 1
        kept = []
        for t in scored:
            did_r = relaxed.get(id(t[3]))
            if did_r is not None and per[did_r] < 2 and not (q1 & _stems(str(t[4].get("title") or ""), icfg, stem_n)):
                rejected.append({"span_id": t[3].get("span_id"), "document_id": t[3].get("document_id"),
                                 "reason": "low_relevance"})
            else:
                kept.append(t)
        scored = kept
    scored.sort(key=lambda t: t[:3])

    mcfg = adm.get("merge_adjacent") or {}
    merge_on = bool(mcfg.get("enabled", False))  # absent/false = no merging (pre-P16R behaviour)
    admitted, per_doc, n_chars, n_merged = [], {}, 0, 0
    for _, _, _, c, d, rel in scored:
        did_s = str(c["document_id"])
        sid = c.get("span_id")
        m = _try_merge(c, admitted, corpus, d, mcfg, n_chars, max_chars) if merge_on else None
        if m is not None:
            members, start, end, text = m
            keep = members[0]  # best-ranked member: admitted is in rank order
            n_chars += len(text) - sum(len(a["text"]) for a in members)
            ids = [x for a in members for x in a.get("merged_span_ids", (a["span_id"],))] + [sid]
            keep.update({"span_id": f"{c['document_id']}:{start}-{end}", "start": start, "end": end, "text": text,
                         "text_sha256": _sha(text), "relevance": max([rel] + [a["relevance"] for a in members]),
                         "merged_span_ids": tuple(ids)})
            admitted[:] = [a for a in admitted if a is keep or all(a is not x for x in members)]
            per_doc[did_s] = sum(1 for a in admitted if str(a["document_id"]) == did_s)
            n_merged += 1
        elif any(a["document_id"] == c["document_id"] and c["start"] < a["end"] and a["start"] < c["end"]
               for a in admitted):
            rejected.append({"span_id": sid, "document_id": c["document_id"], "reason": "overlap"})
        elif per_doc.get(did_s, 0) >= max_doc:
            rejected.append({"span_id": sid, "document_id": c["document_id"], "reason": "cap_doc"})
        elif len(admitted) >= max_total or (max_chars is not None and n_chars + len(c["text"]) > max_chars):
            rejected.append({"span_id": sid, "document_id": c["document_id"], "reason": "cap_total"})
        else:
            per_doc[did_s] = per_doc.get(did_s, 0) + 1
            n_chars += len(c["text"])
            admitted.append({
                "span_id": sid, "document_id": c["document_id"], "canonical_source_id": d.get("canonical_source_id"),
                "title": d.get("title"), "url": d.get("url"), "published": d.get("published"),
                "authors": d.get("authors"), "licence": d.get("licence"), "start": c["start"], "end": c["end"],
                "text": c["text"], "text_sha256": _sha(c["text"]), "relevance": rel,
                "licence_url": d.get("licence_url"), "attribution": d["_attribution"]})

    if weighted:
        thr = float(wcfg.get("min_weighted_coverage", 0.60))
        sal_k = int(wcfg.get("salient_k", 2))
        tot = sum(idf.values())
        salient = sorted(q_ns, key=lambda x: (-idf[x], x))[:max(sal_k, 0)]
        cover = {}
        for a in admitted:
            cover.setdefault(str(a["document_id"]), set()).update(_stems(a["text"], icfg, stem_n) & q_ns)
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
        q_stems = _q_stems(question, icfg, stem_n)
        min_cov = float(adm.get("min_question_coverage", 1.0))
        # review B2: each cited document's admitted passages must together cover the question's content terms
        cover = {}
        for a in admitted:
            cover.setdefault(str(a["document_id"]), set()).update(_stems(a["text"], icfg, stem_n) & q_stems)
        weak = {k for k, v in cover.items() if not q_stems or len(v) / len(q_stems) < min_cov}
        if weak:
            rejected.extend({"span_id": a["span_id"], "document_id": a["document_id"], "reason": "question_coverage"}
                            for a in admitted if str(a["document_id"]) in weak)
            admitted = [a for a in admitted if str(a["document_id"]) not in weak]
            per_doc = {k: v for k, v in per_doc.items() if k not in weak}

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
            "candidate_spans": n_cand.get(did_s, 0), "admitted_spans": per_doc.get(did_s, 0)})
    by_reason: dict = {}
    for r in rejected:
        by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
    return EvidenceBundle(
        question=question, examined_full_documents=tuple(examined), admitted=tuple(admitted), unusable=unusable,
        rejected=tuple(rejected), status="admitted" if admitted else "insufficient",
        corpus_source=src,
        trace={"candidates": len(cands), "admitted": len(admitted), "rejected_by_reason": by_reason,
               "examined": len(examined), "unusable": len(unusable),
               **({"merged": n_merged} if n_merged else {}),
               **({"signal": "idf_weighted", "selector_documents": sorted(selector_docs), "et_al": et_al,
                   "non_selector_stems": len(q_ns)} if weighted else {})})
