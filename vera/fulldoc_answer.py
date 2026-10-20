"""D-055 Option C step 3: grounded generation over admitted spans, claim verification, response projection.

Fails closed: no model call unless the evidence is grounded AND the claim check is enabled; an answer is returned
only after fix_citations (>= 1 valid citation) and claim_check.check_answer. Span text is internal: the public
projection (public_evidence) carries only short verbatim quotes (<= answer.quote_max_chars) in citations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from vera.ask_service import AskResult

_CITE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_WORD = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+")
_TRAILING_CITES = re.compile(r"([.!?])[ \t]*((?:\[\d+(?:\s*,\s*\d+)*\][ \t]*)+)(?=\s|$)")
_SEG = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(*#-])|\n+")  # the segmentation claim_check.split_claims uses
_DOC_FIELDS = ("document_id", "canonical_source_id", "title", "url", "doi", "published", "licence", "licence_decision",
               "text_sha256", "chars", "chunks_inspected", "candidate_spans", "admitted_spans")
_ADM_FIELDS = ("span_id", "document_id", "canonical_source_id", "title", "url", "licence", "relevance", "text_sha256")


@dataclass(frozen=True)
class FullDocOutcome:
    result: AskResult | None
    fallback: str | None
    tokens_used: int = 0
    cost_usd: float = 0.0


def sources_from(ev) -> list[dict]:
    """Numbered sources over the admitted spans. `snippet` is internal (prompt + claim check) and never public."""
    return [{"n": i, "title": a.get("title"), "url": a.get("url"), "published": a.get("published"),
             "snippet": a["text"], "source_type": "full_text_passage", "stored_text_kind": "full_text",
             "admission_passed": True, "span": f"chars={a['start']}-{a['end']}", "start": a["start"], "span_id": a["span_id"],
             "document_id": a.get("document_id"), "canonical_source_id": a.get("canonical_source_id"),
             "licence": a.get("licence"), "licence_url": a.get("licence_url"), "attribution": a.get("attribution"),
             "relevance": a.get("relevance")} for i, a in enumerate(ev.admitted, 1)]


def _format_sources(sources: list[dict]) -> str:
    return "\n\n".join(f"[{s['n']}] {s['title']} (published {s.get('published') or 'unknown'}) {s['url']}\n{s['snippet']}"
                       for s in sources)


def per_part_enabled(cfg: dict) -> bool:
    """D8 = A switch (rollback lever): per-part answering needs answer.per_part.enabled AND its prompt."""
    pp = (cfg.get("answer") or {}).get("per_part") or {}
    return bool(pp.get("enabled") and pp.get("system_prompt"))


def messages_for(question: str, sources: list[dict], cfg: dict) -> list[dict]:
    if per_part_enabled(cfg):
        a = cfg["answer"]
        system = a["per_part"]["system_prompt"].format(sources=_format_sources(sources) or "(none)",
                                                       insufficient_marker=a.get("insufficient_marker", ""),
                                                       max_parts=int(a.get("max_parts", 4)))
        return [{"role": "system", "content": system}, {"role": "user", "content": question}]
    system = cfg["answer"]["system_prompt"].format(sources=_format_sources(sources) or "(none)",
                                                   insufficient_marker=cfg["answer"].get("insufficient_marker", ""))
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def _words(text: str) -> set:
    return set(_WORD.findall((text or "").lower()))


def _trim(text: str, cap: int) -> str:
    text = text.strip()
    if len(text) <= cap:
        return text
    cut = text[:cap]
    if not text[cap].isspace():  # mid-word cut: back off to the last word boundary
        sp = cut.rfind(" ")
        if sp > 0:
            cut = cut[:sp]
    return cut.rstrip()


def pick_quote(span_text: str, claim_sentence: str, cap: int) -> str:
    """Short verbatim substring of span_text: the sentence with the highest word overlap with the claim, trimmed to
    cap characters at a word boundary. Never the whole span when the span is longer than cap."""
    claim_w = _words(_CITE.sub(" ", claim_sentence))
    best, best_score = None, -1
    pos = 0
    for part in _SENT.split(span_text):
        i = span_text.find(part, pos)
        if i < 0 or not part.strip():
            continue
        pos = i + len(part)
        score = len(claim_w & _words(part))
        if score > best_score:
            best, best_score = part, score
    quote = _trim(best if best is not None else span_text, cap)
    if not quote:
        quote = span_text.strip()[:cap].rstrip()
    assert quote in span_text
    return quote


def build_citations(records, sources: list[dict], cap: int, total_cap: int | None = None, *, profiles=None) -> list[dict]:
    """One citation per (kept claim, cited passage), each with the document's CC BY attribution (from the acquisition
    record, never the model; a passage without it is never cited) and a short verbatim quote (<= cap chars). Quotes
    are budgeted per response (<= total_cap chars): citations ranked lowest (partial before supported, then lower
    passage relevance, then later claims) lose their quote first but keep their span reference. With `profiles` (the
    verbatim Profiles of the kept claim sentences, C2) the budget is the union count of generated verbatim text plus
    quotes and the total must stay strictly below total_cap."""
    by_n = {s["n"]: s for s in sources}
    out = []
    for r in records:
        if r["verdict"] not in ("supported", "partial"):
            continue
        for n in r["citations"]:
            s = by_n.get(n)
            if s is None or not isinstance(s.get("attribution"), dict):
                continue
            q = pick_quote(s["snippet"], r["sentence"], cap)
            at = s["start"] + s["snippet"].find(q)
            out.append({"claim_id": r["id"], "n": n, "span_id": s["span_id"], "document_id": s["document_id"],
                        "canonical_source_id": s["canonical_source_id"], "licence": s["licence"], "span": s["span"],
                        "verdict": r["verdict"], "quote": q, "quote_span": f"chars={at}-{at + len(q)}",
                        "attribution": dict(s["attribution"]), "_rank": (r["verdict"] != "supported",
                                                                         -float(s.get("relevance") or 0.0), r["id"], n)})
    if total_cap is not None and profiles is not None:
        from vera.fulldoc_verbatim import combined_total

        accepted: list = []
        for c in sorted(out, key=lambda c: c["_rank"]):
            if combined_total(profiles, accepted + [c["quote"]])[0] < total_cap:
                accepted.append(c["quote"])
            else:  # over the verbatim budget: keep the span reference, drop the quote
                c["quote"], c["quote_span"] = None, None
    elif total_cap is not None:
        used = 0
        for c in sorted(out, key=lambda c: c["_rank"]):
            if used + len(c["quote"]) <= total_cap:
                used += len(c["quote"])
            else:  # over the per-response quote budget: keep the span reference, drop the quote
                c["quote"], c["quote_span"] = None, None
    for c in out:
        c.pop("_rank")
    return out


def attach_trailing_citations(text: str) -> str:
    """'...error. [1] Next' -> '...error [1]. Next': claim_check splits before '[', so a citation written after the full
    stop would otherwise become a claim of its own ('[1]') and its sentence would go unchecked."""
    return _TRAILING_CITES.sub(lambda m: " " + m.group(2).strip() + m.group(1), text or "")


def _copies_passage(sentence: str, sources: list[dict], cap: int, step: int = 10) -> bool:
    """True if the sentence reproduces more than `cap` consecutive characters of an admitted passage (checked on
    windows of cap+1 chars every `step` chars, plus the passage tail)."""
    if len(sentence) <= cap:
        return False
    for s in sources:
        t = s.get("snippet") or ""
        starts = list(range(0, max(0, len(t) - cap), step)) + ([len(t) - cap - 1] if len(t) > cap else [])
        if any(t[i:i + cap + 1] in sentence for i in starts):
            return True
    return False


_NAME_STOP = frozenset("and with from that this the for van von der den del della dos das jr sr phd md".split())
_META_KEYS = ("title", "doi", "journal", "container_title", "creator", "licence", "licence_url", "source", "url", "changes")


def _metadata_strings(*dict_groups) -> list:
    """Document metadata that never counts as verbatim reuse: title, authors, DOI, journal, licence, our attribution."""
    out: list = []
    for group in dict_groups:
        for d in group or ():
            if not isinstance(d, dict):
                continue
            for k in _META_KEYS:
                if isinstance(d.get(k), str):
                    out.append(d[k])
            au = d.get("authors")
            for a in (au if isinstance(au, (list, tuple)) else [au]):
                if isinstance(a, str):
                    out.append(a)
                    # surnames / name parts only: capitalised, >= 4 letters, not a common word ("and", "the", ...)
                    out += [w for w in re.findall(r"[^\W\d_]{4,}", a) if w[0].isupper() and w.lower() not in _NAME_STOP]
            at = d.get("attribution")
            if isinstance(at, dict):
                out += [v for v in at.values() if isinstance(v, str)]
    return out


def _bundle_reference(ev, sources, cfg, reference_texts=None):
    """(reference texts, metadata strings): the FULL text of every bundle document when the corpus is configured
    (or `reference_texts` is supplied); else only the admitted passages (the weaker fallback, used by unit tests)."""
    acquired: list = []
    texts = None
    if reference_texts is not None:
        texts = list(reference_texts.values() if isinstance(reference_texts, dict) else reference_texts)
    elif isinstance(cfg, dict) and cfg.get("bundle_dir_env"):
        try:
            from vera import fulldoc_corpus

            c = fulldoc_corpus.get_corpus(cfg)
            if c is not None and c.status == fulldoc_corpus.STATUS_OK:
                acquired = list(c.acquired)
                texts = [c.text(a["document_id"]) for a in acquired]
        except Exception:  # noqa: BLE001  (fail closed to the admitted passages: still a check, never none)
            texts = None
    texts = list(texts or [])
    texts += [s["snippet"] for s in sources]
    return texts, _metadata_strings(acquired, getattr(ev, "examined_full_documents", ()), sources)


def _rank_worst(r, by_n):
    rel = max([float(by_n[n].get("relevance") or 0.0) for n in r["citations"] if n in by_n] or [0.0])
    return (r["verdict"] == "partial", -rel, r["id"])


def enforce_verbatim(records, sources, checker, cfg):
    """C2 enforcement steps 1 and 3. Returns (kept records, {id: Profile}, dropped count).
    1. a sentence with a verbatim run > quote_max_chars is dropped;
    3. while generated verbatim chars alone reach the cap, drop the lowest-ranked verbatim-heavy sentence.
    (Step 2, quotes filling the rest, is build_citations; step 4, nothing left, is the caller's fallback.)"""
    cap = int(cfg["answer"]["quote_max_chars"])
    total_cap = int(cfg["answer"].get("max_total_quote_chars") or 700)
    by_n = {s["n"]: s for s in sources}
    kept, profs, dropped = [], {}, 0
    for r in records:
        if r.get("verdict") not in ("supported", "partial"):
            continue
        if _copies_passage(r["sentence"], sources, cap):  # short-quote rule on the admitted passages
            dropped += 1
            continue
        if checker is not None:
            p = checker.profile(r["sentence"])
            if p.max_run > cap:
                dropped += 1
                continue
            profs[r["id"]] = p
        kept.append(r)
    while checker is not None and kept and sum(profs[r["id"]].counted for r in kept) >= total_cap:
        heavy = [r for r in kept if profs[r["id"]].counted > 0]
        worst = max(heavy, key=lambda r: _rank_worst(r, by_n))
        kept.remove(worst)
        dropped += 1
    return kept, profs, dropped


def checked_only(draft: str, records, sources, cc_cfg: dict, cfg: dict, summary: dict, keep_ids=None,
                 dropped: int | None = None) -> tuple[str | None, dict]:
    """The published answer is rebuilt from VERIFIED claims only (supported, or partial with its marker), in draft
    order. claim_check judges only sentences that cite a passage, so any uncited sentence of the draft (an unchecked
    claim) is dropped, never shown; a note says so. None when no verified claim remains."""
    from vera.claim_check import split_claims

    claims = split_claims(draft, len(sources), 10 ** 6)  # every cited segment, uncapped, to count what is dropped
    segments = [p.strip() for p in _SEG.split(draft or "") if re.search(r"\w", p or "")]
    cap = int(cfg["answer"]["quote_max_chars"])
    kept, copied = [], 0
    for r in records:
        if r.get("verdict") not in ("supported", "partial"):
            continue
        if keep_ids is not None:
            if r["id"] not in keep_ids:
                continue
        elif _copies_passage(r["sentence"], sources, cap):  # short-quote rule: no long verbatim passage in the answer
            copied += 1
            continue
        kept.append(r["sentence"] + (cc_cfg["partial_marker"] if r["verdict"] == "partial" else ""))
    summary = dict(summary, verbatim_dropped=copied if dropped is None else dropped)
    if not kept:
        return None, summary
    uncited = max(0, len(segments) - len(claims))
    unchecked = max(0, len(claims) - len(records))  # cited but beyond max_claims: not checked, not shown
    notes = [cc_cfg["removed_note"]] if summary.get("removed") else []
    if uncited or unchecked:
        notes.append(cfg["answer"]["unchecked_note"])
    text = "\n\n".join(kept + notes)
    return text, dict(summary, uncited_dropped=uncited, unchecked_dropped=unchecked)


_LABEL = re.compile(r"(?im)^[ \t>*_`#-]*P\s*(\d+)[ \t*_`]*[:.)\u2013-][ \t*_`]*")
_HDR_PARTS = re.compile(r"(?im)^[ \t>*_`#]*parts[ \t*_`]*:[ \t*_`]*$")
_HDR_ANS = re.compile(r"(?im)^[ \t>*_`#]*answers[ \t*_`]*:[ \t*_`]*$")


def _labelled(text: str) -> dict:
    """{k: body} for the first occurrence of each 'Pk:' label."""
    ms = list(_LABEL.finditer(text))
    out: dict = {}
    for i, m in enumerate(ms):
        k = int(m.group(1))
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        out.setdefault(k, text[m.end():end].strip())
    return out


def is_marker(text: str, marker: str) -> bool:
    """A part's answer IS the per-part marker (it leads the part's text); a marker word buried in a supported
    part's text does not drop anything (D8 replaces the whole-draft substring rule)."""
    t = (text or "").strip().strip("*_`.\"' ")
    return bool(marker) and t.startswith(marker) and len(t) <= len(marker) + 30


def parse_parts(reply: str, question: str, marker: str):
    """-> (parts [(k, text)], answers {k: str | None (marker)}, structured). A reply without a parsable
    Parts:/Answers: structure is treated as a single part P1 holding the whole reply."""
    reply = reply or ""
    mp, ma = _HDR_PARTS.search(reply), _HDR_ANS.search(reply)
    if mp and ma and ma.start() > mp.start():
        restated, ans = _labelled(reply[mp.end():ma.start()]), _labelled(reply[ma.end():])
        if restated:
            keys = sorted(restated)
            answers = {k: (None if is_marker(ans[k], marker) else (ans[k] or None)) for k in keys if k in ans}
            return [(k, restated[k] or question) for k in keys], answers, True
    whole = None if is_marker(reply, marker) else (reply.strip() or None)
    return [(1, question)], {1: whole}, False


def _clip(t: str, n: int = 160) -> str:
    t = " ".join((t or "").split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "\u2026"


def _answer_parts(question, res, ev, sources, cfg, grounding_cfg, cc, check_call, reference_texts, fall, totals):
    """D8 = A: per-part answering. A part is answered only with >= 1 kept `supported` claim; every other part is
    explicitly marked; >= 1 answered part is required (D7); a partly answered question is never presented as full."""
    from vera.claim_check import RECORDS, check_answer
    from vera.fulldoc_policy import require_supported_claim
    from vera.grounded_ask import count_valid_citations, fix_citations

    a, fb = cfg["answer"], cfg["fallback_tokens"]
    pp = a["per_part"]
    marker = a.get("insufficient_marker") or ""
    mi_token = fb.get("model_insufficient") or fb["insufficient"]
    parts, answers, structured = parse_parts(res.answer or "", question, marker)
    max_parts = max(1, int(a.get("max_parts", 4)))
    considered = [k for k, _ in parts][:max_parts]  # extra parts stay in `parts`: reported as not addressed
    texts = {k: answers[k] for k in considered if answers.get(k)}
    if not texts:
        return fall(mi_token if any(k in answers and answers[k] is None for k in considered) or not structured and
                    is_marker(res.answer or "", marker) else fb["claim_check_insufficient"])
    cite_note = (grounding_cfg or {})["removed_citation_note"]
    fixed, bad = {}, False
    for k in sorted(texts):
        t, b = fix_citations(attach_trailing_citations(texts[k]), len(sources), cite_note)
        if b:
            bad = True
            i = t.rfind(cite_note)
            t = t[:i] if i >= 0 else t
        if count_valid_citations(t, len(sources)):
            fixed[k] = t.strip()
    if not fixed:
        return fall(fb["no_valid_citations"])
    draft = "\n".join(fixed[k] for k in sorted(fixed))
    cc_cfg = dict(cc, partial_marker=a["partial_marker"], removed_note=a["removed_note"])
    records: list = []
    tok = RECORDS.set(records)
    try:
        checked, summary = check_answer(draft, sources, cc_cfg, call=check_call)
    finally:
        RECORDS.reset(tok)
    if checked is None:
        return fall(fb["checker_unavailable"] if summary.get("outcome") == "checker_unavailable"
                    else fb["claim_check_insufficient"])
    for r in records:  # tag every claim record with its part (records of unanswered parts stay internal)
        sent = r["sentence"]
        r["part"] = next((k for k in sorted(fixed) if sent in fixed[k]), None)
    vcfg = a.get("verbatim") or {}
    checker = None
    if vcfg.get("enabled", True):
        from vera.fulldoc_verbatim import VerbatimChecker

        ref_texts, meta = _bundle_reference(ev, sources, cfg, reference_texts)
        checker = VerbatimChecker(ref_texts, meta, vcfg)
    cand = [r for r in records if r.get("part") is not None]
    dropped = 0
    while True:
        kept, profs, d = enforce_verbatim(cand, sources, checker, cfg)
        dropped += d
        answered = {r["part"] for r in kept if r["verdict"] == "supported"}
        narrowed = [r for r in kept if r["part"] in answered]
        if len(narrowed) == len(kept):
            break
        cand = narrowed
    if not require_supported_claim(kept):  # D7, enforced by C2 itself
        return fall(fb["claim_check_insufficient"])
    order = [k for k, _ in parts]
    ptext = dict(parts)
    blocks, uncited, unchecked, multi = [], 0, 0, len(parts) > 1
    for k in order:
        if k not in answered:
            continue
        pr = [r for r in records if r.get("part") == k]
        t, sm = checked_only(fixed[k], pr, sources, cc_cfg, cfg, {"removed": False} if multi else summary,
                             keep_ids={r["id"] for r in kept if r["part"] == k}, dropped=0)
        if t is None:
            continue
        uncited += sm.get("uncited_dropped", 0)
        unchecked += sm.get("unchecked_dropped", 0)
        blocks.append((k, t))
    unanswered = [k for k in order if k not in {k2 for k2, _ in blocks}]
    if not blocks:
        return fall(fb["claim_check_insufficient"])
    if not multi:
        text, summary = blocks[0][1], dict(summary, uncited_dropped=uncited, unchecked_dropped=unchecked)
    else:
        lines = {k: pp["part_heading"].format(n=k, part=_clip(ptext[k])) + "\n" + t for k, t in blocks}
        for k in unanswered:
            lines[k] = pp["unanswered_line"].format(n=k, part=_clip(ptext[k]))
        body = [lines[k] for k in order]
        if summary.get("removed"):
            body.append(a["removed_note"])
        if uncited or unchecked:
            body.append(a["unchecked_note"])
        text = "\n\n".join(body)
        summary = dict(summary, uncited_dropped=uncited, unchecked_dropped=unchecked)
        if unanswered:
            text = pp["partial_statement"] + "\n\n" + text
    if bad:
        text += "\n\n" + cite_note
    complete = not unanswered
    summary = dict(summary, verbatim_dropped=dropped, answer_completeness="complete" if complete else "partial",
                   parts={"total": len(parts), "answered": [k for k, _ in blocks], "unanswered": unanswered})
    total = a.get("max_total_quote_chars")
    total = int(total) if total is not None else (700 if checker is not None else None)
    profiles = [profs[r["id"]] for r in kept if r["id"] in profs] if checker is not None else None
    final = [r for r in kept if r["part"] in {k for k, _ in blocks}]
    citations = build_citations(final, sources, int(a["quote_max_chars"]), total, profiles=profiles)
    part_of = {r["id"]: r["part"] for r in records}
    for c in citations:
        c["part"] = part_of.get(c["claim_id"])
    if checker is not None:
        from vera.fulldoc_verbatim import combined_total

        tot, gen, quoted = combined_total(profiles, [c["quote"] for c in citations if c["quote"]])
        summary["verbatim_budget"] = {"quoted": quoted, "generated_verbatim": gen, "total": tot, "cap": total}
    kind = "answer" if complete else (pp.get("partial_result_kind") or "answer")
    tokens, cost = totals()
    result = AskResult(answer=text, tokens_used=tokens, cost_usd=round(cost, 6), sources=tuple(sources),
                       claim_check=summary, accepted=True, claims=tuple(records), result_kind=kind,
                       contract=None, evidence=ev, citations=tuple(citations))
    return FullDocOutcome(result, None, tokens, round(cost, 6))


def answer(question: str, ev, model: str, pricing, *, chained: dict, grounding_cfg: dict | None, cfg: dict,
           base_tokens: int = 0, base_cost: float = 0.0, complete_fn=None, check_call=None,
           reference_texts=None) -> FullDocOutcome:
    fb = cfg["fallback_tokens"]
    tokens, cost = base_tokens, base_cost

    def fall(token):
        return FullDocOutcome(None, token, tokens, round(cost, 6))

    if not ev.grounded:
        return fall(fb["corpus_unusable"] if ev.status == "corpus_unusable" else fb["insufficient"])
    cc = (grounding_cfg or {}).get("claim_check") or {}
    if not cc.get("enabled"):  # fail closed: never an unchecked answer
        return fall(fb["claim_check_disabled"])

    if complete_fn is None:
        from vera.ask_service import complete as complete_fn
    sources = sources_from(ev)
    res = complete_fn(messages_for(question, sources, cfg), model, pricing, **(chained or {}))
    tokens += res.tokens_used
    cost += res.cost_usd

    from vera.claim_check import RECORDS, check_answer
    from vera.grounded_ask import count_valid_citations, fix_citations

    if per_part_enabled(cfg):  # D8 = A: per-part marker handler replaces the whole-draft substring rule
        return _answer_parts(question, res, ev, sources, cfg, grounding_cfg, cc, check_call, reference_texts, fall,
                             lambda: (tokens, cost))
    marker = cfg["answer"].get("insufficient_marker")
    if marker and marker in (res.answer or ""):  # the model judged the passages off-question: no answer, no check call
        return fall(fb.get("model_insufficient") or fb["insufficient"])  # C2 (F14): its own token for the marker branch
    cite_note = (grounding_cfg or {})["removed_citation_note"]
    text, bad_cites = fix_citations(attach_trailing_citations(res.answer or ""), len(sources), cite_note)
    if count_valid_citations(text, len(sources)) == 0:
        return fall(fb["no_valid_citations"])

    cc_cfg = dict(cc, partial_marker=cfg["answer"]["partial_marker"], removed_note=cfg["answer"]["removed_note"])
    records: list = []
    tok = RECORDS.set(records)
    try:
        checked, summary = check_answer(text, sources, cc_cfg, call=check_call)
    finally:
        RECORDS.reset(tok)
    if checked is None:
        return fall(fb["checker_unavailable"] if summary.get("outcome") == "checker_unavailable"
                    else fb["claim_check_insufficient"])
    draft = text[: text.rfind(cite_note)] if bad_cites else text  # the fix_citations note is not a draft claim
    vcfg = cfg["answer"].get("verbatim") or {}
    checker = None
    if vcfg.get("enabled", True):
        from vera.fulldoc_verbatim import VerbatimChecker

        ref_texts, meta = _bundle_reference(ev, sources, cfg, reference_texts)
        checker = VerbatimChecker(ref_texts, meta, vcfg)
    kept, profs, dropped = enforce_verbatim(records, sources, checker, cfg)
    if cfg["answer"].get("require_supported_claim", True):  # D7 (POL hook): >= 1 fully supported claim, else fall back
        from vera.fulldoc_policy import require_supported_claim

        if not require_supported_claim(kept):
            return fall(fb["claim_check_insufficient"])
    checked, summary = checked_only(draft, records, sources, cc_cfg, cfg, summary, keep_ids={r["id"] for r in kept},
                                    dropped=dropped)
    if checked is not None and bad_cites:
        checked += "\n\n" + cite_note
    if checked is None:  # nothing verified survived (step 4)
        return fall(fb["claim_check_insufficient"])
    total = cfg["answer"].get("max_total_quote_chars")
    total = int(total) if total is not None else (700 if checker is not None else None)
    profiles = [profs[r["id"]] for r in kept] if checker is not None else None
    citations = build_citations(kept, sources, int(cfg["answer"]["quote_max_chars"]), total, profiles=profiles)
    if checker is not None:
        from vera.fulldoc_verbatim import combined_total

        tot, gen, quoted = combined_total(profiles, [c["quote"] for c in citations if c["quote"]])
        summary = dict(summary, verbatim_budget={"quoted": quoted, "generated_verbatim": gen, "total": tot, "cap": total})
    result = AskResult(answer=checked, tokens_used=tokens, cost_usd=round(cost, 6), sources=tuple(sources),
                       claim_check=summary, accepted=True, claims=tuple(records), result_kind="answer",
                       contract=None, evidence=ev, citations=tuple(citations))
    return FullDocOutcome(result, None, tokens, round(cost, 6))


def public_evidence(ev, citations=()) -> dict:
    """Response projection: allowlisted fields only; no span text beyond the capped quotes inside citations."""
    return {"evidence_status": ev.status,
            "examined_full_documents": [{k: d.get(k) for k in _DOC_FIELDS} for d in ev.examined_full_documents],
            "admitted_evidence": [dict({k: a.get(k) for k in _ADM_FIELDS}, span=f"chars={a['start']}-{a['end']}")
                                  for a in ev.admitted],
            "citations": [dict(c) for c in citations]}
