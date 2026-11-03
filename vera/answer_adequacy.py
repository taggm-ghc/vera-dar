"""Answer adequacy: pure, zero-LLM rule check (ANS-LAYERS-1101 Stage 2, job J1).

No I/O, no model calls, no DB. Every threshold and lexicon comes from the ``cfg``
dict supplied by the caller; this module holds no numeric defaults.

    assess(question, summary, admitted, cfg, route) -> dict

Result keys:
    adequacy     adequate | partial | inadequate | correct_no_answer | unknown
    limitations  plain user-language sentences; the ONLY text safe to show users
    scope        always "unknown" here (see TODO below)
    reasons      INTERNAL ONLY: short machine codes, never shown to users
    form         INTERNAL ONLY: question-form class used by the rules

A low document count never sets the label to partial; it only adds a coverage
limitation note. It never claims the sources share one context.

TODO(stage2-plan section 2b): labelling narrow scope (a general question answered
from narrow evidence) belongs in a scope field on the existing claim-check call,
not in this zero-LLM module. Gated on calibration; no real data exists yet
(caldata-inventory.md). Until then ``scope`` is "unknown".

Inputs (plain values):
    question  str
    summary   dict, all keys optional:
                premise_unverified (bool), full_texts_read (int),
                candidates_total (int), removed_count (int),
                uncited_dropped (int), parts (list of {"answered": bool})
    admitted  iterable of dicts: document_id (optional), title, kind
              ("scholarly"|"web")
    cfg       dict, all keys optional (missing/partial config -> "unknown" when
              the verdict would otherwise depend on it, never an exception):
              form_lexicon {specific, broad, comparative, definitional: [phrases]},
              definitional_max_subject_words, min_docs_general (int; other types
              such as a config reference string are ignored),
              acq_round_target_new_docs (int, fallback for min_docs_general)
    route     "scholarly" | "general" | "mixed" | None (router is OFF today)
"""
from __future__ import annotations

import re

ADEQUATE = "adequate"
PARTIAL = "partial"
INADEQUATE = "inadequate"
UNKNOWN = "unknown"
SCOPE_UNKNOWN = "unknown"
CORRECT_NO_ANSWER = "correct_no_answer"

FORM_SPECIFIC = "specific"
FORM_BROAD = "broad"
FORM_COMPARATIVE = "comparative"
FORM_DEFINITIONAL = "definitional"
FORM_UNKNOWN = "unknown"

_LIMITATIONS = {
    "premise_unsupported": "The question assumes something that the sources found do not support.",
    "no_full_text": "No full text could be read for this question, so no evidence-based answer is given.",
    "source_mismatch": "The sources found are academic papers, which do not suit this kind of question.",
    "one_side_missing": "Only one side of the comparison could be found in the evidence.",
    "removed_claims": "Some statements could not be supported by the sources and were left out.",
    "uncited_dropped": "Some statements had no supporting source and were left out.",
    "unread_candidates": "Some sources found could not be read yet.",
}


def _phrase_hit(text: str, phrases) -> bool:
    for p in phrases or ():
        if re.search(r"(?<![a-z0-9])" + re.escape(str(p).lower()) + r"(?![a-z0-9])", text):
            return True
    return False


def _tokens(text: str):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def classify_form(question: str, cfg: dict) -> str:
    q = (question or "").lower().strip()
    lex = cfg.get("form_lexicon") or {}
    for form in (FORM_SPECIFIC, FORM_BROAD, FORM_COMPARATIVE):
        if _phrase_hit(q, lex.get(form)):
            return form
    for pre in lex.get(FORM_DEFINITIONAL) or ():
        pre = str(pre).lower()
        if q.startswith(pre):
            rest = _tokens(q[len(pre):])
            cap = cfg.get("definitional_max_subject_words")
            if cap is None or len(rest) <= cap:
                return FORM_DEFINITIONAL
    return FORM_UNKNOWN


def _int(v):
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _min_docs(cfg: dict):
    v = _int(cfg.get("min_docs_general"))
    return _int(cfg.get("acq_round_target_new_docs")) if v is None else v


def _distinct_docs(docs) -> int:
    # Docs without a document_id each count once; they never collapse together.
    ids = {d["document_id"] for d in docs if d.get("document_id") is not None}
    return len(ids) + sum(1 for d in docs if d.get("document_id") is None)


def _count_note(n: int) -> str:
    if n == 1:
        return "Only 1 full text was read for this question, so the answer may not cover the whole topic."
    return f"Only {n} full texts were read for this question, so the answer may not cover the whole topic."


def assess(question, summary, admitted, cfg, route=None) -> dict:
    summary = summary if isinstance(summary, dict) else {}
    cfg = cfg if isinstance(cfg, dict) else {}
    docs = [d for d in (admitted or ()) if isinstance(d, dict)]
    reasons: list[str] = []
    extra: list[str] = []
    form = classify_form(question, cfg)

    def out(adequacy):
        seen, lims = set(), []
        for r in reasons:
            t = _LIMITATIONS.get(r)
            if t and t not in seen:
                seen.add(t)
                lims.append(t)
        lims.extend(extra)
        return {"adequacy": adequacy, "reasons": reasons, "limitations": lims,
                "form": form, "scope": SCOPE_UNKNOWN}

    if summary.get("premise_unverified"):
        reasons.append("premise_unsupported")
        return out(CORRECT_NO_ANSWER)

    read = summary.get("full_texts_read")
    if (read is not None and read == 0) or not docs:
        reasons.append("no_full_text")
        return out(INADEQUATE)

    if route == "general" and all(d.get("kind") == "scholarly" for d in docs):
        reasons.append("source_mismatch")
        return out(INADEQUATE)

    partial = False
    if form == FORM_COMPARATIVE:
        parts = summary.get("parts") or []
        if any(not p.get("answered") for p in parts if isinstance(p, dict)):
            reasons.append("one_side_missing")
            partial = True
    if (summary.get("removed_count") or 0) > 0:
        reasons.append("removed_claims")
        partial = True
    if (summary.get("uncited_dropped") or 0) > 0:
        reasons.append("uncited_dropped")
        partial = True
    if form in (FORM_DEFINITIONAL, FORM_BROAD):
        n = _distinct_docs(docs)
        floor = _min_docs(cfg)
        if floor is not None and n < floor:
            reasons.append("low_coverage")  # note only; never sets partial
            extra.append(_count_note(n))
            total = _int(summary.get("candidates_total"))
            if total is not None and total > n:
                reasons.append("unread_candidates")
    if partial:
        return out(PARTIAL)
    if not cfg.get("form_lexicon"):
        return out(UNKNOWN)  # cannot judge question form without the lexicon
    return out(ADEQUATE)
