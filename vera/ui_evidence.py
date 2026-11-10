"""AE-1020/WP-D: truthful display of full-document answers in the Ask page (main answer area, never the sidebar).

Reads only the public /ask projection (citations, examined_full_documents, admitted_evidence, evidence_status,
claim_check, contract.fallback). Every accessor is defensive: malformed or missing data renders less, never raises.
Visitor text comes from config/fulldoc_answer.json only for the fallback reasons insufficient_text and
corpus_unusable_text ({n_inspected} is filled from the examined-document count); the remaining fallback sentences are module constants below
(recorded divergence: no config key exists yet for them; config is owned by another workstream).
No internal token, gate name or model name is ever printed. Quotes are trimmed to quote_max_chars from config/fulldoc_answer.json (200 if absent) before display."""
import json
import re
from pathlib import Path

from vera.sources_sidebar import _link, plain, plain_url
from vera.ui_safety import safe_markdown

FULLDOC_CONFIG = Path(__file__).resolve().parent.parent / "config" / "fulldoc_answer.json"
ACQ_FULL = "full_document_acquired"

# Fallback token -> (config answer-key or None, constant default). Unknown token -> GENERIC_FALLBACK. Tokens are never shown.
_TOKEN_TEXT = {
    "insufficient_full_document_evidence": "insufficient_text",
    "full_document_corpus_unusable": "corpus_unusable_text",
    "no_valid_citations": None,
    "checker_unavailable": None,
    "claim_check_insufficient": None,
    "claim_check_disabled": None,
    "premise_unverified": "insufficient_text",  # OD-1104-TRACE: an evidence miss, never premise refusal text
    "model_insufficient": "unsupported_text",  # NONASK-1103: passages were selected; never "found no passage"
}
_TOKEN_DEFAULT = {
    "insufficient_full_document_evidence": "VERA-DAR found no passage in the documents it searched that answers this question well enough, so it has not written an answer.",
    "full_document_corpus_unusable": "VERA-DAR's full-document collection could not be verified right now, so it has not written an answer.",
    "no_valid_citations": "The draft answer did not cite the inspected passages, so it was not shown.",
    "checker_unavailable": "The automated claim check was unavailable, so no answer is shown.",
    "claim_check_insufficient": "The automated claim check did not support enough of the draft, so no answer is shown.",
    "claim_check_disabled": "The automated claim check is off, so no answer is shown.",
    "model_insufficient": "The documents inspected did not give enough to answer, so no answer is shown.",
}
_TOKEN_DEFAULT["premise_unverified"] = _TOKEN_DEFAULT["insufficient_full_document_evidence"]  # OD-1104-TRACE
GENERIC_FALLBACK = "VERA-DAR did not write an answer to this question. The sources below may still help; read them yourself."
ANSWER_SCOPE_NOTE = ("Automated checks, not human review, compared each claim with the passages shown, and can be wrong. "
                     "Quotes are short excerpts, not whole documents. Open the sources and read them before relying on a claim.")
NOT_CHECKED_NOTE = "The claims in this answer were not checked against the sources."


def _cfg() -> dict:
    try:
        return (json.loads(FULLDOC_CONFIG.read_text(encoding="utf-8")).get("answer")) or {}
    except (OSError, ValueError, AttributeError):
        return {}


def _count(v) -> int:
    if isinstance(v, bool):
        return 0
    if isinstance(v, int):
        return max(v, 0)
    if isinstance(v, (list, tuple, dict)):
        return len(v)
    return 0


def _list(v) -> list:
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


def is_fulldoc(data: dict | None, sources=None) -> bool:
    """True only when the response itself says full documents were inspected/acquired."""
    data = data if isinstance(data, dict) else {}
    if data.get("evidence_status") is not None:
        return True
    return any(isinstance(s, dict) and (s.get("acquisition") == ACQ_FULL or s.get("provider") == "full_document")
               for s in (sources if sources is not None else _list(data.get("sources"))))


# NINE-DOCS-RC4-1108: the per-document field that counts toward user-facing "relevant documents" counts. Passages that
# cleared the question's relevance rules (vera.fulldoc_admission.admit), never raw retrieval candidates (candidate_spans
# match on any non-stopword token such as "did" or "come", so they reached the whole pinned 9-document base set).
RELEVANT_FIELD = "relevant_spans"


def matched_documents(examined) -> int | None:
    """Distinct examined documents with >= 1 passage that cleared THIS question's relevance rules (relevant_spans > 0;
    NINE-DOCS-RC4-1108). None when the rows are absent or any row lacks a valid relevant_spans (legacy payloads and
    candidate-only rows: no count is shown at all). Every document of the stored corpus is scored, so neither the
    examined list nor the raw candidate count says anything about relevance."""
    rows = [x for x in examined if isinstance(x, dict)] if isinstance(examined, (list, tuple)) else []
    if not rows or any(not _nat(r.get(RELEVANT_FIELD)) for r in rows):
        return None
    return _distinct_docs([r for r in rows if r[RELEVANT_FIELD] > 0])


def search_counts(base_rows, acq_rows=None) -> dict | None:
    """{"matched": distinct documents of the base corpus and, when it was searched, the stored acquired corpus with
    >= 1 passage that cleared this question's relevance rules} (NINE-DOCS-RC4-1108; NOANSWER-RC-1107 r2). Never the
    size of a set searched regardless of subject (owner 2026-11-07). None when a searched set's rows lack
    relevant_spans (no exact count: show none)."""
    rows = []
    for part in (base_rows, acq_rows):
        if part is None:
            continue
        got = [x for x in part if isinstance(x, dict)] if isinstance(part, (list, tuple)) else []
        if got and matched_documents(got) is None:
            return None
        rows += got
    if base_rows is None or not rows:
        return None
    return {"matched": _distinct_docs([r for r in rows if r[RELEVANT_FIELD] > 0])}


def _valid_counts(sc) -> dict | None:
    """A well-formed contract.search_counts block, else None (never raises)."""
    return sc if isinstance(sc, dict) and _nat(sc.get("matched")) else None


def matched_for(data) -> int | None:
    """Documents with relevance-cleared passages for this question (relevant_spans): contract.search_counts (base + acquired) when present,
    else the examined rows; None = unknown (then no count is shown at all)."""
    data = data if isinstance(data, dict) else {}
    contract = data.get("contract") if isinstance(data.get("contract"), dict) else {}
    sc = _valid_counts(contract.get("search_counts"))
    return sc["matched"] if sc is not None else matched_documents(data.get("examined_full_documents"))


def count_matching_on(cfg) -> bool:
    """answer.count_matching_documents, exactly true (ships true; false = the previous wording, the only way back)."""
    return isinstance(cfg, dict) and cfg.get("count_matching_documents") is True


# NOANSWER-RC-1107 r2: used only if the configured text is missing or malformed (never the old set-size wording)
_MATCH_DEFAULT = {
    "insufficient_no_match_text": "No stored full document was relevant to this question, so VERA-DAR has not "
                                  "written an answer or answered from memory.",
    "insufficient_matched_text": "{n_stored_phrase} had passages matching this question's key terms, but none "
                                 "answered it well enough, so VERA-DAR has not written an answer or answered from "
                                 "memory.",
    "insufficient_nocount_text": "VERA-DAR found no stored passage that answers this question well enough, so it has "
                                 "not written an answer or answered from memory.",
    "unsupported_matched_text": "Passages from {n_stored_phrase} were selected as possible evidence, but they did "
                                "not support an answer, so VERA-DAR has not answered.",
    "unsupported_nocount_text": "The passages VERA-DAR selected did not support an answer, so it has not answered.",
}


def relevant_scope_phrase(n: int) -> str:
    """NINE-DOCS-RC4-1108: the Evidence scope count of stored documents with relevance-cleared passages."""
    if n == 0:
        return "no stored document relevant to the question"
    return f"{n} stored document{'s' if n != 1 else ''} with passages matching the question's key terms"


def _stored_phrase(n: int) -> str:
    return f"{n} stored full document" if n == 1 else f"{n} stored full documents"


def _match_text(cfg, key: str, n: int | None = None) -> str:
    val = cfg.get(key) if isinstance(cfg, dict) else None
    for t in (val, _MATCH_DEFAULT[key]):
        if isinstance(t, str) and t.strip():
            out = t.replace("{n_stored_phrase}", _stored_phrase(_count(n)))
            if "{" not in out:
                return out
    return _MATCH_DEFAULT["insufficient_nocount_text"]


def insufficient_matching_text(cfg, n_matched, n_fetched: int = 0) -> str | None:
    """NOANSWER-RC-1107: the no-passage headline (flag on) counting only documents with relevance-cleared passages
    (relevant_spans, NINE-DOCS-RC4-1108) for THIS question; none: "No stored full document was relevant"; count unknown (or documents fetched for the
    question, whose matches are not counted here): no count at all. None only when the flag is off."""
    if not count_matching_on(cfg):
        return None
    if n_matched is None or _count(n_fetched) > 0:
        return _match_text(cfg, "insufficient_nocount_text")
    n = _count(n_matched)
    return _match_text(cfg, "insufficient_no_match_text" if n == 0 else "insufficient_matched_text", n)


def unsupported_matching_text(cfg, n_selected) -> str | None:
    """Passages selected, no supported answer (flag on): count only the documents the passages came from."""
    if not count_matching_on(cfg):
        return None
    if _count(n_selected) < 1:  # no passage document known: no count at all
        return _MATCH_DEFAULT["unsupported_nocount_text"]
    return _match_text(cfg, "unsupported_matched_text", n_selected)


PREMISE_TEXT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "premise_text.json"


def premise_unverified_text() -> str:
    """Distinct no-answer reason for a flagged but unverifiable question premise (OD-1104-TRACE reopened behind
    config/premise_text.json `enabled`, shipped false). Returns "" unless enabled is exactly True and text is a non-empty
    string without placeholders; any read or parse error returns "" (old behaviour). Never asserts the premise is false."""
    try:
        c = json.loads(PREMISE_TEXT_CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    t = c.get("text") if isinstance(c, dict) and c.get("enabled") is True else None
    return t.strip() if isinstance(t, str) and t.strip() and "{" not in t else ""


def stored_checked(data: dict | None) -> int:
    """NOANSWER-1109 R1: stored documents reused by this request's inline round whose passages were selected (listed in the
    trace) with no fresh document admitted. The user cares that these stored documents were checked, not which search
    stage found them, so they are counted as checked, never as "no stored document relevant"."""
    data = data if isinstance(data, dict) else {}
    contract = data.get("contract") if isinstance(data.get("contract"), dict) else {}
    ia = contract.get("inline_acquisition") if isinstance(contract.get("inline_acquisition"), dict) else {}
    re_ = _count(ia.get("reused"))
    fe_ = _count((ia.get("counts") or {}).get("fetched", ia.get("admitted"))) if isinstance(ia.get("counts"), dict) or "admitted" in ia else 1
    return re_ if re_ and not fe_ and trace_selected(data) > 0 and _no_answer(data) else 0


def fallback_text(token, cfg: dict | None = None, n_inspected: int = 0, n_selected: int = 0, n_fetched: int = 0,
                  n_matched: int | None = None, n_stored_checked: int = 0) -> str:
    """Visitor sentence for a contract.fallback token (None/unknown -> generic). Never returns the token.
    n_matched (NOANSWER-RC-1107): documents with relevance-cleared passages (relevant_spans) for this question; None = unknown (old wording)."""
    if not isinstance(token, str) or not token:
        return GENERIC_FALLBACK
    if token not in _TOKEN_DEFAULT:
        return GENERIC_FALLBACK
    if token == "premise_unverified":  # PROCFIX-RELKIT-1108 (OD-1108-HITL item 1): honest reason, flag OFF by default
        why = premise_unverified_text()
        if why:
            return why
    cfg = _cfg() if cfg is None else cfg
    key = _TOKEN_TEXT.get(token)
    if n_stored_checked > 0 and key == "insufficient_text" and count_matching_on(cfg):
        return (f"{_stored_phrase(n_stored_checked).replace(' full', '')} "
                f"{'was' if n_stored_checked == 1 else 'were'} checked for this question and passages from "
                f"{'it' if n_stored_checked == 1 else 'them'} were selected, but none answered it, so VERA-DAR has not "
                "written an answer or answered from memory.")
    alt = (insufficient_matching_text(cfg, n_matched, n_fetched) if key == "insufficient_text"
           else unsupported_matching_text(cfg, n_selected) if key == "unsupported_text" else None)
    if alt:
        return alt
    val = cfg.get(key) if key else None
    if isinstance(val, str) and val.strip():
        try:
            n = _count(n_inspected)  # NONASK-1103 WP-C: {n_docs_phrase} = "1 full document" / "N full documents"
            ph = f"{n} full document" if n == 1 else f"{n} full documents"
            f = _count(n_fetched)  # R8-HEADLINE-1106: keep the "from VERA-DAR's collection and M fetched" wording
            if f > 0:
                ph += f" from VERA-DAR's collection and {f} fetched for this question"
            out = val.replace("{n_docs_phrase}", ph).replace("{n_selected}", str(_count(n_selected))).format(n_inspected=n)
        except (KeyError, IndexError, ValueError):
            return _TOKEN_DEFAULT[token]
        return out if "{" not in out else _TOKEN_DEFAULT[token]
    return _TOKEN_DEFAULT[token]


def _distinct_docs(items) -> int:
    """Distinct documents among dict rows (same canonical id / document id / DOI / URL counts once). Rows with no
    identifying field count one each; non-list input counts like _count."""
    if not isinstance(items, list):
        return _count(items)
    seen, extra = set(), 0
    for d in items:
        key = next(((k, str(d[k]).lower()) for k in ("canonical_source_id", "document_id", "doi", "url")
                    if isinstance(d, dict) and d.get(k)), None)
        if key is None:
            extra += 1
        else:
            seen.add(key)
    return len(seen) + extra


def fetched_for_request(data: dict | None) -> int:
    """SCOPE-COUNT-1106: documents fetched for this request whose text was checked for passages but that are NOT in
    examined_full_documents. Only when the examined set is the held collection (corpus_source.kind 'bundle'): the
    inline round's answer-only documents (reason below_persist_min) are then absent from it. Otherwise 0 (no double count)."""
    data = data if isinstance(data, dict) else {}
    cs = data.get("corpus_source")
    contract = data.get("contract") if isinstance(data.get("contract"), dict) else {}
    # SELFQ-RCA-1107 D2: a non-answer (contract.fallback set) always carries the held-collection attempt as its evidence
    # (ask_service.answer_in_scope: evidence=ev of _fulldoc), so the inline round's answer-only documents are never in
    # examined_full_documents there either, whatever the corpus kind; counting them cannot double count.
    no_answer = isinstance(contract.get("fallback"), str) and bool(contract.get("fallback"))
    if not (isinstance(cs, dict) and cs.get("kind") == "bundle") and not no_answer:
        return 0
    ia = contract.get("inline_acquisition")
    docs = ia.get("documents") if isinstance(ia, dict) and isinstance(ia.get("documents"), list) else []
    return sum(1 for d in docs if isinstance(d, dict) and d.get("reason") == "below_persist_min")


def _trace_n(data: dict) -> int:
    contract = data.get("contract") if isinstance(data.get("contract"), dict) else {}
    tr = contract.get("trace") if isinstance(contract.get("trace"), dict) else {}
    ps = tr.get("passages")
    return sum(1 for p in ps if isinstance(p, dict)) if isinstance(ps, list) else 0


def trace_fetched(data: dict | None) -> bool:
    """SELFQ-RCA-1107 D2: the evidence trace shown came from an attempt on documents fetched for this request
    (contract.trace_attempt "fetched", set by the server), so its passages are NOT the response's admitted_evidence
    (the held-collection attempt's). Legacy bodies without the tag: fetched when admitted_evidence is empty but the
    trace lists passages (the only shape in which the two can differ without the tag)."""
    data = data if isinstance(data, dict) else {}
    contract = data.get("contract") if isinstance(data.get("contract"), dict) else {}
    tag = contract.get("trace_attempt")
    if tag in ("fetched", "stored"):
        return tag == "fetched"
    return _count(data.get("admitted_evidence")) == 0 and _trace_n(data) > 0


def trace_selected(data: dict | None) -> int:
    """SELFQ-RCA-1107 D2: passages of documents fetched for this request listed in the evidence trace (0 when the
    trace is the held attempt's: those passages are admitted_evidence and are counted from it)."""
    data = data if isinstance(data, dict) else {}
    return _trace_n(data) if trace_fetched(data) else 0


def _no_answer(data: dict) -> bool:
    contract = data.get("contract") if isinstance(data.get("contract"), dict) else {}
    return isinstance(contract.get("fallback"), str) and bool(contract.get("fallback"))


def _attempt_docs(data: dict) -> str:
    """NOANSWER-1109: what the inline attempt's documents were. Its corpus includes stored documents reused for this
    request, so with reuse and no fresh document it is not "fetched"; the origin comes from the counts, not the trace tag."""
    contract = data.get("contract") if isinstance(data.get("contract"), dict) else {}
    ia = contract.get("inline_acquisition") if isinstance(contract.get("inline_acquisition"), dict) else {}
    re_ = _count(ia.get("reused"))
    fe_ = _count((ia.get("counts") or {}).get("fetched", ia.get("admitted"))) if isinstance(ia.get("counts"), dict) or "admitted" in ia else 1
    if re_ and not fe_:
        return "documents already stored and checked for this question"
    if re_ and fe_:
        return "stored documents and documents fetched for this question"
    return "documents fetched for this question"


def evidence_scope_line(data: dict | None) -> str | None:
    """'N documents inspected, M passages from K documents selected as possible evidence' from the response, or None without evidence fields.
    N and K count distinct documents, so inspected vs used are never conflated."""
    data = data if isinstance(data, dict) else {}
    if data.get("evidence_status") is None and "examined_full_documents" not in data and "admitted_evidence" not in data:
        return None
    n, m = _distinct_docs(data.get("examined_full_documents")), _count(data.get("admitted_evidence"))
    k = _distinct_docs(data.get("admitted_evidence")) if isinstance(data.get("admitted_evidence"), list) else 0
    f = fetched_for_request(data)
    sc_ = stored_checked(data)
    if sc_ and not m and count_matching_on(_cfg()):  # R1: stored documents were checked; say so instead of "none relevant"
        t_ = trace_selected(data)
        return (f"Evidence scope: {sc_} stored document{'s were' if sc_ != 1 else ' was'} checked for this question; "
                f"{t_} of {'their' if sc_ != 1 else 'its'} passages {'were' if t_ != 1 else 'was'} selected as possible evidence; none was used in an answer.")
    if count_matching_on(_cfg()):  # NOANSWER-RC-1107 r2: never the size of a set searched regardless of subject
        mt = matched_for(data)
        parts = []
        if mt is not None:  # NINE-DOCS-RC4-1108: relevance-cleared passages only, never shared function words
            parts.append(relevant_scope_phrase(mt))
        if f:
            parts.append(f"{f} document{'s' if f != 1 else ''} fetched for this question")
        head = "Evidence scope: " + (" and ".join(parts) + "; " if parts else "")
    elif f:
        head = f"Evidence scope: {n} document{'s' if n != 1 else ''} from VERA-DAR's collection and {f} fetched for this question inspected; "
    else:
        head = f"Evidence scope: {n} document{'s' if n != 1 else ''} inspected; "
    t = trace_selected(data)  # SELFQ-RCA-1107 D2: the trace's fetched-document passages are named, never denied
    tail = "; none was used in an answer." if _no_answer(data) else "."
    fetched_part = f"{t} passage{'s' if t != 1 else ''} from {_attempt_docs(data)}" if t else ""
    if not m:
        if t:
            return head + fetched_part + " selected as possible evidence" + tail
        return head + "no passages selected as possible evidence."
    frm = f" from {k} document{'s' if k != 1 else ''}" if k else ""
    if t:
        return head + f"{m} passage{'s' if m != 1 else ''}{frm} in VERA-DAR's collection and {fetched_part} selected as possible evidence" + tail
    return head + f"{m} passage{'s' if m != 1 else ''}{frm} selected as possible evidence."


_SOURCE_LABEL = {"db": "stored database", "bundle": "local files"}


def evidence_source_line(data: dict | None) -> str | None:
    """Accurate source mix. A contract.inline_acquisition block (documents fetched for this request) decides between
    'stored database', 'this request only (not saved)' or both; else corpus_source.kind ('stored database' / 'local files')."""
    data = data if isinstance(data, dict) else {}
    contract = data.get("contract") if isinstance(data.get("contract"), dict) else {}
    ia = contract.get("inline_acquisition")
    if isinstance(ia, dict):
        docs = [d for d in (ia.get("documents") if isinstance(ia.get("documents"), list) else []) if isinstance(d, dict)]
        only = sum(1 for d in docs if d.get("reason") == "below_persist_min")
        held = _count(ia.get("reused")) + _count(ia.get("n_persisted"))
        admitted = _count(data.get("admitted_evidence")) > 0  # SEC U3: "used for this answer" needs admitted evidence
        t = trace_selected(data)  # SELFQ-RCA-1107 D2: passages of the fetched documents listed in the trace
        if _no_answer(data) and not only:
            # NOANSWER-1109: documents fetched for this request and saved are not "stored database" evidence; with no
            # answer, the source is where the selected passages came from (none: no source line at all).
            if t:  # origin from the counts, not the trace tag (the inline attempt includes reused stored documents)
                re_, fe_ = _count(ia.get("reused")), _count((ia.get("counts") or {}).get("fetched", ia.get("admitted")))
                what = ("documents already stored, checked for this question" if re_ and not fe_
                        else "documents fetched for this question" if fe_ and not re_
                        else "stored documents and documents fetched for this question" if re_ and fe_
                        else "documents checked for this question")
                return (f"Evidence source: {what}; passages from them were selected as "
                        "possible evidence but not used in an answer.")
            if not admitted:
                return None
        if only and not admitted and t:
            return ("Evidence source: documents fetched for this request (not saved); passages from them were selected "
                    "as possible evidence but not used in an answer.")
        if only and not admitted:
            return "Evidence source: documents fetched for this request (not saved); none were selected as possible evidence."
        if only and _no_answer(data):  # SELFQ-RCA-1107 D2: no answer was written, so nothing was "used for this answer"
            return ("Evidence source: stored database and documents fetched for this request (not saved); no answer was "
                    "written.")
        if only and held:
            return "Evidence source: stored database and documents used for this answer only."
        if only:
            return "Evidence source: documents used for this answer only (not saved)."
        if held:
            return "Evidence source: stored database."
    cs = data.get("corpus_source")
    label = _SOURCE_LABEL.get(cs.get("kind")) if isinstance(cs, dict) and isinstance(cs.get("kind"), str) else None
    return f"Evidence source: {label}." if label else None


_STATUS_CLAUSE = {
    "resource_limited": "supply was limited.",
    "relevance_rejected": "none of the documents fetched had enough relevant passages to keep.",
    "no_route": "no identifier VERA-DAR can fetch automatically was available for the sources found.",
    "incomplete_deadline": "the search stopped at the time limit.",
    "search_failed": "the search could not be completed.",
    "capped": "the limit on new acquisition runs was reached.",
}


def _stop_clause(aq: dict) -> str:
    """MIDTERMS-WP8-1106: vera.inline_acquire.stop_clause (one wording for the shortfall line and the notice)."""
    from vera import inline_acquire
    return inline_acquire.stop_clause(aq)


def _nat(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def acquisition_notice_line(data: dict | None) -> str | None:
    """OD-1031-P1 item 4: plain notice when a live search found fewer sources than targeted. Counts only; no
    titles, hosts, URLs or provider names. None when the target was met or the block is absent or malformed.
    MIDTERMS-WP10-1106 (owner D3): with a valid examined-source ledger summary, "VERA-DAR obtained N documents for this question (aim: at least T)"
    counts only qualifying full- or partial-text sources actually examined (deduplicated; news, official and web pages
    included); otherwise the legacy "VERA-DAR obtained N new full documents ... (aim: at least T)" (stored) text. WP8 appends the stop clause."""
    data = data if isinstance(data, dict) else {}
    contract = data.get("contract") if isinstance(data.get("contract"), dict) else {}
    ia = contract.get("inline_acquisition") if isinstance(contract.get("inline_acquisition"), dict) else {}
    aq = ia.get("acquisition") if isinstance(ia.get("acquisition"), dict) else {}
    clause = _STATUS_CLAUSE.get(aq.get("status"))
    t = aq.get("target")
    from vera import evidence_ledger
    led = ia.get("ledger")
    if evidence_ledger.valid_summary(led) and _nat(t):
        c = led["counted"]
        if c >= t:
            return None
        head = f"VERA-DAR obtained {c} {'document' if c == 1 else 'documents'} for this question (aim: at least {t})"
        if aq.get("status") == "relevance_rejected":
            # NOANSWER-PAGE-1108 D3: the status also counts documents judged not relevant from their title or abstract
            # BEFORE any fetch (inline_acquire._supply_status), so the "fetched" clause is stated only for documents
            # whose text was really examined (the same ledger as "Read N"), with that count; with none fetched, the
            # line is the read count alone (the non-answer notice already itemises found / not relevant / not retrieved).
            f = led["examined_unqualified"]
            if f < 1:
                return f"{head}.{_stop_clause(aq)}"
            oth = "other " if c else ""  # "other" only when some were read
            clause = (f"{f} {oth}document fetched did not have enough relevant passages to keep." if f == 1
                      else f"{f} {oth}documents fetched did not have enough relevant passages to keep.")
        if clause is None:  # review F2: the stop disclosure does not depend on a status clause
            return _stop_clause(aq).strip() or None
        if led["partial_text"] > 0:
            head += f" ({led['full_text']} in full, {led['partial_text']} in part)"
        kinds = [(k, led[k]) for k in evidence_ledger.KINDS if led[k] > 0]
        if len(kinds) >= 2:
            head += " (" + ", ".join(f"{k} {v}" for k, v in kinds) + ")"
        if led["stored"] > 0:
            head += f", {led['stored']} saved to VERA-DAR's collection"
        return f"{head}; {clause}{_stop_clause(aq)}"
    n = aq.get("stored")
    if not all(_nat(x) for x in (n, t)) or n >= t:
        return None
    if clause is None:  # review F2
        return _stop_clause(aq).strip() or None
    return f"VERA-DAR obtained {n} new full {'document' if n == 1 else 'documents'} for this question (aim: at least {t}); {clause}{_stop_clause(aq)}"


def inline_counts_line(data: dict | None) -> str | None:
    """NONASK-1103 A6-4 (D3): stored and fetched source counts, separately (present only when answer.inline_honest_counts
    was on for the request). Counts only; None when absent or malformed."""
    data = data if isinstance(data, dict) else {}
    contract = data.get("contract") if isinstance(data.get("contract"), dict) else {}
    ia = contract.get("inline_acquisition") if isinstance(contract.get("inline_acquisition"), dict) else {}
    c = ia.get("counts") if isinstance(ia.get("counts"), dict) else {}
    n, f = c.get("stored"), c.get("fetched")
    if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (n, f)):
        return None
    if _no_answer(data):  # NOANSWER-1109: nothing was used; stored = held before this request, fetched = read for it
        return f"Sources checked for this question: {n} already stored, {f} newly fetched."
    return f"Sources used for this answer: {n} stored, {f} newly fetched."


def _shortfall_in_limitations(data, note) -> bool:
    """ANS-LAYERS-1101-FC: with the layered fields present and the line already in limitations, show it once there."""
    return bool(note) and has_layers(data) and note in _note_texts(data.get("limitations"))


def render_evidence_scope(st, data) -> None:
    note = acquisition_notice_line(data)
    if note and not _shortfall_in_limitations(data, note):
        st.caption(note)
    line = evidence_scope_line(data)
    if line:
        st.caption(line)
        src = evidence_source_line(data)
        if src:
            st.caption(src)
        cnt = inline_counts_line(data)
        if cnt:
            st.caption(cnt)


def render_fallback_reason(st, data) -> None:
    """Non-answer result from the full-document path: say why there is no answer. No-op for plain web-only responses."""
    data = data if isinstance(data, dict) else {}
    contract = data.get("contract") if isinstance(data.get("contract"), dict) else {}
    token = contract.get("fallback")
    if token is None and data.get("evidence_status") is None:
        note = acquisition_notice_line(data)  # SA-FIX: a plain reading-list non-answer still shows the acquisition notice
        if note and not _shortfall_in_limitations(data, note):
            st.caption(note)
        return
    ae = data.get("admitted_evidence")
    st.info(fallback_text(token, n_inspected=_count(data.get("examined_full_documents")),
                          n_selected=_distinct_docs(ae) if isinstance(ae, list) else 0,
                          n_fetched=fetched_for_request(data),
                          n_matched=matched_for(data), n_stored_checked=stored_checked(data)))
    render_evidence_scope(st, data)


def _quote_cap() -> int:
    try:
        return max(int(_cfg()["quote_max_chars"]), 1)
    except (KeyError, TypeError, ValueError):
        return 200


def _clean(text, limit: int = 400) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()[:limit]


def _src_by_n(data) -> dict:
    from vera import references_ui

    return references_ui.sources_by_n((data if isinstance(data, dict) else {}).get("sources"))


EXCERPT_MARK = "excerpted"
_SA = re.compile(r"\bby[\s\-]*sa\b", re.I)


def share_alike_notice(groups) -> str:
    """COMP-1103 (orchestrator compliance decision, option ii): a short notice when an answer uses a CC BY-SA source.
    `groups` = dicts with licence / attribution{licence} / title / n. Returns "" when no BY-SA source is used."""
    refs = []
    for g in groups if isinstance(groups, list) else []:
        if not isinstance(g, dict):
            continue
        at = g.get("attribution") if isinstance(g.get("attribution"), dict) else {}
        if _SA.search(str(g.get("licence") or "")) or _SA.search(str(at.get("licence") or "")):
            refs.append(f"[{g.get('n', '?')}] {_clean(g.get('title') or 'source', 120)}")
    if not refs:
        return ""
    return ("Parts of this answer adapt CC BY-SA material from: " + "; ".join(refs)
            + "; that material remains under CC BY-SA 4.0.")


def render_citations(st, data) -> None:
    """Per cited claim: [n] verdict, short quote (or span reference only), then CC BY attribution once per source."""
    cites = _list((data if isinstance(data, dict) else {}).get("citations"))
    if not cites:
        return
    st.markdown("**Cited passages** (short excerpts)")
    shown: set = set()
    lines_seen: set = set()  # NOANSWER-1109: the same passage line is never repeated; one attribution per document
    intext = {n: s["in_text"] for n, s in _src_by_n(data).items() if isinstance(s.get("in_text"), str) and s["in_text"].strip()}
    for c in cites:
        n = c.get("n") if isinstance(c.get("n"), int) and not isinstance(c.get("n"), bool) else "?"
        verdict = {"supported": "supported", "partial": "partly supported"}.get(c.get("verdict"), "checked")
        span = _clean(c.get("quote_span") or c.get("span") or "", 80)
        where = f" (passage {plain(span)})" if span else ""
        quote = _clean(c.get("quote"), _quote_cap()) if isinstance(c.get("quote"), str) else ""
        lk = (n, verdict, span, quote)
        at0 = c.get("attribution") if isinstance(c.get("attribution"), dict) else {}
        dk = c.get("canonical_source_id") or at0.get("source") or at0.get("title") or n
        if lk in lines_seen:
            continue
        lines_seen.add(lk)
        it = safe_markdown(intext[n]) + " " if n in intext else ""  # NONASK-1103 C4: APA in-text citation after "[n] "
        if quote:
            st.markdown(f"[{n}] {it}{verdict}: “{plain(quote)}”{where}")
        else:
            ref = _clean(c.get("span") or "", 80)
            st.markdown(f"[{n}] {it}{verdict}: excerpt not shown{f' (passage {plain(ref)})' if ref else ''}")
        at = c.get("attribution")
        if isinstance(at, dict) and dk not in shown:
            shown.add(dk)
            parts = [plain(at["creator"]) if at.get("creator") else None,
                     _link(at.get("title") or "source", at.get("source")) if at.get("title") or at.get("source") else None,
                     (_link(at["licence"], at.get("licence_url")) if at.get("licence") else None)]
            line = ". ".join(p for p in parts if p)
            if at.get("changes") or quote:  # COMP-1103: a shown excerpt is marked as an excerpt where the licence needs it
                line += f". Changes: {plain(at.get('changes') or EXCERPT_MARK)}"
            if line:
                st.caption(f"[{n}] " + line)


CITATION_CHECK_DISCLAIMER = ("This checks whether the cited claims are supported by their cited sources, "
                             "not whether the question was answered.")


def claim_check_text(cc, cap: int | None = None, fulldoc: bool = False, layered: bool = False) -> str | None:
    """Caption text for the claim check. Legacy abstract wording is kept for non-full-document answers."""
    if not isinstance(cc, dict):
        return None
    if cc.get("outcome") != "checked":
        return NOT_CHECKED_NOTE if cc else None
    checked, supported, partial, removed = (_count(cc.get(k)) for k in ("checked", "supported", "partial", "removed"))
    if layered:  # ANS-LAYERS-1101: grounding statement only; never implies the question was answered
        text = (f"Citation check: {supported} of {checked} cited claims are supported by their cited sources "
                f"({partial} partly supported, {removed} removed). " + CITATION_CHECK_DISCLAIMER)
        left = _count(cc.get("uncited_dropped")) + _count(cc.get("unchecked_dropped"))
        return text + (f" {left} further claims were left out because they could not be checked." if left else "")
    if fulldoc:
        head = f"Claim check: {checked} cited claims checked by a model against the cited passages (full span text; excerpts shown may be abridged)"
    else:
        head = (f"Claim check: {checked} cited claims checked by a model against the cited abstract text"
                f"{f' (first {cap:,} characters)' if cap else ''}")
    text = f"{head}; {supported} supported, {partial} partly supported, {removed} removed."
    extra = []
    unc, unk = _count(cc.get("uncited_dropped")), _count(cc.get("unchecked_dropped"))
    if unc or unk:
        extra.append(f"{unc + unk} left out because they could not be checked")
    if _count(cc.get("verbatim_dropped")):
        extra.append(f"{_count(cc.get('verbatim_dropped'))} left out as too close to the source wording")
    parts = cc.get("parts")
    if isinstance(parts, dict) and _count(parts.get("total")):
        extra.append(f"parts answered: {_count(parts.get('answered'))} of {_count(parts.get('total'))}")
    return text + (" " + "; ".join(extra) + "." if extra else "")


def render_claim_check(st, cc, cap: int | None = None, fulldoc: bool = False, layered: bool = False) -> None:
    text = claim_check_text(cc, cap, fulldoc, layered)
    if text:
        st.caption(text)


def render_answer_scope_note(st) -> None:
    st.caption(ANSWER_SCOPE_NOTE)


def _note_texts(items) -> list[str]:
    """ANS-LAYERS-1101: list[str] or list[{kind,text}] -> plain non-empty strings; anything else is ignored."""
    out = []
    for it in items if isinstance(items, list) else []:
        t = it.get("text") if isinstance(it, dict) else it
        if isinstance(t, str) and t.strip():
            out.append(t.strip())
    return out


STATUS_LABELS = {"partial": "Partial answer", "not_answered": "Limited evidence"}


def has_layers(data) -> bool:
    """True when the API sent the layered fields; absent = older API, render today's way."""
    return isinstance(data, dict) and isinstance(data.get("answer_status"), str)


def answer_status_label(data) -> str | None:
    """Short label above the answer; None for complete or unknown values (unknown is not styled as partial)."""
    return STATUS_LABELS.get(data.get("answer_status")) if isinstance(data, dict) else None


def render_limitations(st, data) -> None:
    for t in _note_texts(data.get("limitations") if isinstance(data, dict) else None):
        st.markdown("- " + safe_markdown(t))


def render_process_notes(st, data) -> None:
    for t in _note_texts(data.get("process_notes") if isinstance(data, dict) else None):
        st.caption(plain(t))
