"""K2.I5: the Ask page's non-answer branch (reading list, decline, discovery notes). Fail-closed: anything that is not
an evidentiary answer lands here. This module never calls st.success, render_sources or render_memory_panel and never
reads sources, traced_sources, claim_check, recalled or memory_submitted (the answer branch owns those)."""
import re

from vera import answer_contract
from vera.memory_panel import _as_count
from vera.sources_sidebar import LICENCE_SIDEBAR, _link, plain, plain_url
from vera import apa7, references_ui
from vera.ui_safety import safe_markdown

KIND_ANSWER, KIND_DECLINE, KIND_READING_LIST = "answer", "decline", "reading_list"
KIND_SELF_DESCRIPTION = "self_description"  # SELFREF-1028: fixed public self-description, shown as plain text


# DAR-PIPE-1022-UIFIX: one table from internal reason codes (config/auto_acquire.json reason_codes) to plain text.
# Internal codes are keys only; they are never rendered. Unknown codes use REASON_GENERIC.
OBTAINED_TEXT = "Full text obtained; not used in this answer."
# R8-HEADLINE-1106: the document supplied passages that reached the writer, but they did not support an answer
OBTAINED_CONSIDERED_TEXT = "Full text obtained; its passages were considered but did not support an answer."
_OBTAINED_ANY = (OBTAINED_TEXT, OBTAINED_CONSIDERED_TEXT)
# Page-level disclosure when some or all listed documents were obtained (the fixed "has not acquired or read the
# documents in this list" line would contradict an "obtained" row).
# The banner is composed from per-row counts: n_obt (obtained), n_held (stored / in collection), n_none (not acquired).
BANNER_TAIL = "This is a discovery result, not an evidence-grounded answer."
HEAD_OTHERS = "VERA-DAR has not acquired or read the other documents in this list."  # obtained + not acquired
HEAD_UNMARKED = ("VERA-DAR has not acquired or read the documents in this list that are not marked as obtained or "
                 "stored.")  # obtained + stored + not acquired
HELD_OBTAINED = "Documents marked as obtained were not used in this answer."
HELD_BOTH = "Documents marked as obtained or stored were not used in this answer."
NOT_READ_OTHERS = f"{HEAD_OTHERS} {HELD_OBTAINED} {BANNER_TAIL}"
OBTAINED_ALL = ("VERA-DAR obtained the full text of the documents in this list but did not use them in this answer. "
                "This is a discovery result, not an evidence-grounded answer.")
REASON_GENERIC = "This source could not be read for this answer."
REASON_TEXT = {
    "R-ACQ-NODOI": "No DOI, PubMed Central or arXiv identifier that VERA-DAR can fetch automatically.",
    "R-ACQ-PID-NOROUTE": "Persistent identifier (Handle) found, but VERA-DAR has no automatic full-text route for it yet.",
    "R-ACQ-PREPRINT_LICENCE_UNVERIFIED": "Preprint: its reuse terms could not be confirmed.",
    "R-ACQ-PREPRINT_NOT_PERMITTED": "Preprint: reuse of the full text is not permitted.",
    "R-ACQ-ARXIV_LICENCE_HOLD": "Preprint: its arXiv licence does not permit reuse of the full text, so it was not used.",
    "R-ACQ-ARXIV_HTML_POLICY": "Preprint: the full text was found, but arXiv full text is not yet used as evidence.",
    "R-ACQ-NOTOA": "Full text is not openly available.",
    "R-ACQ-NOLIC": "Full text not openly licensed.",
    "R-ACQ-LICENCE_ABSENT": "Full text not openly licensed.",
    "R-ACQ-LICENCE_MISMATCH": "Full text not openly licensed.",
    "R-ACQ-PUBTYPE": "Not a type of document that can be used as evidence.",
    "R-ACQ-UNREACHABLE": "Source unreachable.",
    "R-ACQ-RATELIMITED": "Source temporarily unavailable (too many requests).",
    "R-ACQ-DEADLINE": "Not checked in time.",
    "R-ACQ-RATE": "Not checked: the limit on new acquisition runs was reached.",  # NONASK-1103 S6: a quota, not a time stop
    "R-ACQ-DAILY": "Not checked: the limit on new acquisition runs was reached.",
    "R-ACQ-FETCH_TIMEOUT": "Source unreachable.",  # P0DTA-DIAG: the per-fetch wall clock
    "R-ACQ-DOC_BUDGET": "Not checked: the limit on documents read for one request was reached.",
    "R-ACQ-CRAWL_DELAY": "Not checked in time: the source asks crawlers to wait.",
    "R-ACQ-HOST": "Blocked by network safety check.",
    "R-ACQ-ROBOTS": "Blocked by network safety check.",
    "R-ACQ-TDM": "Not used: the publisher reserves text and data mining rights.",  # CONTROL-GAPS-1107
    "R-ACQ-CYCLE": "Blocked by network safety check.",
    "R-ACQ-LARGE": "Document too large to read.",
    "R-ACQ-CONVERT": "Document could not be converted to readable text.",
    "R-ACQ-ENCODING": "Document could not be converted to readable text.",
    "R-ACQ-FORMAT_UNSUPPORTED": "Document format not supported.",
    "R-ACQ-IDENTITY_MISMATCH": "The document did not match the listed source.",
    "R-ACQ-INJECT": "Document rejected by a safety check.",
    "R-ACQ-IRRELEVANT": "Not relevant enough to your question.",
    "R-ACQ-FEWSPANS": "Not relevant enough to your question.",
    "R-ACQ-DUPLICATE": "Duplicate of another listed source.",
    "R-ACQ-KNOWN": "Already in the collection.",
    "R-ACQ-DEFER_CAP": "Not checked: too many candidates for one request.",
    "R-ACQ-STOPPED": "Not checked: the search was stopped before it reached this document.",
    "R-ACQ-NO_PROGRESS": "Not checked: the search ended after further rounds found nothing new.",
    "R-ACQ-WRITE": "Could not be added for this answer.",
    # loader-level refusal codes (config/fulltext_ingest.json reason_codes) that acquisition emits from the fetched document
    "R-LIC-REJECT": "The licence does not permit reuse of the full text.",
    "R-LIC-HOLD": "Full text licence could not be confirmed, so it was not used.",
    "R-LIC-MISMATCH": "Full text licence could not be confirmed, so it was not used.",
    "R-ABSTRACT": "Only an abstract was available; abstracts are not used as evidence.",
}
OUTCOME_TEXT = {  # outcomes that carry no reason code
    "admitted": OBTAINED_TEXT,  # a reading list means no answer was written, so "read for this answer" would be false
    "already_stored": OBTAINED_TEXT,
    "not_used": OBTAINED_TEXT,
    # NONASK-1103 S6: "in time" is for time stops only (R-ACQ-DEADLINE, R-ACQ-CRAWL_DELAY); with no recorded cause the
    # candidate was left unchecked by a limit of this request (quota, candidate cut or document budget), not by time
    "not_evaluated": "Not checked: a limit for this request was reached.",
    "not_reached": "Not checked: a limit for this request was reached.",
    "deferred": "Not checked: too many candidates for one request.",
}


def _reason_line(entry) -> str | None:
    """Plain-language outcome for one discovered entry; None when the entry is unusable."""
    if not isinstance(entry, dict):
        return None
    r = entry.get("reason")
    if isinstance(r, str) and r:
        return REASON_TEXT.get("R-ACQ-RATELIMITED" if r.upper().startswith("R-ACQ-RATELIMITED-") else r, REASON_GENERIC)
    o = entry.get("outcome")
    return OUTCOME_TEXT.get(o) if isinstance(o, str) and o in OUTCOME_TEXT else (REASON_GENERIC if o else None)


def _key(v) -> str:
    return v.strip().rstrip("/").lower() if isinstance(v, str) else ""


def _passage_keys(data: dict) -> set:
    """URLs and ids of documents whose passages were selected for the writer (public admitted_evidence); empty if none."""
    ae = data.get("admitted_evidence")
    out = set()
    for a in ae if isinstance(ae, list) else []:
        if isinstance(a, dict):
            out.update(_key(a.get(k)) for k in ("url", "canonical_source_id"))
    out.discard("")
    return out


def _reasons_by_url(data: dict) -> dict:
    c_ = data.get("contract")
    ia = c_.get("inline_acquisition") if isinstance(c_, dict) else None
    disc = ia.get("discovered") if isinstance(ia, dict) else None
    out = {}
    for e in disc if isinstance(disc, list) else []:
        line = _reason_line(e)
        if line and isinstance(e.get("url"), str) and e["url"].strip():
            out.setdefault(e["url"].strip().rstrip("/"), line)
    # DDGFIX-1023 DF-6: candidates found by DuckDuckGo (or any acquisition-search portal) get the same per-URL reason
    # line, keyed by the result URL the visitor sees (discovered_url) and the candidate URL. Discovery entries win.
    ro = data.get("research_outcome")
    for c in (ro.get("candidates") if isinstance(ro, dict) else None) or []:
        if not isinstance(c, dict):
            continue
        line = _reason_line(c)
        ids = c.get("ids") if isinstance(c.get("ids"), dict) else {}
        for u in (c.get("discovered_url"), ids.get("url"), c.get("url")):
            if line and isinstance(u, str) and u.strip():
                out.setdefault(u.strip().rstrip("/"), line)
    return out


_HANDLE = re.compile(r"(?:https?://)?hdl\.handle\.net/\S+", re.I)


def _is_handle(identifier) -> bool:
    """A Handle (hdl.handle.net) is a stable persistent identifier for display, even with no automatic route."""
    return isinstance(identifier, str) and _HANDLE.fullmatch(identifier.strip()) is not None


def _candidates(raw) -> list[dict]:
    """Public candidate fields only; non-lists and malformed items dropped; unknown acquisition -> not acquired."""
    if not isinstance(raw, list):
        return []
    out = []
    for c in raw:
        if not isinstance(c, dict) or not isinstance(c.get("title"), str) or not isinstance(c.get("url"), str):
            continue
        if not (c["title"].strip() or c["url"].strip()):
            continue
        state = c.get("acquisition")
        state = state if state in (answer_contract.ACQ_NOT_ACQUIRED, answer_contract.ACQ_ACQUIRED, answer_contract.ACQ_IN_CORPUS) \
            else answer_contract.ACQ_NOT_ACQUIRED
        out.append({"title": c["title"].strip(), "url": c["url"].strip(), "acquisition": state,
                    "published": c.get("published") if isinstance(c.get("published"), (str, int)) else None,
                    "identifier": c.get("identifier") if isinstance(c.get("identifier"), (str, int)) else None,
                    "has_licence": bool(c.get("licence_decision")),
                    "characterisation": c["characterisation"].strip() if isinstance(c.get("characterisation"), str) and c["characterisation"].strip() else None,
                    "reference": c["reference"].strip()[:2000] if isinstance(c.get("reference"), str) and c["reference"].strip() else None})
    return out


def _render_memory(st, data: dict) -> None:
    if _as_count(data.get("discovery_note_submitted")) > 0:
        st.info(answer_contract.text("discovery_note_submitted"))
    notes = data.get("recalled_discovery")
    if isinstance(notes, list) and notes:
        st.markdown("**Earlier reading-list notes**")
        st.caption(answer_contract.text("recalled_note_caption"))
        for item in notes:
            if not isinstance(item, dict) or not isinstance(item.get("note"), str):
                continue
            st.markdown(safe_markdown(apa7._p(item["note"])))  # C4: _p() already entity-encodes < >; plain() then safe_markdown() showed "&lt;"
            urls = [u for u in (item.get("urls") or []) if isinstance(u, str)] if isinstance(item.get("urls"), list) else []
            links = [_link(u, u) for u in urls]
            if links:
                st.markdown(" · ".join(links))


def render_discovery(st, data: dict) -> None:
    kind = data.get("result_kind")
    if kind in (KIND_DECLINE, KIND_SELF_DESCRIPTION):  # SELFREF-1028: text only, no reading-list heading or warning
        st.info(safe_markdown(data.get("answer") or ""))
        return
    cands = _candidates(data.get("candidates"))
    why = _reasons_by_url(data)
    if cands:  # NOANSWER-PAGE-1108 D1: the heading only over a list (never over an empty section)
        st.markdown("### Reading list")
    c_ = data.get("contract")
    if cands and isinstance(c_, dict) and c_.get("offtopic") is True and isinstance(data.get("answer"), str) \
            and data["answer"].strip():  # NOANSWER-PAGE-1108: no empty box once the page said it
        st.info(safe_markdown(data["answer"]))  # FIX1-1021: outside-scope notice shown above the list
    if cands:
        whys = [why.get(c["url"].rstrip("/")) if c["acquisition"] == answer_contract.ACQ_NOT_ACQUIRED else None
                for c in cands]
        used = _passage_keys(data)
        whys = [OBTAINED_CONSIDERED_TEXT if w == OBTAINED_TEXT and (_key(c["url"]) in used or _key(c["identifier"]) in used)
                else w for c, w in zip(cands, whys)]
        n_obt = sum(w in _OBTAINED_ANY for w in whys)
        n_held = sum(c["acquisition"] != answer_contract.ACQ_NOT_ACQUIRED for c in cands)
        n_none = len(cands) - n_obt - n_held
        any_stored_only = any(c["acquisition"] == answer_contract.ACQ_ACQUIRED for c in cands)
        if n_none == 0 and (n_obt or any_stored_only):  # every row holds a full text: the fixed "not acquired" line is false
            st.warning(OBTAINED_ALL)
        elif n_obt == 0:
            st.warning(answer_contract.not_read_line([{"acquisition": c["acquisition"]} for c in cands]))
        else:  # obtained rows together with unacquired rows (stored rows are named in the text when present)
            st.warning(" ".join((HEAD_UNMARKED if n_held else HEAD_OTHERS,
                                 HELD_BOTH if n_held else HELD_OBTAINED, BANNER_TAIL)))
        for i, c in enumerate(cands, 1):
            if c.get("reference"):  # NONASK-1103 C4: APA 7 entry (server-formatted, escaped) + type/outlet/date line
                st.markdown(f"{i}. " + safe_markdown(c["reference"]))
                parts = [safe_markdown(c["characterisation"])] if c.get("characterisation") else []
            else:
                line = f"{i}. {_link(c['title'] or c['url'], c['url'])}"
                if c["published"] not in (None, ""):
                    line += f" ({plain(c['published'])})"
                st.markdown(line)
                parts = []
            if c["identifier"] not in (None, "") and not c.get("reference"):  # the reference already ends with the identifier
                parts.append(plain_url(c["identifier"]))
            line_why = whys[i - 1]
            if line_why == REASON_TEXT["R-ACQ-NODOI"] and _is_handle(c["identifier"]):
                line_why = REASON_TEXT["R-ACQ-PID-NOROUTE"]  # a Handle is an identifier, so no 'No stable identifier'
            if line_why not in _OBTAINED_ANY:  # never pair "Not acquired" with "Full text obtained"
                parts.append(answer_contract.representation(c["acquisition"]))
            if line_why:
                parts.append(line_why)
            st.caption(" · ".join(parts))
        if any(c["has_licence"] for c in cands):
            st.caption(LICENCE_SIDEBAR)
    elif kind == KIND_READING_LIST:
        if (data.get("answer") or "").strip():  # NOANSWER-PAGE-1108: nothing left to say once the page said it
            st.info(safe_markdown(data["answer"]))
    else:
        st.warning(answer_contract.text("not_read_line"))
    _render_memory(st, data)
