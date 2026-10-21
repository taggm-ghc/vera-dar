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

from vera.sources_sidebar import _link, plain
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
    "model_insufficient": "insufficient_text",
}
_TOKEN_DEFAULT = {
    "insufficient_full_document_evidence": "VERA-DAR found no passage in the documents it holds that answers this question well enough, so it has not written an answer.",
    "full_document_corpus_unusable": "VERA-DAR's full-document collection could not be verified right now, so it has not written an answer.",
    "no_valid_citations": "The draft answer did not cite the inspected passages, so it was not shown.",
    "checker_unavailable": "The automated claim check was unavailable, so no answer is shown.",
    "claim_check_insufficient": "The automated claim check did not support enough of the draft, so no answer is shown.",
    "claim_check_disabled": "The automated claim check is off, so no answer is shown.",
    "model_insufficient": "The documents inspected did not give enough to answer, so no answer is shown.",
}
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


def fallback_text(token, cfg: dict | None = None, n_inspected: int = 0) -> str:
    """Visitor sentence for a contract.fallback token (None/unknown -> generic). Never returns the token."""
    if not isinstance(token, str) or not token:
        return GENERIC_FALLBACK
    if token not in _TOKEN_DEFAULT:
        return GENERIC_FALLBACK
    cfg = _cfg() if cfg is None else cfg
    key = _TOKEN_TEXT.get(token)
    val = cfg.get(key) if key else None
    if isinstance(val, str) and val.strip():
        try:
            out = val.format(n_inspected=_count(n_inspected))
        except (KeyError, IndexError, ValueError):
            return _TOKEN_DEFAULT[token]
        return out if "{" not in out else _TOKEN_DEFAULT[token]
    return _TOKEN_DEFAULT[token]


def evidence_scope_line(data: dict | None) -> str | None:
    """'N documents inspected, M passages admitted' from the response, or None when it carries no evidence fields."""
    data = data if isinstance(data, dict) else {}
    if data.get("evidence_status") is None and "examined_full_documents" not in data and "admitted_evidence" not in data:
        return None
    n, m = _count(data.get("examined_full_documents")), _count(data.get("admitted_evidence"))
    return (f"Evidence scope: {n} document{'s' if n != 1 else ''} inspected, "
            f"{m} passage{'s' if m != 1 else ''} admitted as evidence.")


_SOURCE_LABEL = {"db": "stored database", "bundle": "local files"}


def evidence_source_line(data: dict | None) -> str | None:
    """'Evidence source: stored database / local files' from the response's corpus_source.kind; None otherwise."""
    cs = data.get("corpus_source") if isinstance(data, dict) else None
    label = _SOURCE_LABEL.get(cs.get("kind")) if isinstance(cs, dict) and isinstance(cs.get("kind"), str) else None
    return f"Evidence source: {label}." if label else None


def render_evidence_scope(st, data) -> None:
    line = evidence_scope_line(data)
    if line:
        st.caption(line)
        src = evidence_source_line(data)
        if src:
            st.caption(src)


def render_fallback_reason(st, data) -> None:
    """Non-answer result from the full-document path: say why there is no answer. No-op for plain web-only responses."""
    data = data if isinstance(data, dict) else {}
    contract = data.get("contract") if isinstance(data.get("contract"), dict) else {}
    token = contract.get("fallback")
    if token is None and data.get("evidence_status") is None:
        return
    st.info(fallback_text(token, n_inspected=_count(data.get("examined_full_documents"))))
    render_evidence_scope(st, data)


def _quote_cap() -> int:
    try:
        return max(int(_cfg()["quote_max_chars"]), 1)
    except (KeyError, TypeError, ValueError):
        return 200


def _clean(text, limit: int = 400) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()[:limit]


def render_citations(st, data) -> None:
    """Per cited claim: [n] verdict, short quote (or span reference only), then CC BY attribution once per source."""
    cites = _list((data if isinstance(data, dict) else {}).get("citations"))
    if not cites:
        return
    st.markdown("**Cited passages** (short excerpts)")
    shown: set = set()
    for c in cites:
        n = c.get("n") if isinstance(c.get("n"), int) and not isinstance(c.get("n"), bool) else "?"
        verdict = {"supported": "supported", "partial": "partly supported"}.get(c.get("verdict"), "checked")
        span = _clean(c.get("quote_span") or c.get("span") or "", 80)
        where = f" (passage {plain(span)})" if span else ""
        quote = _clean(c.get("quote"), _quote_cap()) if isinstance(c.get("quote"), str) else ""
        if quote:
            st.markdown(f"[{n}] {verdict}: “{plain(quote)}”{where}")
        else:
            ref = _clean(c.get("span") or "", 80)
            st.markdown(f"[{n}] {verdict}: excerpt not shown{f' (passage {plain(ref)})' if ref else ''}")
        at = c.get("attribution")
        if isinstance(at, dict) and n not in shown:
            shown.add(n)
            parts = [plain(at["creator"]) if at.get("creator") else None,
                     _link(at.get("title") or "source", at.get("source")) if at.get("title") or at.get("source") else None,
                     (_link(at["licence"], at.get("licence_url")) if at.get("licence") else None)]
            line = ". ".join(p for p in parts if p)
            if at.get("changes"):
                line += f". Changes: {plain(at['changes'])}"
            if line:
                st.caption(f"[{n}] " + line)


def claim_check_text(cc, cap: int | None = None, fulldoc: bool = False) -> str | None:
    """Caption text for the claim check. Legacy abstract wording is kept for non-full-document answers."""
    if not isinstance(cc, dict):
        return None
    if cc.get("outcome") != "checked":
        return NOT_CHECKED_NOTE if cc else None
    checked, supported, partial, removed = (_count(cc.get(k)) for k in ("checked", "supported", "partial", "removed"))
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


def render_claim_check(st, cc, cap: int | None = None, fulldoc: bool = False) -> None:
    text = claim_check_text(cc, cap, fulldoc)
    if text:
        st.caption(text)


def render_answer_scope_note(st) -> None:
    st.caption(ANSWER_SCOPE_NOTE)
