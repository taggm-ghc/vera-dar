"""NONASK-1103 E1: the evidence trace, a pure projection of fulldoc outcome data. Stdlib only, no I/O except reading
config/evidence_trace.json (labels). No model call, no threshold, no network.

project(evidence, claim_records, citations, adequacy=None, inline=None, public=True, *, sources=None, labels=None)
  public=True   built from a per-field allowlist: passage rows (n, cited, supports, verdict label, sentence-snapped excerpt,
                plain reason for "selected, not cited"), declared parts and the adequacy verdict. No relevance, span_id,
                document ids, raw codes, factors or LLM reason text. Counts (claims, documents, not_used by category) appear
                ONLY when they back a visible caveat (adequacy qualify/abstain, an uncovered part, or a removed claim).
  public=False  dev view: adds raw codes, per-candidate Gate A/B rows, relevance, span_id and all counts. Built only
                when public mode is off (otherwise None).
Returns None (fail closed) when the label file is missing/invalid/disabled. No-answer outcomes (claim_records and
citations None, evidence only) are accepted. An unknown reason code falls into the "other reason" category.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LABELS_PATH = Path(__file__).resolve().parent.parent / "config" / "evidence_trace.json"
QUOTE_CAP = 200  # existing answer.quote_max_chars
QUOTE_TOTAL_CAP = 1200  # existing max_total_quote_chars
_SENT_END = re.compile(r"(?<=[.!?])\s+")
_CLAUSE = re.compile(r"(?<=[;:])\s+|(?<=,)\s+")
_CITE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_WORD = re.compile(r"[a-z0-9]+")
_NOT_CITED_KEYS = ("removed_by_check", "unused")


def load_labels(path=None) -> dict | None:
    """Validated labels, or None (missing, unreadable, invalid or enabled != true)."""
    try:
        d = json.loads(Path(path or LABELS_PATH).read_text(encoding="utf-8"))
        if not isinstance(d, dict) or d.get("enabled") is not True:
            return None
        cats = d["categories"]
        if not isinstance(cats, dict) or not cats:
            return None
        for c in cats.values():
            if not (isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip()
                    and isinstance(c.get("codes"), list) and all(isinstance(x, str) for x in c["codes"])):
                return None
        if not (isinstance(d["other"], dict) and isinstance(d["other"].get("text"), str) and d["other"]["text"].strip()):
            return None
        nc = d["not_cited"]
        if not (isinstance(nc, dict) and all(isinstance(nc.get(k), str) and nc[k].strip() for k in _NOT_CITED_KEYS)):
            return None
        v = d["verdicts"]
        if not (isinstance(v, dict) and all(isinstance(v.get(k), str) and v[k].strip() for k in ("supported", "partial"))):
            return None
        ep = d.get("epistemic")
        if ep is not None and not (isinstance(ep, dict) and isinstance(ep.get("reported"), str) and ep["reported"].strip()):
            return None
        return d
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _category(code, labels) -> tuple[str, str]:
    """-> (category id, public text); unknown code -> ("other", other text)."""
    for cid, c in labels["categories"].items():
        if code in c["codes"]:
            return cid, c["text"]
    return "other", labels["other"]["text"]


# ----------------------------------------------------------------------------------------------- excerpt snapping
def _words(t: str) -> set:
    return set(_WORD.findall((t or "").lower()))


def _word_trim(text: str, cap: int) -> str:
    text = text.strip()
    if len(text) <= cap:
        return text
    cut = text[:cap]
    if not text[cap].isspace():
        sp = cut.rfind(" ")
        if sp > 0:
            cut = cut[:sp]
    return cut.rstrip()


def snap_excerpt(span_text: str, claim_sentence: str = "", cap: int = QUOTE_CAP) -> str:
    """Verbatim substring of span_text, starting and ending on a sentence boundary when one fits inside `cap`.
    Picks the sentence with the highest word overlap with the claim. A sentence longer than cap is cut at the last
    clause boundary inside cap; with none, at a word boundary (today's trim). No minimum length, no ellipsis."""
    span_text = span_text or ""
    cw = _words(_CITE.sub(" ", claim_sentence or ""))
    best, best_score, pos = None, -1, 0
    for part in _SENT_END.split(span_text):
        i = span_text.find(part, pos)
        if i < 0 or not part.strip():
            continue
        pos = i + len(part)
        score = len(cw & _words(part))
        if score > best_score:
            best, best_score = part.strip(), score
    sent = best if best is not None else span_text.strip()
    if len(sent) > cap:
        cuts = [m.start() for m in _CLAUSE.finditer(sent) if m.start() + 1 <= cap]
        sent = sent[:cuts[-1] + 1].rstrip() if cuts else _word_trim(sent, cap)
    if not sent:
        sent = span_text.strip()[:cap].rstrip()
    assert sent in span_text
    return sent


def _budget_excerpts(rows: list[dict], total_cap: int) -> None:
    """Keep the total of excerpt characters strictly below total_cap: later rows lose their excerpt first."""
    used = 0
    for r in rows:
        q = r.get("excerpt")
        if q:
            if used + len(q) < total_cap:
                used += len(q)
            else:
                r["excerpt"] = None


# ----------------------------------------------------------------------------------------------- helpers
def _get(o, k, default=None):
    if o is None:
        return default
    m = o if isinstance(o, dict) else vars(o) if hasattr(o, "__dict__") else {}  # dict or dataclass fields; no dynamic getattr
    return m.get(k, default)


def _dicts(seq) -> list[dict]:
    return [x for x in (seq or ()) if isinstance(x, dict)]


def _is_public_mode() -> bool:
    try:
        from vera import public_mode
        return bool(public_mode.is_public_mode())
    except Exception:  # unknown state counts as public: fail closed
        return True


def _bool(x):
    return x if isinstance(x, bool) else None


def _reference(src) -> str | None:
    try:
        from vera import references_ui

        return references_ui.citation_strings([src])[0]["reference"] if src else None
    except Exception:  # noqa: BLE001  (a formatter fault never breaks the trace)
        return None


def project(evidence, claim_records=None, citations=None, adequacy=None, inline=None, public: bool = True, *,
            sources=None, labels=None, quote_cap: int = QUOTE_CAP, total_cap: int = QUOTE_TOTAL_CAP,
            labels_path=None, refusal=None) -> dict | None:
    lab = labels if labels is not None else load_labels(labels_path)
    if lab is None:
        return None
    if public is not True and _is_public_mode():
        return None  # the dev view is never built in public mode
    dev = public is not True
    admitted = _dicts(_get(evidence, "admitted"))
    rejected = _dicts(_get(evidence, "rejected"))
    unusable = _dicts(_get(evidence, "unusable"))
    examined = _dicts(_get(evidence, "examined_full_documents"))
    records = _dicts(claim_records)
    cites = _dicts(citations)
    by_n = {s.get("n"): s for s in _dicts(sources)}
    cited_ns = {c.get("n") for c in cites}

    rows = []
    for i, a in enumerate(admitted, 1):
        n = a.get("n", i)
        src = by_n.get(n) or {}
        span_text = src.get("snippet") or a.get("text") or ""
        mine = [r for r in records if n in (r.get("citations") or ())]
        kept = [r for r in mine if r.get("verdict") in ("supported", "partial")
                and any(c.get("n") == n and c.get("claim_id") == r.get("id") for c in cites)]
        cited = n in cited_ns
        row = {"n": n, "cited": cited}
        if cited:
            row["supports"] = [str(r.get("sentence", "")) for r in kept]
            verdicts = {r.get("verdict") for r in kept}
            row["verdict"] = lab["verdicts"]["supported" if verdicts == {"supported"} else "partial"] if kept else None
            if (src.get("origin") or a.get("origin")) == "reported":  # E-S1: a reported source is never labelled verified
                row["verdict"] = (lab.get("epistemic") or {}).get("reported", "reported, not independently verified")
                row["epistemic"] = "reported"
            elif "web" in (src.get("origin") or a.get("origin"), src.get("lane") or a.get("lane")):  # S3: a web source is never labelled verified
                row["verdict"] = "web source"
                row["epistemic"] = "web"
            first = kept[0].get("sentence", "") if kept else ""
            cq = next((c.get("quote") for c in cites if c.get("n") == n and c.get("quote")), None)
            base = span_text or cq or ""
            row["excerpt"] = snap_excerpt(base, first, quote_cap) if (cq or span_text) and not src.get("answer_only") else None  # R-037: no lane page text
            row["reason"] = None
        else:
            removed = any(r.get("verdict") in ("unsupported", "partial") for r in mine)
            by_refusal = None
            if refusal and not removed:  # F3: fixed label per code; an unknown code gets the neutral one, never the code
                br = lab.get("by_refusal") or {}
                by_refusal = br.get(str(refusal)) or lab.get("refusal_other")
            row["reason"] = by_refusal or lab["not_cited"]["removed_by_check" if removed else "unused"]
            row["supports"], row["verdict"], row["excerpt"] = [], None, None
        if dev:
            ref = _reference(src)  # NONASK-1103 C6: dev record carries the APA 7 reference (public rows stay URL-free: E allowlist)
            if ref:
                row["reference"] = ref
            row.update(span_id=a.get("span_id"), relevance=a.get("relevance"), document_id=a.get("document_id"),
                       span=src.get("span"), quote_span=next((c.get("quote_span") for c in cites if c.get("n") == n), None),
                       not_cited_kind=None if cited else ("removed_by_check" if row["reason"] == lab["not_cited"]["removed_by_check"]
                                                                  else (str(refusal) if by_refusal else "unused")))
        rows.append(row)
    _budget_excerpts(rows, total_cap)

    # reasons for candidates/documents/passages not used (many-to-few)
    not_used: dict[str, int] = {}
    coded: list[tuple[str, str]] = []  # (stage, code)
    for r in rejected:
        coded.append(("gate_b_passage", str(r.get("reason"))))
    for u in unusable:
        coded.append(("document", str(u.get("code"))))
    for c in _dicts(_get(inline, "candidates")):
        if c.get("reason"):
            coded.append(("gate_a_candidate", str(c["reason"])))
    for stage, code in coded:
        _cid, text = _category(code, lab)
        not_used[text] = not_used.get(text, 0) + 1

    ad = adequacy if isinstance(adequacy, dict) else {}
    parts = []
    for p in _dicts(ad.get("parts")):
        parts.append({"label": str(p.get("label", p.get("part", "")))[:300],
                      "covered": bool(p.get("supported", p.get("covered", False)))})
    verdict = ad.get("verdict") if isinstance(ad.get("verdict"), str) else None
    counts = {"supported": sum(1 for r in records if r.get("verdict") == "supported"),
              "partial": sum(1 for r in records if r.get("verdict") == "partial"),
              "unsupported": sum(1 for r in records if r.get("verdict") == "unsupported")}
    removed = counts["partial"] + counts["unsupported"] > 0 and any(
        not any(c.get("claim_id") == r.get("id") for c in cites) for r in records if r.get("verdict") != "supported")
    caveat = (verdict in ("qualify", "abstain")) or any(not p["covered"] for p in parts) or removed

    subs = [{"label": str(q.get("label", ""))[:300], "verdict": q.get("verdict") if isinstance(q.get("verdict"), str) else None}
            for q in _dicts(ad.get("subqueries"))]  # public allowlist: label + verdict only
    out: dict = {"version": 1, "view": "dev" if dev else "public", "passages": rows,
                 "parts": parts,
                 "adequacy": {"verdict": verdict, "all_declared_parts_supported": _bool(ad.get("all_declared_parts_supported"))}
                 if ad else None}
    if subs:
        out["adequacy"]["subqueries"] = subs
    if caveat or dev:
        out["claims"] = counts
        out["documents"] = {"examined_by_corpus": len(examined), "used":  # NOANSWER-RC-1107 r3: never shown as "inspected"
                             len({a.get("document_id") for a in admitted
                                                                    if a.get("document_id") is not None})}
        out["not_used"] = not_used
    if dev:
        out["codes"] = [{"stage": s, "code": c, "category": _category(c, lab)[0]} for s, c in coded]
        out["gate_a"] = [{k: d.get(k) for k in ("url", "outcome", "reason", "gate_a", "gate_b", "service", "portal")}
                         for d in _dicts(_get(inline, "candidates")) + _dicts(_get(inline, "discovered"))]
        out["gate_b"] = [{"span_id": r.get("span_id"), "reason": r.get("reason")} for r in rejected]
        out["admitted_spans"] = len(admitted)
    return out
