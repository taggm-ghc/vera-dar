"""ANSWER-SCOPE-1109: evidence breadth and question scope for a written answer.

Live failure (build 731863f): "What are components of RAG?" was answered from ONE stored paper about one specific
robot-action RAG system, listing that paper's own system components as if they were RAG's general components. The
claim check passed every claim (supported is not the same as answers-the-question) and the answer path never asked
for more documents, because the owner's >= 6 full-text aim (project_ingest_six_then_answer) was enforced only after a
stored NON-answer.

This module holds the generic, data-driven pieces; no topic, question or name list lives here:

* distinct_documents(): how many distinct full documents an answer's admitted passages come from (the evidence floor
  in vera.research_record.sufficient and the limited-evidence note in vera.ask_service compare it with the configured
  minimum, config/auto_acquire.json freshness.sufficiency.min_distinct_documents).
* title_qualifiers(): the content words of a passage's title that the question does not use, after generic
  acronym matching (a question acronym covers the title words whose initials spell it). A hint to the writer and the
  mechanical fallback of the scope note, never a filter.
* scope_note(): the "this evidence describes one particular system" note, decided by the claim checker's per-claim
  "scope" verdict first and by the mechanical title signal only when the checker gave none.

Config: config/fulldoc_answer.json answer.scope_check (enabled=false restores the previous behaviour exactly).
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z0-9]+")
# Generic English function words and question words (language, not topic). Content words are everything else.
_STOP = frozenset("""a an and are as at be been but by can could did do does for from had has have how i if in into is
it its me my of on or our should so than that the their them then there these they this to up us was we were what when
where which who whom whose why will with would you your about via using use used based towards toward vs versus
""".split())
_STEM = 5  # prefix-stem length for matching question words to title words (generic morphology: component/components)


def scope_cfg(cfg: dict | None) -> dict | None:
    """answer.scope_check when it is a dict with enabled exactly True, else None (feature off)."""
    a = (cfg or {}).get("answer") if isinstance(cfg, dict) else None
    s = a.get("scope_check") if isinstance(a, dict) else None
    return s if isinstance(s, dict) and s.get("enabled") is True else None


def _content(text) -> list[str]:
    return [w for w in _WORD.findall(str(text or "")) if len(w) > 1 and w.lower() not in _STOP]


def _acronyms(question: str) -> list[str]:
    """Upper-case tokens of 2 to 6 letters written as such in the question (RAG, LLM): generic shape, no list."""
    return [w for w in _WORD.findall(str(question or "")) if 2 <= len(w) <= 6 and w.isalpha() and w.isupper()]


def _acronym_cover(acronyms: list[str], title_words: list[str]) -> set[int]:
    """Indexes of title words covered by a question acronym: the acronym itself, or a run of consecutive title words
    whose initials spell it ("retrieval augmented generation" for RAG)."""
    low = [w.lower() for w in title_words]
    out: set[int] = set()
    for ac in acronyms:
        a = ac.lower()
        for i, w in enumerate(low):
            if w == a:
                out.add(i)
            if i + len(a) <= len(low) and "".join(x[0] for x in low[i:i + len(a)]) == a:
                out.update(range(i, i + len(a)))
    return out


def title_qualifiers(question: str, title) -> list[str]:
    """Content words of `title` (lower case, in order, no repeats) that the question does not use."""
    words = _content(title)
    q = {w.lower()[:_STEM] for w in _content(question)}
    covered = _acronym_cover(_acronyms(question), _WORD.findall(str(title or "")))
    # _acronym_cover indexes all title words; map them onto the content-word positions
    all_words = _WORD.findall(str(title or ""))
    cov_words = {all_words[i].lower() for i in covered}
    out: list[str] = []
    for w in words:
        lw = w.lower()
        if lw[:_STEM] in q or lw in cov_words or lw in out:
            continue
        out.append(lw)
    return out


def is_specific(question: str, title, scfg: dict) -> bool:
    """Mechanical signal: the title carries at least min_title_qualifiers content words the question does not use."""
    try:
        n = int(scfg.get("min_title_qualifiers", 3))
    except (TypeError, ValueError):
        n = 3
    return bool(title) and len(title_qualifiers(question, title)) >= max(1, n)


def doc_key(a) -> str | None:
    """One key per full document (stored id, canonical source id, else URL, else title)."""
    if not isinstance(a, dict):
        return None
    for k in ("document_id", "canonical_source_id", "url", "title"):
        v = a.get(k)
        if v not in (None, ""):
            return f"{k}:{v}"
    return None


def distinct_documents(admitted, *, count_lane_pages: bool = True) -> int:
    """Distinct full documents behind the admitted passages. Lane pages (answer_only) count for answering
    (project_ingest_six_then_answer) unless count_lane_pages is False."""
    keys = set()
    for a in admitted or ():
        if not count_lane_pages and isinstance(a, dict) and a.get("answer_only") is True:
            continue
        k = doc_key(a)
        if k is not None:
            keys.add(k)
    return len(keys)


def payload_hint(question: str, source: dict, scfg: dict | None) -> dict:
    """Per-passage writer hint {"title_terms_not_in_question": [...]} (empty dict when off or nothing to say)."""
    if scfg is None or scfg.get("writer_hint", True) is not True:
        return {}
    q = title_qualifiers(question, source.get("title"))
    return {"title_terms_not_in_question": q[:12]} if q else {}


def scope_note(question: str, shown, sources, scfg: dict | None) -> str | None:
    """The scope note text when every shown claim describes one particular system rather than the question's subject
    in general; None otherwise. Checker verdicts ("scope": general|specific) decide; only when the checker gave no
    scope verdict at all does the mechanical title signal decide (every cited source is is_specific)."""
    if scfg is None or not shown:
        return None
    vals = [r.get("scope") for r in shown]
    by_n = {s.get("n"): s for s in sources or () if isinstance(s, dict)}
    cited = sorted({n for r in shown for n in (r.get("citations") or ())})
    titles = [str(by_n[n].get("title") or "").strip() for n in cited if n in by_n]
    if any(v is not None for v in vals):
        specific = all(v == "specific" for v in vals)
    elif scfg.get("mechanical_fallback") is True:  # off by default: titles alone over-flag (general surveys)
        specific = bool(titles) and all(is_specific(question, t, scfg) for t in titles)
    else:
        specific = False
    if not specific:
        return None
    uniq = []
    for t in titles:
        if t and t not in uniq:
            uniq.append(t)
    text = str(scfg.get("specific_scope_text") or "")
    if not text:
        return None
    return text.format(titles="; ".join(f"“{t[:160]}”" for t in uniq[:3]) or "the cited documents")


def floor_of(fcfg: dict | None) -> int | None:
    """freshness.sufficiency.min_distinct_documents (a positive int) or None (floor off)."""
    v = ((fcfg or {}).get("sufficiency") or {}).get("min_distinct_documents")
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None


def below_floor(result, fcfg: dict | None) -> tuple[int, int] | None:
    """(documents used, floor) when a grounded answer draws on fewer distinct documents than the floor, else None."""
    m = floor_of(fcfg)
    ev = getattr(result, "evidence", None)
    if m is None or ev is None or not getattr(ev, "admitted", None):
        return None
    lanes = ((fcfg or {}).get("sufficiency") or {}).get("count_lane_pages", True) is not False
    n = distinct_documents(ev.admitted, count_lane_pages=lanes)
    return (n, m) if n < m else None
