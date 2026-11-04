"""NONASK-1103 C2/C5a: the bridge from VERA source dicts to the APA 7 formatter (vera/apa7.py) and the answer-area
References block. Pure functions. Never invents metadata; every string comes from apa7 (escaped there)."""
from __future__ import annotations

import re

from vera import apa7

_TYPES = {"preprint": "preprint", "article-journal": "article-journal", "journal-article": "article-journal",
          "journal": "article-journal", "article": "article-journal", "news": "article-newspaper",
          "article-newspaper": "article-newspaper", "webpage": "webpage", "dataset": "dataset",
          "software": "software", "book": "book", "chapter": "chapter", "report": "report", "thesis": "thesis"}
_ARXIV_ID = re.compile(r"^arxiv:\s*(\S+)$", re.I)


def _parts(value) -> list:
    m = re.match(r"\s*(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", str(value or ""))
    if not m:
        return []
    return [int(g) for g in m.groups() if g is not None][:3]


def csl_from_source(s: dict) -> dict:
    """CSL-JSON-shaped record from a VERA source / candidate dict. Authors are carried as given (a joined string is
    ONE literal author, never split). Type only from a named input field (source_type, else provider arXiv)."""
    s = s if isinstance(s, dict) else {}
    typ = _TYPES.get(str(s.get("source_type") or s.get("type") or "").lower(), "")
    if not typ and str(s.get("provider") or "").lower().startswith("arxiv"):
        typ = "preprint"
    rec: dict = {"title": s.get("title") or "", "URL": s.get("url") or ""}
    if typ:
        rec["type"] = typ
    au = s.get("authors") if s.get("authors") not in (None, "", []) else s.get("author")
    if au:
        rec["author"] = au if isinstance(au, str) else [a for a in au if a]
    p = _parts(s.get("published"))
    if p:
        rec["issued"] = {"date-parts": [p]}
    if s.get("doi"):
        rec["DOI"] = s["doi"]
    ident = str(s.get("identifier") or "")
    m = _ARXIV_ID.match(ident)
    if m:
        rec["arxiv"] = m.group(1)
    elif ident.lower().startswith("doi:") and "DOI" not in rec:
        rec["DOI"] = ident[4:]
    from vera.grounded_ask import _public_pmcid  # PMCID-PUBLIC-1023: only a pmcid the public path already shows

    if _public_pmcid(s) and not (rec.get("DOI") or rec.get("arxiv")):  # PMCID only as the primary id; with a DOI it stays API-only (g23 leak pin)
        rec["PMCID"] = _public_pmcid(s)
    cont = s.get("container_title") or s.get("journal")
    pub = s.get("publisher")
    if typ == "article-journal" and not cont and pub:
        cont, pub = pub, None  # providers put the venue name in `publisher`
    if cont:
        rec["container_title"] = cont
    if pub and typ != "article-journal":
        host = re.sub(r"^https?://(?:www\.)?([^/:]+).*$", r"\1", str(s.get("url") or "").lower())
        if str(pub).lower().removeprefix("www.") != host:  # a bare domain is not an outlet name (plan 2a)
            rec["publisher"] = pub
    a = _parts(s.get("retrieved_at"))
    if len(a) == 3:
        rec["accessed"] = {"date-parts": [a]}
    return rec


def legacy() -> bool:
    return apa7.citation_style() == "legacy"


def _doc_key(s: dict):
    return s.get("canonical_source_id") or s.get("doi") or (s.get("url") or "").rstrip("/").lower() or id(s)


def citation_strings(sources) -> list:
    """Aligned [{reference, in_text}] for a list of source dicts. Passages of one document share one entry; a/b
    suffixes are assigned across the DISTINCT documents of this answer (author-year clashes). {None, None} when the
    style flag is 'legacy' or formatting fails (a formatter fault never breaks the answer path)."""
    none = {"reference": None, "in_text": None, "characterisation": None}
    srcs = [s if isinstance(s, dict) else {} for s in (sources or [])]
    if legacy():
        return [dict(none) for _ in srcs]
    try:
        keys, docs = [], []
        for s in srcs:
            k = _doc_key(s)
            if k not in keys:
                keys.append(k)
                docs.append(csl_from_source(s))
        sfx = apa7.suffixes(docs)
        refs = [apa7.reference(d, suffix=sf) or None for d, sf in zip(docs, sfx)]
        its = [apa7.in_text(d, context=docs) or None for d in docs]
        chs = [apa7.characterise({k: v for k, v in d.items() if k != "URL"}) or None for d in docs]  # no URL: never a domain guess (M1)
        return [{"reference": refs[i], "in_text": its[i], "characterisation": chs[i]}
                for i in (keys.index(_doc_key(s)) for s in srcs)]
    except Exception:  # noqa: BLE001
        return [dict(none) for _ in srcs]


def sources_by_n(sources) -> dict:
    """n -> public source row (only rows with an int n)."""
    return {s["n"]: s for s in (sources if isinstance(sources, list) else [])
            if isinstance(s, dict) and isinstance(s.get("n"), int) and not isinstance(s.get("n"), bool)}


def cited_set(answer) -> set | None:
    """Source numbers cited by the answer text, from the [n] markers BEFORE any display substitution (the same
    function the sidebar uses: sources_sidebar.cited_numbers)."""
    from vera.sources_sidebar import cited_numbers

    return cited_numbers(answer)


_RUN = re.compile(r"(?:\[\d+(?:\s*,\s*\d+)*\])+")


def display_answer(answer, sources):
    """Display-only string: each run of [n] markers becomes one APA parenthetical built from the PUBLIC `in_text`
    strings of the sources plus the [n] locator, e.g. "(Alpha, 2020; Zeta, 2018) [1, 2]". Returns a new string; the
    caller's answer and sources are never changed. Unknown n, or rows without in_text, leave the run as it was."""
    if not isinstance(answer, str):
        return answer
    by_n = sources_by_n(sources)

    def inner(n):
        s = by_n.get(n)
        v = s.get("in_text") if s else None
        return v[1:-1] if isinstance(v, str) and v.startswith("(") and v.endswith(")") else None

    def sub(m):
        nums = sorted({int(x) for x in re.findall(r"\d+", m.group(0))})
        known = [n for n in nums if inner(n)]
        if not known:
            return m.group(0)
        parts = sorted({inner(n) for n in known}, key=str.casefold)
        return "(" + "; ".join(parts) + ") [" + ", ".join(str(n) for n in known) + "]"
    return _RUN.sub(sub, answer)


def reference_block(sources, cited=None) -> list[str]:
    """Alphabetical, de-duplicated APA references of the (cited) public source rows (their `reference` strings)."""
    out = {s["reference"] for n, s in sources_by_n(sources).items()
           if (cited is None or n in cited) and isinstance(s.get("reference"), str) and s["reference"].strip()}
    return sorted(out, key=str.casefold)


def answer_view(data) -> dict:
    """C5a display hook. Order matters (KC-3): the cited set comes from the ORIGINAL [n] markers first; then a local
    display string is built by substitution; `data` is never modified (data["answer"] stays the raw text)."""
    data = data if isinstance(data, dict) else {}
    answer = data.get("answer")
    cited = cited_set(answer)
    return {"cited": cited, "display": display_answer(answer, data.get("sources")),
            "references": reference_block(data.get("sources"), cited)}


def render_references(st, refs) -> None:
    """References block in the ANSWER area (never the sidebar), with the house-variant caption."""
    from vera.ui_safety import safe_markdown

    refs = [r for r in (refs or []) if isinstance(r, str) and r.strip()]
    if not refs:
        return
    st.markdown("**References**")
    for r in refs:
        st.markdown(safe_markdown(r))
    st.caption(apa7.HOUSE_CAPTION)
