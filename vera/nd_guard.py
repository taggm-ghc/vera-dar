"""H19-T5 (OD-1106-NEWS, design section 4 and AD-4/AD-6): the "no rewritten version" output guard for spans with
licence_class == "nd" (CC BY-NC-ND news). Pure and offline: no I/O, nothing stored, never edits a sentence (drops only).

Contract with the span producer (T3): a source dict has `n`, `snippet`, optional `title`, `url`, and `licence_class`
"nd"; `canonical_text` (the FULL canonical article text, A11-filtered) is used for G1/G4/G5 and is REQUIRED (fail closed: without it the ND-citing sentences are dropped).
A record (claim) has `id`, `sentence`, `citations` (list of source numbers), `verdict`.

Checks (all thresholds are PRIORS, uncalibrated; see config/fulldoc_answer.json answer.nd_guard._calibration_note):
  G1 verbatim run >= min_verbatim_tokens against the canonical text -> drop sentence
  G2 sentence share <= max_share, G3 absolute count <= max_sentences per ND source -> drop lowest-ranked citing sentences
  G4 article coverage <= max_coverage (stem-set Jaccard >= match_jaccard) -> drop highest-coverage answer sentences
  G5 longest in-article-order run of answer sentences <= max_sequence -> drop the run's tail
  G6 ND quotes: each <= quote_max_chars, total per source <= quote_total_chars (limit_quotes, after build_citations)
  G8 nothing left -> the caller's existing require_supported_claim / honest-notice path
  G9 (AD-4) a request to summarise, paraphrase, rewrite, retell, translate or restyle ONE ND article: every sentence that
     cites that source is dropped and the answer for it is short attributed quotes plus the link (g9_quote_and_link);
     an ambiguous single-source request fails closed to the same form.
Fail closed (review F2/F5): an ND source with no canonical_text, or answer.nd_guard.enabled=false, DROPS every sentence
that cites that ND source (apply never passes ND-derived sentences unchecked).
G4/G5 use ASYMMETRIC containment, not symmetric Jaccard: an article sentence is "covered" by an answer sentence when
>= article_containment of the article sentence's stems occur in the answer sentence (so one long condensed sentence can
cover many article sentences); an answer sentence is a near-copy when >= near_copy of its own stems occur in ONE article
sentence. G10 word budget: content words of kept sentences citing one ND source <= max_words_per_source.
No ND span in the sources -> apply() returns its input unchanged (same list object).
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

_WORD = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_STOP = frozenset("a an the and or but of to in on at by for with from as is are was were be been it its this that these "
                  "those he she they them his her their we you i not no".split())
DEFAULTS = {"enabled": True, "min_verbatim_tokens": 8, "max_share": 0.4, "max_sentences": 4, "max_coverage": 0.25,
            "match_jaccard": 0.5, "max_sequence": 3, "quote_max_chars": 200, "quote_total_chars": 400,
            "article_containment": 0.6, "near_copy": 0.8, "near_copy_min_stems": 5, "max_cover_per_sentence": 3,
            "min_article_sentence_stems": 3, "max_words_per_source": 60, "union_containment": 0.7,
            "g9_cues": [r"summari[sz]\w*", r"summary of", r"sum (?:it |this |that )?up", r"paraphras\w*", r"rephras\w*",
                        r"rewr(?:ite|iting|ites|itten|ote)", r"retell\w*", r"retold", r"recap\w*", r"tl;?\s?dr",
                        r"translat\w*", r"in my own words", r"in simple terms", r"reword\w*", r"restyl\w*",
                        r"shorten\w*", r"condens\w*", r"abridg\w*", r"key (?:points|takeaways|facts|findings)",
                        r"(?:main |key )?takeaways?", r"gist", r"overview of", r"explain (?:this|the|that) "
                        r"(?:article|story|piece|post|report)", r"break(?:ing)? (?:it|this|that|the "
                        r"(?:article|story|piece)) down", r"what (?:does|did) (?:this|the|that) "
                        r"(?:article|story|piece|post|report) say", r"what (?:is|was) (?:this|the|that) "
                        r"(?:article|story|piece|post|report) about"]}
_DEICTIC = re.compile(r"\b(this|that|the|above|attached|linked) (article|story|piece|post|link|report)\b|\bthis link\b",
                      re.I)
_URL = re.compile(r"https?://[^\s<>\")\]]+", re.I)


def cfg_of(cfg: dict | None) -> dict:
    nd = ((cfg or {}).get("answer") or {}).get("nd_guard") if isinstance(cfg, dict) and "answer" in cfg else cfg
    return {**DEFAULTS, **(nd if isinstance(nd, dict) else {})}


def nd_sources(sources) -> dict:
    return {s["n"]: s for s in sources or [] if isinstance(s, dict) and s.get("licence_class") == "nd" and "n" in s}


def _toks(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def _stem(t: str) -> str:
    for suf in ("ing", "edly", "ed", "es", "ly", "s"):
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def _stems(text: str) -> frozenset:
    text = re.sub(r"\[\d+\]", " ", text or "")  # citation markers are not content
    return frozenset(_stem(t) for t in _toks(text) if t not in _STOP)


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _text(s: dict) -> str:
    return str(s.get("canonical_text") or s.get("snippet") or "")


def _ngrams(toks: list[str], n: int) -> set:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)} if n > 0 else set()


def _article_sents(s: dict) -> list[frozenset]:
    return [st for st in (_stems(x) for x in _SENT.split(_text(s))) if st]


def _rank_worst(r: dict, by_n: dict, pos: int) -> tuple:
    rel = max([float(by_n[n].get("relevance") or 0.0) for n in r.get("citations") or [] if n in by_n] or [0.0])
    return (r.get("verdict") == "partial", -rel, pos)


def _union_near(sentence: str, art: list[frozenset], c: dict) -> bool:
    """Asymmetric containment of the answer sentence in the WHOLE article vocabulary: >= union_containment of its stems
    occur somewhere in the article (a condensed paraphrase re-uses the article's words across several sentences)."""
    a = _stems(sentence)
    if len(a) < int(c["near_copy_min_stems"]):
        return False
    u = set().union(*art) if art else set()
    return len(a & u) / len(a) >= float(c["union_containment"])


def _cover(sentence: str, art: list[frozenset], c: dict) -> tuple[set, int | None, bool]:
    """(indexes of article sentences covered, best covered index or None, near_copy) for one answer sentence.
    covered: |S & A| / |S| >= article_containment (asymmetric: S is the article sentence, A the answer sentence).
    near_copy: |A| >= near_copy_min_stems and |A & S| / |A| >= near_copy for ONE article sentence."""
    a = _stems(sentence)
    if not a:
        return set(), None, False
    thr, nc = float(c["article_containment"]), float(c["near_copy"])
    cov, best, bs, near = set(), None, 0.0, False
    for i, st in enumerate(art):
        inter = len(st & a)
        if len(st) >= int(c["min_article_sentence_stems"]) and inter / len(st) >= thr:
            cov.add(i)
            if inter / len(st) > bs:
                best, bs = i, inter / len(st)
        if len(a) >= int(c["near_copy_min_stems"]) and inter / len(a) >= nc:
            near = True
    return cov, best, near


_CUE_CACHE: dict = {}


def request_cue(question: str, cfg: dict | None = None) -> bool:
    cues = tuple(x for x in cfg_of(cfg).get("g9_cues") or [] if isinstance(x, str) and x)
    if cues not in _CUE_CACHE:
        try:
            _CUE_CACHE[cues] = re.compile(r"\b(?:" + "|".join(cues) + r")\b", re.I)
        except re.error:  # a broken config list must not disable the check: fall back to the shipped list
            _CUE_CACHE[cues] = re.compile(r"\b(?:" + "|".join(DEFAULTS["g9_cues"]) + r")\b", re.I)
    return bool(_CUE_CACHE[cues].search(question or ""))


def g9_sources(question: str, sources, cfg: dict | None = None) -> list[int]:
    """Source numbers G9 forces to quote-and-link: a summarise/paraphrase/rewrite/translate/restyle cue AND the question
    names one ND source (its URL host+path or its title). A cue with no resolvable target but a deictic reference
    ("this article", a link) while ND sources are present fails closed: every ND source is forced. [] otherwise."""
    nd = nd_sources(sources)
    if not nd or not request_cue(question, cfg):
        return []
    q = question or ""
    low = q.lower()
    urls = [u.rstrip(".,;") for u in _URL.findall(q)]
    hit = []
    for n, s in nd.items():
        su = urlsplit(str(s.get("url") or ""))
        title = str(s.get("title") or "").strip().lower()
        by_url = any(urlsplit(u).hostname == su.hostname and urlsplit(u).path.rstrip("/") == su.path.rstrip("/")
                     for u in urls if su.hostname)
        if by_url or (len(title) >= 8 and title in low):
            hit.append(n)
    if hit:
        return sorted(hit)
    if urls or _DEICTIC.search(q):
        return sorted(nd)  # ambiguous single-source request: fail closed
    return []


def g9_quote_and_link(question: str, sources, cfg: dict | None = None) -> dict:
    """{"forced": [n..], "items": [{"n","quote","url","title"}]}: for each forced source a short verbatim quote (the first
    sentence of its snippet, <= quote_max_chars, budget quote_total_chars across all) and the link. No paraphrase."""
    c = cfg_of(cfg)
    forced = g9_sources(question, sources, cfg)
    nd = nd_sources(sources)
    items, used = [], 0
    for n in forced:
        s = nd[n]
        first = (_SENT.split(str(s.get("snippet") or "").strip(), maxsplit=1) or [""])[0]
        q = first[: int(c["quote_max_chars"])].rstrip()
        if q and used + len(q) <= int(c["quote_total_chars"]) * max(1, len(forced)):
            used += len(q)
            items.append({"n": n, "quote": q, "url": s.get("url"), "title": s.get("title")})
        else:
            items.append({"n": n, "quote": None, "url": s.get("url"), "title": s.get("title")})
    return {"forced": forced, "items": items}


def apply(records, sources, cfg: dict | None = None, question: str = "") -> tuple[list, dict]:
    """Returns (kept records, report). The report counts drops per check. No ND source = (records, {}) unchanged.
    Fail closed: guard disabled, or an ND source without canonical_text, drops its citing sentences."""
    nd = nd_sources(sources)
    c = cfg_of(cfg)
    if not nd:
        return records, {}
    by_n = {s["n"]: s for s in sources}
    kept = list(records)
    rep = {"g1": 0, "g2g3": 0, "g4": 0, "g5": 0, "g9": 0, "g10": 0, "closed": 0}
    closed = {n for n, s in nd.items() if c.get("enabled") is False or not str(s.get("canonical_text") or "").strip()}
    if closed:
        n0 = len(kept)
        kept = [r for r in kept if not (closed & set(r.get("citations") or []))]
        rep["closed"] = n0 - len(kept)
        if c.get("enabled") is False:
            return kept, rep
    nd = {n: s for n, s in nd.items() if n not in closed}
    forced = set(g9_sources(question, sources, cfg))
    if forced:  # G9
        n0 = len(kept)
        kept = [r for r in kept if not (forced & set(r.get("citations") or []))]
        rep["g9"] = n0 - len(kept)
        rep["g9_forced"] = sorted(forced)
    if not nd:
        return kept, rep
    k = int(c["min_verbatim_tokens"])
    grams = set().union(*(_ngrams(_toks(_text(s)), k) for s in nd.values()))
    n0 = len(kept)  # G1
    kept = [r for r in kept if not (_ngrams(_toks(r["sentence"]), k) & grams)]
    rep["g1"] = n0 - len(kept)
    for n, s in nd.items():  # G2 and G3, per ND source
        while True:
            citing = [r for r in kept if n in (r.get("citations") or [])]
            if not citing or (len(citing) <= int(c["max_sentences"]) and len(citing) / len(kept) <= float(c["max_share"])):
                break
            worst = max(citing, key=lambda r: _rank_worst(r, by_n, kept.index(r)))
            kept.remove(worst)
            rep["g2g3"] += 1
    for n, s in nd.items():  # G10 word budget on paraphrased (non-quoted) text from one source
        while True:
            citing = [r for r in kept if n in (r.get("citations") or [])]
            if not citing or sum(len(_stems(r["sentence"])) for r in citing) <= int(c["max_words_per_source"]):
                break
            kept.remove(max(citing, key=lambda r: _rank_worst(r, by_n, kept.index(r))))
            rep["g10"] += 1
    for n, s in nd.items():  # G4: near-copy, multi-sentence cover, total coverage
        art = _article_sents(s)
        if not art:
            continue
        keep2 = []
        for r in kept:
            cov, _b, near = _cover(r["sentence"], art, c)
            if near or len(cov) >= int(c["max_cover_per_sentence"]) or _union_near(r["sentence"], art, c):
                rep["g4"] += 1
            else:
                keep2.append(r)
        kept = keep2
        while kept:
            ms = [_cover(r["sentence"], art, c)[0] for r in kept]
            if len(set().union(*ms)) / len(art) <= float(c["max_coverage"]):
                break
            kept.pop(max(range(len(kept)), key=lambda i: len(ms[i])))
            rep["g4"] += 1
    for n, s in nd.items():  # G5 sequence
        art = _article_sents(s)
        if not art:
            continue
        while True:
            idx = [_cover(r["sentence"], art, c)[1] for r in kept]
            run, best, best_end = 0, 0, -1
            for i, x in enumerate(idx):
                run = (run + 1 if run and x is not None and idx[i - 1] is not None and x > idx[i - 1]
                       else (1 if x is not None else 0))
                if run > best:
                    best, best_end = run, i
            if best <= int(c["max_sequence"]):
                break
            kept.pop(best_end)
            rep["g5"] += 1
    return kept, rep


def limit_quotes(citations, sources, cfg: dict | None = None) -> int:
    """G6: for citations of ND sources, drop the quote (keep the span reference) when it exceeds quote_max_chars or the
    per-source total quote_total_chars. Mutates the citation dicts; returns the number of quotes dropped."""
    c = cfg_of(cfg)
    nd = nd_sources(sources)
    used: dict = {}
    dropped = 0
    for ct in citations or []:
        n = ct.get("n")
        if n not in nd or not ct.get("quote"):
            continue
        q = ct["quote"]
        if len(q) > int(c["quote_max_chars"]) or used.get(n, 0) + len(q) > int(c["quote_total_chars"]):
            ct["quote"], ct["quote_span"] = None, None
            dropped += 1
        else:
            used[n] = used.get(n, 0) + len(q)
    return dropped
