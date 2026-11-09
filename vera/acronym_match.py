"""ANSWER-SCOPE-1109 follow-up (root cause): question acronyms match their spelled-out forms, and back, mechanically.

Live case (731863f): "What are components of RAG?" admitted only the one paper that literally wrote "RAG"; general
documents that spell out "retrieval-augmented generation" failed the matched-stem floor. No acronym or topic list
lives here; everything is derived from the question's shape and the corpus text:

* question_acronyms(): tokens of 2 to 6 upper-case letters written as such (optional plural "s"); none when the whole
  question is upper case (shouting is not an acronym signal).
* initial_runs(): contiguous content-word runs in a text whose initials spell an acronym. Contiguous = only spaces
  or one hyphen between the words (no punctuation, no skipped word); every word is a content word (not a tokenizer
  stopword and at least min_word_len letters), so stopword initials never spell anything. The last word may be plural.
* definitions(): expansions LEARNED from in-corpus definitions "long form (ACR)" whose initials spell ACR.
* groups(): term_expansion.Group objects for vera.fulldoc_admission (the existing E1 term-group machinery):
  - direction 1, question has "RAG": the upper-case acronym (case-sensitive) and the expansions the corpus itself
    defines match. Bare initial runs with no corpus definition match only with bare_initials true (default false:
    "MMA" = methylmalonic acid or mixed martial arts; "AI" = average income);
  - direction 2, question spells "retrieval augmented generation": the upper-case acronym matches only when the corpus
    defines that phrase as that acronym, so an unrelated "RAG" or "LMS" never matches;
  - ambiguity guard: an acronym the corpus also defines with a long form that does not spell it ("Methylmalonic acid
    (MMA)"), or with more than max_senses different long forms, is not expanded at all.
  The group token weighs as the question's OWN form (an acronym counts one stem, as today; a spelled phrase its
  content stems, as E1), so a question's arithmetic is unchanged and an equivalent form gains the same weight.

Config: config/fulldoc_answer.json admission.acronym_match (enabled exactly true; false restores the old behaviour).
"""
from __future__ import annotations

import re

_ACR = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,6})(s?)(?![A-Za-z0-9])")
_WORD = re.compile(r"[A-Za-z]+")
_GAP_OK = re.compile(r"\s+|\s*-\s*")  # what may separate two words of a run: spaces, or one hyphen
ID_PREFIX = "acr"


def cfg_of(cfg) -> dict | None:
    a = ((cfg or {}).get("admission") or {}).get("acronym_match") if isinstance(cfg, dict) else None
    return a if isinstance(a, dict) and a.get("enabled") is True else None


def question_acronyms(question: str) -> list[str]:
    q = str(question or "")
    letters = [c for c in q if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return []
    out: list[str] = []
    for m in _ACR.finditer(q):
        a = m.group(1).lower()
        if a not in out:
            out.append(a)
    return out


def _words(text: str) -> list:
    return [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(text)]


def _content(w: str, stop: frozenset, min_len: int) -> bool:
    lw = w.lower()
    return len(lw) >= min_len and lw not in stop and lw.rstrip("s") not in stop


def initial_runs(text: str, acr: str, stop: frozenset, min_len: int = 3) -> list[tuple[int, int]]:
    """(start, end) of every contiguous content-word run whose initials spell `acr` (case-insensitive)."""
    text = str(text or "")
    a = acr.lower()
    n = len(a)
    ws = _words(text)
    out = []
    i = 0
    while i + n <= len(ws):
        run = ws[i:i + n]
        ok = all(w[0][0].lower() == a[j] and _content(w[0], stop, min_len) for j, w in enumerate(run))
        if ok:
            ok = all(_GAP_OK.fullmatch(text[run[j][2]:run[j + 1][1]]) for j in range(n - 1))
        if ok:
            # the run must not be the tail of a longer run that is itself an expansion of something else: keep it simple
            out.append((run[0][1], run[-1][2]))
            i += n
            continue
        i += 1
    return out


def _norm(phrase: str) -> str:
    return " ".join(w.lower() for w in _WORD.findall(phrase))


def definitions(texts, acr: str, stop: frozenset, min_len: int = 3, max_defs: int = 8, other: list | None = None
                ) -> list[str]:
    """Expansions defined in `texts` as "long form (ACR)" / "(ACRs)" whose initials spell ACR (normalised, unique).
    `other` (a list) receives one entry per "(ACR)" whose preceding words do NOT spell it ("Methylmalonic acid
    (MMA)"): a second sense, so the caller treats the acronym as ambiguous."""
    up = acr.upper()
    pat = re.compile(r"\(\s*" + up + r"s?\s*\)")
    out: list[str] = []
    for t in texts or ():
        t = str(t or "")
        if up not in t:
            continue
        for m in pat.finditer(t):
            head = t[max(0, m.start() - 200):m.start()].rstrip()
            runs = initial_runs(head, acr, stop, min_len)
            if runs and runs[-1][1] == len(head):
                p = _norm(head[runs[-1][0]:runs[-1][1]])
                if p and p not in out:
                    out.append(p)
                    if len(out) >= max_defs:
                        return out
            elif other is not None:
                other.append(m.start())
    return out


def _question_runs(question: str, stop: frozenset, min_len: int) -> list[tuple[str, str]]:
    """(initials, normalised phrase) for every contiguous content-word run of 2..6 words in the question."""
    q = str(question or "")
    ws = _words(q)
    out = []
    for i in range(len(ws)):
        for n in range(2, 7):
            run = ws[i:i + n]
            if len(run) < n:
                break
            if not all(_content(w[0], stop, min_len) for w in run):
                break
            if not all(_GAP_OK.fullmatch(q[run[j][2]:run[j + 1][1]]) for j in range(n - 1)):
                break
            out.append(("".join(w[0][0].lower() for w in run), _norm(q[run[0][1]:run[-1][2]])))
    return out


def corpus_texts(corpus, acquired) -> list:
    """Full texts of the acquired documents (definitions may sit anywhere, e.g. an abstract), cached on the corpus."""
    cache = getattr(corpus, "__dict__", None)
    key = ("_acr_texts", len(acquired or ()))
    if cache is not None and key in cache.get("_acr_cache", {}):
        return cache["_acr_cache"][key]
    out = []
    for d in acquired or ():
        try:
            out.append(str(corpus.text(d["document_id"]) or ""))
        except Exception:  # noqa: BLE001
            continue
    if cache is not None:
        cache.setdefault("_acr_cache", {})[key] = out
    return out


def groups(question: str, texts, acfg: dict, stop, seed_groups=()) -> tuple:
    """Dynamic term groups for this question (see the module doc). Never raises (() on any fault)."""
    try:
        from vera.term_expansion import Group
        stop = frozenset(str(s).lower() for s in (stop or ()))
        min_len = int(acfg.get("min_word_len", 3))
        min_scan = int(acfg.get("min_scan_len", 3))
        seeded = {t for g in seed_groups or () for t, _a in g.phrases}
        max_senses = max(1, int(acfg.get("max_senses", 1)))
        out, ids = [], set()
        for a in question_acronyms(question):
            if a in seeded:
                continue  # an owner-seeded group already covers it
            other: list = []
            learned = tuple(definitions(texts, a, stop, min_len, other=other))
            if other or len(learned) > max_senses:
                continue  # the corpus uses this acronym for another (or several) long forms: ambiguous, no expansion
            scan = a if (acfg.get("bare_initials") is True and len(a) >= min_scan) else ""
            if not learned and not scan:
                continue
            gid = ID_PREFIX + a
            ids.add(gid)
            out.append(Group(gid, tuple((p, False) for p in learned), (), initials=scan, upper=a.upper(), own=a,
                             stop=stop, min_len=min_len))
        for ini, phrase in _question_runs(question, stop, min_len):
            gid = ID_PREFIX + ini
            if gid in ids or not (2 <= len(ini) <= 6) or phrase in seeded:
                continue
            other2: list = []
            learned = definitions(texts, ini, stop, min_len, other=other2)
            if phrase not in learned:
                continue  # direction 2 only on the corpus's own definition of this phrase
            if other2 or len(learned) > max_senses:
                continue  # the same ambiguity guard as direction 1: two senses are never treated as equivalent
            ids.add(gid)
            out.append(Group(gid, ((phrase, False),), (),
                             initials="", upper=ini.upper(), own=phrase, stop=stop, min_len=min_len))
        return tuple(out)
    except Exception:  # noqa: BLE001  (odd input: the feature adds nothing)
        return ()


def retrieval_terms(gs) -> str:
    """Extra words for candidate retrieval (BM25 query side): every learned phrase and acronym of the groups."""
    words = []
    for g in gs or ():
        for t, _a in g.phrases:
            words.append(t)
        if getattr(g, "upper", ""):
            words.append(g.upper)
    return " ".join(words)
