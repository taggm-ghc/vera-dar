"""NONASK-1103 S7: extract outbound search terms from a sub-query (plan 6.4, F9).

Portals (GDELT, HN Algolia) get ONLY these terms, never the user's sentence. The PII filter drops e-mails, phone
numbers, ID-like tokens and personal names that are not in the entity registry. Pure, offline, no I/O.
"""
from __future__ import annotations

import re
from typing import Iterable

MAX_TERMS = 6
MIN_LEN = 3

_EMAIL = re.compile(r"\S+@\S+")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-_.]*")
_STOP = frozenset("""a an and are as at be been but by can could did do does for from had has have how i if in into is it its
me my of on or our should so than that the their them then there these they this to up us was we were what when where
which who whom whose why will with would you your about latest current currently now today recent recently news
tell give show explain please any some more most much many also just not no yes""".split())


def _is_id_like(tok: str) -> bool:
    digits = sum(c.isdigit() for c in tok)
    if digits >= 5:  # long numbers, ids, account numbers
        return True
    if digits and any(c.isalpha() for c in tok) and len(tok) >= 6:  # mixed alnum code, e.g. AB123456
        return True
    return False


def extract_terms(sub_query: str, known_entities: Iterable[str] = (), *, max_terms: int = MAX_TERMS) -> list[str]:
    """Return a short list of safe search terms (possibly empty). Never the full sentence.

    known_entities: registry entity names; a capitalised word that is not stop-word, not at sentence start-only
    heuristic, is treated as a possible personal name and dropped unless it occurs in a known entity name.
    """
    text = str(sub_query or "")
    text = _EMAIL.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _PHONE.sub(" ", text)
    known = set()
    for e in known_entities or ():
        known.update(w.lower() for w in _TOKEN.findall(str(e)))
    toks = _TOKEN.findall(text)
    out: list[str] = []
    seen = set()
    for i, raw in enumerate(toks):
        tok = raw.strip(".'-_")
        low = tok.lower()
        if len(tok) < MIN_LEN or low in _STOP or low in seen:
            continue
        if _is_id_like(tok):
            continue
        # capitalised, not the first word: possible personal name -> keep only if registry knows it
        if i > 0 and tok[0].isupper() and not tok.isupper() and low not in known:
            continue
        if i == 0 and tok[0].isupper() and not tok.isupper() and low not in known and _looks_like_name_pair(toks):
            continue
        seen.add(low)
        out.append(tok)
        if len(out) >= max_terms:
            break
    return out


def _looks_like_name_pair(toks: list[str]) -> bool:
    """First word capitalised and followed by another capitalised word: 'John Smith ...'."""
    return len(toks) > 1 and toks[1][:1].isupper() and not toks[1].isupper()


def build_query(terms: Iterable[str]) -> str:
    """Join terms with spaces (implicit AND). Quotes/operators are stripped from the terms."""
    clean = [re.sub(r"[^\w\-]", "", t) for t in terms]
    return " ".join(t for t in clean if t)
