"""NONASK-1103 S7: extract outbound search terms from a sub-query (plan 6.4, F9).

Portals (GDELT, HN Algolia) get ONLY these terms, never the user's sentence. The PII filter drops e-mails, phone
numbers, ID-like tokens and personal names that are not in the entity registry. Pure and offline; the only I/O is a
one-time read of config/query_terms.json (cached).

MIDTERMS-WP2-1106 (RC3, privacy review work/WP2-PRIVACY-REVIEW-1106/review.md): Deny-All allowlists.
- A capitalised word is kept only when it is a registry entity or a `common_capitalised` lexicon common noun. Every
  other capitalised word is dropped as a possible personal name (first word: when followed by another capital).
- An all-caps token in mixed-case text is kept only when it is in `known_acronyms` (US, UK, EU, ...), a registry entity
  or a lexicon word; otherwise it is dropped wherever it stands (JOHN, MARIA, JR, AOC). Dotted initials (J.R) are
  dropped unless their undotted form is an allowlisted acronym (U.S -> US) or a registry entity.
- The stop list applies to everything except allowlisted acronyms, and the GDELT operators OR, AND and NOT never pass.
The lexicon is read only from config/query_terms.json, so an edit to another lane's word lists cannot widen this filter.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
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
_OPERATORS = frozenset({"or", "and", "not"})  # GDELT boolean operators: never a term, whatever the allowlist says

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "query_terms.json"


def _words(node, *, upper: bool) -> frozenset:
    out = set()
    for w in node if isinstance(node, list) else ():
        if isinstance(w, str) and w.strip() and " " not in w.strip():
            w = w.strip()
            out.add(w.upper() if upper else w.lower())
    return frozenset(out)


@lru_cache(maxsize=1)
def _config_lists() -> tuple[frozenset, frozenset]:
    """(lexicon lower-case, known acronyms upper-case) from config/query_terms.json, no DB overlay.
    A missing or broken file gives empty lists, which keeps the stricter drop (fails closed)."""
    try:
        cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    acr = frozenset(a for a in _words(cfg.get("known_acronyms"), upper=True) if a.lower() not in _OPERATORS)
    return _words(cfg.get("common_capitalised"), upper=False), acr


def _default_lexicon() -> frozenset:
    return _config_lists()[0]


def _known_acronyms() -> frozenset:
    return _config_lists()[1]


def _is_id_like(tok: str) -> bool:
    digits = sum(c.isdigit() for c in tok)
    if digits >= 5:  # long numbers, ids, account numbers
        return True
    if digits and any(c.isalpha() for c in tok) and len(tok) >= 6:  # mixed alnum code, e.g. AB123456
        return True
    return False


def extract_terms(sub_query: str, known_entities: Iterable[str] = (), *, max_terms: int = MAX_TERMS,
                  lexicon: Iterable[str] | None = None, acronyms: Iterable[str] | None = None) -> list[str]:
    """Return a short list of safe search terms (possibly empty). Never the full sentence.

    known_entities: registry entity names. lexicon: lower-case common nouns allowed when capitalised; acronyms:
    allowlisted upper-case acronyms (defaults: config/query_terms.json). See the module docstring for the rules.
    """
    text = str(sub_query or "")
    try:  # R2: framing idioms (config/query_terms.json framing_phrases) are no terms ("de facto", "status quo")
        from vera.search_query import strip_framing
        text = strip_framing(text)
    except Exception:  # noqa: BLE001
        pass
    text = _EMAIL.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _PHONE.sub(" ", text)
    known = set()
    for e in known_entities or ():
        known.update(w.lower() for w in _TOKEN.findall(str(e)))
    lex = _default_lexicon() if lexicon is None else frozenset(str(w).lower() for w in lexicon)
    acr = _known_acronyms() if acronyms is None else frozenset(str(w).upper() for w in acronyms)
    shouted = any(c.isalpha() for c in text) and not any(c.islower() for c in text)
    toks = _TOKEN.findall(text)
    out: list[str] = []
    seen = set()
    for i, raw in enumerate(toks):
        tok = raw.strip(".'-_")
        low = tok.lower()
        if not tok or low in seen or low in _OPERATORS:
            continue
        has_alpha = any(c.isalpha() for c in tok)
        caps = has_alpha and tok.upper() == tok  # all letters upper-case (digits and dots allowed)
        undotted = tok.replace(".", "")
        acronym = not shouted and caps and undotted.isalpha() and undotted in acr
        if acronym:
            tok, low = undotted, undotted.lower()  # U.S -> US
            if low in seen:
                continue
        elif len(tok) < MIN_LEN or low in _STOP:
            continue
        if _is_id_like(tok):
            continue
        allowed = acronym or low in known or low in lex
        if not allowed:
            if "." in tok and has_alpha and tok[0].isupper() and undotted.lower() not in known:
                continue  # dotted initials (J.R): possible personal name
            if caps and not shouted:
                continue  # all-caps in mixed-case text, not allowlisted: JOHN, MARIA, JR, AOC
            if tok[0].isupper() and (i > 0 or shouted or _looks_like_name_pair(toks)):
                continue  # capitalised: possible personal name
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
