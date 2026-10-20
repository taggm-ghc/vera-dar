"""Provider-neutral query translation (p3m3 item #71, gap G1).

`build_query_terms(question, cfg)` turns a reader-facing question into search terms: stopwords and meta-words
are dropped, domain phrases from config are kept whole as the CORE concept, and the remaining content words
are the EXTRA terms. Every word list comes from config `search.query_terms` (validated by
vera.eval_config); nothing here encodes a topic.

Fallback is never silent and never empty: when no core phrase and no extra term survives, `mode` is
"fallback_raw" and the caller uses its old behaviour (recorded in the provider's `last_query`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlencode

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")


@dataclass(frozen=True)
class QueryTerms:
    core: tuple[str, ...]    # whole domain phrases found in the question (AND-ed)
    extra: tuple[str, ...]   # other content words (OR-ed)
    mode: str                # "core_and_extra" | "core_only" | "extra_only" | "fallback_raw"

    @property
    def empty(self) -> bool:
        return not self.core and not self.extra


def _phrase_regex(phrase: str) -> re.Pattern:
    words = [re.escape(w) for w in phrase.lower().split()]
    return re.compile(r"\b" + r"\s+".join(words) + r"s?\b")


def build_query_terms(question: str, cfg: Mapping) -> QueryTerms:
    text = " ".join(question.lower().split())
    drop = set(cfg["stopwords"]) | set(cfg["meta_words"])
    core: list[str] = []
    # longest phrases first so "ai coding assistant" wins over "coding assistant"
    for phrase in sorted(cfg["core_phrases"], key=lambda p: (-len(p.split()), p)):
        m = _phrase_regex(phrase).search(text)
        if m:
            core.append(m.group(0))
            text = text[:m.start()] + " " + text[m.end():]  # consumed: not repeated among the extra terms
    # keep a core phrase once even if a shorter phrase matched inside another (dedupe by containment)
    core = [c for i, c in enumerate(core) if not any(c != d and c in d for d in core)]
    extra, seen = [], set()
    for tok in _TOKEN.findall(text):
        if len(tok) < cfg["min_term_chars"] or tok in drop or tok in seen:
            continue
        seen.add(tok)
        extra.append(tok)
        if len(extra) >= cfg["max_extra_terms"]:
            break
    if core and extra:
        mode = "core_and_extra"
    elif core:
        mode = "core_only"
    elif extra:
        mode = "extra_only"
    else:
        mode = "fallback_raw"
    return QueryTerms(tuple(core), tuple(extra), mode)


def scholar_search_url(question: str, terms_cfg: Mapping) -> str:
    """Human link-out to Google Scholar (item #71 R10-b). A PURE STRING BUILDER: VERA-DAR never requests, fetches or
    parses this URL (Scholar's robots.txt disallows /scholar and Google's terms forbid automated access against
    robots.txt). A person clicks it in their own browser. This is the only place in vera/ that names the host."""
    qt = build_query_terms(question, terms_cfg)
    words = list(qt.core) + list(qt.extra)
    q = " ".join(words) if words else " ".join(question.split())
    return "https://scholar.google.com/scholar?" + urlencode({"q": q})
