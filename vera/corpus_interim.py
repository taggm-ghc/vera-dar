"""CORPUS-INTERIM-1108 (owner 2026-11-08): the Corpus Description page without the subject classifier.

While subject_areas.enabled is false (config/ask-provider-chain.json, corpus_description.subject_areas) the page shows no
per-subject counts, shares or assignments and no list of subject areas. It describes coverage with (a) the checked written
description when one exists, plus the document count and publication-year range, or (b) when none exists, a deterministic
line built from the stored titles: the most frequent multi-word subject phrases, with no number per phrase and no source.
Pure functions plus two read-only SELECTs; no model call. Phrase thresholds are PROVISIONAL (see the config note)."""
import json
import re
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "corpus_frequent_subjects.json"
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)  # alphabetic runs only; digits and punctuation split a phrase
_BOUNDS = {"max_phrases": (1, 30), "min_docs": (2, 1000), "min_words": (2, 6), "max_words": (2, 6), "title_limit": (1, 20000)}


def load_cfg(path=None, rules_path=None) -> dict:
    """Validated settings plus `generic` (frozenset) and `min_token_len`. Raises OSError/ValueError (fail closed: the
    caller then shows no phrases rather than guessing)."""
    from vera import chunk_topics
    raw = json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))
    out = {}
    for k, (lo, hi) in _BOUNDS.items():
        v = raw.get(k)
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            raise ValueError(f"corpus_frequent_subjects: bad {k}")
        out[k] = v
    if out["min_words"] > out["max_words"]:
        raise ValueError("corpus_frequent_subjects: min_words > max_words")
    extra = raw.get("extra_generic_terms")
    if not isinstance(extra, list) or not all(isinstance(w, str) for w in extra):
        raise ValueError("corpus_frequent_subjects: bad extra_generic_terms")
    rules = chunk_topics.load_rules(rules_path)
    out["generic"] = frozenset(rules["generic"]) | {w.strip().lower() for w in extra if w.strip()}
    out["min_token_len"] = rules["min_token_len"]
    return out


def _runs(title: str, cfg: dict) -> list[list[str]]:
    """Runs of consecutive content words of one title (stopwords and generic terms end a run)."""
    runs, cur = [], []
    for w in _WORD.findall(str(title).casefold()):
        if len(w) < cfg["min_token_len"] or w in cfg["generic"]:
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(w)
    if cur:
        runs.append(cur)
    return runs


def frequent_phrases(titles, cfg: dict) -> list[str]:
    """Pure. The most frequent multi-word phrases (document frequency over distinct titles), longest-first on ties, with
    phrases contained in an already chosen phrase dropped. No counts are returned."""
    seen, df, whole = set(), {}, set()
    for t in titles or []:
        key = " ".join(str(t).casefold().split())  # distinct titles only (digits kept, so "study 1" and "study 2" differ)
        if not key or key in seen:
            continue
        seen.add(key)
        runs = _runs(t, cfg)
        if len(runs) == 1 and cfg["min_words"] <= len(runs[0]) <= cfg["max_words"] and len(runs[0]) == len(_WORD.findall(key)) and not re.search(r"\d", key):
            whole.add(" ".join(runs[0]))  # the title is only this phrase: never show it
        found = set()
        for r in runs:
            for n in range(cfg["min_words"], cfg["max_words"] + 1):
                for i in range(len(r) - n + 1):
                    found.add(" ".join(r[i:i + n]))
        for p in found:
            df[p] = df.get(p, 0) + 1
    ranked = sorted((p for p, c in df.items() if c >= cfg["min_docs"] and p not in whole),
                    key=lambda p: (-df[p], -len(p.split()), p))
    chosen: list[str] = []
    for p in ranked:
        if len(chosen) >= cfg["max_phrases"]:
            break
        if not any(f" {p} " in f" {c} " for c in chosen):
            chosen.append(p)
    return chosen


def read_interim(engine, cfg: dict) -> dict:
    """{"published_years": [min, max] or None, "frequent_subjects": [phrases]} from vera_vjay.sources (read-only)."""
    from sqlalchemy import text
    with engine.connect() as c:
        y = c.execute(text("SELECT min(substring(provenance->>'published' from '^[0-9]{4}')), "
                           "max(substring(provenance->>'published' from '^[0-9]{4}')) FROM vera_vjay.sources")).one()
        rows = c.execute(text("SELECT coalesce(s.provenance->>'title', k.title) FROM vera_vjay.sources s "
                              "LEFT JOIN vera_vjay.candidates k ON k.candidate_id = s.candidate_id "
                              "WHERE coalesce(s.provenance->>'title', k.title) IS NOT NULL "
                              "ORDER BY md5(s.source_id::text) LIMIT :n"), {"n": cfg["title_limit"]}).all()
    years = [str(y[0]), str(y[1])] if y[0] and y[1] else None
    return {"published_years": years, "frequent_subjects": frequent_phrases([r[0] for r in rows], cfg)}
