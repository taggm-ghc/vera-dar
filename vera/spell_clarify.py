"""SELFQ-RELEVANCE-1108: a clarification round for unknown question words (owner 2026-11-08).

check(question) -> None (no pause) or {"word", "suggestions": [{"word", "question"}], "as_typed": question, "text"}.
A content word (the shared admission tokenizer: casefold, hyphen/underscore/space split, stopwords and generic question
words removed) is KNOWN when it, or its relevance stem, is in the stored-document vocabulary, the help vocabulary or the
A-P1 lay-term vocabulary (vera.query_norm). The first unknown word with at least one known word within the edit bound
pauses the request; suggestions are ranked by edit distance, then corpus frequency, then alphabet. Never a silent
correction: the caller re-asks with the chosen question, or with spell_check off ("search as typed"), a per-request
parameter that is never stored.

The stored-document vocabulary is built from the corpora vera.fulldoc_corpus already loaded from the database, in a
background thread, at most every refresh_s. Until it exists (cold start, warming, corpus off) no pause is made: fail
open to the normal path. In-process cache: recorded divergence from "dynamic memory in the DB" (the corpus it is derived
from is itself an in-process cache of DB rows). Every public function never raises.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import Counter
from contextvars import ContextVar
from pathlib import Path

log = logging.getLogger("vera")
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "spell_clarify.json"
# Per-request only (AskRequest.spell_check); default on. Never stored, never tied to a caller.
SPELL_CHECK: ContextVar[bool] = ContextVar("vera_spell_check", default=True)

_VOCAB: dict = {"words": None, "stems": None, "corpus_stems": None, "key": None, "at": None, "building": False,
                "failed_at": None}
_LOCK = threading.Lock()


def load_config(path: Path = CONFIG_PATH) -> dict:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def osa_distance(a: str, b: str, bound: int | None = None) -> int:
    """Damerau-Levenshtein, optimal string alignment (adjacent transposition = 1). With `bound`, returns bound + 1 as
    soon as every cell of a row exceeds it."""
    la, lb = len(a), len(b)
    if bound is not None and abs(la - lb) > bound:
        return bound + 1
    prev2: list | None = None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + c)
            if prev2 is not None and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                v = min(v, prev2[j - 2] + 1)
            cur[j] = v
        if bound is not None and min(cur) > bound:
            return bound + 1
        prev2, prev = prev, cur
    return prev[lb]


def max_edits(word: str, cfg: dict) -> int:
    for b in cfg.get("edit_bounds") or []:
        if not isinstance(b, dict):
            continue
        ml = b.get("max_len")
        if ml is None or len(word) <= int(ml):
            return int(b.get("max_edits", 0))
    return 0


def _icfg():
    from vera import help_evidence
    icfg, stem_n, *_ = help_evidence._scoring(help_evidence._admission_cfg())
    return icfg, stem_n


def _tokens(text: str, icfg: dict) -> list:
    """Every word (stopwords included: they are known words) by the shared token pattern, casefolded."""
    pat = icfg.get("token_pattern") or r"[^\W\d_]+"
    return re.findall(pat, str(text or "").casefold())


def help_words(icfg: dict) -> Counter:
    from vera import help_evidence
    c: Counter = Counter()
    for p in help_evidence.passages():
        c.update(_tokens(p["text"], icfg))
    return c


def lay_words() -> set:
    try:
        from vera import query_norm
        return {w for t in query_norm.vocabulary(query_norm.load_entries()) for w in query_norm.words(t)}
    except Exception:  # noqa: BLE001
        return set()


def _corpora():
    """The base and acquired corpora, only when already loaded and not warming (never triggers or waits on a load)."""
    try:
        from vera import ask_service, fulldoc_corpus
        cfg = fulldoc_corpus.load_config()
        if not fulldoc_corpus.flag_enabled(cfg) or ask_service._base_warming():
            return ()
        return tuple(c for c in (fulldoc_corpus.get_corpus(cfg), fulldoc_corpus.get_acquired_corpus(cfg))
                     if c is not None)
    except Exception:  # noqa: BLE001
        return ()


def build_vocabulary(corpora, icfg: dict, stem_n: int) -> tuple[Counter, set, dict]:
    """(all known words with counts, their relevance stems, stem shares {"corpus": Counter, "help": Counter} counted
    over the stored documents alone and over the help alone)."""
    c: Counter = Counter()
    for corpus in corpora:
        for text in (getattr(corpus, "_texts", None) or {}).values():  # the verified document texts it holds
            c.update(_tokens(text, icfg))
        for d in getattr(corpus, "acquired", None) or []:
            if isinstance(d, dict):
                c.update(_tokens(d.get("title"), icfg))
    corpus_stems: Counter = Counter()
    for w, n in c.items():
        corpus_stems[w[:stem_n]] += n
    help_stems: Counter = Counter()
    hw = help_words(icfg)
    for w, n in hw.items():
        help_stems[w[:stem_n]] += n
    c.update(hw)
    c.update(lay_words())
    return c, {w[:stem_n] for w in c}, {"corpus": corpus_stems, "help": help_stems}


def _rebuild(corpora, key, icfg, stem_n, now) -> None:
    try:
        words, stems, cstems = build_vocabulary(corpora, icfg, stem_n)
        with _LOCK:
            _VOCAB.update(words=words, stems=stems, corpus_stems=cstems, key=key, at=now)
        log.info("spell vocabulary built: %d words", len(words))
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            _VOCAB["failed_at"] = now  # back off for refresh_s: a persistent failure never rebuilds on every request
        log.warning("spell vocabulary build failed (%s)", type(exc).__name__)
    finally:
        with _LOCK:
            _VOCAB["building"] = False


def vocabulary(cfg: dict | None = None, corpora=None, sync: bool = False, now=time.monotonic):
    """(words Counter, stems set, stored-document stems set) or None while not built. Starts a background rebuild when missing, stale or the
    loaded corpora changed (one at a time)."""
    cfg = load_config() if cfg is None else cfg
    corpora = _corpora() if corpora is None else tuple(corpora)
    if not corpora:
        return None
    key = tuple(id(c) for c in corpora)
    t = now()
    with _LOCK:
        have = _VOCAB["words"] is not None
        stale = (not have or _VOCAB["key"] != key
                 or t - (_VOCAB["at"] or 0) >= float(cfg.get("refresh_s") or 600))
        backoff = _VOCAB["failed_at"] is not None and t - _VOCAB["failed_at"] < float(cfg.get("refresh_s") or 600)
        start = stale and not _VOCAB["building"] and not backoff
        if start:
            _VOCAB["building"] = True
    if start:
        icfg, stem_n = _icfg()
        if sync:
            _rebuild(corpora, key, icfg, stem_n, t)
        else:
            threading.Thread(target=_rebuild, args=(corpora, key, icfg, stem_n, t), daemon=True,
                             name="vera-spell-vocab").start()
    with _LOCK:
        return (_VOCAB["words"], _VOCAB["stems"], _VOCAB["corpus_stems"]) if _VOCAB["words"] is not None else None


def reset_for_tests() -> None:
    with _LOCK:
        _VOCAB.update(words=None, stems=None, corpus_stems=None, key=None, at=None, building=False, failed_at=None)


def known(word: str, vocab, stem_n: int) -> bool:
    words, stems = vocab[0], vocab[1]
    return word in words or word[:stem_n] in stems


def known_to_corpus(word: str, vocab=None) -> bool | None:
    """False when the word is specific to the help: its share of the help's words is more than
    config/help_evidence.json help_specific_ratio times its share of the stored documents' words (a name such as
    'vera' that a few stored papers also use stays help-specific); True otherwise; None while the vocabulary is not
    built or the ratio is not configured."""
    try:
        vocab = vocabulary() if vocab is None else vocab
        if vocab is None or len(vocab) < 3 or not isinstance(vocab[2], dict):
            return None
        from vera import help_evidence
        k = help_evidence.load_config().get("help_specific_ratio")
        if isinstance(k, bool) or not isinstance(k, (int, float)) or k <= 0:
            return None
        _, stem_n = _icfg()
        st = str(word).casefold()[:stem_n]
        cs, hs = vocab[2]["corpus"], vocab[2]["help"]
        c_tot, h_tot = sum(cs.values()) or 1, sum(hs.values()) or 1
        return not (hs.get(st, 0) / h_tot > k * cs.get(st, 0) / c_tot)
    except Exception:  # noqa: BLE001
        return None

def suggestions(word: str, words: Counter, cfg: dict) -> list:
    b = max_edits(word, cfg)
    if b <= 0:
        return []
    n = cfg.get("max_suggestions")
    n = n if isinstance(n, int) and not isinstance(n, bool) and n >= 1 else 3
    found = []
    for w, cnt in words.items():
        if w == word or abs(len(w) - len(word)) > b:
            continue
        d = osa_distance(word, w, b)
        if d <= b:
            found.append((d, -cnt, w))
    return [w for _, _, w in sorted(found)[:n]]


def replace_word(question: str, word: str, new: str) -> str:
    """The question with the first whole-word, case-insensitive occurrence of `word` replaced by `new`."""
    return re.sub(r"(?<![^\W\d_])" + re.escape(word) + r"(?![^\W\d_])", lambda m: new, question, count=1,
                  flags=re.IGNORECASE)


def check(question: str, cfg: dict | None = None, vocab=None) -> dict | None:
    try:
        cfg = load_config() if cfg is None else cfg
        if cfg.get("enabled") is not True or not SPELL_CHECK.get():
            return None
        icfg, stem_n = _icfg()
        if not icfg or stem_n <= 0:
            return None
        vocab = vocabulary(cfg) if vocab is None else vocab
        if vocab is None:
            return None
        from vera import fulldoc_admission as fa
        gen = {str(g).casefold() for g in icfg.get("question_generic_terms") or [] if isinstance(g, str)}
        seen = set()
        for w in fa.tokenize(question, icfg):
            if w in gen or w in seen:
                continue
            seen.add(w)
            if known(w, vocab, stem_n):
                continue
            sug = suggestions(w, vocab[0], cfg)
            if not sug:
                continue  # nothing to offer: no pause
            text = str(cfg.get("text") or "").replace("{word}", w)
            return {"word": w, "text": text, "as_typed": question,
                    "as_typed_label": str(cfg.get("search_as_typed") or "search as typed"),
                    "suggestions": [{"word": s, "question": replace_word(question, w, s)} for s in sug]}
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("spell check unavailable (%s)", type(exc).__name__)
        return None


def reply_text(c: dict) -> str:
    opts = " · ".join(f"[{s['word']}]" for s in c.get("suggestions") or [])
    return f"{c['text']} {opts} · {c.get('as_typed_label', 'search as typed')}"


def clarification_of(data) -> dict | None:
    """The clarification block of an /ask body (result_kind clarification, contract.clarification), else None."""
    if not isinstance(data, dict) or data.get("result_kind") != "clarification":
        return None
    c = (data.get("contract") or {}).get("clarification")
    return c if isinstance(c, dict) and isinstance(c.get("suggestions"), list) else None


def render_clarification(st, data, queue) -> bool:
    """Streamlit: the question, one button per suggestion and a "search as typed" button. A click queues the chosen
    question through `queue(payload)` (on_click, so it fires on the next rerun even though the result is not drawn
    again). True when drawn. Payloads carry spell_check only per request; nothing is stored."""
    c = clarification_of(data)
    if c is None:
        return False
    from vera.ui_safety import safe_markdown
    st.info(safe_markdown(str(c.get("text") or "")))
    opts = [(str(s.get("word")), {"question": str(s.get("question"))}) for s in c["suggestions"]
            if isinstance(s, dict) and s.get("question")]
    opts.append((str(c.get("as_typed_label") or "search as typed"),
                 {"question": str(c.get("as_typed") or ""), "spell_check": False}))
    cols = st.columns(len(opts))
    tag = abs(hash(str(c.get("as_typed")))) % 10**8
    for i, ((label, payload), col) in enumerate(zip(opts, cols)):
        col.button(label, key=f"vera_clarify_{tag}_{i}", on_click=queue, args=(payload,))
    return True
