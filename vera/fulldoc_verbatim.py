"""PLAN-FDA C2: substantially-verbatim accounting against the FULL text of every bundle document.

Pure stdlib, no I/O. Every threshold is read from a `verbatim` config dict with the defaults in DEFAULTS (all are
priors, not calibrated values).

Counted text of a draft sentence (after normalisation on both sides):
  (a) maximal shared token runs of >= min_verbatim_tokens (1-token gap tolerance) with any reference document;
  (b) any sentence whose character-level LCS ratio (LCS / sentence length) against its best source window is
      >= sentence_lcs_ratio counts WHOLE (defeats "edit every 5th word").
Characters are the union of intervals (a quote that is also inline is counted once). A normalised interval is mapped
back to the original text and the LARGER of the two lengths is counted. Exclusions (document metadata, pure
number/unit runs, a configurable technical-phrase list) subtract only the matching characters.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

DEFAULTS = {
    "min_verbatim_tokens": 8,        # prior (plan)
    "gap_tolerance_tokens": 1,       # prior (plan)
    "seed_tokens": 3,                # exact k-gram seed used to locate candidate runs / windows
    "max_index_hits": 64,            # k-grams more frequent than this in the reference are ignored as seeds
    "sentence_lcs_ratio": 0.8,       # prior (plan)
    "sentence_min_chars": 40,        # sentences shorter than this are not LCS-tested
    "sentence_min_votes": 2,         # seed hits agreeing on one reference position before a window is LCS-tested
    "window_slack_chars": 12,
    "window_scale": 1.1,
    "max_windows": 3,
    "technical_phrases": [],
    "unit_words": ["%", "percent", "fold", "mg", "kg", "g", "ml", "l", "mm", "cm", "km", "m", "s", "ms", "min", "h",
                   "hz", "khz", "mhz", "ghz", "kb", "mb", "gb", "tb", "years", "year", "days", "day", "weeks", "week",
                   "months", "month", "hours", "hour", "minutes", "seconds"],
    "exclude_numbers": True,
}

_SUPER = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_DEL = re.compile(
    "­"                                                              # soft hyphen
    r"|(?<=[^\W\d_])[-‐‑]\s*\n\s*(?=[^\W\d_])"                  # line-break hyphenation
    r"|\[\s*\d+(?:\s*[,;–—-]\s*\d+)*\s*\]"                      # [n], [19,20], [3-5]
    r"|\(\s*S\d+\s+[A-Za-z]+\s*\)"                                        # (S1 Text)
    "|[" + _SUPER + "]+")                                                  # superscript digits
_TR = str.maketrans({**{c: "'" for c in "‘’‚‛′`´"},
                     **{c: '"' for c in "“”„‟″"},
                     **{c: "-" for c in "‐‑‒–—―−"}})
_WS = re.compile(r"\s+")
_TOK = re.compile(r"[^\W_]+")
_SENT_SEP = re.compile(r"(?<=[.!?])\s+")


def normalize(text: str) -> str:
    """Fast whole-string normalisation (used for the reference); identical in result to normalize_mapped()."""
    t = _DEL.sub("", text or "")
    t = unicodedata.normalize("NFKC", t).casefold().translate(_TR)
    return _WS.sub(" ", t).strip()


def normalize_mapped(text: str):
    """Normalise and keep, per normalised char, the [start, end) of the original cluster it came from."""
    text = text or ""
    n = len(text)
    dele = bytearray(n)
    for m in _DEL.finditer(text):
        for k in range(m.start(), m.end()):
            dele[k] = 1
    out, st, en = [], [], []
    prev_space = True
    i = 0
    while i < n:
        if dele[i]:
            i += 1
            continue
        j = i + 1
        while j < n and not dele[j] and unicodedata.category(text[j]).startswith("M"):
            j += 1
        for ch in unicodedata.normalize("NFKC", text[i:j]).casefold().translate(_TR):
            if ch.isspace():
                if prev_space:
                    continue
                ch, prev_space = " ", True
            else:
                prev_space = False
            out.append(ch)
            st.append(i)
            en.append(j)
        i = j
    while out and out[-1] == " ":
        out.pop(), st.pop(), en.pop()
    return "".join(out), st, en


def union(intervals) -> list:
    out: list = []
    for a, b in sorted((a, b) for a, b in intervals if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def subtract(intervals, cuts) -> list:
    out = []
    cuts = union(cuts)
    for a, b in intervals:
        cur = a
        for c, d in cuts:
            if d <= cur or c >= b:
                continue
            if c > cur:
                out.append((cur, c))
            cur = max(cur, d)
            if cur >= b:
                break
        if cur < b:
            out.append((cur, b))
    return out


def lcs_len(a: str, b: str) -> int:
    """Length of the longest common subsequence (bit-parallel, Hyyro 2004)."""
    if not a or not b:
        return 0
    masks: dict = {}
    for i, ch in enumerate(a):
        masks[ch] = masks.get(ch, 0) | (1 << i)
    full = (1 << len(a)) - 1
    v = full
    for ch in b:
        m = masks.get(ch)
        if m is None:
            continue
        v = ((v + (v & m)) | (v & ~m)) & full
    return len(a) - bin(v).count("1")


def _tokens(norm: str):
    spans = [(m.start(), m.end()) for m in _TOK.finditer(norm)]
    return [norm[a:b] for a, b in spans], spans


def _cfg(cfg) -> dict:
    return {**DEFAULTS, **{k: v for k, v in (cfg or {}).items() if k in DEFAULTS and v is not None}}


class Reference:
    """Normalised full texts of every bundle document plus a seed-gram index."""

    def __init__(self, texts, cfg=None):
        c = _cfg(cfg)
        self.k = max(2, int(c["seed_tokens"]))
        self.cap = int(c["max_index_hits"])
        self.norm, self.words, self.spans = [], [], []
        self.index: dict = {}
        for d, t in enumerate(texts):
            norm = normalize(t)
            words, spans = _tokens(norm)
            self.norm.append(norm)
            self.words.append(words)
            self.spans.append(spans)
            for j in range(len(words) - self.k + 1):
                key = hash(tuple(words[j:j + self.k]))
                lst = self.index.get(key)
                if lst is None:
                    self.index[key] = [(d, j)]
                elif lst is not False:
                    if len(lst) >= self.cap:
                        self.index[key] = False  # too common to be a useful seed
                    else:
                        lst.append((d, j))


_REF_CACHE: dict = {}


def reference_for(texts, cfg=None) -> Reference:
    c = _cfg(cfg)
    texts = [t for t in texts if isinstance(t, str) and t]
    key = (int(c["seed_tokens"]), int(c["max_index_hits"]),
           tuple(hashlib.sha1(t.encode("utf-8", "replace"), usedforsecurity=False).hexdigest()  # cache key only
                 for t in texts))
    ref = _REF_CACHE.get(key)
    if ref is None:
        if len(_REF_CACHE) >= 4:
            _REF_CACHE.pop(next(iter(_REF_CACHE)))
        ref = _REF_CACHE[key] = Reference(texts, cfg)
    return ref


@dataclass
class Profile:
    norm: str
    starts: list
    ends: list
    intervals: list = field(default_factory=list)  # normalised-coordinate intervals counted as verbatim (post-exclusion)
    raw_intervals: list = field(default_factory=list)  # the same BEFORE exclusions
    run_intervals: list = field(default_factory=list)  # raw intervals joined across excluded strings sitting between them

    def span_len(self, a: int, b: int) -> int:
        """Counted length of normalised [a, b): the larger of the normalised and the original lengths."""
        if b <= a:
            return 0
        return max(b - a, self.ends[b - 1] - self.starts[a])

    @property
    def counted(self) -> int:
        return sum(self.span_len(a, b) for a, b in self.intervals)

    @property
    def max_run(self) -> int:
        """Longest contiguous verbatim run, measured BEFORE exclusions (a number or metadata string inside a copied
        passage must not split it); intervals separated by at most one char are one run. Exclusions only reduce
        the counted total."""
        best, run = 0, None
        for a, b in (self.run_intervals or self.raw_intervals or self.intervals):
            if run is not None and a - run[1] <= 1:
                run = (run[0], b)
            else:
                run = (a, b)
            best = max(best, self.span_len(*run))
        return best


class VerbatimChecker:
    def __init__(self, reference_texts, metadata_strings=(), cfg=None):
        self.cfg = _cfg(cfg)
        self.ref = reference_for(list(reference_texts), self.cfg)
        self._excl = [normalize(s) for s in metadata_strings if isinstance(s, str) and len(s.strip()) >= 3]
        self._excl += [normalize(p) for p in (self.cfg.get("technical_phrases") or []) if isinstance(p, str) and p.strip()]
        self._excl = sorted({s for s in self._excl if len(s) >= 3}, key=len, reverse=True)
        units = sorted({re.escape(u) for u in (self.cfg.get("unit_words") or []) if u}, key=len, reverse=True)
        unit = r"(?:\s*(?:" + "|".join(units) + r")(?![^\W\d_]))?" if units else ""
        num = r"\d[\d.,]*" + unit
        self._num = re.compile(r"(?<![^\W_])" + num + r"(?:\s*[-/x×]\s*" + num + r")*") \
            if self.cfg.get("exclude_numbers") else None

    # -- detection -------------------------------------------------------------------------------------------
    def _hits(self, words):
        k, idx, hits = self.ref.k, self.ref.index, {}
        for i in range(len(words) - k + 1):
            lst = idx.get(hash(tuple(words[i:i + k])))
            if lst:
                hits[i] = lst
        return hits

    def _runs(self, words, hits):
        """Chains of exact seed blocks (verified token-for-token) with <= gap_tolerance tokens between them."""
        k, gap, ref = self.ref.k, int(self.cfg["gap_tolerance_tokens"]), self.ref
        diag: dict = {}
        for i, lst in hits.items():
            for d, j in lst:
                diag.setdefault((d, j - i), []).append(i)
        blocks = []
        for (d, off), idxs in diag.items():
            idxs.sort()
            s = p = idxs[0]
            for i in idxs[1:] + [None]:
                if i is not None and i <= p + 1:
                    p = i
                    continue
                ds, de = s, p + k
                if words[ds:de] == ref.words[d][ds + off:de + off]:
                    blocks.append((ds, de, d, ds + off, de + off))
                if i is not None:
                    s = p = i
        blocks.sort(key=lambda b: (b[0], -b[1]))
        chains: list = []
        for ds, de, d, rs, re_ in blocks:
            for c in chains:
                if c[2] == d and ds <= c[1] + gap and rs <= c[4] + gap and rs >= c[3] and ds >= c[0]:
                    if de > c[1] and re_ > c[4]:
                        c[1], c[4] = de, re_
                    break
            else:
                chains.append([ds, de, d, rs, re_])
        return [(c[0], c[1]) for c in chains if c[1] - c[0] >= int(self.cfg["min_verbatim_tokens"])]

    def _sentence_hit(self, norm, a, b, words_lo, words_hi, hits, spans) -> bool:
        """LCS test of the sentence norm[a:b] (tokens words_lo..words_hi) against the best candidate windows."""
        c = self.cfg
        sent = norm[a:b]
        if len(sent) < int(c["sentence_min_chars"]):
            return False
        votes: dict = {}
        for i in range(words_lo, words_hi):
            for d, j in hits.get(i, ()):
                e = j - (i - words_lo)
                v = votes.setdefault((d, e // 6), [0, 0])
                v[0] += 1
                v[1] += e
        ranked = sorted(votes.items(), key=lambda kv: -kv[1][0])[:int(c["max_windows"])]
        slack = int(c["window_slack_chars"])
        for (d, _), (n, tot) in ranked:
            if n < int(c["sentence_min_votes"]):
                break
            e = max(0, min(len(self.ref.spans[d]) - 1, round(tot / n))) if self.ref.spans[d] else 0
            cs = self.ref.spans[d][e][0] if self.ref.spans[d] else 0
            win = self.ref.norm[d][max(0, cs - slack): cs + int(len(sent) * float(c["window_scale"])) + slack]
            if lcs_len(sent, win) / len(sent) >= float(c["sentence_lcs_ratio"]):
                return True
        return False

    def _exclusions(self, norm: str) -> list:
        cuts = []
        for s in self._excl:
            pos = norm.find(s)
            while pos >= 0:
                end = pos + len(s)
                # whole-token matches only: "and" must not cut "standard"
                if (pos == 0 or not norm[pos - 1].isalnum()) and (end >= len(norm) or not norm[end].isalnum()):
                    cuts.append((pos, end))
                pos = norm.find(s, pos + 1)
        if self._num is not None:
            cuts += [(m.start(), m.end()) for m in self._num.finditer(norm)]
        return cuts

    def profile(self, text: str) -> Profile:
        norm, st, en = normalize_mapped(text)
        p = Profile(norm, st, en)
        if not norm:
            return p
        words, spans = _tokens(norm)
        hits = self._hits(words)
        iv = [(spans[a][0], spans[b - 1][1]) for a, b in self._runs(words, hits)]
        # (b) sentence-level LCS
        pos, bounds = 0, []
        for m in _SENT_SEP.finditer(norm):
            bounds.append((pos, m.start()))
            pos = m.end()
        bounds.append((pos, len(norm)))
        starts = [s for s, _ in spans]
        covered = union(iv)
        for a, b in bounds:
            if b <= a or any(c <= a and b <= d for c, d in covered):
                continue
            lo = next((i for i, s in enumerate(starts) if s >= a), len(starts))
            hi = next((i for i, s in enumerate(starts) if s >= b), len(starts))
            if hi > lo and self._sentence_hit(norm, a, b, lo, hi, hits, spans):
                iv.append((a, b))
        p.raw_intervals = union(iv)
        cuts = self._exclusions(norm)
        merged: list = []
        for c, d in union(cuts):  # adjacent excluded strings (separated only by non-alphanumerics) act as one cut
            if merged and not any(ch.isalnum() for ch in norm[merged[-1][1]:c]):
                merged[-1] = (merged[-1][0], d)
            else:
                merged.append((c, d))
        bridges = [(c, d) for c, d in merged
                   if any(c - 1 <= b <= d for a, b in p.raw_intervals)  # a run ending at/inside the cut
                   and any(c <= a <= d + 1 for a, b in p.raw_intervals)]  # a run starting inside/at the cut end
        p.run_intervals = union(p.raw_intervals + bridges)  # a title/DOI/number inserted mid-run does not split it
        p.intervals = [(a, b) for a, b in subtract(p.raw_intervals, cuts)
                       if any(ch.isalnum() for ch in norm[a:b])]  # drop leftover whitespace/punctuation slivers
        return p


def combined_total(profiles, quotes=(), checker=None):
    """Counted chars of `profiles` (generated text) plus `quotes` (strings), as a union: a quote found inside a
    profile's text is merged into that profile's intervals (counted once); other quotes add their own length.
    Returns (total, generated_verbatim, quoted)."""
    gen = sum(p.counted for p in profiles)
    extra = [[] for _ in profiles]
    outside = 0
    quoted = 0
    for q in quotes:
        nq = normalize(q)
        ql = max(len(q), len(nq))
        quoted += ql
        for k, p in enumerate(profiles):
            pos = p.norm.find(nq) if nq else -1
            if pos >= 0:
                extra[k].append((pos, pos + len(nq)))
                break
        else:
            outside += ql
    total = 0
    for p, ex in zip(profiles, extra):
        total += sum(p.span_len(a, b) for a, b in union(list(p.intervals) + ex))
    return total + outside, gen, quoted
