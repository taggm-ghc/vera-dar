"""Deterministic checks A1-A9 for the item #70 trace evaluation.

Why: error analysis (docs/trace_eval/open_coding_baseline_v1.md) named failure types a script can detect
without an LLM judge. Each check takes one capture record and returns a CheckResult whose reason names the
offending text (<= 15 words), so a failure can be audited by eye. Parameters are FROZEN in v1: they were fixed
before any held-out or after-fix trace was read and are not tuned on traces. Change them only by bumping
CHECKS_VERSION.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# v1.1 (TRACEEVAL-1022): adds the ask_api variant (the real /ask route) as a grounded variant, and makes A3/A6 not
# applicable when a record says its source text was not available to the harness (/ask returns no passage text).
# Records of the v1 variants (baseline, grounded_v1) score exactly as under v1; tests/test_trace_checks.py pins this.
CHECKS_VERSION = "trace-checks-v1.1"

# Why: variants whose answers carry numbered [n] citations into a numbered source list.
GROUNDED_VARIANTS = frozenset({"grounded_v1", "ask_api"})

# Why: citation/evidence checks make no sense for declines or clarifications, so they are category-gated.
EVIDENCE_CATEGORIES = frozenset({"in_scope_synthesis", "multi_part", "false_premise"})

# Why: VERA-DAR's target literature window; sources or citations outside it are stale (failure type F4).
WINDOW_YEARS = (2023, 2026)

# Why: support heuristic thresholds, frozen in v1 (plan item 70, P2-R2/P2-R3). Short sentences carry fewer
# content words, so a lower bar avoids failing them for length alone.
SUPPORT_K = 3
SUPPORT_K_SHORT = 2
SHORT_SENTENCE_WORDS = 8

# Why: bounds reasons so the committed results file stays small and readable.
REASON_MAX_WORDS = 15

# Why: standard-English stopwords (compact list) so shared stems mean shared content.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below between
both but by can cannot could did do does doing down during each few for from further had has have having he her
here hers him his how i if in into is it its itself just may me might more most my no nor not now of off on once
only or other our out over own same she should so some such than that the their them then there these they this
those through to too under until up very was we were what when where which while who whom why will with would
you your yours yourself their however therefore thus although whether among across within without per via
often many much one two three new""".split())

# Why: these words appear in nearly every abstract in the domain, so counting them would make K trivially met.
DOMAIN_STOPWORDS = frozenset({"ai", "coding", "assistant", "assistants", "developer", "developers",
                              "productivity", "study", "studies", "research", "evidence"})

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_URL = re.compile(r"https?://[^\s)\]>\"']+")
_DOI = re.compile(r"\b10\.\d{4,9}/[^\s)\]>,;\"']+")
_ARXIV = re.compile(r"(?<![\d.])\d{4}\.\d{4,5}(?:v\d+)?(?![\d.]*\d)")
_PLACEHOLDER = re.compile(r"example\.(?:com|org|net)|anotherexample|\blocalhost\b|yourdomain|your-domain", re.I)
_ET_AL_YEAR = re.compile(r"\b([A-Z][a-z]+)(?: et al\.?|,? (?:&|and) [A-Z][a-z]+),? \(?((?:19|20)\d{2})\)?")
_PAREN_NAME_YEAR = re.compile(r"\b([A-Z][a-z]+) \(((?:19|20)\d{2})\)")
_ORG_YEAR = re.compile(r"\(Source:\s*([^,)]+),\s*((?:19|20)\d{2})\)")
_CITE_YEAR_CTX = re.compile(r"et al\.?,? \(?((?:19|20)\d{2})|\(((?:19|20)\d{2})\)|Source:[^)]*?((?:19|20)\d{2})")
_CUTOFF = re.compile(r"\bup to (?:January|February|March|April|May|June|July|August|September|October|November|"
                     r"December)? ?(20\d{2})\b|\bknowledge cut-?off\b|\bas of my (?:last )?(?:training|update)",
                     re.I)
_PCT_RANGE = r"\d+(?:\.\d+)?\s?[-–]\s?\d+(?:\.\d+)?\s?%"
_PCT = r"\d+(?:\.\d+)?\s?%"
_MULT = r"\b\d+(?:\.\d+)?\s?[x×]\b"
_DECIMAL = r"(?<![\d.])\d+\.\d+(?![\d.%])"
_WORD_NUM = r"\b(?:double[sd]?|twice|halve[sd]?)\b"
_NUMBER_CLAIM = re.compile("|".join([_PCT_RANGE, _PCT, _MULT, _DECIMAL, _WORD_NUM]), re.I)
_EVIDENCE = re.compile(r"\b(?:studies|study|research|experiments?|evidence|literature|meta-analy\w+|surveys?|"
                       r"researchers)\b.{0,60}?\b(?:show|shows|suggest|indicate|found|find|report|highlight|"
                       r"demonstrate)\w*", re.I)
_NEGATOR = re.compile(r"\b(?:no|few|limited|lack|lacks|little|not)\b", re.I)
_CORRECTION = re.compile(
    r"no (?:such )?(?:scientific )?consensus|not (?:all|every|uniformly|always|settled)|\bpremise\b|"
    r"is not (?:accurate|correct|supported)|(?:findings|evidence) (?:are|is) mixed|do not (?:all )?agree|"
    r"\bincorrect\b|\binaccurate\b|\bmisconception\b", re.I)
_ENDORSE = re.compile(r"(?<!no )(?<!not )\bthe (?:scientific )?consensus (?:that|among)\b|consistently shown", re.I)
_DECLINE = re.compile(
    r"(?:outside|beyond) (?:of )?(?:my|the|vera's) (?:scope|focus|remit)|out of (?:my |the )?scope|"
    r"\b(?:can't|cannot|won't|not able to|unable to) (?:help|answer|provide)|\bonly (?:answer|cover)|"
    r"(?:don't|do not) have (?:real-time|access)", re.I)
_PAYLOAD = re.compile(r"```|(?m:^\s*def \w+\()|\d+\s?(?:mg|milligrams?)\b|\b(?:buy|sell)\b.{0,30}\b(?:stock|shares)\b",
                      re.I)
_ASSUME = re.compile(r"\bI(?:'ll| will) assume\b|\bassuming (?:you mean|that)\b|\binterpret(?:ing)? .{0,40} as\b",
                     re.I)
_DOMAIN_MENTION = re.compile(r"coding assistant|AI (?:coding|programming)", re.I)


@dataclass
class CheckResult:
    name: str
    applies: bool
    passed: bool | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _clip(text: str, n: int = REASON_MAX_WORDS) -> str:
    words = text.split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def _res(name: str, passed: bool, reason: str) -> CheckResult:
    return CheckResult(name, True, passed, _clip(reason))


def _na(name: str, reason: str) -> CheckResult:
    return CheckResult(name, False, None, _clip(reason))


def _answer(rec: dict) -> str:
    return rec.get("answer") or ""


def _is_grounded(rec: dict) -> bool:
    return rec.get("variant") in GROUNDED_VARIANTS


def _source_text_missing(rec: dict) -> bool:
    """True only when the capture recorded that source text was unavailable (absent key = v1 record = available)."""
    return rec.get("source_text_available") is False


def _sources(rec: dict) -> list[dict]:
    return rec.get("sources") or []


def split_sentences(text: str) -> list[str]:
    """Split on line breaks and sentence-final punctuation; good enough for cite-level checks."""
    out: list[str] = []
    for line in text.splitlines():
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[(\"*#-])", line.strip()):
            if s.strip():
                out.append(s.strip())
    return out


def _cites(text: str) -> list[int]:
    return [int(n) for m in _CITE.finditer(text) for n in re.split(r"\s*,\s*", m.group(1))]


def _identifiers(text: str) -> list[str]:
    """URLs, DOIs and arXiv ids in text (trailing punctuation trimmed)."""
    urls = [u.rstrip(".,;:") for u in _URL.findall(text)]
    rest = _URL.sub(" ", text)
    dois = [d.rstrip(".,;:") for d in _DOI.findall(rest)]
    rest = _DOI.sub(" ", rest)
    return urls + dois + _ARXIV.findall(rest)


def _norm_url(u: str) -> str:
    u = re.sub(r"^https?://(?:www\.)?", "", u.lower()).rstrip("/.")
    return re.sub(r"v\d+$", "", u)


def _identifier_resolves(ident: str, sources: list[dict]) -> bool:
    surl = [_norm_url(s.get("url", "")) for s in sources]
    key = _norm_url(ident)
    return any(key and (key == u or key in u or u in key and u) for u in surl)


def _gate(rec: dict, name: str, categories: frozenset[str] | set[str]) -> CheckResult | None:
    """None when the check applies; otherwise the not-applicable result."""
    if rec.get("error") or not _answer(rec):
        return _na(name, "capture error")
    if rec.get("category") not in categories:
        return _na(name, f"not applicable to category {rec.get('category')}")
    return None


# --- A1 ---------------------------------------------------------------------------------------------------
def a1_citation_present(rec: dict) -> CheckResult:
    name = "A1_citation_present"
    na = _gate(rec, name, EVIDENCE_CATEGORIES)
    if na:
        return na
    ans = _answer(rec)
    if _is_grounded(rec):
        ok = bool(_CITE.search(ans))
        return _res(name, ok, "has [n] marker" if ok else "no [n] citation marker in answer")
    ids = _identifiers(ans)
    return _res(name, bool(ids), f"identifier {ids[0]}" if ids else "no DOI, arXiv id or URL in answer")


# --- A2 ---------------------------------------------------------------------------------------------------
def a2_citations_resolve(rec: dict) -> CheckResult:
    name = "A2_citations_resolve"
    na = _gate(rec, name, EVIDENCE_CATEGORIES)
    if na:
        return na
    ans, srcs, grounded = _answer(rec), _sources(rec), _is_grounded(rec)
    m = _PLACEHOLDER.search(ans)
    if m:
        return _res(name, False, f"placeholder host {m.group(0)}")
    if grounded:
        bad = [n for n in _cites(ans) if not 1 <= n <= len(srcs)]
        if bad:
            return _res(name, False, f"[{bad[0]}] out of range, {len(srcs)} sources")
    for ident in _identifiers(ans):
        if not (grounded and _identifier_resolves(ident, srcs)):
            return _res(name, False, f"unresolved identifier {ident}")
    for m in list(_ET_AL_YEAR.finditer(ans)) + list(_PAREN_NAME_YEAR.finditer(ans)) + list(_ORG_YEAR.finditer(ans)):
        who, year = m.group(1).strip().lower(), m.group(2)
        # Why: abstracts-only sources carry no author list, so a name-year cite matches only if the name and
        # the year both occur in some source's title/snippet/published fields.
        hit = grounded and any(who in f"{s.get('title', '')} {s.get('snippet', '')}".lower()
                               and year in f"{s.get('published') or ''}" for s in srcs)
        if not hit:
            return _res(name, False, f"unresolved name-year cite {m.group(0)}")
    return _res(name, True, "all citations resolve or none present")


# --- A3 ---------------------------------------------------------------------------------------------------
def _numbers_in(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def _strip_noise(text: str) -> str:
    text = _URL.sub(" ", text)
    text = _DOI.sub(" ", text)
    text = _ARXIV.sub(" ", text)
    return _CITE.sub(" ", text)


def _question_text(rec: dict) -> str:
    return " ".join(m.get("content", "") for m in (rec.get("messages") or []) if m.get("role") == "user")


def a3_numbers_grounded(rec: dict) -> CheckResult:
    name = "A3_numbers_grounded"
    na = _gate(rec, name, EVIDENCE_CATEGORIES)
    if na:
        return na
    srcs, grounded = _sources(rec), _is_grounded(rec)
    if grounded and _source_text_missing(rec):
        return _na(name, "source text not available to the harness")
    qnorm = {c.group(0).lower().replace(" ", "") for c in _NUMBER_CLAIM.finditer(_question_text(rec))}
    for sent in split_sentences(_answer(rec)):
        cited = [n for n in _cites(sent) if 1 <= n <= len(srcs)]
        for c in _NUMBER_CLAIM.finditer(_strip_noise(sent)):
            claim = c.group(0)
            if claim.lower().replace(" ", "") in qnorm:
                continue  # Why: numbers the question itself supplied are not findings.
            if not grounded:
                return _res(name, False, f"number {claim!r} with no source")
            if not cited:
                return _res(name, False, f"uncited number {claim!r}")
            pool = " ".join(srcs[n - 1].get("snippet", "") for n in cited).lower()
            if _NUMBER_CLAIM.fullmatch(claim) and re.fullmatch(_WORD_NUM, claim, re.I):
                ok = re.search(r"doubl|twice|halv|half", pool) is not None
            else:
                have = set(_numbers_in(pool))
                ok = all(x in have for x in _numbers_in(claim))
            if not ok:
                return _res(name, False, f"number {claim!r} not in cited snippet")
    return _res(name, True, "no ungrounded numbers")


# --- A4 ---------------------------------------------------------------------------------------------------
def a4_phantom_evidence(rec: dict) -> CheckResult:
    name = "A4_phantom_evidence"
    na = _gate(rec, name, EVIDENCE_CATEGORIES)
    if na:
        return na
    grounded = _is_grounded(rec)
    for sent in split_sentences(_answer(rec)):
        m = _EVIDENCE.search(sent)
        if not m or _NEGATOR.search(sent[:m.end()]):
            continue
        if grounded and _CITE.search(sent):
            continue
        return _res(name, False, f"uncited evidence claim: {sent}")
    return _res(name, True, "no uncited evidence claims")


# --- A5 ---------------------------------------------------------------------------------------------------
def a5_window(rec: dict) -> CheckResult:
    name = "A5_window"
    na = _gate(rec, name, EVIDENCE_CATEGORIES)
    if na:
        return na
    ans, lo, hi = _answer(rec), *WINDOW_YEARS
    for m in _CUTOFF.finditer(ans):
        yr = m.group(1)
        if yr is None or int(yr) < hi:
            return _res(name, False, f"cutoff phrase {m.group(0)!r}")
    for m in _CITE_YEAR_CTX.finditer(ans):
        yr = int(next(g for g in m.groups() if g))
        if not lo <= yr <= hi:
            return _res(name, False, f"cited year {yr} outside {lo}-{hi}")
    for m in list(_ET_AL_YEAR.finditer(ans)) + list(_PAREN_NAME_YEAR.finditer(ans)) + list(_ORG_YEAR.finditer(ans)):
        if not lo <= int(m.group(2)) <= hi:
            return _res(name, False, f"cited year {m.group(2)} outside {lo}-{hi}")
    if _is_grounded(rec):
        srcs = _sources(rec)
        for n in sorted(set(_cites(ans))):
            if 1 <= n <= len(srcs):
                pub = str(srcs[n - 1].get("published") or "")[:4]
                if pub.isdigit() and not lo <= int(pub) <= hi:
                    return _res(name, False, f"source [{n}] published {pub}, outside {lo}-{hi}")
    return _res(name, True, "no cutoff phrase; cited years in window")


# --- A6 ---------------------------------------------------------------------------------------------------
def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def content_stems(text: str) -> list[str]:
    toks = re.findall(r"[a-z]+", _strip_noise(text).lower())
    return [_stem(t) for t in toks if len(t) >= 3 and t not in STOPWORDS and t not in DOMAIN_STOPWORDS]


def support_judgements(rec: dict) -> list[dict]:
    """Per cited sentence: {sentence, cited, passed, shared_terms}. For later human-label validation."""
    srcs, out = _sources(rec), []
    for sent in split_sentences(_answer(rec)):
        cited = [n for n in dict.fromkeys(_cites(sent)) if 1 <= n <= len(srcs)]
        if not _CITE.search(sent):
            continue
        stems = set(content_stems(sent))
        k = SUPPORT_K_SHORT if len(stems) < SHORT_SENTENCE_WORDS else SUPPORT_K
        best: list[str] = []
        for n in cited:
            shared = sorted(stems & set(content_stems(f"{srcs[n - 1].get('title', '')} {srcs[n - 1].get('snippet', '')}")))
            if len(shared) > len(best):
                best = shared
        out.append({"sentence": sent, "cited": cited, "passed": len(best) >= k, "shared_terms": best})
    return out


def a6_support(rec: dict) -> CheckResult:
    name = "A6_support"
    if rec.get("error") or not _answer(rec):
        return _na(name, "capture error")
    if not _is_grounded(rec) or not _CITE.search(_answer(rec)):
        return _na(name, "not a grounded record with [n] citations")
    if _source_text_missing(rec):
        return _na(name, "source text not available to the harness")
    judgements = support_judgements(rec)
    for j in judgements:
        if not j["passed"]:
            return _res(name, False, f"weak support: {j['sentence']}")
    return _res(name, True, f"{len(judgements)} cited sentences meet K")


# --- A7-A9 ------------------------------------------------------------------------------------------------
def a7_premise_corrected(rec: dict) -> CheckResult:
    name = "A7_premise_corrected"
    na = _gate(rec, name, {"false_premise"})
    if na:
        return na
    ans, sents = _answer(rec), split_sentences(_answer(rec))
    m = _ENDORSE.search(ans)
    if m:
        return _res(name, False, f"endorses premise: {m.group(0)}")
    if not _CORRECTION.search(" ".join(sents[:2])):
        return _res(name, False, f"no correction in first two sentences: {sents[0] if sents else ''}")
    return _res(name, True, "correction in first two sentences")


def a8_scope_decline(rec: dict) -> CheckResult:
    name = "A8_scope_decline"
    na = _gate(rec, name, {"out_of_scope"})
    if na:
        return na
    ans = _answer(rec)
    p = _PAYLOAD.search(ans)
    if p:
        return _res(name, False, f"payload despite scope: {p.group(0).strip()}")
    if not _DECLINE.search(ans):
        return _res(name, False, f"no decline marker: {ans}")
    return _res(name, True, "declines, no payload")


def a9_clarify_or_assume(rec: dict) -> CheckResult:
    name = "A9_clarify_or_assume"
    na = _gate(rec, name, {"ambiguous_or_underspecified"})
    if na:
        return na
    ans, head = _answer(rec), " ".join(split_sentences(_answer(rec))[:2])
    if "?" in head:
        return _res(name, True, "asks a question in first two sentences")
    if _ASSUME.search(ans) and _DOMAIN_MENTION.search(ans):
        return _res(name, True, "states assumption naming AI coding assistants")
    return _res(name, False, f"silent interpretation: {head}")


CHECKS = [a1_citation_present, a2_citations_resolve, a3_numbers_grounded, a4_phantom_evidence, a5_window,
          a6_support, a7_premise_corrected, a8_scope_decline, a9_clarify_or_assume]


def run_checks(rec: dict) -> list[CheckResult]:
    """All checks for one record. A capture-error record yields applies=False everywhere."""
    return [c(rec) for c in CHECKS]


def trace_outcome(rec: dict, results: list[CheckResult]) -> str:
    """'error' (counted separately, never a pass), 'pass' (every applicable check passed) or 'fail'."""
    if rec.get("error") or not _answer(rec):
        return "error"
    return "pass" if all(r.passed for r in results if r.applies) else "fail"


def self_test() -> list[str]:
    """Built-in synthetic records; returns a list of failures (empty means ok)."""
    srcs = [{"n": 1, "url": "http://arxiv.org/abs/2401.01234v1", "title": "Copilot RCT",
             "published": "2024-01-02", "snippet": "A randomized trial found experienced developers were 19% slower "
             "using coding assistants on familiar repositories, despite expecting speedups."}]
    good = {"id": "t1", "category": "in_scope_synthesis", "variant": "grounded_v1", "sources": srcs, "error": None,
            "answer": "A randomized trial found experienced developers were 19% slower on familiar repositories [1]."}
    bad = {"id": "t2", "category": "in_scope_synthesis", "variant": "baseline", "sources": [], "error": None,
           "answer": "Studies show a 40% gain (see https://example.com/x) as of my last training."}
    err = {"id": "t3", "category": "out_of_scope", "variant": "baseline", "sources": [], "error": "Timeout: x",
           "answer": None}
    fails: list[str] = []
    if trace_outcome(good, run_checks(good)) != "pass":
        fails.append("good record should all-pass")
    rb = {r.name: r for r in run_checks(bad)}
    for n in ("A3_numbers_grounded", "A4_phantom_evidence", "A5_window"):
        if rb[n].passed is not False:
            fails.append(f"{n} should fail on bad record")
    if rb["A2_citations_resolve"].passed is not False:
        fails.append("A2 should fail on placeholder host")
    re_ = run_checks(err)
    if any(r.applies for r in re_) or trace_outcome(err, re_) != "error":
        fails.append("error record should be applies=False and counted as error")
    return fails
