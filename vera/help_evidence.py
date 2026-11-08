"""SELFQ-RELEVANCE-1108 (owner 2026-11-08: "we shouldn't have to hard-code response processing for self-referencing
questions"): VERA-DAR's published help is ordinary evidence.

Passages = the paragraphs the About, How to use, Terms and Corpus Description pages render (config/ui_help_pages.json,
placeholders filled as the pages fill them) plus the About disclosure sections (vera.disclosure). Each passage is
scored by vera.fulldoc_admission's relevance code with the shipped config/fulldoc_answer.json admission values: the same
tokenizer (casefold; hyphen, underscore and space all split words), the same min_relevance and the same (adaptive)
matched-stem floor that admit stored-document spans. There is no name list, pattern list or threshold of its own.

Help is the best evidence when (a) a help passage clears those gates, (b1) at least one matched word is specific to
the help (absent from the stored-document vocabulary: the question is about VERA-DAR, not a topic that merely shares
words with the help), and (b2) when the help leaves a question word uncovered, the stored collection's best admitted
passage (the same admission code, same question) scores lower than the best help passage. Before the vocabulary is
built, only a fully covered question can use help, and only when the collection scores lower (b2). Otherwise None is
returned and the normal path runs. The reply quotes the best passages verbatim (no LLM,
search, storage or memory) and names any question word the help has no information on. Every public function never
raises.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger("vera")
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "help_evidence.json"
_MD = re.compile(r"\*\*|__|`")


def load_config(path: Path = CONFIG_PATH) -> dict:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _admission_cfg() -> dict:
    """The full-document config (admission block = the shared relevance settings); {} when unreadable (fail closed)."""
    try:
        from vera import fulldoc_corpus
        return fulldoc_corpus.load_config() or {}
    except Exception:  # noqa: BLE001
        return {}


def passages(help_cfg: dict | None = None) -> list[dict]:
    """[{page, heading, text}] in page order: what the help pages publish. [] on any problem."""
    try:
        from vera import disclosure, help_pages
        cfg = help_pages.load() if help_cfg is None else help_cfg
        cfg = cfg if isinstance(cfg, dict) else {}
        out: list[dict] = []

        def add(page, heading, text):
            t = _MD.sub("", str(text or "")).strip()
            if t and "[[" not in t and "{" not in t:  # an unfilled placeholder is never published, so never quoted
                out.append({"page": str(page), "heading": str(heading or ""), "text": t})

        def fill(p):
            p = str(p)
            if "{wake_minutes}" in p and help_pages.wake_minutes() is None:
                return ""
            return help_pages._fill(p).replace("{repo_issues}", "the project's public GitHub repository")

        about = cfg.get("about") if isinstance(cfg.get("about"), dict) else {}
        at = about.get("title") or "About VERA-DAR"
        for k in ("opening",):
            add(at, "", about.get(k))
        for line in about.get("opening_expansions") or []:
            add(at, "", line)
        add(at, "", about.get("intro"))
        for heading, text in disclosure.disclosure_sections():
            add(at, heading, text)
        for sec in about.get("after_disclosure") or []:
            for p in (sec or {}).get("paragraphs") or []:
                add(at, sec.get("heading"), fill(p))
        for key, default in (("how_to_use", "How to use VERA-DAR"), ("terms", "Terms")):
            page = cfg.get(key) if isinstance(cfg.get(key), dict) else {}
            title = page.get("title") or default
            add(title, "", page.get("intro"))
            for sec in page.get("sections") or []:
                for p in (sec or {}).get("paragraphs") or []:
                    add(title, sec.get("heading"), fill(p))
        cp = cfg.get("corpus_page") if isinstance(cfg.get("corpus_page"), dict) else {}
        for p in cp.get("paragraphs") or []:
            add(cp.get("title") or "Corpus Description", "", p)
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("help passages unavailable (%s)", type(exc).__name__)
        return []


def _scoring(fcfg: dict):
    """(icfg, stem_n, min_rel, min_hits, adaptive) from the admission block, mirroring fulldoc_admission.admit."""
    from vera import fulldoc_admission as fa
    adm = fcfg.get("admission") or {}
    routing = fa._routing(fcfg)
    icfg = routing.get("identification") or {}
    stem_n = int((routing.get("corpus") or {}).get("relevance_stem_len") or 0)
    gv2 = adm.get("generic_terms_v2")
    if isinstance(gv2, dict) and gv2.get("enabled") is True and icfg:
        icfg = dict(icfg, question_generic_terms=list(icfg.get("question_generic_terms") or [])
                    + [str(x) for x in (gv2.get("add") or []) if isinstance(x, str)])
    return (icfg, stem_n, float(adm.get("min_relevance", 0.4)), int(adm.get("min_matched_stems", 2)),
            adm.get("adaptive_min_matched_stems") is True)


def resolve_addressee(question: str, cfg: dict, icfg: dict, stem_n: int) -> str:
    """The person addressed ('Who are you?', 'What is your name?') is the subject: its name joins the question. Other
    content words still need the help to be the best evidence ('Can you find papers on X' scores X against the
    collection, which then wins: b2)."""
    from vera import fulldoc_admission as fa
    ad = cfg.get("addressee") if isinstance(cfg.get("addressee"), dict) else {}
    terms = {str(t).casefold() for t in ad.get("terms") or [] if isinstance(t, str)}
    subject = ad.get("subject")
    if not terms or not isinstance(subject, str) or not subject.strip():
        return question
    words = re.findall(icfg.get("token_pattern") or r"[^\W\d_]+", question.casefold())
    if not (terms & set(words)):
        return question
    return f"{question} {subject.strip()}"


def content_words(question: str, icfg: dict, stem_n: int) -> dict:
    """Question content stem -> the first question word that produced it (for the 'no information on' line)."""
    from vera import fulldoc_admission as fa
    q = fa._q_stems(question, icfg, stem_n)
    out: dict = {}
    for w in fa.tokenize(question, icfg):
        s = w[:stem_n]
        if s in q and s not in out:
            out[s] = w
    return out


def evaluate(question: str, corpus_best=None, cfg: dict | None = None, fcfg: dict | None = None,
             help_passages: list | None = None, known_to_corpus=None, covered_only: bool = False) -> dict | None:
    """{"passages": [...], "uncovered": [words], "relevance": r} when help is the best evidence, else None.
    corpus_best(question) -> float | None: the relevance of the stored collection's best admitted passage for the
    question (0.0 = none admitted; None = the collection is not available, then help decides alone). Asked only when
    the help leaves a question word uncovered; a collection score >= the help score keeps the normal path.
    known_to_corpus(word) -> True | False | None (None = vocabulary not built): with uncovered words, at least one
    matched word must be False (help-specific); otherwise, or when unknown, the normal path runs (fail closed).
    covered_only: None unless the help covers every content word (the pre-routing check: a question with words the
    help does not cover is judged by the scope classifier first)."""
    try:
        cfg = load_config() if cfg is None else cfg
        if cfg.get("enabled") is not True:
            return None
        fcfg = _admission_cfg() if fcfg is None else fcfg
        from vera import fulldoc_admission as fa
        icfg, stem_n, min_rel, min_hits, adaptive = _scoring(fcfg)
        if not icfg or stem_n <= 0:
            return None
        q = resolve_addressee(str(question or ""), cfg, icfg, stem_n)
        tg = fa._term_ctx(q, fcfg, icfg, stem_n)
        scored = []
        for i, p in enumerate(passages() if help_passages is None else help_passages):
            rel = fa._relevance(q, p["text"], icfg, stem_n, tg)
            hits, q_n = fa._matched(q, p["text"], icfg, stem_n, tg)
            need = max(1, min(min_hits, q_n)) if adaptive else min_hits
            if q_n and rel >= min_rel and hits >= need:
                scored.append((-rel, -hits, i, p, rel))
        if not scored:
            return None
        scored.sort(key=lambda x: x[:3])
        words = content_words(q, icfg, stem_n)
        covered: set = set()
        for *_, p, _r in scored:
            covered |= set(words) & fa._stems(p["text"], icfg, stem_n)
        uncovered = [w for s, w in words.items() if s not in covered]
        if uncovered and covered_only:
            return None
        matched = [w for st, w in words.items() if st in covered]
        known = [known_to_corpus(w) for w in matched] if callable(known_to_corpus) else [None]
        vocab_ok = bool(known) and all(k is not None for k in known)
        if vocab_ok and not any(k is False for k in known):
            return None  # every matched word is a collection word: the question only shares words with the help
        if uncovered and not vocab_ok:
            return None  # cannot tell whether it is about VERA-DAR: fail closed to the normal path
        if uncovered or not vocab_ok:
            cb = corpus_best(q) if callable(corpus_best) else None
            if cb is None and not vocab_ok:
                return None  # neither the vocabulary nor the collection is available: normal path
            if cb is not None and float(cb) >= scored[0][4]:  # the collection is at least as good: normal path
                return None
        n = cfg.get("max_passages")
        n = n if isinstance(n, int) and not isinstance(n, bool) and n >= 1 else 3
        return {"passages": [x[3] for x in scored[:n]], "uncovered": uncovered, "relevance": scored[0][4]}
    except Exception as exc:  # noqa: BLE001
        log.warning("help evidence unavailable (%s)", type(exc).__name__)
        return None


def reply_text(ev: dict, cfg: dict | None = None) -> str:
    cfg = load_config() if cfg is None else cfg
    parts = [str(cfg.get("intro") or "").strip()]
    for p in ev.get("passages") or []:
        where = p["page"] + (f", {p['heading']}" if p.get("heading") else "")
        parts.append(f"> {p['text']}\n\n({where})")
    if ev.get("uncovered") and cfg.get("no_information"):
        terms = ", ".join(f"'{w}'" for w in ev["uncovered"])
        parts.append(str(cfg["no_information"]).replace("{terms}", terms))
    if cfg.get("pages_pointer"):
        parts.append(str(cfg["pages_pointer"]).strip())
    return "\n\n".join(x for x in parts if x)
