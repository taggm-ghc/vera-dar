"""MULTIQ-1108: separate questions in one prompt (owner 2026-11-08: "Sometimes, people pose multi-question or
multi-part question prompts. VD needs to be able to handle both.").

Pure and deterministic: no database, network, model or storage. A prompt is split only from its grammatical structure:

1. sentence boundaries ('?', '!', '.', ';' followed by whitespace, or a newline). A boundary separates two questions
   when the next sentence is interrogative-led (it starts, after optional leading conjunctions, with an interrogative
   or an auxiliary) or when both sentences end in '?'. Anything else ('Dr. Smith', 'U.S. vaccine policy', 'Please be
   brief.') stays with the sentence before it, so abbreviations and decimals need no list.
2. conjoined questions inside one sentence (', and what are their side effects'): the clause after ',' / ';' /
   ' and ' counts only when it starts with an auxiliary, or with an interrogative followed by an auxiliary within
   `aux_window` words. Joined noun phrases or predicates ('pros and cons', 'safe and effective') never split.
3. a part must stand alone: one with no content word (admission tokenizer, stopwords and question-generic terms
   removed) and no pronoun ('And why?', 'How so?') is merged into its neighbour. A pronoun part ('Who can use it?',
   'what are their side effects') keeps the user's words for display and carries the previous part's content words
   for retrieval and answering only (every carried word is from the prompt itself).

The word sets in config/multi_question.json are closed grammatical classes, never topic, name or keyword lists
(feedback_no_hardcoded_selfq). A multi-PART question (one question, several sides or facets) is not split here;
`shape` only reports it, from vera.subquery.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "multi_question.json"
MODES = ("off", "shadow", "on")
_WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")
_SENT_BOUNDARY = re.compile(r"(?<=[?!.;])[ \t]+|[ \t]*(?:\r?\n)+[ \t]*")
_QUOTE = re.compile(r'"[^"]*"|“[^”]*”')
_IN_SENT =re.compile(r"\s*[,;]\s*|\s+(?=(?:and|also|plus|then)\s)", re.IGNORECASE)


@dataclass(frozen=True)
class Part:
    index: int  # 1-based position among the answered parts
    text: str  # the user's own words (displayed)
    query: str  # the text routed and answered: `text`, plus carried subject words for a pronoun part
    carry: tuple = ()


@dataclass(frozen=True)
class Split:
    parts: tuple  # Part, at most max_questions
    over_cap: tuple = ()  # the user's words of each question beyond the cap (listed to ask separately)
    pronoun_carry: int = 0
    trace: tuple = field(default_factory=tuple)

    @property
    def n_questions(self) -> int:
        return len(self.parts) + len(self.over_cap)


def load_config(path: Path = CONFIG_PATH) -> dict:
    try:
        d = json.loads(Path(path).read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def mode_of(cfg: dict | None) -> str:
    """off | shadow | on; anything unreadable or unknown is 'off' (fail closed: today's behaviour)."""
    m = (cfg or {}).get("mode")
    return m if m in MODES else "off"


def _set(cfg: dict, key: str) -> frozenset:
    return frozenset(str(w).casefold() for w in cfg.get(key) or () if isinstance(w, str))


def _words(text: str) -> list:
    return _WORD.findall(str(text or "").casefold())


def _lead(words: list, conj: frozenset) -> list:
    i = 0
    while i < len(words) and words[i] in conj:
        i += 1
    return words[i:]


def clause_is_question(text: str, cfg: dict, in_sentence: bool = False) -> bool:
    """True when the clause is interrogative-led. Inside a sentence the clause also needs its own auxiliary verb."""
    interrog, aux, conj = _set(cfg, "interrogatives"), _set(cfg, "auxiliaries"), _set(cfg, "leading_conjunctions")
    w = _lead(_words(text), conj)
    if not w:
        return False
    if w[0] in aux:
        return len(w) > 1
    if w[0] not in interrog:
        return False
    if not in_sentence:
        return True
    win = cfg.get("aux_window")
    win = int(win) if isinstance(win, int) and not isinstance(win, bool) and win > 0 else 3
    return any(x in aux for x in w[1:1 + win])


def _quoted(text: str) -> list:
    """Spans of balanced double-quoted text; punctuation inside never separates questions."""
    return [m.span() for m in _QUOTE.finditer(text)]


def _in_spans(spans: list, pos: int) -> bool:
    return any(a <= pos < b for a, b in spans)


def _sentences(text: str) -> list:
    spans, out, start = _quoted(text), [], 0
    for m in _SENT_BOUNDARY.finditer(text):
        if _in_spans(spans, m.start()):
            continue
        out.append(text[start:m.start()])
        start = m.end()
    out.append(text[start:])
    return [s for s in (p.strip() for p in out) if s]


def _is_question(s: str, cfg: dict) -> bool:
    return s.rstrip().endswith("?") or clause_is_question(s, cfg)


def _questions_of_sentences(text: str, cfg: dict) -> list:
    """A statement before a question ('I have a cough. Should I see a doctor?') is context, not a question: the
    question takes it in. Two sentences are two questions only when the earlier one is itself a question."""
    out: list = []
    for s in _sentences(text):
        if not out:
            out.append(s)
        elif not _is_question(out[-1], cfg):
            out[-1] = f"{out[-1]} {s}"
        elif clause_is_question(s, cfg) or (out[-1].endswith("?") and s.endswith("?")):
            out.append(s)
        else:
            out[-1] = f"{out[-1]} {s}"
    return out


def _left_can_split(left: str, cfg: dict) -> bool:
    """The clause before a candidate split must itself be a complete question: interrogative-led, and with no
    interrogative or subordinator after its first word (those make the right clause a coordinate inside it:
    'between what is legal and what is ethical', 'when it rains and is cold'). A leading context phrase ('For adults
    over sixty,', 'If it rains,', 'Tell me about X,') is not a question, so it never splits off."""
    tail = (_sentences(left) or [left])[-1]
    if not clause_is_question(tail, cfg):
        return False
    w = _lead(_words(tail), _set(cfg, "leading_conjunctions"))
    return not any(x in _set(cfg, "interrogatives") | _set(cfg, "subordinators") for x in w[1:])


def _conjoined(sentence: str, cfg: dict) -> list:
    pieces, start, spans = [], 0, _quoted(sentence)
    for m in _IN_SENT.finditer(sentence):
        if m.start() <= start or _in_spans(spans, m.start()):
            continue
        if _left_can_split(sentence[start:m.start()], cfg) and clause_is_question(sentence[m.end():], cfg, in_sentence=True):
            pieces.append(sentence[start:m.start()].strip())
            start = m.end()
    pieces.append(sentence[start:].strip())
    return [p for p in pieces if p]


def _default_content() -> Callable[[str], list]:
    """The admission tokenizer (stopwords and min_token_len) minus the question-generic terms, as the spell round uses
    it; when that config is unreadable, words outside the closed classes of 3+ letters."""
    try:
        from vera import fulldoc_admission as fa, spell_clarify
        icfg, _n = spell_clarify._icfg()
        if icfg:
            gen = {str(g).casefold() for g in icfg.get("question_generic_terms") or () if isinstance(g, str)}
            return lambda t: [w for w in fa.tokenize(t, icfg) if w not in gen]
    except Exception:  # noqa: BLE001
        pass
    return lambda t: [w for w in _words(t) if len(w) >= 3]


def _merge_followons(parts: list, content, cfg: dict) -> list:
    pron = _set(cfg, "pronouns")
    closed = _set(cfg, "interrogatives") | _set(cfg, "auxiliaries") | _set(cfg, "leading_conjunctions") | pron
    out: list = []
    for p in parts:
        stands = bool([w for w in content(p) if w not in closed]) or bool(pron & set(_words(p)))
        if stands or not out:
            out.append(p)
        else:
            out[-1] = f"{out[-1]} {p}"
    if len(out) > 1 and not [w for w in content(out[0]) if w not in closed] and not (pron & set(_words(out[0]))):
        out[1] = f"{out[0]} {out[1]}"  # a leading fragment with no content ('How, when and why did X happen?')
        out = out[1:]
    return out


def split_text(text: str, cfg: dict | None = None, content=None) -> list:
    """Every question of the prompt (the user's words), in order, before the cap. One item = not a multi-question."""
    cfg = load_config() if cfg is None else cfg
    content = content or _default_content()
    raw: list = []
    for s in _questions_of_sentences(str(text or "").strip(), cfg):
        raw.extend(_conjoined(s, cfg))
    return _merge_followons(raw, content, cfg)


def detect(text: str, cfg: dict | None = None, content=None) -> Split | None:
    """The bounded split, or None when the prompt holds one question (or the detector is off or fails)."""
    try:
        cfg = load_config() if cfg is None else cfg
        if mode_of(cfg) == "off":
            return None
        content = content or _default_content()
        qs = split_text(text, cfg, content)
        if len(qs) < 2:
            return None
        cap = cfg.get("max_questions")
        cap = int(cap) if isinstance(cap, int) and not isinstance(cap, bool) and cap >= 2 else 3
        pron = _set(cfg, "pronouns")
        parts, carried, prev = [], 0, ()
        for i, q in enumerate(qs[:cap], 1):
            own = tuple(dict.fromkeys(content(q)))
            has_pron = bool(pron & set(_words(q)))
            carry: tuple = ()
            if i > 1 and has_pron:
                carry = tuple(w for w in prev if w not in own)
            if carry:
                carried += 1
            parts.append(Part(index=i, text=q, query=f"{q} ({' '.join(carry)})" if carry else q, carry=carry))
            if own and not has_pron:  # the antecedent is the latest part that names its own subject
                prev = own
        return Split(parts=tuple(parts), over_cap=tuple(qs[cap:]), pronoun_carry=carried)
    except Exception:  # noqa: BLE001  never breaks the Ask path: one question, as before
        return None


def shape(text: str, split: Split | None) -> str:
    """multi_question | multi_part | single, for the request trace (counts and this token only)."""
    if split is not None:
        return "multi_question"
    try:
        from vera import subquery
        return "multi_part" if len(subquery.decompose(text).parts) > 1 else "single"
    except Exception:  # noqa: BLE001
        return "single"
