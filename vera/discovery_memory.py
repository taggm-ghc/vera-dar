"""K2.I4 discovery notes (plan-K2-contract.md r2, closure R-I4): a non-evidentiary note that lists which published
documents VERA-DAR listed for a topic. A note is never a finding and never evidence.

Privacy (OD-K15c): `build_note` takes NO question parameter, so a note cannot be derived from the visitor's question;
topic terms come from candidate titles only. The question reaches this module only as a gate input
(`check_discovery_note`) for the echo-overlap refusal and is never stored, logged or returned.
"""
import re
from pathlib import Path

from vera import answer_contract
from vera.memory_gate import (ECHOES_QUESTION, INJECTION_PATTERN, INVISIBLE_UNICODE, LICENCE, MARKUP, NO_CITATION,
                              NO_IDENTIFIER, PERSONAL, PROVIDER_NOT_ALLOWED, TOOL_DUMP, VOLATILE, GateResult,
                              _has_invisible, _injection_hit, _markup, _overlap, _personal, _provider_allowed,
                              _squash, _tool_dump, load_gate_config, normalise_claim)

CONFIG_PATH = answer_contract.CONFIG_PATH
PREFIX_IN_TITLE = "prefix_in_title"  # V7: a title must not forge the note prefix
_BAD_URL_CHARS = set(" ()<>\"'")
_WORD = re.compile(r"[A-Za-z0-9]{3,}")


def _block(contract_path) -> dict:
    return answer_contract.load_config(contract_path).get("discovery_memory") or {}


def safe_url(url: str) -> bool:
    u = str(url or "")
    return u.startswith(("https://", "http://")) and not (set(u) & _BAD_URL_CHARS) and not any(c.isspace() for c in u)


def _canonical(cand: dict) -> str:
    doi = str(cand.get("doi") or "").strip().lower()
    return f"https://doi.org/{doi}" if doi else str(cand.get("url") or "").strip().rstrip("/")


def eligible_candidates(candidates, gate_cfg) -> list:
    """Candidates whose provider, licence and identifier pass the memory gate rules (excluded before build_note)."""
    out = []
    for c in candidates or ():
        if (_provider_allowed(c.get("provider") or "", gate_cfg) and c.get("licence_decision") in gate_cfg.allowed_licence
                and str(c.get("identifier") or "").strip()):
            out.append(c)
    return out


def _join(items: list) -> str:
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def topic_terms(titles, store_cfg: dict) -> str:
    """Same rules as RoMemoryReader.query_terms (stopwords, min length, max terms), over titles only."""
    stop = {w.lower() for w in store_cfg["query_stopwords"]}
    short = int(store_cfg["min_query_word_len"])
    words = list(dict.fromkeys(w for w in (x.lower() for x in _WORD.findall(" ".join(titles)))
                               if len(w) >= short and w not in stop))
    return ", ".join(words[: int(store_cfg["max_query_terms"])])


def _compose(entries: list, store_cfg: dict, contract_path) -> str:
    titles = [t for t, _ in entries]
    topic = topic_terms(titles, store_cfg)
    sentence = (answer_contract.text("note_prefix", contract_path)
                + answer_contract.text("note_template", contract_path).format(topic=topic, candidates=_join(titles)))
    return sentence + "\n" + " ".join(u for _, u in entries)


def build_note(candidates, store_cfg: dict, contract_path=CONFIG_PATH, gate_cfg=None) -> str | None:
    """'<sentence>\\n<urls>' or None. No question parameter (OD-K15c). Inputs: public candidate fields only."""
    blk = _block(contract_path)
    max_c, max_chars = int(blk.get("max_candidates", 0)), int(blk.get("max_note_chars", 0))
    if gate_cfg is None:
        try:
            gate_cfg = load_gate_config()
        except Exception:  # noqa: BLE001  (fall back to plain whitespace squeeze)
            gate_cfg = None
    entries = []
    for c in candidates or ():
        url = _canonical(c)
        title = str(c.get("title") or "")
        title = normalise_claim(title, gate_cfg) if gate_cfg is not None else re.sub(r"\s+", " ", title).strip()
        if title and safe_url(url):
            entries.append((title, url))
    entries = entries[:max_c]
    if not answer_contract.text("note_template", contract_path) or not answer_contract.text("note_prefix", contract_path):
        return None
    while entries:
        text = _compose(entries, store_cfg, contract_path)
        if len(text) <= max_chars:
            return text
        entries.pop()
    return None


def check_discovery_note(sentence: str, candidates, question: str, gate_cfg,
                         contract_path=CONFIG_PATH) -> GateResult:
    """Structural gate on the note SENTENCE (prefix plus template). The question is used only for the overlap check."""
    reasons: set = set()
    cands = list(candidates or ())
    if not cands:
        reasons.add(NO_CITATION)
    for c in cands:
        if not _provider_allowed(c.get("provider") or "", gate_cfg):
            reasons.add(PROVIDER_NOT_ALLOWED)
        if c.get("licence_decision") not in gate_cfg.allowed_licence:
            reasons.add(LICENCE)
        if not str(c.get("identifier") or "").strip():
            reasons.add(NO_IDENTIFIER)
    prefix = answer_contract.text("note_prefix", contract_path).casefold()
    if not prefix or any(prefix in str(c.get("title") or "").casefold() for c in cands) \
            or sentence.casefold().count(prefix) != 1:
        reasons.add(PREFIX_IN_TITLE)
    if _has_invisible(sentence, gate_cfg):
        reasons.add(INVISIBLE_UNICODE)
    if _injection_hit(sentence, gate_cfg):
        reasons.add(INJECTION_PATTERN)
    if _tool_dump(sentence, gate_cfg):
        reasons.add(TOOL_DUMP)
    if _personal(sentence, gate_cfg):
        reasons.add(PERSONAL)
    if _overlap(sentence, question or "", gate_cfg) > gate_cfg.max_question_overlap:
        reasons.add(ECHOES_QUESTION)
    low = _squash(sentence)
    if any(re.search(r"\b%s\b" % re.escape(t), low) for t in gate_cfg.volatile_terms):
        reasons.add(VOLATILE)
    if _markup(sentence, gate_cfg):
        reasons.add(MARKUP)
    return GateResult(not reasons, tuple(sorted(reasons)))
