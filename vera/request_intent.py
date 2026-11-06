"""MIDTERMS-WP1-1106 RC1: split a trailing request or meta instruction ("Keep searching until you find an answer")
from the research question. The phrase list is config (config/request_intent.json), not code. Only a whole sentence
or a trailing clause that is made up entirely of listed phrases and filler words is split off; a question that
merely mentions searching is unchanged. Returns (research_text, flags). Pure, offline, never raises."""
from __future__ import annotations

import json
import re
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "request_intent.json"
_cache: dict | None = None
_SENT = re.compile(r"[^.!?\n]+[.!?]*\s*|\n+")


def _cfg() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            _cache = {}
    return _cache


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", s.lower().replace("'", "")).split())


def _kind(piece: str, cfg: dict) -> str | None:
    """'persist', 'plain' or None: is the whole piece request phrases plus filler?"""
    t = " " + _norm(piece) + " "
    if t.strip() == "":
        return None
    found = None
    phrases = [(p, "persist") for p in cfg.get("persist_phrases", [])] + [(p, "plain") for p in cfg.get("plain_phrases", [])]
    for p, k in sorted(phrases, key=lambda x: -len(x[0])):
        n = " " + _norm(p) + " "
        if n in t:
            t = t.replace(n, " ")
            if found != "persist":
                found = k
    if found is None:
        return None
    fill = set(_norm(" ".join(cfg.get("filler_words", []))).split())
    return found if all(w in fill for w in t.split()) else None


def split_request(text: str) -> tuple[str, dict]:
    flags = {"persist": False}
    cfg = _cfg()
    if not isinstance(text, str) or not text.strip() or not cfg:
        return text, flags
    sents = _SENT.findall(text)
    keep: list[str] = []
    removed = False
    for s in sents:
        k = _kind(s, cfg)
        if k is None and "," in s:  # trailing clause: "..., and keep searching until you find an answer"
            head, _, tail = s.rpartition(",")
            k2 = _kind(tail, cfg)
            if k2 and _norm(head):
                keep.append(head.rstrip() + ("?" if text.rstrip().endswith("?") and s.rstrip().endswith("?") else "") + " ")
                removed = True
                flags["persist"] = flags["persist"] or k2 == "persist"
                continue
        if k is None:
            keep.append(s)
        else:
            removed = True
            flags["persist"] = flags["persist"] or k == "persist"
    out = "".join(keep).strip()
    if not removed or not _norm(out):
        return text, {"persist": False}
    return out, flags
