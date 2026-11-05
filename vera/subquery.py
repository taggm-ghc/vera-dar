"""NONASK-1103 S1/S1b: deterministic question decomposition into sub-queries.

Pure Python: no DB, network or model call. The ORIGINAL question stays authoritative: decompose() returns it
unchanged in SubQueryPlan.original, and every part is built only from the original's own words (drift guard,
CTQ-S4). Rules, in order:
  1. entity list: >=2 items joined by ',', ', and', 'and', 'or', '&' or '/' (items are active registry entities, or
     capitalised non-initial single tokens that are not generic terms); each item becomes one part, sharing the
     prefix and tail of the original wording;
  2. comparison: 'X vs Y', 'compare X with/and/to Y', 'difference between X and Y';
  3. otherwise one part (the original).
More than max_parts items: the overflow is merged into the last part, nothing is dropped.
The model fallback (S3b) is a stub behind config flag model_fallback.enabled, which ships OFF; with no injected
decomposer it returns nothing. Registry rows come from config/entity_registry.json (seed/fallback for the proposed
DB table); only rows with state == 'active' count.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from vera import temporality

_ROOT = Path(__file__).resolve().parent.parent
_CFG_PATH = _ROOT / "config" / "subquery.json"
_REG_PATH = _ROOT / "config" / "entity_registry.json"

_DEFAULT_CFG = {"enabled": True, "max_parts": 4, "max_comparison_side_words": 8, "generic_terms": [],
                "model_fallback": {"enabled": False}}

_SEP = r"(?:\s*,\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+|\s*&\s*|\s*/\s*)"
_SEP_RE = re.compile(r"^" + _SEP + r"$", re.I)
_CAP_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9+#.\-]*[A-Za-z0-9+#]|\b[A-Z]\b")
_COMPARE = [
    re.compile(r"\b(?:compare|comparing|contrast)\s+(?P<a>.+?)\s+(?:with|to|and|against|versus|vs\.?)\s+(?P<b>.+?)"
               r"(?:\s+(?:in terms of|on|for|regarding|about|in)\s+(?P<c>.+?))?\s*[?.!]?\s*$", re.I),
    re.compile(r"\bdifferences?\s+between\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)"
               r"(?:\s+(?:in terms of|on|for|regarding|about|in)\s+(?P<c>.+?))?\s*[?.!]?\s*$", re.I),
    re.compile(r"^(?P<a>.+?)\s+(?:vs\.?|versus)\s+(?P<b>.+?)\s*[?.!]?\s*$", re.I),
]


@dataclass(frozen=True)
class SubPart:
    label: str
    text: str
    entities: tuple = ()
    temporal: bool = False


@dataclass(frozen=True)
class SubQueryPlan:
    original: str
    parts: tuple
    rule: str  # entity_list | comparison | single | model_fallback
    trace: tuple = field(default_factory=tuple)

    def entity_hits(self) -> list:
        seen: list = []
        for p in self.parts:
            for e in p.entities:
                if e not in seen:
                    seen.append(e)
        return seen


# ---------------------------------------------------------------- config and registry
def load_config(path: Path | None = None) -> dict:
    try:
        cfg = json.loads((path or _CFG_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    out = dict(_DEFAULT_CFG)
    out.update({k: v for k, v in cfg.items() if not k.startswith("_")})
    return out


def load_entity_registry(path: Path | None = None) -> list:
    """Active rows only (draft/suspended/retired never count). An alias claimed by two active entities is dropped
    from both (an alias collision never maps an entity to another entity's host). Rows without a host list are kept."""
    try:
        data = json.loads((path or _REG_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if path is None:  # LISTS-TO-DB-1104: DB list when enabled, else the file rows
        try:
            from vera import list_tables
            data = {**data, "entities": list_tables.get("entity.registry", data.get("entities", []))}
        except Exception:  # noqa: BLE001  loader missing or faulty: keep the file value
            pass
    rows = [r for r in data.get("entities", []) if isinstance(r, dict) and r.get("state") == "active"
            and r.get("entity")]
    owners: dict = {}
    for r in rows:
        for a in set(x.lower() for x in r.get("aliases", []) + [r["entity"]]):
            owners.setdefault(a, set()).add(r["entity"])
    clash = {a for a, o in owners.items() if len(o) > 1}
    out = []
    for r in rows:
        r = dict(r)
        r["aliases"] = [a for a in dict.fromkeys(r.get("aliases", []) + [r["entity"]]) if a.lower() not in clash]
        if r["aliases"]:
            out.append(r)
    return out


def match_entities(text: str, registry: list) -> list:
    """Non-overlapping registry matches as (start, end, entity), longest alias first."""
    t = text or ""
    taken: list = []
    cands = sorted(((a, r) for r in registry for a in r["aliases"]), key=lambda x: -len(x[0]))
    for alias, r in cands:
        flags = 0 if r.get("match_case") else re.I
        for m in re.finditer(r"(?<![\w-])" + re.escape(alias) + r"(?![\w-])", t, flags):
            s, e = m.span()
            if any(s < te and ts < e for ts, te, _ in taken):
                continue
            nxt = re.match(r"\s+([A-Za-z]+)", t[e:])
            if nxt and nxt.group(1).lower() in [x.lower() for x in r.get("exclude_next", [])]:
                continue
            taken.append((s, e, r["entity"]))
    return sorted(taken)


# ---------------------------------------------------------------- helpers
def _toks(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _tidy(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" \t,;:/&")


def drift_ok(original: str, parts: list) -> bool:
    """CTQ-S4: every part is non-empty and its terms are a subset of the original question's terms."""
    o = _toks(original)
    return bool(parts) and all(p.strip() and _toks(p) <= o for p in parts)


def _items(text: str, registry: list, generic: set) -> list:
    """Candidate items (start, end, entity_or_None) in order: registry matches plus capitalised non-initial tokens."""
    items = [(s, e, ent) for s, e, ent in match_entities(text, registry)]
    for m in _CAP_TOKEN.finditer(text):
        s, e = m.span()
        if s == 0 or text[:s].strip() == "" or m.group(0).lower() in generic:
            continue
        if any(s < ie and is_ < e for is_, ie, _ in items):
            continue
        items.append((s, e, None))
    return sorted(items)


def _find_run(text: str, items: list) -> list:
    """Longest run of adjacent items joined only by list separators; needs >=2 items and (>=1 registry entity)."""
    best: list = []
    i = 0
    while i < len(items):
        run = [items[i]]
        j = i + 1
        while j < len(items) and _SEP_RE.match(text[run[-1][1]:items[j][0]]):
            run.append(items[j])
            j += 1
        if len(run) >= 2 and any(x[2] for x in run) and len(run) > len(best):
            best = run
        i = max(j, i + 1)
    return best


def _volatile(ents: tuple, registry: list) -> bool:
    return any(r.get("volatile") for r in registry if r["entity"] in ents)


def _mk(label: str, text: str, ents: tuple, registry: list, cfg_t: dict | None) -> SubPart:
    t = temporality.detect(text, volatile_entity=_volatile(ents, registry), cfg=cfg_t)
    return SubPart(label, text, ents, t.current)


# ---------------------------------------------------------------- rules
def _rule_entity_list(q: str, registry: list, cfg: dict, cfg_t):
    generic = {g.lower() for g in cfg.get("generic_terms", [])}
    run = _find_run(q, _items(q, registry, generic))
    if not run:
        return None
    prefix, tail = q[:run[0][0]], re.sub(r"[?.!\s]+$", "", q[run[-1][1]:])
    maxp = max(1, int(cfg.get("max_parts", 4)))
    groups = [[x] for x in run]
    if len(groups) > maxp:  # overflow merges into the last part; nothing dropped
        groups = groups[:maxp - 1] + [run[maxp - 1:]]
    parts, texts = [], []
    for g in groups:
        if len(g) == 1:
            body = q[g[0][0]:g[0][1]]
        else:
            body = q[g[0][0]:g[-1][1]]
        text = _tidy(f"{prefix} {body} {tail}")
        ents = tuple(dict.fromkeys(x[2] for x in g if x[2]))
        label = ", ".join(q[x[0]:x[1]] for x in g)
        texts.append(text)
        parts.append(_mk(label, text, ents, registry, cfg_t))
    if not drift_ok(q, texts):
        return None
    return parts, ("entity_list:%d items" % len(run),)


def _rule_comparison(q: str, registry: list, cfg: dict, cfg_t):
    maxw = int(cfg.get("max_comparison_side_words", 8))
    for rx in _COMPARE:
        m = rx.search(q)
        if not m:
            continue
        a, b, c = _tidy(m.group("a")), _tidy(m.group("b")), _tidy(m.groupdict().get("c") or "")
        if not a or not b or len(a.split()) > maxw or len(b.split()) > maxw:
            continue
        texts = [_tidy(f"{a} {c}"), _tidy(f"{b} {c}")]
        if not drift_ok(q, texts):
            continue
        parts = []
        for side, text in ((a, texts[0]), (b, texts[1])):
            ents = tuple(dict.fromkeys(e for _, _, e in match_entities(side, registry)))
            parts.append(_mk(side, text, ents, registry, cfg_t))
        return parts, ("comparison",)
    return None


def _has_list_cue(q: str) -> bool:
    return len(re.findall(r",|\band\b|\bor\b|&|/", q or "", re.I)) >= 1


def model_fallback(question: str, cfg: dict, decomposer: Callable | None = None) -> list | None:
    """S3b stub. Ships OFF. Returns validated part texts, or None. No decomposer is bundled."""
    if not (cfg.get("model_fallback") or {}).get("enabled") or decomposer is None:
        return None
    try:
        out = [str(x) for x in decomposer(question)]
    except Exception:  # noqa: BLE001 - a failing optional helper must never break a request
        return None
    return out if drift_ok(question, out) else None


def decompose(question: str, registry: list | None = None, cfg: dict | None = None, cfg_t: dict | None = None,
              decomposer: Callable | None = None) -> SubQueryPlan:
    q = question or ""
    cfg = cfg if cfg is not None else load_config()
    registry = load_entity_registry() if registry is None else registry
    single = lambda tr: SubQueryPlan(q, (_mk("question", q, tuple(dict.fromkeys(e for _, _, e in match_entities(q, registry))),
                                              registry, cfg_t),), "single", tr)  # noqa: E731
    if not cfg.get("enabled", True) or not q.strip():
        return single(("disabled or empty",))
    for name, fn in (("entity_list", _rule_entity_list), ("comparison", _rule_comparison)):
        r = fn(q, registry, cfg, cfg_t)
        if r:
            return SubQueryPlan(q, tuple(r[0]), name, r[1])
    if _has_list_cue(q):
        texts = model_fallback(q, cfg, decomposer)
        if texts and len(texts) > 1:
            parts = tuple(_mk(t, t, tuple(dict.fromkeys(e for _, _, e in match_entities(t, registry))), registry, cfg_t)
                          for t in texts)
            return SubQueryPlan(q, parts, "model_fallback", ("validated by drift guard",))
    return single(("no rule matched",))


def entity_context(name: str, registry: list | None = None) -> dict | None:
    """The disambiguation context of an active registry entity (by entity name or alias, case-insensitive) for
    vera.search_query.entity_hit, or None when the registry does not describe it."""
    registry = load_entity_registry() if registry is None else registry
    low = str(name or "").strip().lower()
    for r in registry:
        names = [r.get("entity", ""), *(r.get("aliases") or ())]
        if low in {str(n).lower() for n in names}:
            return {"names": [str(n).lower() for n in names if n], "full_aliases": list(r.get("full_aliases") or ()),
                    "parents": list(r.get("parents") or ()), "context_terms": list(r.get("context_terms") or ()),
                    "generic_phrases": list(r.get("generic_phrases") or ())}
    return None
