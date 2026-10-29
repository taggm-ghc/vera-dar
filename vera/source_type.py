"""T1 (IMPL-1028-T1O1): rule-based source-type router for acquisition. Pure: no I/O besides the (optional) tracked
config read, no model call, no stored state. The route is a per-request parameter derived from the question text only;
it is never associated with caller identity and never granted or denied access (AGENTS.md rule 1: not a visitor
profile). Switch: config/auto_acquire.json source_type_router.enabled (default false); the caller checks enabled().

route(question) -> "scholarly" | "general" | "mixed"
  general   ONLY on a general-domain lexicon hit with no scholarly signal (SECREV C8).
  scholarly on a scholarly signal with no general-domain hit.
  mixed     everything else (ambiguity, no signal, both): fail-open, the pool is today's plus web candidates.
The trace carries rule ids and counts only, never the question text.
The lexicons are an owner-reviewable SEED / FALLBACK (config); the target is a DB table (owner: DDL gated)."""
from __future__ import annotations

import re

ROUTES = ("scholarly", "general", "mixed")
_WORD = re.compile(r"[a-z0-9][a-z0-9'\-]*")


def _block(cfg) -> dict:
    """The source_type_router block, from the whole acquisition config, the block itself, or (None) the tracked file."""
    if cfg is None:
        from vera import acq_config
        try:
            cfg = acq_config.load_config()
        except Exception:  # noqa: BLE001 - a bad config means the router is off / lexicon empty
            cfg = None
    if isinstance(cfg, dict) and isinstance(cfg.get("source_type_router"), dict):
        return cfg["source_type_router"]
    return cfg if isinstance(cfg, dict) else {}


def enabled(cfg) -> bool:
    """True only when source_type_router.enabled is exactly true."""
    b = cfg.get("source_type_router") if isinstance(cfg, dict) else None
    return isinstance(b, dict) and b.get("enabled") is True


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _norm(text: str) -> list:
    return [_stem(w) for w in _WORD.findall((text or "").lower())]


def _hits(tokens: list, terms) -> list:
    """Matched lexicon terms. A term ending in '*' is a token prefix; a multi-word term is a phrase on stemmed tokens."""
    joined = " " + " ".join(tokens) + " "
    out = []
    for t in terms or ():
        if not isinstance(t, str) or not t.strip():
            continue
        t = t.strip().lower()
        if t.endswith("*"):
            p = t[:-1]
            if any(w.startswith(p) for w in tokens):
                out.append(t)
        elif " " + " ".join(_norm(t)) + " " in joined:
            out.append(t)
    return out


def classify(question: str, cfg=None) -> dict:
    """{"route", "basis": [rule ids], "scores": {"s", "g"}}. Rule ids and counts only (no question text)."""
    b = _block(cfg)
    lex = b.get("lexicon") if isinstance(b.get("lexicon"), dict) else {}
    toks = _norm(question)
    sch = _hits(toks, lex.get("scholarly"))
    gen = _hits(toks, lex.get("general"))
    # ambiguous general terms (e.g. an abbreviation shared with a biomedical term) count only with a general cue
    amb = _hits(toks, lex.get("general_ambiguous"))
    if amb and gen:
        gen = gen + amb
    s, g = len(sch), len(gen)
    if g and not s:
        r, basis = "general", ["G-LEX"]
    elif s and not g:
        r, basis = "scholarly", ["S-LEX"]
    elif s and g:
        r, basis = "mixed", ["S-LEX", "G-LEX", "CONFLICT"]
    else:
        r, basis = "mixed", ["NO-SIGNAL"]
    return {"route": r, "basis": basis, "scores": {"s": s, "g": g}}


def route(question: str, cfg=None) -> str:
    return classify(question, cfg)["route"]


def fit(route_name: str, cand, cfg=None) -> int:
    """Topic-fit rank of a candidate for a route (lower is better). Kind: 'web' = DuckDuckGo-origin candidate
    (acq_source == 'ddg'), else 'scholarly'. Unknown route/kind -> 0 (neutral)."""
    tbl = (_block(cfg).get("fit_table") or {}) if cfg is not None else {}
    kind = "web" if isinstance(cand, dict) and cand.get("acq_source") == "ddg" else "scholarly"
    try:
        return int((tbl.get(route_name) or {}).get(kind, 0))
    except (TypeError, ValueError):
        return 0
