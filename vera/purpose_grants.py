"""PURPOSE-GRANTS-1108 (owner 2026-11-08: "Allow educational|research|etc should also be a green light for us").

VERA+ is non-commercial educational research, so a site that explicitly grants access or use for research,
educational, scholarly, academic, teaching or non-commercial purposes may be granting it to VERA+. This module finds
such grants with ONE config-driven vocabulary (config/purpose_grants.json; no per-site lists) and, in shadow mode,
records what a grant WOULD change. It never changes an access or use decision: there is no "on" mode until the shadow
evidence is calibrated and the owner decides.

Grants count only from site-owner channels (config site_channels): robots.txt comments and Content-Signal /
Content-Usage fields (robots.txt is always fetched from the page's own host), the fetched page's own HTTP response
headers, and the host's own /.well-known/tdmrep.json. Page meta, page body and anything a third party can write
(comments, forums, wiki text) are never scanned for grants.

A grant can only lift a restriction listed under `overridable` (a generic AI or TDM use restriction), and only when
it is linked to that restriction: a carve-out on the same line ("AI crawlers are blocked, except for research") or a
grant that itself names AI (or, for TDM, names TDM). Everything in `never_overridable` (own-token and * disallows,
hyperscaler-only groups pending owner decision H13, the licence reject list, login/paywall, deny list, registry
forbid, rate and Crawl-delay limits) stays binding whatever a grant says. A negated statement ("not for research")
is never a grant; a conflict, a non-standard structured value, an unread tdm-policy or a "contact us first" condition
goes to review, never to admit.

Recording: request_trace event stage "purpose_grant" (codes only: decision, restriction, channel, purposes,
conditions, evidence ids), and an owner-only in-process ring (shadow_evidence()) with the verbatim line, its channel and
host, keyed by evidence id. Nothing here raises into a caller: every public function swallows its own errors.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "purpose_grants.json"
MODES = ("off", "shadow")
STAGE = "purpose_grant"
KINDS_POSITIVE = ("grant", "carveout")
KINDS_REVIEW = ("ambiguous", "structured", "policy_unread")
_CFG: dict | None = None


@dataclass(frozen=True)
class Grant:
    channel: str
    kind: str                      # grant | carveout | negated | ambiguous | structured | policy_unread
    purposes: tuple = ()
    conditions: tuple = ()
    ai_scoped: bool = False
    tdm_scoped: bool = False
    verbatim: str = field(default="", repr=False)
    evidence_id: str = ""


def load_config(path: Path | None = None) -> dict:
    """Raises on a missing or malformed file; mode() turns any error into off (shadow decides nothing either way)."""
    global _CFG
    path = path or CONFIG_PATH
    if path == CONFIG_PATH and _CFG is not None:
        return _CFG
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    need = {"purposes": dict, "permit_terms": list, "negation_terms": list, "carveout_terms": list, "ai_terms": list,
            "tdm_terms": list, "conditions": dict, "overridable": dict, "never_overridable": list,
            "site_channels": list, "limits": dict}
    if not isinstance(c, dict) or any(not isinstance(c.get(k), t) for k, t in need.items()):
        raise ValueError("purpose_grants config malformed")
    if set(c["overridable"]) & set(c["never_overridable"]):
        raise ValueError("purpose_grants config: a restriction is both overridable and never_overridable")
    if path == CONFIG_PATH:
        _CFG = c
    return c


def mode(cfg: dict | None = None) -> str:
    try:
        m = str((cfg if cfg is not None else load_config()).get("mode") or "off").strip().lower()
    except Exception:  # noqa: BLE001
        return "off"
    return m if m in MODES else "off"


# ------------------------------------------------------------------------------------------------ text matching
def _toks(s: str) -> list:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower().replace("'", "").replace("’", "")).split()


def _bounds(raw: str) -> set:
    """Token indices (into _toks(raw)) that begin a new sentence or clause (after . ; : ! ? or a dash)."""
    out, n = set(), 0
    for part in re.split(r"[.;:!?\u2013\u2014]|\s-+\s", str(raw or "")):
        out.add(n)
        n += len(_toks(part))
    return out


def _spans(toks: list, terms) -> list:
    """(start, end) token spans of every term (multi-word, exact token match)."""
    out = []
    for t in terms:
        tt = _toks(t)
        n = len(tt)
        if not n:
            continue
        for i in range(len(toks) - n + 1):
            if toks[i:i + n] == tt:
                out.append((i, i + n))
    return out


def _overlaps(a, spans) -> bool:
    return any(a[0] < b[1] and b[0] < a[1] for b in spans)


def _dist(a, b) -> int:
    """Tokens between two spans (0 = adjacent)."""
    return b[0] - a[1] if a[1] <= b[0] else a[0] - b[1] if b[1] <= a[0] else 0


def _purposes(cfg, toks) -> list:
    out = []
    for name, terms in cfg["purposes"].items():
        for sp in _spans(toks, terms):
            out.append((name, sp))
    return out


def _evid(channel: str, verbatim: str) -> str:
    return hashlib.sha256(f"{channel}\n{verbatim}".encode("utf-8", "replace")).hexdigest()[:16]


_REDACT = (
    (re.compile(r"[\w.+-]+\s*(?:@|\(at\)|\[at\])\s*[\w-]+(?:\s*(?:\.|\(dot\)|\[dot\])\s*[\w-]+)+", re.I), "[email]"),
    (re.compile(r"\b(?:mailto|tel|sms|callto):\S*", re.I), "[contact]"),
    (re.compile(r"(?<![\w@])@[\w.-]{2,}"), "[handle]"),
    (re.compile(r"(?<!\w)\+?\(?\d[\d\s().-]{6,}\d"), "[phone]"),
)


def redact(text: str) -> str:
    """N1: strip email addresses, phone numbers, URLs, @handles and mailto/tel links (personal contact details a
    site could put in a robots.txt comment) before the text is stored anywhere."""
    s = str(text or "")
    for rx, sub in _REDACT:
        s = rx.sub(sub, s)
    return s


def _verbatim(cfg, text: str) -> str:
    s = re.sub(r"[\x00-\x1f\x7f]", " ", redact(text)).strip()
    return s[: int(cfg["limits"].get("verbatim_chars", 300))]


def _conditions(cfg, toks, purposes) -> tuple:
    conds = {name for name, terms in cfg["conditions"].items() if _spans(toks, terms)}
    if any(p == "non_commercial" for p, _ in purposes):
        conds.add("non_commercial")
    return tuple(sorted(conds))


_CLAUSE_SPLIT = re.compile(r"\s[-\u2013\u2014]+\s|[\u2013\u2014;]")


def classify_line(text: str, channel: str, cfg: dict | None = None) -> Grant | None:
    """One free-text statement (a robots.txt comment) -> a Grant, or None when it states no purpose permission or
    negation (a bare mention such as "our research department" is None).

    F3 (narrow): a line that reads as negated as a whole becomes a carve-out only when it splits on a dash or
    semicolon into clauses, an earlier clause restricts AI/bots, and a LATER clause on its own is a plain grant
    ("ai bots blocked - research ok"). No clause may itself be negated or ambiguous. Without a separator nothing
    changes, so "Disallow research bots but allow everyone else" and "AI crawlers are not blocked for research" stay
    negated."""
    cfg = cfg if cfg is not None else load_config()
    g = _classify(text, channel, cfg)
    if g is None or g.kind != "negated":
        return g
    try:
        raw = str(text or "")[: int(cfg["limits"].get("max_line_chars", 500))]
        parts = [p for p in _CLAUSE_SPLIT.split(raw) if p.strip()]
        if len(parts) < 2:
            return g
        sub = [_classify(p, channel, cfg) for p in parts]
        if any(x is not None and x.kind not in ("grant",) for x in sub):
            return g
        for i, x in enumerate(sub):
            if x is None or x.kind != "grant":
                continue
            earlier = _toks(" ".join(parts[:i]))
            if i and _spans(earlier, cfg["negation_terms"]) and _spans(earlier, cfg["ai_terms"]):
                return Grant(channel, "carveout", g.purposes, g.conditions, g.ai_scoped, g.tdm_scoped, g.verbatim,
                             g.evidence_id)
    except Exception:  # noqa: BLE001
        pass
    return g


def _classify(text: str, channel: str, cfg: dict) -> Grant | None:
    lim = cfg["limits"]
    raw = str(text or "")[: int(lim.get("max_line_chars", 500))]
    toks = _toks(raw)
    purp = _purposes(cfg, toks)
    if not purp:
        return None
    pspans = [sp for _, sp in purp]
    cspans_cond = [sp for terms in cfg["conditions"].values() for sp in _spans(toks, terms)]
    # a negation word inside a purpose or condition phrase ("not for profit", "no redistribution") is not a negation
    negs = [n for n in _spans(toks, cfg["negation_terms"]) if not _overlaps(n, pspans) and not _overlaps(n, cspans_cond)]
    perms = [p for p in _spans(toks, cfg["permit_terms"]) if not _overlaps(p, pspans)]
    carves = _spans(toks, cfg["carveout_terms"])
    bounds = _bounds(raw)
    w, cw = int(cfg.get("window", 5)), int(cfg.get("carveout_window", 4))

    def neg_of_perm(p):  # "not permitted", "may not", "never allowed"
        return any((n[1] <= p[0] and p[0] - n[1] <= 1) or (n[0] >= p[1] and n[0] - p[1] == 0) for n in negs)

    kinds = set()
    for _name, ps in purp:
        carve = [c for c in carves if c[1] <= ps[0] and ps[0] - c[1] <= cw
                 and any(n[1] <= c[0] for n in negs)]  # a restriction earlier on the line, then "except ..."
        after_neg = [n for n in negs if n[0] >= ps[1] and n[0] - ps[1] <= w]
        if carve:
            kinds.add("ambiguous" if after_neg else "carveout")
            continue
        near_p = [p for p in perms if _dist(ps, p) <= w]
        # F1: a restriction verb right before the purpose ("Disallow research bots ...", "block research") makes
        # the purpose the OBJECT of the restriction; a later permit ("... but allow everyone else") grants nothing.
        if any(n[1] <= ps[0] and ps[0] - n[1] <= 2 and not any(n[1] <= p[0] < ps[0] for p in perms)
               and not any(n[1] <= b <= ps[0] for b in bounds) for n in negs):
            kinds.add("negated")
            continue
        clean = [p for p in near_p if not neg_of_perm(p)
                 and not any(min(ps[1], p[1]) <= n[0] and n[1] <= max(ps[0], p[0]) for n in negs)]
        negated_p = [p for p in near_p if neg_of_perm(p)]
        near_n = [n for n in negs if _dist(ps, n) <= w]
        if clean:
            kinds.add("ambiguous" if negated_p else "grant")
        elif negated_p or near_n:
            kinds.add("negated")
        elif near_p:
            kinds.add("ambiguous")
    if not kinds:
        return None
    if len(kinds) == 1:
        kind = kinds.pop()
    elif kinds <= {"grant", "carveout"}:
        kind = "carveout"
    else:
        kind = "ambiguous"
    vb = _verbatim(cfg, raw)
    return Grant(channel, kind, tuple(sorted({n for n, _ in purp})), _conditions(cfg, toks, purp),
                 bool(_spans(toks, cfg["ai_terms"])), bool(_spans(toks, cfg["tdm_terms"])), vb, _evid(channel, vb))


def _structured(value: str, channel: str, cfg: dict) -> list:
    """key=value fields (Content-Signal, Content-Usage, X-Robots-Tag): a purpose-named key with a permit value is a
    non-standard structured grant (review only); with n/no it is negated. Standard keys are never grants."""
    out = []
    permit = {str(v).lower() for v in cfg.get("structured_permit_values", [])}
    v0 = re.sub(r"\s*=\s*", "=", str(value or "").lower()).replace('"', "")
    for part in re.split(r"[,;\s]+", v0):
        k, sep, v = part.partition("=")
        if not sep:
            continue
        kt = _toks(k)
        names = [n for n, terms in cfg["purposes"].items() if any(_toks(t) == kt for t in terms)]
        if not names:
            continue
        kind = "structured" if v in permit else "negated" if v in ("n", "no") else "ambiguous"
        vb = _verbatim(cfg, part)
        out.append(Grant(channel, kind, tuple(sorted(names)), (), False, False, vb, _evid(channel, vb)))
    return out


def scan_robots(text: str, cfg: dict | None = None) -> tuple:
    """Grants stated in a robots.txt: every comment (full-line or trailing) is one statement; Content-Signal and
    Content-Usage values are scanned for purpose-named keys. Never raises: () on any error."""
    try:
        cfg = cfg if cfg is not None else load_config()
        lim = cfg["limits"]
        out = []
        for raw in str(text or "").lstrip("﻿").splitlines()[: int(lim.get("max_lines", 2000))]:
            body, hsh, comment = raw.partition("#")
            if hsh:
                g = classify_line(comment, "robots_comment", cfg)
                if g:
                    out.append(g)
            k, sep, v = body.strip().partition(":")
            if sep and k.strip().lower() in ("content-signal", "content-usage"):
                out.extend(_structured(v, "robots_field", cfg))
            if len(out) >= int(lim.get("max_grants", 16)):
                break
        return tuple(out[: int(lim.get("max_grants", 16))])
    except Exception:  # noqa: BLE001
        return ()


def scan_headers(headers, cfg: dict | None = None) -> tuple:
    """Grants in the fetched page's own response headers (structured keys only; a tdm-policy header is an unread
    policy -> review). Never raises."""
    try:
        cfg = cfg if cfg is not None else load_config()
        h = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        out = []
        for key in ("content-signal", "content-usage", "x-robots-tag"):
            if h.get(key):
                out.extend(_structured(h[key], "http_header", cfg))
        if h.get("tdm-policy", "").strip():
            vb = _verbatim(cfg, "TDM-Policy: " + h["tdm-policy"].strip())
            out.append(Grant("http_header", "policy_unread", (), (), False, True, vb, _evid("http_header", vb)))
        return tuple(out[: int(cfg["limits"].get("max_grants", 16))])
    except Exception:  # noqa: BLE001
        return ()


def scan_tdmrep(rules, path: str, cfg: dict | None = None) -> tuple:
    """The host's own tdmrep.json: a rule matching `path` that reserves TDM and names a tdm-policy is a possible
    research permission. The policy document is NOT fetched here (no new request in shadow mode): review. Never raises."""
    try:
        cfg = cfg if cfg is not None else load_config()
        from vera import public_fetch
        out = []
        for r in rules if isinstance(rules, list) else []:
            if not isinstance(r, dict) or not public_fetch._tdm_loc_match(r.get("location"), path or "/"):
                continue
            pol = r.get("tdm-policy")
            if str(r.get("tdm-reservation")).strip() == "1" and isinstance(pol, str) and pol.strip():
                vb = _verbatim(cfg, json.dumps({k: r.get(k) for k in ("location", "tdm-reservation", "tdm-policy")}))
                out.append(Grant("tdmrep", "policy_unread", (), (), False, True, vb, _evid("tdmrep", vb)))
        return tuple(out[: int(cfg["limits"].get("max_grants", 16))])
    except Exception:  # noqa: BLE001
        return ()


# ------------------------------------------------------------------------------------------------ assessment
def assess(restriction: str, grants, cfg: dict | None = None) -> dict | None:
    """What a purpose grant WOULD do to `restriction` (a gate reason code). None when there is nothing to record.
    decision: would_admit | review | negated | not_overridable. Pure; never changes the restriction."""
    cfg = cfg if cfg is not None else load_config()
    site = [g for g in grants or () if isinstance(g, Grant) and g.channel in cfg["site_channels"]]
    if not site:
        return None
    ov = cfg["overridable"].get(restriction)
    if restriction in cfg["never_overridable"] or not isinstance(ov, dict):
        return _result("not_overridable", restriction, site)
    cand = [g for g in site if g.channel in ov.get("channels", [])]
    if not cand:
        return _result("not_overridable", restriction, site)
    need = ov.get("needs", "linked")

    def linked(g):
        if need == "tdm_linked":
            return g.tdm_scoped
        return g.kind == "carveout" or g.ai_scoped or g.tdm_scoped

    pos = [g for g in cand if g.kind in KINDS_POSITIVE]
    neg = [g for g in cand if g.kind == "negated"]
    rev = [g for g in cand if g.kind in KINDS_REVIEW]
    lp = [g for g in pos if linked(g)]
    if pos and neg:
        return _result("review", restriction, pos + neg, why="conflict")
    if lp and not any("contact_first" in g.conditions for g in lp):
        return _result("would_admit", restriction, lp)
    if lp:
        return _result("review", restriction, lp, why="contact_first")
    if pos:
        return _result("review", restriction, pos, why="unlinked")
    if rev:
        return _result("review", restriction, rev, why=rev[0].kind)
    if neg:
        return _result("negated", restriction, neg)
    return None


def _result(decision, restriction, grants, why="") -> dict:
    return {"decision": decision, "restriction": restriction, "why": why or decision, "grants": list(grants),
            "purposes": tuple(sorted({p for g in grants for p in g.purposes})),
            "conditions": tuple(sorted({c for g in grants for c in g.conditions}))}


# ------------------------------------------------------------------------------------------------ shadow record
_LOCK = threading.Lock()
_EVIDENCE: "OrderedDict[str, dict]" = OrderedDict()
_SEEN: "OrderedDict[tuple, None]" = OrderedDict()


def _host(url) -> str:
    try:
        return (urlsplit(str(url or "")).hostname or "").lower().rstrip(".")
    except Exception:  # noqa: BLE001
        return ""


def _tok(s: str) -> str:
    s = re.sub(r"[^a-z0-9_:.-]+", "_", str(s or "").lower())[:64]
    return s or "none"


def shadow(restriction, grants, url: str | None = None) -> dict | None:
    """Record what a purpose grant would do to `restriction` (a reason code, or a list of every restriction that
    applies to this source) for `url`; return the assessment (or None). With several restrictions the source would be
    admitted only if EVERY one would be lifted; otherwise the first that would not is recorded. Shadow only: the
    caller's decision is unchanged whatever this returns. Never raises."""
    try:
        cfg = load_config()
        if mode(cfg) != "shadow" or not grants:
            return None
        rs = [restriction] if isinstance(restriction, str) else [r for r in restriction or () if r]
        results = [assess(r, grants, cfg) for r in rs]
        if not rs or any(x is None for x in results):
            # a restriction with nothing to say about it: record only when no restriction is left unexamined
            results = [x for x in results if x is not None]
            if not results or len(results) < len(rs):
                return None
        a = next((x for x in results if x["decision"] != "would_admit"), results[0])
        restriction = a["restriction"]
        host = _host(url)
        ids = [g.evidence_id for g in a["grants"]][:8]
        from vera import request_trace as rt
        key = (rt.current_id(), restriction, host, tuple(ids))
        ring = int(cfg["limits"].get("evidence_ring", 200))
        with _LOCK:
            if key in _SEEN:
                return a
            _SEEN[key] = None
            while len(_SEEN) > 4 * ring:
                _SEEN.popitem(last=False)
            for g in a["grants"]:
                _EVIDENCE[g.evidence_id] = {"evidence_id": g.evidence_id, "channel": g.channel, "kind": g.kind,
                                            "verbatim": g.verbatim, "host": host, "restriction": restriction,
                                            "decision": a["decision"], "purposes": g.purposes,
                                            "conditions": g.conditions}
                _EVIDENCE.move_to_end(g.evidence_id)
            while len(_EVIDENCE) > ring:
                _EVIDENCE.popitem(last=False)
        rt.event(STAGE, a["decision"], {"grants": len(a["grants"])},
                 {"mode": "shadow", "restriction": _tok(restriction), "why": _tok(a["why"]),
                  "channel": _tok(a["grants"][0].channel), "kind": _tok(a["grants"][0].kind),
                  "purposes": _tok(":".join(a["purposes"])), "conditions": _tok(":".join(a["conditions"])),
                  "evidence": ids})
        return a
    except Exception:  # noqa: BLE001
        return None


def shadow_evidence() -> list:
    """Owner-only, in-process: the recorded shadow evidence (verbatim line, channel, host), oldest first. Not served
    by any route, not logged, not stored."""
    with _LOCK:
        return [dict(v) for v in _EVIDENCE.values()]


def reset_for_tests() -> None:
    global _CFG
    with _LOCK:
        _EVIDENCE.clear()
        _SEEN.clear()
    _CFG = None
