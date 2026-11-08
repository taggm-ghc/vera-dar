"""robots.txt verdicts for VERA+ (RFC 9309), with AI-specific additions.

OD-1108-DISPOSITIONS-2 H13 (owner, 2026-11-08): "Hyperscaler-specific robots blocks are advisory for independently
identified VERA-DAR agents. Continue enforcing applicable wildcard, VERA-specific, AI-use, TDM, and terms
restrictions." (Supersedes OD-1106-ROBOTS-AI, under which hyperscaler groups bound VERA-DAR.)

Binding: the group(s) naming any of our own product tokens (own_tokens; RFC 9309, all five as one agent; every
matching own group must allow), else the * groups (allowed()); and a full-line comment stating that AI crawlers or
assistants are blocked (block_reason(), whole host). Advisory only: a group naming only a hyperscaler token
(advisory_agent_tokens) never denies; advisory() names the tokens whose group disallows a URL, for recording. Their
Crawl-delays still count, as polite pacing (crawl_delay()). Use signals (Content-Signal, Content-Usage, TDM) and
terms are enforced elsewhere (source_permissions, public_fetch, selective_fetch, acq_discovered).
Tokens and phrases come from config/robots_ai.json. Any error loading config or evaluating raises; callers fail closed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "robots_ai.json"
_CFG: dict | None = None
_DEFAULT_REASONS = {"statement": "robots_ai_statement"}


def load_config(path: Path | None = None) -> dict:
    """Raises on a missing or malformed file (fail closed); cached for the default path."""
    global _CFG
    path = path or CONFIG_PATH
    if path == CONFIG_PATH and _CFG is not None:
        return _CFG
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(c, dict) or not isinstance(c.get("advisory_agent_tokens"), list) \
            or not isinstance(c.get("ai_statement_phrases"), list):
        raise ValueError("robots_ai config malformed")
    ot, ut = c.get("own_tokens"), c.get("usage_tokens")  # ROBOTS-TIGHTEN-1108 S1b / A1; missing or malformed fails closed
    if not isinstance(ot, list) or not ot or not all(isinstance(t, str) and t.strip() for t in ot) \
            or not isinstance(ut, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in ut.items()):
        raise ValueError("robots_ai config malformed")
    if path == CONFIG_PATH:
        _CFG = c
    return c


def reason_code(kind: str) -> str:
    return str((load_config().get("reasons") or {}).get(kind) or _DEFAULT_REASONS[kind])


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _base(name: str) -> str:
    """Product token of a user-agent value: lower-cased, version suffix removed (GPTBot/1.1 -> gptbot)."""
    return name.strip().split("/")[0].strip().lower()


def _strip_bom(text: str) -> str:
    """R2-2: a leading UTF-8 BOM (U+FEFF) would hide the first user-agent line (fail open); drop it."""
    return (text or "").lstrip("\ufeff")


def parse_groups(text: str) -> list:
    """RFC 9309 groups of text: [{"agents": [base names], "rules": [(allow, pattern)]}]. Consecutive user-agent
    lines share one group; unknown lines are ignored."""
    groups, cur, in_agents = [], None, False
    for raw in _strip_bom(text).splitlines():
        line = raw.split("#", 1)[0].strip()
        k, sep, v = line.partition(":")
        if not sep:
            continue
        k, v = k.strip().lower(), v.strip()
        if k == "user-agent":
            if not in_agents or cur is None:
                cur = {"agents": [], "rules": []}
                groups.append(cur)
            if v:
                cur["agents"].append(_base(v))
            in_agents = True
        elif k in ("allow", "disallow"):
            in_agents = False
            if cur is not None and v:  # an empty Disallow/Allow carries no rule
                cur["rules"].append((k == "allow", v))
        else:
            in_agents = False
    return groups


def agent_groups(text: str) -> set:
    """Lower-cased base user-agent names that head a group in text."""
    return {a for g in parse_groups(text) for a in g["agents"] if a}


def _decode(s: str) -> str:
    return unquote(s)


_MAX_PATTERN = 512  # R2-1 cap: characters of a pattern (as written); a Disallow over a cap blocks (fail closed)
_MAX_STARS = 32  # R2-1 cap: '*' wildcards per pattern


def _over_cap(pattern: str) -> bool:
    return len(pattern) > _MAX_PATTERN or pattern.count("*") > _MAX_STARS


def _match(pattern: str, path: str) -> bool:
    """RFC 9309 match of pattern against path, linear time (no backtracking, R2-1): the part before the first '*' is a
    prefix, each later part is found left to right with str.find, and a trailing '$' anchors the last part at the end."""
    pat = _decode(pattern)
    end = pat.endswith("$")
    if end:
        pat = pat[:-1]
    parts = pat.split("*")
    if not path.startswith(parts[0]):
        return False
    pos = len(parts[0])
    if len(parts) == 1:
        return pos == len(path) if end else True
    for p in parts[1:-1]:
        i = path.find(p, pos)
        if i < 0:
            return False
        pos = i + len(p)
    last = parts[-1]
    if end:
        return len(path) - len(last) >= pos and path.endswith(last)
    return path.find(last, pos) >= 0


def _path_of(url: str) -> str:
    sp = urlsplit(url)
    return _decode((sp.path or "/") + (("?" + sp.query) if sp.query else ""))


def _applies(groups: list, names, own_ua: str | None) -> list:
    """Rules of every group matching: any base name in names (exact), or, for own_ua, a name equal to the UA's
    product token (case-insensitive, version stripped; RFC 9309, R2-3, no substring match); with no match, the * groups."""
    tok = _base(own_ua) if own_ua else ""
    hit = [g for g in groups if any((a in names) or (tok and a and a == tok) for a in g["agents"])]
    if not hit:
        hit = [g for g in groups if "*" in g["agents"]]
    return [r for g in hit for r in g["rules"]]


def _decide(rules: list, url: str) -> bool:
    """RFC 9309: longest matching pattern wins, Allow wins a tie, no match = allowed."""
    path = _path_of(url)
    best = (-1, True)
    for allow, pat in rules:
        if _over_cap(pat):
            if not allow:
                return False  # R2-1: an over-cap Disallow blocks; an over-cap Allow is ignored
            continue
        if _match(pat, path):
            n = len(_decode(pat))
            if n > best[0] or (n == best[0] and allow):
                best = (n, allow)
    return best[1]


def allowed(rp, ua: str, url: str) -> bool:
    """RFC 9309 verdict for our own user agent (else the * groups) on url; replaces RobotFileParser.can_fetch.
    A bare True means allow-all; anything un-annotated raises (callers fail closed)."""
    if rp is True:
        return True
    if rp is False:
        return False
    toks = {_base(str(t)) for t in load_config()["own_tokens"]} | {_base(ua)}
    toks.discard("")
    hit = [t for t in sorted(toks) if any(t in g["agents"] for g in rp.vera_groups)]
    if not hit:  # no group names any own token: the * groups
        return _decide(_applies(rp.vera_groups, (), None), url)
    return all(_decide(_applies(rp.vera_groups, {t}, None), url) for t in hit)  # S1b: every matched own group must allow


def usage_signals(rp, url: str) -> str:
    """A1: "ai-train=no" (config usage_tokens value) when a group naming a usage token exactly disallows url, else "".
    Informational only: never consulted for access, never falls back to *."""
    if rp in (None, True, False):
        return ""
    out = []
    for tok, val in load_config()["usage_tokens"].items():
        b = _base(tok)
        gs = [g for g in rp.vera_groups if b in g["agents"]]
        if gs and not _decide([r for g in gs for r in g["rules"]], url) and val not in out:
            out.append(val)
    return ", ".join(out)


def statement_in(text: str, cfg: dict | None = None) -> bool:
    """True when a FULL-LINE comment says AI crawlers/assistants are blocked (phrases from config), one line at a
    time, not preceded by a negation, and never inside a URL or after directive text."""
    cfg = cfg or load_config()
    phrases = [_norm(str(p)) for p in cfg["ai_statement_phrases"] if _norm(str(p))]
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line.startswith("#"):
            continue
        c = " " + _norm(line.lstrip("#")) + " "
        for p in phrases:
            at = c.find(f" {p} ")
            while at >= 0:
                before = c[:at].split()[-3:]
                if not any(w in _NEG for w in before):
                    return True
                at = c.find(f" {p} ", at + 1)
    return False


_NEG = {"no", "not", "never", "don", "dont", "doesn", "doesnt", "won", "cannot", "without"}


def annotate(rp, text: str):
    """Attach the parsed groups and AI facts of text to the RobotFileParser rp (returns rp). Raises on errors."""
    cfg = load_config()
    if (text or "").startswith("\ufeff"):
        text = _strip_bom(text)
        if hasattr(rp, "parse"):  # R2-2: re-parse the stdlib parser (crawl_delay) without the BOM
            rp.entries, rp.default_entry = [], None
            rp.disallow_all = rp.allow_all = False
            rp.parse(text.splitlines())
    groups = parse_groups(text)
    names = {a for g in groups for a in g["agents"]}
    rp.vera_groups = groups
    rp.vera_ai_statement = statement_in(text, cfg)
    rp.vera_advisory_tokens = tuple(t for t in cfg["advisory_agent_tokens"] if _base(str(t)) in names)
    rp.vera_purpose_grants = _purpose_grants(text)  # PURPOSE-GRANTS-1108: shadow only, never consulted for access
    rp.vera_own_delay = _own_delay(text, cfg)
    return rp


def _own_delay(text: str, cfg: dict):
    """ROBOTS-FOLLOWUP-1108: the largest Crawl-delay among groups naming ANY own token (inf = malformed), else None."""
    import math
    own = {_base(str(t)) for t in cfg["own_tokens"]}
    out, mine, in_agents = [], False, False
    for raw in _strip_bom(text).splitlines():
        k, sep, v = raw.split("#", 1)[0].strip().partition(":")
        if not sep:
            continue
        k, v = k.strip().lower(), v.strip()
        if k == "user-agent":
            if not in_agents:
                mine = False
            mine = mine or _base(v) in own
            in_agents = True
        else:
            in_agents = False
            if k == "crawl-delay" and mine:
                try:
                    d = float(v)
                except ValueError:
                    d = math.inf
                out.append(d if math.isfinite(d) and d >= 0 else math.inf)
    return max(out) if out else None


def _purpose_grants(text: str) -> tuple:
    try:
        from vera import purpose_grants
        return purpose_grants.scan_robots(text) if purpose_grants.mode() == "shadow" else ()
    except Exception:  # noqa: BLE001  shadow evidence is optional; access is decided without it
        return ()


def _shadow(kind: str, rp, url: str) -> None:
    """PURPOSE-GRANTS-1108: record what a robots.txt purpose grant WOULD do to this block. The block stands."""
    try:
        g = getattr(rp, "vera_purpose_grants", ())
        if g:
            from vera import purpose_grants
            purpose_grants.shadow(_DEFAULT_REASONS[kind], g, url)
    except Exception:  # noqa: BLE001
        pass


def block_reason(rp, url: str) -> str | None:
    """None when no AI statement blocks url; else robots_ai_statement. Hyperscaler-only groups never block (H13; see
    advisory()). A bare True/False marker gives None; an un-annotated parser raises (callers fail closed)."""
    if rp in (None, True, False):
        return None
    if rp.vera_ai_statement:
        _shadow("statement", rp, url)
        return reason_code("statement")
    return None


def advisory(rp, url: str) -> tuple:
    """H13: the advisory (hyperscaler) tokens whose own group disallows url, for recording only. NEVER used to deny."""
    if rp in (None, True, False):
        return ()
    return tuple(t for t in getattr(rp, "vera_advisory_tokens", ())
                 if not _decide(_applies(rp.vera_groups, {_base(t)}, None), url))


def crawl_delay(rp, ua: str) -> float | None:
    """Largest Crawl-delay among the groups that apply: this UA (else *) and the advisory (hyperscaler) groups of the
    file, kept as polite pacing (H13). None when none declares one."""
    if rp in (None, True, False):
        return None
    ds = [rp.crawl_delay(ua), getattr(rp, "vera_own_delay", None)] \
        + [rp.crawl_delay(t) for t in getattr(rp, "vera_advisory_tokens", ())]
    ds = [float(d) for d in ds if d is not None]
    return max(ds) if ds else None
