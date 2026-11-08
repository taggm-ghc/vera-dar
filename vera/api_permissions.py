"""API-ALLOW-1107 (owner OD-1107-HITL, 2026-11-07): deny-by-default permission gate for outbound API calls.

Website robots.txt does not by itself settle API permission. Every API request is checked against
config/api_permissions.json (host + longest path prefix) BEFORE anything is sent:

- no matching entry, an unresolved or held verdict, or an unreadable/invalid file -> ApiDenied, code R-API-DENIED;
- a permitted entry whose conditions do not validate against config/rate_limits.json (missing portal, portal not
  covering the host, rate or concurrency above the documented ceiling, 429 not treated as a throttle, an email
  requirement) -> ApiDenied, code R-API-COND;
- a purpose outside the entry's content_scope (e.g. full text from a metadata-only API) -> ApiDenied, R-API-SCOPE.

A permitted call is then sent through the entry's rate-limit portal (vera.rate_limit) with the entry's User-Agent.
The owner flips an entry with owner_switch ("permit" needs owner_decision) and no code change.

Logs carry only the fixed code and the entry id (never the URL, query or response)."""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urljoin, urlsplit

logger = logging.getLogger("vera")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "api_permissions.json"
DENIED, COND, SCOPE = "R-API-DENIED", "R-API-COND", "R-API-SCOPE"
PERMITTED_VERDICTS = frozenset({"PERMITTED", "PERMITTED-WITH-CONDITIONS"})
_SWITCHES = frozenset({"default", "permit", "deny"})
PURPOSES = frozenset({"metadata", "fulltext"})
# r2 (2): an owner permit must name a decision ID of the form OD-<4 digits>-... The runtime cannot check that the ID
# exists in records/decisions.md (records/ is not in the public repository) and a hash pin of the decision text would
# sit in the same file as the switch, so whoever can flip the switch can also re-pin the hash: it adds no trust. The
# trust root is therefore the reviewed commit of config/api_permissions.json; the pattern only stops a blank or
# free-text permit. A reviewer checks the ID against records/decisions.md before the commit.
OWNER_DECISION_RE = re.compile(r"^OD-\d{4}-[A-Z0-9][A-Z0-9-]*$")  # r3: full match (fullmatch below)
MAX_API_REDIRECTS = 3
_REDIRECTS = (301, 302, 303, 307, 308)


class ApiDenied(RuntimeError):
    """Raised BEFORE any request: the API call is not permitted. code is R-API-DENIED, R-API-COND or R-API-SCOPE."""

    def __init__(self, code: str, entry: str | None, why: str):
        super().__init__(f"{code} ({entry or 'no_entry'}: {why})")
        self.code, self.entry, self.why = code, entry, why


@dataclass(frozen=True)
class Decision:
    allowed: bool
    code: str | None
    entry: str | None
    why: str
    conditions: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class Policy:
    entries: Mapping            # id -> entry dict
    api_hosts: frozenset
    error: str | None = None    # set when the file was unreadable/invalid: everything is denied


def _validate(raw: Any) -> Policy:
    if not isinstance(raw, dict) or raw.get("default") != "deny" or not isinstance(raw.get("entries"), dict):
        raise ValueError("need default 'deny' and an 'entries' object")
    hosts = raw.get("api_hosts") or []
    if not isinstance(hosts, list) or not all(isinstance(h, str) and h == h.lower() and h for h in hosts):
        raise ValueError("api_hosts: lowercase host names")
    entries = {}
    for eid, e in raw["entries"].items():
        if eid.startswith("_"):
            continue
        if not isinstance(e, dict):
            raise ValueError(f"entries.{eid}: object")
        h, p = e.get("host"), e.get("path_prefix")
        if not (isinstance(h, str) and h and h == h.lower() and isinstance(p, str) and p.startswith("/")):
            raise ValueError(f"entries.{eid}: host (lowercase) and path_prefix (starts with /)")
        if e.get("owner_switch", "default") not in _SWITCHES:
            raise ValueError(f"entries.{eid}.owner_switch: default, permit or deny")
        for k in ("verdict", "read_level", "read", "review_by"):
            if not isinstance(e.get(k), str) or not e[k]:
                raise ValueError(f"entries.{eid}.{k}: required")
        if not isinstance(e.get("evidence"), list) or not e["evidence"]:
            raise ValueError(f"entries.{eid}.evidence: at least one evidence record")
        if not isinstance(e.get("conditions"), dict):
            raise ValueError(f"entries.{eid}.conditions: object")
        entries[eid] = e
    return Policy(entries=entries, api_hosts=frozenset(hosts))


def load_policy(path: Path | str = CONFIG_PATH) -> Policy:
    """The policy; an unreadable or invalid file yields a policy that denies everything (fail closed)."""
    try:
        return _validate(json.loads(Path(path).read_text(encoding="utf-8")))
    except Exception as e:  # noqa: BLE001  fail closed
        logger.error("api_permissions config unusable (%s): every API call denied", type(e).__name__)
        return Policy(entries={}, api_hosts=frozenset(), error=type(e).__name__)


_POLICY: Policy | None = None
_LOCK = threading.Lock()


def get_policy() -> Policy:
    global _POLICY
    with _LOCK:
        if _POLICY is None:
            _POLICY = load_policy()
        return _POLICY


def set_policy_for_tests(policy: Policy | None) -> None:
    global _POLICY, _LIKE
    with _LOCK:
        _POLICY = policy
        if policy is None:  # 1108b: a reset also re-reads config/api_like_patterns.json
            _LIKE = None


def normalise_path(path: str) -> str:
    """The path as a server would resolve it, for prefix matching (r2 (4)): percent-decoded until stable (bounded, so
    %2561 -> %61 -> a), ';' parameters cut, dot-segments resolved (RFC 3986 5.2.4: "/x/../api/" -> "/api/"), empty
    segments dropped (//), lower-cased, and ending in "/" (so "/api" matches the prefix "/api/" and "/apix" does not).
    Prefixes in the config are lowercase. A "..", ever, cannot climb above the root."""
    p = str(path or "/")
    for _ in range(4):
        q = unquote(p)
        if q == p:
            break
        p = q
    out: list[str] = []
    for seg in p.replace("\\", "/").split("/"):
        seg = seg.split(";", 1)[0]
        if seg in ("", "."):
            continue
        if seg == "..":
            if out:
                out.pop()
            continue
        out.append(seg)
    return ("/" + "/".join(out) + "/").replace("//", "/").lower()


def _split(url: str) -> tuple[str, str]:
    sp = urlsplit(str(url or ""))
    # r3: a fully-qualified host ("api.openalex.org.") is the same host: strip trailing dots before matching
    host = (sp.hostname or "").lower()
    if not host.isascii():  # API-URL-GATE-1108 review (3): IDNA/NFKC mapping, so a look-alike or full-width spelling
        try:                # of an API host (U+FF41, U+3002 dot) is matched as that host
            host = host.encode("idna").decode("ascii").lower()
        except UnicodeError:  # 1108b review (5): an unmappable host cannot be classed; gate_url refuses it
            raise ValueError("host not IDNA-encodable") from None
    return host.rstrip("."), normalise_path(sp.path)


def match(url: str, policy: Policy | None = None) -> tuple[str, Mapping] | None:
    """(entry id, entry) with the same host and the longest matching path prefix, else None."""
    policy = policy or get_policy()
    host, path = _split(url)
    best = None
    for eid, e in policy.entries.items():
        if e["host"] == host and path.startswith(e["path_prefix"].lower()):
            if best is None or len(e["path_prefix"]) > len(best[1]["path_prefix"]):
                best = (eid, e)
    return best


def is_api_url(url: str, policy: Policy | None = None) -> bool:
    """True when the URL is an API call: its host is an API-only host or it falls under an entry's path prefix."""
    policy = policy or get_policy()
    host, _ = _split(url)
    return host in policy.api_hosts or match(url, policy) is not None


def _conditions_ok(eid: str, e: Mapping, host: str) -> str | None:
    """None when the entry's conditions are enforceable from config/rate_limits.json; else why not."""
    c = e.get("conditions") or {}
    if c.get("send_email"):
        return "email_required"  # the project sends no email anywhere
    portal = c.get("rate_portal")
    if not portal:
        return "no_rate_portal"
    try:
        from vera import rate_limit
        ad = rate_limit.get_registry().portal(str(portal))
    except Exception:  # noqa: BLE001
        return "rate_portal_missing"
    pc = ad.c
    if host not in pc.get("hosts", ()):
        return "rate_portal_host"
    mx = c.get("max_rate_per_s")
    if mx is None or float(pc["rate_per_s"]) > float(mx):
        return "rate_above_ceiling"
    mc = c.get("max_concurrency")
    if mc is not None and int(pc["max_concurrency"]) > int(mc):
        return "concurrency_above_ceiling"
    if c.get("respect_429") and 429 not in tuple(pc.get("throttle_statuses") or ()):
        return "429_not_throttle"
    return None


def decide(url: str, purpose: str | None = None, policy: Policy | None = None) -> Decision:
    policy = policy or get_policy()
    if policy.error:
        return Decision(False, DENIED, None, "config_unusable")
    m = match(url, policy)
    if m is None:
        return Decision(False, DENIED, None, "no_entry")
    eid, e = m
    sw = e.get("owner_switch", "default")
    if sw == "deny":
        return Decision(False, DENIED, eid, "owner_deny")
    if sw == "permit":
        if not OWNER_DECISION_RE.fullmatch(str(e.get("owner_decision") or "")):
            return Decision(False, DENIED, eid, "permit_without_decision")
    elif e["verdict"] not in PERMITTED_VERDICTS:
        return Decision(False, DENIED, eid, "verdict_" + e["verdict"].lower())
    host, _ = _split(url)
    bad = _conditions_ok(eid, e, host)
    if bad:
        return Decision(False, COND, eid, bad)
    scope = tuple((e.get("conditions") or {}).get("content_scope") or ())
    if purpose is not None and purpose not in scope:
        return Decision(False, SCOPE, eid, "scope_" + str(purpose))
    return Decision(True, None, eid, "permitted", e.get("conditions") or {})


def attribution_for(url: str, policy: Policy | None = None) -> dict | None:
    """OA5-ALLOW-1108: the attribution/licence fields the matching entry's conditions require, read from
    config/api_permissions.json only (no strings in code). None when no entry matches, the call is not permitted,
    or the entry names no attribution. Keys: attribution, placement (or None), licence (or None), no_branding."""
    d = decide(url, None, policy)
    if not d.allowed:
        return None
    c = d.conditions or {}
    text = c.get("attribution")
    if not isinstance(text, str) or not text:
        return None
    return {"attribution": text, "placement": c.get("attribution_placement"), "licence": c.get("licence"),
            "no_branding": bool(c.get("no_branding"))}


def require(url: str, purpose: str | None = None, policy: Policy | None = None) -> Decision:
    """decide(); a refusal is logged (fixed code + entry id) and raised as ApiDenied. Nothing is sent."""
    d = decide(url, purpose, policy)
    if not d.allowed:
        logger.warning("%s entry=%s why=%s", d.code, d.entry or "none", d.why)
        raise ApiDenied(d.code, d.entry, d.why)
    return d


def headers_for(d: Decision, headers: Mapping | None) -> dict:
    """The request headers with the entry's required User-Agent (conditions.user_agent), when it names one."""
    out = dict(headers or {})
    ua = d.conditions.get("user_agent")
    if ua:
        out = {k: v for k, v in out.items() if str(k).lower() != "user-agent"}
        out["User-Agent"] = str(ua)
    return out


def call(fn: Callable, url: str, *a, headers: Mapping | None = None, purpose: str | None = "metadata", **kw):
    """THE gated API call: require(), then fn(url, *a, headers=..., allow_redirects=False, **kw) through the entry's
    rate-limit portal. Redirects are never followed by fn (r2 (1)): a 3xx is followed here only when the caller did not
    pass allow_redirects=False, at most MAX_API_REDIRECTS hops, and EVERY hop is gated again (its own entry, purpose,
    User-Agent and rate portal). A redirect to a host or path with no permitted entry (a denied API, or a website,
    which this path does not robots-check) raises ApiDenied with nothing sent to the target.
    Raises ApiDenied (nothing sent) or vera.rate_limit.RateLimited (skipped, nothing sent)."""
    from vera import rate_limit
    follow = kw.pop("allow_redirects", True) is not False
    current, hops = url, 0
    while True:
        d = require(current, purpose)
        ad = rate_limit.get_registry().portal(str(d.conditions["rate_portal"]))
        resp = rate_limit.call_with_limit(ad, fn, current, *a, headers=headers_for(d, headers),
                                          allow_redirects=False, **kw)
        loc = (getattr(resp, "headers", None) or {}).get("Location")
        if not (follow and getattr(resp, "status_code", None) in _REDIRECTS and loc):
            return resp
        hops += 1
        close = getattr(resp, "close", None)
        if callable(close):
            close()
        if hops > MAX_API_REDIRECTS:
            logger.warning("%s entry=%s why=too_many_redirects", DENIED, d.entry)
            raise ApiDenied(DENIED, d.entry, "too_many_redirects")
        current = urljoin(current, str(loc))
        kw.pop("params", None)  # already part of the first URL; the Location carries its own query


# API-URL-GATE-1108: the purpose a generic fetch hop is checked against. Callers pass it explicitly (api_purpose=);
# a generic fetch reads a document, so a hop that names none is checked as "fulltext". 1108b (6): the contextvar is
# only a guarded fallback: gate_api_urls binds (url, purpose) for the duration of its call, and the binding applies
# to THAT url only, so it can never widen any other hop (a redirect target, robots.txt) beyond "fulltext".
_PURPOSE: contextvars.ContextVar = contextvars.ContextVar("vera_api_purpose", default=None)
# 1108b (4): the rate portal already held for a permitted API hop (set by limited_transport while it holds the entry's
# portal), so the transport underneath (pinned_get) does not take the same portal twice.
_HELD: contextvars.ContextVar = contextvars.ContextVar("vera_api_portal_held", default=None)


def current_purpose(url: str, default: str = "fulltext") -> str:
    bound = _PURPOSE.get()
    if bound and bound[0] == url:
        return bound[1]
    return default


@contextlib.contextmanager
def purpose_scope(purpose: str, url: str):
    """Bind purpose for exactly this url (never a wildcard)."""
    tok = _PURPOSE.set((str(url), purpose))
    try:
        yield
    finally:
        _PURPOSE.reset(tok)


def held_portal() -> str | None:
    return _HELD.get()


@contextlib.contextmanager
def holding(portal: str):
    tok = _HELD.set(portal)
    try:
        yield
    finally:
        _HELD.reset(tok)


# ------------------------------------------------------------------ 1108b (1)(2): unlisted API-like URLs, IP literals
LIKE_PATH = Path(__file__).resolve().parent.parent / "config" / "api_like_patterns.json"


@dataclass(frozen=True)
class LikePatterns:
    host_prefixes: Mapping
    path_patterns: Mapping      # id -> compiled regex
    query_markers: Mapping      # id -> compiled regex
    ip_literal_allow: frozenset
    error: str | None = None    # unreadable/invalid file: every generic fetch is refused


def load_like(path: Path | str = LIKE_PATH) -> LikePatterns:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        hp, pp, qm = raw["host_prefixes"], raw["path_patterns"], raw["query_markers"]
        ipa = raw.get("ip_literal_allow", [])
        if not all(isinstance(x, dict) and x for x in (hp, pp, qm)) or not isinstance(ipa, list):
            raise ValueError("host_prefixes, path_patterns, query_markers: non-empty objects; ip_literal_allow: list")
        if not all(isinstance(v, str) and v and v == v.lower() for v in hp.values()):
            raise ValueError("host_prefixes: lowercase strings")
        return LikePatterns(dict(hp), {k: re.compile(v) for k, v in pp.items()},
                            {k: re.compile(v) for k, v in qm.items()}, frozenset(str(x).lower() for x in ipa))
    except Exception as e:  # noqa: BLE001  fail closed
        logger.error("api_like_patterns config unusable (%s): every generic fetch refused", type(e).__name__)
        return LikePatterns({}, {}, {}, frozenset(), error=type(e).__name__)


_LIKE: LikePatterns | None = None


def get_like() -> LikePatterns:
    global _LIKE
    with _LOCK:
        if _LIKE is None:
            _LIKE = load_like()
        return _LIKE


def set_like_for_tests(like: LikePatterns | None) -> None:
    global _LIKE
    with _LOCK:
        _LIKE = like


_IP_LIKE = re.compile(r"(0x[0-9a-f]+|[0-9]+)(\.(0x[0-9a-f]*|[0-9]+)){0,3}\.?")


def is_ip_literal(host: str) -> bool:
    """True for an IPv4/IPv6 literal and for the numeric forms a resolver also reads as an address (127.1, 2130706433,
    0x7f.0.0.1)."""
    import ipaddress
    h = str(host or "").strip("[]").lower()
    try:
        ipaddress.ip_address(h.split("%", 1)[0])
        return True
    except ValueError:
        return bool(h) and _IP_LIKE.fullmatch(h) is not None


def api_like(url: str, like: LikePatterns | None = None) -> str | None:
    """The id of the first pattern that classes the URL as an API call (host prefix, path, query marker), else None."""
    like = like or get_like()
    host, path = _split(url)
    for pid, pre in like.host_prefixes.items():
        if host.startswith(pre):
            return pid
    for pid, rx in like.path_patterns.items():
        if rx.search(path):
            return pid
    q = normalise_query(urlsplit(str(url or "")).query)
    for pid, rx in like.query_markers.items():
        if q and rx.search(q):
            return pid
    return None


def normalise_query(query: str) -> str:
    """1108b review (1): the query as a server reads it, for marker matching: split on & and ;, each key and value
    percent-decoded (+ as space) until stable (bounded), lower-cased, re-joined with &."""
    from urllib.parse import unquote_plus
    out = []
    for part in re.split(r"[&;]", str(query or "")):
        if not part:
            continue
        k, _, v = part.partition("=")
        for _ in range(4):
            k2, v2 = unquote_plus(k), unquote_plus(v)
            if (k2, v2) == (k, v):
                break
            k, v = k2, v2
        out.append(f"{k.strip().lower()}={v.strip().lower()}")
    return "&".join(out)


def _refuse(why: str, entry: str | None = None):
    logger.warning("%s entry=%s why=%s", DENIED, entry or "none", why)
    raise ApiDenied(DENIED, entry, why)


def gate_url(url: str, purpose: str | None = None, policy: Policy | None = None,
             like: LikePatterns | None = None) -> Decision | None:
    """API-URL-GATE-1108: the check for EVERY generic outbound fetch hop (a URL that may come from search results,
    portal links or a redirect). None: not an API URL (robots/read gates apply as before). A Decision: an API URL
    whose entry permits this purpose. Otherwise ApiDenied (logged with its fixed code and why), nothing sent:
    - config_unusable: config/api_permissions.json or config/api_like_patterns.json unusable (EVERY URL refused);
    - unparseable_url;
    - ip_literal: an IP-literal host not in ip_literal_allow (1108b (2));
    - an entry's refusal (owner_deny, verdict_*, scope_*, ...): the URL matches an entry, which alone decides;
    - unlisted_api_like:<pattern id>: no entry, but the URL looks like an API call (1108b (1)).
    purpose: the caller's explicit purpose; None falls back to current_purpose(url) (default "fulltext")."""
    policy = policy or get_policy()
    like = like or get_like()
    if policy.error or like.error:
        _refuse("config_unusable")
    try:
        host, _ = _split(url)
        api = is_api_url(url, policy)
    except ValueError:  # an unparseable URL (e.g. invalid IPv6 literal) cannot be classed: refused
        _refuse("unparseable_url")
    if is_ip_literal(host) and host not in like.ip_literal_allow:
        _refuse("ip_literal")
    if api:
        return require(url, purpose or current_purpose(url), policy)
    pid = api_like(url, like)
    if pid:
        _refuse("unlisted_api_like:" + pid)
    return None


def accepts_purpose(fn: Callable) -> bool:
    """True for a transport that declares an api_purpose parameter by name (pinned_get, gate_api_urls,
    limited_transport); a **kwargs stub does not count, so it is never handed a keyword it does not expect."""
    if getattr(fn, "accepts_api_purpose", False):
        return True
    import inspect
    try:
        p = inspect.signature(fn).parameters.get("api_purpose")
    except (TypeError, ValueError):
        return False
    return p is not None and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)


def gate_api_urls(fn: Callable) -> Callable:
    """Wrap a requests.get-compatible fetcher used for websites AND APIs (auto_acquire's hop transport): a URL that is
    an API call (is_api_url) must pass require() and gets the entry's User-Agent; other URLs pass unchanged (they
    stay under the robots/TDM gate). The wrapper is marked api_gated so callers can pre-check before robots.txt.
    API-URL-GATE-1108: fails closed (gate_url). The purpose is passed on explicitly to a transport that takes it, and
    otherwise bound to this url only (guarded fallback, current_purpose)."""
    def wrapped(url, *a, api_purpose: str = "fulltext", **kw):
        # r2 (3): the caller's purpose ("metadata" for JSON lookups, "fulltext" for documents) is checked against the
        # entry's content_scope here too; a caller that names none gets the stricter "fulltext".
        d = gate_url(url, api_purpose)
        if d is not None:
            kw["headers"] = headers_for(d, kw.get("headers"))
        if accepts_purpose(fn):
            return fn(url, *a, api_purpose=api_purpose, **kw)
        with purpose_scope(api_purpose, url):
            return fn(url, *a, **kw)
    wrapped.__wrapped__ = fn
    wrapped.api_gated = True
    wrapped.accepts_api_purpose = True
    return wrapped


def limited_transport(fn: Callable, fixed_portal: str | None = None, *, strict: bool = False) -> Callable:
    """1108b (4): rate-limit a requests.get-compatible transport. A permitted API hop always goes through ITS entry's
    portal (never fixed_portal, never the host's portal) and the portal is marked held, so a gated transport below does
    not take it twice. Any other URL: fixed_portal when given, else the host's portal (rate_limit.limited_by_url
    semantics), else unlimited; with strict=True (OA-1, acquisition) a URL in no portal fails CLOSED with
    rate_limit.UnlistedHost, nothing sent. A refused API URL raises ApiDenied, nothing sent."""
    def wrapped(url, *a, api_purpose: str | None = None, **kw):
        from vera import rate_limit
        reg = rate_limit.get_registry()
        if accepts_purpose(fn) and api_purpose is not None:
            kw["api_purpose"] = api_purpose
        d = gate_url(url, api_purpose)
        if d is not None:
            name = str(d.conditions["rate_portal"])
            with holding(name):
                return rate_limit.call_with_limit(reg.portal(name), fn, url, *a, **kw)
        ad = reg.portal(fixed_portal) if fixed_portal else reg.portal_for_url(url)
        if ad is None:
            if strict:
                raise rate_limit.UnlistedHost((urlsplit(str(url)).hostname or "").lower())
            return fn(url, *a, **kw)
        return rate_limit.call_with_limit(ad, fn, url, *a, **kw)
    wrapped.__wrapped__ = fn
    wrapped.accepts_api_purpose = True
    wrapped.rate_portal = fixed_portal  # as rate_limit.limited marks its portal (None: per-host)
    return wrapped


def review_due(today: str, policy: Policy | None = None) -> list[str]:
    """Entry ids whose review_by date (ISO) is on or before today. A warning, never a denial (owner rule:
    corrections are not blocked by cutoffs)."""
    policy = policy or get_policy()
    return sorted(eid for eid, e in policy.entries.items() if str(e.get("review_by")) <= today)
