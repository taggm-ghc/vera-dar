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
    global _POLICY
    with _LOCK:
        _POLICY = policy


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
    return (sp.hostname or "").lower().rstrip("."), normalise_path(sp.path)


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


def gate_api_urls(fn: Callable) -> Callable:
    """Wrap a requests.get-compatible fetcher used for websites AND APIs (auto_acquire's hop transport): a URL that is
    an API call (is_api_url) must pass require() and gets the entry's User-Agent; other URLs pass unchanged (they
    stay under the robots/TDM gate). The wrapper is marked api_gated so callers can pre-check before robots.txt."""
    def wrapped(url, *a, api_purpose: str = "fulltext", **kw):
        # r2 (3): the caller's purpose ("metadata" for JSON lookups, "fulltext" for documents) is checked against the
        # entry's content_scope here too; a caller that names none gets the stricter "fulltext".
        if is_api_url(url):
            d = require(url, api_purpose)
            kw["headers"] = headers_for(d, kw.get("headers"))
        return fn(url, *a, **kw)
    wrapped.__wrapped__ = fn
    wrapped.api_gated = True
    return wrapped


def review_due(today: str, policy: Policy | None = None) -> list[str]:
    """Entry ids whose review_by date (ISO) is on or before today. A warning, never a denial (owner rule:
    corrections are not blocked by cutoffs)."""
    policy = policy or get_policy()
    return sorted(eid for eid, e in policy.entries.items() if str(e.get("review_by")) <= today)
