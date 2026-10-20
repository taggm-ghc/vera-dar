"""Endpoint classification (PLAN-ENV v2.1 §2; stdlib only; guard file).

Two classes only (C1, OR-1): every result is PRODUCTION. DISPOSABLE_VERIFIED is defined for interface stability but is
UNREACHABLE: by binding owner decision (2026-10-19) `verify_disposable` refuses unconditionally with VERIFY_DISABLED,
and no flag, env var, argument or config path enables it. Enabling needs a later, independently reviewed isolation
design. The marker/HMAC helpers below (`_inert_*`) are inert: nothing in classification reads a marker file or nonce.

Hygiene: no host, port, user, dbname, password, IP, nonce or hash ever appears in a message, repr or exception. Hashes
are salted per process and held in memory only.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import unquote

PRODUCTION = "PRODUCTION"
DISPOSABLE_VERIFIED = "DISPOSABLE_VERIFIED"   # unreachable (owner decision 2026-10-19)
VERIFY_DISABLED = "DISPOSABLE-VERIFY-DISABLED-pending-isolation-review"

INDICATOR_REASONS = ("fp", "suffix", "host_pattern", "dbname", "role", "ip")
PARSE_REASONS = ("unparsable", "bad_scheme", "bad_param", "percent", "empty_host", "multihost")
OTHER_REASONS = ("inconclusive", "no_sources", "pg_env", "no_marker", "tag_mismatch", "negative_check",
                 "not_read_only", "probe_error", VERIFY_DISABLED)
REASONS = frozenset(INDICATOR_REASONS + PARSE_REASONS + OTHER_REASONS)

URL_NAMES = ("EXTERNAL_DB_URL", "INTERNAL_DB_URL", "DATABASE_URL", "VERA_DB_URL_RO", "VERA_DB_URL_RW",
             "VERA_DB_URL_WO", "VERA_DB_URL_ADMIN")
INTERN_PARAM_NAMES = ("DB_HOST", "DB_HOST_EXTERNAL_SUFFIX", "DB_PORT", "DB_NAME", "DB_ACCOUNT", "DB_ADMIN_ROLE")
AMBIENT_PREFIXES = ("PG", "KRB5", "GSS")

_SCHEMES = {"postgresql": "postgresql", "postgres": "postgresql", "postgresql+psycopg2": "postgresql"}
_ALLOWED_PARAMS = ("sslmode", "connect_timeout", "application_name")
SSLMODES = ("disable", "allow", "prefer", "require", "verify-ca", "verify-full")   # weakest -> strongest
_HOST_RE = re.compile(r"[a-z0-9.-]+")
_USER_RE = re.compile(r"[A-Za-z0-9_.-]+")
_DB_RE = re.compile(r"[A-Za-z0-9_]+")
_PORT_RE = re.compile(r"[0-9]{1,5}")
_APPNAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,63}")
_IPV4_RE = re.compile(r"[0-9]{1,3}(\.[0-9]{1,3}){3}")
HOST_FLOOR_RE = re.compile(r"(?i)(\.render\.com$|^dpg-)")   # PH code floor, applied after trailing-dot stripping

# One registry per process, shared by every copy of the safety modules (package import and by-path private loads),
# so salts, defaults and exception classes agree across copies.
import sys as _sys
import types as _types
_SHARED = _sys.modules.setdefault("_vera_safety_shared_state", _types.ModuleType("_vera_safety_shared_state"))
_SALT = _SHARED.__dict__.setdefault("ec_salt", secrets.token_bytes(32))


def _h(value: str) -> str:
    return hmac.new(_SALT, value.encode("utf-8", "surrogatepass"), hashlib.sha256).hexdigest()


class ParseError(Exception):
    """A strict-parse refusal. Carries a code only."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code

    def __repr__(self) -> str:
        return f"ParseError({self.code!r})"


ParseError = _SHARED.__dict__.setdefault("ParseError", ParseError)


@dataclass(frozen=True)
class Verdict:
    cls: str
    reasons: tuple = ()
    name: str = ""

    def as_dict(self) -> dict:
        return {"cls": self.cls, "reasons": list(self.reasons), "name": self.name}


def _verdict(reasons: Iterable[str], name: str = "") -> Verdict:
    # Only PRODUCTION is ever constructed (C1 plus the owner decision).
    return Verdict(PRODUCTION, tuple(sorted(set(reasons))), name)


class ParsedURL:
    """Parsed endpoint. Values live in memory only; str/repr never show them."""
    __slots__ = ("scheme", "host", "port", "dbname", "user", "password", "params")

    def __init__(self, scheme, host, port, dbname, user, password, params):
        self.scheme, self.host, self.port, self.dbname = scheme, host, port, dbname
        self.user, self.password, self.params = user, password, params

    def __repr__(self) -> str:
        return "ParsedURL(<redacted>)"

    __str__ = __repr__

    def __reduce__(self):   # never pickled or copied into another channel
        raise TypeError("ParsedURL is not serializable")


def parse(url) -> ParsedURL:
    """Strict C4/V2 parse. Raises ParseError(code); the code is the only payload."""
    if not isinstance(url, str) or not url or any(ord(c) < 33 or ord(c) == 127 for c in url):
        raise ParseError("unparsable")
    scheme, sep, rest = url.partition("://")
    if not sep or scheme.lower() not in _SCHEMES:
        raise ParseError("bad_scheme")
    if "#" in rest:
        raise ParseError("unparsable")
    netloc, _slash, pathq = rest.partition("/")
    path, _q, query = pathq.partition("?")
    if "?" in netloc:                      # no path before the query: libpq would default the dbname
        raise ParseError("unparsable")
    params = _parse_query(query)
    if "@" not in netloc:
        if "%" in netloc:
            raise ParseError("percent")
        raise ParseError("empty_host" if not netloc else "unparsable")
    userinfo, _at, hostport = netloc.rpartition("@")
    user, _c, pw = userinfo.partition(":")
    if "%" in user or "%" in hostport or "%" in path:
        raise ParseError("percent")
    if "," in hostport:
        raise ParseError("multihost")
    if "[" in hostport or "]" in hostport:
        raise ParseError("unparsable")
    host, _c2, port = hostport.partition(":")
    if not host:
        raise ParseError("empty_host")
    host = host.lower()
    if host.endswith("."):
        host = host[:-1]
    if not host or _HOST_RE.fullmatch(host) is None or ":" in port:
        raise ParseError("empty_host" if not host else "unparsable")
    if port == "":
        port_n = 5432
    elif _PORT_RE.fullmatch(port) is None or not 0 < int(port) <= 65535:
        raise ParseError("unparsable")
    else:
        port_n = int(port)
    if _USER_RE.fullmatch(user or "") is None or _DB_RE.fullmatch(path or "") is None:
        raise ParseError("unparsable")
    password = unquote(pw) if pw else ""      # V2(a): decoded once, never inspected further
    return ParsedURL(_SCHEMES[scheme.lower()], host, port_n, path, user, password, params)


def _parse_query(query: str) -> dict:
    params: dict = {}
    if not query:
        return params
    for part in query.split("&"):
        k, eq, v = part.partition("=")
        if not eq or k not in _ALLOWED_PARAMS or k in params:
            raise ParseError("bad_param")
        if k == "sslmode" and v not in SSLMODES:
            raise ParseError("bad_param")
        if k == "connect_timeout" and (_PORT_RE.fullmatch(v) is None):
            raise ParseError("bad_param")
        if k == "application_name" and _APPNAME_RE.fullmatch(v) is None:
            raise ParseError("bad_param")
        params[k] = int(v) if k == "connect_timeout" else v
    return params


def ambient_names(environ: Mapping | None = None) -> list:
    env = os.environ if environ is None else environ
    return sorted(n for n in env if n.upper().startswith(AMBIENT_PREFIXES))


def _read_names_values(path: Path | None, wanted: tuple) -> dict:
    """Tiny KEY=VALUE reader for fingerprinting. Values are returned to the caller, which hashes and drops them."""
    out: dict = {}
    if path is None:
        return out
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        if s.startswith("export ") or s.startswith("export\t"):
            s = s[7:].lstrip()
        k, _, v = s.partition("=")
        k = k.strip()
        if k in wanted:
            out[k] = v.strip().strip("'\"")
    return out


class Sources:
    """Fingerprint sets PF, PD, PR, PH (exact hosts and suffixes) and PI, as salted hashes only."""

    def __init__(self):
        self._pf: set = set()
        self._pd: set = set()
        self._pr: set = set()
        self._ph: set = set()
        self._suffix: set = set()      # (length, hash) pairs
        self._pi: set = set()
        self._n_sources = 0

    def __repr__(self) -> str:
        return (f"Sources(sources={self._n_sources}, pf={len(self._pf)}, pd={len(self._pd)}, pr={len(self._pr)}, "
                f"ph={len(self._ph)}, suffix={len(self._suffix)}, pi={len(self._pi)})")

    @classmethod
    def empty(cls) -> "Sources":
        return cls()

    @property
    def has_sources(self) -> bool:
        return self._n_sources > 0

    def _add_endpoint(self, host: str, port, db: str, user: str, resolver=None) -> None:
        host = (host or "").lower()
        if host.endswith("."):
            host = host[:-1]
        if host:
            self._ph.add(_h(host))
            if db:
                self._pf.add(_h(f"{host}|{port}|{db}"))
            if resolver is not None:
                self._resolve(host, resolver)
        if db:
            self._pd.add(_h(db))
        if user:
            self._pr.add(_h(user))

    def _resolve(self, host: str, resolver) -> None:
        try:
            for ip in resolver(host) or ():
                self._pi.add(_h(str(ip)))
        except Exception:      # resolution failure adds nothing; indicators only ever add reasons
            pass

    def add_resolved_ips(self, ips: Iterable[str]) -> None:
        for ip in ips:
            self._pi.add(_h(str(ip)))

    def _add_mapping(self, values: Mapping, resolver=None) -> bool:
        used = False
        for name in URL_NAMES:
            v = values.get(name)
            if not v:
                continue
            used = True
            try:
                p = parse(v)
            except ParseError:
                continue
            self._add_endpoint(p.host, p.port, p.dbname, p.user, resolver)
        host = values.get("DB_HOST") or ""
        suffix = values.get("DB_HOST_EXTERNAL_SUFFIX") or ""
        port = values.get("DB_PORT") or "5432"
        db = values.get("DB_NAME") or ""
        for role_name in ("DB_ACCOUNT", "DB_ADMIN_ROLE"):
            if values.get(role_name):
                self._pr.add(_h(values[role_name]))
                used = True
        if host or db:
            used = True
            self._add_endpoint(host, port, db, "", resolver)
            if host and suffix:
                self._add_endpoint(host + suffix, port, db, "", resolver)
        if suffix:
            s = suffix.lower().rstrip(".")
            if s:
                self._suffix.add((len(s), _h(s)))
                used = True
        return used

    @classmethod
    def from_files(cls, vera_env: Path | None, intern_env: Path | None, environ: Mapping | None = None, *,
                   resolver=None) -> "Sources":
        src = cls()
        names = URL_NAMES + INTERN_PARAM_NAMES
        for mapping in (_read_names_values(vera_env, URL_NAMES), _read_names_values(intern_env, names),
                        {k: v for k, v in (environ or {}).items() if k in names}):
            if src._add_mapping(mapping, resolver):
                src._n_sources += 1
            mapping.clear()
        return src

    # membership tests used by classify and the inert marker helper (hash comparison only)
    def has_dbname(self, db: str) -> bool:
        return bool(db) and _h(db) in self._pd

    def has_role(self, role: str) -> bool:
        return bool(role) and _h(role) in self._pr

    def indicators(self, p: ParsedURL, resolver=None) -> set:
        hits = set()
        if _h(f"{p.host}|{p.port}|{p.dbname}") in self._pf:
            hits.add("fp")
        if _h(p.host) in self._ph:
            hits.add("host_pattern")
        for length, hsh in self._suffix:
            if len(p.host) >= length and _h(p.host[-length:]) == hsh:
                hits.add("suffix")
        if self.has_dbname(p.dbname):
            hits.add("dbname")
        if self.has_role(p.user):
            hits.add("role")
        if self._pi:
            ips = [p.host] if _IPV4_RE.fullmatch(p.host) else []
            if resolver is not None:
                try:
                    ips += [str(i) for i in (resolver(p.host) or ())]
                except Exception:
                    pass
            if any(_h(i) in self._pi for i in ips):
                hits.add("ip")
        return hits


_DEFAULT: dict = _SHARED.__dict__.setdefault("ec_default", {"sources": None})


def set_default_sources(sources: Sources | None) -> None:
    _DEFAULT["sources"] = sources


def get_default_sources() -> Sources | None:
    return _DEFAULT["sources"]


def classify(url, *, sources: Sources | None = None, environ: Mapping | None = None, name: str = "",
             resolver=None) -> Verdict:
    """Always PRODUCTION. Reasons: parse code, or indicator hits, else 'inconclusive'; plus no_sources and pg_env."""
    reasons: set = set()
    if ambient_names(environ):
        reasons.add("pg_env")
    src = sources if sources is not None else _DEFAULT["sources"]
    try:
        p = parse(url)
    except ParseError as e:
        reasons.add(e.code)
        return _verdict(reasons, name)
    if src is None or not src.has_sources:
        reasons.add("no_sources")
    hits = src.indicators(p, resolver) if src is not None else set()
    if HOST_FLOOR_RE.search(p.host):
        hits.add("host_pattern")
    reasons |= hits if hits else {"inconclusive"}
    return _verdict(reasons, name)


def verify_disposable(conn, url, *, marker_file=None, sources: Sources | None = None,
                      environ: Mapping | None = None) -> Verdict:
    """Owner decision 2026-10-19: refuses unconditionally. It never touches `conn` and never reads `marker_file`."""
    base = classify(url, sources=sources, environ=environ)
    return _verdict(set(base.reasons) | {VERIFY_DISABLED})


# ---------------------------------------------------------------------------------------------------------------
# INERT marker/HMAC helpers (plan §2.4). Not reachable from classify, verify_disposable, connect_guard or check_config.
# Tested directly with fakes only. They take the marker as an in-memory dict; no function here reads a marker file.
# ---------------------------------------------------------------------------------------------------------------
MARKER_TAG_PREFIX = "vera-disposable-v1|"
_MARKER_OK = "marker_ok"


def _inert_marker_tag(nonce: bytes, dbname: str, oid: int) -> str:
    return hmac.new(nonce, f"{MARKER_TAG_PREFIX}{dbname}|{int(oid)}".encode(), hashlib.sha256).hexdigest()


def _inert_marker_check(conn, marker: Mapping, *, sources: Sources, events: list | None = None) -> str:
    """Returns 'marker_ok' or a reason code. Negative checks run first (ordering recorded in `events`)."""
    ev = events if events is not None else []
    try:
        cur = conn.cursor()
        ev.append("read_only")
        cur.execute("SHOW transaction_read_only")
        if str((cur.fetchone() or [""])[0]).lower() != "on":
            return "not_read_only"
        ev.append("identity")
        cur.execute("SELECT current_database(), current_user, "
                    "(SELECT oid FROM pg_database WHERE datname = current_database()), inet_server_port()")
        db, role, oid, port = cur.fetchone()
        if sources.has_dbname(db) or sources.has_role(role):
            return "negative_check"
        ev.append("marker")
        cur.execute("SELECT tag FROM vera_local_meta.disposable_marker")
        row = cur.fetchone()
        if not row or not row[0]:
            return "no_marker"
        if (marker.get("dbname") != db or int(marker.get("oid", -1)) != int(oid)
                or int(marker.get("port", -1)) != int(port)):
            return "tag_mismatch"
        expected = _inert_marker_tag(bytes.fromhex(str(marker["nonce"])), db, oid)
        return _MARKER_OK if hmac.compare_digest(expected, str(row[0])) else "tag_mismatch"
    except Exception:
        return "probe_error"
