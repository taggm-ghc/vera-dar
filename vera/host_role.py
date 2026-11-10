"""PORTAL-RULE-1110 WP1: one host-role policy for every evidence entry point (pure; no network, no DB).

Roles: ORIGINAL | REPOSITORY_OF_RECORD | PORTAL | RESOLVER | EXCLUDED (plus UNKNOWN: no row). Policy lives in the
reviewed file config/host_roles.json, loaded once, validated, sha256 recorded. A missing or invalid file fails closed
(every host PORTAL, rule "config_invalid"). Nothing here raises. Not wired into any lane yet (WP3-WP7).

Public API:
  role_for_url(url, meta=None) -> HostRole(role, rule_id)
  role_for_record(meta)        -> HostRole      (DOI prefix, source flag, type, version, then url)
  is_evidence_allowed(target, use) -> bool      (target: url str or record dict; use: "store" | "answer")
  is_hop_allowed(url) -> bool                   (redirect hop: not PORTAL/EXCLUDED; RESOLVER passes)
  config_hash() / config_error()
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import threading
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote, urlsplit

ROLES = ("ORIGINAL", "REPOSITORY_OF_RECORD", "PORTAL", "RESOLVER", "EXCLUDED")
UNKNOWN = "UNKNOWN"
_EVIDENCE_ROLES = ("ORIGINAL", "REPOSITORY_OF_RECORD")
_RANK = {"EXCLUDED": 0, "PORTAL": 1, "RESOLVER": 2, "REPOSITORY_OF_RECORD": 3, "ORIGINAL": 4}  # lower wins ties
_USES = ("store", "answer")
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "host_roles.json"
_DOI_HOSTS = ("doi.org", "dx.doi.org")


class HostRole(NamedTuple):
    role: str
    rule_id: str


class ConfigError(ValueError):
    pass


class Registry:
    """Validated, immutable view of the policy file."""

    def __init__(self, cfg: dict, sha256: str = "", error: str | None = None):
        self.sha256, self.error = sha256, error
        self.hosts: list[dict] = []
        self.patterns: list[dict] = []
        self.doi_rules: list[tuple] = []
        self.flags: list[dict] = []
        self.type_portal: set[str] = set()
        self.rv: dict = {}
        self.unknown = {"store": "deny", "answer": "allow"}
        if error is None:
            try:
                self._build(cfg)
            except Exception as e:  # noqa: BLE001 - any defect means fail closed
                self.error = f"{type(e).__name__}: {e}"

    def _build(self, cfg: dict) -> None:
        if not isinstance(cfg, dict):
            raise ConfigError("top level must be an object")
        seen = set()
        for i, row in enumerate(cfg.get("hosts") or []):
            m = str(row.get("match", "")).strip().lower().strip(".")
            if not m or "/" in m or " " in m:
                raise ConfigError(f"hosts[{i}]: bad match")
            if row.get("role") not in ROLES:
                raise ConfigError(f"hosts[{i}] {m}: bad role {row.get('role')!r}")
            if not row.get("kind"):
                raise ConfigError(f"hosts[{i}] {m}: missing kind")
            pr = row.get("path_regex")
            rx = re.compile(pr, re.I) if pr else None
            key = (m, pr)
            if key in seen:
                raise ConfigError(f"duplicate match {key}")
            seen.add(key)
            spec = (m.count(".") + 1) * 1 + (1000 if rx else 0)
            self.hosts.append({**row, "match": m, "_rx": rx, "_spec": spec, "_id": f"host:{m}" + (f"#{pr}" if pr else "")})
        for i, row in enumerate(cfg.get("host_patterns") or []):
            if row.get("role") not in ROLES or not row.get("id"):
                raise ConfigError(f"host_patterns[{i}]: bad role or id")
            self.patterns.append({**row, "_rx": re.compile(row["regex"], re.I)})
        for row in cfg.get("doi_prefix_roles") or []:
            if row.get("role") not in ROLES:
                raise ConfigError("doi_prefix_roles: bad role")
            self.doi_rules.append((re.compile(row["regex"], re.I), row["role"], str(row.get("id", "doi"))))
        for row in cfg.get("source_flags") or []:
            if row.get("role") not in ROLES:
                raise ConfigError("source_flags: bad role")
            self.flags.append({**row, "value": str(row["value"]).upper()})
        self.type_portal = {str(x).lower() for x in cfg.get("record_type_portal") or []}
        rv = cfg.get("repository_version") or {}
        self.rv = {k: {str(x).lower() for x in rv.get(k, [])} for k in ("portal_versions", "accepted_versions", "non_article_types")}
        u = (cfg.get("use_defaults") or {}).get("unknown_host") or {}
        self.unknown = {"store": u.get("store", "deny"), "answer": u.get("answer", "allow")}
        for v in self.unknown.values():
            if v not in ("allow", "deny"):
                raise ConfigError("unknown_host values must be allow|deny")
        if not self.hosts:
            raise ConfigError("no host rows")


def _failed(error: str, sha: str = "") -> Registry:
    return Registry({}, sha, error)


def load_registry(path=None) -> Registry:
    """Read and validate the file; never raises."""
    try:
        raw = Path(path or DEFAULT_PATH).read_bytes()
    except Exception as e:  # noqa: BLE001
        return _failed(f"unreadable: {type(e).__name__}")
    sha = hashlib.sha256(raw).hexdigest()
    try:
        return Registry(json.loads(raw.decode("utf-8")), sha)
    except Exception as e:  # noqa: BLE001
        return _failed(f"invalid: {type(e).__name__}: {e}", sha)


_lock = threading.Lock()
_REG: Registry | None = None


def get_registry() -> Registry:
    global _REG
    if _REG is None:
        with _lock:
            if _REG is None:
                _REG = load_registry()
    return _REG


def set_registry_for_tests(reg: Registry | None) -> None:
    """Test hook: install a registry (None resets to lazy load of the real file)."""
    global _REG
    _REG = reg


def config_hash() -> str:
    return get_registry().sha256


def config_error() -> str | None:
    return get_registry().error


# ------------------------------------------------------------------------------------------------ URL handling
def _split(url) -> tuple[str | None, list, str]:
    """(host, path, problem). host None when unusable."""
    if not isinstance(url, str) or not url.strip() or len(url) > 4096 or re.search(r"[\x00-\x20\x7f\\]", url.strip()):
        return None, [], "malformed_url"
    u = url.strip()
    if "://" not in u and not u.startswith("//"):
        u = "//" + u
    try:
        p = urlsplit(u)
        if p.scheme and p.scheme.lower() not in ("http", "https"):
            return None, [], "malformed_url"
        host = p.hostname
        p.port  # noqa: B018 - validates the port
    except Exception:  # noqa: BLE001
        return None, [], "malformed_url"
    if not host:
        return None, [], "malformed_url"
    host = re.sub("[\u3002\uff0e\uff61]", ".", host.lower()).rstrip(".")
    if not host or "." not in host or ".." in host or "%" in host:
        return None, [], "malformed_url"
    try:
        host = host.encode("idna").decode("ascii").lower().rstrip(".")  # strip again: IDNA can map to "."
    except Exception:  # noqa: BLE001
        return None, [], "malformed_url"
    if not host or "." not in host or ".." in host:
        return None, [], "malformed_url"
    try:
        ipaddress.ip_address(host)
        return None, [], "ip_literal"
    except ValueError:
        pass
    if _NUMERIC_LABEL.match(host.rsplit(".", 1)[-1]) or all(_NUMERIC_LABEL.match(x) for x in host.split(".")):
        return None, [], "ip_literal"  # inet_aton forms such as 0x7f.1, 2130706433
    raw = p.path or "/"
    paths = []
    for q in (_norm_path(raw), _literal_path(raw, decode=False), _literal_path(raw, decode=True)):
        if p.query and q == "/":
            q = "/?" + p.query  # ERIC-style /?id=ED123
        if q not in paths:
            paths.append(q)
    return host, paths, ""


_NUMERIC_LABEL = re.compile(r"^(?:0x[0-9a-f]*|[0-9]+)$", re.I)


def _norm_path(raw: str) -> str:
    """Percent-decode (up to 3 rounds), collapse //, resolve . and .. (RFC 3986 5.2.4) before rule matching."""
    path = raw
    for _ in range(3):
        d = unquote(path)
        if d == path:
            break
        path = d
    return _resolve(path.replace("\\", "/"), path)


_SEP = re.compile(r"%(2f|5c|25)", re.I)


def _literal_path(raw: str, decode: bool) -> str:
    """A reading that keeps encoded separators (%2f, %5c, %25) literal, so they cannot form dot segments:
    decode=False decodes only %2e; decode=True also decodes everything else once."""
    s = re.sub(r"%2e", ".", raw, flags=re.I)
    if decode:
        s = _SEP.sub(lambda m: "\x00" + m.group(1), s)
        s = unquote(s).replace("\x00", "%")
    return _resolve(s, s)


def _resolve(path: str, orig: str) -> str:
    out: list[str] = []
    for seg in path.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if out:
                out.pop()
            continue
        out.append(seg)
    return "/" + "/".join(out) + ("/" if orig.endswith("/") and out else "")


def _suffix_ok(host: str, match: str) -> bool:
    return host == match or host.endswith("." + match)


def _qualifies(row: dict, reg: Registry, meta) -> bool:
    """Version test for a version_gated repository row: True when the record is an accepted/published version
    (or a non-article deposit with no preprint marker). Anything undecidable is False (fail closed)."""
    m = meta if isinstance(meta, dict) else {}
    ver = str(m.get("version") or "").strip().lower()
    typ = str(m.get("type") or m.get("resource_type") or m.get("source_type") or "").strip().lower()
    if typ in reg.type_portal or ver in reg.rv["portal_versions"] and ver:
        return False
    if ver in reg.rv["accepted_versions"]:
        return True
    return not ver and typ in reg.rv["non_article_types"]


def _match_row(host: str, path: str, reg: Registry):
    best = None
    for row in reg.hosts:
        if not _suffix_ok(host, row["match"]):
            continue
        if row["_rx"] is not None and not row["_rx"].search(path):
            continue
        key = (-row["_spec"], -int(row.get("priority", 0)), _RANK[row["role"]])
        if best is None or key < best[0]:
            best = (key, row)
    for row in reg.patterns:
        if row["_rx"].search(host):
            key = (-1, 0, _RANK[row["role"]])  # patterns rank as a one-label suffix
            if best is None or key < best[0]:
                best = (key, row)
    return best[1] if best else None


def _doi_of(text) -> str:
    t = str(text or "").strip()
    t = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", t, flags=re.I)
    return unquote(t).strip()


def _doi_role(doi: str, reg: Registry):
    d = _doi_of(doi)
    if d:
        for rx, role, rid in reg.doi_rules:
            if rx.search(d):
                return HostRole(role, f"doi:{rid}")
    return None


def role_for_url(url, meta=None) -> HostRole:
    """Role of a URL's host (and path). Never raises. meta feeds the version test for version-gated rows."""
    try:
        reg = get_registry()
        if reg.error:
            return HostRole("PORTAL", "config_invalid")
        host, paths, problem = _split(url)
        if host is None:
            return HostRole(UNKNOWN, problem)
        # classify every reading of the path (fully decoded; encoded separators kept literal) and keep the stricter
        best = None
        for path in paths:
            hr = _role_for_path(host, path, reg, meta)
            if best is None or _STRICT.get(hr.role, 1.5) < _STRICT.get(best.role, 1.5):
                best = hr
        return best
    except Exception:  # noqa: BLE001
        return HostRole("PORTAL", "internal_error")


_STRICT = {"EXCLUDED": 0, "PORTAL": 1, UNKNOWN: 1.5, "RESOLVER": 2, "REPOSITORY_OF_RECORD": 3, "ORIGINAL": 4}


def _role_for_path(host: str, path: str, reg: Registry, meta) -> HostRole:
    if True:
        row = _match_row(host, path, reg)
        if row is None:
            return HostRole(UNKNOWN, "no_row")
        if host in _DOI_HOSTS or (row["role"] == "RESOLVER" and row.get("match") in _DOI_HOSTS):
            hit = _doi_role(path.lstrip("/"), reg)
            if hit:
                return hit
        rid = row.get("_id") or f"pattern:{row.get('id')}"
        if row.get("version_gated") and not _qualifies(row, reg, meta):
            return HostRole("PORTAL", f"version_unqualified:{rid}")
        return HostRole(row["role"], rid)


def role_for_record(meta) -> HostRole:
    """Record-level role: DOI prefix, source flag (Europe PMC PPR), preprint type, then the URL's host."""
    try:
        reg = get_registry()
        if reg.error:
            return HostRole("PORTAL", "config_invalid")
        if not isinstance(meta, dict):
            return HostRole(UNKNOWN, "no_record")
        hit = _doi_role(meta.get("doi") or "", reg)
        if hit:
            return hit
        oa = meta.get("oa") if isinstance(meta.get("oa"), dict) else {}
        for f in reg.flags:
            if f["value"] in (str(meta.get(f["field"]) or "").strip().upper(), str(oa.get(f["field"]) or "").strip().upper()):
                return HostRole(f["role"], f"flag:{f.get('id', f['field'])}")
        ptypes = meta.get("pub_types") or meta.get("pubTypeList") or []
        ptypes = ptypes.get("pubType", []) if isinstance(ptypes, dict) else ptypes
        ptypes = [ptypes] if isinstance(ptypes, str) else list(ptypes or [])
        typ = str(meta.get("source_type") or meta.get("type") or "").strip().lower()
        if typ in reg.type_portal or any(str(p).strip().lower() in reg.type_portal for p in ptypes) or meta.get("is_preprint") is True:
            return HostRole("PORTAL", "record_type:preprint")
        for k in ("url", "landing_url", "fulltext_url", "full_text_url"):
            if meta.get(k):
                return role_for_url(meta[k], meta)
        return HostRole(UNKNOWN, "no_url")
    except Exception:  # noqa: BLE001
        return HostRole("PORTAL", "internal_error")


def is_evidence_allowed(target, use: str = "store") -> bool:
    """May text from this URL/record be evidence for `use`? ORIGINAL and REPOSITORY_OF_RECORD yes; PORTAL, EXCLUDED
    and RESOLVER no (a resolver has no content of its own); UNKNOWN follows config (store deny, answer allow);
    malformed URLs and an invalid config deny. Never raises."""
    try:
        if use not in _USES:
            return False
        hr = role_for_record(target) if isinstance(target, dict) else role_for_url(target)
        if hr.role in _EVIDENCE_ROLES:
            return True
        if hr.role != UNKNOWN or hr.rule_id in ("malformed_url", "ip_literal", "no_record", "no_url"):
            return False
        reg = get_registry()
        return reg.unknown.get(use) == "allow" and not reg.error
    except Exception:  # noqa: BLE001
        return False


def is_hop_allowed(url, use: str = "answer", meta=None) -> bool:
    """Redirect-hop check. store: only ORIGINAL, REPOSITORY_OF_RECORD or RESOLVER hops (unknown hops refused; the
    route allowlist still applies). answer: PORTAL/EXCLUDED/malformed refused, RESOLVER and unknown hosts pass, and
    the caller must re-check the final URL with is_evidence_allowed(final, "answer"). Never raises."""
    try:
        if use not in _USES:
            return False
        hr = role_for_url(url, meta)
        if use == "store":
            return hr.role in _EVIDENCE_ROLES or hr.role == "RESOLVER"
        if hr.role in ("PORTAL", "EXCLUDED"):
            return False
        return not (hr.role == UNKNOWN and hr.rule_id in ("malformed_url", "ip_literal"))
    except Exception:  # noqa: BLE001
        return False
