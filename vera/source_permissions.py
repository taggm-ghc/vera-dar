"""Shared per-source permission record (NONASK-1103 I-A6 D4; owner R-049/R-050). Pure and offline.

Three independent gates, each with a dev-trace reason code and a plain public phrase (no host, no URL):
  READ            unknown = allowed. Restricted: robots disallow, registry access_verdict forbid, owner deny-list host,
                  login/paywall page, explicit no-crawl signal. A read-restricted source is never fetched.
  USE_FOR_ANSWER  unknown = allowed. Barred: ND licence, no-AI signal (noai/noimageai in robots meta or X-Robots-Tag,
                  Content-Signal ai-input=no in header or robots), registry derivative_verdict forbid.
  INGEST          only when the licence explicitly permits storage (vera.licence_gate) AND registry ingest_verdict is
                  not forbid AND the source is readable and usable; otherwise answer-only.
read_gate() is for BEFORE the fetch; evaluate() is for AFTER it. Both return a frozen PermissionRecord.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from vera import licence_gate

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "source_permissions.json"
REGISTRY_PATH = Path(__file__).resolve().parent.parent / "config" / "entity_registry.json"

_FALLBACK_PHRASE = {
    "read": "not read: the site does not permit automated access",
    "use": "not used: the source does not permit reuse",
    "ingest": "not stored: the source does not permit storage",
}


@dataclass(frozen=True)
class PermissionRecord:
    read: bool
    use_for_answer: bool
    ingest: bool
    read_reason: str
    use_reason: str
    ingest_reason: str
    read_phrase: str
    use_phrase: str
    ingest_phrase: str


def load_config(path: Path = CONFIG_PATH) -> dict:
    try:
        c = json.loads(Path(path).read_text(encoding="utf-8"))
        return c if isinstance(c, dict) else {}
    except Exception:  # noqa: BLE001  missing/corrupt config: empty deny-list, built-in phrases
        return {}


def load_registry(path: Path = REGISTRY_PATH) -> list[dict]:
    try:
        return _rows(json.loads(Path(path).read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return []


def _rows(registry) -> list[dict]:
    if registry is None:  # fail closed: no registry argument means the SHIPPED registry, never "no restrictions"
        try:
            registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []
    rows = registry.get("entities", []) if isinstance(registry, dict) else registry
    return [r for r in rows if isinstance(r, dict) and r.get("state", "active") == "active"]


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".")
    except Exception:  # noqa: BLE001
        return ""


def _host_match(host: str, domain: str) -> bool:
    d = (domain or "").lower().strip().rstrip(".")
    return bool(host and d) and (host == d or host.endswith("." + d))


def registry_row_for(url: str, registry=None) -> dict | None:
    """Active registry row whose domain equals the URL host or is a parent domain of it."""
    host = _host(url)
    for r in _rows(registry):
        if any(_host_match(host, d) for d in r.get("domains", [])):
            return r
    return None


def _deny_hit(host: str, deny_list) -> bool:
    for e in deny_list or []:
        h = e.get("host") if isinstance(e, dict) else e
        if isinstance(h, str) and _host_match(host, h):
            return True
    return False


def _phrase(cfg: dict, gate: str, code: str) -> str:
    key = "use_for_answer" if gate == "use" else gate
    p = (cfg.get("public_phrases") or {}).get(key, {})
    if code in p:
        return p[code]
    return "" if code.endswith("_ok") else _FALLBACK_PHRASE[gate]


def _lower_headers(headers) -> dict:
    return {str(k).lower(): str(v) for k, v in (headers or {}).items()}


_META = re.compile(r"<meta\b[^>]*>", re.I)
_ATTR = re.compile(r"([a-zA-Z\-:]+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))")


def _meta_robots(html_head: str | None) -> str:
    out = []
    for m in _META.findall(html_head or ""):
        a = {k.lower(): (v1 or v2 or v3) for k, v1, v2, v3 in _ATTR.findall(m)}
        if a.get("name", "").lower() in ("robots", "googlebot", "vera-dar", "ai"):
            out.append(a.get("content", "").lower())
    return ",".join(out)


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[\s,;]+", s.lower()) if t}


def _content_signal_no_ai(value: str) -> bool:
    for part in re.split(r"[,\s]+", value.lower()):
        k, _, v = part.partition("=")
        if k.strip() == "ai-input" and v.strip() == "no":
            return True
    return False


def _robots_lines(robots_txt_entries) -> list[str]:
    if not robots_txt_entries:
        return []
    if isinstance(robots_txt_entries, str):
        return robots_txt_entries.splitlines()
    return [str(x) for x in robots_txt_entries]


def _robots_no_ai(robots_txt_entries) -> bool:
    for ln in _robots_lines(robots_txt_entries):
        ln = ln.split("#", 1)[0].strip()
        k, _, v = ln.partition(":")
        if k.strip().lower() == "content-signal" and _content_signal_no_ai(v):
            return True
    return False


def robots_content_signal(robots_txt: str) -> str:
    """WP6-P2b D2: the Content-Signal values of a robots.txt (comments dropped), joined by ", ". Signal text only;
    the caller keeps nothing else of the file. Any group counts (fail-safe: a no-AI line anywhere restricts use)."""
    vals = []
    for ln in _robots_lines(robots_txt):
        k, _, v = ln.split("#", 1)[0].strip().partition(":")
        if k.strip().lower() == "content-signal" and v.strip():
            vals.append(v.strip())
    return ", ".join(vals)


def robots_entries(signal: str) -> list[str]:
    """robots_txt_entries for evaluate() rebuilt from a robots_content_signal() string ([] when empty)."""
    return [f"Content-Signal: {signal}"] if str(signal or "").strip() else []


def robots_no_ai(robots_txt_entries) -> bool:
    """Public form of the robots.txt no-AI test (Content-Signal ai-input=no)."""
    return _robots_no_ai(robots_txt_entries)


def _lic(licence) -> tuple[str | None, str | None]:
    if isinstance(licence, str):
        return (licence, licence) if licence.startswith("http") else (licence, None)
    if isinstance(licence, dict):
        return licence.get("id"), licence.get("url")
    return None, None


def _verdict(row, key: str) -> str:
    return str((row or {}).get(key) or "unknown").lower()


_TERMS_TO_ACCESS = {"links_only": "forbid", "allow": "allow"}


def access_verdict(row) -> str:
    """WP6-P2 (R-050): the row's access verdict; when access_verdict is missing it is derived from terms_verdict
    (links_only -> forbid, allow -> allow, otherwise unknown). terms_verdict links_only marks a host whose terms are
    known to forbid automated access, so such a row is forbid even if access_verdict says otherwise (never fetched)."""
    row = row or {}
    terms = str(row.get("terms_verdict") or "").lower()
    if terms == "links_only":
        return "forbid"
    if row.get("access_verdict"):
        return _verdict(row, "access_verdict")
    return _TERMS_TO_ACCESS.get(terms, "unknown")


def _read(url, robots_allowed, page_signals, row, deny_list, cfg_deny) -> str:
    host = _host(url)
    if _deny_hit(host, deny_list if deny_list is not None else cfg_deny):
        return "read_deny_list"
    if access_verdict(row) == "forbid":
        return "read_registry_forbid"
    if robots_allowed is False:
        return "read_robots_disallow"
    s = page_signals or {}
    if s.get("login_required") or s.get("paywall"):
        return "read_login_paywall"
    if s.get("no_crawl"):
        return "read_no_crawl"
    return "read_ok"


def _record(cfg, read_c, use_c, ing_c) -> PermissionRecord:
    return PermissionRecord(
        read_c == "read_ok", use_c == "use_ok", ing_c == "ingest_ok", read_c, use_c, ing_c,
        _phrase(cfg, "read", read_c), _phrase(cfg, "use", use_c), _phrase(cfg, "ingest", ing_c))


def read_gate(url: str, *, robots_allowed: bool | None = None, page_signals: dict | None = None,
              registry=None, deny_list=None, config: dict | None = None) -> PermissionRecord:
    """Pre-fetch gate: host/registry/deny-list/robots (robots_allowed=None means robots not consulted = allowed).
    use_for_answer here reflects the registry only; ingest is 'ingest_unevaluated' until evaluate() sees the page."""
    cfg = load_config() if config is None else config
    row = registry_row_for(url, registry)
    read_c = _read(url, robots_allowed, page_signals, row, deny_list, cfg.get("deny_hosts"))
    use_c = "use_registry_forbid" if _verdict(row, "derivative_verdict") == "forbid" else "use_ok"
    return _record(cfg, read_c, use_c, "ingest_unevaluated")


def evaluate(url: str, *, licence=None, headers=None, html_head: str | None = None, robots_txt_entries=None,
             registry=None, registry_row: dict | None = None, robots_allowed: bool | None = None,
             page_signals: dict | None = None, deny_list=None, config: dict | None = None) -> PermissionRecord:
    """Post-fetch record. `licence` is {"id","url"} or an id/url string; `headers` response headers;
    `html_head` the page <head> markup; `robots_txt_entries` robots.txt text or lines (Content-Signal lines)."""
    cfg = load_config() if config is None else config
    row = registry_row if registry_row is not None else registry_row_for(url, registry)
    h = _lower_headers(headers)
    xrt = _tokens(h.get("x-robots-tag", ""))
    meta = _tokens(_meta_robots(html_head))
    signals = dict(page_signals or {})
    head = html_head or ""
    if re.search(r"type\s*=\s*[\"']?password", head, re.I):
        signals["login_required"] = True
    if re.search(r"isAccessibleForFree[\"']?\s*:\s*[\"']?false", head, re.I):
        signals["paywall"] = True
    if {"noai-crawl", "nocrawl"} & (xrt | meta):
        signals["no_crawl"] = True
    read_c = _read(url, robots_allowed, signals, row, deny_list, cfg.get("deny_hosts"))

    lid, lurl = _lic(licence)
    nd = licence_gate.reject_reason(lid, lurl) == "ND (no derivatives)"
    no_ai = (bool({"noai", "noimageai"} & (xrt | meta)) or _content_signal_no_ai(h.get("content-signal", ""))
             or _robots_no_ai(robots_txt_entries))
    if nd:
        use_c = "use_nd_licence"
    elif no_ai:
        use_c = "use_no_ai_signal"
    elif _verdict(row, "derivative_verdict") == "forbid":
        use_c = "use_registry_forbid"
    else:
        use_c = "use_ok"

    if read_c != "read_ok":
        ing_c = "ingest_not_readable"
    elif use_c != "use_ok":
        ing_c = "ingest_not_usable"
    elif _verdict(row, "ingest_verdict") == "forbid":
        ing_c = "ingest_registry_forbid"
    elif licence_gate.licence_decision({"url": url, "licence": {"content": {"id": lid, "url": lurl}}})[0] != "allow":
        ing_c = "ingest_no_storage_licence"
    else:
        ing_c = "ingest_ok"
    return _record(cfg, read_c, use_c, ing_c)
