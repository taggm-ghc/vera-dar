"""Shared per-source permission record (NONASK-1103 I-A6 D4; owner R-049/R-050). Pure and offline.

Three independent gates, each with a dev-trace reason code and a plain public phrase (no host, no URL):
  READ            unknown = allowed. Restricted: robots disallow, registry access_verdict forbid, owner deny-list host,
                  login/paywall page, explicit no-crawl signal. A read-restricted source is never fetched.
  USE_FOR_ANSWER  unknown = allowed. Barred: ND licence, no-AI signal (noai/noimageai in robots meta or X-Robots-Tag,
                  Content-Signal ai-input=no in header or robots, Content-Usage ai-use=n, Bing ai-grounding
                  Disallow/DisallowExtended in a Bingbot meta or X-Robots-Tag, H11), registry derivative_verdict forbid.
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
        if not isinstance(c, dict):
            return {}
        if Path(path) == CONFIG_PATH:  # LISTS-TO-DB-1104: permissions.deny_hosts (union with the file) when enabled
            try:
                from vera import list_tables
                c = list_tables.overlay("source_permissions.json", c)
            except Exception:  # noqa: BLE001  loader missing or faulty: keep the file value
                pass
        return c
    except Exception:  # noqa: BLE001  missing/corrupt config: empty deny-list, built-in phrases
        return {}


def load_registry(path: Path = REGISTRY_PATH) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if Path(path) == REGISTRY_PATH and isinstance(data, dict):  # LISTS-TO-DB-1104: entity.registry from the DB when enabled
            try:
                from vera import list_tables
                data = list_tables.overlay("entity_registry.json", data)
            except Exception:  # noqa: BLE001  loader missing or faulty: keep the file value
                pass
        return _rows(data)
    except Exception:  # noqa: BLE001
        return []


def _rows(registry) -> list[dict]:
    if registry is None:  # fail closed: no registry argument means the SHIPPED registry, never "no restrictions"
        try:
            registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            if isinstance(registry, dict):  # LISTS-TO-DB-1104: entity.registry from the DB when enabled
                try:
                    from vera import list_tables
                    registry = list_tables.overlay("entity_registry.json", registry)
                except Exception:  # noqa: BLE001  loader missing or faulty: keep the file value
                    pass
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
        name = a.get("name", "").lower()
        if name in ("robots", "googlebot", "vera-dar", "ai"):
            out.append(a.get("content", "").lower())
        elif name == "bingbot":  # H11: only the ai-grounding directive is read from a Bingbot meta
            out.extend(ai_grounding_tokens(a.get("content", "")))
    return ",".join(out)


_AI_GROUNDING = re.compile(r"ai-grounding\s*:\s*[\"']?\s*(allow|disallowextended|disallow)\b", re.I)


def ai_grounding_tokens(value: str) -> list[str]:
    """OD-1108-DISPOSITIONS-2 H11: Bing's page-level ai-grounding directive (<meta name="Bingbot"
    content="ai-grounding:disallow">, or an X-Robots-Tag value, with or without a "bingbot:" prefix) as normalised
    tokens "ai-grounding:allow|disallow|disallowextended". Evidence for the syntax is secondary only
    (work/AIPREF-VERIFY-1108/findings.md claim (e))."""
    return [f"ai-grounding:{m.lower()}" for m in _AI_GROUNDING.findall(str(value or ""))]


def ai_grounding_disallow(value: str) -> bool:
    """H11: True when value carries ai-grounding Disallow or DisallowExtended; both bind answer-evidence use (the
    same effect as Content-Signal ai-input=no). Allow or absent: False."""
    return any(t != "ai-grounding:allow" for t in ai_grounding_tokens(value))


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[\s,;]+", s.lower()) if t}


def _content_signal_no_ai(value: str) -> bool:
    for part in re.split(r"[,\s]+", value.lower()):
        k, _, v = part.partition("=")
        if k.strip() == "ai-input" and v.strip() == "no":
            return True
    return False


def content_usage_signal(value: str) -> str:
    """A2: Content-Usage (IETF AIPREF) value -> "ai-input=no" when ai-use is n/no (equivalent to Content-Signal
    ai-input=no); "" when it parses without barring; "info:content-usage-unparsed" when nothing parses (recorded, no bar)."""
    v0 = re.sub(r"\s*=\s*", "=", str(value or "").lower()).replace('"', "")  # lenient: spaces around '=', quoted values
    parts = [p for p in re.split(r"[,;\s]+", v0) if p]
    if not parts:
        return ""
    ok = False
    for part in parts:
        k, sep, v = part.partition("=")
        if sep and k.strip() and v.strip():
            ok = True
            if k.strip() == "ai-use" and v.strip() in ("n", "no"):
                return "ai-input=no"
    return "" if ok else "info:content-usage-unparsed"


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
        if k.strip().lower() == "content-usage" and content_usage_signal(v) == "ai-input=no":  # A2: raw robots.txt lines
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
        elif k.strip().lower() == "content-usage" and content_usage_signal(v):  # A2, any group counts
            vals.append(content_usage_signal(v))
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


_USE_OK = ("use_ok", "use_ok_nd_news")  # H19: use_ok_nd_news = ND news with a satisfied access verdict (answer-only)


def _tdm_reserved(lower_headers: dict) -> bool:
    """TDM Reservation Protocol signal as a header-style value: "1" (reserved) or "unreadable" (tdmrep.json could not be
    read; fail closed). Anything else (absent, "", "0") = not reserved."""
    return str(lower_headers.get("tdm-reservation", "")).strip().lower() in ("1", "unreadable")


def _record(cfg, read_c, use_c, ing_c) -> PermissionRecord:
    return PermissionRecord(
        read_c == "read_ok", use_c in _USE_OK, ing_c == "ingest_ok", read_c, use_c, ing_c,
        _phrase(cfg, "read", read_c), _phrase(cfg, "use", use_c), _phrase(cfg, "ingest", ing_c))


def _purpose_shadow(url, lower_headers, robots_txt_entries, restrictions) -> None:
    """PURPOSE-GRANTS-1108: what a site-owner purpose grant (this page's own headers; raw robots.txt text when the
    caller passed it) WOULD do to these restrictions. Recorded only; the record above is unchanged. Never raises."""
    try:
        from vera import purpose_grants
        if purpose_grants.mode() != "shadow":
            return
        g = purpose_grants.scan_headers(lower_headers)
        if isinstance(robots_txt_entries, str) or isinstance(robots_txt_entries, (list, tuple)):
            g = g + purpose_grants.scan_robots("\n".join(_robots_lines(robots_txt_entries)))
        if g:
            purpose_grants.shadow(restrictions() if callable(restrictions) else restrictions, g, url)
    except Exception:  # noqa: BLE001
        pass


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
             or _content_signal_no_ai(content_usage_signal(h.get("content-usage", "")))
             or ai_grounding_disallow(h.get("x-robots-tag", ""))  # H11: Bing ai-grounding, header form
             or ai_grounding_disallow(_meta_robots(html_head))  # H11: <meta name="Bingbot"> form
             or _robots_no_ai(robots_txt_entries))
    nd_news = nd and licence_gate.nd_news_permitted(
        {"url": url, "licence": {"content": {"id": lid, "url": lurl}}})  # H19: looked up from the URL host; flag off = False
    tdm = _tdm_reserved(h)  # H19 A9 / T2b: a publisher opt-out, bars use for every lane
    if tdm:
        use_c = "use_tdm_reserved"
    elif nd and not nd_news:
        use_c = "use_nd_licence"
    elif no_ai:
        use_c = "use_no_ai_signal"
    elif _verdict(row, "derivative_verdict") == "forbid":
        use_c = "use_registry_forbid"
    else:
        use_c = "use_ok_nd_news" if nd_news else "use_ok"
    if use_c in ("use_tdm_reserved", "use_no_ai_signal"):  # PURPOSE-GRANTS-1108: shadow record only
        _purpose_shadow(url, h, robots_txt_entries, lambda: [c for c, on in (  # built inside the hook's try (N2)
            (read_c, read_c != "read_ok"), ("use_tdm_reserved", tdm), ("use_nd_licence", nd and not nd_news),
            ("use_no_ai_signal", no_ai), ("use_registry_forbid", _verdict(row, "derivative_verdict") == "forbid"))
            if on])

    if read_c != "read_ok":
        ing_c = "ingest_not_readable"
    elif use_c != "use_ok":  # includes use_ok_nd_news: ND news is never ingested (answer-only)
        ing_c = "ingest_not_usable"
    elif _verdict(row, "ingest_verdict") == "forbid":
        ing_c = "ingest_registry_forbid"
    elif licence_gate.licence_decision({"url": url, "licence": {"content": {"id": lid, "url": lurl}}})[0] != "allow":
        ing_c = "ingest_no_storage_licence"
    else:
        ing_c = "ingest_ok"
    return _record(cfg, read_c, use_c, ing_c)
