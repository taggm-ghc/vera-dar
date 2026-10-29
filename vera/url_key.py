"""R3NORM-1028: the single normaliser for canonical-source keys (canonical_sources.canonical_url, the R3 key).

Why: R3 is a single-text UNIQUE; two spellings of one URL became two rows (KEYDUP-DB-1028, OD-1028(7)/(8)). The DDL
fix is owner-gated; this is the code-side first step. `canonical_key` is the ONLY function that builds a key for a new
canonical source, and every lookup also tries `legacy_keys` (read-time compatibility): rows already stored under the
old normalisations are still found, so no duplicate is inserted for a stored source. No row is migrated or rewritten.

Conservative by design: only transformations that cannot join two distinct documents are applied.
  - scheme and host lower-cased; default port (80 http, 443 https) dropped; fragment dropped;
  - trailing slashes dropped (matches both legacy forms for the root path);
  - known tracking parameters (utm_*, fbclid, gclid, ...) dropped (matched on the decoded, lower-cased name); the remaining
    query parameters are sorted by raw name only (stable: repeated names keep their relative order, so ?id=2&id=1 is
    NOT ?id=1&id=2); the query is never decoded or re-encoded, so 'a+b', 'a%20b', 'a/b' and 'a%2Fb' stay distinct and
    '?flag' stays distinct from '?flag='; empty segments (stray '&') are dropped;
  - percent-escape hex upper-cased in path and query (no decoding);
  - DOI forms (doi:, dx.doi.org, doi.org, case) -> https://doi.org/<lower-case doi>;
  - arXiv abs / pdf / .pdf -> https://arxiv.org/abs/<id>; a version suffix (v2) is KEPT (versions can differ);
  - http -> https only for hosts known to serve both identically (doi.org, arxiv.org).
Everything else (www., path case, path dot-segments, other hosts' scheme, non-tracking query values) is left alone.
"""
from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit, urlunsplit

TRACKING_PARAMS = frozenset({"fbclid", "gclid", "dclid", "msclkid", "yclid", "mc_cid", "mc_eid", "_hsenc", "_hsmi",
                             "igshid", "ref_src"})
TRACKING_PREFIXES = ("utm_",)
_SAFE_HTTPS_HOSTS = frozenset({"doi.org", "arxiv.org"})
_DOI_HOSTS = frozenset({"doi.org", "dx.doi.org"})
_DOI_PREFIX = re.compile(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)", re.I)
_ARXIV_PATH = re.compile(r"^/(?:abs|pdf)/(.+?)(?:\.pdf)?$", re.I)
_PCT = re.compile(r"%[0-9a-fA-F]{2}")


def normalize_doi(doi) -> str:
    """Bare lower-case DOI from any common form ('' when none)."""
    return _DOI_PREFIX.sub("", str(doi or "").strip()).strip().lower()


def _is_tracking(name: str) -> bool:
    n = unquote(name).lower()
    return n in TRACKING_PARAMS or n.startswith(TRACKING_PREFIXES)


def _upper_pct(s: str) -> str:
    return _PCT.sub(lambda m: m.group(0).upper(), s)


def normalize_url(url) -> str:
    """Stronger canonical form of one URL (see module docstring). Never raises; unparseable input is stripped only."""
    raw = str(url or "").strip()
    try:
        p = urlsplit(raw)
        scheme, host = p.scheme.lower(), (p.hostname or "").lower()
        port = p.port
    except ValueError:
        return raw.rstrip("/")
    if scheme not in ("http", "https") or not host:
        return raw.rstrip("/")
    if host in _DOI_HOSTS and p.path.strip("/") and not p.query:
        return "https://doi.org/" + normalize_doi(p.path.strip("/"))
    if host in _SAFE_HTTPS_HOSTS:
        scheme = "https"
    if port is not None and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    if ":" in (p.hostname or ""):  # IPv6 literal
        host = f"[{p.hostname.lower()}]" + host[len(p.hostname):]
    path = _upper_pct(p.path).rstrip("/")
    if host == "arxiv.org":
        m = _ARXIV_PATH.match(path)
        if m:
            path = "/abs/" + m.group(1).lower()
    segs = [(_upper_pct(seg.split("=", 1)[0]), _upper_pct(seg)) for seg in p.query.split("&") if seg]
    segs = [(k, seg) for k, seg in segs if not _is_tracking(k)]
    query = "&".join(seg for _, seg in sorted(segs, key=lambda kv: kv[0]))  # stable sort on the raw name only
    user = f"{p.username}@" if p.username and p.password is None else ""
    if p.password is not None:
        user = f"{p.username or ''}:{p.password}@"
    return urlunsplit((scheme, user + host, path, query, ""))


def canonical_key(src: dict) -> str:
    """The key for a canonical source: DOI form when a DOI is known, else the normalised URL."""
    doi = normalize_doi(src.get("doi"))
    return f"https://doi.org/{doi}" if doi else normalize_url(src.get("url") or "")


def _legacy_corpus(src: dict) -> str:
    """Pre-R3NORM corpus_admission.canonical_url, verbatim."""
    doi = str(src.get("doi") or "").strip().lower()
    return f"https://doi.org/{doi}" if doi else str(src.get("url") or "").strip().rstrip("/")


def _legacy_m2(url: str) -> str:
    """Pre-R3NORM search_and_fetch.normalize_url, verbatim (m2_runner key; keeps '/' for the root path)."""
    p = urlsplit(str(url or "").strip())
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/") or "/", p.query, ""))


def legacy_keys(src: dict) -> list[str]:
    """Old spellings a row for this source may be stored under (excluding the current key)."""
    out = [_legacy_corpus(src)]
    if str(src.get("url") or "").strip():
        try:
            out.append(_legacy_m2(src["url"]))
        except ValueError:
            pass
    cur = canonical_key(src)
    return [k for k in dict.fromkeys(out) if k and k != cur]


def lookup_keys(src: dict) -> list[str]:
    """Current key first, then legacy keys, de-duplicated. Use for every read/lookup of canonical_sources."""
    return [canonical_key(src)] + legacy_keys(src)
