"""LLM base_url host allowlist (LLM-ALLOWLIST-1113, problem P-1112-02).

Every code path that builds an LLM client from a base_url read from configuration calls `require_allowed` FIRST
(before any key is read from the environment and before any transport is built) and passes the RETURNED value to the
client constructor. The allowlist is derived at call time from the active provider entries (`providers`) of
config/ask-provider-chain.json: no host is written in code. A refusal raises LlmHostRefused (an OpenAIError, so the
/ask chain treats it as a provider failure and moves on) with a distinct, fixed reason code. Fail closed: an unreadable
file or an empty list refuses everything.

Limit (stated in the plan): for the chain entries themselves the list is derived from the same entries, so that check
is shape-only (https, port, no userinfo, key-name binding); the checker, stance, draft and generator sites are checked
against the provider entries for real. The text, repr and args of a refusal never contain the URL or host.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

from openai import OpenAIError

logger = logging.getLogger("vera")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "ask-provider-chain.json"

HOST_NOT_ALLOWED = "R-LLM-HOST-NOT-ALLOWED"
URL_INVALID = "R-LLM-URL-INVALID"
ALLOWLIST_UNAVAILABLE = "R-LLM-ALLOWLIST-UNAVAILABLE"
KEY_NOT_BOUND = "R-LLM-KEY-NOT-BOUND"

_TEXT = {
    HOST_NOT_ALLOWED: "LLM base URL refused: its host is not on the allowlist derived from the configured providers",
    URL_INVALID: "LLM base URL refused: not a plain https URL",
    ALLOWLIST_UNAVAILABLE: "LLM base URL refused: the provider allowlist is unavailable or empty",
    KEY_NOT_BOUND: "LLM base URL refused: the API key variable is not bound to that host in the provider entries",
}
_KEY_ENV_RE = re.compile(r"[A-Z][A-Z0-9_]*_API_KEY")
_STRICT_RE = re.compile(r"(?i)https://([^/?#:@\[\]]+)(?::0*443)?(?:[/?#]|$)")
_HOST_LDH_RE =re.compile(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*")


class LlmHostRefused(OpenAIError):
    """Raised before any client is built. `.code` is one of the R-LLM-* constants; the text is fixed."""

    def __init__(self, code: str):
        super().__init__(_TEXT[code])
        self.code = code


def _refuse(code: str):
    logger.warning("LLM base_url refused (%s)", code)
    return LlmHostRefused(code)


def _clean_host(base_url) -> str | None:
    """Lowercased hostname of a plain https URL, or None. Refuses rather than normalises odd input."""
    if not isinstance(base_url, str) or not base_url or not base_url.isascii():
        return None
    if any(ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in base_url) or "\\" in base_url:
        return None
    try:
        parts = urlsplit(base_url)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    if parts.scheme.lower() != "https" or not host or "@" in parts.netloc or "%" in parts.netloc:
        return None
    if port not in (None, 443):
        return None
    # second opinion, independent of urlsplit: a strict pattern over the raw string must see the same host (the SDK's
    # own parser may differ from urlsplit on odd input; agreement of two parsers plus the character refusals above
    # removes that differential). Not httpx: importing it would make this module an outbound-inventory site.
    m = _STRICT_RE.match(base_url)
    if m is None or m.group(1).lower() != host.lower():
        return None
    host = host.lower()
    return host if _HOST_LDH_RE.fullmatch(host) else None


_cache: dict = {}
_EMPTY = MappingProxyType({})


def allowed_hosts(path=None) -> dict:
    """{host: frozenset(api_key_env)} from providers[*] of the chain file. Empty when unreadable. Cached on the
    identity (resolved path, mtime_ns, size) of the file actually read; an empty result is never cached."""
    p = Path(path if path is not None else CONFIG_PATH)
    try:
        p = p.resolve()
        with open(p, "rb") as fh:
            st = os.fstat(fh.fileno())
            key = (str(p), st.st_mtime_ns, st.st_size)
            hit = _cache.get(key)
            if hit:
                return hit
            raw = json.loads(fh.read().decode("utf-8"))
    except (OSError, ValueError, RecursionError, RuntimeError):  # RecursionError: absurdly deep JSON; fail closed
        return _EMPTY
    provs = raw.get("providers") if isinstance(raw, dict) else None
    if not isinstance(provs, list):  # int, bool, str, dict, null: unavailable, never a TypeError
        return _EMPTY
    out: dict = {}
    for ent in provs:
        if not isinstance(ent, dict):
            continue
        host = _clean_host(ent.get("base_url"))
        env = ent.get("api_key_env")
        if host and isinstance(env, str) and _KEY_ENV_RE.fullmatch(env):
            out[host] = out.get(host, frozenset()) | {env}
    if not out:
        return _EMPTY
    view = MappingProxyType(out)  # immutable view of frozensets: a caller cannot widen the cached list
    _cache.clear()
    _cache[key] = view
    return view


def require_allowed(base_url, api_key_env, path=None) -> str:
    """Return `base_url` unchanged when its host is allowed and `api_key_env` is bound to it; else raise
    LlmHostRefused. Call this first at every site and pass the returned value on."""
    host = _clean_host(base_url)
    if host is None:
        raise _refuse(URL_INVALID) from None
    allow = allowed_hosts(path)
    if not allow:
        raise _refuse(ALLOWLIST_UNAVAILABLE) from None
    if host not in allow:
        raise _refuse(HOST_NOT_ALLOWED) from None
    if not (isinstance(api_key_env, str) and _KEY_ENV_RE.fullmatch(api_key_env)) or api_key_env not in allow[host]:
        raise _refuse(KEY_NOT_BOUND) from None
    return base_url
