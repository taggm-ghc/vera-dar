"""ACQ-1027 K3/K4: pure URL scrubbing and secret redaction for acquisition adapters. No I/O, no network.

scrub() removes key-like query parameters from a URL; redact_text() does the same inside free text (exception
messages, log lines); key_like_value() flags config values that look like secrets. The GovInfo link mapping is
deliberately not here (no new hosts)."""
from __future__ import annotations

import contextlib
import logging
import re
import threading
from typing import Iterable
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

KEY_PARAMS = frozenset({"api_key", "apikey", "key", "token", "access_token", "mailto"})
REDACTED = "[redacted]"
_URL_RE = re.compile(r"https?://[^\s'\"<>)\]]+")
_PARAM_RE = re.compile(r"([?&;#\s,{(']|^)([A-Za-z0-9_.%\-]+)(['\"]?\s*[=:]\s*['\"]?)([^&;?#\s'\",})]+)")


def _keys(loaded_keys: Iterable[str]) -> list[str]:
    return [k for k in (loaded_keys or ()) if isinstance(k, str) and len(k.strip()) >= 4]


_SECRET_WORDS = frozenset({"key", "apikey", "token", "secret", "signature", "sig", "password", "passwd", "pwd", "auth",
                           "authorization", "authtoken", "credential", "credentials", "mailto"})
_SECRET_SUFFIXES = ("apikey", "token", "secret", "password", "signature")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _is_key_param(name: str) -> bool:
    """D3: a param NAME that denotes a secret: after percent-decoding, any word (split on separators and camelCase)
    in the secret-word set, or a word ending in apikey/token/secret/password/signature. 'keyword', 'monkey', 'author'
    are ordinary (whole-word matching, not substring)."""
    n = unquote(str(name)).strip()
    if n.lower().replace("-", "_") in KEY_PARAMS:
        return True
    for w in re.split(r"[^A-Za-z0-9]+", _CAMEL.sub("_", n)):
        w = w.lower()
        if w and (w in _SECRET_WORDS or (w.endswith(_SECRET_SUFFIXES))):
            return True
    return False


def has_key_like_param(url: str) -> bool:
    try:
        return any(_is_key_param(k) for k, _ in parse_qsl(urlsplit(str(url)).query, keep_blank_values=True))
    except Exception:  # noqa: BLE001
        return True  # unparseable: treat as suspect


def scrub(url: str, loaded_keys: Iterable[str] = ()) -> str:
    """Drop key-like query params and any param whose value equals a loaded key. Pure; returns str."""
    url = str(url)
    keys = _keys(loaded_keys)
    try:
        parts = urlsplit(url)
        kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                if not _is_key_param(k) and v not in keys]
        out = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept), parts.fragment))
    except Exception:  # noqa: BLE001
        out = url
    for k in keys:
        out = out.replace(k, REDACTED)
    return out


def redact_text(text, keys: Iterable[str] = ()) -> str:
    """Replace key values and key-like params inside arbitrary text."""
    s = str(text)
    ks = sorted(_keys(keys), key=len, reverse=True)
    for k in ks:
        s = s.replace(k, REDACTED)
    s = _URL_RE.sub(lambda m: _redact_url_in_text(m.group(0)), s)
    s = _PARAM_RE.sub(lambda m: (f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}" if _is_key_param(m.group(2))
                                 else m.group(0)), s)
    return s


_URL_PARAM = re.compile(r"([?&;#])([^=&;#\s]*)=([^&;#\s]*)")
_USERINFO = re.compile(r"^(https?://)[^/?#@\s]*@", re.I)


def _redact_url_in_text(u: str) -> str:
    """Marker-preserving URL redaction inside text: key-like param values (also percent-encoded / hyphenated /
    mixed-case names, in the query or fragment) and any userinfo become [redacted]; nothing else is touched."""
    u = _USERINFO.sub(lambda m: m.group(1) + REDACTED + "@", u)

    def sub(m):
        return f"{m.group(1)}{m.group(2)}={REDACTED}" if _is_key_param(unquote(m.group(2))) else m.group(0)
    return _URL_PARAM.sub(sub, u)


_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*|R-[A-Z]+-[A-Z0-9_]+")  # env var names; R-<AREA>-<CODE> reason codes
_ENV_ONLY = re.compile(r"[A-Z][A-Z0-9_]*")
_REASON_CODE = re.compile(r"R-[A-Z]+-[A-Z0-9_]+")
_WORDLIKE = re.compile(r"(?:[a-z]+|v[0-9]+)(?:[-_](?:[a-z]+|v[0-9]+))+")  # kebab/snake identifiers made of words (+vN)
_KEY_PATH = re.compile(r"(?i)key|token|secret|password")
_PATTERNS = re.compile(r"(?:\bsk-[A-Za-z0-9_-]{8,}|\bghp_[A-Za-z0-9]{16,}|\bBearer\s+[A-Za-z0-9._~+/=-]{12,})")
_LONG = re.compile(r"[A-Za-z0-9_-]{24,}")
# Config paths whose values are sha-256 pins (hex digests by design); the only place a bare hex digest is allowed.
SHA_PIN_PATHS = frozenset({
    "pins.manifest_sha256", "pins.sums_sha256", "scope_set.sha256", "scope_set_v2.sha256", "freeze_pin.file_sha256"})
_SHA_PIN_DOC = re.compile(r"pins\.documents\.[A-Za-z0-9_]+\.text_sha256")


def _pin_path(path: str) -> bool:
    p = path.lstrip(".")
    return p in SHA_PIN_PATHS or bool(_SHA_PIN_DOC.fullmatch(p)) or any(p.endswith("." + x) for x in SHA_PIN_PATHS)


# Explicit, narrow exemption: values (state names) under exactly micro_topic.registry_table.refused_state_map.<category code>. Only a
# refused state name or a category-code-shaped value passes there; any credential-like value under that path is still refused.
_REFUSED_MAP_PATH = re.compile(r"micro_topic\.registry_table\.refused_state_map\.[a-z][a-z_]{1,31}")
_CATEGORY_CODE_VALUE = re.compile(r"[a-z][a-z_]{1,31}")


_REFUSED_STATE_NAMES = ("banned", "prohibited", "secrets")  # equals the registry module's REFUSED_STATES (a test pins it; no import here)


def _refused_map_value_ok(s: str, path: str) -> bool:
    return bool(_REFUSED_MAP_PATH.fullmatch(path.lstrip("."))) and (s in _REFUSED_STATE_NAMES or bool(_CATEGORY_CODE_VALUE.fullmatch(s)))


def key_like_value(v, path: str | None = None) -> bool:
    """Does a config value look like a secret?
    path=None: shape-only heuristic (weak: lowercase kebab/snake and hex digests pass).
    path given (dotted config path; config scans always pass it): path rules first. A leaf named *_env must hold an
    env var NAME. Any other value under a path containing key|token|secret|password is refused unless it is an env var
    name (a leaf starting with '_' is a comment and uses the shape rules). Otherwise only R-<AREA>-<CODE> reason codes,
    word-like kebab/snake identifiers and allowlisted sha-pin paths (SHA_PIN_PATHS) are exempt from the long-token test."""
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not s:
        return False
    if _PATTERNS.search(s):
        return True
    if re.match(r"(?i)https?://", s):
        return has_key_like_param(s)
    if path is None:
        if not _LONG.fullmatch(s):
            return False
        if _ENV_NAME.fullmatch(s) or re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)+", s):
            return False
        return not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", s)
    if _REFUSED_MAP_PATH.fullmatch(path.lstrip(".")):
        return not _refused_map_value_ok(s, path)
    leaf = path.rsplit(".", 1)[-1]
    if leaf.lower().endswith("_env"):
        return not _ENV_ONLY.fullmatch(s)
    if not leaf.startswith("_") and _KEY_PATH.search(path):
        return not _ENV_NAME.fullmatch(s)
    if not _LONG.fullmatch(s):
        return False
    if _REASON_CODE.fullmatch(s) or _WORDLIKE.fullmatch(s) or _pin_path(path):
        return False
    return True


_URLLIB3 = ("urllib3", "urllib3.connection", "urllib3.connectionpool")
_ACTIVE: dict = {}
_ACTIVE_LOCK = threading.Lock()


class _RedactFilter(logging.Filter):
    """Thread-safe: never touches logger levels. A record that carries an active key or a key-like param is rewritten
    (message redacted, args dropped) before any handler sees it."""

    def filter(self, record: logging.LogRecord) -> bool:
        with _ACTIVE_LOCK:
            keys = list(_ACTIVE)
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            msg = str(record.msg)
        red = redact_text(msg, keys)
        if red != msg:
            record.msg, record.args = red, None
        return True


_FILTER = _RedactFilter()


@contextlib.contextmanager
def quiet_urllib3(keys: Iterable[str] = ()):
    """urllib3 logs the raw request URL at DEBUG. For the duration, records on its loggers are redacted (loaded `keys`
    and key-like params) by a filter; global logger levels are not changed, so concurrent callers cannot race."""
    ks = _keys(keys)
    for n in _URLLIB3:
        lg = logging.getLogger(n)
        if _FILTER not in lg.filters:
            lg.addFilter(_FILTER)
    with _ACTIVE_LOCK:
        for k in ks:
            _ACTIVE[k] = _ACTIVE.get(k, 0) + 1
    try:
        yield
    finally:
        with _ACTIVE_LOCK:
            for k in ks:
                if _ACTIVE.get(k, 0) <= 1:
                    _ACTIVE.pop(k, None)
                else:
                    _ACTIVE[k] -= 1
