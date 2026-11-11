"""ENV-PATHS-1110: containment check for any path or file URI that comes from an env var or config.

`safe_path(value, allowed_roots, key=...)` resolves the value (symlinks followed) and returns the resolved Path only if it lies
inside one of the named allowed roots. Otherwise it raises PathRefused. Callers must use the RETURNED Path, never the original
string. There is no default root: every call site names its own, anchored on Path(__file__), never on cwd or the env value.

Refusals are counted per (key, reason code) and logged as `safe_path refused key=<K> reason=<code>`. The value, the resolved
path and the roots are never logged or placed in the exception. The counters are in-process only (refusal_counts()); they are
not exposed on any endpoint; the post-deploy check greps the service log for the line.

Residual risk: a symlink swapped between the check and the later open (TOCTOU). Writes/appends use append_text(), which opens
with O_NOFOLLOW on the final component; reads of directories baked into the image are not hardened further.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

logger = logging.getLogger(__name__)

# Test-only hook: tests monkeypatch this list (for example to [tmp_path]). It is deliberately a module attribute and is never
# read from the environment or a config file, because an env-settable allowlist would defeat the control.
EXTRA_TEST_ROOTS: list = []

MODES = ("read", "write", "append")
_SCHEME_URI = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
_DRIVE = re.compile(r"^[A-Za-z]:")
_ENCODED_BAD = re.compile(r"%(2e|2f|5c|00)", re.IGNORECASE)
_KEY_OK = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

_lock = threading.Lock()
_counts: Counter = Counter()


class PathRefused(Exception):
    """Carries the key name and a reason code only, never the value."""

    def __init__(self, key: str, reason: str):
        self.key, self.reason = key, reason
        super().__init__(f"safe_path refused key={key} reason={reason}")


def refusal_counts() -> dict:
    """Snapshot {(key, reason): n} of refusals in this process."""
    with _lock:
        return dict(_counts)


def reset_refusal_counts() -> None:
    with _lock:
        _counts.clear()


def _refuse(key: str, reason: str):
    k = key if isinstance(key, str) and _KEY_OK.match(key) else "invalid-key"
    with _lock:
        _counts[(k, reason)] += 1
    logger.warning("safe_path refused key=%s reason=%s", k, reason)
    raise PathRefused(k, reason)


def _uri_to_path(value: str, key: str) -> str:
    if value.lower().startswith("file:") and not value.lower().startswith("file://"):
        _refuse(key, "uri_form")
    parts = urlsplit(value)
    if parts.scheme.lower() != "file":
        _refuse(key, "scheme")
    if parts.netloc.lower() not in ("", "localhost"):
        _refuse(key, "uri_host")
    if parts.query or parts.fragment:
        _refuse(key, "uri_extra")
    if _ENCODED_BAD.search(parts.path):
        _refuse(key, "uri_encoded")
    decoded = unquote(parts.path)  # decoded exactly once
    if "\x00" in decoded or "\\" in decoded:
        _refuse(key, "uri_encoded")
    if _DRIVE.match(decoded.lstrip("/")):
        _refuse(key, "drive_letter")
    return decoded


def safe_path(value, allowed_roots, *, key: str, must_exist: bool = False, mode: str = "read") -> Path:
    """Return the resolved Path for `value` if it is inside one of `allowed_roots`, else raise PathRefused."""
    if mode not in MODES:
        _refuse(key, "bad_mode")
    if not allowed_roots:
        _refuse(key, "no_roots")
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        _refuse(key, "not_str")
    if not value:
        _refuse(key, "empty")
    if "\x00" in value:
        _refuse(key, "nul")
    if value != value.strip():
        _refuse(key, "whitespace")
    if _DRIVE.match(value):
        _refuse(key, "drive_letter")
    if _SCHEME_URI.match(value) or value.lower().startswith("file:"):
        value = _uri_to_path(value, key)
    if "\x00" in value:
        _refuse(key, "nul")
    if not os.path.isabs(value):
        _refuse(key, "relative")  # a relative value would depend on cwd
    try:
        roots = [Path(r).resolve() for r in list(allowed_roots) + list(EXTRA_TEST_ROOTS)]
        p = Path(value)
        if mode == "read":
            target = p.resolve()
        else:
            parent = p.parent.resolve()
            if not parent.is_dir():
                _refuse(key, "parent_missing")
            target = parent / p.name
            if p.name in ("", ".", ".."):
                _refuse(key, "bad_name")
            if target.is_symlink():
                _refuse(key, "symlink_final")
    except PathRefused:
        raise
    except (OSError, RuntimeError, ValueError):
        _refuse(key, "resolve_error")
    if not any(target.is_relative_to(r) for r in roots):
        _refuse(key, "outside_root")
    if must_exist and mode == "read" and not target.exists():
        _refuse(key, "missing")
    return target


def append_text(path: Path, text: str) -> None:
    """Append `text` to a path already returned by safe_path(mode="append"); never follows a symlink at the final component."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(text)
