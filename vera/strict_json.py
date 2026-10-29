"""STRICTJSON-1027: one strict JSON loader for untrusted model / provider replies.

A plain json.loads keeps the LAST of duplicate keys and accepts NaN / Infinity and overflowing numbers (1e999 -> inf),
so a reply can say "unsupported" and then "supported" for the same field. loads() rejects, at every nesting level,
duplicate keys (exact, or equal after normalising), non-ASCII keys, non-finite numbers, oversize input and
pathological nesting. Every rejection is a StrictJSONError (a ValueError, so existing `except ValueError` paths still
fail closed) whose message is a fixed code, never key, value or reply text. Callers log `exc.code` only.

The normaliser and pairs hook are moved verbatim from HOTFIX-DUPKEY-1027 (scope_router); scope_router now imports
them from here. Named code constants only: the security floor must not be config-weakenable.
"""
import json
import math
import re
import unicodedata

_DROP_CATS = frozenset({"Cf", "Cc", "Mn"})  # SEC-H-1: format, control and combining-mark characters are dropped
CODES = frozenset({"not_str", "oversize", "duplicate_key", "non_ascii_key", "non_finite", "bad_json", "too_deep",
                   "bad_type"})


class StrictJSONError(ValueError):
    """A rejected reply. str(exc) == exc.code, a fixed string from CODES (module literals only)."""

    def __init__(self, code: str):
        assert code in CODES
        super().__init__(code)
        self.code = code


def base_norm(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", s) if unicodedata.category(ch) not in _DROP_CATS)


def norm_key(s: str) -> str:
    """Keys and equality checks: no whitespace at all, case-folded."""
    return re.sub(r"\s+", "", base_norm(s)).casefold()


def _pairs_hook(pairs):  # json applies it at every nesting level
    keys = [norm_key(k) for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise StrictJSONError("duplicate_key")
    if any(not k.isascii() for k, _ in pairs):
        raise StrictJSONError("non_ascii_key")
    return dict(pairs)


def _pairs_hook_exact(pairs):  # free-text keys (e.g. OpenAlex abstract_inverted_index words): exact duplicates only
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise StrictJSONError("duplicate_key")
    return dict(pairs)


def _no_constant(_name):  # NaN / Infinity / -Infinity literals
    raise StrictJSONError("non_finite")


def _finite_float(s: str) -> float:  # 1e999 / -1e999 overflow to inf
    v = float(s)
    if not math.isfinite(v):
        raise StrictJSONError("non_finite")
    return v


def loads(text, *, allow_constants: bool = False, max_chars: int | None = None, keys: str = "normalised"):
    """Decoded JSON, or StrictJSONError. keys="exact" (A1c-2) is for objects whose keys are free text: it rejects only
    EXACT duplicate keys (no normalisation, no ASCII rule); non-finite numbers are still rejected. Default "normalised"
    is unchanged. allow_constants=True keeps json's own number handling (NaN, 1e999 accepted)."""
    if keys not in ("normalised", "exact"):
        raise StrictJSONError("bad_type")
    if not isinstance(text, str):
        raise StrictJSONError("not_str")
    if max_chars is not None and len(text) > max_chars:  # checked before parsing
        raise StrictJSONError("oversize")
    kw = {} if allow_constants else {"parse_constant": _no_constant, "parse_float": _finite_float}
    try:
        return json.loads(text, object_pairs_hook=_pairs_hook_exact if keys == "exact" else _pairs_hook, **kw)
    except StrictJSONError:
        raise
    except RecursionError:
        raise StrictJSONError("too_deep") from None
    except TypeError:
        raise StrictJSONError("bad_type") from None
    except ValueError:  # JSONDecodeError, int-digit limit
        raise StrictJSONError("bad_json") from None


def loads_or_none(text, **kw):
    """loads(text, **kw), or None when the text is rejected. Never raises StrictJSONError."""
    try:
        return loads(text, **kw)
    except StrictJSONError:
        return None


def response_json(resp, **kw):
    """Strict equivalent of requests' resp.json(): decodes resp.text with loads(**kw). A response with no text body
    (e.g. a test double that only has .json()) falls back to resp.json(); a real empty body fails there as before."""
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return loads(text, **kw)
    return resp.json()
