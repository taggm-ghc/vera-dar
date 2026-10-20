"""Item #84 D6: operator (owner-key) access control for memory review endpoints.

Identity: one principal, the operator, proven by the X-Owner-Key header matching the env var named by
memory.store.owner_key_env (default VERA_OWNER_KEY), compared in constant time. The public UI's VERA_API_KEY is
never accepted here. Deny-by-default: an unset, placeholder or short owner key, or one equal to VERA_API_KEY,
disables every operator route (503) before anything runs. Each route declares this dependency itself
(per-call authorisation); no key, IP or claim text is logged.
"""
import logging
import os
import secrets

from fastapi import Header, HTTPException

logger = logging.getLogger("vera")

DEFAULT_OWNER_KEY_ENV = "VERA_OWNER_KEY"
# Key-strength floor, not a traffic threshold: secrets.token_urlsafe(24) is 32 chars (192 bits). Plan s20.
OWNER_KEY_MIN_LENGTH = 32
_PLACEHOLDERS = {"changeme", "change-me", "change_me", "your-secret-here", "secret", "password", "todo", "fixme",
                 "placeholder", "test", "testkey", "owner", "ownerkey", "owner-key"}
DISABLED_DETAIL = "Operator endpoints are disabled on this server"


def classify_owner_key(key: str | None, api_key: str | None = None) -> str:
    """'missing', 'placeholder', 'weak', 'same_as_api_key' or 'ok'."""
    key = (key or "").strip()
    if not key:
        return "missing"
    if key.lower() in _PLACEHOLDERS:
        return "placeholder"
    if len(key) < OWNER_KEY_MIN_LENGTH:
        return "weak"
    if api_key and secrets.compare_digest(key.encode(), api_key.strip().encode()):
        return "same_as_api_key"
    return "ok"


def owner_key_env(memory_block: dict | None) -> str:
    return ((memory_block or {}).get("store") or {}).get("owner_key_env") or DEFAULT_OWNER_KEY_ENV


def make_verify_owner_key(memory_block: dict | None, env=os.environ):
    """FastAPI dependency factory. The env is read on every call, so rotating or removing the key takes effect
    without a code change."""
    var = owner_key_env(memory_block)
    _logged: set = set()

    def verify_owner_key(x_owner_key: str | None = Header(default=None, alias="X-Owner-Key")):
        expected = env.get(var)
        status = classify_owner_key(expected, env.get("VERA_API_KEY"))
        if status != "ok":
            if status not in _logged:  # once per status per process: anonymous calls cannot flood the log
                _logged.add(status)
                logger.warning("operator routes disabled: %s is %s", var, status)
            raise HTTPException(status_code=503, detail=DISABLED_DETAIL)
        if not x_owner_key or not secrets.compare_digest(x_owner_key.encode(), expected.strip().encode()):
            raise HTTPException(status_code=401, detail="Invalid or missing owner key")

    return verify_owner_key
