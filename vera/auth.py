"""API-key classification and the /ask authentication dependency, shared
schema so key-format rules and error messages live in one place instead of
being duplicated between validation and error messaging.
"""
import logging
import os
import re
import secrets

from fastapi import Header, HTTPException

logger = logging.getLogger("vera")


_OPENAI_KEY_PLACEHOLDERS = {
    "your-api-key-here", "your_api_key_here", "your-openai-api-key",
    "youropenaikey", "changeme", "change-me", "change_me", "todo", "fixme",
    "placeholder", "insert-key-here", "insert_your_key_here", "api-key-here",
    "<your-api-key>", "<api_key>", "none", "null", "n/a", "openai_api_key",
}
_OPENAI_KEY_PLACEHOLDER_RE = re.compile(r"^sk-x{10,}$", re.IGNORECASE)
_OPENAI_KEY_MIN_LENGTH = 40  # real OpenAI keys run 50+ chars; this is a floor, not a format check


def classify_openai_api_key(key: str | None) -> str:
    """Classify OPENAI_API_KEY as 'missing', 'placeholder', 'malformed', or 'ok'."""
    if key is None:
        return "missing"
    key = key.strip()
    if not key:
        return "missing"
    if key.lower() in _OPENAI_KEY_PLACEHOLDERS or _OPENAI_KEY_PLACEHOLDER_RE.match(key):
        return "placeholder"
    if not key.startswith("sk-") or len(key) < _OPENAI_KEY_MIN_LENGTH:
        return "malformed"
    return "ok"


OPENAI_KEY_ERROR_DETAIL = {
    "missing": "Server misconfigured: OPENAI_API_KEY is not set.",
    "placeholder": "Server misconfigured: OPENAI_API_KEY still holds a placeholder value.",
    "malformed": "Server misconfigured: OPENAI_API_KEY does not look like a valid OpenAI key.",
}

_VERA_KEY_PLACEHOLDERS = {
    "changeme", "change-me", "change_me", "your-secret-here", "secret",
    "password", "todo", "fixme", "placeholder", "test", "testkey",
}
_VERA_KEY_MIN_LENGTH = 16


def classify_vera_api_key(key: str | None) -> str:
    """Classify VERA_API_KEY as 'missing', 'placeholder', 'weak', or 'ok'."""
    if key is None:
        return "missing"
    key = key.strip()
    if not key:
        return "missing"
    if key.lower() in _VERA_KEY_PLACEHOLDERS:
        return "placeholder"
    if len(key) < _VERA_KEY_MIN_LENGTH:
        return "weak"
    return "ok"


def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    expected = os.getenv("VERA_API_KEY")
    status = classify_vera_api_key(expected)
    if status != "ok":
        logger.error("Rejected /ask: VERA_API_KEY is %s", status)
        raise HTTPException(status_code=500, detail="Server misconfigured: VERA_API_KEY is not set to a usable secret")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
