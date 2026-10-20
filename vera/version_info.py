"""Deployed-commit marker. Reads Render's built-in RENDER_GIT_COMMIT; no secrets, no host."""
import os
import re

UNKNOWN = "unknown"
_SHA = re.compile(r"^[0-9a-fA-F]{7,40}$")


def deployed_commit(env=None) -> str:
    """Full commit hash from RENDER_GIT_COMMIT, or "unknown" when unset or not a hex hash."""
    v = ((env if env is not None else os.environ).get("RENDER_GIT_COMMIT") or "").strip()
    return v.lower() if _SHA.match(v) else UNKNOWN


def short_commit(commit: str) -> str:
    return commit[:7] if commit and commit != UNKNOWN else UNKNOWN
