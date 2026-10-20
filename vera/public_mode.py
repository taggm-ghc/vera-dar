"""Public-mode access control for VERA-DAR (item #72 W2; OWASP LLM10 caps, key and URL exposure F72-5).

Mode: VERA_PUBLIC_MODE. SAFE DEFAULT = PUBLIC when unset. Only an explicit 0/false/no/off/dev/local
turns dev mode on. In public mode the UI has no key widget and no editable URL; the Streamlit
process attaches X-API-Key from its own environment; the API enforces per-client and per-day caps.

Client identity (recorded limitation): the peer address of the TCP connection by default. Behind a
reverse proxy that is the proxy, so all visitors would share one limit. Set trust_proxy_headers=true
(config) or VERA_TRUST_PROXY=1 only when a trusted proxy overwrites X-Forwarded-For; then the entry
`trusted_proxy_hops` from the right is used (not the left, which a client can forge). Shared-IP
clients (NAT, campus) share a limit. In-memory counters: per process, reset on restart, not shared
across workers.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from fastapi import HTTPException, Request

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "public_mode.json"
DEFAULT_LOCAL_URL = "http://127.0.0.1:8001"
_DEV_VALUES = {"0", "false", "no", "off", "dev", "local"}
_TRUE = {"1", "true", "yes", "on"}


def is_public_mode(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("VERA_PUBLIC_MODE", "")).strip().lower() not in _DEV_VALUES


@dataclass(frozen=True)
class Caps:
    req_per_min_per_client: int = 6
    req_per_day_per_client: int = 40
    req_per_day_global: int = 400
    cost_usd_per_day_global: float = 2.0
    cost_usd_per_day_per_client: float = 0.25
    trust_proxy_headers: bool = False
    trusted_proxy_hops: int = 1
    rate_limit_detail: str = "Rate limit reached. Please try again later."


def load_caps(path: Path = CONFIG_PATH, env=None) -> Caps:
    env = os.environ if env is None else env
    raw = json.loads(Path(path).read_text())
    fields = {k: raw[k] for k in Caps.__dataclass_fields__ if k in raw}
    if str(env.get("VERA_TRUST_PROXY", "")).strip().lower() in _TRUE:
        fields["trust_proxy_headers"] = True
    return Caps(**fields)


def client_id(request: Request, caps: Caps) -> str:
    peer = request.client.host if request.client else "unknown"
    if caps.trust_proxy_headers:
        parts = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        hops = max(1, caps.trusted_proxy_hops)
        if len(parts) >= hops:
            return parts[-hops]
    return peer


class CapLimiter:
    """Sliding minute window plus per-UTC-day counters, per client and global. Thread-safe."""

    def __init__(self, caps: Caps, clock=time.time):
        self.caps, self.clock = caps, clock
        self._lock = threading.Lock()
        self._minute: dict[str, list[float]] = {}
        self._day = ""
        self._day_req: dict[str, int] = {}
        self._day_cost: dict[str, float] = {}
        self._global_req = 0
        self._global_cost = 0.0

    def _roll(self, now: float) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        if day != self._day:
            self._day, self._day_req, self._day_cost = day, {}, {}
            self._global_req, self._global_cost = 0, 0.0

    def allow(self, client: str) -> bool:
        c, now = self.caps, self.clock()
        with self._lock:
            self._roll(now)
            if (self._global_req >= c.req_per_day_global or self._global_cost >= c.cost_usd_per_day_global
                    or self._day_req.get(client, 0) >= c.req_per_day_per_client
                    or self._day_cost.get(client, 0.0) >= c.cost_usd_per_day_per_client):
                return False
            recent = [t for t in self._minute.get(client, []) if now - t < 60]
            if len(recent) >= c.req_per_min_per_client:
                self._minute[client] = recent
                return False
            recent.append(now)
            self._minute[client] = recent
            self._global_req += 1
            self._day_req[client] = self._day_req.get(client, 0) + 1
            return True

    def record_cost(self, client: str, usd: float) -> None:
        with self._lock:
            self._roll(self.clock())
            self._global_cost += usd
            self._day_cost[client] = self._day_cost.get(client, 0.0) + usd


_limiter: CapLimiter | None = None
_limiter_lock = threading.Lock()


def get_limiter() -> CapLimiter:
    global _limiter
    with _limiter_lock:
        if _limiter is None:
            _limiter = CapLimiter(load_caps())
        return _limiter


def reset_limiter(limiter: CapLimiter | None = None) -> None:
    global _limiter
    with _limiter_lock:
        _limiter = limiter


def enforce_caps(request: Request) -> None:
    """FastAPI dependency for /ask. Public mode only; 429 with a generic message, no provider text."""
    if not is_public_mode():
        return
    lim = get_limiter()
    cid = client_id(request, lim.caps)
    if not lim.allow(cid):
        raise HTTPException(status_code=429, detail=lim.caps.rate_limit_detail)
    request.state.vera_client_id = cid


def record_request_cost(request: Request, usd: float) -> None:
    cid = getattr(request.state, "vera_client_id", None)
    if cid is not None:
        get_limiter().record_cost(cid, float(usd))


# --- Streamlit side -------------------------------------------------------------------------

def upstream_base_url(env=None, local_url_file: Path | None = None) -> str:
    """Public mode: only VERA_API_BASE_URL (server env) or the local default; never user input."""
    env = os.environ if env is None else env
    url = str(env.get("VERA_API_BASE_URL", "")).strip().rstrip("/")
    if url:
        return url
    if local_url_file is not None:
        try:
            return local_url_file.read_text().strip().rstrip("/") or DEFAULT_LOCAL_URL
        except OSError:
            pass
    return DEFAULT_LOCAL_URL


def post_ask(base_url: str, question: str, mode: str, api_key: str, timeout: int = 60):  # #79: search + link following
    """Single place the key is attached to an outgoing request. `api_key` is server-side (public)
    or typed by the user (dev); the caller decides, this never reads a widget."""
    return requests.post(
        f"{base_url.rstrip('/')}/ask",
        json={"question": question, "mode": mode},
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        timeout=timeout,
    )


def get_corpus_summary(base_url: str, api_key: str, timeout: int = 30) -> dict | None:
    """Item #83: server-side GET of the API's corpus description (key attached here, never in a widget)."""
    if not (base_url and api_key):
        return None
    try:
        r = requests.get(f"{base_url.rstrip('/')}/corpus-summary", headers={"X-API-Key": api_key}, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except (requests.RequestException, ValueError):
        return None
