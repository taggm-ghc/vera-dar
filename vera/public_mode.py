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
    req_per_hour_per_client: int = 30
    req_per_day_per_client: int = 180
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
    """Sliding minute and hour windows plus per-UTC-day counters, per client and global. Thread-safe."""

    def __init__(self, caps: Caps, clock=time.time):
        self.caps, self.clock = caps, clock
        self._lock = threading.Lock()
        self._minute: dict[str, list[float]] = {}
        self._hour: dict[str, list[float]] = {}  # OD-1031-PROMPTLIM: rolling 3600 s window, in-memory, per process
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
            hour = [t for t in self._hour.get(client, []) if now - t < 3600]
            if len(hour) >= c.req_per_hour_per_client:
                self._minute[client] = recent
                self._hour[client] = hour
                return False
            recent.append(now)
            hour.append(now)
            self._minute[client] = recent
            self._hour[client] = hour
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


# DAR-1021-ASKDIAG: client wait for POST /ask. Liveness bound, not a decision threshold. 60 s was below the server's
# bounded worst case on a cold or warming process (classifier 20 s + base-corpus single-flight wait 60 s + search
# 10 s, before any answer or claim-check call), so a slow answer surfaced as "service unavailable". 180 s covers
# that sum with margin; lower it once live /ask latencies are measured in the API log.
ASK_TIMEOUT_S = 180


def _ask_body(question: str, mode: str, spell_check: bool = True) -> dict:
    """SELFQ-RELEVANCE-1108: `spell_check` is a per-request choice ("search as typed" sends false); the body is unchanged
    when it is on, so older APIs keep accepting it."""
    return {"question": question, "mode": mode, **({} if spell_check else {"spell_check": False})}


def post_ask(base_url: str, question: str, mode: str, api_key: str, timeout: int = ASK_TIMEOUT_S,
             spell_check: bool = True):  # #79: search + link following
    """Single place the key is attached to an outgoing request. `api_key` is server-side (public)
    or typed by the user (dev); the caller decides, this never reads a widget."""
    return requests.post(
        f"{base_url.rstrip('/')}/ask",
        json=_ask_body(question, mode, spell_check),
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        timeout=timeout,
    )


def post_ask_tracked(base_url: str, question: str, mode: str, api_key: str, rid: str, session_token: str,
                     timeout: int = ASK_TIMEOUT_S, spell_check: bool = True):
    """PROG-1027 W4a: post_ask plus the two progress headers (request id, per-browser-session token)."""
    return requests.post(
        f"{base_url.rstrip('/')}/ask",
        json=_ask_body(question, mode, spell_check),
        headers={"Content-Type": "application/json", "X-API-Key": api_key,
                 "X-VERA-Request-Id": rid, "X-VERA-Session": session_token},
        timeout=timeout,
    )


# --- OD-1031-STREAM-ON: NDJSON stream client (contract v1.3). No total timeout: only connect and no-data bounds. ---------

STREAM_V = "1.3"
_LIT_STAGES = frozenset({"received", "scope", "corpus", "corpus_verdict", "search", "acquire", "write", "check", "retry", "done"})
_LIT_STATES = frozenset({"start", "update", "done", "skipped", "sufficient", "insufficient", "stale", "answered", "partial",
                         "no_answer", "declined", "error", "limited", "busy"})
_LIT_COUNTS = frozenset({"admitted", "found", "checked", "usable", "budget", "stored", "target", "queued",
                         "attempt", "shortlisted", "held", "step", "ceiling", "why"})
_RESULT_KINDS = frozenset({"busy", "error", "provider_error", "unavailable"})
_RESULT_STATUSES = frozenset({200, 429, 500, 502, 503})
_MAX_COUNT = 1_000_000
_MAX_LINE = 4_000_000
_MAX_EVENT_LINE = 65_536


class StreamNeverConnected(Exception):
    """No stream line was ever received (connect failure or non-200). Carries only a status int; never host or text."""

    def __init__(self, status: int | None = None):
        super().__init__("never_connected")
        self.status = status


class StreamLost(Exception):
    """The stream broke after it started. reason is a closed code: no_data | connection | ended."""

    def __init__(self, reason: str = "connection"):
        super().__init__("stream_lost")
        self.reason = reason


def _stream_sets():
    """Closed sets: the progress contract when present, else the literal sets above (union, so both are tolerated)."""
    stages, states, counts = set(_LIT_STAGES), set(_LIT_STATES), set(_LIT_COUNTS)
    try:
        from vera import progress

        stages |= {s.value for s in progress.Stage}
        states |= {s.value for s in progress.State}
        counts |= set(progress.COUNT_KEYS)
    except Exception:  # noqa: BLE001
        pass
    return stages, states, counts


def validate_stream_line(obj, sets=None) -> bool:
    """True only for a well-formed v1.3 event | hb | result line (closed sets, int counts)."""
    if not isinstance(obj, dict) or obj.get("v") != STREAM_V:
        return False
    t = obj.get("type")
    if t == "hb":
        extra = {k: v for k, v in obj.items() if k not in ("v", "type", "seq", "elapsed_s")}  # NOISY-UI: activity ints only
        return (type(obj.get("seq")) is int and type(obj.get("elapsed_s")) is int and obj["elapsed_s"] >= 0
                and all(k in ("act", "wait_s", "k", "n", "elapsed") and type(v) is int and 0 <= v <= _MAX_COUNT
                        for k, v in extra.items()))
    if t == "event":
        stages, states, counts = sets or _stream_sets()
        seq = obj.get("seq")
        if type(seq) is not int or seq < 0 or obj.get("stage") not in stages or obj.get("state") not in states:
            return False
        if not isinstance(obj.get("ts", ""), str):
            return False
        c = obj.get("counts", {})
        if not isinstance(c, dict):
            return False
        return all(k in counts and type(v) is int and 0 <= v <= _MAX_COUNT for k, v in c.items())
    if t == "result":
        st = obj.get("status")
        if type(st) is not int or st not in _RESULT_STATUSES:
            return False
        if st == 200:
            return isinstance(obj.get("body"), dict)
        return obj.get("kind") in _RESULT_KINDS
    return False


class AskStream:
    """Iterator of validated line dicts. `.dropped` counts invalid lines; `.close()` ends the connection."""

    def __init__(self, base_url, question, mode, api_key, *, connect_timeout, no_data_timeout, should_stop=None, session=None,
                 spell_check: bool = True):
        self._args = (base_url, question, mode, api_key)
        self._spell_check = spell_check
        self._ct, self._nd = connect_timeout, no_data_timeout
        self._should_stop = should_stop
        self._http = session if session is not None else requests
        self.dropped = 0
        self.stopped = False  # True when should_stop ended the iteration
        self._resp = None
        self._gen = self._run()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._gen)

    def close(self) -> None:
        try:
            self._gen.close()
        finally:
            self._close_resp()

    def _close_resp(self) -> None:
        r, self._resp = self._resp, None
        if r is not None:
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass

    def _run(self):
        base_url, question, mode, api_key = self._args
        sets = _stream_sets()
        try:
            try:
                r = self._http.post(
                    f"{base_url.rstrip('/')}/ask/stream", json=_ask_body(question, mode, self._spell_check),
                    headers={"Content-Type": "application/json", "X-API-Key": api_key, "Accept": "application/x-ndjson"},
                    stream=True, timeout=(self._ct, self._nd))
            except Exception:  # noqa: BLE001  no text kept: the message can carry the host
                raise StreamNeverConnected(None) from None
            self._resp = r
            if getattr(r, "status_code", None) != 200:
                st = getattr(r, "status_code", None)
                raise StreamNeverConnected(st if type(st) is int else None)
            got = False
            saw_result = False
            try:
                for raw in r.iter_lines():
                    got = True
                    if self._should_stop is not None and self._should_stop():
                        self.stopped = True
                        return
                    if not raw:
                        continue
                    if isinstance(raw, bytes):
                        if len(raw) > _MAX_LINE:
                            self.dropped += 1
                            continue
                        try:
                            raw = raw.decode("utf-8")
                        except UnicodeDecodeError:
                            self.dropped += 1
                            continue
                    try:
                        obj = json.loads(raw)
                    except ValueError:
                        self.dropped += 1
                        continue
                    if not validate_stream_line(obj, sets) or (obj["type"] != "result" and len(raw) > _MAX_EVENT_LINE):
                        self.dropped += 1
                        continue
                    if saw_result:  # exactly once, last
                        self.dropped += 1
                        continue
                    yield obj
                    if obj["type"] == "result":
                        saw_result = True
                        return
            except requests.exceptions.ReadTimeout:
                raise StreamLost("no_data") from None
            except (StreamLost, StreamNeverConnected, GeneratorExit):
                raise
            except Exception:  # noqa: BLE001
                if not got:
                    raise StreamNeverConnected(200) from None
                raise StreamLost("connection") from None
            raise StreamLost("ended")
        finally:
            self._close_resp()


def iter_ask_stream(base_url: str, question: str, mode: str, api_key: str, *, connect_timeout: float,
                    no_data_timeout: float, should_stop=None, session=None, spell_check: bool = True) -> AskStream:
    """POST /ask/stream and iterate validated v1.3 lines. Bounds: connect_timeout and per-read no_data_timeout only
    (no total timeout). Raises StreamNeverConnected / StreamLost; neither carries the key, host or exception text."""
    return AskStream(base_url, question, mode, api_key, connect_timeout=connect_timeout,
                     no_data_timeout=no_data_timeout, should_stop=should_stop, session=session, spell_check=spell_check)


class _Gone:
    """Sentinel type: the progress GET returned 404 (no record visible)."""
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self) -> str:
        return "GONE"

    def __bool__(self) -> bool:
        return False


GONE = _Gone()
_EVENT_KEYS = frozenset({"v", "seq", "ts", "stage", "state", "counts"})


def _valid_event(ev, max_count: int) -> bool:
    from vera import progress  # local import: keeps public_mode import-light

    if not isinstance(ev, dict) or set(ev) != _EVENT_KEYS:
        return False
    if ev["v"] != progress.CONTRACT_VERSION:
        return False
    seq = ev["seq"]
    if type(seq) is not int or seq < 1 or not isinstance(ev["ts"], str):
        return False
    try:
        stage, state = progress.Stage(ev["stage"]), progress.State(ev["state"])
    except (ValueError, TypeError):
        return False
    if state not in progress.VALID.get(stage, frozenset()):
        return False
    counts = ev["counts"]
    if not isinstance(counts, dict):
        return False
    for k, v in counts.items():
        if k not in progress.COUNT_KEYS or type(v) is not int or not 0 <= v <= max_count:
            return False
    return True


def get_progress(base_url: str, api_key: str, rid: str, session_token: str, after: int, timeout: float | None = None):
    """PROG-1027 W4a: list of validated events | GONE (404) | None (anything else). Whole body guarded; never raises and
    never carries exception text (the URL contains the host)."""
    try:
        from urllib.parse import quote

        from vera import progress

        cfg = progress.CFG
        t = timeout if timeout is not None else (cfg.get_timeout_s if cfg is not None else 2)
        max_count = cfg.max_count if cfg is not None else 10000
        r = requests.get(
            f"{base_url.rstrip('/')}/ask/progress/{quote(str(rid), safe='')}",
            params={"after": int(after)},
            headers={"X-API-Key": api_key, "X-VERA-Session": session_token},
            timeout=t,
        )
        if r.status_code == 404:
            return GONE
        if r.status_code != 200:
            return None
        body = r.json()
        evs = body.get("events") if isinstance(body, dict) else None
        if not isinstance(evs, list) or not all(_valid_event(e, max_count) for e in evs):
            return None
        return evs
    except Exception:  # noqa: BLE001
        return None


def get_corpus_summary(base_url: str, api_key: str, timeout: int = 30) -> dict | None:
    """Item #83: server-side GET of the API's corpus description (key attached here, never in a widget)."""
    if not (base_url and api_key):
        return None
    try:
        r = requests.get(f"{base_url.rstrip('/')}/corpus-summary", headers={"X-API-Key": api_key}, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except (requests.RequestException, ValueError):
        return None
