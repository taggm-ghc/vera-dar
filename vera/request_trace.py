"""PERF-MISS-1104-TRACE (owner decision OD-1104-TRACE, 2026-11-04): per-request diagnostic trace.

One trace per /ask or /ask/stream request:
  tid = start(question, meta)        # main.py entry points only; returns the server-generated request id
  event(stage, code, counts, detail)  # anywhere inside the request; also fed automatically by progress.emit
  finish(outcome_branch, message_code)

Request id: a server-generated random hex string (secrets.token_hex(16)), separate from the client's x-vera-request-id
header (recorded as `client_request_id` when present and well formed). It is owner-visible only: the response header
X-VERA-Trace-Id, the `vera_trace` log lines and the DB tables. It is never put into a response body or the UI.

Question text (OD-1104-TRACE (b)): kept for diagnosing errors, readable by the owner only, purged after 30 days by an
owner-run script. It is in the per-request DB row and the owner route's recent window, never in event details. The start
event's log line carries it only when config log_sink.include_question is true (ships false: host logs are not purgeable
and R-028 tests forbid the question in logs); otherwise it carries question_chars.

Detail is bounded and scrubbed: identifier keys only; keys naming secrets, URLs, hosts, DSNs or credentials are
dropped; string values are cut to `max_str_chars` with URLs, hosts, IP addresses, bearer tokens and key-shaped strings
replaced by fixed markers; lists are cut; total JSON size is capped. It may hold passage/document ids, scores, reason
codes, provider status, latency, model and tokens.

Sinks:
  * log sink (config log_sink.enabled, ships on): one JSON line per event on logger "vera.trace", prefixed "vera_trace".
  * DB sink (config db_sink.enabled, ships OFF until the owner applies migration 015): app-path INSERTs through the
    write-only account, buffered in a bounded queue and written by one background thread. It never blocks or fails a
    request: a full queue or a DB error drops rows and bumps a counter.
Every public call never raises.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import secrets
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from vera import retention_purge

log = logging.getLogger("vera.trace")
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "request_trace.json"
LOG_PREFIX = "vera_trace"
HEADER = "X-VERA-Trace-Id"
SCHEMA_VERSION = "1"

TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}\Z")
CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}\Z")
IDENT_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}\Z")
TABLE_RE = re.compile(r"^vera_vjay\.[a-z_][a-z0-9_]{0,62}\Z")
_META_KEYS = ("endpoint", "client_request_id", "stream")
_ENDPOINTS = ("/ask", "/ask/stream", "other")

DEFAULT_LIMITS = {"max_question_chars": 4000, "max_events_per_request": 400, "max_detail_bytes": 2048,
                  "max_detail_keys": 24, "max_str_chars": 200, "max_list_items": 20, "recent_max": 50,
                  "recent_ttl_s": 900}

# --------------------------------------------------------------------------------------------- scrubbing
# Key names that may carry a secret, a URL/host or a credential (dropped whole, value never inspected).
_DENY_KEY_RE = re.compile(r"(password|passwd|secret|api_?key|authorization|cookie|credential|bearer|dsn|url|uri|"
                          r"host|endpoint|session|(^|_)(key|token|pw|pwd|auth)$)")
_VALUE_SUBS = (
    (re.compile(r"(?i)\b(?:password|passwd|pwd|secret|token|api_?key)\s*[=:]\s*\S+"), "<secret>"),  # libpq keyword=value
    (re.compile(r"(?i)\b[a-z][a-z0-9+.-]{1,15}://\S+"), "<url>"),            # any scheme://... incl. postgres URIs
    (re.compile(r"(?i)\bbearer\s+\S+"), "<secret>"),
    (re.compile(r"\b(?:sk|gsk|pk|rk|ghp|gho|xox[abpr])[-_][A-Za-z0-9_-]{8,}"), "<secret>"),
    (re.compile(r"\b[^\s@]+@[^\s@]+\.[A-Za-z]{2,}\b"), "<email>"),             # user@host forms (also DSN leftovers)
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<ip>"),
    (re.compile(r"(?i)\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|ai|dev|app|cloud|internal|local|edu|gov|co|uk|de|info|"
                r"biz|xyz|site|online|tech|onrender\.com)\b"), "<host>"),
    (re.compile(r"[A-Za-z0-9+/_=-]{48,}"),                                       # key-shaped blobs; pure hex (hashes) kept
     lambda m: m.group(0) if re.fullmatch(r"[0-9a-fA-F]+", m.group(0)) else "<long>"),
)


def scrub_str(s: str, max_chars: int) -> str:
    out = str(s)
    for rx, rep in _VALUE_SUBS:
        out = rx.sub(rep, out)
    out = "".join(ch if ch.isprintable() else " " for ch in out)
    return out[:max_chars]


# DENYALL-TRACE-1104 (owner standard 2026-11-04: "Deny All; approve specified exceptions-from-Deny"):
# a string detail value is stored only if it is a short lowercase token; everything else becomes "<str>".
STR_OK_RE = re.compile(r"^[a-z0-9_:.-]{1,64}$")
STR_PLACEHOLDER = "<str>"
_HOSTLIKE_RE = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}$")  # extra layer: dotted names ending in a letter-only label


def _safe_str(v: str) -> str:
    """Deny-all string gate. The secret scan stays as an extra layer: any value it would alter, or that looks like a
    host name, is replaced too. Known limit: an opaque lowercase token (e.g. a short password) matches the token
    shape and is stored; only the per-stage key allowlist keeps such values out."""
    if not STR_OK_RE.fullmatch(v) or _HOSTLIKE_RE.match(v) or scrub_str(v, 10**6) != v:
        return STR_PLACEHOLDER
    return v


# Narrow exception to the string gate: for these keys only, a value that EXACTLY equals a provider/model id in the
# provider-chain config is stored (ids such as "openai/gpt-oss-20b" fail the token shape). Derived at load time.
ID_KEYS = ("model", "check_model", "provider")
CHAIN_PATH = Path(__file__).resolve().parent.parent / "config" / "ask-provider-chain.json"


def _load_chain_ids(path: Path | None = None) -> frozenset:
    ids: set = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("provider", "model") and isinstance(v, str) and v and not v.startswith("<") and len(v) <= 100:
                    ids.add(v)
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)
    try:
        walk(json.loads(Path(path or CHAIN_PATH).read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001  (missing or invalid file: no exception ids, fail closed)
        pass
    return frozenset(ids)


CHAIN_IDS: frozenset = _load_chain_ids()


def _key_ok(k) -> bool:
    return isinstance(k, str) and IDENT_RE.match(k) is not None and _DENY_KEY_RE.search(k) is None


def _key_allowlist():
    """LISTS-TO-DB-1104: trace.key_allowlist from the DB when enabled (per call, loader TTL cache), else CFG."""
    try:
        from vera import list_tables
        return list_tables.get("trace.key_allowlist", CFG.get("key_allowlist"))
    except Exception:  # noqa: BLE001  loader missing or faulty: keep the file value
        return CFG.get("key_allowlist")


def _stage_keys(stage) -> frozenset:
    """Allowed top-level detail keys for `stage` from config key_allowlist (missing or malformed = none: fail closed)."""
    al = _key_allowlist()
    ent = al.get(stage) if isinstance(al, dict) and isinstance(stage, str) else None
    ks = ent.get("keys") if isinstance(ent, dict) else None
    return frozenset(k for k in ks if isinstance(k, str)) if isinstance(ks, list) else frozenset()


def _nested_keys(stage, key):
    """Optional config `nested` map (stage entry): allowed keys inside the mapping stored under detail[key]. None = free
    identifier tokens (reason-code and phase-ms maps, whose keys are data)."""
    al = _key_allowlist()
    ent = al.get(stage) if isinstance(al, dict) and isinstance(stage, str) else None
    nm = ent.get("nested") if isinstance(ent, dict) else None
    ks = nm.get(key) if isinstance(nm, dict) else None
    return frozenset(k for k in ks if isinstance(k, str)) if isinstance(ks, list) else None


def _clean_value(v, lim: dict, depth: int, dk: list, nested=None):
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return round(v, 6) if v == v and abs(v) != float("inf") else None
    if isinstance(v, str):
        return _safe_str(v)
    if isinstance(v, (list, tuple, set, frozenset)):
        if depth >= 3:  # per_claim: list -> record -> cites list
            return f"<{type(v).__name__}>"
        items = list(v)[: lim["max_list_items"]]
        return [_clean_value(x, lim, depth + 1, dk, nested) for x in items]
    if isinstance(v, dict):
        if depth >= 2:
            return "<dict>"
        return _clean_mapping(v, lim, depth + 1, dk, nested)
    return f"<{type(v).__name__}>"


def _clean_mapping(d: dict, lim: dict, depth: int, dk: list, allowed, stage=None) -> dict:
    """`allowed` is the stage key allowlist at the top level (depth 0); nested mappings (reason-code maps, per-claim
    records) are walked under the same value rules, with keys limited to identifier tokens that pass the key deny list."""
    out: dict = {}
    for k, v in d.items():
        if len(out) >= lim["max_detail_keys"]:
            out["_truncated"] = True
            break
        if not _key_ok(k) or (allowed is not None and k not in allowed):
            dk[0] += 1
            _drop("detail_key")
            continue
        if k in ID_KEYS and isinstance(v, str) and v in CHAIN_IDS:
            out[k] = v
            continue
        out[k] = _clean_value(v, lim, depth, dk, _nested_keys(stage, k) if depth == 0 else None)
    return out


def clean_detail(detail, lim: dict | None = None, stage=None) -> dict | None:
    """Bounded copy of `detail` under deny-all: only keys allowlisted for `stage`, string values token-shaped. Unknown
    keys are dropped and counted in `dropped_keys`. None when empty or not a dict. Never raises."""
    lim = lim or _limits()
    try:
        if not isinstance(detail, dict) or not detail:
            return None
        dk = [0]
        out = _clean_mapping(detail, lim, 0, dk, _stage_keys(stage), stage)
        if dk[0]:
            out["dropped_keys"] = dk[0]
        cap = lim["max_detail_bytes"]
        while out and len(json.dumps(out, separators=(",", ":"), default=str)) > cap:
            keys = [k for k in out if k not in ("_truncated", "dropped_keys")]
            if not keys:
                break
            out.pop(keys[-1])
            out["_truncated"] = True
        return out or None
    except Exception:  # noqa: BLE001
        _drop("detail_error")
        return None


def clean_counts(counts) -> dict:
    if not isinstance(counts, dict):
        return {}
    out = {}
    for k, v in list(counts.items())[:32]:
        if _key_ok(k) and not isinstance(v, bool) and isinstance(v, (int, float)) and v == v and abs(v) < 1e15:
            out[k] = v if isinstance(v, int) else round(v, 6)
        else:
            _drop("count")
    return out


def _ident(s, fallback: str) -> str:
    s = getattr(s, "value", s)  # str Enums (progress.Stage / State)
    return s if isinstance(s, str) and IDENT_RE.match(s) else fallback


# --------------------------------------------------------------------------------------------- config
def load_cfg(path: Path | None = None) -> dict:
    """Missing or invalid file: tracing off (fail closed for storage; requests are unaffected)."""
    try:
        cfg = json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))
        return cfg if isinstance(cfg, dict) else {"enabled": False}
    except Exception as exc:  # noqa: BLE001
        log.warning("request_trace config unusable, tracing off (%s)", type(exc).__name__)
        return {"enabled": False}


CFG: dict = load_cfg()


def _limits() -> dict:
    raw = CFG.get("limits") if isinstance(CFG.get("limits"), dict) else {}
    out = dict(DEFAULT_LIMITS)
    for k, dv in DEFAULT_LIMITS.items():
        v = raw.get(k)
        if not isinstance(v, bool) and isinstance(v, (int, float)) and 0 < v <= 10**7:
            out[k] = type(dv)(v)
    return out


def enabled() -> bool:
    return CFG.get("enabled") is True


def log_enabled() -> bool:
    ls = CFG.get("log_sink")
    return enabled() and isinstance(ls, dict) and ls.get("enabled") is True


def db_enabled() -> bool:
    ds = CFG.get("db_sink")
    return enabled() and isinstance(ds, dict) and ds.get("enabled") is True


def configure(cfg: dict | None, *, engine_factory=None, env=None) -> None:
    """Replace the active config and reset state (tests and owner tooling; app code never calls it)."""
    global CFG, _WRITER
    CFG = cfg if isinstance(cfg, dict) else {"enabled": False}
    w, _WRITER = _WRITER, None
    if w is not None:
        w.stop()
    _FACTORY_OVERRIDE["f"] = engine_factory
    _FACTORY_OVERRIDE["env"] = env
    _RO_ENGINE.clear()
    with _recent_lock:
        _RECENT.clear()
    with _drop_lock:
        DROPS.clear()


# --------------------------------------------------------------------------------------------- counters
_drop_lock = threading.Lock()
DROPS: dict[str, int] = {}


def _drop(kind: str) -> None:
    with _drop_lock:
        DROPS[kind] = DROPS.get(kind, 0) + 1


# --------------------------------------------------------------------------------------------- trace state
def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_trace_id() -> str:
    return secrets.token_hex(16)


class Trace:
    __slots__ = ("trace_id", "client_request_id", "endpoint", "question", "started_at", "t0", "seq", "events",
                 "dropped", "finished", "outcome_branch", "message_code", "elapsed_ms", "mono", "_lock")

    def __init__(self, question: str, meta: dict):
        lim = _limits()
        self.trace_id = new_trace_id()
        cid = meta.get("client_request_id")
        self.client_request_id = cid if isinstance(cid, str) and CLIENT_ID_RE.match(cid) else None
        ep = meta.get("endpoint")
        self.endpoint = ep if ep in _ENDPOINTS else "other"
        self.question = (question if isinstance(question, str) else "")[: lim["max_question_chars"]]
        self.started_at = _utc_iso()
        self.t0 = time.monotonic()
        self.mono = self.t0
        self.seq = 0
        self.events: list[dict] = []  # kept only for the owner route's short recent window
        self.dropped = 0
        self.finished = False
        self.outcome_branch = self.message_code = None
        self.elapsed_ms = None
        self._lock = threading.Lock()

    def owner_view(self) -> dict:
        with self._lock:
            return {"request_id": self.trace_id, "client_request_id": self.client_request_id,
                    "endpoint": self.endpoint, "started_at": self.started_at, "question": self.question,
                    "finished": self.finished, "outcome_branch": self.outcome_branch,
                    "message_code": self.message_code, "elapsed_ms": self.elapsed_ms,
                    "dropped_events": self.dropped, "events": [dict(e) for e in self.events]}


CURRENT: ContextVar[Trace | None] = ContextVar("vera_request_trace", default=None)
_recent_lock = threading.Lock()
_RECENT: dict[str, Trace] = {}  # insertion order; bounded by recent_max and recent_ttl_s (owner route only)


def _remember(tr: Trace) -> None:
    lim = _limits()
    with _recent_lock:
        now = time.monotonic()
        for tid in [t for t, x in _RECENT.items() if now - x.mono > lim["recent_ttl_s"]]:
            del _RECENT[tid]
        while len(_RECENT) >= lim["recent_max"]:
            _RECENT.pop(next(iter(_RECENT)))
        _RECENT[tr.trace_id] = tr


def recent(trace_id) -> dict | None:
    """Owner route: one recent trace by id (in-process window only; the DB holds the durable copy). Never raises."""
    try:
        if not (isinstance(trace_id, str) and TRACE_ID_RE.match(trace_id)):
            return None
        lim = _limits()
        with _recent_lock:
            tr = _RECENT.get(trace_id)
        if tr is None or time.monotonic() - tr.mono > lim["recent_ttl_s"]:
            return None
        return tr.owner_view()
    except Exception:  # noqa: BLE001
        return None


def owner_db_view_enabled() -> bool:
    """DEFERRED-DB-1104: db_sink.owner_route_db_view (default false) gates the owner route's DB read."""
    ds = CFG.get("db_sink")
    return db_enabled() and isinstance(ds, dict) and ds.get("owner_route_db_view") is True


def db_view(trace_id, *, env=None, engine_factory=None) -> dict | None:
    """INTEGRATE-1104 (owner: migration 015 applied): the owner route's durable read. One trace (request row + its
    events in seq order) from the db_sink tables through the read-only account (db_sink.db_url_env_ro). None when the
    DB sink is off, the id is malformed, no row exists yet (the writer is asynchronous) or on ANY error (e.g. no SELECT
    grant): the caller then falls back to the in-process window. Never raises; the error class only is logged."""
    try:
        if not db_enabled() or not (isinstance(trace_id, str) and TRACE_ID_RE.match(trace_id)):
            return None
        dcfg = CFG.get("db_sink") or {}
        env = env if env is not None else (_FACTORY_OVERRIDE.get("env") or os.environ)
        # DEFERRED-DB-1104: an optional dedicated owner-read login (db_url_env_owner_read) wins over the ro login, which
        # migration 015 does not allow to SELECT these tables
        url = (env.get(str(dcfg.get("db_url_env_owner_read") or dcfg.get("db_url_env_ro") or "VERA_DB_URL_RO")) or "").strip()
        if not url:
            return None
        rt = str(dcfg.get("requests_table") or "vera_vjay.request_trace_requests")
        et = str(dcfg.get("events_table") or "vera_vjay.request_trace_events")
        if not (TABLE_RE.match(rt) and TABLE_RE.match(et)):
            return None
        factory = engine_factory or _FACTORY_OVERRIDE.get("f")
        if factory is None:
            from sqlalchemy import create_engine as factory
        ms = int(dcfg.get("statement_timeout_ms") or 2000)
        eng = _RO_ENGINE.get(url)
        if eng is None:
            eng = factory(url, hide_parameters=True, pool_pre_ping=True, pool_size=1, max_overflow=0,
                          connect_args={"connect_timeout": int(dcfg.get("connect_timeout_s") or 2),
                                        "options": f"-c statement_timeout={ms}"})
            _RO_ENGINE[url] = eng
        from sqlalchemy import text
        with eng.connect() as conn:
            req = conn.execute(text(f"SELECT request_id, client_request_id, endpoint, started_at, question, "
                                    f"outcome_branch, message_code, event_count, dropped_events, elapsed_ms FROM {rt} "
                                    "WHERE request_id = :rid ORDER BY row_id DESC LIMIT 1"), {"rid": trace_id}).first()
            if req is None:
                return None
            evs = conn.execute(text(f"SELECT seq, ts, stage, code, counts, detail FROM {et} WHERE request_id = :rid "
                                    "ORDER BY seq"), {"rid": trace_id}).fetchall()
        m = dict(req._mapping)

        def _iso(v):
            return v.isoformat() if hasattr(v, "isoformat") else v

        def _js(v):
            return json.loads(v) if isinstance(v, str) else v
        return {"request_id": m["request_id"], "client_request_id": m.get("client_request_id"),
                "endpoint": m.get("endpoint"), "started_at": _iso(m.get("started_at")), "question": m.get("question"),
                "finished": True, "outcome_branch": m.get("outcome_branch"), "message_code": m.get("message_code"),
                "elapsed_ms": m.get("elapsed_ms"), "dropped_events": m.get("dropped_events"), "source": "db",
                "events": [{"seq": e["seq"], "ts": _iso(e["ts"]), "stage": e["stage"], "code": e["code"],
                            "counts": _js(e["counts"]), "detail": _js(e["detail"])}
                           for e in (dict(r._mapping) for r in evs)]}
    except Exception as exc:  # noqa: BLE001
        log.warning("request_trace owner read failed: RT_READ_ERROR (%s)", type(exc).__name__)
        return None


_RO_ENGINE: dict = {}


def current_id() -> str | None:
    tr = CURRENT.get()
    return tr.trace_id if tr is not None else None


# --------------------------------------------------------------------------------------------- sinks
def _log_line(obj: dict) -> None:
    if not log_enabled():
        return
    try:
        log.info("%s %s", LOG_PREFIX, json.dumps(obj, separators=(",", ":"), ensure_ascii=True, default=str))
    except Exception:  # noqa: BLE001
        _drop("log_error")


def _emit(tr: Trace, stage: str, code: str, counts: dict, detail: dict | None, extra: dict | None = None,
          terminal: bool = False) -> None:
    lim = _limits()
    with tr._lock:
        if tr.finished:
            _drop("after_finish")
            return
        if not terminal and tr.seq >= lim["max_events_per_request"] - 1:  # one slot kept for the finish event
            tr.dropped += 1
            _drop("full")
            return
        tr.seq += 1
        ev = {"seq": tr.seq, "ts": _utc_iso(), "stage": stage, "code": code, "counts": counts, "detail": detail}
        tr.events.append(ev)
        if terminal:
            tr.finished = True
    line = {"v": SCHEMA_VERSION, "request_id": tr.trace_id, **ev}
    if extra:
        line.update(extra)
    _log_line(line)
    if db_enabled():
        _writer_put(("evt", {"request_id": tr.trace_id, "seq": ev["seq"], "ts": ev["ts"], "stage": stage,
                             "code": code, "counts": json.dumps(counts, separators=(",", ":")),
                             "detail": None if detail is None else json.dumps(detail, separators=(",", ":"))}))


# --------------------------------------------------------------------------------------------- public API
def start(question, meta: dict | None = None) -> str | None:
    """Open the trace for this request context and return its id (None when tracing is off). Never raises."""
    try:
        if not enabled():
            return None
        meta = {k: v for k, v in (meta or {}).items() if k in _META_KEYS} if isinstance(meta, dict) else {}
        tr = Trace(question, meta)
        CURRENT.set(tr)
        _remember(tr)
        ls = CFG.get("log_sink") if isinstance(CFG.get("log_sink"), dict) else {}
        q = {"question": tr.question} if ls.get("include_question") is True else {"question_chars": len(tr.question)}
        extra = {**q, "client_request_id": tr.client_request_id, "endpoint": tr.endpoint}
        try:  # SETTINGS-W3-1110: settings digest in force; absent (byte-identical line) while every read_from_db flag is off
            from vera import settings as _settings
            sv = _settings.version()
            if sv:
                extra["settings_version"] = sv
        except Exception:  # noqa: BLE001  never break a trace
            pass
        _emit(tr, "start", "received", {}, None, extra=extra)
        return tr.trace_id
    except Exception as exc:  # noqa: BLE001
        _drop("start_error")
        log.warning("request_trace start failed (%s)", type(exc).__name__)
        return None


def event(stage, code, counts=None, detail=None) -> None:
    """Append one event to this context's trace. No-op outside a traced request. Never raises."""
    try:
        tr = CURRENT.get()
        if tr is None:
            return
        st = _ident(stage, "other")
        _emit(tr, st, _ident(code, "other"), clean_counts(counts), clean_detail(detail, stage=st))
    except Exception:  # noqa: BLE001
        _drop("event_error")


def on_progress(stage, state, counts) -> None:
    """Hook called by progress.emit for every stage event (counts only; the progress contract carries no text)."""
    event(stage, state, counts)


def finish(outcome_branch, message_code) -> None:
    """Close the trace: one terminal event, then the per-request row. Idempotent. Never raises."""
    try:
        tr = CURRENT.get()
        if tr is None or tr.finished:
            return
        ob, mc = _ident(outcome_branch, "other"), _ident(message_code, "other")
        elapsed = int((time.monotonic() - tr.t0) * 1000)
        with tr._lock:
            tr.outcome_branch, tr.message_code, tr.elapsed_ms = ob, mc, elapsed
        _emit(tr, "finish", ob, {"elapsed_ms": elapsed, "dropped": tr.dropped}, {"message_code": mc}, terminal=True)
        if db_enabled():
            _writer_put(("req", {"request_id": tr.trace_id, "client_request_id": tr.client_request_id,
                                 "endpoint": tr.endpoint, "started_at": tr.started_at, "question": tr.question,
                                 "outcome_branch": ob, "message_code": mc, "event_count": tr.seq,
                                 "dropped_events": tr.dropped, "elapsed_ms": elapsed}))
    except Exception as exc:  # noqa: BLE001
        _drop("finish_error")
        log.warning("request_trace finish failed (%s)", type(exc).__name__)


def end_context() -> None:
    """Detach the trace from this context (entry points call it in `finally`)."""
    try:
        CURRENT.set(None)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------------------------- DB writer
_FACTORY_OVERRIDE: dict = {"f": None, "env": None}
_WRITER = None
_writer_lock = threading.Lock()

_EVT_COLS = ("request_id", "seq", "ts", "stage", "code", "counts", "detail")
_REQ_COLS = ("request_id", "client_request_id", "endpoint", "started_at", "question", "outcome_branch",
             "message_code", "event_count", "dropped_events", "elapsed_ms")


class _Writer:
    """One daemon thread draining a bounded queue into batched app-path INSERTs. put() never blocks."""

    def __init__(self, dcfg: dict, factory=None, env=None):
        self.cfg = dcfg
        self.q: queue.Queue = queue.Queue(maxsize=int(dcfg.get("queue_max") or 5000))
        self._factory, self._env = factory, env if env is not None else os.environ
        self._engine = None
        self._stop = threading.Event()
        self.inflight = 0
        self._purger = retention_purge.Purger(CFG.get("retention_purge"), dcfg, factory, self._env)
        self.thread = threading.Thread(target=self._run, name="request-trace-writer", daemon=True)
        self.thread.start()

    def put(self, item) -> None:
        try:
            self.q.put_nowait(item)
        except queue.Full:
            _drop("queue_full")

    def stop(self) -> None:
        self._stop.set()

    def _engine_or_none(self):
        if self._engine is not None:
            return self._engine
        url = (self._env.get(str(self.cfg.get("db_url_env_wo") or "VERA_DB_URL_WO")) or "").strip()
        if not url:
            return None
        factory = self._factory
        if factory is None:
            from sqlalchemy import create_engine as factory
        self._engine = factory(url, hide_parameters=True, pool_pre_ping=True, pool_size=1, max_overflow=0,
                               connect_args={"connect_timeout": int(self.cfg.get("connect_timeout_s") or 2),
                                             "options": "-c statement_timeout="
                                                        + str(int(self.cfg.get("statement_timeout_ms") or 2000))})
        return self._engine

    def _tables(self) -> tuple[str, str]:
        rt = str(self.cfg.get("requests_table") or "vera_vjay.request_trace_requests")
        et = str(self.cfg.get("events_table") or "vera_vjay.request_trace_events")
        if not (TABLE_RE.match(rt) and TABLE_RE.match(et)):
            raise ValueError("bad table name")
        return rt, et

    def flush(self, batch: list) -> None:
        eng = self._engine_or_none()
        if eng is None:
            _drop("no_writer")
            return
        from sqlalchemy import text
        rt, et = self._tables()
        evts = [r for k, r in batch if k == "evt"]
        reqs = [r for k, r in batch if k == "req"]
        with eng.begin() as conn:
            conn.execute(text("SET LOCAL statement_timeout = " + str(int(self.cfg.get("statement_timeout_ms") or 2000))))
            if evts:
                conn.execute(text(f"INSERT INTO {et} (request_id, seq, ts, stage, code, counts, detail) VALUES "
                                  "(:request_id, :seq, CAST(:ts AS timestamptz), :stage, :code, CAST(:counts AS jsonb), "
                                  "CAST(:detail AS jsonb))"), evts)
            if reqs:
                conn.execute(text(f"INSERT INTO {rt} (request_id, client_request_id, endpoint, started_at, question, "
                                  "outcome_branch, message_code, event_count, dropped_events, elapsed_ms) VALUES "
                                  "(:request_id, :client_request_id, :endpoint, CAST(:started_at AS timestamptz), "
                                  ":question, :outcome_branch, :message_code, :event_count, :dropped_events, "
                                  ":elapsed_ms)"), reqs)

    def _purge_tick(self) -> None:
        # PURGE-WIRING-1105: daily retention purge on this thread; no-op (no connection) while the flag is off.
        def emit(ev: dict) -> None:
            _log_line({"v": SCHEMA_VERSION, "request_id": None, "ts": _utc_iso(), **ev})
        self._purger.tick(emit)

    def _run(self) -> None:
        interval = float(self.cfg.get("flush_interval_s") or 2)
        bmax = int(self.cfg.get("batch_max") or 200)
        while not self._stop.is_set():
            self._purge_tick()
            try:
                first = self.q.get(timeout=interval)
            except queue.Empty:
                continue
            batch = [first]
            while len(batch) < bmax:
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            self.inflight = len(batch)
            try:
                self.flush(batch)
            except Exception as exc:  # noqa: BLE001  (dropped, counted; type only in the log, never the row)
                _drop("write_error")
                log.warning("request_trace write failed: RT_WRITE_ERROR (%s), %d rows dropped",
                            type(exc).__name__, len(batch))
            finally:
                self.inflight = 0


def _writer_put(item) -> None:
    global _WRITER
    try:
        w = _WRITER
        if w is None:
            with _writer_lock:
                if _WRITER is None:
                    _WRITER = _Writer(CFG.get("db_sink") or {}, _FACTORY_OVERRIDE["f"], _FACTORY_OVERRIDE["env"])
                w = _WRITER
        w.put(item)
    except Exception as exc:  # noqa: BLE001
        _drop("writer_error")
        log.warning("request_trace writer unavailable (%s)", type(exc).__name__)


def start_purge_at_boot() -> bool:
    """PURGE-BOOT-1105: when retention_purge.enabled and the DB sink is on, start the writer thread (which hosts the
    daily purge scheduler) now, so an idle API still purges. Idempotent with the first-request start (same lock and
    singleton). Non-blocking: only spawns a daemon thread. Flag off: nothing starts. Never raises."""
    global _WRITER
    try:
        rp = CFG.get("retention_purge")
        if not (isinstance(rp, dict) and rp.get("enabled") is True and db_enabled()):
            return False
        with _writer_lock:
            if _WRITER is not None:
                return False
            _WRITER = _Writer(CFG.get("db_sink") or {}, _FACTORY_OVERRIDE["f"], _FACTORY_OVERRIDE["env"])
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("request_trace purge boot start failed (%s)", type(exc).__name__)
        return False


def drain_for_tests(timeout_s: float = 5.0) -> bool:
    """Wait until the writer queue is empty (tests only)."""
    w = _WRITER
    if w is None:
        return True
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if w.q.empty() and not w.inflight:
            time.sleep(0.05)
            if w.q.empty() and not w.inflight:
                return True
        time.sleep(0.02)
    return False


start_purge_at_boot()  # PURGE-BOOT-1105: import-time start (main.py imports this module); no-op while the flag is off
