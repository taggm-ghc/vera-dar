"""PROG-1027 W4b-2: pure progress UI logic (no st.*, no I/O except the worker's injected client call).

`step` is a state machine over polled events; `classify` derives the terminal kind from the worker Holder in a fixed
check order (plan s3a table T). Terminals are returned as label keys (see config/progress-stages.json `terminals`).
Exception text is never stored: the Holder keeps the exception type name and a small closed kind only.
"""
from __future__ import annotations

import json
import string
import threading
from dataclasses import dataclass, field
from pathlib import Path

import requests

from vera import progress
from vera.public_mode import ASK_TIMEOUT_S, GONE

LABELS_PATH = Path(__file__).resolve().parent.parent / "config" / "progress-stages.json"
TERMINAL_KINDS = ("answered", "partial", "no_answer", "declined", "error", "provider_error", "busy", "expired",
                  "poll_timeout_received", "poll_timeout_unreceived")
_TOP_KEYS = ("sending", "finishing", "progress_unavailable", "reload_caption", "already_running", "background_line")
_FIELDS = frozenset(progress.COUNT_KEYS) | {"rid"}
_TERMINAL_FIELDS = _FIELDS | {"minutes"}  # F5: wording derives from ASK_TIMEOUT_S (no duplicated literal), terminals only
ERROR_KINDS = frozenset({"connect_timeout", "read_timeout", "connection_error", "chunked", "other"})


def _check_text(name: str, text, fields=_FIELDS) -> None:
    if not isinstance(text, str) or not text:
        raise ValueError(f"label {name}: must be a non-empty string")
    try:
        for _, field_name, spec, conv in string.Formatter().parse(text):
            if field_name is None:
                continue
            if field_name not in fields or spec or conv:
                raise ValueError(f"label {name}: bad placeholder")
    except ValueError as exc:
        raise ValueError(f"label {name}: malformed ({exc})") from None


def _fields_of(text: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


def _minutes_phrase(seconds) -> str:
    """Whole minutes, rounded up (min 1), with correct singular/plural: 60 -> "1 minute", 90 -> "2 minutes"."""
    n = max(1, -(-int(seconds) // 60))
    return f"{n} minute" + ("" if n == 1 else "s")


class Labels:
    """Validated view of config/progress-stages.json. Raises ValueError (naming the keys) on a bad shape."""

    def __init__(self, raw):
        if not isinstance(raw, dict):
            raise ValueError("labels must be a JSON object")
        for k in _TOP_KEYS:
            _check_text(k, raw.get(k))
        stages, terms = raw.get("stages"), raw.get("terminals")
        if not isinstance(stages, dict) or not isinstance(terms, dict):
            raise ValueError("labels: stages and terminals must be objects")
        for st, states in stages.items():
            if st not in {s.value for s in progress.Stage} or not isinstance(states, dict):
                raise ValueError(f"labels: unknown stage {st!r}")
            for sa, text in states.items():
                if sa not in {s.value for s in progress.VALID[progress.Stage(st)]}:
                    raise ValueError(f"labels: invalid state {st}.{sa}")
                _check_text(f"{st}.{sa}", text)
        for kind in TERMINAL_KINDS:
            _check_text(f"terminals.{kind}", terms.get(kind), _TERMINAL_FIELDS)
        self._raw = raw
        self._stages, self._terms = stages, terms

    @classmethod
    def load(cls, path: Path = LABELS_PATH) -> "Labels":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def _top(self, key: str, rid: str = "") -> str:
        return self._raw[key].format_map({"rid": rid})

    def sending(self, rid: str) -> str:
        return self._top("sending", rid)

    @property
    def finishing(self) -> str:
        return self._raw["finishing"]

    @property
    def progress_unavailable(self) -> str:
        return self._raw["progress_unavailable"]

    @property
    def reload_caption(self) -> str:
        return self._raw["reload_caption"]

    @property
    def already_running(self) -> str:
        return self._raw["already_running"]

    @property
    def background_line(self) -> str:
        return self._raw["background_line"]

    def stage_label(self, stage: str, state: str, counts: dict | None, rid: str) -> str | None:
        """Label for (stage, state) or None when unknown. Counts are shown only when the event carries them: a label
        whose placeholders are not all present falls back to the stage's `start` label, else None."""
        text = (self._stages.get(stage) or {}).get(state)
        if text is None:
            return None
        values = {k: v for k, v in (counts or {}).items() if k in progress.COUNT_KEYS}
        values["rid"] = rid
        if _fields_of(text) <= set(values):
            return text.format_map(values)
        start = (self._stages.get(stage) or {}).get("start")
        if start is not None and _fields_of(start) <= set(values):
            return start.format_map(values)
        return None

    def terminal_text(self, kind: str, rid: str = "") -> str:
        text = self._terms.get(kind)
        if text is None:
            text = self._terms["error"]
        return text.format_map({"rid": rid, "minutes": _minutes_phrase(ASK_TIMEOUT_S)})


@dataclass
class Holder:
    """Written only by the worker thread; `done` is set LAST (after the fields)."""
    result: dict | None = None
    status: int | None = None
    error_type: str | None = None  # exception class name only
    error_kind: str | None = None  # closed set ERROR_KINDS, computed from isinstance checks in the worker
    done: threading.Event = field(default_factory=threading.Event)


@dataclass
class Inflight:
    rid: str
    holder: Holder
    t0: float
    seq: int = 0
    label: str = ""
    received_seen: bool = False
    first_event_at: float | None = None
    server_done: str | None = None  # state value of a server `done` event
    queued: int = 0


def error_kind_of(exc: BaseException) -> str:
    """ConnectTimeout is checked first: it subclasses both ConnectionError and Timeout."""
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return "connect_timeout"
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return "read_timeout"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "connection_error"
    if isinstance(exc, requests.exceptions.ChunkedEncodingError):
        return "chunked"
    return "other"


def run_worker(holder: Holder, client_fn, args) -> None:
    """Worker body (plan s4 `_worker`): no text kept, nothing logged, `done` always set."""
    try:
        r = client_fn(*args)
        holder.status = r.status_code
        holder.result = r.json() if r.status_code == 200 else None  # a bad 200 body raises -> error_type
    except BaseException as e:  # noqa: BLE001  incl. SystemExit/KeyboardInterrupt in the thread
        holder.error_type = type(e).__name__
        try:
            holder.error_kind = error_kind_of(e)
        except Exception:  # noqa: BLE001
            holder.error_kind = "other"
    finally:
        holder.done.set()


def _poll_timeout_kind(inf: Inflight) -> str:
    return "poll_timeout_received" if inf.received_seen else "poll_timeout_unreceived"


def classify(inf: Inflight, final_get_404: bool) -> str:
    """Fixed check order; returns a terminal label key. Status codes and exception kinds only."""
    h = inf.holder
    st, kind = h.status, h.error_kind
    if st == 429:
        return "busy"
    if kind == "connect_timeout":
        return "error"
    if kind == "read_timeout":
        return _poll_timeout_kind(inf)
    if st in (502, 503) and inf.server_done == "error":
        return "provider_error"
    lost = kind in ("connection_error", "chunked") or st in (502, 503, 504)
    if lost and inf.received_seen and inf.server_done is None and final_get_404:
        return "expired"
    if st == 200 and isinstance(h.result, dict):
        cc = h.result.get("claim_check")
        comp = cc.get("answer_completeness") if isinstance(cc, dict) else None
        return progress.outcome_of(h.result.get("result_kind"), comp).value
    return "error"


def classify_safe(inf: Inflight, final_get_404: bool = False) -> str:
    try:
        return classify(inf, final_get_404)
    except Exception:  # noqa: BLE001
        return "error"


def step(inf: Inflight, events, now: float, cfg, labels: Labels, final: bool = False) -> tuple[str, str | None]:
    """Pure state machine. Returns (label_to_show, terminal_key_or_None) and updates inf. A terminal only when `final`
    (holder was done before this tick's GET) or at the backstop; a server `done` event never ends the run."""
    if not inf.label:
        inf.label = labels.sending(inf.rid)
    if isinstance(events, list):
        for ev in events:
            try:
                seq = ev["seq"]
                if seq <= inf.seq:
                    continue
                inf.seq = seq
                if inf.first_event_at is None:
                    inf.first_event_at = now
                stage, state, counts = ev["stage"], ev["state"], ev.get("counts") or {}
                if stage == progress.Stage.RECEIVED.value:
                    inf.received_seen = True
                if stage == progress.Stage.DONE.value:
                    inf.server_done = state
                    q = counts.get("queued", 0)
                    inf.queued = q if type(q) is int and q > 0 else 0
                    inf.label = labels.finishing
                    continue
                if inf.server_done is not None:
                    continue
                text = labels.stage_label(stage, state, counts, inf.rid)
                if text is not None:
                    inf.label = text
            except Exception:  # noqa: BLE001  a malformed event is ignored, never raised
                continue
    elif events is GONE or events is None:
        in_grace = inf.first_event_at is None and now - inf.t0 < cfg.first_event_grace_s
        if not in_grace and inf.server_done is None:
            inf.label = labels.progress_unavailable
    if final:
        return inf.label, classify_safe(inf, final_get_404=events is GONE)
    if now - inf.t0 >= ASK_TIMEOUT_S + cfg.poll_margin_s:
        return inf.label, _poll_timeout_kind(inf)
    return inf.label, None


@dataclass
class LastResult:
    """Durable copy of a finished run (session state). `rendered` flips only after a successful render."""
    result: dict | None
    terminal: str
    queued: int = 0
    rid: str = ""
    status: int | None = None
    rendered: bool = False


FIN_PREFIX = "vera_fin_"


def poll_tick(inf: Inflight, ses, ui, fetch, now, cfg, labels: Labels) -> bool:
    """One fragment tick (W4b-3), free of Streamlit so it is testable directly. `ses` is a dict-like session state,
    `ui` has `caption`, `fetch(after)` returns list | GONE | None, `now()` is the clock. Nothing raised in the body
    reaches the page (fixed text only). Returns True when a completion rerun is needed; at most once per rid (fin guard)."""
    terminal = None
    try:
        final = inf.holder.done.is_set()  # read BEFORE the GET: the final GET follows the POST response
        label, terminal = step(inf, fetch(inf.seq), now(), cfg, labels, final=final)
        if terminal is None:
            ui.caption(label)
            ui.caption(labels.reload_caption)
    except Exception:  # noqa: BLE001
        ui.caption(labels.progress_unavailable)
        terminal = classify_safe(inf) if inf.holder.done.is_set() else None
    if terminal and not ses.get(FIN_PREFIX + inf.rid):
        ses[FIN_PREFIX + inf.rid] = True
        ses["vera_last_result"] = LastResult(inf.holder.result, terminal, inf.queued, inf.rid, inf.holder.status)
        ses.pop("vera_inflight", None)
        return True
    return False


def start_tracked(ses, mk_rid, token: str, now, client_fn, args_tail, ui, labels: Labels, spawn=None) -> Inflight:
    """Single-flight start keyed on the session: returns the existing in-flight entry when there is one. Otherwise mints
    the rid, renders the Sending label FIRST, then starts the worker (R4-PROG-Q-3). The worker touches no Streamlit state."""
    cur = ses.get("vera_inflight")
    if cur is not None:
        return cur
    rid = mk_rid()
    inf = Inflight(rid=rid, holder=Holder(), t0=now())
    inf.label = labels.sending(rid)
    ses["vera_inflight"] = inf
    ui.caption(inf.label)
    args = (*args_tail[:4], rid, token)  # base_url, question, mode, api_key, rid, session_token
    (spawn or _spawn)(run_worker, inf.holder, client_fn, args)
    return inf


def _spawn(fn, *a) -> None:
    threading.Thread(target=fn, args=a, daemon=True).start()
