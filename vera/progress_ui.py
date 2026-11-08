"""PROG-1027 W4b-2: pure progress UI logic (no st.*, no I/O except the worker's injected client call).

`step` is a state machine over polled events; `classify` derives the terminal kind from the worker Holder in a fixed
check order (plan s3a table T). Terminals are returned as label keys (see config/progress-stages.json `terminals`).
Exception text is never stored: the Holder keeps the exception type name and a small closed kind only.
"""
from __future__ import annotations

import json
import logging
import string
import threading
from dataclasses import dataclass, field
from pathlib import Path

import requests

from vera import progress
from vera.public_mode import ASK_TIMEOUT_S, GONE

log = logging.getLogger("vera.progress_ui")

LABELS_PATH = Path(__file__).resolve().parent.parent / "config" / "progress-stages.json"
TERMINAL_KINDS = ("answered", "partial", "no_answer", "declined", "error", "provider_error", "busy", "expired",
                  "poll_timeout_received", "poll_timeout_unreceived")
_TOP_KEYS = ("sending", "finishing", "progress_unavailable", "reload_caption", "already_running", "background_line")
_STREAM_COUNT_KEYS = frozenset({"shortlisted", "held", "step", "ceiling", "why"})
_COUNT_KEYS = frozenset(progress.COUNT_KEYS) | _STREAM_COUNT_KEYS | {"act", "wait_s", "k", "n", "elapsed"}
_FIELDS = _COUNT_KEYS | {"rid", "why_text"}
_EXTRA_STATES = {"acquire": frozenset({"limited", "busy"})}  # stream states; tolerated until progress.VALID carries them
STREAM_TERMINALS = ("unavailable", "connection_lost", "never_connected", "stalled", "orphaned", "cancelled")
# Generic fallbacks (no exception text, no names) when config/progress-stages.json lacks a stream terminal key.
_GENERIC_TERMINALS = {
    "unavailable": "VERA-DAR cannot answer right now; please try again shortly",
    "connection_lost": "The connection was interrupted. Documents VERA-DAR already saved stay in its collection; ask again to use them.",
    "never_connected": "VERA-DAR could not be reached; please try again shortly",
    "stalled": "VERA-DAR stopped searching because no further progress was made; ask again to use what it already saved.",
    "orphaned": "This page stopped receiving updates, so the request was closed; ask again to use what VERA-DAR already saved.",
    "cancelled": "The request was cancelled before it finished; ask again to use what VERA-DAR already saved.",
    "busy": "VERA-DAR is busy right now; please try again shortly",
    "error": "Something went wrong; please try again",
    "provider_error": "The answer could not be generated right now; please try again shortly",
}
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
                if sa not in {s.value for s in progress.VALID[progress.Stage(st)]} | _EXTRA_STATES.get(st, frozenset()):
                    raise ValueError(f"labels: invalid state {st}.{sa}")
                _check_text(f"{st}.{sa}", text)
        for kind in TERMINAL_KINDS:
            _check_text(f"terminals.{kind}", terms.get(kind), _TERMINAL_FIELDS)
        why = raw.get("why", {})
        if not isinstance(why, dict):
            raise ValueError("labels: why must be an object")
        for k, text in why.items():
            _check_text(f"why.{k}", text)
        acts = raw.get("activities", {})  # NOISY-UI: generic live-activity text by closed code
        if not isinstance(acts, dict) or any(k not in {str(c) for c in progress.ACTIVITY_CODES} for k in acts):
            raise ValueError("labels: activities must be an object keyed by activity code")
        for k, text in acts.items():
            _check_text(f"activities.{k}", text)
        self._acts = acts
        for kind in STREAM_TERMINALS:
            if kind in terms:
                _check_text(f"terminals.{kind}", terms.get(kind), _TERMINAL_FIELDS)
        self._raw = raw
        self._stages, self._terms, self._why = stages, terms, why

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

    def activity_label(self, counts: dict | None) -> str | None:
        """Text for the activity code `act` in counts (an event's counts or a heartbeat's fields); None when absent/unknown
        or when a needed count is missing."""
        c = counts or {}
        text = getattr(self, "_acts", {}).get(str(c.get("act")))
        if text is None:
            return None
        values = {k: v for k, v in c.items() if type(v) is int}
        if "wait_s" in values and values["wait_s"] < 1:  # SELFQ-VARIANTS-1108: never "about 0 s"
            text = text.replace("in about {wait_s} s", "shortly").replace(" about {wait_s} s", " briefly")
        return text.format_map(values) if _fields_of(text) <= set(values) else None

    def stage_label(self, stage: str, state: str, counts: dict | None, rid: str) -> str | None:
        """Label for (stage, state) or None when unknown. Counts are shown only when the event carries them: a label
        whose placeholders are not all present falls back to the stage's `start` label, else None."""
        text = (self._stages.get(stage) or {}).get(state)
        if text is None:
            return None
        values = {k: v for k, v in (counts or {}).items() if k in _COUNT_KEYS and type(v) is int}
        if "why" in values and str(values["why"]) in self._why:
            values["why_text"] = self._why[str(values["why"])]
        values["rid"] = rid
        if type(values.get("stored")) is int and type(values.get("target")) is int and values["target"] > 0:
            values["stored"] = min(values["stored"], values["target"])  # PROGRESS-COUNT-1106: never "7 of 6 saved"
        if _fields_of(text) <= set(values):
            return text.format_map(values)
        start = (self._stages.get(stage) or {}).get("start")
        if start is not None and _fields_of(start) <= set(values):
            return start.format_map(values)
        return None

    def terminal_text(self, kind: str, rid: str = "") -> str:
        text = self._terms.get(kind)
        if text is None:
            text = _GENERIC_TERMINALS.get(kind) or self._terms["error"]
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


# ----------------------------------------------------------------------------------------------- OD-1031-STREAM-ON
# Stream path: a worker thread (never calls st.*) fills a StreamHolder; a fragment renders it once a second.
STREAM_HOLDER_KEY = "vera_stream"
STREAM_LAST_KEY = "vera_stream_last"
_LOG_MAX = 12
_LOG_REFRESH_S = 10.0  # PROGRESS-COUNT-1106: at most one extra line per 10 s for repeated count updates
_RESULT_KIND_TERMINAL = {"busy": "busy", "error": "error", "provider_error": "provider_error", "unavailable": "unavailable"}


@dataclass
class StreamHolder:
    """Appended to by the worker only (list append / attribute set; `done` set LAST); `last_read` is stamped by the
    fragment each tick. Holds line dicts and closed codes, never exception text."""
    t0: float
    last_read: float
    events: list = field(default_factory=list)
    event_at: list = field(default_factory=list)  # PROGRESS-COUNT-1106: arrival time of each events[i] (same length)
    last_line_at: float = 0.0
    result: dict | None = None
    status: int | None = None
    kind: str | None = None
    terminal: str | None = None
    dropped: int = 0
    activity: dict | None = None  # NOISY-UI: latest act fields (event counts or heartbeat); cleared by a later plain line
    done: threading.Event = field(default_factory=threading.Event)


def stream_worker(holder: StreamHolder, stream_fn, args, now, orphan_s: float) -> None:
    """`stream_fn(*args, should_stop=...)` returns an iterable of validated lines (public_mode.iter_ask_stream) that
    raises StreamNeverConnected / StreamLost. The stream is closed when the fragment has not read for > orphan_s."""
    from vera.public_mode import StreamLost, StreamNeverConnected

    def orphaned() -> bool:
        return now() - holder.last_read > orphan_s

    stream = None
    try:
        stream = stream_fn(*args, should_stop=orphaned)
        for line in stream:
            holder.last_line_at = now()
            if line.get("type") == "hb":
                a = {k: line[k] for k in ("act", "wait_s", "k", "n", "elapsed") if k in line}
                holder.activity = a if "act" in a else None
            elif line.get("type") == "event":
                holder.events.append(line)
                getattr(holder, "event_at", []).append(now()) if hasattr(holder, "event_at") else None
                c = line.get("counts") or {}
                holder.activity = dict(c) if "act" in c else None
            elif line.get("type") == "result":
                holder.status = line.get("status")
                if holder.status == 200:
                    holder.result = line.get("body")
                    holder.terminal = "answer"
                else:
                    holder.kind = line.get("kind")
                    holder.terminal = _RESULT_KIND_TERMINAL.get(holder.kind, "error")
            if orphaned():
                break
        if holder.terminal is None:
            holder.terminal = "orphaned" if orphaned() else "connection_lost"
            # PERF-MISS-1105-FIX2 F4: a stream that ended without a result line is logged with its closed code
            log.info("stream ended without a result terminal=%s lines=%d", holder.terminal, len(holder.events))
    except StreamNeverConnected as e:
        holder.status = e.status
        holder.terminal = "busy" if e.status == 429 else "never_connected"
    except StreamLost as e:
        holder.terminal = "stalled" if e.reason == "no_data" else "connection_lost"
    except BaseException:  # noqa: BLE001  closed code only, no text
        holder.terminal = "connection_lost"
    finally:
        try:
            holder.dropped = int(getattr(stream, "dropped", 0) or 0)
            if stream is not None and hasattr(stream, "close"):
                stream.close()
        except Exception:  # noqa: BLE001
            pass
        holder.done.set()


def _latest_counts(events) -> dict:
    out: dict = {}
    for ev in events:
        for k, v in (ev.get("counts") or {}).items():
            out[k] = v
    return out


def _duration(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60} min {s % 60} s"


def stream_view(holder: StreamHolder, labels: Labels, now: float) -> dict:
    """Pure view model: label, log lines, bars [(text, fraction)], and the still-working caption."""
    log: list[str] = []
    label = "Working on your question"
    evs = list(holder.events)
    ats = list(getattr(holder, "event_at", None) or [])
    last_key = None  # (stage, state, held, appended_at) of the last appended line
    for i, ev in enumerate(evs):
        if ev.get("stage") == "received":  # its label carries the request id, which the stream does not use
            continue
        if "act" in (ev.get("counts") or {}):  # activity events are the live status line, not log lines
            continue
        text = labels.stage_label(ev.get("stage"), ev.get("state"), ev.get("counts"), "")
        if text is None:
            continue
        label = text
        at = ats[i] if i < len(ats) else None
        held = (ev.get("counts") or {}).get("held")
        key = (ev.get("stage"), ev.get("state"))
        # PROGRESS-COUNT-1106: a run of count-only updates of one stage shares one line, refreshed in place; a new line
        # starts only when "held" changes or about every _LOG_REFRESH_S since the line was appended.
        if (log and last_key and key == last_key[:2] and ev.get("state") == "update" and held == last_key[2]
                and (at is None or last_key[3] is None or at - last_key[3] < _LOG_REFRESH_S)):
            log[-1] = text
            continue
        if text not in log:  # SELFQ-VARIANTS-1108: a repeated line (also non-adjacent) is never appended again
            log.append(text)
            last_key = (*key, held, at)
    counts = _latest_counts(list(holder.events))
    bars = []
    # MIDTERMS-WP10-1106: once the examined-source count arrives, "Read n of 6" replaces the "Saved" bar
    pairs = (("checked", "shortlisted", "Checked"),
             ("counted", "target", "Read") if type(counts.get("counted")) is int else ("stored", "target", "Saved"))
    for done_k, total_k, name in pairs:
        total = counts.get(total_k)
        if type(total) is int and total > 0 and type(counts.get(done_k)) is int:
            n = min(counts[done_k], total)
            text = f"{name} {n} of {total}"
            # NOANSWER-PAGE-1108 D5: a fact the status line already states is not repeated as the bar's text (the bar
            # itself still shows the progress); "Saved n of t" matches the status wording "n of t saved".
            said = label.casefold()
            if text.casefold() in said or (name == "Saved" and f"{n} of {total} saved" in said):
                text = ""
            bars.append((text, min(1.0, counts[done_k] / total)))
    since = now - (holder.last_line_at or holder.t0)
    caption = f"Still working · {_duration(now - holder.t0)} · last update {max(0, int(since))} s ago"
    act = labels.activity_label(holder.activity) if holder.activity else None
    if act is None and not holder.events and since >= 5:
        act = "Waiting for a connection"  # nothing received yet: say so instead of showing a silent spinner
    line = f"{act or 'Working on your question'} · {_duration(now - holder.t0)} elapsed"  # SELFQ-VARIANTS-1108: not a third copy of the status label
    return {"label": label, "log": log[-_LOG_MAX:], "bars": bars, "caption": caption, "activity": line}


@dataclass
class StreamLast:
    """Durable copy of a finished stream run (session state); `rendered` flips only after a successful render."""
    terminal: str
    result: dict | None = None
    status: int | None = None
    rendered: bool = False


def stream_tick(holder: StreamHolder, ses, ui, now, labels: Labels) -> bool:
    """One fragment tick, free of Streamlit. Stamps last_read; returns True when a completion rerun is needed."""
    t = now()
    holder.last_read = t
    done = holder.done.is_set()  # read BEFORE the view: events precede done
    if not done:
        try:
            v = stream_view(holder, labels, t)
            with ui.status(v["label"], state="running", expanded=True):
                for line in v["log"]:
                    if line != v["label"]:  # SELFQ-VARIANTS-1108: the status header already shows the latest line
                        ui.write(line)
            for text, frac in v["bars"]:
                ui.progress(frac, text=text or None)
            ui.info(v["activity"]) if hasattr(ui, "info") else ui.caption(v["activity"])  # live status line: latest activity + elapsed
            ui.caption(v["caption"])
            ui.caption(labels.reload_caption)
        except Exception:  # noqa: BLE001  nothing raised in the body reaches the page
            ui.caption(labels.progress_unavailable)
        return False
    term = holder.terminal or "connection_lost"
    ses[STREAM_LAST_KEY] = StreamLast(term, holder.result, holder.status)
    ses.pop(STREAM_HOLDER_KEY, None)
    return True


def start_stream(ses, now, stream_fn, args, orphan_s: float, spawn=None) -> StreamHolder:
    """Single flight keyed on the session: a rerun returns the existing holder and never POSTs a second time."""
    cur = ses.get(STREAM_HOLDER_KEY)
    if cur is not None:
        return cur
    t = now()
    holder = StreamHolder(t0=t, last_read=t, last_line_at=t)
    ses[STREAM_HOLDER_KEY] = holder
    (spawn or _spawn)(stream_worker, holder, stream_fn, args, now, orphan_s)
    return holder
