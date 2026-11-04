"""Reusable sentinel leak helper (NONASK-1103 S6, plan 6.8). Offline, stdlib only.

    s = leak_sentinel.new()                      # unique marker; plant it in a stubbed page body
    with leak_sentinel.capture_logs() as logs:   # captures root handlers + a catch-all handler
        ... run the request ...
    leak_sentinel.assert_no_leak(s, db=fake_db.calls, logs=logs, caplog=caplog, trace=contract["trace"],
                                 caches=[lru, disk], live_stats=rows, exceptions=[exc], extra={"panel": payload})
Any hit raises AssertionError naming only the SINK label, never the surrounding text."""
from __future__ import annotations

import base64
import contextlib
import io
import logging
import secrets
import sys
import urllib.parse
import warnings


def new(prefix: str = "LEAKSENTINEL") -> str:
    return f"{prefix}-{secrets.token_hex(8)}"


def variants(s: str) -> list[str]:
    b = s.encode()
    out = [s, s.lower(), s.upper(), urllib.parse.quote(s), b.hex(), s.replace("-", " ")]
    for k, drop in ((0, 0), (1, 2), (2, 3)):  # the three base64 alignments of an embedded value
        enc = base64.b64encode(b"\0" * k + b).decode().rstrip("=")
        enc = enc[drop:]
        out.append(enc[:-1] if len(enc) > 8 else enc)
    return out


def flatten(obj, _seen=None, _depth=0) -> str:
    """Deep text rendering of dicts, lists, sets, tuples, bytes, exceptions (args, cause, context), objects (__dict__)."""
    _seen = _seen if _seen is not None else {}
    if _depth > 200:
        raise AssertionError("leak_sentinel: nesting too deep to scan; sink not provably clean")
    if id(obj) in _seen:
        return ""
    _seen[id(obj)] = obj  # strong ref: a freed temporary must never donate its id to a later object
    if obj is None:
        return ""
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", "replace") + " " + bytes(obj).hex()
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return " ".join(flatten(k, _seen, _depth + 1) + " " + flatten(v, _seen, _depth + 1) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return " ".join(flatten(v, _seen, _depth + 1) for v in obj)
    if isinstance(obj, BaseException):
        import traceback
        return " ".join([repr(obj), str(obj), "".join(traceback.format_exception(obj)),
                         flatten(obj.args, _seen, _depth + 1), flatten(obj.__cause__, _seen, _depth + 1),
                         flatten(obj.__context__, _seen, _depth + 1)])
    if isinstance(obj, logging.LogRecord):
        return " ".join([obj.getMessage(), flatten(obj.args, _seen, _depth + 1), obj.exc_text or "",
                         flatten(obj.__dict__, _seen, _depth + 1)])
    d = getattr(obj, "__dict__", None)
    slots = {n: getattr(obj, n, None) for c in type(obj).__mro__ for n in getattr(c, "__slots__", ()) if isinstance(n, str)}
    if slots:
        d = {**(d or {}), **slots}
    return repr(obj) + (" " + flatten(d, _seen, _depth + 1) if d else "")


class Capture:
    """Everything the run emitted: log records (every logger, any level, propagate or not, handlers filtered or not),
    stdout, stderr and warnings."""

    def __init__(self):
        self.records: list[logging.LogRecord] = []
        self.out = io.StringIO()
        self.err = io.StringIO()
        self.warnings: list = []

    @property
    def stdout(self):
        return self.out.getvalue()

    @property
    def stderr(self):
        return self.err.getvalue()


@contextlib.contextmanager
def capture_all():
    """Patch Logger.isEnabledFor (level/disable bypass) and Logger.handle (records first, before filters and
    propagation), redirect stdout/stderr, record warnings. Self-checks with a probe on a propagate=False logger with
    a high level and raises RuntimeError if the probe is not captured. (A capsys/capfd fixture can also be passed to
    assert_no_leak for output written below Python's sys.stdout.)"""
    cap = Capture()
    orig_handle, orig_enabled = logging.Logger.handle, logging.Logger.isEnabledFor

    def handle(self, record):
        cap.records.append(record)
        return orig_handle(self, record)
    logging.Logger.handle = handle
    logging.Logger.isEnabledFor = lambda self, level: True
    try:
        with contextlib.redirect_stdout(cap.out), contextlib.redirect_stderr(cap.err), \
                warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cap.warnings = w
            probe = logging.getLogger("leak_sentinel.probe")
            probe.propagate, probe.disabled = False, False
            probe.setLevel(logging.CRITICAL)
            probe.debug("probe")
            print("probe", end="")
            if not cap.records or cap.stdout != "probe":
                raise RuntimeError("leak_sentinel: capture is not working; refusing to report a clean run")
            cap.records.clear()
            cap.out.seek(0); cap.out.truncate()
            yield cap
    finally:
        logging.Logger.handle, logging.Logger.isEnabledFor = orig_handle, orig_enabled


capture_logs = capture_all  # back-compat name; yields a Capture (use .records)


def root_handler_text() -> str:
    """Text held by in-memory handlers on the root logger and every named logger (incl. FileHandlers)."""
    loggers = [logging.getLogger()] + [x for x in logging.Logger.manager.loggerDict.values() if isinstance(x, logging.Logger)]
    out = []
    for lg in loggers:
        for h in lg.handlers:
            for attr in ("records", "buffer"):
                out.append(flatten(getattr(h, attr, None)))
            st = getattr(h, "stream", None)
            if hasattr(st, "getvalue"):
                out.append(st.getvalue())
            fn = getattr(h, "baseFilename", None)
            if fn:
                try:
                    h.flush()
                    out.append(open(fn, encoding="utf-8", errors="replace").read())
                except OSError:
                    raise AssertionError("leak_sentinel: cannot read a file log handler") from None
    return " ".join(out)


def find_leaks(sentinel: str, **sinks) -> list[str]:
    """Sink labels whose rendering contains the sentinel (or an encoded variant)."""
    vs = variants(sentinel)
    hit = []
    for name, obj in sinks.items():
        if name == "caplog":
            text = flatten(getattr(obj, "records", [])) + " " + getattr(obj, "text", "")
        elif name in ("capsys", "capfd"):
            o, e = obj.readouterr()
            text = o + " " + e
        elif isinstance(obj, Capture):
            text = flatten(obj.records) + " " + obj.stdout + " " + obj.stderr + " " + flatten(
                [(str(w.message), w.category.__name__) for w in obj.warnings])
        else:
            text = flatten(obj)
        if any(v in text for v in vs):
            hit.append(name)
    rh = root_handler_text()
    if any(v in rh for v in vs):
        hit.append("root_handlers")
    return hit


def assert_no_leak(sentinel: str, **sinks) -> None:
    hit = find_leaks(sentinel, **sinks)
    assert not hit, f"sentinel reached sink(s): {', '.join(hit)}"
