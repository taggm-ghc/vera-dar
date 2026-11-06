"""NOISY-UI-1031 (owner 2026-10-31): every wait / slow step is visible on the stream and in the Streamlit status line.
Offline: fake clocks and sleeps; counts and generic text only."""
import asyncio
import json

import main
from vera import inline_acquire as ia
from vera import progress
from vera import public_mode as pm
from vera import selective_fetch as sf
from vera.progress_ui import Labels, StreamHolder, stream_tick, stream_view


import pytest


@pytest.fixture(autouse=True)
def _reset_sink():
    tok = progress.SINK.set(None)
    yield
    progress.SINK.reset(tok)


def _sink():
    s = main._StreamSink(100)
    progress.SINK.set(s)
    return s


def _events(s):
    out = []
    while not s.items.empty():
        out.append(s.items.get_nowait())
    return [e for e in out if e[0] == "event"]


def test_wait_emits_event_immediately_then_clears():
    s = _sink()
    with progress.waiting(progress.ACT_RATE, 3):
        ev = _events(s)
        assert len(ev) == 1 and ev[0][3] == {"act": 1, "wait_s": 3}  # visible at wait start (<= 10 s by construction)
        assert s.hb_fields()["act"] == 1 and s.hb_fields()["wait_s"] == 3
    assert s.hb_fields() == {}


def test_politeness_wait_in_pacer_is_visible_during_the_sleep():
    s = _sink()
    seen = []
    clock = [100.0]
    p = sf.HostPacer(intervals={"example.invalid": 5.0}, clock=lambda: clock[0],
                     sleep=lambda g: seen.append(dict(s.hb_fields())))
    p.last["example.invalid"] = 100.0
    p.wait("https://example.invalid/x")
    assert seen and seen[0]["act"] == 2 and seen[0]["wait_s"] == 5 and s.hb_fields() == {}


def test_slow_read_reports_at_10s_and_every_10s_after():
    s = _sink()
    t = [0.0]
    started = {"f": (3, 0.0)}
    slow = {"mark": 0}
    marks = []
    for now in (5, 10, 15, 20, 31):
        t[0] = now
        ia._slow_read({"f"}, started, 8, slow, now=lambda: t[0])
        marks.append(s.hb_fields().get("elapsed"))
    assert marks == [None, 10, 10, 20, 31]
    ev = [e[3] for e in _events(s)]
    assert [e["elapsed"] for e in ev] == [10, 20, 31] and all(e["k"] == 1 and e["n"] == 8 and e["act"] == 3 for e in ev)
    ia._slow_read(set(), started, 8, slow)
    assert s.hb_fields() == {}


def test_heartbeat_carries_activity_and_validates():
    s = _sink()
    s.set_activity(3, {"k": 2, "n": 5})
    s.put_result(200, body={"answer": "A"})
    s.close()

    async def run():
        out = []
        async for ln in main._stream_lines(s, None, None, 0.05):
            out.append(json.loads(ln))
        return out
    # the queued event and result are served first; add a heartbeat by draining after the event with a held-open sink
    s2 = main._StreamSink(100)
    s2.set_activity(3, {"k": 2, "n": 5})

    async def hb_run():
        out = []
        gen = main._stream_lines(s2, None, None, 0.05)
        async for ln in gen:
            d = json.loads(ln)
            out.append(d)
            if d["type"] == "hb":
                s2.put_result(0)
                break
        return out
    lines = asyncio.run(hb_run())
    hb = [x for x in lines if x["type"] == "hb"][0]
    assert hb["act"] == 3 and hb["k"] == 2 and hb["n"] == 5 and "elapsed" in hb and pm.validate_stream_line(hb)
    assert not any(isinstance(v, str) for k, v in hb.items() if k not in ("v", "type"))
    assert pm.validate_stream_line({"v": hb["v"], "type": "hb", "seq": 1, "elapsed_s": 1, "url": "x"}) is False


def test_labels_render_generic_activity_text_for_every_code():
    lab = Labels.load()
    for code, counts in ((1, {"wait_s": 4}), (2, {"wait_s": 4}), (3, {"k": 2, "n": 5, "elapsed": 20}), (4, {"wait_s": 4}),
                         (5, {}), (6, {"elapsed": 12})):
        t = lab.activity_label({"act": code, **counts})
        assert t and "{" not in t
    assert "20 s" in lab.activity_label({"act": 3, "k": 2, "n": 5, "elapsed": 20})
    assert "rate limit" in lab.activity_label({"act": 1, "wait_s": 4}) and "backup" in lab.activity_label({"act": 5})


class FakeUI:
    def __init__(self):
        self.calls = []

    def status(self, label, state, expanded):
        self.calls.append(("status", label))
        import contextlib
        return contextlib.nullcontext()

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a[0] if a else None))


def test_streamlit_status_line_shows_latest_activity_and_elapsed():
    lab = Labels.load()
    h = StreamHolder(t0=0.0, last_read=0.0, last_line_at=0.0)
    h.events.append({"stage": "acquire", "state": "start", "counts": {"budget": 6}})
    h.activity = {"act": 1, "wait_s": 7}
    ui = FakeUI()
    assert stream_tick(h, {}, ui, lambda: 95.0, lab) is False
    infos = [c[1] for c in ui.calls if c[0] == "info"]
    assert infos and "rate limit" in infos[0] and "1 min 35 s elapsed" in infos[0]
    h.activity = {"act": 3, "k": 1, "n": 4, "elapsed": 30}  # the line changes with the activity
    ui2 = FakeUI()
    stream_tick(h, {}, ui2, lambda: 96.0, lab)
    assert "Still reading 1 source(s) at once; the slowest has taken 30 s" in [c[1] for c in ui2.calls if c[0] == "info"][0]


def test_no_events_yet_says_waiting_for_a_connection():
    lab = Labels.load()
    h = StreamHolder(t0=0.0, last_read=0.0, last_line_at=0.0)
    assert "Waiting for a connection" in stream_view(h, lab, 9.0)["activity"]


def test_grace_fallback_is_visible(monkeypatch):
    from types import SimpleNamespace

    from vera import ask_service as svc
    s = _sink()
    svc.complete_after_exhaustion("q", None, {"work": True, "decision": SimpleNamespace(accepted=False, tokens_used=0, cost_usd=0.0)})
    assert s.hb_fields()["act"] == 5 and any(e[3].get("act") == 5 for e in _events(s))


def test_event_elapsed_is_preserved_in_heartbeat_fields():  # EXG-FIX-1031 N-D1
    s = _sink()
    s.set_activity(3, {"k": 1, "n": 2, "elapsed": 20})
    assert s.hb_fields()["elapsed"] == 20
