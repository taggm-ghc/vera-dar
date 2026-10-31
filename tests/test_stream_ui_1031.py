"""OD-1031-STREAM-ON T10/T10f: stream UI (offline; fake clock, fake stream; AppTest for the rerun-resume case)."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import streamlit
from streamlit.testing.v1 import AppTest

from vera import progress, progress_ui
from vera import public_mode as pm
from vera import sources_sidebar as sb
from vera import ui_wake

ROOT = Path(__file__).resolve().parent.parent
ANSWER = {"answer": "Batching reduces latency.", "model": "m", "tokens_used": 5, "cost_usd": 0.0,
          "result_kind": "answer", "sources": [], "traced_sources": [], "recalled": []}


def ev(seq, stage, state, counts=None):
    return {"v": "1.3", "type": "event", "seq": seq, "ts": "t", "stage": stage, "state": state, "counts": counts or {}}


def result(status=200, kind=None):
    return {"v": "1.3", "type": "result", "status": status, **({"body": ANSWER} if status == 200 else {"kind": kind})}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakeUI:
    def __init__(self):
        self.out = []

    def status(self, label, **k):
        self.out.append(("status", label))
        return _Ctx()

    def write(self, t):
        self.out.append(("write", t))

    def progress(self, f, text=""):
        self.out.append(("progress", round(f, 2), text))

    def caption(self, t):
        self.out.append(("caption", t))


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture(scope="module")
def labels():
    return progress_ui.Labels.load()


def make_fn(lines, raises=None, gate=None):
    calls = []

    def fn(*args, should_stop=None):
        calls.append(args)

        def gen():
            for ln in lines:
                if gate is not None:
                    gate.wait(5)
                if should_stop and should_stop():
                    return
                yield ln
            if raises:
                raise raises
        g = gen()
        return SimpleNamespace(__iter__=None) if False else _It(g)
    fn.calls = calls
    return fn


class _It:
    dropped = 0

    def __init__(self, g):
        self.g, self.closed = g, False

    def __iter__(self):
        return self.g

    def close(self):
        self.closed = True


def run_sync(fn, clock, orphan_s=120, holder=None):
    h = holder or progress_ui.StreamHolder(t0=clock(), last_read=clock(), last_line_at=clock())
    progress_ui.stream_worker(h, fn, ("u", "q", "normal", "k"), clock, orphan_s)
    return h


def test_answer_terminal_and_view(labels):
    clk = Clock()
    lines = [ev(1, "received", "done"), ev(2, "search", "done", {"found": 12, "shortlisted": 9}),
             ev(3, "acquire", "update", {"checked": 7, "shortlisted": 9, "usable": 2, "held": 3, "stored": 2, "target": 6}),
             result()]
    h = run_sync(make_fn(lines), clk)
    assert h.terminal == "answer" and h.result == ANSWER and h.done.is_set()
    clk.t += 252
    v = progress_ui.stream_view(h, labels, clk())
    assert v["label"].startswith("Checked 7 of 9") and "12 possible sources found; 9 to check" in v["log"]
    assert ("Checked 7 of 9", pytest.approx(7 / 9)) == (v["bars"][0][0], pytest.approx(v["bars"][0][1]))
    assert v["bars"][1][0] == "Saved 2 of 6"
    assert v["caption"] == "Still working · 4 min 12 s · last update 252 s ago"


def test_acquire_limited_why_text(labels):
    t = labels.stage_label("acquire", "limited", {"stored": 3, "target": 6, "why": 1}, "")
    assert t and "3" in t and "no more open sources" in t.lower()
    assert labels.stage_label("acquire", "busy", {}, "")


@pytest.mark.parametrize("res,term", [(result(429, "busy"), "busy"), (result(500, "error"), "error"),
                                      (result(502, "provider_error"), "provider_error"),
                                      (result(503, "unavailable"), "unavailable")])
def test_result_terminals(res, term):
    assert run_sync(make_fn([ev(1, "search", "start", {"step": 1}), res]), Clock()).terminal == term


def test_stream_failures_map_to_terminals(labels):
    NC, SL = pm.StreamNeverConnected, pm.StreamLost
    assert run_sync(make_fn([], raises=NC(None)), Clock()).terminal == "never_connected"
    assert run_sync(make_fn([], raises=NC(429)), Clock()).terminal == "busy"
    assert run_sync(make_fn([ev(1, "search", "start")], raises=SL("connection")), Clock()).terminal == "connection_lost"
    assert run_sync(make_fn([ev(1, "search", "start")], raises=SL("no_data")), Clock()).terminal == "stalled"
    assert run_sync(make_fn([ev(1, "search", "start")]), Clock()).terminal == "connection_lost"  # ended, no result
    assert run_sync(make_fn([], raises=RuntimeError("boom http://x.invalid")), Clock()).terminal == "connection_lost"
    for k in ("answer", "orphaned"):
        pass
    for k in ("busy", "error", "provider_error", "unavailable", "connection_lost", "never_connected", "stalled"):
        t = labels.terminal_text(k)
        assert t and "{" not in t


def test_t10f_reset_mid_stream_is_connection_lost():
    h = run_sync(make_fn([ev(1, "search", "start")], raises=requests.exceptions.ConnectionError("reset")), Clock())
    # fake raises a raw requests error (not mapped by the client): still a closed terminal, no text kept
    assert h.terminal == "connection_lost" and not hasattr(h, "error")


def test_t10f_orphan_close_after_orphan_s_fake_clock():
    clk = Clock()
    gate = threading.Event()
    gate.set()
    h = progress_ui.StreamHolder(t0=clk(), last_read=clk(), last_line_at=clk())
    stops = []

    def fn(*args, should_stop=None):
        def gen():
            for i in range(1, 6):
                if i == 3:
                    clk.t += 121  # nobody ticked the fragment for > orphan_s
                if should_stop():
                    stops.append(i)
                    return
                yield ev(i, "search", "start")
        it = _It(gen())
        fn.it = it
        return it
    progress_ui.stream_worker(h, fn, (), clk, 120)
    assert stops == [3] and h.terminal == "orphaned" and fn.it.closed and h.done.is_set()


def test_tick_stamps_last_read_and_completes(labels):
    clk = Clock()
    ses, ui = {}, FakeUI()
    h = progress_ui.StreamHolder(t0=clk(), last_read=clk() - 50, last_line_at=clk())
    ses[progress_ui.STREAM_HOLDER_KEY] = h
    clk.t += 5
    assert progress_ui.stream_tick(h, ses, ui, clk, labels) is False and h.last_read == clk()
    assert any(o[0] == "status" for o in ui.out) and any(o[0] == "caption" and "Still working" in o[1] for o in ui.out)
    h.terminal, h.result = "answer", ANSWER
    h.done.set()
    assert progress_ui.stream_tick(h, ses, ui, clk, labels) is True
    assert progress_ui.STREAM_HOLDER_KEY not in ses and ses[progress_ui.STREAM_LAST_KEY].terminal == "answer"


def test_start_stream_single_flight():
    clk, ses, spawned = Clock(), {}, []
    sp = lambda *a: spawned.append(a)  # noqa: E731
    h1 = progress_ui.start_stream(ses, clk, lambda *a, **k: None, ("a",), 120, spawn=sp)
    h2 = progress_ui.start_stream(ses, clk, lambda *a, **k: None, ("a",), 120, spawn=sp)
    assert h1 is h2 and len(spawned) == 1


# ---- T10: AppTest rerun resumes from the holder without a second POST ------------------------------------------------
def test_t10_rerun_resumes_without_second_post(monkeypatch):
    gate = threading.Event()
    posts = []

    def fake_stream(*args, connect_timeout=None, no_data_timeout=None, should_stop=None):
        posts.append(args)

        def gen():
            yield ev(1, "search", "start", {"step": 1})
            gate.wait(10)
            yield result()
        return _It(gen())
    import dataclasses
    base = progress.CFG if progress.CFG is not None else progress.validate_config(
        {"enabled": False, "max_records": 8, "max_events_per_record": 8, "max_events_per_poll": 8, "ttl_s": 600,
         "poll_interval_s": 1.0, "poll_margin_s": 5, "first_event_grace_s": 10, "get_timeout_s": 2, "max_count": 10000})
    cfg = dataclasses.replace(base, stream=progress.StreamCfg(
        enabled=True, heartbeat_s=10, client_no_data_s=60, connect_timeout_s=10, orphan_s=120))
    monkeypatch.setattr(progress, "CFG", cfg)
    monkeypatch.setattr(progress, "stream_enabled", lambda: True, raising=False)
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    monkeypatch.setenv("VERA_API_KEY", "k")
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: {"commit": "abcdef1234"}))
    monkeypatch.setattr(ui_wake, "http_get", lambda *a, **k: SimpleNamespace(status_code=200))
    monkeypatch.setattr(pm, "iter_ask_stream", fake_stream)
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    at.text_input[0].set_value("a question")
    at.button[0].click()
    at.run()
    assert not at.exception and len(posts) == 1
    holder = at.session_state[progress_ui.STREAM_HOLDER_KEY]
    for _ in range(100):
        if holder.events:
            break
        threading.Event().wait(0.02)
    at.run()
    at.run()  # reruns while the stream is open: same holder, no second POST
    assert not at.exception and len(posts) == 1
    assert at.session_state[progress_ui.STREAM_HOLDER_KEY] is holder
    assert at.button[0].disabled
    gate.set()
    holder.done.wait(5)
    at.run()  # AppTest does not run the fragment timer: this rerun's tick moves the finished stream to the last result
    if progress_ui.STREAM_HOLDER_KEY in at.session_state:
        at.run()
    assert not at.exception and len(posts) == 1
    assert "Batching reduces latency." in " ".join(m.value for m in at.markdown) + " ".join(s.value for s in at.success)


# ---- lint: no host/URL/portal names in labels, events or UI strings --------------------------------------------------
def _portal_names():
    p = ROOT / "config" / "rate_limits.json"
    try:
        raw = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return set()
    return {str(k).lower() for k in (raw.get("portals") or {}) if len(str(k)) >= 4}


def test_lint_no_host_url_portal_in_ui_strings(labels):
    texts = [json.dumps(json.loads((ROOT / "config" / "progress-stages.json").read_text()))]
    texts += list(progress_ui._GENERIC_TERMINALS.values())
    clk = Clock()
    h = run_sync(make_fn([ev(1, "search", "done", {"found": 3, "shortlisted": 2}),
                          ev(2, "acquire", "limited", {"stored": 1, "target": 6, "why": 2}), result()]), clk)
    v = progress_ui.stream_view(h, labels, clk())
    texts += [v["label"], v["caption"], *v["log"], *[b[0] for b in v["bars"]]]
    blob = " ".join(texts).lower()
    for bad in ("http://", "https://", "www.", ".org", ".com", "doi", "arxiv", "pubmed", "openalex", "crossref", "unpaywall"):
        assert bad not in blob, bad
    for n in _portal_names():
        assert n not in blob, n
