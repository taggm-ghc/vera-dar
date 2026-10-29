"""PROG-1027 W4b-3: the Ask page with progress ON (AppTest, full reruns only: AppTest does not run the client-side
run_every timer). Offline: fake client, patched vera.ui_wake.http_get (with a status_code), no network, no waits."""
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import streamlit
from streamlit.testing.v1 import AppTest

from vera import progress
from vera import public_mode as pm
from vera import sources_sidebar as sb
from vera import ui_wake

ROOT = Path(__file__).resolve().parent.parent
HOST = "https://sentinel-host.invalid"
CFG = progress.validate_config({"enabled": True, "max_records": 8, "max_events_per_record": 8,
                                "max_events_per_poll": 8, "ttl_s": 600, "poll_interval_s": 1.0, "poll_margin_s": 5,
                                "first_event_grace_s": 10, "get_timeout_s": 2, "max_count": 10000})
ANSWER = {"answer": "Batching reduces latency.", "model": "m", "tokens_used": 5, "cost_usd": 0.0,
          "result_kind": "answer", "sources": [], "traced_sources": [], "recalled": []}
NOANS = {"answer": "", "model": "m", "tokens_used": 0, "cost_usd": 0.0, "result_kind": "reading_list"}


def ev(seq, stage, state, counts=None):
    return {"v": "1", "seq": seq, "ts": "t", "stage": stage, "state": state, "counts": counts or {}}


class Env:
    """Controls the fake client: POST blocks on `gate`; GET returns `events` (list | None | pm.GONE)."""

    def __init__(self):
        self.gate = threading.Event()
        self.posts, self.gets = [], []
        self.status, self.body, self.raises = 200, ANSWER, None
        self.json_raises = None
        self.events = []
        self.captions = []

    def post(self, base, question, mode, key, rid, token, timeout=180):
        self.cap_at_post = len(self.captions)
        self.posts.append({"rid": rid, "token": token, "question": question})
        self.gate.wait(10)
        if self.raises:
            raise self.raises
        if self.json_raises:
            def bad_json():
                raise self.json_raises
            return SimpleNamespace(status_code=self.status, json=bad_json)
        return SimpleNamespace(status_code=self.status, json=lambda: self.body)

    def get(self, base, key, rid, token, after, timeout=None):
        self.gets.append((rid, after))
        ev_ = self.events
        return ev_ if ev_ is None or ev_ is pm.GONE else [e for e in ev_ if e["seq"] > after]


@pytest.fixture
def env(monkeypatch):
    e = Env()
    prev = progress.CFG
    progress.configure(CFG)
    monkeypatch.delenv("VERA_PROGRESS_DISABLED", raising=False)
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    monkeypatch.setenv("VERA_API_KEY", "k")
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: {"commit": "abcdef1234"}))
    monkeypatch.setattr(ui_wake, "http_get", lambda *a, **k: SimpleNamespace(status_code=200))
    monkeypatch.setattr(pm, "post_ask_tracked", e.post)
    monkeypatch.setattr(pm, "get_progress", e.get)
    orig = streamlit.caption

    def spy(body, *a, **k):
        e.captions.append(("caption", body))
        e.last_cap = len(e.captions)
        return orig(body, *a, **k)
    monkeypatch.setattr(streamlit, "caption", spy)
    yield e
    e.gate.set()
    progress.configure(prev)


def _texts(at):
    return [c.value for c in at.caption] + [m.value for m in at.markdown] + [i.value for i in at.info] \
        + [x.value for x in at.error] + [w.value for w in at.warning] + [s.value for s in at.success]


def _all(at):
    return " ".join(_texts(at))


def _ask(env):
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    at.text_input[0].set_value("a question")
    at.button[0].click()
    at.run()
    assert not at.exception
    return at


def _finish(env, at):
    env.gate.set()
    for _ in range(200):  # the worker is a real thread; wait for it (bounded) before the next full rerun
        if "vera_inflight" not in at.session_state or at.session_state["vera_inflight"].holder.done.is_set():
            break
        threading.Event().wait(0.01)
    at.run()
    assert not at.exception
    return at


def test_flag_off_keeps_spinner_path(env, monkeypatch):
    progress.configure(progress.validate_config({**CFG.__dict__, "enabled": False}))
    monkeypatch.setattr(pm, "post_ask", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: ANSWER))
    at = _ask(env)
    assert "vera_inflight" not in at.session_state and not env.posts
    assert "Batching reduces latency." in _all(at)


def test_ac1_sending_label_with_rid_before_post_and_received_only_after_event(env):
    at = _ask(env)
    rid = env.posts[0]["rid"]
    sending = [b for _, b in env.captions if rid in b]
    assert sending and sending[0].startswith("Sending your question")
    first = next(i for i, (_, b) in enumerate(env.captions) if rid in b)
    assert first < env.cap_at_post  # the Sending label was drawn before the worker called POST
    assert len(env.posts) == 1 and len(rid) == 32
    assert "Question received" not in _all(at)  # no event yet: only the Sending label
    env.events = [ev(1, "received", "done")]
    at.run()
    assert f"Question received (request {rid})" in _all(at)
    assert len(env.posts) == 1


def test_ac1_received_shown_on_429_after_event_then_busy_terminal(env):
    env.status, env.body = 429, None
    env.events = [ev(1, "received", "done")]
    at = _ask(env)
    at.run()
    assert "Question received" in _all(at)
    at = _finish(env, at)
    assert "VERA-DAR is busy right now" in _all(at)
    assert "Batching" not in _all(at)


def test_label_sequence_and_background_line_absent(env):
    at = _ask(env)
    seen = []
    for evs in ([ev(1, "received", "done")], [ev(1, "received", "done"), ev(2, "write", "start")],
                [ev(1, "received", "done"), ev(2, "write", "start"), ev(3, "check", "start")]):
        env.events = evs
        at.run()
        seen.append(_all(at))
    assert "Question received" in seen[0] and "Writing the answer" in seen[1] and "Checking each claim" in seen[2]
    at = _finish(env, at)
    assert "VERA-DAR will keep looking" not in _all(at)


def test_ac6_two_runs_resume_with_one_post_same_rid_and_last_label(env):
    at = _ask(env)
    rid = env.posts[0]["rid"]
    env.events = [ev(1, "received", "done"), ev(2, "write", "start")]
    at.run()
    first = at.session_state["vera_inflight"].label
    at.run()  # interruption: a plain full rerun resumes the same entry
    assert len(env.posts) == 1 and at.session_state["vera_inflight"].rid == rid
    assert at.session_state["vera_inflight"].label == first == "Writing the answer from the usable passages"
    assert "Writing the answer from the usable passages" in _all(at)


def test_ac6_completion_only_after_holder_done_and_exactly_once(env):
    at = _ask(env)
    env.events = [ev(1, "received", "done"), ev(2, "done", "answered")]
    at.run()
    assert "vera_inflight" in at.session_state  # server done never ends the run
    assert "Finishing" in _all(at)
    at = _finish(env, at)
    assert "vera_inflight" not in at.session_state
    assert _all(at).count("Batching reduces latency.") == 1
    assert at.session_state["vera_last_result"].rendered is True


def test_ac6_raise_in_render_keeps_vera_last_result_and_renders_next_run(env, monkeypatch):
    at = _ask(env)
    env.gate.set()
    # make the first render raise via a sensitive renderer
    import vera.memory_panel as mp
    real = mp.render_memory_panel
    calls = {"n": 0}

    def boom(st, data):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("render failed " + HOST)
        return real(st, data)
    monkeypatch.setattr(mp, "render_memory_panel", boom)
    for _ in range(200):
        if at.session_state["vera_inflight"].holder.done.is_set():
            break
        threading.Event().wait(0.01)
    at.run()
    assert at.exception  # the page shows the failure, but the result is durable
    assert at.session_state["vera_last_result"].rendered is False
    at.run()
    assert not at.exception and "Batching reduces latency." in _all(at)
    assert at.session_state["vera_last_result"].rendered is True


def test_ac6b_ask_during_inflight_button_disabled_one_post_no_pending(env):  # the "already running" note (click race) is untested: unreachable under AppTest
    at = _ask(env)
    assert at.button[0].disabled is True
    at.text_input[0].set_value("second")
    at.button[0].click()
    at.run()
    assert len(env.posts) == 1
    assert ui_wake.PENDING_KEY not in at.session_state or not at.session_state[ui_wake.PENDING_KEY]


@pytest.mark.parametrize("status,body,raises,text", [
    (200, ANSWER, None, "Finished"),
    (200, {**ANSWER, "claim_check": {"answer_completeness": "partial"}}, None, "Partial answer: some parts"),
    (200, NOANS, None, "No answer: not enough usable evidence yet"),
    (500, None, None, "Something went wrong"),
    (200, "not-a-dict", None, "Something went wrong"),
    (200, None, requests.exceptions.ConnectTimeout(HOST), "Something went wrong"),
    (200, None, requests.exceptions.ConnectionError(HOST), "Something went wrong"),
])
def test_ac4_terminal_texts_and_no_host_or_token(env, status, body, raises, text):
    env.status, env.body, env.raises = status, body, raises
    at = _ask(env)
    token = at.session_state["vera_ptoken"]
    at = _finish(env, at)
    out = _all(at)
    assert text in out
    assert HOST not in out and token not in out
    assert "RuntimeError" not in out and "ConnectTimeout" not in out


def test_ac4_poll_timeout_variants_and_provider_error_and_expired(env):
    # ReadTimeout with RECEIVED seen -> "Still working"; unseen -> "No reply"
    env.raises = requests.exceptions.ReadTimeout(HOST)
    env.events = [ev(1, "received", "done")]
    at = _ask(env)
    at.run()
    at = _finish(env, at)
    rid = at.session_state["vera_last_result"].rid
    assert f"Still working after 3 minutes. The server may still finish; ask again later (request {rid})" in _all(at)


def test_ac4_poll_timeout_unreceived(env):
    env.raises = requests.exceptions.ReadTimeout(HOST)
    at = _ask(env)
    at = _finish(env, at)
    assert "No reply after 3 minutes; please ask again later" in _all(at)


def test_ac4_provider_error_after_server_done_error(env):
    env.status, env.body = 503, None
    env.events = [ev(1, "received", "done"), ev(2, "done", "error")]
    at = _ask(env)
    at.run()
    at = _finish(env, at)
    assert "The answer could not be generated right now" in _all(at)


def test_ac4_expired_when_connection_lost_after_received_and_final_get_404(env):
    env.raises = requests.exceptions.ConnectionError(HOST)
    env.events = [ev(1, "received", "done")]
    at = _ask(env)
    at.run()
    env.events = pm.GONE
    at = _finish(env, at)
    assert "The service restarted while answering" in _all(at)


def test_ac4_background_line_only_with_queued_count(env):
    env.events = [ev(1, "received", "done"), ev(2, "done", "answered", {"queued": 2})]
    at = _ask(env)
    at.run()
    at = _finish(env, at)
    out = _all(at)
    assert "VERA-DAR will keep looking for sources on this topic after this answer" in out
    assert "2" not in "".join(c.value for c in at.caption if "keep looking" in c.value)


def test_reload_caption_present_while_in_flight(env):
    at = _ask(env)
    env.events = [ev(1, "received", "done")]
    at.run()
    assert "Keep this tab open; reloading starts over" in _all(at)


def test_ac9_token_header_only_and_unknown_v2_events_show_fixed_text(env):
    env.events = None  # the guarded client returns None for a version-skewed body
    at = _ask(env)
    at.run()
    out = _all(at)
    assert at.session_state["vera_ptoken"] not in out and HOST not in out
    assert env.posts[0]["token"] == at.session_state["vera_ptoken"]
    assert len(env.posts[0]["token"]) == 43


DECLINE = {"answer": "", "model": "m", "tokens_used": 0, "cost_usd": 0.0, "result_kind": "decline",
           "evidence_status": "none", "contract": {"fallback": "insufficient_full_document_evidence"}}


def test_ac4_declined_shows_finished_and_verbatim_decline_body(env):
    from vera.ui_evidence import fallback_text
    env.body = DECLINE
    at = _ask(env)
    at = _finish(env, at)
    body = fallback_text("insufficient_full_document_evidence", n_inspected=0)
    assert body and body in [i.value for i in at.info]  # verbatim decline body, not just the caption
    assert "Finished" in [c.value for c in at.caption]


class _Json(ValueError):
    pass


@pytest.mark.parametrize("name,status,body,raises,json_raises,text", [
    ("post_404", 404, None, None, None, "Something went wrong"),
    ("html_200", 200, None, None, _Json("<html>" + HOST + "</html>"), "Something went wrong"),
    ("missing_key_200", 200, {"answer": "x"}, None, None, "No answer: not enough usable evidence yet"),
    ("read_timeout", 200, None, requests.exceptions.ReadTimeout(HOST), None, "No reply after 3 minutes"),
])
def test_ac4_ac9_sentinel_paths_fixed_text_no_host_no_token(env, name, status, body, raises, json_raises, text):
    env.status, env.body, env.raises, env.json_raises = status, body, raises, json_raises
    at = _ask(env)
    token = at.session_state["vera_ptoken"]
    at = _finish(env, at)
    out = _all(at)
    assert text in out, (name, out)
    assert HOST not in out and token not in out and "<html>" not in out and "_Json" not in out
    assert "Traceback" not in out and not at.exception


def _triage_spy(monkeypatch):
    import vera.api_triage as tri
    calls = []

    def fake(base, path, http_get=None):
        calls.append((base, path))
        return [{"step": "1. API address configured", "ok": True, "detail": "set"},
                {"step": "2. API reachable", "ok": False, "detail": "nothing answered"}]
    monkeypatch.setattr(tri, "triage_api", fake)
    return calls


@pytest.mark.parametrize("status,raises", [
    (500, None), (503, None), (200, requests.exceptions.ConnectTimeout(HOST)),
    (200, requests.exceptions.ConnectionError(HOST))])
def test_f2_item83_public_flag_on_5xx_or_no_response_runs_ordered_triage(env, monkeypatch, status, raises):
    calls = _triage_spy(monkeypatch)
    env.status, env.body, env.raises = status, None, raises
    at = _ask(env)
    at = _finish(env, at)
    out = _all(at)
    assert "Something went wrong" in out
    assert len(calls) == 1 and calls[0][1] == "/ask"
    assert "❌ 2. API reachable: nothing answered" in out and "✅ 1. API address configured: set" in out
    assert HOST not in out


@pytest.mark.parametrize("status,body", [(429, None), (200, "not-a-dict"), (404, None)])
def test_f2_no_triage_for_vera_429_or_non_5xx(env, monkeypatch, status, body):
    calls = _triage_spy(monkeypatch)
    env.status, env.body = status, body
    at = _ask(env)
    at = _finish(env, at)
    assert calls == []
    if status == 429:
        assert "VERA-DAR is busy right now" in _all(at)


def test_f5_timeout_wording_derives_from_ask_timeout_s(monkeypatch):
    import json
    import vera.progress_ui as pu
    raw = json.loads(pu.LABELS_PATH.read_text(encoding="utf-8"))
    assert "3 minutes" not in json.dumps(raw)  # no duplicated literal in the config
    labels = pu.Labels.load()
    assert "after 3 minutes" in labels.terminal_text("poll_timeout_received", "r1")
    monkeypatch.setattr(pu, "ASK_TIMEOUT_S", 240)
    assert "after 4 minutes" in labels.terminal_text("poll_timeout_received", "r1")
    assert "after 4 minutes" in labels.terminal_text("poll_timeout_unreceived", "r1")
    for secs, phrase in ((60, "1 minute"), (90, "2 minutes"), (180, "3 minutes")):
        monkeypatch.setattr(pu, "ASK_TIMEOUT_S", secs)
        text = labels.terminal_text("poll_timeout_received", "r1")
        assert f"after {phrase}." in text, (secs, text)
    bad = dict(raw, stages={**raw["stages"], "write": {**raw["stages"]["write"], "start": "{minutes}"}})
    with pytest.raises(ValueError):
        pu.Labels(bad)  # {minutes} is terminals-only


def test_t_public_flag_on_poll_timeout_status_none_runs_triage(env, monkeypatch):
    """Current behaviour: a poll timeout (terminal status None) on the public flag-on path also runs the ordered triage."""
    calls = _triage_spy(monkeypatch)
    env.raises = requests.exceptions.ReadTimeout(HOST)
    at = _ask(env)
    at = _finish(env, at)
    assert at.session_state["vera_last_result"].status is None
    assert "after 3 minutes" in _all(at)
    assert len(calls) == 1 and calls[0][1] == "/ask"
