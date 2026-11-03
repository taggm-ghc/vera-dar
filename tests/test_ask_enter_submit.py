"""WP-J: Enter in the question box is a form submit identical to clicking Ask; in-flight submits queue nothing."""
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from tests.test_discovery_panel import ROOT, body
from vera import sources_sidebar as sb


def _app(monkeypatch, calls):
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    monkeypatch.setenv("VERA_API_KEY", "k")
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    import requests

    def post(*a, **k):
        calls.append(a)
        return SimpleNamespace(status_code=200, json=lambda: body("reading_list"))
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: {"commit": "abcdef1234"}))
    return AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()


def test_input_and_ask_share_one_form(monkeypatch):
    at = _app(monkeypatch, [])
    assert not at.exception
    src = (ROOT / "streamlit_app.py").read_text()
    assert 'st.form("ask_form"' in src and "st.form_submit_button(\"Ask\"" in src
    assert at.button[0].label == "Ask"


def _spy(monkeypatch):
    from vera import ui_wake
    queued = []
    monkeypatch.setattr(ui_wake, "queue_question", lambda state, payload: queued.append(dict(payload)))
    return queued


def test_submit_queues_question(monkeypatch):
    queued = _spy(monkeypatch)
    at = _app(monkeypatch, [])
    at.text_input[0].set_value("What is VERA-DAR?")
    at.button[0].click().run()
    assert not at.exception
    assert queued == [{"question": "What is VERA-DAR?", "force_bad": False}]


def test_empty_submit_warns(monkeypatch):
    queued = _spy(monkeypatch)
    at = _app(monkeypatch, [])
    at.button[0].click().run()
    assert any("Please enter a question" in w.value for w in at.warning)
    assert queued == []


def _in_flight(monkeypatch, key):
    from vera import progress_ui
    queued = _spy(monkeypatch)
    at = _app(monkeypatch, [])
    at.session_state[key] = SimpleNamespace(rid="r")
    monkeypatch.setattr(progress_ui, "stream_tick", lambda *a, **k: False)
    monkeypatch.setattr(progress_ui, "poll_tick", lambda *a, **k: False)
    at.text_input[0].set_value("late question")
    at.button[0].click().run()
    return at, queued


def test_submit_while_stream_running_queues_nothing(monkeypatch):
    # race: a forced submit while the stream holder exists must not queue a question that fires afterwards
    from vera import progress_ui
    at, queued = _in_flight(monkeypatch, progress_ui.STREAM_HOLDER_KEY)
    assert queued == []


def test_race_holder_set_queues_nothing():
    src = (ROOT / "streamlit_app.py").read_text()
    assert "if _clicked and (_inflight is not None or _sholder is not None):" in src
