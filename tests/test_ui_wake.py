"""UIWAKE-1022: the UI wakes a sleeping API before /ask; optional owner restart hook with a cooldown.
Offline: mocked HTTP, fake clock, no real sleeps, no network."""
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

from vera import ui_wake

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "streamlit_app.py")
DEMO = str(ROOT / "pages" / "vera_demo.py")
HOST = "upstream-sentinel.invalid"
HOOK = "https://hook-sentinel.invalid/deploy/srv-x?key=HOOKSECRET123"
CFG = ui_wake.load_config()


class R:
    def __init__(self, code=200, body=None):
        self.status_code, self._body = code, body or {}

    def json(self):
        return self._body


class Clock:
    def __init__(self):
        self.t, self.slept = 1000.0, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("VERA_API_KEY", "k-test")
    monkeypatch.setenv("VERA_API_BASE_URL", f"http://{HOST}:9")
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    monkeypatch.delenv(CFG.restart_hook_env, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    clk = Clock()
    monkeypatch.setattr(ui_wake, "clock", clk)
    monkeypatch.setattr(ui_wake, "sleep", clk.sleep)
    calls = {"health": 0, "ask": [], "hook": []}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["ask"].append(json["question"])
        return R(200, {"result_kind": "answer", "answer": "ok answer", "tokens_used": 1, "cost_usd": 0.0})

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(ui_wake, "http_post", lambda url, timeout=None: calls["hook"].append(url) or R(201))
    monkeypatch.setattr("requests.get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError(HOST)))
    return clk, calls


def script_health(monkeypatch, calls, plan):
    """plan: list of bools consumed per probe; the last value repeats."""
    def get(url, timeout=None):
        assert url.endswith(CFG.health_path) and timeout == CFG.health_timeout_s
        i = calls["health"]
        calls["health"] += 1
        if plan[min(i, len(plan) - 1)]:
            return R(200, {"status": "ok"})
        raise requests.ConnectionError(f"cannot reach {HOST}")
    monkeypatch.setattr(ui_wake, "http_get", get)


def blob(at):
    parts = [repr(at.session_state.filtered_state)]
    parts += [repr(el.proto) for el in at.main] + [repr(el.proto) for el in at.sidebar]
    return "\n".join(parts)


def ask(at, q="what is vera?"):
    at.text_input[0].set_value(q)
    at.button[0].click().run()
    return at


def test_config_values_and_cooldown_floor(tmp_path):
    assert sum(CFG.backoff_s) <= CFG.max_total_s <= 60
    assert CFG.restart_hook_cooldown_s >= ui_wake.COOLDOWN_FLOOR_S >= 900
    p = tmp_path / "w.json"
    p.write_text((ROOT / "config" / "ui_wake.json").read_text().replace('"restart_hook_cooldown_s": 900.0',
                                                                       '"restart_hook_cooldown_s": 5'))
    assert ui_wake.load_config(p).restart_hook_cooldown_s == ui_wake.COOLDOWN_FLOOR_S


def test_healthy_no_wait(env, monkeypatch):
    clk, calls = env
    script_health(monkeypatch, calls, [True])
    at = ask(AppTest.from_file(APP, default_timeout=30).run())
    assert not at.exception
    assert calls["ask"] == ["what is vera?"] and clk.slept == [] and calls["hook"] == []
    assert any("ok answer" in e.value for e in at.success)


def test_asleep_then_healthy_sends_pending_question(env, monkeypatch):
    clk, calls = env
    script_health(monkeypatch, calls, [True, False, False, True])  # page load ok; asleep at Ask; wakes on 2nd retry
    at = ask(AppTest.from_file(APP, default_timeout=30).run(), "pending question")
    assert not at.exception
    assert clk.slept == list(CFG.backoff_s[:2])
    assert calls["ask"] == ["pending question"]
    assert ui_wake.PENDING_KEY not in at.session_state
    assert any("ok answer" in e.value for e in at.success)
    assert calls["hook"] == []


def test_pending_question_survives_and_is_sent_on_next_run(env, monkeypatch):
    """A question queued in session state (e.g. a rerun cut the wait short) is sent on the next run, once healthy."""
    _, calls = env
    script_health(monkeypatch, calls, [True])
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.session_state[ui_wake.PENDING_KEY] = {"question": "kept", "force_bad": False}
    at.run()
    assert calls["ask"] == ["kept"] and ui_wake.PENDING_KEY not in at.session_state


@pytest.mark.parametrize("hook", [None, HOOK])
def test_never_healthy_friendly_error_no_host_or_hook(env, monkeypatch, hook):
    clk, calls = env
    if hook:
        monkeypatch.setenv(CFG.restart_hook_env, hook)
    script_health(monkeypatch, calls, [False])
    at = AppTest.from_file(APP, default_timeout=30).run()
    first_info = [e.value for e in at.info]
    clk.t += CFG.page_load_recheck_s + 1
    ask(at)
    assert not at.exception
    assert calls["ask"] == []  # never sent to a dead API
    assert sum(clk.slept) <= 2 * CFG.max_total_s  # page load + Ask, each within its window
    assert any(CFG.failed_message in e.value for e in at.error)
    shown = blob(at)
    for secret in (HOST, "hook-sentinel", "HOOKSECRET123", "ConnectionError"):
        assert secret not in shown
    if hook:
        assert calls["hook"] == [hook]  # page load window fired it; the Ask window is inside the cooldown
        assert CFG.restart_requested_message in first_info
    else:
        assert CFG.restart_requested_message not in first_info
        assert calls["hook"] == []


def test_hook_unset_never_called(env, monkeypatch):
    clk, calls = env
    script_health(monkeypatch, calls, [False])
    for _ in range(3):
        at = AppTest.from_file(APP, default_timeout=30).run()
        ask(at)
        clk.t += CFG.restart_hook_cooldown_s + 1
    assert calls["hook"] == []
    assert ui_wake.kick_restart(CFG, env={}) is False and calls["hook"] == []


def test_hook_set_called_at_most_once_within_cooldown(env, monkeypatch):
    clk, calls = env
    script_health(monkeypatch, calls, [False])
    monkeypatch.setenv(CFG.restart_hook_env, HOOK)
    t0 = clk.t
    for _ in range(2):  # two sessions, each with a page-load window and an Ask window
        ask(AppTest.from_file(APP, default_timeout=30).run())
    assert clk.t - t0 < CFG.restart_hook_cooldown_s  # all four windows fell inside one cooldown
    assert calls["hook"] == [HOOK]
    hook_env = {CFG.restart_hook_env: HOOK}
    assert ui_wake.kick_restart(CFG, env=hook_env) is False
    clk.t += CFG.restart_hook_cooldown_s + 1
    assert ui_wake.kick_restart(CFG, env=hook_env) is True and calls["hook"] == [HOOK, HOOK]


def test_hook_non_https_or_dev_mode_never_called(env, monkeypatch):
    _, calls = env
    script_health(monkeypatch, calls, [False])
    assert ui_wake.kick_restart(CFG, env={CFG.restart_hook_env: "http://plain.invalid/x"}) is False
    monkeypatch.setenv(CFG.restart_hook_env, HOOK)
    monkeypatch.setenv("VERA_PUBLIC_MODE", "0")  # dev mode: the URL is user-editable, so no hook
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].set_value("q")
    at.sidebar.text_input[1].set_value("typed-key")
    at.button[0].click().run()
    assert calls["hook"] == [] and calls["ask"] == []


def test_hook_failure_is_silent_and_counts(env, monkeypatch):
    _, calls = env

    def boom(url, timeout=None):
        calls["hook"].append(url)
        raise requests.ConnectionError(url)
    monkeypatch.setattr(ui_wake, "http_post", boom)
    hook_env = {CFG.restart_hook_env: HOOK}
    assert ui_wake.kick_restart(CFG, env=hook_env) is True
    assert ui_wake.kick_restart(CFG, env=hook_env) is False and len(calls["hook"]) == 1


def test_page_load_check_at_most_once_a_minute(env, monkeypatch):
    clk, calls = env
    script_health(monkeypatch, calls, [True])
    at = AppTest.from_file(APP, default_timeout=30).run()
    n = calls["health"]
    assert n == 1
    at.run()
    assert calls["health"] == n  # cached in session state
    clk.t += CFG.page_load_recheck_s + 1
    at.run()
    assert calls["health"] == n + 1


def test_report_page_wakes_then_sends(env, monkeypatch):
    clk, calls = env
    monkeypatch.setattr("vera.m7.live_report.build_live_report", lambda body, q, lat: {"question": q})
    monkeypatch.setattr("vera.m7.live_report.render_live_report", lambda st, rep, r: st.write("report shown"))
    script_health(monkeypatch, calls, [True, False, True])
    at = AppTest.from_file(DEMO, default_timeout=30).run()
    at.text_input[0].set_value("report q")
    at.button[0].click().run()
    assert not at.exception
    assert calls["ask"] == ["report q"] and clk.slept == [CFG.backoff_s[0]]
    assert HOST not in blob(at)



def test_bare_execution_never_probes_or_waits(env, monkeypatch):
    """Outside a Streamlit script run (e.g. a spawn child re-running __main__) nothing probes, waits or kicks."""
    clk, calls = env
    script_health(monkeypatch, calls, [False])
    monkeypatch.setenv(CFG.restart_hook_env, HOOK)

    class St:
        session_state = {}
    assert ui_wake.ensure_awake(St, f"http://{HOST}:9", allow_restart_hook=True) is True
    assert calls["health"] == 0 and clk.slept == [] and calls["hook"] == []
