"""WAKE-1023: Streamlit AppTest drive of the real Ask page (streamlit_app.py) against the SHIPPED config/ui_wake.json
with fake HTTP and a fake clock/sleep. No sockets, no network. Checks: question preserved, "~3 minutes" starting message,
no /ask before ready, clean failure, no host shown."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import streamlit
from streamlit.testing.v1 import AppTest

from vera import sources_sidebar as sb
from vera import ui_wake

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "streamlit_app.py")
CFG_PATH = ROOT / "config" / "ui_wake.json"
CFG = ui_wake.load_config(CFG_PATH)
HOST = "sentinel-wakeui-host.invalid"
URL = f"https://{HOST}"
Q = "What is the sentinel question about wake-up?"


class Net:
    """API is down until `ready_at` (fake-clock seconds). unready='hang': probe burns its timeout; 'refuse': fails fast."""
    def __init__(self, clk, ready_at, unready="hang"):
        self.clk, self.ready_at, self.unready = clk, ready_at, unready
        self.health, self.posts, self.statuses = [], [], []

    def get(self, url, timeout=None):
        if not url.endswith(CFG.health_path):
            raise requests.ConnectionError("n/a")  # /version footer etc.
        if self.clk.t < self.ready_at:
            if self.unready == "hang":
                self.clk.t += timeout
            self.health.append((round(self.clk.t - self.clk.t0, 1), "down"))
            raise requests.ConnectionError(f"down {HOST}")
        self.health.append((round(self.clk.t - self.clk.t0, 1), "200"))
        return SimpleNamespace(status_code=200)

    def post(self, url, **kw):
        self.posts.append((round(self.clk.t - self.clk.t0, 1), url.rsplit("/", 1)[-1]))
        return SimpleNamespace(status_code=200, json=lambda: {"result_kind": "decline", "answer": "", "sources": []})


class Clk:
    def __init__(self):
        self.t0 = self.t = 5000.0
    def __call__(self):
        return self.t


@pytest.fixture
def env(monkeypatch):
    clk = Clk()
    labels = []
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    monkeypatch.setenv("VERA_API_KEY", "k")
    monkeypatch.setenv("VERA_API_BASE_URL", URL)
    monkeypatch.delenv("VERA_API_RESTART_HOOK", raising=False)
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    monkeypatch.setattr(ui_wake, "load_config", lambda path=CFG_PATH: ui_wake.WakeConfig(**{**CFG.__dict__}))
    monkeypatch.setattr(ui_wake, "clock", clk)
    monkeypatch.setattr(ui_wake, "sleep", lambda s: setattr(clk, "t", clk.t + s))
    orig_status = streamlit.status
    def rec_status(label, *a, **k):
        labels.append(label)
        return orig_status(label, *a, **k)
    monkeypatch.setattr(streamlit, "status", rec_status)
    ui_wake.reset_hook_cooldown()
    return SimpleNamespace(clk=clk, labels=labels, mp=monkeypatch)


def install(env, net):
    env.mp.setattr(ui_wake, "http_get", net.get)
    env.mp.setattr(requests, "get", net.get)
    env.mp.setattr(requests, "post", net.post)


def visible(at):
    out = []
    for kind in ("markdown", "error", "info", "warning", "success", "caption", "text"):
        out += [str(e.value) for e in getattr(at, kind)]
    for s in at.get("status"):
        out.append(str(getattr(s.proto, "label", "")))
    return out


def ask(at, text=Q):
    at.text_input[0].set_value(text)
    at.button[0].click()
    at.run()
    assert not at.exception, at.exception


def scenario(env, name, *, ready_delay, unready="hang", url_unreachable=False):
    """Load the page with API healthy (page-load check ok), then API goes down for `ready_delay` s from the click."""
    net = Net(env.clk, ready_at=-1)
    install(env, net)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    click_t = env.clk.t
    net.unready = unready
    net.ready_at = float("inf") if ready_delay is None else click_t + ready_delay
    net.health.clear(); env.labels.clear()
    ask(at)
    return at, net, click_t


def no_host(vis, at):
    blob = "\n".join(vis) + "\n" + "\n".join(str(e.value) for e in at.get("code")) if True else ""
    assert HOST not in blob and "https://" not in blob and "http://" not in blob


@pytest.mark.parametrize("delay", [30, 75, 120])
def test_ready_at(env, delay):
    at, net, t = scenario(env, f"ready{delay}", ready_delay=delay)
    vis = visible(at)
    ok = True  # shipped 160 s window reaches 30/75/120 s
    posts = [p for p in net.posts if p[1] == "ask"]
    assert at.text_input[0].value == Q                                   # (a) preserved
    assert env.labels == [CFG.starting_message.format(max_total_s=CFG.max_total_s)]  # (b) starting message
    no_host(vis, at)                                                     # (d)
    assert posts or not ok
    if ok and posts:  # woke in time: exactly one /ask, and only after readiness
        assert len(posts) == 1 and posts[0][0] >= delay                  # (c)
        assert CFG.ready_message in vis
    else:
        assert not posts                                                  # (c) none while not ready
        assert CFG.failed_message in vis
        net.ready_at = -1                                                 # now ready; user clicks Ask again
        env.labels.clear()
        ask(at)
        assert len([p for p in net.posts if p[1] == "ask"]) == 1          # (c) exactly one after ready
        assert at.text_input[0].value == Q


def test_never_ready(env):
    at, net, t = scenario(env, "never", ready_delay=None)
    vis = visible(at)
    assert not net.posts and at.text_input[0].value == Q
    assert CFG.failed_message in vis
    assert env.labels == [CFG.starting_message.format(max_total_s=CFG.max_total_s)]
    assert env.clk.t - t <= CFG.max_total_s + (len(CFG.backoff_s) + 1) * CFG.health_timeout_s   # bounded
    no_host(vis, at)
    net.ready_at = -1; ask(at)
    assert len(net.posts) == 1


def test_unreachable_url(env):
    at, net, t = scenario(env, "unreach", ready_delay=None, unready="refuse")
    vis = visible(at)
    assert not net.posts and at.text_input[0].value == Q
    assert CFG.failed_message in vis
    no_host(vis, at)


def test_cold_load_never_ready(env):
    """Page opened while API is down (no click): page-load check wakes/fails; nothing sent."""
    net = Net(env.clk, ready_at=float("inf")); install(env, net)
    at = AppTest.from_file(APP, default_timeout=60).run()
    vis = visible(at)
    assert not at.exception and not net.posts
    no_host(vis, at)


def test_mutation_probe_message(env):
    """Mutation check: the failed-message assertion must be able to fail (expects a wrong text -> must not be visible)."""
    at, net, t = scenario(env, "mut", ready_delay=None)
    assert "THIS TEXT IS NOT SHOWN" not in visible(at)
    assert "~3 minutes" in CFG.starting_message
    assert CFG.failed_message in visible(at)
