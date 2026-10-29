"""PROG-1027 W4a: public_mode.post_ask_tracked / get_progress (offline, monkeypatched requests)."""
import logging

import pytest
import requests

from vera import progress
from vera import public_mode as pm

HOST = "https://sentinel-host.invalid"
TOKEN = "SENTINELTOKEN" + "x" * 30
RID = "a" * 32


def ev(seq=1, stage="received", state="done", counts=None, **over):
    e = {"v": "1", "seq": seq, "ts": "2026-10-28T00:00:00+00:00", "stage": stage, "state": state, "counts": {} if counts is None else counts}
    e.update(over)
    return e


class Resp:
    def __init__(self, status=200, body=None, raises=False):
        self.status_code, self._b, self._r = status, body, raises

    def json(self):
        if self._r:
            raise ValueError("bad json " + HOST)
        return self._b


def run(monkeypatch, resp=None, exc=None):
    seen = {}

    def fake_get(url, **kw):
        seen.update(url=url, **kw)
        if exc:
            raise exc
        return resp

    monkeypatch.setattr(pm.requests, "get", fake_get)
    out = pm.get_progress(HOST, "KEY", RID, TOKEN, 3)
    return out, seen


def test_ok_and_request_shape(monkeypatch):
    out, seen = run(monkeypatch, Resp(200, {"events": [ev(), ev(2, "scope", "start"), ev(3, "acquire", "update", {"checked": 1, "usable": 0})]}))
    assert [e["seq"] for e in out] == [1, 2, 3]
    assert seen["params"] == {"after": 3}
    assert seen["headers"]["X-VERA-Session"] == TOKEN and seen["headers"]["X-API-Key"] == "KEY"
    assert seen["url"].endswith(f"/ask/progress/{RID}")


def test_empty_events_ok(monkeypatch):
    assert run(monkeypatch, Resp(200, {"events": []}))[0] == []


@pytest.mark.parametrize("bad", [
    ev(v="2"), ev(v=1), ev(seq=0), ev(seq=True), ev(seq="1"), ev(stage="nope"), ev(state="nope"),
    ev(stage="scope", state="answered"), ev(stage="done", state="start"), ev(counts={"round": 1}),
    ev(stage="acquire", state="update", counts={"checked": True}), ev(stage="acquire", state="update", counts={"checked": -1}),
    ev(stage="acquire", state="update", counts={"checked": 10**9}), ev(stage="acquire", state="update", counts={"checked": "1"}),
    ev(counts=[]), ev(ts=5), {**ev(), "extra": 1}, {k: v for k, v in ev().items() if k != "ts"}, "str", None,
])
def test_bad_event_rejects_whole_body(monkeypatch, bad):
    assert run(monkeypatch, Resp(200, {"events": [ev(), bad]}))[0] is None


@pytest.mark.parametrize("body", [{}, {"events": "x"}, [], None, {"events": None}])
def test_bad_body(monkeypatch, body):
    assert run(monkeypatch, Resp(200, body))[0] is None


def test_404_gone(monkeypatch):
    out, _ = run(monkeypatch, Resp(404, {}))
    assert out is pm.GONE and not out


@pytest.mark.parametrize("code", [401, 422, 429, 500, 502, 503])
def test_other_status_none(monkeypatch, code):
    assert run(monkeypatch, Resp(code, {"events": []}))[0] is None


@pytest.mark.parametrize("exc", [requests.exceptions.ReadTimeout(HOST), requests.exceptions.ConnectTimeout(HOST),
                                 requests.exceptions.ConnectionError(HOST + TOKEN), requests.RequestException(HOST),
                                 RuntimeError(HOST)])
def test_exceptions_none_no_leak(monkeypatch, caplog, exc):
    with caplog.at_level(logging.DEBUG):
        out, _ = run(monkeypatch, exc=exc)
    assert out is None
    assert HOST not in caplog.text and TOKEN not in caplog.text


def test_bad_json_none_no_leak(monkeypatch, caplog):
    with caplog.at_level(logging.DEBUG):
        out, _ = run(monkeypatch, Resp(200, raises=True))
    assert out is None and HOST not in caplog.text and TOKEN not in repr(out)


def test_return_values_hold_no_host_or_token(monkeypatch):
    for r in (run(monkeypatch, Resp(200, {"events": [ev()]}))[0], run(monkeypatch, Resp(404))[0], run(monkeypatch, exc=OSError(HOST))[0]):
        assert HOST not in repr(r) and TOKEN not in repr(r)


def test_post_ask_tracked_headers(monkeypatch):
    seen = {}
    monkeypatch.setattr(pm.requests, "post", lambda url, **kw: seen.update(url=url, **kw) or "R")
    assert pm.post_ask_tracked(HOST + "/", "q", "m", "KEY", RID, TOKEN) == "R"
    assert seen["url"] == HOST + "/ask" and seen["json"] == {"question": "q", "mode": "m"}
    h = seen["headers"]
    assert h["X-API-Key"] == "KEY" and h["X-VERA-Request-Id"] == RID and h["X-VERA-Session"] == TOKEN
    assert seen["timeout"] == pm.ASK_TIMEOUT_S


def test_uses_cfg_timeout(monkeypatch):
    _, seen = run(monkeypatch, Resp(200, {"events": []}))
    assert seen["timeout"] == progress.CFG.get_timeout_s
