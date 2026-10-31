"""STREAM-1031 T6/T7/T10e/T11/TB7: POST /ask/stream wire format, order, caps, errors. Offline."""
import json
import threading
import time

import pytest

import main
from vera import progress
from vera import public_mode as pm
from tests.test_stream_route_support import API, CFG, KEY, client, lines, post, stub_impl, stream_env  # noqa: F401


def _minimal():
    return {"answer": "A", "model": "m", "tokens_used": 0, "cost_usd": 0.0, "result_kind": "answer"}


def test_default_config_stream_enabled_true_polling_false():
    c = progress.load_config(progress.CONFIG_PATH)
    assert c.stream.enabled is True and c.enabled is False
    assert (c.stream.heartbeat_s, c.stream.client_no_data_s, c.stream.connect_timeout_s, c.stream.orphan_s) == (10, 60, 10, 120)
    raw = json.loads(progress.CONFIG_PATH.read_text())
    for k in ("enabled", "heartbeat_s", "client_no_data_s", "connect_timeout_s", "orphan_s"):
        assert k in raw["stream"]["_calibration"]
    assert progress.STREAM_CONTRACT_VERSION == "1.3" and progress.CONTRACT_VERSION == "1"


def test_t6_validate_stream_rejects_bad_values_verbosely():
    base = json.loads(progress.CONFIG_PATH.read_text())
    for patch, needle in [({"enabled": "yes"}, "stream.enabled"), ({"heartbeat_s": True}, "stream.heartbeat_s"),
                          ({"orphan_s": "x"}, "stream.orphan_s"), ({"client_no_data_s": 5}, "client_no_data_s must be >"),
                          ({"bogus": 1}, "unknown key")]:
        raw = {**base, "stream": {**base["stream"], **patch}}
        with pytest.raises(ValueError, match=needle):
            progress.validate_config(raw)
    with pytest.raises(ValueError, match="stream must be a JSON object"):
        progress.validate_config({**base, "stream": 3})
    assert progress.validate_config({k: v for k, v in base.items() if k != "stream"}).stream == progress.DEFAULT_STREAM


def test_t6_acquire_states_and_count_keys():
    ok = progress.VALID[progress.Stage.ACQUIRE]
    assert progress.State.LIMITED in ok and progress.State.BUSY in ok
    assert {"shortlisted", "held", "step", "ceiling", "why"} <= progress.COUNT_KEYS
    assert progress.State.LIMITED not in progress.VALID[progress.Stage.SEARCH]


def test_stream_disabled_is_404_and_kill_switch(monkeypatch):
    progress.configure(progress.Config(**{**CFG.__dict__, "stream": progress.StreamCfg(**{**CFG.stream.__dict__, "enabled": False})}))
    stub_impl(monkeypatch, lambda *a: pytest.fail("must not run"))
    assert post(client()).status_code == 404
    progress.configure(CFG)
    monkeypatch.setenv(progress.KILL_SWITCH_ENV, "1")
    assert post(client()).status_code == 404 and not progress.stream_enabled()


def test_t7_order_headers_single_result(monkeypatch):
    def impl(req, request, bg):
        progress.emit(progress.Stage.SEARCH, progress.State.START, step=1)
        progress.emit(progress.Stage.ACQUIRE, progress.State.LIMITED, stored=1, target=6, why=1)
        bg.add_task(lambda: ran.append(1))
        return main.AskResponse(**_minimal())
    ran = []
    stub_impl(monkeypatch, impl)
    r = post(client())
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    assert r.headers["x-accel-buffering"] == "no" and r.headers["cache-control"] == "no-cache"
    ls = lines(r)
    assert all(x["v"] == "1.3" for x in ls)
    ev = [x for x in ls if x["type"] == "event"]
    assert [(e["stage"], e["state"]) for e in ev] == [("received", "done"), ("search", "start"), ("acquire", "limited"), ("done", "answered")]
    assert ev[2]["counts"] == {"stored": 1, "target": 6, "why": 1}
    assert [x["seq"] for x in ls if "seq" in x] == list(range(1, len(ls)))
    assert ls[-1]["type"] == "result" and ls[-1]["status"] == 200 and sum(x["type"] == "result" for x in ls) == 1
    assert ran == [1]  # after-response background task ran


def test_t7_heartbeat_when_silent(monkeypatch):
    def impl(req, request, bg):
        time.sleep(0.4)
        return main.AskResponse(**_minimal())
    stub_impl(monkeypatch, impl)
    ls = lines(post(client()))
    hb = [x for x in ls if x["type"] == "hb"]
    assert len(hb) >= 2 and all(set(h) == {"v", "type", "seq", "elapsed_s"} for h in hb)
    assert ls[-1]["type"] == "result"


def test_t7_401_and_429_before_streaming(monkeypatch, stream_env):
    stub_impl(monkeypatch, lambda *a: pytest.fail("must not run"))
    r = post(client(), headers={"X-API-Key": "wrong" * 5})
    assert r.status_code == 401 and not r.headers["content-type"].startswith("application/x-ndjson")
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    pm.reset_limiter(pm.CapLimiter(pm.Caps(req_per_min_per_client=0, req_per_day_per_client=0, req_per_day_global=0)))
    r = post(client())
    assert r.status_code == 429 and not r.headers["content-type"].startswith("application/x-ndjson")


def test_t10e_cap_slot_counted_once_per_post(monkeypatch):
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    lim = pm.CapLimiter(pm.Caps(req_per_min_per_client=1000, req_per_day_per_client=1000, req_per_day_global=1000))
    pm.reset_limiter(lim)
    stub_impl(monkeypatch, lambda req, request, bg: main.AskResponse(**_minimal()))
    c = client()
    for n in (1, 2):
        assert post(c).status_code == 200
        assert lim._global_req == n and sum(lim._day_req.values()) == n


def test_tb7_new_limiter_resets_counters():
    caps = pm.Caps(req_per_min_per_client=1000, req_per_day_per_client=1000, req_per_day_global=1000)
    a = pm.CapLimiter(caps)
    assert a.allow("c") and a._global_req == 1
    b = pm.CapLimiter(caps)  # process restart: in-memory counters start over (accepted divergence)
    assert b._global_req == 0 and b._day_req == {} and b._hour == {}


@pytest.mark.parametrize("exc,status,kind", [
    (RuntimeError("SECRET-TEXT-1 sk-abc"), 500, "error"),
    (main.HTTPException(status_code=502, detail="SECRET-TEXT-2"), 502, "provider_error"),
    (main.HTTPException(status_code=503, detail="SECRET-TEXT-3"), 503, "unavailable"),
    (main.HTTPException(status_code=429, detail="SECRET-TEXT-4"), 429, "busy"),
])
def test_t11_errors_closed_kind_no_exception_text(monkeypatch, exc, status, kind):
    def impl(*a):
        raise exc
    stub_impl(monkeypatch, impl)
    r = post(client())
    ls = lines(r)
    assert ls[-1] == {"v": "1.3", "type": "result", "status": status, "kind": kind}
    assert "SECRET-TEXT" not in r.text and "sk-abc" not in r.text
    assert (ls[-2]["stage"], ls[-2]["state"]) == ("done", "error")


def test_providers_exhausted_maps_to_503_unavailable(monkeypatch):
    import sys
    import types
    mod = types.ModuleType("vera.provider_exhaustion")

    class ProvidersExhausted(RuntimeError):
        pass
    mod.ProvidersExhausted = ProvidersExhausted
    monkeypatch.setitem(sys.modules, "vera.provider_exhaustion", mod)

    def impl(*a):
        raise ProvidersExhausted("SECRET-TEXT-5")
    stub_impl(monkeypatch, impl)
    r = post(client())
    assert lines(r)[-1]["status"] == 503 and lines(r)[-1]["kind"] == "unavailable" and "SECRET" not in r.text


def test_stream_cancelled_is_silent_stop(monkeypatch):
    from vera import stream_ctx

    def impl(*a):
        raise stream_ctx.StreamCancelled()
    stub_impl(monkeypatch, impl)
    ls = lines(post(client()))
    assert [x["type"] for x in ls] == ["event"] and ls[0]["stage"] == "received"
