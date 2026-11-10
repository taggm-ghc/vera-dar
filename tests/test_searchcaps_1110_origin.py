"""SEARCHCAPS-1110 (2): origin=seed is set only by an authenticated POST /ask; seed jobs run at low priority under the
seed origin (OD-1110-SEARCHCAPS). Offline: stubbed handler, fake jobs; no DB, network or env files."""
import inspect

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import main
from tests import acq_support as A
from vera import auto_acquire, public_mode as pm, request_origin

KEY = "k" * 20
Q = "What is photosynthesis?"


@pytest.fixture(autouse=True)
def api_env(monkeypatch):
    monkeypatch.setenv("VERA_API_KEY", KEY)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    pm.reset_limiter(pm.CapLimiter(pm.Caps(req_per_min_per_client=1000, req_per_day_per_client=1000,
                                           req_per_day_global=1000)))
    auto_acquire.reset_for_tests()
    yield
    pm.reset_limiter(None)
    auto_acquire.reset_for_tests()


def _capture(monkeypatch):
    seen = []

    def impl(req, request, bg):
        seen.append(request_origin.current())
        raise HTTPException(status_code=503, detail="stub")
    monkeypatch.setattr(main, "_ask_impl", impl)
    return seen


def test_authenticated_ask_accepts_seed_and_scopes_it(monkeypatch):
    seen = _capture(monkeypatch)
    c = TestClient(main.app, raise_server_exceptions=False)
    r = c.post("/ask", json={"question": Q, "origin": "seed"}, headers={"X-API-Key": KEY})
    assert r.status_code == 503 and seen == ["seed"]
    c.post("/ask", json={"question": Q}, headers={"X-API-Key": KEY})
    assert seen == ["seed", "visitor"]
    assert request_origin.current() == "visitor"  # nothing leaks out of the request


def test_seed_without_key_is_refused_before_the_handler(monkeypatch):
    seen = _capture(monkeypatch)
    c = TestClient(main.app, raise_server_exceptions=False)
    assert c.post("/ask", json={"question": Q, "origin": "seed"}).status_code == 401
    assert c.post("/ask", json={"question": Q, "origin": "seed"}, headers={"X-API-Key": "x" * 20}).status_code == 401
    assert seen == []


def test_unknown_origin_is_a_422(monkeypatch):
    seen = _capture(monkeypatch)
    c = TestClient(main.app, raise_server_exceptions=False)
    assert c.post("/ask", json={"question": Q, "origin": "owner"}, headers={"X-API-Key": KEY}).status_code == 422
    assert seen == []


def test_stream_route_refuses_seed(monkeypatch):
    seen = _capture(monkeypatch)
    monkeypatch.setattr(main, "_require_stream", lambda: None)
    c = TestClient(main.app, raise_server_exceptions=False)
    r = c.post("/ask/stream", json={"question": Q, "origin": "seed"}, headers={"X-API-Key": KEY})
    assert r.status_code == 422 and seen == []


def test_public_ui_never_sends_origin():
    for sc in (True, False):
        assert "origin" not in pm._ask_body(Q, "normal", sc)
    assert "origin" not in inspect.signature(pm.post_ask).parameters
    assert "origin" not in inspect.signature(pm.post_ask_tracked).parameters


def test_seed_job_carries_origin_and_runs_after_visitor_jobs(monkeypatch):
    c = A.cfg(queue_max=5, jobs_per_hour=10)
    clk = lambda: 5.0  # noqa: E731
    with request_origin.scope("seed"):
        assert auto_acquire.enqueue("seed q", "decline", cfg=c, env=A.ENV_ON, clock=clk,
                                    start_worker=False) == "ACQ-QUEUED"
    assert auto_acquire.enqueue("visitor q", "decline", cfg=c, env=A.ENV_ON, clock=clk,
                                start_worker=False) == "ACQ-QUEUED"
    ran = []
    monkeypatch.setattr(auto_acquire, "run_job",
                        lambda job, cfg, deps, clock=None, env=None: ran.append((job.question, job.origin,
                                                                                 request_origin.current())) or {})
    auto_acquire.drain_for_tests(c, deps=None)
    assert ran == [("visitor q", "visitor", "visitor"), ("seed q", "seed", "seed")]


def test_seed_job_never_takes_the_last_hourly_slots():
    c = A.cfg(queue_max=5, jobs_per_hour=3)
    clk = lambda: 5.0  # noqa: E731
    assert auto_acquire.enqueue("v1", "decline", cfg=c, env=A.ENV_ON, clock=clk, start_worker=False) == "ACQ-QUEUED"
    assert auto_acquire.enqueue("v2", "decline", cfg=c, env=A.ENV_ON, clock=clk, start_worker=False) == "ACQ-QUEUED"
    with request_origin.scope("seed"):
        assert auto_acquire.enqueue("s1", "decline", cfg=c, env=A.ENV_ON, clock=clk,
                                    start_worker=False) == "R-ACQ-RATE"
    assert auto_acquire.enqueue("v3", "decline", cfg=c, env=A.ENV_ON, clock=clk, start_worker=False) == "ACQ-QUEUED"


def test_continue_jobs_inherit_seed_origin():
    c = A.cfg(queue_max=5, jobs_per_hour=10)
    with request_origin.scope("seed"):
        out = auto_acquire.enqueue_candidates("seed q", [{"doi": "10.9999/x.1"}], cfg=c, env=A.ENV_ON,
                                              clock=lambda: 5.0, start_worker=False)
    assert out["status"] == "ACQ-QUEUED"
    assert [j.origin for j in list(auto_acquire._S.queue.queue)] == ["seed"]
