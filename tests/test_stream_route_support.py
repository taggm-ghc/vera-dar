"""Shared offline fixtures for POST /ask/stream tests (STREAM-1031). Stubs only; no DB, network or env files."""
import json

import pytest
from fastapi.testclient import TestClient

import main
from vera import ask_service, progress
from vera import public_mode as pm

KEY = "k" * 20
API = {"X-API-Key": KEY}
Q = "SENTINEL-QUERY-ZX81 how do caches reduce latency?"
STREAM = progress.StreamCfg(enabled=True, heartbeat_s=0.05, client_no_data_s=60, connect_timeout_s=10, orphan_s=120)
CFG = progress.Config(enabled=False, max_records=8, max_events_per_record=64, max_events_per_poll=64, ttl_s=600,
                      poll_interval_s=1.0, poll_margin_s=5, first_event_grace_s=10, get_timeout_s=2, max_count=10000,
                      stream=STREAM)


@pytest.fixture(autouse=True)
def stream_env(monkeypatch):
    original = progress.CFG
    progress.configure(CFG)
    monkeypatch.setenv("VERA_API_KEY", KEY)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    monkeypatch.delenv(progress.KILL_SWITCH_ENV, raising=False)
    monkeypatch.setattr(main, "_stream_stall_s", lambda: 300.0)
    lim = pm.CapLimiter(pm.Caps(req_per_min_per_client=1000, req_per_day_per_client=1000, req_per_day_global=1000))
    pm.reset_limiter(lim)
    yield lim
    progress.configure(original)
    pm.reset_limiter(None)


def client():
    return TestClient(main.app, raise_server_exceptions=False)


def stub_impl(monkeypatch, fn):
    monkeypatch.setattr(main, "_ask_impl", fn)


def ok_response():
    return ask_service.AskResult(answer="A", tokens_used=0, cost_usd=0.0, result_kind="answer", claim_check=None)


def post(c, headers=API):
    return c.post("/ask/stream", json={"question": Q}, headers=headers)


def lines(r):
    return [json.loads(x) for x in r.text.splitlines() if x.strip()]
