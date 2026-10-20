"""Health v3: public /health is a coarse enum only; /health/dar (full dict) is behind the API key; startup never blocks.
No DB, no network: readiness is monkeypatched."""
import re
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

KEY = "k" * 24
FULL = {"dar": "degraded", "routing": "on", "migrations": {"006": True, "007": False}, "stored_corpus": "ready",
        "corpus": {"documents": 3, "chunks": 40, "domains": 2}, "fallback": True, "reason_code": "migration_007_missing"}
_LEAK = re.compile(r"(006|007|migration|reason|corpus|chunks|documents|domains|routing|runtime_|fallback|\d)", re.I)


@pytest.fixture()
def client(monkeypatch):
    for k in ("VERA_DB_URL_RO", "DATABASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VERA_API_KEY", KEY)
    import main
    return TestClient(main.app), main


def _set(monkeypatch, main, value):
    def f(*a, **k):
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(main.dar_health, "cached_readiness", f)


@pytest.mark.parametrize("state", ["ready", "degraded", "unavailable", "disabled", "unknown"])
def test_enum_mapping_each_state(client, monkeypatch, state):
    c, main = client
    _set(monkeypatch, main, dict(FULL, dar=state))
    r = c.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "dar": state}


@pytest.mark.parametrize("value", [None, {}, {"dar": None}, {"dar": "weird"}, {"dar": 5}, {"dar": {}}, {"dar": []}, {"dar": ["ready"]}, "x", RuntimeError("boom")])
def test_null_uncached_garbage_maps_to_unknown(client, monkeypatch, value):
    c, main = client
    _set(monkeypatch, main, value)
    r = c.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "dar": "unknown"}


def test_public_body_has_no_ids_codes_or_counts(client, monkeypatch):
    c, main = client
    _set(monkeypatch, main, FULL)
    r = c.get("/health")
    assert set(r.json()) == {"status", "dar"}
    assert not _LEAK.search(r.text), r.text


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong-key-wrong-key-wrong"}])
def test_health_dar_requires_key(client, monkeypatch, headers):
    c, main = client
    _set(monkeypatch, main, FULL)
    r = c.get("/health/dar", headers=headers)
    assert r.status_code in (401, 403)
    assert "migration" not in r.text and "007" not in r.text


def test_health_dar_full_dict_with_key(client, monkeypatch):
    c, main = client
    _set(monkeypatch, main, FULL)
    r = c.get("/health/dar", headers={"X-API-Key": KEY})
    body = r.json()
    assert r.status_code == 200 and body.pop("fulldoc") is not None and body == FULL  # fulldoc: /health/dar only (AE-1020 WP-H)


def test_health_dar_failure_does_not_raise(client, monkeypatch):
    c, main = client
    _set(monkeypatch, main, RuntimeError("boom"))
    r = c.get("/health/dar", headers={"X-API-Key": KEY})
    assert r.status_code == 200 and "boom" not in r.text


def test_startup_uses_async_path_and_does_not_block(monkeypatch):
    from vera import dar_health
    src = (Path(__file__).resolve().parent.parent / "main.py").read_text()
    assert "dar_health.startup_log_async()" in src and "dar_health.startup_line()" not in src
    gate = threading.Event()
    monkeypatch.setattr(dar_health, "startup_line", lambda *a, **k: gate.wait(10))
    t0 = time.monotonic()
    dar_health.startup_log_async()
    elapsed = time.monotonic() - t0
    gate.set()
    assert elapsed < 1.0, elapsed


def test_uncached_real_cached_readiness_is_unknown_publicly_and_null_in_full(client, monkeypatch):
    """No monkeypatched readiness: the real not_probed placeholder (no refresh thread started) maps to "unknown"."""
    c, main = client
    from vera import dar_health
    dar_health.reset_for_tests()
    monkeypatch.setattr(dar_health.threading, "Thread", lambda *a, **k: type("T", (), {"start": lambda self: None})())
    assert c.get("/health").json() == {"status": "ok", "dar": "unknown"}
    full = c.get("/health/dar", headers={"X-API-Key": KEY}).json()
    assert full["dar"] is None and full["reason_code"] == "not_probed"
    dar_health.reset_for_tests()
