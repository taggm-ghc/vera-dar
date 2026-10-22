"""DIAG-1021: a hung DB connect or full-document status computation never blocks a request thread past its bound.
Offline: stubbed hanging connections and computations; no network, no DB."""
import threading
import time

import pytest

from vera import dar_health, fulldoc_db

CFG = {"db_bounds": {"db_load_deadline_s": 0.3, "single_flight_wait_s": 0.4, "health_wait_s": 0.3}}


@pytest.fixture
def gate():
    ev = threading.Event()
    yield ev
    ev.set()  # release every hung stub so no daemon thread outlives the test longer than needed


def test_deadline_s_reads_config_and_falls_back():
    assert fulldoc_db.deadline_s(CFG, "db_load_deadline_s", 30) == 0.3
    for bad in ({}, {"db_bounds": {"db_load_deadline_s": 0}}, {"db_bounds": {"db_load_deadline_s": True}},
                {"db_bounds": {"db_load_deadline_s": "5"}}, None):
        assert fulldoc_db.deadline_s(bad, "db_load_deadline_s", 30) == 30.0


def test_hung_connect_returns_unusable_within_deadline(gate, monkeypatch):
    monkeypatch.setattr(fulldoc_db, "load_from_rows", lambda *a, **k: (_ for _ in ()).throw(AssertionError("built")))
    t0 = time.monotonic()
    c = fulldoc_db.load_db(lambda: gate.wait(30), 1, CFG)
    assert time.monotonic() - t0 < 2.0
    assert c.status != fulldoc_db.STATUS_OK and not c.acquired
    assert any("LoadTimeout" in str(u) for u in c.unusable)


def test_cached_load_followers_return_within_bound_and_are_not_cached(gate):
    fulldoc_db.reset_for_tests()
    cache: dict = {}
    results, times = [], []

    def call():
        t0 = time.monotonic()
        results.append(fulldoc_db.cached_load(lambda: gate.wait(30), 7, CFG, cache))
        times.append(time.monotonic() - t0)
    ts = [threading.Thread(target=call) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    assert len(results) == 4 and max(times) < 2.0
    assert all(r.status != fulldoc_db.STATUS_OK for r in results) and cache == {}
    fulldoc_db.reset_for_tests()


def test_health_fulldoc_status_bounded_when_leader_never_returns(gate, monkeypatch):
    monkeypatch.setattr(dar_health, "_fulldoc_compute", lambda: (gate.wait(30), {"status": "ok"})[1])
    monkeypatch.setattr(dar_health, "fulldoc_wait_s", lambda: 0.3)
    with dar_health._FD_LOCK:
        dar_health._FD_INFLIGHT.clear()
    out, times = [], []

    def call():
        t0 = time.monotonic()
        out.append(dar_health._fulldoc_status())
        times.append(time.monotonic() - t0)
    ts = [threading.Thread(target=call) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    assert len(out) == 3 and max(times) < 2.0 and all(o["status"] == "timeout" for o in out)
    assert set(out[0]) == set(dar_health._fulldoc_error())  # codes only, same keys
    gate.set()
    for _ in range(50):  # the background computation finishes and clears the slot
        with dar_health._FD_LOCK:
            if not dar_health._FD_INFLIGHT:
                break
        time.sleep(0.02)
    assert dar_health._fulldoc_status()["status"] == "ok"


def test_health_wait_default_and_shipped_bounds():
    import json
    from pathlib import Path
    cfg = json.loads((Path(__file__).resolve().parent.parent / "config" / "fulldoc_answer.json").read_text())
    b = cfg["db_bounds"]
    assert 0 < b["health_wait_s"] < 20 and b["db_load_deadline_s"] > 0 and b["single_flight_wait_s"] > 0
    assert dar_health.fulldoc_wait_s() == float(b["health_wait_s"])


def test_keepalives_configured():
    import inspect
    from vera import auto_acquire
    for src in (inspect.getsource(fulldoc_db.default_connect), inspect.getsource(auto_acquire.default_deps)):
        assert '"keepalives": 1' in src and '"keepalives_idle"' in src
