"""DEFERRED-DB-1104: digest persistence ON (stored digest shown at once, background refresh, INSERT via the _wo URL only)
and the owner trace route's flag-gated DB read with window fallback. Offline: fake engines, no DB or network."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

import main
from vera import corpus_description as cd
from vera import digest_store, request_trace as rt
from tests.test_integrate_1104_db_flags import Eng, OV, CFG, Row, RID

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = json.loads((ROOT / "config" / "request_trace.json").read_text())


def _wait(pred, n=200):
    for _ in range(n):
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_shipped_flags():
    assert SHIPPED["digest_store"]["enabled"] is True
    assert SHIPPED["db_sink"]["owner_route_db_view"] is False


def test_write_goes_through_wo_url_only_as_insert():
    digest_store.reset_for_tests()
    urls, eng = [], Eng()

    def factory(url, **kw):
        urls.append(url)
        return eng
    env = {"VERA_DB_URL_WO": "postgresql://wo", "VERA_DB_URL_RO": "postgresql://ro"}
    assert digest_store.save({"summary": "S"}, "7|t|pv", cfg=SHIPPED["digest_store"], env=env, engine_factory=factory)
    assert urls == ["postgresql://wo"] and len(eng.inserts) == 1
    stmts = [q for q, _ in eng.sql]
    assert all(q.startswith(("INSERT", "SET LOCAL")) for q in stmts) and any(q.startswith("INSERT INTO vera_vjay.corpus_digest_state") for q in stmts)
    assert not any(w in q.upper() for q in stmts for w in ("UPDATE ", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"))
    # read goes through the ro URL
    urls.clear()
    digest_store.reset_for_tests()
    digest_store.load_latest(cfg=SHIPPED["digest_store"], env=env, engine_factory=factory)
    assert urls == ["postgresql://ro"]


def test_no_wo_url_means_no_write():
    digest_store.reset_for_tests()
    eng = Eng()
    assert digest_store.save({"summary": "S"}, "k", cfg=SHIPPED["digest_store"], env={"VERA_DB_URL_RO": "r"},
                             engine_factory=lambda *a, **k: eng) is False and eng.sql == []


@pytest.fixture
def desc(monkeypatch):
    digest_store.reset_for_tests()
    monkeypatch.setattr(cd, "stamp_and_profile", lambda *a, **k: (dict(OV), {}))
    monkeypatch.setattr(cd, "agentic_cfg", lambda cfg: {"sample_excerpt_chars": 0})
    calls, gate = [], threading.Event()

    def pipeline(overview, profile, llm, cfg, now=None):
        gate.wait(5)
        calls.append(1)
        return {"summary": "NEW", "topics": [], "coverage": None, "topic_shares": None, "model": "m", "trace": []}
    monkeypatch.setattr(cd, "run_digest_pipeline", pipeline)
    d = cd.CorpusDescriber(CFG, engine_factory=lambda *a, **k: object(), env={"VERA_DB_URL_RO": "ro", "VERA_DB_URL_WO": "wo"})
    eng = Eng()
    d.store_cfg = {"enabled": True, "table": "vera_vjay.corpus_digest_state"}
    d.store_engine_factory = lambda *a, **k: eng
    eng.digest_rows = [Row({"cache_key": "6|old|pv1", "digest": json.dumps({"summary": "OLD", "topics": []}), "created_at": "x"})]
    return d, eng, calls, gate


def test_stale_stored_shown_immediately_while_regeneration_is_blocked(desc):
    d, eng, calls, gate = desc
    t0 = time.monotonic()
    assert d.current(llm=None)["summary"] == "OLD"          # pipeline is blocked on the gate: page did not wait
    assert time.monotonic() - t0 < 2 and calls == []
    assert d.current(llm=None)["summary"] == "OLD" and d._refreshing   # second view: no second thread, no DB read
    n_sql = len(eng.sql)
    d.current(llm=None)
    assert len(eng.sql) == n_sql
    gate.set()
    assert _wait(lambda: len(eng.inserts) == 1) and calls == [1]
    assert _wait(lambda: not d._refreshing)
    assert d.current(llm=None)["summary"] == "NEW"


def test_background_failure_keeps_stored_and_sets_cooldown(desc, monkeypatch):
    d, eng, calls, gate = desc
    monkeypatch.setattr(cd, "run_digest_pipeline", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    assert d.current(llm=None)["summary"] == "OLD"
    assert _wait(lambda: not d._refreshing and d._retry_at > 0)
    assert d.current(llm=None)["summary"] == "OLD" and eng.inserts == []


# ------------------------------------------------------------------ owner route
def test_route_db_first_when_flag_on(monkeypatch):
    monkeypatch.setattr(rt, "owner_db_view_enabled", lambda: True)
    monkeypatch.setattr(rt, "db_view", lambda tid: {"source": "db"})
    monkeypatch.setattr(rt, "recent", lambda tid: pytest.fail("window used"))
    assert main.trace_request(RID) == {"source": "db"}


def test_route_falls_back_to_window_when_db_none(monkeypatch):
    monkeypatch.setattr(rt, "owner_db_view_enabled", lambda: True)
    monkeypatch.setattr(rt, "db_view", lambda tid: None)
    monkeypatch.setattr(rt, "recent", lambda tid: {"source": "window"})
    assert main.trace_request(RID) == {"source": "window"}
    monkeypatch.setattr(rt, "recent", lambda tid: None)
    with pytest.raises(main.HTTPException):
        main.trace_request(RID)


def test_flag_helper_requires_both_flags():
    old = rt.CFG
    try:
        rt.configure({"enabled": True, "db_sink": {"enabled": True, "owner_route_db_view": True}})
        assert rt.owner_db_view_enabled() is True
        rt.configure({"enabled": True, "db_sink": {"enabled": True}})
        assert rt.owner_db_view_enabled() is False
        rt.configure({"enabled": True, "db_sink": {"enabled": False, "owner_route_db_view": True}})
        assert rt.owner_db_view_enabled() is False
    finally:
        rt.configure(old)


def test_db_view_prefers_owner_read_env():
    old = rt.CFG
    try:
        rt.configure({"enabled": True, "db_sink": {"enabled": True, "db_url_env_ro": "VERA_DB_URL_RO",
                                                  "db_url_env_owner_read": "VERA_DB_URL_OWNER_RO"}},
                     env={"VERA_DB_URL_RO": "postgresql://ro", "VERA_DB_URL_OWNER_RO": "postgresql://own"})
        urls = []

        def f(url, **kw):
            urls.append(url)
            return Eng()
        assert rt.db_view(RID, engine_factory=f)["source"] == "db" and urls == ["postgresql://own"]
    finally:
        rt.configure(old)
