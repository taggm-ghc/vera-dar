"""WED-P161 (P1.6.1): startup warm-up of the DB-backed full-document corpus and its evidence index.
SYNTHETIC bundle -> row plan -> in-memory fake DB (gated so a load can be held "slow"). No DB, no network, no env files."""
import copy
import json
import logging
import threading
import time
from pathlib import Path

import pytest

from tests import test_fulldoc_db as tdb
from tests import fulldoc_bundle_synth as synth
from vera import dar_corpus, dar_health, fulldoc_corpus as fc, fulldoc_db, fulldoc_warm

RUN = tdb.RUN
ENV_DB = {"VERA_FULLDOC_DB_RUN_ID": RUN}
WAIT = 10.0


@pytest.fixture()
def world(tmp_path):
    fc.reset_for_tests()
    fulldoc_warm.reset_for_tests()
    root = synth.write_bundle(tmp_path / "b")
    cfg = synth.synth_cfg()
    rows, _ = tdb._script().build_plan(root, cfg, target=700)
    yield root, cfg, rows
    fc.reset_for_tests()
    fulldoc_warm.reset_for_tests()


class Gate:
    """A fake connect whose SOURCES select blocks until released: a slow loader, counted."""

    def __init__(self, rows):
        self.srcs = tdb._db(rows)
        self.entered, self.release, self.log = threading.Event(), threading.Event(), []

    def connect(self):
        gate = self

        class C(tdb.FakeConn):
            def fetchall(self):
                if self._sql == fulldoc_db.SQL_SOURCES:
                    gate.entered.set()
                    assert gate.release.wait(WAIT), "gate never released"
                return super().fetchall()
        return C(*self.srcs, self.log)

    def n_loads(self):
        return sum(1 for sql, _ in self.log if sql == fulldoc_db.SQL_SOURCES)


def _gate(monkeypatch, rows, released=False):
    g = Gate(rows)
    if released:
        g.release.set()
    monkeypatch.setattr(fulldoc_db, "default_connect", lambda cfg, env: g.connect)
    return g


def _thread_names():
    return [t.name for t in threading.enumerate() if t.name == "fulldoc-warm"]


# ------------------------------------------------------------------------------------------------ AC1 / AC2: when it starts
def test_wanted_matrix(world):
    root, cfg, _ = world
    assert fulldoc_warm.wanted(cfg, ENV_DB) is True
    assert fulldoc_warm.wanted(cfg, {}) is False                                   # no source
    assert fulldoc_warm.wanted(cfg, {cfg["bundle_dir_env"]: str(root)}) is False   # bundle source: not warmed
    assert fulldoc_warm.wanted(cfg, {**ENV_DB, cfg["bundle_dir_env"]: str(root)}) is False  # R-SOURCE conflict
    for en in (False, "false", None, "maybe"):
        off = copy.deepcopy(cfg)
        off["fulldoc"]["enabled"] = en
        assert fulldoc_warm.wanted(off, ENV_DB) is False, en
    absent = copy.deepcopy(cfg)
    absent.pop("fulldoc")
    assert fulldoc_warm.wanted(absent, ENV_DB) is False
    for flag in (False, "false", "0", "off"):
        noflag = copy.deepcopy(cfg)
        noflag["fulldoc"]["warm_at_startup"] = flag
        assert fulldoc_warm.wanted(noflag, ENV_DB) is False, flag
    on = copy.deepcopy(cfg)
    on["fulldoc"].pop("warm_at_startup", None)
    assert fulldoc_warm.wanted(on, ENV_DB) is True  # absent flag = on


def test_starts_once_only_when_enabled(world, monkeypatch):
    _, cfg, rows = world
    g = _gate(monkeypatch, rows)
    t = fulldoc_warm.start(cfg, ENV_DB)
    assert t is not None and t.daemon and t.name == "fulldoc-warm"
    assert g.entered.wait(WAIT) and fulldoc_warm.state() == fulldoc_warm.WARMING
    assert fulldoc_warm.start(cfg, ENV_DB) is None  # idempotent: one thread per process
    g.release.set()
    t.join(WAIT)
    assert not t.is_alive() and fulldoc_warm.state() == fulldoc_warm.DONE and g.n_loads() == 1


@pytest.mark.parametrize("how", ["disabled", "no_source", "bundle", "flag_off"])
def test_disabled_modes_start_no_thread_and_touch_nothing(world, monkeypatch, how):
    root, cfg, _ = world
    cfg, env = copy.deepcopy(cfg), dict(ENV_DB)
    if how == "disabled":
        cfg["fulldoc"]["enabled"] = False
    elif how == "no_source":
        env = {}
    elif how == "bundle":
        env = {cfg["bundle_dir_env"]: str(root)}
    else:
        cfg["fulldoc"]["warm_at_startup"] = False

    def boom(*a, **k):
        raise AssertionError("corpus touched")
    monkeypatch.setattr(fc, "get_corpus", boom)
    monkeypatch.setattr(fulldoc_db, "default_connect", boom)
    started = []
    real = threading.Thread
    monkeypatch.setattr(fulldoc_warm.threading, "Thread", lambda *a, **k: started.append(1) or real(*a, **k))
    before = len(_thread_names())
    assert fulldoc_warm.start(cfg, env) is None
    assert started == [] and len(_thread_names()) == before and fulldoc_warm.state() == fulldoc_warm.IDLE
    assert dar_health._fulldoc_compute() != fulldoc_warm.loading_status()


def test_shipped_config_flag_and_main_wiring():
    cfg = json.loads(Path(fc.CONFIG_PATH).read_text(encoding="utf-8"))
    assert cfg["fulldoc"]["warm_at_startup"] is True
    src = (Path(__file__).resolve().parent.parent / "main.py").read_text()
    assert src.count("fulldoc_warm.start()") == 1


def test_shipped_env_has_no_db_source_so_import_starts_no_thread():
    """The test environment (and any deploy without the run-id env var) never warms: same behaviour as P1.6."""
    import os
    assert fulldoc_warm.wanted(fc.load_config(), os.environ) is False


# ------------------------------------------------------------------------------------------------ AC3: /health/dar never blocks
def test_health_fulldoc_is_loading_and_does_not_block_while_warming(world, monkeypatch):
    _, cfg, rows = world
    g = _gate(monkeypatch, rows)
    t = fulldoc_warm.start(cfg, ENV_DB)
    assert g.entered.wait(WAIT)
    out = {}
    h = threading.Thread(target=lambda: out.setdefault("r", dar_health._fulldoc_status()), daemon=True)
    t0 = time.monotonic()
    h.start()
    h.join(2.0)
    assert not h.is_alive() and time.monotonic() - t0 < 2.0  # returned while the load is still held
    r = out["r"]
    assert r["status"] == "loading" and r["source"] == "db" and r["n_acquired"] == 0
    assert set(r) == {"enabled", "bundle_configured", "db_configured", "status", "n_acquired", "n_unusable", "pins_ok",
                      "source", "codes"}  # the readiness() DB shape
    assert RUN not in str(list(r.values()))  # no run id
    g.release.set()
    t.join(WAIT)
    assert fulldoc_warm.state() == fulldoc_warm.DONE
    after = fc.readiness(cfg, ENV_DB)  # the cache is warm: no second load
    assert after["status"] == "ok" and after["n_acquired"] == 3 and g.n_loads() == 1


def test_health_dar_endpoint_while_warming(monkeypatch):
    from fastapi.testclient import TestClient
    key = "k" * 24
    for k in ("VERA_DB_URL_RO", "DATABASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VERA_API_KEY", key)
    import main
    monkeypatch.setattr(main.dar_health, "cached_readiness", lambda *a, **k: {"dar": "ready"})
    monkeypatch.setattr(fulldoc_warm, "warming", lambda: True)

    def blocked(*a, **k):
        raise AssertionError("readiness() would block on the in-flight load")
    monkeypatch.setattr(fc, "readiness", blocked)
    r = TestClient(main.app).get("/health/dar", headers={"X-API-Key": key})
    assert r.status_code == 200 and r.json()["fulldoc"]["status"] == "loading"


# ------------------------------------------------------------------------------------------------ AC4 / AC5: no double work
def test_no_double_load_with_requests_during_warming(world, monkeypatch):
    _, cfg, rows = world
    g = _gate(monkeypatch, rows)
    t = fulldoc_warm.start(cfg, ENV_DB)
    assert g.entered.wait(WAIT)
    got = []
    reqs = [threading.Thread(target=lambda: got.append(fc.get_corpus(cfg, ENV_DB)), daemon=True) for _ in range(4)]
    for r in reqs:
        r.start()
    time.sleep(0.2)
    assert all(r.is_alive() for r in reqs)  # requests wait on the same in-flight load, as before
    g.release.set()
    for r in reqs:
        r.join(WAIT)
    t.join(WAIT)
    assert len(got) == 4 and len({id(c) for c in got}) == 1 and got[0].status == "ok"
    assert g.n_loads() == 1  # ONE load_db for warm-up + four concurrent requests


def test_first_ask_retrieval_after_warmup_does_not_rebuild(world, monkeypatch):
    _, cfg, rows = world
    g = _gate(monkeypatch, rows, released=True)
    t = fulldoc_warm.start(cfg, ENV_DB)
    t.join(WAIT)
    assert fulldoc_warm.state() == fulldoc_warm.DONE
    c = fc.get_corpus(cfg, ENV_DB)
    ev = c._ev_dar
    assert ev is not None  # evidence index already built by the warm-up

    def boom(*a, **k):
        raise AssertionError("rebuilt on the request path")
    monkeypatch.setattr(dar_corpus.DarCorpus, "ingest_document", boom)
    monkeypatch.setattr(fulldoc_db, "load_db", boom)
    assert fc.get_corpus(cfg, ENV_DB) is c
    spans = c.retrieve(tdb.Q, evidence_only=True)
    assert spans and all(s["in_evidence"] for s in spans) and c._ev_dar is ev and g.n_loads() == 1


def test_retrieval_unchanged_by_warmup(world, monkeypatch):
    """Same candidates with and without the warm-up (no ranking/retrieval change)."""
    _, cfg, rows = world
    _gate(monkeypatch, rows, released=True)
    lazy = fc.get_corpus(cfg, ENV_DB)
    want = [(q, ev, lazy.retrieve(q, evidence_only=ev)) for q in tdb.QS for ev in (False, True)]
    fc.reset_for_tests()
    fulldoc_warm.reset_for_tests()
    fulldoc_warm.start(cfg, ENV_DB).join(WAIT)
    warm = fc.get_corpus(cfg, ENV_DB)
    assert warm is not lazy
    assert [(q, ev, warm.retrieve(q, evidence_only=ev)) for q, ev, _ in want] == want


# ------------------------------------------------------------------------------------------------ AC6: failures non-fatal
def test_db_failure_is_non_fatal_logged_by_code_and_negative_cached(world, monkeypatch, caplog):
    _, cfg, _ = world
    calls = []

    def bad_connect():
        calls.append(1)
        raise ConnectionError("host db.secret.example password=hunter2 run 7")
    monkeypatch.setattr(fulldoc_db, "default_connect", lambda cfg, env: bad_connect)
    caplog.set_level(logging.INFO, logger="vera.fulldoc_warm")
    t = fulldoc_warm.start(cfg, ENV_DB)
    t.join(WAIT)
    assert fulldoc_warm.state() == fulldoc_warm.FAILED and calls == [1]
    text = caplog.text
    assert fulldoc_warm.R_WARM_UNUSABLE in text
    assert "secret" not in text and "hunter2" not in text and "run 7" not in text
    c = fc.get_corpus(cfg, ENV_DB)  # within the failure-cache TTL: no second DB hit (existing behaviour)
    assert c.status == "corpus_unusable" and calls == [1]
    assert dar_health._fulldoc_compute()["status"] != "loading"  # health no longer reports loading


def test_exception_in_warm_thread_is_non_fatal_and_logs_class_only(world, monkeypatch, caplog):
    _, cfg, _ = world

    def explode(*a, **k):
        raise RuntimeError("db.secret.example key=abc")
    monkeypatch.setattr(fc, "get_corpus", explode)
    caplog.set_level(logging.INFO, logger="vera.fulldoc_warm")
    fulldoc_warm.start(cfg, ENV_DB).join(WAIT)
    assert fulldoc_warm.state() == fulldoc_warm.FAILED
    assert "warm_error (RuntimeError)" in caplog.text and "secret" not in caplog.text and "key=abc" not in caplog.text


def test_start_never_raises(monkeypatch):
    monkeypatch.setattr(fulldoc_warm, "wanted", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    fulldoc_warm.reset_for_tests()
    assert fulldoc_warm.start({}, {}) is None
    fulldoc_warm.reset_for_tests()


def test_single_flight_follower_survives_leader_failure(world, monkeypatch):
    """A leader that dies without a result never leaves followers waiting: they retry and load once more."""
    _, cfg, rows = world
    srcs = tdb._db(rows)
    entered, release, n = threading.Event(), threading.Event(), []
    real = fulldoc_db.load_db

    def load(connect, run_id, cfg_, ingest_cfg=None):
        n.append(1)
        if len(n) == 1:
            entered.set()
            release.wait(WAIT)
            raise KeyboardInterrupt  # non-Exception: escapes load_db's own handling
        return real(tdb._factory(*srcs), run_id, cfg_)
    monkeypatch.setattr(fulldoc_db, "load_db", load)
    cache = {}
    died = []

    def leader():
        try:
            fulldoc_db.cached_load(None, RUN, cfg, cache)
        except KeyboardInterrupt:
            died.append(1)
    lt = threading.Thread(target=leader, daemon=True)
    lt.start()
    assert entered.wait(WAIT)
    got = []
    ft = threading.Thread(target=lambda: got.append(fulldoc_db.cached_load(None, RUN, cfg, cache)), daemon=True)
    ft.start()
    time.sleep(0.1)
    release.set()
    lt.join(WAIT)
    ft.join(WAIT)
    assert died == [1] and got and got[0].status == "ok" and len(n) == 2 and not fulldoc_db._INFLIGHT
