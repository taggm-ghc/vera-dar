"""PURGE-WIRING-1105 offline tests: fake engine only, no DB."""
import logging
import time

import pytest

from vera import request_trace as rt
from vera import retention_purge as rp


class Conn:
    def __init__(self, eng):
        self.eng = eng

    def execute(self, sql, params=None):
        if self.eng.fail:
            raise RuntimeError("db down")
        self.eng.calls.append(str(sql))
        row = self.eng.row

        class R:
            def first(_s):
                return row
        return R()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Eng:
    def __init__(self, fail=False, row=(3, 2)):
        self.calls, self.fail, self.row, self.disposed = [], fail, row, 0

    def begin(self):
        return Conn(self)

    connect = begin

    def dispose(self):
        self.disposed += 1


def mk(eng, made):
    def f(url, **kw):
        made.append((url, kw))
        return eng
    return f


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


ENV = {"VERA_DB_URL_WO": "postgresql://fake"}
P = {"enabled": True, "interval_s": 100, "initial_delay_s": 10, "statement_timeout_ms": 60000}


def test_flag_off_no_connection():
    made = []
    p = rp.Purger({"enabled": False}, {}, mk(Eng(), made), ENV, Clock())
    assert p.tick() is False and made == []
    p2 = rp.Purger(None, {}, mk(Eng(), made), ENV, Clock())
    assert p2.tick() is False and made == []


def test_due_time_once_per_interval_and_only_018_function():
    eng, made, c, ev = Eng(), [], Clock(), []
    p = rp.Purger(P, {"connect_timeout_s": 2}, mk(eng, made), ENV, c)
    assert p.tick(ev.append) is False and made == []          # before initial delay
    c.t += 10
    assert p.tick(ev.append) is True
    assert eng.calls == [rp.PURGE_SQL] and "purge_expired_trace()" in eng.calls[0]
    assert made[0][0] == "postgresql://fake"
    assert "statement_timeout=60000" in made[0][1]["connect_args"]["options"]
    assert eng.disposed == 1
    assert ev == [{"stage": "retention_purge", "code": "purged", "counts": {"events_deleted": 3, "requests_deleted": 2}}]
    c.t += 99
    assert p.tick() is False and len(made) == 1                # not due again
    c.t += 1
    assert p.tick() is True and len(made) == 2


def test_error_logs_class_only_and_backs_off(caplog):
    eng, made, c = Eng(fail=True), [], Clock()
    p = rp.Purger(P, {}, mk(eng, made), ENV, c)
    c.t += 10
    with caplog.at_level(logging.INFO, logger="vera.trace"):
        assert p.tick() is False                               # never raises
    assert "RuntimeError" in caplog.text and "db down" not in caplog.text and "fake" not in caplog.text
    assert eng.disposed == 1
    c.t += 9
    assert p.tick() is False and len(made) == 1                # first retry after 10 s
    c.t += 1
    assert p.tick() is False and len(made) == 2 and p.fails == 2   # next wait 20 s
    c.t += 20
    eng.fail = False
    assert p.tick() is True and p.fails == 0


def test_missing_url_and_factory_error_back_off():
    c = Clock()
    p = rp.Purger(P, {}, mk(Eng(), []), {}, c)
    c.t += 10
    assert p.tick() is False and p.fails == 1

    def boom(url, **kw):
        raise ValueError("x")
    p = rp.Purger(P, {}, boom, ENV, c)
    c.t += 10
    assert p.tick() is False and p.fails == 1


def test_backoff_capped_at_interval():
    c = Clock()
    p = rp.Purger(P, {}, mk(Eng(fail=True), []), ENV, c)
    for _ in range(8):
        c.t = p.next_due
        p.tick()
    assert p.next_due - c.t == 100


def test_bad_numbers_fall_back_to_defaults():
    p = rp.Purger({"enabled": True, "interval_s": -5, "initial_delay_s": True}, {}, None, ENV, Clock())
    assert p.interval == 86400 and p.delay == 300


def test_shipped_config_ships_off():
    import json
    cfg = json.loads(rt.CONFIG_PATH.read_text())
    assert cfg["retention_purge"]["enabled"] is False


def test_writer_thread_runs_purge_on_own_engine_not_flush_engine():
    flush, made_flush, made_purge = Eng(), [], []
    purge_eng = Eng()
    calls = []

    def factory(url, **kw):
        kw_opts = kw["connect_args"]["options"]
        if "60000" in kw_opts:
            made_purge.append(url)
            return purge_eng
        made_flush.append(url)
        return flush
    cfg = {"enabled": True, "db_sink": {"enabled": True, "flush_interval_s": 0.02,
                                        "db_url_env_wo": "VERA_DB_URL_WO"},
           "log_sink": {"enabled": True},
           "retention_purge": {"enabled": True, "interval_s": 100, "initial_delay_s": 0.05, "statement_timeout_ms": 60000},
           "limits": {}}
    rt.configure(cfg, engine_factory=factory, env=ENV)
    try:
        rt._writer_put(("evt", {}))       # starts the writer; the row itself may fail to insert (fake) and is dropped
        end = time.monotonic() + 3
        while time.monotonic() < end and not purge_eng.calls:
            time.sleep(0.02)
        assert purge_eng.calls == [rp.PURGE_SQL]
        assert all("purge_expired_trace" not in c for c in flush.calls)
        assert made_purge == ["postgresql://fake"]
    finally:
        rt.configure({"enabled": False})


def test_writer_flag_off_opens_no_purge_connection():
    made = []
    cfg = {"enabled": True, "db_sink": {"enabled": True, "flush_interval_s": 0.02},
           "retention_purge": {"enabled": False, "initial_delay_s": 0.01}, "limits": {}}
    rt.configure(cfg, engine_factory=lambda url, **kw: made.append(kw["connect_args"]["options"]) or Eng(), env=ENV)
    try:
        rt._writer_put(("evt", {}))
        time.sleep(0.3)
        assert all("60000" not in o for o in made)
    finally:
        rt.configure({"enabled": False})


def _boot_cfg(on):
    return {"enabled": True, "db_sink": {"enabled": True, "flush_interval_s": 0.02},
            "retention_purge": {"enabled": on, "interval_s": 100, "initial_delay_s": 0.05, "statement_timeout_ms": 60000},
            "limits": {}}


def test_boot_flag_off_starts_nothing():
    rt.configure(_boot_cfg(False), engine_factory=lambda url, **kw: Eng(), env=ENV)
    try:
        assert rt.start_purge_at_boot() is False and rt._WRITER is None
    finally:
        rt.configure({"enabled": False})


def test_boot_flag_on_starts_once_purges_idle_and_request_start_is_noop():
    purge_eng, made = Eng(), []

    def factory(url, **kw):
        made.append(kw["connect_args"]["options"])
        return purge_eng
    rt.configure(_boot_cfg(True), engine_factory=factory, env=ENV)
    try:
        t0 = time.monotonic()
        assert rt.start_purge_at_boot() is True
        assert time.monotonic() - t0 < 0.5                       # boot not blocked
        w = rt._WRITER
        assert w is not None and w.thread.is_alive()
        assert rt.start_purge_at_boot() is False                 # second boot call: no second scheduler
        rt._writer_put(("evt", {}))                              # first request: no-op start
        assert rt._WRITER is w
        assert rt._WRITER is w and w.thread.is_alive()           # same singleton, no second scheduler
        end = time.monotonic() + 3                               # purge ran with no request traffic at all
        while time.monotonic() < end and rp.PURGE_SQL not in purge_eng.calls:
            time.sleep(0.02)
        assert rp.PURGE_SQL in purge_eng.calls
    finally:
        rt.configure({"enabled": False})
