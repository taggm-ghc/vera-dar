"""PURGE-TEST-LAND-1105: enabled + migration 018 NOT applied (function missing). Offline, fake engine only.

The purge is enabled in memory only (rt.configure / Purger config dicts); config/request_trace.json is never read
or written by this module."""
import logging
import time

from vera import request_trace as rt
from vera import retention_purge as rp


class ProgrammingError(Exception):
    """Stands in for sqlalchemy.exc.ProgrammingError (psycopg UndefinedFunction)."""


MSG = "function vera_vjay.purge_expired_trace() does not exist SECRETHOST"


class Conn:
    def __init__(self, e): self.e = e
    def execute(self, sql, params=None):
        self.e.calls.append(str(sql))
        if "purge_expired_trace" in str(sql):
            raise ProgrammingError(MSG)
        class R:
            def first(_s): return None
        return R()
    def __enter__(self): return self
    def __exit__(self, *a): return False


class Eng:
    def __init__(self): self.calls, self.disposed = [], 0
    def begin(self): return Conn(self)
    connect = begin
    def dispose(self): self.disposed += 1


ENV = {"VERA_DB_URL_WO": "postgresql://fake"}


def test_missing_function_fails_closed_bounded_log(caplog):
    t = [1000.0]
    eng = Eng()
    p = rp.Purger({"enabled": True, "interval_s": 100, "initial_delay_s": 10}, {}, lambda u, **k: eng, ENV, lambda: t[0])
    t[0] += 10
    with caplog.at_level(logging.INFO, logger="vera.trace"):
        assert p.tick() is False
    assert "ProgrammingError" in caplog.text and "RT_PURGE_ERROR" in caplog.text
    assert "SECRETHOST" not in caplog.text and "does not exist" not in caplog.text
    assert eng.disposed == 1 and p.runs == 0
    waits = []
    for _ in range(6):                      # bounded retry: doubles, capped at interval_s
        waits.append(p.next_due - t[0])
        t[0] = p.next_due
        p.tick()
    assert waits == [10, 20, 40, 80, 100, 100]


def test_writer_boot_and_requests_unaffected(caplog):
    flush, purge = Eng(), Eng()
    # flush engine and purge engine are told apart by statement_timeout option
    def factory(url, **kw):
        return purge if "60000" in kw["connect_args"]["options"] else flush
    cfg = {"enabled": True, "db_sink": {"enabled": True, "flush_interval_s": 0.02},
           "retention_purge": {"enabled": True, "interval_s": 100, "initial_delay_s": 0.05, "statement_timeout_ms": 60000},
           "limits": {}}
    rt.configure(cfg, engine_factory=factory, env=ENV)
    try:
        with caplog.at_level(logging.INFO, logger="vera.trace"):
            t0 = time.monotonic()
            assert rt.start_purge_at_boot() is True          # boot returns at once
            assert time.monotonic() - t0 < 0.5
            w = rt._WRITER
            end = time.monotonic() + 3
            while time.monotonic() < end and not purge.calls:
                time.sleep(0.02)
            assert any("purge_expired_trace" in c for c in purge.calls)
            time.sleep(0.2)
            assert w.thread.is_alive()                        # writer survives the failure
            t1 = time.monotonic()
            rt._writer_put(("evt", {}))                       # request-path enqueue never blocks or raises
            assert time.monotonic() - t1 < 0.1
        assert "SECRETHOST" not in caplog.text
        assert w.thread.is_alive()
    finally:
        rt.configure({"enabled": False})
