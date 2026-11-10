"""TAXO-TRIM-1109: the post-pass release hook runs after every pass exit and is a no-op off glibc. Offline; no DB, no network."""
from __future__ import annotations

import logging
from itertools import count

from tests.test_chunk_taxo_job_1109 import ENV, FakeDB, FakeEmb, StubCore, cfg, factory_for, make, spans
from vera import chunk_taxo_job as job


def _spy(monkeypatch):
    calls = []
    monkeypatch.setattr(job, "malloc_trim_release", lambda enabled=True: calls.append(enabled) or True)
    return calls


def test_hook_after_normal_pass(monkeypatch):
    calls = _spy(monkeypatch)
    assert make(FakeDB(spans())).run()["state"] != "error"
    assert calls == [True]


def test_hook_after_early_stop(monkeypatch):
    calls = _spy(monkeypatch)
    db = FakeDB(spans())
    ticks = count(0, 100)
    j = job.Job(cfg(max_seconds=150), core=StubCore(), embedder=FakeEmb(), engine_factory=factory_for(db), env=dict(ENV),
                clock=lambda: next(ticks), rss_reader=lambda: 1.0)
    assert j.run()["stop"] == "max_seconds"
    assert calls == [True]


def test_hook_after_error_paths(monkeypatch):
    calls = _spy(monkeypatch)
    db = FakeDB(spans())
    db.fail_on = "link_insert_chunk_subtopic"
    assert make(db).run()["state"] == "error"
    assert len(calls) == 1

    def boom(url, **kw):
        raise RuntimeError("x")
    assert job.Job(cfg(), core=StubCore(), embedder=FakeEmb(), engine_factory=boom, env=dict(ENV)).run()["state"] == "error"
    assert len(calls) == 2


def test_hook_not_called_when_off(monkeypatch):
    calls = _spy(monkeypatch)
    assert job.Job({"enabled": False}, core=StubCore(), env=dict(ENV)).run()["state"] == "off"
    assert calls == []


def test_flag_default_on_and_disable(monkeypatch):
    calls = _spy(monkeypatch)
    assert make(FakeDB(spans())).trim_on is True
    j = make(FakeDB(spans()), malloc_trim=False)
    j.run()
    assert j.trim_on is False and calls == [False]
    assert make(FakeDB(spans()), env=dict(ENV, CHUNK_TAXO_MALLOC_TRIM="0")).trim_on is False


def test_noop_off_glibc(monkeypatch):
    import ctypes
    loaded = []
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: loaded.append(a) or (_ for _ in ()).throw(AssertionError("loaded")))
    monkeypatch.setattr(job.sys, "platform", "darwin")
    assert job.malloc_trim_release(True) is False
    monkeypatch.setattr(job.sys, "platform", "linux")
    import platform
    monkeypatch.setattr(platform, "libc_ver", lambda *a, **k: ("musl", ""))
    assert job.malloc_trim_release(True) is False
    assert job.malloc_trim_release(False) is False
    assert loaded == []


def test_failure_ignored_and_logged_once(monkeypatch, caplog):
    import ctypes
    import platform
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(job.sys, "platform", "linux")
    monkeypatch.setattr(platform, "libc_ver", lambda *a, **k: ("glibc", "2.39"))
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    monkeypatch.setattr(job, "_TRIM_WARNED", [False])
    assert job.malloc_trim_release(True) is False and job.malloc_trim_release(True) is False
    assert caplog.text.count("malloc_trim unavailable") == 1
