"""DAR runtime corpus provider (vera.dar_runtime). No DB, no network, no real env reads: fake engines/readers."""
import copy
import logging
import sys
import threading
import time

import pytest

from vera import dar_runtime, domain_registry

URL = "postgresql://user:secret-pw@host.invalid:5432/db"
CFG_REAL = domain_registry.load_config()


def _cfg(on=True, **rt):
    cfg = copy.deepcopy(CFG_REAL)
    cfg["enabled"] = on
    cfg.setdefault("corpus", {})["stored_branch_enabled"] = on
    cfg["runtime"] = {"db_url_env": "VERA_DB_URL_RO", "connect_timeout_s": 5, "statement_timeout_ms": 10000,
                      "failure_retry_ttl_s": 60, **rt}
    return cfg


class FakeEngine:
    def __init__(self):
        self.disposed = 0

    def dispose(self):
        self.disposed += 1


class Factory:
    def __init__(self):
        self.engines, self.calls = [], []
        self.delay = 0.0

    def __call__(self, url, **kw):
        self.calls.append((url, kw))
        if self.delay:
            time.sleep(self.delay)
        e = FakeEngine()
        self.engines.append(e)
        return e


class FakeCorpus:
    def __init__(self, cfg=None):
        self.restored = None

    status = "restored"

    def restore(self, snap):
        self.restored = snap
        return {"status": FakeCorpus.status}


class Reader:
    boom = False
    version = "v1"
    version_boom = False
    version_calls = 0

    def corpus_version(self):
        Reader.version_calls += 1
        if Reader.version_boom:
            raise RuntimeError(f"version failed for {URL}")
        return Reader.version

    def __init__(self, engine):
        self.engine = engine

    def load_snapshot(self, kinds, span_text=None, **kw):
        if Reader.boom:
            raise RuntimeError(f"connect failed for {URL}")
        return {"kinds": list(kinds)}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    dar_runtime.reset_for_tests()
    Reader.boom = False
    Reader.version, Reader.version_boom, Reader.version_calls = "v1", False, 0
    yield
    dar_runtime.reset_for_tests()


@pytest.fixture
def on(monkeypatch):
    import vera.dar_corpus as dc
    import vera.dar_store as ds
    monkeypatch.setattr(dc, "DarCorpus", FakeCorpus)
    monkeypatch.setattr(ds, "DarReader", Reader)
    cfg = _cfg()
    monkeypatch.setattr(domain_registry, "load_config", lambda *a, **k: cfg)
    return cfg


class NoEnv:
    def get(self, *a, **k):
        raise AssertionError("env must not be read")

    __getitem__ = __contains__ = get


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def test_flag_off_touches_nothing(monkeypatch):
    for m in ("vera.dar_store", "vera.dar_corpus", "sqlalchemy"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.setattr(domain_registry, "load_config", lambda *a, **k: _cfg(on=False))
    assert dar_runtime.get_corpus(env=NoEnv(), engine_factory=Factory()) is None
    for m in ("vera.dar_store", "vera.dar_corpus", "sqlalchemy"):
        assert m not in sys.modules


@pytest.mark.parametrize("enabled,branch", [(True, False), (False, True), ("true", True), (True, 1)])
def test_flag_requires_both_json_true(monkeypatch, enabled, branch):
    cfg = _cfg()
    cfg["enabled"], cfg["corpus"]["stored_branch_enabled"] = enabled, branch
    monkeypatch.setattr(domain_registry, "load_config", lambda *a, **k: cfg)
    assert dar_runtime.get_corpus(env=NoEnv(), engine_factory=Factory()) is None


def test_success_cached_and_disposed(on):
    f = Factory()
    c1 = dar_runtime.get_corpus(env={"VERA_DB_URL_RO": URL}, engine_factory=f)
    c2 = dar_runtime.get_corpus(env=NoEnv(), engine_factory=f)
    assert isinstance(c1, FakeCorpus) and c1 is c2 and len(f.calls) == 1
    assert [e.disposed for e in f.engines] == [1]
    assert c1.restored["kinds"]  # non-document kinds passed


def test_timeouts_in_connect_args(on):
    f = Factory()
    dar_runtime.get_corpus(env={"VERA_DB_URL_RO": URL}, engine_factory=f)
    url, kw = f.calls[0]
    assert url == URL and kw["pool_pre_ping"] is True
    assert kw["connect_args"] == {"connect_timeout": 5, "options": "-c statement_timeout=10000"}


def test_concurrent_first_requests_build_once(on):
    f = Factory()
    f.delay = 0.1
    out = []
    ts = [threading.Thread(target=lambda: out.append(
        dar_runtime.get_corpus(env={"VERA_DB_URL_RO": URL}, engine_factory=f))) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(f.calls) == 1 and len(out) == 8 and all(o is out[0] and o is not None for o in out)


def test_failure_ttl_and_dispose_and_no_url_in_logs(on, caplog):
    caplog.set_level(logging.DEBUG)
    f, clk = Factory(), Clock()
    env = {"VERA_DB_URL_RO": URL}
    Reader.boom = True
    assert dar_runtime.get_corpus(env=env, engine_factory=f, now=clk) is None
    assert f.engines[0].disposed == 1
    clk.t += 59
    assert dar_runtime.get_corpus(env=NoEnv(), engine_factory=f, now=clk) is None
    assert len(f.calls) == 1  # no retry, DB untouched before TTL
    clk.t += 2
    Reader.boom = False
    assert isinstance(dar_runtime.get_corpus(env=env, engine_factory=f, now=clk), FakeCorpus)
    assert len(f.calls) == 2 and f.engines[1].disposed == 1
    text = caplog.text
    assert "RuntimeError" in text and URL not in text and "secret-pw" not in text


def test_missing_url_returns_none_cached(on):
    f = Factory()
    assert dar_runtime.get_corpus(env={}, engine_factory=f) is None
    assert dar_runtime.get_corpus(env={"VERA_DB_URL_RO": URL}, engine_factory=f) is None  # cached
    assert f.calls == []


def test_non_restored_status_is_a_cached_failure_not_a_partial_corpus(on, monkeypatch):
    monkeypatch.setattr(FakeCorpus, "status", "chunking_mismatch")
    f = Factory()
    assert dar_runtime.get_corpus(env={"VERA_DB_URL_RO": URL}, engine_factory=f) is None
    assert dar_runtime.get_corpus(env={"VERA_DB_URL_RO": URL}, engine_factory=f) is None
    assert len(f.calls) == 1 and f.engines[0].disposed == 1  # TTL-cached, engine disposed


def test_get_corpus_never_raises_even_for_bad_config(on, monkeypatch):
    bad = _cfg()
    bad["runtime"] = {}  # missing timeout keys: no hard-coded fallbacks, so the build fails, loudly but not fatally
    monkeypatch.setattr(domain_registry, "load_config", lambda *a, **k: bad)
    dar_runtime.reset_for_tests()
    assert dar_runtime.get_corpus(env={"VERA_DB_URL_RO": URL}, engine_factory=Factory()) is None


# ---- reload / refresh ----
ENV = {"VERA_DB_URL_RO": URL}


def _with_ttl(monkeypatch, **rt):
    cfg = _cfg(**rt)
    monkeypatch.setattr(domain_registry, "load_config", lambda *a, **k: cfg)


def test_ttl_expiry_one_rebuild_and_swap(on, monkeypatch):
    _with_ttl(monkeypatch, reload_ttl_s=300)
    f, clk = Factory(), Clock()
    c1 = dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)
    clk.t += 299
    assert dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk) is c1 and len(f.calls) == 1
    clk.t += 2
    c2 = dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)
    assert c2 is not c1 and isinstance(c2, FakeCorpus) and len(f.calls) == 2
    assert dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk) is c2 and len(f.calls) == 2
    assert [e.disposed for e in f.engines] == [1, 1]


def test_failed_reload_keeps_old_and_retries_after_failure_ttl(on, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    _with_ttl(monkeypatch, reload_ttl_s=300)
    f, clk = Factory(), Clock()
    c1 = dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)
    clk.t += 301
    Reader.boom = True
    assert dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk) is c1 and len(f.calls) == 2
    clk.t += 59
    assert dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk) is c1 and len(f.calls) == 2
    clk.t += 2
    Reader.boom = False
    c3 = dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)
    assert c3 is not c1 and len(f.calls) == 3
    assert "RuntimeError" in caplog.text and URL not in caplog.text and "secret-pw" not in caplog.text


def test_non_restored_reload_keeps_old(on, monkeypatch):
    _with_ttl(monkeypatch, reload_ttl_s=10)
    f, clk = Factory(), Clock()
    c1 = dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)
    clk.t += 11
    monkeypatch.setattr(FakeCorpus, "status", "chunking_mismatch")
    assert dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk) is c1
    assert f.engines[1].disposed == 1


def test_refresh_forces_rebuild(on, monkeypatch):
    _with_ttl(monkeypatch, reload_ttl_s=300)
    f, clk = Factory(), Clock()
    c1 = dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)
    c2 = dar_runtime.refresh(env=ENV, engine_factory=f, now=clk)
    assert c2 is not c1 and len(f.calls) == 2
    assert dar_runtime.get_corpus(env=NoEnv(), engine_factory=f, now=clk) is c2
    Reader.boom = True
    assert dar_runtime.refresh(env=ENV, engine_factory=f, now=clk) is c2  # failure: old kept, no raise
    assert dar_runtime.refresh(env=ENV, engine_factory=lambda *a, **k: 1 / 0, now=clk) is c2


def test_refresh_with_no_prior_corpus_failure_returns_none(on):
    Reader.boom = True
    assert dar_runtime.refresh(env=ENV, engine_factory=Factory()) is None


def test_concurrent_callers_during_rebuild_get_old_and_build_once(on, monkeypatch):
    _with_ttl(monkeypatch, reload_ttl_s=300)
    f, clk = Factory(), Clock()
    c1 = dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)
    clk.t += 301
    f.delay = 0.3
    out = []
    first = threading.Thread(target=lambda: out.append(dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)))
    first.start()
    time.sleep(0.1)  # first holds the lock, inside the slow build
    others = [threading.Thread(target=lambda: out.append(
        dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk))) for _ in range(5)]
    [t.start() for t in others]
    [t.join() for t in others]
    assert out == [c1] * 5 and len(f.calls) == 2  # others not blocked on the DB, served the old corpus
    first.join()
    assert out[-1] is not c1 and len(f.calls) == 2


@pytest.mark.parametrize("rt", [{}, {"reload_ttl_s": 0}, {"reload_ttl_s": -5}])
def test_no_reload_ttl_never_reloads(on, monkeypatch, rt):
    _with_ttl(monkeypatch, **rt)
    f, clk = Factory(), Clock()
    c1 = dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)
    clk.t += 10 ** 9
    assert dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk) is c1 and len(f.calls) == 1


def test_refresh_flag_off_touches_nothing(monkeypatch):
    monkeypatch.setattr(domain_registry, "load_config", lambda *a, **k: _cfg(on=False))
    assert dar_runtime.refresh(env=NoEnv(), engine_factory=Factory()) is None


# ---- per-request version check / status ----
def test_version_unchanged_no_rebuild_and_check_engine_disposed(on, monkeypatch):
    _with_ttl(monkeypatch, version_check_s=0)
    f, clk = Factory(), Clock()
    c1 = dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)
    for _ in range(3):
        assert dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk) is c1
    assert Reader.version_calls == 4 and len(f.calls) == 4  # 1 build + 3 checks
    assert all(e.disposed == 1 for e in f.engines)
    url, kw = f.calls[1]
    assert kw["connect_args"] == {"connect_timeout": 5, "options": "-c statement_timeout=10000"} and kw["pool_pre_ping"]


def test_version_change_one_rebuild_visible_next_call(on, monkeypatch):
    _with_ttl(monkeypatch, version_check_s=0)
    f, clk = Factory(), Clock()
    c1 = dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)
    Reader.version = "v2"  # ingest from another process
    c2 = dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)
    assert c2 is not c1 and len(f.calls) == 3  # build, check, rebuild
    assert dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk) is c2 and len(f.calls) == 4  # check only
    assert all(e.disposed == 1 for e in f.engines)


def test_failing_check_keeps_old_and_backs_off(on, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    _with_ttl(monkeypatch, version_check_s=0)
    f, clk = Factory(), Clock()
    c1 = dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)
    Reader.version_boom = True
    assert dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk) is c1
    n = len(f.calls)
    clk.t += 59
    assert dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk) is c1 and len(f.calls) == n
    clk.t += 2
    Reader.version_boom = False
    assert dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk) is c1 and len(f.calls) == n + 1
    assert "RuntimeError" in caplog.text and URL not in caplog.text and "secret-pw" not in caplog.text
    assert dar_runtime.status() == "ok" and all(e.disposed == 1 for e in f.engines)


def test_version_check_absent_means_no_check(on):
    f, clk = Factory(), Clock()
    dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)
    clk.t += 10 ** 6
    dar_runtime.get_corpus(env=NoEnv(), engine_factory=f, now=clk)
    assert Reader.version_calls == 1 and len(f.calls) == 1  # build-time version only


def test_version_check_interval(on, monkeypatch):
    _with_ttl(monkeypatch, version_check_s=5)
    f, clk = Factory(), Clock()
    c1 = dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)
    Reader.version = "v2"
    clk.t += 4
    assert dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk) is c1 and len(f.calls) == 1
    clk.t += 1.5
    c2 = dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)
    assert c2 is not c1 and len(f.calls) == 3


def test_reader_without_corpus_version_is_tolerated(on, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    _with_ttl(monkeypatch, version_check_s=0)
    monkeypatch.delattr(Reader, "corpus_version")
    f, clk = Factory(), Clock()
    c1 = dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk)
    assert dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk) is c1
    n = len(f.calls)
    assert dar_runtime.get_corpus(env=ENV, engine_factory=f, now=clk) is c1 and len(f.calls) == n  # skipped
    assert caplog.text.count("unsupported") == 1


def test_status_codes(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(domain_registry, "load_config", lambda *a, **k: _cfg(on=False))
    assert dar_runtime.status() == "disabled"
    import vera.dar_corpus as dc
    import vera.dar_store as ds
    monkeypatch.setattr(dc, "DarCorpus", FakeCorpus)
    monkeypatch.setattr(ds, "DarReader", Reader)
    monkeypatch.setattr(domain_registry, "load_config", lambda *a, **k: _cfg())
    dar_runtime.get_corpus(env={}, engine_factory=Factory())
    assert dar_runtime.status() == "unavailable:no_url"
    dar_runtime.reset_for_tests()
    Reader.boom = True
    dar_runtime.get_corpus(env=ENV, engine_factory=Factory())
    assert dar_runtime.status() == "unavailable:RuntimeError"
    dar_runtime.reset_for_tests()
    Reader.boom = False
    monkeypatch.setattr(FakeCorpus, "status", "chunking_mismatch")
    dar_runtime.get_corpus(env=ENV, engine_factory=Factory())
    assert dar_runtime.status() == "unavailable:restore_chunking_mismatch"
    dar_runtime.reset_for_tests()
    monkeypatch.setattr(FakeCorpus, "status", "restored")
    assert dar_runtime.get_corpus(env=ENV, engine_factory=Factory()) is not None
    assert dar_runtime.status() == "ok"
    assert caplog.text.count("DAR runtime status") == 4
    assert URL not in caplog.text and "secret-pw" not in caplog.text
