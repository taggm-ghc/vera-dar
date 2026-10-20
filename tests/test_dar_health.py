"""P0-E: DAR readiness codes (fake engine; no real DB, no network). Overall /health stays independent of the DB."""
import re

import pytest

from vera import dar_health, dar_runtime

SENTINEL_HOST = "sentinel-host.invalid"
URL = f"postgresql://u:pw@{SENTINEL_HOST}:5432/db"
ENV = {"VERA_DB_URL_RO": URL}
_URLISH = re.compile(r"(postgres|://|sentinel-host|pw@)", re.I)


class _Res:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Conn:
    def __init__(self, tables, counts):
        self.tables, self.counts = tables, counts

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "to_regclass" in sql:
            return _Res([(params["n"].split(".")[1] in self.tables,)])
        t = sql.split(".")[-1].strip()
        return _Res([(self.counts.get(t, 0),)])


class _Engine:
    disposed = False

    def __init__(self, tables, counts, fail=False):
        self.tables, self.counts, self.fail = tables, counts, fail

    def connect(self):
        if self.fail:
            raise RuntimeError(f"could not connect to {URL}")
        return _Conn(self.tables, self.counts)

    def dispose(self):
        _Engine.disposed = True


ALL = {"licence_observations", "research_domains", "domain_corpus", "domain_chunks"}
BUILT = {"domain_corpus": 3, "domain_chunks": 40, "research_domains": 2}


def _factory(tables=ALL, counts=BUILT, fail=False):
    return lambda url, **kw: _Engine(tables, counts, fail)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    dar_health.reset_for_tests()
    monkeypatch.setattr(dar_runtime, "status", lambda: "ok")
    yield
    dar_health.reset_for_tests()


def test_ready_when_all_present_and_built():
    r = dar_health.dar_readiness(ENV, _factory())
    assert r == {"dar": "ready", "migrations": {"006": True, "007": True}, "stored_corpus": "ready", "routing": "on",
                 "corpus": {"documents": 3, "chunks": 40, "domains": 2}, "fallback": False, "reason_code": "ok"}
    assert _Engine.disposed


def test_007_missing_never_ready():
    r = dar_health.dar_readiness(ENV, _factory(tables={"licence_observations"}))
    assert r["dar"] != "ready" and r["fallback"] is True
    assert r["migrations"] == {"006": True, "007": False} and r["corpus"] is None
    assert r["reason_code"] == "migration_007_missing"


def test_partial_007_missing_one_table():
    r = dar_health.dar_readiness(ENV, _factory(tables=ALL - {"domain_chunks"}))
    assert r["migrations"]["007"] is False and r["dar"] != "ready"


def test_empty_corpus_degraded():
    r = dar_health.dar_readiness(ENV, _factory(counts={}))
    assert r["dar"] == "degraded" and r["stored_corpus"] == "empty" and r["fallback"] is True


def test_runtime_not_ok_degraded(monkeypatch):
    monkeypatch.setattr(dar_runtime, "status", lambda: "unavailable:restore_chunking_mismatch")
    r = dar_health.dar_readiness(ENV, _factory())
    assert r["dar"] == "degraded" and r["fallback"] is True and r["reason_code"].startswith("runtime_")


def test_disabled_routing(monkeypatch):
    monkeypatch.setattr(dar_runtime, "status", lambda: "disabled")
    r = dar_health.dar_readiness(ENV, _factory())
    assert r["dar"] == "disabled" and r["routing"] == "off" and r["fallback"] is False


def test_db_unreachable_unavailable_no_leak():
    r = dar_health.dar_readiness(ENV, _factory(fail=True))
    assert r["dar"] == "unavailable" and r["fallback"] is True and r["reason_code"] == "db_unreachable"
    assert r["migrations"] == {"006": None, "007": None}
    assert not _URLISH.search(repr(r))


def test_no_url_unavailable():
    r = dar_health.dar_readiness({}, _factory())
    assert r["dar"] == "unavailable" and r["reason_code"] == "no_db_url"


def test_no_url_or_host_in_any_output(caplog):
    caplog.set_level("DEBUG")
    outs = [dar_health.dar_readiness(ENV, _factory(), force=True), dar_health.dar_readiness(ENV, _factory(fail=True), force=True),
            dar_health.startup_line(ENV, _factory(fail=True)), dar_health.startup_line(ENV, _factory())]
    assert not _URLISH.search(repr(outs)) and not _URLISH.search(caplog.text)


def test_never_raises_and_cached():
    calls = []

    def f(url, **kw):
        calls.append(1)
        return _Engine(ALL, BUILT)
    t = [0.0]
    dar_health.dar_readiness(ENV, f, now=lambda: t[0])
    dar_health.dar_readiness(ENV, f, now=lambda: t[0] + 1)
    assert len(calls) == 1
    assert dar_health.dar_readiness(ENV, lambda url, **kw: 1 / 0, force=True)["dar"] == "unavailable"


def test_health_endpoint_dar_fields_and_status_independent(monkeypatch):
    """Requires the owner-gated main.py /health hook. status stays ok whatever the DAR state."""
    from fastapi.testclient import TestClient
    import main
    if "dar" not in TestClient(main.app).get("/health").json():
        pytest.skip("owner-gated main.py hook (l4-main-health-dar.diff) not applied")
    for k in list(__import__("os").environ):
        if "DB_URL" in k:
            monkeypatch.delenv(k)
    monkeypatch.setattr(dar_health, "cached_readiness", lambda *a, **k: dar_health.dar_readiness({}, None, force=True))
    r = TestClient(main.app).get("/health")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok"
    assert body["dar"] in {"degraded", "unavailable", "disabled", "unknown"}  # v3: coarse public enum only
    monkeypatch.setenv("VERA_API_KEY", "k" * 24)
    full = TestClient(main.app).get("/health/dar", headers={"X-API-Key": "k" * 24}).json()
    assert set(full) == {"dar", "migrations", "stored_corpus", "routing", "corpus", "fallback", "reason_code"}
    assert full["dar"] != "ready"
    monkeypatch.setattr(dar_health, "cached_readiness", lambda *a, **k: 1 / 0)
    assert TestClient(main.app).get("/health").json()["status"] == "ok"


def test_startup_line_names_missing_migrations_by_code(caplog):
    caplog.set_level("INFO")
    line = dar_health.startup_line(ENV, _factory(tables=set()))
    assert "m007=False" in line and "missing_migrations=007,006" in line
    assert any("missing_migrations=007,006" in r.getMessage() for r in caplog.records)
    assert "missing_migrations=007 " in dar_health.startup_line(ENV, _factory(tables={"licence_observations"}), )


def test_env_none_uses_runtime_url_resolution(monkeypatch):
    """env=None (production): the URL comes from dar_runtime's own env handling (also what an injected env reaches)."""
    seen = []

    def fake_engine_for(rt, env, engine_factory):
        seen.append(1)
        return engine_factory(URL, connect_args={})
    monkeypatch.setattr(dar_runtime, "_engine_for", fake_engine_for, raising=False)
    r = dar_health.dar_readiness(None, _factory(tables=ALL - {"research_domains"}), force=True)
    assert seen and r["migrations"]["007"] is False and r["reason_code"] == "migration_007_missing"
    assert not _URLISH.search(repr(r))


def test_runtime_not_built_yet_is_still_ready(monkeypatch):
    monkeypatch.setattr(dar_runtime, "status", lambda: "unavailable:not_built")
    assert dar_health.dar_readiness(ENV, _factory())["dar"] == "ready"


def test_not_probed_state_is_unknown_not_unavailable(monkeypatch):
    """Uncached readiness: dar/routing/fallback are None (unknown), distinct from a real outage or routing off."""
    monkeypatch.setattr(dar_health, "_CACHE", {})
    monkeypatch.setattr(dar_health, "_compute", lambda env, ef: (_ for _ in ()).throw(RuntimeError("no probe in test")))
    r = dar_health.cached_readiness()
    assert r["reason_code"] == "not_probed"
    assert r["dar"] is None and r["routing"] is None and r["fallback"] is None and r["corpus"] is None


def test_health_error_state_is_unknown_not_unavailable(monkeypatch):
    """A failing readiness computation reports unknown (None), never "unavailable"/"off"."""
    monkeypatch.setattr(dar_health, "_compute", lambda env, ef: (_ for _ in ()).throw(RuntimeError("boom")))
    r = dar_health.dar_readiness(ENV, None, force=True)
    assert r["reason_code"] == "health_error"
    assert r["dar"] is None and r["routing"] is None and r["fallback"] is None
