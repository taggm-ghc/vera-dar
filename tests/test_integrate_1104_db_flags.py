"""INTEGRATE-1104 round 2 (owner: migration 015 applied 2026-11-04): db_sink and digest_store flags on, the corpus digest
persisted behind digest_store.enabled, and the owner trace route reading the DB with the in-process window as fallback.
Offline: fake engines only, no DB or network."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vera import corpus_description as cd
from vera import digest_store
from vera import request_trace as rt

ROOT = Path(__file__).resolve().parent.parent
RID = "0123456789abcdef0123456789abcdef"


def test_shipped_flags_on():
    cfg = json.loads((ROOT / "config" / "request_trace.json").read_text())
    assert cfg["db_sink"]["enabled"] is True and cfg["digest_store"]["enabled"] is False  # digest persistence deferred
    assert cfg["db_sink"]["db_url_env_ro"] == "VERA_DB_URL_RO"
    assert cfg["log_sink"]["include_question"] is False  # unchanged (orchestrator-accepted)


def test_015_recorded_applied_and_rollback_kept():
    readme = (ROOT / "db" / "README.md").read_text()
    row = [ln for ln in readme.splitlines() if ln.startswith("| 015 ")]
    assert len(row) == 1 and "applied 2026-11-04 by the owner" in row[0] and "verification not recorded" in row[0]
    assert (ROOT / "db/migrations/rollback/015-DRAFT-request-trace-rollback.sql.draft").exists()
    assert (ROOT / "db/migrations/pending/015-DRAFT-request-trace.sql.draft").exists()  # convention: files stay put


# ------------------------------------------------------------------------------- fake DB
class Row:
    def __init__(self, d):
        self._mapping = d

    def __getitem__(self, k):
        return self._mapping[k]


class Res:
    def __init__(self, rows):
        self.rows = rows

    def first(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class Conn:
    def __init__(self, eng):
        self.eng = eng

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        q = str(sql)
        self.eng.sql.append((q, params))
        if self.eng.fail:
            raise RuntimeError("permission denied")
        if "FROM vera_vjay.request_trace_requests" in q:
            return Res(self.eng.req)
        if "FROM vera_vjay.request_trace_events" in q:
            return Res(self.eng.evs)
        if q.startswith("SELECT cache_key"):
            return Res(self.eng.digest_rows)
        if q.startswith("INSERT"):
            self.eng.inserts.append(params)
        return Res([])


class Eng:
    def __init__(self, fail=False):
        self.fail, self.sql, self.inserts = fail, [], []
        self.req = [Row({"request_id": RID, "client_request_id": None, "endpoint": "/ask", "started_at": "2026-11-04T10:00:00+00:00",
                         "question": "Q", "outcome_branch": "answer", "message_code": "ok", "event_count": 1,
                         "dropped_events": 0, "elapsed_ms": 12})]
        self.evs = [Row({"seq": 1, "ts": "2026-11-04T10:00:00+00:00", "stage": "scope", "code": "basis",
                         "counts": "{}", "detail": '{"basis": "safe"}'})]
        self.digest_rows = []

    def connect(self):
        return Conn(self)

    def begin(self):
        return Conn(self)


@pytest.fixture
def trace_on():
    old = rt.CFG
    rt.configure({"enabled": True, "db_sink": {"enabled": True, "db_url_env_ro": "VERA_DB_URL_RO"}},
                 env={"VERA_DB_URL_RO": "postgresql://fake-ro"})
    yield
    rt.configure(old)


def test_db_view_reads_request_and_events_through_ro(trace_on):
    eng = Eng()
    v = rt.db_view(RID, engine_factory=lambda *a, **k: eng)
    assert v["request_id"] == RID and v["source"] == "db" and v["events"][0]["detail"] == {"basis": "safe"}
    assert all(":rid" in q for q, _ in eng.sql) and all(p == {"rid": RID} for _, p in eng.sql)  # bound parameter only


@pytest.mark.parametrize("tid", ["nothex", "", None, RID + "0"])
def test_db_view_rejects_bad_ids_without_a_query(trace_on, tid):
    eng = Eng()
    assert rt.db_view(tid, engine_factory=lambda *a, **k: eng) is None and eng.sql == []


def test_db_view_error_or_off_returns_none(trace_on):
    assert rt.db_view(RID, engine_factory=lambda *a, **k: Eng(fail=True)) is None
    rt.configure({"enabled": True, "db_sink": {"enabled": False}}, env={"VERA_DB_URL_RO": "x"})
    assert rt.db_view(RID, engine_factory=lambda *a, **k: Eng()) is None


def test_owner_route_still_reads_the_window_only(monkeypatch):  # DB read deferred to the next increment
    import main
    monkeypatch.setattr(main.request_trace, "db_view", lambda tid: pytest.fail("DB read wired early"))
    monkeypatch.setattr(main.request_trace, "recent", lambda tid: {"source": "window"})
    assert main.trace_request(RID) == {"source": "window"}
    monkeypatch.setattr(main.request_trace, "recent", lambda tid: None)
    with pytest.raises(main.HTTPException):
        main.trace_request(RID)


# ------------------------------------------------------------------------------- digest persistence
OV = {"source_count": 7, "corpus_updated_at": "2026-11-04T09:00:00+00:00"}
CFG = {"read_db_url_env": "VERA_DB_URL_RO", "max_titles": 5, "prompt_version": "pv1", "retry_s": 60}


@pytest.fixture
def describer(monkeypatch):
    digest_store.reset_for_tests()
    monkeypatch.setattr(cd, "stamp_and_profile", lambda *a, **k: (dict(OV), {}))
    monkeypatch.setattr(cd, "agentic_cfg", lambda cfg: {"sample_excerpt_chars": 0})
    calls = []

    def pipeline(overview, profile, llm, cfg, now=None):
        calls.append(1)
        return {"summary": "S", "topics": ["t"], "coverage": "C", "topic_shares": None, "model": "m", "trace": []}
    monkeypatch.setattr(cd, "run_digest_pipeline", pipeline)
    d = cd.CorpusDescriber(CFG, engine_factory=lambda *a, **k: object(), env={"VERA_DB_URL_RO": "ro", "VERA_DB_URL_WO": "wo"})
    eng = Eng()
    d.store_cfg = {"enabled": True, "table": "vera_vjay.corpus_digest_state"}
    d.store_engine_factory = lambda *a, **k: eng
    return SimpleNamespace(d=d, calls=calls, eng=eng)


def test_new_digest_is_saved_when_flag_on(describer):
    out = describer.d.current(llm=None)
    assert out["summary"] == "S" and describer.calls == [1]
    assert len(describer.eng.inserts) == 1


def test_restart_reuses_the_stored_digest_without_an_llm_call(describer):
    key = describer.d._store_key(OV)
    stored = {**OV, "summary": "STORED", "topics": [], "coverage": None, "topic_shares": None,
              "summary_source": "agentic_model", "summary_generated_at": "x", "summary_model": "m",
              "prompt_version": describer.d.cfg["prompt_version"]}  # the describer may extend the configured version
    describer.eng.digest_rows = [Row({"cache_key": key, "digest": json.dumps(stored), "created_at": "2026-11-04"})]
    out = describer.d.current(llm=None)
    assert out["summary"] == "STORED" and describer.calls == [] and describer.eng.inserts == []


def test_stale_stored_digest_is_ignored(describer):
    describer.eng.digest_rows = [Row({"cache_key": "6|old|pv1", "digest": json.dumps({"summary": "OLD"}),
                                      "created_at": "x"})]
    assert describer.d.current(llm=None)["summary"] == "S" and describer.calls == [1]


def test_flag_off_never_touches_the_store(describer):
    describer.d.store_cfg = {"enabled": False}
    describer.d.current(llm=None)
    assert describer.eng.sql == [] and describer.calls == [1]
