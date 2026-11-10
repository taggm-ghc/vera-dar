"""CHUNK-USAGE-1110: chunk_usage counters, safety and demotion, with fakes only (no database, no network)."""
import math
import threading
from types import SimpleNamespace

import pytest

from vera import chunk_usage as cu

STAMPS = [{"span_id": 7, "start": 0, "end": 4000}, {"span_id": 8, "start": 4000, "end": 8000},
          {"span_id": None, "start": None, "end": None, "basis": "document"}]


def holder(corpus, did):
    return {"corpus_stamps": STAMPS}


class FakeConn:
    def __init__(self, eng):
        self.e = eng

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.e.calls.append((sql, params))
        if self.e.fail and "INSERT" in sql:
            raise RuntimeError("boom")
        r = SimpleNamespace()
        r.scalar = lambda: self.e.table
        r.fetchall = lambda: self.e.rows
        return r


class FakeEngine:
    def __init__(self, table=True, fail=False, rows=()):
        self.table, self.fail, self.rows, self.calls = table, fail, list(rows), []

    def connect(self):
        return FakeConn(self)

    begin = connect

    def dispose(self):
        pass


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    cu.reset_for_tests()
    monkeypatch.setattr(cu, "_ensure_thread", lambda: None)  # no background thread in tests
    cu.set_config_for_tests({"enabled": True})
    yield
    cu.reset_for_tests()


def test_chunk_key_midpoint_and_skips():
    assert cu.chunk_key(STAMPS, 100, 500) == (7, 100, 500)
    assert cu.chunk_key(STAMPS, 3900, 4300) == (8, 3900, 4300)  # midpoint 4100 -> second partition
    assert cu.chunk_key([STAMPS[2]], 0, 400) is None  # document-level stamp has no id
    assert cu.chunk_key(STAMPS, 500, 100) is None
    assert cu.chunk_key(STAMPS, True, 5) is None


def test_note_admission_counts_matched_and_selected():
    matched = [{"document_id": "d", "start": 0, "end": 400}, {"document_id": "d", "start": 4000, "end": 4400}]
    admitted = [{"document_id": "d", "span_id": "d:0-400", "start": 0, "end": 400, "corpus_chunks": STAMPS}]
    cu.note_admission(None, matched, admitted, holder)
    assert cu.pending_for_tests() == {(7, 0, 400): (1, 1, 0), (8, 4000, 4400): (1, 0, 0)}


def test_merged_entry_expands_and_used_counts():
    ent = {"document_id": "d", "span_id": "d:0-900", "start": 0, "end": 900, "corpus_chunks": STAMPS,
           "merged_span_ids": ("d:0-400", "d:400-900")}
    ev = SimpleNamespace(admitted=(ent, {"document_id": "d", "span_id": "d:5000-5400", "start": 5000, "end": 5400, "corpus_chunks": STAMPS}))
    cu.note_used(ev, [{"span_id": "d:0-900"}, {"span_id": "d:0-900"}])
    assert cu.pending_for_tests() == {(7, 0, 400): (0, 0, 1), (7, 400, 900): (0, 0, 1)}


def test_disabled_and_suppressed_record_nothing():
    cu.set_config_for_tests({"enabled": False})
    cu.note_admission(None, [{"document_id": "d", "start": 0, "end": 400}], [], holder)
    assert cu.pending_for_tests() == {}
    cu.set_config_for_tests({"enabled": True})
    with cu.suppressed():
        cu.note_admission(None, [{"document_id": "d", "start": 0, "end": 400}], [], holder)
    assert cu.pending_for_tests() == {}


def test_note_never_raises_on_garbage():
    cu.note_admission(None, [None, 3], [object()], holder)
    cu.note_used(None, None)
    cu.note_used(SimpleNamespace(admitted=5), [{"span_id": "x"}])


def test_queue_is_bounded():
    cu.set_config_for_tests({"enabled": True, "queue_max": 10})
    cu._note([(i, 0, 5) for i in range(50)], 0)
    assert len(cu.pending_for_tests()) == 10
    assert cu._STATE["dropped"] == 40


def test_lock_contention_drops_instead_of_waiting(monkeypatch):
    cu._LOCK.acquire()
    try:
        monkeypatch.setattr(cu, "LOCK_TIMEOUT_S", 0.01)
        cu._note([(1, 0, 5)], 0)
    finally:
        cu._LOCK.release()
    assert cu.pending_for_tests() == {} and cu._STATE["dropped"] == 1


def test_flush_batches_and_upserts():
    eng = FakeEngine()
    cu.set_engine_factory_for_tests(lambda: eng)
    cu.set_config_for_tests({"enabled": True, "batch_max": 2})
    cu._note([(1, 0, 5), (2, 0, 5), (3, 0, 5)], 0)
    cu._note([(1, 0, 5)], 2)
    assert cu.flush_once() == 3
    ups = [c for c in eng.calls if "INSERT" in c[0]]
    assert len(ups) == 2 and len(ups[0][1]) == 2 and len(ups[1][1]) == 1  # bounded batches
    assert ups[0][1][0] == {"span_id": 1, "chunk_start": 0, "chunk_end": 5, "m": 1, "s": 0, "u": 1}
    assert "ON CONFLICT" in ups[0][0] and "question" not in ups[0][0].lower()
    assert cu.pending_for_tests() == {}


def test_table_absent_means_inert_and_queue_cleared():
    eng = FakeEngine(table=False)
    cu.set_engine_factory_for_tests(lambda: eng)
    cu._note([(1, 0, 5)], 0)
    assert cu.flush_once() == 0
    assert not [c for c in eng.calls if "INSERT" in c[0]]
    assert cu.pending_for_tests() == {}


def test_db_error_drops_batch_without_raising():
    eng = FakeEngine(fail=True)
    cu.set_engine_factory_for_tests(lambda: eng)
    cu._note([(1, 0, 5)], 0)
    assert cu.flush_once() == 0
    assert cu._STATE["failed_batches"] == 1 and cu.pending_for_tests() == {}


def test_engine_factory_error_is_swallowed():
    def bad():
        raise RuntimeError("no_url")
    cu.set_engine_factory_for_tests(bad)
    cu._note([(1, 0, 5)], 0)
    assert cu.flush_once() == 0


def test_demotion_off_by_default_and_with_zero_weight():
    assert cu.demoter(holder, None) is None
    cu.set_config_for_tests({"enabled": True, "demote": {"weight": 0}})
    assert cu.demoter(holder, None) is None


def test_demotion_factor_formula(monkeypatch):
    cu.set_config_for_tests({"enabled": True, "demote": {"weight": 0.5, "k_min": 50, "threshold": 1.0}})
    monkeypatch.setattr(cu, "_refresh_snapshot_async", lambda: None)
    assert cu.demoter(holder, None) is None  # no snapshot yet => unchanged
    cu._STATE["snap"] = {(7, 0, 400): 99}
    f = cu.demoter(holder, None)
    got = f({"document_id": "d", "start": 0, "end": 400})
    assert math.isclose(got, 1 / (1 + 0.5 * (math.log(100) - 1.0)))
    assert f({"document_id": "d", "start": 400, "end": 800}) == 1.0  # not in snapshot


def test_snapshot_refresh_reads_rows_and_keeps_old_on_error():
    eng = FakeEngine(rows=[(7, 0, 400, 80)])
    cu.set_engine_factory_for_tests(lambda: eng)
    assert cu.refresh_snapshot() == 1 and cu._STATE["snap"] == {(7, 0, 400): 80}
    cu.set_engine_factory_for_tests(lambda: (_ for _ in ()).throw(RuntimeError("x")))
    cu._STATE["table_ok"] = None
    cu.refresh_snapshot()
    assert cu._STATE["snap"] == {(7, 0, 400): 80}


def test_shipped_config_is_demotion_off():
    cu.reset_for_tests()
    c = cu.config()
    assert c["enabled"] is True and c["demote"]["weight"] == 0


def test_admission_hook_demotes_rank_only(monkeypatch):
    """With weight > 0 the admission sort key of a known never-used chunk is scaled; nothing is removed."""
    from vera import fulldoc_admission as fa
    monkeypatch.setattr(cu, "_refresh_snapshot_async", lambda: None)
    cu.set_config_for_tests({"enabled": True, "demote": {"weight": 10, "k_min": 1, "threshold": 0}})
    cu._STATE["snap"] = {(7, 0, 400): 1000}
    f = cu.demoter(fa._stamp_holder, SimpleNamespace(stamps={"d": {"corpus_stamps": STAMPS}}))
    a = {"document_id": "d", "start": 0, "end": 400}
    b = {"document_id": "d", "start": 400, "end": 800}
    scored = [(-0.9, 0.0, "a", a), (-0.8, 0.0, "b", b)]
    scored.sort(key=lambda t: (t[0] * f(t[3]), t[1], t[2]))
    assert [t[2] for t in scored] == ["b", "a"] and len(scored) == 2
