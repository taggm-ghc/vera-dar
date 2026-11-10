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


def test_note_admission_counts_matched_and_selected_per_node():
    matched = [{"document_id": "d", "start": 0, "end": 400, "branches": ["fallback", "domain:3", "vector"]},
               {"document_id": "d", "start": 4000, "end": 4400, "branches": ["fallback"]}]
    admitted = [{"document_id": "d", "span_id": "d:0-400", "start": 0, "end": 400, "corpus_chunks": STAMPS}]
    cu.note_admission(None, matched, admitted, holder)
    assert cu.pending_for_tests() == {(7, 0, 400, 0): (1, 1, 0), (7, 0, 400, 3): (1, 1, 0), (8, 4000, 4400, 0): (1, 0, 0)}


def test_used_is_attributed_to_the_nodes_of_the_latest_admission():
    matched = [{"document_id": "d", "start": 0, "end": 400, "nodes": [5], "branches": ["domain:3"]}]
    admitted = [{"document_id": "d", "span_id": "d:0-400", "start": 0, "end": 400, "corpus_chunks": STAMPS}]
    cu.note_admission(None, matched, admitted, holder)
    cu._STATE["pending"].clear()
    cu.note_used(SimpleNamespace(admitted=tuple(admitted)), [{"span_id": "d:0-400"}])
    assert cu.pending_for_tests() == {(7, 0, 400, 0): (0, 0, 1), (7, 0, 400, 3): (0, 0, 1), (7, 0, 400, 5): (0, 0, 1)}


def test_merged_entry_expands_and_used_counts():
    ent = {"document_id": "d", "span_id": "d:0-900", "start": 0, "end": 900, "corpus_chunks": STAMPS,
           "merged_span_ids": ("d:0-400", "d:400-900")}
    ev = SimpleNamespace(admitted=(ent, {"document_id": "d", "span_id": "d:5000-5400", "start": 5000, "end": 5400, "corpus_chunks": STAMPS}))
    cu.note_used(ev, [{"span_id": "d:0-900"}, {"span_id": "d:0-900"}])
    assert cu.pending_for_tests() == {(7, 0, 400, 0): (0, 0, 1), (7, 400, 900, 0): (0, 0, 1)}


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
    cu._note([(i, 0, 5, 0) for i in range(50)], 0)
    assert len(cu.pending_for_tests()) == 10
    assert cu._STATE["dropped"] == 40


def test_lock_contention_drops_instead_of_waiting(monkeypatch):
    cu._LOCK.acquire()
    try:
        monkeypatch.setattr(cu, "LOCK_TIMEOUT_S", 0.01)
        cu._note([(1, 0, 5, 0)], 0)
    finally:
        cu._LOCK.release()
    assert cu.pending_for_tests() == {} and cu._STATE["dropped"] == 1


def test_flush_batches_and_upserts():
    eng = FakeEngine()
    cu.set_engine_factory_for_tests(lambda: eng)
    cu.set_config_for_tests({"enabled": True, "batch_max": 2})
    cu._note([(1, 0, 5, 0), (2, 0, 5, 0), (3, 0, 5, 7)], 0)
    cu._note([(1, 0, 5, 0)], 2)
    assert cu.flush_once() == 3
    ups = [c for c in eng.calls if "INSERT" in c[0]]
    assert len(ups) == 2 and len(ups[0][1]) == 2 and len(ups[1][1]) == 1  # bounded batches
    p0 = ups[0][1][0]
    assert (p0["span_id"], p0["node_id"], p0["m"], p0["u"], p0["init"], p0["msu0"]) == (1, 0, 1, 1, 1.0, 0)
    assert ups[0][1][1]["span_id"] == 2 and ups[1][1][0]["node_id"] == 7 and ups[1][1][0]["msu0"] == 1
    assert "ON CONFLICT" in ups[0][0] and "question" not in ups[0][0].lower()
    assert cu.pending_for_tests() == {}


def test_table_absent_means_inert_and_queue_cleared():
    eng = FakeEngine(table=False)
    cu.set_engine_factory_for_tests(lambda: eng)
    cu._note([(1, 0, 5, 0)], 0)
    assert cu.flush_once() == 0
    assert not [c for c in eng.calls if "INSERT" in c[0]]
    assert cu.pending_for_tests() == {}


def test_db_error_drops_batch_without_raising():
    eng = FakeEngine(fail=True)
    cu.set_engine_factory_for_tests(lambda: eng)
    cu._note([(1, 0, 5, 0)], 0)
    assert cu.flush_once() == 0
    assert cu._STATE["failed_batches"] == 1 and cu.pending_for_tests() == {}


def test_engine_factory_error_is_swallowed():
    def bad():
        raise RuntimeError("no_url")
    cu.set_engine_factory_for_tests(bad)
    cu._note([(1, 0, 5, 0)], 0)
    assert cu.flush_once() == 0


CFG = {"enabled": True, "assoc": {"k": 2, "decay": 0.5, "floor": 0.1, "restore": 0.5, "remove_below": 0.3, "relative": True}}


def S(**kw):
    return dict({"initial": 0.8, "weight": 0.8, "msu": 0, "removed": False}, **kw)


def test_no_decay_before_k_then_multiplicative_decay():
    cu.set_config_for_tests(CFG)
    st = cu.step(S(), 1, 0)
    assert st["weight"] == 0.8 and st["msu"] == 1  # 1 match < k=2: no decay
    st = cu.step(st, 1, 0)
    assert math.isclose(st["weight"], 0.4) and st["msu"] == 2  # k reached: one decay step
    st = cu.step(st, 1, 0)
    assert math.isclose(st["weight"], 0.2)
    assert math.isclose(cu.step(S(msu=1), 3, 0)["weight"], 0.8 * 0.5 ** 3)  # crossing k inside a batch: matches from k on decay


def test_floor_is_relative_to_initial():
    cu.set_config_for_tests(CFG)
    st = S()
    for _ in range(30):
        st = cu.step(st, 1, 0)
    assert math.isclose(st["weight"], cu.config()["assoc"]["floor"] * 0.8)  # floor clamped below remove_below


def test_threshold_removal_is_relative_by_default_and_soft():
    cu.set_config_for_tests(CFG)
    st = S(weight=0.5, msu=5)
    st = cu.step(st, 1, 0)  # 0.25 >= 0.3 * 0.8 = 0.24 -> kept
    assert st["removed"] is False
    st = cu.step(st, 1, 0)  # 0.125 < 0.24 -> removed
    assert st["removed"] is True and st["weight"] > 0  # soft: weight stays, nothing is deleted
    cu.set_config_for_tests(dict(CFG, assoc=dict(CFG["assoc"], relative=False, remove_below=0.3)))
    assert cu.step(S(weight=0.35, msu=5), 1, 0)["removed"] is True  # absolute: 0.175 < 0.3 (relative would need < 0.24)


def test_use_restores_toward_initial_never_above_and_reverses_removal():
    cu.set_config_for_tests(CFG)
    st = S(weight=0.1, msu=9, removed=True)
    st = cu.step(st, 0, 1)
    assert math.isclose(st["weight"], 0.1 + 0.7 * 0.5) and st["msu"] == 0 and st["removed"] is False  # 0.45 >= 0.24 -> cleared
    st = cu.step(S(weight=0.1, msu=9, removed=True), 0, 1)
    assert st["removed"] is False  # reversible by use
    top = cu.step(S(weight=0.79), 0, 5)
    assert top["weight"] <= 0.8 and math.isclose(top["weight"], 0.8, abs_tol=0.02)
    assert cu.step(S(weight=0.1, removed=True), 0, 1)["weight"] > 0.1


def test_a_use_that_stays_below_threshold_keeps_the_mark():
    cu.set_config_for_tests(dict(CFG, assoc=dict(CFG["assoc"], restore=0.1, remove_below=0.9)))
    assert cu.step(S(weight=0.05, removed=True), 0, 1)["removed"] is True


def test_sql_carries_the_same_rules_and_never_deletes():
    sql = cu.SQL_UPSERT
    for frag in ("power(", "LEAST(", "GREATEST(", "removed_at", "matched_since_use", "initial_weight", "EXCLUDED.node_id = 0"):
        assert frag in sql
    for q in (cu.SQL_UPSERT, cu.SQL_SNAPSHOT, cu.SQL_REFRESH_INIT, cu.SQL_RESTORE):
        assert "DELETE" not in q.upper() and "TRUNCATE" not in q.upper()
    assert "question" not in sql.lower()


def test_flush_never_issues_a_delete_and_passes_assoc_config():
    eng = FakeEngine()
    cu.set_engine_factory_for_tests(lambda: eng)
    cu.set_config_for_tests(CFG)
    cu.set_similarity_source(lambda sid, node: 0.62 if node == 4 else None)
    cu._note([(1, 0, 5, 4), (1, 0, 5, 0), (1, 0, 5, 9)], 0)
    cu.flush_once()
    ps = {p["node_id"]: p for c in eng.calls if "INSERT" in c[0] for p in c[1]}
    assert ps[4]["init"] == 0.62 and ps[0]["init"] == 1.0 and ps[9]["init"] == 1.0  # unknown similarity => 1.0; total row 1.0
    assert (ps[4]["k"], ps[4]["decay"], ps[4]["rel"]) == (2, 0.5, True)
    assert not [c for c in eng.calls if "DELETE" in c[0].upper()]


def test_similarity_baseline_is_clamped_and_never_raises():
    cu.set_similarity_source(lambda s, n: -0.4)
    assert cu._baseline(1, 3) == cu.MIN_INITIAL
    cu.set_similarity_source(lambda s, n: 7)
    assert cu._baseline(1, 3) == 1.0
    cu.set_similarity_source(lambda s, n: (_ for _ in ()).throw(RuntimeError("x")))
    assert cu._baseline(1, 3) == 1.0
    assert cu._baseline(1, 0) == 1.0


def test_refresh_initial_weights_rebases_and_restore_clears():
    eng = FakeEngine()
    cu.set_engine_factory_for_tests(lambda: eng)
    assert cu.refresh_initial_weights([(7, 3, 0.5), (7, 4, "bad"), (7, 0, 0.9)]) == 0  # fake rowcount unknown => 0, but one UPDATE ran
    ups = [c for c in eng.calls if "UPDATE" in c[0]]
    assert len(ups) == 1 and ups[0][1] == {"span_id": 7, "node_id": 3, "new": 0.5}
    assert "weight / initial_weight" in ups[0][0]  # decay ratio kept
    cu.restore_association(7, 0, 400, 3)
    assert "removed_at = NULL" in [c for c in eng.calls if "UPDATE" in c[0]][-1][0]


def test_read_api_and_removed_set_from_snapshot():
    eng = FakeEngine(rows=[(7, 0, 400, 3, 0.8, 0.1, True), (7, 0, 400, 4, 0.5, 0.4, False)])
    cu.set_engine_factory_for_tests(lambda: eng)
    assert cu.refresh_snapshot() == 2
    cu._STATE["snap_at"] = cu._STATE["clock"]()  # fresh: the read API does not refresh again
    assert cu.removed_associations() == frozenset({(7, 0, 400, 3)})
    a = cu.association(7, 0, 400, 3)
    assert a["removed"] is True and math.isclose(a["ratio"], 0.125)
    assert math.isclose(cu.weight_ratio(7, 0, 400, 4), 0.8) and cu.association(7, 0, 400, 99) is None
    assert cu.weight_ratio(7, 0, 400, 99) == 1.0


def test_snapshot_error_keeps_old_snapshot():
    cu.set_engine_factory_for_tests(lambda: FakeEngine(rows=[(7, 0, 400, 3, 0.8, 0.4, False)]))
    cu.refresh_snapshot()
    cu.set_engine_factory_for_tests(lambda: (_ for _ in ()).throw(RuntimeError("x")))
    cu._STATE["table_ok"] = None
    cu.refresh_snapshot()
    assert (7, 0, 400, 3) in cu._STATE["snap"]


def test_demotion_off_by_default():
    assert cu.demoter(holder, None) is None
    cu.set_config_for_tests({"enabled": True, "demote": {"scale": 0}})
    assert cu.demoter(holder, None) is None
    assert cu.config()["demote"]["scale"] == 0


def test_demotion_uses_best_association_weight(monkeypatch):
    cu.set_config_for_tests({"enabled": True, "demote": {"scale": 2.0}})
    monkeypatch.setattr(cu, "_refresh_snapshot_async", lambda: None)
    assert cu.demoter(holder, None) is None  # no lowered association known
    cu._STATE["snap"] = {(7, 0, 400, 3): (0.8, 0.2, False), (7, 0, 400, 4): (0.5, 0.5, False)}
    f = cu.demoter(holder, None)
    c = {"document_id": "d", "start": 0, "end": 400}
    assert f(dict(c, branches=["domain:3"])) == pytest.approx(1 / (1 + 2.0 * 0.75))
    assert f(dict(c, branches=["domain:3", "domain:9"])) == 1.0  # one healthy node (domain 9: no row) => no demotion
    assert f(dict(c, branches=["fallback"])) == 1.0  # no routing node => not demoted
    assert f({"document_id": "d", "start": 400, "end": 800, "branches": ["domain:3"]}) == 1.0


def test_shipped_config_is_everything_off_but_counting():
    cu.reset_for_tests()
    c = cu.config()
    assert c["enabled"] is True and c["demote"]["scale"] == 0 and c["assoc"]["relative"] is True


def test_admission_hook_demotes_rank_only(monkeypatch):
    from vera import fulldoc_admission as fa
    monkeypatch.setattr(cu, "_refresh_snapshot_async", lambda: None)
    cu.set_config_for_tests({"enabled": True, "demote": {"scale": 10}})
    cu._STATE["snap"] = {(7, 0, 400, 3): (0.8, 0.08, True)}
    f = cu.demoter(fa._stamp_holder, SimpleNamespace(stamps={"d": {"corpus_stamps": STAMPS}}))
    a = {"document_id": "d", "start": 0, "end": 400, "branches": ["domain:3"]}
    b = {"document_id": "d", "start": 400, "end": 800, "branches": ["domain:3"]}
    scored = [(-0.9, 0.0, "a", a), (-0.8, 0.0, "b", b)]
    scored.sort(key=lambda t: (t[0] * f(t[3]), t[1], t[2]))
    assert [t[2] for t in scored] == ["b", "a"] and len(scored) == 2  # reordered, none removed
