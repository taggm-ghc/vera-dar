"""CHUNK-USAGE-1110: chunk_usage staging/commit, association weights, safety and demotion, with fakes only (no database, no network)."""
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from vera import chunk_usage as cu

STAMPS = [{"span_id": 7, "start": 0, "end": 4000}, {"span_id": 8, "start": 4000, "end": 8000},
          {"span_id": None, "start": None, "end": None, "basis": "document"}]
DOM, TOP = cu.NODE_KINDS["tax_domain"], cu.NODE_KINDS["tax_topic"]
DAR = cu.NODE_KINDS["dar_domain"]


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
        r.rowcount = 1
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


def entry(start, end, sid=None, **kw):
    return dict({"document_id": "d", "span_id": sid or f"d:{start}-{end}", "start": start, "end": end, "corpus_chunks": STAMPS}, **kw)


def cand(start, end, **kw):
    return dict({"document_id": "d", "start": start, "end": end}, **kw)


def ev_of(*entries):
    return SimpleNamespace(admitted=tuple(entries))


# ------------------------------------------------------------------------------------------------ keys, nodes, staging
def test_chunk_key_midpoint_and_skips():
    assert cu.chunk_key(STAMPS, 100, 500) == (7, 100, 500)
    assert cu.chunk_key(STAMPS, 3900, 4300) == (8, 3900, 4300)
    assert cu.chunk_key([STAMPS[2]], 0, 400) is None
    assert cu.chunk_key(STAMPS, 500, 100) is None and cu.chunk_key(STAMPS, True, 5) is None


def test_nodes_carry_a_kind_so_equal_ids_do_not_collide():
    n = cu._nodes_of(cand(0, 400, branches=["domain:3"], nodes=[("tax_domain", 3), (3, 3), {"kind": "tax_topic", "id": 3}, ("bad", 3), ("tax_topic", 0)]))
    assert n == ((DAR, 3), (DOM, 3), (TOP, 3))


def test_staging_counts_nothing_and_commit_counts_the_delivered_bundle():
    a = entry(0, 400)
    cu.stage_admission(None, [cand(0, 400, nodes=[("tax_domain", 5)]), cand(4000, 4400)], [a], holder)
    assert cu.pending_for_tests() == {"tot": {}, "assoc": {}}  # staged only
    cu.commit_answer(ev_of(a), [{"span_id": a["span_id"]}])
    p = cu.pending_for_tests()
    assert p["tot"] == {(7, 0, 400): (1, 1, 1), (8, 4000, 4400): (1, 0, 0)}
    assert p["assoc"] == {(7, DOM, 5): (1, 1, 1)}  # partition level; only the routed chunk makes an association


def test_selected_but_not_cited_counts_selected_only():
    a = entry(0, 400)
    cu.stage_admission(None, [cand(0, 400, nodes=[("tax_domain", 5)])], [a], holder)
    cu.commit_answer(ev_of(a), [])
    assert cu.pending_for_tests()["assoc"] == {} or cu.pending_for_tests()["assoc"][(7, DOM, 5)] == (1, 1, 0)


def test_discarded_passes_are_never_counted_and_a_stage_commits_once():
    a, b = entry(0, 400), entry(4000, 4400)
    cu.stage_admission(None, [cand(0, 400)], [a], holder)  # e.g. an acquisition pass whose bundle is discarded
    cu.stage_admission(None, [cand(4000, 4400)], [b], holder)
    cu.commit_answer(ev_of(b), [{"span_id": b["span_id"]}])
    assert cu.pending_for_tests()["tot"] == {(8, 4000, 4400): (1, 1, 1)}  # a's pass is not in the answer: nothing counted
    cu.commit_answer(ev_of(b), [{"span_id": b["span_id"]}])
    assert cu.pending_for_tests()["tot"] == {(8, 4000, 4400): (1, 1, 1)}  # committed once
    cu.commit_answer(ev_of(), [])
    cu.commit_answer(None, None)


def test_merged_bundle_commits_the_union_of_stages():
    a, b = entry(0, 400), entry(4000, 4400)
    cu.stage_admission(None, [cand(0, 400)], [a], holder)
    cu.stage_admission(None, [cand(4000, 4400)], [b], holder)
    cu.commit_answer(ev_of(a, b), [{"span_id": a["span_id"]}])
    assert cu.pending_for_tests()["tot"] == {(7, 0, 400): (1, 1, 1), (8, 4000, 4400): (1, 1, 0)}


def test_merged_entry_expands_to_member_chunks():
    ent = entry(0, 900, merged_span_ids=("d:0-400", "d:400-900"))
    cu.stage_admission(None, [cand(0, 400), cand(400, 900)], [ent], holder)
    cu.commit_answer(ev_of(ent), [{"span_id": "d:0-900"}])
    assert cu.pending_for_tests()["tot"] == {(7, 0, 400): (1, 1, 1), (7, 400, 900): (1, 1, 1)}


def test_node_attribution_is_per_stage_not_global():
    a1, a2 = entry(0, 400), entry(0, 400)  # two concurrent requests admit the same chunk through different nodes
    cu.stage_admission(None, [cand(0, 400, nodes=[("tax_domain", 1)])], [a1], holder)
    cu.stage_admission(None, [cand(0, 400, nodes=[("tax_domain", 2)])], [a2], holder)
    cu.commit_answer(ev_of(a1), [{"span_id": a1["span_id"]}])
    assert set(cu.pending_for_tests()["assoc"]) == {(7, DOM, 1)}


def test_stage_cache_is_bounded():
    cu.set_config_for_tests({"enabled": True, "stage_max": 8})
    for i in range(30):
        cu.stage_admission(None, [cand(0, 400)], [entry(0, 400)], holder)
    assert len(cu._STAGES) == 8


def test_disabled_and_suppressed_stage_nothing():
    cu.set_config_for_tests({"enabled": False})
    a = entry(0, 400)
    cu.stage_admission(None, [cand(0, 400)], [a], holder)
    assert len(cu._STAGES) == 0
    cu.set_config_for_tests({"enabled": True})
    with cu.suppressed():
        cu.stage_admission(None, [cand(0, 400)], [a], holder)
    assert len(cu._STAGES) == 0


def test_hooks_never_raise_on_garbage():
    cu.stage_admission(None, [None, 3], [object()], holder)
    cu.commit_answer(SimpleNamespace(admitted=5), [{"span_id": "x"}])
    cu.commit_answer(SimpleNamespace(admitted=(1, "x")), None)


def test_queue_is_bounded_and_lock_contention_drops(monkeypatch):
    cu.set_config_for_tests({"enabled": True, "queue_max": 10})
    cu._add([(i, 0, 5) for i in range(50)], 0, {})
    assert len(cu.pending_for_tests()["tot"]) == 10 and cu._STATE["dropped"] == 40
    cu._STATE["tot"].clear()
    cu._LOCK.acquire()
    try:
        monkeypatch.setattr(cu, "LOCK_TIMEOUT_S", 0.01)
        cu._add([(1, 0, 5)], 0, {})
    finally:
        cu._LOCK.release()
    assert cu.pending_for_tests()["tot"] == {}


# ------------------------------------------------------------------------------------------------ flush
def test_flush_batches_totals_and_requires_a_baseline_for_associations():
    eng = FakeEngine()
    cu.set_engine_factory_for_tests(lambda: eng)
    cu.set_config_for_tests({"enabled": True, "batch_max": 2, "assoc": {"k": 2}})
    cu._add([(1, 0, 5), (2, 0, 5), (3, 0, 5)], 0, {(1, 0, 5): ((DOM, 4), (DOM, 9)), (2, 0, 5): ((TOP, 4),)})
    n = cu.flush_once()
    ins = [c for c in eng.calls if "INSERT" in c[0]]
    tot = [c for c in ins if "vera_vjay.chunk_use (" in c[0]]
    assert sum(len(c[1]) for c in tot) == 3 and len(tot) == 2  # bounded batches
    assert not [c for c in ins if "chunk_use_assoc" in c[0]]  # F5: no source wired => no association row at all
    assert n == 3 and cu._STATE["dropped"] == 3
    cu.set_similarity_source(lambda sid, kind, node: 0.62 if (kind, node) == (DOM, 4) else None)
    cu._add([(1, 0, 5), (2, 0, 5)], 1, {(1, 0, 5): ((DOM, 4), (DOM, 9)), (2, 0, 5): ((TOP, 4),)})
    cu.flush_once()
    ass = [p for c in eng.calls if "chunk_use_assoc" in c[0] and "INSERT" in c[0] for p in c[1]]
    assert len(ass) == 1 and (ass[0]["node_kind"], ass[0]["node_id"], ass[0]["init"], ass[0]["ssu0"], ass[0]["k"]) == (DOM, 4, 0.62, 1, 2)


def test_table_absent_means_inert_and_queue_cleared():
    eng = FakeEngine(table=False)
    cu.set_engine_factory_for_tests(lambda: eng)
    cu._add([(1, 0, 5)], 0, {})
    assert cu.flush_once() == 0 and not [c for c in eng.calls if "INSERT" in c[0]]
    assert cu.pending_for_tests() == {"tot": {}, "assoc": {}}


def test_db_error_drops_batch_and_factory_error_is_swallowed():
    eng = FakeEngine(fail=True)
    cu.set_engine_factory_for_tests(lambda: eng)
    cu._add([(1, 0, 5)], 0, {})
    assert cu.flush_once() == 0 and cu._STATE["failed_batches"] == 1 and cu.pending_for_tests()["tot"] == {}

    def bad():
        raise RuntimeError("no_url")
    cu.set_engine_factory_for_tests(bad)
    cu._add([(1, 0, 5)], 0, {})
    assert cu.flush_once() == 0


# ------------------------------------------------------------------------------------------------ weights
CFG = {"enabled": True, "assoc": {"k": 2, "decay": 0.5, "floor": 0.1, "restore": 0.5, "remove_below": 0.3, "relative": True}}


def S(**kw):
    return dict({"initial": 0.8, "weight": 0.8, "ssu": 0, "removed": False}, **kw)


def test_no_decay_before_k_then_multiplicative_decay_on_selected_not_cited():
    cu.set_config_for_tests(CFG)
    st = cu.step(S(), 1, 0)
    assert st["weight"] == 0.8 and st["ssu"] == 1
    st = cu.step(st, 1, 0)
    assert math.isclose(st["weight"], 0.4) and st["ssu"] == 2
    assert math.isclose(cu.step(st, 1, 0)["weight"], 0.2)
    assert math.isclose(cu.step(S(ssu=1), 3, 0)["weight"], 0.8 * 0.5 ** 3)
    assert cu.step(S(), 0, 0)["weight"] == 0.8  # matched only: no decay


def test_floor_is_relative_to_initial():
    cu.set_config_for_tests(CFG)
    st = S()
    for _ in range(30):
        st = cu.step(st, 1, 0)
    assert math.isclose(st["weight"], cu.config()["assoc"]["floor"] * 0.8) and st["weight"] > 0


def test_threshold_removal_is_relative_by_default_and_soft():
    cu.set_config_for_tests(CFG)
    st = cu.step(S(weight=0.5, ssu=5), 1, 0)
    assert st["removed"] is False  # 0.25 >= 0.3 * 0.8
    st = cu.step(st, 1, 0)
    assert st["removed"] is True and st["weight"] > 0
    cu.set_config_for_tests(dict(CFG, assoc=dict(CFG["assoc"], relative=False, remove_below=0.3)))
    assert cu.step(S(weight=0.35, ssu=5), 1, 0)["removed"] is True


def test_use_restores_toward_initial_never_above_and_reverses_removal():
    cu.set_config_for_tests(CFG)
    st = cu.step(S(weight=0.1, ssu=9, removed=True), 0, 1)
    assert math.isclose(st["weight"], 0.45) and st["ssu"] == 0 and st["removed"] is False
    assert cu.step(S(weight=0.79), 0, 5)["weight"] <= 0.8
    cu.set_config_for_tests(dict(CFG, assoc=dict(CFG["assoc"], restore=0.1, remove_below=0.9)))
    assert cu.step(S(weight=0.05, removed=True), 0, 1)["removed"] is True


def test_sql_files_carry_the_rules_and_never_delete():
    sql = cu.SQL_UPSERT_ASSOC
    for frag in ("power(", "LEAST(", "GREATEST(", "removed_at", "selected_since_use", "initial_weight", "ON CONFLICT (span_id, node_kind, node_id)"):
        assert frag in sql
    for q in (cu.SQL_UPSERT_ASSOC, cu.SQL_UPSERT_TOTAL, cu.SQL_SNAPSHOT, cu.SQL_REFRESH_INIT, cu.SQL_RESTORE):
        assert not re.search(r"\b(DELETE|TRUNCATE|DROP)\b", q.upper()) and "question" not in q.lower()
        assert "--" not in q and not q.endswith(";")
    assert set(re.findall(r"(?<!:):([a-z0-9_]+)", sql)) == {"span_id", "node_kind", "node_id", "m", "s", "u", "ssu0", "init", "restore", "k", "floor", "decay", "rel", "thr"}


def test_kit_sql_copy_matches_the_repo_files_when_present():
    kit = Path(__file__).resolve().parents[3] / "work" / "CHUNK-USAGE-MIG025-1110" / "sql"
    for name in ("chunk_usage_upsert_assoc.sql", "chunk_usage_upsert_total.sql"):
        if (kit / name).exists():
            assert (kit / name).read_bytes() == (Path(cu.__file__).resolve().parent / "chunk_usage_sql" / name).read_bytes()


def test_similarity_helpers_accept_only_cosine_kinds():
    assert cu.link_similarity(0.7, "cosine") == 0.7 and cu.link_similarity(0.7, "cosine_r2") == 0.7 and cu.link_similarity(0.7, "cosine_soft") == 0.7
    assert cu.link_similarity(0.7, "share") is None and cu.link_similarity(0.7, None) is None and cu.link_similarity(None, "cosine") is None
    assert cu.link_similarity(-0.4, "cosine") == cu.MIN_INITIAL and cu.link_similarity(7, "cosine") == 1.0
    assert cu.domain_similarity([(0.4, "cosine"), (0.9, "cosine_r1"), (0.99, "share")]) == 0.9  # max over the topic links
    assert cu.domain_similarity([(0.4, "share")]) is None and cu.domain_similarity(None) is None


def test_refresh_initial_weights_rebases_and_restore_clears():
    eng = FakeEngine()
    cu.set_engine_factory_for_tests(lambda: eng)
    assert cu.refresh_initial_weights([(7, "tax_domain", 3, 0.5), (7, "tax_domain", 4, "bad"), (7, "nope", 3, 0.5), (7, DOM, 3, -1)]) == 1
    ups = [c for c in eng.calls if "UPDATE" in c[0]]
    assert len(ups) == 1 and ups[0][1] == {"span_id": 7, "node_kind": DOM, "node_id": 3, "new": 0.5} and "weight / initial_weight" in ups[0][0]
    assert cu.restore_association(7, "tax_domain", 3) is True
    assert "removed_at = NULL" in [c for c in eng.calls if "UPDATE" in c[0]][-1][0]


def test_read_api_and_removed_set_from_snapshot():
    eng = FakeEngine(rows=[(7, DOM, 3, 0.8, 0.1, True), (7, TOP, 3, 0.5, 0.4, False)])
    cu.set_engine_factory_for_tests(lambda: eng)
    assert cu.refresh_snapshot() == 2
    cu._STATE["snap_at"] = cu._STATE["clock"]()
    assert cu.removed_associations() == frozenset({(7, DOM, 3)})
    a = cu.association(7, "tax_domain", 3)
    assert a["removed"] is True and math.isclose(a["ratio"], 0.125)
    assert math.isclose(cu.weight_ratio(7, TOP, 3), 0.8) and cu.association(7, DAR, 3) is None  # same id, other kind: no collision
    assert cu.weight_ratio(7, DOM, 99) == 1.0
    cu.set_engine_factory_for_tests(lambda: (_ for _ in ()).throw(RuntimeError("x")))
    cu._STATE["table_ok"] = None
    cu.refresh_snapshot()
    assert (7, DOM, 3) in cu._STATE["snap"]  # an error keeps the old snapshot


def test_demotion_off_by_default_and_best_association_weight(monkeypatch):
    assert cu.demoter(holder, None) is None
    cu.set_config_for_tests({"enabled": True, "demote": {"scale": 0}})
    assert cu.demoter(holder, None) is None
    cu.set_config_for_tests({"enabled": True, "demote": {"scale": 2.0}})
    monkeypatch.setattr(cu, "_refresh_snapshot_async", lambda: None)
    assert cu.demoter(holder, None) is None
    cu._STATE["snap"] = {(7, DOM, 3): (0.8, 0.2, False)}
    f = cu.demoter(holder, None)
    c = cand(0, 400)
    assert f(dict(c, nodes=[("tax_domain", 3)])) == pytest.approx(1 / (1 + 2.0 * 0.75))
    assert f(dict(c, nodes=[("tax_domain", 3), ("tax_topic", 3)])) == 1.0  # a healthy node (other kind, same id) => not demoted
    assert f(dict(c, branches=["fallback"])) == 1.0 and f(cand(400, 800, nodes=[("tax_domain", 3)])) == pytest.approx(0.4) and f(cand(4000, 4400, nodes=[("tax_domain", 3)])) == 1.0  # same partition => same weight; other partition untouched


def test_shipped_config_is_everything_off_but_counting():
    cu.reset_for_tests()
    c = cu.config()
    assert c["enabled"] is True and c["demote"]["scale"] == 0 and c["assoc"]["relative"] is True and c["snapshot"]["max_rows"] <= 25000


# ------------------------------------------------------------------------------------------------ through admit()
class FakeCorpus:
    status = "ok"

    def __init__(self):
        t = "x" * 10 + "Forecasting rainfall accuracy improved." + " y" * 20
        self.texts = {"d1": t}
        d = {"document_id": "d1", "canonical_source_id": "https://x/d1", "title": "T", "url": "https://x/d1", "doi": None, "published": "2020",
             "authors": ["A"], "licence": "CC-BY", "licence_url": "https://creativecommons.org/licenses/by/4.0/", "licence_decision": "allow",
             "attribution": "a", "text_sha256": "h", "chars": len(t), "evidence_ranges": [[0, 10 ** 9]]}
        self.acquired = [d]
        self.stamps = {"d1": {"corpus_added_at": None, "corpus_stamps": [{"span_id": 7, "start": 0, "end": 4000}]}}
        self.unusable = []

    def text(self, i):
        return self.texts[i]

    def inspection(self):
        return {"d1": {"chunks_inspected": 1}}

    def retrieve(self, q, k):
        s = "Forecasting rainfall accuracy improved."
        return [{"span_id": "d1:10-49", "document_id": "d1", "start": 10, "end": 10 + len(s), "text": s, "score": 1.0,
                 "branches": ["fallback", "domain:3"]}]


ROUTING = {"identification": {"token_pattern": r"[a-z0-9]+", "min_token_len": 3, "stopwords": ["the", "and", "what"]},
           "corpus": {"relevance_stem_len": 5}}
ACFG = {"routing": ROUTING, "retrieval": {"candidate_k": 12}, "admission": {"min_relevance": 0.5, "max_spans_total": 3, "max_spans_per_doc": 2},
        "answer": {"attribution_changes_text": "x {format}"}}
Q = "forecasting rainfall accuracy"


def test_admit_output_is_identical_with_the_feature_off_and_stages_only_when_on():
    from vera.fulldoc_admission import admit
    cu.set_config_for_tests({"enabled": False})
    off = admit(Q, FakeCorpus(), ACFG)
    assert len(cu._STAGES) == 0 and off.admitted
    cu.set_config_for_tests({"enabled": True})
    on = admit(Q, FakeCorpus(), ACFG)
    assert on == off  # byte-identical bundle: demotion scale 0, staging is invisible
    assert cu.pending_for_tests() == {"tot": {}, "assoc": {}} and len(cu._STAGES) == 1
    a = on.admitted[0]
    cu.commit_answer(on, [{"span_id": a["span_id"]}])
    p = cu.pending_for_tests()
    assert p["tot"] == {(7, 10, 49): (1, 1, 1)} and p["assoc"] == {(7, DAR, 3): (1, 1, 1)}


def test_admit_demotion_scale_zero_does_not_touch_the_order():
    from vera.fulldoc_admission import admit
    cu.set_config_for_tests({"enabled": True, "demote": {"scale": 0}})
    cu._STATE["snap"] = {(7, DAR, 3): (0.8, 0.01, True)}
    assert admit(Q, FakeCorpus(), ACFG).admitted[0]["relevance"] == 1.0
