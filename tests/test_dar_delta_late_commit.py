"""D-053: late-commit freshness (lookback overlap) and bounded incremental load. FakeDB (no DB, network, env)."""
import copy
import json

import pytest

from vera import dar_runtime, domain_registry
from vera.dar_corpus import DarCorpus
from vera.dar_store import DarReader, _lower_ts
from tests.test_dar_freshness import DOCS, ENV, Calls, FakeDB, Ingestor, _cfg, _factory, Clock, StubCorpus, StubReader, _Eng, _get  # noqa: F401

EARLY = "2026-10-19T00:00:00+00:00"
WIN = {"delta_lookback_ids": 10, "delta_lookback_s": 120, "delta_max_documents": 50, "delta_max_assocs": 500}


def _late_scenario(monkeypatch, **rt):
    """B (id 2) commits first; the runtime builds; A (id 1, older txn start => older created_at) commits late."""
    db, clk = FakeDB(), Clock()
    a = Ingestor(db)
    a.n = 1
    a.add(DOCS[1])                      # id 2
    cfg = _cfg(**rt)
    monkeypatch.setattr(domain_registry, "load_config", lambda *x, **k: cfg)
    dar_runtime.reset_for_tests()
    runtime = dar_runtime.DarRuntime()
    c0 = runtime.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    assert c0.stats()["documents"] == 1
    a.n = 0
    a.add(DOCS[0])                      # id 1, commits after the cursor passed it
    sp1 = {s["span_id"] for s in db.spans if s["source_id"] == 1}
    for r in db.dcorp:
        if r["c"] == 1:
            r["ts"] = EARLY             # created_at = txn START (older than the cursor timestamp)
    for r in db.dchunks:
        if r["s"] in sp1:
            r["ts"] = EARLY
    return db, clk, runtime


def test_late_commit_missed_without_lookback(monkeypatch):
    """Pre-fix behaviour (lookback keys absent = legacy): the late document is never read by the delta."""
    db, clk, runtime = _late_scenario(monkeypatch)
    c = runtime.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    assert c.stats()["documents"] == 1


def test_late_commit_found_with_lookback_and_idempotent(monkeypatch):
    db, clk, runtime = _late_scenario(monkeypatch, **WIN)
    c = runtime.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    assert c.stats()["documents"] == 2
    first = json.dumps(c.stats(), sort_keys=True, default=str)
    # force more deltas over the same window: nothing changes (idempotent re-read)
    for _ in range(2):
        db.domains.append({"domain_id": len(db.domains) + 1, "name": "Zed%d" % len(db.domains), "name_norm": "zed%d" % len(db.domains),
                           "description": "", "origin": "derived"})
        c = runtime.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    s = c.stats()
    assert s["documents"] == 2 and s["skipped"] == 0
    assert json.loads(first)["documents"] == 2


def test_doc_only_late_commit_changes_version_token():
    db = FakeDB()
    a = Ingestor(db)
    a.n = 1
    a.add(DOCS[1])
    v1 = DarReader(db).corpus_version()
    a.n = 0
    a.add(DOCS[0])
    # a lower id leaves every max unchanged; the counts must still change the token
    assert DarReader(db).corpus_version() != v1


def test_bound_hit_returns_partial_never_truncated():
    db = FakeDB()
    a = Ingestor(db)
    a.add(DOCS[0])
    cur = DarReader(db).load_snapshot(["full_text"])["cursor"]
    a.add(DOCS[1]); a.add(DOCS[2])
    rd = DarReader(db)
    ok = rd.load_delta(cur, ["full_text"], max_documents=2)
    assert "partial" not in ok and len(ok["documents"]) == 2
    p = rd.load_delta(cur, ["full_text"], max_documents=1)
    assert p["partial"] is True and p["reason"] == "delta_max_documents"
    assert p["documents"] == [] and p["corpus_assoc"] == [] and p["chunk_assoc"] == [] and p["cursor"] == cur
    p2 = rd.load_delta(cur, ["full_text"], max_assocs=1)
    assert p2["partial"] is True and p2["reason"] == "delta_max_assocs" and p2["documents"] == []
    sql = [c for c in db.sql_log if "LIMIT CAST(:lim_assoc" in c[0]]
    assert sql and all(c[1]["lim_assoc"] == 2 for c in sql)


def test_lookback_rows_do_not_count_toward_bound():
    db = FakeDB()
    a = Ingestor(db)
    a.add(DOCS[0]); a.add(DOCS[1])
    cur = DarReader(db).load_snapshot(["full_text"])["cursor"]
    a.add(DOCS[2])
    d = DarReader(db).load_delta(cur, ["full_text"], max_documents=1, lookback_ids=5, lookback_s=60)
    assert "partial" not in d and [x["source_id"] for x in d["documents"]] == [1, 2, 3]  # 2 re-read + 1 new


def test_delta_sql_params_lowered_and_cursor_monotone():
    db = FakeDB()
    a = Ingestor(db)
    a.add(DOCS[0]); a.add(DOCS[1])
    cur = DarReader(db).load_snapshot(["full_text"])["cursor"]
    db.sql_log.clear()
    d = DarReader(db).load_delta(cur, ["full_text"], lookback_ids=1, lookback_s=30)
    prm = [p for _, p in db.sql_log]
    assert any(p.get("since_source_id") == cur["source_id"] - 1 for p in prm)
    assert any(p.get("since_assoc_ts") == _lower_ts(cur["assoc_ts"], 30) for p in prm)
    assert d["cursor"]["source_id"] == cur["source_id"] and d["cursor"]["assoc_ts"] == cur["assoc_ts"]


def test_naive_cursor_timestamp_is_partial_not_silently_unwindowed():
    cur = {"source_id": 5, "span_id": 0, "domain_id": 1, "assoc_ts": "2026-10-19 00:00:05"}
    r = DarReader(FakeDB()).load_delta(cur, ["full_text"], lookback_s=10)
    assert r["partial"] is True and r["reason"] == "cursor_ts_unparsable"


def test_delta_limits_config_and_clamp():
    assert dar_runtime.delta_limits({}) == {}
    assert dar_runtime.delta_limits({"delta_lookback_ids": 0, "delta_max_documents": True}) == {}
    out = dar_runtime.delta_limits({"delta_max_documents": 10, "delta_lookback_ids": 50, "delta_lookback_s": 5,
                                    "delta_max_assocs": 7})
    assert out == {"max_documents": 10, "lookback_ids": 9, "lookback_s": 5.0, "max_assocs": 7}
    shipped = domain_registry.load_config()["runtime"]
    lim = dar_runtime.delta_limits(shipped)
    assert set(lim) == {"max_documents", "max_assocs", "lookback_ids", "lookback_s"}
    assert lim["lookback_ids"] < lim["max_documents"]


def test_runtime_partial_delta_rebuilds_and_never_advances_version(monkeypatch):
    db, clk = FakeDB(), Clock()
    a = Ingestor(db)
    a.add(DOCS[0])
    cfg = _cfg(delta_max_documents=1)
    monkeypatch.setattr(domain_registry, "load_config", lambda *x, **k: cfg)
    dar_runtime.reset_for_tests()
    snaps = []
    import vera.dar_store as ds

    class Spy(DarReader):
        def load_snapshot(self, *x, **k):
            snaps.append(1)
            return super().load_snapshot(*x, **k)
    monkeypatch.setattr(ds, "DarReader", Spy)
    rt = dar_runtime.DarRuntime()
    rt.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    a.add(DOCS[1]); a.add(DOCS[2])      # 2 new docs > bound 1
    c = rt.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    assert len(snaps) == 2 and c.stats()["documents"] == 3   # rebuilt, not truncated


def test_partial_delta_from_reader_is_never_applied(stub):
    rt = stub
    _get(rt, Clock())
    StubReader.version = "v2"
    orig = StubReader.load_delta
    StubReader.load_delta = lambda self, since, kinds=None, span_text=None, **kw: {
        "documents": [], "domains": [], "corpus_assoc": [], "chunk_assoc": [], "cursor": dict(since),
        "partial": True, "reason": "delta_max_documents"}
    try:
        _get(rt, Clock())
    finally:
        StubReader.load_delta = orig
    assert StubCorpus.applied == [] and StubReader.snapshots == 2


def test_overlap_reread_is_idempotent_and_rejected_docs_not_duplicated():
    c = DarCorpus(copy.deepcopy(domain_registry.load_config()))
    bad = {"source_key": "http://x/abstract", "url": "http://x/abstract", "title": "t", "text": "short",
           "stored_text_kind": "abstract", "admission_passed": False, "db_source_id": 9, "db_canonical_source_id": 9}
    delta = {"domains": [], "documents": [bad], "chunk_assoc": [],
             "corpus_assoc": [{"name_norm": "nope", "source_key": bad["source_key"], "asserted_by": "dar"}]}
    for _ in range(3):
        assert c.apply_delta(copy.deepcopy(delta))["status"] == "applied"
    assert c.stats()["skipped"] == 1


def test_assoc_for_unknown_domain_needs_full():
    c = DarCorpus(copy.deepcopy(domain_registry.load_config()))
    r = c.apply_delta({"domains": [], "documents": [], "chunk_assoc": [], "corpus_assoc": [
        {"name_norm": "nope", "source_key": "nokey", "asserted_by": "dar"}]})
    assert r["status"] == "needs_full"  # unknown document is reported first; never silently skipped


@pytest.fixture
def stub(monkeypatch):
    import vera.dar_corpus as dc
    import vera.dar_store as ds
    StubCorpus.mode, StubCorpus.applied = "applied", []
    StubReader.version, StubReader.snapshots, StubReader.deltas, StubReader.delta_boom = "v1", 0, 0, False
    monkeypatch.setattr(dc, "DarCorpus", StubCorpus)
    monkeypatch.setattr(ds, "DarReader", StubReader)
    cfg = _cfg()
    monkeypatch.setattr(domain_registry, "load_config", lambda *a, **k: cfg)
    return dar_runtime.DarRuntime()


# ------------------------------------------------------------------ v2: new versions, snapshot bound, one transaction
def test_new_version_of_known_document_needs_full_then_rebuild_sees_it():
    cfg = copy.deepcopy(domain_registry.load_config())
    c = DarCorpus(cfg)
    base = {"source_key": "http://x/p", "url": "http://x/p", "title": "t", "text": "alpha beta gamma " * 80,
            "year": 2020, "published": "2020", "source_type": "journal", "retrieved_at": "2026-10-19",
            "stored_text_kind": "full_text", "admission_passed": True, "db_source_id": 5, "db_canonical_source_id": 5}
    assert c.apply_delta({"domains": [], "documents": [base], "corpus_assoc": [], "chunk_assoc": []})["status"] == "applied"
    same = dict(base)   # overlap re-read of the SAME version: ignored
    assert c.apply_delta({"domains": [], "documents": [same], "corpus_assoc": [], "chunk_assoc": []})["status"] == "applied"
    newer = dict(base, db_source_id=9, text="delta epsilon " * 80)
    r = c.apply_delta({"domains": [], "documents": [newer], "corpus_assoc": [], "chunk_assoc": []})
    assert r == {"status": "needs_full", "reason": "new_version_of_known_document"}


def test_runtime_applies_new_version_through_rebuild(monkeypatch):
    db, clk = FakeDB(), Clock()
    a = Ingestor(db)
    a.add(DOCS[0])
    cfg = _cfg(**WIN)
    monkeypatch.setattr(domain_registry, "load_config", lambda *x, **k: cfg)
    dar_runtime.reset_for_tests()
    import vera.dar_store as ds
    snaps = []

    class Spy(DarReader):
        def load_snapshot(self, *x, **k):
            snaps.append(1)
            return super().load_snapshot(*x, **k)
    monkeypatch.setattr(ds, "DarReader", Spy)
    rt = dar_runtime.DarRuntime()
    rt.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    # same canonical source, version 2 (new source_id 50, newer text)
    old = db.sources[0]
    db.sources.append({**old, "source_id": 50, "version": 2, "content": old["content"]})  # same text: old-version span rows stay valid
    c = rt.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    assert len(snaps) == 2   # delta -> needs_full -> rebuild picked the latest version
    assert any(d.get("db_source_id") == 50 for d in c._docs.values())


def test_snapshot_bound_marks_partial_and_surfaces_it(monkeypatch):
    db, clk = FakeDB(), Clock()
    a = Ingestor(db)
    a.add(DOCS[0]); a.add(DOCS[1]); a.add(DOCS[2])
    snap = DarReader(db).load_snapshot(["full_text"], max_documents=2)
    assert snap["partial"] is True and snap["reason"] == "snapshot_max_documents" and snap["cursor"] is None
    assert len(snap["documents"]) == 2
    keys = {d["source_key"] for d in snap["documents"]}
    assert all(x["source_key"] in keys for x in snap["corpus_assoc"] + snap["chunk_assoc"])
    full = DarReader(db).load_snapshot(["full_text"], max_documents=3)
    assert "partial" not in full and len(full["documents"]) == 3 and full["cursor"] is not None
    cfg = _cfg(snapshot_max_documents=2)
    monkeypatch.setattr(domain_registry, "load_config", lambda *x, **k: cfg)
    dar_runtime.reset_for_tests()
    rt = dar_runtime.DarRuntime()
    c = rt.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    assert c.stats()["documents"] == 2 and rt.status() == "partial:snapshot_max_documents"
    assert rt._state["cursor"] is None   # no deltas on a truncated corpus
    cfg2 = _cfg(snapshot_max_documents=5)
    monkeypatch.setattr(domain_registry, "load_config", lambda *x, **k: cfg2)
    rt2 = dar_runtime.DarRuntime()
    rt2.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    assert rt2.status() == "ok"


def test_partial_runtime_status_is_degraded_in_health(monkeypatch):
    from tests import test_dar_health as th
    from vera import dar_health
    dar_health.reset_for_tests()
    monkeypatch.setattr(dar_runtime, "status", lambda: "partial:snapshot_max_documents")
    r = dar_health.dar_readiness(th.ENV, th._factory(), force=True)
    assert r["dar"] == "degraded" and r["reason_code"] == "runtime_snapshot_max_documents" and r["fallback"] is True
    dar_health.reset_for_tests()


class _OneConn:
    def __init__(self, eng):
        self.eng = eng

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execution_options(self, **kw):
        self.eng.opts.append(kw)
        return self

    def execute(self, stmt, params=None):
        self.eng.stmts += 1
        sql = str(stmt)
        from tests.test_dar_freshness import _Res
        return _Res([(1, 0, 0, 0, 0)] if False else [])


class _PgLike:
    class dialect:  # noqa: N801
        name = "postgresql"

    def __init__(self, name="postgresql"):
        self.dialect = type("D", (), {"name": name})
        self.opts, self.connects, self.stmts = [], 0, 0

    def connect(self):
        self.connects += 1
        return _OneConn(self)


def test_reads_share_one_repeatable_read_transaction_on_postgres():
    e = _PgLike("postgresql")
    DarReader(e).load_delta({"source_id": 1, "span_id": 0, "domain_id": 1, "assoc_ts": EARLY}, ["full_text"], lookback_ids=1)
    assert e.connects == 1 and e.stmts >= 4
    assert e.opts == [{"isolation_level": "REPEATABLE READ", "postgresql_readonly": True}]
    e2 = _PgLike("postgresql")
    DarReader(e2).load_snapshot(["full_text"])
    assert e2.connects == 1 and e2.stmts >= 4


def test_sqlite_degrades_to_plain_single_connection():
    e = _PgLike("sqlite")
    DarReader(e).load_delta({"source_id": 1, "span_id": 0, "domain_id": 1, "assoc_ts": EARLY}, ["full_text"])
    assert e.connects == 1 and e.opts == []


# ------------------------------------------------------------------ v3: re-versioned text must not wedge the rebuild
def test_new_version_with_different_text_rebuilds_and_later_docs_are_not_blocked(monkeypatch):
    db, clk = FakeDB(), Clock()
    a = Ingestor(db)
    a.add(DOCS[0])
    cfg = _cfg(**WIN)
    monkeypatch.setattr(domain_registry, "load_config", lambda *x, **k: cfg)
    dar_runtime.reset_for_tests()
    rt = dar_runtime.DarRuntime()
    rt.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    old = db.sources[0]   # version 2 of the same canonical source, DIFFERENT text (old v1 span offsets no longer fit)
    db.sources.append({**old, "source_id": 50, "version": 2, "content": old["content"] + " appended revision text " * 40})
    a.n = 59
    a.add(DOCS[1])        # and a later, unrelated new document
    c = rt.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    ids = {d.get("db_source_id") for d in c._docs.values()}
    assert 50 in ids and 60 in ids and c.stats()["documents"] == 2
    assert rt.status() == "ok"


def test_refused_rebuild_is_stale_not_ok_and_health_surfaces_it(monkeypatch):
    db, clk = FakeDB(), Clock()
    a = Ingestor(db)
    a.add(DOCS[0])
    cfg = _cfg(**WIN)
    monkeypatch.setattr(domain_registry, "load_config", lambda *x, **k: cfg)
    dar_runtime.reset_for_tests()
    rt = dar_runtime.DarRuntime()
    rt.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    monkeypatch.setattr(DarCorpus, "restore", lambda self, snap: {"status": "chunking_mismatch"})
    a.n = 1
    a.add(DOCS[1])        # forces a delta; make that fail too so a rebuild is attempted
    monkeypatch.setattr(DarCorpus, "apply_delta", lambda self, d: {"status": "needs_full"})
    c = rt.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    assert c is not None and c.stats()["documents"] == 1          # old corpus kept
    assert rt.status() == "stale:rebuild_restore_chunking_mismatch"
    from tests import test_dar_health as th
    from vera import dar_health
    dar_health.reset_for_tests()
    monkeypatch.setattr(dar_runtime, "status", rt.status)
    r = dar_health.dar_readiness(th.ENV, th._factory(), force=True)
    assert r["dar"] == "degraded" and r["reason_code"].startswith("runtime_rebuild_restore")
    dar_health.reset_for_tests()


def test_snapshot_chunk_query_is_restricted_to_latest_version():
    db = FakeDB()
    Ingestor(db).add(DOCS[0])
    DarReader(db).load_snapshot(["full_text"])
    DarReader(db).load_delta({"source_id": 0, "span_id": 0, "domain_id": 0, "assoc_ts": None}, ["full_text"])
    q = [s for s, _ in db.sql_log if "FROM domain_chunks" in s or "FROM vera_vjay.domain_chunks" in s or "domain_chunks dk" in s]
    assert q and all("max(v.version)" in s for s in q)
