"""vera/dar_store.py with a FAKE engine (no DB, no network, no env)."""
import re
from pathlib import Path

from sqlalchemy.exc import IntegrityError

from vera import dar_store
from vera.dar_store import DarReader, DarWriter, persist, smoke_roundtrip

SRC = Path(dar_store.__file__).read_text(encoding="utf-8")


class _Res:
    def __init__(self, rows):
        self.rows = rows

    def first(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class _SP:
    def __init__(self, conn):
        self.conn = conn

    def commit(self):
        self.conn.events.append("sp_commit")

    def rollback(self):
        self.conn.events.append("sp_rollback")


class _Conn:
    def __init__(self, eng):
        self.eng, self.events = eng, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def begin_nested(self):
        return _SP(self)

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.eng.calls.append((sql, dict(params or {})))
        return _Res(self.eng.handler(sql, dict(params or {})))


class FakeEngine:
    def __init__(self, handler=None):
        self.calls, self.handler = [], handler or (lambda s, p: [])

    def begin(self):
        return _Conn(self)

    connect = begin


def _integrity():
    return IntegrityError("stmt", {}, Exception("dup"))


def _plan(**kw):
    p = {"status": "indexed", "reason": None, "source_key": "k1",
         "doc": {"canonical_source_id": "k1", "url": "u", "title": "t", "db_source_id": 7, "db_canonical_source_id": 3},
         "domains": [{"domain_id": 1, "name": "Alpha", "name_norm": "alpha", "description": "d", "origin": "derived",
                      "created": True}],
         "corpus_assoc": [{"domain_id": 1, "canonical_source_id": "k1", "asserted_by": "dar"}],
         "chunks": [{"span_id": "k1:0-5", "source_key": "k1", "start": 0, "end": 5, "text": "whole", "domains": [1]},
                    {"span_id": "k1:6-12", "source_key": "k1", "start": 6, "end": 12, "text": "second", "domains": [1]}],
         "promoted": []}
    p.update(kw)
    return p


def _ok_handler(sql, p):
    if "research_domains" in sql and "INSERT" in sql:
        return [(101,)]
    if "evidence_spans" in sql and "INSERT" in sql:
        return [(500 + p["a"],)]
    return [(1,)]


def test_insert_returning_only_and_no_forbidden_sql():
    eng = FakeEngine(_ok_handler)
    r = persist(_plan(), DarWriter(eng))
    assert r["status"] == "persisted" and r["domains_written"] == 1 and r["spans_written"] == 2
    assert r["corpus_assoc_written"] == 1 and r["chunk_assoc_written"] == 2
    for sql, _ in eng.calls:
        assert sql.lstrip().upper().startswith("INSERT INTO"), sql
        assert "SELECT" not in sql.upper().replace("RETURNING", "")
    code = re.sub(r'""".*?"""', "", SRC, flags=re.S)
    for word in ("UPDATE ", "DELETE ", "TRUNCATE", "DROP ", "ALTER ", "CREATE TABLE", "GRANT "):
        assert word not in code.upper(), word


def test_no_db_ids_skips_and_writes_nothing():
    for bad in (None, "x"):
        eng = FakeEngine(_ok_handler)
        plan = _plan()
        plan["doc"]["db_source_id"] = bad
        r = persist(plan, DarWriter(eng))
        assert r == {**r, "status": "skipped", "reason": "no_db_ids"} and eng.calls == []
    plan = _plan()
    plan["doc"]["db_canonical_source_id"] = None
    assert persist(plan, DarWriter(FakeEngine()))["reason"] == "no_db_ids"


def test_existing_domain_and_span_resolved_via_reader():
    def wo(sql, p):
        if "INSERT" in sql and ("research_domains" in sql or "evidence_spans" in sql or "domain_" in sql):
            raise _integrity()
        return []

    def ro(sql, p):
        if "research_domains" in sql:
            assert p == {"n": "alpha"}
            return [(42,)]
        if "evidence_spans" in sql:
            return [(900 + p["a"],)]
        return []
    ro_eng = FakeEngine(ro)
    ctx = {}
    eng = FakeEngine(wo)
    r = persist(_plan(), DarWriter(eng, reader=DarReader(ro_eng)), ctx=ctx)
    assert r["domains_existing"] == 1 and r["domains_written"] == 0 and r["spans_existing"] == 2
    assert r["corpus_assoc_written"] == 0 and r["chunk_assoc_written"] == 0  # duplicates -> False, no raise
    assert ctx["domains"] == {1: 42} and ctx["spans"] == {"k1:0-5": 900, "k1:6-12": 906} and ctx["canonical"] == {"k1": 3}
    assert all("SELECT" in s.upper() for s, _ in ro_eng.calls)


def test_rollback_on_duplicate_savepoint():
    conns = []

    class E(FakeEngine):
        def begin(self):
            c = _Conn(self)
            conns.append(c)
            return c
    eng = E(lambda s, p: (_ for _ in ()).throw(_integrity()))
    assert DarWriter(eng).write_domain("A", "a", "", "derived") is None
    assert conns[0].events == ["sp_rollback"]
    assert DarWriter(eng).write_corpus_assoc(1, 2, "x") is False


def test_placeholder_span_skipped_with_span_text_param():
    plan = _plan()
    plan["chunks"][0]["text"] = "whole abstract"
    eng = FakeEngine(_ok_handler)
    r = persist(plan, DarWriter(eng), span_text="whole abstract")
    assert r["skipped_placeholder"] == 1 and r["spans_written"] == 1 and r["chunk_assoc_written"] == 1
    assert not any(p.get("t") == "whole abstract" for _, p in eng.calls)
    # without span_text nothing is skipped
    r2 = persist(plan, DarWriter(FakeEngine(_ok_handler)))
    assert r2["skipped_placeholder"] == 0 and r2["spans_written"] == 2
    # start != 0 with same text is not a placeholder
    plan["chunks"][0]["start"] = 1
    assert persist(plan, DarWriter(FakeEngine(_ok_handler)), span_text="whole abstract")["skipped_placeholder"] == 0


def test_ctx_mapping_and_promoted():
    plan = _plan()
    ctx = {}
    persist(plan, DarWriter(FakeEngine(_ok_handler)), ctx=ctx)
    assert ctx["domains"] == {1: 101} and set(ctx["spans"]) == {"k1:0-5", "k1:6-12"} and ctx["canonical"]["k1"] == 3
    plan2 = _plan(source_key="k2", doc={"canonical_source_id": "k2", "url": "u2", "title": "t", "db_source_id": 8,
                                        "db_canonical_source_id": 4}, corpus_assoc=[], chunks=[], domains=[],
                  promoted=[{"source_key": "k1", "corpus_assoc": [{"domain_id": 1, "canonical_source_id": "k1",
                                                                   "asserted_by": "dar"}],
                             "chunk_assoc": [{"span_id": "k1:0-5", "domain_id": 1, "asserted_by": "dar"}]},
                            {"source_key": "gone", "corpus_assoc": [{"domain_id": 1, "canonical_source_id": "g",
                                                                     "asserted_by": "dar"}], "chunk_assoc": []}])
    eng = FakeEngine(_ok_handler)
    r = persist(plan2, DarWriter(eng), ctx=ctx)
    assert r["corpus_assoc_written"] == 1 and r["chunk_assoc_written"] == 1 and r["promoted_skipped"] == 1
    assert eng.calls[0][1]["c"] == 3
    # no ctx at all: promoted reported skipped, never raised
    r3 = persist(plan2, DarWriter(FakeEngine(_ok_handler)))
    assert r3["promoted_skipped"] == 3 and r3["status"] == "partial" and r3["unresolved"] >= 3
    assert "unresolved=" in r3["reason"]


def test_load_snapshot_parameterised_and_shape():
    def ro(sql, p):
        if "FROM s1.research_domains ORDER" in sql:
            return [(1, "Alpha", "alpha", "d", "derived")]
        if "DISTINCT ON" in sql:
            return [("k1", 7, 3, "full text", "u", "10.1/x", "T", "2024-05-01", "journal", "2024-06-01", "full_text")]
        if "domain_corpus" in sql:
            return [("alpha", "k1", "dar")]
        if "domain_chunks" in sql:
            return [("alpha", "k1", 0, 4, "dar")]
        return []
    eng = FakeEngine(ro)
    snap = DarReader(eng, schema="s1").load_snapshot(["abstract_metadata"], span_text="whole abstract")
    assert set(snap) == {"domains", "documents", "corpus_assoc", "chunk_assoc", "cursor"}
    assert snap["cursor"] == {"source_id": 7, "span_id": None, "domain_id": 1, "assoc_ts": None}
    assert snap["domains"] == [{"name": "Alpha", "name_norm": "alpha", "description": "d", "origin": "derived"}]
    d = snap["documents"][0]
    assert d["source_key"] == "k1" and d["text"] == "full text" and d["year"] == 2024 and d["db_source_id"] == 7
    assert d["db_canonical_source_id"] == 3 and d["stored_text_kind"] == "full_text"
    assert snap["corpus_assoc"] == [{"name_norm": "alpha", "source_key": "k1", "asserted_by": "dar"}]
    assert snap["chunk_assoc"] == [{"name_norm": "alpha", "source_key": "k1", "start": 0, "end": 4, "asserted_by": "dar"}]
    doc_sql, doc_p = next(c for c in eng.calls if "DISTINCT ON" in c[0])
    assert "= ANY(:kinds)" in doc_sql and "IS NOT NULL" in doc_sql and doc_p == {"kinds": ["abstract_metadata"]}
    ch_sql, ch_p = next(c for c in eng.calls if "domain_chunks" in c[0])
    assert ch_p == {"span_text": "whole abstract"} and "whole abstract" not in ch_sql and "abstract_metadata" not in doc_sql
    assert "s1.research_domains" in doc_sql or all("s1." in c[0] for c in eng.calls)


def test_smoke_roundtrip():
    def ro(sql, p):
        if "research_domains" in sql:
            return [(101, "Alpha", "alpha", "d", "derived")]
        if "domain_corpus" in sql:
            return [(101, 3, "dar")]
        if "domain_chunks" in sql:
            return [(101, 500, "dar")]
        return []
    out = smoke_roundtrip(FakeEngine(_ok_handler), FakeEngine(ro), _plan())
    assert out["persist"]["status"] == "persisted"
    assert out["domains"][0]["name_norm"] == "alpha" and out["corpus_assocs"][0]["canonical_source_id"] == 3
    assert out["chunk_assocs"][0]["span_id"] == 500


def test_static_no_env_url_or_engine_creation():
    for bad in ("os.environ", "getenv", "create_engine", "import os", "http://", "https://", "postgresql://", "URL("):
        assert bad not in SRC, bad
    assert "sqlalchemy" not in SRC.split("class DarReader")[0].replace('"""', "").split("from __future__")[1]


def _promoted_plan():
    return _plan(source_key="k2", doc={"canonical_source_id": "k2", "url": "u2", "title": "t", "db_source_id": 8,
                                       "db_canonical_source_id": 4}, corpus_assoc=[], chunks=[], domains=[],
                 promoted=[{"source_key": "k1",
                            "corpus_assoc": [{"domain_id": 1, "name_norm": "alpha", "asserted_by": "dar"}],
                            "chunk_assoc": [{"span_id": "k1:0-5", "domain_id": 1, "name_norm": "alpha",
                                             "asserted_by": "dar"},
                                            {"span_id": "k1:6-12", "domain_id": 1, "name_norm": "alpha",
                                             "asserted_by": "dar"}]}])


def test_promoted_resolved_with_fresh_ctx_via_reader():
    persist(_plan(), DarWriter(FakeEngine(_ok_handler)), ctx={})  # doc 1 with ctx A

    def ro(sql, p):
        if "canonical_sources" in sql:
            assert p == {"u": "k1"}
            return [(3,)]
        if "FROM s1.sources" in sql or "{S}.sources" in sql or "sources WHERE" in sql:
            assert p == {"c": 3}
            return [(7,)]
        if "evidence_spans" in sql:
            assert p["s"] == 7
            return [(500 + p["a"],)]
        if "research_domains" in sql:
            return [(101,)]
        return []
    ro_eng, wo_eng = FakeEngine(ro), FakeEngine(_ok_handler)
    ctx_b = {}
    r = persist(_promoted_plan(), DarWriter(wo_eng, reader=DarReader(ro_eng, schema="s1")), ctx=ctx_b)
    assert r["status"] == "persisted" and r["promoted_skipped"] == 0 and r["unresolved"] == 0
    assert r["corpus_assoc_written"] == 1 and r["chunk_assoc_written"] == 2
    assert ctx_b["canonical"]["k1"] == 3 and ctx_b["spans"]["k1:6-12"] == 506
    assert all("SELECT" in s.upper() for s, _ in ro_eng.calls)
    assert all("s1." in s for s, _ in ro_eng.calls)


def test_promoted_without_reader_is_partial_not_silent():
    r = persist(_promoted_plan(), DarWriter(FakeEngine(_ok_handler)), ctx={})
    assert r["status"] == "partial" and r["unresolved"] > 0 and "unresolved=" in r["reason"]
    # reader that raises: still no raise
    def boom(sql, p):
        raise RuntimeError("down")
    r2 = persist(_promoted_plan(), DarWriter(FakeEngine(_ok_handler), reader=DarReader(FakeEngine(boom))), ctx={})
    assert r2["status"] == "partial" and r2["unresolved"] == 3 and r2["chunk_assoc_written"] == 0


def _promo9():
    pl = _promoted_plan()
    pl["promoted"][0]["db_source_id"] = 77
    pl["promoted"][0]["db_canonical_source_id"] = 3
    pl["promoted"][0]["corpus_assoc"] = []
    return pl


def _ro_nospan(seen):
    def ro(sql, p):
        if "canonical_sources" in sql:
            return [(3,)]
        if "sources WHERE" in sql:
            return [(7,)]  # reader's "latest" id, must not be used
        if "evidence_spans" in sql:
            seen.append(p["s"])
            return []
        if "research_domains" in sql:
            return [(101,)]
        return []
    return ro


def test_promoted_uses_plan_db_source_id_not_reader_latest():
    seen = []
    pl = _promo9()
    for a in pl["promoted"][0]["chunk_assoc"]:
        a["text"] = "txt"
    eng = FakeEngine(_ok_handler)
    r = persist(pl, DarWriter(eng, reader=DarReader(FakeEngine(_ro_nospan(seen)))), ctx={})
    assert seen and set(seen) == {77}
    ins = [p for s, p in eng.calls if "evidence_spans" in s]
    assert ins and all(p["s"] == 77 for p in ins) and r["spans_written"] == 2


def test_missing_span_without_text_is_partial():
    seen = []
    r = persist(_promo9(), DarWriter(FakeEngine(_ok_handler), reader=DarReader(FakeEngine(_ro_nospan(seen)))), ctx={})
    assert r["status"] == "partial" and r["missing_span_text"] == 2 and "missing_span_text=2" in r["reason"]
    assert r["chunk_assoc_written"] == 0


def test_missing_span_with_chunk_text_callable_inserts_and_applies_placeholder():
    seen = []
    eng = FakeEngine(_ok_handler)
    got = []

    def ct(sid):
        got.append(sid)
        return "body " + sid
    r = persist(_promo9(), DarWriter(eng, reader=DarReader(FakeEngine(_ro_nospan(seen)))), ctx={}, chunk_text=ct,
                span_text="whole abstract")
    assert r["status"] == "persisted" and r["spans_written"] == 2 and r["chunk_assoc_written"] == 2
    assert got == ["k1:0-5", "k1:6-12"]
    ins = [p for s, p in eng.calls if "INSERT INTO" in s and "evidence_spans" in s]
    assert [p["t"] for p in ins] == ["body k1:0-5", "body k1:6-12"] and all(p["s"] == 77 for p in ins)
    # placeholder text (start 0) is not inserted
    eng2 = FakeEngine(_ok_handler)
    r2 = persist(_promo9(), DarWriter(eng2, reader=DarReader(FakeEngine(_ro_nospan([])))), ctx={},
                 chunk_text=lambda sid: "whole abstract" if sid == "k1:0-5" else "other", span_text="whole abstract")
    assert r2["skipped_placeholder"] == 1 and r2["spans_written"] == 1
    assert not any(p.get("t") == "whole abstract" for _, p in eng2.calls)


def _snap_eng(lic_row):
    def ro(sql, p):
        if "DISTINCT ON" in sql:
            return [("k1", 7, 3, "full text", "u", None, "T", "2024", "journal", None, "full_text") + lic_row]
        return []
    return FakeEngine(ro)


def test_load_snapshot_provenance_keys():
    lic = {"spdx": "CC-BY-4.0"}
    eng = _snap_eng((lic, "accept"))
    d = DarReader(eng, schema="s1").load_snapshot([])["documents"][0]
    assert d["canonical_source_id"] == 3 and d["source_id"] == 7 and d["db_source_id"] == 7
    assert d["licence"] == lic and d["licence_decision"] == "accept" and d["admission_passed"] is True
    sql = next(c[0] for c in eng.calls if "DISTINCT ON" in c[0])
    assert "provenance->'licence'" in sql and "provenance->'licence_decision'" in sql
    d2 = DarReader(_snap_eng((None, None)), schema="s1").load_snapshot([])["documents"][0]
    assert d2["licence"] is None and d2["licence_decision"] is None and d2["admission_passed"] is True


def test_corpus_version():
    state = {"row": (10, 4, 7)}
    eng = FakeEngine(lambda s, p: [state["row"]])
    rd = DarReader(eng, schema="s1")
    assert rd.corpus_version() == "10:4:7"
    state["row"] = (11, 4, 7)
    assert rd.corpus_version() == "11:4:7"
    sql, params = eng.calls[0]
    assert sql.lstrip().upper().startswith("SELECT") and params == {} and "s1.evidence_spans" in sql
    assert "s1.domain_chunks" in sql and "s1.sources" in sql
    assert DarReader(FakeEngine()).corpus_version() == "0:0:0"
