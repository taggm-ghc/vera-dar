"""Incremental freshness (P0-D): DarReader.load_delta cursor semantics + two independent DarRuntime instances over one
shared in-memory fake DB. No DB, no network, no env reads."""
import copy
import json
import logging
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from vera import answer_contract as ac
from vera import dar_runtime, domain_registry
from vera.dar_corpus import DarCorpus
from vera.dar_store import DarReader, DarWriter, persist

FIX = Path(__file__).resolve().parent / "fixtures" / "dar_acceptance"
DOCS = json.loads((FIX / "demo9.json").read_text(encoding="utf-8"))
URL = "postgresql://user:secret-pw@host.invalid:5432/db"
ENV = {"VERA_DB_URL_RO": URL}
SCHEMA = "vera_vjay"
CFG_REAL = domain_registry.load_config()


# ------------------------------------------------------------------ shared in-memory fake DB (engine)
class _Res:
    def __init__(self, rows):
        self.rows = rows

    def first(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class _SP:
    def commit(self):
        pass

    def rollback(self):
        pass


class _Conn:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def begin_nested(self):
        return _SP()

    def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split()).replace(SCHEMA + ".", "")
        p = dict(params or {})
        self.db.sql_log.append((sql, p))
        return _Res(self.db.run(sql, p))


class FakeDB:
    """Tables + the statements dar_store.py issues. created_at = ISO string from a monotonically increasing tick."""

    def __init__(self):
        self.canonical, self.sources = {}, []
        self.domains, self.spans, self.dcorp, self.dchunks = [], [], [], []
        self.tick, self.sql_log = 0, []

    def connect(self):
        return _Conn(self)

    begin = connect

    def dispose(self):
        pass

    def now(self):
        self.tick += 1
        return f"2026-10-19T00:00:{self.tick:02d}+00:00"

    def add_source(self, n, d):
        self.canonical[n] = d["url"]
        self.sources.append({"source_id": n, "canonical_source_id": n, "version": 1, "content": d["content"],
                             "prov": {"url": d["url"], "doi": d["doi"], "title": d["title"], "published": str(d["year"]),
                                      "source_type": d["source_type"], "retrieved_at": "2026-10-19T00:00:00+00:00",
                                      "stored_text_kind": d["stored_text_kind"]}})

    def _dup(self):
        return IntegrityError("stmt", {}, Exception("duplicate key"))

    def run(self, sql, p):  # noqa: C901
        if sql.startswith("INSERT INTO research_domains"):
            if any(d["name_norm"] == p["nn"] for d in self.domains):
                raise self._dup()
            self.domains.append({"domain_id": len(self.domains) + 1, "name": p["n"], "name_norm": p["nn"],
                                 "description": p["d"], "origin": p["o"]})
            return [(len(self.domains),)]
        if sql.startswith("INSERT INTO evidence_spans"):
            if any((s["source_id"], s["start"], s["end"]) == (p["s"], p["a"], p["b"]) for s in self.spans):
                raise self._dup()
            self.spans.append({"span_id": len(self.spans) + 1, "source_id": p["s"], "text": p["t"],
                               "start": p["a"], "end": p["b"]})
            return [(len(self.spans),)]
        if sql.startswith("INSERT INTO domain_corpus"):
            if any((r["d"], r["c"]) == (p["d"], p["c"]) for r in self.dcorp):
                raise self._dup()
            self.dcorp.append({"d": p["d"], "c": p["c"], "by": p["a"], "ts": self.now()})
            return [(1,)]
        if sql.startswith("INSERT INTO domain_chunks"):
            if any((r["d"], r["s"]) == (p["d"], p["s"]) for r in self.dchunks):
                raise self._dup()
            self.dchunks.append({"d": p["d"], "s": p["s"], "by": p["a"], "ts": self.now()})
            return [(1,)]
        assert sql.startswith("SELECT"), sql
        if "coalesce(max(span_id)" in sql:
            return [(max([s["span_id"] for s in self.spans] or [0]), len(self.dchunks),
                     max([s["source_id"] for s in self.sources] or [0]), len(self.dcorp),
                     max([d["domain_id"] for d in self.domains] or [0]),
                     len(self.sources), len(self.spans), len(self.domains))]
        if sql.startswith("SELECT domain_id FROM research_domains WHERE name_norm"):
            return [(d["domain_id"],) for d in self.domains if d["name_norm"] == p["n"]]
        if sql.startswith("SELECT span_id FROM evidence_spans WHERE source_id"):
            return [(s["span_id"],) for s in self.spans if (s["source_id"], s["start"], s["end"]) == (p["s"], p["a"], p["b"])]
        if sql.startswith("SELECT canonical_source_id FROM canonical_sources WHERE canonical_url"):
            return [(c,) for c, u in self.canonical.items() if u == p["u"]]
        if sql.startswith("SELECT source_id FROM sources WHERE canonical_source_id"):
            return [(s["source_id"],) for s in self.sources if s["canonical_source_id"] == p["c"]][:1]
        if sql.startswith("SELECT domain_id, name, name_norm, description, origin FROM research_domains"):
            lo = p.get("since_domain_id", 0)
            return [(d["domain_id"], d["name"], d["name_norm"], d["description"], d["origin"])
                    for d in self.domains if d["domain_id"] > lo]
        if "SELECT DISTINCT ON (cs.canonical_source_id)" in sql:
            out = []
            for s in sorted(self.sources, key=lambda s: (s["canonical_source_id"], -s["version"])):
                kind = s["prov"].get("stored_text_kind")
                if kind is None or kind in p["kinds"] or s["source_id"] <= p.get("since_source_id", 0):
                    continue
                if any(o[2] == s["canonical_source_id"] for o in out):
                    continue
                pr = s["prov"]
                out.append((self.canonical[s["canonical_source_id"]], s["source_id"], s["canonical_source_id"],
                            s["content"], pr["url"], pr["doi"], pr["title"], pr["published"], pr["source_type"],
                            pr["retrieved_at"], kind))
            return out
        lo = p.get("since_source_id_lo")  # id-window predicate (EXISTS ... w.source_id > lo): docs inside the window
        in_win = lambda cid: lo is not None and any(  # noqa: E731
            x["canonical_source_id"] == cid and x["source_id"] > lo for x in self.sources)
        sp_src = {x["span_id"]: x["source_id"] for x in self.spans}
        newer = lambda r: ("since_assoc_ts" not in p) or r["ts"] > p["since_assoc_ts"] or in_win(  # noqa: E731
            r["c"] if "c" in r else next(x["canonical_source_id"] for x in self.sources if x["source_id"] == sp_src[r["s"]]))
        nm = {d["domain_id"]: d["name_norm"] for d in self.domains}
        if sql.startswith("SELECT d.name_norm, cs.canonical_url, dc.asserted_by"):
            return [(nm[r["d"]], self.canonical[r["c"]], r["by"], r["ts"])
                    for r in sorted(self.dcorp, key=lambda r: (r["d"], r["c"])) if newer(r)]
        if sql.startswith("SELECT d.name_norm, cs.canonical_url, es.start_index"):
            sp = {s["span_id"]: s for s in self.spans}
            sid = {s["source_id"]: s["canonical_source_id"] for s in self.sources}
            out = []
            latest_only = "max(v.version)" in sql  # emulate the version restriction of the SQL join
            top = {}
            for x in self.sources:
                top[x["canonical_source_id"]] = max(top.get(x["canonical_source_id"], 0), x["version"])
            src = {x["source_id"]: x for x in self.sources}
            for r in sorted(self.dchunks, key=lambda r: (r["d"], r["s"])):
                s = sp[r["s"]]
                if latest_only and src[s["source_id"]]["version"] != top[src[s["source_id"]]["canonical_source_id"]]:
                    continue
                if not newer(r) or (p["span_text"] is not None and s["start"] == 0 and s["text"] == p["span_text"]):
                    continue
                out.append((nm[r["d"]], self.canonical[sid[s["source_id"]]], s["start"], s["end"], r["by"], r["ts"],
                            s["span_id"]))
            return out
        raise AssertionError("FakeDB: unsupported SQL: " + sql)


class Ingestor:
    """Process A: ingest + persist documents into the shared fake DB."""

    def __init__(self, db):
        self.db, self.ctx, self.n = db, {}, 0
        self.dar = DarCorpus(copy.deepcopy(CFG_REAL))
        from vera import dar_runtime as drt
        self.span_text = drt._span_text(ac.CHAIN_CONFIG_PATH)

    def add(self, d):
        self.n += 1
        self.db.add_source(self.n, d)
        plan = self.dar.ingest_document({
            "source_key": d["url"], "url": d["url"], "doi": d["doi"], "title": d["title"], "text": d["content"],
            "year": d["year"], "published": str(d["year"]), "source_type": d["source_type"],
            "retrieved_at": "2026-10-19T00:00:00+00:00", "stored_text_kind": d["stored_text_kind"],
            "admission_passed": d["admission_passed"], "db_source_id": self.n, "db_canonical_source_id": self.n})
        assert plan["status"] == "indexed", plan
        res = persist(plan, DarWriter(self.db, reader=DarReader(self.db), span_text=self.span_text),
                      span_text=self.span_text, ctx=self.ctx)
        assert res["status"] == "persisted", res
        return plan


# ------------------------------------------------------------------ runtime plumbing
def _cfg(**rt):
    cfg = copy.deepcopy(CFG_REAL)
    cfg["enabled"] = True
    cfg.setdefault("corpus", {})["stored_branch_enabled"] = True
    cfg["runtime"] = {"db_url_env": "VERA_DB_URL_RO", "connect_timeout_s": 5, "statement_timeout_ms": 10000,
                      "failure_retry_ttl_s": 60, "reload_ttl_s": 300, "version_check_s": 0, **rt}
    return cfg


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


class Calls:
    snapshots = 0
    deltas = []


@pytest.fixture
def spy(monkeypatch):
    import vera.dar_store as ds
    Calls.snapshots, Calls.deltas = 0, []

    class SpyReader(DarReader):
        def load_snapshot(self, *a, **k):
            Calls.snapshots += 1
            return super().load_snapshot(*a, **k)

        def load_delta(self, since, *a, **k):
            out = super().load_delta(since, *a, **k)
            Calls.deltas.append((copy.deepcopy(since), out))
            return out

    monkeypatch.setattr(ds, "DarReader", SpyReader)
    cfg = _cfg()
    monkeypatch.setattr(domain_registry, "load_config", lambda *a, **k: cfg)
    dar_runtime.reset_for_tests()
    yield
    dar_runtime.reset_for_tests()


def _factory(db):
    return lambda url, **kw: db


# ------------------------------------------------------------------ end-to-end: A ingests, B picks it up incrementally
def test_second_runtime_gets_new_document_via_one_delta(spy):
    db, clk = FakeDB(), Clock()
    a = Ingestor(db)
    for d in DOCS[:3]:
        a.add(d)
    rt_b = dar_runtime.DarRuntime()
    c_old = rt_b.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    assert set(c_old._docs) == {d["url"] for d in DOCS[:3]} and Calls.snapshots == 1 and Calls.deltas == []
    assert DOCS[3]["url"] not in c_old._docs
    assert rt_b.get_corpus(env=ENV, engine_factory=_factory(db), now=clk) is c_old  # no change: no delta
    assert Calls.deltas == [] and Calls.snapshots == 1
    a.add(DOCS[3])  # process A commits a new document
    c_new = rt_b.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    assert DOCS[3]["url"] in c_new._docs and set(c_new._docs) == {d["url"] for d in DOCS[:4]}
    assert Calls.snapshots == 1 and len(Calls.deltas) == 1  # NOT a full reload; exactly one delta
    since, delta = Calls.deltas[0]
    assert [d["source_key"] for d in delta["documents"]] == [DOCS[3]["url"]]  # only the new row was requested
    assert since["source_id"] == 3
    assert {a_["source_key"] for a_ in delta["corpus_assoc"]} <= {DOCS[3]["url"]}
    assert c_new.retrieve(DOCS[3]["title"])  # retrievable from the incrementally updated corpus
    n_sql = len(db.sql_log)
    assert rt_b.get_corpus(env=ENV, engine_factory=_factory(db), now=clk) is c_new
    assert len(Calls.deltas) == 1 and Calls.snapshots == 1
    assert not any("created_at" in s or "since_" in json.dumps(p) for s, p in db.sql_log[n_sql:])  # version check only


def test_instances_are_independent(spy):
    db, clk = FakeDB(), Clock()
    Ingestor(db).add(DOCS[0])
    r1, r2 = dar_runtime.DarRuntime(), dar_runtime.DarRuntime()
    c1 = r1.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    assert r2.status().startswith("unavailable") and r2._state.get("corpus") is None
    c2 = r2.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    assert c1 is not c2 and r1._lock is not r2._lock
    r1.reset_for_tests()
    assert r2._state.get("corpus") is c2


def test_late_promotion_row_for_old_document_arrives_in_delta(spy):
    db, clk = FakeDB(), Clock()
    a = Ingestor(db)
    a.add(DOCS[0])
    rt = dar_runtime.DarRuntime()
    rt.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    # a later secondary label on the OLD document is a new association row
    db.domains.append({"domain_id": len(db.domains) + 1, "name": "Late", "name_norm": "late", "description": "",
                       "origin": "derived"})
    db.dcorp.append({"d": len(db.domains), "c": 1, "by": "dar", "ts": db.now()})
    rt.get_corpus(env=ENV, engine_factory=_factory(db), now=clk)
    _, delta = Calls.deltas[-1]
    assert delta["documents"] == [] and [x["name_norm"] for x in delta["domains"]] == ["late"]
    assert [(x["name_norm"], x["source_key"]) for x in delta["corpus_assoc"]] == [("late", DOCS[0]["url"])]


# ------------------------------------------------------------------ cursor semantics of load_delta
def test_load_delta_sql_uses_bound_since_params_and_first_load_has_none():
    db = FakeDB()
    a = Ingestor(db)
    a.add(DOCS[0])
    snap = DarReader(db).load_snapshot([], span_text=a.span_text)
    cur = snap["cursor"]
    assert cur["source_id"] == 1 and cur["domain_id"] == len(db.domains) and cur["span_id"] >= 1
    assert isinstance(cur["assoc_ts"], str)
    assert not any("since_" in k for _, p in db.sql_log for k in p)  # first (snapshot) load carries no 'since'
    db.sql_log.clear()
    out = DarReader(db).load_delta(cur, [], span_text=a.span_text)
    assert out["documents"] == [] and out["corpus_assoc"] == [] and out["chunk_assoc"] == [] and out["domains"] == []
    assert out["cursor"] == cur  # nothing new: cursor unchanged
    sqls = {s: p for s, p in db.sql_log}
    doc_sql = next(s for s in sqls if "DISTINCT ON" in s)
    assert "s.source_id > :since_source_id" in doc_sql and sqls[doc_sql]["since_source_id"] == 1
    dom_sql = next(s for s in sqls if "FROM research_domains" in s)
    assert "domain_id > :since_domain_id" in dom_sql
    for s, p in db.sql_log:
        if "domain_corpus" in s or "domain_chunks" in s:
            assert "created_at > CAST(:since_assoc_ts AS timestamptz)" in s and p["since_assoc_ts"] == cur["assoc_ts"]
        assert s.lstrip().upper().startswith("SELECT")
    a.add(DOCS[1])
    out2 = DarReader(db).load_delta(cur, [], span_text=a.span_text)
    assert [d["source_key"] for d in out2["documents"]] == [DOCS[1]["url"]]
    assert out2["cursor"]["source_id"] == 2 and out2["cursor"]["assoc_ts"] > cur["assoc_ts"]


def test_load_delta_empty_since_and_none_ts_has_no_ts_clause():
    db = FakeDB()
    Ingestor(db).add(DOCS[0])
    out = DarReader(db).load_delta({}, [])
    assert len(out["documents"]) == 1 and out["cursor"]["source_id"] == 1
    assert not any("since_assoc_ts" in p for _, p in db.sql_log)


# ------------------------------------------------------------------ fallback behaviour
class StubCorpus:
    """Minimal corpus: restore() ok; apply_delta behaviour configurable."""
    mode = "applied"
    applied = []

    def __init__(self, cfg=None):
        self.n = 0

    def restore(self, snap):
        return {"status": "restored"}

    def apply_delta(self, delta):
        if StubCorpus.mode == "raise":
            raise RuntimeError(f"boom {URL}")
        StubCorpus.applied.append(delta)
        return {"status": StubCorpus.mode}


class StubReader:
    version, snapshots, deltas, delta_boom = "v1", 0, 0, False

    def __init__(self, engine):
        pass

    def corpus_version(self):
        return StubReader.version

    def load_snapshot(self, kinds, span_text=None, **kw):
        StubReader.snapshots += 1
        return {"cursor": {"source_id": StubReader.snapshots, "span_id": 0, "domain_id": 0, "assoc_ts": None}}

    def load_delta(self, since, kinds=None, span_text=None):
        StubReader.deltas += 1
        if StubReader.delta_boom:
            raise RuntimeError("delta failed")
        return {"documents": [], "domains": [], "corpus_assoc": [], "chunk_assoc": [],
                "cursor": {"source_id": 99, "span_id": 0, "domain_id": 0, "assoc_ts": "t"}}


class _Eng:
    def dispose(self):
        pass


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


def _get(rt, clk):
    return rt.get_corpus(env=ENV, engine_factory=lambda u, **k: _Eng(), now=clk)


def test_applied_delta_keeps_corpus_and_advances_cursor(stub):
    clk = Clock()
    c1 = _get(stub, clk)
    StubReader.version = "v2"
    assert _get(stub, clk) is c1
    assert StubReader.snapshots == 1 and StubReader.deltas == 1 and len(StubCorpus.applied) == 1
    assert stub._state["cursor"]["source_id"] == 99 and stub._state["version"] == "v2"
    assert _get(stub, clk) is c1 and StubReader.deltas == 1  # no new version: no delta


@pytest.mark.parametrize("mode", ["needs_full", "raise", "off"])
def test_apply_delta_failure_falls_back_to_full_rebuild(stub, mode, caplog):
    caplog.set_level(logging.DEBUG)
    clk = Clock()
    c1 = _get(stub, clk)
    StubCorpus.mode = mode
    StubReader.version = "v2"
    c2 = _get(stub, clk)
    assert c2 is not c1 and StubReader.snapshots == 2 and StubReader.deltas == 1  # swapped for a full rebuild
    assert stub._state["cursor"]["source_id"] == 2 and stub._state["version"] == "v2"
    assert URL not in caplog.text and "secret-pw" not in caplog.text


def test_delta_load_failure_falls_back_to_full_rebuild(stub):
    clk = Clock()
    c1 = _get(stub, clk)
    StubReader.delta_boom, StubReader.version = True, "v2"
    c2 = _get(stub, clk)
    assert c2 is not c1 and StubReader.snapshots == 2 and StubCorpus.applied == []


def test_failed_fallback_keeps_old_corpus(stub, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    clk = Clock()
    c1 = _get(stub, clk)
    StubCorpus.mode, StubReader.version, StubReader.delta_boom = "needs_full", "v2", False

    def boom(self, kinds, span_text=None, **kw):
        raise RuntimeError(f"connect failed {URL}")
    monkeypatch.setattr(StubReader, "load_snapshot", boom)
    assert _get(stub, clk) is c1  # old corpus kept, never raises
    assert stub.status() == "stale:rebuild_RuntimeError"  # D-053 v3: a refused refresh is surfaced, never "ok"
    assert "RuntimeError" in caplog.text and URL not in caplog.text


def test_no_cursor_means_full_rebuild_path(stub, monkeypatch):
    clk = Clock()

    def nocur(self, kinds, span_text=None, **kw):
        StubReader.snapshots += 1
        return {}
    monkeypatch.setattr(StubReader, "load_snapshot", nocur)
    c1 = _get(stub, clk)
    StubReader.version = "v2"
    c2 = _get(stub, clk)
    assert c2 is not c1 and StubReader.deltas == 0 and StubReader.snapshots == 2


def test_ttl_backstop_is_a_full_rebuild(stub):
    clk = Clock()
    c1 = _get(stub, clk)
    clk.t += 301
    c2 = _get(stub, clk)
    assert c2 is not c1 and StubReader.snapshots == 2 and StubReader.deltas == 0


def test_module_functions_delegate_to_default_instance(stub, monkeypatch):
    dar_runtime.reset_for_tests()
    c = dar_runtime.get_corpus(env=ENV, engine_factory=lambda u, **k: _Eng(), now=Clock())
    assert c is dar_runtime._DEFAULT._state["corpus"] and dar_runtime.status() == "ok"
    assert dar_runtime.refresh(env=ENV, engine_factory=lambda u, **k: _Eng()) is not c
    dar_runtime.reset_for_tests()
    assert dar_runtime._DEFAULT._state == {}
