"""Section 3 atomicity: dar_store.persist() runs one plan in ONE transaction (DarWriter.atomic). Fake engine only."""
import logging

from sqlalchemy.exc import IntegrityError

from vera.dar_store import DarWriter, persist

URL = "postgresql://user:secret-pw@host.invalid:5432/db"


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
        self.conn.eng.events.append("sp_commit")

    def rollback(self):
        self.conn.eng.events.append("sp_rollback")


class _Conn:
    def __init__(self, eng):
        self.eng = eng

    def __enter__(self):
        self.eng.events.append("begin")
        return self

    def __exit__(self, et, ev, tb):
        self.eng.events.append("rollback" if et else "commit")
        return False

    def begin_nested(self):
        return _SP(self)

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.eng.executed.append(sql)
        if sql.lstrip().upper().startswith("INSERT"):
            self.eng.n_insert += 1
            if self.eng.fail_at == self.eng.n_insert:
                raise RuntimeError(f"db exploded {URL}")
            if self.eng.dup_at == self.eng.n_insert:
                raise IntegrityError("stmt", {}, Exception("dup"))
        return _Res(self.eng.handler(sql, dict(params or {})))


class Eng:
    def __init__(self, fail_at=None, dup_at=None):
        self.events, self.executed, self.n_insert = [], [], 0
        self.fail_at, self.dup_at = fail_at, dup_at

    def handler(self, sql, p):
        if "research_domains" in sql:
            return [(101,)]
        if "evidence_spans" in sql:
            return [(500 + p["a"],)]
        return [(1,)]

    def begin(self):
        return _Conn(self)

    connect = begin


def _plan():
    return {"status": "indexed", "reason": None, "source_key": "k1",
            "doc": {"db_source_id": 7, "db_canonical_source_id": 3},
            "domains": [{"domain_id": 1, "name": "Alpha", "name_norm": "alpha", "description": "d", "origin": "derived"}],
            "corpus_assoc": [{"domain_id": 1, "canonical_source_id": "k1", "asserted_by": "dar"}],
            "chunks": [{"span_id": "k1:0-5", "source_key": "k1", "start": 0, "end": 5, "text": "whole", "domains": [1]},
                       {"span_id": "k1:6-12", "source_key": "k1", "start": 6, "end": 12, "text": "second", "domains": [1]}],
            "promoted": []}


ZERO = {k: 0 for k in ("domains_written", "domains_existing", "spans_written", "spans_existing", "corpus_assoc_written",
                       "chunk_assoc_written", "skipped_placeholder", "unresolved", "promoted_skipped",
                       "missing_span_text")}


def test_success_commits_exactly_once():
    eng = Eng()
    ctx = {}
    r = persist(_plan(), DarWriter(eng), ctx=ctx)
    assert r["status"] == "persisted" and r["spans_written"] == 2 and r["chunk_assoc_written"] == 2
    assert eng.events.count("begin") == 1 and eng.events.count("commit") == 1 and "rollback" not in eng.events
    assert eng.n_insert == 6
    assert ctx["domains"] == {1: 101} and ctx["canonical"] == {"k1": 3} and set(ctx["spans"]) == {"k1:0-5", "k1:6-12"}


def test_mid_persist_failure_rolls_back_everything_and_leaves_ctx(caplog):
    caplog.set_level(logging.DEBUG)
    for n in range(1, 7):  # every INSERT position: domain, corpus_assoc, spans, chunk assocs
        eng = Eng(fail_at=n)
        ctx = {"domains": {}, "canonical": {}, "spans": {}}
        r = persist(_plan(), DarWriter(eng), ctx=ctx)
        assert r == {"status": "failed", "reason": "RuntimeError", "rolled_back": True, **ZERO}, (n, r)
        assert eng.events == ["begin", "sp_commit"] * 0 + [e for e in eng.events if e != "commit"]  # never committed
        assert "commit" not in eng.events and eng.events.count("rollback") == 1 and eng.events.count("begin") == 1
        assert ctx == {"domains": {}, "canonical": {}, "spans": {}}  # ctx not poisoned
    assert "RuntimeError" in caplog.text and URL not in caplog.text and "secret-pw" not in caplog.text


def test_failed_then_retry_with_same_ctx_succeeds():
    ctx = {}
    assert persist(_plan(), DarWriter(Eng(fail_at=4)), ctx=ctx)["status"] == "failed"
    assert persist(_plan(), DarWriter(Eng()), ctx=ctx)["status"] == "persisted"
    assert ctx["domains"] == {1: 101}


def test_failure_on_begin_is_also_failed_not_raised():
    class Bad(Eng):
        def begin(self):
            raise ConnectionError(URL)
    r = persist(_plan(), DarWriter(Bad()), ctx={})
    assert r["status"] == "failed" and r["reason"] == "ConnectionError" and r["rolled_back"] is True


def test_duplicate_inside_atomic_transaction_is_handled_by_savepoint():
    eng = Eng(dup_at=3)  # first span insert is a unique-violation
    ro = Eng()
    ro.handler = lambda sql, p: [(777,)]
    from vera.dar_store import DarReader
    r = persist(_plan(), DarWriter(eng, reader=DarReader(ro)), ctx={})
    assert r["status"] == "persisted" and r["spans_existing"] == 1 and r["spans_written"] == 1
    assert "sp_rollback" in eng.events and eng.events.count("begin") == 1
    assert eng.events.count("commit") == 1 and "rollback" not in eng.events  # the transaction was not aborted


def test_skipped_plans_open_no_transaction():
    eng = Eng()
    assert persist({"status": "off"}, DarWriter(eng))["status"] == "skipped" and eng.events == []
    bad = _plan()
    bad["doc"]["db_source_id"] = None
    assert persist(bad, DarWriter(eng))["reason"] == "no_db_ids" and eng.events == []


class _ConnWriterLike:
    """The loader's ConnWriter shape: no atomic(), exceptions propagate to the caller's single transaction."""

    def __init__(self, fail_at=None):
        self.calls, self.fail_at, self.reader = [], fail_at, None

    def _hit(self, name, ret):
        self.calls.append(name)
        if self.fail_at == len(self.calls):
            raise RuntimeError("x")
        return ret

    def write_domain(self, *a):
        return self._hit("domain", 101)

    def write_span(self, *a):
        return self._hit("span", 500 + len(self.calls))

    def write_corpus_assoc(self, *a):
        return self._hit("corpus", True)

    def write_chunk_assoc(self, *a):
        return self._hit("chunk", True)


def test_writer_without_atomic_is_unchanged_and_exceptions_propagate():
    w = _ConnWriterLike()
    assert persist(_plan(), w, ctx={})["status"] == "persisted" and len(w.calls) == 6
    try:
        persist(_plan(), _ConnWriterLike(fail_at=3), ctx={})
    except RuntimeError:
        pass
    else:
        raise AssertionError("a writer inside the caller's transaction must let the exception propagate")


def test_real_connwriter_shape_has_passthrough_atomic_and_propagates():
    """DarWriter subclass bound to a caller's connection (engine None): atomic() is a passthrough, no engine use."""
    from vera.dar_store import DarWriter as DW

    class CW(DW):
        def __init__(self):
            super().__init__(None)
            self.calls = 0

        def _insert_new(self, sql, params, returning=True):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("x")
            return 1

    with CW().atomic() as w:
        assert w is not None
    try:
        persist(_plan(), CW(), ctx={})
    except RuntimeError:
        return
    raise AssertionError("must propagate to the loader's transaction")
