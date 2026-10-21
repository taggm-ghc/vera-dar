"""DEV-A: real-PostgreSQL integration of the full-document write/read path (M1 writer, fulldoc_db reader, doc pins).

SKIPS unless VERA_PG_REHEARSAL_DSN names an approved, NON-production rehearsal database that already carries the
vera_vjay schema (db/migrations). It never creates, drops or alters schema, never starts a server, and every case that
writes leaves nothing behind: fault cases never reach COMMIT, the other cases run inside one outer transaction that is
rolled back. Optional: VERA_PG_REHEARSAL_ALLOW_COMMIT=1 additionally commits one ingest and reads it back through a
second connection (only for a database that may be discarded afterwards; the sources table is immutable by trigger).

Version note (UNVERIFIED until run): the local binaries at /usr/lib/postgresql/16/bin are PostgreSQL 16; production is
PostgreSQL 18.6. A pass here proves transaction/rollback/jsonb/ANY() behaviour on the server major actually used (recorded
in the test report as pg_server_version_num); it does NOT prove PG18-specific behaviour. Rerun on PG18 before relying on it.

Under tests/vera_test_boundary.py every non-SQLite connection is refused (owner decision 2026-10-19) and the sanctioned
runner passes no VERA_* variable, so this module is expected to SKIP in the runner. Running it needs a separately approved
rehearsal lane (not added here).
"""
import contextlib
import copy
import hashlib
import ipaddress
import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

from vera import fulldoc_corpus as fc, fulldoc_db, fulltext_loader

DSN = (os.environ.get("VERA_PG_REHEARSAL_DSN") or "").strip()
ALLOW_COMMIT = os.environ.get("VERA_PG_REHEARSAL_ALLOW_COMMIT") == "1"
REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "ingest_fulldoc_chunks.py"
REAL_BUNDLE = REPO.parent / "work" / "fd-1020" / "bundle"
EXPECT = (9, 110)
TABLES = ["questions", "runs", "candidates", "canonical_sources", "sources", "evidence_spans"]
SCHEMA = "vera_vjay"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
NAME_OK = re.compile(r"[a-z0-9_]*rehears[a-z0-9_]*")
NAME_DENY = re.compile(r"prod|live|render|main|public_|^postgres$|^vera$|^vera_vjay$")

pytestmark = pytest.mark.vera_sockets
NO_DB = "UNVERIFIED: no rehearsal DB (VERA_PG_REHEARSAL_DSN is not set)"
NO_BUNDLE = "UNVERIFIED: real DAR-9 bundle (work/fd-1020/bundle) absent; 9 sources / 110 spans not checked"


def production_name_refusal(dbname) -> str | None:
    """None when `dbname` is acceptable as a rehearsal database; otherwise the reason. Allowlist AND denylist: the name
    must say rehearsal and must not look like any production-like name (identity never rests on the name alone: the
    host must also be loopback, and the connected server reports the same name back)."""
    n = (dbname or "").lower()
    if not n:
        return "no database name in the DSN"
    if NAME_DENY.search(n):
        return "database name looks like a production name"
    if not NAME_OK.fullmatch(n):
        return "database name does not contain 'rehears' (not an approved rehearsal name)"
    return None


def dsn_refusal(dsn) -> str | None:
    from sqlalchemy.engine import make_url
    try:
        u = make_url(dsn)
    except Exception as e:  # noqa: BLE001  never echo the DSN
        return f"DSN does not parse ({type(e).__name__})"
    if not u.drivername.startswith("postgresql"):
        return "DSN is not a PostgreSQL URL"
    why = production_name_refusal(u.database)
    if why:
        return why
    if dict(u.query):
        return "DSN carries query parameters (host, hostaddr, service, ... could redirect the connection)"
    host = (u.host or "").lower()
    if not host:
        return "DSN has no explicit host (loopback is required)"
    if host not in LOCAL_HOSTS and not host.startswith("/"):
        return "host is not local (loopback only; a socket path cannot be expressed without a query parameter)"
    return None


# ---- the guard itself needs no database -----------------------------------------------------------------------------
def test_guard_refuses_production_like_names_and_hosts():
    good = "postgresql+psycopg2://u@localhost:5543/vera_rehearsal_a"
    assert dsn_refusal(good) is None
    for bad in ("postgresql://u@localhost/vera", "postgresql://u@localhost/vera_prod", "postgresql://u@localhost/postgres",
                "postgresql://u@localhost/vera_vjay", "postgresql://u@localhost/mydb", "postgresql://u@localhost/",
                "postgresql://u@db.example.com/vera_rehearsal", "postgresql://u@localhost/live_rehearsal",
                "postgresql://u@/vera_rehearsal", "postgresql://u@localhost/vera_rehearsal?host=db.example.internal",
                "postgresql://u@localhost/vera_rehearsal?service=prod", "postgresql://u@localhost/vera_rehearsal?hostaddr=192.0.2.5",
                "sqlite:///x.db", "not a url"):
        assert dsn_refusal(bad), bad


def _script():
    spec = importlib.util.spec_from_file_location("ingest_fulldoc_chunks_pg", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _engine():
    from sqlalchemy import create_engine
    return create_engine(DSN, **_script().ENGINE_KW)


@pytest.fixture(scope="module")
def eng():
    if not DSN:
        pytest.skip(NO_DB)
    why = dsn_refusal(DSN)
    assert why is None, f"REFUSED rehearsal DSN: {why}"  # a failure, not a skip: a wrong DSN must be loud
    from sqlalchemy import text
    e = _engine()
    try:
        with e.connect() as c:
            name, vnum = c.execute(text("SELECT current_database(), current_setting('server_version_num')::int")).one()
            assert production_name_refusal(name) is None, "connected database reports a production-like name"
            addr = c.execute(text("SELECT inet_server_addr()::text")).scalar()  # NULL on a unix socket
            assert addr is None or ipaddress.ip_interface(addr).ip.is_loopback, "connected server address is not loopback"
            assert int(vnum) >= 160000, f"server_version_num {vnum}: PostgreSQL 16 or newer required"
            if c.execute(text("SELECT to_regclass(:t)"), {"t": f"{SCHEMA}.sources"}).scalar() is None:
                pytest.skip("UNVERIFIED: rehearsal DB has no vera_vjay schema (apply db/migrations first; not done here)")
            held = c.execute(text(f"SELECT count(*) FROM {SCHEMA}.sources WHERE provenance->>'stored_text_kind' = :k"),
                             {"k": fulldoc_db.FULLDOC_CHUNKS_KIND}).scalar()
            if held:
                pytest.skip("UNVERIFIED: rehearsal DB already holds full-document chunk rows; need an empty target")
        e.pg_server_version_num = int(vnum)
        yield e
    finally:
        e.dispose()


@pytest.fixture(scope="module")
def plan():
    if not (REAL_BUNDLE / "manifest-filled.https.json").is_file():
        pytest.skip(NO_BUNDLE)
    fc.reset_for_tests()
    s = _script()
    rows, report = s.build_plan(REAL_BUNDLE)
    assert (report["n_sources"], report["n_spans"]) == EXPECT
    bundle = fc.load_bundle(REAL_BUNDLE, fc.load_config())
    assert bundle.status == "ok" and len(bundle.acquired) == 9, bundle.unusable
    return s, rows, report, bundle


def _counts(eng) -> dict:
    from sqlalchemy import text
    with eng.connect() as c:
        return {t: c.execute(text(f"SELECT count(*) FROM {SCHEMA}.{t}")).scalar() for t in TABLES}


class _OuterTx:
    """Stands in for engine_wo: every begin() is a SAVEPOINT inside ONE outer transaction that the test rolls back.
    A failure inside begin() rolls back to the savepoint and re-raises, exactly as engine.begin() rolls back."""

    def __init__(self, conn):
        self.conn = conn

    @contextlib.contextmanager
    def begin(self):
        with self.conn.begin_nested():
            yield self.conn


@pytest.fixture()
def outer(eng, record_property):
    record_property("pg_server_version_num", eng.pg_server_version_num)
    conn = eng.connect()
    tx = conn.begin()
    try:
        yield conn
    finally:
        tx.rollback()
        conn.close()


def _conn_factory(conn):
    """A fulldoc_db `connect()` over the SAME outer-transaction connection; close() must not end it."""
    class _W:
        def __init__(self, raw):
            self._raw = raw

        def cursor(self):
            return self._raw.cursor()

        def close(self):
            pass
    return lambda: _W(conn.connection.dbapi_connection)


def _write(plan, conn, eng):
    s, rows, report, _ = plan
    return s.write_rows(_OuterTx(conn), eng, rows, fulltext_loader.load_config(), report["n_spans"])


def test_single_transaction_ingest_9_110_reads_back_exactly(eng, plan, outer):
    _, rows, report, bundle = plan
    done = _write(plan, outer, eng)
    assert (done["sources"], done["spans"]) == EXPECT
    # read-back through fulldoc_db with the shipped config (manifest, sums AND per-document pins)
    cfg = fc.load_config()
    assert len(cfg["pins"]["documents"]) == 9
    c = fulldoc_db.load_db(_conn_factory(outer), str(done["run_id"]), cfg)
    assert c.status == "ok" and not c.unusable, c.unusable
    assert len(c.acquired) == 9 and len(c._texts) == 9
    assert c._texts == bundle._texts  # exact reconstruction: every document, byte for byte
    assert c.acquired == bundle.acquired


@pytest.mark.parametrize("where", ["first", "middle", "last"])
def test_fault_at_first_middle_last_document_leaves_zero_rows(eng, plan, where):
    from sqlalchemy import event
    s, rows, report, _ = plan
    at = {"first": 0, "middle": len(rows) // 2, "last": len(rows) - 1}[where]
    before = _counts(eng)
    wo, seen = _engine(), {"n": 0, "fired": False}

    @event.listens_for(wo, "before_cursor_execute")
    def _boom(conn, cur, statement, params, ctx, many):
        if statement.lstrip().startswith(f"INSERT INTO {SCHEMA}.sources"):
            if seen["n"] == at:
                seen["fired"] = True
                raise RuntimeError("injected fault")
            seen["n"] += 1
        elif "count(*) FROM" in statement and not seen["fired"]:  # the pre-commit count check: the fault never fired
            raise AssertionError("injected fault never fired; aborting before COMMIT")
    try:
        with pytest.raises(s.WriteFailed) as e:
            s.write_rows(wo, eng, rows, fulltext_loader.load_config(), report["n_spans"])
    finally:
        wo.dispose()
    assert seen["fired"] and e.value.cause_name == "RuntimeError" and e.value.doc_index == at
    assert e.value.attempted_sources == at
    assert _counts(eng) == before  # nothing committed: all six tables unchanged


def test_coordinated_tamper_is_detected_by_doc_pins(eng, plan, outer):
    done = _write(plan, outer, eng)
    cfg = fc.load_config()
    sources, spans = fulldoc_db.fetch_rows(_conn_factory(outer), done["run_id"])
    assert len(sources) == 9 and len(spans) == 110
    sid = sources[0]["source_id"]
    mine = [x for x in spans if x["source_id"] == sid]
    sp = next(x for x in mine if re.search("[a-z]", x["text"]))
    i = re.search("[a-z]", sp["text"]).start()
    sp["text"] = sp["text"][:i] + ("z" if sp["text"][i] != "z" else "y") + sp["text"][i + 1:]
    text = "".join(x["text"] for x in sorted(mine, key=lambda x: x["start_index"]))
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    prov = sources[0]["provenance"]
    prov["text_sha256"] = h
    prov["meta"]["text_sha256"] = h  # span text + both recorded hashes rewritten together: self-consistent
    unpinned = copy.deepcopy(cfg)
    unpinned["pins"] = {"manifest_sha256": None, "sums_sha256": None}
    accepted = fulldoc_db.load_from_rows(copy.deepcopy(sources), copy.deepcopy(spans), unpinned)
    assert accepted.status == "ok" and not accepted.unusable  # the tamper is invisible without the config pins
    bad = fulldoc_db.load_from_rows(sources, spans, cfg)
    assert bad.status == "corpus_unusable" and not bad.acquired and bad._texts == {}
    assert {u["code"] for u in bad.unusable} == {fulldoc_db.R_DBDOCPIN}


def test_uncoordinated_span_edit_in_the_database_is_detected(eng, plan, outer):
    from sqlalchemy import text
    done = _write(plan, outer, eng)
    outer.execute(text(f"UPDATE {SCHEMA}.evidence_spans SET text = 'X' || substr(text, 2) WHERE span_id = "
                       f"(SELECT min(e.span_id) FROM {SCHEMA}.evidence_spans e JOIN {SCHEMA}.sources s "
                       f"ON s.source_id = e.source_id JOIN {SCHEMA}.candidates k ON k.candidate_id = s.candidate_id "
                       f"WHERE k.run_id = :r)"), {"r": done["run_id"]})
    c = fulldoc_db.load_db(_conn_factory(outer), str(done["run_id"]), fc.load_config())
    assert len(c.unusable) >= 1 and len(c.acquired) <= 8
    assert all(u["code"].startswith("R-") and "text" not in u for u in c.unusable)


def test_reingest_is_refused_with_zero_new_rows(eng, plan, outer):
    s, rows, report, _ = plan
    _write(plan, outer, eng)
    from sqlalchemy import text
    after_first = {t: outer.execute(text(f"SELECT count(*) FROM {SCHEMA}.{t}")).scalar() for t in TABLES}
    # `eng` (the read-only preflight) cannot see the uncommitted first ingest, so the refusal comes from the duplicate
    # check repeated inside the write transaction (advisory lock + same-connection SELECT)
    with pytest.raises(s.WriteFailed) as e:
        s.write_rows(_OuterTx(outer), eng, rows, fulltext_loader.load_config(), report["n_spans"])
    assert e.value.cause_name == "IngestError" and e.value.doc_index == 0
    assert {t: outer.execute(text(f"SELECT count(*) FROM {SCHEMA}.{t}")).scalar() for t in TABLES} == after_first


@pytest.mark.skipif(not (DSN and ALLOW_COMMIT), reason="UNVERIFIED: commit visibility not run (needs VERA_PG_REHEARSAL_DSN "
                                                       "and VERA_PG_REHEARSAL_ALLOW_COMMIT=1)")
def test_committed_ingest_is_visible_to_a_second_connection_and_reingest_is_refused(eng, plan):
    s, rows, report, bundle = plan
    wo = _engine()
    try:
        done = s.write_rows(wo, eng, rows, fulltext_loader.load_config(), report["n_spans"])
    finally:
        wo.dispose()
    assert (done["sources"], done["spans"]) == EXPECT
    raw_engine = _engine()

    def connect():
        return raw_engine.raw_connection()
    try:
        c = fulldoc_db.load_db(connect, str(done["run_id"]), fc.load_config())
    finally:
        raw_engine.dispose()
    assert c.status == "ok" and c._texts == bundle._texts
    before = _counts(eng)
    wo2 = _engine()
    try:
        with pytest.raises(fulltext_loader.IngestError):
            s.write_rows(wo2, eng, rows, fulltext_loader.load_config(), report["n_spans"])
    finally:
        wo2.dispose()
    assert _counts(eng) == before
