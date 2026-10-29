"""DBC-M1: atomic, recoverable full-document write. SYNTHETIC bundle, SQLite file (schema ATTACHed as vera_vjay), no network,
no PostgreSQL. SQLite stands in for the transaction semantics only; jsonb, ANY(), grants, PG rollback and the timeouts of
ENGINE_KW are NOT exercised here (they need an isolated PG16 run)."""
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, text

from tests import fulldoc_bundle_synth as synth
from vera import fulldoc_corpus as fc, fulldoc_db, fulltext_loader

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "ingest_fulldoc_chunks.py"
REAL_BUNDLE = Path(__file__).resolve().parent.parent.parent / "work" / "fd-1020" / "bundle"
DDL = """
CREATE TABLE questions (question_id INTEGER PRIMARY KEY, research_question TEXT);
CREATE TABLE runs (run_id INTEGER PRIMARY KEY, question_id INTEGER, question TEXT);
CREATE TABLE candidates (candidate_id INTEGER PRIMARY KEY, run_id INTEGER, source_url TEXT, title TEXT, snippet TEXT,
  gate_a_decision TEXT, gate_a_rationale TEXT, fetch_status TEXT);
CREATE TABLE canonical_sources (canonical_source_id INTEGER PRIMARY KEY, canonical_url TEXT UNIQUE);
CREATE TABLE sources (source_id INTEGER PRIMARY KEY, canonical_source_id INTEGER, candidate_id INTEGER, version INTEGER,
  content TEXT, content_hash TEXT, provenance TEXT);
CREATE TABLE evidence_spans (span_id INTEGER PRIMARY KEY, source_id INTEGER, text TEXT, start_index INTEGER,
  end_index INTEGER, relevance_score REAL, evidence_type TEXT, UNIQUE(source_id, start_index, end_index));
"""
TABLES = ["questions", "runs", "candidates", "canonical_sources", "sources", "evidence_spans"]


def _script():
    spec = importlib.util.spec_from_file_location("ingest_fulldoc_chunks_atomic", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _make_db(d: Path, name: str) -> str:
    """A SQLite file whose schema file is ATTACHed as vera_vjay on every connection."""
    d.mkdir(parents=True, exist_ok=True)
    sch = d / f"{name}.schema.db"
    con = sqlite3.connect(sch)
    con.executescript(DDL)
    con.commit()
    con.close()
    return f"sqlite:///{d / (name + '.main.db')}|{sch}"


def _engine(spec: str):
    url, sch = spec.split("|")
    eng = create_engine(url)

    @event.listens_for(eng, "connect")
    def _attach(dbapi, _):  # noqa: ANN001
        dbapi.execute(f"ATTACH DATABASE '{sch}' AS vera_vjay")

    @event.listens_for(eng, "before_cursor_execute", retval=True)
    def _jsonb(conn, cur, statement, params, ctx, many):  # test-only: SQLite has no jsonb; production SQL is unchanged
        return statement.replace("CAST(? AS jsonb)", "?"), params
    return eng


def _counts(spec: str) -> dict:
    eng = _engine(spec)
    try:
        with eng.connect() as c:
            return {t: c.execute(text(f"SELECT count(*) FROM vera_vjay.{t}")).scalar() for t in TABLES}
    finally:
        eng.dispose()


class Factory:
    """Recording engine factory; `fail_at` makes the Nth (0-based) INSERT INTO sources on the WRITE engine raise."""

    def __init__(self, fail_at=None):
        self.calls, self.fail_at, self.n = [], fail_at, 0

    def __call__(self, url, **kw):
        self.calls.append((url, kw))
        eng = _engine(url)
        if self.fail_at is not None and len(self.calls) == 1:
            @event.listens_for(eng, "before_cursor_execute")
            def _boom(conn, cur, statement, params, ctx, many):
                if statement.lstrip().startswith("INSERT INTO vera_vjay.sources"):
                    if self.n == self.fail_at:
                        raise RuntimeError("injected fault")
                    self.n += 1
        return eng


@pytest.fixture()
def world(tmp_path):
    fc.reset_for_tests()
    root = synth.write_bundle(tmp_path / "b")
    cfg = synth.synth_cfg()
    s = _script()
    rows, report = s.build_plan(root, cfg, target=700)
    spec = _make_db(tmp_path / "db", "w")
    return s, root, cfg, rows, report, spec, tmp_path


def _run(w, factory, extra=(), ro=None, expect=None):
    s, root, cfg, rows, report, spec, tmp = w
    env = {"VERA_DB_URL_WO": spec, "VERA_DB_URL_RO": ro or spec}
    argv = ["--bundle", str(root), "--plan-dir", str(tmp / "p"), "--target-chars", "700", "--write", "--new-run",
            "--i-have-owner-authorization", "tok", "--expect-spans", str(expect or report["n_spans"]), *extra]
    return s.main(argv, env=env, engine_factory=factory, cfg=cfg, guard=lambda: None)


# (a) fault injection: nothing persists, counts and label are reported, no text leaks
@pytest.mark.parametrize("at", [0, 1, 2])
def test_fault_at_first_middle_last_doc_rolls_back(world, capsys, at):
    s, _, _, rows, report, spec, _ = world
    rc = _run(world, Factory(fail_at=at))
    out = capsys.readouterr().out
    assert rc == s.EXIT_ERROR
    assert all(v == 0 for v in _counts(spec).values())
    assert "rolled_back" in out and "run_id=" in out and "(not persisted)" in out
    assert f"doc_index={at} " in out
    assert "committed sources=0 spans=0" in out and "RuntimeError" in out and "written:" not in out
    assert f"attempted sources={at} " in out
    for r in rows:
        for sp in r["spans"][:5]:
            assert sp["text"][:40] not in out


def test_write_failed_carries_counts(world):
    s, _, _, rows, report, spec, _ = world
    eng_ro = _engine(spec)
    eng_wo = Factory(fail_at=1)(spec)  # the first factory call gets the fault hook
    with pytest.raises(s.WriteFailed) as e:
        s.write_rows(eng_wo, eng_ro, rows, fulltext_loader.load_config(), report["n_spans"])
    w = e.value
    assert w.attempted_sources == 1 and w.attempted_spans == len(rows[0]["spans"]) and w.doc_index == 1
    assert w.rid is not None and w.cause_name == "RuntimeError"
    assert all(v == 0 for v in _counts(spec).values())


def test_pre_commit_count_check_failure_rolls_back(world, capsys, monkeypatch):
    s, _, _, rows, report, spec, _ = world
    monkeypatch.setattr(s, "_run_counts", lambda c, S, rid: (0, 0))
    rc = s.main(["--bundle", str(world[1]), "--plan-dir", str(world[6] / "p"), "--target-chars", "700", "--write",
                 "--new-run", "--i-have-owner-authorization", "tok", "--expect-spans", str(report["n_spans"])],
                env={"VERA_DB_URL_WO": spec, "VERA_DB_URL_RO": spec}, engine_factory=Factory(), cfg=world[2],
                guard=lambda: None)
    out = capsys.readouterr().out
    assert rc == s.EXIT_ERROR and "rolled_back" in out and "(not persisted)" in out
    assert f"attempted sources={len(rows)} spans={report['n_spans']}" in out
    assert all(v == 0 for v in _counts(spec).values())


# (b) success
def test_success_on_synthetic_bundle(world, capsys):
    s, _, _, rows, report, spec, _ = world
    assert _run(world, Factory()) == 0
    out = capsys.readouterr().out
    assert f"written: sources=3 spans={report['n_spans']}" in out
    c = _counts(spec)
    assert c["sources"] == 3 and c["evidence_spans"] == report["n_spans"] and c["canonical_sources"] == 3
    assert c["questions"] == 1 and c["runs"] == 1 and c["candidates"] == 3


@pytest.mark.skipif(not (REAL_BUNDLE / "manifest-filled.https.json").exists(),
                    reason="real DAR-9 bundle absent: 9 sources / 110 spans UNVERIFIED")
def test_success_on_real_bundle_9_110(tmp_path):
    fc.reset_for_tests()
    s = _script()
    spec = _make_db(tmp_path / "db", "r")
    rows, report = s.build_plan(REAL_BUNDLE)
    assert (report["n_sources"], report["n_spans"]) == (9, 110)
    eng = _engine(spec)
    try:
        done = s.write_rows(eng, _engine(spec), rows, fulltext_loader.load_config(), 110)
    finally:
        eng.dispose()
    assert (done["sources"], done["spans"]) == (9, 110)


def test_real_bundle_exact_reconstruction(tmp_path):
    """9 real docs: plan -> SQLite -> load_from_rows with the shipped pins; every text byte-identical to the bundle."""
    import hashlib
    import os
    RB = Path(os.environ.get("VERA_REAL_DAR9_BUNDLE") or REAL_BUNDLE)
    if not (RB / "manifest-filled.https.json").exists():
        pytest.skip("UNVERIFIED: real bundle not mounted")
    fc.reset_for_tests()
    s = _script()
    cfg = fc.load_config()  # shipped config, shipped per-document pins
    pins = cfg["pins"]["documents"]
    assert len(pins) == 9
    spec = _make_db(tmp_path / "db", "x")
    rows, report = s.build_plan(RB)
    assert (report["n_sources"], report["n_spans"]) == (9, 110)
    eng = _engine(spec)
    try:
        done = s.write_rows(eng, _engine(spec), rows, fulltext_loader.load_config(), 110)
    finally:
        eng.dispose()
    assert (done["sources"], done["spans"]) == (9, 110)
    eng = _engine(spec)
    with eng.connect() as c:
        srows = c.execute(text("SELECT source_id, canonical_source_id, content, content_hash, provenance "
                               "FROM vera_vjay.sources ORDER BY source_id")).fetchall()
        urls = dict(c.execute(text("SELECT canonical_source_id, canonical_url FROM vera_vjay.canonical_sources")).fetchall())
        sprows = c.execute(text("SELECT source_id, start_index, end_index, text FROM vera_vjay.evidence_spans "
                                "ORDER BY source_id, start_index")).fetchall()
    eng.dispose()
    assert len(sprows) == 110
    sources = [{"source_id": r[0], "canonical_url": urls[r[1]], "content": r[2], "content_hash": r[3],
                "provenance": json.loads(r[4])} for r in srows]
    spans = [{"source_id": r[0], "start_index": r[1], "end_index": r[2], "text": r[3]} for r in sprows]
    db = fulldoc_db.load_from_rows(sources, spans, cfg)
    assert db.status == "ok" and not db.unusable and len(db.acquired) == 9
    bundle = fc.load_bundle(RB, cfg)
    assert bundle.status == "ok" and len(bundle.acquired) == 9
    mismatches = 0
    for a in db.acquired:
        did = a["document_id"]
        t = db.text(did)
        if t.encode("utf-8") != bundle.text(did).encode("utf-8"):
            mismatches += 1
        if hashlib.sha256(t.encode("utf-8")).hexdigest() != pins[did]["text_sha256"] or a["text_sha256"] != pins[did]["text_sha256"]:
            mismatches += 1
    assert mismatches == 0
    assert {a["document_id"] for a in db.acquired} == set(pins)


# (c) round trip through the reader
def test_round_trip_into_load_from_rows(world):
    s, _, cfg, rows, report, spec, _ = world
    assert _run(world, Factory()) == 0
    eng = _engine(spec)
    with eng.connect() as c:
        srows = c.execute(text("SELECT source_id, canonical_source_id, content, content_hash, provenance "
                               "FROM vera_vjay.sources ORDER BY source_id")).fetchall()
        urls = dict(c.execute(text("SELECT canonical_source_id, canonical_url FROM vera_vjay.canonical_sources")).fetchall())
        sprows = c.execute(text("SELECT source_id, start_index, end_index, text FROM vera_vjay.evidence_spans "
                                "ORDER BY source_id, start_index")).fetchall()
    eng.dispose()
    assert all(isinstance(r[4], str) and isinstance(json.loads(r[4]), dict) for r in srows)  # a JSON object, not 0
    sources = [{"source_id": r[0], "canonical_url": urls[r[1]], "content": r[2], "content_hash": r[3],
                "provenance": json.loads(r[4])} for r in srows]
    spans = [{"source_id": r[0], "start_index": r[1], "end_index": r[2], "text": r[3]} for r in sprows]
    corpus = fulldoc_db.load_from_rows(sources, spans, cfg)
    assert corpus.status == "ok" and len(corpus.acquired) == 3
    want = {d["document_id"]: d["text_sha256"] for d in report["documents"]}
    got = {a["document_id"]: a["text_sha256"] for a in corpus.acquired}
    assert got == want


# (d) rerun refused, zero new rows
def test_rerun_refused_by_hash_and_by_url(world, capsys):
    s, _, _, rows, report, spec, _ = world
    assert _run(world, Factory()) == 0
    before = _counts(spec)
    assert _run(world, Factory()) == s.EXIT_ERROR  # same text_sha256 (and URL): refused in preflight
    assert _counts(spec) == before
    # URL only: change every hash so only the canonical URL can match
    import copy
    r2 = copy.deepcopy(rows)
    for r in r2:
        r["source"]["provenance"]["text_sha256"] = "0" * 63 + "1"
    with pytest.raises(fulltext_loader.IngestError, match="canonical URL"):
        s.write_rows(_engine(spec), _engine(spec), r2, fulltext_loader.load_config(), report["n_spans"])
    # hash only: new URLs, same text hash
    r3 = copy.deepcopy(rows)
    for r in r3:
        r["source"]["canonical_url"] += "-other"  # R3NORM-1028: was "#other"; a fragment-only variant is now (correctly) the same key
    with pytest.raises(fulltext_loader.IngestError, match="same text"):
        s.write_rows(_engine(spec), _engine(spec), r3, fulltext_loader.load_config(), report["n_spans"])
    assert _counts(spec) == before
    capsys.readouterr()


def test_empty_string_hash_refused(world):
    s, _, _, rows, report, spec, _ = world
    import copy
    r2 = copy.deepcopy(rows)
    r2[0]["source"]["provenance"]["text_sha256"] = fulldoc_db.EMPTY_SHA256
    with pytest.raises(fulltext_loader.IngestError, match="empty-string"):
        s.write_rows(_engine(spec), _engine(spec), r2, fulltext_loader.load_config(), report["n_spans"])
    assert all(v == 0 for v in _counts(spec).values())


def test_duplicate_rechecked_inside_write_transaction(world):
    """A stale/empty read-only view must not defeat the check: it is repeated on the write connection."""
    s, _, _, rows, report, spec, tmp = world
    assert _run(world, Factory()) == 0
    before = _counts(spec)
    empty_ro = _make_db(tmp / "db2", "empty")
    with pytest.raises(s.WriteFailed) as e:
        s.write_rows(_engine(spec), _engine(empty_ro), rows, fulltext_loader.load_config(), report["n_spans"])
    assert e.value.cause_name == "IngestError" and e.value.doc_index == 0
    assert _counts(spec) == before


# (e) engine keywords
def test_factory_receives_engine_kw_with_hide_parameters(world):
    s = world[0]
    f = Factory()
    assert _run(world, f) == 0
    assert len(f.calls) == 2
    for _, kw in f.calls:
        assert kw == s.ENGINE_KW and kw["hide_parameters"] is True
        opts = kw["connect_args"]["options"]
        assert "statement_timeout=30000" in opts and "lock_timeout=5000" in opts
        assert "idle_in_transaction_session_timeout=60000" in opts and kw["connect_args"]["connect_timeout"] == 10


def test_content_hash_only_row_refused_as_same_text(world):
    """P3-a: a stored row matching only by content_hash (no provenance text_sha256 key) still refuses; zero new rows."""
    import copy
    import hashlib
    s, _, _, rows, report, spec, _ = world
    h = hashlib.sha256(b"some stored full text").hexdigest()
    con = sqlite3.connect(spec.split("|")[1])
    con.execute("INSERT INTO sources (source_id, canonical_source_id, candidate_id, version, content, content_hash, provenance)"
                " VALUES (1, 1, 1, 1, ?, ?, '{}')", ("some stored full text", h))
    con.commit()
    con.close()
    before = _counts(spec)
    r2 = copy.deepcopy(rows)
    r2[0]["source"]["provenance"]["text_sha256"] = h
    with pytest.raises(fulltext_loader.IngestError, match="same text"):
        s.write_rows(_engine(spec), _engine(spec), r2, fulltext_loader.load_config(), report["n_spans"])
    assert _counts(spec) == before


class _Stop(Exception):
    pass


class _FakeConn:
    def __init__(self, name, log):
        self.dialect, self.log = type("D", (), {"name": name})(), log

    def execute(self, stmt, params=None):
        self.log.append(str(stmt))
        if "pg_advisory_xact_lock" not in str(stmt):
            raise _Stop()  # stop after the first non-lock statement; nothing else is needed


class _FakeEngine:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def begin(self):
        import contextlib
        return contextlib.nullcontext(_FakeConn(self.name, self.log))


@pytest.mark.parametrize("dialect,expect_lock", [("postgresql", True), ("sqlite", False)])
def test_advisory_lock_only_on_postgresql(world, dialect, expect_lock):
    """P3-c: the per-hash advisory lock is issued first inside the write transaction on postgresql only."""
    s, _, _, rows, report, spec, _ = world
    log = []
    with pytest.raises(s.WriteFailed) as e:
        s.write_rows(_FakeEngine(dialect, log), _engine(spec), rows, fulltext_loader.load_config(), report["n_spans"])
    assert e.value.cause_name == "_Stop"
    locks = [x for x in log if "pg_advisory_xact_lock" in x]
    assert bool(locks) is expect_lock
    if expect_lock:
        assert "hashtext" in log[0] and len(locks) == 1  # before the duplicate check, once per document reached


def test_fragment_only_variant_refused_as_same_canonical_url(world):
    """R3NORM-FIX-1028: a fragment-only spelling normalises to the stored key, so it is refused as the same canonical URL."""
    import copy
    s, _, _, rows, report, spec, _ = world
    assert _run(world, Factory()) == 0
    before = _counts(spec)
    r2 = copy.deepcopy(rows)
    for r in r2:
        r["source"]["canonical_url"] += "#other"
        r["source"]["provenance"]["text_sha256"] = "0" * 63 + "1"  # only the URL can match
    with pytest.raises(fulltext_loader.IngestError, match="canonical URL"):
        s.write_rows(_engine(spec), _engine(spec), r2, fulltext_loader.load_config(), report["n_spans"])
    assert _counts(spec) == before
    from vera.url_key import normalize_url
    assert normalize_url(rows[0]["source"]["canonical_url"] + "#other") == rows[0]["source"]["canonical_url"]
