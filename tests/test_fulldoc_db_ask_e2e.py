"""DEV-B: DB-backed full-document /ask, end to end, OFFLINE/STUB, NOT LIVE.

SYNTHETIC bundle -> build_plan -> M1 write_rows (the real INSERT path) into a SQLite file (schema ATTACHed as vera_vjay)
-> the bundle directory is DELETED -> /ask via main:app TestClient with VERA_FULLDOC_DB_RUN_ID set, so get_corpus takes
the DB branch: fulldoc_db.cached_load -> fetch_rows (the two production SELECT constants, issued through a read-only
SQLite DB-API adapter) -> load_from_rows -> FullDocCorpus -> admission -> fulldoc_answer (stub model) -> claim_check
(stub checker) -> AskResponse.

Stubbed: scope router, answering LLM, claim checker, web retrieval, DAR/memory stores. NOT exercised: PostgreSQL itself
(jsonb, ANY(), grants, statement timeout, the real read-only account), the real model, the real checker, real documents.
"""
import json
import re
import shutil
import sqlite3

import pytest

from tests import fulldoc_bundle_synth as synth
from tests import test_fulldoc_e2e as e2e
from tests import test_fulldoc_write_atomic as wa
from vera import fulldoc_corpus, fulldoc_db, fulltext_loader

SPAN = re.compile(r"^chars=(\d+)-(\d+)$")
SID = re.compile(r"^(.+):(\d+)-(\d+)$")


class SqliteRO:
    """DB-API connection over the SQLite file written by write_rows. Accepts ONLY the two production SELECT constants
    (verbatim) and translates their PostgreSQL placeholders/ANY() for SQLite; anything else fails the test."""

    def __init__(self, main_path, schema_path, log):
        self.con = sqlite3.connect(f"file:{main_path}?mode=ro", uri=True)
        self.con.execute(f"ATTACH DATABASE 'file:{schema_path}?mode=ro' AS vera_vjay")
        self.log = log

    def cursor(self):
        return self

    def execute(self, sql, params):
        assert sql in (fulldoc_db.SQL_SOURCES, fulldoc_db.SQL_SPANS), "unexpected SQL"
        self.log.append(sql)
        q, p = sql.replace("%(run_id)s", ":run_id").replace("%(kind)s", ":kind"), dict(params)
        if "%(ids)s" in q:
            ids = list(p.pop("ids"))
            p.update({f"i{n}": v for n, v in enumerate(ids)})
            q = q.replace("= ANY(%(ids)s)", "IN (" + ",".join(f":i{n}" for n in range(len(ids))) + ")")
        self._cur = self.con.execute(q, p)

    def fetchall(self):
        return self._cur.fetchall()

    def close(self):
        self.con.close()


@pytest.fixture
def dbw(monkeypatch, tmp_path):
    """Persist the synthetic bundle through write_rows, delete the bundle, point the DB source at the SQLite file."""
    fulldoc_corpus.reset_for_tests()
    monkeypatch.delenv("VERA_FULLDOC_BUNDLE_DIR", raising=False)
    root = synth.write_bundle(tmp_path / "bundle")
    cfg = synth.synth_cfg()
    cfg.setdefault("fulldoc", {})["enabled"] = True
    s = wa._script()
    rows, report = s.build_plan(root, cfg, target=700)
    spec = wa._make_db(tmp_path / "db", "w")
    eng = wa._engine(spec)
    done = s.write_rows(eng, eng, rows, fulltext_loader.load_config(), report["n_spans"])
    eng.dispose()
    assert done["sources"] == 3 and done["spans"] == report["n_spans"] and done["run_id"]
    shutil.rmtree(root)  # from here on, a bundle read is impossible
    url, schema = spec.split("|")
    main_path = url[len("sqlite:///"):]
    con = sqlite3.connect(main_path)
    con.execute(f"ATTACH DATABASE '{schema}' AS vera_vjay")
    persisted = {}  # document_id -> [(span_id INTEGER, start, end, text)] as stored in evidence_spans
    for sid, prov in con.execute("SELECT source_id, provenance FROM vera_vjay.sources").fetchall():
        did = json.loads(prov)["document_id"]
        persisted[did] = con.execute("SELECT span_id, start_index, end_index, text FROM vera_vjay.evidence_spans "
                                     "WHERE source_id = ? ORDER BY start_index", (sid,)).fetchall()
    con.close()
    log = []
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(fulldoc_db, "default_connect", lambda c, env: (lambda: SqliteRO(main_path, schema, log)))
    monkeypatch.setenv(fulldoc_corpus.DB_RUN_ID_ENV_DEFAULT, str(done["run_id"]))
    h = e2e.Harness(monkeypatch, tmp_path)
    yield h, cfg, persisted, log, done["run_id"]
    fulldoc_corpus.reset_for_tests()


def _text(persisted, did):
    return "".join(r[3] for r in persisted[did])


def test_db_backed_ask_answers_from_persisted_chunks(dbw):
    """offline/stub; not live."""
    h, cfg, persisted, log, _ = dbw
    raw, body = h.ask(e2e.Q_EPI)
    assert body["result_kind"] == "answer" and body["outcome"] == "answer", body.get("contract")
    # source is the DB, not the bundle: the two SELECTs ran, the bundle dir no longer exists, readiness says "db"
    assert log == [fulldoc_db.SQL_SOURCES, fulldoc_db.SQL_SPANS]
    r = fulldoc_corpus.readiness(cfg)
    assert r["source"] == "db" and r["status"] == "ok" and r["n_acquired"] == 3 and not r["bundle_configured"]
    assert sorted(e2e._ids(body["examined_full_documents"])) == sorted(persisted)
    assert body["corpus_source"] == {"kind": "db"} and "run_id" not in body["corpus_source"]  # loaded from the DB branch
    assert "source" in r and "run_id" not in r  # readiness behaviour unchanged (Part B awaits the owner)
    # the passages in the model context are the persisted chunk texts (sliced from the concatenated evidence_spans rows)
    prompt = json.dumps(h.answer_calls[0], ensure_ascii=False)
    adm = body["admitted_evidence"]
    assert adm and len(h.answer_calls) == 1
    for a in adm:
        m = SPAN.match(a["span"])
        assert m and SID.match(a["span_id"]).group(1) == a["document_id"]
        passage = _text(persisted, a["document_id"])[int(m.group(1)):int(m.group(2))]
        # WED-P16-RET: compare against the decoded passages (a passage holding a newline or quote is escaped twice
        # in the json.dumps of the whole message list, so a single-encoded substring test misses it)
        texts = [p["text"] for p in json.loads(h.answer_calls[0][-1]["content"])["passages"]]
        assert passage.strip() and any(passage.strip() in t for t in texts)
    # citations map to persisted evidence_spans rows (integer span_id) whose text holds the verbatim quote
    adm_ids = {a["span_id"] for a in adm}
    assert body["citations"]
    for c in body["citations"]:
        assert c["span_id"] in adm_ids
        a, b = map(int, SPAN.match(c["span"]).groups())
        doc = next(x["document_id"] for x in adm if x["span_id"] == c["span_id"])
        rows = [r for r in persisted[doc] if r[1] < b and r[2] > a]
        stored_ids = {x[0] for x in persisted[doc]}  # integer evidence_spans row ids as persisted
        assert rows and all(type(r[0]) is int and r[0] in stored_ids for r in rows)  # each citation -> persisted row ids
        joined, base = "".join(r[3] for r in rows), rows[0][1]
        assert c["quote"] and c["quote"] in joined[a - base:b - base]
    assert body["sources"] and all(s["span_id"] in adm_ids for s in body["sources"])
    e2e._no_long_doc_text(raw, {d: _text(persisted, d) for d in persisted})


def test_db_backed_no_relevant_passage_is_explicit_insufficient(dbw):
    """offline/stub; not live. A reading list is never an answer."""
    h = dbw[0]
    raw, body = h.ask(e2e.Q_NONE)
    e2e._assert_non_answer(body, {"insufficient_full_document_evidence"})
    assert body["outcome"] == "insufficient_evidence" and body["candidates"]  # web reading list, not an answer
    assert h.answer_calls == [] and h.check_calls == []


def test_db_backed_model_insufficient_is_not_an_answer(dbw):
    """offline/stub; not live."""
    h = dbw[0]
    h.answer_text = "INSUFFICIENT_EVIDENCE"
    raw, body = h.ask(e2e.Q_EPI)
    e2e._assert_non_answer(body, {"model_insufficient"})
    assert body["outcome"] == "insufficient_evidence"


def test_db_backed_unknown_run_is_corpus_unusable(dbw, monkeypatch):
    """offline/stub; not live. A run id with no rows never grounds anything."""
    h, cfg, _, log, rid = dbw
    monkeypatch.setenv(fulldoc_corpus.DB_RUN_ID_ENV_DEFAULT, str(rid + 999))
    raw, body = h.ask(e2e.Q_EPI, web=False)
    sql = list(log)  # record the SQL first, so a stray statement cannot be swallowed by load_db's except
    assert sql == [fulldoc_db.SQL_SOURCES]  # no source rows, so no span SELECT
    e2e._assert_non_answer(body, {"full_document_corpus_unusable"})
    assert body["outcome"] == "corpus_unusable" and h.answer_calls == []
    assert "corpus_source" not in body  # the label means "loaded from": omitted when the corpus failed
    assert list(log) == sql


def test_no_corpus_and_no_web_is_no_sources(dbw, monkeypatch):
    """offline/stub; not live. Path disabled (no run id) and no web candidates => explicit no_sources."""
    h = dbw[0]
    monkeypatch.delenv(fulldoc_corpus.DB_RUN_ID_ENV_DEFAULT)
    raw, body = h.ask(e2e.Q_EPI, web=False)
    assert body["result_kind"] == "reading_list" and body["outcome"] == "no_sources"
    assert not body.get("citations") and body["sources"] == [] and h.answer_calls == []


def test_bundle_source_is_never_labelled_db(monkeypatch, tmp_path):
    """offline/stub; not live. The local bundle path is labelled bundle, in /ask; readiness unchanged."""
    fulldoc_corpus.reset_for_tests()
    monkeypatch.delenv(fulldoc_corpus.DB_RUN_ID_ENV_DEFAULT, raising=False)
    h = e2e.Harness(monkeypatch, tmp_path)
    h.bundle()
    raw, body = h.ask(e2e.Q_EPI)
    assert body["result_kind"] == "answer" and body["corpus_source"] == {"kind": "bundle"}
    fulldoc_corpus.reset_for_tests()


def test_read_path_carries_evidence_spans_created_at_as_corpus_added_at(dbw):
    """FRESH-AMEND-1023: the DB reader carries each stored chunk's evidence_spans.created_at (+ span_id) as its
    corpus_added_at stamp (basis 'chunk'); admitted passages reference those stamps. offline/stub; not live."""
    from vera import fulldoc_admission
    h, cfg, persisted, log, _ = dbw
    c = fulldoc_corpus.get_corpus(cfg)
    assert c.status == "ok" and c.stamps and set(c.stamps) == {a["document_id"] for a in c.acquired}
    assert all("created_at" not in a and "corpus_stamps" not in a for a in c.acquired)  # bundle parity kept
    for did, rec in c.stamps.items():
        ids = {r[0] for r in persisted[did]}
        assert {x["span_id"] for x in rec["corpus_stamps"]} == ids
        assert all(x["basis"] == "chunk" and x["corpus_added_at"] for x in rec["corpus_stamps"])
    ev = fulldoc_admission.admit(e2e.Q_EPI, c, cfg)
    assert ev.admitted
    for a in ev.admitted:
        assert a["corpus_chunks"] and all(x in c.stamps[str(a["document_id"])]["corpus_stamps"] for x in a["corpus_chunks"])
