"""End-to-end M2 against real Postgres. Opt-in and double-guarded:
  VERA_TEST_DB_URL=<url of a DISPOSABLE database>   VERA_TEST_DB_DISPOSABLE=1

Deliberately does NOT read EXTERNAL_DB_URL (shared production instance). Everything runs in one
transaction and is rolled back. It applies the real M1 migrations (001 + 002, which DROP and
recreate schema vera_vjay - hence the disposable-DB requirement) inside that transaction, then
runs M2 with stubbed search/LLM/fetch and checks candidates, sources and iterations.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

from vera.cost_ledger import CostLedger
from vera.gate_a import score_candidates
from vera.m2_runner import PostgresStore, run_m2
from vera.selective_fetch import FetchResult

URL = os.getenv("VERA_TEST_DB_URL")
pytestmark = pytest.mark.skipif(not (URL and os.getenv("VERA_TEST_DB_DISPOSABLE") == "1"),
                                reason="needs VERA_TEST_DB_URL and VERA_TEST_DB_DISPOSABLE=1")
MIGRATIONS = Path(__file__).resolve().parent.parent / "db" / "migrations"


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()


def _llm(messages):
    n = len(json.loads(messages[1]["content"])["candidates"])
    sc = [(.9, .9, .9, "low"), (.8, .7, .9, "low"), (.4, .4, .5, "medium"), (.05, .05, .1, "low")]
    rows = [{"id": i, "query_fit": sc[i % 4][0], "evidentiary_value": sc[i % 4][1], "cost_risk": sc[i % 4][2],
             "uncertainty": sc[i % 4][3], "rationale": {"query_fit": "a", "evidentiary_value": "b", "cost_risk": "c"}}
            for i in range(n)]
    return json.dumps({"scores": rows}), 500, 200


def test_end_to_end_in_rolled_back_transaction():
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    engine = create_engine(URL)
    conn = engine.connect()
    tx = conn.begin()
    try:
        cur = conn.connection.cursor()  # raw DBAPI cursor, same transaction; no %-interpolation without args
        for name in ("001_vera_schema_init.sql", "002_vera_indices.sql"):
            cur.execute((MIGRATIONS / name).read_text())
        s = Session(bind=conn)
        schema = "vera_vjay"
        qid = s.execute(text(f"INSERT INTO {schema}.questions (research_question) VALUES ('q') RETURNING question_id")).scalar()
        run_id = s.execute(text(f"INSERT INTO {schema}.runs (question_id) VALUES (:q) RETURNING run_id"), {"q": qid}).scalar()

        def search(q, n, ledger=None):
            ledger.record(kind="search", provider="arxiv", cost_usd=0.008)
            return [{"url": f"https://example.org/{i}", "title": f"T{i}", "snippet": "s", "rank": i + 1,
                     "provider": "arxiv"} for i in range(4)]

        def fetch(url, ledger=None):
            return FetchResult(url=url, ok=True, content_text="text " + url, content_hash=sha("text " + url),
                               provenance={"final_url": url})

        store = PostgresStore(s, schema=schema, commit=False)
        led = CostLedger()
        res = run_m2("q", run_id, store, search_fn=search, fetch_fn=fetch, ledger=led,
                     score_fn=lambda q, c, cfg=None, ledger=None: score_candidates(q, c, _llm, cfg=cfg, ledger=ledger))

        rows = s.execute(text(f"SELECT gate_a_decision, fetch_status, count(*) FROM {schema}.candidates "
                              "WHERE run_id=:r GROUP BY 1,2"), {"r": run_id}).all()
        assert {(d, f): n for d, f, n in rows} == {("fetch", "fetched"): 2, ("defer", "pending"): 1, ("reject", "pending"): 1}
        assert len(res.admitted) == 2
        assert s.execute(text(f"SELECT count(*) FROM {schema}.sources")).scalar() == 2
        joined = s.execute(text(
            f"SELECT c.gate_a_score, c.gate_a_rationale, s.content_hash, s.version FROM {schema}.candidates c "
            f"JOIN {schema}.sources s ON s.candidate_id=c.candidate_id "
            "WHERE c.run_id=:r AND c.gate_a_decision='fetch' AND c.fetch_status='fetched'"), {"r": run_id}).all()
        assert len(joined) == 2 and all(r[1] and len(r[2]) == 64 and r[3] == 1 for r in joined)
        it = s.execute(text(f"SELECT results_count, iteration_no FROM {schema}.search_iterations WHERE run_id=:r"),
                       {"r": run_id}).one()
        assert it == (4, 1)
        assert res.cost_usd > 0.008 and len(res.cost_calls) == 2
        # re-search with identical results adds no candidates/sources
        res2 = run_m2("q", run_id, store, iteration=2, search_fn=search, fetch_fn=fetch,
                      score_fn=lambda q, c, cfg=None, ledger=None: score_candidates(q, c, _llm, cfg=cfg, ledger=ledger))
        assert res2.admitted == []
        assert s.execute(text(f"SELECT count(*) FROM {schema}.candidates")).scalar() == 4
        # changed content under same URL -> version 2 (new run + candidate)
        run2 = s.execute(text(f"INSERT INTO {schema}.runs (question_id) VALUES (:q) RETURNING run_id"), {"q": qid}).scalar()
        def fetch_changed(url, ledger=None):
            return FetchResult(url=url, ok=True, content_text="new " + url, content_hash=sha("new " + url),
                               provenance={})
        res3 = run_m2("q", run2, store, search_fn=search, fetch_fn=fetch_changed,
                      score_fn=lambda q, c, cfg=None, ledger=None: score_candidates(q, c, _llm, cfg=cfg, ledger=ledger))
        assert sorted(c["source_version"] for c in res3.admitted) == [2, 2]
    finally:
        tx.rollback()
        conn.close()
