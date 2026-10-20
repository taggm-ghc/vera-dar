"""Integration: M2-shaped sources in a DB (SQLite standing in for Postgres schema vera_vjay)
-> run_m3 -> rows persisted -> evidence_corpus JSON ready for M4."""
import json

import pytest
import sqlalchemy as sa
from sqlalchemy.pool import StaticPool

from tests.m3_helpers import FakeLLM, domains
from vera.m3 import m3_runner, store

GOOD = "Methods: randomized trial of 16 developers. Result: developers were 19% slower with AI tools."
BAD = "Buy our amazing tool! Everyone is 10x faster, trust us."
EMPTY = "Nothing relevant at all here."


@pytest.fixture
def engine():
    e = sa.create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    with e.connect() as c:
        c.exec_driver_sql("ATTACH DATABASE ':memory:' AS vera_vjay")
        c.commit()
    store.metadata.create_all(e)  # candidates, sources (M2 shape), evidence_spans, appraisals (003 shape)
    with e.begin() as c:
        for i, (t, body) in enumerate([("RCT", GOOD), ("Vendor", BAD), ("Blank", EMPTY)], 1):
            c.execute(sa.insert(store.candidates).values(candidate_id=i, run_id=7, source_url=f"https://x/{i}", title=t))
            c.execute(sa.insert(store.sources).values(source_id=i, candidate_id=i, version=1, content=body,
                                                      content_hash=f"{i:064x}", provenance={"published_date": "2025-07-10"}))
    return e


def llm():
    def extract(user):
        if "randomized trial" in user:
            return {"spans": [{"quote": "developers were 19% slower with AI tools.", "relevance_score": 0.9, "evidence_type": "fact"},
                              {"quote": "randomized trial of 16 developers", "relevance_score": 0.8, "evidence_type": "methodology"}]}
        if "amazing tool" in user:
            return {"spans": [{"quote": "Everyone is 10x faster", "relevance_score": 0.6, "evidence_type": "fact"}]}
        return {"spans": []}

    def appraise(user):
        return domains(bias=1, inc=2, ind=3, imp=1, pub=1, app=3) if "Vendor" in user.split("<excerpts>")[0] else domains()

    return FakeLLM(**{"m3.extract_spans": extract, "m3.appraise": appraise})


def test_end_to_end_persist_and_corpus(engine):
    sources = store.load_sources(engine)
    assert [s["source_id"] for s in sources] == [1, 2, 3]
    corpus = m3_runner.run_m3(sources, engine=engine, llm=llm())

    assert corpus["counts"] == {"sources_in": 3, "admit": 1, "qualify": 0, "reject": 2, "failed": 0}
    assert [s["source_id"] for s in corpus["sources"]] == [1]
    assert {s["source_id"] for s in corpus["excluded"]} == {2, 3}
    json.dumps(corpus)  # clean JSON for M4

    admitted = corpus["sources"][0]
    assert admitted["decision"] == "admit" and admitted["rationale"] and admitted["spans"]
    for sp in admitted["spans"]:  # highlightable exact locations + persisted ids
        assert GOOD[sp["start_index"]:sp["end_index"]] == sp["text"] and sp["span_id"] is not None

    with engine.connect() as c:
        spans = c.execute(sa.select(store.evidence_spans)).all()
        aps = {r.source_id: r for r in c.execute(sa.select(store.appraisals))}
    assert len(spans) == 3 and set(aps) == {1, 2, 3}
    # M1 003 mapping: risk domains stored as concern (5 - rubric score); overall as GRADE label
    assert aps[1].gate_b_decision == "admit" and float(aps[1].bias_score) == 1 and float(aps[1].applicability_score) == 4
    assert aps[1].overall_quality == "high" and aps[1].rubric_version == "grade-casp-v1" and aps[1].policy_version == "gate-b-v1"
    assert "grade-casp-v1" not in aps[1].rationale
    assert aps[2].gate_b_decision == "reject" and aps[3].gate_b_decision == "reject"
    assert aps[3].bias_score is None and aps[3].overall_quality is None  # unknown stays NULL, not 0
    assert corpus["llm_cost"]["calls"] == 5 and corpus["llm_cost"]["by_purpose"]["m3.appraise"]["calls"] == 2


def test_rerun_upserts_spans_appends_appraisals_dry_run_writes_nothing(engine):
    sources = store.load_sources(engine)
    m3_runner.run_m3(sources, engine=engine, llm=llm(), dry_run=True)
    with engine.connect() as c:
        assert c.execute(sa.select(sa.func.count()).select_from(store.appraisals)).scalar() == 0
    c1 = m3_runner.run_m3(sources, engine=engine, llm=llm())
    c2 = m3_runner.run_m3(sources, engine=engine, llm=llm())
    with engine.connect() as c:
        assert c.execute(sa.select(sa.func.count()).select_from(store.appraisals)).scalar() == 6  # history kept
        assert c.execute(sa.select(sa.func.count()).select_from(store.evidence_spans)).scalar() == 3  # not duplicated
    assert [s["span_id"] for s in c1["sources"][0]["spans"]] == [s["span_id"] for s in c2["sources"][0]["spans"]]
    assert store.load_sources(engine, skip_appraised=True) == []
    assert [s["source_id"] for s in store.load_sources(engine, run_id=8)] == []


def test_loaded_source_shape(engine):
    s = store.load_sources(engine, run_id=7)[0]
    assert s["title"] == "RCT" and s["url"] == "https://x/1" and s["content"] == GOOD and s["published_date"] == "2025-07-10"


def test_cost_limit_stops_verbosely(engine):
    corpus = m3_runner.run_m3(store.load_sources(engine), engine=engine, llm=llm(), max_cost_usd=0.0002)
    assert corpus["stopped_early"] and "cost limit" in corpus["stopped_early"]


def test_llm_failure_is_reported_not_swallowed(engine):
    from vera.m3.llm import LLMError

    def boom(user):
        raise LLMError("m3.extract_spans: LLM call failed: timeout")

    corpus = m3_runner.run_m3(store.load_sources(engine), engine=engine,
                              llm=FakeLLM(**{"m3.extract_spans": boom, "m3.appraise": domains()}))
    assert len(corpus["failures"]) == 3 and corpus["failures"][0]["stage"] == "extract"


def test_missing_sources_table_fails_verbosely():
    e = sa.create_engine("sqlite://")
    with e.connect() as c:
        c.exec_driver_sql("ATTACH DATABASE ':memory:' AS vera_vjay")
    with pytest.raises(RuntimeError, match="M1/M2 have not shipped"):
        store.load_sources(e)


def test_db_score_mapping_and_certainty_labels():
    s = store.db_scores({"risk_of_bias": 5, "inconsistency": 1, "indirectness": None, "imprecision": 3,
                         "publication_bias": 4, "applicability": 2})
    assert s == {"bias_score": 0, "inconsistency_score": 4, "indirectness_score": None, "imprecision_score": 2,
                 "publication_bias_score": 1, "applicability_score": 2}
    assert [store.certainty_label(x) for x in (4.0, 3.99, 3.0, 2.0, 1.9, None)] == ["high", "moderate", "moderate", "low", "very_low", None]
