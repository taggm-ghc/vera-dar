"""DARINT end-to-end (synthetic): ingest -> persist (fake engine) -> restore (dar_runtime, injected engine factory) ->
retrieve_all (search stub). 9 documents in 3 domains mirroring the demo (epidemiology forecasting, climate projection
uncertainty, reproducibility/provenance), then ONE document from an unrepresented 4th domain (ecology forecasting).
DAR ON (shipped config); no DB, network or env values."""
import copy
import json
from pathlib import Path

import pytest

from tests.test_dar_store import FakeEngine
from vera import dar_runtime, dar_store, domain_registry, grounded_ask as ga
from vera.dar_corpus import DarCorpus
from vera.dar_store import DarWriter, persist

BASE = Path(__file__).parent / "fixtures" / "dar"
GCFG = {"enabled": True, "providers": ["p"], "num_results": 5, "deadline_s": 5, "snippet_max_chars": 300,
        "resolve": {"enabled": False}}
URL = "postgresql://u:pw@host.invalid:5432/db"


def _docs():
    rows = json.loads((BASE / "demo9.json").read_text()) + json.loads((BASE / "eco1.json").read_text())
    return [{"source_key": r["doc_id"], "url": r["url"], "title": r["title"], "text": r["content"], "year": r["year"],
             "source_type": r["source_type"], "retrieved_at": "2026-10-18", "stored_text_kind": "full_text",
             "admission_passed": True, "db_source_id": 100 + i, "db_canonical_source_id": 200 + i}
            for i, r in enumerate(rows)]


def _words(doc_id, n=8):
    d = next(x for x in _docs() if x["source_key"] == doc_id)
    return " ".join(sorted({w for w in d["text"].lower().split() if w.isalpha() and len(w) > 6})[:n])


def _cfg():
    c = copy.deepcopy(domain_registry.load_config())
    assert c["enabled"] is True  # DAR ON (owner decision OD-DARINT-DAR-ON)
    c["corpus"]["stored_branch_enabled"] = True
    return c


def _handler(counter):
    def h(sql, params):
        if "RETURNING" in sql:
            counter["n"] += 1
            return [(counter["n"],)]
        return []
    return h


def _snapshot(c, docs):
    reg = c.registry
    return {"domains": [{"name": d.name, "name_norm": d.name_norm, "description": d.description, "origin": d.origin}
                        for d in reg.list_domains()],
            "documents": [{k: v for k, v in d.items() if k != "admission_passed"} for d in docs],
            "corpus_assoc": [{"name_norm": d.name_norm, "source_key": x["source_key"], "asserted_by": "x"}
                             for x in docs for d in reg.domains_for_corpus(x["source_key"])],
            "chunk_assoc": [{"name_norm": d.name_norm, "source_key": ch["source_key"], "start": ch["start"],
                             "end": ch["end"], "asserted_by": "x"}
                            for ch in c._chunks for d in reg.domains_for_chunk(ch["span_id"])],
            "chunking": c.chunking()}


@pytest.fixture
def flow(monkeypatch):
    cfg = _cfg()
    docs = _docs()
    live = DarCorpus(cfg)
    eng, ctx, counter = None, {}, {"n": 0}
    eng = FakeEngine(_handler(counter))
    writer = DarWriter(eng)
    results = []
    for d in docs[:-1]:  # the 9 demo documents first
        plan = live.ingest_document(d)
        assert plan["status"] == "indexed"
        results.append(persist(plan, writer, ctx=ctx))
    before = {d.domain_id for d in live.registry.list_domains()}
    plan4 = live.ingest_document(docs[-1])  # ONE document from an unrepresented domain
    results.append(persist(plan4, writer, ctx=ctx))
    # restore through the real runtime path with an injected engine factory and a reader serving the persisted state
    class Eng:
        disposed = 0

        def dispose(self):
            Eng.disposed += 1

    class Reader:
        def __init__(self, engine):
            pass

        def load_snapshot(self, kinds, span_text=None, **kw):
            return _snapshot(live, docs)
    monkeypatch.setattr(dar_store, "DarReader", Reader)
    rcfg = copy.deepcopy(cfg)
    rcfg["runtime"] = {"db_url_env": "VERA_DB_URL_RO", "connect_timeout_s": 1, "statement_timeout_ms": 1000,
                       "failure_retry_ttl_s": 1}
    monkeypatch.setattr(domain_registry, "load_config", lambda *a, **k: rcfg)
    dar_runtime.reset_for_tests()
    restored = dar_runtime.get_corpus(env={"VERA_DB_URL_RO": URL}, engine_factory=lambda url, **kw: Eng())
    yield live, restored, plan4, before, results, eng
    dar_runtime.reset_for_tests()


def test_persist_writes_succeeded_for_every_plan(flow):
    live, restored, plan4, before, results, eng = flow
    assert all(r["status"] == "persisted" for r in results), [r for r in results if r["status"] != "persisted"]
    sqls = " ".join(s for s, _ in eng.calls).upper()
    assert "UPDATE " not in sqls and "DELETE " not in sqls


def test_three_domains_form_unmixed_and_the_fourth_document_gets_a_provisional_domain(flow):
    live, restored, plan4, before, results, eng = flow
    assert restored is not None
    keys = [x["source_key"] for x in _docs()]
    for c in (live, restored):
        prim = lambda d, k: not (c.registry.association_actor("corpus", d.domain_id, k) or ":secondary").endswith(":secondary")  # noqa: E731
        part = {frozenset(k for k in keys if prim(d, k)) for d in c.registry.list_domains()}
        assert part == {frozenset({"D1-1", "D1-2", "D1-3"}), frozenset({"D2-1", "D2-2", "D2-3"}),
                        frozenset({"D3-1", "D3-2", "D3-3"}), frozenset({"E-1"})}, part  # 3 unmixed domains + the 4th
    new = [d for d in live.registry.list_domains() if d.domain_id not in before]
    assert len(new) == 1 and live.domain_status(new[0].domain_id) == "provisional"
    assert any(x["created"] for x in plan4["domains"])


def test_fourth_document_is_retrievable_immediately_through_retrieve_all_after_restore(flow):
    live, restored, plan4, before, results, eng = flow
    q = _words("E-1")
    for corpus in (live, restored):
        out, _ = ga.retrieve_all(q, GCFG, search=lambda _q: [], dar_corpus=corpus)
        assert out and any(s["url"] == "https://fixtures.invalid/e-1" for s in out), [s["url"] for s in out]
        assert all(s["stored_text_kind"] == "full_text" and s["discovered_via"] == "stored_corpus" for s in out)


def test_restored_corpus_matches_live_selection_and_topk(flow):
    live, restored, plan4, before, results, eng = flow
    for doc_id in ("D1-1", "D2-2", "D3-3", "E-1"):
        q = _words(doc_id)
        a, b = live.retrieve(q), restored.retrieve(q)
        assert [x["span_id"] for x in a["candidates"]] == [x["span_id"] for x in b["candidates"]]
        assert a["trace"]["question_domains"] == b["trace"]["question_domains"]
