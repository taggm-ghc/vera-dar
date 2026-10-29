"""FIX-1028-FOLLOWUP item 1: 'newly found' only when a cited passage comes from a document not held before."""
from types import SimpleNamespace as NS

from vera import ask_service as a


def _p(doc, sha):
    return {"document_id": doc, "text_sha256": sha, "text": "t"}


def _ev(*ps):
    return NS(admitted=tuple(ps))


def test_new_doc_ids_empty_merge_reused_and_genuinely_new():
    base = _ev(_p("d1", "a"))
    assert a._new_doc_ids((), base) == set()                           # merge with empty new ids
    assert a._new_doc_ids((_p("d1", "a"),), base) == set()             # de-duplicated passage
    assert a._new_doc_ids((_p("d1", "b"),), base) == set()             # another passage of a held document
    assert a._new_doc_ids((_p("d2", "c"),), base, _ev(_p("d2", "c"))) == set()  # reused candidate (held via ev_acq)
    assert a._new_doc_ids((_p("d1", "a"), _p("d3", "x")), base) == {"d3"}      # genuinely new document


def _run(monkeypatch, base, merged):
    seen = {}
    fd = NS(evidence=base, sources=())
    res = NS(evidence=merged, sources=(), answer="A")
    monkeypatch.setattr(a, "_research_attempt", lambda *x: ({"status": "ok"}, NS(corpus=object())))
    monkeypatch.setattr(a, "_combined_answer", lambda *x: (res, False))
    monkeypatch.setattr(a, "_annotate", lambda r, fr: r)
    monkeypatch.setattr(a, "_with_research", lambda r, rec, suffix=None: seen.update(rec=rec) or r)
    monkeypatch.setattr(a, "_source_freshness", lambda *x: [])
    monkeypatch.setattr(a, "_backfill", lambda *x: None)
    from vera import research_record as rr
    monkeypatch.setattr(rr, "assess", lambda adm, cfg, at=None, new_ids=(): {"counts": {}, "documents": [],
                                                                         "authored_counts": {}})
    monkeypatch.setattr(rr, "sufficient", lambda *x: False)
    monkeypatch.setattr(rr, "triggers", lambda *x: ["insufficient"])
    real = rr.summarize
    monkeypatch.setattr(rr, "summarize", lambda **k: seen.update(used_new=k["used_new"], basis=k["answer_basis"])
                        or real(**k))
    a._freshness_research("q", None, None, None, None, None, None, {"texts": {}}, fd, {}, None)
    return seen


def test_freshness_flow_not_new_when_only_held_documents(monkeypatch):
    base = _ev(_p("d1", "a"))
    rec = _run(monkeypatch, base, _ev(_p("d1", "a"), _p("d1", "b")))
    assert rec["used_new"] is False and rec["basis"] == "generated_existing_only"


def test_freshness_flow_new_when_new_document(monkeypatch):
    base = _ev(_p("d1", "a"))
    rec = _run(monkeypatch, base, _ev(_p("d1", "a"), _p("d2", "b")))
    assert rec["used_new"] is True and rec["basis"] == "generated_new_evidence"


def test_held_document_reacquired_under_new_id_is_not_new():
    base = _ev({"document_id": "d1", "canonical_source_id": "cs1", "text_sha256": "a", "text": "t"})
    re_acq = {"document_id": "acq-9", "canonical_source_id": "cs1", "text_sha256": "z", "text": "t"}
    assert a._new_doc_ids((re_acq,), base) == set()
    other = {"document_id": "acq-10", "canonical_source_id": "cs2", "text_sha256": "y", "text": "t"}
    assert a._new_doc_ids((re_acq, other), base) == {"acq-10"}


def test_reused_stored_acquired_document_is_not_new(monkeypatch):
    from vera import fulldoc_corpus
    corpus = NS(acquired=[{"document_id": "acq-5", "canonical_source_id": "cs5"}])
    monkeypatch.setattr(fulldoc_corpus, "get_acquired_corpus", lambda cfg=None, **k: corpus)
    monkeypatch.setattr(fulldoc_corpus, "load_config", lambda: {})
    held = a._acq_held_ids()
    reused = {"document_id": "acq-5", "canonical_source_id": "cs5", "text_sha256": "q", "text": "t"}
    fresh = {"document_id": "acq-6", "canonical_source_id": "cs6", "text_sha256": "r", "text": "t"}
    assert a._new_doc_ids((reused,), _ev(), held_keys=held) == set()
    assert a._new_doc_ids((reused, fresh), _ev(), held_keys=held) == {"acq-6"}
    monkeypatch.setattr(a, "_acq_held_ids", lambda: held)
    rec = _run(monkeypatch, _ev(_p("d1", "a")), _ev(_p("d1", "a"), reused))
    assert rec["used_new"] is False
