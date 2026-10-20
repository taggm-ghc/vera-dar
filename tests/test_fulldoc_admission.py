"""Tests for vera.fulldoc_admission (D-055 Option C). Fake corpus duck type; no network/DB."""
import dataclasses

import pytest

from vera.fulldoc_admission import admit
from vera.fulldoc_evidence import EvidenceBundle

ROUTING = {"identification": {"token_pattern": r"[a-z0-9]+", "min_token_len": 3, "stopwords": ["the", "and", "what"]},
           "corpus": {"relevance_stem_len": 5}}
CFG = {"routing": ROUTING, "retrieval": {"candidate_k": 12},
       "admission": {"min_relevance": 0.5, "max_spans_total": 3, "max_spans_per_doc": 2},
       "answer": {"attribution_changes_text": "Short excerpt; converted from {format}."}}
Q = "forecasting rainfall accuracy"
GOOD = "Forecasting rainfall accuracy improved."


def doc(i, lic="allow"):
    return {"document_id": i, "canonical_source_id": f"https://x/{i}", "title": f"T{i}", "url": f"https://x/{i}",
            "doi": None, "published": "2020", "authors": ["A"], "licence": "CC-BY",
            "licence_url": "https://creativecommons.org/licenses/by/4.0/",
            "licence_decision": lic, "attribution": "a", "text_sha256": "h" + i, "chars": 100,
            "evidence_ranges": [[0, 10 ** 9]]}


class Fake:
    def __init__(self, texts, cands, status="ok", licences=None, unusable=None):
        self.status, self.texts, self.cands = status, texts, cands
        self.acquired = [doc(i, (licences or {}).get(i, "allow")) for i in texts]
        self.unusable = unusable or []

    def text(self, i):
        return self.texts[i]

    def retrieve(self, q, k):
        return list(self.cands)

    def inspection(self):
        return {i: {"chunks_inspected": 3} for i in self.texts}


def span(i, sid, start, text, score=1.0, full=None):
    return {"span_id": sid, "document_id": i, "start": start, "end": start + len(text), "text": text, "score": score}


def corp(cands, **kw):
    t = "x" * 10 + GOOD + " " + "y" * 20 + GOOD + " zzz " + GOOD
    return Fake({"d1": t, "d2": "unrelated text here"}, cands, **kw)


def test_examined_not_admitted_subset():
    c = corp([span("d1", "s1", 10, GOOD)])
    b = admit(Q, c, CFG)
    assert b.status == "admitted" and b.grounded
    assert {e["document_id"] for e in b.examined_full_documents} == {"d1", "d2"}
    e = {e["document_id"]: e for e in b.examined_full_documents}
    assert e["d1"]["admitted_spans"] == 1 and e["d2"]["admitted_spans"] == 0 and e["d1"]["chunks_inspected"] == 3
    assert [a["document_id"] for a in b.admitted] == ["d1"]
    a = b.admitted[0]
    assert len(a["text_sha256"]) == 64 and a["relevance"] == 1.0 and a["canonical_source_id"] == "https://x/d1"


def test_low_relevance_insufficient():
    c = corp([span("d1", "s1", 10, "x" * 10)])
    c.texts["d1"] = "x" * 10 + "y" * 5
    c.cands = [span("d1", "s1", 0, "x" * 10)]
    b = admit(Q, c, CFG)
    assert b.status == "insufficient" and not b.grounded and b.examined_full_documents
    assert b.rejected[0]["reason"] == "low_relevance" and b.trace["rejected_by_reason"] == {"low_relevance": 1}


def test_tampered_span_rejected():
    s = span("d1", "s1", 10, GOOD)
    s["text"] = GOOD.replace("improved", "worsened")
    b = admit(Q, corp([s]), CFG)
    assert b.rejected[0]["reason"] == "span_mismatch" and not b.admitted
    s2 = span("d1", "s2", 10, GOOD)
    s2["end"] = 10**6
    assert admit(Q, corp([s2]), CFG).rejected[0]["reason"] == "span_mismatch"


def test_not_acquired_and_bad_dicts():
    b = admit(Q, corp([span("dX", "s1", 0, GOOD), "junk", {"document_id": "d1"}, None]), CFG)
    assert [r["reason"] for r in b.rejected] == ["not_acquired", "span_mismatch", "span_mismatch", "span_mismatch"]
    assert b.status == "insufficient"


def test_licence_not_allow():
    b = admit(Q, corp([span("d1", "s1", 10, GOOD)], licences={"d1": "hold"}), CFG)
    assert b.rejected[0]["reason"] == "licence" and not b.grounded
    assert b.examined_full_documents[0]["licence_decision"] in ("hold", "allow")


def test_caps_and_overlap():
    t = corp([]).texts["d1"]
    p2, p3 = 10 + len(GOOD) + 1 + 20, 10 + len(GOOD) + 1 + 20 + len(GOOD) + 5
    cands = [span("d1", "a", 10, GOOD, 3), span("d1", "b", 10, GOOD[:29], 0.5),  # overlaps a
             span("d1", "c", p2, GOOD, 2), span("d1", "d", p3, GOOD, 1)]
    b = admit(Q, corp(cands), CFG)
    r = {x["span_id"]: x["reason"] for x in b.rejected}
    assert [a["span_id"] for a in b.admitted] == ["a", "c"]
    assert r.get("b") == "overlap" and r.get("d") == "cap_doc"
    cfg = {**CFG, "admission": {**CFG["admission"], "max_spans_total": 1}}
    b = admit(Q, corp([span("d1", "a", 10, GOOD), span("d1", "c", p2, GOOD)]), cfg)
    assert len(b.admitted) == 1 and b.rejected[0]["reason"] == "cap_total"


def test_corpus_unusable():
    un = [{"document_id": "u", "code": "sha", "reason": "bad"}]
    for c in (None, Fake({"d1": "t"}, [], status="corpus_unusable", unusable=un), Fake({}, [], unusable=un)):
        b = admit(Q, c, CFG)
        assert b.status == "corpus_unusable" and not b.examined_full_documents and not b.admitted and not b.grounded
    assert b.unusable == ({"document_id": "u", "code": "sha", "reason": "bad"},)


def test_grounded_truth_table():
    ex, ad = ({"document_id": "d"},), ({"span_id": "s"},)
    G = lambda **k: EvidenceBundle(question="q", **k).grounded
    assert G(examined_full_documents=ex, admitted=ad, status="admitted")
    assert not G(examined_full_documents=ex, admitted=ad, status="insufficient")
    assert not G(examined_full_documents=ex, admitted=(), status="admitted")
    assert not G(examined_full_documents=(), admitted=ad, status="admitted")
    assert not G()
    with pytest.raises(dataclasses.FrozenInstanceError):
        EvidenceBundle(question="q").status = "admitted"


def test_deterministic_and_no_relevance_config_fails_closed():
    c = corp([span("d1", "s1", 10, GOOD)])
    assert admit(Q, c, CFG) == admit(Q, c, CFG)
    b = admit(Q, c, {**CFG, "routing": {}})
    assert b.status == "insufficient"


def test_single_shared_word_never_admitted():  # D-055 lead: min_matched_stems floor (one shared word is not evidence)
    from vera import fulldoc_admission as fa
    from vera import fulldoc_corpus
    r = fa._routing(fulldoc_corpus.load_config())
    icfg, n = r["identification"], int(r["corpus"]["relevance_stem_len"])
    hits, q_n = fa._matched("capital of France", "the capital stock grew", icfg, n)
    assert q_n >= 2 and hits == 1


def test_span_outside_body_sections_never_admitted():  # review B1: title/abstract/back matter are never evidence
    from vera import fulldoc_admission as fa
    text = "x" * 10 + GOOD + " " * 70 + GOOD + " tail"
    i2 = text.index(GOOD, 20)
    c = Fake({"d1": text}, [span("d1", "d1:10-49", 10, text[10:10 + len(GOOD)]),
                           span("d1", "d1:late", i2, text[i2:i2 + len(GOOD)])])
    c.acquired[0]["evidence_ranges"] = [[i2 - 5, len(text)]]
    b = fa.admit(Q, c, CFG)
    assert [a["span_id"] for a in b.admitted] == ["d1:late"]
    assert {r["span_id"]: r["reason"] for r in b.rejected}["d1:10-49"] == "not_body_section"
    del c.acquired[0]["evidence_ranges"]  # no section record: fail closed
    assert fa.admit(Q, c, CFG).admitted == ()


def test_document_without_complete_attribution_is_not_citable():
    from vera import fulldoc_admission as fa
    c = corp([span("d1", "d1:a", 10, GOOD)])
    c.acquired[0]["licence_url"] = None
    b = fa.admit(Q, c, CFG)
    assert b.admitted == () and [r["reason"] for r in b.rejected] == ["attribution_incomplete"]
    c.acquired[0]["licence_url"] = "https://creativecommons.org/licenses/by/4.0/"
    a = fa.admit(Q, c, CFG).admitted[0]["attribution"]
    assert a["creator"] == "A" and a["source"] == "https://x/d1" and a["changes"].endswith("published format.")


def test_partial_question_coverage_is_not_admitted():  # review B2: shared words on a subset of the question
    from vera import fulldoc_admission as fa
    c = corp([span("d1", "d1:a", 10, GOOD)])
    b = fa.admit("forecasting rainfall accuracy in wedding photography studios", c, dict(CFG, admission=dict(
        CFG["admission"], min_relevance=0.3, min_question_coverage=0.8)))
    assert b.admitted == () and "question_coverage" in [r["reason"] for r in b.rejected]
