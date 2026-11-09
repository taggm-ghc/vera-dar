"""NONASK-1103 E1: evidence trace projection. Benign synthetic fixtures only (verifier authors hostile/deny-list ones)."""
import json
import re

import pytest

from vera import evidence_trace as et
from vera.fulldoc_evidence import EvidenceBundle

CODES = ("not_acquired", "licence", "attribution_incomplete", "span_mismatch", "not_body_section", "excluded_section",
         "question_coverage", "salient_stem", "low_relevance", "pointer_only", "selector_mismatch", "overlap", "cap_doc",
         "cap_total")
SPAN1 = "Alpha systems repeat their checks daily. Beta systems vary widely across sites; Gamma is rare. Delta ends here."
SPAN2 = "An unrelated sentence about weather."


def ev(**kw):
    adm = ({"span_id": "SPAN-A", "document_id": "DOC-1", "text": SPAN1, "start": 0, "end": len(SPAN1), "relevance": 0.91},
           {"span_id": "SPAN-B", "document_id": "DOC-2", "text": SPAN2, "start": 0, "end": len(SPAN2), "relevance": 0.4})
    base = dict(question="q", examined_full_documents=({"document_id": "DOC-1"}, {"document_id": "DOC-2"}, {"document_id": "DOC-3"}),
                admitted=adm, rejected=({"span_id": "S9", "document_id": "DOC-3", "reason": "low_relevance"},
                                        {"span_id": "S8", "document_id": "DOC-3", "reason": "overlap"},
                                        {"span_id": "S7", "document_id": "DOC-3", "reason": "weird_new_code"}),
                unusable=({"document_id": "DOC-4", "code": "licence", "reason": "x"},), status="admitted")
    base.update(kw)
    return EvidenceBundle(**base)


SOURCES = [{"n": 1, "snippet": SPAN1, "span": "chars=0-10"}, {"n": 2, "snippet": SPAN2}]
RECS = [{"id": 0, "sentence": "Beta systems vary across sites [1].", "verdict": "supported", "citations": [1]},
        {"id": 1, "sentence": "Weather matters [2].", "verdict": "unsupported", "citations": [2]}]
CITES = [{"claim_id": 0, "n": 1, "span_id": "SPAN-A", "quote": "Beta systems vary widely across sites;", "quote_span": "chars=1-2"}]


def build(public=True, **kw):
    a = dict(adequacy=None, inline=None, sources=SOURCES)
    a.update(kw)
    return et.project(ev(), RECS, CITES, public=public, **a)


def test_public_allowlist_and_no_internals(monkeypatch):
    t = build()
    s = json.dumps(t)
    for bad in ("SPAN-A", "DOC-1", "0.91", "span_id", "relevance", "low_relevance", "weird_new_code", "quote_span", "chars="):
        assert bad not in s
    assert [r["n"] for r in t["passages"]] == [1, 2]
    p1, p2 = t["passages"]
    assert p1["cited"] and p1["supports"] == ["Beta systems vary across sites [1]."] and p1["verdict"] == "supported"
    assert p1["excerpt"] in SPAN1
    assert not p2["cited"] and "citation check" in p2["reason"]  # H2 case 1: removed by the check


def test_selected_not_cited_unused_has_no_invented_cause():
    t = et.project(ev(), [], [], public=True, sources=SOURCES)
    assert all(r["reason"] == "the answer did not use this passage" for r in t["passages"])


def test_caveat_gates_public_counts():
    assert "claims" in build()  # an unsupported claim was removed -> caveat
    t = et.project(ev(), RECS[:1], CITES, public=True, sources=SOURCES)
    assert not any(k in t for k in ("claims", "documents", "not_used"))
    t = et.project(ev(), RECS[:1], CITES, adequacy={"verdict": "qualify", "parts": [{"label": "P1", "supported": False}]},
                   public=True, sources=SOURCES)
    assert t["not_used"] and t["documents"] == {"examined_by_corpus": 3, "used": 2}
    assert t["parts"] == [{"label": "P1", "covered": False}] and t["adequacy"]["verdict"] == "qualify"


def test_every_code_maps_to_four_categories_and_unknown_to_other():
    lab = et.load_labels()
    texts = set()
    for c in CODES:
        cid, text = et._category(c, lab)
        assert cid != "other", c
        texts.add(text)
    assert len(texts) == 4
    assert et._category("zzz", lab)[1] == "other reason"
    t = build(adequacy={"verdict": "qualify"})
    assert t["not_used"]["other reason"] == 1


def test_no_answer_outcome_still_gives_decisions_and_reasons(monkeypatch):
    monkeypatch.setattr(et, "_is_public_mode", lambda: False)
    inline = {"candidates": [{"url": "u", "outcome": "rejected", "reason": "licence_rejected"}],
              "discovered": [{"gate_a": {"decision": "defer"}, "outcome": "held"}]}
    t = et.project(ev(), None, None, inline=inline, public=True, sources=SOURCES)
    assert [r["reason"] for r in t["passages"]] == ["the answer did not use this passage"] * 2
    assert "claims" not in t  # no caveat -> counts dev-only
    d = et.project(ev(), None, None, inline=inline, public=False, sources=SOURCES)
    assert d["view"] == "dev" and len(d["gate_a"]) == 2 and len(d["gate_b"]) == 3 and d["claims"]["supported"] == 0


def test_dev_view_keeps_codes_and_is_refused_in_public_mode(monkeypatch):
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    monkeypatch.setattr(et, "_is_public_mode", lambda: False)
    d = build(public=False)
    assert d["passages"][0]["span_id"] == "SPAN-A" and {"stage": "gate_b_passage", "code": "overlap", "category": "repeated"} in d["codes"]
    monkeypatch.setattr(et, "_is_public_mode", lambda: True)
    assert build(public=False) is None
    assert build(public=True) is not None


def test_missing_invalid_or_disabled_labels_fail_closed(tmp_path):
    assert et.project(ev(), RECS, CITES, public=True, labels_path=tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert et.project(ev(), RECS, CITES, public=True, labels_path=bad) is None
    d = json.loads(et.LABELS_PATH.read_text())
    off = tmp_path / "off.json"
    off.write_text(json.dumps(dict(d, enabled=False)))
    assert et.project(ev(), RECS, CITES, public=True, labels_path=off) is None
    d2 = dict(d)
    del d2["other"]
    p = tmp_path / "inc.json"
    p.write_text(json.dumps(d2))
    assert et.load_labels(p) is None


def test_snap_excerpt_sentence_clause_word_and_caps():
    claim = "Beta systems vary"
    q = et.snap_excerpt(SPAN1, claim)
    assert q == "Beta systems vary widely across sites; Gamma is rare."
    # long sentence: clause boundary inside cap, no ellipsis
    long = "First clause is here, second clause follows; " + "word " * 60 + "end."
    q = et.snap_excerpt(long, "first clause", 60)
    assert q in long and len(q) <= 60 and q.endswith(";") and "..." not in q and "…" not in q
    # no boundary: word-boundary trim
    nb = "w" * 10 + " " + "x" * 10 + " " + "y" * 50
    q = et.snap_excerpt(nb, "", 25)
    assert q == "w" * 10 + " " + "x" * 10
    assert et.snap_excerpt("Hi.", "") == "Hi."  # no minimum length


def test_total_cap_strictly_below_1200_and_each_at_most_200():
    sent = ("S%03d " + "z" * 150 + ". ")
    span = "".join(sent % i for i in range(3))
    admitted = tuple({"span_id": f"S{i}", "document_id": f"D{i}", "text": span} for i in range(10))
    srcs = [{"n": i, "snippet": span} for i in range(1, 11)]
    recs = [{"id": i, "sentence": "z", "verdict": "supported", "citations": [i + 1]} for i in range(10)]
    cites = [{"claim_id": i, "n": i + 1, "quote": "x"} for i in range(10)]
    t = et.project(EvidenceBundle(question="q", admitted=admitted, status="admitted"), recs, cites, public=True, sources=srcs)
    qs = [r["excerpt"] for r in t["passages"] if r["excerpt"]]
    assert qs and all(len(q) <= 200 for q in qs) and sum(map(len, qs)) < 1200 and len(qs) < 10


def test_dict_evidence_accepted():
    t = et.project({"admitted": [], "rejected": [], "unusable": []}, None, None, public=True)
    assert t["passages"] == []


def test_es2_public_subquery_allowlist_and_reported_label():
    from vera import adequacy_gate as ag, evidence_trace as et
    subs = [{"label": "A", "declared": [1], "supported": [1]}, {"label": "B", "declared": [1], "supported": []}]
    ad = ag.evaluate([1], [1], subqueries=subs)
    ev = {"admitted": [{"n": 1, "span_id": "s", "text": "Vendor says X. More."}]}
    recs = [{"id": 0, "sentence": "X [1]", "verdict": "supported", "citations": [1]}]
    cites = [{"n": 1, "claim_id": 0, "quote": "Vendor says X."}]
    out = et.project(ev, recs, cites, ad, None, True, sources=[{"n": 1, "snippet": "Vendor says X.", "origin": "reported"}])
    assert out["adequacy"]["subqueries"] == [{"label": "A", "verdict": "finalize"}, {"label": "B", "verdict": "abstain"}]
    assert "not_found" not in str(out["adequacy"]) and "parts" not in out["adequacy"]
    row = out["passages"][0]
    assert row["epistemic"] == "reported" and "reported" in row["verdict"] and row["verdict"] != "supported"
    plain = et.project(ev, recs, cites, ad, None, True, sources=[{"n": 1, "snippet": "Vendor says X."}])
    assert plain["passages"][0]["verdict"] == "supported" and "epistemic" not in plain["passages"][0]
