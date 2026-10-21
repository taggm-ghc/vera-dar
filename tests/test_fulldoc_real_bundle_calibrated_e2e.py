"""DR-FD: offline end-to-end of the full-document pathway over the REAL DAR-9 bundle with the CALIBRATED admission
config (work/fda-cal/CALIBRATION.md) applied as an IN-TEST override; config/fulldoc_answer.json is never edited.
Stub answering model and stub claim checker only; no provider call, no network, no DB. Skips unless the runner mounted
the bundle (VERA_TEST_RO_BUNDLE=1).

LIMITATIONS (not general research readiness): under this calibration 2 of the 16 answerable calset questions are
admitted (c02, c06); 6 never retrieved a candidate; the stub model's text is not live synthesis and the stub claim
check is not the live checker. Questions are copied from the calset (the calset is not mounted in the runner)."""
import copy
import json
import os
from pathlib import Path

import pytest

from vera import ask_service, claim_check, fulldoc_answer, fulldoc_corpus, scope_router
from vera.ask_service import AskResult
from vera.pricing.config import PricingRecord

REAL = os.environ.get("VERA_REAL_DAR9_BUNDLE") or ""
pytestmark = pytest.mark.skipif(not REAL or not (Path(REAL) / "manifest-filled.https.json").is_file(),
                                reason="UNVERIFIED: real bundle not mounted")

_REAL_GET_CORPUS = fulldoc_corpus.get_corpus
PRICING = PricingRecord(provider="p", model="m", input=0.0, output=0.0, source_url="u",
                        retrieved_at="2026-01-01", effective_from="2026-01-01")
GROUNDING = {"insufficient_text": "INSUFFICIENT", "removed_citation_note": "NOTE", "no_sources_text": "NO SOURCES",
             "claim_check": {"enabled": True, "unavailable_text": "UNAVAILABLE"}}
ADMIT = {
    "c02": "In Gostic et al., how does the size of the smoothing window trade off accuracy in Rt estimates?",
    "c06": "In Petchey et al., what is a forecast proficiency threshold?",
}
NEG = {
    "c20": "Which software container tool do Wilson et al. recommend for packaging an analysis environment?",
    "c21": "Do Bracher et al. recommend a specific preregistration platform for forecast evaluation?",
    "c24": "What are the DOI and volume number of the paper that introduced the FAIR Guiding Principles?",
}
STUB_REPLY = "The cited passage addresses the question [1]."


def calibrated_cfg():
    cfg = copy.deepcopy(fulldoc_corpus.load_config())
    cfg["fulldoc"] = {"enabled": True}
    adm = cfg["admission"]
    adm.update({"signal": "idf_weighted", "min_relevance": 0.4, "min_matched_stems": 2, "max_spans_per_doc": 2})
    adm["idf_signal"].update({"min_weighted_coverage": 0.70, "salient_k": 2, "min_relevance": 0.40})
    adm["pointer_filter"]["enabled"] = True
    cfg["retrieval"].update({"candidate_k": 12, "evidence_only": True})
    return cfg


@pytest.fixture(scope="module")
def world():
    cfg = calibrated_cfg()
    corpus = fulldoc_corpus.load_bundle(Path(REAL), cfg)  # shipped pins apply: a wrong bundle is unusable
    assert corpus.status == "ok" and len(corpus.acquired) == 9 and not corpus.unusable
    texts = {d["document_id"]: corpus.text(d["document_id"]) for d in corpus.acquired}
    return cfg, corpus, texts


def run(mp, world, q, reply=STUB_REPLY):
    cfg, corpus, texts = world
    calls = []
    mp.setattr(fulldoc_corpus, "load_config", lambda *a, **k: cfg)
    mp.setattr(fulldoc_corpus, "get_corpus",  # the real get_corpus (flag, env, cache, "bundle" label); only the env is injected
               lambda *a, **k: _REAL_GET_CORPUS(cfg, env={cfg["bundle_dir_env"]: REAL}))
    mp.setattr(scope_router, "route", lambda *a, **k: scope_router.ScopeDecision(True, "layer1", "ok"))

    def complete(messages, model, pricing, **kw):
        calls.append(messages)
        return AskResult(reply, 0, 0.0)

    cc_seen = []

    def check_answer(t, sources, cc, *a, **k):
        recs = claim_check.RECORDS.get()
        kept = []
        for c in claim_check.split_claims(t, len(sources), 30):
            recs.append({"id": c["id"], "sentence": c["text"], "citations": c["cites"], "verdict": "supported"})
            cc_seen.append({"id": c["id"], "cites": c["cites"], "verdict": "supported"})
            kept.append(c["text"])
        return " ".join(kept), {"outcome": "checked", "checked": len(kept), "supported": len(kept), "partial": 0, "removed": 0}

    mp.setattr(ask_service, "complete", complete)
    mp.setattr(claim_check, "check_answer", check_answer)
    res = ask_service.answer_in_scope(q, "m", PRICING, grounding=(lambda: [], GROUNDING))
    return res, calls, cc_seen


def context_passages(messages):
    user = next(m for m in messages if m["role"] == "user")
    return json.loads(user["content"])["passages"]


@pytest.mark.parametrize("qid", sorted(ADMIT))
def test_calibrated_admits_and_grounds_on_exact_bundle_slices(world, monkeypatch, qid):
    cfg, corpus, texts = world
    res, calls, cc = run(monkeypatch, world, ADMIT[qid])
    ev = res.evidence
    assert ev.status == "admitted" and ev.corpus_source == "bundle" and ev.grounded
    assert 1 <= len(ev.admitted) <= 2 * 1 + 4 and all(a["start"] < a["end"] for a in ev.admitted)
    per_doc = {}
    for a in ev.admitted:
        per_doc[a["document_id"]] = per_doc.get(a["document_id"], 0) + 1
        assert a["text"] == texts[a["document_id"]][a["start"]:a["end"]]  # admitted text is an exact bundle slice
    assert max(per_doc.values()) <= 2
    assert len(calls) == 1
    passages = context_passages(calls[0])
    assert [p["text"] for p in passages] == [a["text"] for a in ev.admitted]  # model context == admitted slices, in order
    assert res.result_kind == "answer" and res.contract["fallback"] is None
    assert res.claim_check["outcome"] == "checked" and cc and all(c["verdict"] == "supported" for c in cc)
    assert res.citations
    by_n = {s["n"]: s for s in res.sources}
    for c in res.citations:
        s = by_n[c["n"]]
        a = next(x for x in ev.admitted if x["span_id"] == s["span_id"])
        assert c["span_id"] == s["span_id"] and c["span"] == f"chars={a['start']}-{a['end']}"
        assert c["quote"] and c["quote"] in texts[a["document_id"]][a["start"]:a["end"]]
    public = fulldoc_answer.public_evidence(ev, res.citations)
    assert public["corpus_source"] == {"kind": "bundle"}
    print("FDREC " + json.dumps({
        "q": qid, "outcome": res.result_kind, "fallback": res.contract["fallback"], "status": ev.status,
        "corpus_source": ev.corpus_source,
        "admitted": [{"span_id": a["span_id"], "doc": a["document_id"], "start": a["start"], "end": a["end"],
                      "relevance": a.get("relevance"), "context_exact_slice": p["text"] == texts[a["document_id"]][a["start"]:a["end"]]}
                     for a, p in zip(ev.admitted, passages)],
        "citations": [{"n": c["n"], "span_id": c["span_id"], "span": c["span"], "quote": c["quote"][:60]} for c in res.citations],
        "claim_check": {"outcome": res.claim_check["outcome"], "claims": cc}}))


@pytest.mark.parametrize("qid", sorted(NEG))
def test_calibrated_negatives_come_back_insufficient_without_generation(world, monkeypatch, qid):
    cfg, corpus, texts = world
    res, calls, cc = run(monkeypatch, world, NEG[qid])
    ev = res.evidence
    assert ev.status == "insufficient" and ev.corpus_source == "bundle" and ev.admitted == ()
    assert calls == [] and cc == []  # no model call, no claim check
    assert res.result_kind == "reading_list" and res.contract["fallback"] == "insufficient_full_document_evidence"
    assert not res.citations
    print("FDREC " + json.dumps({"q": qid, "outcome": res.result_kind, "fallback": res.contract["fallback"],
                                  "status": ev.status, "corpus_source": ev.corpus_source, "admitted": [],
                                  "rejected_reasons": sorted({r.get("reason") for r in ev.rejected})}))


def test_shipped_config_file_is_not_the_calibrated_one():
    """The calibration is an in-test override only: the shipped file still has fulldoc.enabled false."""
    shipped = fulldoc_corpus.load_config()
    assert shipped["fulldoc"]["enabled"] is False
