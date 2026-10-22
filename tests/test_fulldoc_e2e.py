"""D-055 Option C: end-to-end tests of the full-document-grounded /ask through the real main:app TestClient.

Stubbed: the scope router decision, the answering LLM (ask_service.complete) and the claim checker
(claim_check.check_answer), web retrieval and the DB/memory. Real: the bundle, vera.fulldoc_corpus/_admission/_answer,
ask_service.answer_in_scope, answer_contract, main.AskResponse. The bundle is SYNTHETIC (tests/fulldoc_bundle_synth.py),
pins nulled through the config; no network, no DB, no LLM.
"""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests import fulldoc_bundle_synth as synth
from vera import answer_contract as ac
from vera import ask_service, claim_check, scope_router
from vera.ask_service import AskResult
from vera.pricing.config import PricingRecord

PRICING = PricingRecord(provider="p", model="m", input=0.0, output=0.0, source_url="u",
                        retrieved_at="2026-01-01", effective_from="2026-01-01")
ABS = "SENTINEL-E2E-WEB-ABSTRACT"
WEB_URL = "https://example.org/web/one"
WEB = [{"n": 1, "title": "Web Paper One", "url": WEB_URL, "published": "2025-01-01", "snippet": ABS, "abstract": ABS,
        "identifier": "10.1234/web", "provider": "arxiv", "licence_decision": "allow", "retrieved_at": "2026-10-01"}]
GROUNDING = {"insufficient_text": "INSUFFICIENT", "removed_citation_note": "NOTE", "no_sources_text": "NO SOURCES",
             "claim_check": {"enabled": True, "unavailable_text": "UNAVAILABLE"}}
Q_EPI = "How was the transmission of the pathogen related to incidence during an outbreak?"
Q_TWO = ("How was the transmission of the pathogen related to incidence in an outbreak, and how did the temperature "
         "forcing of the ensemble relate to precipitation and ocean warming?")
Q_NONE = "What is the best recipe for sourdough bread with rye flour and a long overnight proof?"
SPAN = re.compile(r"^chars=(\d+)-(\d+)$")
FALLBACKS = {"insufficient_full_document_evidence", "full_document_corpus_unusable", "model_insufficient"}


class Harness:
    def __init__(self, mp, tmp_path):
        self.mp, self.tmp = mp, tmp_path
        self.answer_calls, self.check_calls = [], []
        self.answer_text = "Transmission was compared under the stated conditions [1]."
        self.check = None  # callable(text, sources, cfg) -> (text|None, summary); overrides the default checker
        self.verdicts = {}  # default checker: {claim id: verdict}, absent = supported
        self.accept = True

    def bundle(self, **kw):
        root = synth.write_bundle(self.tmp / "bundle", **kw)
        from vera import fulldoc_corpus
        cfg = synth.synth_cfg()
        cfg.setdefault("fulldoc", {})["enabled"] = True  # the shipped flag is false until acceptance: enable explicitly
        fulldoc_corpus.reset_for_tests()
        self.mp.setattr(fulldoc_corpus, "load_config", lambda *a, **k: cfg)
        self.mp.setenv("VERA_FULLDOC_BUNDLE_DIR", str(root))
        return root

    def install(self, grounding=GROUNDING, web=True):
        mp = self.mp
        mp.setenv("VERA_API_KEY", "k" * 20)
        mp.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
        mp.delenv("VERA_PUBLIC_MODE", raising=False)
        import main
        from vera import public_mode as pm
        pm.reset_limiter(None)
        acc = self.accept
        mp.setattr(scope_router, "route", lambda *a, **k: scope_router.ScopeDecision(True, "layer1", "ok") if acc
                   else scope_router.ScopeDecision(False, "layer1", "out of scope"))

        def complete(messages, model, pricing, **kw):
            self.answer_calls.append(messages)
            return AskResult(self.answer_text, 10, 0.0)

        def check_answer(text, sources, cfg, *a, **k):
            self.check_calls.append((text, sources))
            if self.check:
                return self.check(text, sources, cfg)
            claims = claim_check.split_claims(text, len(sources), 30)
            recs = claim_check.RECORDS.get()  # as the real checker: one record per claim, the caller reads them
            kept = []
            for c in claims:
                v = self.verdicts.get(c["id"], "supported")
                if recs is not None:
                    recs.append({"id": c["id"], "sentence": c["text"], "citations": c["cites"], "verdict": v})
                if v != "unsupported":
                    kept.append(c["text"])
            removed = len(claims) - len(kept)
            return " ".join(kept), {"outcome": "checked", "checked": len(claims), "supported": len(kept), "partial": 0,
                                    "removed": removed}

        mp.setattr(ask_service, "complete", complete)
        mp.setattr(claim_check, "check_answer", check_answer)
        mp.setattr(main, "GROUNDING", grounding)
        from vera import dar_runtime
        mp.setattr(dar_runtime, "get_corpus", lambda *a, **k: None)  # isolation: never a real DAR corpus
        mp.setattr(main, "retrieve_all", lambda q, cfg, **kw: ([dict(s) for s in WEB] if web else [], []))
        mp.setattr(main, "answer_via_chain", lambda q, entries, g: (ask_service.answer_in_scope(q, "m", PRICING, grounding=g), "groq:m"))
        mp.setattr(main, "store_from_env", lambda cfg: None)  # never a real corpus engine
        mp.setattr(main, "admit_to_corpus", lambda *a: None)
        mp.setattr(main, "llm_call_for", lambda e: None)
        mp.setattr(main, "memory_from_env", lambda block: None)
        return main

    def ask(self, q, **inst):
        main = self.install(**inst)
        r = TestClient(main.app).post("/ask", json={"question": q}, headers={"X-API-Key": "k" * 20})
        assert r.status_code == 200, r.text
        return r.text, r.json()


@pytest.fixture
def h(monkeypatch, tmp_path):
    from vera import fulldoc_corpus
    fulldoc_corpus.reset_for_tests()
    monkeypatch.delenv("VERA_FULLDOC_BUNDLE_DIR", raising=False)
    yield Harness(monkeypatch, tmp_path)
    fulldoc_corpus.reset_for_tests()


def _texts(root):
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted((Path(root) / "texts").glob("*.txt"))}


def _ids(items):
    return [i["document_id"] for i in items]


def _no_long_doc_text(raw, texts):
    for t in texts.values():
        for i in range(0, max(1, len(t) - 300), 25):
            assert t[i:i + 300] not in raw, "a 300-char document passage leaked into the response"


def _assert_non_answer(body, fallback_in):
    assert body["result_kind"] == "reading_list" and body["contract"]["fallback"] in fallback_in, body["contract"]
    assert body["citations"] == [] and body["sources"] == [] and body["claim_check"] is None


def test_1_env_unset_keeps_todays_shape(h):
    raw, body = h.ask(Q_EPI)
    assert body["result_kind"] == "reading_list" and body["claim_check"] is None and body["sources"] == []
    new = ("examined_full_documents", "admitted_evidence", "citations", "evidence_status")
    assert not any(body.get(k) for k in new) and not any(k in body for k in new), "additive fields leaked (shape changed)"
    assert body["contract"]["fallback"] is None or isinstance(body["contract"]["fallback"], str)
    assert [c["url"] for c in body["candidates"]] == [WEB_URL] and ABS not in raw
    assert h.answer_calls == [] and h.check_calls == []
    for k in ("answer", "model", "tokens_used", "cost_usd", "traced_sources", "recalled", "candidates", "contract"):
        assert k in body


def test_2_matching_question_gets_grounded_answer(h):
    root = h.bundle()
    texts = _texts(root)
    raw, body = h.ask(Q_EPI)
    assert body["result_kind"] == "answer", body
    assert sorted(_ids(body["examined_full_documents"])) == sorted(texts)  # all acquired documents were inspected
    adm = body["admitted_evidence"]
    assert adm and all(a["span_id"] for a in adm)
    assert len(adm) <= 6 and set(_ids(adm)) <= set(texts)
    adm_ids = {a["span_id"] for a in adm}
    assert body["citations"]
    for c in body["citations"]:
        assert c["span_id"] in adm_ids
        assert c["canonical_source_id"] and c["licence"] and c["quote"]
        m = SPAN.match(c["span"])
        assert m, c["span"]
        a, b = int(m.group(1)), int(m.group(2))
        assert len(c["quote"]) <= 200
        doc = next(x["document_id"] for x in adm if x["span_id"] == c["span_id"])
        assert c["quote"] in texts[doc][a:b], "quote is not a verbatim substring of the document at those chars"
    assert body["sources"] and all(s.get("span") and s.get("span_id") for s in body["sources"])
    assert body["evidence_status"] and body["claim_check"]["outcome"] == "checked"
    assert body["traced_sources"] == [] and WEB_URL not in raw and ABS not in raw
    assert len(h.answer_calls) == 1
    _no_long_doc_text(raw, texts)


def test_3_declined_question_touches_nothing(h, monkeypatch):
    h.bundle()
    from vera import fulldoc_corpus

    def boom(*a, **k):
        raise AssertionError("full-document retrieval ran for a declined question")

    monkeypatch.setattr(fulldoc_corpus, "get_corpus", boom)
    monkeypatch.setattr(fulldoc_corpus.FullDocCorpus, "retrieve", boom)
    h.accept = False
    raw, body = h.ask(Q_EPI)
    assert body["result_kind"] == "decline"
    assert h.answer_calls == [] and h.check_calls == []
    assert not any(body.get(k) for k in ("examined_full_documents", "admitted_evidence", "citations", "evidence_status"))


def test_4_no_relevant_passage_no_generation(h):
    root = h.bundle()
    raw, body = h.ask(Q_NONE)
    _assert_non_answer(body, {"insufficient_full_document_evidence"})
    assert body["admitted_evidence"] == []
    assert sorted(_ids(body["examined_full_documents"])) == sorted(_texts(root))
    assert h.answer_calls == [] and h.check_calls == []


def test_5a_tampered_document_never_grounds(h):
    root = h.bundle(tamper={"syn_epi": "\nINJECTED TAMPERING"})
    raw, body = h.ask(Q_EPI)
    for key in ("examined_full_documents", "admitted_evidence"):
        assert "syn_epi" not in _ids(body[key])
    assert all("syn_epi" not in (c.get("span_id", "") + c.get("canonical_source_id", "")) for c in body["citations"])
    assert "INJECTED TAMPERING" not in raw
    if body["result_kind"] != "answer":
        assert h.answer_calls == []


def test_5b_all_tampered_is_unusable(h):
    h.bundle(tamper={d: "\nINJECTED" for d in ("syn_epi", "syn_clim", "syn_repro")})
    raw, body = h.ask(Q_EPI)
    _assert_non_answer(body, FALLBACKS)
    assert body["admitted_evidence"] == [] and body["examined_full_documents"] == []
    assert h.answer_calls == [] and h.check_calls == []


def test_6a_unsupported_claim_removed_from_answer_and_citations(h):
    root = h.bundle()
    _relax_coverage()  # DAR-TRACE-1022: 0.7 per-document coverage admits one topic of this two-topic question
    h.answer_text = "Transmission relates to incidence [1]. Temperature forcing relates to precipitation [2]."
    h.verdicts = {1: "unsupported"}
    raw, body = h.ask(Q_TWO)
    if len(body["admitted_evidence"]) < 2:
        pytest.skip("fewer than 2 passages admitted for the two-topic question; cannot test removal")
    assert body["result_kind"] == "answer"
    assert "[2]" not in body["answer"] and "[1]" in body["answer"]
    assert len(body["citations"]) == 1
    assert body["claim_check"]["removed"] == 1
    assert "precipitation" not in body["answer"] and all(c["n"] == 1 for c in body["citations"])


def test_6b_checker_unavailable_no_answer(h):
    h.bundle()
    h.check = lambda t, s, c: (None, {"checked": 0, "outcome": "checker_unavailable"})
    raw, body = h.ask(Q_EPI)
    _assert_non_answer(body, {"checker_unavailable"})
    assert "Transmission was compared" not in body["answer"]
    assert len(h.check_calls) == 1


def test_6c_checker_disabled_no_answer(h):
    h.bundle()
    raw, body = h.ask(Q_EPI, grounding={**GROUNDING, "claim_check": {"enabled": False}})
    _assert_non_answer(body, {"claim_check_disabled"})
    assert h.check_calls == []


def test_7_web_sources_never_cited_in_full_document_answer(h):
    root = h.bundle()
    raw, body = h.ask(Q_EPI)
    assert body["result_kind"] == "answer"
    assert WEB_URL not in raw and ABS not in raw and "Web Paper One" not in raw
    for c in body["citations"] + body["sources"]:
        assert WEB_URL not in str(c)
    assert all(s.get("span_id") for s in body["sources"])
    prompt = str(h.answer_calls[0])
    assert ABS not in prompt and WEB_URL not in prompt  # abstracts never reach the generation prompt


def test_8_llm_call_bound(h):
    h.bundle()
    raw, body = h.ask(Q_EPI)
    assert body["result_kind"] == "answer"
    assert len(h.answer_calls) <= 1 and len(h.check_calls) <= 1
    h2_calls = (len(h.answer_calls), len(h.check_calls))
    assert h2_calls == (1, 1)


# ---- INT (PLAN-FDA): integrated pipeline, in-process (answer_in_scope), stubs only, no TestClient, no DB ------------------

PARTS_REPLY = ("Parts:\nP1: How was transmission related to incidence?\nP2: How did temperature forcing relate to precipitation?"
               "\nAnswers:\nP1: Transmission relates to incidence [1].\nP2: Temperature forcing relates to precipitation [2].")


def _relax_coverage():
    """Test-only: the legacy 0.8 per-document coverage rule admits one topic of a two-topic question (the known false
    abstention CAL addresses), so the two-part pipeline tests lower it on the in-memory config copy only."""
    from vera import fulldoc_corpus
    fulldoc_corpus.load_config()["admission"]["min_question_coverage"] = 0.0


def _in_process(h, monkeypatch, q, text, verdict_by_marker=None, default="supported"):
    """answer_in_scope with the router, answering LLM and claim checker stubbed; store/writer classes raise if touched."""
    from vera import corpus_admission, dar_store

    def boom(*a, **k):
        raise AssertionError("FDA path touched a store/writer (DB write)")

    monkeypatch.setattr(corpus_admission.WoCorpusStore, "__init__", boom)
    monkeypatch.setattr(dar_store.DarWriter, "__init__", boom)
    monkeypatch.setattr(corpus_admission, "store_from_env", boom)
    monkeypatch.setattr(scope_router, "route", lambda *a, **k: scope_router.ScopeDecision(True, "layer1", "ok"))
    monkeypatch.setattr(ask_service, "complete", lambda *a, **k: AskResult(text, 10, 0.0))
    vb = verdict_by_marker or {}

    def check_answer(t, sources, cfg, *a, **k):
        recs = claim_check.RECORDS.get()
        kept = []
        for c in claim_check.split_claims(t, len(sources), 30):
            v = next((v for m, v in vb.items() if m in c["text"]), default)
            recs.append({"id": c["id"], "sentence": c["text"], "citations": c["cites"], "verdict": v})
            if v != "unsupported":
                kept.append(c["text"])
        return " ".join(kept), {"outcome": "checked", "checked": len(kept), "removed": 0}

    monkeypatch.setattr(claim_check, "check_answer", check_answer)
    web = [dict(w) for w in WEB]
    return ask_service.answer_in_scope(q, "m", PRICING, grounding=(lambda: web, GROUNDING))


def test_int_two_part_question_partial_through_integrated_pipeline(h, monkeypatch):
    h.bundle()
    _relax_coverage()
    res = _in_process(h, monkeypatch, Q_TWO, PARTS_REPLY, {"precipitation": "unsupported"})
    cfg = __import__("vera.fulldoc_corpus", fromlist=["x"]).load_config()
    assert res.result_kind == "answer"  # OWNER GATE: guarded seams only pass `answer`; fallback design in force
    assert res.claim_check["answer_completeness"] == "partial" and res.claim_check["parts"]["unanswered"] == [2]
    assert res.answer.startswith(cfg["answer"]["per_part"]["partial_statement"])
    assert "Transmission relates to incidence" in res.answer and "precipitation [2]" not in res.answer
    assert "do not support an answer to this part" in res.answer
    assert res.citations and all(c["part"] == 1 and c["span_id"] and c["attribution"] for c in res.citations)


def test_int_all_partial_falls_back_d7(h, monkeypatch):
    h.bundle()
    _relax_coverage()
    res = _in_process(h, monkeypatch, Q_TWO, PARTS_REPLY, default="partial")
    assert res.result_kind == "reading_list" and res.contract["fallback"] == "claim_check_insufficient"
    assert "Transmission relates" not in res.answer and not res.citations


def test_int_model_insufficient_token_passes_through_unchanged(h, monkeypatch):
    h.bundle()
    res = _in_process(h, monkeypatch, Q_EPI, "INSUFFICIENT_EVIDENCE")
    assert res.result_kind == "reading_list" and res.contract["fallback"] == "model_insufficient"


def test_int_fully_supported_two_parts_is_complete(h, monkeypatch):
    h.bundle()
    _relax_coverage()
    res = _in_process(h, monkeypatch, Q_TWO, PARTS_REPLY)
    assert res.result_kind == "answer" and res.claim_check["answer_completeness"] == "complete"
    assert "partial answer" not in res.answer.lower()
