"""K2.T (R-T, closure-K2-T3.md §2.5 / §7.5): integrated contract test.

Each case drives POST /ask through TestClient(main.app), then feeds the exact JSON body into the Streamlit page
(AppTest, requests.post stubbed). Where the kind is reachable through vera.ask_service, the fake chain calls the real
answer_in_scope (router, retrieval, model and claim checker stubbed; the L3 seam opened only where stated), so the
C-F1 fallbacks run through the real code. Memory is a real MemoryService with fake reader and store (V3 is the real
control). No DB, no network: store_from_env, memory_from_env, admit_to_corpus and llm_call_for are stubbed.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from tests.public_terms import find_terms
from vera import answer_contract as ac
from vera import ask_service
from vera import memory_store as ms
from vera import sources_sidebar as sb
from vera.ask_service import AskResult
from vera.grounded_ask import public_candidates
from vera.memory_gate import load_gate_config

ROOT = Path(ask_service.__file__).resolve().parent.parent
STORE_CFG = json.loads((ROOT / "config" / "ask-provider-chain.json").read_text())["memory"]["store"]
ABS = "SENTINEL-K2T-ABSTRACT"          # abstract/snippet text: never in any body or page
CHECK = "SENTINEL-K2T-CLAIMCHECK"      # claim_check payload: only in the answer body
RECALL = "SENTINEL-K2T-RECALLED"       # evidence recall rows: never returned (V3 off)
TRACED = "SENTINEL-K2T-TRACED"         # traced (unused) source title: only in the answer body
SRC = [{"n": 1, "title": "Paper One", "url": "https://example.org/papers/one", "published": "2025-01-01",
        "snippet": ABS, "abstract": ABS, "identifier": "10.1234/one", "provider": "arxiv",
        "licence_decision": "allow", "retrieved_at": "2026-10-01"}]
UNUSED = [{"title": TRACED, "url": "https://doi.org/10.1/t", "published": None, "discovered_via": "x"}]
GROUNDING_CFG = {"insufficient_text": "INSUFFICIENT", "removed_citation_note": "NOTE", "no_sources_text": "NO SOURCES",
                 "claim_check": {"enabled": True, "unavailable_text": "UNAVAILABLE"}}
NOTE = {"kind": "discovery_note", "note": "Discovery note (not evidence). Earlier list", "urls": ["https://a.org/n"],
        "saved_at": "2026-10-01"}

CF1 = ("cf1_no_valid_citations", "cf1_checker_unavailable", "cf1_claim_check_insufficient")
NON_ANSWER = ("reading_list", "reading_list_empty", "decline", "unspecified", "discovery", "bogus", "missing") + CF1
CASES = NON_ANSWER + ("answer",)


class Reader:
    def __init__(self): self.recall_calls = 0
    def recall(self, q): self.recall_calls += 1; return [{"claim_text": RECALL, "claim_id": 1}]
    def recall_discovery(self, q): return [dict(NOTE)]
    def pending_count(self): return 0
    def duplicate(self, text, spans): return False
    def answer(self): return 1


class Store:
    def __init__(self): self.notes = []
    def answer_id(self): return 1
    def add_discovery_note(self, answer_id, text): self.notes.append(text); return 1


def _pricing():
    from tests.test_item79_claim_check import PRICING
    return PRICING


def _setup(monkeypatch, case):
    """Stub the router, retrieval, model and checker for `case`; return the fake chain for main.answer_via_chain."""
    from vera import claim_check, scope_router
    accept = scope_router.ScopeDecision(True, "layer1", "ok")
    monkeypatch.setattr(scope_router, "route", lambda *a, **k: accept if case != "decline"
                        else scope_router.ScopeDecision(False, "layer1", "out of scope"))
    monkeypatch.setattr(ask_service, "complete", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model call")))
    if case in CF1 or case == "answer":  # open the L3 seam (test only; the shipped stub returns [])
        monkeypatch.setattr(ac, "examined_full_documents", lambda c: ["examined"])
        text = "No citation here." if case == "cf1_no_valid_citations" else "Faster [1]."
        monkeypatch.setattr(ask_service, "complete", lambda *a, **k: AskResult(text, 10, 0.0))
        if case == "answer":
            checked = ("Faster [1].", {"outcome": "checked", "checked": 1, "supported": 1, "partial": 0,
                                       "removed": 0, "detail": CHECK})
        else:
            checked = (None, {"checked": 0, "detail": CHECK,
                              "outcome": "checker_unavailable" if case == "cf1_checker_unavailable" else "insufficient"})
        monkeypatch.setattr(claim_check, "check_answer", lambda t, s, c: checked)
    if case in ("unspecified", "discovery", "bogus", "missing"):  # kinds not produced by ask_service: direct results
        kw = {} if case in ("unspecified", "missing") else {"result_kind": case}

        def chain(q, entries, grounding):
            grounding[0]()  # fills main's traced-source cache, as a real chain would
            return AskResult("SERVER-TEXT", 1, 0.0, sources=tuple(SRC), claim_check={"detail": CHECK}, accepted=True,
                             claims=({"id": 0, "sentence": "s", "citations": [1], "verdict": "supported"},),
                             candidates=public_candidates(SRC), contract={"layers": ["L0"], "fallback": None},
                             **kw), "groq:m"
        return chain

    def chain(q, entries, grounding):
        return ask_service.answer_in_scope(q, "m", _pricing(), grounding=grounding), "groq:m"
    return chain


def _post(monkeypatch, case):
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)  # fake value; the chain is stubbed
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    import main
    from vera import public_mode as pm
    pm.reset_limiter(None)
    srcs = [] if case == "reading_list_empty" else SRC
    monkeypatch.setattr(main, "GROUNDING", GROUNDING_CFG)
    monkeypatch.setattr(main, "retrieve_all", lambda q, cfg: ([dict(s) for s in srcs], [dict(u) for u in UNUSED]))
    monkeypatch.setattr(main, "answer_via_chain", _setup(monkeypatch, case))
    admitted = []
    monkeypatch.setattr(main, "store_from_env", lambda cfg: None)  # never a real corpus engine (TEST-DB-PROD)
    monkeypatch.setattr(main, "admit_to_corpus", lambda *a: admitted.append(a))
    monkeypatch.setattr(main, "llm_call_for", lambda e: None)
    reader, store = Reader(), Store()
    svc = ms.MemoryService(dict(STORE_CFG), load_gate_config(), reader, store)
    monkeypatch.setattr(main, "memory_from_env", lambda block: svc)
    r = TestClient(main.app).post("/ask", json={"question": "What is VERA-DAR?"}, headers={"X-API-Key": "k" * 20})
    assert r.status_code == 200, r.text
    body = r.json()
    if case == "missing":  # an older API without result_kind: the page must still fail closed
        body.pop("result_kind")
    return r.text, body, reader


def _page(monkeypatch, body):
    monkeypatch.setenv("VERA_PUBLIC_MODE", "1")
    monkeypatch.setenv("VERA_API_KEY", "k")
    monkeypatch.setattr(sb, "_corpus", lambda st: None)
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: body))
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(json=lambda: {"commit": "abcdef1234"}))
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    at.text_input[0].set_value("What is VERA-DAR?")
    at.button[0].click()
    at = at.run()
    assert not at.exception
    return at


def _texts(at, *names):
    return [str(e.value) for n in names for e in getattr(at, n)]


ALL = ("markdown", "caption", "info", "warning", "success", "error")


@pytest.mark.parametrize("case", CASES)
def test_ask_body_into_page(monkeypatch, case):
    raw, body, reader = _post(monkeypatch, case)
    at = _page(monkeypatch, body)
    page = "\n".join(_texts(at, *ALL))
    answer_heading = any("✅ Answer" in t for t in _texts(at, "markdown"))
    # (c) no abstract/snippet text, and no evidence recall rows (V3 off), in any case
    assert ABS not in raw and ABS not in page and RECALL not in raw and RECALL not in page
    assert body["recalled"] == [] and reader.recall_calls == 0
    # (e) no internal or vendor term on the page
    assert find_terms(page) == [], find_terms(page)
    # R-T 4 (I5c): sidebar is truthful
    side = _texts(at.sidebar, "caption")
    assert any("does not use abstracts or snippets as evidence" in t for t in side)
    assert not any("characters each" in t for t in side)
    if case == "answer":  # (f) the gate is a kind check, not a deletion
        assert body["result_kind"] == "answer" and body["candidates"] == []
        assert body["sources"] and body["traced_sources"] and body["claim_check"]["detail"] == CHECK
        assert at.success and answer_heading and "Reading list" not in page
        return
    # (b) C-F2 API: no prohibited evidence field on any non-answer path
    assert body.get("result_kind", "missing") != "answer"
    assert body["sources"] == [] and body["traced_sources"] == [] and body["claim_check"] is None
    assert body["memory_submitted"] == 0 and CHECK not in raw and TRACED not in raw
    # (a) C-F2 UI: never a green Answer; a reading list or the decline text instead
    assert not at.success and not answer_heading and CHECK not in page and TRACED not in page
    if case == "decline":
        assert body["candidates"] == [] and body["answer"] in _texts(at, "info") and "Reading list" not in page
    else:
        assert "Reading list" in page
    # (d) the D1 line verbatim, first, when candidates exist
    if body["candidates"]:
        d1 = ac.not_read_line(body["candidates"])
        assert d1 and _texts(at, "warning")[0] == d1
        assert "Paper One" in page  # the provider value ("arxiv") is never shown: find_terms above
    if case in CF1:  # C-F1 end to end: the fallback is a reading list with its token, never an answer
        assert body["result_kind"] == "reading_list" and body["contract"]["fallback"] == case[4:]
        assert body["answer"] == ac.text("not_read_line") and "INSUFFICIENT" not in raw and "UNAVAILABLE" not in raw
        assert body["candidates"] == [json.loads(json.dumps(c)) for c in public_candidates(SRC)]


def test_kind_matrix_api_fields_and_page_heading():
    """Owner ruling 2026-10-16: one row per result_kind with the API evidence fields and the page heading."""
    rows = {}
    for case in ("answer", "reading_list", "discovery", "decline", "unspecified", "missing", "bogus"):
        with pytest.MonkeyPatch.context() as mp:  # fresh stubs per kind (the L3 seam must not leak)
            _, body, _ = _post(mp, case)
            at = _page(mp, body)
        md = _texts(at, "markdown")
        heading = "Answer" if any("✅ Answer" in t for t in md) else \
            "Reading list" if "### Reading list" in md else "decline text" if at.info else "none"
        rows[case] = (bool(body["sources"]), bool(body["traced_sources"]), body["claim_check"] is not None,
                      bool(body["candidates"]), heading, bool(at.success))
    assert rows == {
        "answer": (True, True, True, False, "Answer", True),
        "reading_list": (False, False, False, True, "Reading list", False),
        "discovery": (False, False, False, True, "Reading list", False),
        "decline": (False, False, False, False, "decline text", False),
        "unspecified": (False, False, False, True, "Reading list", False),
        "missing": (False, False, False, True, "Reading list", False),
        "bogus": (False, False, False, True, "Reading list", False),
    }, rows


def test_static_wording_absent():
    """R-T item 4 (r2): no abstract-grounding wording in the public modules; no old sidebar or memory-panel text."""
    import re
    for rel in ("vera/disclosure.py", "streamlit_app.py", "vera/discovery_panel.py"):
        src = (ROOT / rel).read_text()
        assert not re.search(r"1,500|answers only from the abstracts|grounded only on index records", src), rel
    assert not re.search(r"answers only from the abstracts|Answers are drafted|rather than the full papers",
                         (ROOT / "vera/sources_sidebar.py").read_text())
    assert not re.search(r"Saved|1,500|1500|icence", (ROOT / "vera/memory_panel.py").read_text())


def test_stored_candidate_fields_survive_the_public_model_and_web_shape_is_unchanged():
    import main
    stored = [{"n": 1, "title": "S", "url": "https://stored.example/a", "span": "chars=0-9", "span_id": "sp1",
               "disposition": "proposed", "canonical_source_id": 3, "source_id": 4, "admission_passed": True,
               "retrieved_at": "2025-01-01", "licence": {"content": {"id": "CC-BY-4.0"}}, "snippet": ABS}]
    c = main.AskCandidate(**public_candidates(stored)[0]).model_dump()
    assert (c["span"], c["span_id"], c["disposition"], c["canonical_source_id"], c["source_id"],
            c["admission_passed"], c["retrieved_at"]) == ("chars=0-9", "sp1", "proposed", 3, 4, True, "2025-01-01")
    assert c["licence"] == "CC-BY-4.0"
    web = public_candidates(SRC)[0]
    assert not {"span", "span_id", "disposition", "canonical_source_id", "source_id", "admission_passed", "licence"} & set(web)
