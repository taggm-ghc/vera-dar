"""K2.I6: /ask response behaviour per result kind; C-F1 fallback closure; discovery memory wiring. No streamlit."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vera import answer_contract as ac
from vera import ask_service
from vera import memory_store as ms
from vera.ask_service import AskResult
from vera.grounded_ask import public_candidates
from vera.memory_gate import load_gate_config

ROOT = Path(ask_service.__file__).resolve().parent.parent
CFG_CHAIN = json.loads((ROOT / "config" / "ask-provider-chain.json").read_text())
STORE_CFG = CFG_CHAIN["memory"]["store"]
SENT = "SENTINEL-ABSTRACT-TEXT"
SRC = [{"n": 1, "title": "Paper One", "url": "https://arxiv.org/abs/2401.00001", "published": "2025-01-01",
        "snippet": SENT, "identifier": "arXiv:2401.00001", "provider": "arxiv", "licence_decision": "allow",
        "retrieved_at": "2026-10-01"}]
CAND = public_candidates(SRC)
GROUNDING_CFG = {"insufficient_text": "INSUFFICIENT", "removed_citation_note": "NOTE", "no_sources_text": "NO SOURCES",
                 "claim_check": {"enabled": True, "unavailable_text": "UNAVAILABLE"}}
HEAD = {"answer", "model", "tokens_used", "cost_usd", "sources", "traced_sources", "recalled", "memory_written",
        "memory_submitted", "claim_check"}
NEW = {"result_kind", "candidates", "contract", "recalled_discovery", "discovery_note_submitted", "outcome"}  # outcome: additive, FDC-1020 2026-10-20


class Mem:
    def __init__(self, reader=True, store=True, notes=None, raises=False):
        self.reader, self.store = (object() if reader else None), (object() if store else None)
        self.cfg = {"max_writes_per_answer": 3}
        self.notes, self.raises = notes or [], raises
        self.claim_writes, self.note_writes, self.recall_calls = [], [], 0

    def recall(self, q): self.recall_calls += 1; return []
    def recall_discovery(self, q):
        if self.raises:
            raise RuntimeError("boom")
        return self.notes
    def write_claims(self, r, s, q): self.claim_writes.append(r); return 0, {}
    def write_discovery_note(self, c, q): self.note_writes.append((list(c), q)); return 1, {}


@pytest.fixture
def contract(monkeypatch):
    state = {"layers": ["L0", "L2"], "discovery_memory": {"write_on_l0": True}, "memory": {"evidence_recall_enabled": False},
             "texts": json.loads((ROOT / "config" / "discovery_contract.json").read_text())["texts"]}
    monkeypatch.setattr(ac, "load_config", lambda path=ac.CONFIG_PATH: state)
    return state


def ask(monkeypatch, result, mem=None, admit=None):
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    import main
    from vera import public_mode as pm
    pm.reset_limiter(None)
    monkeypatch.setattr(main, "answer_via_chain", lambda q, e, g: (result, "groq:m"))
    monkeypatch.setattr(main, "store_from_env", lambda cfg: object() if admit is not None else None)
    if admit is not None:
        monkeypatch.setattr(main, "admit_to_corpus", lambda *a: admit.append(a))
        monkeypatch.setattr(main, "llm_call_for", lambda e: None)
    monkeypatch.setattr(main, "memory_from_env", lambda block: mem)
    r = TestClient(main.app).post("/ask", json={"question": "q"}, headers={"X-API-Key": "k" * 20})
    assert r.status_code == 200, r.text
    return r


def res(kind="reading_list", **kw):
    base = dict(answer="A", tokens_used=1, cost_usd=0.0, sources=tuple(SRC), claim_check={"outcome": "SENTCHECK"},
                accepted=True, claims=({"id": 0, "sentence": "s", "citations": [1], "verdict": "supported"},),
                candidates=CAND, contract={"layers": ["L0"], "fallback": None})
    base.update(kw)
    return AskResult(**base) if kind is None else AskResult(result_kind=kind, **base)


def test_response_key_set_is_exact_union(monkeypatch, contract):
    assert set(ask(monkeypatch, res("answer")).json()) == HEAD | NEW


@pytest.mark.parametrize("kind", ["reading_list", "discovery", "decline", "unspecified", "bogus", None])
def test_non_answer_response_has_no_prohibited_evidence_fields(monkeypatch, contract, kind):
    r = ask(monkeypatch, res(kind), Mem())
    b = r.json()
    assert b["sources"] == [] and b["traced_sources"] == [] and b["claim_check"] is None
    assert b["memory_submitted"] == 0 and SENT not in r.text and "SENTCHECK" not in r.text
    assert b["result_kind"] == (kind or "unspecified")


def test_recalled_empty_with_real_memory_service_when_v3_off(monkeypatch, contract):
    class Rd:
        calls = 0
        def recall(self, q): Rd.calls += 1; return [{"claim_text": "x"}]
        def recall_discovery(self, q): return []
    svc = ms.MemoryService(dict(STORE_CFG), load_gate_config(), Rd(), object())
    assert ask(monkeypatch, res("answer"), svc).json()["recalled"] == [] and Rd.calls == 0


def test_admission_args_present_iff_internal_sources(monkeypatch, contract):
    seen = []
    ask(monkeypatch, res("reading_list"), None, admit=seen)  # sources internal -> admission still runs (V1)
    assert len(seen) == 1 and seen[0][1] == list(SRC)
    seen2 = []
    ask(monkeypatch, res("reading_list", sources=()), None, admit=seen2)
    assert seen2 == []


def test_answer_kind_keeps_public_sources_and_claim_check(monkeypatch, contract):
    b = ask(monkeypatch, res("answer", claim_check={"outcome": "checked"}), Mem()).json()
    assert b["sources"][0]["title"] == "Paper One" and "snippet" not in b["sources"][0]
    assert b["claim_check"] == {"outcome": "checked"} and b["candidates"] == [] and b["memory_submitted"] == 1


def test_evidence_write_only_for_answer_kind(monkeypatch, contract):
    m = Mem()
    b = ask(monkeypatch, res("reading_list"), m).json()
    assert m.claim_writes == [] and b["memory_submitted"] == 0
    m2 = Mem()
    ask(monkeypatch, res("answer"), m2)
    assert len(m2.claim_writes) == 1


@pytest.mark.parametrize("breaker", ["accepted", "kind", "no_candidates", "no_l2", "no_write_on_l0", "no_memory",
                                     "no_store", "no_reader"])
def test_discovery_note_scheduled_only_when_all_conditions(monkeypatch, contract, breaker):
    m = Mem(store=breaker != "no_store", reader=breaker != "no_reader")
    kw = {}
    if breaker == "accepted": kw["accepted"] = False
    if breaker == "no_candidates": kw["candidates"] = ()
    if breaker == "no_l2": contract["layers"] = ["L0"]
    if breaker == "no_write_on_l0": contract["discovery_memory"]["write_on_l0"] = False
    kind = "answer" if breaker == "kind" else "reading_list"
    b = ask(monkeypatch, res(kind, **kw), None if breaker == "no_memory" else m).json()
    assert b["discovery_note_submitted"] == 0 and m.note_writes == []
    contract["layers"], contract["discovery_memory"]["write_on_l0"] = ["L0", "L2"], True
    m2 = Mem()
    b2 = ask(monkeypatch, res("discovery"), m2).json()
    assert b2["discovery_note_submitted"] == 1 and m2.note_writes == [(list(CAND), "q")]


def test_discovery_note_submitted_is_0_or_1_int(monkeypatch, contract):
    for r, m in ((res("reading_list"), Mem()), (res("reading_list", candidates=()), Mem())):
        v = ask(monkeypatch, r, m).json()["discovery_note_submitted"]
        assert type(v) is int and v in (0, 1)


def test_recalled_discovery_shape_and_separate_from_recalled(monkeypatch, contract):
    note = {"kind": "discovery_note", "note": "Discovery note (not evidence). x", "urls": ["https://a.org"],
            "saved_at": "2026-10-01", "claim_id": 9, "extra": "drop"}
    b = ask(monkeypatch, res("reading_list"), Mem(notes=[note, "junk", {"note": 5}])).json()
    assert b["recalled"] == []
    assert b["recalled_discovery"] == [{"kind": "discovery_note", "note": "Discovery note (not evidence). x",
                                        "urls": ["https://a.org"], "saved_at": "2026-10-01"}]
    contract["layers"] = ["L0"]
    assert ask(monkeypatch, res("reading_list"), Mem(notes=[note])).json()["recalled_discovery"] == []
    contract["layers"] = ["L0", "L2"]
    assert ask(monkeypatch, res("reading_list", accepted=False), Mem(notes=[note])).json()["recalled_discovery"] == []
    assert ask(monkeypatch, res("reading_list"), Mem(raises=True)).json()["recalled_discovery"] == []


def test_candidates_allowlisted_drop_extras(monkeypatch, contract):
    dirty = ({**CAND[0], "snippet": SENT, "abstract": SENT, "stored_text_kind": "abstract"},)
    r = ask(monkeypatch, res("reading_list", candidates=dirty), Mem())
    c = r.json()["candidates"][0]
    assert set(c) == {"n", "title", "url", "published", "identifier", "provider", "licence_decision", "acquisition",
                      "representation", "retrieved_at"} and SENT not in r.text


# ---- C-F1 ----
TOKENS = ["no_valid_citations", "checker_unavailable", "claim_check_insufficient"]


def _l3(monkeypatch, token):
    from vera import claim_check, scope_router
    monkeypatch.setattr(ac, "examined_full_documents", lambda c: ["examined"])
    monkeypatch.setattr(scope_router, "route", lambda *a, **k: scope_router.ScopeDecision(True, "layer1", "ok"))
    text = "No citation here." if token == "no_valid_citations" else "Faster [1]."
    monkeypatch.setattr(ask_service, "complete", lambda *a, **k: AskResult(text, 10, 0.0))
    if token != "no_valid_citations":
        outcome = "checker_unavailable" if token == "checker_unavailable" else "insufficient"
        monkeypatch.setattr(claim_check, "check_answer", lambda t, s, c: (None, {"checked": 0, "outcome": outcome}))


def _pricing():
    from tests.test_item79_claim_check import PRICING
    return PRICING


@pytest.mark.parametrize("token", TOKENS)
def test_cf1_fallbacks_satisfy_non_answer_invariant(monkeypatch, token):
    _l3(monkeypatch, token)
    r = ask_service.answer_in_scope("q", "m", _pricing(), grounding=(lambda: SRC, GROUNDING_CFG))
    assert r.result_kind == "reading_list" and r.claim_check is None and r.claims == ()
    assert r.candidates == public_candidates(SRC) and r.sources == tuple(SRC) and r.contract["fallback"] == token
    assert r.answer == ac.text("not_read_line") and "INSUFFICIENT" not in r.answer and "UNAVAILABLE" not in r.answer


@pytest.mark.parametrize("token", TOKENS)
def test_cf1_fallback_through_ask_has_no_evidence_fields(monkeypatch, contract, token):
    _l3(monkeypatch, token)
    r = ask_service.answer_in_scope("q", "m", _pricing(), grounding=(lambda: SRC, GROUNDING_CFG))
    resp = ask(monkeypatch, r, Mem())
    b = resp.json()
    assert b["result_kind"] == "reading_list" and b["sources"] == [] and b["traced_sources"] == []
    assert b["claim_check"] is None and b["memory_submitted"] == 0 and SENT not in resp.text
    assert b["contract"]["fallback"] == token and len(b["candidates"]) == 1


def test_fallback_paths_unreachable_with_shipped_seam(monkeypatch):
    from vera import scope_router
    monkeypatch.setattr(scope_router, "route", lambda *a, **k: scope_router.ScopeDecision(True, "layer1", "ok"))
    calls = []
    monkeypatch.setattr(ask_service, "_answer_grounded", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(ask_service, "complete", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model called")))
    assert ac.examined_full_documents(CAND) == []
    for grounding in (None, (lambda: [], GROUNDING_CFG), (lambda: SRC, GROUNDING_CFG)):
        r = ask_service.answer_in_scope("q", "m", _pricing(), grounding=grounding)
        assert r.result_kind == "reading_list"
    assert calls == []
