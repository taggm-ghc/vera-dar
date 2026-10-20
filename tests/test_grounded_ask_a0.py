"""K2.I1: L0 dispatch in ask_service (plan-K2-contract.md r2 sections 3, 8.1): reading list, no claims, V1/V2/V9/D2/D5."""
import ast
import json
from pathlib import Path

import pytest

from vera import answer_contract, ask_service, grounded_ask
from vera.ask_service import AskResult
from vera.pricing.config import PricingRecord

ROOT = Path(__file__).resolve().parent.parent
PRICING = PricingRecord(provider="p", model="m", input=0.0, output=0.0, source_url="u",
                        retrieved_at="2026-01-01", effective_from="2026-01-01")
CFG = {"no_sources_text": "NO SOURCES", "removed_citation_note": "NOTE", "insufficient_text": "INSUFFICIENT",
       "claim_check": {"enabled": True}}
ABSTRACT = "SENTINEL-ABSTRACT-TEXT findings: assistants raise productivity"
SRC = [{"n": 1, "title": "T1", "url": "https://arxiv.org/abs/1", "published": "2025-01-01", "snippet": ABSTRACT,
        "source_type": "preprint", "provider": "arxiv", "identifier": "arXiv:1", "retrieved_at": "2026-10-15T00:00:00",
        "licence_decision": "allow", "stored_text_kind": "abstract_metadata", "fetch_status": "fetched"},
       {"n": 2, "title": "T2", "url": "https://arxiv.org/abs/2", "published": None, "snippet": ABSTRACT,
        "provider": "openalex", "identifier": "doi:x", "licence_decision": "hold"}]


@pytest.fixture(autouse=True)
def accept_router(monkeypatch):
    from vera import scope_router
    monkeypatch.setattr(scope_router, "route", lambda *a, **k: scope_router.ScopeDecision(True, "layer1", "ok"))
    monkeypatch.setattr(ask_service, "complete", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model called after router")))
    monkeypatch.setattr(ask_service, "answer_question", lambda *a, **k: (_ for _ in ()).throw(AssertionError("bare model")))
    from vera import claim_check
    monkeypatch.setattr(claim_check, "check_answer", lambda *a, **k: (_ for _ in ()).throw(AssertionError("claim check")))


def _ask(grounding):
    return ask_service.answer_in_scope("Do AI assistants help?", "m", PRICING, grounding=grounding)


def test_default_askresult_is_non_answer():
    r = AskResult("x", 1, 0.0)
    assert r.result_kind == "unspecified" and r.claims == () and r.candidates == () and r.contract is None


def test_reading_list_no_claims_no_claim_check_no_model_call():
    r = _ask((lambda: SRC, CFG))
    assert r.result_kind == "reading_list" and r.claims == () and r.claim_check is None and r.accepted
    assert r.answer == answer_contract.text("not_read_line") and r.answer  # C-7: fixed non-empty string
    assert r.contract == {"layers": answer_contract.layers(), "fallback": None}


def test_v1_internal_sources_stay_populated_for_admission():
    r = _ask((lambda: SRC, CFG))
    assert r.sources == tuple(SRC) and len(r.candidates) == 2


def test_candidates_allowlisted_and_carry_no_abstract():
    r = _ask((lambda: SRC, CFG))
    assert ABSTRACT not in json.dumps(r.candidates) and ABSTRACT.split()[0] not in r.answer
    assert set(r.candidates[0]) == set(grounded_ask.PUBLIC_CANDIDATE_FIELDS) | {"acquisition", "representation", "retrieved_at"}
    assert "snippet" not in r.candidates[0] and "stored_text_kind" not in r.candidates[0]


def test_acquisition_per_candidate_not_from_fetch_status():  # D2
    r = _ask((lambda: SRC, CFG))
    assert [c["acquisition"] for c in r.candidates] == ["not_acquired", "not_acquired"]  # fetch_status='fetched' ignored
    assert all("not acquired" in c["representation"] for c in r.candidates)


def test_partial_acquisition_uses_partial_line_and_labels_each_candidate():
    cands = [dict(c) for c in grounded_ask.public_candidates(SRC)]
    cands[0]["acquisition"] = "full_document_acquired"
    cands[0]["representation"] = answer_contract.representation("full_document_acquired")
    assert answer_contract.not_read_line(cands) == answer_contract.text("not_read_line_partial")
    assert cands[0]["representation"] == "full document stored (not inspected for this answer)" and "not acquired" in cands[1]["representation"]


def test_no_sources_uses_cfg_text_and_reading_list_kind():
    r = _ask((lambda: [], CFG))
    assert r.answer == "NO SOURCES" and r.result_kind == "reading_list" and r.candidates == () and r.sources == ()


def test_grounding_none_returns_l0_no_sources_never_bare_model():  # V2
    r = _ask(None)
    assert r.result_kind == "reading_list" and r.claims == () and r.answer == answer_contract.no_sources_text()


def test_decline_kind():  # V2
    from vera import scope_router
    scope_router_route = lambda *a, **k: scope_router.ScopeDecision(False, "layer1", "no")  # noqa: E731
    import pytest as _p
    mp = _p.MonkeyPatch()
    mp.setattr(scope_router, "route", scope_router_route)
    try:
        r = _ask((lambda: SRC, CFG))
    finally:
        mp.undo()
    assert r.result_kind == "decline" and r.claims == () and r.sources == ()


@pytest.mark.parametrize("grounding", [None, (lambda: [], CFG), (lambda: SRC, CFG),
                                       (lambda: [dict(SRC[0], stored_text_kind="full_text", admission_passed=True)], CFG)])
def test_every_dispatch_input_never_yields_answer_or_claims(grounding):  # L3 safety test (8.1 d)
    r = _ask(grounding)
    assert r.result_kind != "answer" and r.claims == () and r.claim_check is None


def test_answer_reachable_only_through_l3_seam(monkeypatch):
    monkeypatch.setattr(answer_contract, "examined_full_documents", lambda c: ["doc"])
    monkeypatch.setattr(ask_service, "complete", lambda *a, **k: AskResult("Claim [1].", 5, 0.0))
    from vera import claim_check
    monkeypatch.setattr(claim_check, "check_answer", lambda t, s, c: (t, {"outcome": "checked", "checked": 1}))
    assert _ask((lambda: SRC, CFG)).result_kind == "answer"


# ---- V9 source guards (AST) ----
def _funcs_calling(path, names):
    tree = ast.parse((ROOT / path).read_text())
    hits = {}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        src = ast.unparse(fn)
        for nm in names:
            if nm in src:
                hits.setdefault(nm, set()).add(fn.name)
    return hits


def test_state_a_calls_reachable_only_from_l3_function():
    hits = _funcs_calling("vera/ask_service.py", ["messages_for", "check_answer", "fix_citations"])
    assert hits == {"messages_for": {"_answer_grounded"}, "check_answer": {"_answer_grounded"},
                    "fix_citations": {"_answer_grounded"}}
    callers = [n for n in ast.walk(ast.parse((ROOT / "vera/ask_service.py").read_text()))
               if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_answer_grounded"]
    assert len(callers) == 1


def test_write_claims_not_reachable_from_contract_or_ask_service():
    for f in ("vera/ask_service.py", "vera/answer_contract.py", "vera/grounded_ask.py"):
        assert "write_claims" not in (ROOT / f).read_text() and "add_claim" not in (ROOT / f).read_text()


def test_l3_function_never_receives_notes():  # D8c
    src = (ROOT / "vera/ask_service.py").read_text()
    assert "recalled_discovery" not in src and "discovery_note" not in src


def test_d5_l0_path_does_not_import_l1_modules():
    for f in ("vera/ask_service.py", "vera/answer_contract.py", "vera/grounded_ask.py"):
        for n in ast.walk(ast.parse((ROOT / f).read_text())):
            mods = ([a.name for a in n.names] if isinstance(n, ast.Import) else
                    [(n.module or "") + "." + a.name for a in n.names] if isinstance(n, ast.ImportFrom) else [])
            assert not [m for m in mods if "vera.discovery" in m or "discovery_leak_check" in m
                        or m.startswith("vera.discovery")], (f, mods)


def test_with_count_substitutes_count_only_and_tolerates_missing_placeholder():
    from types import SimpleNamespace

    from vera.ask_service import _with_count

    ev = SimpleNamespace(examined_full_documents=[object(), object(), object()])
    assert _with_count("Inspected {n_inspected} documents {x}.", ev) == "Inspected 3 documents {x}."
    assert _with_count("No placeholder.", ev) == "No placeholder."
    assert _with_count("N={n_inspected}", None) == "N=0"
