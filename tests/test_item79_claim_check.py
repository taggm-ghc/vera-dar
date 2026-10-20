"""Item #79: R1's grounded-default criteria: per-citation provenance, insufficient-evidence reply, claim check."""
import json

from vera import ask_service, claim_check as cc, grounded_ask as ga
from vera.ask_service import AskResult
from vera.pricing.config import PricingRecord
from vera.search_providers import OPENALEX_METADATA_LICENCE, licence_record

PRICING = PricingRecord(provider="p", model="m", input=0.0, output=0.0, source_url="u",
                        retrieved_at="2026-01-01", effective_from="2026-01-01")
CC = {"enabled": True, "timeout_s": 1, "max_claims": 30, "partial_marker": " (partly)", "removed_note": "REMOVED",
      "unavailable_text": "UNAVAILABLE",
      "checkers": [{"provider": "a", "base_url": "x", "model": "first", "api_key_env": "KEY_A", "family": "qwen"},
                   {"provider": "b", "base_url": "x", "model": "second", "api_key_env": "KEY_B", "family": "openai"}]}
CFG = {"no_sources_text": "NO SOURCES", "removed_citation_note": "NOTE", "insufficient_text": "INSUFFICIENT",
       "claim_check": CC}
SRC = [{"n": 1, "title": "A", "url": "https://arxiv.org/abs/2401.00001", "snippet": "Copilot users were 55% faster."},
       {"n": 2, "title": "B", "url": "https://doi.org/10.1/b", "snippet": "No effect on code quality."}]


def _verdicts(mapping):
    return lambda checker, msgs: json.dumps({"claims": [{"id": i, "verdict": v, "reason": "r"} for i, v in mapping.items()]})


def test_provenance_fields():
    p = ga.provenance({"url": "https://arxiv.org/abs/2401.00001v2", "provider": "arxiv", "retrieved_at": "T",
                       "licence": licence_record(metadata=OPENALEX_METADATA_LICENCE), "stored_text_kind": "abstract_metadata"})
    assert p == {"provider": "arxiv", "identifier": "arXiv:2401.00001v2", "retrieved_at": "T", "licence_decision": "allow"}
    d = ga.provenance({"url": "https://doi.org/10.1/X", "doi": "10.1/X", "discovered_via": "news.example",
                       "abstract_source": "semanticscholar", "licence": None})
    assert d["identifier"] == "doi:10.1/x" and d["provider"] == "openalex+semanticscholar" and d["licence_decision"] == "defer"


def test_public_sources_carry_provenance_not_abstracts():
    out = ga.public_sources([dict(SRC[0], provider="arxiv", identifier="arXiv:1", retrieved_at="T", licence_decision="allow")])
    assert out[0]["identifier"] == "arXiv:1" and "snippet" not in out[0]


def test_split_claims_only_valid_citations():
    claims = cc.split_claims("Faster [1]. Quality unchanged [2, 9]. Bogus [7]. No cite here.", 2, 30)
    assert [(c["text"], c["cites"]) for c in claims] == [("Faster [1].", [1]), ("Quality unchanged [2, 9].", [2])]


def test_apply_removes_unsupported_and_marks_partial(monkeypatch):
    monkeypatch.setenv("KEY_A", "k")
    text, summary = cc.check_answer("Faster [1]. Quality improved [2]. Shorter [1].", SRC, CC,
                                    call=_verdicts({0: "supported", 1: "unsupported", 2: "partial"}))
    assert text == "Faster [1]. Shorter [1]. (partly)\n\nREMOVED"
    assert summary == {"checked": 3, "supported": 1, "partial": 1, "removed": 1, "checker": "a:first", "outcome": "checked"}


def test_incomplete_reply_moves_to_next_checker(monkeypatch):
    monkeypatch.setenv("KEY_A", "k")
    monkeypatch.setenv("KEY_B", "k")
    replies = {"first": json.dumps({"claims": []}), "second": json.dumps({"claims": [{"id": 0, "verdict": "supported"}]})}
    text, summary = cc.check_answer("Faster [1].", SRC, CC, call=lambda ch, m: replies[ch["model"]])
    assert summary["checker"] == "b:second" and text == "Faster [1]."


def test_fail_closed_when_no_checker(monkeypatch):
    monkeypatch.delenv("KEY_A", raising=False)
    monkeypatch.delenv("KEY_B", raising=False)
    assert cc.check_answer("Faster [1].", SRC, CC) == (None, {"checked": 0, "outcome": "checker_unavailable"})


def test_all_unsupported_fails_closed(monkeypatch):
    monkeypatch.setenv("KEY_A", "k")
    text, summary = cc.check_answer("Faster [1].", SRC, CC, call=_verdicts({0: "unsupported"}))
    assert text is None and summary["outcome"] == "no_supported_claims"


def test_checker_sees_only_cited_abstracts(monkeypatch):
    monkeypatch.setenv("KEY_A", "k")
    seen = {}

    def call(ch, msgs):
        seen["payload"] = json.loads(msgs[1]["content"])
        return json.dumps({"claims": [{"id": 0, "verdict": "supported"}]})
    cc.check_answer("Faster [1].", SRC, CC, call=call)
    assert list(seen["payload"]["sources"]) == ["1"]


def _l3_open(monkeypatch):
    """These tests exercise the retained state-A function; open the L3 seam in the test only (the shipped stub returns [])."""
    from vera import answer_contract
    monkeypatch.setattr(answer_contract, "examined_full_documents", lambda c: ["examined"])


def _router_accepts(monkeypatch):
    _l3_open(monkeypatch)
    from vera import scope_router
    monkeypatch.setattr(scope_router, "route", lambda *a, **k: scope_router.ScopeDecision(True, "layer1", "ok"))


def _assert_fallback(r, token):
    """C-F1 invariant: a fallback is a reading-list non-answer, never an answer with a claim check."""
    from vera.grounded_ask import public_candidates
    assert r.result_kind == "reading_list" and r.claim_check is None and r.claims == ()
    assert r.candidates == public_candidates(SRC) and r.sources == tuple(SRC) and r.contract["fallback"] == token


def test_answer_without_valid_citation_becomes_insufficient(monkeypatch):
    _router_accepts(monkeypatch)
    monkeypatch.setattr(ask_service, "complete", lambda *a, **k: AskResult("Sources do not address this.", 10, 0.0))
    r = ask_service.answer_in_scope("q", "m", PRICING, grounding=(lambda: SRC, CFG))
    _assert_fallback(r, "no_valid_citations")


def test_grounded_path_runs_claim_check_and_fails_closed(monkeypatch):
    _router_accepts(monkeypatch)
    monkeypatch.setattr(ask_service, "complete", lambda *a, **k: AskResult("Faster [1].", 10, 0.0))
    from vera import claim_check
    monkeypatch.setattr(claim_check, "check_answer", lambda text, s, c: (None, {"checked": 0, "outcome": "checker_unavailable"}))
    r = ask_service.answer_in_scope("q", "m", PRICING, grounding=(lambda: SRC, CFG))
    _assert_fallback(r, "checker_unavailable")
    monkeypatch.setattr(claim_check, "check_answer", lambda text, s, c: ("Faster [1].", {"outcome": "checked", "checked": 1}))
    assert ask_service.answer_in_scope("q", "m", PRICING, grounding=(lambda: SRC, CFG)).claim_check["outcome"] == "checked"
