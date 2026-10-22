"""INT-1022-B: a base-corpus 'nonresponsive' fallback starts inline acquisition (like in-scope insufficient), at most
once per request; a nonresponsive inline answer gives the reading list. Offline: everything below the router is stubbed."""
from types import SimpleNamespace

import pytest

from vera import ask_service, fulldoc_corpus, inline_acquire
from vera.fulldoc_answer import RESP_FALLBACK


@pytest.fixture
def wired(monkeypatch):
    calls = {"fulldoc": [], "acquire": 0}
    dec = SimpleNamespace(accepted=True, layer="layer2_llm", reason="llm_in:x", tokens_used=3, cost_usd=0.0)
    import vera.scope_router as sr
    monkeypatch.setattr(sr, "route", lambda *a, **k: dec)
    monkeypatch.setattr(ask_service, "_acq_cfg_if_on", lambda: {"inline": {"in_scope": True}})
    monkeypatch.setattr(ask_service, "_layer1_clean", lambda q: True)
    monkeypatch.setattr(ask_service, "_acq_on_insufficient", lambda acq: False)
    monkeypatch.setattr(inline_acquire, "icfg_of", lambda acq: {"in_scope": True})
    monkeypatch.setattr(fulldoc_corpus, "flag_enabled", lambda cfg: True)
    monkeypatch.setattr(inline_acquire, "notice", lambda acq, status: f"NOTICE:{status}")

    def acquire(question, acq, served=None):
        calls["acquire"] += 1
        return SimpleNamespace(corpus=object(), status="ok")
    monkeypatch.setattr(inline_acquire, "acquire", acquire)

    def reading_list(question, model, pricing, decision, chained, grounding, contract, tokens=None, cost=None):
        return ask_service.AskResult(answer="RL", tokens_used=tokens or 0, cost_usd=cost or 0.0,
                                     result_kind="reading_list", contract={"layers": ["L0"]})
    monkeypatch.setattr(ask_service, "_reading_list", reading_list)
    return calls


def _fd(calls, base_token, inline_token):
    def fulldoc(question, model, pricing, decision, chained, grounding, acquired_only=False, ephemeral=None, used=None):
        calls["fulldoc"].append("inline" if ephemeral is not None else "base")
        return (None, inline_token if ephemeral is not None else base_token, 1, 0.0, "TEXT")
    return fulldoc


def test_nonresponsive_base_result_starts_one_inline_attempt_then_reading_list(wired, monkeypatch):
    monkeypatch.setattr(ask_service, "_fulldoc", _fd(wired, RESP_FALLBACK, RESP_FALLBACK))
    r = ask_service.answer_in_scope("What is X?", "m", SimpleNamespace(input=0, output=0))
    assert wired["acquire"] == 1 and wired["fulldoc"] == ["base", "inline"]  # exactly one inline attempt, no loop
    assert r.result_kind == "reading_list" and r.contract["fallback"] == RESP_FALLBACK
    assert r.contract.get("inline_fallback") == RESP_FALLBACK and "NOTICE:unanswered" in r.answer


def test_other_fallback_tokens_still_do_not_start_inline(wired, monkeypatch):
    monkeypatch.setattr(ask_service, "_fulldoc", _fd(wired, "claim_check_insufficient", RESP_FALLBACK))
    r = ask_service.answer_in_scope("What is X?", "m", SimpleNamespace(input=0, output=0))
    assert wired["acquire"] == 0 and wired["fulldoc"] == ["base"] and r.result_kind == "reading_list"
