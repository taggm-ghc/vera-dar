"""EXHAUST-GRACE-1031 / OD-1031-EXHAUST-GRACE: exhaustion after work started completes the response; a fresh prompt is unchanged.
Offline: stubbed providers, writer and search; no network, no DB, no env files."""
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import RateLimitError

from vera import ask_service as svc
from vera import claim_check
from vera import inference_chain as ic
from vera import provider_exhaustion as px
from vera.ask_service import AskResult
from vera.pricing.config import PricingRecord

PR = PricingRecord(provider="groq", model="m", input=0.0, output=0.0, source_url="u",
                   retrieved_at="2026-01-01", effective_from="2026-01-01")
DEC = SimpleNamespace(accepted=True, tokens_used=5, cost_usd=0.0)
CAND = ({"title": "T", "url": "https://example.invalid/a"},)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    px.reset_for_tests()
    px.set_grace_openai("gpt-x", PR)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    yield
    px.set_grace_openai(None, None)
    px.reset_for_tests()


def _entry(name):
    return ic.ChainEntry("groq", name, PR, "https://x.invalid/v1", "K", {}, 5.0)


def _rl():
    resp = httpx.Response(429, request=httpx.Request("POST", "https://p.invalid/v1/c"))
    return RateLimitError("x", response=resp, body={"error": {"code": "rate_limit_exceeded"}})


@pytest.fixture
def seams(monkeypatch):
    calls = {"fulldoc": [], "openai_clients": 0}
    monkeypatch.setattr(svc, "_reading_list",
                        lambda q, m, p, d, c, g, contract, tokens=None, cost=None:
                        svc.AskResult("Listed sources.", tokens or 0, cost or 0.0, result_kind="reading_list", candidates=CAND,
                                      contract={"layers": [], "fallback": None}))
    monkeypatch.setattr(ic.ChainEntry, "client", lambda self: None)
    return calls


def _fd(calls, ret):
    def fake(question, model, pricing, decision, chained, grounding, acquired_only=False, ephemeral=None, used=None, lanes=None):  # INTEGRATE-1104: _fulldoc signature (NONASK-1103 lanes)
        calls["fulldoc"].append({"model": model, "client": chained.get("client"), "grace": px.GRACE.get() is not None,
                                 "ephemeral": ephemeral})
        if isinstance(ret, Exception):
            raise ret
        return ret
    return fake


def test_fresh_prompt_exhausted_is_unchanged(monkeypatch, seams):
    px.register_chain(["groq:a"])
    px._until["groq:a"] = 10 ** 9
    monkeypatch.setattr(ic, "answer_in_scope", lambda *a, **k: pytest.fail("no work for a fresh exhausted prompt"))
    monkeypatch.setattr(svc, "_fulldoc", _fd(seams, None))
    with pytest.raises(px.ProvidersExhausted):
        ic.answer_via_chain("q", [_entry("a")])
    assert seams["fulldoc"] == []


def test_exhaustion_before_scope_decision_is_not_grace(monkeypatch, seams):
    def inner(*a, **k):
        raise _rl()  # router call itself 429s: no work started
    monkeypatch.setattr(ic, "answer_in_scope", inner)
    monkeypatch.setattr(svc, "_fulldoc", _fd(seams, None))
    with pytest.raises(px.ProvidersExhausted):
        ic.answer_via_chain("q", [_entry("a")])
    assert seams["fulldoc"] == []


def test_mid_acquisition_exhaustion_completes_with_fallback_without_key(monkeypatch, seams):
    monkeypatch.delenv("OPENAI_API_KEY")
    def inner(*a, **k):
        px.note_work(decision=DEC, ephemeral=object(), inline={"status": "partial", "stored": 1})
        raise _rl()
    monkeypatch.setattr(ic, "answer_in_scope", inner)
    monkeypatch.setattr(svc, "_fulldoc", _fd(seams, None))
    res, by = ic.answer_via_chain("q", [_entry("a")])
    assert by == "exhaustion_grace" and res.result_kind == "reading_list"
    assert res.contract["exhaustion_grace"]["mode"] == "reading_list"
    assert res.contract["inline_acquisition"]["stored"] == 1  # acquisition outcome carried
    assert "too busy" in res.answer and seams["fulldoc"] == []  # V1-V4 wording; OpenAI never touched without the key


def test_mid_writing_unverified_draft_gives_fallback_not_claims(monkeypatch, seams):
    unverified = ("ev", "claim_check_insufficient", 40, 0.001, "Unverified draft text that must never show.")
    monkeypatch.setattr(svc, "_fulldoc", _fd(seams, unverified))
    res = svc.complete_after_exhaustion("q", None, {"work": True, "decision": DEC})
    assert res.result_kind == "reading_list" and "Unverified draft" not in res.answer
    assert res.contract["exhaustion_grace"]["mode"] == "reading_list"
    assert res.tokens_used == 40 and res.cost_usd == 0.001  # spend of the attempted writer/check recorded
    assert px.GRACE.get() is None


def test_verified_answer_via_openai_returned_once(monkeypatch, seams):
    eph = object()
    ans = AskResult("Verified answer [1].", 100, 0.002, result_kind="answer", accepted=True,
                    claim_check={"outcome": "ok"}, contract={"layers": [], "fallback": None})
    monkeypatch.setattr(svc, "_fulldoc", _fd(seams, ans))
    res = svc.complete_after_exhaustion("q", None, {"work": True, "decision": DEC, "ephemeral": eph})
    assert res.result_kind == "answer" and res.cost_usd == 0.002 and res.claim_check == {"outcome": "ok"}
    assert res.contract["exhaustion_grace"]["mode"] == "openai_completed" and "still checked against its sources" in res.answer  # V1-V4 wording
    assert len(seams["fulldoc"]) == 1 and seams["fulldoc"][0]["model"] == "gpt-x" and seams["fulldoc"][0]["grace"]
    assert seams["fulldoc"][0]["ephemeral"] is eph and px.GRACE.get() is None


def test_openai_also_rate_limited_falls_back(monkeypatch, seams):
    monkeypatch.setattr(svc, "_fulldoc", _fd(seams, _rl()))
    res = svc.complete_after_exhaustion("q", None, {"work": True, "decision": DEC})
    assert res.result_kind == "reading_list" and len(seams["fulldoc"]) == 1


def test_flag_off_never_calls_openai(monkeypatch, seams):
    monkeypatch.setattr(px, "grace_openai_enabled", lambda path=None: False)
    monkeypatch.setattr(svc, "_fulldoc", _fd(seams, None))
    res = svc.complete_after_exhaustion("q", None, {"work": True, "decision": DEC})
    assert res.result_kind == "reading_list" and seams["fulldoc"] == []


def test_no_evidence_is_the_no_answer_path(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    res = svc.complete_after_exhaustion("q", None, {"work": True, "decision": DEC})  # grounding None: real L0 no-sources result
    assert res.result_kind == "reading_list" and not res.candidates
    assert res.contract["exhaustion_grace"]["mode"] == "no_answer" and "too busy" in res.answer  # V1-V4 wording


def test_config_ships_grace_on_and_openai_fallback_off():
    cfg = json.load(open(px.CONFIG_PATH))
    assert cfg["exhaustion"]["grace"]["openai_completion"] is True and cfg["openai_fallback"]["enabled"] is False
    assert "OD-1031-EXHAUST-GRACE" in cfg["exhaustion"]["grace"]["note"]


def test_guard_passes_and_checker_uses_only_openai_in_grace(monkeypatch):
    px.register_chain(["a"]); px._until["a"] = 10 ** 9
    with pytest.raises(px.ProvidersExhausted):
        px.guard()
    used = []

    class C:
        def __init__(self, label): self.label = label
        class chat:  # noqa: N801
            pass
    def fake_client(checker, t):
        used.append(checker["model"])
        msg = SimpleNamespace(content='{"claims": []}')
        comp = SimpleNamespace(create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None))
        return SimpleNamespace(chat=SimpleNamespace(completions=comp))
    monkeypatch.setattr(claim_check, "_client", fake_client)
    monkeypatch.setenv("GROQ_API_KEY", "g")
    cfg = {"timeout_s": 5, "checkers": [
        {"provider": "groq", "base_url": "https://x.invalid", "model": "free", "api_key_env": "GROQ_API_KEY"},
        {"provider": "openai", "base_url": None, "model": "gpt-n", "api_key_env": "OPENAI_API_KEY"}]}
    tok = px.GRACE.set({"client": object()})
    try:
        px.guard()  # no raise in grace
        claim_check.judge([], [], cfg)
    finally:
        px.GRACE.reset(tok)
    assert used == ["gpt-n"]


# ---- end to end through main (legacy /ask and /ask/stream), real answer_via_chain, stubbed inner work
from tests.test_stream_route_support import client, lines, post, stream_env  # noqa: E402,F401


def _wire(monkeypatch, work_started):
    import main

    def inner(*a, **k):
        if work_started:
            px.note_work(decision=DEC)
        raise _rl()
    monkeypatch.setattr(ic, "answer_in_scope", inner)
    monkeypatch.setattr(ic.ChainEntry, "client", lambda self: None)
    monkeypatch.setattr(svc, "_fulldoc", lambda *a, **k: pytest.fail("OpenAI/writer must not run here"))
    monkeypatch.setattr(svc, "_reading_list",
                        lambda q, m, p, d, c, g, contract, tokens=None, cost=None:
                        svc.AskResult("Listed sources.", 0, 0.0, result_kind="reading_list",
                                      contract={"layers": [], "fallback": None}))
    monkeypatch.delenv("OPENAI_API_KEY")  # non-LLM completion keeps these two tests offline
    px.set_grace_openai("gpt-x", PR)
    chain = [_entry("a")]
    monkeypatch.setattr(main, "ASK_CHAIN", chain)
    monkeypatch.setattr(main, "answer_via_chain", lambda q, e, g: ic.answer_via_chain(q, chain, g))
    monkeypatch.setattr(main, "store_from_env", lambda cfg: None)
    monkeypatch.setattr(main, "memory_from_env", lambda block: None)
    return main


def test_stream_mid_response_exhaustion_final_result_is_completed(monkeypatch, stream_env):
    _wire(monkeypatch, True)
    ls = lines(post(client()))
    assert ls[-1]["type"] == "result" and ls[-1]["status"] == 200, ls[-1]
    body = ls[-1]["body"]
    assert body["result_kind"] == "reading_list" and body["contract"]["exhaustion_grace"]["mode"] == "no_answer"
    assert "too busy" in body["answer"]  # V1-V4 wording


def test_stream_fresh_exhaustion_is_still_503_unavailable(monkeypatch, stream_env):
    _wire(monkeypatch, False)
    ls = lines(post(client()))
    assert ls[-1]["status"] == 503 and ls[-1]["kind"] == "unavailable"


@pytest.fixture(autouse=True)
def _legacy_concatenated_notes(monkeypatch):
    """ANS-LAYERS-1101: pins the pre-layering answer text (notes inside `answer`); the separated mode is covered by
    tests/test_ans_layers_s1.py."""
    from vera import answer_contract
    monkeypatch.setattr(answer_contract, "notes_separate", lambda *a, **k: False)
