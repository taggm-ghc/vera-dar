"""EXG-FIX-1031: D1 (grace sees the research-eligible override), D2 (unpriced OpenAI checker is costed), D3 (disconnect
cancels grace work), N-D1 (event elapsed preserved). Offline: stubs only; no network, no DB, no env files."""
import asyncio
import logging
from types import SimpleNamespace

import pytest

from vera import ask_service as svc
from vera import claim_check as cc
from vera import provider_exhaustion as px
from vera import stream_ctx
from vera.ask_service import AskResult

from vera.scope_router import ScopeDecision

DEC = ScopeDecision(accepted=False, layer="layer2_llm", reason="r", tokens_used=1)


def test_d1_note_work_decision_is_refreshed_after_research_eligible_override(monkeypatch):
    seen = {}
    px.reset_for_tests()
    monkeypatch.setattr("vera.scope_router.route", lambda *a, **k: DEC)
    monkeypatch.setattr(svc, "_catstep", lambda d, *a, **k: d)
    monkeypatch.setattr(svc, "_acq_cfg_if_on", lambda: None)
    monkeypatch.setattr(svc, "_research_eligible", lambda *a, **k: True)
    monkeypatch.setattr(svc, "_offline_coverage_eligible", lambda *a, **k: True)

    def fd(question, model, pricing, decision, *a, **k):
        seen["state"] = px.WORK.get()["decision"]
        return AskResult("x", 0, 0.0, result_kind="answer", accepted=True)
    monkeypatch.setattr(svc, "_fulldoc", fd)
    tok = px.WORK.set({})
    try:
        svc.answer_in_scope("a research question", "m", None)
        assert px.WORK.get()["decision"].accepted is True
        assert seen["state"].accepted is True
    finally:
        px.WORK.reset(tok)


def _usage(p, c):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=p, completion_tokens=c))


def test_d2_openai_checker_costed_from_authoritative_price():
    rows = []
    tok = cc.USAGE.set(rows)
    try:
        cc._record_usage(_usage(1_000_000, 1_000_000), "openai:gpt-4.1-nano", {"provider": "openai", "model": "gpt-4.1-nano"})
    finally:
        cc.USAGE.reset(tok)
    assert rows[0]["cost_usd"] == pytest.approx(0.1 + 0.4)  # config/model-pricing.json gpt-4.1-nano


def test_d2_unknown_model_fails_safe_at_highest_known_price_with_warning(caplog):
    from vera.pricing.config import load_model_pricing
    recs = [r for r in load_model_pricing().records if r.provider == "openai"]
    hi_in, hi_out = max(r.input for r in recs), max(r.output for r in recs)
    rows = []
    tok = cc.USAGE.set(rows)
    try:
        with caplog.at_level(logging.WARNING, logger="vera"):
            cc._record_usage(_usage(1_000_000, 1_000_000), "openai:nope", {"provider": "openai", "model": "nope"})
    finally:
        cc.USAGE.reset(tok)
    assert rows[0]["cost_usd"] == pytest.approx(hi_in + hi_out) and rows[0]["cost_usd"] > 0.5
    assert "no price" in caplog.text


@pytest.mark.parametrize("prov,model", [("groq", "llama-x"), ("nopriced", "m")])
def test_d2_provider_without_price_records_stays_unpriced_without_warning(caplog, prov, model):
    rows = []
    tok = cc.USAGE.set(rows)
    try:
        with caplog.at_level(logging.WARNING):
            cc._record_usage(_usage(1000, 1000), f"{prov}:{model}", {"provider": prov, "model": model})
    finally:
        cc.USAGE.reset(tok)
    assert rows[0]["cost_usd"] is None
    assert "no price" not in caplog.text


def test_d3_disconnect_after_providers_exhausted_is_recorded():
    c = stream_ctx.StreamCtx(stall_s=60)
    c.stop(stream_ctx.PROVIDERS_EXHAUSTED)
    c.cancel()
    assert c.cancelled
    with pytest.raises(stream_ctx.StreamCancelled):
        tok = stream_ctx.CTX.set(c)
        try:
            stream_ctx.check_cancelled()
        finally:
            stream_ctx.CTX.reset(tok)


def test_d3_first_non_disconnect_reason_still_wins():
    c = stream_ctx.StreamCtx(stall_s=60)
    c.stop(stream_ctx.PROVIDERS_EXHAUSTED)
    c.stop(stream_ctx.STALLED)
    assert c.stop_reason() == stream_ctx.PROVIDERS_EXHAUSTED


@pytest.mark.parametrize("exc", [stream_ctx.StreamCancelled, asyncio.CancelledError])
def test_d3_grace_completion_does_not_swallow_cancellation(monkeypatch, exc):
    from tests.test_exhaust_grace_1031 import DEC as GDEC, PR
    px.reset_for_tests()
    px.set_grace_openai("gpt-x", PR)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setattr(svc, "_reading_list", lambda *a, **k: pytest.fail("must not fall through to the reading list"))

    def boom(*a, **k):
        raise exc()
    monkeypatch.setattr(svc, "_fulldoc", boom)
    try:
        with pytest.raises(exc):
            svc.complete_after_exhaustion("q", None, {"work": True, "decision": GDEC})
    finally:
        px.set_grace_openai(None, None)
        px.reset_for_tests()


def test_d3_claim_checker_makes_no_call_after_disconnect(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setattr(cc, "_client", lambda *a, **k: pytest.fail("paid call after disconnect"))
    c = stream_ctx.StreamCtx(stall_s=60)
    c.stop(stream_ctx.PROVIDERS_EXHAUSTED)
    c.cancel()
    cfg = {"timeout_s": 5, "checkers": [{"provider": "groq", "base_url": "https://x.invalid", "model": "m",
                                         "api_key_env": "GROQ_API_KEY"}]}
    tok = stream_ctx.CTX.set(c)
    try:
        with pytest.raises(stream_ctx.StreamCancelled):
            cc.judge([], [], cfg)
    finally:
        stream_ctx.CTX.reset(tok)

