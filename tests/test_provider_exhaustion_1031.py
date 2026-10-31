"""STREAM-1031 OD-1031-TOKENS: provider-exhaustion stop (TB3, TB4, TB8-TB11). Offline; stubbed providers, fake clock."""
import json
import logging
import threading
from pathlib import Path

import httpx
import pytest
from openai import RateLimitError, APIConnectionError

from vera import inference_chain as ic
from vera import provider_exhaustion as px
from vera import stream_ctx
from vera.ask_service import AskResult
from vera.pricing.config import PricingRecord

WT = Path(__file__).resolve().parent.parent
PRICING = PricingRecord(provider="groq", model="m", input=0.0, output=0.0, source_url="u",
                        retrieved_at="2026-01-01", effective_from="2026-01-01")
SECRET = "gsk_" + "s" * 40


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture(autouse=True)
def _px():
    px.reset_for_tests()
    clk = Clock()
    px.set_clock(clk)
    yield clk
    px.reset_for_tests()


def _e(name, base="https://x.invalid/v1"):
    return ic.ChainEntry("groq", name, PRICING, base, "K", {}, 5.0)


def _rl(retry_after=None):
    h = {"retry-after": str(retry_after)} if retry_after is not None else {}
    resp = httpx.Response(429, headers=h, request=httpx.Request("POST", "https://p.invalid/v1/c"))
    return RateLimitError(f"secret {SECRET}", response=resp, body={"error": {"code": "rate_limit_exceeded"}})


def _ok():
    return AskResult("fine", 10, 0.0)


@pytest.fixture
def stub(monkeypatch):
    calls = []
    behaviour = {}

    def fake(question, model, pricing, client=None, extra_body=None, **kw):
        calls.append(model)
        b = behaviour.get(model)
        if isinstance(b, Exception):
            raise b
        return _ok()

    monkeypatch.setattr(ic, "answer_in_scope", fake)
    monkeypatch.setattr(ic.ChainEntry, "client", lambda self: None)
    return calls, behaviour


def test_tb8_all_429_ends_providers_exhausted_and_logs(stub, caplog, _px):
    calls, beh = stub
    beh["a"] = _rl(30)
    beh["b"] = _rl()
    ctx = stream_ctx.StreamCtx(60)
    tok = stream_ctx.CTX.set(ctx)
    try:
        with caplog.at_level(logging.WARNING, logger="vera"):
            with pytest.raises(px.ProvidersExhausted):
                ic.answer_via_chain("q-secret-question", [_e("a"), _e("b")])
    finally:
        stream_ctx.CTX.reset(tok)
    assert ctx.stop_reason() == "providers_exhausted"
    line = [r.getMessage() for r in caplog.records if "providers exhausted" in r.getMessage()]
    assert len(line) == 1 and "[429, 429]" in line[0] and "30.0" in line[0] and "60.0" in line[0]
    assert SECRET not in caplog.text and "q-secret-question" not in caplog.text


def test_tb9_retry_after_honoured_single_probe(stub, _px, monkeypatch):
    calls, beh = stub
    beh["a"] = _rl(45)
    with pytest.raises(px.ProvidersExhausted):
        ic.answer_via_chain("q", [_e("a")])
    assert calls == ["a"]
    _px.t += 44
    for _ in range(3):
        with pytest.raises(px.ProvidersExhausted):
            ic.answer_via_chain("q", [_e("a")])
    assert calls == ["a"]  # zero calls before Retry-After
    _px.t += 2
    # two concurrent prompts at recovery: only one probe goes out
    beh.pop("a")
    gate, entered = threading.Event(), threading.Event()
    n = []

    def slow(question, model, pricing, client=None, extra_body=None, **kw):
        n.append(model)
        entered.set()
        gate.wait(5)
        return _ok()

    monkeypatch.setattr(ic, "answer_in_scope", slow)
    res = {}
    t1 = threading.Thread(target=lambda: res.setdefault("r1", ic.answer_via_chain("q", [_e("a")])))
    t1.start()
    assert entered.wait(5)
    with pytest.raises(px.ProvidersExhausted):
        ic.answer_via_chain("q", [_e("a")])  # second prompt: no second probe
    gate.set()
    t1.join(5)
    assert n == ["a"] and res["r1"][1] == "groq:a"
    assert not px.exhausted()


def test_tb10_exhaustion_stops_both_ctxs_via_guard():
    px.register_chain(["a"])
    px.note_failure("a", _rl(30))
    ctxs = [stream_ctx.StreamCtx(60), stream_ctx.StreamCtx(60)]
    for c in ctxs:
        tok = stream_ctx.CTX.set(c)
        try:
            with pytest.raises(px.ProvidersExhausted):
                px.guard()
        finally:
            stream_ctx.CTX.reset(tok)
    assert [c.stop_reason() for c in ctxs] == ["providers_exhausted"] * 2


def test_tb11_new_prompt_no_call_recovery_and_partial(stub, _px):
    calls, beh = stub
    beh["a"] = _rl(10)
    beh["b"] = _rl(10)
    with pytest.raises(px.ProvidersExhausted):
        ic.answer_via_chain("q", [_e("a"), _e("b")])
    n = len(calls)
    with pytest.raises(px.ProvidersExhausted):
        ic.answer_via_chain("q2", [_e("a"), _e("b")])
    assert len(calls) == n  # no LLM call
    _px.t += 11
    beh.clear()
    assert ic.answer_via_chain("q3", [_e("a"), _e("b")])[1] == "groq:a"
    # one exhausted + one healthy -> no stop, fallback answers, guard silent
    px.reset_for_tests()
    px.set_clock(_px)
    beh["a"] = _rl(100)
    r = ic.answer_via_chain("q", [_e("a"), _e("b")])
    assert r[1] == "groq:b" and not px.exhausted()
    px.guard()


def test_non_exhaustion_errors_do_not_mark(stub):
    calls, beh = stub
    beh["a"] = APIConnectionError(request=httpx.Request("POST", "https://p.invalid"))
    with pytest.raises(APIConnectionError):
        ic.answer_via_chain("q", [_e("a")])
    assert not px.exhausted()
    assert px.try_acquire("a")


def test_tb3_calls_bounded_by_chain_length(stub):
    calls, beh = stub
    beh["a"] = APIConnectionError(request=httpx.Request("POST", "https://p.invalid"))
    beh["b"] = APIConnectionError(request=httpx.Request("POST", "https://p.invalid"))
    with pytest.raises(APIConnectionError):
        ic.answer_via_chain("q", [_e("a"), _e("b")])
    assert len(calls) == 2


def test_tb4_fallback_off_no_paid_entry_and_cost_bound(tmp_path):
    cfg = json.loads((WT / "config" / "ask-provider-chain.json").read_text())
    assert cfg["openai_fallback"]["enabled"] is False
    assert cfg["exhaustion"]["recheck_s"] == 60
    entries = ic.load_chain("gpt-4o-mini", PRICING, env={"GROQ_API_KEY": SECRET})
    assert entries and all(e.pricing.input == 0.0 and e.pricing.output == 0.0 and e.base_url for e in entries)
    # enabled stub: bound = 13 calls x (8000 p_in + 1024 p_out) at the recorded gpt-4o-mini rates
    rec = next(r for r in json.loads((WT / "config" / "model-pricing.json").read_text())["records"]
               if r["model"] == "gpt-4o-mini")
    per_call = 8000 * rec["input"] / 1e6 + 1024 * rec["output"] / 1e6
    bound = 13 * per_call
    assert bound == pytest.approx(0.0236, abs=1e-3)
    paid = PricingRecord(provider="openai", model="gpt-4o-mini", input=rec["input"], output=rec["output"],
                         source_url="u", retrieved_at="2026-01-01", effective_from="2026-01-01")
    chain = ic.load_chain("gpt-4o-mini", paid, env={}, openai_enabled=True)
    assert [e.label for e in chain] == ["gpt-4o-mini"]
    # worst case one prompt: 13 calls at the maximum tokens
    assert 13 * (8000 * paid.input / 1e6 + 1024 * paid.output / 1e6) <= bound + 1e-12


def test_legacy_ask_all_providers_exhausted_is_429_not_500(monkeypatch, caplog):
    """REL-FIX-1031: legacy POST /ask maps ProvidersExhausted to the pre-diff 429 (no unhandled 500)."""
    from fastapi.testclient import TestClient
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    import main
    from vera import public_mode as pm
    pm.reset_limiter(None)

    def boom(q, e, g):
        raise px.ProvidersExhausted("all exhausted")

    monkeypatch.setattr(main, "answer_via_chain", boom)
    monkeypatch.setattr(main, "store_from_env", lambda cfg: None)
    monkeypatch.setattr(main, "memory_from_env", lambda block: None)
    with caplog.at_level(logging.WARNING):
        r = TestClient(main.app, raise_server_exceptions=False).post(
            "/ask", json={"question": "q"}, headers={"X-API-Key": "k" * 20})
    assert r.status_code == 429, r.text
    want = ("The free answering provider is rate-limited; retry shortly." if not main.OPENAI_FALLBACK
            else "OpenAI rate limit hit (too many requests). Wait a few seconds and retry.")
    assert r.json() == {"detail": want}
    assert "Internal server error" not in r.text
    assert not any(rec.exc_info for rec in caplog.records)  # generic unhandled-exception handler not hit
