"""Item #77: free-tier-first provider chain for /ask (offline; no network)."""
import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APITimeoutError, BadRequestError, RateLimitError

from vera import inference_chain as ic
from vera.ask_service import AskResult
from vera.pricing.config import PricingRecord

OPENAI_PRICING = PricingRecord(provider="openai", model="gpt-4.1-nano", input=0.1, output=0.4, source_url="u",
                               retrieved_at="2026-01-01", effective_from="2026-01-01")
SECRET = "gsk_" + "s" * 40


def _status_error(cls, code, err_code="x"):
    resp = httpx.Response(code, request=httpx.Request("POST", "https://provider.invalid/v1/chat/completions"))
    return cls(f"provider said {SECRET}", response=resp, body={"error": {"code": err_code}})


def _chain_file(tmp_path, providers):
    p = tmp_path / "chain.json"
    p.write_text(json.dumps({"checked_at": "2026-10-07", "timeout_s": 5, "providers": providers}))
    return p


GROQ = {"provider": "groq", "base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-20b",
        "api_key_env": "GROQ_API_KEY", "input_per_1m": 0.0, "output_per_1m": 0.0,
        "extra_body": {"reasoning_effort": "low"}}
ROUTER = dict(GROQ, provider="openrouter", base_url="https://openrouter.ai/api/v1", model="m:free",
              api_key_env="OPENROUTER_API_KEY", extra_body={})


def test_load_chain_skips_entries_without_key_and_appends_openai(tmp_path):
    path = _chain_file(tmp_path, [GROQ, ROUTER])
    labels = [e.label for e in ic.load_chain("gpt-4.1-nano", OPENAI_PRICING, path, env={"GROQ_API_KEY": SECRET})]
    assert labels == ["groq:openai/gpt-oss-20b", "gpt-4.1-nano"]
    labels = [e.label for e in ic.load_chain("gpt-4.1-nano", OPENAI_PRICING, path, env={})]
    assert labels == ["gpt-4.1-nano"]  # no free keys: identical to the pre-#77 behaviour


def test_load_chain_unreadable_file_means_openai_only(tmp_path):
    entries = ic.load_chain("gpt-4.1-nano", OPENAI_PRICING, tmp_path / "missing.json", env={"GROQ_API_KEY": SECRET})
    assert [e.provider for e in entries] == ["openai"]


def test_free_entries_cost_zero_and_carry_extra_body(tmp_path):
    e = ic.load_chain("gpt-4.1-nano", OPENAI_PRICING, _chain_file(tmp_path, [GROQ]), env={"GROQ_API_KEY": SECRET})[0]
    assert e.pricing.input == 0 and e.pricing.output == 0 and e.extra_body == {"reasoning_effort": "low"}


def test_shipped_config_lists_current_groq_models_with_low_reasoning():
    raw = json.loads(ic.CONFIG_PATH.read_text())
    models = [p["model"] for p in raw["providers"]]
    assert models and all(p["extra_body"].get("reasoning_effort") == "low" for p in raw["providers"])
    assert not any("mixtral" in m or m.startswith("llama-3") for m in models)  # retired ids (2025-03 / 2026-08)


def _entries():
    free = ic.ChainEntry("groq", "a", OPENAI_PRICING.model_copy(update={"input": 0, "output": 0}),
                         "https://api.groq.com/openai/v1", "GROQ_API_KEY")
    free2 = ic.ChainEntry("groq", "b", free.pricing, "https://api.groq.com/openai/v1", "GROQ_API_KEY")
    return [free, free2, ic.ChainEntry("openai", "gpt-4.1-nano", OPENAI_PRICING)]


@pytest.mark.parametrize("exc", [
    _status_error(RateLimitError, 429),
    _status_error(BadRequestError, 400, "model_decommissioned"),
    APITimeoutError(request=httpx.Request("POST", "https://provider.invalid")),
])
def test_falls_through_on_provider_errors(monkeypatch, caplog, exc):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    calls = []

    def fake(q, model, pricing, client=None, extra_body=None):
        calls.append(model)
        if model == "a":
            raise exc
        return AskResult("ok", 10, 0.0)

    monkeypatch.setattr(ic, "answer_in_scope", fake)
    with caplog.at_level(logging.WARNING, logger="vera"):
        result, label = ic.answer_via_chain("q", _entries())
    assert calls == ["a", "b"] and label == "groq:b" and result.answer == "ok"
    assert SECRET not in caplog.text and "provider said" not in caplog.text


def test_empty_answer_falls_through(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    replies = {"a": "", "b": "  ", "gpt-4.1-nano": "paid answer"}
    monkeypatch.setattr(ic, "answer_in_scope", lambda q, m, p, client=None, extra_body=None: AskResult(replies[m], 1, 0.0))
    result, label = ic.answer_via_chain("q", _entries())
    assert label == "gpt-4.1-nano" and result.answer == "paid answer"


def test_last_entry_error_propagates(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)

    def boom(q, m, p, client=None, extra_body=None):
        raise _status_error(RateLimitError, 429)

    monkeypatch.setattr(ic, "answer_in_scope", boom)
    with pytest.raises(RateLimitError):
        ic.answer_via_chain("q", _entries())


def test_openai_entry_uses_default_client_and_free_entry_does_not():
    entries = _entries()
    assert entries[-1].client() is None  # default OpenAI client, as before #77


def test_ask_response_names_the_provider_that_answered(monkeypatch):
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    import main
    from vera import public_mode as pm
    pm.reset_limiter(None)
    monkeypatch.setattr(main, "ASK_CHAIN", _entries())
    monkeypatch.setattr(main, "answer_via_chain", lambda q, entries, grounding=None: (AskResult("free answer", 42, 0.0), "groq:a"))
    r = TestClient(main.app).post("/ask", json={"question": "Do AI coding assistants help developers?"},
                                  headers={"X-API-Key": "k" * 20})
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "groq:a" and r.json()["cost_usd"] == 0.0


def test_eval_paths_do_not_read_the_chain():
    from pathlib import Path
    root = Path(ic.__file__).resolve().parent
    for rel in ("trace_eval/capture.py", "m6/baseline_collection.py", "gate_a.py", "pipeline_llm.py", "m6/judge.py"):
        src = (root / rel).read_text()
        assert "inference_chain" not in src and "ask-provider-chain" not in src, rel
