"""WED-OAI-OPT (G7 option b): the OpenAI fallback is optional; OFF means no OpenAI call and no OpenAI key needed.
Offline: every client is a stub or a recorder; no network, no DB."""
import json
import logging

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError, BadRequestError, RateLimitError

from vera import ask_service, claim_check as cc, corpus_description as cd, inference_chain as ic, scope_router
from vera.pricing.config import PricingRecord

SHIPPED = json.loads(ic.CONFIG_PATH.read_text())
CHECK_CFG = SHIPPED["grounding"]["claim_check"]
OPENAI_PRICING = PricingRecord(provider="openai", model="gpt-4.1-nano", input=0.1, output=0.4, source_url="u",
                               retrieved_at="2026-01-01", effective_from="2026-01-01")
GROQ_KEY = "gsk_" + "g" * 40
OPENAI_SENTINEL = "sk-" + "o" * 40
SRC = [{"n": 1, "title": "A", "url": "https://arxiv.org/abs/2401.00001", "snippet": "Copilot users were 55% faster."}]
CLAIMS = [{"id": 0, "text": "Faster [1].", "cites": [1]}]


def _cfg_file(tmp_path, block):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"checked_at": "2026-10-21", "timeout_s": 5, **block, "providers": [SHIPPED["providers"][0]]}))
    return p


def _status_error(cls, code):
    resp = httpx.Response(code, request=httpx.Request("POST", "https://provider.invalid/v1/chat/completions"))
    return cls("provider said secret", response=resp, body={"error": {"code": "x"}})


@pytest.fixture
def recorder(monkeypatch):
    """Class-level patch of openai.OpenAI.__init__: records every construction. A default client (base_url None) is a
    failure; a Groq-style client is refused with a connection error so callers fall through exactly as on an outage."""
    calls = []

    def init(self, *a, **kw):
        calls.append(kw.get("base_url"))
        raise APIConnectionError(request=httpx.Request("POST", "https://provider.invalid"))

    monkeypatch.setattr(openai.OpenAI, "__init__", init)
    return calls


@pytest.fixture
def main_off(monkeypatch):
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    import main
    from vera import public_mode as pm
    pm.reset_limiter(None)
    monkeypatch.setattr(main, "OPENAI_FALLBACK", False)
    return main


def test_shipped_switch_is_off_and_t3_reader_fails_safe(tmp_path):
    assert SHIPPED["openai_fallback"]["enabled"] is False and ic.openai_fallback_enabled() is False
    assert ic.openai_fallback_enabled(_cfg_file(tmp_path, {"openai_fallback": {"enabled": True}})) is True
    for block in ({}, {"openai_fallback": {}}, {"openai_fallback": {"enabled": "true"}}, {"openai_fallback": {"enabled": 1}},
                  {"openai_fallback": None}, {"openai_fallback": []}):
        assert ic.openai_fallback_enabled(_cfg_file(tmp_path, block)) is False, block
    assert ic.openai_fallback_enabled(tmp_path / "missing.json") is False
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert ic.openai_fallback_enabled(bad) is False


def test_t1_t2_load_chain_off_and_on(tmp_path):
    off = _cfg_file(tmp_path, {"openai_fallback": {"enabled": False}})
    env = {"GROQ_API_KEY": GROQ_KEY, "OPENAI_API_KEY": OPENAI_SENTINEL}
    labels = [e.label for e in ic.load_chain("gpt-4.1-nano", OPENAI_PRICING, off, env=env)]
    assert labels and all(x.startswith("groq:") for x in labels)  # openai_enabled=None reads the switch from `path`
    assert ic.load_chain("gpt-4.1-nano", OPENAI_PRICING, off, env={"OPENAI_API_KEY": OPENAI_SENTINEL}) == []
    on = _cfg_file(tmp_path, {"openai_fallback": {"enabled": True}})
    assert [e.label for e in ic.load_chain("gpt-4.1-nano", OPENAI_PRICING, on, env=env)][-1] == "gpt-4.1-nano"
    assert ic.load_chain("gpt-4.1-nano", OPENAI_PRICING, on, env=env, openai_enabled=False)[-1].base_url is not None


def test_t4_judge_off_skips_openai_checker(monkeypatch, caplog):
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_SENTINEL)
    tried = []

    class Down:
        def __init__(self, checker):
            self.checker, self.chat, self.completions = checker, self, self

        def create(self, **kw):
            tried.append(kw["model"])
            raise RuntimeError("down")

    monkeypatch.setattr(cc, "_client", lambda checker, t: Down(checker))
    with caplog.at_level(logging.WARNING, logger="vera"):
        assert cc.judge(CLAIMS, SRC, CHECK_CFG) == (None, None)
        text, summary = cc.check_answer("Faster [1].", SRC, CHECK_CFG)
    assert text is None and summary["outcome"] == "checker_unavailable"
    assert "gpt-4.1-nano" not in tried and "gpt-4.1-nano" not in caplog.text
    assert tried.count("openai/gpt-oss-120b") == 2 and tried.count("qwen/qwen3.8-27b") == 2
    # ON: the OpenAI checker is tried last
    monkeypatch.setattr(ic, "openai_fallback_enabled", lambda path=ic.CONFIG_PATH: True)
    tried.clear()
    cc.judge(CLAIMS, SRC, CHECK_CFG)
    assert tried[-1] == "gpt-4.1-nano"


def test_t5_startup_off_needs_no_openai_key_and_logs_providers(monkeypatch, caplog, main_off):
    main = main_off
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with caplog.at_level(logging.INFO, logger="vera"):
        main.startup_checks()
    assert "OPENAI_API_KEY looks" not in caplog.text
    assert "openai_fallback=off" in caplog.text and "OPENAI_API_KEY=unset" in caplog.text
    assert ask_service._DEFAULT_CLIENT_FORBIDDEN is True  # latch set by startup
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    monkeypatch.setattr(main, "ASK_CHAIN", ic.load_chain("gpt-4.1-nano", OPENAI_PRICING, ic.CONFIG_PATH, openai_enabled=False))
    monkeypatch.setattr(main, "answer_via_chain", lambda q, e, g=None: (ask_service.AskResult("free answer", 5, 0.0), "groq:a"))
    r = TestClient(main.app).post("/ask", json={"question": "Do AI coding assistants help developers?"},
                                  headers={"X-API-Key": "k" * 20})
    assert r.status_code == 200 and r.json()["model"].startswith("groq:")


def test_t8_startup_log_has_no_key_values(monkeypatch, caplog, main_off):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_SENTINEL)
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    with caplog.at_level(logging.INFO, logger="vera"):
        main_off.startup_checks()
    assert "openai_fallback=off" in caplog.text and "OPENAI_API_KEY=set" in caplog.text and "ignored" in caplog.text
    assert OPENAI_SENTINEL not in caplog.text and GROQ_KEY not in caplog.text and "checkers=" in caplog.text
    assert "gpt-4.1-nano" not in caplog.text.split("checkers=")[1]


@pytest.mark.parametrize("exc,status,fragment", [
    (_status_error(RateLimitError, 429), 429, "rate-limited"),
    (_status_error(BadRequestError, 400), 502, "failed"),
    (ask_service.OpenAIFallbackDisabled("x"), 503, "No answering provider"),
])
def test_t6_off_errors_are_provider_neutral(monkeypatch, main_off, exc, status, fragment):
    main = main_off
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def boom(q, e, g=None):
        raise exc

    monkeypatch.setattr(main, "answer_via_chain", boom)
    r = TestClient(main.app).post("/ask", json={"question": "q"}, headers={"X-API-Key": "k" * 20})
    assert r.status_code == status and fragment in r.json()["detail"] and "openai" not in r.json()["detail"].lower()


def test_t6_off_empty_chain_is_503_and_other_runtime_errors_are_not_masked(monkeypatch, main_off):
    main = main_off
    monkeypatch.setattr(main, "ASK_CHAIN", [])
    r = TestClient(main.app).post("/ask", json={"question": "q"}, headers={"X-API-Key": "k" * 20})
    assert r.status_code == 503 and r.json()["detail"] == "No answering provider is configured."

    def bug(q, e, g=None):
        raise RuntimeError("unrelated bug")

    monkeypatch.setattr(main, "answer_via_chain", bug)
    with pytest.raises(RuntimeError, match="unrelated"):
        TestClient(main.app, raise_server_exceptions=True).post("/ask", json={"question": "q"}, headers={"X-API-Key": "k" * 20})


def test_t12_on_keeps_openai_wording_and_falls_through_to_openai_entry(monkeypatch, main_off):
    main = main_off
    monkeypatch.setattr(main, "OPENAI_FALLBACK", True)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_SENTINEL)
    monkeypatch.setattr(main, "answer_via_chain", lambda q, e, g=None: (_ for _ in ()).throw(_status_error(BadRequestError, 400)))
    r = TestClient(main.app).post("/ask", json={"question": "q"}, headers={"X-API-Key": "k" * 20})
    assert r.status_code == 502 and r.json()["detail"] == "OpenAI request failed. Please try again."
    free = ic.ChainEntry("groq", "a", OPENAI_PRICING, "https://api.groq.com/openai/v1", "GROQ_API_KEY")
    entries = [free, ic.ChainEntry("openai", "gpt-4.1-nano", OPENAI_PRICING)]
    seen = []

    def fake(q, m, p, client=None, extra_body=None, **kw):
        seen.append(m)
        if m == "a":
            raise _status_error(BadRequestError, 400)
        return ask_service.AskResult("ok", 1, 0.0)

    monkeypatch.setattr(ic, "answer_in_scope", fake)
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    monkeypatch.setattr(openai.OpenAI, "__init__", lambda self, **kw: None)
    assert ic.answer_via_chain("q", entries)[1] == "gpt-4.1-nano" and seen == ["a", "gpt-4.1-nano"]


def test_latch_blocks_default_client_before_the_cached_constructor(monkeypatch):
    built = []
    monkeypatch.setattr(ask_service, "OpenAI", lambda **kw: built.append(kw) or object())
    ask_service._get_client.cache_clear()
    assert ask_service._get_client() is ask_service._get_client() and len(built) == 1  # cached when not latched
    ask_service.forbid_default_client()
    with pytest.raises(ask_service.OpenAIFallbackDisabled):
        ask_service._get_client()  # latch beats a warm cache
    assert len(built) == 1
    ask_service.reset_default_client_latch()
    ask_service._get_client.cache_clear()


def test_latch_does_not_leak_between_tests_a():
    ask_service.forbid_default_client()


def test_latch_does_not_leak_between_tests_b():
    assert ask_service._DEFAULT_CLIENT_FORBIDDEN is False  # autouse fixture in tests/conftest.py resets it


@pytest.mark.parametrize("latched", [True, False], ids=["latch_on", "latch_off_filters_alone"])
def test_t7_zero_spend_guard(monkeypatch, recorder, caplog, main_off, latched):
    """OFF + a well-formed OPENAI_API_KEY: no default (base_url None) OpenAI client is ever constructed, and the OpenAI
    chain entry / checker are never even attempted. With latched=False the filters alone are proven."""
    main = main_off
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_SENTINEL)
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    if latched:
        ask_service.forbid_default_client()
    else:
        ask_service.reset_default_client_latch()
    entries = ic.load_chain("gpt-4.1-nano", OPENAI_PRICING, ic.CONFIG_PATH)
    assert entries and all(e.base_url is not None for e in entries)
    attempted = []
    real_e_client = ic.ChainEntry.client
    monkeypatch.setattr(ic.ChainEntry, "client", lambda self: attempted.append(self.label) or real_e_client(self))
    with caplog.at_level(logging.WARNING, logger="vera"):
        # chain: every Groq client construction is refused; the OpenAI entry must not exist or be tried
        with pytest.raises(APIConnectionError):
            ic.answer_via_chain("Do AI coding assistants help developers?", entries)
        # claim checker: Groq checkers fail, OpenAI checker is skipped
        assert cc.judge(CLAIMS, SRC, CHECK_CFG) == (None, None)
        # corpus description generator over the same chain
        with pytest.raises(RuntimeError, match="no provider produced"):
            cd.generate_with_chain(entries)([{"role": "user", "content": "x"}])
        # the API route, end to end with real chain and router
        monkeypatch.setattr(main, "ASK_CHAIN", entries)
        monkeypatch.setattr(main, "GROUNDING", None)
        r = TestClient(main.app).post("/ask", json={"question": "Do AI coding assistants help developers?"},
                                      headers={"X-API-Key": "k" * 20})
    assert r.status_code == 502 and "openai" not in r.json()["detail"].lower()
    assert recorder and all(b is not None for b in recorder), recorder  # zero default-client constructions
    assert "gpt-4.1-nano" not in attempted and "gpt-4.1-nano" not in caplog.text
    assert "openai:gpt-4.1-nano" not in caplog.text
    if latched:
        with pytest.raises(ask_service.OpenAIFallbackDisabled):
            ask_service._get_client()
        with pytest.raises(ask_service.OpenAIFallbackDisabled):
            scope_router.default_chat([{"role": "user", "content": "x"}], "m", {"temperature": 0, "max_tokens": 1, "timeout_s": 1})
        assert all(b is not None for b in recorder)


class _Stub:
    def __init__(self, replies):
        self.replies, self.kw = list(replies), []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.kw.append(kw)
        content = self.replies.pop(0)
        msg = type("M", (), {"content": content})
        usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})
        return type("R", (), {"choices": [type("C", (), {"message": msg})], "usage": usage})


def test_t10_router_cap_and_checker_json_mode(monkeypatch):
    cfg = json.loads(scope_router.CONFIG_PATH.read_text())["layer2"]
    assert cfg["max_tokens"] == 512
    stub = _Stub(['{"in_scope": true, "reason": "r"}'])
    scope_router.default_chat([{"role": "user", "content": "x"}], "m", cfg, client=stub)
    assert stub.kw[0]["max_tokens"] == cfg["max_tokens"] and stub.kw[0]["response_format"] == {"type": "json_object"}

    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    good = json.dumps({"claims": [{"id": 0, "verdict": "supported", "reason": "r"}]})
    stub = _Stub(["<think>x</think> not json", good])
    monkeypatch.setattr(cc, "_client", lambda checker, t: stub)
    verdicts, label = cc.judge(CLAIMS, SRC, CHECK_CFG)
    assert label == "groq:openai/gpt-oss-120b" and verdicts[0]["verdict"] == "supported"
    assert stub.kw[0]["model"] == "qwen/qwen3.8-27b" and "response_format" not in stub.kw[0]
    assert stub.kw[1]["model"] == "openai/gpt-oss-120b" and stub.kw[1]["response_format"] == {"type": "json_object"}
