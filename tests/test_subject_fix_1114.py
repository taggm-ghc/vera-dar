"""SUBJECT-BUILD-1114 fix cycle: warm-up wiring (defect 3), _ask_impl-level T14/T25 (defect 4), clamps (6), engine bounds (2).
Fakes only: no DB, no network, no model."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from vera import subject, subject_names
from vera.ask_service import AskResult

ROOT = Path(main.__file__).resolve().parent
CFG_BOTH = {"enabled": True, "answer_subject": {"enabled": True}, "stored_taxonomy": {"enabled": True},
            "subject_names": {"enabled": True, "max_calls_per_version": 5}}
CFG_OFF = {"enabled": True, "answer_subject": {"enabled": False}, "stored_taxonomy": {"enabled": True},
           "subject_names": {"enabled": True}}


def OPEN():
    return {"display_ok": True, "version": "v1"}


@pytest.fixture(autouse=True)
def _reset():
    main._SUBJ_WARM["version"] = None


def test_hook_flags_off_no_thread_no_model(monkeypatch):
    started = []
    monkeypatch.setattr(main, "json_with_chain", lambda e: started.append("llm"))
    assert main._maybe_warm_subject_names(CFG_OFF, OPEN, starter=lambda *a, **k: started.append("t")) is False
    closed = main._maybe_warm_subject_names(CFG_BOTH, lambda: {"display_ok": False, "version": "v1"},
                                            starter=lambda *a, **k: started.append("t"))
    assert closed is False and started == []


def test_hook_starts_once_per_version(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "json_with_chain", lambda e: "LLM")

    def st(*a, **k):
        calls.append(a)
    assert main._maybe_warm_subject_names(CFG_BOTH, OPEN, starter=st) is True
    assert main._maybe_warm_subject_names(CFG_BOTH, OPEN, starter=st) is False
    assert len(calls) == 1 and calls[0][0] is CFG_BOTH and calls[0][2] == "LLM" and calls[0][3] is OPEN
    assert main._maybe_warm_subject_names(CFG_BOTH, lambda: {"display_ok": True, "version": "v2"}, starter=st) is True
    assert len(calls) == 2


def test_hook_swallows_exceptions(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(main, "json_with_chain", lambda e: "LLM")
    assert main._maybe_warm_subject_names(CFG_BOTH, OPEN, starter=boom) is False


def test_clamps():
    b = subject_names._bounds
    assert b({"max_calls_per_version": 10**9, "call_timeout_s": 999}) == (1000, 30.0)
    assert b({"max_calls_per_version": -5, "call_timeout_s": 0}) == (0, 1.0)
    assert b({"max_calls_per_version": "9", "call_timeout_s": "x"}) == (0, subject_names.DEFAULT_TIMEOUT_S)
    assert b({"max_calls_per_version": True})[0] == 0 and b({})[0] == 0
    assert b({"max_calls_per_version": 200, "call_timeout_s": 10}) == (200, 10.0)


def test_subject_engine_bounds():
    src = (ROOT / "vera" / "subject.py").read_text()
    assert "pool_timeout=0.5" in src and '"connect_timeout": 1}' in src


SRC = [{"n": 1, "title": "Paper One", "url": "https://arxiv.org/abs/2401.00001", "published": "2025-01-01",
        "snippet": "S", "identifier": "arXiv:2401.00001", "provider": "arxiv", "licence_decision": "allow",
        "retrieved_at": "2026-10-01", "evidence_eligible": True}]


def _ask(monkeypatch):
    monkeypatch.setenv("VERA_API_KEY", "k" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    from vera import public_mode as pm
    pm.reset_limiter(None)
    result = AskResult(answer="A", tokens_used=1, cost_usd=0.0, sources=tuple(SRC), claim_check={"outcome": "X"},
                       accepted=True, claims=(), candidates=[], contract={"layers": ["L0"], "fallback": None},
                       result_kind="answer")
    monkeypatch.setattr(main, "answer_via_chain", lambda q, e, g: (result, "groq:m"))
    monkeypatch.setattr(main, "store_from_env", lambda cfg: None)
    monkeypatch.setattr(main, "memory_from_env", lambda block: None)
    r = TestClient(main.app).post("/ask", json={"question": "q"}, headers={"X-API-Key": "k" * 20})
    assert r.status_code == 200, r.text
    return r


def test_ask_flag_off_untouched(monkeypatch):
    """T14 at _ask_impl level: flag off: no subject code runs and the body has no assoc keys; the body equals the one produced
    when every subject helper would raise if called."""
    monkeypatch.setattr(main, "_DESC_CFG", CFG_OFF)
    base = _ask(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("subject code ran with the flag off")
    monkeypatch.setattr(main, "_attach_subjects", boom)
    monkeypatch.setattr(main, "_maybe_warm_subject_names", boom)
    monkeypatch.setattr(subject, "subjects_for", boom)
    again = _ask(monkeypatch)
    assert [s for s in base.json()["sources"]] == [s for s in again.json()["sources"]]
    assert all("assoc_" not in k for s in base.json()["sources"] for k in s)


def test_ask_flags_on_empty_cache_makes_no_model_call(monkeypatch):
    """T25 at _ask_impl level: flags on, empty name cache: the model-call counter stays 0 (the warm-up thread is only started,
    here by a recorder, and never awaited)."""
    monkeypatch.setattr(main, "_DESC_CFG", CFG_BOTH)
    from vera import corpus_taxonomy
    monkeypatch.setattr(corpus_taxonomy, "job_status", OPEN)
    llm_calls, started = [], []
    monkeypatch.setattr(main, "json_with_chain", lambda e: (lambda *a, **k: llm_calls.append(1)))
    monkeypatch.setattr(subject_names, "start_warm_up", lambda *a, **k: started.append(a))
    r = _ask(monkeypatch)
    assert llm_calls == [] and len(started) == 1
    assert subject_names.get_display("topic", 1) is None
    assert all("assoc_" not in k for s in r.json()["sources"] for k in s)  # web source: no field
