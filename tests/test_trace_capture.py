"""Fakes-only tests for the item #70 capture harness (no network, no LLM)."""
import json

import httpx
import pytest
from openai import AuthenticationError

from vera.ask_service import answer_question
from vera.pricing.config import PricingRecord
from vera.trace_eval import capture

PRICING = PricingRecord(provider="p", model="m", input=1.0, output=2.0, source_url="u",
                        retrieved_at="2026-01-01", effective_from="2026-01-01")
SRC = [{"url": "http://a", "title": "T1", "snippet": "abs one", "published": "2025-01-01"},
       {"url": "http://b", "title": "T2", "snippet": "abs two"}]


def _doc(n=3, status="FROZEN"):
    return {"status": status, "items": [
        {"id": f"q{i}", "text": f"question {i}", "category": "c", "split": "dev" if i % 2 else "heldout"}
        for i in range(1, n + 1)]}


def _chat(model, messages):
    return "answer", 1000, 500  # cost 0.002


def _write(tmp_path, doc):
    p = tmp_path / "q.json"
    p.write_text(json.dumps(doc))
    return p


def test_baseline_messages_match_ask_shape(monkeypatch):
    seen = {}

    class C:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages):
                    seen["m"] = messages
                    raise RuntimeError("stop")

    monkeypatch.setattr("vera.ask_service._get_client", lambda: C)
    with pytest.raises(RuntimeError):
        answer_question("hello?", "m", PRICING)
    assert capture.baseline_messages("hello?") == seen["m"] == [{"role": "user", "content": "hello?"}]


def test_grounded_prompt_has_numbered_sources_and_cite_rule():
    srcs = [{"n": 1, **SRC[0]}, {"n": 2, **SRC[1]}]
    msgs = capture.grounded_v1_messages("q?", srcs)
    assert [m["role"] for m in msgs] == ["system", "user"] and msgs[1]["content"] == "q?"
    s = msgs[0]["content"]
    assert "[1] T1 (published 2025-01-01) http://a\nabs one" in s and "[2] T2" in s
    assert "ONLY the numbered sources" in s and "Never cite anything that is not listed" in s


def test_frozen_refusal_and_sha(tmp_path):
    with pytest.raises(ValueError):
        capture.load_questions(_write(tmp_path, _doc(status="DRAFT")))
    p = _write(tmp_path, _doc())
    doc, sha = capture.load_questions(p)
    import hashlib
    assert sha == hashlib.sha256(p.read_bytes()).hexdigest() and doc["status"] == "FROZEN"


def test_schema_and_sha_recorded(tmp_path):
    out = tmp_path / "o.jsonl"
    s = capture.run_capture(_doc(2), "grounded_v1", out, chat=_chat, search=lambda q: SRC, pricing=PRICING,
                            model="m", questions_sha256="abc")
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert s["n"] == 2 and s["errors"] == 0 and s["total_cost_usd"] == 0.004
    keys = {"id", "split", "category", "variant", "prompt_version", "questions_sha256", "model", "messages",
            "sources", "answer", "prompt_tokens", "completion_tokens", "cost_usd", "latency_s",
            "search_meta", "error"}
    assert keys <= set(rows[0]) and rows[0]["questions_sha256"] == "abc"
    assert rows[0]["sources"][0]["n"] == 1 and rows[0]["prompt_version"] == capture.PROMPT_VERSION_GROUNDED


def test_cost_guard_stops(tmp_path):
    out = tmp_path / "o.jsonl"
    s = capture.run_capture(_doc(5), "baseline", out, chat=_chat, search=lambda q: [], pricing=PRICING,
                            model="m", max_cost_usd=0.005)
    assert s["n"] == 2 and "cost guard" in s["stopped"]


def test_auth_error_aborts(tmp_path):
    def bad(model, messages):
        r = httpx.Response(401, request=httpx.Request("POST", "http://x"))
        raise AuthenticationError("nope", response=r, body=None)

    with pytest.raises(AuthenticationError):
        capture.run_capture(_doc(2), "baseline", tmp_path / "o.jsonl", chat=bad, search=lambda q: [],
                            pricing=PRICING, model="m")


def test_per_question_error_recorded_and_run_continues(tmp_path):
    calls = []

    def flaky(model, messages):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("slow")
        return "ok", 10, 10

    out = tmp_path / "o.jsonl"
    s = capture.run_capture(_doc(2), "baseline", out, chat=flaky, search=lambda q: [], pricing=PRICING, model="m")
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert s["n"] == 2 and s["errors"] == 1
    assert rows[0]["error"].startswith("TimeoutError") and rows[1]["error"] is None
