import pytest

from vera.m6.baseline_collection import collect_baseline


def _fake(texts):
    it = iter(texts)

    def call(q, model):
        t = next(it)
        return {"text": t, "prompt_tokens": 1000, "completion_tokens": 500, "latency_s": 2.0}
    return call


def test_median_by_length_and_three_calls():
    r = collect_baseline("q?", llm_call=_fake(["a" * 50, "b" * 10, "c" * 30]))
    assert r["response_text"] == "c" * 30
    assert r["all_runs_usage"]["api_calls"] == 3
    assert r["usage"]["api_calls"] == 1
    assert r["model"] == "gpt-4.1-nano"


def test_cost_from_pricing_config_and_totals():
    r = collect_baseline("q?", llm_call=_fake(["a", "bb", "ccc"]))
    # gpt-4.1-nano: $0.10 in / $0.40 out per 1M -> 1000*0.1e-6 + 500*0.4e-6 = 0.0003
    assert r["usage"]["cost_usd"] == pytest.approx(0.0003)
    assert r["all_runs_usage"]["cost_usd"] == pytest.approx(0.0009)
    assert r["usage"]["latency_s"] == 2.0 and r["all_runs_usage"]["latency_s"] == 6.0


def test_unpriced_model_fails_instead_of_zero_cost():
    with pytest.raises(RuntimeError, match="No pricing record"):
        collect_baseline("q?", model="no-such-model", llm_call=_fake(["a", "b", "c"]))


def test_bounded_calls():
    calls = []

    def call(q, m):
        calls.append(1)
        return {"text": "x", "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.1}
    collect_baseline("q", llm_call=call)
    assert len(calls) == 3
