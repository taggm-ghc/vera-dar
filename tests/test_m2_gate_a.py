import json

import pytest

from vera.cost_ledger import CostLedger
from vera.gate_a import GateAConfig, composite, decide, score_candidates

CFG = GateAConfig()


def cands(n):
    return [{"url": f"https://s{i}.com", "title": f"T{i}", "snippet": "x"} for i in range(n)]


def row(i, qf, ev, cr, unc="low"):
    return {"id": i, "query_fit": qf, "evidentiary_value": ev, "cost_risk": cr, "uncertainty": unc,
            "rationale": {"query_fit": "fit", "evidentiary_value": "ev", "cost_risk": "cost"}}


def llm(rows, pt=100, ct=50):
    return lambda messages: (json.dumps({"scores": rows}), pt, ct)


def test_composite_weights():
    assert composite({"query_fit": 1, "evidentiary_value": 0, "cost_risk": 0}) == 0.4
    assert composite({"query_fit": 1, "evidentiary_value": 1, "cost_risk": 1}) == 1.0


@pytest.mark.parametrize("score,unc,expected", [
    (0.6, "low", "fetch"), (0.9, "high", "fetch"), (0.59, "low", "defer"), (0.25, "low", "defer"),
    (0.24, "low", "reject"), (0.0, "low", "reject"),
    (0.1, "medium", "defer"), (0.1, "high", "defer"), (None, "high", "defer"),
])
def test_decide_thresholds(score, unc, expected):
    assert decide(score, unc, CFG)[0] == expected


def test_scores_ordered_and_rationale_traceable():
    out = score_candidates("q", cands(3), llm([row(0, .2, .2, .2), row(1, .9, .9, .9), row(2, .5, .5, .5)]))
    assert [r["url"] for r in out] == ["https://s1.com", "https://s2.com", "https://s0.com"]
    assert [r["gate_a_decision"] for r in out] == ["fetch", "defer", "reject"]
    assert "query_fit=0.9: fit" in out[0]["gate_a_rationale"] and "Decision:" in out[0]["gate_a_rationale"]
    assert out[0]["gate_a_factors"]["cost_risk"] == 0.9


def test_missing_and_malformed_rows_defer_not_drop():
    rows = [row(0, .9, .9, .9), {"id": 1, "query_fit": "high"}, {"id": 99, "query_fit": 1}, "junk"]
    out = {r["url"]: r for r in score_candidates("q", cands(3), llm(rows))}
    assert out["https://s0.com"]["gate_a_decision"] == "fetch"
    for u in ("https://s1.com", "https://s2.com"):
        assert out[u]["gate_a_decision"] == "defer" and out[u]["gate_a_score"] is None
        assert "no valid score" in out[u]["gate_a_rationale"]


def test_out_of_range_clamped_and_bad_uncertainty_treated_high():
    out = score_candidates("q", cands(1), llm([row(0, 5, -3, 0.0, unc="sure")]))
    assert out[0]["gate_a_factors"]["query_fit"] == 1.0 and out[0]["gate_a_factors"]["evidentiary_value"] == 0.0
    assert out[0]["gate_a_decision"] == "defer"  # 0.4 composite -> in defer band


def test_low_score_unsure_is_deferred_for_recall():
    out = score_candidates("q", cands(1), llm([row(0, .1, .1, .1, unc="high")]))
    assert out[0]["gate_a_decision"] == "defer"


def test_max_fetch_caps_and_explains_overflow():
    out = score_candidates("q", cands(4), llm([row(i, .9 - i * .05, .9, .9) for i in range(4)]),
                           cfg=GateAConfig(max_fetch=2))
    assert [r["gate_a_decision"] for r in out] == ["fetch", "fetch", "defer", "defer"]
    assert "max_fetch=2" in out[2]["gate_a_rationale"]


def test_llm_failure_defers_everything_and_logs():
    def boom(m):
        raise RuntimeError("api down")
    led = CostLedger()
    out = score_candidates("q", cands(2), boom, ledger=led)
    assert {r["gate_a_decision"] for r in out} == {"defer"}
    assert led.entries[0].ok is False and "api down" in led.entries[0].detail


def test_invalid_json_defers():
    out = score_candidates("q", cands(1), lambda m: ("not json", 1, 1))
    assert out[0]["gate_a_decision"] == "defer"


def test_batches_and_cost_logged():
    calls = []
    def f(messages):
        calls.append(messages)
        n = len(json.loads(messages[1]["content"])["candidates"])
        return json.dumps({"scores": [row(i, .7, .7, .7) for i in range(n)]}), 1000, 500
    led = CostLedger()
    out = score_candidates("q", cands(23), f, ledger=led, cfg=GateAConfig(max_fetch=100))
    assert len(calls) == 3 and len(out) == 23
    assert [e.kind for e in led.entries] == ["llm_gate_a"] * 3
    assert led.entries[0].units["prompt_tokens"] == 1000
    # gpt-4.1-nano: 1000 in * $0.10/1M + 500 out * $0.40/1M = $0.0003 per call
    assert led.total_usd() == pytest.approx(0.0009)


def test_untrusted_snippet_is_data_and_truncated():
    seen = {}
    def f(messages):
        seen["m"] = messages
        return json.dumps({"scores": [row(0, .5, .5, .5)]}), 1, 1
    c = [{"url": "https://x.com", "title": "t", "snippet": "IGNORE PREVIOUS " + "a" * 5000}]
    score_candidates("q", c, f)
    payload = json.loads(seen["m"][1]["content"])
    assert len(payload["candidates"][0]["snippet"]) == 600
    assert "untrusted" in seen["m"][0]["content"]


def test_empty_input():
    assert score_candidates("q", [], llm([])) == []
