import pytest

from vera.m6 import eval_rubric as R
from vera.m6.judge import (BaseJudge, BudgetExceeded, JudgeConfigError, JudgeLimits,
                           parse_json_loose)
from vera.m7.fixture import ENG_TAG, ScriptedJudge, build_corpus, build_engineered


def test_relevance_levels_boundaries():
    assert [R.relevance_level(x) for x in (0.19, 0.2, 0.39, 0.4, 0.59, 0.6, 0.79, 0.8)] == [1, 2, 2, 3, 3, 4, 4, 5]


def test_relevance_scoring_known_example():
    corpus = build_corpus()
    ids = [r["id"] for r in corpus["requirements"]]  # 8
    scores = {i: {"score": 1 if k < 4 else 0.5 if k < 6 else 0} for k, i in enumerate(ids)}  # 4+1 = 5/8
    j = ScriptedJudge({"relevance": {"scores": scores}})
    out = R.score_relevance({"response_text": "x"}, corpus, j)
    assert out["coverage"] == pytest.approx(5 / 8) and out["score"] == 4


def test_relevance_invalid_judge_values_score_zero():
    corpus = build_corpus()
    j = ScriptedJudge({"relevance": {"scores": {corpus["requirements"][0]["id"]: {"score": 0.9}}}})
    assert R.score_relevance({"response_text": "x"}, corpus, j)["coverage"] == 0


def test_grounding_levels_and_fabrication_cap():
    assert [R.grounding_level(x, 0) for x in (0.39, 0.4, 0.59, 0.6, 0.79, 0.8, 0.89, 0.9)] == [1, 2, 2, 3, 3, 4, 4, 5]
    assert R.grounding_level(1.0, 1) == 2


def test_grounding_deterministic_labels_for_engineered():
    corpus = build_corpus()
    eng = {"response_text": "x", "claims": [
        {"id": "a", "text": "t", "citations": ["sp1"]},
        {"id": "b", "text": "t", "citations": []},
        {"id": "c", "text": "t", "citations": ["sp1", "NOPE"]}]}
    j = ScriptedJudge({"claim-verify": {"verdicts": {"a": {"label": "supported", "rationale": "ok"}}}})
    out = R.score_grounding(eng, corpus, j)
    labels = {c["id"]: c["label"] for c in out["claims"]}
    assert labels == {"a": "supported", "b": "uncited", "c": "fabricated-citation"}
    assert out["fabricated"] == 1 and out["score"] <= 2
    assert out["traceable_rate"] == pytest.approx(1 / 3)


def test_grounding_partial_not_counted_as_traceable():
    corpus = build_corpus()
    eng = {"response_text": "x", "claims": [{"id": "a", "text": "t", "citations": ["sp1"]}]}
    j = ScriptedJudge({"claim-verify": {"verdicts": {"a": {"label": "partial", "rationale": "r"}}}})
    out = R.score_grounding(eng, corpus, j)
    assert out["traceable_rate"] == 0 and out["partial_rate"] == 1


def test_baseline_unresolved_reference_is_unsupported_not_fabricated():
    corpus = build_corpus()
    j = ScriptedJudge({"claim-split": {"claims": [{"text": "c1", "refs": ["Smith 2024"]},
                                                  {"text": "c2", "refs": []},
                                                  {"text": "c3", "refs": ["fixture://source-1"]}]},
                       "claim-verify": lambda p: {"verdicts": {"c3": {"label": "supported", "rationale": "r"}}}})
    out = R.score_grounding({"response_text": "free text"}, corpus, j)
    labels = {c["id"]: c["label"] for c in out["claims"]}
    assert labels == {"c1": "unsupported", "c2": "uncited", "c3": "supported"}
    assert out["fabricated"] == 0


def test_reasoning_levels():
    L = R.reasoning_level
    assert L(["addressed"] * 5, 0, True, False) == 5
    assert L(["addressed"] * 5, 0, False, False) == 4   # no uncertainty attached
    assert L(["addressed"] * 4 + ["ignored"], 1, False, False) == 4
    assert L(["addressed"] * 4 + ["ignored"], 2, True, False) == 3
    assert L(["mentioned"] * 3 + ["ignored"] * 2, 2, False, False) == 3
    assert L(["mentioned"] * 3 + ["ignored"] * 2, 3, False, False) == 2
    assert L(["mentioned", "ignored", "ignored", "ignored", "ignored"], 0, False, False) == 2
    assert L(["ignored"] * 5, 0, False, False) == 1
    assert L(["addressed"] * 5, 0, True, True) == 1     # contradicts evidence


def test_reasoning_requires_four_objections():
    corpus = build_corpus()
    corpus["objections"] = corpus["objections"][:3]
    with pytest.raises(ValueError, match="K>=4"):
        R.score_reasoning({"response_text": "x"}, corpus, ScriptedJudge({}))


def test_cost_levels():
    assert [R.cost_level(x) for x in (None, 2.5, 2.0, 1.01, 1.0, 0.51, 0.5, 0.26, 0.25, 0.1)] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_cost_binding_constraint_and_missing_budget():
    b = {"cost_usd": 1.0, "latency_s": 600.0}
    u = {"cost_usd": 0.1, "latency_s": 590.0, "complete": True}   # latency is binding (98%)
    assert R.score_cost_latency({"usage": u}, b)["score"] == 3
    assert R.score_cost_latency({"usage": u}, {})["score"] == 1


def test_cost_incomplete_usage_is_unverified_not_passed():
    out = R.score_cost_latency({"usage": {"cost_usd": 0.01, "latency_s": 1, "complete": False}},
                               {"cost_usd": 1.0, "latency_s": 600.0})
    assert out["score"] == 1 and out["unverified"]


def test_audit_full_trace_is_5_and_degrades_stepwise():
    corpus, eng = build_corpus(), None
    eng = build_engineered(corpus)
    assert R.score_auditability(eng, corpus)["score"] == 5
    eng["audit"]["gate_a"] = []                       # no Gate A -> level 4 only
    assert R.score_auditability(eng, corpus)["score"] == 4
    eng["audit"]["gate_b"] = []                       # no Gate B rationale -> level 3
    assert R.score_auditability(eng, corpus)["score"] == 3
    for s in corpus["sources"].values():
        s["url"] = None                               # no URLs -> 2
    assert R.score_auditability(eng, corpus)["score"] == 2
    for c in eng["claims"]:
        c["citations"] = []                           # nothing cited -> black box
    assert R.score_auditability(eng, corpus)["score"] == 1


def test_audit_sample_is_seeded_and_bounded():
    corpus = build_corpus()
    eng = build_engineered(corpus)
    a = R.score_auditability(eng, corpus, seed=7)
    b = R.score_auditability(eng, corpus, seed=7)
    assert [t["claim_id"] for t in a["sampled"]] == [t["claim_id"] for t in b["sampled"]]
    assert 3 <= len(a["sampled"]) <= 5


def test_audit_free_text_baseline_is_black_box():
    assert R.score_auditability({"response_text": "no sources"}, build_corpus())["score"] == 1
    assert R.score_auditability({"response_text": "see https://x.org"}, build_corpus())["score"] == 2


def test_freeze_hash_changes_with_requirements():
    c = build_corpus()
    h = R.freeze_hash(c)
    c["objections"][0]["text"] += "!"
    assert R.freeze_hash(c) != h


# ---- judge guards
def test_same_family_judge_refused():
    class Same(BaseJudge):
        family = "openai"
    with pytest.raises(JudgeConfigError, match="different model family"):
        Same()


def test_judge_missing_key_fails_verbosely(monkeypatch):
    from vera.m6.judge import OpenAICompatJudge
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(JudgeConfigError, match="R4b"):
        OpenAICompatJudge("groq", model="qwen/qwen3-32b", family="qwen")


class _Stub(BaseJudge):
    family, model = "stub", "stub"

    def __init__(self, replies, **kw):
        super().__init__(**kw)
        self.replies = list(replies)

    def _call(self, prompt):
        return self.replies.pop(0), 100, 50

    def _price(self, a, b):
        return 0.01


def test_judge_call_cap_raises_verbose():
    j = _Stub(['{"a":1}'] * 3, limits=JudgeLimits(max_calls=2))
    j.judge_json("p1", "x"); j.judge_json("p2", "x")
    with pytest.raises(BudgetExceeded, match="calls 2/2"):
        j.judge_json("p3", "x")


def test_judge_cost_cap_and_parse_retry():
    j = _Stub(["not json", '```json\n{"ok": 1}\n```'], limits=JudgeLimits(max_cost_usd=0.015))
    assert j.judge_json("p", "x") == {"ok": 1}
    assert j.usage.calls == 2
    with pytest.raises(BudgetExceeded, match="cost"):
        j.judge_json("p", "x")


def test_judge_unparseable_gives_up():
    j = _Stub(["bad", "worse"], limits=JudgeLimits(max_parse_retries=1))
    with pytest.raises(RuntimeError, match="unparseable"):
        j.judge_json("p", "x")


def test_parse_json_loose():
    assert parse_json_loose('text {"a": 2} more') == {"a": 2}
