"""FIXTURE report for the demo UI and tests. NOT a real VERA-DAR run.

Every text here is a placeholder labelled FIXTURE; no real study, number, or
URL is asserted. It is produced by running the real M6 scorers over scripted
judge output, so the structure matches a genuine eval_report exactly. The UI
shows a banner whenever it renders this. It is never written to the database.
"""
from __future__ import annotations

from vera.m6.adapters import load_freeze
from vera.m6.comparative_eval import evaluate_both
from vera.m6.judge import JudgeUsage
from vera.m6.m6_runner import build_report

QUESTION = ("What does published evidence (2023-2026) show about AI coding assistants' effect on "
            "developer productivity, and why do the findings disagree?")
ENG_TAG, BASE_TAG = "[FIXTURE-ENGINEERED]", "[FIXTURE-BASELINE]"


class ScriptedJudge:
    """Test double standing in for the different-family judge. Replies by purpose."""
    family, model = "fixture", "scripted-judge"

    def __init__(self, replies: dict):
        self.replies, self.usage, self.prompts = replies, JudgeUsage(), []

    def judge_json(self, purpose: str, prompt: str) -> dict:
        self.prompts.append((purpose, prompt))
        self.usage.calls += 1
        r = self.replies[purpose]
        return r(prompt) if callable(r) else r


def _side(prompt, eng, base):
    return eng if ENG_TAG in prompt else base


def build_corpus() -> dict:
    fz = load_freeze()
    spans, sources = {}, {}
    for i in range(1, 7):
        sid = f"src{i}"
        sources[sid] = {"url": f"fixture://source-{i}", "title": f"FIXTURE source {i}",
                        "content_hash": f"{i:x}" * 64, "version": 1, "gate_b_decision": "admit" if i != 5 else "qualify",
                        "aliases": []}
        spans[f"sp{i}"] = {"text": f"FIXTURE SPAN {i}: placeholder evidence text, illustrative only.",
                           "source_id": sid, "start": 100 * i, "end": 100 * i + 60, "evidence_type": "fact"}
    return {"requirements": fz["requirements"], "objections": fz["objections"], "spans": spans,
            "sources": sources, "rubric_version": "fixture-rubric", "policy_version": "fixture-policy"}


def build_engineered(corpus: dict) -> dict:
    claims = [{"id": f"c{i}", "text": f"{ENG_TAG} fixture claim {i}.", "citations": [f"sp{i}"],
               "consequential": True, "verification_status": "supported"} for i in range(1, 7)]
    return {
        "response_text": f"{ENG_TAG} FIXTURE engineered answer (placeholder, not a real finding).",
        "claims": claims,
        "usage": {"api_calls": 42, "tokens": 180_000, "cost_usd": 0.31, "latency_s": 240.0, "complete": True},
        "audit": {
            "gate_a": [{"source_id": f"src{i}", "score": round(0.9 - 0.05 * i, 2), "decision": "fetch",
                        "rationale": f"FIXTURE Gate A rationale {i}", "url": f"fixture://source-{i}"}
                       for i in range(1, 7)],
            "gate_b": [{"span_id": f"sp{i}", "source_id": f"src{i}",
                        "decision": "admit" if i != 5 else "qualify",
                        "rubric": {"risk_of_bias": 4, "indirectness": 4, "applicability": 3},
                        "overall_quality": 3.6, "rationale": f"FIXTURE Gate B rationale {i}",
                        "rubric_version": "fixture-rubric", "policy_version": "fixture-policy"}
                       for i in range(1, 7)],
            "gate_c": {"trace": ["search_again", "adequate"], "decision": "adequate",
                       "detail": [{"coverage": {"sq_controlled": "met", "sq_field": "met",
                                                "sq_quality": "partial"},
                                   "missing_evidence": [{"id": "sq_quality", "need": "FIXTURE: quality evidence"}]}]},
            "relations": [{"span1_id": "sp1", "span2_id": "sp2", "relation_type": "refutes", "confidence": 0.8,
                           "rationale": "FIXTURE relation"},
                          {"span1_id": "sp3", "span2_id": "sp1", "relation_type": "qualifies", "confidence": 0.7,
                           "rationale": "FIXTURE relation"},
                          {"span1_id": "sp4", "span2_id": "sp3", "relation_type": "supports", "confidence": 0.9,
                           "rationale": "FIXTURE relation"}],
            "verification_status": "revised_after_flags", "unsupported_claims": []}}


def build_judge() -> ScriptedJudge:
    reqs = [r["id"] for r in load_freeze()["requirements"]]
    objs = [o["id"] for o in load_freeze()["objections"]]
    rel = lambda p: {"scores": {r: {"score": (1 if k < 6 else 0.5) if ENG_TAG in p else (0.5 if k < 2 else 0),
                                    "rationale": "FIXTURE scripted score"} for k, r in enumerate(reqs)}}
    split = {"claims": [{"text": f"{BASE_TAG} fixture baseline claim {i}.", "refs": []} for i in range(4)]}
    ver = lambda p: {"verdicts": {f"c{i}": {"label": "supported", "rationale": "FIXTURE scripted verdict"}
                                  for i in range(1, 7)}}
    rea = lambda p: {"objections": {o: {"label": ("addressed" if k < 5 else "mentioned") if ENG_TAG in p
                                         else ("mentioned" if k < 2 else "ignored"),
                                         "rationale": "FIXTURE scripted label"} for k, o in enumerate(objs)},
                     "overclaims": 0 if ENG_TAG in p else 2, "uncertainty_attached": ENG_TAG in p,
                     "contradicts_evidence": False, "overclaim_examples": []}
    return ScriptedJudge({"relevance": rel, "claim-split": split, "claim-verify": ver, "reasoning": rea})


def build_fixture_report() -> dict:
    corpus, eng = build_corpus(), None
    eng = build_engineered(corpus)
    base = {"response_text": f"{BASE_TAG} FIXTURE baseline answer (placeholder).",
            "usage": {"api_calls": 1, "tokens": 1500, "cost_usd": 0.0006, "latency_s": 9.0}}
    judge = build_judge()
    ev = evaluate_both(eng, base, corpus, QUESTION, judge=judge)
    baseline = {"response_text": base["response_text"], "usage": base["usage"], "model": "gpt-4.1-nano",
                "method": "FIXTURE", "all_runs_usage": {"api_calls": 3, "tokens": 4500, "cost_usd": 0.0018,
                                                         "latency_s": 27.0}}
    rep = build_report({"run_id": "FIXTURE", "question": QUESTION}, eng, baseline, ev, judge.usage.as_dict(), corpus)
    rep["is_fixture"] = True
    return rep
