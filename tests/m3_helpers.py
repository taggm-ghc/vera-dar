"""Shared fakes for the M3 tests (kept out of conftest.py, which other milestones own)."""
from vera.m3.llm import CallLog, CallRecord


class FakeLLM:
    """Scripted JSON-mode LLM: handlers map purpose -> dict | callable(user_prompt) -> dict."""

    def __init__(self, **handlers):
        self.handlers = handlers
        self.log = CallLog()
        self.prompts = []

    def complete_json(self, system, user, purpose):
        self.prompts.append((purpose, user))
        h = self.handlers[purpose]
        out = h(user) if callable(h) else h
        self.log.add(CallRecord(purpose, "fake", 100, 50, 0.0001, 0.0))
        return out


def domains(bias=4, inc=4, ind=4, imp=4, pub=4, app=4, why="because excerpts say so"):
    def d(s):
        return {"score": s, "rationale": why if s is not None else "", "cited_spans": [0]}
    return {"domains": {"risk_of_bias": d(bias), "inconsistency": d(inc), "indirectness": d(ind),
                        "imprecision": d(imp), "publication_bias": d(pub), "applicability": d(app)},
            "flags": {"independence": "academic", "temporal_validity": "2025"}}
