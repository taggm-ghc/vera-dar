import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vera.pipeline_llm import LLMResult  # noqa: E402


def span(sid, text, sqs, src=None, year=2025, typ="rct", **kw):
    return {"span_id": sid, "source_id": src or f"src_{sid}", "text": text, "sub_question_ids": sqs,
            "source_title": f"Title {sid}", "year": year, "source_type": typ, **kw}


def good_corpus():
    return {"run_id": "t", "question": "q", "spans": [
        span("s1", "Developers using the assistant completed the task 55.8% faster in a controlled experiment.", ["sq_controlled"]),
        span("s2", "Experienced open-source developers took 19% longer with AI tools in a randomized trial.", ["sq_controlled"], typ="rct"),
        span("s3", "Field data from three companies showed 26% more completed tasks per week.", ["sq_field"]),
        span("s4", "Telemetry across a large organisation showed modest throughput gains.", ["sq_field"]),
        span("s5", "Code churn rose after adoption, suggesting lower maintainability.", ["sq_quality"]),
        span("s6", "Security defect rates were unchanged in a controlled comparison.", ["sq_quality"]),
        span("s7", "Task type and developer experience moderate measured productivity effects.", ["sq_disagreement"]),
        span("s8", "Developers perceived speedups that measured outcomes did not confirm.", ["sq_disagreement"]),
    ]}


@pytest.fixture
def corpus():
    return good_corpus()


class FakeLLM:
    """Routes on system-prompt keywords; records calls. Override handlers per test."""
    def __init__(self, **handlers):
        self.handlers, self.calls = handlers, []

    def __call__(self, system, user):
        self.calls.append((system, user))
        for key, data in self.handlers.items():
            if key.replace("_", " ") in system.lower():
                data = data(system, user) if callable(data) else data
                return LLMResult(data=data, model="fake", tokens=10, cost_usd=0.001, latency_s=0.01)
        raise AssertionError(f"no fake handler for system prompt: {system[:80]}")
