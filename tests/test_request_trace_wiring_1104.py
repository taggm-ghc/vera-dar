"""INTEGRATE-1104-TRACE: request_trace wiring in fulldoc_answer, claim_check, search_and_fetch, search_providers and
inline_acquire. Offline stubs only. Asserts the events appear with ids/scores/codes only and carry no URL/title/text."""
import contextvars
import json
import time
from types import SimpleNamespace

import pytest

from tests.test_fulldoc_premise import run
from tests.test_fulldoc_answer import key  # noqa: F401  (autouse env fixture)
from vera import inline_acquire, request_trace as rt, search_and_fetch as saf, search_providers as sp

FORBIDDEN_KEYS = ("url", "uri", "host", "doi", "title", "text", "snippet", "sentence", "reason_text", "question",
                  "answer", "quote")


@pytest.fixture(autouse=True)
def _reset():
    yield
    rt.end_context()


def traced(fn):
    def body():
        rt.start("CANARY-QUESTION-TEXT?", {"endpoint": "/ask"})
        fn()
        return list(rt.CURRENT.get().events)
    return contextvars.copy_context().run(body)


def codes(events):
    return {(e["stage"], e["code"]) for e in events}


def assert_clean(events):
    blob = json.dumps(events)
    assert "CANARY-QUESTION-TEXT" not in blob
    for e in events:
        for k in (e.get("detail") or {}):
            assert k not in FORBIDDEN_KEYS, (e, k)
    for bad in ("http://", "https://", "10.1000", "Alpha", "gamma results"):
        assert bad not in blob, bad


def test_premise_claim_check_and_passages_events():
    ev = traced(lambda: run("PREMISE: false - The passage says gamma results are short, not long [2]."))
    c = codes(ev)
    assert {("premise", "verdict_flagged"), ("premise", "verified"), ("passages", "sent_to_writer"),
            ("claim_check", "verdicts"), ("claim_check", "checked"), ("passages", "evidence_trace"),
            ("passages", "verbatim_drop"), ("writer", "draft")} <= c
    sent = next(e for e in ev if e["code"] == "sent_to_writer")
    assert sent["detail"]["span_ids"] and len(sent["detail"]["span_ids"]) == len(sent["detail"]["relevance"])
    pc = next(e for e in ev if e["code"] == "verdicts")["detail"]["per_claim"]
    assert all(set(x) == {"id", "verdict", "cites"} for x in pc)
    assert_clean(ev)


def test_premise_ok_unverified_and_writer_fall():
    assert ("premise", "verdict_ok") in codes(traced(lambda: run("Plain answer [1].")))
    ev = traced(lambda: run("PREMISE: false - uncited correction"))
    assert ("premise", "unverified") in codes(ev)
    assert_clean(ev)
    ev = traced(lambda: run("Plain answer [1].", verdicts=("unsupported",)))
    fall = next(e for e in ev if e["code"] == "fall")
    assert fall["detail"]["fallback"] and "claim_check" in {e["stage"] for e in ev}
    assert_clean(ev)


def test_noop_outside_trace():
    assert rt.CURRENT.get() is None
    out, _ = run("PREMISE: ok\nPlain answer [1].")  # must not raise, nothing recorded
    assert out is not None and rt.CURRENT.get() is None
    sp.record_portal("p", "completed", 3)


def test_record_portal_and_fan_out_provider_events(monkeypatch):
    class P:
        def __init__(self, name):
            self.name, self.endpoint = name, "https://example.invalid/x"

    def fake_attempts(p, q, n, **kw):
        if p.name == "bad":
            raise saf._ProviderFailed(["HTTP 429 from https://example.invalid/secret"])
        return [{"url": "https://example.invalid/a", "title": "Alpha Title", "snippet": "x", "rank": 1}] if p.name == "ok" else []
    monkeypatch.setattr(saf, "_attempts", fake_attempts)
    monkeypatch.setattr(saf, "_finalize", lambda raw, name, n: list(raw))
    monkeypatch.setattr(saf, "rrf_merge", lambda per, k, n: [i for v in per.values() for i in v])
    monkeypatch.setattr(saf, "apply_soft_share", lambda m, s: m)
    monkeypatch.setattr(saf, "_route_queries", lambda c: [])
    monkeypatch.setattr(saf, "_relaxed_flags", lambda c: [])
    monkeypatch.setattr(saf, "_dedupe_merges", lambda i: 0)

    def go():
        saf._fan_out([P("ok"), P("empty"), P("bad")], "q", 5, 1, 60, {})
        sp.record_portal("openalex", "completed", 4)
    ev = traced(go)
    prov = {e["detail"]["provider"]: e for e in ev if e["code"] == "provider"}
    assert prov["ok"]["detail"]["status"] == "ok" and prov["ok"]["counts"]["results"] == 1
    assert prov["empty"]["detail"]["status"] == "empty"
    assert prov["bad"]["detail"]["status"] == "rate_limited"
    assert all(isinstance(p["detail"]["latency_ms"], int) for p in prov.values())
    assert ("search", "portal") in codes(ev)
    assert_clean(ev)


def test_inline_done_event_bounded_and_clean():
    cands = [{"portal": "openalex", "doi": "10.1000/x", "url": "https://example.invalid/p", "title": "Alpha",
              "outcome": "rejected", "reason": "licence", "status": "rejected"} for _ in range(30)]
    o = SimpleNamespace(admitted=2, stored=1, reused=0, candidates=cands, status="ok", codes={"licence": 3}, ms={"total": 5})
    ev = traced(lambda: inline_acquire._trace_done(o, {}))
    assert len([e for e in ev if e["code"] == "candidate"]) == 20
    assert next(e for e in ev if e["code"] == "done")["counts"]["candidates"] == 30
    assert_clean(ev)
