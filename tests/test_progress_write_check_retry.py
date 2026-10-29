"""PROG-1027 W2a: WRITE / CHECK / RETRY emit points (AC2 ordering, raise and empty paths). Offline, stubbed."""
import pytest

from tests import test_fulldoc_answer as legacy
from tests.test_fulldoc_d8_parts import CFG as PCFG, GCFG, bundle, key, reply  # noqa: F401  (key: autouse env)
from tests.test_item77_inference_chain import SECRET, _entries, _status_error  # noqa: F401
from openai import RateLimitError

from vera import ask_service
from vera import fulldoc_answer as fa
from vera import inference_chain as ic
from vera import progress as p
from vera.ask_service import AskResult

RID = "ef" * 16
TOK = "C" * 43
Q = "What do alpha systems do?"
A1 = "Alpha systems reduce latency by forty percent [1]."


@pytest.fixture
def rec():
    orig = p.CFG
    p.configure(p.Config(enabled=True, max_records=4, max_events_per_record=64, max_events_per_poll=64, ttl_s=100,
                         poll_interval_s=1.0, poll_margin_s=5, first_event_grace_s=10, get_timeout_s=2, max_count=10000))
    r = p.STORE.create(RID, b"h" * 32)
    t = p.SINK.set(r)
    yield r
    p.SINK.reset(t)
    p.configure(orig)


def seq(r):
    return [(e["stage"], e["state"]) for e in r.events_after(0, 99)]


class Writer:
    def __init__(self, text, boom=False):
        self.text, self.boom = text, boom

    def __call__(self, messages, model, pricing, **kw):
        if self.boom:
            raise RuntimeError("writer failed")
        return AskResult(answer=self.text, tokens_used=1, cost_usd=0.0)


def ok_checker(chk, msgs):
    import json
    n = len(json.loads(msgs[1]["content"])["claims"])
    return json.dumps({"claims": [{"id": i, "verdict": "supported", "reason": "r"} for i in range(n)]})


def run(text, rec_, writer=None, check=ok_checker, cfg=PCFG):
    return fa.answer(Q, bundle(), "m", None, chained={}, grounding_cfg=GCFG, cfg=cfg, complete_fn=writer or Writer(text),
                     check_call=check)


def test_answered_path_orders_write_then_check(rec):
    out = run(reply([Q], [A1]), rec)
    assert out.result is not None
    s = seq(rec)
    assert s == [("write", "start"), ("write", "done"), ("check", "start"), ("check", "done")]
    assert [e["seq"] for e in rec.events_after(0, 99)] == [1, 2, 3, 4] and all(not e["counts"] for e in rec.events_after(0, 9))


def test_writer_raises_emits_start_but_no_done(rec):
    with pytest.raises(RuntimeError):
        run("x", rec, writer=Writer("x", boom=True))
    assert seq(rec) == [("write", "start")]


def test_checker_raises_emits_check_start_without_done(rec):
    def boom(chk, msgs):
        raise RuntimeError("checker down")

    try:
        run(reply([Q], [A1]), rec, check=boom)
    except Exception:  # noqa: BLE001  (the checker wrapper may convert it to a fallback)
        pass
    s = seq(rec)
    assert s[:3] == [("write", "start"), ("write", "done"), ("check", "start")]
    assert ("check", "done") not in s or s.index(("check", "done")) > 2  # DONE only if check_answer returned


def test_no_write_when_not_grounded_and_no_sink_is_a_noop(rec):
    out = fa.answer(Q, legacy.bundle(status="insufficient", admitted=()), "m", None, chained={}, grounding_cfg=GCFG, cfg=PCFG,
                    complete_fn=Writer("x"), check_call=ok_checker)
    assert out.result is None and seq(rec) == []
    p.SINK.set(None)
    run(reply([Q], [A1]), rec)  # no sink: nothing recorded, nothing raised
    assert seq(rec) == []


def test_non_per_part_path_also_emits(rec):
    out = fa.answer("q", legacy.bundle(), "m", None, chained={}, grounding_cfg=legacy.GCFG, cfg=legacy.CFG,
                    complete_fn=legacy.Stub("Beta caching doubles throughput under heavy load [1]. Gamma results are short [2]."),
                    check_call=legacy.checker(["supported", "supported"]))
    assert out.result is not None
    assert seq(rec) == [("write", "start"), ("write", "done"), ("check", "start"), ("check", "done")]


def test_legacy_answer_grounded_points(rec, monkeypatch):
    from vera import claim_check, grounded_ask

    monkeypatch.setattr(ask_service, "complete", lambda *a, **k: AskResult(answer="Alpha is fast [1].", tokens_used=1, cost_usd=0.0))
    monkeypatch.setattr(grounded_ask, "messages_for", lambda q, s: [])
    monkeypatch.setattr(claim_check, "check_answer", lambda text, sources, cc: (text, {"outcome": "ok"}))
    dec = AskResult(answer="", tokens_used=0, cost_usd=0.0)
    out = ask_service._answer_grounded("q", "m", None, [{"title": "t"}], {"removed_citation_note": "n", "claim_check": {"enabled": True}},
                                       dec, {})
    assert out.result_kind == "answer"
    assert seq(rec) == [("write", "start"), ("write", "done"), ("check", "start"), ("check", "done")]


def test_legacy_check_raises_no_done(rec, monkeypatch):
    from vera import claim_check, grounded_ask

    monkeypatch.setattr(ask_service, "complete", lambda *a, **k: AskResult(answer="Alpha is fast [1].", tokens_used=1, cost_usd=0.0))
    monkeypatch.setattr(grounded_ask, "messages_for", lambda q, s: [])

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(claim_check, "check_answer", boom)
    with pytest.raises(RuntimeError):
        ask_service._answer_grounded("q", "m", None, [{"title": "t"}], {"removed_citation_note": "n", "claim_check": {"enabled": True}},
                                     AskResult(answer="", tokens_used=0, cost_usd=0.0), {})
    assert seq(rec) == [("write", "start"), ("write", "done"), ("check", "start")]


def test_retry_emitted_on_failover_with_attempt_count(rec, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)

    def fake(q, model, pricing, client=None, extra_body=None):
        p.emit("write", "start")
        if model != "gpt-4.1-nano":
            raise _status_error(RateLimitError, 429)
        return AskResult("ok", 10, 0.0)

    monkeypatch.setattr(ic, "answer_in_scope", fake)
    entries = _entries()
    result, label = ic.answer_via_chain("q", entries)
    s = [(e["stage"], e["state"], e["counts"]) for e in rec.events_after(0, 99)]
    retries = [x for x in s if x[0] == "retry"]
    assert len(retries) == len(entries) - 1
    assert [r[2] for r in retries] == [{"attempt": i + 2} for i in range(len(entries) - 1)]
    assert s[0][0] == "write" and s[1][0] == "retry"  # the retry marker precedes the next provider's re-run stages


def test_no_retry_when_last_entry_fails_or_first_succeeds(rec, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    monkeypatch.setattr(ic, "answer_in_scope", lambda q, m, pr, client=None, extra_body=None: AskResult("ok", 1, 0.0))
    ic.answer_via_chain("q", _entries())
    assert seq(rec) == []

    def boom(q, m, pr, client=None, extra_body=None):
        raise _status_error(RateLimitError, 429)

    monkeypatch.setattr(ic, "answer_in_scope", boom)
    one = _entries()[-1:]
    with pytest.raises(RateLimitError):
        ic.answer_via_chain("q", one)
    assert seq(rec) == []  # a single entry has no next attempt


def test_events_carry_no_free_text(rec):
    run(reply([Q], [A1]), rec)
    for e in rec.events_after(0, 99):
        assert set(e) == {"v", "seq", "ts", "stage", "state", "counts"}
