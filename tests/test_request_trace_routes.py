"""PERF-MISS-1104-TRACE (OD-1104-TRACE): X-VERA-Trace-Id header on /ask and /ask/stream, never in the body; owner-key
route /trace/request/{id}; a failing trace sink never fails a request. Offline: _ask_impl stubbed, no DB, no network."""
import json
import logging

from fastapi import HTTPException

import main
from vera import progress, request_trace as rt
from tests.test_stream_route_support import API, CFG, KEY, Q, client, lines, post, stub_impl, stream_env  # noqa: F401

OWNER = "o" * 40
BASE = json.loads(rt.CONFIG_PATH.read_text(encoding="utf-8"))


def _resp():
    return main.AskResponse(answer="A", model="m", tokens_used=3, cost_usd=0.0, result_kind="answer",
                            outcome="answer", answer_status="complete")


def _impl(req, request, bg):
    progress.emit(progress.Stage.SEARCH, progress.State.START, step=1)
    rt.event("premise", "verdict", detail={"verdict": "ok"})
    return _resp()


def _trace_lines(caplog):
    return [json.loads(r.getMessage()[len(rt.LOG_PREFIX) + 1:]) for r in caplog.records
            if r.name == "vera.trace" and r.getMessage().startswith(rt.LOG_PREFIX + " ")]


def test_ask_returns_trace_header_not_in_body(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="vera.trace")
    rt.configure(BASE)
    stub_impl(monkeypatch, _impl)
    r = client().post("/ask", json={"question": Q}, headers={**API, "x-vera-request-id": "a" * 32})
    assert r.status_code == 200
    tid = r.headers.get(rt.HEADER)
    assert tid and rt.TRACE_ID_RE.match(tid) and tid != "a" * 32
    assert tid not in r.text
    ls = _trace_lines(caplog)
    assert [x["stage"] for x in ls] == ["start", "search", "premise", "outcome", "finish"]
    assert ls[0]["question_chars"] == len(Q) and ls[0]["client_request_id"] == "a" * 32
    assert Q not in caplog.text  # default include_question=false (R-028)
    assert ls[-1]["code"] == "answer" and ls[-1]["detail"] == {"message_code": "complete"}
    assert all(x["request_id"] == tid for x in ls)


def test_ask_error_carries_header_and_error_terminal(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="vera.trace")
    rt.configure(BASE)

    def boom(req, request, bg):
        raise HTTPException(status_code=503, detail="No answering provider is configured.")
    stub_impl(monkeypatch, boom)
    r = client().post("/ask", json={"question": Q}, headers=API)
    assert r.status_code == 503 and rt.TRACE_ID_RE.match(r.headers.get(rt.HEADER, ""))
    assert _trace_lines(caplog)[-1]["stage"] == "finish" and _trace_lines(caplog)[-1]["code"] == "error"


def test_unhandled_error_still_500_with_header(monkeypatch):
    rt.configure(BASE)
    stub_impl(monkeypatch, lambda *a: (_ for _ in ()).throw(ValueError("x")))
    r = client().post("/ask", json={"question": Q}, headers=API)
    assert r.status_code == 500 and rt.TRACE_ID_RE.match(r.headers.get(rt.HEADER, ""))


def test_stream_returns_trace_header_and_worker_events(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="vera.trace")
    rt.configure(BASE)
    stub_impl(monkeypatch, _impl)
    r = post(client())
    assert r.status_code == 200
    tid = r.headers.get(rt.HEADER)
    assert tid and rt.TRACE_ID_RE.match(tid) and tid not in r.text
    assert lines(r)[-1]["type"] == "result"
    ls = [x for x in _trace_lines(caplog) if x["request_id"] == tid]
    # the stream worker always emits the progress `done` terminal, which the trace also records
    assert [x["stage"] for x in ls] == ["start", "search", "premise", "done", "outcome", "finish"]
    assert ls[0]["endpoint"] == "/ask/stream"


def test_tracing_off_no_header(monkeypatch):
    rt.configure({**BASE, "enabled": False})
    stub_impl(monkeypatch, _impl)
    r = client().post("/ask", json={"question": Q}, headers=API)
    assert r.status_code == 200 and rt.HEADER.lower() not in {k.lower() for k in r.headers}
    rt.configure(BASE)


def test_broken_trace_module_never_fails_request(monkeypatch):
    """main.py's entry-point wrappers and the progress hook absorb a raising trace module; the public event() API
    absorbs its own internal failures (_emit / log sink)."""
    rt.configure(BASE)

    def impl(req, request, bg):
        progress.emit(progress.Stage.SEARCH, progress.State.START, step=1)  # -> on_progress (raising)
        rt.event("premise", "verdict")  # real event() with a raising internal _emit
        return _resp()
    stub_impl(monkeypatch, impl)
    boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("trace down"))  # noqa: E731
    for name in ("start", "finish", "on_progress", "_emit", "_log_line"):
        monkeypatch.setattr(rt, name, boom)
    r = client().post("/ask", json={"question": Q}, headers=API)
    assert r.status_code == 200 and r.json()["answer"] == "A"
    r = post(client())
    assert r.status_code == 200 and lines(r)[-1]["status"] == 200


def test_in_process_call_without_response_object_still_works(monkeypatch):
    rt.configure(BASE)
    stub_impl(monkeypatch, _impl)
    from types import SimpleNamespace
    from fastapi import BackgroundTasks
    out = main.ask(main.AskRequest(question=Q), SimpleNamespace(state=SimpleNamespace()), BackgroundTasks())
    assert out.answer == "A"


def test_owner_route(monkeypatch):
    rt.configure(BASE)
    monkeypatch.setenv("VERA_OWNER_KEY", OWNER)
    stub_impl(monkeypatch, _impl)
    c = client()
    tid = c.post("/ask", json={"question": Q}, headers=API).headers[rt.HEADER]
    assert c.get(f"/trace/request/{tid}").status_code == 401
    assert c.get(f"/trace/request/{tid}", headers={"X-Owner-Key": "w" * 40}).status_code == 401
    ok = c.get(f"/trace/request/{tid}", headers={"X-Owner-Key": OWNER})
    assert ok.status_code == 200 and ok.json()["question"] == Q and ok.json()["outcome_branch"] == "answer"
    assert [e["stage"] for e in ok.json()["events"]][-1] == "finish"
    assert c.get("/trace/request/" + "0" * 32, headers={"X-Owner-Key": OWNER}).status_code == 404
