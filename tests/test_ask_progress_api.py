"""PROG-1027 W3a/W3b: POST /ask wrapper and GET /ask/progress/{rid}. Stubs only; no DB, no network, no env files."""
import hashlib
import logging
import secrets

import pytest
from fastapi.testclient import TestClient

import main
from vera import ask_service, progress
from vera import public_mode as pm

Q = "SENTINEL-QUERY-ZX81 how do caches reduce latency?"
KEY = "k" * 20
TEST_CFG = progress.Config(enabled=True, max_records=8, max_events_per_record=64, max_events_per_poll=64, ttl_s=600,
                           poll_interval_s=1.0, poll_margin_s=5, first_event_grace_s=10, get_timeout_s=2, max_count=10000)
OFF_CFG = progress.Config(**{**TEST_CFG.__dict__, "enabled": False})
NOT_FOUND = {"detail": "Not Found"}


def _rid():
    return secrets.token_hex(16)


def _tok():
    return secrets.token_urlsafe(32)


def _hdr(rid, tok):
    return {"X-VERA-Request-Id": rid, "X-VERA-Session": tok}


API = {"X-API-Key": KEY}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    original = progress.CFG
    progress.configure(TEST_CFG)
    monkeypatch.setenv("VERA_API_KEY", KEY)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    monkeypatch.delenv("VERA_PUBLIC_MODE", raising=False)
    monkeypatch.delenv(progress.KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv("VERA_AUTOACQ_DISABLED", "1")
    lim = pm.CapLimiter(pm.Caps(req_per_min_per_client=1000, req_per_day_per_client=1000, req_per_day_global=1000))
    pm.reset_limiter(lim)
    monkeypatch.setattr(main, "GROUNDING", None)
    monkeypatch.setattr(main, "store_from_env", lambda cfg: None)
    monkeypatch.setattr(main, "admit_to_corpus", lambda *a: None)
    monkeypatch.setattr(main, "llm_call_for", lambda e: None)
    monkeypatch.setattr(main, "memory_from_env", lambda block: None)
    _chain(monkeypatch, "answer")
    yield lim
    progress.configure(original)
    pm.reset_limiter(None)


def _result(kind="answer", completeness=None):
    cc = {"answer_completeness": completeness} if completeness else None
    return ask_service.AskResult(answer="A", tokens_used=0, cost_usd=0.0, result_kind=kind, claim_check=cc)


def _chain(monkeypatch, kind, completeness=None):
    res = _result(kind, completeness)
    monkeypatch.setattr(main, "answer_via_chain", lambda q, entries, g: (res, "groq:m"))


def _client():
    return TestClient(main.app, raise_server_exceptions=False)


def _ask(c, rid=None, tok=None, headers=True):
    h = dict(API)
    if headers:
        h.update(_hdr(rid, tok))
    return c.post("/ask", json={"question": Q}, headers=h)


def _poll(c, rid, tok, after=None, hdr=True, key=True):
    h = dict(API) if key else {}
    if hdr and tok is not None:
        h["X-VERA-Session"] = tok
    url = f"/ask/progress/{rid}" + ("" if after is None else f"?after={after}")
    return c.get(url, headers=h)


def _events(c, rid, tok):
    r = _poll(c, rid, tok)
    assert r.status_code == 200, r.text
    return r.json()


def _done(ev):
    return [e for e in ev["events"] if e["stage"] == "done"]


# ---- AC5: POST unchanged ------------------------------------------------------------------------------------
def _norm(r):
    return r.status_code, r.json()


def test_post_identical_with_headers_and_flag_on_off():
    c = _client()
    plain = _norm(_ask(c, headers=False))
    on = _norm(_ask(c, _rid(), _tok()))
    progress.configure(OFF_CFG)
    off_h = _norm(_ask(c, _rid(), _tok()))
    off_nh = _norm(_ask(c, headers=False))
    assert plain[0] == 200
    assert plain == on == off_h == off_nh


def test_post_without_headers_creates_no_record():
    c = _client()
    _ask(c, headers=False)
    assert len(progress.STORE) == 0


def test_received_seq_1_and_answered_done():
    c, rid, tok = _client(), _rid(), _tok()
    assert _ask(c, rid, tok).status_code == 200
    ev = _events(c, rid, tok)
    assert ev["closed"] is True
    assert (ev["events"][0]["seq"], ev["events"][0]["stage"], ev["events"][0]["state"]) == (1, "received", "done")
    d = _done(ev)
    assert len(d) == 1 and d[0]["state"] == "answered" and d[0]["counts"] == {"queued": 0}
    assert ev["events"][-1]["stage"] == "done"


@pytest.mark.parametrize("kind,comp,state,queued", [
    ("answer", None, "answered", True), ("answer", "complete", "answered", True), ("answer", "partial", "partial", True),
    ("decline", None, "declined", False), ("reading_list", None, "no_answer", True),
    ("unspecified", None, "no_answer", True), ("something_new", None, "no_answer", True),
])
def test_tmap_and_queued_only_on_allowed_terminals(monkeypatch, kind, comp, state, queued):
    _chain(monkeypatch, kind, comp)
    c, rid, tok = _client(), _rid(), _tok()
    assert _ask(c, rid, tok).status_code == 200
    d = _done(_events(c, rid, tok))
    assert len(d) == 1 and d[0]["state"] == state
    assert ("queued" in d[0]["counts"]) is queued


def test_outcome_unknown_kind_and_garbage():
    assert progress.outcome({"result_kind": "zzz"}) == progress.State.NO_ANSWER
    assert progress.outcome(object()) == progress.State.NO_ANSWER


# ---- AC8: access control ------------------------------------------------------------------------------------
def test_two_sessions_distinct_records_cross_read_404():
    c = _client()
    r1, t1, r2, t2 = _rid(), _tok(), _rid(), _tok()
    _ask(c, r1, t1)
    _ask(c, r2, t2)
    assert _events(c, r1, t1)["events"] and _events(c, r2, t2)["events"]
    unknown = _poll(c, _rid(), t1)
    for r in (_poll(c, r1, t2), _poll(c, r2, t1)):
        assert (r.status_code, r.content) == (unknown.status_code, unknown.content) == (404, unknown.content)


def test_all_refusals_identical_to_unmatched_route():
    c = _client()
    rid, tok = _rid(), _tok()
    _ask(c, rid, tok)
    unmatched = c.get("/no/such/route", headers=API)
    assert unmatched.status_code == 404 and unmatched.json() == NOT_FOUND
    unknown = _poll(c, _rid(), tok)
    bad = [
        _poll(c, _rid(), tok),                       # unknown rid
        _poll(c, "NOT-A-RID", tok),                  # malformed rid
        _poll(c, "0" * 31, tok),
        _poll(c, rid, tok, after=-1),
        _poll(c, rid, tok, after="abc"),
        _poll(c, rid, tok, after="1.5"),
        _poll(c, rid, tok, after=""),
        _poll(c, rid, tok, after="%EF%BC%91"),       # non-ASCII digit
        _poll(c, rid, tok, hdr=False),               # missing session header
        _poll(c, rid, "x" * 43),                     # bad token
        _poll(c, rid, ""),
    ]
    for r in bad:
        assert r.status_code == 404 and r.content == unknown.content == unmatched.content, r.text
    assert _poll(c, rid, tok, after=0).status_code == 200


def test_bad_api_key_still_401():
    c = _client()
    rid, tok = _rid(), _tok()
    _ask(c, rid, tok)
    assert _poll(c, rid, tok, key=False).status_code == 401


def test_flag_off_get_is_identical_404():
    c = _client()
    rid, tok = _rid(), _tok()
    _ask(c, rid, tok)
    unknown = _poll(c, _rid(), tok)
    progress.configure(OFF_CFG)
    r = _poll(c, rid, tok)
    assert (r.status_code, r.content) == (unknown.status_code, unknown.content)
    assert r.json() == NOT_FOUND


def test_after_filters_and_limit():
    c, rid, tok = _client(), _rid(), _tok()
    _ask(c, rid, tok)
    ev = _poll(c, rid, tok, after=1).json()
    assert [e["seq"] for e in ev["events"]] == [2]
    assert _poll(c, rid, tok, after=99).json()["events"] == []
    progress.configure(progress.Config(**{**TEST_CFG.__dict__, "max_events_per_poll": 2}))
    rid2, tok2 = _rid(), _tok()
    rec = progress.STORE.create(rid2, hashlib.sha256(tok2.encode()).digest())
    for _ in range(5):
        rec.append("write", "start", {})
    assert [e["seq"] for e in _poll(c, rid2, tok2).json()["events"]] == [1, 2]


def test_token_never_in_logs_or_responses(caplog):
    caplog.set_level(logging.DEBUG)
    c, rid, tok = _client(), _rid(), _tok()
    r1 = _ask(c, rid, tok)
    r2 = _poll(c, rid, tok)
    r3 = _poll(c, rid, tok + "x")
    r4 = _poll(c, rid, tok, after="bad")
    blob = " ".join(x.text for x in (r1, r2, r3, r4)) + caplog.text
    assert tok not in blob and tok + "x" not in blob
    assert progress.STORE._records[rid].token_hash == hashlib.sha256(tok.encode()).digest()


# ---- AC7: caps ----------------------------------------------------------------------------------------------
def test_cap_counter_once_per_ask_every_path(env, monkeypatch):
    lim = env
    c = _client()
    n = 0
    for mode in ("headers", "noheaders", "flagoff", "mismatch"):
        if mode == "flagoff":
            progress.configure(OFF_CFG)
        if mode == "headers" or mode == "flagoff":
            _ask(c, _rid(), _tok())
        elif mode == "noheaders":
            _ask(c, headers=False)
        else:
            progress.configure(TEST_CFG)
            c.post("/ask", json={"question": Q}, headers={**API, "X-VERA-Request-Id": "bad", "X-VERA-Session": "bad"})
        n += 1
        assert lim._global_req == n
    # error path
    def boom(q, e, g):
        raise RuntimeError("x")
    monkeypatch.setattr(main, "answer_via_chain", boom)
    _ask(c, _rid(), _tok())
    assert lim._global_req == n + 1


def test_polls_do_not_touch_caps(env):
    lim = env
    c, rid, tok = _client(), _rid(), _tok()
    _ask(c, rid, tok)
    before = (lim._global_req, dict(lim._day_req), {k: list(v) for k, v in lim._minute.items()})
    for _ in range(200):
        assert _poll(c, rid, tok).status_code == 200
    assert (lim._global_req, dict(lim._day_req), {k: list(v) for k, v in lim._minute.items()}) == before


# ---- AC9: errors --------------------------------------------------------------------------------------------
def test_raising_stub_gives_done_error_and_no_query_in_logs(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)

    def boom(q, e, g):
        raise RuntimeError("provider exploded")
    monkeypatch.setattr(main, "answer_via_chain", boom)
    c, rid, tok = _client(), _rid(), _tok()
    r = _ask(c, rid, tok)
    assert r.status_code == 500
    ev = _events(c, rid, tok)
    d = _done(ev)
    assert ev["closed"] and len(d) == 1 and d[0]["state"] == "error" and d[0]["counts"] == {}
    assert "SENTINEL-QUERY-ZX81" not in caplog.text


@pytest.mark.parametrize("status", [502, 503])
def test_http_exception_after_received_gives_done_error_same_status(monkeypatch, status):
    from fastapi import HTTPException

    def boom(q, e, g):
        raise HTTPException(status_code=status, detail="d")
    monkeypatch.setattr(main, "answer_via_chain", boom)
    c, rid, tok = _client(), _rid(), _tok()
    assert _ask(c, rid, tok).status_code == status
    ev = _events(c, rid, tok)
    assert ev["events"][0]["stage"] == "received"
    assert [e["state"] for e in _done(ev)] == ["error"]


def test_open_record_failure_does_not_break_ask(monkeypatch):
    def bad(self, rid, h):
        raise RuntimeError("store down")
    monkeypatch.setattr(progress.Store, "create", bad)
    c = _client()
    assert _ask(c, _rid(), _tok()).status_code == 200


# ---- AC10: store full ---------------------------------------------------------------------------------------
def test_store_full_of_open_records_post_200_get_404():
    progress.configure(progress.Config(**{**TEST_CFG.__dict__, "max_records": 2}))
    for _ in range(2):
        assert progress.STORE.create(_rid(), b"h" * 32) is not None  # open records, never closed
    c, rid, tok = _client(), _rid(), _tok()
    assert _ask(c, rid, tok).status_code == 200
    assert _poll(c, rid, tok).status_code == 404


def test_single_terminal_and_closed_record_drops_late_events():
    c, rid, tok = _client(), _rid(), _tok()
    _ask(c, rid, tok)
    rec = progress.STORE._records[rid]
    tk = progress.SINK.set(rec)
    try:
        progress.emit(progress.Stage.WRITE, progress.State.START)
        progress.emit(progress.Stage.DONE, progress.State.ERROR)
    finally:
        progress.SINK.reset(tk)
    assert len(_done(_events(c, rid, tok))) == 1
    assert progress.SINK.get() is None


def test_no_db_store_factory_called_with_real_url(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "store_from_env", lambda cfg: calls.append(cfg))
    monkeypatch.setattr(main, "memory_from_env", lambda block: calls.append(block))
    c, rid, tok = _client(), _rid(), _tok()
    _ask(c, rid, tok)
    for _ in range(3):
        _poll(c, rid, tok)
    assert not any(isinstance(a, str) and "://" in a for a in calls)
