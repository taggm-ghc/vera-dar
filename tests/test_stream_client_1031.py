"""OD-1031-STREAM-ON T9: stream client (offline, fake session; no network)."""
import json

import pytest
import requests

from vera import public_mode as pm

HOST = "https://sentinel-host.invalid"
KEY = "sentinel-key-123"


def ev(seq, stage="search", state="start", counts=None):
    return {"v": "1.3", "type": "event", "seq": seq, "ts": "t", "stage": stage, "state": state, "counts": counts or {}}


HB = {"v": "1.3", "type": "hb", "seq": 2, "elapsed_s": 10}
RES = {"v": "1.3", "type": "result", "status": 200, "body": {"answer": "x"}}


def enc(o):
    return json.dumps(o).encode() if not isinstance(o, (bytes, str)) else (o if isinstance(o, bytes) else o.encode())


class Resp:
    def __init__(self, lines, status=200, then=None):
        self.status_code, self._lines, self._then, self.closed = status, lines, then, False

    def iter_lines(self):
        for ln in self._lines:
            yield enc(ln)
        if self._then:
            raise self._then

    def close(self):
        self.closed = True


class Sess:
    def __init__(self, resp=None, raises=None):
        self.resp, self.raises, self.calls = resp, raises, []

    def post(self, url, **kw):
        self.calls.append((url, kw))
        if self.raises:
            raise self.raises
        return self.resp


def run(sess, **kw):
    it = pm.iter_ask_stream(HOST, "q", "normal", KEY, connect_timeout=10, no_data_timeout=60, session=sess, **kw)
    return it, list(it)


def test_valid_lines_and_timeouts_tuple_no_total():
    s = Sess(Resp([ev(1, "acquire", "limited", {"stored": 1, "target": 6, "why": 3}), HB, RES]))
    it, out = run(s)
    assert [o["type"] for o in out] == ["event", "hb", "result"] and it.dropped == 0
    url, kw = s.calls[0]
    assert url.endswith("/ask/stream") and kw["stream"] is True and kw["timeout"] == (10, 60)  # connect, no-data only
    assert kw["headers"]["X-API-Key"] == KEY


def test_invalid_lines_dropped_and_counted():
    bad = [b"not json", b"[1]", ev(1, "nostage", "start"), ev(2, "search", "bogus"), ev(3, "search", "start", {"url": 1}),
           ev(4, "search", "start", {"found": "x"}), {**ev(5), "v": "1"}, {"v": "1.3", "type": "result", "status": 418},
           {"v": "1.3", "type": "result", "status": 500, "kind": "boom"}, b"\xff\xfe", ev(6, "acquire", "busy")]
    it, out = run(Sess(Resp(bad + [RES])))
    assert [o["type"] for o in out] == ["event", "result"] and it.dropped == len(bad) - 1


def test_second_result_dropped():
    it, out = run(Sess(Resp([RES, RES])))
    assert len(out) == 1


def test_error_results_valid():
    for st, kind in ((429, "busy"), (500, "error"), (502, "provider_error"), (503, "unavailable")):
        _, out = run(Sess(Resp([{"v": "1.3", "type": "result", "status": st, "kind": kind}])))
        assert out[0]["kind"] == kind


def test_no_data_timeout_is_stream_lost():
    s = Sess(Resp([ev(1)], then=requests.exceptions.ReadTimeout("read timed out " + HOST + KEY)))
    it = pm.iter_ask_stream(HOST, "q", "m", KEY, connect_timeout=1, no_data_timeout=1, session=s)
    assert next(it)["type"] == "event"
    with pytest.raises(pm.StreamLost) as ei:
        next(it)
    assert ei.value.reason == "no_data" and HOST not in repr(ei.value) and KEY not in str(ei.value)
    assert s.resp.closed


def test_mid_stream_connection_error_is_lost_and_ended_is_lost():
    s = Sess(Resp([ev(1)], then=requests.exceptions.ChunkedEncodingError(HOST)))
    with pytest.raises(pm.StreamLost) as ei:
        run(s)
    assert ei.value.reason == "connection" and HOST not in str(ei.value)
    with pytest.raises(pm.StreamLost) as ei:
        run(Sess(Resp([ev(1)])))  # stream ended without a result
    assert ei.value.reason == "ended"


def test_never_connected_cases():
    with pytest.raises(pm.StreamNeverConnected) as ei:
        run(Sess(raises=requests.exceptions.ConnectTimeout(HOST + KEY)))
    assert HOST not in str(ei.value) and ei.value.__cause__ is None and ei.value.status is None
    with pytest.raises(pm.StreamNeverConnected) as ei:
        run(Sess(Resp([], status=429)))
    assert ei.value.status == 429
    with pytest.raises(pm.StreamNeverConnected):
        run(Sess(Resp([], status=500)))
    with pytest.raises(pm.StreamNeverConnected):
        run(Sess(Resp([], then=requests.exceptions.ConnectionError(HOST))))


def test_should_stop_closes_stream():
    r = Resp([ev(1), ev(2), RES])
    stop = {"v": False}
    it = pm.iter_ask_stream(HOST, "q", "m", KEY, connect_timeout=1, no_data_timeout=1, session=Sess(r),
                            should_stop=lambda: stop["v"])
    assert next(it)["seq"] == 1
    stop["v"] = True
    assert list(it) == [] and it.stopped and r.closed


def test_legacy_clients_unchanged():
    import inspect
    # SELFQ-RELEVANCE-1108: one optional per-request spell_check (default on; the body is unchanged when on)
    assert list(inspect.signature(pm.post_ask).parameters) == ["base_url", "question", "mode", "api_key", "timeout",
                                                              "spell_check"]
    assert "rid" in inspect.signature(pm.post_ask_tracked).parameters
