"""PROG-1027 W4b-2: pure progress_ui state machine, classify, Labels, run_worker (AC4, AC6, AC9)."""
import logging

import pytest
import requests

from vera import progress
from vera import progress_ui as pu
from vera.public_mode import ASK_TIMEOUT_S, GONE

LABELS = pu.Labels.load()
CFG = progress.validate_config({"enabled": True, "max_records": 8, "max_events_per_record": 8, "max_events_per_poll": 8,
                                "ttl_s": 600, "poll_interval_s": 1.0, "poll_margin_s": 5, "first_event_grace_s": 10,
                                "get_timeout_s": 2, "max_count": 10000})
RID = "ab12cd34" + "0" * 24
HOST = "https://sentinel-host.invalid"
TOKEN = "SENTINELTOKEN" + "y" * 30


def ev(seq, stage, state, counts=None):
    return {"v": "1", "seq": seq, "ts": "t", "stage": stage, "state": state, "counts": counts or {}}


def inflight(**hk):
    h = pu.Holder()
    for k, v in hk.items():
        setattr(h, k, v)
    return pu.Inflight(rid=RID, holder=h, t0=100.0)


def hdone(**hk):
    inf = inflight(**hk)
    inf.holder.done.set()
    return inf


# ---------------------------------------------------------------------------------------------- Labels
def test_labels_cover_every_terminal_and_stage_state():
    for k in pu.TERMINAL_KINDS:
        assert LABELS.terminal_text(k, RID)
    for st, states in progress.VALID.items():
        for sa in states:
            assert LABELS.stage_label(st.value, sa.value, {"inspected": 1, "admitted": 2, "found": 3, "checked": 4, "usable": 5,
                                                           "budget": 2, "stored": 1, "target": 6, "attempt": 2}, RID)


def test_label_missing_counts_never_invented():
    assert LABELS.stage_label("corpus", "update", {}, RID) == "Searching VERA-DAR's collection"
    assert LABELS.stage_label("acquire", "update", {}, RID) is None  # no start fallback with counts either
    assert LABELS.stage_label("nope", "x", {}, RID) is None
    assert "3 documents inspected" in LABELS.stage_label("corpus", "update", {"inspected": 3, "admitted": 1}, RID)


def test_labels_reject_bad_shape():
    import json
    import pathlib
    raw = json.loads(pathlib.Path(pu.LABELS_PATH).read_text())
    for mut in (lambda r: r.pop("sending"), lambda r: r["terminals"].pop("busy"), lambda r: r["stages"].update(bogus={"x": "y"}),
                lambda r: r["terminals"].update(busy="{0.__class__}"), lambda r: r["stages"]["scope"].update(answered="x"),
                lambda r: r["terminals"].update(busy="{secret}")):
        r = json.loads(json.dumps(raw))
        mut(r)
        with pytest.raises(ValueError):
            pu.Labels(r)


def test_terminal_text_variants_have_rid():
    assert RID in LABELS.terminal_text("poll_timeout_received", RID) and RID in LABELS.terminal_text("poll_timeout_unreceived", RID)
    assert RID in LABELS.sending(RID)


# ---------------------------------------------------------------------------------------------- step
def test_event_less_ticks_keep_sending_then_unavailable_never_terminal():
    inf = inflight()
    for i in range(30):
        label, term = step_(inf, [], 101 + i)
        assert term is None and label == LABELS.sending(RID)
    assert inf.seq == 0


def step_(inf, events, now, final=False):
    return pu.step(inf, events, now, CFG, LABELS, final=final)


def test_404_within_grace_keeps_sending_after_grace_unavailable_and_keeps_waiting():
    inf = inflight()
    assert step_(inf, GONE, 105) == (LABELS.sending(RID), None)
    assert step_(inf, None, 109.9) == (LABELS.sending(RID), None)
    label, term = step_(inf, GONE, 111)
    assert label == LABELS.progress_unavailable and term is None


def test_unavailable_after_events_then_recovers():
    inf = inflight()
    step_(inf, [ev(1, "received", "done")], 101)
    assert step_(inf, GONE, 102)[0] == LABELS.progress_unavailable
    assert step_(inf, [ev(2, "scope", "start")], 103)[0] == "Checking what the question asks"


def test_received_label_only_after_event():
    inf = inflight()
    assert not inf.received_seen and "received" not in step_(inf, [], 101)[0].lower()
    label, _ = step_(inf, [ev(1, "received", "done")], 102)
    assert inf.received_seen and label == f"Question received (request {RID})"


def test_several_stages_in_one_tick_show_last():
    inf = inflight()
    label, _ = step_(inf, [ev(1, "received", "done"), ev(2, "scope", "start"), ev(3, "write", "start")], 101)
    assert label == "Writing the answer from the usable passages" and inf.seq == 3


def test_unknown_stage_state_ignored_and_seq_deduped():
    inf = inflight()
    step_(inf, [ev(1, "scope", "start")], 101)
    before = inf.label
    assert step_(inf, [ev(2, "zzz", "start"), ev(2, "scope", "bogus")], 102)[0] == before
    assert step_(inf, [ev(1, "write", "start")], 103)[0] == before  # replayed seq ignored


def test_interruption_between_any_two_steps_resumes_same_state():
    seq = [[ev(1, "received", "done")], [], [ev(2, "scope", "start"), ev(3, "scope", "done")], None, [ev(4, "write", "start")]]
    ref = inflight()
    trace = [step_(ref, e, 101 + i) for i, e in enumerate(seq)]
    for cut in range(len(seq)):
        inf = inflight()
        for i, e in enumerate(seq[:cut]):
            step_(inf, e, 101 + i)
        # "interruption": nothing but the Inflight survives; rid, seq and label persist
        rid, s, lab = inf.rid, inf.seq, inf.label
        label, _ = step_(inf, [], 101 + cut)  # a tick that carries nothing
        assert (inf.rid, inf.seq) == (rid, s) and (label == lab or not lab)
        for i, e in enumerate(seq[cut:], start=cut):
            out = step_(inf, e, 101 + i)
        assert out == trace[-1] and inf.rid == RID


def test_server_done_with_holder_not_done_no_terminal_then_final_one_terminal():
    inf = inflight()
    label, term = step_(inf, [ev(1, "received", "done"), ev(2, "done", "answered", {"queued": 2})], 101)
    assert term is None and label == LABELS.finishing and inf.server_done == "answered" and inf.queued == 2
    assert step_(inf, [], 102) == (LABELS.finishing, None)
    inf.holder.status, inf.holder.result = 200, {"result_kind": "answer", "claim_check": {"answer_completeness": "complete"}}
    inf.holder.done.set()
    assert step_(inf, [], 103, final=True) == (LABELS.finishing, "answered")


def test_final_without_events_uses_holder_result():
    inf = hdone(status=200, result={"result_kind": "answer", "claim_check": {"answer_completeness": "partial"}})
    assert step_(inf, None, 101, final=True)[1] == "partial"


def test_backstop_poll_timeout_variants_without_final():
    inf = inflight()
    assert step_(inf, [], 100 + ASK_TIMEOUT_S + 4.9)[1] is None
    assert step_(inf, [], 100 + ASK_TIMEOUT_S + 5)[1] == "poll_timeout_unreceived"
    inf2 = inflight()
    step_(inf2, [ev(1, "received", "done")], 101)
    assert step_(inf2, [], 100 + ASK_TIMEOUT_S + 5)[1] == "poll_timeout_received"


def test_done_error_event_plus_429_busy_only():
    inf = hdone(status=429)
    step_(inf, [ev(1, "received", "done"), ev(2, "done", "error")], 101)
    assert step_(inf, [], 102, final=True)[1] == "busy"


def test_final_get_404_drives_expired():
    inf = hdone(status=502)
    step_(inf, [ev(1, "received", "done")], 101)
    assert step_(inf, GONE, 102, final=True)[1] == "expired"


# ---------------------------------------------------------------------------------------------- classify
def mk(received=False, server_done=None, **hk):
    inf = inflight(**hk)
    inf.received_seen, inf.server_done = received, server_done
    return inf


def test_classify_t_map():
    def res(kind, comp=None):
        return {"result_kind": kind, "claim_check": {"answer_completeness": comp}}
    cases = [(res("answer", "complete"), "answered"), (res("answer", "partial"), "partial"), (res("decline"), "declined"),
             (res("reading_list"), "no_answer"), (res("unspecified"), "no_answer"), (res("???"), "no_answer"),
             ({"result_kind": "answer", "claim_check": None}, "answered"), ({}, "no_answer")]
    for r, want in cases:
        assert pu.classify(mk(status=200, result=r), False) == want


def test_classify_200_non_dict_and_bad_body_error():
    for r in ([], "x", None, 5):
        assert pu.classify(mk(status=200, result=r), False) == "error"
    assert pu.classify(mk(status=200, error_type="JSONDecodeError", error_kind="other"), False) == "error"


def test_classify_429_busy_ignores_server_done_error():
    assert pu.classify(mk(True, "error", status=429), False) == "busy"
    assert pu.classify(mk(False, None, status=429), True) == "busy"


@pytest.mark.parametrize("code", [502, 503])
def test_classify_provider_error(code):
    assert pu.classify(mk(True, "error", status=code), False) == "provider_error"
    assert pu.classify(mk(True, "error", status=code), True) == "provider_error"


def test_classify_expired_and_error_variants():
    assert pu.classify(mk(True, None, status=502), True) == "expired"
    assert pu.classify(mk(True, None, status=503), True) == "expired"
    assert pu.classify(mk(True, None, status=504), True) == "expired"
    assert pu.classify(mk(True, None, status=502), False) == "error"  # final GET 200/None
    assert pu.classify(mk(False, None, status=502), True) == "error"  # RECEIVED not seen
    assert pu.classify(mk(True, "answered", status=502), True) == "error"
    assert pu.classify(mk(True, None, status=500), True) == "error"
    assert pu.classify(mk(True, None, status=418), True) == "error"
    for kind in ("connection_error", "chunked"):
        assert pu.classify(mk(True, None, error_kind=kind), True) == "expired"
        assert pu.classify(mk(True, None, error_kind=kind), False) == "error"
        assert pu.classify(mk(False, None, error_kind=kind), True) == "error"
    assert pu.classify(mk(True, None, error_kind="other", error_type="OSError"), True) == "error"


def real_kind(exc):
    h = pu.Holder()

    def boom(*a):
        raise exc
    pu.run_worker(h, boom, ())
    return h


def test_classify_with_real_requests_exceptions():
    ct = real_kind(requests.exceptions.ConnectTimeout("x"))
    assert ct.error_kind == "connect_timeout" and ct.error_type == "ConnectTimeout"
    assert pu.classify(pu.Inflight(RID, ct, 0.0, received_seen=True), True) == "error"  # before Timeout / expired
    rt = real_kind(requests.exceptions.ReadTimeout("x"))
    assert rt.error_kind == "read_timeout"
    assert pu.classify(pu.Inflight(RID, rt, 0.0, received_seen=True), False) == "poll_timeout_received"
    assert pu.classify(pu.Inflight(RID, rt, 0.0, received_seen=False), False) == "poll_timeout_unreceived"
    ce = real_kind(requests.exceptions.ConnectionError("x"))
    assert ce.error_kind == "connection_error"
    assert pu.classify(pu.Inflight(RID, ce, 0.0), True) == "error"  # ConnectionError without RECEIVED
    assert pu.classify(pu.Inflight(RID, ce, 0.0, received_seen=True), True) == "expired"
    assert real_kind(requests.exceptions.ChunkedEncodingError("x")).error_kind == "chunked"
    assert real_kind(requests.exceptions.SSLError("x")).error_kind == "connection_error"  # SSLError is a ConnectionError
    assert real_kind(ValueError("x")).error_kind == "other"


def test_classify_safe_never_raises():
    class Bad:
        @property
        def holder(self):
            raise RuntimeError("x")
    assert pu.classify_safe(Bad()) == "error"
    assert pu.classify_safe(None) == "error"
    inf = mk(status=200, result={"result_kind": "answer", "claim_check": {"answer_completeness": "partial"}})
    assert pu.classify_safe(inf) == "partial"


# ---------------------------------------------------------------------------------------------- run_worker
class Fake:
    def __init__(self, status, body=None, bad_json=False):
        self.status_code, self._b, self._bad = status, body, bad_json

    def json(self):
        if self._bad:
            raise ValueError("body " + HOST)
        return self._b


def test_worker_ok_non200_and_bad_body():
    h = pu.Holder()
    pu.run_worker(h, lambda *a: Fake(200, {"a": 1}), ("x",))
    assert (h.status, h.result, h.error_type, h.done.is_set()) == (200, {"a": 1}, None, True)
    h = pu.Holder()
    pu.run_worker(h, lambda *a: Fake(429), ())
    assert (h.status, h.result, h.error_type, h.done.is_set()) == (429, None, None, True)
    h = pu.Holder()
    pu.run_worker(h, lambda *a: Fake(200, bad_json=True), ())
    assert h.status == 200 and h.result is None and h.error_type == "ValueError" and h.done.is_set()


class Stop(BaseException):
    pass


@pytest.mark.parametrize("exc,name", [(RuntimeError(HOST + TOKEN), "RuntimeError"), (Stop(HOST), "Stop"),
                                      (SystemExit(HOST), "SystemExit"), (KeyboardInterrupt(), "KeyboardInterrupt")])
def test_worker_raising_client_type_name_done_no_leak(exc, name, caplog):
    h = pu.Holder()

    def boom(*a):
        raise exc
    with caplog.at_level(logging.DEBUG):
        pu.run_worker(h, boom, (HOST, TOKEN))
    assert h.error_type == name and h.done.is_set() and h.status is None and h.result is None
    assert HOST not in caplog.text and TOKEN not in caplog.text
    assert HOST not in repr(h) and TOKEN not in repr(h)
    assert pu.classify(pu.Inflight(RID, h, 0.0), False) == "error"


def test_worker_passes_args_to_client():
    got = []
    h = pu.Holder()
    pu.run_worker(h, lambda *a: got.append(a) or Fake(200, {}), (1, 2, 3))
    assert got == [(1, 2, 3)]
