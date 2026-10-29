"""PROG-1027 W1a: closed emitter, record and store (vera/progress.py). Offline; no DB, no network."""
import logging
import threading
from hashlib import sha256
from types import SimpleNamespace

import pytest

from vera import progress as p

RID = "ab" * 16
RID2 = "cd" * 16
TOK = "A" * 43
TOK2 = "B" * 43
SENTINEL = "sentinel-title-https://sentinel.invalid"


def cfg(**kw):
    base = dict(enabled=True, max_records=4, max_events_per_record=6, max_events_per_poll=6, ttl_s=100, poll_interval_s=1.0,
                poll_margin_s=5, first_event_grace_s=10, get_timeout_s=2, max_count=10000)
    base.update(kw)
    return p.Config(**base)


@pytest.fixture(autouse=True)
def fresh():
    orig = p.CFG
    p.configure(cfg())
    p.DROPS.clear()
    yield
    p.configure(orig)


def hdrs(rid=RID, tok=TOK):
    return SimpleNamespace(headers={"x-vera-request-id": rid, "x-vera-session": tok})


def sink(rid=RID, tok=TOK):
    rec = p.open_record(hdrs(rid, tok))
    assert rec is not None
    return rec, p.SINK.set(rec)


def test_emit_is_a_noop_without_sink():
    p.emit("write", "start")
    p.set_queued(3)
    assert p.DROPS == {}


def test_enum_and_valid_pairs_enforced():
    rec, tok = sink()
    p.emit("write", "start")
    p.emit("write", "answered")  # not valid for write
    p.emit("nope", "start")  # unknown stage
    p.emit("retry", "done")  # retry allows start only
    p.emit("received", "start")  # received allows done only
    p.emit("corpus_verdict", "start")
    p.emit("corpus_verdict", "stale")
    p.SINK.reset(tok)
    assert [(e["stage"], e["state"]) for e in rec.events_after(0, 99)] == [("write", "start"), ("corpus_verdict", "stale")]
    assert p.DROPS["state"] == 4 and p.DROPS["exception"] == 1


def test_event_shape_seq_and_ts():
    rec, tok = sink()
    p.emit("received", "done")
    p.emit("corpus", "done", inspected=3, admitted=2)
    p.SINK.reset(tok)
    evs = rec.events_after(0, 99)
    assert [e["seq"] for e in evs] == [1, 2]
    assert set(evs[0]) == {"v", "seq", "ts", "stage", "state", "counts"} and evs[0]["v"] == "1"
    assert evs[0]["ts"].endswith("+00:00") and evs[1]["counts"] == {"inspected": 3, "admitted": 2}
    assert [e["seq"] for e in rec.events_after(1, 99)] == [2]


@pytest.mark.parametrize("bad", [{"round": 1}, {"candidates": 2}, {"found": True}, {"found": -1}, {"found": 10001},
                                 {"found": 1.5}, {"found": SENTINEL}, {"title": SENTINEL}, {"reason": 1}])
def test_non_contract_counts_dropped(bad):
    rec, tok = sink()
    p.emit("search", "done", **bad)
    p.SINK.reset(tok)
    (e,) = rec.events_after(0, 9)
    assert e["counts"] == {} and p.DROPS["counts"] == 1
    assert SENTINEL not in repr(e)


def test_max_count_boundary_and_mixed_counts():
    rec, tok = sink()
    p.emit("search", "done", found=10000, round=1)
    p.SINK.reset(tok)
    assert rec.events_after(0, 9)[0]["counts"] == {"found": 10000}


def test_closed_record_rejects_appends_and_second_done_rejected():
    rec, tok = sink()
    p.emit("done", "answered")
    p.emit("done", "error")  # second done
    p.emit("write", "start")  # after done is allowed until close
    rec.close()
    p.emit("write", "done")
    p.SINK.reset(tok)
    assert [e["state"] for e in rec.events_after(0, 9)] == ["answered", "start"]
    assert p.DROPS["second_done"] == 1 and p.DROPS["closed"] == 1 and rec.closed


def test_terminal_always_kept_when_full():
    rec, tok = sink()
    for _ in range(10):
        p.emit("write", "update")
    p.emit("done", "answered")
    p.SINK.reset(tok)
    evs = rec.events_after(0, 99)
    assert len(evs) == 7 and evs[-1]["stage"] == "done" and p.DROPS["full"] == 4


def test_set_queued_bounds_and_terminal_counts():
    rec, tok = sink()
    p.set_queued(True)
    p.set_queued(-1)
    p.set_queued(10001)
    assert rec.queued == 0
    p.set_queued(1)
    p.SINK.reset(tok)
    assert rec.queued == 1
    assert p.terminal_counts(p.State.ANSWERED, rec) == {"queued": 1}
    assert p.terminal_counts(p.State.NO_ANSWER, rec) == {"queued": 1}
    assert p.terminal_counts(p.State.DECLINED, rec) == {} and p.terminal_counts(p.State.ERROR, rec) == {}


def test_store_full_of_open_records_refuses_then_closed_evicted_oldest_first():
    recs = [p.open_record(hdrs(f"{i:032x}")) for i in range(4)]
    assert all(recs)
    assert p.open_record(hdrs(RID)) is None and p.DROPS["store_full"] == 1
    recs[1].close()
    recs[2].close()
    new = p.open_record(hdrs(RID))
    assert new is not None and len(p.STORE) == 4
    assert p.STORE.get_authorized(f"{1:032x}", TOK) is None  # oldest closed evicted
    assert p.STORE.get_authorized(f"{2:032x}", TOK) is not None


def test_ttl_purge(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(p, "clock", lambda: now[0])
    rec = p.open_record(hdrs())
    rec.close()
    now[0] += 99
    assert p.STORE.get_authorized(RID, TOK) is rec
    now[0] += 2
    assert p.STORE.get_authorized(RID, TOK) is None  # expired
    assert p.open_record(hdrs()) is not None  # same rid reusable after purge


def test_duplicate_rid_first_writer_wins():
    first = p.open_record(hdrs())
    assert p.open_record(hdrs(tok=TOK2)) is None
    assert p.STORE.get_authorized(RID, TOK) is first and p.STORE.get_authorized(RID, TOK2) is None


def test_get_authorized_paths(monkeypatch):
    p.open_record(hdrs())
    assert p.STORE.get_authorized(RID, TOK) is not None
    for rid, h in [(RID, None), (RID, ""), (RID, TOK2), (RID2, TOK), ("zz", TOK), (None, TOK), (RID.upper(), TOK), (RID, 5)]:
        assert p.STORE.get_authorized(rid, h) is None
    seen = []
    real = p.hmac.compare_digest
    monkeypatch.setattr(p.hmac, "compare_digest", lambda a, b: seen.append(b) or real(a, b))
    p.STORE.get_authorized(RID2, TOK)  # unknown rid
    p.STORE.get_authorized("bad", TOK)  # malformed rid
    p.STORE.get_authorized(RID, None)  # missing header, known rid
    assert seen[0] == seen[1] == p.DUMMY_HASH and seen[2] == sha256(TOK.encode()).digest()


def test_open_record_none_cases(monkeypatch):
    assert p.open_record(SimpleNamespace()) is None  # in-process callers: no headers attribute
    assert p.open_record(SimpleNamespace(headers={})) is None
    assert p.open_record(SimpleNamespace(headers={"x-vera-request-id": RID})) is None
    assert p.open_record(hdrs(rid="g" * 32)) is None
    assert p.open_record(hdrs(tok="short")) is None
    assert p.open_record(hdrs(tok="!" * 43)) is None
    p.configure(cfg(enabled=False))
    assert p.open_record(hdrs()) is None
    p.configure(cfg())
    monkeypatch.setenv("VERA_PROGRESS_DISABLED", "1")
    assert p.open_record(hdrs()) is None and not p.enabled()
    monkeypatch.delenv("VERA_PROGRESS_DISABLED")
    assert p.enabled()


def test_open_record_with_raising_store_returns_none(monkeypatch, caplog):
    class Boom:
        def create(self, *a):
            raise RuntimeError(SENTINEL)

    monkeypatch.setattr(p, "STORE", Boom())
    with caplog.at_level(logging.WARNING, logger="vera.progress"):
        assert p.open_record(hdrs()) is None
    assert "RuntimeError" in caplog.text and SENTINEL not in caplog.text and TOK not in caplog.text


def test_emit_never_raises_when_record_misbehaves(monkeypatch):
    rec, tok = sink()
    monkeypatch.setattr(rec, "append", lambda *a: (_ for _ in ()).throw(ValueError("x")))
    p.emit("write", "start")
    p.SINK.reset(tok)
    assert p.DROPS["exception"] == 1


def test_outcome_t_map():
    cc = {"answer_completeness": "partial"}
    assert p.outcome({"result_kind": "answer", "claim_check": cc}) is p.State.PARTIAL
    assert p.outcome(SimpleNamespace(result_kind="answer", claim_check={})) is p.State.ANSWERED
    assert p.outcome(SimpleNamespace(result_kind="answer", claim_check=None)) is p.State.ANSWERED
    assert p.outcome({"result_kind": "decline"}) is p.State.DECLINED
    for k in ("reading_list", "unspecified", "weird", None, 3):
        assert p.outcome({"result_kind": k}) is p.State.NO_ANSWER
    assert p.outcome(object()) is p.State.NO_ANSWER
    assert p.outcome_of("decline", "partial") is p.State.DECLINED


def test_threads_do_not_share_sink():
    rec, tok = sink()
    seen = []
    t = threading.Thread(target=lambda: seen.append(p.SINK.get()))
    t.start()
    t.join()
    p.SINK.reset(tok)
    assert seen == [None]


def test_store_keeps_only_token_hash():
    rec = p.open_record(hdrs())
    assert rec.token_hash == sha256(TOK.encode()).digest()
    assert TOK not in repr(vars(rec))


def test_config_validation_fails_verbosely_and_disables_only_progress(tmp_path, caplog):
    good = dict(enabled=False, max_records=1, max_events_per_record=2, max_events_per_poll=2, ttl_s=1, poll_interval_s=1,
                poll_margin_s=0, first_event_grace_s=0, get_timeout_s=2, max_count=1, _comment="x")
    assert isinstance(p.validate_config(good), p.Config)
    for k, v in [("max_records", 0), ("max_records", True), ("ttl_s", "1"), ("enabled", 1), ("bogus", 1), ("get_timeout_s", 3),
                 ("max_events_per_poll", 1)]:
        with pytest.raises(ValueError):
            p.validate_config({**good, k: v})
    with pytest.raises(ValueError) as ei:
        p.validate_config({**good, "max_records": -5, "ttl_s": None})
    assert "max_records" in str(ei.value) and "ttl_s" in str(ei.value)  # every bad key is named
    f = tmp_path / "c.json"
    f.write_text("{not json")
    with caplog.at_level(logging.ERROR, logger="vera.progress"):
        assert p.load_config(f) is None
    assert p.load_config(tmp_path / "missing.json") is None


def test_shipped_config_is_valid_and_ships_disabled():
    c = p.load_config()
    assert c is not None and c.enabled is False
