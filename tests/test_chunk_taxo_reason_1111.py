"""TAXO-REASON-1111: a failed taxonomy embedding run stores a specific, text-free reason (and integer-only batch shape and
transport metrics) instead of the bare word 'failed'. Plan: work/TAXO-REASON-1111 (T1-T13, leak rules L1-L6).
Offline: the FakeDB, FakeEmb, spans, make, Opener, cf and http_err helpers of test_chunk_taxo_job_1109.py are imported, not
edited; no DB, no network."""
from __future__ import annotations

import json
import logging
import re
import urllib.error

import pytest

from tests.test_chunk_taxo_job_1109 import (SENTINEL_TEXT, FakeDB, FakeEmb, Opener, cf, cf_ok, http_err, spans)
from tests.test_chunk_taxo_job_1109 import make as _make
from vera import chunk_taxo_embed as emb
from vera import chunk_taxo_job as job

LOGGER = "vera.chunk_taxo"
MARKER_SQL = "INSERT INTO secret_table VALUES (:v) -- SQLMARKER"
MARKER_VAL = "BOUNDVALUEMARKER"
MARKER_MSG = "secret-text https://host.example/x"
INT_KEYS = {"fail_rows_embedded", "fail_span_id", "fail_batch_n", "fail_min_chars", "fail_max_chars", "fail_total_chars",
            "fail_nonascii_chars", "last_http_status", "attempts",
            # WP-I D6 run-row contract: written on every close (flips INT_KEYS and T7 below on purpose)
            "scan_complete", "skipped_portal", "skipped_portal_error", "skipped_no_source", "skipped_empty", "no_vector_yet",
            "no_vector_yet_min_span_id", "stale_vector", "write_incomplete", "no_rebuild_snapshot_incomplete",
            "excluded_with_vector_portal", "excluded_with_vector_portal_error", "excluded_with_vector_no_source", "excluded_with_vector_empty"}


class RecDB(FakeDB):
    """FakeDB that also keeps the parameters of job_run_close and can raise a given exception on a named statement."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.closes, self.raise_map = [], {}

    def h_any(self, name, batch):
        if name == "job_run_close":
            self.closes.append(dict(batch[0]))
        if name in self.raise_map:
            raise self.raise_map[name]
        return super().h_any(name, batch)


class RaisingEmb(FakeEmb):
    def __init__(self, exc, ok_calls=0):
        super().__init__()
        self.exc, self.ok_calls = exc, ok_calls

    def embed(self, texts):
        if len(self.calls) >= self.ok_calls:
            raise self.exc
        return super().embed(texts)


class ShortEmb(FakeEmb):
    def embed(self, texts):
        return super().embed(texts)[:-1]


class MixedDimEmb(FakeEmb):
    def embed(self, texts):
        self.calls.append(len(texts))
        return [[1.0, 0.0, 0.5] if "beta" in t else [1.0, 0.0] for t in texts]


def fail_run(e, caplog, db=None, **jobkw):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    db = db or RecDB(spans())
    j = _make(db, e=e, **jobkw)
    res = j.run()
    return res, j, db


def fail_records(caplog):
    return [r for r in caplog.records if "CT_JOB_ERROR" in r.getMessage()]


def stored(db):
    c = db.closes[-1]
    return c["reasons"], json.loads(c["metrics"])


def everything(db, j, caplog):
    parts = [repr(db.closes), repr(j.status())]
    for r in caplog.records:
        parts += [r.getMessage(), str(r.exc_text), repr(r.exc_info), repr(r.stack_info), repr(r.args)]
    return "\n".join(parts)


def assert_no_traceback(caplog):
    for r in caplog.records:
        assert r.exc_info is None and r.stack_info is None  # L4


# ---------------------------------------------------------------------------------------------------------------- T1
def test_t1_embed_http_429_reason_phase_rows_span(caplog):
    # two successful batches of 2, then a 429: rows_embedded 4, the failing batch starts at span 5
    res, j, db = fail_run(RaisingEmb(emb.EmbedError("http_429"), ok_calls=2), caplog, embed_batch=2)
    assert res["state"] == "error" and res["rows_embedded"] == 4
    reasons, metrics = stored(db)
    assert reasons == ["embed:http_429"] and db.closes[-1]["state"] == "failed"
    assert j.status()["state"] == "error" and j.status()["reasons"] == ["embed:http_429"]
    assert metrics["fail_rows_embedded"] == 4 and metrics["fail_span_id"] == 5
    msg = fail_records(caplog)[-1].getMessage()
    assert "reason=embed:http_429" in msg and "phase=scan" in msg and "rows_embedded=4" in msg and "span_id=5" in msg
    assert "batch_n=2" in msg and "last_http_status=0" in msg and "attempts=0" in msg
    assert_no_traceback(caplog)


# ---------------------------------------------------------------------------------------------------------------- T2
@pytest.mark.parametrize("code", ["http_429", "dim_mismatch", "request_failed", "send_not_allowed",
                                  "credentials_missing", "response_too_large", "timeout", "model_unknown"])
def test_t2_each_embed_code(code, caplog):
    res, j, db = fail_run(RaisingEmb(emb.EmbedError(code)), caplog)
    reasons, _ = stored(db)
    assert res["state"] == "error" and reasons == ["embed:" + code] and reasons != ["failed"]
    assert "reason=embed:" + code in fail_records(caplog)[-1].getMessage()
    assert_no_traceback(caplog)


# ---------------------------------------------------------------------------------------------------------------- T3
def test_t3_named_runtime_codes(caplog):
    res, j, db = fail_run(ShortEmb(), caplog)
    assert stored(db)[0] == ["embed_count_mismatch"] and res["state"] == "error"
    res, j, db = fail_run(MixedDimEmb(), caplog)
    assert stored(db)[0] == ["dim_mismatch"]
    # bad_phase: an unknown phase value; the logged phase is cut to the closed charset (L5)
    db = RecDB(spans())
    j = _make(db)
    j._do_start = lambda: setattr(j, "_phase", "bad phase/ x\n" + SENTINEL_TEXT)
    caplog.clear()
    j.run()
    assert j.status()["reasons"] == ["bad_phase"]  # no run row exists: _do_start was replaced
    msg = fail_records(caplog)[-1].getMessage()
    assert "reason=bad_phase" in msg and "phase=badphasexSENTINELCHUNKBODY" in msg and "\n" not in msg
    # node_not_active: direct (driving _do_nodes with the fake store is impractical)
    for code in ("embed_count_mismatch", "dim_mismatch", "bad_phase", "node_not_active"):
        assert job._reason_for(RuntimeError(code)) == code


# ---------------------------------------------------------------------------------------------------------------- T4
def test_t4_generic_exception_message_never_read(caplog):
    res, j, db = fail_run(RaisingEmb(ValueError(MARKER_MSG)), caplog)
    reasons, metrics = stored(db)
    assert reasons == ["ValueError"] and res["state"] == "error"
    blob = everything(db, j, caplog)
    for frag in ("secret-text", "host.example", MARKER_MSG):
        assert frag not in blob
    assert_no_traceback(caplog)


# ---------------------------------------------------------------------------------------------------------------- T5
@pytest.mark.parametrize("msg", ["has space", "a/b", "x" * 41, "UPPER", "", "http://h/x", "ünï", "http_429\n", "timeout\n"])
def test_t5_off_pattern_embed_error(msg, caplog):
    assert job._reason_for(emb.EmbedError(msg)) == "embed:unknown"
    res, j, db = fail_run(RaisingEmb(emb.EmbedError(msg)), caplog)
    assert stored(db)[0] == ["embed:unknown"]


# ---------------------------------------------------------------------------------------------------------------- T6
def test_t6_span_text_never_in_reasons_metrics_logs(caplog):
    res, j, db = fail_run(RaisingEmb(emb.EmbedError("http_429")), caplog)
    reasons, metrics = stored(db)
    assert SENTINEL_TEXT not in everything(db, j, caplog)
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in metrics.values())
    assert set(metrics) <= INT_KEYS


# ---------------------------------------------------------------------------------------------------------------- T7
def test_t7_old_outcomes_unchanged(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    db = RecDB(spans())
    j = _make(db, e=RaisingEmb(emb.Deferred("quota_reached", 100.0)))
    res = j.run()
    assert res["state"] == "deferred" and j.status()["reasons"] == ["deferred_until_utc_midnight"]
    # WP-I flip: a deferred close now carries the scan contract numbers (scan_complete, skipped_*, excluded_with_vector_*) and nothing else
    m7 = json.loads(db.closes[-1]["metrics"])  # (the scan went on reuse-only to its end, so the unvectored spans are counted)
    assert stored(db)[0] == ["deferred"] and m7["scan_complete"] == 1 and m7["no_vector_yet"] == 10 and m7["no_vector_yet_min_span_id"] == 1
    assert not fail_records(caplog)
    db = RecDB(spans())
    j = _make(db, max_rows=2, embed_batch=2)
    res = j.run()
    assert res["stop"] == "max_rows" and j.status()["reasons"] == ["max_rows"] and stored(db)[0] == ["max_rows"]
    # no failure test stores the bare state word
    res, j, db = fail_run(RaisingEmb(RuntimeError("x")), caplog)
    assert stored(db)[0] != ["failed"] and j.status()["reasons"] != ["failed"]


# ---------------------------------------------------------------------------------------------------------------- T8
def test_t8_outer_handler(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    db = RecDB(spans())
    j = _make(db)

    def boom():
        raise ValueError(MARKER_MSG)

    j._release_buffers = boom  # fails after the step loop, outside the inner handler
    res = j.run()
    assert res["state"] == "error" and res["complete"] is False
    msg = fail_records(caplog)[-1].getMessage()
    assert "reason=ValueError" in msg and "phase=" in msg and "rows_embedded=" in msg
    assert j.status()["reasons"] == ["ValueError"] and MARKER_MSG not in everything(db, j, caplog)
    assert_no_traceback(caplog)


# ---------------------------------------------------------------------------------------------------------------- T9
def test_t9_reason_length_and_charset(caplog):
    long_cls = type("A" * 100, (Exception,), {})
    odd_cls = type("Bad Name/with-chars\nx", (Exception,), {})
    seen = []
    for exc in (long_cls("m"), odd_cls("m"), emb.EmbedError("http_503"), RuntimeError("x"), KeyError("k"),
                emb.EmbedError("y" * 40), emb.EmbedError("y" * 41)):
        res, j, db = fail_run(RaisingEmb(exc), caplog)
        reasons, _ = stored(db)
        seen += reasons + j.status()["reasons"]
    assert seen and all(len(r) <= 64 and re.fullmatch(r"[A-Za-z0-9_.:]+", r) for r in seen)
    assert job._reason_for(long_cls("m")) == "A" * 40
    assert job._reason_for(odd_cls("m")) == "BadNamewith-charsx".replace("-", "")


# --------------------------------------------------------------------------------------------------------------- T10
class IntegrityError(Exception):
    def __init__(self):
        super().__init__("(psycopg.errors.UniqueViolation) " + MARKER_SQL + " " + MARKER_VAL)
        self.orig, self.statement, self.params, self.pgerror = MARKER_MSG, MARKER_SQL, {"v": MARKER_VAL}, MARKER_SQL


class OperationalError(IntegrityError):
    pass


OperationalError.__name__ = "OperationalError"


@pytest.mark.parametrize("cls", [IntegrityError, OperationalError])
def test_t10_db_error_class_name_only(cls, caplog):
    db = RecDB(spans())
    db.raise_map["vectors_upsert"] = cls()
    res, j, db = fail_run(FakeEmb(), caplog, db=db)
    reasons, metrics = stored(db)
    assert res["state"] == "error" and reasons == [cls.__name__] and metrics["fail_span_id"] == 1
    blob = everything(db, j, caplog)
    for frag in (MARKER_SQL, MARKER_VAL, MARKER_MSG, "SQLMARKER", "secret_table"):
        assert frag not in blob
    assert_no_traceback(caplog)


def test_t10_failure_inside_close_run_keeps_reason_in_log(caplog):
    db = RecDB(spans())
    db.raise_map["job_run_close"] = IntegrityError()
    res, j, db = fail_run(RaisingEmb(emb.EmbedError("http_429")), caplog, db=db)
    assert res["state"] == "error"
    assert "reason=embed:http_429" in fail_records(caplog)[-1].getMessage()
    assert j.status()["reasons"] == ["embed:http_429"]
    assert MARKER_SQL not in everything(db, j, caplog)


# --------------------------------------------------------------------------------------------------------------- T11
def test_t11_lookalikes(caplog):
    class SubRT(RuntimeError):
        pass

    class BadStr(Exception):
        def __str__(self):
            raise ValueError(MARKER_MSG)

    class BadStrRT(RuntimeError):
        def __str__(self):
            raise ValueError(MARKER_MSG)

    assert job._reason_for(RuntimeError("embed_count_mismatch extra text")) == "RuntimeError"
    assert job._reason_for(SubRT("dim_mismatch")) == "SubRT"
    assert job._reason_for(BadStr()) == "BadStr"
    assert job._reason_for(RuntimeError()) == "RuntimeError"
    assert job._reason_for(None) in ("NoneType", "unknown")
    # a plain RuntimeError whose __str__ raises: the classifier never raises (L3)
    class RuntimeError2(RuntimeError):
        pass

    assert job._reason_for(BadStrRT()) == "BadStrRT"
    res, j, db = fail_run(RaisingEmb(BadStr()), caplog)
    assert res["state"] == "error" and stored(db)[0] == ["BadStr"] and MARKER_MSG not in everything(db, j, caplog)
    assert_no_traceback(caplog)


# --------------------------------------------------------------------------------------------------------------- T12
def test_t12_batch_shape_metrics(caplog):
    long_txt = "é" * 3 + "ß" + "x" * 20 + "ZZNONASCIIMARK"
    sp = [(1, 100, " x\t"), (2, 101, long_txt)]  # ROBUST-1111: was blank "   \t", now filtered before batching
    db = RecDB(sp)
    res, j, db = fail_run(RaisingEmb(emb.EmbedError("http_401")), caplog, db=db)
    reasons, m = stored(db)
    assert reasons == ["embed:http_401"]
    assert m["fail_batch_n"] == 2 and m["fail_min_chars"] == 1 and m["fail_max_chars"] == len(long_txt)
    assert m["fail_total_chars"] == 3 + len(long_txt) and m["fail_nonascii_chars"] == 4 and m["fail_span_id"] == 1
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in m.values()) and set(m) <= INT_KEYS
    blob = everything(db, j, caplog)
    assert "ZZNONASCIIMARK" not in blob and "é" not in blob and "ß" not in blob
    # reset after a committed batch: the second batch's shape is reported, not the first's
    res, j, db = fail_run(RaisingEmb(emb.EmbedError("http_401"), ok_calls=1), caplog, embed_batch=3)
    _, m = stored(db)
    assert m["fail_span_id"] == 4 and m["fail_batch_n"] == 3
    # a later failure outside _do_scan stores no batch shape and no span id
    db = RecDB(spans())
    db.fail_on = "link_insert_chunk_subtopic"
    res, j, db = fail_run(FakeEmb(), caplog, db=db)
    reasons, m = stored(db)
    assert res["state"] == "error" and reasons == ["RuntimeError"]
    assert "fail_batch_n" not in m and "fail_span_id" not in m and "fail_max_chars" not in m and "fail_rows_embedded" in m
    assert j._fail_span is None and j._fail_batch is None


# --------------------------------------------------------------------------------------------------------------- T13
def test_t13_transport_codes_real_embedder():
    n = 4  # max_attempts default
    e = cf(Opener(*[TimeoutError("t") for _ in range(n)]))
    with pytest.raises(emb.EmbedError, match="^timeout$"):
        e.embed(["a"])
    assert e.stats_run["attempts"] == n and e.stats_run["last_http_status"] == 0
    e = cf(Opener(*[urllib.error.URLError(TimeoutError("t")) for _ in range(n)]))
    with pytest.raises(emb.EmbedError, match="^timeout$"):
        e.embed(["a"])
    e = cf(Opener(*[urllib.error.URLError("refused") for _ in range(n)]))
    with pytest.raises(emb.EmbedError, match="^request_failed$"):
        e.embed(["a"])
    assert e.stats_run["attempts"] == n and e.stats_run["last_http_status"] == 0
    e = cf(Opener(http_err(400)))
    with pytest.raises(emb.EmbedError, match="^http_400$"):
        e.embed(["a"])
    assert e.stats_run["attempts"] == 1 and e.stats_run["last_http_status"] == 400
    # 5xx retried to the end: request_failed with the last status; a later success resets the counters per request
    e = cf(Opener(*[http_err(503) for _ in range(n)]))
    with pytest.raises(emb.EmbedError, match="^request_failed$"):
        e.embed(["a"])
    assert e.stats_run["last_http_status"] == 503
    e = cf(Opener(cf_ok(1)))
    e.embed(["a"])
    assert e.stats_run["attempts"] == 1 and e.stats_run["last_http_status"] == 0
    assert set(e.stats()) == {"neurons_today", "neurons", "requests", "deferred"}  # stats() output shape unchanged
    for k in ("attempts", "last_http_status"):
        assert isinstance(e.stats_run[k], int) and not isinstance(e.stats_run[k], bool)


def test_t13_run_row_metrics_carry_transport_ints(caplog):
    e = cf(Opener(http_err(401)))
    res, j, db = fail_run(e, caplog)
    reasons, m = stored(db)
    assert reasons == ["embed:http_401"] and m["last_http_status"] == 401 and m["attempts"] == 1 and m["fail_span_id"] == 1
    msg = fail_records(caplog)[-1].getMessage()
    assert "last_http_status=401" in msg and "attempts=1" in msg
    e = cf(Opener(*[TimeoutError("t") for _ in range(4)]))
    res, j, db = fail_run(e, caplog)
    reasons, m = stored(db)
    assert reasons == ["embed:timeout"] and m["attempts"] == 4 and m["last_http_status"] == 0
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in m.values())
    assert_no_traceback(caplog)
