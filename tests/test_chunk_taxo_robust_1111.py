"""TAXO-ROBUST-1111: the taxonomy embedding job survives empty, oversize and poison spans (plan: work/TAXO-ROBUST-1111, T1-T16).
Offline: the FakeDB, FakeEmb, spans, make, StubCore, Opener, cf, cf_ok and http_err helpers of test_chunk_taxo_job_1109.py and the
RecDB/stored/fail_run helpers of test_chunk_taxo_reason_1111.py are imported, not edited. No DB, no network."""
from __future__ import annotations

import json
import logging
import math

import pytest

from tests.test_chunk_taxo_job_1109 import (ENV, FakeEmb, Opener, StubCore, cf, cf_ok, factory_for, http_err, spans)
from tests.test_chunk_taxo_job_1109 import make as _make
from tests.test_chunk_taxo_reason_1111 import RecDB, everything, fail_run, stored
from vera import chunk_taxo as ct
from vera import chunk_taxo_embed as emb
from vera import chunk_taxo_job as job

LEAK = "LEAKMARKER"


class PoisonEmb(FakeEmb):
    """Raises EmbedError(code) for any request that holds the marker (or for every request); records every request."""

    def __init__(self, marker="POISON", code="http_400", always=False, limit=None, then=None):
        super().__init__()
        self.marker, self.code, self.always, self.requests, self.then = marker, code, always, [], then
        if limit is not None:
            self.batch_limit = limit

    def embed(self, texts):
        texts = list(texts)
        self.requests.append(texts)
        if self.then is not None:
            self.then(self)
        if self.always or any(self.marker in t for t in texts):
            raise emb.EmbedError(self.code)
        return super().embed(texts)


def poison_spans(n=32, at=(7,), tag="POISON"):
    sp = spans(n)
    for i in at:
        sp[i - 1] = (i, 100 + i % 3, f"alpha {tag} {i}")
    return sp


def make_core(db, e, core, **jobkw):
    return job.Job({"enabled": True, "params": {}, "display_gate": {}, "embedder": "x", "job": dict(jobkw)}, core=core, embedder=e,
                   engine_factory=factory_for(db, []), env=dict(ENV), rss_reader=lambda: 100.0)


def flat_ints(x):
    if isinstance(x, dict):
        return all(flat_ints(v) for v in x.values())
    return isinstance(x, (int, bool)) and not isinstance(x, float)


# ---------------------------------------------------------------------------------------------------------------- T1
def test_t1_empty_spans_never_sent():
    sp = spans(30) + [(31, 100, ""), (32, 101, "   \n")]
    db, e = RecDB(sp), PoisonEmb(always=False)
    j = _make(db, e=e)
    res = j.run()
    assert res["complete"] and j.skipped_empty == 2
    assert all(t.strip() for req in e.requests for t in req) and sum(len(r) for r in e.requests) == 30
    assert len(db.vectors) == 30 and 31 not in db.vectors and 32 not in db.vectors


# ---------------------------------------------------------------------------------------------------------------- T2
def test_t2_one_poison_span_quarantined_by_bisection():
    db, e = RecDB(poison_spans()), PoisonEmb()
    j = _make(db, e=e)
    res = j.run()
    assert res["complete"] and set(j._quarantine) == {7} and j._quarantine[7][1] == "http_400"
    assert len(db.vectors) == 31 and 7 not in db.vectors
    assert len(e.requests) <= 2 * math.ceil(math.log2(32)) + 1
    run = json.loads(db.runs[-1]["metrics"])["run"]
    assert run["quarantined"] == 1 and run["q"] == {"q7": 400} and run["q_last_status"] == 400


# ---------------------------------------------------------------------------------------------------------------- T3
@pytest.mark.parametrize("code,status", [("http_400", 400), ("http_413", 413), ("http_422", 422), ("bad_response", 0)])
def test_t3_systemic_4xx_consecutive_cap_and_rollback(caplog, code, status):
    e = PoisonEmb(always=True, code=code)
    res, j, db = fail_run(e, caplog, db=RecDB(spans(10)))
    reasons, m = stored(db)
    assert res["state"] == "error" and reasons == ["quarantine_cap_consecutive"] and reasons != ["failed"]
    assert sum(1 for r in e.requests if len(r) == 1) <= j.q_max_consec
    assert m["q_last_status"] == status and m["q_consecutive"] == j.q_max_consec
    assert j._quarantine == {} and not db.vectors  # streak rolled back, nothing stored
    e.always = False  # the fault clears: a second pass embeds every span
    res = j.run()
    assert res["complete"] and len(db.vectors) == 10


# ---------------------------------------------------------------------------------------------------------------- T4
def test_t4_total_cap(caplog):
    sp = spans(8)
    for i in (1, 3, 5):
        sp[i - 1] = (i, 100, f"alpha POISON {i}")
    res, j, db = fail_run(PoisonEmb(), caplog, db=RecDB(sp), embed_batch=2, quarantine_max=2)
    assert res["state"] == "error" and stored(db)[0] == ["quarantine_cap_total"] and set(j._quarantine) == {1, 3}


# ---------------------------------------------------------------------------------------------------------------- T5
@pytest.mark.parametrize("code", ["timeout", "request_failed", "http_429", "http_401", "http_403", "credentials_missing",
                                  "dim_mismatch", "send_not_allowed", "response_too_large", "model_unknown", "http_500"])
def test_t5_never_quarantined(code, caplog):
    res, j, db = fail_run(PoisonEmb(code=code, always=True), caplog)
    assert res["state"] == "error" and stored(db)[0] == ["embed:" + code] and j._quarantine == {}


def test_t5_deferred_and_runtime_error_not_quarantined(caplog):
    res, j, db = fail_run(PoisonEmb(code="x", then=lambda s: (_ for _ in ()).throw(emb.Deferred("quota_reached", 5.0))), caplog)
    assert res["state"] == "deferred" and j._quarantine == {}
    res, j, db = fail_run(PoisonEmb(then=lambda s: (_ for _ in ()).throw(RuntimeError("embed_count_mismatch"))), caplog)
    assert stored(db)[0] == ["embed_count_mismatch"] and j._quarantine == {}


def test_t5_503_surfaces_as_request_failed(caplog):
    e = cf(Opener(http_err(503), http_err(503)), {"max_attempts": 2})
    res, j, db = fail_run(e, caplog)
    assert stored(db)[0] == ["embed:request_failed"] and j._quarantine == {}


# ---------------------------------------------------------------------------------------------------------------- T6
def test_t6_timeout_through_real_embedder(caplog):
    e = cf(Opener(*[TimeoutError("t") for _ in range(4)]))
    res, j, db = fail_run(e, caplog)
    reasons, m = stored(db)
    assert reasons == ["embed:timeout"] and m["attempts"] == 4 and j._quarantine == {}


# ---------------------------------------------------------------------------------------------------------------- T7
def sized(lengths):
    return [(i + 1, 100, ("alpha " + "x" * n)[:n]) for i, n in enumerate(lengths)]


def test_t7_char_budget_and_count_ceiling():
    lens = [400, 400, 400, 900, 50, 50, 1500, 300, 300, 300]
    e = PoisonEmb(always=False)
    db = RecDB(sized(lens))
    assert make_core(db, e, StubCore(), embed_batch_chars=1000).run()["complete"]
    for req in e.requests:
        assert sum(len(t) for t in req) <= 1000 or len(req) == 1
    assert sorted(len(t) for r in e.requests for t in r) == sorted(lens) and len(db.vectors) == len(lens)
    # every row longer than half the budget: one span per request
    e, db = PoisonEmb(always=False), RecDB(sized([600] * 5))
    assert make_core(db, e, StubCore(), embed_batch_chars=1000).run()["complete"]
    assert [len(r) for r in e.requests] == [1] * 5
    # count ceiling and the embedder's batch_limit
    e, db = PoisonEmb(always=False, limit=4), RecDB(spans(20))
    assert _make(db, e=e, embed_batch=32).run()["complete"] and max(len(r) for r in e.requests) == 4
    e, db = PoisonEmb(always=False), RecDB(spans(20))
    assert _make(db, e=e, embed_batch=3).run()["complete"] and max(len(r) for r in e.requests) == 3


# ---------------------------------------------------------------------------------------------------------------- T8
def test_t8_truncate_inputs_in_body():
    op = Opener(lambda n: cf_ok(n, 4))
    cf(op).embed(["a"])
    assert json.loads(op.reqs[0].data) == {"text": ["a"], "truncate_inputs": True}
    op = Opener(lambda n: cf_ok(n, 4))
    cf(op, {"truncate_inputs": False}).embed(["a"])
    assert "truncate_inputs" not in json.loads(op.reqs[0].data)


# ---------------------------------------------------------------------------------------------------------------- T9
def test_t9_resume_and_restart_after_quarantine():
    db, e = RecDB(poison_spans()), PoisonEmb()
    j = _make(db, e=e)
    assert j.run()["complete"]
    n = len(e.requests)
    assert j.run()["complete"]
    assert len(e.requests) == n and set(j._quarantine) == {7}  # nothing re-sent, count persists
    e2 = PoisonEmb()
    j2 = _make(RecDB(poison_spans()), e=e2)  # restart on an empty store: re-found within the bound
    assert j2.run()["complete"] and set(j2._quarantine) == {7} and len(e2.requests) <= 11
    j3 = _make(db, e=PoisonEmb())  # restart on the same store: only the missing span is retried
    assert j3.run()["complete"] and set(j3._quarantine) == {7}


# ---------------------------------------------------------------------------------------------------------------- T10
def test_t10_quarantine_released_on_text_change():
    db, e = RecDB(poison_spans()), PoisonEmb()
    j = _make(db, e=e)
    j.run()
    assert 7 in j._quarantine and 7 not in db.vectors
    db.spans[6] = (7, 100 + 7 % 3, "alpha fixed text 7")
    assert j.run()["complete"] and j._quarantine == {} and 7 in db.vectors


# ---------------------------------------------------------------------------------------------------------------- T11
class SpyCore(StubCore):
    def build(self, chunks, vecs, params, prior=None):
        self.ids, self.rows = [int(c[0]) for c in chunks], len(vecs)
        return super().build(chunks, vecs, params, prior)


def test_t11_block_build_skips_rows_without_vector():
    sp = [(1, 100, ""), (2, 101, "alpha ok 2"), (3, 102, "alpha POISON 3"), (4, 100, "beta ok 4"), (5, 101, "  ")]
    db, core = RecDB(sp), SpyCore()
    j = make_core(db, PoisonEmb(), core)
    res = j.run()
    assert res["complete"] and core.ids == [2, 4] and core.rows == 2 and j._plan == [2, 4]
    assert j._cursor == 5 and j._scan_done and j.vectors_missing == 3
    # a page of only skipped rows adds no block and does not hang; the next page is read
    db, core = RecDB([(1, 100, ""), (2, 100, " "), (3, 100, "alpha 3"), (4, 100, "beta 4")]), SpyCore()
    assert make_core(db, PoisonEmb(), core, page_size=2).run()["complete"] and core.ids == [3, 4]
    # a corpus of only empty spans ends idle with no_chunks
    db = RecDB([(1, 100, ""), (2, 100, "\n")])
    j = make_core(db, PoisonEmb(), SpyCore())
    res = j.run()
    assert res["complete"] and j.status()["state"] == "idle" and j.status()["reasons"] == ["no_chunks"]


# ---------------------------------------------------------------------------------------------------------------- T12
def test_t12_normal_corpus_unchanged():
    db, e = RecDB(spans(10)), FakeEmb()
    j = _make(db, e=e)
    assert j.run()["complete"] and e.calls == [10] and len(db.vectors) == 10 and j._quarantine == {}
    assert "quarantined" not in json.loads(db.runs[-1]["metrics"])["run"]
    db, e = RecDB(spans(10)), FakeEmb()
    assert _make(db, e=e, embed_batch=3).run()["complete"] and e.calls == [3, 3, 3, 1]


# ---------------------------------------------------------------------------------------------------------------- T13
def test_t13_no_leak_ints_only(caplog):
    caplog.set_level(logging.DEBUG, logger="vera.chunk_taxo")
    sp = poison_spans(at=(7, 9), tag=LEAK) + [(33, 100, "")]
    db = RecDB(sp)
    j = _make(db, e=PoisonEmb(marker=LEAK))
    assert j.run()["complete"]
    blob = everything(db, j, caplog) + repr(j._quarantine) + repr(db.runs)
    assert LEAK not in blob
    run = json.loads(db.runs[-1]["metrics"])["run"]
    assert flat_ints(run["q"]) and set(run["q"]) == {"q7", "q9"} and len(run["q"]) <= 64
    assert [k for k in run if k.startswith("q") and k not in ("q", "q_consecutive", "q_last_status", "quarantined")] == []
    assert flat_ints({k: v for k, v in run.items() if k not in ("seconds", "rows_per_s", "peak_rss_mb", "embedder")})
    for r in caplog.records:
        assert r.exc_info is None and r.stack_info is None


# ---------------------------------------------------------------------------------------------------------------- T14
def test_t14_bounds_and_config_validation(monkeypatch):
    for key, default in (("embed_batch_chars", 48000), ("quarantine_max", 50), ("quarantine_max_consecutive", 4)):
        lo, hi = job._BOUNDS[key]
        assert job._num({}, key) == default
        for bad in (lo - 1, hi + 1, "9", None, True, float("nan")):
            assert job._num({key: bad}, key) == default
        assert job._num({key: lo}, key) == lo and ct._JOB_NUM[key] == job._BOUNDS[key]
    assert ct._validate_job({"embed_batch_chars": 2000, "quarantine_max": 5, "quarantine_max_consecutive": 2})
    with pytest.raises(ValueError):
        ct._validate_job({"embed_batch_chars": 5})
    with pytest.raises(ValueError):
        ct._validate_job({"quarantine_bogus": 5})
    assert ct._validate_cloudflare({"truncate_inputs": False}) == {"truncate_inputs": False}
    for bad in (1, "yes", None):
        with pytest.raises(ValueError):
            ct._validate_cloudflare({"truncate_inputs": bad})
    with pytest.raises(ValueError):
        ct._validate_cloudflare({"bogus": True})
    # CHUNK_TAXO_EMBED_BATCH=512 cannot defeat the char budget or the embedder's batch_limit
    sp = [(i, 100, "alpha " + "x" * 1994) for i in range(1, 201)]
    e = PoisonEmb(always=False, limit=64)
    j = job.Job({"enabled": True, "params": {}, "display_gate": {}, "embedder": "x", "job": {}}, core=StubCore(), embedder=e,
                engine_factory=factory_for(RecDB(sp), []), env=dict(ENV, CHUNK_TAXO_EMBED_BATCH="512"), rss_reader=lambda: 100.0)
    assert j.batch == 512 and j.run()["complete"]
    assert all(sum(len(t) for t in r) <= 48000 and len(r) <= 64 for r in e.requests) and max(len(r) for r in e.requests) == 24


# ---------------------------------------------------------------------------------------------------------------- T15
def test_t15_counts_durable_on_deferred_partial_and_finished(caplog):
    sp = spans(6)
    sp[0] = (1, 100, "alpha POISON 1")
    sp.append((7, 100, ""))

    def defer(s):
        if len(s.requests) >= 3:
            raise emb.Deferred("quota_reached", 5.0)
    res, j, db = fail_run(PoisonEmb(then=defer), caplog, db=RecDB(sp), embed_batch=2)
    assert res["state"] == "deferred"
    c = db.closes[-1]
    m = json.loads(c["metrics"])
    assert c["reasons"] == ["deferred"] and m["quarantined"] == 1 and m["skipped_empty"] == 1 and m["q"] == {"q1": 400} and flat_ints(m)
    res, j, db = fail_run(PoisonEmb(), caplog, db=RecDB(sp), embed_batch=2, max_rows=1)
    assert res["state"] == "partial"
    m = json.loads(db.closes[-1]["metrics"])
    assert m["quarantined"] == 1 and m["q"] == {"q1": 400} and flat_ints(m)
    db = RecDB(sp)
    assert _make(db, e=PoisonEmb()).run()["complete"]
    assert json.loads(db.runs[-1]["metrics"])["run"]["quarantined"] == 1


# ---------------------------------------------------------------------------------------------------------------- T16
def test_t16_failed_4xx_charged_and_persisted(monkeypatch, caplog):
    est = 3 / 3.0 / 1e6 * 1075
    for code, charged in ((400, True), (413, True), (422, True), (401, False), (403, False)):
        e = cf(Opener(http_err(code)))
        with pytest.raises(emb.EmbedError):
            e.embed(["abc"])
        assert (e.stats_run["neurons"] == pytest.approx(est)) is charged
    for code in (429, 503):
        e = cf(Opener(http_err(code)), {"max_attempts": 1})
        with pytest.raises(emb.EmbedError):
            e.embed(["abc"])
        assert e.stats_run["neurons"] == 0
    monkeypatch.setattr(emb, "CHARGE_FAILED_4XX", False)
    e = cf(Opener(http_err(400)))
    with pytest.raises(emb.EmbedError):
        e.embed(["abc"])
    assert e.stats_run["neurons"] == 0
    monkeypatch.setattr(emb, "CHARGE_FAILED_4XX", True)
    # a consecutive-cap failure still persists the charged total in the closed run row
    e = cf(Opener(*[http_err(400) for _ in range(60)]))
    res, j, db = fail_run(e, caplog, db=RecDB(spans(10)))
    assert stored(db)[0] == ["quarantine_cap_consecutive"]
    assert e.stats_run["neurons"] > 0 and db.closes[-1]["neurons"] == job._ceil_int(e.stats_run["neurons"]) > 0


# ------------------------------------------------------------------------------------------------------------ D1 (verifier)
def test_d1_partial_vectors_stored_when_deferred_or_capped_midway(caplog):
    def defer(s):
        if len(s.requests) >= 3:
            raise emb.Deferred("quota_reached", 5.0)
    res, j, db = fail_run(PoisonEmb(then=defer), caplog, db=RecDB(poison_spans(8, at=(8,))))
    assert res["state"] == "deferred" and sorted(db.vectors) == [1, 2, 3, 4] and j.rows_embedded == 4
    sp = poison_spans(8, at=(5, 6, 7, 8))
    res, j, db = fail_run(PoisonEmb(), caplog, db=RecDB(sp), quarantine_max_consecutive=2)
    assert stored(db)[0] == ["quarantine_cap_consecutive"] and sorted(db.vectors) == [1, 2, 3, 4]
    e2 = PoisonEmb()
    j2 = _make(db, e=e2, quarantine_max_consecutive=10)  # a restart does not re-send the stored four
    j2.run()
    assert all(t not in sum(e2.requests, []) for t in [s[2] for s in sp[:4]])


# ------------------------------------------------------------------------------------------------------------ D2 (verifier)
def test_d2_reason_for_quarantinable_embed_codes():
    assert job._reason_for(emb.EmbedError("http_400")) == "embed:http_400"
    assert job._reason_for(emb.EmbedError("bad_response")) == "embed:bad_response"
    assert job._reason_for(RuntimeError("quarantine_cap_total")) == "quarantine_cap_total"


# ------------------------------------------------------------------------------------------------------------ D3 (verifier)
def test_d3_malformed_200_charged_like_failed_4xx(monkeypatch):
    est = 3 / 3.0 / 1e6 * 1075
    bad = {"result": {"data": []}, "success": True}
    e = cf(Opener(bad))
    with pytest.raises(emb.EmbedError, match="bad_response"):
        e.embed(["abc"])
    assert e.stats_run["neurons"] == pytest.approx(est)
    monkeypatch.setattr(emb, "CHARGE_FAILED_4XX", False)
    e = cf(Opener(bad))
    with pytest.raises(emb.EmbedError):
        e.embed(["abc"])
    assert e.stats_run["neurons"] == 0
