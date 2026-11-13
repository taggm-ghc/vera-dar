"""WP-F (CHUNK-ASSOC-1113, PLAN Amendment 1 A3, T44-T50 and T50a): the finish-now fallback of the chunk-taxonomy job.
With config job.finish_now literally true, no active build, and a pass whose embedding stopped on a budget code (budget_reached or
quota_reached), the scan finishes reuse-only and the first build writes links for the vectored spans only. Everything else behaves as
main. Offline: FakeDB, RecDB, StubCore, FakeEmb and helpers are imported from the 1109 and 1111 tests (not edited); real span tuples
(span_id, source_id, text), real statement names; reserved fixture values only. No DB, no network."""
from __future__ import annotations

import copy
import hashlib
import json
from itertools import count
from types import SimpleNamespace

import pytest

from tests.test_arxiv_defense_1113 import mixed_db
from tests.test_chunk_taxo_job_1109 import ENV, Conn, FakeEmb, StubCore, factory_for, spans
from tests.test_chunk_taxo_reason_1111 import MixedDimEmb, RaisingEmb, RecDB, stored
from vera import chunk_taxo as ct
from vera import chunk_taxo_embed as emb
from vera import chunk_taxo_job as job

FLAG = "finish_now"
PREFIX = "auto-chunk-taxo:"


class DeferEmb(FakeEmb):
    """Embeds `ok_calls` batches, then raises Deferred(code); counts every attempt."""

    def __init__(self, ok_calls=0, code="budget_reached"):
        super().__init__()
        self.ok_calls, self.code, self.attempts = ok_calls, code, 0

    def embed(self, texts):
        self.attempts += 1
        if len(self.calls) >= self.ok_calls:
            raise emb.Deferred(self.code, 4321.0)
        return super().embed(texts)


class SpyCore(StubCore):
    def __init__(self):
        self.built = []

    def build(self, chunks, vecs, params, prior=None):
        self.built.append([int(c[0]) for c in chunks])
        return super().build(chunks, vecs, params, prior=prior)


class AbstainCore(SpyCore):
    """Abstains (no sub-topic, no topic) on span 4, otherwise the stub rule."""

    def build(self, chunks, vecs, params, prior=None):
        m = super().build(chunks, vecs, params, prior=prior)
        m.assoc[4] = dict(subtopic=None, topic=None, domain="D", level="domain", distance=0.9, margin=0.0, reason="ABSTAIN")
        return m


def vec_of(text):
    return [1.0, 0.0] if "alpha" in text else ([-1.0, 0.0] if "beta" in text else [0.0, 0.0])


def preload(db, ids):
    """Stored vectors (model "fake", sha of the text) for the given span ids, as an earlier pass would have left them."""
    for sid, _src, text in db.spans:
        if sid in ids:
            db.vectors[sid] = ("fake", hashlib.sha256(text.encode("utf-8")).hexdigest(), vec_of(text))


def mk(db, e, core=None, **jobkw):
    j = job.Job({"enabled": True, "params": {}, "display_gate": {}, "embedder": "x", "job": dict(jobkw)}, core=core or StubCore(),
                embedder=e, engine_factory=factory_for(db), env=dict(ENV), rss_reader=lambda: 100.0)
    return j


def linked(db):
    return {a for kind in ("chunk_subtopic", "chunk_topic") for a, _b in db.active(kind)}


def row(db):
    return json.loads(db.runs[-1]["metrics"])["run"]


def build_node(db, origin="induction", ref=PREFIX + "abc", state="active", lvl="topic", slug="n1"):
    db.nodes[(lvl, slug)] = dict(id=900, label="x", origin=origin, origin_ref=ref, state=state)


WRITES = ("node_", "link_insert", "link_retire", "anchor_", "set_ctx", "job_run_insert")


def write_names(db):
    return [n for n in db.names() if n.startswith(WRITES)]


# ------------------------------------------------------------------------------------------------ T44 flag off / not literal true
@pytest.mark.parametrize("kw", [{}, {FLAG: False}, {FLAG: "true"}, {FLAG: 1}, {FLAG: None}])
def test_t44_flag_off_or_not_literally_true_equals_main(kw):
    db = RecDB(spans())
    preload(db, {1, 2, 3, 4, 5, 6})
    j = mk(db, DeferEmb(), **kw)
    r = j.run()
    assert r["state"] == "deferred" and r["retry_in_s"] == 4321.0 and not r["complete"] and "finish_now" not in r
    assert j.status()["reasons"] == ["deferred_until_utc_midnight"]
    assert not write_names(db) and not db.nodes and not db.links
    reasons, metrics = stored(db)
    assert reasons == ["deferred"] and "no_vector_yet" not in metrics and "partial_snapshot" not in metrics


def test_t44_default_false_shipped_and_validator_rejects_non_bool():
    assert ct.load_cfg()["job"].get(FLAG) is not True  # shipped config leaves the key out: off (an existing test pins job == {})
    assert job.JOB_DEFAULTS[FLAG] is False
    assert job.Job({"enabled": True}).finish_now is False  # missing key = off
    assert ct._validate_job({FLAG: True}) == {FLAG: True} and ct._validate_job({FLAG: False}) == {FLAG: False}
    for bad in ("false", 1, None, 0, "true"):
        with pytest.raises(ValueError):
            ct._validate_job({FLAG: bad})


# ------------------------------------------------------------------------------------------------ T45 / T46
@pytest.mark.parametrize("code", ["budget_reached", "quota_reached"])
def test_t45_t46_first_build_on_vectored_spans_once_and_counts_the_rest(code):
    sp = spans(12)
    sp[8] = (9, sp[8][1], "   ")  # span 9: empty, no vector -> skipped_empty, not no_vector_yet
    db = RecDB(sp)
    preload(db, {1, 2, 6, 7, 12})  # page 1 holds stored vectors and needs embedding (3, 4); later pages hold more stored vectors
    core, e = SpyCore(), DeferEmb(code=code)
    j = mk(db, e, core=core, page_size=4, **{FLAG: True})
    j._quarantine[11] = (hashlib.sha256(sp[10][2].encode("utf-8")).hexdigest(), "http_400")  # span 11: quarantined, no vector
    r = j.run()
    assert r["state"] == "complete" and r["finish_now"] is True and r["retry_in_s"] == 4321.0
    assert core.built == [[1, 2, 6, 7, 12]]  # each vectored span in the build input exactly once, none of the others
    assert linked(db) == {1, 2, 6, 7, 12} and {3, 4, 5, 8, 9, 10, 11}.isdisjoint(db.vectors)
    assert {n for n in write_names(db) if n.startswith("link_insert_chunk")} <= {"link_insert_chunk_subtopic", "link_insert_chunk_topic"}
    m = row(db)
    assert m["no_vector_yet"] == 5 and m["no_vector_yet_min_span_id"] == 3 and m["partial_snapshot"] == 1  # 3, 4, 5, 8, 10; not empty 9, not quarantined 11
    assert m["skipped_empty"] == 1 and m["quarantined"] == 1
    assert e.attempts == 1  # zero embed calls after the budget stop; zero neurons spent
    assert db.run_rows[-1]["neurons"] == 0 and db.run_rows[-1]["state"] == "finished"


def test_t45_partial_batch_before_the_stop_is_kept_once():
    db = RecDB(spans())
    preload(db, {1, 2, 3, 4, 5, 6})
    core, e = SpyCore(), DeferEmb(ok_calls=1)
    j = mk(db, e, core=core, embed_batch=2, **{FLAG: True})
    r = j.run()
    assert r["finish_now"] is True and r["rows_embedded"] == 2
    assert core.built == [[1, 2, 3, 4, 5, 6, 7, 8]] and set(db.vectors) == set(range(1, 9))
    assert row(db)["no_vector_yet"] == 2 and row(db)["no_vector_yet_min_span_id"] == 9


# ------------------------------------------------------------------------------------------------ T47 build present = flag off
@pytest.mark.parametrize("seed", [True])
def test_t47_active_build_present_equals_flag_false(seed):
    out = []
    for flag in (False, True):
        db = RecDB(spans())
        preload(db, {1, 2, 3})
        build_node(db)
        j = mk(db, DeferEmb(), **{FLAG: flag})
        r = j.run()
        out.append((r["state"], j.status()["reasons"], [n for n in db.names() if not n.startswith("nodes_")], stored(db)))  # nodes_<lvl> = the read-only active-build probe
    assert out[0] == out[1] and out[0][0] == "deferred"
    assert not [n for n in out[0][2] if n.startswith(("link_insert", "node_insert"))]


@pytest.mark.parametrize("kw", [dict(origin="curator"), dict(origin="registry"), dict(state="proposed"), dict(ref="seed:other")])
def test_t47_non_induction_or_inactive_nodes_count_as_no_build(kw):
    db = RecDB(spans())
    preload(db, {1, 2, 3, 4, 5, 6})
    build_node(db, **kw)
    r = mk(db, DeferEmb(), **{FLAG: True}).run()
    assert r["finish_now"] is True and linked(db) == {1, 2, 3, 4, 5, 6}


# ------------------------------------------------------------------------------------------------ T48 complete snapshot
def test_t48_complete_snapshot_same_links_and_partial_snapshot_key():
    res = {}
    for flag in (False, True):
        db = RecDB(spans())
        mk(db, FakeEmb(), **{FLAG: flag}).run()
        res[flag] = (db, sorted(db.links.values(), key=repr) and sorted(((k[0], k[1], k[2]) for k in db.links)), row(db))
    assert res[False][1] == res[True][1] and res[True][1]
    assert res[True][2]["partial_snapshot"] == 0 and "partial_snapshot" not in res[False][2]
    assert "no_vector_yet" not in res[True][2]


# T49 (single begin() transaction around the whole write) is retired by WP-W (CHUNK-ASSOC-1113 A3): D1 commits per phase on purpose;
# crash behaviour is covered by tests/test_chunk_taxo_wpw_1113.py (T1-T3).

# ------------------------------------------------------------------------------------------------ T50 negative probes
def test_t50_portal_abstain_and_empty_spans_get_no_link_and_values_are_numbers():
    db = mixed_db()
    db.__class__ = RecDB
    db.closes, db.raise_map = [], {}
    preload(db, {1, 3})  # ok spans 1, 3, 5; span 5 has no vector; spans 2, 4, 6, 7 are portal spans (span 2 holds a stale vector)
    preload(db, {2})
    j = mk(db, DeferEmb(), core=AbstainCore(), page_size=3, **{FLAG: True})
    assert j.run()["finish_now"] is True
    assert linked(db) == {1, 3}  # portal span 2 (vectored) gets none; span 5 has no vector
    m = row(db)
    assert m["skipped_portal"] == 4 and m["no_vector_yet"] == 1 and m["no_vector_yet_min_span_id"] == 5

    def nums(x):
        return all(nums(v) for v in x.values()) if isinstance(x, dict) else isinstance(x, (int, float)) and not isinstance(x, bool) or isinstance(x, bool)
    assert nums(m)


def test_t50_vectored_span_the_build_abstains_on_gets_no_link():
    db = RecDB(spans())
    preload(db, {1, 2, 3, 4, 5, 6})
    mk(db, DeferEmb(), core=AbstainCore(), **{FLAG: True}).run()
    assert 4 not in linked(db) and {1, 2, 3, 5, 6} <= linked(db)


# ------------------------------------------------------------------------------------------------ T50a other stops behave as main
def _clock():
    c = count(1)
    return lambda: next(c) * 1000.0


STOPS = {
    "max_rows": lambda: dict(e=FakeEmb(), kw=dict(max_rows=1, embed_batch=2)),
    "rss_hard": lambda: dict(e=FakeEmb(), kw=dict(rss_hard_mb=50)),
    "rss_build": lambda: dict(e=FakeEmb(), kw=dict(rss_hard_mb=100.0001)),
    "embed_error": lambda: dict(e=RaisingEmb(RuntimeError("embedder died")), kw={}),
    "dim_mismatch": lambda: dict(e=MixedDimEmb(), kw={}),
    "max_seconds": lambda: dict(e=FakeEmb(), kw={}, clock=_clock()),
}


@pytest.mark.parametrize("name", sorted(STOPS))
def test_t50a_non_budget_stops_give_no_build_and_equal_flag_false(name):
    out = []
    for flag in (False, True):
        spec = STOPS[name]()
        db = RecDB(spans())
        j = job.Job({"enabled": True, "params": {}, "display_gate": {}, "embedder": "x", "job": {FLAG: flag, **spec["kw"]}}, core=StubCore(),
                    embedder=spec["e"], engine_factory=factory_for(db), env=dict(ENV), rss_reader=lambda: 100.0,
                    **({"clock": spec["clock"]} if "clock" in spec else {}))
        r = j.run()
        out.append((r["state"], r.get("stop"), j.status()["reasons"], db.names()))
        assert not write_names(db) and not db.links and not db.nodes and "finish_now" not in r
    assert out[0] == out[1] and out[0][0] != "complete"


def test_t50a_budget_stop_without_embedder_progress_codes_other_than_budget_stay_as_main():
    db = RecDB(spans())
    preload(db, {1, 2, 3})
    j = mk(db, DeferEmb(code="other_code"), **{FLAG: True})
    r = j.run()
    assert r["state"] == "deferred" and not write_names(db) and "finish_now" not in r


# ------------------------------------------------------------------------------------------------ empty vector set, later pass, loop
def test_flag_on_empty_vector_set_builds_nothing_and_says_why():
    db = RecDB(spans())
    j = mk(db, DeferEmb(), **{FLAG: True})
    r = j.run()
    assert r["state"] == "deferred" and not r["complete"]
    assert j.status()["reasons"] == ["deferred_until_utc_midnight", "no_vectors"]
    assert not write_names(db) and not db.nodes
    reasons, metrics = stored(db)
    assert reasons == ["no_vectors"] and metrics["no_vector_yet"] == 10 and metrics["no_vector_yet_min_span_id"] == 1


def test_later_pass_embeds_and_assigns_the_unvectored_chunks():
    db = RecDB(spans())
    preload(db, {1, 2, 3, 4, 5, 6})
    mk(db, DeferEmb(), **{FLAG: True}).run()
    assert linked(db) == {1, 2, 3, 4, 5, 6}
    r = mk(db, FakeEmb(), **{FLAG: True}).run()  # budget back: the normal path on a complete snapshot, now a rebuild
    assert r["state"] == "complete" and "finish_now" not in r and linked(db) == set(range(1, 11))
    assert row(db)["partial_snapshot"] == 0


def test_loop_waits_for_budget_reset_after_finish_now_but_backs_off_on_partial_or_error():
    waits = []
    job._STOP.clear()
    runs = iter([{"state": "complete", "finish_now": True, "retry_in_s": 5000.0},
                 {"state": "partial", "finish_now": True, "retry_in_s": 5000.0}, {"state": "inert"}])
    job.loop(SimpleNamespace(jcfg={"interval_s": 3600, "initial_delay_s": 5}, run=runs.__next__), sleep=lambda w: waits.append(w) or False)
    assert waits == [5, 5000.0, 5]
