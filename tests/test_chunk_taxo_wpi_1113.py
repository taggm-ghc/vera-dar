"""WP-I (CHUNK-ASSOC-1113 D2, D3, D5 and the job-side D6 counts; PLAN T7-T13, T20, T24-T26, T28-T30, T34, T36, T38, T40, T42).
Incremental assignment ships OFF: the mode `incremental` is off | shadow | on. Offline: the transactional fake DB with per-commit snapshots
and the held-connection engine of tests/test_chunk_taxo_wpw_1113.py (imported, not edited), a core whose `associate` is a real
nearest-center rule over the centers the job rebuilds from the stored strict members, deterministic 2-d vectors, reserved fixture values.
Control flow only: the SQL (the new statements) and the guard trigger stay "unverified" until the owner-run WP-V rehearsal. No DB, no network."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_arxiv_defense_1113 import PORTAL_SOURCES
from tests.test_chunk_taxo_job_1109 import FakeEmb, StubCore, spans
from tests.test_chunk_taxo_wpw_1113 import WDB, KeyedCore, Locks, mk, visible
from vera import chunk_taxo_embed as emb
from vera import chunk_taxo_job as job

VEC = {"alpha": [1.0, 0.2], "beta": [-1.0, 0.2], "gamma": [0.0, 1.0]}


class VEmb(FakeEmb):
    """Deterministic 2-d vectors by keyword (alpha, beta, gamma); `name` is the model id the job stores."""
    name = "fake"

    def embed(self, texts):
        self.calls.append(len(texts))
        return [next((v for k, v in VEC.items() if k in t), [0.5, 0.5]) for t in texts]


class OtherEmb(VEmb):
    name = "other-model"


class BudgetEmb(VEmb):
    def embed(self, texts):
        raise emb.Deferred("budget_reached", 4321.0)


class AssocCore(StubCore):
    """StubCore plus stored cutoffs and a nearest-center `associate` (the contract of vera.chunk_taxo.associate on a model-like object)."""

    def __init__(self, min_support=2, sub_dq=0.3):
        self.min_support, self.sub_dq, self.seen = min_support, sub_dq, []

    def build(self, chunks, vecs, params, prior=None):
        m = super().build(chunks, vecs, params, prior=prior)
        m.null = {"subtopic": {"dist_q": self.sub_dq, "margin_q": None}, "topic": {"dist_q": 0.6, "margin_q": 0.05},
                  "domain": {"dist_q": 0.95, "margin_q": None}}
        m.params = {"min_support": self.min_support}
        return m

    def associate(self, model, vec):
        self.seen.append(model)
        v = np.asarray(vec, dtype=np.float64)
        v = v / np.linalg.norm(v)
        by_lvl = {}
        for n in model.nodes:
            by_lvl.setdefault(n["level"], []).append(n)
        for lv in ("subtopic", "topic", "domain"):
            nodes = by_lvl.get(lv, [])
            if not nodes:
                continue
            sims = [float(np.dot(v, np.asarray(n["center"]))) for n in nodes]
            i = int(np.argmax(sims))
            n = nodes[i]
            if 1.0 - sims[i] <= model.null[lv]["dist_q"] and len(n["members"]) >= model.params["min_support"]:
                top = n["key"] if lv == "topic" else next((x["parent_key"] for x in nodes if x["key"] == n["key"]), None)
                return {"subtopic": n["key"] if lv == "subtopic" else None, "topic": top if lv != "domain" else None,
                        "domain": n["key"] if lv == "domain" else None, "level": lv, "distance": round(1.0 - sims[i], 6),
                        "margin": None, "reason": None}
        return {"subtopic": None, "topic": None, "domain": None, "level": None, "distance": 0.9, "margin": None, "reason": "NO_FIT"}


def sp(i):
    return (i, 100 + i % 3, f"{['alpha', 'beta', 'gamma'][i % 3]} WPI {i}")


def build_db(n=12, core=None, **kw):
    db = WDB(spans(n), locks=Locks())
    j = mk(db, e=VEmb(), core=core or AssocCore(), **kw)
    assert j.run()["state"] == "complete"
    return db


def add_spans(db, ids):
    db.spans = db.spans + [sp(i) for i in ids]
    db.snaps, db.tx_names, db.log, db._mark = [], [], [], 0


def linked(db, kinds=("chunk_subtopic", "chunk_topic")):
    return {k[1] for k, v in db.links.items() if k[0] in kinds and v["state"] == "active" and v["score_kind"] in (None, "cosine")}


def last_close(db):
    return json.loads(db.closes[-1]["metrics"]), db.closes[-1]["reasons"]


# ------------------------------------------------------------------------------------------------ T8, T9, T13
def test_t8_t9_new_spans_get_links_without_a_rebuild_and_linked_spans_are_never_written():
    core = AssocCore()
    db = build_db(core=core)
    before = {k: dict(v) for k, v in db.links.items()}
    add_spans(db, [13, 14, 15, 16])
    e = VEmb()
    j = mk(db, e=e, core=core, incremental="on")
    r = j.run()
    assert r["complete"] and j.status()["state"] == "incremental"
    names = db.names()
    assert not [n for n in names if n.startswith(("node_", "anchor_", "job_run_insert"))] and "link_insert_domain_topic" not in names
    assert {13, 14, 15, 16} <= linked(db)  # the new spans were assigned on the last build's centers
    for k, v in before.items():  # T9: no previously linked span's row was touched
        assert db.links[k] == v
    m, reasons = last_close(db)
    assert reasons == ["incremental"] and m["run"]["inc_targets"] == 4 and m["run"]["inc_assigned"] == 4
    assert len(db.runs) == 1  # no new build row


def test_t9_a_linked_span_re_embedded_after_the_build_is_not_a_target_and_its_rows_stay_untouched():
    core = AssocCore()
    db = build_db(core=core)
    before = {k: dict(v) for k, v in db.links.items()}
    db.vec_after_ids = {3, 4}  # spans 3 and 4 hold active strict links and a vector written after the build's scan end (text changed)
    j = mk(db, e=VEmb(), core=core, incremental="on")
    j.run()
    assert db.links == before  # the guard `no active strict link` kept them out of the target set: no retire, no insert
    assert last_close(db)[0]["run"]["inc_targets"] == 0


def test_a1_incremental_pages_set_the_022_guard_context_before_the_first_link_write():
    core = AssocCore()
    db = build_db(core=core)
    add_spans(db, [13])
    db.ctx = None
    mk(db, e=VEmb(), core=core, incremental="on").run()
    names = db.names()
    assert db.ctx is not None and db.ctx[1]  # vera.tax_actor / vera.tax_batch were set (batch = sha of the active version)
    assert names.index("set_ctx") < names.index("link_insert_chunk_subtopic")
    import hashlib
    assert db.ctx[1] == hashlib.sha256(("auto-chunk-taxo:" + db.runs[0]["version"]).encode()).hexdigest()


def test_t13_an_incremental_pass_with_no_new_spans_spends_zero_neurons_and_writes_nothing():
    core = AssocCore()
    db = build_db(core=core)
    db.log = []
    e = VEmb()
    j = mk(db, e=e, core=core, incremental="on")
    assert j.run()["complete"]
    assert e.calls == [] and db.run_rows[-1]["neurons"] == 0
    assert not [n for n in db.names() if n.startswith(("node_", "link_insert", "link_retire", "job_run_insert"))]
    assert last_close(db)[0]["run"]["inc_targets"] == 0


# ------------------------------------------------------------------------------------------------ T10 three modes
def test_t10_off_rebuilds_shadow_rebuilds_and_records_agreement_on_assigns():
    out = {}
    for mode in ("off", "shadow", "on"):
        db = build_db(core=AssocCore())
        add_spans(db, [13, 14, 15])
        j = mk(db, e=VEmb(), core=AssocCore(), incremental=mode)
        assert j.run()["complete"]
        out[mode] = db
    assert len(out["off"].runs) == 2 and len(out["shadow"].runs) == 2 and len(out["on"].runs) == 1  # off, shadow: a new build row
    assert out["off"].names().count("link_insert_chunk_subtopic") == out["shadow"].names().count("link_insert_chunk_subtopic")
    sm = json.loads(out["shadow"].runs[-1]["metrics"])["run"]
    assert sm["shadow_targets"] == 3 and sm["shadow_compared"] >= 1 and "shadow_targets" not in json.loads(out["off"].runs[-1]["metrics"])["run"]
    assert last_close(out["on"])[1] == ["incremental"]


def test_t10_shadow_computes_in_memory_only_no_link_write_before_the_rebuild():
    db = build_db(core=AssocCore())
    add_spans(db, [13, 14])
    mk(db, e=VEmb(), core=AssocCore(), incremental="shadow").run()
    names = db.names()
    w1 = min(names.index(x) for x in ("link_retire_pair_other_domain_topic", "link_insert_domain_topic") if x in names)  # the rebuild W1 hierarchy write
    pre = names[:w1]
    assert not [n for n in pre if n.startswith(("link_insert", "link_retire", "anchor_", "job_run_insert"))] and "advisory_lock" in pre  # nothing written before the rebuild


def test_t10_extended_incomplete_snapshot_with_a_build_off_and_shadow_write_nothing_on_assigns():
    res = {}
    for mode in ("off", "shadow", "on"):
        core = AssocCore()
        db = build_db(core=core)
        add_spans(db, [13, 14, 15, 16])
        db.vectors[15] = ("fake", __import__("hashlib").sha256(sp(15)[2].encode()).hexdigest(), VEC["alpha"] if "alpha" in sp(15)[2] else VEC["gamma"])
        db.log = []
        j = mk(db, e=BudgetEmb(), core=core, incremental=mode)
        r = j.run()
        res[mode] = (db, r, j)
    for mode in ("off", "shadow"):
        db, r, j = res[mode]
        assert r["state"] == "deferred" and r["retry_in_s"] == 4321.0
        assert not [n for n in db.names() if n.startswith(("link_insert", "node_", "job_run_insert"))]
        assert json.loads(db.closes[-1]["metrics"])["no_rebuild_snapshot_incomplete"] == 1
    db, r, j = res["on"]
    assert r["state"] == "deferred" and 15 in linked(db) and not ({13, 14, 16} & linked(db))  # T7: only the stored vector is assigned
    m, reasons = last_close(db)
    assert reasons == ["incremental"] and m["run"]["no_vector_yet"] == 3 and m["run"]["scan_complete"] == 1


# ------------------------------------------------------------------------------------------------ T7 centers use every stored vector
def test_t7_after_a_budget_stop_the_scan_reaches_the_end_and_centers_use_every_stored_vector():
    core = AssocCore()
    db = build_db(core=core)
    add_spans(db, [13, 14, 15, 16, 17, 18])
    for i in (16, 17, 18):  # vectors that sit on pages AFTER the span where the budget stops embedding
        text = sp(i)[2]
        key = next(k for k in VEC if k in text)
        db.vectors[i] = ("fake", __import__("hashlib").sha256(text.encode()).hexdigest(), VEC[key])
    j = mk(db, e=BudgetEmb(), core=core, incremental="on", page_size=2)
    j.run()
    model = core.seen[-1]
    s0_members = {s for k, s in ((k, k[1]) for k, v in db.links.items() if k[0] == "chunk_subtopic" and v["state"] == "active")}
    assert model.nodes and all(len(n["members"]) >= 1 for n in model.nodes)
    assert {16, 17, 18} <= linked(db)  # the stored vectors beyond the stop were read and assigned
    assert last_close(db)[0]["run"]["no_vector_yet"] == 3 and s0_members


# ------------------------------------------------------------------------------------------------ T11 identity, routing rows
def test_t11_routing_rows_do_not_move_prior_members_or_node_identity_on_rebuild():
    db = build_db(core=AssocCore())
    ids_before = {k: v["id"] for k, v in db.nodes.items() if v["state"] == "active"}
    some = next(k for k in db.nodes if k[0] == "subtopic")
    sid = db.nodes[some]["id"]
    for span in (1, 2, 3):  # same-version routing rows (generous links and a soft outlier) on other nodes
        db.links[("chunk_subtopic", span, sid, 900 + span)] = dict(version="x", state="active", score=0.4, score_kind="cosine_r2")
    j = mk(db, e=VEmb(), core=AssocCore())
    j._chunks = [{"span_id": i, "source_id": 100, "text": "x"} for i in range(1, 13)]
    prior = j._load_prior()
    strict_members = {k[1] for k, v in db.links.items() if k[0] in ("chunk_subtopic", "chunk_topic") and v["state"] == "active" and v["score_kind"] == "cosine"
                      and k[2] == sid and k[0] == "chunk_subtopic"}
    got = {n["key"]: set(n["members"]) for n in prior["nodes"]}
    assert got[some[1]] == strict_members  # routing rows (cosine_r2) are not members
    add_spans(db, [13])
    mk(db, e=VEmb(), core=AssocCore()).run()
    assert {k: v["id"] for k, v in db.nodes.items() if v["state"] == "active"} == ids_before


# ------------------------------------------------------------------------------------------------ T12, T40
def test_t12_t40_portal_empty_and_vectored_portal_spans_get_no_link_and_are_counted_once():
    core = AssocCore()
    db = build_db(core=core)
    db.sources = {s: ("https://doi.org/10.1000/example.%d" % s, {"document_id": "acq-%d" % s}) for s in (100, 101, 102)}
    db.sources[101] = PORTAL_SOURCES[201]  # source 101 becomes a portal source
    add_spans(db, [13, 14, 15, 16])
    db.spans[-1] = (16, 100, "   ")  # an empty span
    j = mk(db, e=VEmb(), core=core, incremental="on")
    j.run()
    portal_new = {i for i, src, _t in db.spans if src == 101 and i >= 13}
    assert not (portal_new & linked(db)) and 16 not in linked(db)
    run = last_close(db)[0]["run"]
    n_portal = sum(1 for i, src, _t in db.spans if src == 101)
    assert run["skipped_portal"] == n_portal and run["skipped_empty"] == 1
    # the old portal spans still hold stored vectors: counted once, as portal (never above skipped_portal)
    assert run["excluded_with_vector_portal"] == len([i for i, src, _t in db.spans if src == 101 and i in db.vectors])
    assert run["excluded_with_vector_portal"] <= run["skipped_portal"] and run["scan_complete"] == 1


# ------------------------------------------------------------------------------------------------ T20 shadow agreement
def test_t20_shadow_records_agreement_after_identity_matching():
    db = build_db(core=AssocCore())
    add_spans(db, [13, 14, 15, 16])
    mk(db, e=VEmb(), core=AssocCore(), incremental="shadow").run()
    m = json.loads(db.runs[-1]["metrics"])["run"]
    assert m["shadow_compared"] == m["shadow_agree"] and m["shadow_compared"] >= 3  # StubCore reuses every node: same node by identity
    db2 = build_db(core=AssocCore())
    add_spans(db2, [13, 14, 15, 16])
    mk(db2, e=VEmb(), core=KeyedCoreAssoc("z"), incremental="shadow").run()
    m2 = json.loads(db2.runs[-1]["metrics"])["run"]
    assert m2["shadow_compared"] >= 3 and m2["shadow_agree"] == 0  # nothing reused: no shadow node matches a rebuilt node


class KeyedCoreAssoc(KeyedCore, AssocCore):
    def __init__(self, tag):
        KeyedCore.__init__(self, tag)
        self.min_support, self.sub_dq, self.seen = 2, 0.3, []


# ------------------------------------------------------------------------------------------------ T24, T34
def test_t24_a_node_below_min_support_is_not_a_target_the_chunk_backs_off():
    core = AssocCore(min_support=5)  # a sub-topic needs 5 strict members; each sub-topic of the first build has 4
    db = WDB(spans(12), locks=Locks())
    mk(db, e=VEmb(), core=core).run()
    add_spans(db, [13, 14])  # 13 is a beta span (its sub-topic has 4 members, below 5), 14 a gamma span (a topic-level direction)
    assert "beta" in sp(13)[2] and "gamma" in sp(14)[2]
    mk(db, e=VEmb(), core=core, incremental="on").run()
    kinds = lambda s: sorted(k[0] for k, v in db.links.items() if k[1] == s and v["state"] == "active")  # noqa: E731
    assert "chunk_subtopic" not in kinds(13) and "chunk_subtopic" not in kinds(14)  # never a sub-topic row for a node below min_support
    assert kinds(14) == ["chunk_topic"]  # the chunk backs off to the topic level


def test_t34_only_a_new_span_is_a_target_a_build_abstained_span_with_a_vector_is_not_and_two_node_levels_are_assignable():
    core = AssocCore()
    db = build_db(core=core)
    # span 99: has a vector and no link, id above max_span_id: a target. span 5's link is dropped: id below max, vector before scan end: not a target
    add_spans(db, [99])
    for k in [k for k in db.links if k[0].startswith("chunk_") and k[1] == 5]:
        db.links[k]["state"] = "retired"
    j = mk(db, e=VEmb(), core=core, incremental="on")
    j.run()
    assert 99 in linked(db) and 5 not in linked(db)
    build = json.loads(db.runs[0]["metrics"])
    assert build["has_margin_subtopic"] == 0 and build["has_margin_topic"] == 1 and "margin_q_subtopic" not in build  # presence flag, not absence
    assert build["dist_q_subtopic"] == 0.3 and build["min_support"] == 2


# ------------------------------------------------------------------------------------------------ T25
def test_t25_no_build_params_and_write_incomplete_block_assignment_an_incremental_row_does_not():
    core = AssocCore()
    db = WDB(spans(6), locks=Locks())
    j = mk(db, e=VEmb(), core=core, incremental="on")
    assert j.run()["state"] == "complete" and len(db.runs) == 1  # no build row yet: the first pass is a build, not an assignment
    assert json.loads(db.closes[-1]["metrics"]) is not None
    db = build_db(core=core)
    add_spans(db, [13])
    mk(db, e=VEmb(), core=core, incremental="on").run()  # an incremental run row now sits after the build row
    add_spans(db, [14])
    j2 = mk(db, e=VEmb(), core=core, incremental="on")
    assert j2.run()["state"] != "error" and 14 in linked(db) and len(db.runs) == 1  # the incremental row did not hide the build row
    # write_incomplete: a closed row of an aborted pass after the build blocks assignment ...
    db.rec.append(("failed", json.dumps({"write_incomplete": 1})))
    add_spans(db, [15])
    j3 = mk(db, e=VEmb(), core=core, incremental="on")
    j3._chunks = []
    r, d = j3._inc_inputs()
    assert r == "write_incomplete" and d is None
    db.rec.pop()
    # ... and so do active nodes beyond the build's own (a pass killed after W1 leaves only a running row)
    db.nodes[("topic", "ct-t-stray")] = dict(id=997, label="s", origin="induction", origin_ref="auto-chunk-taxo:zz", state="active")
    r2, d2 = mk(db, e=VEmb(), core=core, incremental="on")._inc_inputs()
    assert r2 == "write_incomplete" and d2 is None


# ------------------------------------------------------------------------------------------------ T26 wake
def test_t26_many_wakes_inside_the_debounce_window_give_one_early_pass():
    job._JOB = SimpleNamespace()
    job._WAKE.clear()
    t = [0.0]
    waits = []

    def stop_wait(s):
        waits.append(s)
        t[0] += s
        if len(waits) == 2:  # three wakes arrive while the loop waits, all inside the window
            job.wake(), job.wake(), job.wake()
        return False

    try:
        stopped = job._wait_or_wake(3600.0, 0.0, 600.0, clock=lambda: t[0], stop_wait=stop_wait, slice_s=100.0)
    finally:
        job._JOB = None
    assert stopped is False and 599.0 < t[0] <= 700.0  # woke once, at the window end (600 s after the pass began), not at 3600 s
    assert not job._WAKE.is_set()


def test_t26_a_wake_never_changes_the_shipped_waits_only_a_normal_wait_of_a_job_with_the_mode_not_off():
    assert job._wake_allowed(SimpleNamespace(incremental="off"), {"state": "complete"}) is False  # shipping D5 with the mode off: no change
    assert job._wake_allowed(SimpleNamespace(incremental="shadow"), {"state": "complete"}) is True
    for res in ({"state": "deferred"}, {"state": "error"}, {"state": "partial"}, {"state": "complete", "finish_now": True}):
        assert job._wake_allowed(SimpleNamespace(incremental="on"), res) is False  # never cuts a budget-deferral or backoff wait
    job._JOB = SimpleNamespace()
    job._WAKE.clear()
    t = [0.0]

    def stop_wait(s):
        t[0] += s
        job.wake()
        return False

    try:
        job._wait_or_wake(80000.0, 0.0, 3600.0, clock=lambda: t[0], stop_wait=stop_wait, slice_s=1000.0, allow_wake=False)
        assert t[0] == 80000.0  # a wake during a deferral wait did not end it (the probe's 3600 s case)
        t[0] = 0.0
        job._wait_or_wake(80000.0, 0.0, 3600.0, clock=lambda: t[0], stop_wait=stop_wait, slice_s=1000.0, allow_wake=True)
        assert t[0] < 80000.0  # while with the permission the same wake ends it at the window end
    finally:
        job._JOB = None
        job._WAKE.clear()


def test_t26_wake_is_inert_without_a_running_job_and_the_ingest_hand_off_keeps_notify_ingested():
    job._JOB = None
    job._WAKE.clear()
    job.wake()
    assert not job._WAKE.is_set()
    src = open(__import__("vera.auto_acquire", fromlist=["x"]).__file__, encoding="utf-8").read()
    a, b = src.index("topic_tagging.notify_ingested(sid)"), src.index("chunk_taxo_wake.wake()")
    assert a < b and "except Exception" in src[a:b]  # both calls kept; the wake sits after, in its own never-raise block
    assert job._wake_min({}, 3600.0) == 3600.0 and job._wake_min({"wake_min_interval_s": 90}, 3600.0) == 90.0  # default = interval_s


# ------------------------------------------------------------------------------------------------ T28
def test_t28_a_same_version_soft_routing_row_on_a_target_pair_is_retired_before_the_strict_insert():
    core = AssocCore()
    db = build_db(core=core)
    version = db.runs[0]["version"]
    add_spans(db, [15])  # 15 % 3 == 0 -> alpha
    s0 = next(v["id"] for k, v in db.nodes.items() if k[0] == "subtopic" and v["state"] == "active"
              and any(kk[0] == "chunk_subtopic" and kk[2] == v["id"] and kk[1] == 3 for kk in db.links))
    db.links[("chunk_subtopic", 15, s0, 700)] = dict(version=version, state="active", score=0.5, score_kind="cosine_soft")
    mk(db, e=VEmb(), core=core, incremental="on").run()
    rows = [(k, v) for k, v in db.links.items() if k[0] == "chunk_subtopic" and k[1] == 15 and k[2] == s0]
    states = {v["score_kind"]: v["state"] for k, v in rows}
    assert states.get("cosine_soft") == "retired" and states.get("cosine") == "active"
    names = db.names()
    assert "link_retire_pair_routing_chunk_subtopic" in names
    assert not [n for n in names if n.startswith("link_retire_pair_other_")]  # the W1-W3 by-pair statement is not used here


# ------------------------------------------------------------------------------------------------ T36, T29, T30
def test_t29_t36_t30_targets_are_new_spans_or_spans_with_a_vector_written_after_the_build_scan_end():
    core = AssocCore()
    db = build_db(core=core)
    for k in [k for k in db.links if k[0].startswith("chunk_") and k[1] in (4, 5)]:
        db.links[k]["state"] = "retired"
    db.vec_after_ids = {5}  # span 5 was re-embedded after the build's scan end (a late lower id): a target; span 4 abstained: not
    j = mk(db, e=VEmb(), core=core, incremental="on")
    j.run()
    assert 5 in linked(db) and 4 not in linked(db)


# ------------------------------------------------------------------------------------------------ T38
def test_t38_topic_and_domain_centers_include_their_descendants_strict_members():
    core = AssocCore()
    db = build_db(core=core)
    add_spans(db, [13])
    j = mk(db, e=VEmb(), core=core, incremental="on")
    j.run()
    model = core.seen[-1]
    by = {n["key"]: n for n in model.nodes}
    sub_members = {s for n in model.nodes if n["level"] == "subtopic" for s in n["members"]}
    topic = next(n for n in model.nodes if n["level"] == "topic")
    domain = next(n for n in model.nodes if n["level"] == "domain")
    assert sub_members <= set(topic["members"]) and set(topic["members"]) <= set(domain["members"])
    vecs = {i: np.asarray(VEC[next(k for k in VEC if k in t)]) for i, _s, t in db.spans}
    mean = np.mean([vecs[s] for s in topic["members"]], axis=0)
    assert np.allclose(np.asarray(topic["center"]), mean / np.linalg.norm(mean), atol=1e-6) and by


# ------------------------------------------------------------------------------------------------ T42
def test_t42_after_an_embedder_change_no_rebuild_runs_no_label_is_lost_and_stale_vectors_are_counted():
    for mode in ("off", "on"):
        core = AssocCore()
        db = build_db(core=core)
        before = {s for s in range(1, 13) if visible(db.nodes, db.links, s)}
        add_spans(db, [13, 14])
        j = mk(db, e=OtherEmb.__new__(OtherEmb), core=core, incremental=mode)
        j._emb = OtherEmbBudget()
        r = j.run()
        assert r["state"] == "deferred"
        assert not [n for n in db.names() if n.startswith(("node_", "job_run_insert", "link_retire"))]
        assert all(visible(db.nodes, db.links, s) for s in before)
        run = last_close(db)[0]
        assert run["stale_vector"] == 12 and run["no_vector_yet"] == 2 and run["scan_complete"] == 1


class OtherEmbBudget(OtherEmb):
    def __init__(self):
        super().__init__()

    def embed(self, texts):
        raise emb.Deferred("budget_reached", 4321.0)


# ------------------------------------------------------------------------------------------------ config
def test_incremental_defaults_off_and_config_validation():
    from vera import chunk_taxo as ct
    assert job.Job({"enabled": True}).incremental == "off" and job.Job({"enabled": True, "job": {"incremental": "bogus"}}).incremental == "off"
    assert job.Job({"enabled": True, "job": {"incremental": "shadow"}}).incremental == "shadow"
    assert ct.load_cfg()["job"].get("incremental", "off") == "off"
    assert ct._validate_job({"incremental": "on", "rebuild_interval_s": None, "rebuild_unassigned_share": 0.5, "wake_min_interval_s": 60})
    for bad in ({"incremental": "ON"}, {"incremental": True}, {"rebuild_interval_s": 0}, {"rebuild_unassigned_share": "x"}, {"wake_min_interval_s": 0}):
        with pytest.raises(ValueError):
            ct._validate_job(bad)
    assert job.Job({"enabled": True, "job": {"rebuild_interval_s": 0}}).rebuild_interval_s is None
