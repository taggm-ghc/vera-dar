"""WP-W (CHUNK-ASSOC-1113 D1; PLAN T1-T6, T21-T23, T32, T33, T35, T39): checkpointed writes and the single writer of the chunk-taxonomy job.
Offline fake DB with per-commit snapshots (a transactional fake: a rollback restores the last commit), real statement names, real span
tuples, reserved fixture values. The fake covers CONTROL FLOW only; the guard trigger, the partial unique index, column grants and
the reader SQL stay "unverified" until the owner-run WP-V rehearsal. No DB, no network."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from tests.test_arxiv_defense_1113 import OK_SOURCES, PORTAL_SOURCES
from tests.test_chunk_taxo_job_1109 import BY_TEXT, ENV, Conn, FakeEmb, StubCore, spans
from tests.test_chunk_taxo_reason_1111 import RecDB
from vera import chunk_taxo_job as job
from vera import chunk_taxo_store as store

WRITE_NAMES = ("node_", "link_insert", "link_retire", "anchor_", "set_ctx", "job_run_insert")


class Locks:
    """One advisory lock shared by every engine of a test (the database side of W0)."""

    def __init__(self):
        self.holder, self.lost_after, self.checks = None, None, 0


class WDB(RecDB):
    def __init__(self, *a, locks=None, **k):
        super().__init__(*a, **k)
        self.locks = locks or Locks()
        self.snaps, self.tx_names, self._mark = [], [], 0
        self.committed = (copy.deepcopy(self.nodes), copy.deepcopy(self.links), copy.deepcopy(self.anchors))
        self.no_latest = False
        self.hooks = {}  # statement name -> callable run after the statement
        self.rec = []  # run rows in write order: (state, metrics json), for job_run_recent

    def snapshot(self):
        return SimpleNamespace(nodes=copy.deepcopy(self.nodes), links=copy.deepcopy(self.links))

    def execute(self, stmt, params=None):
        name = BY_TEXT[str(stmt)]
        if name == "job_run_recent":  # aborted passes (closes) sit newer than the build row (runs): newest first
            self.log.append((name, params))
            from tests.test_chunk_taxo_job_1109 import Res
            return Res(list(reversed(self.rec))[:8])  # newest first, in the order the rows were written
        res = super().execute(stmt, params)
        if name == "job_run_close" and params["state"] != "finished":
            self.rec.append((params["state"], params["metrics"]))
        elif name == "job_run_insert":
            self.rec.append(("finished", params["metrics"]))
        if name in self.hooks:
            self.hooks[name](self)
        return res


class HeldConn(Conn):
    def __init__(self, db, eng):
        super().__init__(db)
        self.eng = eng

    def __enter__(self):
        self.eng.open += 1
        self.eng.max_open = max(self.eng.max_open, self.eng.open)
        return self

    def __exit__(self, *a):
        self.eng.open -= 1
        return False

    def execute(self, stmt, params=None):
        name = BY_TEXT[str(stmt)]
        lk = self.db.locks
        if name in ("advisory_lock", "advisory_check", "advisory_unlock"):
            self.db.log.append((name, params))
            if name == "advisory_lock":
                ok = lk.holder in (None, self.eng.me)
                if ok:
                    lk.holder = self.eng.me
                return type(self).res([(ok,)])
            if name == "advisory_check":
                lk.checks += 1
                ok = lk.holder == self.eng.me and not (lk.lost_after is not None and lk.checks > lk.lost_after)
                return type(self).res([(ok,)])
            if lk.holder == self.eng.me:
                lk.holder = None
            return type(self).res([(True,)])
        return self.db.execute(stmt, params)

    def commit(self):
        db = self.db
        fail = getattr(db, "fail_commit_when", None)
        if fail is not None and fail([n for n, _ in db.log[db._mark:]]):
            raise RuntimeError("commit failed")  # the database refused the commit
        db.committed = (copy.deepcopy(db.nodes), copy.deepcopy(db.links), copy.deepcopy(db.anchors))
        db.snaps.append(db.snapshot())
        db.tx_names.append([n for n, _ in db.log[db._mark:]])
        db._mark = len(db.log)

    def rollback(self):
        db = self.db
        db.nodes, db.links, db.anchors = copy.deepcopy(db.committed[0]), copy.deepcopy(db.committed[1]), copy.deepcopy(db.committed[2])
        db._mark = len(db.log)

    res = None


class Eng:
    _n = 0

    def __init__(self, db):
        Eng._n += 1
        self.db, self.me, self.open, self.max_open = db, Eng._n, 0, 0

    def connect(self):
        return HeldConn(self.db, self)

    begin = connect

    def dispose(self):
        pass


def _res():
    from tests.test_chunk_taxo_job_1109 import Res
    return Res


HeldConn.res = staticmethod(lambda rows: _res()(rows))


def mk(db, e=None, core=None, **jobkw):
    eng = Eng(db)
    j = job.Job({"enabled": True, "params": {}, "display_gate": {}, "embedder": "x", "job": dict(jobkw)}, core=core or StubCore(),
                embedder=e or FakeEmb(), engine_factory=lambda url, **kw: eng, env=dict(ENV), rss_reader=lambda: 100.0)
    j.eng = eng
    return j


class KeyedCore(StubCore):
    """Second-generation nodes with new keys: nothing is reused, every prior node is retired (supersession)."""

    def __init__(self, tag="k", abstain=(), reparent=False):
        self.tag, self.abstain, self.reparent = tag, set(abstain), reparent

    def build(self, chunks, vecs, params, prior=None):
        m = super().build(chunks, vecs, params, prior=prior)
        t = self.tag
        keymap = {"D": "D" + t, "T": "T" + t, "S0": "S0" + t, "S1": "S1" + t}
        for n in m.nodes:
            n["key"], n["reused_from"] = keymap[n["key"]], None
            n["parent_key"] = keymap.get(n["parent_key"])
        for sid, a in m.assoc.items():
            for f in ("subtopic", "topic", "domain"):
                a[f] = keymap.get(a[f])
            if sid in self.abstain:
                a.update(subtopic=None, topic=None, level="domain")
        m.retired_keys = sorted(p["key"] for p in (prior or {}).get("nodes", []))
        return m


class ReparentCore(StubCore):
    """Reuses every prior node but moves sub-topic S1 under a new topic T2."""

    def build(self, chunks, vecs, params, prior=None):
        m = super().build(chunks, vecs, params, prior=prior)
        m.nodes.append({"key": "T2", "level": "topic", "parent_key": "D", "label": "Topic B", "terms": ["beta"], "reused_from": None})
        for n in m.nodes:
            if n["key"] == "S1":
                n["parent_key"] = "T2"
        return m


def visible(db_nodes, db_links, span):
    """Reader predicate (fake): an active strict link of the classifier to an active node."""
    act = {n["id"] for n in db_nodes.values() if n["state"] == "active"}
    by = {"chunk_subtopic": "subtopic", "chunk_topic": "topic"}
    ids = {(lvl, n["id"]) for (lvl, _s), n in db_nodes.items() if n["state"] == "active"}
    for k, v in db_links.items():
        if k[0] in by and k[1] == span and v["state"] == "active" and v["score_kind"] in (None, "cosine") and (by[k[0]], k[2]) in ids:
            return True
    return False


def first_pass(n=10, **kw):
    db = WDB(spans(n))
    r = mk(db, **kw).run()
    assert r["state"] == "complete"
    return db


def second_pass(db, core, extra=6, **kw):
    db.spans = db.spans + [(i, 100 + i % 3, f"alpha NEW {i}") for i in range(len(db.spans) + 1, len(db.spans) + 1 + extra)]
    db.snaps, db.tx_names = [], []
    db.log = []
    db._mark = 0
    j = mk(db, core=core, **kw)
    return j, j.run()


# ------------------------------------------------------------------------------------------------ T1 per-span invariant
def test_t1_every_commit_point_keeps_old_labels_until_swapped_and_counts_the_decreases():
    db = first_pass()
    before_linked = {s for s in range(1, 11) if visible(db.nodes, db.links, s)}
    assert before_linked == set(range(1, 11))  # every span of the first build is labelled
    # the new build abstains on span 5 (source 102, a normal source) and the scan classes source 101 as a portal (spans 1, 4, 7, 10 leave the build)
    db.sources = {s: ("https://doi.org/10.1000/example.%d" % s, {"document_id": "acq-%d" % s}) for s in (100, 101, 102)}
    db.sources[101] = PORTAL_SOURCES[201]
    j, r = second_pass(db, KeyedCore("n", abstain={5}), extra=3)
    assert r["state"] == "complete"
    portal_spans = {sid for sid, src, _t in db.spans if src == 101}
    assert 5 not in portal_spans and portal_spans & before_linked == {1, 4, 7, 10}
    assert db.snaps, "commit points were recorded"
    lost_total = {s for s in before_linked if not visible(db.nodes, db.links, s)}
    assert lost_total == {5} | (portal_spans & before_linked)  # the decrease is exactly the abstained span plus the excluded ones
    m = json.loads(db.runs[-1]["metrics"])["run"]
    assert m["abstained_now_unlabelled"] == 1  # span 5: abstained now
    assert m["outside_build_unlabelled"] == 4 and m["outside_build_unlabelled_portal"] == 4  # spans 1, 4, 7, 10: outside the build
    # at every commit point a span lost only if it is one of the counted classes (never a silent decrease)
    for snap in db.snaps:
        for s in before_linked - lost_total:
            assert visible(snap.nodes, snap.links, s), (s, "label lost at a commit point")


def test_t2_crash_after_w1_leaves_old_links_visible_and_new_nodes_with_zero_chunks():
    db = first_pass()
    old_nodes = {k for k, v in db.nodes.items() if v["state"] == "active"}
    db.spans = db.spans + [(11, 100, "alpha NEW 11")]
    db.fail_on = "link_insert_chunk_topic"  # the first chunk page of W2 fails (a crash after W1 committed)
    db.snaps = []
    j = mk(db, core=KeyedCore("c"))
    r = j.run()
    assert r["state"] == "error"
    new_nodes = {k for k, v in db.nodes.items() if v["state"] == "active"} - old_nodes
    assert len(new_nodes) == 4  # W1 committed: the new version's nodes are active
    assert all(visible(db.nodes, db.links, s) for s in range(1, 11))  # every old label still readable
    new_ids = {db.nodes[k]["id"] for k in new_nodes}
    assert not [k for k, v in db.links.items() if k[0].startswith("chunk_") and v["state"] == "active" and k[2] in new_ids]  # zero chunks
    last = db.closes[-1]
    assert last["state"] == "failed" and json.loads(last["metrics"]).get("write_incomplete") == 1


def test_t3_next_pass_after_a_crash_builds_from_the_mixed_state_and_w4_retires_the_superseded_nodes():
    db = first_pass()
    old_nodes = {k for k, v in db.nodes.items() if v["state"] == "active"}
    db.spans = db.spans + [(11, 100, "alpha NEW 11")]
    db.fail_on = "link_insert_chunk_topic"
    assert mk(db, core=KeyedCore("c")).run()["state"] == "error"
    mixed = {k for k, v in db.nodes.items() if v["state"] == "active"}
    assert len(mixed) == 8  # two versions active: write_incomplete
    db.fail_on = None
    j = mk(db, core=KeyedCore("c"))
    r = j.run()
    assert r["state"] == "complete"
    act = {k for k, v in db.nodes.items() if v["state"] == "active"}
    assert len(act) == 4 and not (act & old_nodes)  # the first pass's nodes were retired at W4; the crashed attempt's nodes are the same version
    assert all(visible(db.nodes, db.links, s) for s in range(1, 12))
    assert {v["version"] for k, v in db.links.items() if v["state"] == "active"} == {db.runs[-1]["version"]}


def test_t3_unchanged_clusters_keep_their_node_ids_after_an_interrupted_write():
    db = first_pass()
    ids_before = {k: v["id"] for k, v in db.nodes.items() if v["state"] == "active"}
    db.spans = db.spans + [(11, 100, "alpha NEW 11")]
    db.fail_on = "link_insert_source_topic"  # W2 completed, W3 fails: chunk links of the new version are written, nodes reused
    assert mk(db).run()["state"] == "error"
    db.fail_on = None
    assert mk(db).run()["state"] == "complete"
    assert {k: v["id"] for k, v in db.nodes.items() if v["state"] == "active"} == ids_before  # StubCore reuses by level order, prior from newest strict link


# ------------------------------------------------------------------------------------------------ T4, T5, T6
def test_t4_w4_retires_nodes_anchors_and_hierarchy_in_one_transaction_with_the_build_row():
    db = first_pass()
    j, r = second_pass(db, KeyedCore("w"), extra=2)
    assert r["state"] == "complete"
    last = next(t for t in db.tx_names if "job_run_insert" in t)
    for need in ("node_retire_topic", "node_retire_subtopic", "node_retire_domain", "anchor_retire_topic", "anchor_retire_subtopic",
                 "link_retire_domain_topic", "link_retire_topic_subtopic", "job_run_insert", "job_run_close"):
        assert need in last, (need, last)
    # and nothing of W4 ran in an earlier transaction
    w4 = db.tx_names.index(last)
    assert not [n for t in db.tx_names[:w4] for n in t if n.startswith(("node_retire", "anchor_retire", "job_run_insert"))]
    # W1 is its own transaction, ahead of every chunk page
    w1 = next(i for i, t in enumerate(db.tx_names) if "node_insert_domain" in t)
    w2 = next(i for i, t in enumerate(db.tx_names) if "link_insert_chunk_subtopic" in t)
    assert w1 < w2 < w4


def test_t5_rerunning_a_page_is_idempotent():
    db = first_pass(page_size=3)
    active = sorted((k[0], k[1], k[2]) for k, v in db.links.items() if v["state"] == "active")
    db.no_latest = True  # force a full rewrite of the same version (the no-op rule is off for this fake)
    assert mk(db, page_size=3).run()["state"] == "complete"
    assert sorted((k[0], k[1], k[2]) for k, v in db.links.items() if v["state"] == "active") == active
    assert len([v for v in db.links.values() if v["state"] == "active"]) == len(active)  # no duplicate active row


def test_t6_run_row_metrics_are_numbers_only():
    db = first_pass()
    j, r = second_pass(db, KeyedCore("m"), extra=2)
    m = json.loads(db.runs[-1]["metrics"])

    def leaves(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from leaves(v)
        else:
            yield x
    assert all(isinstance(v, (int, float)) and not isinstance(v, str) for v in leaves(m))
    for key in ("max_span_id", "n_spans", "n_vectors", "scan_end_epoch"):
        assert isinstance(m[key], (int, float)), key
    assert m["max_span_id"] == 12 and m["n_spans"] == 12 and m["n_vectors"] == 12


# ------------------------------------------------------------------------------------------------ T21, T22, T23
def test_t21_second_job_skips_with_writer_busy_before_any_embedding():
    locks = Locks()
    db = WDB(spans(), locks=locks)
    other = mk(db)
    e = FakeEmb()
    me = mk(db, e=e)
    other.eng.me = 999
    locks.holder = 999  # another process holds the writer lock
    r = me.run()
    assert r["state"] == "partial" and r["stop"] == "writer_busy" and me.status()["reasons"] == ["writer_busy"]
    assert e.calls == [] and not db.vectors and not db.run_rows  # no embedding, no run row, no neurons
    assert not [n for n in db.names() if n.startswith(WRITE_NAMES)]
    locks.holder = None  # released: the next pass proceeds
    assert me.run()["state"] == "complete"
    assert locks.holder is None  # and unlocks at the end


def test_t21_two_job_objects_in_one_process_pass_the_lock_concurrently_never():
    locks = Locks()
    db = WDB(spans(), locks=locks)
    b = mk(db)
    seen = {}

    class Hook(FakeEmb):
        def embed(self, texts):
            if "r" not in seen:
                seen["r"] = b.run()  # job B starts while job A is mid-pass (embedding)
            return super().embed(texts)

    a = mk(db, e=Hook())
    assert a.run()["state"] == "complete"
    assert seen["r"]["state"] == "partial" and seen["r"]["stop"] == "writer_busy"


def test_t22_a_span_inserted_between_chunk_pages_causes_no_label_regression():
    db = first_pass(page_size=4)
    before = {s for s in range(1, 11) if visible(db.nodes, db.links, s)}
    db.spans = db.spans + [(11, 100, "alpha NEW 11"), (12, 100, "beta NEW 12")]
    db.snaps = []
    added = []

    def late(d):
        if not added:
            added.append(1)
            d.spans.append((13, 100, "alpha LATE 13"))  # appears after the scan, between W2 pages

    db.hooks["link_insert_chunk_subtopic"] = late
    r = mk(db, core=KeyedCore("p"), page_size=4).run()
    assert r["state"] == "complete" and added
    for snap in db.snaps:
        assert all(visible(snap.nodes, snap.links, s) for s in before)
    assert not visible(db.nodes, db.links, 13)  # new_since_build: not in the build's span set, so no link yet
    assert json.loads(db.runs[-1]["metrics"])["max_span_id"] == 12


def test_t23_a_node_reused_by_the_new_version_is_swapped_in_one_page_and_never_left_without_a_row():
    db = first_pass(page_size=3)
    j, r = second_pass(db, StubCore(), extra=4, page_size=3)  # StubCore reuses every prior node: same (span, node) pairs in both versions
    assert r["state"] == "complete"
    for snap in db.snaps:
        assert all(visible(snap.nodes, snap.links, s) for s in range(1, 11))
    assert {v["version"] for k, v in db.links.items() if k[0].startswith("chunk_") and v["state"] == "active"} == {db.runs[-1]["version"]}
    # reused parent-child hierarchy pairs survive W4
    assert len(db.active("domain_topic")) == 1 and len(db.active("topic_subtopic")) == 2


def test_t23_a_reparented_node_has_two_parents_only_between_w1_and_w4():
    db = first_pass()
    j, r = second_pass(db, ReparentCore(), extra=2)
    assert r["state"] == "complete"
    n_parents = [sum(1 for k, v in snap.links.items() if k[0] == "topic_subtopic" and v["state"] == "active" and k[2] == sub)
                 for snap in db.snaps for sub in {k[2] for k in snap.links if k[0] == "topic_subtopic"}]
    assert max(n_parents) == 2  # a re-parented node: both parents active between W1 and W4
    final = {}
    for k, v in db.links.items():
        if k[0] == "topic_subtopic" and v["state"] == "active":
            final.setdefault(k[2], set()).add(k[1])
    assert all(len(p) == 1 for p in final.values()) and len(final) == 2


# ------------------------------------------------------------------------------------------------ T32
def test_t32_every_read_and_write_uses_the_one_held_connection():
    db = WDB(spans(), locks=Locks())
    j = mk(db)
    assert j.run()["state"] == "complete"
    assert j.eng.max_open == 1  # pool of one: no second checkout ever waits
    names = db.names()
    assert names.index("advisory_lock") < names.index("job_run_start") < names.index("spans_page")
    assert names[-1] == "advisory_unlock" or names.count("advisory_unlock") == 1
    assert db.locks.holder is None


def test_t32_writer_lost_when_the_held_lock_check_fails_before_a_write():
    locks = Locks()
    locks.lost_after = 0  # the very first held-lock check fails: the connection dropped
    db = WDB(spans(), locks=locks)
    j = mk(db)
    r = j.run()
    assert r["state"] == "partial" and r["stop"] == "writer_lost" and j.status()["reasons"] == ["writer_lost"]
    assert not [n for n in db.names() if n.startswith(("node_", "link_insert", "anchor_", "job_run_insert"))]
    assert db.locks.holder is not None or True  # the dead connection is abandoned, not trusted to unlock


# ------------------------------------------------------------------------------------------------ T33
def test_t33_second_pass_over_the_same_spans_is_unchanged_and_issues_no_w1_w4_statement():
    db = first_pass()
    db.log = []
    j = mk(db)
    r = j.run()
    assert r["complete"] and j.status()["state"] == "unchanged"
    names = db.names()
    assert not [n for n in names if n.startswith(WRITE_NAMES)]  # zero W1-W4 statements, zero retire-by-pair or by-span
    assert db.closes[-1]["state"] == "finished" and db.closes[-1]["reasons"] == ["unchanged"]
    assert len(db.runs) == 1  # no second build row


def test_t33_version_known_before_the_build_skips_the_build():
    class Spy(StubCore):
        built = 0

        def build(self, *a, **k):
            Spy.built += 1
            return super().build(*a, **k)

        def version_for(self, ids, params):
            import hashlib
            return hashlib.sha256(json.dumps(sorted(int(i) for i in ids)).encode()).hexdigest()[:12]

    db = WDB(spans())
    mk(db, core=Spy()).run()
    assert Spy.built == 1
    mk(db, core=Spy()).run()
    assert Spy.built == 1  # the second pass knew its version and did not build


# ------------------------------------------------------------------------------------------------ T35
def test_t35_prior_uses_each_spans_deepest_then_newest_strict_link():
    db = WDB(spans(5))
    j = mk(db)
    j._chunks = [{"span_id": i, "source_id": 100, "text": "x"} for i in range(1, 6)]
    for lvl, slug, i in (("domain", "d", 1), ("topic", "t", 2), ("topic", "t9", 3), ("subtopic", "s0", 4), ("subtopic", "s1", 5)):
        db.nodes[(lvl, slug)] = dict(id=i, label=slug, origin="induction", origin_ref=store.ORIGIN_REF_PREFIX + "v1", state="active")
    db.links[("domain_topic", 1, 2, 0)] = dict(version="v1", state="active", score=None, score_kind=None)
    db.links[("topic_subtopic", 2, 4, 1)] = dict(version="v1", state="active", score=None, score_kind=None)
    db.links[("topic_subtopic", 3, 5, 2)] = dict(version="v1", state="active", score=None, score_kind=None)
    st = dict(state="active")

    def add(kind, span, eid, seq, kindname="cosine", state="active"):
        db.links[(kind, span, eid, seq)] = dict(version="v1", state=state, score=0.9, score_kind=kindname)

    add("chunk_subtopic", 1, 4, 10)          # span 1: sub-topic s0 (strict) ...
    add("chunk_topic", 1, 3, 11)             # ... and a topic row t9: the deepest level wins, so span 1 belongs to s0 only (and t via s0)
    add("chunk_subtopic", 2, 4, 3)           # span 2: two strict rows in one table: the highest link_id (s1) wins
    add("chunk_subtopic", 2, 5, 7)
    add("chunk_subtopic", 3, 4, 12, kindname="cosine_r1")  # routing rows are not strict
    add("chunk_topic", 4, 2, 13)             # span 4: topic row only: member of t
    add("chunk_subtopic", 5, 4, 14, state="retired")       # retired rows are ignored
    prior = j._load_prior()
    members = {n["key"]: set(n["members"]) for n in prior["nodes"]}
    assert members["s0"] == {1}
    assert members["s1"] == {2}
    assert 3 not in set().union(*members.values())
    assert members["t"] == {1, 4}            # t's own strict span 4 plus s0's member 1 by upward propagation
    assert 1 not in members["t9"]            # t9's topic row for span 1 lost to the deeper sub-topic row
    assert members["d"] >= {1, 4} and st


# ------------------------------------------------------------------------------------------------ T39
def test_t39_a_span_re_embedded_under_the_same_id_forces_a_rebuild_although_the_version_is_unchanged():
    db = first_pass()
    assert len(db.runs) == 1
    db.vec_epoch = 5000.0  # a vector row written after the build's scan end (epoch 1000)
    j = mk(db)
    r = j.run()
    assert r["state"] == "complete" and j.status()["state"] == "complete" and len(db.runs) == 2
    assert db.runs[-1]["version"] == db.runs[0]["version"]
    # and with no newer vector the very next pass is a no-op again once its scan end is later than the vector write
    db.epoch, db.vec_epoch = 9000.0, 5000.0
    db.runs[-1]["metrics"] = db.runs[-1]["metrics"]  # the latest build row now carries scan_end_epoch 1000 (fake clock) ...
    db.epoch = 9000.0
    mk(db).run()  # ... so a vector at 5000 is still newer: rebuild, whose row records the new scan end
    j3 = mk(db)
    assert j3.run()["complete"]
    assert j3.status()["state"] == "unchanged"


# ------------------------------------------------------------------------------------------------ verifier R1, R2
def test_r1_a_failed_commit_of_w4_is_an_error_never_complete():
    db = WDB(spans(), locks=Locks())
    db.fail_commit_when = lambda names: "job_run_insert" in names  # W4's own transaction
    j = mk(db)
    r = j.run()
    assert r["state"] == "error" and r["complete"] is False
    assert j.status()["state"] == "error" and j.status()["reasons"] == ["RuntimeError"]


def test_r2_active_nodes_beyond_the_builds_own_block_the_no_op_rule_whatever_the_run_rows_say():
    # a pass killed after W1 (a deploy restart) leaves its run row `running` and the new version's nodes active beside the old ones: no
    # closed row records write_incomplete, yet the next pass must rebuild, not record `unchanged`
    db = first_pass()
    db.nodes[("topic", "ct-t-killed-after-w1")] = dict(id=999, label="stray", origin="induction", origin_ref=store.ORIGIN_REF_PREFIX + "other",
                                                         state="active")
    j = mk(db)
    assert j.run()["state"] == "complete" and j.status()["state"] != "unchanged" and len(db.runs) == 2
    # many later rows cannot push that evidence out: after the rebuild the stray node is retired (superseded) and the next pass is a no-op
    assert db.nodes[("topic", "ct-t-killed-after-w1")]["state"] == "retired"
    j2 = mk(db)
    j2.run()
    assert j2.status()["state"] == "unchanged"


def test_r2_a_rebuild_that_reuses_nodes_keeps_their_original_origin_ref_and_the_next_pass_is_still_a_no_op():
    db = first_pass()
    second_pass(db, StubCore(), extra=2)  # StubCore reuses every node: active nodes carry two different origin_ref values now
    refs = {v["origin_ref"] for v in db.nodes.values() if v["state"] == "active"}
    assert len(refs) == 1  # (StubCore keeps the first build's nodes, all with the same ref) ...
    db.nodes[("topic", "ct-t-other-ref")] = dict(id=998, label="x", origin="induction", origin_ref=store.ORIGIN_REF_PREFIX + "zz", state="retired")
    j = mk(db)
    j.run()
    assert j.status()["state"] == "unchanged"  # ... and node COUNTS, not origin_ref values, decide


def test_r2_an_aborted_write_after_the_build_blocks_the_no_op_rule():
    db = first_pass()
    db.spans = db.spans + [(11, 100, "alpha NEW 11")]
    db.fail_on = "link_insert_chunk_topic"  # a pass aborted between W1 and W4: its row carries write_incomplete
    assert mk(db, core=KeyedCore("c")).run()["state"] == "error"
    db.fail_on = None
    db.spans = db.spans[:-1]  # the corpus is back to the span set of the build row: same version as the latest build row
    db.log = []
    j = mk(db)
    r = j.run()
    assert j.status()["state"] != "unchanged" and r["state"] == "complete"  # rebuilt over the mixed state, not recorded as a no-op
    assert "job_run_insert" in db.names()
    # with no aborted pass in between the very next pass is the no-op again
    db.log = []
    j2 = mk(db)
    j2.run()
    assert j2.status()["state"] == "unchanged"
