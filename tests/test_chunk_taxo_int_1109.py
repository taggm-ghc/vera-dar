"""CHUNK-TAXO-INT-1109: integration of the REAL engine (vera.chunk_taxo) with the REAL job, store, status reader and Cloudflare
adapter, on ~300 synthetic chunks. The database is a recording mock that interprets the allowlisted statements, models the
024 constraints (rejects violating writes) and is transactional (a failed transaction rolls back); the HTTP layer is a fake opener.
No DB, no network, no env files."""
from __future__ import annotations

import copy
import json
import math
import random
import re

import pytest

from tests.test_chunk_taxo_job_1109 import ENV, FakeDB, FakeResp, Res
from vera import chunk_taxo as core
from vera import chunk_taxo_embed as emb
from vera import chunk_taxo_job as job
from vera import chunk_taxo_status as cstatus
from vera import chunk_taxo_store as store

DIM = 64
N_CLUSTER, PER = 6, 50
CF_ENV = {"CLOUDFLARE_ACCOUNT_ID": "acct123", "CLOUDFLARE_API_TOKEN": "TOKEN-SECRET-VALUE"}
WORDS = [("quantum", "entanglement", "qubit", "photon"), ("genome", "sequencing", "protein", "enzyme"),
         ("glacier", "permafrost", "arctic", "sediment"), ("tariff", "inflation", "currency", "bonds"),
         ("telescope", "galaxy", "nebula", "pulsar"), ("scholarship", "curriculum", "teacher", "classroom")]


def _centers():
    rng = random.Random(7)
    return [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(N_CLUSTER)]


CENTERS = _centers()


def make_spans(n_per=PER, start=1):
    out, sid = [], start
    for k in range(N_CLUSTER):
        for j in range(n_per):
            w = WORDS[k]
            out.append((sid, 100 + (k * 7 + j) % 12, f"c{k} {w[j % 4]} {w[(j + 1) % 4]} {w[(j + 2) % 4]} item {sid}"))
            sid += 1
    return out


def vec_for(text):
    k = int(text.split()[0][1:])
    rng = random.Random(hash_int(text))
    return [c + rng.gauss(0, 0.12) for c in CENTERS[k]]


def hash_int(t):
    return int.from_bytes(__import__("hashlib").sha256(t.encode()).digest()[:8], "big")


class CfOpener:
    """Fake HTTP: the Cloudflare Workers AI embedding endpoint."""

    def __init__(self):
        self.reqs = 0

    def __call__(self, req, timeout=None):
        assert req.full_url.startswith("https://api.cloudflare.com/client/v4/accounts/acct123/ai/run/@cf/baai/bge-m3")
        texts = json.loads(req.data)["text"]
        self.reqs += 1
        data = [vec_for(t) for t in texts]
        usage = {"total_tokens": sum(len(t) // 3 + 1 for t in texts)}
        return FakeResp({"result": {"shape": [len(texts), DIM], "data": data, "usage": usage}, "success": True, "errors": []})


# ---------------------------------------------------------------------------------------------- transactional, constrained mock DB
class Violation(ValueError):
    pass


SHA = re.compile(r"^[0-9a-f]{64}$")
MODEL = re.compile(r"^@cf/baai/[a-z0-9.-]{1,64}$")


class TxDB(FakeDB):
    SNAP = ("vectors", "nodes", "links", "runs", "anchors", "run_rows", "ctx", "ids")

    def __init__(self, spans, **kw):
        super().__init__(spans, **kw)
        self.commits = self.rollbacks = 0

    def snapshot(self):
        return {k: copy.deepcopy(getattr(self, k)) for k in self.SNAP}

    def restore(self, s):
        for k, v in s.items():
            setattr(self, k, v)

    def check_runs(self):
        for r in self.run_rows:
            if (r["display_ok"] is None) != (not r["finished"]):
                raise Violation("tax_job_runs_disp_chk")
            if r["state"] != "running" and not r["finished"]:
                raise Violation("tax_job_runs_statefin_chk")
            if r["state"] not in ("running", "finished", "deferred", "failed"):
                raise Violation("tax_job_runs_state_chk")
            if not isinstance(r["neurons"], int) or r["neurons"] < 0:
                raise Violation("tax_job_runs_neurons_chk")
            if not re.fullmatch(r"[ -~]{1,128}", r["version"]):
                raise Violation("tax_job_runs_cver_chk")

    def h_any(self, name, batch):
        p = batch[0]
        if name == "vectors_upsert":
            for q in batch:
                if not MODEL.match(q["model"]):
                    raise Violation("tax_chunk_vectors_model_chk")
                if not (64 <= q["dim"] <= 4096) or len(q["vec"]) != q["dim"]:
                    raise Violation("tax_chunk_vectors_dim_chk/vec_chk")
                if not all(isinstance(x, float) and math.isfinite(x) for x in q["vec"]):
                    raise Violation("tax_chunk_vectors_vec_chk")
                if not SHA.match(q["sha"]):
                    raise Violation("tax_chunk_vectors_sha_chk")
            return super().h_any(name, batch)
        if name == "job_run_start":
            self.run_rows.append(dict(state="running", finished=False, display_ok=None, version="pending", neurons=0))
            self.check_runs()
            return Res([(len(self.run_rows),)])
        if name == "job_run_neurons":
            r = self.run_rows[p["run_id"] - 1]
            if r["state"] == "running":
                r["neurons"] = p["neurons"]
            self.check_runs()
            return Res([])
        if name == "job_run_close":
            r = self.run_rows[p["run_id"] - 1]
            if r["state"] == "running":
                r.update(state=p["state"], neurons=p["neurons"], finished=True, display_ok=False, reasons=p["reasons"],
                         metrics=json.loads(p["metrics"]))
            self.check_runs()
            return Res([])
        if name == "job_run_insert":
            self.run_rows.append(dict(state="finished", finished=True, display_ok=p["display_ok"], version=p["version"], neurons=0,
                                      reasons=p["reasons"], metrics=json.loads(p["metrics"])))
            self.runs.append(p)
            self.check_runs()
            return Res([])
        if name == "neurons_today":
            return Res([(sum(r["neurons"] for r in self.run_rows),)])
        if name == "job_run_latest":
            fin = [r for r in self.run_rows if r["state"] == "finished" and r["finished"] and r["version"] != "pending"]
            if not fin:
                return Res([])
            r = fin[-1]
            return Res([(r["version"], _Ts(), r["metrics"], r["display_ok"], r["reasons"])])
        return super().h_any(name, batch)


class _Ts:
    def isoformat(self):
        return "2026-11-09T00:00:00+00:00"


class TxConn:
    def __init__(self, db):
        self.db = db

    def execute(self, stmt, params=None):
        return self.db.execute(stmt, params)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TxBegin(TxConn):
    def __enter__(self):
        self.snap = self.db.snapshot()
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            self.db.commits += 1
        else:
            self.db.restore(self.snap)
            self.db.rollbacks += 1
        return False


class TxEngine:
    def __init__(self, db):
        self.db = db

    def connect(self):
        return TxConn(self.db)

    def begin(self):
        return TxBegin(self.db)

    def dispose(self):
        pass


def factory_for(db):
    return lambda url, **kw: TxEngine(db)


def cfg_real(**jobkw):
    c = core.load_cfg()
    assert c["enabled"] is True and c["embedder"] == "cloudflare" and c["cloudflare"]["send_allowed"] is True
    c = dict(c, enabled=True, job=dict(c["job"], page_size=100, tick_row_cap=1000, embed_batch=40, **jobkw))
    return c


def make(db, opener, **jobkw):
    emb.reset_budget_for_tests()
    e = emb.CloudflareEmbedder(core.load_cfg()["cloudflare"], env=dict(CF_ENV), opener=opener, clock=lambda: 1_700_000_000.0,
                               sleep=lambda s: None)
    return job.Job(cfg_real(**jobkw), core=core, embedder=e, engine_factory=factory_for(db), env=dict(ENV), rss_reader=lambda: 100.0)


def level_of(db, nid):
    for (lvl, _s), v in db.nodes.items():
        if v["id"] == nid:
            return lvl


@pytest.fixture(scope="module")
def first_run():
    db = TxDB(make_spans())
    op = CfOpener()
    j = make(db, op)
    res = j.run()
    return db, op, j, res


def test_config_sections_load_and_are_validated(tmp_path):
    c = core.load_cfg()
    assert c["cloudflare"]["model"] == "@cf/baai/bge-m3" and c["cloudflare"]["daily_neuron_budget"] == 8000
    assert c["embedder"] == "cloudflare" and c["job"] == {"finish_now": True}
    for bad in ({"cloudflare": {"nope": 1}}, {"cloudflare": {"send_allowed": "yes"}}, {"job": {"page_size": 0}},
                {"job": {"zzz": 1}}, {"embedder": "Bad Name"}, {"bogus": 1}):
        p = tmp_path / "c.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            core.load_cfg(p)


def test_real_pair_completes_and_writes_nodes_proposed_then_active_with_actor(first_run):
    db, op, j, res = first_run
    assert res["complete"] and res["state"] == "complete", res
    n = db.names()
    ins = [p for nm, p in db.log if nm.startswith("node_insert_")]
    assert ins and all(q["state"] == "proposed" and q["origin"] == "induction" for rows in ins for q in rows)
    assert n.index("set_ctx") < n.index("node_insert_domain") < n.index("node_activate_domain")
    assert db.ctx[0] == "auto:chunk_taxo" and len(db.ctx[1]) == 64
    assert db.nodes and all(v["state"] == "active" and v["origin_ref"].startswith("auto-chunk-taxo:") for v in db.nodes.values())
    assert {lvl for (lvl, _s) in db.nodes} == {"domain", "topic", "subtopic"}
    assert op.reqs >= 300 // 40 and len(db.vectors) == 300
    assert all(m == "@cf/baai/bge-m3" for m, _sha, _v in db.vectors.values())


def test_chunk_links_deepest_level_only(first_run):
    db = first_run[0]
    # strict links only (score_kind 'cosine'); routing-only multi-membership links (cosine_r<k>, cosine_soft) are extra (TAXO-ROUTE-R1-1110)
    sub = sorted((k[1], k[2]) for k, v in db.links.items() if k[0] == "chunk_subtopic" and v["state"] == "active" and v["score_kind"] == "cosine")
    top = sorted((k[1], k[2]) for k, v in db.links.items() if k[0] == "chunk_topic" and v["state"] == "active" and v["score_kind"] == "cosine")
    spans_sub, spans_top = {a for a, _b in sub}, {a for a, _b in top}
    assert not (spans_sub & spans_top)                       # one strict link per chunk: never both levels
    assert len(sub) == len(spans_sub) and len(top) == len(spans_top)
    assert len(spans_sub | spans_top) >= 0.8 * 300 and len(spans_sub) > 0
    assert all(level_of(db, b) == "subtopic" for _a, b in sub) and all(level_of(db, b) == "topic" for _a, b in top)
    assert not [1 for k in db.links if k[0] in ("chunk_domain",)]


def test_hierarchy_votes_and_run_row(first_run):
    db, _op, j, _res = first_run
    assert db.active("domain_topic") and db.active("topic_subtopic")
    st, sd = db.active("source_topic"), db.active("source_domain")
    assert st and sd and {a for a, _b in st} <= set(range(100, 112))
    assert all(level_of(db, b) == "topic" for _a, b in st) and all(level_of(db, b) == "domain" for _a, b in sd)
    for k, v in db.links.items():
        if k[0] in ("source_topic", "source_domain"):
            assert v["score_kind"] == "share" and 0 < v["score"] <= 1
    assert len(db.runs) == 1
    r = db.runs[0]
    m = json.loads(r["metrics"])
    assert isinstance(r["display_ok"], bool) and isinstance(r["reasons"], list)
    assert {"silhouette", "coverage", "run"} <= set(m) and m["run"]["rows_embedded"] == 300
    # the running row was closed in the same transaction as the finished row
    assert [x["state"] for x in db.run_rows] == ["finished", "finished"] and db.run_rows[0]["version"] == "pending"
    assert db.run_rows[0]["neurons"] > 0 and db.run_rows[1]["neurons"] == 0
    assert j.status()["display_ok"] is r["display_ok"]


def test_whole_write_is_one_transaction(first_run):
    db = first_run[0]
    # begin() blocks: run start, one per embed batch, ONE write transaction (nodes..links..votes..run rows)
    assert db.rollbacks == 0
    wr = [i for i, nm in enumerate(db.names()) if nm in ("node_insert_domain", "link_insert_chunk_subtopic", "link_insert_source_topic",
                                                          "job_run_insert", "job_run_close")]
    assert wr and db.names().index("job_run_insert") > db.names().index("link_insert_source_topic")


def test_rerun_is_idempotent_and_does_not_reembed(first_run):
    db0, op0, _j, _r = first_run
    db = copy.deepcopy(db0)  # keep the module fixture intact
    db.log = []
    op = CfOpener()
    r = make(db, op).run()
    assert r["complete"] and op.reqs == 0
    assert db.nodes == db0.nodes and db.active("chunk_subtopic") == db0.active("chunk_subtopic")
    assert db.active("chunk_topic") == db0.active("chunk_topic")
    assert db.active("source_topic") == db0.active("source_topic") and db.active("domain_topic") == db0.active("domain_topic")
    assert len(db.runs) == 2 and db.runs[1]["version"] == db.runs[0]["version"]
    assert not [nm for nm in db.names() if nm.startswith("node_insert")] or all(
        not rows for nm, rows in db.log if nm.startswith("node_insert"))
    assert [x["state"] for x in db.run_rows].count("running") == 0


def test_status_reader_returns_the_run(first_run):
    db = first_run[0]
    cstatus.reset_for_tests()
    s = cstatus.status(engine_factory=factory_for(db), env=dict(ENV))
    r = db.runs[0]
    assert s["state"] == "ok" and s["version"] == r["version"] and s["display_ok"] is r["display_ok"]
    assert s["metrics"]["run"]["rows_embedded"] == 300 and s["last_run"]
    cstatus.reset_for_tests()


def test_mid_write_failure_leaves_previous_version_active():
    db = TxDB(make_spans())
    assert make(db, CfOpener()).run()["complete"]
    v1 = db.runs[0]["version"]
    before = (copy.deepcopy(db.nodes), db.active("chunk_subtopic"), db.active("chunk_topic"), db.active("source_topic"),
              db.active("domain_topic"), len(db.runs))
    # more chunks => a new version (and a new pass); the write fails after nodes, hierarchy and chunk links were written
    db.spans = db.spans + make_spans(10, start=1001)
    db.fail_on = "link_insert_source_topic"
    emb_op = CfOpener()
    r = make(db, emb_op).run()
    assert r["state"] == "error" and db.rollbacks >= 1
    assert db.nodes == before[0]
    assert all(not (v["origin_ref"] or "").endswith(("auto-chunk-taxo:" + "x")) for v in db.nodes.values())
    assert (db.active("chunk_subtopic"), db.active("chunk_topic"), db.active("source_topic"), db.active("domain_topic")) == before[1:5]
    assert len(db.runs) == before[5] == 1 and db.runs[0]["version"] == v1
    # no link or run row of the new version is visible; the failed pass's running row is closed as failed, not left running
    assert all(v["version"] == v1 for k, v in db.links.items() if v["state"] == "active")
    assert [x["state"] for x in db.run_rows] == ["finished", "finished", "failed"]
    assert db.run_rows[2]["finished"] and db.run_rows[2]["display_ok"] is False
    # embedding batches are a cache and stay committed (vectors for the new chunks exist); the old taxonomy is still readable
    assert len(db.vectors) == 360
    cstatus.reset_for_tests()
    assert cstatus.status(engine_factory=factory_for(db), env=dict(ENV))["version"] == v1
    cstatus.reset_for_tests()


def test_daily_neuron_counter_is_read_from_the_db_and_defers():
    db = TxDB(make_spans())
    db.run_rows.append(dict(state="finished", finished=True, display_ok=False, version="old", neurons=8000))  # today's earlier usage
    op = CfOpener()
    r = make(db, op).run()
    assert r["state"] == "deferred" and op.reqs == 0 and not r["complete"]
    assert [x["state"] for x in db.run_rows] == ["finished", "deferred"] and db.run_rows[1]["finished"]
    assert db.nodes == {} and not db.runs


def test_mock_rejects_024_constraint_violations():
    db = TxDB([])
    good = dict(span_id=1, model="@cf/baai/bge-m3", dim=DIM, sha="a" * 64, vec=[0.1] * DIM)
    store.run_sql(TxConn(db), "vectors_upsert", [good])
    for bad in (dict(good, model="fake"), dict(good, dim=8, vec=[0.1] * 8), dict(good, vec=[0.1] * 63),
                dict(good, vec=[float("nan")] + [0.1] * 63), dict(good, vec=[float("inf")] + [0.1] * 63), dict(good, sha="A" * 64)):
        with pytest.raises(Violation):
            store.run_sql(TxConn(db), "vectors_upsert", [bad])
    db.run_rows.append(dict(state="finished", finished=True, display_ok=None, version="v", neurons=0))
    with pytest.raises(Violation):
        db.check_runs()
    db.run_rows[-1].update(display_ok=True, finished=False)
    with pytest.raises(Violation):
        db.check_runs()
    db.run_rows[-1].update(display_ok=None, finished=False)
    with pytest.raises(Violation):
        db.check_runs()
    db.run_rows[-1].update(state="running")
    db.check_runs()  # a running row with finished_at NULL and display_ok NULL is valid
    db.run_rows[-1].update(neurons=1.5)
    with pytest.raises(Violation):
        db.check_runs()


def test_wp6_page_gate_uses_the_single_status_entry_point(first_run, monkeypatch):
    """WP6 (vera.corpus_taxonomy.job_status) reads the SAME latest-'finished'-row reader the job module exposes."""
    from vera import corpus_taxonomy
    db = copy.deepcopy(first_run[0])
    ok_run = db.run_rows[1]
    for flag in (True, False):
        ok_run["display_ok"] = flag
        cstatus.reset_for_tests()
        monkeypatch.setattr(cstatus, "status", lambda: cstatus.__dict__["_orig"](engine_factory=factory_for(db), env=dict(ENV)))
        monkeypatch.setitem(cstatus.__dict__, "_orig", _ORIG_STATUS)
        st = corpus_taxonomy.job_status()
        assert st["display_ok"] is flag and st["version"] == ok_run["version"]
        cstatus.reset_for_tests()
    # a later deferred or failed row never replaces the finished one
    db.run_rows.append(dict(state="failed", finished=True, display_ok=False, version="pending", neurons=0, reasons=[], metrics={}))
    ok_run["display_ok"] = True
    cstatus.reset_for_tests()
    assert corpus_taxonomy.job_status()["display_ok"] is True
    cstatus.reset_for_tests()
    # 024 absent or unreachable -> not displayable (interim page)
    db.has = (True, True, False)
    assert corpus_taxonomy.job_status()["display_ok"] is False
    cstatus.reset_for_tests()


_ORIG_STATUS = cstatus.status
