"""CHUNK-TAXO-JOB-1109: offline tests for the chunk-taxonomy job, store, status reader, embedder slot and M1 script.
A recording in-memory fake stands in for the database (it interprets the allowlisted statements by name); a stub core and a
deterministic fake embedder or a fake HTTP opener replace C1 and Cloudflare. No DB, no network, no model."""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from itertools import count
from types import SimpleNamespace

import pytest

from vera import chunk_taxo_embed as emb
from vera import chunk_taxo_job as job
from vera import chunk_taxo_status as cstatus
from vera import chunk_taxo_store as store

SENTINEL_TEXT = "SENTINELCHUNKBODY"
BY_TEXT = {t: n for n, t in store.SQL.items()}
ENV = {"VERA_DB_URL_RW": "postgresql://fake/none"}


class Res:
    def __init__(self, rows):
        self.rows = list(rows)

    def fetchall(self):
        return list(self.rows)

    def first(self):
        return self.rows[0] if self.rows else None


class FakeDB:
    """Interprets allowlisted statements; records (name, params)."""

    def __init__(self, spans, user_rw=True, user_wo=False, has=(True, True, True)):
        self.spans, self.log, self.has = spans, [], has
        self.acct = ("vera_claude_code_rw", user_rw, user_wo)
        self.vectors, self.nodes, self.links, self.runs, self.anchors = {}, {}, {}, [], []
        self.run_rows = []
        self.inert_rows = []
        self.ids = count(1)
        self.ctx = None
        self.fail_on = None

    def names(self):
        return [n for n, _ in self.log]

    def execute(self, stmt, params=None):
        sql = str(stmt)
        name = BY_TEXT[sql]  # KeyError => a statement outside the allowlist
        batch = params if isinstance(params, list) else ([params] if params is not None else [None])
        self.log.append((name, params))
        if self.fail_on and name == self.fail_on:
            raise RuntimeError("boom")
        return self.h_any(name, batch)

    def h_any(self, name, batch):
        p = batch[0]
        if name == "probe_account":
            return Res([self.acct])
        if name == "probe_schema":
            return Res([self.has])
        if name == "set_ctx":
            self.ctx = (p["actor"], p["batch"])
            return Res([])
        if name == "spans_page":
            rows = [r for r in self.spans if r[0] > p["after"]][: p["lim"]]
            return Res([(a, b, c[: p["n"]]) for a, b, c in rows])
        if name == "sources_portal":
            # ARXIV-DEFENSE-1113: default = a non-portal doi.org row per id; self.sources (id -> row | None) overrides, and an
            # id mapped to None (or omitted from the map when overriding) is absent from the result
            ov = getattr(self, "sources", None)
            out = []
            for i in p["ids"]:
                if ov is not None and i in ov:
                    if ov[i] is not None:
                        out.append((i, *ov[i]))
                else:
                    u = f"https://doi.org/10.1000/example.{i}"
                    out.append((i, u, {"document_id": f"acq-{i}", "url": u}))
            return Res(out)
        if name == "vectors_select":
            return Res([(i, *self.vectors[i]) for i in p["ids"] if i in self.vectors])
        if name == "vectors_upsert":
            for q in batch:
                self.vectors[q["span_id"]] = (q["model"], q["sha"], q["vec"])
            return Res([])
        if name == "job_run_insert":
            self.runs.append(p)
            return Res([])
        if name == "job_run_inert":
            self.inert_rows.append(p)
            return Res([])
        if name == "job_run_start":
            self.run_rows.append(dict(state="running", neurons=0))
            return Res([(len(self.run_rows),)])
        if name == "job_run_neurons":
            self.run_rows[p["run_id"] - 1]["neurons"] = p["neurons"]
            return Res([])
        if name == "job_run_close":
            self.run_rows[p["run_id"] - 1].update(state=p["state"], neurons=p["neurons"])
            return Res([])
        if name == "neurons_today":
            return Res([(sum(r["neurons"] for r in self.run_rows),)])
        lvl = name.split("_")[-1]
        if name.startswith("nodes_"):
            return Res([(s, v["id"], v["label"], v["origin"], v["origin_ref"], v["state"]) for (l, s), v in self.nodes.items()
                        if l == lvl and v["state"] in ("proposed", "active")])
        if name.startswith("node_ids_"):
            return Res([(s, v["id"], v["state"]) for (l, s), v in self.nodes.items() if l == lvl and s in p["slugs"]])
        if name.startswith("node_insert_"):
            for q in batch:
                if (lvl, q["slug"]) not in self.nodes:
                    self.nodes[(lvl, q["slug"])] = dict(id=next(self.ids), label=q["label"], origin=q["origin"],
                                                         origin_ref=q["origin_ref"], state=q["state"])
            return Res([])
        if name.startswith("node_activate_"):
            for (l, s), v in self.nodes.items():
                if l == lvl and s in p["slugs"] and v["state"] == "proposed" and v["origin"] == "induction":
                    v["state"] = "active"
            return Res([])
        if name.startswith("node_retire_"):
            for (l, s), v in self.nodes.items():
                if l == lvl and s in p["slugs"] and v["state"] == "active" and v["origin"] == "induction" \
                        and (v["origin_ref"] or "").startswith("auto-chunk-taxo:"):
                    v["state"] = "retired"
            return Res([])
        if name.startswith("anchor_insert_"):
            self.anchors.extend(batch)
            return Res([])
        if name.startswith(("anchor_activate_", "anchor_retire_")):
            return Res([])
        if name.startswith("link_insert_"):
            kind = name[len("link_insert_"):]
            for q in batch:
                if not any(k[0] == kind and k[1] == q["a"] and k[2] == q["b"] and v["state"] == "active" for k, v in self.links.items()):
                    self.links[(kind, q["a"], q["b"], len(self.links))] = dict(version=q["version"], state="active", score=q["score"],
                                                                                score_kind=q["score_kind"])
            return Res([])
        if name.startswith("link_retire_"):
            kind = name[len("link_retire_"):]
            for k, v in self.links.items():
                if k[0] == kind and v["state"] == "active" and v["version"] != p["version"] and ("ids" not in p or k[1] in p["ids"]):
                    v["state"] = "retired"
            return Res([])
        if name.startswith("link_select_"):
            kind = name[len("link_select_"):]
            rows = sorted((k[1], k[2]) for k, v in self.links.items() if k[0] == kind and v["state"] == "active")
            if p:
                rows = [r for r in rows if r[0] > p["after"]][: p["lim"]]
            return Res(rows)
        raise AssertionError(name)

    def active(self, kind):
        return sorted((k[1], k[2]) for k, v in self.links.items() if k[0] == kind and v["state"] == "active")


class Conn:
    def __init__(self, db):
        self.db = db

    def execute(self, stmt, params=None):
        return self.db.execute(stmt, params)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Engine:
    def __init__(self, db, kw):
        self.db, self.kw, self.disposed = db, kw, False

    def connect(self):
        return Conn(self.db)

    begin = connect

    def dispose(self):
        self.disposed = True


def factory_for(db, seen=None):
    def f(url, **kw):
        eng = Engine(db, kw)
        if seen is not None:
            seen.append(eng)
        return eng
    return f


class FakeEmb:
    name = "fake"

    def __init__(self, fail_after=None):
        self.calls, self.fail_after = [], fail_after

    def embed(self, texts):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("embedder died")
        self.calls.append(len(texts))
        return [[1.0, 0.0] if "alpha" in t else ([-1.0, 0.0] if "beta" in t else [0.0, 0.0]) for t in texts]


class StubCore:
    """Same shape as vera.chunk_taxo (C1 contract) with a trivial deterministic rule."""

    def build(self, chunks, vecs, params, prior=None):
        ids = sorted(int(c[0]) for c in chunks)
        version = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12]
        prior_nodes = {(n["level"]): n for n in (prior or {}).get("nodes", [])} if prior else {}
        nodes = [{"key": "D", "level": "domain", "parent_key": None, "label": "Domain A", "terms": ["alpha", "beta"], "reused_from": None},
                 {"key": "T", "level": "topic", "parent_key": "D", "label": "Topic A", "terms": ["alpha", "beta"], "reused_from": None},
                 {"key": "S0", "level": "subtopic", "parent_key": "T", "label": "Sub alpha", "terms": ["alpha", "alphas"], "reused_from": None},
                 {"key": "S1", "level": "subtopic", "parent_key": "T", "label": "Sub beta", "terms": ["beta", "betas"], "reused_from": None}]
        reused = set()
        if prior:
            by_lvl = {}
            for n in prior["nodes"]:
                by_lvl.setdefault(n["level"], []).append(n)
            for n in nodes:
                cands = [p for p in by_lvl.get(n["level"], []) if p["key"] not in reused]
                if cands:
                    n["reused_from"] = cands[0]["key"]
                    reused.add(cands[0]["key"])
        retired = sorted(p["key"] for p in (prior or {}).get("nodes", []) if p["key"] not in reused)
        assoc = {}
        for (sid, src, text), v in zip(chunks, vecs):
            sid = int(sid)
            if v[0] > 0:
                assoc[sid] = dict(subtopic="S0", topic="T", domain="D", level="subtopic", distance=0.1, margin=0.5, reason=None)
            elif v[0] < 0:
                assoc[sid] = dict(subtopic="S1", topic="T", domain="D", level="subtopic", distance=0.2, margin=0.5, reason=None)
            else:
                assoc[sid] = dict(subtopic=None, topic="T", domain="D", level="topic", distance=0.6, margin=0.0, reason="NO_FIT")
        return SimpleNamespace(version=version, params=params, nodes=nodes, assoc=assoc, null={}, retired_keys=retired)

    def quality(self, model, vecs):
        return {"silhouette": {"subtopic": 0.5}, "note": "text-must-not-leak"}

    def display_gate(self, metrics, gate):
        return True, []

    def doc_subjects(self, model, span_source):
        out = {}
        for sid, src in span_source.items():
            out[src] = {"union": {"T": 1}, "vote": {"topic": "T", "domain": "D", "share": 0.9, "domain_share": 0.95}}
        return out

    def associate(self, model, vec):
        return dict(subtopic="S0", topic="T", domain="D", level="subtopic", distance=0.1, margin=0.5, reason=None)


def spans(n=10):
    kinds = ["alpha", "beta", "gamma"]
    return [(i, 100 + i % 3, f"{kinds[i % 3]} {SENTINEL_TEXT} {i}") for i in range(1, n + 1)]


def cfg(**jobkw):
    return {"enabled": True, "params": {}, "display_gate": {}, "embedder": "x", "job": dict(jobkw)}


def make(db, e=None, c=None, env=None, **jobkw):
    seen = []
    j = job.Job(c or cfg(**jobkw), core=StubCore(), embedder=e or FakeEmb(), engine_factory=factory_for(db, seen),
                env=env if env is not None else dict(ENV), rss_reader=lambda: 100.0)
    j.seen = seen
    return j


# ----------------------------------------------------------------------------------------------- flag / account / schema
def test_flag_off_creates_no_engine():
    seen = []
    j = job.Job({"enabled": False}, core=StubCore(), embedder=FakeEmb(), engine_factory=factory_for(FakeDB(spans()), seen), env=ENV)
    assert j.run()["state"] == "off" and j.engines_created == 0 and not seen
    job.reset_for_tests()
    assert job.start({"enabled": False}) is False and job._THREAD is None
    assert job.start({}) is False
    assert job.status()["state"] == "off"


def test_start_never_raises(monkeypatch):
    job.reset_for_tests()
    def bad(*a, **k):
        raise ValueError("bad")
    from vera import chunk_taxo as real_core  # FLAGON-TAXO-1109: the shipped config is on, so patch the module attribute that start() reads
    monkeypatch.setattr(real_core, "load_cfg", bad)
    assert job.start() is False
    monkeypatch.setattr(real_core, "load_cfg", lambda: {"enabled": False})
    assert job.start() is False and job._THREAD is None


def test_embedder_none_is_inert_with_one_log_line(caplog):
    caplog.set_level(logging.INFO, logger="vera.chunk_taxo")
    db = FakeDB(spans())
    j = job.Job({"enabled": True, "params": {}}, core=StubCore(), engine_factory=factory_for(db), env=dict(ENV))
    assert j.run()["state"] == "inert"
    assert [r for r in caplog.records if "inert" in r.getMessage()].__len__() == 1
    assert db.names() == ["job_run_inert"] and db.inert_rows == [{"reasons": ["embedder_none"]}]  # self-report row only


@pytest.mark.parametrize("rw,wo,why", [(False, False, "account_not_rw"), (True, True, "account_is_wo")])
def test_wrong_account_inert(rw, wo, why):
    db = FakeDB(spans(), user_rw=rw, user_wo=wo)
    j = make(db)
    r = j.run()
    assert r["state"] == "inert" and j.status()["reasons"] == [why]
    assert db.names() == ["probe_account", "job_run_inert"] and db.inert_rows == [{"reasons": [why]}]


def test_missing_024_or_020_inert():
    for has, why in (((True, True, False), "schema_024_missing"), ((False, False, True), "schema_020_missing")):
        db = FakeDB(spans(), has=has)
        j = make(db)
        assert j.run()["state"] == "inert" and j.status()["reasons"] == [why]
        assert "node_insert_topic" not in db.names()


def test_020_only_is_compute_only_and_writes_no_taxonomy():
    db = FakeDB(spans(), has=(True, False, True))
    r = make(db).run()
    assert r["state"] == "compute_only" and r["complete"]
    assert not [n for n in db.names() if n.startswith(("node_", "link_", "anchor_", "job_run_insert", "set_ctx"))]  # run counter rows only (INT-1109 b)
    assert not db.nodes and not db.links and not db.runs


def test_020_only_curator_switch_writes_curator_origin_active():
    db = FakeDB(spans(), has=(True, False, True))
    j = make(db, allow_curator_origin_on_020=True)
    assert j.run()["complete"]
    assert db.nodes and all(v["origin"] == "curator" and v["state"] == "active" for v in db.nodes.values())
    assert "set_ctx" not in db.names() and not db.anchors
    j2 = make(FakeDB(spans(), has=(True, False, True)), allow_curator_origin_on_020="true")  # not literally True => compute only
    assert j2.run()["state"] == "compute_only"


# ----------------------------------------------------------------------------------------------- 022 write path
def test_write_path_sequence_actor_and_links():
    db = FakeDB(spans())
    j = make(db)
    r = j.run()
    assert r["complete"] and r["state"] == "complete"
    n = db.names()
    first = {k: n.index(k) for k in ("probe_account", "probe_schema", "set_ctx", "node_insert_domain", "node_insert_subtopic",
                                      "anchor_insert_topic", "node_activate_topic", "anchor_activate_topic", "link_insert_domain_topic",
                                      "link_insert_chunk_subtopic", "link_insert_source_topic", "job_run_insert")}
    assert first["probe_account"] < first["probe_schema"] < first["set_ctx"] < first["node_insert_domain"]
    assert first["node_insert_subtopic"] < first["anchor_insert_topic"] < first["node_activate_topic"] < first["anchor_activate_topic"]
    assert first["anchor_activate_topic"] < first["link_insert_domain_topic"] < first["link_insert_chunk_subtopic"] \
        < first["link_insert_source_topic"] < first["job_run_insert"]
    assert db.ctx[0] == "auto:chunk_taxo" and len(db.ctx[1]) == 64
    for (l, s), v in db.nodes.items():
        assert v["origin"] == "induction" and v["state"] == "active" and v["origin_ref"].startswith("auto-chunk-taxo:")
        assert s.startswith("ct-") and len(s) <= 64
    inserts = [p for nm, p in db.log if nm.startswith("node_insert_")]
    assert all(q["state"] == "proposed" for rows in inserts for q in rows)
    sub = db.active("chunk_subtopic")
    top = db.active("chunk_topic")
    assert len(sub) + len(top) == 10 and len(top) == len([s for s in spans() if "gamma" in s[2]])  # deepest level only
    assert {k[0] for k in db.links} <= {"domain_topic", "topic_subtopic", "chunk_subtopic", "chunk_topic", "source_topic", "source_domain"}
    for k, v in db.links.items():
        if k[0] in ("chunk_subtopic", "chunk_topic"):
            assert v["score_kind"] == "cosine" and -1 <= v["score"] <= 1
    assert len(db.active("source_topic")) == 3 and len(db.active("source_domain")) == 3
    assert len(db.runs) == 1 and db.runs[0]["display_ok"] is True
    assert "run" in json.loads(db.runs[0]["metrics"]) and "peak_rss_mb" in json.loads(db.runs[0]["metrics"])["run"]
    assert all(a["text"].isascii() and 20 <= len(a["text"]) <= 1200 for a in db.anchors)
    st = j.status()
    assert set(st) == {"state", "version", "metrics", "display_ok", "reasons", "last_run"} and st["display_ok"] is True


def test_idempotent_rerun_and_resume_after_crash():
    db = FakeDB(spans())
    assert make(db).run()["complete"]
    snap = (sorted(db.nodes), db.active("chunk_subtopic"), db.active("chunk_topic"), db.active("source_topic"),
            db.active("domain_topic"), db.active("topic_subtopic"))
    e2 = FakeEmb()
    assert make(db, e=e2).run()["complete"]
    assert e2.calls == []  # vectors reused from the DB cache (same text sha256 and model)
    assert snap == (sorted(db.nodes), db.active("chunk_subtopic"), db.active("chunk_topic"), db.active("source_topic"),
                    db.active("domain_topic"), db.active("topic_subtopic"))
    assert len([1 for v in db.links.values() if v["state"] == "active"]) == len(db.links)  # nothing duplicated, nothing retired
    assert len(db.runs) == 2


def test_resume_embeds_only_missing_after_crash():
    db = FakeDB(spans())
    bad = FakeEmb(fail_after=1)
    r = make(db, e=bad, embed_batch=4).run()
    assert r["state"] == "error" and len(db.vectors) == 4 and not db.nodes  # one committed batch survives, nothing half-written
    good = FakeEmb()
    r2 = make(db, e=good, embed_batch=4).run()
    assert r2["complete"] and sum(good.calls) == 6 and len(db.vectors) == 10


def test_changed_text_is_reembedded():
    db = FakeDB(spans())
    make(db).run()
    db.spans[0] = (1, 101, "beta changed " + SENTINEL_TEXT)
    e = FakeEmb()
    make(db, e=e).run()
    assert sum(e.calls) == 1


def test_retire_scope():
    db = FakeDB(spans())
    make(db).run()
    for lvl, s in (("subtopic", "ct-s-old000000000"), ("topic", "ct-t-reg")):
        db.nodes[(lvl, s)] = dict(id=next(db.ids), label="old " + s, origin="induction" if "old" in s else "registry",
                                  origin_ref="auto-chunk-taxo:v0" if "old" in s else "topreg:x", state="active")
    db.links[("chunk_subtopic", 1, 999, 99)] = dict(version="v0", state="active", score=None, score_kind=None)
    db.links[("chunk_subtopic", 1, 998, 98)] = dict(version="v0", state="active", score=None, score_kind=None)
    db.spans.append((11, 100, "alpha new " + SENTINEL_TEXT))  # new corpus => new version => full swap
    assert make(db).run()["complete"]
    assert db.nodes[("subtopic", "ct-s-old000000000")]["state"] == "retired"  # in retired_keys, auto origin
    assert db.nodes[("topic", "ct-t-reg")]["state"] == "active"  # registry row never touched
    assert all(v["state"] == "retired" for k, v in db.links.items() if v["version"] == "v0")
    for nm, p in db.log:
        if nm.startswith("node_retire_"):
            assert "origin = 'induction'" in store.SQL[nm] and "auto-chunk-taxo:%" in store.SQL[nm] and "DELETE" not in store.SQL[nm]
        if nm.startswith("link_retire_"):
            assert "classifier_name = 'chunk_taxo_auto'" in store.SQL[nm] and "classifier_version <> :version" in store.SQL[nm]


# ----------------------------------------------------------------------------------------------- bounds
def test_max_rows_stops_partial_without_build():
    db = FakeDB(spans())
    j = make(db, max_rows=3, embed_batch=2)
    r = j.run()
    assert r["state"] == "partial" and r["stop"] == "max_rows" and not r["complete"]
    assert len(db.vectors) == 3 and not db.nodes and not db.runs
    r2 = make(db, max_rows=100).run()
    assert r2["complete"] and len(db.vectors) == 10


def test_page_and_row_cap_and_embed_batch_bounds():
    db = FakeDB(spans(23))
    e = FakeEmb()
    j = make(db, e=e, page_size=5, tick_row_cap=7, embed_batch=3)
    assert j.run()["complete"]
    pages = [p for n, p in db.log if n == "spans_page"]
    assert pages and all(p["lim"] <= 5 for p in pages) and max(e.calls) <= 3
    assert j.seen[0].kw["connect_args"]["options"] == "-c statement_timeout=30000 -c lock_timeout=5000"
    assert j.seen[0].kw["pool_size"] == 1


def test_rss_guard_halves_then_stops():
    db = FakeDB(spans(20))
    e = FakeEmb()
    j = make(db, e=e, embed_batch=8)
    j._rss = lambda: 400.0  # above soft (380), below hard (420)
    assert j.run()["complete"] and e.calls[0] == 4 and max(e.calls) <= 4
    db2 = FakeDB(spans(20))
    j2 = make(db2, embed_batch=8)
    j2._rss = lambda: 500.0
    r = j2.run()
    assert r["state"] == "partial" and r["stop"] == "rss_hard" and not db2.vectors and not db2.nodes


def test_max_seconds_stops():
    db = FakeDB(spans())
    ticks = count(0, 100)
    j = job.Job(cfg(max_seconds=150), core=StubCore(), embedder=FakeEmb(), engine_factory=factory_for(db), env=dict(ENV),
                clock=lambda: next(ticks), rss_reader=lambda: 1.0)
    r = j.run()
    assert r["stop"] == "max_seconds" and not db.runs


# ----------------------------------------------------------------------------------------------- never raises / allowlist / privacy
def test_exceptions_never_escape(caplog):
    caplog.set_level(logging.INFO)
    def boom(url, **kw):
        raise RuntimeError("secret-host.example")
    j = job.Job(cfg(), core=StubCore(), embedder=FakeEmb(), engine_factory=boom, env=dict(ENV))
    assert j.run()["state"] == "error"
    assert "secret-host" not in caplog.text and "RuntimeError" in caplog.text
    db = FakeDB(spans())
    db.fail_on = "link_insert_chunk_subtopic"
    assert make(db).run()["state"] == "error"
    j3 = job.Job(cfg(), core=StubCore(), embedder=FakeEmb(), env={})  # no URL env
    assert j3.run()["state"] == "error"


def test_allowlist_and_no_forbidden_sql():
    db = FakeDB(spans())
    make(db).run()
    make(db, c=cfg()).run()
    assert db.log
    for name in db.names():
        assert name in store.SQL
    for name, sql in store.SQL.items():
        up = sql.upper()
        for bad in ("DELETE", "TRUNCATE", "DROP ", "ALTER ", "CREATE ", "GRANT", "REVOKE"):
            assert bad not in up, (name, bad)
        assert up.startswith(("SELECT", "INSERT", "UPDATE")), name
        if up.startswith("UPDATE"):
            assert "WHERE" in up and ("SET STATE =" in up or name == "vectors_upsert" or name.startswith("job_run_"))
        if up.startswith("INSERT") and name.startswith(("link_insert", "node_insert", "anchor_insert")):
            assert "ON CONFLICT" in up
    with pytest.raises(store.SqlRefused):
        store.run_sql(Conn(db), "drop_everything")
    with pytest.raises(store.SqlRefused):
        store.run_sql(Conn(db), "__class__")


def test_no_chunk_text_in_logs_or_status(caplog):
    caplog.set_level(logging.DEBUG)
    db = FakeDB(spans())
    j = make(db)
    j.run()
    assert SENTINEL_TEXT not in caplog.text and SENTINEL_TEXT not in repr(j.status())
    assert SENTINEL_TEXT not in json.dumps([p for n, p in db.log if n == "job_run_insert"], default=str)
    assert "text-must-not-leak" not in repr(j.status())  # non-numeric metrics are dropped


def test_loop_backoff_and_stop_on_inert():
    waits = []
    results = iter([{"state": "error"}, {"state": "error"}, {"state": "complete"}, {"state": "inert"}])
    fake = SimpleNamespace(jcfg={"interval_s": 3600, "initial_delay_s": 10}, run=lambda: next(results))
    job._STOP.clear()
    job.loop(fake, sleep=lambda w: waits.append(w) or False)
    assert waits == [10, 10, 20, 3600]


# ----------------------------------------------------------------------------------------------- status reader / embedder / M1
def test_status_reader_absent_and_ok_and_cached():
    cstatus.reset_for_tests()
    db = FakeDB([], has=(True, True, False))
    assert cstatus.status(engine_factory=factory_for(db), env=dict(ENV))["state"] == "absent"
    cstatus.reset_for_tests()
    fin = SimpleNamespace(isoformat=lambda: "2026-11-09T00:00:00+00:00")
    db2 = FakeDB([])
    orig = db2.h_any
    db2.h_any = lambda name, batch: Res([("v1", fin, {"a": 1, "t": "x"}, True, ["R"])]) if name == "job_run_latest" else orig(name, batch)
    seen = []
    s = cstatus.status(engine_factory=factory_for(db2, seen), env=dict(ENV))
    assert s == {"state": "ok", "version": "v1", "metrics": {"a": 1}, "display_ok": True, "reasons": ["R"],
                 "last_run": "2026-11-09T00:00:00+00:00"}
    cstatus.status(engine_factory=factory_for(db2, seen), env=dict(ENV))
    assert len(seen) == 1  # TTL cache
    cstatus.reset_for_tests()
    assert cstatus.status(engine_factory=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")), env=dict(ENV))["state"] == "absent"
    assert cstatus.status(env={})["display_ok"] is False


# ----------------------------------------------------------------------------------------------- Cloudflare adapter (fake opener)
import urllib.error  # noqa: E402

CF_ENV = {"CLOUDFLARE_ACCOUNT_ID": "acct123", "CLOUDFLARE_API_TOKEN": "TOKEN-SENTINEL_TEXT-VALUE"}


class FakeResp:
    def __init__(self, obj):
        self.b = json.dumps(obj).encode()

    def read(self, n=-1):
        return self.b if n is None or n < 0 else self.b[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def cf_ok(n, dim=4, usage=None):
    res = {"shape": [n, dim], "data": [[1.0] * dim for _ in range(n)]}
    if usage is not None:
        res["usage"] = usage
    return {"result": res, "success": True, "errors": [], "messages": []}


def http_err(code, body=None, retry_after=None):
    import io
    hdr = {"Retry-After": str(retry_after)} if retry_after is not None else {}
    return urllib.error.HTTPError("https://api.cloudflare.com/x", code, "m", hdr, io.BytesIO(json.dumps(body or {}).encode()))


class Opener:
    def __init__(self, *script):
        self.script, self.reqs = list(script), []

    def __call__(self, req, timeout=None):
        self.reqs.append(req)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResp(item(len(json.loads(req.data)["text"])) if callable(item) else item)


def cf(opener, ecfg=None, clock=lambda: 1_700_000_000.0, sleeps=None):
    emb.reset_budget_for_tests()
    return emb.CloudflareEmbedder(ecfg or {}, env=dict(CF_ENV), opener=opener, clock=clock,
                                  sleep=(sleeps.append if sleeps is not None else (lambda s: None)))


def test_embedder_slot_lazy_and_default_none():
    assert emb.load_embedder({})[0] is None and emb.load_embedder({"embedder": "none"})[1] == "embedder_none"
    assert emb.load_embedder({"embedder": "nope"})[1] == "embedder_unknown"
    assert emb.load_embedder({"embedder": "cloudflare"}, env={})[1] == "credentials_missing"
    assert emb.load_embedder({"embedder": "cloudflare", "cloudflare": {"send_allowed": False}}, env=dict(CF_ENV))[1] == "send_not_allowed"
    e, why = emb.load_embedder({"embedder": "cloudflare"}, env=dict(CF_ENV))
    assert why == "ok" and e.name == "@cf/baai/bge-m3"
    for mod in ("fastembed", "model2vec", "onnxruntime"):
        assert mod not in sys.modules
    emb.register("fake-t", lambda c: FakeEmb())
    try:
        assert emb.load_embedder({"embedder": "fake-t"})[1] == "ok"
    finally:
        emb.unregister("fake-t")


def test_cf_success_request_shape_dim_detected_and_secrets_not_logged(caplog):
    caplog.set_level(logging.DEBUG)
    op = Opener(lambda n: cf_ok(n, 5, usage={"total_tokens": 2000}))
    e = cf(op)
    out = e.embed(["a", "b"])
    r = op.reqs[0]
    assert r.full_url == "https://api.cloudflare.com/client/v4/accounts/acct123/ai/run/@cf/baai/bge-m3" and r.get_method() == "POST"
    assert r.get_header("Authorization") == "Bearer TOKEN-SENTINEL_TEXT-VALUE" and json.loads(r.data) == {"text": ["a", "b"], "truncate_inputs": True}
    assert len(out) == 2 and e.dim == 5
    assert abs(e.stats()["neurons_today"] - 2000 / 1e6 * 1075) < 1e-6  # usage field preferred
    assert "TOKEN-SENTINEL_TEXT" not in caplog.text and "acct123" not in caplog.text


def test_cf_dim_mismatch_refused():
    op = Opener(lambda n: cf_ok(n, 4), lambda n: cf_ok(n, 6))
    e = cf(op)
    e.embed(["a"])
    with pytest.raises(emb.EmbedError) as ei:
        e.embed(["b"])
    assert str(ei.value) == "dim_mismatch"
    e2 = cf(Opener(lambda n: cf_ok(n, 100)), {"model": "bge-small-en-v1.5"})  # documented dim 384
    with pytest.raises(emb.EmbedError):
        e2.embed(["a"])


def test_cf_429_retry_after_honoured_then_success_and_5xx_bounded():
    sleeps = []
    e = cf(Opener(http_err(429, retry_after=7), lambda n: cf_ok(n)), sleeps=sleeps)
    assert len(e.embed(["a"])) == 1 and sleeps == [7.0]
    sleeps2 = []
    e2 = cf(Opener(http_err(503), http_err(500), http_err(502), http_err(503)), sleeps=sleeps2)
    with pytest.raises(emb.EmbedError) as ei:
        e2.embed(["a"])
    assert str(ei.value) == "request_failed" and sleeps2 == [1.0, 2.0, 4.0]  # exponential, bounded attempts
    e3 = cf(Opener(http_err(401)))
    with pytest.raises(emb.EmbedError) as ei:
        e3.embed(["a"])
    assert str(ei.value) == "http_401" and "acct123" not in repr(ei.value)


def test_cf_quota_error_is_deferred_until_utc_midnight():
    e = cf(Opener(http_err(429, {"errors": [{"code": 4006, "message": "daily free allocation of 10,000 neurons used"}]})))
    with pytest.raises(emb.Deferred) as ei:
        e.embed(["a"])
    assert 0 < ei.value.retry_in_s <= 86400 + 60
    assert emb.seconds_to_utc_midnight(1_700_006_399.0) < 100  # 23:59:59 UTC => just after midnight


def test_cf_budget_guard_defers_and_resets_next_day():
    t = [1_700_000_000.0]
    e = cf(Opener(lambda n: cf_ok(n), lambda n: cf_ok(n)), {"daily_neuron_budget": 5, "chars_per_token": 1.0}, clock=lambda: t[0])
    e.embed(["x" * 1000])  # est ~1 neuron
    with pytest.raises(emb.Deferred):
        e.embed(["x" * 10_000_000])  # est ~10,750 neurons > budget; nothing is sent
    assert len(e.opener.reqs) == 1
    t[0] += 86400
    e.embed(["y"])  # new UTC day
    assert len(e.opener.reqs) == 2


def test_cf_long_input_split_mean_pooled_and_normalised():
    seen = []

    def resp(n):
        seen.append(n)
        return {"result": {"data": [[3.0, 0.0], [0.0, 4.0]][:n]}, "success": True, "errors": []}
    e = cf(Opener(resp), {"models": {"bge-m3": {"max_tokens": 10}}, "chars_per_token": 1.0})
    out = e.embed(["a" * 15])  # 2 parts of <=10 chars
    assert seen == [2] and len(out) == 1
    assert abs(sum(x * x for x in out[0]) - 1.0) < 1e-9 and abs(out[0][0] - 0.6) < 1e-9 and abs(out[0][1] - 0.8) < 1e-9


def test_cf_send_not_allowed_refuses_and_sends_nothing():
    op = Opener()
    emb.reset_budget_for_tests()
    e = emb.CloudflareEmbedder({"send_allowed": False}, env=dict(CF_ENV), opener=op)
    with pytest.raises(emb.EmbedError):
        e.embed(["a"])
    assert op.reqs == []


def test_job_deferred_is_not_failure_and_model_change_reembeds(caplog):
    caplog.set_level(logging.INFO)
    db = FakeDB(spans())
    class Def(FakeEmb):
        def embed(self, texts):
            raise emb.Deferred("quota_reached", 1234.0)
    j = make(db, e=Def())
    r = j.run()
    assert r["state"] == "deferred" and r["retry_in_s"] == 1234.0 and not r["complete"]
    assert "CT_JOB_ERROR" not in caplog.text and j.status()["reasons"] == ["deferred_until_utc_midnight"]
    make(db).run()
    class NewModel(FakeEmb):
        name = "@cf/baai/other"
    e2 = NewModel()
    make(db, e=e2).run()
    assert sum(e2.calls) == 10  # a model id change re-embeds every span
    waits = []
    job._STOP.clear()
    job.loop(SimpleNamespace(jcfg={"interval_s": 3600, "initial_delay_s": 5}, run=iter([{"state": "deferred", "retry_in_s": 5000.0}, {"state": "inert"}]).__next__),
             sleep=lambda w: waits.append(w) or False)
    assert waits == [5, 5000.0]


def test_cf_response_size_is_bounded():
    e = cf(Opener(lambda n: cf_ok(n)), {"max_response_bytes": 20})
    with pytest.raises(emb.EmbedError) as ei:
        e.embed(["a"])
    assert str(ei.value) == "response_too_large"


def test_cf_default_opener_refuses_redirects_so_the_token_is_not_forwarded():
    import io
    import urllib.request as ur
    h = emb._NoRedirect()
    req = ur.Request("https://api.cloudflare.com/x", data=b"{}", headers={"Authorization": "Bearer T"})
    assert h.redirect_request(req, io.BytesIO(b""), 302, "Found", {}, "https://evil.example.invalid/") is None
    e = cf(Opener(http_err(302)))
    with pytest.raises(emb.EmbedError) as ei:
        e.embed(["a"])
    assert str(ei.value) == "http_302"
    assert emb.CloudflareEmbedder({}, env=dict(CF_ENV)).opener is emb._default_open


# ----------------------------------------------------------------------------------------------- TAXO-DIAG-FIX-1110 self-report
@pytest.fixture(autouse=True)
def _fresh_inert_flag():
    job.reset_for_tests()
    yield
    job.reset_for_tests()


def test_startup_logs_delay_and_missing_cred_names_only(caplog):
    caplog.set_level(logging.INFO, logger="vera.chunk_taxo")
    sentinel = "placeholder-not-a-credential"  # must never appear in the warning
    c = dict(cfg(initial_delay_s=10**6), embedder="cloudflare")
    assert job.start(c, env={"CLOUDFLARE_API_TOKEN": "  "}) is True
    job.reset_for_tests()
    msgs = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert (logging.INFO, "chunk_taxo: first pass in 1000000 s") in msgs
    warn = [m for lv, m in msgs if lv == logging.WARNING and "credentials_missing" in m]
    assert len(warn) == 1 and "CLOUDFLARE_ACCOUNT_ID" in warn[0] and "CLOUDFLARE_API_TOKEN" in warn[0] and sentinel not in warn[0]


def test_startup_warning_never_logs_a_present_credential_value(caplog):
    """Non-vacuous redaction check: the token is SET to a sentinel and only the account id is missing; the warning
    names the missing variable and no log record carries the token's value."""
    caplog.set_level(logging.INFO, logger="vera.chunk_taxo")
    sentinel = "placeholder-not-a-credential"
    c = dict(cfg(initial_delay_s=10**6), embedder="cloudflare")
    assert job.start(c, env={"CLOUDFLARE_API_TOKEN": sentinel}) is True
    job.reset_for_tests()
    msgs = [(r.levelno, r.getMessage()) for r in caplog.records]
    warn = [m for lv, m in msgs if lv == logging.WARNING and "credentials_missing" in m]
    assert len(warn) == 1 and "CLOUDFLARE_ACCOUNT_ID" in warn[0] and "CLOUDFLARE_API_TOKEN" not in warn[0]
    assert not [m for _, m in msgs if sentinel in m]


def test_startup_no_warning_when_creds_present(caplog):
    caplog.set_level(logging.INFO, logger="vera.chunk_taxo")
    c = dict(cfg(initial_delay_s=10**6), embedder="cloudflare")
    assert job.start(c, env={"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}) is True
    job.reset_for_tests()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("first pass in" in r.getMessage() for r in caplog.records)


def test_inert_log_is_warning_and_one_failed_row_per_process(caplog):
    caplog.set_level(logging.INFO, logger="vera.chunk_taxo")
    db = FakeDB(spans())
    e = {"CHUNK_TAXO_EMBEDDER": "cloudflare", **ENV}
    r1 = job.Job({"enabled": True, "params": {}, "embedder": "cloudflare", "send_allowed": True}, core=StubCore(), engine_factory=factory_for(db), env=e)
    assert r1.run()["state"] == "inert" and db.inert_rows == [{"reasons": ["credentials_missing"]}]
    assert [r.levelno for r in caplog.records if "inert:" in r.getMessage()] == [logging.WARNING]
    r2 = job.Job({"enabled": True, "params": {}, "embedder": "cloudflare", "send_allowed": True}, core=StubCore(), engine_factory=factory_for(db), env=e)
    assert r2.run()["state"] == "inert" and len(db.inert_rows) == 1  # second inert in the same process writes nothing


def test_inert_row_insert_failure_tolerated(caplog):
    caplog.set_level(logging.INFO, logger="vera.chunk_taxo")
    db = FakeDB(spans())
    db.fail_on = "job_run_inert"
    j = job.Job({"enabled": True, "params": {}}, core=StubCore(), engine_factory=factory_for(db), env=dict(ENV))
    assert j.run()["state"] == "inert"
    assert [r for r in caplog.records if r.levelno == logging.WARNING and "CT_INERT_ROW_ERROR" in r.getMessage()]


def test_enabled_path_writes_no_inert_row():
    db = FakeDB(spans())
    j = make(db)
    assert j.run()["complete"] and db.inert_rows == [] and "job_run_inert" not in db.names()


def test_embedder_loads_with_account_and_api_token_and_inert_when_either_missing(caplog):
    from vera.chunk_taxo_embed import load_embedder
    c = {"embedder": "cloudflare", "send_allowed": True}
    emb, why = load_embedder(c, env={"CLOUDFLARE_ACCOUNT_ID": "fake-acct", "CLOUDFLARE_API_TOKEN": "fake-token"})
    assert emb is not None and why == "ok"
    for env in ({"CLOUDFLARE_ACCOUNT_ID": "fake-acct"}, {"CLOUDFLARE_API_TOKEN": "fake-token"}):
        emb, why = load_embedder(c, env=env)
        assert emb is None and why == "credentials_missing"
        caplog.clear()
        caplog.set_level(logging.INFO, logger="vera.chunk_taxo")
        job.reset_for_tests()
        assert job.start(dict(cfg(initial_delay_s=10**6), embedder="cloudflare"), env=env) is True
        job.reset_for_tests()
        w = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(w) == 1 and "credentials_missing" in w[0] and "fake-" not in w[0]
        assert ("CLOUDFLARE_API_TOKEN" in w[0]) != ("CLOUDFLARE_ACCOUNT_ID" in w[0])
