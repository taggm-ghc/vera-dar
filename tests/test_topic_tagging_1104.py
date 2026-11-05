"""TOPIC-TAGGING-1104: offline tests (fake DB store, fake model, fake SQLAlchemy engine). No network, no database."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from vera import micro_topic
from vera import topic_tagging as tt

REG = micro_topic.Registry({"anchor_registry_version": "t", "entries": [
    {"id": "epi-outbreak", "label": "Outbreak epidemiology", "anchor_text": "outbreak epidemic transmission cases vaccine"},
    {"id": "ml-models", "label": "Machine learning models", "anchor_text": "neural network training model learning dataset"},
    {"id": "climate", "label": "Climate change", "anchor_text": "climate warming emissions carbon temperature"},
]})
# R4: the test registry has 3 roots (min_active_roots 1 here); limits confirmed so the worker is not paused; the BM25 fallback
# (default off until T9) is switched on where a test exercises it
CFG = {**tt.DEFAULTS, "enabled": True, "calls_per_minute": 6000, "min_active_roots": 1, "limits_confirmed": True,
       "tokens_per_minute": 10 ** 9, "bm25_fallback_after_failures": 3}


def fake_llm(reply):
    calls = []

    def llm(messages, max_tokens=300, timeout_s=None):
        calls.append(messages)
        r = reply(messages) if callable(reply) else reply
        if isinstance(r, BaseException):
            raise r
        return r, "groq:fake-free"
    llm.calls = calls
    return llm


def cand_index(messages, key_label):
    body = messages[1]["content"].split("\n\nPassage:")[0]
    for line in body.splitlines()[1:]:
        n, label = line.split(". ", 1)
        if label == key_label:
            return int(n)
    raise AssertionError(key_label)


# ------------------------------------------------------------------ pure logic
def test_parse_tags_deny_all_and_ranking():
    cands = [("a", "A"), ("b", "B"), ("c", "C")]
    tags, props = tt.parse_tags({"topics": [{"n": 2, "score": 0.9}, {"n": 1, "score": 0.9}, {"n": 3, "score": 0.2}],
                                 "proposed": []}, cands, CFG)
    assert [t["topic_key"] for t in tags] == ["a", "b"]  # ties by key; 0.2 < min_score dropped
    for bad in ({"topics": [{"n": 4, "score": 0.9}]}, {"topics": [{"n": 1, "score": 1.5}]}, {"topics": [{"n": True, "score": 1}]},
                {"topics": [{"n": 1, "score": 0.9}, {"n": 1, "score": 0.8}]}, {"topics": "a"}, [], {"topics": [{"n": 1}]},
                {"topics": [{"n": 1, "score": float("nan")}]}, {"topics": [], "proposed": "x"}):
        with pytest.raises(tt.TaggingError):
            tt.parse_tags(bad, cands, CFG)


def test_model_can_only_pick_offered_candidates():
    part = tt.Part(1, 11, "outbreak cases rose after transmission")
    llm = fake_llm(lambda m: {"topics": [{"n": cand_index(m, "Outbreak epidemiology"), "score": 0.8}], "proposed": [],
                              "extra": "ignored"})
    out = tt.tag_part(part, REG, llm, CFG)
    assert out["outcome"] == "tagged" and out["tags"] == [{"topic_key": "epi-outbreak", "score": 0.8, "rank": 1}]
    assert out["method"] == "llm" and out["model"] == "groq:fake-free"
    assert "outbreak cases rose" in llm.calls[0][1]["content"]  # the part's full text is what the model reads


def test_proposals_off_by_default_and_gated():
    part = tt.Part(1, 11, "quantum chromodynamics lattice")
    reply = {"topics": [], "proposed": [{"label": "Lattice QCD", "field": "physics", "anchor": "lattice quark gluon"}]}
    seen = []
    prop = lambda l, f, a, par=None: seen.append(l) or "lattice-qcd"  # noqa: E731
    assert tt.tag_part(part, REG, fake_llm(reply), CFG, prop)["outcome"] == "no_topic" and not seen
    out = tt.tag_part(part, REG, fake_llm(reply), {**CFG, "propose_new_topics": True}, prop)
    assert seen == ["Lattice QCD"] and out["outcome"] == "no_topic" and out["tags"] == []  # R4 (skeptic 5): registered, never a tag
    out = tt.tag_part(part, REG, fake_llm(reply), {**CFG, "propose_new_topics": True}, lambda *a: None)
    assert out["outcome"] == "no_topic"  # registry did not accept: nothing stored


def test_bm25_fallback_after_failures_uses_no_model():
    part = tt.Part(1, 11, "climate warming emissions carbon temperature rise", failures=3)
    llm = fake_llm(AssertionError("model must not be called"))
    out = tt.tag_part(part, REG, llm, CFG)
    assert out["method"] == "bm25" and out["tags"][0]["topic_key"] == "climate" and 0 < out["tags"][0]["score"] < 1
    assert not llm.calls


def test_composite_weights():
    rows = tt.composite([(4000, [{"topic_key": "a", "score": 1.0}, {"topic_key": "b", "score": 0.5}]),
                         (2000, [{"topic_key": "b", "score": 1.0}]), (2000, [])])
    assert rows == [{"topic_key": "a", "weight": 0.5, "chunk_share": 0.5, "n_chunks": 1},
                    {"topic_key": "b", "weight": 0.5, "chunk_share": 0.75, "n_chunks": 2}]  # tie on weight: key order; shares by chars
    assert tt.composite([]) == []


def test_n2_child_of_retired_parent_is_inactive_for_sweep_status_and_retag():
    """N2 (R5 verify): the ACTIVE key set honours the ancestor chain: 'child' is DB-active but its parent 'par' is retired (absent)."""
    reg = micro_topic.Registry({"entries": [{"id": "child", "label": "C", "anchor_text": "c", "parent_topic_key": "par"},
                                            {"id": "ok", "label": "O", "anchor_text": "o"}]})
    ld = tt.chain_usable_loader(lambda: reg, lambda: {"child", "ok"})
    assert ld() == {"ok"}
    ro = FakeEngine([("WITH sig AS", [(4,)]), ("SELECT count(*) FROM (", [(1,)]), ("NOT EXISTS (SELECT 1 FROM", [(5, 50)])])
    st = tt.PgTagStore(ro, FakeEngine([]), {**CFG}, active_keys_loader=ld)
    assert st.active_keys() == {"ok"}
    st.stale_sources(10); st.inactive_only(10); st.inactive_only_count()
    qs = [p for _, p in ro.log if isinstance(p, dict) and "keys" in p]
    assert len(qs) == 3 and all(p["keys"] == ["ok"] for p in qs)  # 'child' never counts as active in any of the three queries


def test_error_code_is_class_only():
    assert tt.error_code(ValueError("secret host db.example:5432")) == "ValueError"


# ------------------------------------------------------------------ deny-all candidate filter
def test_usable_keys_db_active_and_parent_chain():
    reg = micro_topic.Registry({"entries": [
        {"id": "aa", "label": "A", "anchor_text": "a"}, {"id": "bb", "label": "B", "anchor_text": "b", "parent_topic_key": "aa"},
        {"id": "cc", "label": "C", "anchor_text": "c", "parent_topic_key": "zz"}, {"id": "dd", "label": "D", "anchor_text": "d"}]})
    assert tt.usable_keys(reg, None) == {"aa", "bb", "dd"}  # cc's parent is unknown/retired: not offered
    assert tt.usable_keys(reg, {"bb", "dd"}) == {"dd"}  # bb's parent not active in the DB
    short = tt.shortlist(reg, "a b c d", 12, {"dd"})
    assert [k for k, _ in short] == ["dd"]


# ------------------------------------------------------------------ worker over an in-memory store
class FakeStore:
    def __init__(self, parts, active=None, refuse_keys=()):
        self.parts = {(p.source_id, p.span_id): p for p in parts}
        self.status, self.failures, self.tags, self.docs, self.queued = {}, {}, {}, {}, []
        self.attempt_id, self.last_attempt = 0, {}
        self.active = active if active is not None else {e["id"] for e in REG.entries}
        self.refuse = set(refuse_keys)

    def pending(self, limit, source_id=None):
        out = []
        for k, p in sorted(self.parts.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1)):
            if source_id is not None and k[0] != source_id:
                continue
            if self.status.get(k) in tt.TERMINAL or self.failures.get(k, 0) >= CFG["max_failures"]:
                continue
            out.append(tt.Part(p.source_id, p.span_id, p.text, self.failures.get(k, 0), self.status.get(k)))
        return out[:limit]

    def active_keys(self):
        return set(self.active)

    def _att(self, k, outcome):
        self.attempt_id += 1
        self.status[k] = outcome
        if outcome in tt.TERMINAL:
            self.last_attempt[k] = self.attempt_id

    def enqueue(self, parts):
        new = [(p.source_id, p.span_id) for p in parts if p.status is None]
        for k in new:
            self._att(k, "queued")
        self.queued += new
        return len(new)

    def record_failure(self, part, exc):
        k = (part.source_id, part.span_id)
        self.failures[k] = self.failures.get(k, 0) + 1
        self._att(k, "failed")
        return "failed"

    def write_result(self, part, result):
        k = (part.source_id, part.span_id)
        if self.status.get(k) in tt.TERMINAL:
            return "done_elsewhere"
        if any(t["topic_key"] in self.refuse for t in result["tags"]):
            return self.record_failure(part, RuntimeError("guard"))
        self.tags[k] = result["tags"]
        self._att(k, result["outcome"])
        return result["outcome"]

    def _sig(self, sid):
        term = [k for k in self.status if k[0] == sid and self.status[k] in tt.TERMINAL]
        ntags = sum(len(self.tags.get(k, [])) for k in term)
        return tt.signature(len(term), max([self.last_attempt[k] for k in term] or [0]), ntags), term, ntags

    def stale_sources(self, limit, after=0, keys=None):
        out = [sid for sid in sorted({k[0] for k in self.status}) if sid > after and self._sig(sid)[2]
               and self.docs.get(sid, (None,))[0] != self._sig(sid)[0]]
        return out[:limit]

    def sources_for_hashes(self, hashes, limit):
        return [int(h[-1]) for h in hashes][:limit]

    def refresh_document(self, sid, keys=None):
        sig, term, ntags = self._sig(sid)
        if not ntags or self.docs.get(sid, (None,))[0] == sig:
            return False
        self.docs[sid] = (sig, tt.composite([(len(self.parts[k].text), self.tags.get(k, [])) for k in term]))
        return True


def make_worker(store, reply, **kw):
    changed = []
    clock = [1000.0]
    w = tt.Worker(store, lambda: REG, fake_llm(reply), {**CFG, **kw}, on_documents_changed=lambda: changed.append(1),
                  clock=lambda: clock[0], sleep=lambda s: None)
    w.test_clock = clock
    return w, changed


def epi_reply(m):
    return {"topics": [{"n": cand_index(m, "Outbreak epidemiology"), "score": 0.9}], "proposed": []}


def test_worker_tags_pending_then_composite_and_notifies_digest():
    store = FakeStore([tt.Part(1, 10, "outbreak cases"), tt.Part(1, 11, "transmission"), tt.Part(2, 20, "abstract text")])
    w, changed = make_worker(store, epi_reply)
    w.tick()
    assert set(store.status.values()) == {"tagged"} and store.docs[1][1][0]["topic_key"] == "epi-outbreak"
    assert set(store.docs) == {1, 2} and changed and sorted(store.queued) == [(1, 10), (1, 11), (2, 20)]
    calls = len(w.llm.calls)
    w.tick()  # idempotent: nothing pending, nothing stale
    assert len(w.llm.calls) == calls and w.health()["ticks"] == 2


def test_late_no_topic_part_makes_composite_stale():
    store = FakeStore([tt.Part(1, 10, "outbreak cases"), tt.Part(1, 11, "other")])
    w, _ = make_worker(store, lambda m: epi_reply(m) if "outbreak cases" in m[1]["content"] else {"topics": [], "proposed": []},
                       batch=1)
    w.tick()
    first = store.docs[1]
    w.tick()  # the second part ends no_topic: signature changes, weights are recomputed over both parts
    assert store.status[(1, 11)] == "no_topic" and store.docs[1] != first
    assert store.docs[1][1][0]["chunk_share"] == round(len("outbreak cases") / len("outbreak casesother"), 4)


def test_parse_failure_counts_then_bm25_but_transient_does_not():
    store = FakeStore([tt.Part(3, 30, "climate warming emissions carbon")])
    w, _ = make_worker(store, tt.TaggingError("malformed"))
    for _ in range(3):
        w.tick()
    assert store.failures[(3, 30)] == 3 and store.status[(3, 30)] == "failed" and store.queued == [(3, 30)]
    w.tick()  # failures >= 3 -> bm25 fallback, no model
    assert store.status[(3, 30)] == "tagged" and store.tags[(3, 30)][0]["topic_key"] == "climate"
    store2 = FakeStore([tt.Part(4, 40, "outbreak")])
    w2, _ = make_worker(store2, tt.Transient("Timeout"))
    w2.tick()
    assert (4, 40) not in store2.failures and w2.cooldown_left() == CFG["transient_cooldown_s"]
    calls = len(w2.llm.calls)
    w2.tick()  # cooling down: no model call, nothing recorded
    assert len(w2.llm.calls) == calls


def test_rate_limit_is_not_an_attempt_and_cools_down_process_wide():
    store = FakeStore([tt.Part(4, 40, "outbreak"), tt.Part(4, 41, "outbreak")])
    w, _ = make_worker(store, tt.RateLimited(7), concurrency=1)
    w.tick()
    assert not store.failures and w.cooldown_left() == 7 and len(w.llm.calls) == 1  # second part deferred, no call
    w.test_clock[0] += 8
    w.llm = fake_llm(epi_reply)
    w.tick()
    assert store.status[(4, 40)] == "tagged"


def test_guard_refusal_is_recorded_so_retries_are_bounded():
    store = FakeStore([tt.Part(5, 50, "outbreak")], refuse_keys={"epi-outbreak"})
    w, _ = make_worker(store, epi_reply)
    for _ in range(8):
        w.tick()
    assert store.failures[(5, 50)] >= 3  # recorded each time; then bm25 (no topic for this text) or stop at max_failures
    assert len(w.llm.calls) == 3


def test_only_db_active_keys_are_offered():
    store = FakeStore([tt.Part(6, 60, "outbreak transmission")], active={"ml-models"})
    w, _ = make_worker(store, lambda m: {"topics": [{"n": 1, "score": 0.9}], "proposed": []})
    w.tick()
    assert store.tags[(6, 60)][0]["topic_key"] == "ml-models"
    assert "Outbreak epidemiology" not in w.llm.calls[0][1]["content"]


def test_registry_unavailable_tags_nothing():
    store = FakeStore([tt.Part(7, 70, "outbreak")])
    store.active_keys = lambda: (_ for _ in ()).throw(RuntimeError("db down"))
    w, _ = make_worker(store, epi_reply)
    w.tick()
    assert not w.llm.calls and (7, 70) not in store.status


def test_tick_is_bounded_by_batch():
    store = FakeStore([tt.Part(8, i, "outbreak") for i in range(1, 30)])
    w, _ = make_worker(store, epi_reply, batch=5)
    for sid in range(8, 20):
        w.notify("source", 8)
    w.tick()
    assert len(w.llm.calls) == 5


def test_hooks_are_non_blocking_and_never_raise():
    tt.install(None)
    assert tt.notify_ingested(5) is False and tt.schedule_refresh(["a" * 64]) == 0 and tt.health() is None
    store = FakeStore([tt.Part(5, 50, "outbreak")])
    w, _ = make_worker(store, epi_reply, queue_max=1)
    tt.install(w)
    try:
        assert tt.notify_ingested(5) is True
        assert tt.notify_ingested(6) is False  # queue full: dropped, the sweep still finds it
        assert tt.notify_ingested(True) is False and tt.schedule_refresh([1, None, "zz", object()]) == 0
        assert not w.llm.calls  # nothing ran inline
    finally:
        tt.install(None)


def test_dead_thread_is_restarted_on_notify():
    store = FakeStore([])
    w, _ = make_worker(store, epi_reply)

    class Dead:
        def is_alive(self):
            return False
    w._thread = Dead()
    started = []
    w.start = lambda: started.append(1)
    w.notify("source", 1)
    assert started == [1] and w.stats["restarts"] == 1


def test_answer_time_refresh_by_text_hash():
    store = FakeStore([tt.Part(7, 70, "outbreak")])
    w, _ = make_worker(store, epi_reply)
    w.tick()
    store.docs.clear()  # composite missing: stale
    w.notify("doc", "0" * 63 + "7")
    w.tick()
    assert 7 in store.docs


def test_load_cfg_default_off():
    assert tt.load_cfg(tt.ROOT / "config" / "does-not-exist.json")["enabled"] is False
    assert tt.build_worker({**CFG, "enabled": False}) is None
    assert tt.build_worker(CFG, env={}) is None  # URLs missing: off, names only logged


def test_lock_key_is_64_bit_and_stable():
    a, b = tt.lock_key(1, None), tt.lock_key(1, 2)
    assert a != b and -2 ** 63 <= a < 2 ** 63 and a == tt.lock_key(1, None)


# ------------------------------------------------------------------ PgTagStore SQL path over a fake engine
class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def scalar(self):
        return self.rows[0][0] if self.rows else None

    def one(self):
        return self.rows[0]

    def __iter__(self):
        return iter(self.rows)


class FakeConn:
    def __init__(self, log, answers, fail_on=None):
        self.log, self.answers, self.fail_on = log, answers, fail_on

    def execute(self, sql, params=None):
        s = str(sql)
        if self.fail_on and self.fail_on in s:
            raise RuntimeError("guard refused host=secret")
        self.log.append((" ".join(s.split()[:2]), params))
        for frag, rows in self.answers:
            if frag in s:
                return FakeResult(rows(params) if callable(rows) else rows)
        return FakeResult([])

    def commit(self):
        self.log.append(("COMMIT", None))

    def rollback(self):
        self.log.append(("ROLLBACK", None))

    def begin(self):
        conn = self

        class Tx:
            def __enter__(self):
                conn.log.append(("BEGIN", None))
                return conn

            def __exit__(self, *a):
                conn.log.append(("COMMIT" if a[0] is None else "ROLLBACK", None))
                return False
        return Tx()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeEngine:
    def __init__(self, answers, fail_on=None):
        self.log, self.answers, self.fail_on = [], answers, fail_on
        self.fail_once = fail_on

    def connect(self):
        return FakeConn(self.log, self.answers)

    def begin(self):
        f, self.fail_once = self.fail_once, None  # only the first transaction fails
        return FakeConn(self.log, self.answers, f).begin()


RES = {"outcome": "tagged", "tags": [{"topic_key": "epi-outbreak", "score": 0.9, "rank": 1}], "method": "llm", "model": "m" * 100}


def test_pg_write_result_one_transaction_with_xact_lock():
    wo = FakeEngine([("toptag_span_status_v", []), ("max(set_version)", [(2,)])])
    assert tt.PgTagStore(FakeEngine([]), wo, CFG).write_result(tt.Part(1, 10, "x"), RES) == "tagged"
    kinds = [k for k, _ in wo.log]
    assert kinds[0] == "BEGIN" and kinds[1].startswith("SELECT pg_advisory_xact_lock") and kinds[-1] == "COMMIT"
    ins = [p for k, p in wo.log if k.startswith("INSERT INTO")]
    assert ins[0]["sv"] == 2 and ins[0]["k"] == "epi-outbreak" and len(ins[0]["mo"]) == 80 and ins[-1]["o"] == "tagged"
    assert not [k for k in kinds if "pg_try_advisory_lock" in k or "unlock" in k]  # no session locks at all


def test_pg_write_result_already_done_and_refusal_recorded():
    done = FakeEngine([("toptag_span_status_v", [("tagged",)])])
    assert tt.PgTagStore(FakeEngine([]), done, CFG).write_result(tt.Part(1, 10, "x"), RES) == "done_elsewhere"
    refused = FakeEngine([("toptag_span_status_v", []), ("max(set_version)", [(1,)])], fail_on="toptag_chunk_topics (span_id")
    assert tt.PgTagStore(FakeEngine([]), refused, CFG).write_result(tt.Part(1, None, "x"), RES) == "failed"
    ins = [p for k, p in refused.log if k.startswith("INSERT INTO")]
    assert ins == [{"pid": None, "sid": 1, "o": "failed", "e": "RuntimeError", "tv": "tt1"}]
    assert "secret" not in json.dumps(refused.log, default=str)


def test_pg_refresh_document_signature_one_statement_after_lock():
    ro = FakeEngine([("length(e.text)", [(10, 4000), (11, 4000)])])
    state = [("t", 10, 7, None, None), ("t", 11, 9, None, None), ("g", 10, None, "a", 1.0)]
    fresh = FakeEngine([("SELECT 't' AS kind", state + [("d", None, None, "2:9:1", None)])])
    assert tt.PgTagStore(ro, fresh, CFG).refresh_document(9) is False
    assert not [k for k, _ in fresh.log if k.startswith("INSERT")]
    stale = FakeEngine([("SELECT 't' AS kind", state + [("d", None, None, "1:5:1", None)]), ("max(set_version)", [(3,)])])
    assert tt.PgTagStore(ro, FakeEngine([("SELECT 't' AS kind", state + [("d", None, None, "1:5:1", None)])]), CFG
                         ).refresh_document(9) is False  # R5 (R5K-1): no active key set known -> nothing composed (deny-all)
    assert tt.PgTagStore(ro, stale, CFG, active_keys_loader=lambda: ["a"]).refresh_document(9) is True
    kinds = [k for k, _ in stale.log]
    assert kinds[0] == "BEGIN" and kinds[1].startswith("SELECT pg_advisory_xact_lock") and kinds[2].startswith("SELECT 't'")
    assert kinds[-1] == "COMMIT"
    ins = [p for k, p in stale.log if k.startswith("INSERT INTO")]
    assert ins == [{"sid": 9, "sv": 3, "k": "a", "w": 0.5, "cs": 0.5, "n": 1, "sig": "2:9:1"}]


def test_pg_record_failure_skips_terminal_part():
    done = FakeEngine([("toptag_span_status_v", [("tagged",)])])
    assert tt.PgTagStore(FakeEngine([]), done, CFG).record_failure(tt.Part(1, 10, "x"), ValueError()) == "done_elsewhere"
    assert not [k for k, _ in done.log if k.startswith("INSERT")]


def test_sql_shapes():
    one = tt.pending_sql(False, True)
    assert "evidence_type = 'fulldoc_partition'" in one and "max(v.version)" in one and one.count(":sid") == 1
    assert "UNION ALL" not in one and "UNION ALL" in tt.pending_sql(True, False)
    for s in (tt.SQL_INSERT_TAG, tt.SQL_ATTEMPT, tt.SQL_DOC_INSERT):
        assert s.startswith("INSERT INTO vera_vjay.toptag_") and "DELETE" not in s and "UPDATE" not in s
    assert "created_at" not in tt.SQL_STALE_SWEEP  # no clock watermark


# ---- R3 (convergence round 3, 2026-11-05) -------------------------------------------------------------------------------------
def test_limits_must_be_confirmed_before_any_model_call():
    assert tt.DEFAULTS["limits_confirmed"] is False and not tt.limits_ok(tt.DEFAULTS)
    assert not tt.limits_ok({**CFG, "limits_confirmed": True, "tokens_per_minute": 0})
    assert not tt.limits_ok({**CFG, "limits_confirmed": "true", "tokens_per_minute": 6000})  # literal true only
    assert tt.limits_ok({**CFG, "limits_confirmed": True, "tokens_per_minute": 6000})
    env = {CFG["db_url_env_ro"]: "postgresql://x", CFG["db_url_env_wo"]: "postgresql://y"}
    unconfirmed = {**CFG, "limits_confirmed": False}  # R4: the module CFG confirms limits; this check needs them unconfirmed
    assert tt.build_worker(unconfirmed, env=env, engine_factory=lambda *a, **k: 1 / 0) is None  # refused before any engine is made
    assert tt.main(["--backfill"]) == 2  # staged config: limits_confirmed false -> refused before any connection


def test_repeated_transient_part_is_skipped_so_it_cannot_hold_the_head():
    store = FakeStore([tt.Part(4, 40, "outbreak"), tt.Part(4, 41, "outbreak")])
    w, _ = make_worker(store, tt.Transient("Timeout"), batch=1, concurrency=1)
    for _ in range(2):
        w.tick()
        w.test_clock[0] += CFG["transient_cooldown_s"] + 1
    assert w.skipping() == 1 and not store.failures  # skipped in process, still not an attempt
    w.llm = fake_llm(epi_reply)
    w.tick()
    assert store.status.get((4, 41)) == "tagged" and store.status.get((4, 40)) not in tt.TERMINAL


def test_failing_write_cools_down_instead_of_respending_quota():
    store = FakeStore([tt.Part(6, 60, "outbreak")])

    def boom(part, result):
        raise RuntimeError("db down")
    store.write_result = boom
    w, _ = make_worker(store, epi_reply)
    w.tick()
    calls = len(w.llm.calls)
    assert w.cooldown_left() > 0
    w.tick()
    assert len(w.llm.calls) == calls


def test_proposed_null_is_accepted_like_missing():
    cands = [("epi-outbreak", "Outbreak epidemiology")]
    tags, props = tt.parse_tags({"topics": [{"n": 1, "score": 0.9}], "proposed": None}, cands, CFG)
    assert tags[0]["topic_key"] == "epi-outbreak" and props == []


# ---- R4 (convergence round 4, 2026-11-05) -------------------------------------------------------------------------------------
def test_proposal_never_stores_a_non_offered_key_or_fabricated_score():
    """R4 (skeptic 5): an ACTIVE key returned by the registry for a proposal is not stored on this part."""
    part = tt.Part(1, 11, "outbreak transmission cases")
    reply = lambda m: {"topics": [{"n": cand_index(m, "Outbreak epidemiology"), "score": 0.9}],  # noqa: E731
                       "proposed": [{"label": "Contact tracing", "field": "epi", "anchor": "contact tracing apps",
                                     "parent": cand_index(m, "Outbreak epidemiology")}]}
    out = tt.tag_part(part, REG, fake_llm(reply), {**CFG, "propose_new_topics": True}, lambda *a: "climate")
    assert [t["topic_key"] for t in out["tags"]] == ["epi-outbreak"] and out["tags"][0]["score"] == 0.9


def test_empty_registry_makes_no_call_and_no_terminal_row():
    """R4 (skeptic 1): nothing to offer -> no model call, no attempt, the part stays pending."""
    store = FakeStore([tt.Part(3, 30, "outbreak")], active=set())
    w, _ = make_worker(store, AssertionError("model must not be called"))
    w.tick()
    assert not w.llm.calls and (3, 30) not in store.status
    with pytest.raises(tt.NoCandidates):
        tt.tag_part(tt.Part(3, 30, "x"), REG, fake_llm(AssertionError("no")), CFG, allowed=set())


def test_registry_below_min_active_roots_tags_nothing():
    """R4 (skeptic 1, D2): fewer active top-level topics than min_active_roots -> paused, no call, no row."""
    store = FakeStore([tt.Part(3, 31, "outbreak")])
    w, _ = make_worker(store, epi_reply, min_active_roots=4)  # the test registry has 3 roots
    w.tick()
    assert not w.llm.calls and not store.status and w.stats["last_error"] == "registry_too_small"
    assert tt.DEFAULTS["min_active_roots"] >= 20  # shipped: waits for the curated roots


def test_bm25_fallback_off_by_default():
    """R4 (SEOS 7): no lexical tag from a failed model reply until T9 switches it on."""
    part = tt.Part(1, 11, "climate warming emissions carbon temperature rise", failures=4)
    llm = fake_llm({"topics": [], "proposed": []})
    out = tt.tag_part(part, REG, llm, {**CFG, "bm25_fallback_after_failures": tt.DEFAULTS["bm25_fallback_after_failures"]})
    assert out["method"] != "bm25" and llm.calls


def test_db_control_kill_switch_and_pacing():
    """R4 (SEOS 2): owner DB control rows pause tagging without a restart and retune the pacer; no 016 view = config only."""
    store = FakeStore([tt.Part(5, 50, "outbreak")])
    rows = {"enabled": {"value": False}}
    store.control = lambda: rows
    w, _ = make_worker(store, epi_reply)
    w.tick()
    assert not w.llm.calls and not store.status and w.health()["paused"] == "db_control_disabled"
    rows.clear()
    rows.update({"enabled": {"value": True}, "calls_per_minute": {"value": 3}, "batch": {"value": "x"}})
    w.test_clock[0] += 31  # next control poll
    w.tick()
    assert w.llm.calls and w.cfg["calls_per_minute"] == 3 and w.cfg["batch"] == CFG["batch"] and w.health()["paused"] is None
    assert abs(w.pace.gap - 20.0) < 1e-9
    store2 = FakeStore([tt.Part(5, 51, "outbreak")])
    store2.control = lambda: (_ for _ in ()).throw(RuntimeError("db"))
    w2, _ = make_worker(store2, epi_reply)
    w2.tick()
    assert not w2.llm.calls and w2.health()["paused"] == "db_control_unreadable"
    eff, why = tt.apply_control(CFG, None)
    assert why is None and eff == CFG  # 016 not applied: the config file alone
    assert tt.apply_control(CFG, {"limits_confirmed": {"value": False}})[1] == "limits_unconfirmed"


def test_digest_signal_is_debounced_and_paced():
    """R4 (skeptic 4): at most one digest signal per digest_signal_min_s, carrying the worker's pacer; a due signal is
    sent on a later tick, not lost."""
    store = FakeStore([tt.Part(1, 10, "outbreak cases")])
    got = []
    w, _ = make_worker(store, epi_reply)
    w.on_documents_changed = lambda pace=None: got.append(pace)
    w.tick()
    assert got == [w.pace]
    store.parts[(1, 12)] = tt.Part(1, 12, "outbreak")
    w.tick()
    assert len(got) == 1 and w._digest_due
    w.test_clock[0] += 601
    w.tick()
    assert len(got) == 2 and not w._digest_due


def test_counts_sql_has_exhausted_bucket_and_latest_filters():
    assert "'exhausted'" in tt.SQL_COUNTS and ":maxf" in tt.SQL_COUNTS
    assert "max(v.version)" in tt.SQL_STALE_SWEEP and "max(v.version)" in tt.SQL_HASH_SOURCES
    assert ":maxf" in tt.SQL_EXHAUSTED and tt.SQL_CONTROL.startswith("SELECT") and "lstreg_active_items_v" in tt.SQL_CONTROL


def test_tagger_reaches_the_registry_only_through_micro_topic():
    """R4 (orchestrator, T5 guard tests test_topicreg_1028 / test_topicjudge_1028): no registry SQL, no direct registry or judge
    module use; active keys come from the injected loader (vera.micro_topic.registry_active_keys)."""
    src = open(tt.__file__, encoding="utf-8").read()
    assert "topic_registry" not in src and "topic_judge" not in src and "vera_vjay.topreg" not in src
    store = tt.PgTagStore(None, None, CFG, active_keys_loader=lambda: ["aa", "bb"])
    assert store.active_keys() == {"aa", "bb"} and tt.PgTagStore(None, None, CFG).active_keys() == set()


# ---- R5 (convergence round 5, 2026-11-05; fixed after the bound, NOT re-reviewed) ----------------------------------------------
def test_corpus_reads_use_ro_and_the_sweep_filters_active_keys():
    """R5P-1: every PgTagStore query naming sources or evidence_spans runs on the ro engine (wo has no SELECT on them)."""
    ro = FakeEngine([("WITH sig AS", [(4,), (7,)]), ("SELECT count(*) FROM (", [(3,)]), ("NOT EXISTS (SELECT 1 FROM", [(5, 50)]),
                     ("length(e.text)", [(10, 4000)])])
    wo = FakeEngine([("SELECT 't' AS kind", [("t", 10, 7, None, None), ("g", 10, None, "old", 1.0)])])
    st = tt.PgTagStore(ro, wo, {**CFG}, active_keys_loader=lambda: ["epi-outbreak"])
    assert st.stale_sources(50, after=3) == [4, 7]
    assert st.inactive_only(10) == [(5, 50)] and st.inactive_only_count() == 3
    assert st.refresh_document(9) is False  # the only tag carries an inactive key: nothing composed, nothing written
    assert not [k for k, _ in wo.log if k.startswith("INSERT")]
    sweep = [p for k, p in ro.log if k == "WITH sig"]
    assert sweep == [{"lim": 50, "after": 3, "keys": ["epi-outbreak"]}]
    for name in dir(tt):
        v = getattr(tt, name)
        if name.startswith("SQL_") and isinstance(v, str) and ("sources" in v or "evidence_spans" in v):
            assert name not in ("SQL_DOC_STATE", "SQL_STATUS_ONE", "SQL_NEXT_SET", "SQL_DOC_NEXT_SET"), name
    import inspect
    src = inspect.getsource(tt.PgTagStore)
    assert "self.wo.connect()" not in src  # the wo account is used only in write transactions (begin)
    for s_ in (tt.SQL_STALE_SWEEP, tt.SQL_DOC_STATE, tt.SQL_INACTIVE_ONLY):
        assert "ANY(:keys)" in s_ and "topreg" not in s_


class _SweepStore(FakeStore):
    """60 stale documents whose refresh always fails (retired keys) and one valid stale document after them."""
    def __init__(self):
        super().__init__([])
        self.bad, self.ok_done, self.after_seen = set(range(1, 61)), False, []

    def stale_sources(self, limit, after=0, keys=None):
        self.after_seen.append(after)
        return [s for s in range(1, 62) if s > after and (s in self.bad or not self.ok_done)][:limit]

    def refresh_document(self, sid, keys=None):
        if sid in self.bad:
            raise RuntimeError("guard: key not active")
        self.ok_done = True
        return True


def test_stale_sweep_cursor_reaches_later_documents():
    """R5K-1: a block of always-failing documents no longer starves later ones (stale_sweep_max 50)."""
    store = _SweepStore()
    w, _ = make_worker(store, epi_reply, stale_sweep_max=50)
    w.tick()
    assert not store.ok_done and w._sweep_after == 50
    w.tick()
    assert store.ok_done and store.after_seen == [0, 50] and w._sweep_after == 0  # short page -> wraps


def test_retag_requeues_only_still_tagged_parts_under_the_lock():
    """R5K-2: owner re-tag writes one 'queued' attempt per part (append-only), skips a part that is no longer tagged."""
    wo = FakeEngine([("toptag_span_status_v", [("tagged",)])])
    assert tt.PgTagStore(FakeEngine([]), wo, CFG).requeue(5, 50) is True
    kinds = [k for k, _ in wo.log]
    assert kinds[1].startswith("SELECT pg_advisory_xact_lock") and kinds[-1] == "COMMIT"
    ins = [p for k, p in wo.log if k.startswith("INSERT INTO")]
    assert ins == [{"pid": 50, "sid": 5, "o": "queued", "e": "retag_inactive_key", "tv": "tt1"}]
    gone = FakeEngine([("toptag_span_status_v", [("queued",)])])
    assert tt.PgTagStore(FakeEngine([]), gone, CFG).requeue(5, 50) is False
    assert not [k for k, _ in gone.log if k.startswith("INSERT")]
    assert "UPDATE" not in tt.SQL_INACTIVE_ONLY and "DELETE" not in tt.SQL_INACTIVE_ONLY


def test_malformed_kill_switch_fails_closed():
    """R5K-7: a mistyped enabled row pauses; it never leaves tagging on."""
    assert tt.apply_control(CFG, {"enabled": {"value": "false"}})[1] == "db_control_invalid"
    assert tt.apply_control(CFG, {"enabled": "false"})[1] == "db_control_invalid"
    assert tt.apply_control(CFG, {"enabled": {"value": True}, "batch": {"value": "x"}})[1] is None  # other keys: ignored, logged


def test_long_retry_after_is_recorded_so_the_cli_can_stop():
    """R5K-5: a Retry-After above max_retry_after_s is recorded (the capped cooldown alone never reached the stop threshold)."""
    store = FakeStore([tt.Part(1, 10, "outbreak cases")])
    w, _ = make_worker(store, tt.RateLimited(86400.0))
    w.tick()
    assert w.long_wait_s == 86400.0 and w.cooldown_left() <= CFG["max_retry_after_s"] and (1, 10) not in store.tags


def test_registry_block_is_reported_for_the_cli():
    """R5P-3: run_parts names the registry reason, so --backfill stops instead of busy-looping on reads."""
    store = FakeStore([tt.Part(3, 31, "outbreak")])
    w, _ = make_worker(store, epi_reply, min_active_roots=4)
    w.run_parts(store.pending(5))
    assert w.blocked == "registry_too_small"
