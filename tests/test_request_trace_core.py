"""PERF-MISS-1104-TRACE (OD-1104-TRACE): vera/request_trace.py unit tests. Offline: fake engines only, no DB, no network."""
import contextvars
import json
import logging
import threading

import pytest

from vera import digest_store, progress, request_trace as rt

BASE = json.loads(rt.CONFIG_PATH.read_text(encoding="utf-8"))


def cfg(**over):
    c = json.loads(json.dumps(BASE))
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(c.get(k), dict):
            c[k].update(v)
        else:
            c[k] = v
    return c


class FakeConn:
    def __init__(self, eng):
        self.eng = eng

    def execute(self, sql, params=None):
        if self.eng.fail:
            raise RuntimeError("db down")
        self.eng.calls.append((str(sql), params))

        class R:
            def first(_self):
                return self.eng.row
        return R()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeEngine:
    def __init__(self, fail=False, row=None):
        self.calls, self.fail, self.row = [], fail, row

    def begin(self):
        return FakeConn(self)

    def connect(self):
        return FakeConn(self)


def factory_for(eng, made):
    def f(url, **kw):
        made.append((url, kw))
        return eng
    return f


@pytest.fixture(autouse=True)
def reset():
    yield
    rt.configure(BASE)
    rt.end_context()


def in_ctx(fn):
    return contextvars.copy_context().run(fn)


def lines(caplog):
    out = []
    for r in caplog.records:
        msg = r.getMessage()
        if r.name == "vera.trace" and msg.startswith(rt.LOG_PREFIX + " "):
            out.append(json.loads(msg[len(rt.LOG_PREFIX) + 1:]))
    return out


def test_shipped_config_defaults():
    assert BASE["enabled"] is True and BASE["log_sink"]["enabled"] is True
    # INTEGRATE-1104 round 2: migration 015 applied by the owner 2026-11-04 -> both DB flags ship on
    assert BASE["db_sink"]["enabled"] is True and BASE["digest_store"]["enabled"] is True  # DEFERRED-DB-1104: on
    assert BASE["db_sink"]["db_url_env_wo"] == "VERA_DB_URL_WO"
    assert "_calibration" in BASE


def test_ids_are_server_generated_hex_and_distinct_from_client_header(caplog):
    caplog.set_level(logging.INFO, logger="vera.trace")
    rt.configure(BASE)

    def run():
        tid = rt.start("q?", {"endpoint": "/ask", "client_request_id": "c" * 32, "junk": "x"})
        rt.finish("answer", "complete")
        return tid
    tid = in_ctx(run)
    assert rt.TRACE_ID_RE.match(tid) and tid != "c" * 32
    first = lines(caplog)[0]
    assert first["client_request_id"] == "c" * 32 and first["endpoint"] == "/ask" and "junk" not in first
    bad = in_ctx(lambda: (rt.start("q", {"client_request_id": "has space/../x"}), rt.CURRENT.get().client_request_id)[1])
    assert bad is None


def test_default_log_line_has_no_question_text(caplog):
    caplog.set_level(logging.INFO, logger="vera.trace")
    rt.configure(BASE)
    in_ctx(lambda: (rt.start("SECRET-QUESTION-CANARY text", {}), rt.finish("a", "b")))
    assert "SECRET-QUESTION-CANARY" not in caplog.text
    assert lines(caplog)[0]["question_chars"] == len("SECRET-QUESTION-CANARY text")


def test_event_order_seq_and_question_only_on_start(caplog):
    caplog.set_level(logging.INFO, logger="vera.trace")
    rt.configure(cfg(log_sink={"include_question": True}))

    def run():
        tid = rt.start("What about the 2026 midterm?", {"endpoint": "/ask"})
        progress.emit(progress.Stage.SEARCH, progress.State.START, step=1)  # no progress sink: still traced
        rt.event("premise", "verdict", counts={"n": 2}, detail={"verdict": "ok"})
        rt.finish("answer", "complete")
        rt.event("late", "x")  # after finish: dropped
        rt.finish("answer", "again")  # idempotent
        return tid
    tid = in_ctx(run)
    ls = lines(caplog)
    assert [(x["stage"], x["code"]) for x in ls] == [("start", "received"), ("search", "start"), ("premise", "verdict"),
                                                    ("finish", "answer")]
    assert [x["seq"] for x in ls] == [1, 2, 3, 4] and all(x["request_id"] == tid for x in ls)
    assert ls[0]["question"] == "What about the 2026 midterm?"
    assert all("question" not in x for x in ls[1:])
    assert ls[1]["counts"] == {"step": 1}
    assert ls[3]["detail"] == {"message_code": "complete"} and "elapsed_ms" in ls[3]["counts"]


def test_no_trace_outside_request_and_disabled_flag(caplog):
    caplog.set_level(logging.INFO, logger="vera.trace")
    rt.configure(BASE)
    rt.event("x", "y")
    rt.finish("a", "b")
    progress.emit(progress.Stage.SEARCH, progress.State.START)
    rt.configure(cfg(enabled=False))
    assert in_ctx(lambda: rt.start("q", {})) is None
    rt.configure(cfg(log_sink={"enabled": False}))
    in_ctx(lambda: (rt.start("q", {}), rt.event("a", "b"), rt.finish("c", "d")))
    assert lines(caplog) == []


def test_size_caps(monkeypatch):
    lim = dict(rt.DEFAULT_LIMITS)
    keys = ["s", "lst", "nest"] + [f"k{i}" for i in range(100)]
    monkeypatch.setitem(rt.CFG, "key_allowlist", {"t": {"keys": keys, "reason": "test"}})
    d = rt.clean_detail({"s": "ab" * 2000, "lst": list(range(500)), "nest": {"a": {"b": {"c": 1}}},
                         **{f"k{i}": i for i in range(100)}}, lim, stage="t")
    assert d["s"] == "<str>" and len(d["lst"]) == lim["max_list_items"]
    assert d["nest"] == {"a": {"b": "<dict>"}}
    assert len(json.dumps(d)) <= lim["max_detail_bytes"] and d["_truncated"] is True
    assert len([k for k in d if k != "_truncated"]) <= lim["max_detail_keys"]
    monkeypatch.setitem(rt.CFG, "key_allowlist", {"t": {"keys": [f"k{i}" for i in range(20)], "reason": "test"}})
    big = rt.clean_detail({f"k{i}": ["y" * 60] * 20 for i in range(20)}, lim, stage="t")
    assert len(json.dumps(big, separators=(",", ":"))) <= lim["max_detail_bytes"] and big["_truncated"] is True
    q = in_ctx(lambda: (rt.start("z" * 9000, {}), len(rt.CURRENT.get().question))[1])
    assert q == lim["max_question_chars"]


def test_event_cap_keeps_finish(caplog):
    caplog.set_level(logging.INFO, logger="vera.trace")
    rt.configure(cfg(limits={"max_events_per_request": 5}))

    def run():
        rt.start("q", {})
        for i in range(20):
            rt.event("loop", "tick", counts={"i": i})
        rt.finish("answer", "complete")
        return rt.CURRENT.get().dropped
    dropped = in_ctx(run)
    ls = lines(caplog)
    assert len(ls) == 5 and ls[-1]["stage"] == "finish" and dropped == 17 and ls[-1]["counts"]["dropped"] == 17


@pytest.mark.parametrize("value", [
    "postgresql://vera_pipeline_wo:hunter2@db.example.oregon-postgres.render.com:5432/vera_dar_db",
    "see https://api.groq.com/openai/v1?key=abc failing",
    "Authorization: Bearer abcdefghijklmnop1234",
    "key sk-proj-AAAAbbbbCCCCdddd1234 leaked",
    "gsk_ABCDEFGHIJKLMNOPQRSTUV0123 leaked",
    "host 10.1.2.3:5432 down",
    "server dpg-xyz.oregon-postgres.render.com refused",
    "user vera_pipeline_wo@dpg-abc.render.com",
])
def test_secrets_urls_hosts_redacted(value, monkeypatch):
    monkeypatch.setitem(rt.CFG, "key_allowlist", {"t": {"keys": ["error_code"], "reason": "test"}})
    d = rt.clean_detail({"error_code": value}, stage="t")
    s = json.dumps(d)
    for needle in ("hunter2", "render.com", "groq.com", "abcdefghijklmnop1234", "AAAAbbbb", "ABCDEFGHIJ", "10.1.2.3",
                   "://"):
        assert needle not in s, (needle, s)


def test_secret_named_keys_dropped_and_allowed_keys_kept(monkeypatch):
    ok = ["passage_ids", "score", "reason", "provider", "status", "latency_ms", "model", "tokens", "prompt_tokens",
          "doc_sha", "api_key", "password", "db_url", "host", "access_token", "authorization", "session", "dsn"]
    monkeypatch.setitem(rt.CFG, "key_allowlist", {"t": {"keys": ok, "reason": "test"}})
    d = rt.clean_detail({"api_key": "x", "password": "p", "db_url": "u", "host": "h", "access_token": "t",
                         "authorization": "a", "session": "s", "dsn": "d",
                         "passage_ids": ["d12:p3", "d40:p1"], "score": 0.731234567, "reason": "below_floor",
                         "provider": "openalex", "status": "rate_limited", "latency_ms": 812, "model": "llama-3.1-8b-instant",
                         "tokens": 1200, "prompt_tokens": 900, "doc_sha": "a" * 64, "Bad Key": 1}, stage="t")
    d.pop("dropped_keys")
    assert set(d) == {"passage_ids", "score", "reason", "provider", "status", "latency_ms", "model", "tokens",
                      "prompt_tokens", "doc_sha"}
    assert d["model"] == "llama-3.1-8b-instant" and d["doc_sha"] == "a" * 64 and d["score"] == 0.731235


def test_never_raises_on_hostile_input(caplog):
    class Boom:
        def __repr__(self):
            raise RuntimeError("x")

        def __str__(self):
            raise RuntimeError("x")
    rt.configure(BASE)

    def run():
        rt.start(Boom(), Boom())
        rt.event(Boom(), Boom(), counts=Boom(), detail={"a": Boom(), "b": float("nan")})
        rt.event("ok", "ok", counts={"bad": "str", "b": True, "n": 1})
        rt.finish(Boom(), None)
    in_ctx(run)  # no exception


def test_log_sink_failure_never_raises(monkeypatch):
    rt.configure(BASE)
    monkeypatch.setattr(rt.log, "info", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("log down")))
    in_ctx(lambda: (rt.start("q", {}), rt.event("a", "b"), rt.finish("c", "d")))
    assert rt.DROPS.get("log_error", 0) >= 3


def test_flag_off_means_no_db_writes():
    eng, made = FakeEngine(), []
    rt.configure(cfg(db_sink={"enabled": False}), engine_factory=factory_for(eng, made), env={"VERA_DB_URL_WO": "postgresql://fake"})
    in_ctx(lambda: (rt.start("q", {}), rt.event("a", "b"), rt.finish("c", "d")))
    assert rt._WRITER is None and made == [] and eng.calls == []


def test_db_sink_writes_insert_only_in_background():
    eng, made = FakeEngine(), []
    rt.configure(cfg(db_sink={"enabled": True, "flush_interval_s": 0.05}), engine_factory=factory_for(eng, made),
                 env={"VERA_DB_URL_WO": "postgresql://fake"})
    main_thread = threading.get_ident()
    tid = in_ctx(lambda: (rt.start("the question", {"endpoint": "/ask/stream"}), rt.event("a", "b"),
                          rt.finish("answer", "complete"))[0])
    assert rt.drain_for_tests(5)
    assert made and made[0][0] == "postgresql://fake" and made[0][1]["hide_parameters"] is True
    sqls = [s for s, _ in eng.calls]
    assert all(s.startswith("SET LOCAL") or s.startswith("INSERT INTO vera_vjay.request_trace_") for s in sqls)
    assert not any(w in s.upper() for s in sqls for w in ("UPDATE ", "DELETE", "SELECT"))
    evts = [p for s, p in eng.calls if "request_trace_events" in s]
    reqs = [p for s, p in eng.calls if "request_trace_requests" in s]
    rows = [r for batch in evts for r in batch]
    assert [r["seq"] for r in rows] == [1, 2, 3] and all(r["request_id"] == tid for r in rows)
    assert len(reqs) == 1 and reqs[0][0]["question"] == "the question" and reqs[0][0]["outcome_branch"] == "answer"
    assert all("question" not in r for r in rows)
    assert rt._WRITER.thread.ident != main_thread


def test_db_errors_and_full_queue_never_raise_or_block():
    eng = FakeEngine(fail=True)
    rt.configure(cfg(db_sink={"enabled": True, "flush_interval_s": 0.05, "queue_max": 2}),
                 engine_factory=factory_for(eng, []), env={"VERA_DB_URL_WO": "postgresql://fake"})

    def run():
        rt.start("q", {})
        for _ in range(50):
            rt.event("a", "b")
        rt.finish("x", "y")
    in_ctx(run)
    rt.drain_for_tests(5)
    assert rt.DROPS.get("queue_full", 0) > 0 and rt.DROPS.get("write_error", 0) >= 1


def test_db_sink_without_url_drops_quietly():
    rt.configure(cfg(db_sink={"enabled": True, "flush_interval_s": 0.05}), engine_factory=factory_for(FakeEngine(), []),
                 env={})
    in_ctx(lambda: (rt.start("q", {}), rt.finish("x", "y")))
    rt.drain_for_tests(5)
    assert rt.DROPS.get("no_writer", 0) >= 1


def test_bad_table_name_refused():
    w = rt._Writer({"requests_table": "public.x; DROP TABLE y", "flush_interval_s": 0.05}, factory_for(FakeEngine(), []),
                   {"VERA_DB_URL_WO": "postgresql://fake"})
    try:
        with pytest.raises(ValueError):
            w.flush([("evt", {})])
    finally:
        w.stop()


def test_recent_window_for_owner_route():
    rt.configure(cfg(limits={"recent_max": 2}))
    ids = [in_ctx(lambda: (rt.start(f"q{i}", {}), rt.finish("a", "b"))[0]) for i in range(3)]
    assert rt.recent(ids[0]) is None and rt.recent(ids[2])["question"] == "q2"
    assert rt.recent("not-hex") is None and rt.recent(None) is None


# ---------------------------------------------------------------------------------------------- digest_store
def test_digest_store_off_by_default_and_never_raises():
    eng, made = FakeEngine(), []
    off = {**BASE["digest_store"], "enabled": False}  # INTEGRATE-1104: ships on since 015; the off path is still pinned
    assert digest_store.enabled() is True and digest_store.enabled(off) is False
    assert digest_store.save({"summary": "s"}, "k1", cfg=off, env={"VERA_DB_URL_WO": "x"}, engine_factory=factory_for(eng, made)) is False
    assert digest_store.load_latest(cfg=off, env={"VERA_DB_URL_RO": "x"}, engine_factory=factory_for(eng, made)) is None
    assert made == []
    on = {**BASE["digest_store"], "enabled": True}
    bad = FakeEngine(fail=True)
    digest_store.reset_for_tests()
    assert digest_store.save({"a": 1}, "k", cfg=on, env={"VERA_DB_URL_WO": "x"}, engine_factory=factory_for(bad, [])) is False
    digest_store.reset_for_tests()
    assert digest_store.load_latest(cfg=on, env={"VERA_DB_URL_RO": "y"}, engine_factory=factory_for(bad, [])) is None


def test_digest_store_insert_and_read_latest():
    on = {**BASE["digest_store"], "enabled": True}
    digest_store.reset_for_tests()
    eng = FakeEngine()
    assert digest_store.save({"summary": "S", "topics": ["a"]}, "12|2026-11-04|p3", cfg=on, env={"VERA_DB_URL_WO": "w"},
                             engine_factory=factory_for(eng, [])) is True
    ins = [p for s, p in eng.calls if s.startswith("INSERT INTO vera_vjay.corpus_digest_state")]
    assert len(ins) == 1 and len(ins[0]["h"]) == 64 and json.loads(ins[0]["d"])["summary"] == "S"
    assert digest_store.save({"x": "y" * 70000}, "k", cfg=on, env={"VERA_DB_URL_WO": "w"},
                             engine_factory=factory_for(eng, [])) is False
    assert digest_store.save({"x": 1}, "bad key!", cfg=on, env={"VERA_DB_URL_WO": "w"},
                             engine_factory=factory_for(eng, [])) is False

    class Row:
        _mapping = {"cache_key": "k", "digest": {"summary": "S"}, "created_at": None}
    digest_store.reset_for_tests()
    reader = FakeEngine(row=Row())
    got = digest_store.load_latest(cfg=on, env={"VERA_DB_URL_RO": "r"}, engine_factory=factory_for(reader, []))
    assert got == {"cache_key": "k", "digest": {"summary": "S"}, "created_at": None}
    assert any("ORDER BY digest_id DESC LIMIT 1" in s for s, _ in reader.calls)
    digest_store.reset_for_tests()
